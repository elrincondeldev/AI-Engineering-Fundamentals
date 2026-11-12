"""Run tool calls safely: permissions, validation, approval, timeouts, retries, idempotency.

The model only *asks* for a tool call. This executor decides whether and how it
runs, and always returns a :class:`ToolResult` that can go back to the model as
a ``tool`` message, including when something went wrong.
"""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeout
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from .providers import Message, ToolCall
from .schema import validate
from .store import ToolError, TransientToolError
from .tools import Session, Tool


class Permission(StrEnum):
    ALLOW = "allow"  # run without asking
    APPROVE = "approve"  # run only if a person says yes
    DENY = "deny"  # never run in this context


Approver = Callable[[ToolCall], bool]


@dataclass(frozen=True)
class ToolResult:
    """The outcome of one tool call, ready to be sent back to the model."""

    call_id: str
    name: str
    status: str  # ok | unknown_tool | denied | invalid_arguments | rejected | timeout | error
    data: Any = None
    error: str = ""
    attempts: int = 0

    @property
    def ok(self) -> bool:
        return self.status == "ok"

    def payload(self) -> dict[str, Any]:
        if self.ok:
            return {"ok": True, "data": self.data}
        return {"ok": False, "error": {"type": self.status, "message": self.error}}

    def to_message(self) -> Message:
        content = json.dumps(self.payload(), ensure_ascii=False, sort_keys=True)
        return Message("tool", content, name=self.name, tool_call_id=self.call_id)


def idempotency_key(session: Session, call: ToolCall) -> str:
    """Same conversation + same tool + same arguments = same key, so a retry cannot act twice."""
    canonical = json.dumps({"c": session.conversation_id, "t": call.name, "a": call.arguments}, sort_keys=True)
    return hashlib.sha256(canonical.encode()).hexdigest()[:16]


@dataclass
class ToolExecutor:
    """Executes :class:`ToolCall` objects against a toolbox on behalf of one session."""

    tools: dict[str, Tool]
    session: Session
    policy: dict[str, Permission] = field(default_factory=dict)
    approver: Approver | None = None
    max_retries: int = 2
    backoff_s: float = 0.05
    use_idempotency_keys: bool = True
    sleep: Callable[[float], None] = time.sleep
    audit: list[dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        self._pool = ThreadPoolExecutor(max_workers=8, thread_name_prefix="tool")

    def permission(self, tool: Tool) -> Permission:
        """Explicit policy first; otherwise read-only tools are allowed and side effects need approval."""
        return self.policy.get(tool.name, Permission.APPROVE if tool.side_effect else Permission.ALLOW)

    def specs(self) -> list[dict[str, Any]]:
        """Only advertise tools this session may use (a denied tool is not even shown)."""
        return [t.spec() for t in self.tools.values() if self.permission(t) != Permission.DENY]

    def execute(self, call: ToolCall) -> ToolResult:
        result = self._execute(call)
        self.audit.append({"call_id": call.id, "tool": call.name, "arguments": call.arguments,
                           "status": result.status, "attempts": result.attempts})
        return result

    def _execute(self, call: ToolCall) -> ToolResult:
        tool = self.tools.get(call.name)
        if tool is None:
            return ToolResult(call.id, call.name, "unknown_tool",
                              error=f"Unknown tool '{call.name}'. Available: {sorted(self.tools)}")
        if self.permission(tool) == Permission.DENY:
            return ToolResult(call.id, call.name, "denied", error="This tool is not allowed in this context.")
        problems = validate(call.arguments, tool.parameters)
        if problems:
            return ToolResult(call.id, call.name, "invalid_arguments",
                              error="; ".join(problems) + ". Fix the arguments and call the tool again.")
        if self.permission(tool) == Permission.APPROVE and not (self.approver and self.approver(call)):
            return ToolResult(call.id, call.name, "rejected",
                              error="A person did not approve this action. Do not retry; tell the customer.")
        args = dict(call.arguments)
        if tool.side_effect and self.use_idempotency_keys:
            args["_idempotency_key"] = idempotency_key(self.session, call)
        return self._run_with_retries(tool, call, args)

    def _run_with_retries(self, tool: Tool, call: ToolCall, args: dict[str, Any]) -> ToolResult:
        last = ""
        for attempt in range(1, self.max_retries + 2):
            future = self._pool.submit(tool.handler, args, self.session)
            try:
                return ToolResult(call.id, call.name, "ok", data=future.result(timeout=tool.timeout_s),
                                  attempts=attempt)
            except FutureTimeout:
                last, status = f"timed out after {tool.timeout_s} s", "timeout"
            except TransientToolError as exc:
                last, status = str(exc), "error"
            except ToolError as exc:  # business rule: retrying will not help
                return ToolResult(call.id, call.name, "error", error=str(exc), attempts=attempt)
            except Exception as exc:  # noqa: BLE001 - a crashing tool must not crash the assistant
                return ToolResult(call.id, call.name, "error", error=f"{type(exc).__name__}: {exc}",
                                  attempts=attempt)
            if attempt <= self.max_retries:
                self.sleep(self.backoff_s * 2 ** (attempt - 1))  # exponential backoff
        return ToolResult(call.id, call.name, status, error=last, attempts=self.max_retries + 1)

    def execute_all(self, calls: Sequence[ToolCall], parallel: bool = True) -> list[ToolResult]:
        """Run several calls from one model turn; results keep the order of the calls."""
        if not parallel or len(calls) < 2:
            return [self.execute(c) for c in calls]
        with ThreadPoolExecutor(max_workers=len(calls)) as outer:
            return list(outer.map(self.execute, calls))
