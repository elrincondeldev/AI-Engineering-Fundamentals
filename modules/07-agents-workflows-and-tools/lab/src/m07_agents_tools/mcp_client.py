"""A *teaching* MCP-style client and host helpers. Not the official MCP SDK.

- :class:`StdioClient` launches a server as a subprocess and talks JSON-RPC 2.0
  over its stdin/stdout, one message per line.
- :func:`remote_tools` turns the server's ``tools/list`` into local :class:`Tool`
  objects, so the same executor and tool loop work with remote tools. Whether a
  tool changes data is decided by the **host's** policy, not by the server.
- :func:`lint_description` and :func:`pin_tools` are two host-side defences
  against poisoned or silently changed tool descriptions.
"""

from __future__ import annotations

import hashlib
import json
import os
import queue
import re
import subprocess
import sys
import threading
from dataclasses import dataclass, field
from typing import Any

from .mcp_server import PROTOCOL_VERSION
from .store import ToolError
from .tools import Session, Tool

SERVER_CMD = [sys.executable, "-m", "m07_agents_tools.mcp_server"]
HOST_WRITE_TOOLS = {"create_return"}  # the host's own list of tools that change data


def minimal_env(extra: dict[str, str]) -> dict[str, str]:
    """Only what the server needs: never hand a server the host's whole environment (API keys!)."""
    keep = ("PATH", "SYSTEMROOT", "COURSE_DATASETS_DIR")
    return {**{k: os.environ[k] for k in keep if k in os.environ}, **extra}


class McpError(RuntimeError):
    """A JSON-RPC error response, a dead server or a timeout."""


@dataclass
class StdioClient:
    """One connection to one server process."""

    command: list[str] = field(default_factory=lambda: list(SERVER_CMD))
    env: dict[str, str] = field(default_factory=dict)
    timeout_s: float = 10.0
    transcript: list[tuple[str, dict[str, Any]]] = field(default_factory=list)

    def __post_init__(self) -> None:
        self._next_id = 0
        self._inbox: queue.Queue[str] = queue.Queue()
        self._lock = threading.Lock()  # one request in flight at a time keeps this client simple
        self.proc: subprocess.Popen[str] | None = None

    def start(self) -> StdioClient:
        self.proc = subprocess.Popen(
            self.command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, encoding="utf-8", bufsize=1, env=minimal_env(self.env),
        )
        threading.Thread(target=self._read_loop, daemon=True).start()
        return self

    def _read_loop(self) -> None:
        assert self.proc and self.proc.stdout
        for line in iter(self.proc.stdout.readline, ""):
            self._inbox.put(line)

    def _send(self, message: dict[str, Any]) -> None:
        assert self.proc and self.proc.stdin, "call start() first"
        self.transcript.append(("->", message))
        self.proc.stdin.write(json.dumps(message) + "\n")
        self.proc.stdin.flush()

    def request(self, method: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        """Send a request and wait for the response with the same id."""
        with self._lock:
            return self._request(method, params)

    def _request(self, method: str, params: dict[str, Any] | None) -> dict[str, Any]:
        self._next_id += 1
        msg: dict[str, Any] = {"jsonrpc": "2.0", "id": self._next_id, "method": method}
        if params is not None:
            msg["params"] = params
        self._send(msg)
        while True:
            try:
                reply = json.loads(self._inbox.get(timeout=self.timeout_s))
            except queue.Empty as exc:
                raise McpError(f"{method}: no response within {self.timeout_s} s") from exc
            self.transcript.append(("<-", reply))
            if reply.get("id") == self._next_id:
                break
        if "error" in reply:
            raise McpError(f"{reply['error']['code']}: {reply['error']['message']}")
        return reply["result"]

    def notify(self, method: str, params: dict[str, Any] | None = None) -> None:
        msg: dict[str, Any] = {"jsonrpc": "2.0", "method": method}
        if params is not None:
            msg["params"] = params
        self._send(msg)

    def initialize(self) -> dict[str, Any]:
        result = self.request("initialize", {
            "protocolVersion": PROTOCOL_VERSION, "capabilities": {},
            "clientInfo": {"name": "hearthwise-host", "version": "0.1.0"},
        })
        self.notify("notifications/initialized")
        return result

    def list_tools(self) -> list[dict[str, Any]]:
        return self.request("tools/list")["tools"]

    def call_tool(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        return self.request("tools/call", {"name": name, "arguments": arguments})

    def close(self) -> None:
        if self.proc:
            if self.proc.stdin:
                self.proc.stdin.close()
            try:
                self.proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.proc.kill()
            for stream in (self.proc.stdout, self.proc.stderr):
                if stream:
                    stream.close()

    def __enter__(self) -> StdioClient:
        return self.start()

    def __exit__(self, *exc: object) -> None:
        self.close()


def remote_tools(client: StdioClient, listed: list[dict[str, Any]]) -> dict[str, Tool]:
    """Wrap each remote tool as a local Tool whose handler forwards a ``tools/call``."""

    def make_handler(name: str):
        def handler(args: dict[str, Any], session: Session) -> Any:
            clean = {k: v for k, v in args.items() if not k.startswith("_")}  # keys stay on the host
            result = client.call_tool(name, clean)
            data = json.loads(result["content"][0]["text"])
            if result.get("isError"):
                raise ToolError(data.get("error", "remote tool failed"))
            return data

        return handler

    return {
        t["name"]: Tool(t["name"], t.get("description", ""), t.get("inputSchema", {"type": "object"}),
                        make_handler(t["name"]), side_effect=t["name"] in HOST_WRITE_TOOLS, timeout_s=5.0)
        for t in listed
    }


# --------------------------------------------------------------------------- #
# Host-side defences                                                          #
# --------------------------------------------------------------------------- #

LINT_RULES = [
    ("hidden-instruction markup", re.compile(r"<\s*(important|system|instructions?)\s*>", re.I)),
    ("tells the model to hide something", re.compile(r"\b(do not|don't|never) (tell|mention|show|reveal)", re.I)),
    ("overrides other instructions", re.compile(r"\bignore (all |any )?(previous|prior|other)", re.I)),
    ("asks for secrets", re.compile(r"\b(password|api[_ ]?key|token|secret|credential)s?\b", re.I)),
    ("contains a URL or e-mail", re.compile(r"https?://|[\w.+-]+@[\w-]+\.[\w.]+", re.I)),
]


def lint_description(tool: dict[str, Any], other_tools: set[str]) -> list[str]:
    """Flag description text that tries to steer the model instead of describing the tool."""
    text = tool.get("description", "")
    findings = [label for label, rx in LINT_RULES if rx.search(text)]
    mentioned = sorted(n for n in other_tools if n != tool["name"] and n in text)
    if mentioned:
        findings.append(f"mentions other tools: {', '.join(mentioned)}")
    if len(text) > 600:
        findings.append("unusually long description")
    return findings


def pin_tools(listed: list[dict[str, Any]]) -> dict[str, str]:
    """A fingerprint of each tool definition, to store when a person approves the server."""
    return {t["name"]: hashlib.sha256(json.dumps(t, sort_keys=True).encode()).hexdigest()[:12] for t in listed}


def diff_pins(approved: dict[str, str], current: dict[str, str]) -> dict[str, list[str]]:
    """What changed since approval: new tools, removed tools, changed definitions."""
    return {
        "added": sorted(set(current) - set(approved)),
        "removed": sorted(set(approved) - set(current)),
        "changed": sorted(n for n in set(approved) & set(current) if approved[n] != current[n]),
    }
