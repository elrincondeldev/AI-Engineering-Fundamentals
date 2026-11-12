#!/usr/bin/env python3
"""Check that your machine is ready for the course.

Usage::

    python3 tools/check_setup.py

Required: Git, uv (which downloads Python 3.12 for the labs) and the shared
datasets. Optional: Ollama for local models, an OpenAI-compatible endpoint,
and Node.js (only for contributors who validate diagrams).
"""

from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
OK, WARN, FAIL = "✅", "⚠️ ", "❌"


def run(cmd: list[str]) -> tuple[int, str]:
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=60, check=False)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return 1, str(exc)
    return out.returncode, (out.stdout or out.stderr).strip()


def check_python() -> tuple[str, str]:
    v = sys.version_info
    text = f"Python {v.major}.{v.minor}.{v.micro} runs this script"
    if v < (3, 9):
        return FAIL, text + " (need 3.9+ for the tools; the labs use 3.12 through uv)"
    return OK, text


def check_tool(name: str, required: bool, hint: str) -> tuple[str, str]:
    path = shutil.which(name)
    if not path:
        return (FAIL if required else WARN), f"{name} not found. {hint}"
    code, out = run([name, "--version"])
    return OK, f"{name}: {out.splitlines()[0] if code == 0 and out else path}"


def check_uv_python() -> tuple[str, str]:
    if not shutil.which("uv"):
        return WARN, "Python 3.12 for labs: install uv first"
    code, out = run(["uv", "python", "find", "3.12"])
    if code == 0:
        return OK, f"Python 3.12 for labs: {out}"
    return WARN, "Python 3.12 not installed yet; `uv sync` will download it (or run `uv python install 3.12`)"


def check_datasets() -> tuple[str, str]:
    gen = REPO / "datasets" / "generate.py"
    if not gen.is_file():
        return FAIL, "datasets/generate.py is missing: is this the course repository?"
    code, out = run([sys.executable, str(gen), "--check"])
    if code == 0:
        return OK, f"Datasets: {out}"
    return FAIL, "Datasets are missing or modified. Run: python3 datasets/generate.py"


def check_ollama() -> tuple[str, str]:
    host = os.environ.get("OLLAMA_HOST", "http://localhost:11434").rstrip("/")
    if not shutil.which("ollama"):
        return WARN, "Ollama not installed (optional, for local models): https://ollama.com"
    try:
        with urllib.request.urlopen(f"{host}/api/tags", timeout=2) as resp:  # noqa: S310 (local URL)
            models = [m.get("name", "?") for m in json.loads(resp.read()).get("models", [])]
    except Exception:  # noqa: BLE001
        return WARN, f"Ollama installed but not running at {host} (start it with `ollama serve`)"
    if not models:
        return WARN, "Ollama is running but has no models. Try: ollama pull llama3.2:1b"
    return OK, f"Ollama running with {len(models)} model(s): {', '.join(models[:5])}"


def check_openai_env() -> tuple[str, str]:
    present = [k for k in ("OPENAI_BASE_URL", "OPENAI_API_KEY", "OPENAI_MODEL") if os.environ.get(k)]
    if not present:
        return WARN, "No OpenAI-compatible endpoint configured (optional)"
    missing = {"OPENAI_MODEL"} - set(present)
    detail = f"set: {', '.join(present)}" + (f"; missing: {', '.join(missing)}" if missing else "")
    return (WARN if missing else OK), f"OpenAI-compatible endpoint {detail}"


def check_disk() -> tuple[str, str]:
    free_gb = shutil.disk_usage(REPO).free / 1e9
    status = OK if free_gb >= 3 else WARN
    return status, f"Free disk space: {free_gb:.1f} GB (about 3 GB is enough for all lab environments)"


def main() -> int:
    print(f"AI Engineering Fundamentals · setup check on {platform.system()} {platform.machine()}\n")
    checks = [
        ("Required", check_python()),
        ("Required", check_tool("git", True, "Install Git: https://git-scm.com")),
        ("Required", check_tool("uv", True, "Install uv: https://docs.astral.sh/uv/")),
        ("Required", check_uv_python()),
        ("Required", check_datasets()),
        ("Optional", check_ollama()),
        ("Optional", check_openai_env()),
        ("Optional", check_tool("node", False, "Only needed to validate diagrams (contributors).")),
        ("Info", check_disk()),
    ]
    failed = False
    for kind, (status, text) in checks:
        print(f"{status} [{kind}] {text}")
        failed |= status == FAIL
    print()
    if failed:
        print("Fix the ❌ items above, then run this script again.")
        return 1
    print("You are ready. Start with: cd modules/01-how-llms-are-built/lab && uv sync && uv run pytest")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
