#!/usr/bin/env python3
"""PostToolUse hook. Rate-limited fact observation. Fails silently."""

import json
import os
import subprocess
import sys
import tempfile
import time

RATE_LIMIT_SECONDS = 300


def should_observe(tool_name: str, file_path: str | None) -> bool:
    if tool_name not in ("Write", "Edit", "MultiEdit"):
        return False
    if not file_path:
        return False

    lock_file = os.path.join(
        tempfile.gettempdir(),
        "slm_" + file_path.replace("/", "_")[-50:] + ".lock",
    )
    if os.path.exists(lock_file) and time.time() - os.path.getmtime(lock_file) < RATE_LIMIT_SECONDS:
        return False
    open(lock_file, "w").close()
    return True


def main():
    try:
        hook_input = {}
        if not sys.stdin.isatty():
            hook_input = json.loads(sys.stdin.read())
        tool_name = hook_input.get("tool_name", "")
        tool_input = hook_input.get("tool_input", {})
        file_path = tool_input.get("file_path") or tool_input.get("path")

        if not should_observe(tool_name, file_path):
            return

        subprocess.Popen(
            [
                sys.executable,
                "-m",
                "mnemo",
                "observe",
                "--tool",
                tool_name,
                "--file",
                file_path or "",
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except Exception:
        pass


if __name__ == "__main__":
    main()
