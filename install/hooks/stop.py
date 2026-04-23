#!/usr/bin/env python3
"""Stop hook. Saves session summary asynchronously. Never blocks."""

import json
import os
import subprocess
import sys


def main():
    try:
        hook_input = {}
        if not sys.stdin.isatty():
            hook_input = json.loads(sys.stdin.read())
        project = hook_input.get("cwd") or os.getcwd()

        subprocess.Popen(
            [
                sys.executable,
                "-m",
                "mnemo",
                "save-session",
                "--project",
                project,
                "--with-git-context",
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
    except Exception:
        pass


if __name__ == "__main__":
    main()
