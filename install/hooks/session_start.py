#!/usr/bin/env python3
"""Claude Code SessionStart hook. Retrieves memories and soft prompts. Fails silently."""

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

        result = subprocess.run(
            [sys.executable, "-m", "mnemo", "session-start", "--project", project, "--json"],
            capture_output=True,
            text=True,
            timeout=8,
        )
        if result.returncode == 0 and result.stdout.strip():
            data = json.loads(result.stdout)

            output_lines = []

            if data.get("soft_prompts"):
                output_lines.append("## Memory context")
                output_lines.append(data["soft_prompts"])

            if data.get("recent_memories"):
                output_lines.append("\n## Recent facts on record")
                for mem in data["recent_memories"][:5]:
                    output_lines.append(f"- {mem['text']}")

            if output_lines:
                print("\n".join(output_lines))

    except Exception:
        pass


if __name__ == "__main__":
    main()
