"""Auto-installs hooks into Claude Code's hook configuration."""

import os
import shutil
from pathlib import Path

HOOK_NAMES = {
    "SessionStart": "session_start.py",
    "PostToolUse": "post_tool_use.py",
    "Stop": "stop.py",
}


def install_hooks(scope: str = "global"):
    hook_dir = (
        Path.home() / ".claude" / "hooks" if scope == "global" else Path.cwd() / ".claude" / "hooks"
    )
    hook_dir.mkdir(parents=True, exist_ok=True)

    src_dir = Path(__file__).parent / "hooks"
    for hook_name, filename in HOOK_NAMES.items():
        src = src_dir / filename
        dst = hook_dir / filename
        shutil.copy2(src, dst)
        os.chmod(dst, 0o755)
        print(f"Installed {hook_name} hook -> {dst}")

    print("\nMnemo hooks installed.")
    print("Run 'mnemo status' to verify.")


if __name__ == "__main__":
    install_hooks()
