"""End-to-end tests for Claude Code session hooks.

Tests the three hooks (session_start, post_tool_use, stop) at multiple levels:
  1. Hook script logic (unit tests for rate limiting, filtering)
  2. CLI commands with mock embedder (observe, session-start, save-session)
  3. Full subprocess execution (hook scripts via stdin JSON)
"""

import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# ── Shared fixtures ───────────────────────────────────────────────────


@pytest.fixture
def hook_env(tmp_path, monkeypatch, mock_embedder):
    """Isolated environment with mock embedder and pre-stored facts."""
    data_dir = tmp_path / ".slm"
    data_dir.mkdir()
    monkeypatch.setenv("SLM_DATA_DIR", str(data_dir))

    mock_nlp = MagicMock()
    mock_nlp.return_value.ents = []

    with (
        patch("mnemo.pipeline.encode.get_embedder", return_value=mock_embedder),
        patch("mnemo.pipeline.encode._get_nlp", return_value=mock_nlp),
    ):
        from mnemo.core.engine import MemoryEngine
        from mnemo.storage.db import get_connection, init_schema

        conn = get_connection()
        init_schema(conn)

        engine = MemoryEngine(project="hook_test", auto_scheduler=False)
        engine.remember("We use PostgreSQL for persistence", importance=0.9)
        engine.remember("FastAPI handles all REST endpoints", importance=0.8)
        engine.remember("Redis for session caching", importance=0.7)

    yield {
        "data_dir": data_dir,
        "project": "hook_test",
        "tmp_path": tmp_path,
    }


def _run_hook(
    script_path: str, stdin_data: dict, env: dict | None = None
) -> subprocess.CompletedProcess:
    """Run a hook script with JSON on stdin, return the result."""
    hook_env = os.environ.copy()
    if env:
        hook_env.update(env)
    return subprocess.run(
        [sys.executable, script_path],
        input=json.dumps(stdin_data),
        capture_output=True,
        text=True,
        timeout=15,
        env=hook_env,
    )


def _hook_path(name: str) -> str:
    """Get absolute path to a hook script."""
    return str(Path(__file__).parent.parent.parent / "install" / "hooks" / name)


# ── PostToolUse: should_observe unit tests ─────────────────────────────


class TestShouldObserve:
    """Unit tests for the rate limiting and tool filtering logic."""

    def test_write_tool_observed(self):
        from install.hooks.post_tool_use import should_observe

        # Clean up any existing lock file
        lock_file = os.path.join(
            tempfile.gettempdir(), "slm_" + "/tmp/test.py".replace("/", "_")[-50:] + ".lock"
        )
        if os.path.exists(lock_file):
            os.unlink(lock_file)
        assert should_observe("Write", "/tmp/test.py") is True
        # Cleanup
        if os.path.exists(lock_file):
            os.unlink(lock_file)

    def test_edit_tool_observed(self):
        from install.hooks.post_tool_use import should_observe

        lock_file = os.path.join(
            tempfile.gettempdir(), "slm_" + "/tmp/edit.py".replace("/", "_")[-50:] + ".lock"
        )
        if os.path.exists(lock_file):
            os.unlink(lock_file)
        assert should_observe("Edit", "/tmp/edit.py") is True
        if os.path.exists(lock_file):
            os.unlink(lock_file)

    def test_multiedit_observed(self):
        from install.hooks.post_tool_use import should_observe

        lock_file = os.path.join(
            tempfile.gettempdir(), "slm_" + "/tmp/multi.py".replace("/", "_")[-50:] + ".lock"
        )
        if os.path.exists(lock_file):
            os.unlink(lock_file)
        assert should_observe("MultiEdit", "/tmp/multi.py") is True
        if os.path.exists(lock_file):
            os.unlink(lock_file)

    def test_bash_ignored(self):
        from install.hooks.post_tool_use import should_observe

        assert should_observe("Bash", "/tmp/test.sh") is False

    def test_read_ignored(self):
        from install.hooks.post_tool_use import should_observe

        assert should_observe("Read", "/tmp/test.py") is False

    def test_grep_ignored(self):
        from install.hooks.post_tool_use import should_observe

        assert should_observe("Grep", "/tmp/test.py") is False

    def test_no_file_path_ignored(self):
        from install.hooks.post_tool_use import should_observe

        assert should_observe("Write", None) is False
        assert should_observe("Write", "") is False

    def test_rate_limiting_blocks_second_call(self):
        """Second call within 5 minutes for the same file should be blocked."""
        from install.hooks.post_tool_use import should_observe

        file_path = f"/tmp/rate_test_{time.time()}.py"
        lock_file = os.path.join(
            tempfile.gettempdir(),
            "slm_" + file_path.replace("/", "_")[-50:] + ".lock",
        )
        try:
            assert should_observe("Write", file_path) is True  # First call
            assert os.path.exists(lock_file)
            assert should_observe("Write", file_path) is False  # Second call blocked
        finally:
            if os.path.exists(lock_file):
                os.unlink(lock_file)

    def test_rate_limit_allows_after_expiry(self):
        """After rate limit period, observation should proceed again."""
        from install.hooks.post_tool_use import should_observe

        file_path = "/tmp/expiry_test_unique_12345.py"
        lock_file = os.path.join(
            tempfile.gettempdir(),
            "slm_" + file_path.replace("/", "_")[-50:] + ".lock",
        )
        try:
            # Create an expired lock file (6 minutes old)
            with open(lock_file, "w"):
                pass
            old_time = time.time() - 360  # 6 minutes ago
            os.utime(lock_file, (old_time, old_time))

            assert should_observe("Write", file_path) is True
        finally:
            if os.path.exists(lock_file):
                os.unlink(lock_file)

    def test_lock_file_created(self):
        """should_observe should create a lock file on first observation."""
        from install.hooks.post_tool_use import should_observe

        file_path = "/tmp/lock_create_test_999.py"
        lock_file = os.path.join(
            tempfile.gettempdir(),
            "slm_" + file_path.replace("/", "_")[-50:] + ".lock",
        )
        try:
            if os.path.exists(lock_file):
                os.unlink(lock_file)
            should_observe("Write", file_path)
            assert os.path.exists(lock_file)
        finally:
            if os.path.exists(lock_file):
                os.unlink(lock_file)


# ── CLI Commands (with mock embedder) ─────────────────────────────────


class TestObserveCommand:
    """Test the 'mnemo observe' CLI command in-process with mock embedder."""

    def test_observe_stores_fact(self, hook_env, mock_embedder):
        """observe command should store a low-importance observation."""
        mock_nlp = MagicMock()
        mock_nlp.return_value.ents = []
        with (
            patch("mnemo.pipeline.encode.get_embedder", return_value=mock_embedder),
            patch("mnemo.pipeline.encode._get_nlp", return_value=mock_nlp),
        ):
            from mnemo.cli.commands import observe

            observe(tool="Write", file="/src/api.py")

        from mnemo.storage.db import get_connection

        conn = get_connection()
        row = conn.execute(
            "SELECT * FROM facts WHERE source = 'hook' AND text LIKE '%api.py%'"
        ).fetchone()
        assert row is not None
        assert row["importance"] == pytest.approx(0.2)
        assert row["source"] == "hook"
        assert "Write" in row["text"]
        assert "/src/api.py" in row["text"]

    def test_observe_ignores_non_edit_tools(self, hook_env, mock_embedder):
        """observe should not store facts for Bash/Read tools."""
        mock_nlp = MagicMock()
        mock_nlp.return_value.ents = []
        with (
            patch("mnemo.pipeline.encode.get_embedder", return_value=mock_embedder),
            patch("mnemo.pipeline.encode._get_nlp", return_value=mock_nlp),
        ):
            from mnemo.cli.commands import observe

            observe(tool="Bash", file="/tmp/test.sh")

        from mnemo.storage.db import get_connection

        conn = get_connection()
        rows = conn.execute(
            "SELECT * FROM facts WHERE source = 'hook' AND text LIKE '%test.sh%'"
        ).fetchall()
        assert len(rows) == 0

    def test_observe_no_file(self, hook_env, mock_embedder):
        """observe with no file should not store anything."""
        mock_nlp = MagicMock()
        mock_nlp.return_value.ents = []
        with (
            patch("mnemo.pipeline.encode.get_embedder", return_value=mock_embedder),
            patch("mnemo.pipeline.encode._get_nlp", return_value=mock_nlp),
        ):
            from mnemo.cli.commands import observe

            observe(tool="Write", file="")

        from mnemo.storage.db import get_connection

        conn = get_connection()
        rows = conn.execute("SELECT * FROM facts WHERE source = 'hook'").fetchall()
        assert len(rows) == 0


class TestSessionStartCommand:
    """Test the 'mnemo session-start' CLI command."""

    def test_json_output(self, hook_env):
        """session-start --json should produce valid JSON with expected fields."""
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "mnemo",
                "session-start",
                "--project",
                hook_env["project"],
                "--json",
            ],
            capture_output=True,
            text=True,
            timeout=15,
            env={**os.environ, "SLM_DATA_DIR": str(hook_env["data_dir"])},
        )
        assert result.returncode == 0
        data = json.loads(result.stdout)
        assert "soft_prompts" in data
        assert "recent_memories" in data
        assert isinstance(data["recent_memories"], list)
        # Should have our pre-stored facts
        texts = [m["text"] for m in data["recent_memories"]]
        assert any("PostgreSQL" in t for t in texts)

    def test_recent_memories_limited_to_5(self, hook_env, mock_embedder):
        """session-start should return at most 5 recent memories."""
        mock_nlp = MagicMock()
        mock_nlp.return_value.ents = []
        with (
            patch("mnemo.pipeline.encode.get_embedder", return_value=mock_embedder),
            patch("mnemo.pipeline.encode._get_nlp", return_value=mock_nlp),
        ):
            from mnemo.core.engine import MemoryEngine

            engine = MemoryEngine(project="hook_test", auto_scheduler=False)
            for i in range(10):
                engine.remember(f"Extra fact {i}", importance=0.5)

        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "mnemo",
                "session-start",
                "--project",
                hook_env["project"],
                "--json",
            ],
            capture_output=True,
            text=True,
            timeout=15,
            env={**os.environ, "SLM_DATA_DIR": str(hook_env["data_dir"])},
        )
        data = json.loads(result.stdout)
        assert len(data["recent_memories"]) <= 5

    def test_in_process(self, hook_env, mock_embedder):
        """session-start should return soft prompts and recent memories."""
        import io
        from contextlib import redirect_stdout

        from mnemo.cli.commands import session_start

        f = io.StringIO()
        with redirect_stdout(f):
            session_start(project=hook_env["project"], json_output=True)
        data = json.loads(f.getvalue())
        assert "soft_prompts" in data
        assert len(data["recent_memories"]) >= 3


class TestSaveSessionCommand:
    """Test the 'mnemo save-session' CLI command."""

    def test_saves_session_record(self, hook_env):
        """save-session should create a session record in the DB."""
        result = subprocess.run(
            [sys.executable, "-m", "mnemo", "save-session", "--project", hook_env["project"]],
            capture_output=True,
            text=True,
            timeout=15,
            env={**os.environ, "SLM_DATA_DIR": str(hook_env["data_dir"])},
        )
        assert result.returncode == 0

        from mnemo.storage.db import get_connection

        conn = get_connection()
        row = conn.execute(
            "SELECT * FROM sessions WHERE project = ?",
            (hook_env["project"],),
        ).fetchone()
        assert row is not None
        assert row["project"] == hook_env["project"]

    def test_git_context_captured(self, hook_env):
        """save-session --with-git-context should capture git branch/commit."""
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "mnemo",
                "save-session",
                "--project",
                hook_env["project"],
                "--with-git-context",
            ],
            capture_output=True,
            text=True,
            timeout=15,
            env={**os.environ, "SLM_DATA_DIR": str(hook_env["data_dir"])},
        )
        assert result.returncode == 0

        from mnemo.storage.db import get_connection

        conn = get_connection()
        row = conn.execute(
            "SELECT * FROM sessions WHERE project = ? ORDER BY started_at DESC LIMIT 1",
            (hook_env["project"],),
        ).fetchone()
        assert row is not None


# ── Session Start Hook (subprocess) ────────────────────────────────────


class TestSessionStartHook:
    def test_outputs_memory_context(self, hook_env):
        """Hook subprocess should output recent memories."""
        result = _run_hook(
            _hook_path("session_start.py"),
            {"cwd": hook_env["project"]},
            env={"SLM_DATA_DIR": str(hook_env["data_dir"])},
        )
        assert result.returncode == 0
        assert "PostgreSQL" in result.stdout or "FastAPI" in result.stdout

    def test_outputs_recent_facts_header(self, hook_env):
        result = _run_hook(
            _hook_path("session_start.py"),
            {"cwd": hook_env["project"]},
            env={"SLM_DATA_DIR": str(hook_env["data_dir"])},
        )
        assert "Recent facts" in result.stdout

    def test_handles_empty_db(self, tmp_path):
        data_dir = tmp_path / ".slm"
        data_dir.mkdir()
        result = _run_hook(
            _hook_path("session_start.py"),
            {"cwd": "empty_project"},
            env={"SLM_DATA_DIR": str(data_dir)},
        )
        assert result.returncode == 0

    def test_handles_no_stdin(self, hook_env):
        result = subprocess.run(
            [sys.executable, _hook_path("session_start.py")],
            capture_output=True,
            text=True,
            timeout=15,
            env={**os.environ, "SLM_DATA_DIR": str(hook_env["data_dir"])},
        )
        assert result.returncode == 0

    def test_handles_malformed_stdin(self, hook_env):
        """Hook should not crash on malformed JSON."""
        result = subprocess.run(
            [sys.executable, _hook_path("session_start.py")],
            input="not valid json{{{",
            capture_output=True,
            text=True,
            timeout=15,
            env={**os.environ, "SLM_DATA_DIR": str(hook_env["data_dir"])},
        )
        assert result.returncode == 0


# ── PostToolUse Hook (subprocess) ──────────────────────────────────────


class TestPostToolUseHook:
    def test_does_not_crash(self, hook_env):
        """Hook should never crash, regardless of input."""
        result = _run_hook(
            _hook_path("post_tool_use.py"),
            {"tool_name": "Write", "tool_input": {"file_path": "/tmp/test.py"}},
            env={"SLM_DATA_DIR": str(hook_env["data_dir"])},
        )
        assert result.returncode == 0

    def test_ignores_non_edit_tools(self, hook_env):
        """Non-edit tools should be silently ignored."""
        result = _run_hook(
            _hook_path("post_tool_use.py"),
            {"tool_name": "Bash", "tool_input": {"command": "ls"}},
            env={"SLM_DATA_DIR": str(hook_env["data_dir"])},
        )
        assert result.returncode == 0

    def test_handles_no_stdin(self, hook_env):
        result = subprocess.run(
            [sys.executable, _hook_path("post_tool_use.py")],
            capture_output=True,
            text=True,
            timeout=15,
            env={**os.environ, "SLM_DATA_DIR": str(hook_env["data_dir"])},
        )
        assert result.returncode == 0

    def test_handles_malformed_stdin(self, hook_env):
        result = subprocess.run(
            [sys.executable, _hook_path("post_tool_use.py")],
            input="{{{bad json",
            capture_output=True,
            text=True,
            timeout=15,
            env={**os.environ, "SLM_DATA_DIR": str(hook_env["data_dir"])},
        )
        assert result.returncode == 0

    def test_no_file_path_does_not_crash(self, hook_env):
        result = _run_hook(
            _hook_path("post_tool_use.py"),
            {"tool_name": "Write", "tool_input": {}},
            env={"SLM_DATA_DIR": str(hook_env["data_dir"])},
        )
        assert result.returncode == 0


# ── Stop Hook (subprocess) ────────────────────────────────────────────


class TestStopHook:
    def test_saves_session(self, hook_env):
        """Stop hook should save a session record."""
        _run_hook(
            _hook_path("stop.py"),
            {"cwd": hook_env["project"]},
            env={"SLM_DATA_DIR": str(hook_env["data_dir"])},
        )
        time.sleep(1)

        from mnemo.storage.db import get_connection

        conn = get_connection()
        rows = conn.execute(
            "SELECT * FROM sessions WHERE project = ?",
            (hook_env["project"],),
        ).fetchall()
        assert len(rows) >= 1

    def test_never_blocks(self, hook_env):
        """Stop hook should return immediately."""
        start = time.perf_counter()
        result = _run_hook(
            _hook_path("stop.py"),
            {"cwd": hook_env["project"]},
            env={"SLM_DATA_DIR": str(hook_env["data_dir"])},
        )
        elapsed = time.perf_counter() - start
        assert result.returncode == 0
        assert elapsed < 2.0

    def test_handles_no_stdin(self, hook_env):
        result = subprocess.run(
            [sys.executable, _hook_path("stop.py")],
            capture_output=True,
            text=True,
            timeout=15,
            env={**os.environ, "SLM_DATA_DIR": str(hook_env["data_dir"])},
        )
        assert result.returncode == 0


# ── Hook Installation ─────────────────────────────────────────────────


class TestHookInstallation:
    def test_install_creates_files(self, tmp_path, monkeypatch):
        fake_home = tmp_path / "home"
        fake_home.mkdir()
        monkeypatch.setenv("HOME", str(fake_home))

        from install.postinstall import install_hooks

        install_hooks(scope="global")

        hook_dir = fake_home / ".claude" / "hooks"
        assert (hook_dir / "session_start.py").exists()
        assert (hook_dir / "post_tool_use.py").exists()
        assert (hook_dir / "stop.py").exists()

    def test_installed_hooks_executable(self, tmp_path, monkeypatch):
        fake_home = tmp_path / "home"
        fake_home.mkdir()
        monkeypatch.setenv("HOME", str(fake_home))

        from install.postinstall import install_hooks

        install_hooks(scope="global")

        hook_dir = fake_home / ".claude" / "hooks"
        for f in hook_dir.iterdir():
            assert os.access(f, os.X_OK)

    def test_local_scope(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        from install.postinstall import install_hooks

        install_hooks(scope="local")
        assert (tmp_path / ".claude" / "hooks" / "session_start.py").exists()

    def test_cli_install_hooks(self, tmp_path, monkeypatch):
        fake_home = tmp_path / "home"
        fake_home.mkdir()
        monkeypatch.setenv("HOME", str(fake_home))

        result = subprocess.run(
            [sys.executable, "-m", "mnemo", "install-hooks"],
            capture_output=True,
            text=True,
            timeout=15,
            cwd=str(Path(__file__).parent.parent.parent),
            env={**os.environ, "HOME": str(fake_home)},
        )
        assert result.returncode == 0
        assert "Installed" in result.stdout


# ── Full Lifecycle (in-process, with mock) ────────────────────────────


class TestFullHookLifecycle:
    def test_session_lifecycle(self, hook_env, mock_embedder):
        """Full lifecycle: start → observe → stop, all verified."""
        mock_nlp = MagicMock()
        mock_nlp.return_value.ents = []

        # 1. Session start
        with (
            patch("mnemo.pipeline.encode.get_embedder", return_value=mock_embedder),
            patch("mnemo.pipeline.encode._get_nlp", return_value=mock_nlp),
        ):
            import io
            from contextlib import redirect_stdout

            from mnemo.cli.commands import session_start

            f = io.StringIO()
            with redirect_stdout(f):
                session_start(project=hook_env["project"], json_output=True)
            data = json.loads(f.getvalue())
            assert len(data["recent_memories"]) >= 3

        # 2. Observe file edit
        with (
            patch("mnemo.pipeline.encode.get_embedder", return_value=mock_embedder),
            patch("mnemo.pipeline.encode._get_nlp", return_value=mock_nlp),
        ):
            from mnemo.cli.commands import observe

            observe(tool="Write", file="/src/api.py")

        # Verify observation stored
        from mnemo.storage.db import get_connection

        conn = get_connection()
        obs = conn.execute(
            "SELECT * FROM facts WHERE source = 'hook' AND text LIKE '%api.py%'"
        ).fetchall()
        assert len(obs) >= 1
        assert obs[0]["importance"] == pytest.approx(0.2)

        # 3. Save session
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "mnemo",
                "save-session",
                "--project",
                hook_env["project"],
                "--with-git-context",
            ],
            capture_output=True,
            text=True,
            timeout=15,
            env={**os.environ, "SLM_DATA_DIR": str(hook_env["data_dir"])},
        )
        assert result.returncode == 0

        sessions = conn.execute(
            "SELECT * FROM sessions WHERE project = ?",
            (hook_env["project"],),
        ).fetchall()
        assert len(sessions) >= 1

    def test_next_session_sees_prior_observations(self, hook_env, mock_embedder):
        """Observations from prior session appear in next session's context."""
        mock_nlp = MagicMock()
        mock_nlp.return_value.ents = []

        # Session A: observe a file edit
        with (
            patch("mnemo.pipeline.encode.get_embedder", return_value=mock_embedder),
            patch("mnemo.pipeline.encode._get_nlp", return_value=mock_nlp),
        ):
            from mnemo.cli.commands import observe

            observe(tool="Edit", file="/src/models.py")

        # Session B: session_start should surface the observation
        with (
            patch("mnemo.pipeline.encode.get_embedder", return_value=mock_embedder),
            patch("mnemo.pipeline.encode._get_nlp", return_value=mock_nlp),
        ):
            import io
            from contextlib import redirect_stdout

            from mnemo.cli.commands import session_start

            f = io.StringIO()
            with redirect_stdout(f):
                session_start(project=hook_env["project"], json_output=True)
            data = json.loads(f.getvalue())
            texts = [m["text"] for m in data["recent_memories"]]
            assert any("models.py" in t for t in texts)
