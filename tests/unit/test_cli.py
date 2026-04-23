"""Tests for Typer CLI commands."""

from unittest.mock import MagicMock, patch

import pytest
from typer.testing import CliRunner


@pytest.fixture
def cli_env(tmp_path, monkeypatch, mock_embedder):
    data_dir = tmp_path / ".slm"
    data_dir.mkdir()
    monkeypatch.setenv("SLM_DATA_DIR", str(data_dir))

    mock_nlp = MagicMock()
    mock_nlp.return_value.ents = []

    with (
        patch("mnemo.pipeline.encode.get_embedder", return_value=mock_embedder),
        patch("mnemo.pipeline.encode._get_nlp", return_value=mock_nlp),
        patch("mnemo.core.retrieval.embed_text", side_effect=mock_embedder.embed_text),
    ):
        yield


runner = CliRunner()


class TestCLIInit:
    def test_init_command(self, cli_env):
        from mnemo.cli.commands import app

        result = runner.invoke(app, ["init"])
        assert result.exit_code == 0
        assert "Mnemo initialized" in result.output


class TestCLIRemember:
    def test_remember_new_fact(self, cli_env):
        from mnemo.cli.commands import app

        result = runner.invoke(app, ["remember", "Test fact from CLI"])
        assert result.exit_code == 0
        assert "Remembered" in result.output

    def test_remember_duplicate(self, cli_env):
        from mnemo.cli.commands import app

        runner.invoke(app, ["remember", "Duplicate fact"])
        result = runner.invoke(app, ["remember", "Duplicate fact"])
        assert result.exit_code == 0
        assert "duplicate" in result.output.lower()


class TestCLIRecall:
    def test_recall_with_results(self, cli_env):
        from mnemo.cli.commands import app

        runner.invoke(app, ["remember", "Python is our language"])
        result = runner.invoke(app, ["recall", "Python"])
        assert result.exit_code == 0

    def test_recall_no_results(self, cli_env):
        from mnemo.cli.commands import app

        result = runner.invoke(app, ["recall", "nonexistent topic xyz"])
        assert result.exit_code == 0
        assert "No memories found" in result.output


class TestCLIStats:
    def test_stats_shows_table(self, cli_env):
        from mnemo.cli.commands import app

        runner.invoke(app, ["remember", "Stats test fact"])
        result = runner.invoke(app, ["stats"])
        assert result.exit_code == 0
        assert "Active" in result.output


class TestCLIForget:
    def test_forget_command(self, cli_env):
        from mnemo.cli.commands import app

        remember_result = runner.invoke(app, ["remember", "Fact to forget"])
        assert "Remembered:" in remember_result.output
        fact_id = remember_result.output.split("Remembered: ")[1].strip()

        result = runner.invoke(app, ["forget", fact_id])
        assert result.exit_code == 0
        assert "Forgotten" in result.output


class TestCLIListMemories:
    def test_list_memories(self, cli_env):
        from mnemo.cli.commands import app

        runner.invoke(app, ["remember", "List test fact"])
        result = runner.invoke(app, ["list-memories"])
        assert result.exit_code == 0

    def test_list_empty(self, cli_env):
        from mnemo.cli.commands import app

        result = runner.invoke(app, ["list-memories", "--lifecycle", "Nonexistent"])
        assert result.exit_code == 0
        assert "No" in result.output


class TestCLISessionStart:
    def test_session_start_json(self, cli_env):
        from mnemo.cli.commands import app

        result = runner.invoke(app, ["session-start", "--json"])
        assert result.exit_code == 0


class TestCLIObserve:
    def test_observe_edit(self, cli_env):
        from mnemo.cli.commands import app

        result = runner.invoke(app, ["observe", "--tool", "Edit", "--file", "test.py"])
        assert result.exit_code == 0

    def test_observe_ignores_read(self, cli_env):
        from mnemo.cli.commands import app

        result = runner.invoke(app, ["observe", "--tool", "Read", "--file", "test.py"])
        assert result.exit_code == 0
