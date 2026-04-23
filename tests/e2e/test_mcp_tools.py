"""End-to-end tests for MCP tools."""

import time
from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture(autouse=True)
def _reset_mcp_engine():
    """Reset MCP module singleton between tests."""
    import mnemo.mcp.tools as tools_mod

    tools_mod._engine = None
    yield
    tools_mod._engine = None


@pytest.fixture
def mcp_env(tmp_path, mock_embedder, monkeypatch):
    """Environment with pre-stored facts for MCP tool tests."""
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

        engine = MemoryEngine(project="mcp_test", auto_scheduler=False)
        fid1 = engine.remember("We use FastAPI for all endpoints", importance=0.9)
        fid2 = engine.remember("PostgreSQL is our primary database", importance=0.8)
        fid3 = engine.remember("Never use global state", importance=0.7)

    return {
        "fact_ids": [fid1, fid2, fid3],
        "project": "mcp_test",
    }


class TestMCPTools:
    def test_slm_stats(self, mcp_env, mock_embedder):
        from mnemo.mcp.tools import slm_stats

        result = slm_stats(project=mcp_env["project"])
        assert isinstance(result, dict)
        assert "Active" in result
        assert result["Active"]["count"] >= 3

    def test_slm_list_memories(self, mcp_env, mock_embedder):
        from mnemo.mcp.tools import slm_list_memories

        result = slm_list_memories(project=mcp_env["project"])
        assert len(result) >= 3
        assert all(r["lifecycle"] == "Active" for r in result)

    def test_slm_set_importance(self, mcp_env, mock_embedder):
        from mnemo.mcp.tools import slm_set_importance

        fid = [f for f in mcp_env["fact_ids"] if f][0]
        result = slm_set_importance(fid, 0.99)
        assert result["success"] is True

    def test_slm_confirm_memory(self, mcp_env, mock_embedder):
        from mnemo.mcp.tools import slm_confirm_memory

        fid = [f for f in mcp_env["fact_ids"] if f][0]
        result = slm_confirm_memory(fid)
        assert result["confirmations"] >= 1

    def test_slm_export(self, mcp_env, mock_embedder):
        from mnemo.mcp.tools import slm_export

        result = slm_export(project=mcp_env["project"])
        assert len(result) >= 3
        assert all("text" in r for r in result)

    def test_slm_consolidate(self, mcp_env, mock_embedder):
        from mnemo.mcp.tools import slm_consolidate

        result = slm_consolidate(project=mcp_env["project"])
        assert "patterns_created" in result

    def test_slm_forget(self, mcp_env, mock_embedder):
        from mnemo.mcp.tools import slm_forget

        fid = [f for f in mcp_env["fact_ids"] if f][0]
        result = slm_forget(fid)
        assert result["success"] is True

    def test_slm_recall(self, mcp_env, mock_embedder):
        from mnemo.mcp.tools import slm_recall

        with patch("mnemo.core.retrieval.embed_text", side_effect=mock_embedder.embed_text):
            results = slm_recall("FastAPI endpoints", top_k=5, project=mcp_env["project"])
        assert isinstance(results, list)
        if results:
            assert "id" in results[0]
            assert "text" in results[0]
            assert "lifecycle" in results[0]
            assert "retention" in results[0]

    def test_slm_remember_new(self, mcp_env, mock_embedder):
        from mnemo.mcp.tools import slm_remember

        mock_nlp = MagicMock()
        mock_nlp.return_value.ents = []
        with (
            patch("mnemo.pipeline.encode.get_embedder", return_value=mock_embedder),
            patch("mnemo.pipeline.encode._get_nlp", return_value=mock_nlp),
        ):
            result = slm_remember("New MCP test fact", importance=0.6)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_slm_remember_duplicate(self, mcp_env, mock_embedder):
        from mnemo.mcp.tools import slm_remember

        mock_nlp = MagicMock()
        mock_nlp.return_value.ents = []
        with (
            patch("mnemo.pipeline.encode.get_embedder", return_value=mock_embedder),
            patch("mnemo.pipeline.encode._get_nlp", return_value=mock_nlp),
        ):
            slm_remember("Duplicate MCP fact", importance=0.5)
            result = slm_remember("Duplicate MCP fact", importance=0.5)
        assert result is None

    def test_slm_forget_hard_delete(self, mcp_env, mock_embedder):
        from mnemo.mcp.tools import slm_forget

        fid = [f for f in mcp_env["fact_ids"] if f][0]
        result = slm_forget(fid, hard_delete=True)
        assert result["success"] is True

        # Verify fact is actually deleted
        from mnemo.mcp.tools import get_engine

        engine = get_engine(project=mcp_env["project"])
        row = engine.conn.execute("SELECT * FROM facts WHERE id = ?", (fid,)).fetchone()
        assert row is None

    def test_slm_search_entities(self, mcp_env, mock_embedder):
        """slm_search_entities with no entities should return empty list."""
        from mnemo.mcp.tools import slm_search_entities

        result = slm_search_entities("nonexistent_entity")
        assert isinstance(result, list)

    def test_slm_forgetting_curve(self, mcp_env, mock_embedder):
        from mnemo.mcp.tools import slm_forgetting_curve

        fid = [f for f in mcp_env["fact_ids"] if f][0]
        result = slm_forgetting_curve(fid, days_ahead=7)
        assert isinstance(result, list)
        assert len(result) > 0
        assert all("hours" in pt for pt in result)
        assert all("retention" in pt for pt in result)

    def test_slm_forgetting_curve_nonexistent(self, mcp_env, mock_embedder):
        from mnemo.mcp.tools import slm_forgetting_curve

        result = slm_forgetting_curve("nonexistent-id", days_ahead=7)
        assert result == []

    def test_slm_list_sessions_empty(self, mcp_env, mock_embedder):
        from mnemo.mcp.tools import slm_list_sessions

        result = slm_list_sessions(limit=5)
        assert isinstance(result, list)

    def test_slm_list_sessions_with_session(self, mcp_env, mock_embedder):
        from mnemo.mcp.tools import get_engine

        engine = get_engine(project=mcp_env["project"])
        engine.conn.execute(
            "INSERT INTO sessions (id, project, started_at, ended_at) VALUES (?, ?, ?, ?)",
            ("sess_1", "mcp_test", time.time(), time.time()),
        )
        engine.conn.commit()

        from mnemo.mcp.tools import slm_list_sessions

        result = slm_list_sessions(project="mcp_test", limit=10)
        assert len(result) >= 1
        assert result[0]["project"] == "mcp_test"

    def test_slm_reset_learning(self, mcp_env, mock_embedder):
        from mnemo.mcp.tools import get_engine, slm_reset_learning

        engine = get_engine(project=mcp_env["project"])
        # Insert some patterns to delete
        engine.conn.execute(
            "INSERT INTO patterns (id, pattern_type, description, evidence_count, positive_rate, "
            "confidence, last_updated, source_fact_ids) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            ("p1", "test", "test pattern", 5, 0.5, 0.8, time.time(), "[]"),
        )
        engine.conn.commit()

        result = slm_reset_learning()
        assert "deleted_count" in result
        assert result["deleted_count"] >= 1

    def test_slm_daemon_status(self, mcp_env, mock_embedder):
        from mnemo.mcp.tools import slm_daemon_status

        result = slm_daemon_status()
        assert result["status"] in ("running", "stopped")
        assert result["port"] == 8767
