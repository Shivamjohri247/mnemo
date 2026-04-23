"""Exhaustive tests for MemoryEngine orchestrator."""

import time
from unittest.mock import MagicMock, patch

import pytest

from mnemo.core.engine import MemoryEngine


@pytest.fixture
def engine_env(tmp_path, monkeypatch, mock_embedder):
    """Isolated engine with mock embedder and NLP."""
    data_dir = tmp_path / ".slm"
    data_dir.mkdir()
    monkeypatch.setenv("SLM_DATA_DIR", str(data_dir))

    mock_nlp = MagicMock()
    mock_nlp.return_value.ents = []

    with (
        patch("mnemo.pipeline.encode.get_embedder", return_value=mock_embedder),
        patch("mnemo.pipeline.encode._get_nlp", return_value=mock_nlp),
    ):
        engine = MemoryEngine(project="test_engine", auto_scheduler=False)
        yield engine


# ── remember ──────────────────────────────────────────────────────────


class TestRemember:
    def test_store_and_retrieve(self, engine_env, mock_embedder):
        fid = engine_env.remember("Test fact about Python", importance=0.8)
        assert fid is not None

        row = engine_env.conn.execute("SELECT * FROM facts WHERE id = ?", (fid,)).fetchone()
        assert row["text"] == "Test fact about Python"
        assert row["importance"] == 0.8
        assert row["lifecycle"] == "Active"
        assert row["precision_bits"] == 32

    def test_duplicate_returns_none(self, engine_env, mock_embedder):
        fid1 = engine_env.remember("Duplicate fact")
        fid2 = engine_env.remember("Duplicate fact")
        assert fid1 is not None
        assert fid2 is None

    def test_stores_embedding(self, engine_env, mock_embedder):
        fid = engine_env.remember("Embedding test")
        row = engine_env.conn.execute(
            "SELECT vector_f32 FROM embeddings WHERE fact_id = ?", (fid,)
        ).fetchone()
        import numpy as np

        vec = np.frombuffer(row["vector_f32"], dtype=np.float32)
        assert vec.shape == (384,)

    def test_with_project(self, engine_env):
        fid = engine_env.remember("Project fact")
        row = engine_env.conn.execute("SELECT project FROM facts WHERE id = ?", (fid,)).fetchone()
        assert row["project"] == "test_engine"

    def test_importance_range(self, engine_env):
        fid_low = engine_env.remember("Low importance", importance=0.0)
        fid_high = engine_env.remember("High importance", importance=1.0)
        assert fid_low is not None
        assert fid_high is not None


# ── recall ─────────────────────────────────────────────────────────────


class TestRecall:
    def test_recall_finds_stored_fact(self, engine_env, mock_embedder):
        engine_env.remember("Redis is our caching layer")
        # Use full text so BM25 channel can match via phrase
        with patch("mnemo.core.retrieval.embed_text", side_effect=mock_embedder.embed_text):
            results = engine_env.recall("Redis is our caching layer")
        assert len(results) > 0

    def test_recall_empty_returns_empty(self, engine_env, mock_embedder):
        with patch("mnemo.core.retrieval.embed_text", side_effect=mock_embedder.embed_text):
            results = engine_env.recall("nonexistent query")
        assert results == []

    def test_recall_top_k(self, engine_env, mock_embedder):
        for i in range(15):
            engine_env.remember(f"Fact number {i} about topic {i % 3}")
        with patch("mnemo.core.retrieval.embed_text", side_effect=mock_embedder.embed_text):
            results = engine_env.recall("topic", top_k=5)
        assert len(results) <= 5


# ── forget ─────────────────────────────────────────────────────────────


class TestForget:
    def test_forget_sets_forgotten(self, engine_env):
        fid = engine_env.remember("To be forgotten")
        engine_env.forget(fid)
        row = engine_env.conn.execute("SELECT lifecycle FROM facts WHERE id = ?", (fid,)).fetchone()
        assert row["lifecycle"] == "Forgotten"

    def test_forget_nonexistent_no_error(self, engine_env):
        engine_env.forget("nonexistent-id")  # Should not raise


# ── stats ──────────────────────────────────────────────────────────────


class TestStats:
    def test_stats_returns_dict(self, engine_env):
        engine_env.remember("Stat fact 1", importance=0.7)
        engine_env.remember("Stat fact 2", importance=0.8)
        stats = engine_env.stats()
        assert isinstance(stats, dict)
        assert "Active" in stats
        assert stats["Active"]["count"] >= 2

    def test_stats_avg_retention(self, engine_env):
        engine_env.remember("Retention test", importance=0.9)
        stats = engine_env.stats()
        if "Active" in stats:
            assert 0 <= stats["Active"]["avg_retention"] <= 1.0

    def test_stats_empty_db(self, engine_env):
        stats = engine_env.stats()
        assert isinstance(stats, dict)


# ── list_memories ──────────────────────────────────────────────────────


class TestListMemories:
    def test_lists_active(self, engine_env):
        fid = engine_env.remember("Active memory")
        memories = engine_env.list_memories(lifecycle="Active")
        assert len(memories) >= 1
        assert any(m["id"] == fid for m in memories)

    def test_limit_respected(self, engine_env):
        for i in range(10):
            engine_env.remember(f"Memory {i}")
        memories = engine_env.list_memories(limit=3)
        assert len(memories) <= 3

    def test_empty_for_nonexistent_lifecycle(self, engine_env):
        memories = engine_env.list_memories(lifecycle="Forgotten")
        assert memories == []


# ── get_fact ───────────────────────────────────────────────────────────


class TestGetFact:
    def test_get_existing(self, engine_env):
        fid = engine_env.remember("Get this fact")
        fact = engine_env.get_fact(fid)
        assert fact is not None
        assert fact["text"] == "Get this fact"

    def test_get_nonexistent(self, engine_env):
        fact = engine_env.get_fact("nonexistent")
        assert fact is None


# ── set_importance ─────────────────────────────────────────────────────


class TestSetImportance:
    def test_updates_importance(self, engine_env):
        fid = engine_env.remember("Importance test")
        result = engine_env.set_importance(fid, 0.99)
        assert result is True
        row = engine_env.conn.execute(
            "SELECT importance FROM facts WHERE id = ?", (fid,)
        ).fetchone()
        assert row["importance"] == 0.99

    def test_nonexistent_returns_false(self, engine_env):
        result = engine_env.set_importance("nonexistent", 0.5)
        assert result is False


# ── confirm_memory ────────────────────────────────────────────────────


class TestConfirmMemory:
    def test_increments_confirmations(self, engine_env):
        fid = engine_env.remember("Confirm test")
        c1 = engine_env.confirm_memory(fid)
        c2 = engine_env.confirm_memory(fid)
        assert c2 == c1 + 1

    def test_start_at_zero(self, engine_env):
        fid = engine_env.remember("Confirm test")
        c = engine_env.confirm_memory(fid)
        assert c == 1  # First confirmation


# ── forgetting_curve ──────────────────────────────────────────────────


class TestForgettingCurve:
    def test_returns_curve(self, engine_env):
        fid = engine_env.remember("Curve test", importance=0.8)
        curve = engine_env.forgetting_curve(fid, days_ahead=7)
        assert len(curve) > 0
        # Retention should decrease over time
        for i in range(len(curve) - 1):
            assert curve[i]["retention"] >= curve[i + 1]["retention"]

    def test_nonexistent_returns_empty(self, engine_env):
        curve = engine_env.forgetting_curve("nonexistent")
        assert curve == []

    def test_curve_has_hours(self, engine_env):
        fid = engine_env.remember("Curve test")
        curve = engine_env.forgetting_curve(fid, days_ahead=1)
        assert all("hours" in pt for pt in curve)
        assert all("retention" in pt for pt in curve)


# ── export_memories ───────────────────────────────────────────────────


class TestExport:
    def test_exports_all(self, engine_env):
        engine_env.remember("Export fact 1")
        engine_env.remember("Export fact 2")
        exported = engine_env.export_memories()
        assert len(exported) >= 2

    def test_export_with_lifecycle_filter(self, engine_env):
        fid = engine_env.remember("Active export")
        engine_env.forget(fid)
        engine_env.remember("Still active")
        exported = engine_env.export_memories(lifecycle="Forgotten")
        assert any(r["id"] == fid for r in exported)


# ── run_decay ─────────────────────────────────────────────────────────


class TestRunDecay:
    def test_decay_updates_states(self, engine_env):
        fid = engine_env.remember("Decay test", importance=0.1)
        # Artificially age it
        old_time = time.time() - (30 * 24 * 3600)
        engine_env.conn.execute(
            "UPDATE facts SET accessed_at = ?, access_count = 0 WHERE id = ?",
            (old_time, fid),
        )
        engine_env.conn.commit()

        result = engine_env.run_decay()
        assert result["updated"] >= 1

        row = engine_env.conn.execute("SELECT lifecycle FROM facts WHERE id = ?", (fid,)).fetchone()
        assert row["lifecycle"] == "Forgotten"


class TestEngineShutdown:
    def test_shutdown_with_scheduler(self, tmp_path, monkeypatch, mock_embedder):
        """Engine with auto_scheduler should shut down cleanly."""
        data_dir = tmp_path / ".slm"
        data_dir.mkdir()
        monkeypatch.setenv("SLM_DATA_DIR", str(data_dir))

        mock_nlp = MagicMock()
        mock_nlp.return_value.ents = []

        with (
            patch("mnemo.pipeline.encode.get_embedder", return_value=mock_embedder),
            patch("mnemo.pipeline.encode._get_nlp", return_value=mock_nlp),
        ):
            engine = MemoryEngine(project="shutdown_test", auto_scheduler=True)
            assert engine._scheduler is not None
            assert engine._scheduler.running

            engine.shutdown()
            assert not engine._scheduler.running

    def test_shutdown_without_scheduler(self, engine_env):
        """Engine without auto_scheduler should handle shutdown gracefully."""
        engine_env.shutdown()  # Should not raise
