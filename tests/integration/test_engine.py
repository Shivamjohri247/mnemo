"""Integration tests for the full engine lifecycle."""

import time
from unittest.mock import MagicMock, patch

import pytest

from mnemo.core.engine import MemoryEngine


@pytest.fixture
def engine_env(tmp_path, monkeypatch, mock_embedder):
    data_dir = tmp_path / ".slm"
    data_dir.mkdir()
    monkeypatch.setenv("SLM_DATA_DIR", str(data_dir))

    mock_nlp = MagicMock()
    mock_nlp.return_value.ents = []

    with (
        patch("mnemo.pipeline.encode.get_embedder", return_value=mock_embedder),
        patch("mnemo.pipeline.encode._get_nlp", return_value=mock_nlp),
    ):
        engine = MemoryEngine(project="integration_test", auto_scheduler=False)
        yield engine


class TestRememberRecallRoundtrip:
    def test_remember_then_recall(self, engine_env, mock_embedder):
        engine_env.remember("FastAPI is our REST framework", importance=0.9)
        engine_env.remember("PostgreSQL handles persistence", importance=0.8)

        with patch("mnemo.core.retrieval.embed_text", side_effect=mock_embedder.embed_text):
            results = engine_env.recall("REST framework")
        assert len(results) > 0
        assert any("FastAPI" in r["text"] for r in results)

    def test_remember_forget_recall(self, engine_env, mock_embedder):
        fid = engine_env.remember("To be forgotten fact")
        engine_env.forget(fid)

        with patch("mnemo.core.retrieval.embed_text", side_effect=mock_embedder.embed_text):
            results = engine_env.recall("forgotten")
        # Forgotten facts should not appear in recall
        assert not any(r["id"] == fid for r in results)


class TestDecayLifecycle:
    def test_decay_updates_states(self, engine_env):
        fid = engine_env.remember("Low importance fact", importance=0.1)
        old_time = time.time() - (30 * 24 * 3600)
        engine_env.conn.execute(
            "UPDATE facts SET accessed_at = ?, access_count = 0 WHERE id = ?",
            (old_time, fid),
        )
        engine_env.conn.commit()

        result = engine_env.run_decay()
        assert result["updated"] >= 1

        fact = engine_env.get_fact(fid)
        assert fact["lifecycle"] == "Forgotten"

    def test_decay_preserves_hot_facts(self, engine_env):
        fid = engine_env.remember("Important fact", importance=0.9)
        engine_env.conn.execute(
            "UPDATE facts SET access_count = 100, confirmations = 5 WHERE id = ?",
            (fid,),
        )
        engine_env.conn.commit()

        engine_env.run_decay()
        fact = engine_env.get_fact(fid)
        assert fact["lifecycle"] == "Active"


class TestConsolidationFlow:
    def test_consolidate_and_prompts(self, engine_env):
        # Store enough facts for a pattern
        for i in range(6):
            engine_env.remember(
                f"We prefer Python for scripting task {i}",
                importance=0.8,
            )

        prompts = engine_env.get_soft_prompts()
        # May or may not generate prompts depending on consolidation threshold
        assert isinstance(prompts, str)


class TestStats:
    def test_stats_reflects_lifecycle(self, engine_env):
        fid = engine_env.remember("Active fact 1")
        engine_env.remember("Active fact 2")
        engine_env.forget(fid)

        stats = engine_env.stats()
        assert "Forgotten" in stats
        assert stats["Forgotten"]["count"] >= 1


class TestConfirmAndImportance:
    def test_confirm_boosts_memory(self, engine_env):
        fid = engine_env.remember("Confirmable fact")
        engine_env.confirm_memory(fid)
        engine_env.confirm_memory(fid)
        fact = engine_env.get_fact(fid)
        assert fact["confirmations"] == 2

    def test_importance_update(self, engine_env):
        fid = engine_env.remember("Adjustable fact")
        engine_env.set_importance(fid, 0.99)
        fact = engine_env.get_fact(fid)
        assert fact["importance"] == 0.99
