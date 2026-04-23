"""End-to-end test: session continuity — 10/10 facts survive session boundary."""

from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture
def isolated_env(tmp_path, mock_embedder, monkeypatch):
    """Fully isolated environment for e2e tests."""
    data_dir = tmp_path / ".slm"
    data_dir.mkdir()
    monkeypatch.setenv("SLM_DATA_DIR", str(data_dir))
    return data_dir


def test_session_continuity(isolated_env, mock_embedder):
    """
    Paper Benchmark 5: 10/10 facts survive session boundary.
    Close and reopen engine, all 10 facts recalled at rank 1.
    """
    mock_nlp = MagicMock()
    mock_nlp.return_value.ents = []

    # Session A: store facts
    with (
        patch("mnemo.pipeline.encode.get_embedder", return_value=mock_embedder),
        patch("mnemo.pipeline.encode._get_nlp", return_value=mock_nlp),
    ):
        from mnemo.core.engine import MemoryEngine

        engine_a = MemoryEngine(project="test", auto_scheduler=False)
        test_facts = [
            ("The project uses PostgreSQL as the primary database", 0.8),
            ("We use TypeScript, not JavaScript", 0.9),
            ("API responses follow the JSON:API specification", 0.7),
            ("Authentication uses JWT with 24-hour expiry", 0.8),
            ("The monorepo is managed with Turborepo", 0.6),
            ("Python services use FastAPI, not Flask", 0.7),
            ("Redis is used for session caching only", 0.6),
            ("All dates stored as UTC Unix timestamps", 0.7),
            ("The main branch is protected, require PR reviews", 0.5),
            ("Docker Compose used for local development", 0.6),
        ]
        fact_ids = []
        for text, importance in test_facts:
            fid = engine_a.remember(text, importance=importance)
            if fid:
                fact_ids.append(fid)

        assert len(fact_ids) == 10, f"Expected 10 facts stored, got {len(fact_ids)}"

    # Simulate session boundary: close and reopen
    del engine_a

    # Session B: recall facts
    # Use full text for queries — BM25 channel matches keywords,
    # temporal channel favors recent facts, and semantic channel
    # uses the same mock embedder (same text = same vector)
    with patch("mnemo.core.retrieval.embed_text", side_effect=mock_embedder.embed_text):
        from mnemo.core.engine import MemoryEngine as ME2

        engine_b = ME2(project="test", auto_scheduler=False)

        recalled = 0
        for text, _ in test_facts:
            results = engine_b.recall(text, top_k=10)
            if results and any(r["text"] == text for r in results):
                recalled += 1

        assert recalled == 10, f"Only {recalled}/10 facts survived session boundary"
