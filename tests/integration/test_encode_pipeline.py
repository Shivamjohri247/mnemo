"""Exhaustive integration tests for the encode pipeline and lifecycle scheduler."""

import time
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from mnemo.pipeline.encode import content_hash, normalize_text, store_fact
from mnemo.pipeline.lifecycle import garbage_collect, run_decay_pass
from mnemo.storage.db import init_schema


@pytest.fixture
def test_env(tmp_path, mock_embedder, monkeypatch):
    """Set up isolated test environment with mock embedder."""
    data_dir = tmp_path / ".slm"
    data_dir.mkdir()
    monkeypatch.setenv("SLM_DATA_DIR", str(data_dir))

    mock_nlp = MagicMock()
    mock_nlp.return_value.ents = []

    with (
        patch("mnemo.pipeline.encode.get_embedder", return_value=mock_embedder),
        patch("mnemo.pipeline.encode._get_nlp", return_value=mock_nlp),
    ):
        from mnemo.storage.db import get_connection

        conn = get_connection()
        init_schema(conn)
        yield conn


# ── normalize_text / content_hash ─────────────────────────────────────


class TestNormalizeHash:
    @pytest.mark.parametrize(
        "input_text,expected",
        [
            ("Hello  World", "hello world"),
            ("  Spaces  ", "spaces"),
            ("UPPER CASE", "upper case"),
            ("Mixed\tCase\nLines", "mixed case lines"),
        ],
    )
    def test_normalize(self, input_text, expected):
        assert normalize_text(input_text) == expected

    def test_content_hash_deterministic(self):
        h1 = content_hash("test string")
        h2 = content_hash("test string")
        assert h1 == h2

    def test_content_hash_normalizes(self):
        h1 = content_hash("Hello  World")
        h2 = content_hash("hello world")
        assert h1 == h2

    def test_content_hash_different_for_different_text(self):
        h1 = content_hash("first")
        h2 = content_hash("second")
        assert h1 != h2

    def test_content_hash_is_sha256(self):
        h = content_hash("test")
        assert len(h) == 64  # SHA256 hex length
        assert all(c in "0123456789abcdef" for c in h)


# ── store_fact ────────────────────────────────────────────────────────


class TestStoreFact:
    def test_store_new_fact(self, test_env, mock_embedder):
        with (
            patch("mnemo.pipeline.encode.get_embedder", return_value=mock_embedder),
            patch("mnemo.pipeline.encode._get_nlp", return_value=MagicMock()) as mock_nlp,
        ):
            mock_nlp.return_value.ents = []
            fact_id = store_fact("Test fact", importance=0.7)
            assert fact_id is not None

            row = test_env.execute("SELECT * FROM facts WHERE id = ?", (fact_id,)).fetchone()
            assert row["text"] == "Test fact"
            assert row["importance"] == 0.7
            assert row["lifecycle"] == "Active"
            assert row["precision_bits"] == 32
            assert row["strength"] == pytest.approx(3.5)
            assert row["retention"] == pytest.approx(1.0)

    def test_store_duplicate_returns_none(self, test_env, mock_embedder):
        mock_nlp = MagicMock()
        mock_nlp.return_value.ents = []
        with (
            patch("mnemo.pipeline.encode.get_embedder", return_value=mock_embedder),
            patch("mnemo.pipeline.encode._get_nlp", return_value=mock_nlp),
        ):
            fid1 = store_fact("Same fact here", importance=0.5)
            fid2 = store_fact("Same fact here", importance=0.5)
            assert fid1 is not None
            assert fid2 is None

            row = test_env.execute(
                "SELECT access_count FROM facts WHERE id = ?", (fid1,)
            ).fetchone()
            assert row["access_count"] == 1

    def test_store_with_project(self, test_env, mock_embedder):
        mock_nlp = MagicMock()
        mock_nlp.return_value.ents = []
        with (
            patch("mnemo.pipeline.encode.get_embedder", return_value=mock_embedder),
            patch("mnemo.pipeline.encode._get_nlp", return_value=mock_nlp),
        ):
            fid = store_fact("Project fact", project="myapp", importance=0.6)
            assert fid is not None
            row = test_env.execute("SELECT project FROM facts WHERE id = ?", (fid,)).fetchone()
            assert row["project"] == "myapp"

    def test_store_creates_embedding(self, test_env, mock_embedder):
        mock_nlp = MagicMock()
        mock_nlp.return_value.ents = []
        with (
            patch("mnemo.pipeline.encode.get_embedder", return_value=mock_embedder),
            patch("mnemo.pipeline.encode._get_nlp", return_value=mock_nlp),
        ):
            fid = store_fact("Embedding test")
            row = test_env.execute(
                "SELECT vector_f32 FROM embeddings WHERE fact_id = ?", (fid,)
            ).fetchone()
            assert row is not None
            vec = np.frombuffer(row["vector_f32"], dtype=np.float32)
            assert vec.shape == (384,)

    def test_store_with_trust_score(self, test_env, mock_embedder):
        mock_nlp = MagicMock()
        mock_nlp.return_value.ents = []
        with (
            patch("mnemo.pipeline.encode.get_embedder", return_value=mock_embedder),
            patch("mnemo.pipeline.encode._get_nlp", return_value=mock_nlp),
        ):
            fid = store_fact("Trusted fact", trust_score=0.5)
            row = test_env.execute("SELECT trust_score FROM facts WHERE id = ?", (fid,)).fetchone()
            assert row["trust_score"] == 0.5

    def test_store_with_source(self, test_env, mock_embedder):
        mock_nlp = MagicMock()
        mock_nlp.return_value.ents = []
        with (
            patch("mnemo.pipeline.encode.get_embedder", return_value=mock_embedder),
            patch("mnemo.pipeline.encode._get_nlp", return_value=mock_nlp),
        ):
            fid = store_fact("Agent fact", source="agent")
            row = test_env.execute("SELECT source FROM facts WHERE id = ?", (fid,)).fetchone()
            assert row["source"] == "agent"

    def test_case_insensitive_dedup(self, test_env, mock_embedder):
        mock_nlp = MagicMock()
        mock_nlp.return_value.ents = []
        with (
            patch("mnemo.pipeline.encode.get_embedder", return_value=mock_embedder),
            patch("mnemo.pipeline.encode._get_nlp", return_value=mock_nlp),
        ):
            fid1 = store_fact("Hello World")
            fid2 = store_fact("hello world")
            assert fid1 is not None
            assert fid2 is None  # Normalized to same text

    def test_fts_indexed(self, test_env, mock_embedder):
        """Fact text should be searchable via FTS5."""
        mock_nlp = MagicMock()
        mock_nlp.return_value.ents = []
        with (
            patch("mnemo.pipeline.encode.get_embedder", return_value=mock_embedder),
            patch("mnemo.pipeline.encode._get_nlp", return_value=mock_nlp),
        ):
            store_fact("PostgreSQL database system")
            rows = test_env.execute(
                "SELECT * FROM facts_fts WHERE facts_fts MATCH '\"PostgreSQL\"'"
            ).fetchall()
            assert len(rows) >= 1


# ── Entity extraction ─────────────────────────────────────────────────


class TestEntityExtraction:
    def test_entities_stored(self, test_env, mock_embedder):
        """When NLP returns entities, they should be stored."""

        mock_entity = MagicMock()
        mock_entity.text = "PostgreSQL"
        mock_entity.label_ = "PRODUCT"

        mock_nlp = MagicMock()
        mock_doc = MagicMock()
        mock_doc.ents = [mock_entity]
        mock_nlp.return_value = mock_doc

        with (
            patch("mnemo.pipeline.encode.get_embedder", return_value=mock_embedder),
            patch("mnemo.pipeline.encode._get_nlp", return_value=mock_nlp),
        ):
            fid = store_fact("We use PostgreSQL for data storage")

        # Check entity was stored
        rows = test_env.execute("SELECT * FROM entities").fetchall()
        assert len(rows) >= 1
        assert rows[0]["canonical_name"] == "postgresql"

        # Check entity mention
        mentions = test_env.execute(
            "SELECT * FROM entity_mentions WHERE fact_id = ?", (fid,)
        ).fetchall()
        assert len(mentions) >= 1


class TestEntityEdges:
    def test_co_occurrence_edges_created(self, test_env, mock_embedder):
        """Facts with 2+ entities should create bidirectional co-occurrence edges."""
        mock_pg = MagicMock(text="PostgreSQL", label_="PRODUCT")
        mock_redis = MagicMock(text="Redis", label_="PRODUCT")

        mock_nlp = MagicMock()
        mock_doc = MagicMock()
        mock_doc.ents = [mock_pg, mock_redis]
        mock_nlp.return_value = mock_doc

        with (
            patch("mnemo.pipeline.encode.get_embedder", return_value=mock_embedder),
            patch("mnemo.pipeline.encode._get_nlp", return_value=mock_nlp),
        ):
            store_fact("PostgreSQL and Redis are both used")

        edges = test_env.execute("SELECT * FROM entity_edges").fetchall()
        # 2 entities → 2 directed edges (A→B, B→A)
        assert len(edges) == 2
        weights = [e["weight"] for e in edges]
        assert all(w == 1.0 for w in weights)
        assert all(e["relation"] == "co-occurs" for e in edges)

    def test_co_occurrence_weight_increments(self, test_env, mock_embedder):
        """Repeated co-occurrence should increment edge weight."""
        mock_pg = MagicMock(text="PostgreSQL", label_="PRODUCT")
        mock_redis = MagicMock(text="Redis", label_="PRODUCT")

        mock_nlp = MagicMock()
        mock_doc = MagicMock()
        mock_doc.ents = [mock_pg, mock_redis]
        mock_nlp.return_value = mock_doc

        with (
            patch("mnemo.pipeline.encode.get_embedder", return_value=mock_embedder),
            patch("mnemo.pipeline.encode._get_nlp", return_value=mock_nlp),
        ):
            store_fact("PostgreSQL and Redis together first time")
            store_fact("PostgreSQL and Redis together second time")

        edges = test_env.execute("SELECT * FROM entity_edges").fetchall()
        assert len(edges) == 2
        assert all(e["weight"] == 2.0 for e in edges)

    def test_no_edges_for_single_entity(self, test_env, mock_embedder):
        """Facts with only 1 entity should not create any edges."""
        mock_entity = MagicMock(text="Python", label_="PRODUCT")

        mock_nlp = MagicMock()
        mock_doc = MagicMock()
        mock_doc.ents = [mock_entity]
        mock_nlp.return_value = mock_doc

        with (
            patch("mnemo.pipeline.encode.get_embedder", return_value=mock_embedder),
            patch("mnemo.pipeline.encode._get_nlp", return_value=mock_nlp),
        ):
            store_fact("Python is great")

        edges = test_env.execute("SELECT * FROM entity_edges").fetchall()
        assert len(edges) == 0

    def test_no_edges_for_no_entities(self, test_env, mock_embedder):
        """Facts with no entities should not create any edges."""
        mock_nlp = MagicMock()
        mock_doc = MagicMock()
        mock_doc.ents = []
        mock_nlp.return_value = mock_doc

        with (
            patch("mnemo.pipeline.encode.get_embedder", return_value=mock_embedder),
            patch("mnemo.pipeline.encode._get_nlp", return_value=mock_nlp),
        ):
            store_fact("No named entities here")

        edges = test_env.execute("SELECT * FROM entity_edges").fetchall()
        assert len(edges) == 0


# ── Lifecycle decay pass ──────────────────────────────────────────────


class TestLifecycleDecay:
    def test_decay_transitions_old_fact_to_forgotten(self, test_env, mock_embedder):
        mock_nlp = MagicMock()
        mock_nlp.return_value.ents = []
        with (
            patch("mnemo.pipeline.encode.get_embedder", return_value=mock_embedder),
            patch("mnemo.pipeline.encode._get_nlp", return_value=mock_nlp),
        ):
            fid = store_fact("Old fact", importance=0.1)

        old_time = time.time() - (30 * 24 * 3600)
        test_env.execute(
            "UPDATE facts SET accessed_at = ?, access_count = 0 WHERE id = ?",
            (old_time, fid),
        )
        test_env.commit()

        result = run_decay_pass()
        assert result["updated"] >= 1
        assert result["forgotten"] >= 1

        row = test_env.execute(
            "SELECT lifecycle, retention FROM facts WHERE id = ?", (fid,)
        ).fetchone()
        assert row["lifecycle"] == "Forgotten"
        assert row["retention"] < 0.05

    def test_hot_fact_stays_active(self, test_env, mock_embedder):
        mock_nlp = MagicMock()
        mock_nlp.return_value.ents = []
        with (
            patch("mnemo.pipeline.encode.get_embedder", return_value=mock_embedder),
            patch("mnemo.pipeline.encode._get_nlp", return_value=mock_nlp),
        ):
            fid = store_fact("Important fact", importance=0.9)

        test_env.execute(
            "UPDATE facts SET access_count = 50, confirmations = 5 WHERE id = ?",
            (fid,),
        )
        test_env.commit()

        run_decay_pass()
        row = test_env.execute("SELECT lifecycle FROM facts WHERE id = ?", (fid,)).fetchone()
        assert row["lifecycle"] == "Active"

    def test_decay_with_project_filter(self, test_env, mock_embedder):
        mock_nlp = MagicMock()
        mock_nlp.return_value.ents = []
        with (
            patch("mnemo.pipeline.encode.get_embedder", return_value=mock_embedder),
            patch("mnemo.pipeline.encode._get_nlp", return_value=mock_nlp),
        ):
            fid_a = store_fact("Project A fact", project="proj_a", importance=0.1)
            store_fact("Project B fact", project="proj_b", importance=0.9)

        old_time = time.time() - (30 * 24 * 3600)
        test_env.execute(
            "UPDATE facts SET accessed_at = ?, access_count = 0 WHERE id = ?",
            (old_time, fid_a),
        )
        test_env.commit()

        result = run_decay_pass(project="proj_a")
        assert result["updated"] >= 1

    def test_multiple_facts_decayed(self, test_env, mock_embedder):
        mock_nlp = MagicMock()
        mock_nlp.return_value.ents = []
        with (
            patch("mnemo.pipeline.encode.get_embedder", return_value=mock_embedder),
            patch("mnemo.pipeline.encode._get_nlp", return_value=mock_nlp),
        ):
            for i in range(5):
                store_fact(f"Fact {i}", importance=0.1)

        old_time = time.time() - (720 * 3600)  # 30 days
        test_env.execute(
            "UPDATE facts SET accessed_at = ?, access_count = 0",
            (old_time,),
        )
        test_env.commit()

        result = run_decay_pass()
        assert result["updated"] == 5
        assert result["forgotten"] == 5


# ── Garbage collection ────────────────────────────────────────────────


class TestGarbageCollect:
    def test_deletes_old_forgotten(self, test_env, mock_embedder):
        mock_nlp = MagicMock()
        mock_nlp.return_value.ents = []
        with (
            patch("mnemo.pipeline.encode.get_embedder", return_value=mock_embedder),
            patch("mnemo.pipeline.encode._get_nlp", return_value=mock_nlp),
        ):
            fid = store_fact("To be GC'd", importance=0.1)

        # Age it and mark as forgotten
        old_time = time.time() - (8 * 24 * 3600)  # 8 days
        test_env.execute(
            "UPDATE facts SET lifecycle = 'Forgotten', accessed_at = ? WHERE id = ?",
            ("Forgotten", fid),
        )
        test_env.execute(
            "UPDATE facts SET accessed_at = ? WHERE id = ?",
            (old_time, fid),
        )
        test_env.commit()

        garbage_collect()

        row = test_env.execute("SELECT * FROM facts WHERE id = ?", (fid,)).fetchone()
        assert row is None

    def test_preserves_recent_forgotten(self, test_env, mock_embedder):
        mock_nlp = MagicMock()
        mock_nlp.return_value.ents = []
        with (
            patch("mnemo.pipeline.encode.get_embedder", return_value=mock_embedder),
            patch("mnemo.pipeline.encode._get_nlp", return_value=mock_nlp),
        ):
            fid = store_fact("Recently forgotten")

        test_env.execute("UPDATE facts SET lifecycle = 'Forgotten' WHERE id = ?", (fid,))
        test_env.commit()

        garbage_collect()

        row = test_env.execute("SELECT * FROM facts WHERE id = ?", (fid,)).fetchone()
        assert row is not None  # Too recent to GC

    def test_preserves_active_regardless_of_age(self, test_env, mock_embedder):
        mock_nlp = MagicMock()
        mock_nlp.return_value.ents = []
        with (
            patch("mnemo.pipeline.encode.get_embedder", return_value=mock_embedder),
            patch("mnemo.pipeline.encode._get_nlp", return_value=mock_nlp),
        ):
            fid = store_fact("Active old fact")

        old_time = time.time() - (365 * 24 * 3600)  # 1 year
        test_env.execute(
            "UPDATE facts SET accessed_at = ? WHERE id = ?",
            (old_time, fid),
        )
        test_env.commit()

        garbage_collect()

        row = test_env.execute("SELECT * FROM facts WHERE id = ?", (fid,)).fetchone()
        assert row is not None  # Active, shouldn't be GC'd
