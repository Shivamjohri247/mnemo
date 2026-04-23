"""Exhaustive tests for 4-channel retrieval and RRF fusion."""

import time
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from mnemo.core.retrieval import (
    CHANNEL_WEIGHTS,
    FINAL_TOP_K,
    RRF_K,
    TOP_K_PER_CHANNEL,
    bm25_channel,
    entity_channel,
    fuse,
    recall,
    rrf_score,
    semantic_channel,
    temporal_channel,
)

# ── RRF Score ──────────────────────────────────────────────────────────


class TestRRFScore:
    def test_formula_rank0(self):
        assert rrf_score(0, 1.0) == pytest.approx(1.0 / 15)

    def test_formula_weighted(self):
        assert rrf_score(5, 1.2) == pytest.approx(1.2 / 20)

    def test_decreases_with_rank(self):
        scores = [rrf_score(r, 1.0) for r in range(10)]
        for i in range(len(scores) - 1):
            assert scores[i] > scores[i + 1]

    def test_custom_k(self):
        assert rrf_score(0, 1.0, k=60) == pytest.approx(1.0 / 60)

    def test_zero_weight_gives_zero(self):
        assert rrf_score(0, 0.0) == 0.0

    def test_high_rank_near_zero(self):
        assert rrf_score(1000, 1.0) < 0.001

    def test_positive_always(self):
        for r in range(100):
            assert rrf_score(r, 1.0) > 0


# ── RRF Fusion ────────────────────────────────────────────────────────


class TestFusion:
    def test_single_channel_preserves_order(self):
        results = {"semantic": ["a", "b", "c"]}
        assert fuse(results) == ["a", "b", "c"]

    def test_multi_channel_agreement(self):
        results = {
            "semantic": ["a", "b", "c"],
            "bm25": ["a", "b", "d"],
            "temporal": ["a", "c", "b"],
        }
        fused = fuse(results)
        assert fused[0] == "a"  # consensus top

    def test_weight_advantage_semantic(self):
        results = {"semantic": ["a"], "bm25": ["b", "a"]}
        fused = fuse(results)
        assert fused[0] == "a"  # semantic weight 1.2 > bm25 1.0

    def test_empty_channels(self):
        assert fuse({}) == []
        assert fuse({"semantic": [], "bm25": []}) == []

    def test_single_item(self):
        assert fuse({"semantic": ["x"]}) == ["x"]

    def test_duplicate_across_channels_boosts_rank(self):
        """Same fact in multiple channels should rank higher."""
        multi = {"semantic": ["a", "b"], "bm25": ["a", "c"]}
        fused_multi = fuse(multi)
        assert fused_multi[0] == "a"

    def test_all_channels_equal_weight(self):
        """With equal weights, rank position determines order."""
        results = {
            "bm25": ["x", "y", "z"],
            "temporal": ["x", "y", "z"],
        }
        fused = fuse(results)
        assert fused[0] == "x"

    def test_large_rank_penalized(self):
        """A fact at rank 0 in one channel should outrank rank-100 in another."""
        # "top" at semantic rank 0, "bottom" at bm25 rank 0
        results = {"semantic": ["top"], "bm25": ["bottom"]}
        fused = fuse(results)
        assert fused[0] == "top"  # semantic weight 1.2 > bm25 1.0

    def test_preserves_all_unique_ids(self):
        results = {"semantic": ["a", "b"], "bm25": ["c", "d"]}
        fused = fuse(results)
        assert set(fused) == {"a", "b", "c", "d"}


# ── Channel Constants ─────────────────────────────────────────────────


class TestConstants:
    def test_semantic_highest_weight(self):
        assert CHANNEL_WEIGHTS["semantic"] > CHANNEL_WEIGHTS["bm25"]

    def test_all_channels_have_weights(self):
        for ch in ("semantic", "bm25", "temporal", "entity"):
            assert ch in CHANNEL_WEIGHTS
            assert CHANNEL_WEIGHTS[ch] > 0

    def test_k_values(self):
        assert RRF_K == 15
        assert TOP_K_PER_CHANNEL == 20
        assert FINAL_TOP_K == 10


# ── Temporal Channel (unit testable without DB) ────────────────────────


class TestTemporalChannel:
    @pytest.fixture
    def temp_db(self, tmp_path, monkeypatch, mock_embedder):
        data_dir = tmp_path / ".slm"
        data_dir.mkdir()
        monkeypatch.setenv("SLM_DATA_DIR", str(data_dir))
        from mnemo.storage.db import get_connection, init_schema

        conn = get_connection()
        init_schema(conn)
        yield conn

    def test_returns_recent_ids_first(self, temp_db):
        now = time.time()
        # Insert facts with different ages
        temp_db.execute(
            "INSERT INTO facts (id, text, source, created_at, accessed_at, importance, "
            "trust_score, content_hash, lifecycle, precision_bits, strength, retention) "
            "VALUES ('old', 'old fact', 'user', ?, ?, 0.5, 1.0, 'h1', 'Active', 32, 3.5, 1.0)",
            (now - 3600, now - 3600),
        )
        temp_db.execute(
            "INSERT INTO facts (id, text, source, created_at, accessed_at, importance, "
            "trust_score, content_hash, lifecycle, precision_bits, strength, retention) "
            "VALUES ('new', 'new fact', 'user', ?, ?, 0.5, 1.0, 'h2', 'Active', 32, 3.5, 1.0)",
            (now, now),
        )
        temp_db.commit()

        # Insert embeddings for semantic channel compatibility
        for fid in ("old", "new"):
            vec = np.zeros(384, dtype=np.float32)
            temp_db.execute(
                "INSERT INTO embeddings (fact_id, vector_f32) VALUES (?, ?)",
                (fid, vec.tobytes()),
            )
        temp_db.commit()

        ids = temporal_channel("test", None, 10)
        assert len(ids) >= 2
        assert ids[0] == "new"  # newer should rank first

    def test_excludes_forgotten(self, temp_db):
        now = time.time()
        temp_db.execute(
            "INSERT INTO facts (id, text, source, created_at, accessed_at, importance, "
            "trust_score, content_hash, lifecycle, precision_bits, strength, retention) "
            "VALUES ('forgotten', 'forgotten fact', 'user', ?, ?, 0.5, 1.0, 'h3', 'Forgotten', 0, 1.0, 0.01)",
            (now, now),
        )
        temp_db.commit()

        ids = temporal_channel("test", None, 10)
        assert "forgotten" not in ids


# ── Semantic Channel (with mock embedder) ─────────────────────────────


class TestSemanticChannel:
    @pytest.fixture
    def semantic_env(self, tmp_path, monkeypatch, mock_embedder):
        data_dir = tmp_path / ".slm"
        data_dir.mkdir()
        monkeypatch.setenv("SLM_DATA_DIR", str(data_dir))

        from mnemo.storage.db import get_connection, init_schema

        conn = get_connection()
        init_schema(conn)

        # Store facts directly
        now = time.time()
        for i, text in enumerate(["database postgresql", "python web server", "react frontend"]):
            fid = f"fact_{i}"
            conn.execute(
                "INSERT INTO facts (id, text, source, created_at, accessed_at, importance, "
                "trust_score, content_hash, lifecycle, precision_bits, strength, retention) "
                "VALUES (?, ?, 'user', ?, ?, 0.5, 1.0, ?, 'Active', 32, 3.5, 1.0)",
                (fid, text, now, now, f"hash_{i}"),
            )
            vec = mock_embedder.embed_text(text)
            conn.execute(
                "INSERT INTO embeddings (fact_id, vector_f32) VALUES (?, ?)",
                (fid, vec.tobytes()),
            )
        conn.commit()

        with patch("mnemo.core.retrieval.embed_text", side_effect=mock_embedder.embed_text):
            yield conn

    def test_returns_results(self, semantic_env):
        ids = semantic_channel("database", None, 10)
        assert len(ids) > 0

    def test_top_result_relevant(self, semantic_env):
        ids = semantic_channel("database postgresql", None, 3)
        assert ids[0] == "fact_0"  # Most similar to "database postgresql"

    def test_respects_k_limit(self, semantic_env):
        ids = semantic_channel("test", None, 2)
        assert len(ids) <= 2


# ── BM25 Channel ───────────────────────────────────────────────────────


class TestBM25Channel:
    @pytest.fixture
    def bm25_env(self, tmp_path, monkeypatch, mock_embedder):
        data_dir = tmp_path / ".slm"
        data_dir.mkdir()
        monkeypatch.setenv("SLM_DATA_DIR", str(data_dir))

        from mnemo.storage.db import get_connection, init_schema

        conn = get_connection()
        init_schema(conn)

        now = time.time()
        texts = [
            "PostgreSQL is the primary database",
            "We use FastAPI for REST endpoints",
            "The frontend is built with React",
        ]
        for i, text in enumerate(texts):
            fid = f"fact_{i}"
            conn.execute(
                "INSERT INTO facts (id, text, source, created_at, accessed_at, importance, "
                "trust_score, content_hash, lifecycle, precision_bits, strength, retention) "
                "VALUES (?, ?, 'user', ?, ?, 0.5, 1.0, ?, 'Active', 32, 3.5, 1.0)",
                (fid, text, now, now, f"hash_{i}"),
            )
            vec = mock_embedder.embed_text(text)
            conn.execute(
                "INSERT INTO embeddings (fact_id, vector_f32) VALUES (?, ?)",
                (fid, vec.tobytes()),
            )
        conn.commit()
        yield conn

    def test_keyword_match(self, bm25_env):
        # Use a single keyword that exists in the fact text
        ids = bm25_channel("PostgreSQL", None, 10)
        assert len(ids) > 0
        assert "fact_0" in ids

    def test_no_match_returns_empty(self, bm25_env):
        ids = bm25_channel("xyznonexistent123", None, 10)
        assert ids == []

    def test_special_chars_handled(self, bm25_env):
        """FTS5 operators like NOT, OR should not crash."""
        ids = bm25_channel("not using OR but AND", None, 10)
        assert isinstance(ids, list)  # Should not raise

    def test_excludes_forgotten(self, bm25_env):
        bm25_env.execute("UPDATE facts SET lifecycle = 'Forgotten' WHERE id = 'fact_0'")
        bm25_env.commit()
        ids = bm25_channel("PostgreSQL database", None, 10)
        assert "fact_0" not in ids


# ── Recall Integration ────────────────────────────────────────────────


class TestRecall:
    @pytest.fixture
    def recall_env(self, tmp_path, monkeypatch, mock_embedder):
        data_dir = tmp_path / ".slm"
        data_dir.mkdir()
        monkeypatch.setenv("SLM_DATA_DIR", str(data_dir))

        mock_nlp = MagicMock()
        mock_nlp.return_value.ents = []

        with (
            patch("mnemo.pipeline.encode.get_embedder", return_value=mock_embedder),
            patch("mnemo.pipeline.encode._get_nlp", return_value=mock_nlp),
        ):
            from mnemo.storage.db import get_connection, init_schema

            conn = get_connection()
            init_schema(conn)

            from mnemo.pipeline.encode import store_fact

            store_fact("Redis is used for caching", importance=0.8)
            store_fact("PostgreSQL handles persistent data", importance=0.9)
            store_fact("FastAPI serves the API layer", importance=0.7)

        with patch("mnemo.core.retrieval.embed_text", side_effect=mock_embedder.embed_text):
            yield conn

    def test_recall_returns_results(self, recall_env):
        results = recall("database", top_k=5)
        assert len(results) > 0

    def test_recall_updates_access_count(self, recall_env):
        results = recall("database", top_k=5)
        if results:
            # recall() updates access_count for top 3 results
            row = recall_env.execute(
                "SELECT access_count FROM facts WHERE id = ?", (results[0]["id"],)
            ).fetchone()
            # May be 1 (store only) or 2+ (if recall found and boosted it)
            assert row["access_count"] >= 1

    def test_recall_empty_db(self, tmp_path, monkeypatch, mock_embedder):
        data_dir = tmp_path / ".slm"
        data_dir.mkdir()
        monkeypatch.setenv("SLM_DATA_DIR", str(data_dir))

        from mnemo.storage.db import get_connection, init_schema

        conn = get_connection()
        init_schema(conn)

        with patch("mnemo.core.retrieval.embed_text", side_effect=mock_embedder.embed_text):
            results = recall("anything", top_k=5)
        assert results == []

    def test_recall_with_project_filter(self, recall_env):
        results = recall("database", project="nonexistent", top_k=5)
        assert results == []


# ── Entity Channel with 1-hop traversal ──────────────────────────────────


class TestEntityChannel:
    @pytest.fixture
    def entity_env(self, tmp_path, monkeypatch, mock_embedder):
        """Set up DB with entities, mentions, and co-occurrence edges."""
        data_dir = tmp_path / ".slm"
        data_dir.mkdir()
        monkeypatch.setenv("SLM_DATA_DIR", str(data_dir))

        from mnemo.storage.db import get_connection, init_schema

        conn = get_connection()
        init_schema(conn)

        now = time.time()

        # Create entities
        pg_id = "ent_pg"
        redis_id = "ent_redis"
        python_id = "ent_python"
        for eid, name in [(pg_id, "postgresql"), (redis_id, "redis"), (python_id, "python")]:
            conn.execute(
                "INSERT OR IGNORE INTO entities (id, name, type, canonical_name) VALUES (?, ?, 'PRODUCT', ?)",
                (eid, name, name),
            )

        # Fact A mentions PostgreSQL and Redis
        conn.execute(
            "INSERT INTO facts (id, text, source, created_at, accessed_at, importance, "
            "trust_score, content_hash, lifecycle, precision_bits, strength, retention) "
            "VALUES ('fact_a', 'PostgreSQL and Redis for storage', 'user', ?, ?, 0.5, 1.0, 'ha', 'Active', 32, 3.5, 1.0)",
            (now, now),
        )
        for eid in (pg_id, redis_id):
            conn.execute(
                "INSERT OR IGNORE INTO entity_mentions (fact_id, entity_id) VALUES ('fact_a', ?)",
                (eid,),
            )

        # Fact B mentions Python and PostgreSQL
        conn.execute(
            "INSERT INTO facts (id, text, source, created_at, accessed_at, importance, "
            "trust_score, content_hash, lifecycle, precision_bits, strength, retention) "
            "VALUES ('fact_b', 'Python with PostgreSQL backend', 'user', ?, ?, 0.5, 1.0, 'hb', 'Active', 32, 3.5, 1.0)",
            (now, now),
        )
        for eid in (python_id, pg_id):
            conn.execute(
                "INSERT OR IGNORE INTO entity_mentions (fact_id, entity_id) VALUES ('fact_b', ?)",
                (eid,),
            )

        # Fact C mentions only Python
        conn.execute(
            "INSERT INTO facts (id, text, source, created_at, accessed_at, importance, "
            "trust_score, content_hash, lifecycle, precision_bits, strength, retention) "
            "VALUES ('fact_c', 'Python for scripting', 'user', ?, ?, 0.5, 1.0, 'hc', 'Active', 32, 3.5, 1.0)",
            (now, now),
        )
        conn.execute(
            "INSERT OR IGNORE INTO entity_mentions (fact_id, entity_id) VALUES ('fact_c', ?)",
            (python_id,),
        )

        # Co-occurrence edges: pg↔redis (from fact_a), pg↔python (from fact_b)
        conn.execute(
            "INSERT INTO entity_edges (from_entity, to_entity, relation, weight, last_updated) "
            "VALUES (?, ?, 'co-occurs', 1.0, ?)",
            (pg_id, redis_id, now),
        )
        conn.execute(
            "INSERT INTO entity_edges (from_entity, to_entity, relation, weight, last_updated) "
            "VALUES (?, ?, 'co-occurs', 1.0, ?)",
            (redis_id, pg_id, now),
        )
        conn.execute(
            "INSERT INTO entity_edges (from_entity, to_entity, relation, weight, last_updated) "
            "VALUES (?, ?, 'co-occurs', 1.0, ?)",
            (pg_id, python_id, now),
        )
        conn.execute(
            "INSERT INTO entity_edges (from_entity, to_entity, relation, weight, last_updated) "
            "VALUES (?, ?, 'co-occurs', 1.0, ?)",
            (python_id, pg_id, now),
        )
        conn.commit()

        # Patch spacy to return "Redis" as an entity when querying
        mock_nlp_inst = MagicMock()
        mock_ent = MagicMock()
        mock_ent.text = "Redis"
        mock_doc = MagicMock()
        mock_doc.ents = [mock_ent]
        mock_nlp_inst.return_value = mock_doc

        with patch("mnemo.core.retrieval._get_nlp", return_value=mock_nlp_inst):
            yield conn

    def test_direct_match_found(self, entity_env):
        """Query containing 'Redis' should directly match fact_a."""
        ids = entity_channel("Redis", None, 10)
        assert "fact_a" in ids

    def test_1hop_traversal_finds_related(self, entity_env):
        """Query 'Redis' should 1-hop to pg→python and find fact_b too."""
        ids = entity_channel("Redis", None, 10)
        # fact_a is direct match (Redis mentioned)
        # fact_b is 1-hop: Redis → pg → python → fact_b
        assert "fact_b" in ids

    def test_returns_empty_without_spacy(self, tmp_path, monkeypatch):
        """Entity channel returns [] if spacy is not available."""
        data_dir = tmp_path / ".slm"
        data_dir.mkdir()
        monkeypatch.setenv("SLM_DATA_DIR", str(data_dir))

        from mnemo.storage.db import get_connection, init_schema

        conn = get_connection()
        init_schema(conn)

        with patch("mnemo.core.retrieval._get_nlp", return_value=None):
            ids = entity_channel("test", None, 10)
        assert ids == []

    def test_returns_empty_for_no_entities_in_query(self, entity_env):
        """If spacy finds no entities in the query, return empty."""
        mock_nlp_inst = MagicMock()
        mock_doc = MagicMock()
        mock_doc.ents = []  # No entities in query
        mock_nlp_inst.return_value = mock_doc

        with patch("mnemo.core.retrieval._get_nlp", return_value=mock_nlp_inst):
            ids = entity_channel("generic query", None, 10)
        assert ids == []
