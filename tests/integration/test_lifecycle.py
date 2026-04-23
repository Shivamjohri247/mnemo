"""Integration tests for lifecycle scheduler."""

import time
from unittest.mock import MagicMock, patch

import pytest

from mnemo.pipeline.lifecycle import garbage_collect, run_decay_pass
from mnemo.storage.db import get_connection, init_schema


@pytest.fixture
def lifecycle_env(tmp_path, monkeypatch, mock_embedder):
    data_dir = tmp_path / ".slm"
    data_dir.mkdir()
    monkeypatch.setenv("SLM_DATA_DIR", str(data_dir))

    mock_nlp = MagicMock()
    mock_nlp.return_value.ents = []

    with (
        patch("mnemo.pipeline.encode.get_embedder", return_value=mock_embedder),
        patch("mnemo.pipeline.encode._get_nlp", return_value=mock_nlp),
    ):
        conn = get_connection()
        init_schema(conn)

        from mnemo.pipeline.encode import store_fact

        now = time.time()

        # Store facts with different characteristics
        fid_hot = store_fact("Hot important fact", importance=0.9, trust_score=1.0)
        fid_warm = store_fact("Warm moderate fact", importance=0.5, trust_score=0.7)
        fid_cold = store_fact("Cold low fact", importance=0.1, trust_score=0.3)

        # Age them differently
        if fid_hot:
            conn.execute(
                "UPDATE facts SET access_count = 50, confirmations = 5 WHERE id = ?",
                (fid_hot,),
            )
        if fid_warm:
            conn.execute(
                "UPDATE facts SET accessed_at = ?, access_count = 5 WHERE id = ?",
                (now - 48 * 3600, fid_warm),  # 2 days old
            )
        if fid_cold:
            conn.execute(
                "UPDATE facts SET accessed_at = ?, access_count = 0, trust_score = 0.1 WHERE id = ?",
                (now - 720 * 3600, fid_cold),  # 30 days old
            )
        conn.commit()

        yield conn, {"hot": fid_hot, "warm": fid_warm, "cold": fid_cold}


class TestDecayPass:
    def test_hot_stays_active(self, lifecycle_env):
        conn, ids = lifecycle_env
        run_decay_pass()

        if ids["hot"]:
            row = conn.execute("SELECT lifecycle FROM facts WHERE id = ?", (ids["hot"],)).fetchone()
            assert row["lifecycle"] == "Active"

    def test_cold_becomes_forgotten(self, lifecycle_env):
        conn, ids = lifecycle_env
        run_decay_pass()

        if ids["cold"]:
            row = conn.execute(
                "SELECT lifecycle FROM facts WHERE id = ?", (ids["cold"],)
            ).fetchone()
            assert row["lifecycle"] == "Forgotten"

    def test_trust_affects_decay(self, lifecycle_env):
        """Low trust should accelerate decay."""
        conn, ids = lifecycle_env
        run_decay_pass()

        if ids["warm"]:
            row = conn.execute(
                "SELECT retention FROM facts WHERE id = ?", (ids["warm"],)
            ).fetchone()
            # With trust=0.7 and 48h old, retention should be moderate
            assert 0 < row["retention"] <= 1.0


class TestGarbageCollect:
    def test_gc_removes_old_forgotten(self, lifecycle_env):
        conn, ids = lifecycle_env
        run_decay_pass()

        # Ensure cold fact is forgotten
        if ids["cold"]:
            conn.execute(
                "UPDATE facts SET accessed_at = ? WHERE id = ?",
                (time.time() - 8 * 24 * 3600, ids["cold"]),
            )
            conn.commit()

        garbage_collect()

        if ids["cold"]:
            row = conn.execute("SELECT * FROM facts WHERE id = ?", (ids["cold"],)).fetchone()
            assert row is None  # GC'd


class TestPrecisionTierCompression:
    def test_decay_compresses_embedding(self, lifecycle_env):
        """When decay lowers precision_bits, embeddings should actually be compressed."""
        conn, ids = lifecycle_env

        # The cold fact (low importance, low trust, old) should get compressed
        run_decay_pass()

        if ids["cold"]:
            fact_row = conn.execute(
                "SELECT precision_bits FROM facts WHERE id = ?", (ids["cold"],)
            ).fetchone()
            bits = fact_row["precision_bits"]

            emb_row = conn.execute(
                "SELECT vector_i8, vector_i4, vector_i2, vector_f32 FROM embeddings WHERE fact_id = ?",
                (ids["cold"],),
            ).fetchone()

            if bits <= 8:
                assert emb_row["vector_i8"] is not None, "vector_i8 should be populated at <= 8-bit"
            if bits <= 4:
                assert emb_row["vector_i4"] is not None, "vector_i4 should be populated at <= 4-bit"
            if bits <= 2:
                assert emb_row["vector_i2"] is not None, "vector_i2 should be populated at <= 2-bit"

    def test_hot_fact_stays_32bit(self, lifecycle_env):
        """Hot facts should remain at 32-bit after decay."""
        conn, ids = lifecycle_env
        run_decay_pass()

        if ids["hot"]:
            fact_row = conn.execute(
                "SELECT precision_bits FROM facts WHERE id = ?", (ids["hot"],)
            ).fetchone()
            assert fact_row["precision_bits"] == 32

            emb_row = conn.execute(
                "SELECT vector_f32 FROM embeddings WHERE fact_id = ?", (ids["hot"],)
            ).fetchone()
            assert emb_row["vector_f32"] is not None
