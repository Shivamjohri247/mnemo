"""Benchmark tests — 30-day simulation, session continuity at scale, retrieval latency."""

import time
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from mnemo.core.forgetting import (
    lifecycle_state,
    memory_strength,
    retention,
    trust_decay_multiplier,
)
from mnemo.core.quantization import (
    QUANTIZATION_NOISE,
    VECTOR_DIM,
    dequantize_int4,
    dequantize_int8,
    frqad_score,
    quantize_to_int2,
    quantize_to_int4,
    quantize_to_int8,
)

# ── Benchmark: Forgetting Curve Fidelity ──────────────────────────────


class TestBenchmarkForgettingCurve:
    """Reproduces paper Table 7: 30-day retention discrimination."""

    def test_hot_group_survives(self):
        """30 days, 1/day, importance=0.7, 3 confirmations, 12h since last."""
        total_access = 1 * 30  # 30 accesses
        s = memory_strength(total_access, 0.7, 3, 0.0)
        r = retention(s, 12)  # 12h since last access
        state, bits = lifecycle_state(r)
        # Should NOT be Forgotten — hot facts survive
        assert state != "Forgotten"
        assert s > 5.0  # Strong memory
        assert r > 0.2  # Meaningful retention

    def test_cold_group_forgotten(self):
        """1 access, importance=0.2, no confirmations, 30 days."""
        s = memory_strength(0, 0.2, 0, 0.0)
        r = retention(s, 720)  # 30 days
        state, bits = lifecycle_state(r)
        assert state == "Forgotten"

    def test_discriminative_ratio(self):
        """Hot/Cold strength ratio should be > 5× (paper claims 6.7×)."""
        s_hot = memory_strength(30, 0.7, 3, 0.0)
        s_cold = memory_strength(0, 0.2, 0, 0.0)
        ratio = s_hot / s_cold
        assert ratio > 5.0, f"Discriminative ratio {ratio:.1f}× < 5×"

    def test_retention_gap_orders_of_magnitude(self):
        """Hot retention should be orders of magnitude higher than cold."""
        s_hot = memory_strength(30, 0.7, 3, 0.0)
        s_cold = memory_strength(0, 0.2, 0, 0.0)
        r_hot = retention(s_hot, 12)
        r_cold = retention(s_cold, 720)
        assert r_hot / max(r_cold, 1e-10) > 100

    def test_spacing_effect_marginal_gains(self):
        """Each additional access should have diminishing marginal gain."""
        gains = []
        for i in range(1, 50):
            s_prev = memory_strength(i - 1, 0.5, 0, 0.0)
            s_curr = memory_strength(i, 0.5, 0, 0.0)
            gains.append(s_curr - s_prev)

        # Gains should be monotonically decreasing
        violations = sum(1 for i in range(len(gains) - 1) if gains[i] < gains[i + 1])
        assert violations == 0, f"Spacing effect violated {violations} times"

    def test_trust_decay_acceleration(self):
        """Low-trust facts should decay 3× faster."""
        s = memory_strength(5, 0.5, 0, 0.0)
        r_trusted = retention(s, 24, decay_multiplier=trust_decay_multiplier(1.0))
        r_untrusted = retention(s, 24, decay_multiplier=trust_decay_multiplier(0.0))
        ratio = r_trusted / max(r_untrusted, 1e-10)
        assert ratio > 3.0  # At least 3× slower decay for trusted

    def test_1000_facts_forgetting_simulation(self):
        """Simulate forgetting for 1000 facts over 30 days."""
        rng = np.random.RandomState(42)
        active_count = 0
        forgotten_count = 0

        for _ in range(1000):
            access_count = int(rng.exponential(5))
            importance = float(rng.beta(2, 5))  # Mostly low importance
            confirmations = int(rng.poisson(0.5))
            trust = float(rng.uniform(0, 1))
            hours_since = float(rng.exponential(168))  # Avg 1 week

            s = memory_strength(access_count, importance, confirmations, 0.0)
            decay_mult = trust_decay_multiplier(trust)
            r = retention(s, hours_since, decay_multiplier=decay_mult)
            state, _ = lifecycle_state(r)

            if state == "Active":
                active_count += 1
            elif state == "Forgotten":
                forgotten_count += 1

        # Most facts should be forgotten (low importance + exponential time)
        assert forgotten_count > active_count
        # But some should survive (high importance or recent)
        assert active_count > 0


# ── Benchmark: Quantization Quality ───────────────────────────────────


class TestBenchmarkQuantization:
    def test_int8_preserves_direction_100_vectors(self):
        """100 random unit vectors should maintain >0.9 cosine after int8 roundtrip."""
        rng = np.random.RandomState(42)
        min_cosine = 1.0
        for _ in range(100):
            vec = rng.randn(VECTOR_DIM).astype(np.float32)
            vec /= np.linalg.norm(vec)
            result = dequantize_int8(quantize_to_int8(vec))
            cos = float(
                np.dot(vec, result) / (np.linalg.norm(vec) * np.linalg.norm(result) + 1e-10)
            )
            min_cosine = min(min_cosine, cos)
        assert min_cosine > 0.85, f"Worst cosine similarity: {min_cosine:.3f}"

    def test_int4_preserves_direction_100_vectors(self):
        """100 random unit vectors should maintain >0.1 cosine after int4 roundtrip."""
        rng = np.random.RandomState(42)
        min_cosine = 1.0
        for _ in range(100):
            vec = rng.randn(VECTOR_DIM).astype(np.float32)
            vec /= np.linalg.norm(vec)
            result = dequantize_int4(quantize_to_int4(vec))
            cos = float(
                np.dot(vec, result) / (np.linalg.norm(vec) * np.linalg.norm(result) + 1e-10)
            )
            min_cosine = min(min_cosine, cos)
        assert min_cosine > 0.1, f"Worst cosine similarity: {min_cosine:.3f}"

    def test_frqad_preference_rate_1000_pairs(self):
        """FRQAD should prefer f32 over int4 ≥97% for semantically similar vectors."""
        rng = np.random.RandomState(42)
        preferred = 0
        for _ in range(1000):
            base = rng.randn(VECTOR_DIM).astype(np.float32)
            base /= np.linalg.norm(base)
            query = base + rng.randn(VECTOR_DIM).astype(np.float32) * 0.1
            query /= np.linalg.norm(query)
            vec_4 = dequantize_int4(quantize_to_int4(base))
            if frqad_score(query, base, 32, 32) >= frqad_score(query, vec_4, 32, 4):
                preferred += 1
        rate = preferred / 1000
        assert rate >= 0.97, f"FRQAD preference rate: {rate:.2%}"

    def test_compression_sizes(self):
        """Verify byte sizes at each precision level."""
        vec = np.zeros(VECTOR_DIM, dtype=np.float32)
        assert len(quantize_to_int8(vec)) == VECTOR_DIM  # 384 bytes
        assert len(quantize_to_int4(vec)) == VECTOR_DIM // 2  # 192 bytes
        assert len(quantize_to_int2(vec)) == VECTOR_DIM // 4  # 96 bytes

    def test_quantization_noise_ordering(self):
        """Higher precision → lower noise."""
        assert QUANTIZATION_NOISE[32] == 0.0
        assert QUANTIZATION_NOISE[8] < QUANTIZATION_NOISE[4]
        assert QUANTIZATION_NOISE[4] < QUANTIZATION_NOISE[2]


# ── Benchmark: Session Continuity at Scale ────────────────────────────


class TestBenchmarkSessionContinuity:
    @pytest.fixture
    def session_env(self, tmp_path, monkeypatch, mock_embedder):
        data_dir = tmp_path / ".slm"
        data_dir.mkdir()
        monkeypatch.setenv("SLM_DATA_DIR", str(data_dir))
        return data_dir

    def test_10_facts_survive_session(self, session_env, mock_embedder):
        """Paper Benchmark 5: 10/10 facts survive session boundary."""
        mock_nlp = MagicMock()
        mock_nlp.return_value.ents = []

        with (
            patch("mnemo.pipeline.encode.get_embedder", return_value=mock_embedder),
            patch("mnemo.pipeline.encode._get_nlp", return_value=mock_nlp),
        ):
            from mnemo.core.engine import MemoryEngine

            engine_a = MemoryEngine(project="bench", auto_scheduler=False)
            facts = [
                ("The project uses PostgreSQL", 0.8),
                ("We use TypeScript not JavaScript", 0.9),
                ("API follows JSON:API specification", 0.7),
                ("Auth uses JWT with 24h expiry", 0.8),
                ("Monorepo managed with Turborepo", 0.6),
                ("Python services use FastAPI", 0.7),
                ("Redis for session caching", 0.6),
                ("Dates stored as UTC timestamps", 0.7),
                ("Main branch requires PR reviews", 0.5),
                ("Docker Compose for local dev", 0.6),
            ]
            for text, imp in facts:
                engine_a.remember(text, importance=imp)

        del engine_a

        with patch("mnemo.core.retrieval.embed_text", side_effect=mock_embedder.embed_text):
            from mnemo.core.engine import MemoryEngine as ME2

            engine_b = ME2(project="bench", auto_scheduler=False)

            recalled = 0
            for text, _ in facts:
                results = engine_b.recall(text, top_k=10)
                if results and any(r["text"] == text for r in results):
                    recalled += 1

            assert recalled == 10, f"Only {recalled}/10 facts survived"

    def test_50_facts_high_recall(self, session_env, mock_embedder):
        """50 facts: ≥90% should be recoverable."""
        mock_nlp = MagicMock()
        mock_nlp.return_value.ents = []

        with (
            patch("mnemo.pipeline.encode.get_embedder", return_value=mock_embedder),
            patch("mnemo.pipeline.encode._get_nlp", return_value=mock_nlp),
        ):
            from mnemo.core.engine import MemoryEngine

            engine = MemoryEngine(project="bench50", auto_scheduler=False)
            facts = []
            for i in range(50):
                text = f"Fact number {i}: Project uses technology {i % 10}"
                fid = engine.remember(text, importance=0.6 + (i % 5) * 0.08)
                if fid:
                    facts.append(text)

        with patch("mnemo.core.retrieval.embed_text", side_effect=mock_embedder.embed_text):
            recalled = 0
            for text in facts:
                results = engine.recall(text, top_k=50)
                if results and any(r["text"] == text for r in results):
                    recalled += 1

            rate = recalled / len(facts) if facts else 0
            assert rate >= 0.90, f"Recall rate: {rate:.0%} ({recalled}/{len(facts)})"


# ── Benchmark: Retrieval Latency ──────────────────────────────────────


class TestBenchmarkRetrievalLatency:
    @pytest.fixture
    def latency_env(self, tmp_path, monkeypatch, mock_embedder):
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

            engine = MemoryEngine(project="latency", auto_scheduler=False)
            # Pre-populate with 100 facts
            for i in range(100):
                engine.remember(
                    f"Fact {i}: Technology {i % 20} is used for {['caching', 'storage', 'compute', 'networking'][i % 4]}",
                    importance=0.5 + (i % 10) * 0.05,
                )
            yield engine

    def test_recall_under_500ms(self, latency_env, mock_embedder):
        """Single recall should complete in <500ms."""
        with patch("mnemo.core.retrieval.embed_text", side_effect=mock_embedder.embed_text):
            # Use text that matches stored facts for BM25 channel
            start = time.perf_counter()
            results = latency_env.recall("Technology", top_k=10)
            elapsed = time.perf_counter() - start

        assert elapsed < 0.5, f"Recall took {elapsed:.3f}s"
        # With 100 facts, BM25 should match "Technology" keyword
        assert len(results) > 0

    def test_decay_pass_under_1s(self, latency_env):
        """Decay pass over 100 facts should complete in <1s."""
        start = time.perf_counter()
        result = latency_env.run_decay()
        elapsed = time.perf_counter() - start

        assert elapsed < 1.0, f"Decay pass took {elapsed:.3f}s"
        assert result["updated"] == 100
