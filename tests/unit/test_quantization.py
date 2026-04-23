"""Exhaustive tests for quantization and FRQAD scoring."""

import numpy as np

from mnemo.core.quantization import (
    QUANTIZATION_NOISE,
    VECTOR_DIM,
    dequantize_int2,
    dequantize_int4,
    dequantize_int8,
    frqad_score,
    quantize_to_int2,
    quantize_to_int4,
    quantize_to_int8,
)

# ── Roundtrip fidelity ───────────────────────────────────────────────


class TestInt8Roundtrip:
    def test_direction_preserved(self):
        rng = np.random.RandomState(42)
        vec = _unit_vec(rng)
        result = dequantize_int8(quantize_to_int8(vec))
        assert _cosine(vec, result) > 0.9

    def test_zero_vector(self):
        vec = np.zeros(VECTOR_DIM, dtype=np.float32)
        result = dequantize_int8(quantize_to_int8(vec))
        assert np.allclose(result, 0, atol=0.02)

    def test_all_ones(self):
        vec = np.ones(VECTOR_DIM, dtype=np.float32)
        vec /= np.linalg.norm(vec)
        result = dequantize_int8(quantize_to_int8(vec))
        assert _cosine(vec, result) > 0.95

    def test_negative_values(self):
        rng = np.random.RandomState(99)
        vec = -_unit_vec(rng)
        result = dequantize_int8(quantize_to_int8(vec))
        assert _cosine(vec, result) > 0.9

    def test_byte_size(self):
        data = quantize_to_int8(np.zeros(VECTOR_DIM, dtype=np.float32))
        assert len(data) == VECTOR_DIM  # 384


class TestInt4Roundtrip:
    def test_direction_preserved(self):
        rng = np.random.RandomState(42)
        vec = _unit_vec(rng)
        result = dequantize_int4(quantize_to_int4(vec))
        assert _cosine(vec, result) > 0.3

    def test_byte_size(self):
        data = quantize_to_int4(np.zeros(VECTOR_DIM, dtype=np.float32))
        assert len(data) == VECTOR_DIM // 2  # 192

    def test_not_completely_random(self):
        """Dequantized vector should not be orthogonal to original."""
        rng = np.random.RandomState(42)
        vec = _unit_vec(rng)
        result = dequantize_int4(quantize_to_int4(vec))
        assert _cosine(vec, result) > 0.1  # Some correlation exists


class TestInt2Roundtrip:
    def test_byte_size(self):
        data = quantize_to_int2(np.zeros(VECTOR_DIM, dtype=np.float32))
        assert len(data) == VECTOR_DIM // 4  # 96

    def test_nonzero_norm(self):
        rng = np.random.RandomState(42)
        vec = _unit_vec(rng)
        result = dequantize_int2(quantize_to_int2(vec))
        assert np.linalg.norm(result) > 0


# ── Precision ordering ────────────────────────────────────────────────


class TestPrecisionOrdering:
    def test_int8_better_than_int4(self):
        rng = np.random.RandomState(42)
        vec = _unit_vec(rng)
        r8 = dequantize_int8(quantize_to_int8(vec))
        r4 = dequantize_int4(quantize_to_int4(vec))
        assert _cosine(vec, r8) > _cosine(vec, r4)

    def test_int4_better_than_int2(self):
        rng = np.random.RandomState(42)
        vec = _unit_vec(rng)
        r4 = dequantize_int4(quantize_to_int4(vec))
        r2 = dequantize_int2(quantize_to_int2(vec))
        assert _cosine(vec, r4) >= _cosine(vec, r2)


# ── FRQAD scoring ────────────────────────────────────────────────────


class TestFRQADScoring:
    def test_f32_preferred_over_4bit(self):
        """With semantically similar vectors, f32 should be preferred ≥97%."""
        rng = np.random.RandomState(42)
        preferred = 0
        for _ in range(1000):
            base = _unit_vec(rng)
            query = base + rng.randn(VECTOR_DIM).astype(np.float32) * 0.1
            query /= np.linalg.norm(query)
            vec_4 = dequantize_int4(quantize_to_int4(base))
            if frqad_score(query, base, 32, 32) >= frqad_score(query, vec_4, 32, 4):
                preferred += 1
        assert preferred / 1000 >= 0.97

    def test_noise_reduces_score(self):
        rng = np.random.RandomState(42)
        v1 = _unit_vec(rng)
        v2 = v1 + rng.randn(VECTOR_DIM).astype(np.float32) * 0.2
        v2 /= np.linalg.norm(v2)
        scores = [frqad_score(v1, v2, 32, bits) for bits in (32, 8, 4, 2)]
        for i in range(len(scores) - 1):
            assert scores[i] > scores[i + 1]

    def test_zero_vector_returns_zero(self):
        v = np.zeros(VECTOR_DIM, dtype=np.float32)
        assert frqad_score(v, v, 32, 32) == 0.0

    def test_identical_vectors_high_score(self):
        v = _unit_vec(np.random.RandomState(42))
        score = frqad_score(v, v, 32, 32)
        assert score > 0.99

    def test_opposite_vectors_negative(self):
        v = _unit_vec(np.random.RandomState(42))
        score = frqad_score(v, -v, 32, 32)
        assert score < -0.99

    def test_quantization_noise_values(self):
        assert QUANTIZATION_NOISE[32] == 0.0
        assert QUANTIZATION_NOISE[8] > 0
        assert QUANTIZATION_NOISE[4] > QUANTIZATION_NOISE[8]
        assert QUANTIZATION_NOISE[2] > QUANTIZATION_NOISE[4]

    def test_compression_ratios(self):
        """Verify progressive compression."""
        rng = np.random.RandomState(42)
        vec = _unit_vec(rng)
        sizes = {
            32: len(vec.tobytes()),
            8: len(quantize_to_int8(vec)),
            4: len(quantize_to_int4(vec)),
            2: len(quantize_to_int2(vec)),
        }
        assert sizes[8] < sizes[32]
        assert sizes[4] < sizes[8]
        assert sizes[2] < sizes[4]
        assert sizes[2] == VECTOR_DIM // 4


# ── Helpers ──────────────────────────────────────────────────────────


def _unit_vec(rng: np.random.RandomState) -> np.ndarray:
    vec = rng.randn(VECTOR_DIM).astype(np.float32)
    vec /= np.linalg.norm(vec)
    return vec


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10))
