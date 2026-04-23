"""
Lifecycle-aware embedding compression.

Precision tiers (paper equation 8):
  Active  → 32-bit float (full fidelity)
  Warm    →  8-bit int   (384 bytes)
  Cold    →  4-bit int   (192 bytes)
  Archive →  2-bit int   (96 bytes)

FRQAD approximation: variance-inflated cosine similarity.
"""

import numpy as np

from mnemo.storage.db import get_connection, transaction

VECTOR_DIM = 384

QUANTIZATION_NOISE = {
    32: 0.000,
    8: 0.005,
    4: 0.050,
    2: 0.200,
}


def quantize_to_int8(vec: np.ndarray) -> bytes:
    """Scale float32 [-1,1] to int8 [-127,127]."""
    return (vec * 127).clip(-127, 127).astype(np.int8).tobytes()


def quantize_to_int4(vec: np.ndarray) -> bytes:
    """Pack two 4-bit values per byte. Values mapped to [0,15]."""
    dim = len(vec)
    q = ((vec + 1) * 7.5).clip(0, 15).astype(np.uint8)
    # Pad to even length if needed
    if dim % 2 != 0:
        q = np.append(q, 0)
    pairs = len(q) // 2
    packed = np.zeros(pairs, dtype=np.uint8)
    packed[:] = q[0::2] | (q[1::2] << 4)
    return packed.tobytes()


def quantize_to_int2(vec: np.ndarray) -> bytes:
    """Pack four 2-bit values per byte. Values mapped to [0,3]."""
    dim = len(vec)
    q = ((vec + 1) * 1.5).clip(0, 3).astype(np.uint8)
    # Pad to multiple of 4
    remainder = dim % 4
    if remainder != 0:
        q = np.append(q, np.zeros(4 - remainder, dtype=np.uint8))
    quads = len(q) // 4
    packed = np.zeros(quads, dtype=np.uint8)
    packed[:] = q[0::4] | (q[1::4] << 2) | (q[2::4] << 4) | (q[3::4] << 6)
    return packed.tobytes()


def dequantize_int8(data: bytes, dim: int = VECTOR_DIM) -> np.ndarray:
    return np.frombuffer(data, dtype=np.int8).astype(np.float32)[:dim] / 127.0


def dequantize_int4(data: bytes, dim: int = VECTOR_DIM) -> np.ndarray:
    packed = np.frombuffer(data, dtype=np.uint8)
    q = np.zeros(dim, dtype=np.uint8)
    q[0::2] = packed[: dim // 2] & 0x0F
    q[1::2] = (packed[: dim // 2] >> 4) & 0x0F
    return (q.astype(np.float32) / 7.5) - 1.0


def dequantize_int2(data: bytes, dim: int = VECTOR_DIM) -> np.ndarray:
    packed = np.frombuffer(data, dtype=np.uint8)
    q = np.zeros(dim, dtype=np.uint8)
    q[0::4] = packed[: dim // 4] & 0x03
    q[1::4] = (packed[: dim // 4] >> 2) & 0x03
    q[2::4] = (packed[: dim // 4] >> 4) & 0x03
    q[3::4] = (packed[: dim // 4] >> 6) & 0x03
    return (q.astype(np.float32) / 1.5) - 1.0


def frqad_score(
    v1: np.ndarray,
    v2: np.ndarray,
    bits1: int,
    bits2: int,
) -> float:
    """
    FRQAD approximation: variance-inflated cosine similarity.
    Lower precision → higher noise → lower score.
    """
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    if norm1 < 1e-9 or norm2 < 1e-9:
        return 0.0
    cosine = np.dot(v1, v2) / (norm1 * norm2)

    noise1 = QUANTIZATION_NOISE.get(bits1, 0.0)
    noise2 = QUANTIZATION_NOISE.get(bits2, 0.0)
    effective_variance = noise1 + noise2

    return float(cosine / (1.0 + effective_variance))


def apply_precision_tier(fact_id: str, target_bits: int):
    """Compress or promote a fact's embedding to the target precision tier."""
    conn = get_connection()
    emb_row = conn.execute(
        "SELECT vector_f32 FROM embeddings WHERE fact_id = ?", (fact_id,)
    ).fetchone()

    if not emb_row or not emb_row["vector_f32"]:
        return

    vec_f32 = np.frombuffer(emb_row["vector_f32"], dtype=np.float32)

    with transaction(conn):
        if target_bits <= 8:
            conn.execute(
                "UPDATE embeddings SET vector_i8 = ? WHERE fact_id = ?",
                (quantize_to_int8(vec_f32), fact_id),
            )
        if target_bits <= 4:
            conn.execute(
                "UPDATE embeddings SET vector_i4 = ? WHERE fact_id = ?",
                (quantize_to_int4(vec_f32), fact_id),
            )
        if target_bits <= 2:
            conn.execute(
                "UPDATE embeddings SET vector_i2 = ? WHERE fact_id = ?",
                (quantize_to_int2(vec_f32), fact_id),
            )
        if target_bits == 2:
            conn.execute(
                "UPDATE embeddings SET vector_f32 = NULL WHERE fact_id = ?",
                (fact_id,),
            )
        conn.execute(
            "UPDATE facts SET precision_bits = ? WHERE id = ?",
            (target_bits, fact_id),
        )
