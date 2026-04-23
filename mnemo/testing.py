"""Shared test utilities for mnemo (mock embedder, helpers)."""

from unittest.mock import MagicMock

import numpy as np


def make_mock_embedder():
    """Create a mock embedder that returns deterministic 384-dim vectors."""

    def _embed_text(text: str) -> np.ndarray:
        rng = np.random.RandomState(hash(text) % (2**31))
        vec = rng.randn(384).astype(np.float32)
        vec /= np.linalg.norm(vec)
        return vec

    def _embed_batch(texts: list[str]) -> np.ndarray:
        return np.stack([_embed_text(t) for t in texts])

    embedder = MagicMock()
    embedder.embed_text = _embed_text
    embedder.embed_batch = _embed_batch
    embedder.dim = 384
    return embedder
