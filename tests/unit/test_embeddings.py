"""Tests for embedding backends."""

import numpy as np
import pytest

from mnemo.storage.embeddings import LocalEmbedder, OllamaEmbedder


def test_local_embedder_dim():
    """LocalEmbedder should report correct dimension."""
    embedder = LocalEmbedder()
    assert embedder.dim == 384


def test_local_embedder_output_shape():
    """LocalEmbedder should produce 384-dim normalized vectors."""
    try:
        import sentence_transformers  # noqa: F401
    except (ImportError, ValueError):
        pytest.skip("sentence_transformers not available in this environment")
    embedder = LocalEmbedder()
    vec = embedder.embed_text("hello world")
    assert vec.shape == (384,)
    assert abs(np.linalg.norm(vec) - 1.0) < 1e-5


def test_local_embedder_batch():
    """Batch embedding should produce consistent results."""
    try:
        import sentence_transformers  # noqa: F401
    except (ImportError, ValueError):
        pytest.skip("sentence_transformers not available in this environment")
    embedder = LocalEmbedder()
    texts = ["hello", "world", "test"]
    vecs = embedder.embed_batch(texts)
    assert vecs.shape == (3, 384)

    # Single embed should match batch
    single = embedder.embed_text("hello")
    np.testing.assert_allclose(vecs[0], single, atol=1e-5)


def test_ollama_embedder_init():
    """OllamaEmbedder should initialize with correct defaults."""
    embedder = OllamaEmbedder()
    assert embedder._model_name == "nomic-embed-text"
    assert embedder._url == "http://localhost:11434/api/embeddings"


def test_mock_embedder_fixture(mock_embedder):
    """Verify the mock embedder fixture works."""
    vec = mock_embedder.embed_text("test")
    assert vec.shape == (384,)
    assert abs(np.linalg.norm(vec) - 1.0) < 1e-5

    vecs = mock_embedder.embed_batch(["a", "b"])
    assert vecs.shape == (2, 384)

    # Same text should produce same vector
    vec2 = mock_embedder.embed_text("test")
    np.testing.assert_array_equal(vec, vec2)
