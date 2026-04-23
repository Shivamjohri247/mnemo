"""Embedding helper with dual backend: local sentence-transformers or Ollama."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod

import numpy as np

log = logging.getLogger(__name__)

DEFAULT_LOCAL_MODEL = "all-MiniLM-L6-v2"
DEFAULT_OLLAMA_MODEL = "nomic-embed-text"
OLLAMA_URL = "http://localhost:11434/api/embeddings"


class Embedder(ABC):
    """Abstract embedder interface."""

    @property
    @abstractmethod
    def dim(self) -> int:
        """Embedding dimension."""

    @abstractmethod
    def embed_text(self, text: str) -> np.ndarray:
        """Embed a single text string."""

    @abstractmethod
    def embed_batch(self, texts: list[str]) -> np.ndarray:
        """Embed multiple text strings."""


class LocalEmbedder(Embedder):
    """Sentence-transformers local embedding."""

    def __init__(self, model_name: str = DEFAULT_LOCAL_MODEL):
        self._model_name = model_name
        self._model = None

    def _load(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self._model_name)
        return self._model

    @property
    def dim(self) -> int:
        if self._model is not None:
            return self._model.get_sentence_embedding_dimension()
        return 384  # default for all-MiniLM-L6-v2

    def embed_text(self, text: str) -> np.ndarray:
        model = self._load()
        vec = model.encode(text, normalize_embeddings=True)
        return vec.astype(np.float32)  # type: ignore[no-any-return]

    def embed_batch(self, texts: list[str]) -> np.ndarray:
        model = self._load()
        vecs = model.encode(texts, normalize_embeddings=True, batch_size=32)
        return vecs.astype(np.float32)  # type: ignore[no-any-return]


class OllamaEmbedder(Embedder):
    """Ollama embedding via HTTP API."""

    def __init__(self, model_name: str = DEFAULT_OLLAMA_MODEL, url: str = OLLAMA_URL):
        self._model_name = model_name
        self._url = url
        self._dim: int | None = None

    @property
    def dim(self) -> int:
        if self._dim is None:
            # Probe with a dummy text to determine dimension
            vec = self.embed_text("test")
            self._dim = len(vec)
        return self._dim

    def _request(self, text: str) -> list[float]:
        import httpx

        resp = httpx.post(self._url, json={"model": self._model_name, "prompt": text}, timeout=30)
        resp.raise_for_status()
        return list(resp.json()["embedding"])  # type: ignore[no-any-return]

    def embed_text(self, text: str) -> np.ndarray:
        vec = np.array(self._request(text), dtype=np.float32)
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec /= norm
        # Warn if dimension doesn't match expected schema (384)
        if len(vec) != 384:
            log.warning(
                f"Ollama model '{self._model_name}' returned {len(vec)}-dim vectors; "
                f"expected 384. Vectors will be truncated/padded."
            )
            if len(vec) > 384:
                vec = vec[:384]
                vec /= np.linalg.norm(vec)  # re-normalize after truncation
            else:
                vec = np.pad(vec, (0, 384 - len(vec)))
        return vec

    def embed_batch(self, texts: list[str]) -> np.ndarray:
        return np.stack([self.embed_text(t) for t in texts])


# Module-level singleton
_embedder: Embedder | None = None


def get_embedder(backend: str = "local", model: str | None = None) -> Embedder:
    """Factory: get or create the embedding backend."""
    global _embedder
    if _embedder is not None:
        return _embedder

    if backend == "ollama":
        _embedder = OllamaEmbedder(model_name=model or DEFAULT_OLLAMA_MODEL)
    else:
        _embedder = LocalEmbedder(model_name=model or DEFAULT_LOCAL_MODEL)
    return _embedder


def set_embedder(embedder: Embedder):
    """Override the global embedder (used by tests)."""
    global _embedder
    _embedder = embedder


def embed_text(text: str) -> np.ndarray:
    """Convenience: embed a single text using the global embedder."""
    return get_embedder().embed_text(text)


def embed_batch(texts: list[str]) -> np.ndarray:
    """Convenience: embed a batch of texts using the global embedder."""
    return get_embedder().embed_batch(texts)
