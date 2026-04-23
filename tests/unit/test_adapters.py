"""Tests for agent framework adapters."""

from unittest.mock import MagicMock, patch

import pytest

from mnemo.adapters.protocol import MemoryStore, MnemoStore

# ── Protocol Compliance ───────────────────────────────────────────────


class TestMemoryStoreProtocol:
    def test_is_runtime_checkable(self):
        """MemoryStore should be a runtime_checkable Protocol."""
        assert hasattr(MemoryStore, "__protocol_attrs__") or hasattr(MemoryStore, "_is_protocol")

    def test_mnemo_store_is_instance(self):
        """MnemoStore should satisfy the MemoryStore protocol."""
        assert hasattr(MnemoStore, "store")
        assert hasattr(MnemoStore, "retrieve")
        assert hasattr(MnemoStore, "delete")
        assert hasattr(MnemoStore, "search")


# ── MnemoStore ────────────────────────────────────────────────────────


@pytest.fixture
def store_env(tmp_path, monkeypatch, mock_embedder):
    data_dir = tmp_path / ".slm"
    data_dir.mkdir()
    monkeypatch.setenv("SLM_DATA_DIR", str(data_dir))

    mock_nlp = MagicMock()
    mock_nlp.return_value.ents = []

    with (
        patch("mnemo.pipeline.encode.get_embedder", return_value=mock_embedder),
        patch("mnemo.pipeline.encode._get_nlp", return_value=mock_nlp),
    ):
        store = MnemoStore(project="adapter_test")
        yield store


class TestMnemoStoreOperations:
    def test_store_returns_id(self, store_env):
        mem_id = store_env.store("Test memory content", metadata={"importance": 0.8})
        assert isinstance(mem_id, str)
        assert len(mem_id) > 0

    def test_store_with_metadata(self, store_env):
        mem_id = store_env.store(
            "Agent observation",
            metadata={"importance": 0.9, "source": "agent"},
        )
        assert mem_id  # Should succeed

    def test_retrieve_returns_list(self, store_env, mock_embedder):
        store_env.store("Python is preferred language")
        with patch("mnemo.core.retrieval.embed_text", side_effect=mock_embedder.embed_text):
            results = store_env.retrieve("Python language")
        assert isinstance(results, list)

    def test_retrieve_has_required_fields(self, store_env, mock_embedder):
        store_env.store("Redis for caching", metadata={"importance": 0.8})
        with patch("mnemo.core.retrieval.embed_text", side_effect=mock_embedder.embed_text):
            results = store_env.retrieve("caching")
        if results:
            r = results[0]
            assert "id" in r
            assert "content" in r
            assert "metadata" in r
            assert "score" in r

    def test_delete_returns_true(self, store_env):
        mem_id = store_env.store("To be deleted")
        result = store_env.delete(mem_id)
        assert result is True

    def test_search_with_filters(self, store_env, mock_embedder):
        store_env.store("Active memory", metadata={"importance": 0.9})
        with patch("mnemo.core.retrieval.embed_text", side_effect=mock_embedder.embed_text):
            results = store_env.search("memory", filters={"lifecycle": "Active"})
        assert isinstance(results, list)

    def test_search_without_filters(self, store_env, mock_embedder):
        store_env.store("General memory")
        with patch("mnemo.core.retrieval.embed_text", side_effect=mock_embedder.embed_text):
            results = store_env.search("General memory")
        assert isinstance(results, list)


# ── LangChain Adapter ─────────────────────────────────────────────────


class TestLangChainAdapter:
    @pytest.fixture
    def langchain_env(self, tmp_path, monkeypatch, mock_embedder):
        data_dir = tmp_path / ".slm"
        data_dir.mkdir()
        monkeypatch.setenv("SLM_DATA_DIR", str(data_dir))

        mock_nlp = MagicMock()
        mock_nlp.return_value.ents = []

        with (
            patch("mnemo.pipeline.encode.get_embedder", return_value=mock_embedder),
            patch("mnemo.pipeline.encode._get_nlp", return_value=mock_nlp),
        ):
            from mnemo.adapters.langchain import MnemoLangGraphStore

            store = MnemoLangGraphStore(project="langchain_test")
            yield store

    def test_put_and_get(self, langchain_env):
        langchain_env.put(
            namespace=("memories",),
            key="test1",
            value={"content": "Test memory"},
        )
        result = langchain_env.get(("memories",), "test1")
        assert result is not None

    def test_search(self, langchain_env, mock_embedder):
        langchain_env.put(
            namespace=("memories",),
            key="search_test",
            value={"content": "Python is great"},
        )
        with patch("mnemo.core.retrieval.embed_text", side_effect=mock_embedder.embed_text):
            results = langchain_env.search(("memories",), query="Python")
        assert isinstance(results, list)

    def test_delete(self, langchain_env):
        langchain_env.put(
            namespace=("memories",),
            key="delete_test",
            value={"content": "Temporary"},
        )
        langchain_env.delete(("memories",), "delete_test")
        # Cache should be cleared
        assert langchain_env._cache.get("('memories',)/delete_test") is None

    def test_get_fallback_to_retrieve(self, langchain_env, mock_embedder):
        """When key not in cache, get() should fall back to retrieve."""
        langchain_env.put(
            namespace=("memories",),
            key="cached_key",
            value={"content": "Cached content"},
        )
        # Clear cache to force fallback
        langchain_env._cache.clear()
        with patch("mnemo.core.retrieval.embed_text", side_effect=mock_embedder.embed_text):
            result = langchain_env.get(("memories",), "cached_key")
        # Should return something from retrieve, or None if not found
        assert result is None or isinstance(result, dict)

    def test_get_returns_none_for_unknown(self, langchain_env, mock_embedder):
        """get() returns None for key not in cache and not in DB."""
        with patch("mnemo.core.retrieval.embed_text", side_effect=mock_embedder.embed_text):
            result = langchain_env.get(("memories",), "never_stored")
        assert result is None


class TestMnemoStoreSearchFilter:
    @pytest.fixture
    def search_env(self, tmp_path, monkeypatch, mock_embedder):
        data_dir = tmp_path / ".slm"
        data_dir.mkdir()
        monkeypatch.setenv("SLM_DATA_DIR", str(data_dir))

        mock_nlp = MagicMock()
        mock_nlp.return_value.ents = []

        with (
            patch("mnemo.pipeline.encode.get_embedder", return_value=mock_embedder),
            patch("mnemo.pipeline.encode._get_nlp", return_value=mock_nlp),
        ):
            from mnemo.adapters.protocol import MnemoStore

            store = MnemoStore(project="search_test")
            yield store

    def test_search_with_lifecycle_filter(self, search_env, mock_embedder):
        fid = search_env.store("Active memory for search", metadata={"importance": 0.9})
        search_env.delete(fid)  # Sets to Forgotten
        search_env.store("Still active memory", metadata={"importance": 0.8})

        with patch("mnemo.core.retrieval.embed_text", side_effect=mock_embedder.embed_text):
            results = search_env.search("memory", filters={"lifecycle": "Forgotten"})
        assert all(r["metadata"]["lifecycle"] == "Forgotten" for r in results)

    def test_search_without_filters(self, search_env, mock_embedder):
        search_env.store("Filterless memory", metadata={"importance": 0.7})
        with patch("mnemo.core.retrieval.embed_text", side_effect=mock_embedder.embed_text):
            results = search_env.search("memory")
        assert isinstance(results, list)
