"""Tests for CrewAI, Pydantic AI, and Semantic Kernel adapters."""

from unittest.mock import MagicMock, patch

import pytest


def _make_env(tmp_path, monkeypatch, mock_embedder):
    data_dir = tmp_path / ".slm"
    data_dir.mkdir()
    monkeypatch.setenv("SLM_DATA_DIR", str(data_dir))

    mock_nlp = MagicMock()
    mock_nlp.return_value.ents = []
    return mock_nlp


# ── CrewAI ─────────────────────────────────────────────────────────────


class TestCrewAIAdapter:
    @pytest.fixture
    def crew_env(self, tmp_path, monkeypatch, mock_embedder):
        mock_nlp = _make_env(tmp_path, monkeypatch, mock_embedder)
        with (
            patch("mnemo.pipeline.encode.get_embedder", return_value=mock_embedder),
            patch("mnemo.pipeline.encode._get_nlp", return_value=mock_nlp),
        ):
            from mnemo.adapters.crewai import MnemoCrewAIMemory

            yield MnemoCrewAIMemory(project="crew_test")

    def test_save_stores_content(self, crew_env):
        crew_env.save("CrewAI observation", metadata={"importance": 0.7})

    def test_search_returns_strings(self, crew_env, mock_embedder):
        crew_env.save("Python is preferred for ML")
        with patch("mnemo.core.retrieval.embed_text", side_effect=mock_embedder.embed_text):
            results = crew_env.search("Python ML", limit=3)
        assert isinstance(results, list)
        if results:
            assert isinstance(results[0], str)

    def test_reset_is_noop(self, crew_env):
        crew_env.reset()  # Should not raise


# ── Pydantic AI ────────────────────────────────────────────────────────


class TestPydanticAIAdapter:
    @pytest.fixture
    def pydantic_env(self, tmp_path, monkeypatch, mock_embedder):
        mock_nlp = _make_env(tmp_path, monkeypatch, mock_embedder)
        with (
            patch("mnemo.pipeline.encode.get_embedder", return_value=mock_embedder),
            patch("mnemo.pipeline.encode._get_nlp", return_value=mock_nlp),
        ):
            from mnemo.adapters.pydantic_ai import MnemoDependency

            yield MnemoDependency(project="pydantic_test")

    def test_store_memory_returns_id(self, pydantic_env):
        mem_id = pydantic_env.store_memory("Pydantic AI fact", metadata={"importance": 0.8})
        assert isinstance(mem_id, str)
        assert len(mem_id) > 0

    def test_retrieve_memories_returns_strings(self, pydantic_env, mock_embedder):
        pydantic_env.store_memory("FastAPI for APIs")
        with patch("mnemo.core.retrieval.embed_text", side_effect=mock_embedder.embed_text):
            results = pydantic_env.retrieve_memories("FastAPI")
        assert isinstance(results, list)
        if results:
            assert isinstance(results[0], str)

    def test_forget_memory(self, pydantic_env):
        mem_id = pydantic_env.store_memory("To forget")
        result = pydantic_env.forget(mem_id)
        assert result is True

    def test_store_tool_function(self, pydantic_env):
        from mnemo.adapters.pydantic_ai import mnemo_store_tool

        ctx = MagicMock()
        ctx.deps = pydantic_env
        result = mnemo_store_tool(ctx, "tool stored fact", importance=0.5)
        assert isinstance(result, str)

    def test_retrieve_tool_function(self, pydantic_env, mock_embedder):
        from mnemo.adapters.pydantic_ai import mnemo_retrieve_tool

        pydantic_env.store_memory("Tool retrieve test")
        ctx = MagicMock()
        ctx.deps = pydantic_env
        with patch("mnemo.core.retrieval.embed_text", side_effect=mock_embedder.embed_text):
            results = mnemo_retrieve_tool(ctx, "Tool retrieve")
        assert isinstance(results, list)


# ── Semantic Kernel ────────────────────────────────────────────────────


class TestSemanticKernelAdapter:
    @pytest.fixture
    def sk_env(self, tmp_path, monkeypatch, mock_embedder):
        mock_nlp = _make_env(tmp_path, monkeypatch, mock_embedder)
        with (
            patch("mnemo.pipeline.encode.get_embedder", return_value=mock_embedder),
            patch("mnemo.pipeline.encode._get_nlp", return_value=mock_nlp),
        ):
            from mnemo.adapters.semantic_kernel import MnemoSKStore

            yield MnemoSKStore(project="sk_test")

    def test_save_reference_returns_id(self, sk_env):
        mem_id = sk_env.save_reference(
            collection="docs",
            key="doc1",
            text="Semantic Kernel doc",
            description="A document",
        )
        assert isinstance(mem_id, str)

    def test_get_returns_record(self, sk_env):
        sk_env.save_reference("docs", "key1", "Test content", "desc")
        record = sk_env.get("docs", "key1")
        assert record is not None
        assert record.text == "Test content"

    def test_get_returns_none_for_missing(self, sk_env):
        assert sk_env.get("nonexistent", "missing") is None

    def test_search_returns_records(self, sk_env, mock_embedder):
        sk_env.save_reference("docs", "k1", "Python documentation", "about Python")
        with patch("mnemo.core.retrieval.embed_text", side_effect=mock_embedder.embed_text):
            results = sk_env.search("docs", "Python", limit=5)
        assert isinstance(results, list)

    def test_search_with_min_relevance(self, sk_env, mock_embedder):
        sk_env.save_reference("docs", "k1", "High relevance doc", "important")
        with patch("mnemo.core.retrieval.embed_text", side_effect=mock_embedder.embed_text):
            results = sk_env.search("docs", "relevance", min_relevance_score=99.0)
        assert len(results) == 0

    def test_remove_deletes_record(self, sk_env):
        sk_env.save_reference("docs", "del1", "To be removed", "")
        sk_env.remove("docs", "del1")
        assert sk_env.get("docs", "del1") is None

    @pytest.mark.asyncio
    async def test_async_save_and_get(self, sk_env):
        await sk_env.save_reference_async("docs", "async1", "Async content", "")
        record = await sk_env.get_async("docs", "async1")
        assert record is not None
        assert record.text == "Async content"

    @pytest.mark.asyncio
    async def test_async_search(self, sk_env, mock_embedder):
        sk_env.save_reference("docs", "async_s", "Async search doc", "")
        with patch("mnemo.core.retrieval.embed_text", side_effect=mock_embedder.embed_text):
            results = await sk_env.search_async("docs", "Async")
        assert isinstance(results, list)

    @pytest.mark.asyncio
    async def test_async_remove(self, sk_env):
        sk_env.save_reference("docs", "async_del", "Async delete", "")
        await sk_env.remove_async("docs", "async_del")
        assert sk_env.get("docs", "async_del") is None
