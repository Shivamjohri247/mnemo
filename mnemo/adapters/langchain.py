"""LangChain/LangGraph adapter for Mnemo.

Maps to LangGraph's BaseStore interface:
  put(namespace, key, value) -> None
  get(namespace, key) -> Item
  search(namespace, *, query, limit) -> list[SearchResult]
  delete(namespace, key) -> None
"""

from __future__ import annotations

from mnemo.adapters.protocol import MnemoStore


class MnemoLangGraphStore:
    """Mnemo adapter for LangGraph BaseStore interface.

    Usage:
        from mnemo.adapters.langchain import MnemoLangGraphStore
        store = MnemoLangGraphStore(project="myapp")
        # Use with LangGraph agents
    """

    def __init__(self, project: str | None = None):
        self._store = MnemoStore(project=project)
        self._cache: dict[str, dict] = {}

    def put(self, namespace: tuple[str, ...], key: str, value: dict) -> None:
        """Store a memory item."""
        content = value.get("content", value.get("text", str(value)))
        metadata = {
            "namespace": "/".join(namespace),
            "key": key,
            **value.get("metadata", {}),
        }
        mem_id = self._store.store(content, metadata=metadata)
        self._cache[f"{namespace}/{key}"] = {
            "namespace": namespace,
            "key": key,
            "value": value,
            "created_at": metadata.get("created_at"),
            "updated_at": metadata.get("updated_at"),
            "mem_id": mem_id,
        }

    def get(self, namespace: tuple[str, ...], key: str) -> dict | None:
        """Retrieve a specific item by key."""
        cached = self._cache.get(f"{namespace}/{key}")
        if cached:
            return cached
        # Search for it
        results = self._store.retrieve(key, top_k=1)
        if results:
            return {
                "namespace": namespace,
                "key": key,
                "value": {"content": results[0]["content"]},
            }
        return None

    def search(
        self,
        namespace: tuple[str, ...],
        *,
        query: str = "",
        limit: int = 10,
        **kwargs,
    ) -> list[dict]:
        """Search memories in namespace."""
        results = self._store.retrieve(query, top_k=limit)
        return [
            {
                "namespace": namespace,
                "key": r["id"][:8],
                "value": {"content": r["content"]},
                "created_at": None,
                "updated_at": None,
                "score": r.get("score", 0),
            }
            for r in results
        ]

    def delete(self, namespace: tuple[str, ...], key: str) -> None:
        """Delete a memory item."""
        cached = self._cache.pop(f"{namespace}/{key}", None)
        if cached and cached.get("mem_id"):
            self._store.delete(cached["mem_id"])
