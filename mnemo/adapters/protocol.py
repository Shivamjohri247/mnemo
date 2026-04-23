"""MemoryStore protocol — universal memory provider interface."""

from typing import Protocol, runtime_checkable


@runtime_checkable
class MemoryStore(Protocol):
    """Universal memory provider interface for agent frameworks."""

    def store(self, content: str, metadata: dict | None = None) -> str:
        """Store a memory. Returns memory ID."""
        ...

    def retrieve(self, query: str, top_k: int = 5) -> list[dict]:
        """Retrieve relevant memories. Returns list of {id, content, metadata, score}."""
        ...

    def delete(self, memory_id: str) -> bool:
        """Delete a memory. Returns success."""
        ...

    def search(self, query: str, filters: dict | None = None) -> list[dict]:
        """Search with optional filters. Returns list of {id, content, metadata, score}."""
        ...


class MnemoStore:
    """Mnemo-backed implementation of MemoryStore protocol."""

    def __init__(self, project: str | None = None):
        from mnemo.core.engine import MemoryEngine

        self._engine = MemoryEngine(project=project, auto_scheduler=False)

    def store(self, content: str, metadata: dict | None = None) -> str:
        importance = (metadata or {}).get("importance", 0.5)
        source = (metadata or {}).get("source", "agent")
        result = self._engine.remember(content, importance=importance, source=source)
        return result or ""

    def retrieve(self, query: str, top_k: int = 5) -> list[dict]:
        results = self._engine.recall(query, top_k=top_k)
        return [
            {
                "id": r["id"],
                "content": r["text"],
                "metadata": {
                    "lifecycle": r["lifecycle"],
                    "retention": r.get("retention"),
                    "importance": r["importance"],
                    "access_count": r["access_count"],
                },
                "score": r.get("retention", 0),
            }
            for r in results
        ]

    def delete(self, memory_id: str) -> bool:
        self._engine.forget(memory_id)
        return True

    def search(self, query: str, filters: dict | None = None) -> list[dict]:
        results = self._engine.recall(query, top_k=filters.get("top_k", 10) if filters else 10)
        output = []
        for r in results:
            if filters and filters.get("lifecycle") and r["lifecycle"] != filters["lifecycle"]:
                continue
            output.append(
                {
                    "id": r["id"],
                    "content": r["text"],
                    "metadata": {"lifecycle": r["lifecycle"], "importance": r["importance"]},
                    "score": r.get("retention", 0),
                }
            )
        return output
