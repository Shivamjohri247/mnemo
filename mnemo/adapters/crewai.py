"""CrewAI adapter for Mnemo.

CrewAI uses an ExternalMemory pattern where you provide a custom storage class
with save() and search() methods.

Usage:
    from mnemo.adapters.crewai import MnemoCrewAIMemory
    from crewai.memory.external.external_memory import ExternalMemory

    storage = MnemoCrewAIMemory(project="myapp")
    external_memory = ExternalMemory(storage=storage)
    agent = Agent(memory=external_memory, ...)
"""

from __future__ import annotations

from mnemo.adapters.protocol import MnemoStore


class MnemoCrewAIMemory:
    """Mnemo adapter for CrewAI External Memory.

    Implements the storage interface expected by CrewAI's ExternalMemory:
      save(content, metadata) -> None
      search(query, limit) -> list[str]
    """

    def __init__(self, project: str | None = None):
        self._store = MnemoStore(project=project)

    def save(self, content: str, metadata: dict | None = None) -> None:
        """Save a memory item."""
        self._store.store(content, metadata=metadata)

    def search(self, query: str, limit: int = 5) -> list[str]:
        """Search memories and return content strings."""
        results = self._store.retrieve(query, top_k=limit)
        return [r["content"] for r in results]

    def reset(self) -> None:
        """Clear all memories (for testing)."""
        pass
