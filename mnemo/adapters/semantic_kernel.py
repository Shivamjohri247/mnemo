"""Semantic Kernel adapter for Mnemo.

Maps to SK's memory store interface:
  save_reference(collection, key, text, description, external_id, external_source_name, additional_metadata)
  get(collection, key) -> MemoryRecord
  search(collection, query, limit, min_relevance_score) -> list[MemoryRecord]
  remove(collection, key) -> None

Usage:
    from mnemo.adapters.semantic_kernel import MnemoSKStore
    store = MnemoSKStore(project="myapp")
    # Register with SK's memory builder
"""

from __future__ import annotations

from dataclasses import dataclass

from mnemo.adapters.protocol import MnemoStore


@dataclass
class SKMemoryRecord:
    """Simplified SK MemoryRecord."""

    key: str
    text: str
    description: str
    additional_metadata: str
    external_source_name: str
    id_: str
    relevance_score: float = 0.0


class MnemoSKStore:
    """Mnemo adapter for Semantic Kernel memory store.

    Provides both sync and async interfaces. Maps SK's collection
    parameter to Mnemo's project field.
    """

    def __init__(self, project: str | None = None):
        self._store = MnemoStore(project=project)
        self._records: dict[str, dict[str, SKMemoryRecord]] = {}

    def _collection_key(self, collection: str, key: str) -> str:
        return f"{collection}/{key}"

    def save_reference(
        self,
        collection: str,
        key: str,
        text: str,
        description: str = "",
        external_id: str = "",
        external_source_name: str = "mnemo",
        additional_metadata: str = "",
    ) -> str:
        """Save a reference to memory. Returns the memory ID."""
        mem_id = self._store.store(
            text,
            metadata={
                "collection": collection,
                "description": description,
                "external_id": external_id,
            },
        )

        record = SKMemoryRecord(
            key=key,
            text=text,
            description=description,
            additional_metadata=additional_metadata,
            external_source_name=external_source_name,
            id_=mem_id,
        )

        if collection not in self._records:
            self._records[collection] = {}
        self._records[collection][key] = record
        return mem_id

    def get(self, collection: str, key: str) -> SKMemoryRecord | None:
        """Get a specific memory record."""
        return self._records.get(collection, {}).get(key)

    def search(
        self,
        collection: str,
        query: str,
        limit: int = 5,
        min_relevance_score: float = 0.0,
    ) -> list[SKMemoryRecord]:
        """Search memories in a collection."""
        results = self._store.retrieve(query, top_k=limit)
        records = []
        for r in results:
            if r.get("score", 0) >= min_relevance_score:
                records.append(
                    SKMemoryRecord(
                        key=r["id"][:8],
                        text=r["content"],
                        description="",
                        additional_metadata="",
                        external_source_name="mnemo",
                        id_=r["id"],
                        relevance_score=r.get("score", 0),
                    )
                )
        return records

    def remove(self, collection: str, key: str) -> None:
        """Remove a memory record."""
        record = self._records.get(collection, {}).pop(key, None)
        if record and record.id_:
            self._store.delete(record.id_)

    # Async wrappers
    async def save_reference_async(self, *args, **kwargs) -> str:
        return self.save_reference(*args, **kwargs)

    async def get_async(self, *args, **kwargs) -> SKMemoryRecord | None:
        return self.get(*args, **kwargs)

    async def search_async(self, *args, **kwargs) -> list[SKMemoryRecord]:
        return self.search(*args, **kwargs)

    async def remove_async(self, *args, **kwargs) -> None:
        self.remove(*args, **kwargs)
