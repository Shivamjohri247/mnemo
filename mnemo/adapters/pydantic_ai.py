"""Pydantic AI adapter for Mnemo.

Provides a dependency class that can be injected into Pydantic AI agents
via deps_type and RunContext.

Usage:
    from mnemo.adapters.pydantic_ai import MnemoDependency, mnemo_store_tool, mnemo_retrieve_tool
    from pydantic_ai import Agent, RunContext

    mnemo_dep = MnemoDependency(project="myapp")

    agent = Agent(
        'openai:gpt-4',
        deps_type=MnemoDependency,
    )

    @agent.tool
    async def remember(ctx: RunContext[MnemoDependency], text: str) -> str:
        return ctx.deps.store(text)

    @agent.tool
    async def recall(ctx: RunContext[MnemoDependency], query: str) -> list[str]:
        return ctx.deps.retrieve(query)
"""

from __future__ import annotations

from dataclasses import dataclass, field

from mnemo.adapters.protocol import MnemoStore


@dataclass
class MnemoDependency:
    """Pydantic AI dependency for Mnemo memory.

    Inject into agents via deps_type=MnemoDependency.
    Pass instance to agent.run(prompt, deps=mnemo_dep).
    """

    project: str | None = None
    _store: MnemoStore | None = field(default=None, init=False, repr=False)

    @property
    def store(self) -> MnemoStore:
        if self._store is None:
            self._store = MnemoStore(project=self.project)
        return self._store

    def store_memory(self, content: str, metadata: dict | None = None) -> str:
        """Store a memory. Returns ID."""
        return self.store.store(content, metadata=metadata)

    def retrieve_memories(self, query: str, top_k: int = 5) -> list[str]:
        """Retrieve memories as content strings."""
        results = self.store.retrieve(query, top_k=top_k)
        return [r["content"] for r in results]

    def forget(self, memory_id: str) -> bool:
        """Forget a memory."""
        return self.store.delete(memory_id)


def mnemo_store_tool(ctx, text: str, importance: float = 0.5) -> str:
    """Tool function: store a memory. Register with @agent.tool."""
    dep: MnemoDependency = ctx.deps
    return dep.store_memory(text, metadata={"importance": importance})


def mnemo_retrieve_tool(ctx, query: str, top_k: int = 5) -> list[str]:
    """Tool function: retrieve memories. Register with @agent.tool."""
    dep: MnemoDependency = ctx.deps
    return dep.retrieve_memories(query, top_k=top_k)
