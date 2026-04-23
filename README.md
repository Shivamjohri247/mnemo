# Mnemo — Living Memory for AI Agents

**Biologically-inspired local-first memory engine** based on [SuperLocalMemory V3.3: The Living Brain](https://arxiv.org/abs/2604.04514). No cloud. No API keys. Runs 100% locally with SQLite.

AI coding agents forget everything between sessions. Mnemo gives them a brain that remembers, forgets, and adapts — just like humans do.

## Quick Start

```bash
pip install mnemo
```

```python
from mnemo import MemoryEngine

engine = MemoryEngine()

# Remember
engine.remember("We use PostgreSQL for persistence", importance=0.9)
engine.remember("FastAPI handles all REST endpoints", importance=0.8)
engine.remember("Redis for session caching", importance=0.6)

# Recall
results = engine.recall("what database?")
print(results[0]["text"])  # "We use PostgreSQL for persistence"

# Stats
print(engine.stats())  # {"Active": {"count": 3, "avg_retention": 1.0}}
```

## How It Works

### Ebbinghaus Forgetting Curve

Facts decay like human memories. The forgetting model implements paper equations (4), (5), and (9):

```
S(m) = max(S_min, α·log(1+a) + β·ι + γ_c·γ + δ·ε)
R(t)  = exp(-t · λ_eff / S(m))
```

Where:
- `a` = access count (logarithmic spacing effect)
- `ι` = importance (0.0–1.0)
- `γ` = confirmation count (corroboration)
- `ε` = emotional salience
- `τ` = trust score (untrusted facts decay 3x faster)

Facts progress through lifecycle states as retention drops:

| State | Retention | Precision | Use |
|-------|-----------|-----------|-----|
| Active | > 0.8 | 32-bit | Frequently accessed, high importance |
| Warm | > 0.5 | 8-bit | Still relevant, moderate access |
| Cold | > 0.2 | 4-bit | Background knowledge |
| Archive | > 0.05 | 2-bit | Long-term patterns |
| Forgotten | <= 0.05 | deleted | Garbage collected after 7 days |

### 4-Channel Retrieval

Every recall query runs through four retrieval channels, fused with Weighted Reciprocal Rank Fusion (RRF):

1. **Semantic** (weight 1.2) — sqlite-vec KNN cosine similarity over float32 embeddings
2. **BM25** (weight 1.0) — SQLite FTS5 full-text search
3. **Temporal** (weight 1.0) — Recency scoring (inverse hours since access)
4. **Entity** (weight 1.0) — Knowledge graph traversal (1-hop via entity edges)

```
score(m) = Σ w_c · 1/(k + rank_c(m))    where k=15
```

### Adaptive Quantization

As facts age and retention drops, their embedding vectors are compressed to save memory:

- **32-bit** (384 bytes) — Active facts, full precision
- **8-bit** (384 bytes) — Warm facts, 4x compression
- **4-bit** (192 bytes) — Cold facts, 8x compression
- **2-bit** (96 bytes) — Archived facts, 16x compression

The FRQAD (Fine-to-Rough Quantization-Aware Decision) approximation ensures f32 vectors are preferred ≥97% of the time for semantically similar queries.

### Soft Prompts

Mnemo auto-generates behavioral context from learned patterns. High-confidence patterns are rendered as template-based soft prompts that can be injected into agent sessions.

## Installation

```bash
# Basic install
pip install mnemo

# With all extras (dev tools + optional backends)
pip install mnemo[all]

# Just dev tools
pip install mnemo[dev]
```

Requires Python 3.11+. Uses `all-MiniLM-L6-v2` (384 dims, ~80MB) for embeddings by default.

## CLI Usage

```bash
# Initialize storage
mnemo init

# Store a memory
mnemo remember "We use PostgreSQL for the primary database" --importance 0.9

# Recall memories
mnemo recall "what database?"

# List active memories
mnemo list-memories

# Show statistics
mnemo stats

# View soft prompts
mnemo prompts

# Launch web dashboard
mnemo dashboard

# Start daemon server
mnemo daemon

# Install Claude Code hooks
mnemo install-hooks
```

## Claude Code Integration

### Session Hooks (Automatic)

Hooks give Claude Code automatic memory — every session starts with your important memories pre-loaded, and file edits are observed silently.

```bash
# Install globally (recommended)
mnemo install-hooks

# Install for current project only
mnemo install-hooks --scope local
```

Three hooks are installed:

| Hook | Trigger | Behavior |
|------|---------|----------|
| `session_start` | New session | Loads soft prompts + 5 most recent memories |
| `post_tool_use` | After Write/Edit/MultiEdit | Rate-limited observation (max 1 per file per 5 min) |
| `stop` | Session ends | Saves session record with git context |

All hooks fail silently — they never block or crash your session.

### MCP Tools (15 tools)

Add Mnemo as an MCP server in your Claude Code settings:

```json
{
  "mcpServers": {
    "mnemo": {
      "command": "python",
      "args": ["-m", "mnemo.mcp.tools"]
    }
  }
}
```

| Tool | Description |
|------|-------------|
| `slm_recall` | Recall memories using 4-channel retrieval |
| `slm_remember` | Store a new memory |
| `slm_forget` | Archive or delete a memory |
| `slm_list_memories` | Browse memories by lifecycle state |
| `slm_get_soft_prompts` | Get behavioral context for current session |
| `slm_stats` | Summary statistics by lifecycle |
| `slm_set_importance` | Adjust a memory's importance |
| `slm_confirm_memory` | Increment confirmation count |
| `slm_search_entities` | Entity graph lookup |
| `slm_forgetting_curve` | Retention over time for a fact |
| `slm_list_sessions` | Session history |
| `slm_export` | JSON dump of all memories |
| `slm_reset_learning` | Erase behavioral patterns |
| `slm_consolidate` | Trigger manual consolidation pass |
| `slm_daemon_status` | Health check |

## Agent Framework Adapters

Mnemo provides a universal `MemoryStore` protocol and adapters for popular agent frameworks.

### MemoryStore Protocol

```python
from mnemo.adapters.protocol import MemoryStore, MnemoStore

class MemoryStore(Protocol):
    def store(self, content: str, metadata: dict | None = None) -> str: ...
    def retrieve(self, query: str, top_k: int = 5) -> list[dict]: ...
    def delete(self, memory_id: str) -> bool: ...
    def search(self, query: str, filters: dict | None = None) -> list[dict]: ...
```

### LangChain / LangGraph

```python
from mnemo.adapters.langchain import MnemoLangGraphStore

store = MnemoLangGraphStore(project="myapp")
store.put(("memories",), "key1", {"content": "Important fact"})
results = store.search(("memories",), query="fact")
```

### CrewAI

```python
from mnemo.adapters.crewai import MnemoCrewAIMemory
from crewai.memory.external.external_memory import ExternalMemory

storage = MnemoCrewAIMemory(project="myapp")
external_memory = ExternalMemory(storage=storage)
agent = Agent(memory=external_memory, ...)
```

### Semantic Kernel

```python
from mnemo.adapters.semantic_kernel import MnemoSKStore

store = MnemoSKStore(project="myapp")
store.save_reference("collection", "key", "Important fact", description="...")
results = store.search("collection", "fact query", limit=5)
```

### Pydantic AI

```python
from mnemo.adapters.pydantic_ai import MnemoDependency
from pydantic_ai import Agent, RunContext

agent = Agent("openai:gpt-4", deps_type=MnemoDependency)

@agent.tool
async def remember(ctx: RunContext[MnemoDependency], text: str) -> str:
    return ctx.deps.store_memory(text)

@agent.tool
async def recall(ctx: RunContext[MnemoDependency], query: str) -> list[str]:
    return ctx.deps.retrieve_memories(query)

# Run with dependency
result = await agent.run("What database do we use?", deps=MnemoDependency(project="myapp"))
```

## API Reference

### MemoryEngine

```python
engine = MemoryEngine(project="myapp", auto_scheduler=True)

# Core operations
engine.remember(text, importance=0.5, source="user", trust_score=1.0) -> str | None
engine.recall(query, top_k=10) -> list[dict]
engine.forget(fact_id) -> None

# Memory management
engine.list_memories(lifecycle="Active", limit=20) -> list[dict]
engine.get_fact(fact_id) -> dict | None
engine.set_importance(fact_id, importance) -> bool
engine.confirm_memory(fact_id) -> int

# Analysis
engine.stats() -> dict
engine.forgetting_curve(fact_id, days_ahead=30) -> list[dict]
engine.get_soft_prompts() -> str
engine.export_memories(project=None, lifecycle=None) -> list[dict]

# Maintenance
engine.run_decay() -> dict
engine.shutdown() -> None
```

### Configuration

Environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `SLM_DATA_DIR` | `~/.slm` | Data directory for SQLite databases |
| `SLM_EMBEDDING_BACKEND` | `local` | Embedding backend (`local` or `ollama`) |
| `SLM_EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | Sentence-transformers model name |

## Benchmarks

All benchmarks verified in `tests/e2e/test_benchmark.py`:

| Benchmark | Result |
|-----------|--------|
| 30-day hot fact survival | Active (importance=0.7, 1/day, 12h since access) |
| 30-day cold fact forgetting | Forgotten (importance=0.2, 1 access, 30 days) |
| Hot/Cold discriminative ratio | > 5x (paper claims 6.7x) |
| Retention gap | > 100x between hot and cold |
| FRQAD f32 preference rate | >= 97% over int4 (1000 pairs) |
| Session continuity (10 facts) | 10/10 survive session boundary |
| Session continuity (50 facts) | >= 90% recovery rate |
| Recall latency (100 facts) | < 500ms |
| Decay pass (100 facts) | < 1s |

## Comparison with Alternatives

| Feature | Mnemo | engram-core | Mem0 | Zep |
|---------|-------|-------------|------|-----|
| Ebbinghaus forgetting | Yes | No | No | No |
| Adaptive quantization (32→2 bit) | Yes | No | No | No |
| Soft prompt generation | Yes | No | No | No |
| 4-channel RRF retrieval | Yes | No | Partial | No |
| Trust-weighted decay | Yes | No | No | No |
| Knowledge graph entities | Yes | Partial | No | Yes |
| MCP server | Yes | Yes | No | No |
| Claude Code hooks | Yes | Yes | No | No |
| Self-hosted (zero infra) | SQLite | SQLite | Cloud-only | Graph DB |
| Open source | MIT | MIT | Partial | BSL |

## Development

```bash
# Install with dev dependencies
pip install -e ".[dev]"

# Run all tests (278+ tests)
pytest

# Run with coverage
pytest --cov=mnemo

# Lint
ruff check .
ruff format --check .

# Type check
mypy mnemo/ --ignore-missing-imports
```

## Architecture

```
mnemo/
├── core/                    # Core algorithms
│   ├── engine.py            # MemoryEngine orchestrator
│   ├── forgetting.py        # Ebbinghaus math (eq 4, 5, 9)
│   ├── quantization.py      # Precision tiers + FRQAD
│   ├── retrieval.py         # 4-channel retrieval + RRF
│   ├── consolidation.py     # Episode → pattern extraction
│   └── parameterization.py  # Soft prompt generation
├── storage/                 # Persistence
│   ├── db.py                # SQLite connection manager
│   ├── schema.sql           # Database schema
│   └── embeddings.py        # Local + Ollama embedder
├── pipeline/                # Processing pipelines
│   ├── encode.py            # Fact extraction → NER → embed
│   └── lifecycle.py         # Decay scheduler
├── adapters/                # Agent framework adapters
│   ├── protocol.py          # MemoryStore protocol
│   ├── langchain.py         # LangGraph BaseStore
│   ├── crewai.py            # CrewAI ExternalMemory
│   ├── semantic_kernel.py   # SK memory store
│   └── pydantic_ai.py       # Pydantic AI dependency
├── cli/                     # Typer CLI (13 commands)
├── mcp/                     # 15 MCP tools via fastmcp
├── daemon/                  # FastAPI daemon on :8767
└── dashboard/               # Web dashboard on :8768
```

## License

MIT
