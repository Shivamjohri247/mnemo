"""15 MCP tools via fastmcp for SuperLocalMemory."""

from fastmcp import FastMCP

from mnemo.core.engine import MemoryEngine

mcp = FastMCP("Mnemo")

_engine: MemoryEngine | None = None


def get_engine(project: str | None = None) -> MemoryEngine:
    global _engine
    if _engine is None:
        _engine = MemoryEngine(project=project, auto_scheduler=True)
    return _engine


@mcp.tool()
def slm_recall(query: str, top_k: int = 10, project: str | None = None) -> list[dict]:
    """Recall relevant memories using 4-channel retrieval."""
    engine = get_engine(project)
    results = engine.recall(query, top_k=top_k)
    return [
        {
            "id": r["id"],
            "text": r["text"],
            "lifecycle": r["lifecycle"],
            "retention": round(r.get("retention") or 0, 3),
            "strength": round(r.get("strength") or 0, 2),
            "access_count": r["access_count"],
        }
        for r in results
    ]


@mcp.tool()
def slm_remember(text: str, importance: float = 0.5, source: str = "agent") -> str | None:
    """Store a new memory. Returns fact_id or null if duplicate."""
    engine = get_engine()
    return engine.remember(text, importance=importance, source=source)


@mcp.tool()
def slm_forget(fact_id: str, hard_delete: bool = False) -> dict:
    """Archive or delete a specific memory."""
    engine = get_engine()
    if hard_delete:
        with engine.conn:
            engine.conn.execute("DELETE FROM facts WHERE id = ?", (fact_id,))
        return {"success": True}
    engine.forget(fact_id)
    return {"success": True}


@mcp.tool()
def slm_list_memories(
    lifecycle: str = "Active", limit: int = 20, project: str | None = None
) -> list[dict]:
    """Browse memories by lifecycle state."""
    engine = get_engine(project)
    return engine.list_memories(lifecycle=lifecycle, limit=limit)


@mcp.tool()
def slm_get_soft_prompts(project: str | None = None) -> str:
    """Get current session's injected behavioral context."""
    engine = get_engine(project)
    return engine.get_soft_prompts()


@mcp.tool()
def slm_stats(project: str | None = None) -> dict:
    """Summary statistics by lifecycle state."""
    engine = get_engine(project)
    return engine.stats()


@mcp.tool()
def slm_set_importance(fact_id: str, importance: float) -> dict:
    """Manually adjust a memory's importance."""
    engine = get_engine()
    success = engine.set_importance(fact_id, importance)
    return {"success": success}


@mcp.tool()
def slm_confirm_memory(fact_id: str) -> dict:
    """Increment confirmation count (corroboration)."""
    engine = get_engine()
    count = engine.confirm_memory(fact_id)
    return {"confirmations": count}


@mcp.tool()
def slm_search_entities(entity_name: str, entity_type: str | None = None) -> list[dict]:
    """Entity graph lookup."""
    engine = get_engine()
    conn = engine.conn
    query = "SELECT * FROM entities WHERE canonical_name LIKE ?"
    params: list = [f"%{entity_name.lower()}%"]
    if entity_type:
        query += " AND type = ?"
        params.append(entity_type)
    rows = conn.execute(query, params).fetchall()
    entities = [dict(r) for r in rows]

    for ent in entities:
        facts = conn.execute(
            """
            SELECT f.id, f.text FROM facts f
            JOIN entity_mentions em ON f.id = em.fact_id
            WHERE em.entity_id = ?
            """,
            (ent["id"],),
        ).fetchall()
        ent["related_facts"] = [r["text"] for r in facts]

    return entities


@mcp.tool()
def slm_forgetting_curve(fact_id: str, days_ahead: int = 30) -> list[dict]:
    """Return retention over time for a fact."""
    engine = get_engine()
    return engine.forgetting_curve(fact_id, days_ahead=days_ahead)


@mcp.tool()
def slm_list_sessions(limit: int = 10, project: str | None = None) -> list[dict]:
    """Session history."""
    engine = get_engine()
    conn = engine.conn
    query = "SELECT * FROM sessions WHERE 1=1"
    params: list = []
    if project:
        query += " AND project = ?"
        params.append(project)
    query += " ORDER BY started_at DESC LIMIT ?"
    params.append(limit)
    rows = conn.execute(query, params).fetchall()
    return [dict(r) for r in rows]


@mcp.tool()
def slm_export(project: str | None = None, lifecycle: str | None = None) -> list[dict]:
    """JSON dump of all memories."""
    engine = get_engine()
    return engine.export_memories(project=project, lifecycle=lifecycle)


@mcp.tool()
def slm_reset_learning() -> dict:
    """GDPR erasure of behavioral patterns."""
    engine = get_engine()
    with engine.conn:
        cursor = engine.conn.execute("DELETE FROM patterns")
        count = cursor.rowcount
        engine.conn.execute("DELETE FROM soft_prompts")
        engine.conn.execute("DELETE FROM feedback_signals")
        engine.conn.execute("DELETE FROM retrieval_log")
    return {"deleted_count": count}


@mcp.tool()
def slm_consolidate(project: str | None = None) -> dict:
    """Trigger manual consolidation pass."""
    from mnemo.core.consolidation import run_consolidation_pass

    count = run_consolidation_pass(project=project)
    return {"patterns_created": count}


@mcp.tool()
def slm_daemon_status() -> dict:
    """Health check — is the daemon running?"""
    from mnemo.daemon.client import DAEMON_URL, is_running

    if is_running():
        return {"status": "running", "port": 8767, "url": DAEMON_URL}
    return {"status": "stopped", "port": 8767, "url": DAEMON_URL}


if __name__ == "__main__":
    mcp.run()
