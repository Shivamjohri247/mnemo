"""Web dashboard for mnemo — FastAPI + Jinja2 on port 8768."""

import logging
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from mnemo.core.forgetting import memory_strength, retention
from mnemo.storage.db import get_connection

logger = logging.getLogger(__name__)

dash = FastAPI()

_template_dir = Path(__file__).parent / "templates"
try:
    templates = Jinja2Templates(directory=str(_template_dir))
except Exception:
    logger.warning("Dashboard templates not found at %s", _template_dir)
    templates = None  # type: ignore[assignment]


def _render_template(name: str, context: dict) -> HTMLResponse:
    if templates:
        return templates.TemplateResponse(context["request"], name, context)
    return HTMLResponse(f"<h1>Template {name} not found</h1><pre>{context}</pre>")


@dash.get("/")
def index(request: Request, lifecycle: str = "all", limit: int = 50):
    conn = get_connection()
    if lifecycle == "all":
        facts = conn.execute(
            "SELECT * FROM facts WHERE lifecycle != 'Forgotten' ORDER BY accessed_at DESC LIMIT ?",
            (limit,),
        ).fetchall()
    else:
        facts = conn.execute(
            "SELECT * FROM facts WHERE lifecycle = ? ORDER BY accessed_at DESC LIMIT ?",
            (lifecycle, limit),
        ).fetchall()

    return _render_template(
        "index.html",
        {
            "request": request,
            "facts": [dict(f) for f in facts],
            "lifecycle": lifecycle,
        },
    )


@dash.get("/memory/{fact_id}")
def memory_detail(request: Request, fact_id: str):
    conn = get_connection()
    fact = conn.execute("SELECT * FROM facts WHERE id = ?", (fact_id,)).fetchone()
    if not fact:
        return HTMLResponse("<h1>Not found</h1>", status_code=404)

    s = memory_strength(
        fact["access_count"],
        fact["importance"],
        fact["confirmations"],
        fact["emotional_salience"],
    )
    curve = [
        {"hours": h, "retention": round(retention(s, h) * 100, 1)}
        for h in range(0, 720, 6)  # 30 days in 6-hour steps
    ]

    return _render_template(
        "memory.html",
        {
            "request": request,
            "fact": dict(fact),
            "curve": curve,
            "strength": round(s, 2),
        },
    )


@dash.get("/stats")
def stats(request: Request):
    conn = get_connection()
    rows = conn.execute(
        "SELECT lifecycle, COUNT(*) as count, AVG(retention) as avg_retention "
        "FROM facts GROUP BY lifecycle"
    ).fetchall()

    total_memories = sum(r["count"] for r in rows)
    lifecycle_counts = {r["lifecycle"]: r["count"] for r in rows}

    return _render_template(
        "stats.html",
        {
            "request": request,
            "lifecycle_counts": lifecycle_counts,
            "total_memories": total_memories,
        },
    )


def run_dashboard():
    import uvicorn

    uvicorn.run(dash, host="127.0.0.1", port=8768)
