"""MemoryEngine — single entry point for all memory operations.

Orchestrates fact storage, retrieval, forgetting, consolidation,
and soft prompt generation.
"""

import time

from mnemo.core.consolidation import run_consolidation_pass
from mnemo.core.parameterization import generate_soft_prompts
from mnemo.core.retrieval import recall as _recall
from mnemo.pipeline.encode import store_fact
from mnemo.pipeline.lifecycle import run_decay_pass, start_scheduler
from mnemo.storage.db import get_connection, init_schema


class MemoryEngine:
    def __init__(self, project: str | None = None, auto_scheduler: bool = True):
        self.project = project
        self.conn = get_connection()
        init_schema(self.conn)
        self._scheduler = None
        if auto_scheduler:
            self._scheduler = start_scheduler()

    def remember(
        self, text: str, importance: float = 0.5, source: str = "user", trust_score: float = 1.0
    ) -> str | None:
        return store_fact(
            text,
            source=source,
            project=self.project,
            importance=importance,
            trust_score=trust_score,
        )

    def recall(self, query: str, top_k: int = 10) -> list[dict]:
        return _recall(query, project=self.project, top_k=top_k)

    def forget(self, fact_id: str):
        with self.conn:
            self.conn.execute("UPDATE facts SET lifecycle = 'Forgotten' WHERE id = ?", (fact_id,))

    def get_soft_prompts(self) -> str:
        run_consolidation_pass(project=self.project)
        return generate_soft_prompts(project=self.project)

    def stats(self) -> dict:
        if self.project is not None:
            rows = self.conn.execute(
                """
                SELECT lifecycle, COUNT(*) as count, AVG(retention) as avg_retention
                FROM facts
                WHERE project = ?
                GROUP BY lifecycle
                """,
                (self.project,),
            ).fetchall()
        else:
            rows = self.conn.execute(
                """
                SELECT lifecycle, COUNT(*) as count, AVG(retention) as avg_retention
                FROM facts
                GROUP BY lifecycle
                """
            ).fetchall()
        return {
            r["lifecycle"]: {
                "count": r["count"],
                "avg_retention": round(r["avg_retention"] or 0, 3),
            }
            for r in rows
        }

    def run_decay(self):
        return run_decay_pass(project=self.project)

    def list_memories(self, lifecycle: str = "Active", limit: int = 20) -> list[dict]:
        rows = self.conn.execute(
            """
            SELECT * FROM facts
            WHERE lifecycle = ? AND (project IS NULL OR project = ?)
            ORDER BY accessed_at DESC LIMIT ?
            """,
            (lifecycle, self.project, limit),
        ).fetchall()
        return [dict(r) for r in rows]

    def get_fact(self, fact_id: str) -> dict | None:
        row = self.conn.execute("SELECT * FROM facts WHERE id = ?", (fact_id,)).fetchone()
        return dict(row) if row else None

    def set_importance(self, fact_id: str, importance: float) -> bool:
        with self.conn:
            cursor = self.conn.execute(
                "UPDATE facts SET importance = ? WHERE id = ?",
                (importance, fact_id),
            )
            return cursor.rowcount > 0

    def confirm_memory(self, fact_id: str) -> int:
        with self.conn:
            self.conn.execute(
                "UPDATE facts SET confirmations = confirmations + 1 WHERE id = ?",
                (fact_id,),
            )
        row = self.conn.execute(
            "SELECT confirmations FROM facts WHERE id = ?", (fact_id,)
        ).fetchone()
        return row["confirmations"] if row else 0

    def forgetting_curve(self, fact_id: str, days_ahead: int = 30) -> list[dict]:
        from mnemo.core.forgetting import memory_strength, retention

        row = self.conn.execute("SELECT * FROM facts WHERE id = ?", (fact_id,)).fetchone()
        if not row:
            return []

        s = memory_strength(
            row["access_count"],
            row["importance"],
            row["confirmations"],
            row["emotional_salience"],
        )
        hours_now = (time.time() - row["accessed_at"]) / 3600.0
        return [
            {"hours": h, "retention": round(retention(s, h) * 100, 1)}
            for h in range(int(hours_now), int(hours_now) + days_ahead * 24, 6)
        ]

    def export_memories(
        self, project: str | None = None, lifecycle: str | None = None
    ) -> list[dict]:
        query = "SELECT * FROM facts WHERE 1=1"
        params: list = []
        if project or self.project:
            query += " AND project = ?"
            params.append(project or self.project)
        if lifecycle:
            query += " AND lifecycle = ?"
            params.append(lifecycle)
        rows = self.conn.execute(query, params).fetchall()
        return [dict(r) for r in rows]

    def shutdown(self):
        if self._scheduler:
            self._scheduler.shutdown()
