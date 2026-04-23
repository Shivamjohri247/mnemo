"""
Lifecycle scheduler: runs decay pass and garbage collection on a schedule.
Discrete approximation of the paper's Fokker-Planck continuous dynamics.
"""

import logging
import time

from apscheduler.schedulers.background import BackgroundScheduler

from mnemo.core.forgetting import compute_forget_state
from mnemo.core.quantization import apply_precision_tier
from mnemo.storage.db import get_connection, transaction

log = logging.getLogger(__name__)


def run_decay_pass(project: str | None = None):
    """Recompute forgetting state for all non-Forgotten facts."""
    conn = get_connection()

    if project:
        rows = conn.execute(
            "SELECT * FROM facts WHERE lifecycle != 'Forgotten' AND project = ?",
            (project,),
        ).fetchall()
    else:
        rows = conn.execute("SELECT * FROM facts WHERE lifecycle != 'Forgotten'").fetchall()

    now = time.time()
    updated = forgotten = 0

    with transaction(conn):
        for row in rows:
            fs = compute_forget_state(
                access_count=row["access_count"],
                importance=row["importance"],
                confirmations=row["confirmations"],
                emotional_salience=row["emotional_salience"],
                trust_score=row["trust_score"],
                accessed_at=row["accessed_at"],
                now=now,
            )
            conn.execute(
                """
                UPDATE facts
                SET strength = ?, retention = ?, lifecycle = ?, precision_bits = ?
                WHERE id = ?
                """,
                (fs.strength, fs.retention, fs.state, fs.precision_bits, row["id"]),
            )

            if fs.precision_bits != row["precision_bits"]:
                apply_precision_tier(row["id"], fs.precision_bits)

            if fs.state == "Forgotten":
                forgotten += 1
            updated += 1

    log.info(f"Decay pass complete: {updated} updated, {forgotten} forgotten")
    return {"updated": updated, "forgotten": forgotten}


def garbage_collect():
    """Delete Forgotten facts older than 7 days."""
    cutoff = time.time() - (7 * 24 * 3600)
    conn = get_connection()
    with transaction(conn):
        conn.execute(
            "DELETE FROM facts WHERE lifecycle = 'Forgotten' AND accessed_at < ?",
            (cutoff,),
        )


def start_scheduler():
    """Start the background lifecycle scheduler."""
    scheduler = BackgroundScheduler()
    scheduler.add_job(run_decay_pass, "interval", hours=1, id="decay_pass")
    scheduler.add_job(garbage_collect, "interval", hours=24, id="gc")
    scheduler.start()
    return scheduler
