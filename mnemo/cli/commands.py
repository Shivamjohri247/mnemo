"""Typer CLI for mnemo."""

import json

import typer
from rich.console import Console
from rich.table import Table

app = typer.Typer(name="mnemo", help="Mnemo — Local-first AI agent memory")
console = Console()


def _get_engine(project: str | None = None):
    from mnemo.core.engine import MemoryEngine

    return MemoryEngine(project=project, auto_scheduler=False)


@app.command()
def init(project: str | None = typer.Option(None, help="Default project identifier")):
    """Initialize mnemo storage."""
    engine = _get_engine(project)
    console.print("[green]Mnemo initialized.[/green]")
    console.print(f"Data directory: {engine.conn.execute('PRAGMA database_list').fetchone()[2]}")


@app.command()
def remember(
    text: str = typer.Argument(help="Fact to remember"),
    importance: float = typer.Option(0.5, help="Importance 0.0-1.0"),
    project: str | None = typer.Option(None, help="Project scope"),
):
    """Store a new memory."""
    engine = _get_engine(project)
    fact_id = engine.remember(text, importance=importance)
    if fact_id:
        console.print(f"[green]Remembered:[/green] {fact_id}")
    else:
        console.print("[yellow]Already known (duplicate).[/yellow]")


@app.command()
def recall(
    query: str = typer.Argument(help="Query to search memories"),
    top_k: int = typer.Option(5, help="Number of results"),
    project: str | None = typer.Option(None, help="Project scope"),
):
    """Recall relevant memories."""
    engine = _get_engine(project)
    results = engine.recall(query, top_k=top_k)

    if not results:
        console.print("[yellow]No memories found.[/yellow]")
        return

    for i, r in enumerate(results, 1):
        retention_pct = (r.get("retention") or 0) * 100
        console.print(
            f"[bold]{i}.[/bold] {r['text']}\n"
            f"   [dim]lifecycle={r['lifecycle']}  retention={retention_pct:.1f}%  "
            f"accesses={r['access_count']}[/dim]"
        )


@app.command()
def forget(
    fact_id: str = typer.Argument(help="Fact ID to forget"),
):
    """Forget a specific memory."""
    engine = _get_engine()
    engine.forget(fact_id)
    console.print(f"[red]Forgotten:[/red] {fact_id}")


@app.command()
def stats(
    project: str | None = typer.Option(None, help="Project scope"),
):
    """Show memory statistics."""
    engine = _get_engine(project)
    st = engine.stats()

    table = Table(title="Memory Stats")
    table.add_column("Lifecycle", style="bold")
    table.add_column("Count", justify="right")
    table.add_column("Avg Retention", justify="right")

    for lifecycle, data in sorted(st.items()):
        table.add_row(lifecycle, str(data["count"]), f"{data['avg_retention']:.1%}")

    console.print(table)


@app.command()
def prompts(
    project: str | None = typer.Option(None, help="Project scope"),
):
    """Show generated soft prompts."""
    engine = _get_engine(project)
    text = engine.get_soft_prompts()
    if text:
        console.print(text)
    else:
        console.print("[yellow]No soft prompts generated yet.[/yellow]")


@app.command()
def list_memories(
    lifecycle: str = typer.Option("Active", help="Filter by lifecycle"),
    limit: int = typer.Option(20, help="Max results"),
    project: str | None = typer.Option(None, help="Project scope"),
):
    """List memories by lifecycle state."""
    engine = _get_engine(project)
    memories = engine.list_memories(lifecycle=lifecycle, limit=limit)

    if not memories:
        console.print(f"[yellow]No {lifecycle} memories.[/yellow]")
        return

    table = Table(title=f"{lifecycle} Memories")
    table.add_column("ID", style="dim")
    table.add_column("Text")
    table.add_column("Retention", justify="right")
    table.add_column("Importance", justify="right")

    for m in memories:
        retention_pct = (m.get("retention") or 0) * 100
        table.add_row(
            m["id"][:8],
            m["text"][:60],
            f"{retention_pct:.1f}%",
            f"{m['importance']:.1f}",
        )

    console.print(table)


@app.command()
def dashboard():
    """Open the web dashboard."""
    from mnemo.dashboard.app import run_dashboard

    console.print("[green]Starting dashboard on http://127.0.0.1:8768[/green]")
    run_dashboard()


@app.command()
def daemon():
    """Start the daemon server."""
    from mnemo.daemon.server import run_daemon

    console.print("[green]Starting daemon on http://127.0.0.1:8767[/green]")
    run_daemon()


@app.command("install-hooks")
def install_hooks():
    """Install Claude Code hooks."""
    from install.postinstall import install_hooks as _install

    _install()


@app.command("session-start")
def session_start(
    project: str = typer.Option("", help="Project path"),
    json_output: bool = typer.Option(False, "--json", help="JSON output"),
):
    """Load memories for session start (used by hooks)."""
    engine = _get_engine(project or None)
    data = {
        "soft_prompts": engine.get_soft_prompts(),
        "recent_memories": engine.list_memories(lifecycle="Active", limit=5),
    }
    if json_output:
        print(json.dumps(data, default=str))
    else:
        if data["soft_prompts"]:
            console.print(data["soft_prompts"])
        for m in data["recent_memories"]:
            console.print(f"- {m['text']}")


@app.command("save-session")
def save_session(
    project: str = typer.Option("", help="Project path"),
    with_git_context: bool = typer.Option(False, "--with-git-context"),
):
    """Save session summary (used by hooks)."""
    import time
    import uuid

    from mnemo.storage.db import get_connection, transaction

    conn = get_connection()
    session_id = str(uuid.uuid4())
    now = time.time()

    git_branch = git_commit = None
    if with_git_context:
        import subprocess

        try:
            git_branch = (
                subprocess.check_output(
                    ["git", "rev-parse", "--abbrev-ref", "HEAD"],
                    stderr=subprocess.DEVNULL,
                )
                .decode()
                .strip()
            )
            git_commit = (
                subprocess.check_output(
                    ["git", "rev-parse", "HEAD"],
                    stderr=subprocess.DEVNULL,
                )
                .decode()
                .strip()
            )
        except Exception:
            pass

    with transaction(conn):
        conn.execute(
            """
            INSERT INTO sessions (id, project, started_at, ended_at, git_branch, git_commit)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (session_id, project or None, now, now, git_branch, git_commit),
        )


@app.command("observe")
def observe(
    tool: str = typer.Option("", help="Tool name"),
    file: str = typer.Option("", help="File path"),
):
    """Observe a tool use event (used by hooks)."""
    # Simple observation: store the file edit as a low-importance fact
    if file and tool in ("Write", "Edit", "MultiEdit"):
        try:
            engine = _get_engine()
            engine.remember(
                f"Edited {file} via {tool}",
                importance=0.2,
                source="hook",
            )
        except Exception:
            # Hooks must fail silently — never crash
            pass


if __name__ == "__main__":
    app()
