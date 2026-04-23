"""Daemon client with fallback to direct engine."""

import httpx

HOST = "127.0.0.1"
PORT = 8767
DAEMON_URL = f"http://{HOST}:{PORT}"


def is_running() -> bool:
    try:
        httpx.get(f"{DAEMON_URL}/health", timeout=0.5)
        return True
    except Exception:
        return False


def recall_via_daemon(query: str, **kwargs) -> list[dict] | None:
    if not is_running():
        return None
    try:
        r = httpx.post(
            f"{DAEMON_URL}/recall",
            params={"query": query, **kwargs},
            timeout=5,
        )
        return list(r.json())  # type: ignore[no-any-return]
    except Exception:
        return None


def remember_via_daemon(text: str, **kwargs) -> str | None:
    if not is_running():
        return None
    try:
        r = httpx.post(
            f"{DAEMON_URL}/remember",
            params={"text": text, **kwargs},
            timeout=5,
        )
        return str(r.json().get("fact_id"))
    except Exception:
        return None
