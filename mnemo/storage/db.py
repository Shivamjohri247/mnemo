"""SQLite connection manager for mnemo."""

import os

# Use pysqlite3 on Python 3.13+ where stdlib sqlite3 dropped extension loading
import sqlite3
import threading
from contextlib import contextmanager, suppress
from pathlib import Path

import sqlite_vec

try:
    _test_conn = sqlite3.connect(":memory:")
    if not hasattr(_test_conn, "enable_load_extension"):
        raise ImportError
    _test_conn.close()
except (ImportError, AttributeError):
    from pysqlite3 import dbapi2 as sqlite3  # type: ignore[no-redef]


def _data_dir() -> Path:
    return Path(os.environ.get("SLM_DATA_DIR", str(Path.home() / ".slm")))


DEFAULT_DB = "memory.db"

# Cached connections — one per (data_dir, db_name) pair
_connections: dict[str, sqlite3.Connection] = {}
_lock = threading.Lock()


def get_connection(db_name: str = DEFAULT_DB) -> sqlite3.Connection:
    """Get a cached SQLite connection with sqlite-vec loaded."""
    data_dir = _data_dir()
    cache_key = f"{data_dir}/{db_name}"

    with _lock:
        if cache_key in _connections:
            return _connections[cache_key]

        data_dir.mkdir(exist_ok=True)
        conn = sqlite3.connect(str(data_dir / db_name), check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.enable_load_extension(True)
        sqlite_vec.load(conn)
        conn.enable_load_extension(False)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        _connections[cache_key] = conn
        return conn


def reset_connections():
    """Close and discard all cached connections. Used by tests."""
    with _lock:
        for conn in _connections.values():
            with suppress(Exception):
                conn.close()
        _connections.clear()


@contextmanager
def transaction(conn):
    """Context manager for database transactions."""
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise


def init_schema(conn):
    """Initialize the database schema."""
    schema_path = Path(__file__).parent / "schema.sql"
    schema = schema_path.read_text()
    conn.executescript(schema)
    conn.commit()
