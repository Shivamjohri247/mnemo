"""Shared test fixtures for mnemo tests."""

import pytest

from mnemo.storage.db import sqlite3
from mnemo.testing import make_mock_embedder


@pytest.fixture(autouse=True)
def _reset_db_connections():
    """Reset cached DB connections between tests so each gets a fresh DB."""
    from mnemo.storage.db import reset_connections

    reset_connections()
    yield
    reset_connections()


@pytest.fixture
def tmp_data_dir(monkeypatch, tmp_path):
    """Redirect SLM_DATA_DIR to a temp directory."""
    data_dir = tmp_path / ".slm"
    data_dir.mkdir()
    monkeypatch.setenv("SLM_DATA_DIR", str(data_dir))
    return data_dir


@pytest.fixture
def mock_embedder():
    """Mock embedder that returns deterministic 384-dim vectors."""
    return make_mock_embedder()


@pytest.fixture
def in_memory_conn():
    """In-memory SQLite connection with sqlite-vec loaded."""
    import sqlite_vec

    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


@pytest.fixture
def initialized_conn(in_memory_conn):
    """In-memory connection with schema initialized."""
    from mnemo.storage.db import init_schema

    init_schema(in_memory_conn)
    return in_memory_conn
