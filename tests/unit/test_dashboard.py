"""Tests for the web dashboard."""

from unittest.mock import MagicMock, patch

import pytest

from mnemo.storage.db import get_connection, init_schema


@pytest.fixture
def dashboard_env(tmp_path, monkeypatch, mock_embedder):
    data_dir = tmp_path / ".slm"
    data_dir.mkdir()
    monkeypatch.setenv("SLM_DATA_DIR", str(data_dir))

    mock_nlp = MagicMock()
    mock_nlp.return_value.ents = []

    with (
        patch("mnemo.pipeline.encode.get_embedder", return_value=mock_embedder),
        patch("mnemo.pipeline.encode._get_nlp", return_value=mock_nlp),
    ):
        conn = get_connection()
        init_schema(conn)

        from mnemo.pipeline.encode import store_fact

        fid1 = store_fact("Dashboard fact one", importance=0.8, project="dash_test")
        fid2 = store_fact("Dashboard fact two", importance=0.5, project="dash_test")

        yield conn, {"fact_ids": [fid1, fid2]}


class TestDashboardIndex:
    def test_index_returns_all_facts(self, dashboard_env):
        from fastapi.testclient import TestClient

        from mnemo.dashboard.app import dash

        with TestClient(dash) as client:
            resp = client.get("/")
            assert resp.status_code == 200
            # Should contain our facts
            text = resp.text
            assert "Dashboard fact" in text

    def test_index_filters_by_lifecycle(self, dashboard_env):
        conn, ids = dashboard_env
        from fastapi.testclient import TestClient

        from mnemo.dashboard.app import dash

        with TestClient(dash) as client:
            resp = client.get("/?lifecycle=Active")
            assert resp.status_code == 200

    def test_index_handles_empty_lifecycle(self, dashboard_env):
        from fastapi.testclient import TestClient

        from mnemo.dashboard.app import dash

        with TestClient(dash) as client:
            resp = client.get("/?lifecycle=Nonexistent")
            assert resp.status_code == 200


class TestDashboardMemoryDetail:
    def test_memory_detail_returns_fact(self, dashboard_env):
        conn, ids = dashboard_env
        fid = [f for f in ids["fact_ids"] if f][0]
        from fastapi.testclient import TestClient

        from mnemo.dashboard.app import dash

        with TestClient(dash) as client:
            resp = client.get(f"/memory/{fid}")
            assert resp.status_code == 200
            assert "Dashboard fact" in resp.text

    def test_memory_detail_404_for_unknown(self, dashboard_env):
        from fastapi.testclient import TestClient

        from mnemo.dashboard.app import dash

        with TestClient(dash) as client:
            resp = client.get("/memory/nonexistent-id")
            assert resp.status_code == 404


class TestDashboardStats:
    def test_stats_returns_lifecycle_counts(self, dashboard_env):
        from fastapi.testclient import TestClient

        from mnemo.dashboard.app import dash

        with TestClient(dash) as client:
            resp = client.get("/stats")
            assert resp.status_code == 200
