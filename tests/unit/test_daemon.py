"""Tests for daemon client and server."""

import time
from unittest.mock import MagicMock, patch

import pytest


class TestDaemonClient:
    @pytest.fixture(autouse=True)
    def _reset_db(self):
        from mnemo.storage.db import reset_connections

        reset_connections()
        yield
        reset_connections()

    def test_is_running_returns_false_when_no_server(self):
        from mnemo.daemon.client import is_running

        assert is_running() is False

    def test_recall_via_daemon_returns_none_when_not_running(self):
        from mnemo.daemon.client import recall_via_daemon

        assert recall_via_daemon("test query") is None

    def test_remember_via_daemon_returns_none_when_not_running(self):
        from mnemo.daemon.client import remember_via_daemon

        assert remember_via_daemon("test fact") is None

    @patch("mnemo.daemon.client.httpx.post")
    @patch("mnemo.daemon.client.httpx.get")
    def test_remember_via_daemon_calls_api(self, mock_get, mock_post):
        from mnemo.daemon.client import remember_via_daemon

        mock_get.return_value = MagicMock(status_code=200)
        mock_post.return_value = MagicMock(status_code=200, json=lambda: {"fact_id": "abc-123"})
        result = remember_via_daemon("test fact", importance=0.8)
        assert result == "abc-123"

    @patch("mnemo.daemon.client.httpx.post")
    @patch("mnemo.daemon.client.httpx.get")
    def test_recall_via_daemon_calls_api(self, mock_get, mock_post):
        from mnemo.daemon.client import recall_via_daemon

        mock_get.return_value = MagicMock(status_code=200)
        mock_post.return_value = MagicMock(
            status_code=200, json=lambda: [{"id": "1", "text": "result"}]
        )
        result = recall_via_daemon("test query", top_k=3)
        assert len(result) == 1
        assert result[0]["text"] == "result"


class TestDaemonServerFunctions:
    """Test server endpoint functions directly to avoid lifespan issues."""

    def test_health_reports_uptime(self):
        import mnemo.daemon.server as server_mod

        server_mod.start_time = time.time() - 60
        server_mod.engine = None

        result = server_mod.health()
        assert result["status"] == "running"
        assert result["uptime_s"] >= 59

    def test_health_uptime_is_since_start_not_since_activity(self):
        import mnemo.daemon.server as server_mod

        server_mod.start_time = time.time() - 300
        server_mod.last_activity = time.time()
        server_mod.engine = None

        result = server_mod.health()
        assert result["uptime_s"] > 200

    def test_stats_with_no_engine(self):
        import mnemo.daemon.server as server_mod

        server_mod.engine = None
        result = server_mod.stats()
        assert result == {}

    def test_remember_with_no_engine(self):
        import mnemo.daemon.server as server_mod

        server_mod.engine = None
        result = server_mod.remember("test fact")
        assert result == {"fact_id": None}

    def test_recall_with_no_engine(self):
        import mnemo.daemon.server as server_mod

        server_mod.engine = None
        result = server_mod.recall("test query")
        assert result == []

    def test_touch_updates_activity(self):
        import mnemo.daemon.server as server_mod

        before = server_mod.last_activity
        time.sleep(0.01)
        server_mod.touch()
        assert server_mod.last_activity > before

    def test_stats_with_mock_engine(self):
        import mnemo.daemon.server as server_mod

        mock_engine = MagicMock()
        mock_engine.stats.return_value = {"Active": {"count": 5, "avg_retention": 0.9}}
        server_mod.engine = mock_engine

        result = server_mod.stats()
        assert result["Active"]["count"] == 5

        server_mod.engine = None

    def test_recall_with_mock_engine(self):
        import mnemo.daemon.server as server_mod

        mock_engine = MagicMock()
        mock_engine.recall.return_value = [{"id": "1", "text": "result"}]
        server_mod.engine = mock_engine

        result = server_mod.recall("query", top_k=5)
        assert len(result) == 1
        mock_engine.recall.assert_called_once_with("query", top_k=5)

        server_mod.engine = None

    def test_remember_with_mock_engine(self):
        import mnemo.daemon.server as server_mod

        mock_engine = MagicMock()
        mock_engine.remember.return_value = "fact-id-123"
        server_mod.engine = mock_engine

        result = server_mod.remember("new fact", importance=0.8, source="test")
        assert result == {"fact_id": "fact-id-123"}
        mock_engine.remember.assert_called_once_with("new fact", importance=0.8, source="test")

        server_mod.engine = None


class TestDaemonClientErrorHandling:
    """Test graceful degradation when server returns errors."""

    @patch("mnemo.daemon.client.httpx.post")
    @patch("mnemo.daemon.client.httpx.get")
    def test_remember_returns_none_on_server_error(self, mock_get, mock_post):
        from mnemo.daemon.client import remember_via_daemon

        mock_get.return_value = MagicMock(status_code=200)
        mock_post.side_effect = Exception("connection reset")
        result = remember_via_daemon("test fact")
        assert result is None

    @patch("mnemo.daemon.client.httpx.post")
    @patch("mnemo.daemon.client.httpx.get")
    def test_recall_returns_none_on_server_error(self, mock_get, mock_post):
        from mnemo.daemon.client import recall_via_daemon

        mock_get.return_value = MagicMock(status_code=200)
        mock_post.side_effect = Exception("timeout")
        result = recall_via_daemon("test query")
        assert result is None

    @patch("mnemo.daemon.client.httpx.post")
    @patch("mnemo.daemon.client.httpx.get")
    def test_remember_returns_none_on_http_error(self, mock_get, mock_post):
        from mnemo.daemon.client import remember_via_daemon

        mock_get.return_value = MagicMock(status_code=200)
        mock_post.side_effect = Exception("500")
        result = remember_via_daemon("test fact")
        assert result is None

    @patch("mnemo.daemon.client.httpx.post")
    @patch("mnemo.daemon.client.httpx.get")
    def test_recall_returns_none_on_http_error(self, mock_get, mock_post):
        from mnemo.daemon.client import recall_via_daemon

        mock_get.return_value = MagicMock(status_code=200)
        mock_post.side_effect = Exception("500")
        result = recall_via_daemon("test query")
        assert result is None
