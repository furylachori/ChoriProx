"""Tests for proxy.eviction_coordinator."""
import time
from unittest.mock import MagicMock, patch

import pytest


def _make_mocks():
    """Create mock registry and workspace_store with proper lock context manager."""
    mock_registry = MagicMock()
    mock_ws_store = MagicMock()
    mock_registry._lock = MagicMock()
    mock_registry._lock.__enter__ = MagicMock(return_value=None)
    mock_registry._lock.__exit__ = MagicMock(return_value=False)
    return mock_registry, mock_ws_store


class TestBuildDeferredClears:
    def test_build_deferred_clears_identifies_sessions_needing_clear(self):
        from proxy.eviction_coordinator import _build_deferred_clears

        mock_registry, mock_ws_store = _make_mocks()

        agent = MagicMock()
        agent.snapshot_mode.return_value = "code"

        mock_registry._in_flight_requests = {}
        mock_registry._agents = {"s1": agent}
        mock_ws_store.get_last_access.return_value = time.time()
        mock_ws_store.pop_session = MagicMock()

        with patch.dict(
            "sys.modules",
            {
                "proxy.agent.agent_registry": MagicMock(registry=mock_registry),
                "proxy.workspace_state": MagicMock(workspace_store=mock_ws_store),
            },
        ):
            deferred = _build_deferred_clears(["s1"])

        assert len(deferred["mode_clears"]) == 1
        assert deferred["mode_clears"][0][0] is agent
        assert deferred["mode_clears"][0][1] == "code"
        assert deferred["mode_clears"][0][2] == "s1"
        assert deferred["popped_workspace_ids"] == []
        mock_ws_store.pop_session.assert_called_once_with("s1")

    def test_deferred_clears_skips_in_flight_sessions(self):
        from proxy.eviction_coordinator import _build_deferred_clears

        mock_registry, mock_ws_store = _make_mocks()

        agent = MagicMock()
        agent.snapshot_mode.return_value = "code"

        mock_registry._in_flight_requests = {"s1": 2}
        mock_registry._agents = {"s1": agent}
        mock_ws_store.get_last_access.return_value = 0.0
        mock_ws_store.pop_session = MagicMock()

        with patch.dict(
            "sys.modules",
            {
                "proxy.agent.agent_registry": MagicMock(registry=mock_registry),
                "proxy.workspace_state": MagicMock(workspace_store=mock_ws_store),
            },
        ):
            deferred = _build_deferred_clears(["s1"])

        assert deferred["mode_clears"] == []
        assert deferred["popped_workspace_ids"] == []
        mock_ws_store.pop_session.assert_not_called()

    def test_deferred_clears_calls_clear_active_mode(self):
        from proxy.eviction_coordinator import (
            _build_deferred_clears,
            _process_deferred_clears,
        )

        mock_registry, mock_ws_store = _make_mocks()

        agent = MagicMock()
        agent.snapshot_mode.return_value = "code"

        mock_registry._in_flight_requests = {}
        mock_registry._agents = {"s1": agent}
        mock_ws_store.get_last_access.return_value = 0.0
        mock_ws_store.pop_session = MagicMock()

        with patch.dict(
            "sys.modules",
            {
                "proxy.agent.agent_registry": MagicMock(registry=mock_registry),
                "proxy.workspace_state": MagicMock(workspace_store=mock_ws_store),
            },
        ):
            deferred = _build_deferred_clears(["s1"])

        _process_deferred_clears(deferred)
        agent.clear_active_mode.assert_called_once_with(expected_mode="code")


class TestLazyImports:
    def test_lazy_imports_no_circular_dependency(self):
        import proxy.eviction_coordinator
        assert proxy.eviction_coordinator is not None


class TestEmptyEvictedList:
    def test_empty_evicted_list_does_nothing(self):
        from proxy.eviction_coordinator import build_and_process_deferred_clears

        mock_registry, mock_ws_store = _make_mocks()
        mock_registry._in_flight_requests = {}
        mock_registry._agents = {}
        mock_ws_store.pop_session = MagicMock()

        with patch.dict(
            "sys.modules",
            {
                "proxy.agent.agent_registry": MagicMock(registry=mock_registry),
                "proxy.workspace_state": MagicMock(workspace_store=mock_ws_store),
            },
        ):
            result = build_and_process_deferred_clears([])

        # No-op contract: returns None and touches no side-effect functions.
        assert result is None
        mock_ws_store.pop_session.assert_not_called()


class TestWorkspaceStateCleanup:
    def test_workspace_state_cleaned_on_deferred_clear(self):
        from proxy.eviction_coordinator import _build_deferred_clears

        mock_registry, mock_ws_store = _make_mocks()

        mock_registry._in_flight_requests = {}
        mock_registry._agents = {}
        mock_ws_store.get_last_access.return_value = 0.0
        mock_ws_store.pop_session = MagicMock()

        with patch.dict(
            "sys.modules",
            {
                "proxy.agent.agent_registry": MagicMock(registry=mock_registry),
                "proxy.workspace_state": MagicMock(workspace_store=mock_ws_store),
            },
        ):
            _build_deferred_clears(["s1", "s2"])

        mock_ws_store.pop_session.assert_any_call("s1")
        mock_ws_store.pop_session.assert_any_call("s2")
        assert mock_ws_store.pop_session.call_count == 2

    def test_neo4j_edge_preserved_on_eviction(self):
        """LRU eviction must NOT call cleanup_neo4j_sessions — that would
        DELETE the [:LAST_USED_WORKSPACE] edge, breaking sticky-workspace
        recovery on the next access. Edge cleanup is reserved for explicit
        workspace leave/deactivate only.
        """
        from proxy.eviction_coordinator import (
            _build_deferred_clears,
            _process_deferred_clears,
        )

        mock_registry, mock_ws_store = _make_mocks()

        agent = MagicMock()
        agent.snapshot_mode.return_value = "code"

        mock_registry._in_flight_requests = {}
        mock_registry._agents = {"s1": agent}
        mock_ws_store.get_last_access.return_value = time.time()
        mock_ws_store.pop_session = MagicMock()

        with patch.dict(
            "sys.modules",
            {
                "proxy.agent.agent_registry": MagicMock(registry=mock_registry),
                "proxy.workspace_state": MagicMock(workspace_store=mock_ws_store),
            },
        ):
            deferred = _build_deferred_clears(["s1"])
            _process_deferred_clears(deferred)

        mock_ws_store.pop_session.assert_called_once_with("s1")
        assert deferred["popped_workspace_ids"] == []
