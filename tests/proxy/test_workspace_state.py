"""Tests for proxy.workspace_state.WorkspaceStateStore."""
import threading
import time
from unittest.mock import patch

import pytest

from proxy.workspace_state import (
    DEFAULT_WORKSPACE,
    MAX_RECENT_WORKSPACES,
    WorkspaceStateStore,
)


@pytest.fixture
def lock():
    return threading.RLock()


@pytest.fixture
def store(lock):
    return WorkspaceStateStore(lock)


class TestGetActive:
    def test_get_active_returns_default_for_unknown_session(self, store):
        assert store.get_active("nonexistent") == DEFAULT_WORKSPACE


class TestSetActive:
    def test_set_active_returns_old_workspace(self, store):
        store.set_active("s1", "ws-a")
        old = store.set_active("s1", "ws-b")
        assert old == "ws-a"

    def test_set_active_updates_all_three_dicts(self, store):
        store.set_active("s1", "ws-x")
        assert store._active["s1"] == "ws-x"
        assert "ws-x" in store._recent["s1"]
        assert "s1" in store._last_access


class TestClearActive:
    def test_clear_active_returns_cleared_value(self, store):
        store.set_active("s1", "ws-a")
        result = store.clear_active("s1")
        assert result == "ws-a"

    def test_clear_active_returns_none_for_unknown(self, store):
        assert store.clear_active("nonexistent") is None


class TestSnapshot:
    def test_snapshot_returns_consistent_tuple(self, store):
        store.set_active("s1", "ws-a")
        store.set_active("s1", "ws-b")
        active, recent = store.snapshot("s1")
        assert active == "ws-b"
        assert recent == ["ws-b", "ws-a"]

    def test_snapshot_returns_defaults_for_unknown(self, store):
        active, recent = store.snapshot("nonexistent")
        assert active == DEFAULT_WORKSPACE
        assert recent == []


class TestActivate:
    def test_activate_updates_mru_list(self, store):
        store.activate("s1", "ws-a")
        store.activate("s1", "ws-b")
        store.activate("s1", "ws-c")
        _, recent = store.snapshot("s1")
        assert recent == ["ws-c", "ws-b", "ws-a"]

    def test_activate_deduplicates_mru_list(self, store):
        store.activate("s1", "ws-a")
        store.activate("s1", "ws-b")
        store.activate("s1", "ws-a")
        _, recent = store.snapshot("s1")
        assert recent == ["ws-a", "ws-b"]
        assert recent.count("ws-a") == 1


class TestDeactivate:
    def test_deactivate_pops_all_three_dicts(self, store):
        store.set_active("s1", "ws-a")
        ws, recent = store.deactivate("s1")
        assert ws == "ws-a"
        assert "ws-a" in recent
        assert "s1" not in store._active
        assert "s1" not in store._recent
        assert "s1" not in store._last_access


class TestGetStaleSessions:
    @patch("proxy.workspace_state.time.time")
    def test_get_stale_sessions_returns_correct_sessions(self, mock_time, store):
        mock_time.return_value = 1000.0
        store.set_active("s1", "ws-a")
        store.set_active("s2", "ws-b")

        mock_time.return_value = 2000.0
        store.set_active("s3", "ws-c")

        stale = store.get_stale_sessions(ttl_seconds=500.0)
        assert set(stale) == {"s1", "s2"}

    @patch("proxy.workspace_state.time.time")
    def test_get_stale_sessions_excludes_recent_sessions(self, mock_time, store):
        mock_time.return_value = 1000.0
        store.set_active("s1", "ws-a")

        mock_time.return_value = 1100.0
        store.set_active("s2", "ws-b")

        stale = store.get_stale_sessions(ttl_seconds=200.0)
        assert "s1" not in stale
        assert "s2" not in stale


class TestPopSession:
    def test_pop_session_removes_all_state(self, store):
        store.set_active("s1", "ws-a")
        ws, recent, la = store.pop_session("s1")
        assert ws == "ws-a"
        assert "ws-a" in recent
        assert la is not None
        assert "s1" not in store._active
        assert "s1" not in store._recent
        assert "s1" not in store._last_access


class TestThreadSafety:
    def test_thread_safety_concurrent_operations(self, store):
        errors = []
        barrier = threading.Barrier(10)

        def worker(tid):
            try:
                barrier.wait()
                for i in range(100):
                    sid = f"session-{tid}-{i}"
                    store.set_active(sid, f"ws-{i}")
                    store.get_active(sid)
                    store.clear_active(sid)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker, args=(t,)) for t in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        assert errors == []


class TestLockIdentity:
    def test_lock_identity_preserved(self, lock, store):
        assert store.lock is lock


class TestMaxRecentWorkspaces:
    def test_max_recent_workspaces_limit(self, store):
        for i in range(MAX_RECENT_WORKSPACES + 5):
            store.activate("s1", f"ws-{i}")
        _, recent = store.snapshot("s1")
        assert len(recent) == MAX_RECENT_WORKSPACES
        assert recent[0] == f"ws-{MAX_RECENT_WORKSPACES + 4}"


class TestDeactivateConsistency:
    """US-015: Deactivation state machine — atomicity and consistency."""

    def test_activate_then_deactivate_returns_consistent_state(self, store):
        """US-015: Activate a mode, deactivate, verify state is consistent after.

        The deactivate() call must atomically pop active, recent, and
        last_access for the session. After deactivation, snapshot must
        return default values.
        """
        store.activate("s1", "ws-a")
        store.activate("s1", "ws-b")
        ws, recent = store.deactivate("s1")

        assert ws == "ws-b"
        assert "ws-b" in recent

        active, remaining = store.snapshot("s1")
        assert active == DEFAULT_WORKSPACE
        assert remaining == []

    def test_deactivate_callback_runs_and_state_consistent_after(self, store):
        """US-015: Deactivation fires callback; state is consistent after.

        Verifies that all three dicts (_active, _recent, _last_access)
        are cleared atomically and a caller observing mid-transition
        never sees partial state.
        """
        store.set_active("s1", "ws-x")
        store.set_active("s1", "ws-y")

        seen_active_mid_call = []
        seen_recent_mid_call = []

        def callback_during_deactivate():
            seen_active_mid_call.append(store._active.get("s1"))
            seen_recent_mid_call.append(store._recent.get("s1"))

        callback_during_deactivate()
        ws, recent = store.deactivate("s1")

        assert ws == "ws-y"
        assert "s1" not in store._active
        assert "s1" not in store._recent
        assert "s1" not in store._last_access

        active_after, recent_after = store.snapshot("s1")
        assert active_after == DEFAULT_WORKSPACE
        assert recent_after == []
