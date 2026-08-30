"""Integration tests for the VFS (Virtual Filesystem Layer).

Tests the complete flow across multiple VFS components:
- Data structures (VFSEntry, VFSCache)
- Write-through caching
- Read interception
- Rollback on failure
- Cross-agent cache sharing
- Scope guards
"""

import os
import time
import hashlib
import pytest
import proxy.state as state
from proxy.vfs import (
    VFSEntry, VFSCache,
    _cache_file, _cache_file_write_through, _cache_file_edit_through,
    _rollback_snapshot, _rollback_stale_snapshots, _rollback_oldest_snapshot,
    _vfs_active_for_context, _is_expired, _looks_like_failure,
    _get_or_create_workspace_cache, _log_write,
)

APP = os.path.normpath(os.path.realpath("src/app.py"))
STATE_PY = os.path.normpath(os.path.realpath("src/state.py"))
PIPELINE = os.path.normpath(os.path.realpath("src/pipeline.py"))


@pytest.fixture(autouse=True)
def clean_vfs_state():
    with state.vfs_lock:
        state.vfs_caches.clear()
        state.vfs_write_log.clear()
        state.vfs_rollback_snapshots.clear()
    state.vfs_cache_hits = 0
    state.vfs_cache_misses = 0
    yield
    with state.vfs_lock:
        state.vfs_caches.clear()
        state.vfs_write_log.clear()
        state.vfs_rollback_snapshots.clear()


@pytest.fixture
def vfs_enabled(monkeypatch):
    monkeypatch.setattr(state, "VFS_ENABLED", True)
    monkeypatch.setattr(state, "VFS_SCOPE", "all")
    monkeypatch.setattr(state, "VFS_TTL_SECONDS", 60)
    monkeypatch.setattr(state, "VFS_MAX_ENTRIES", 500)
    monkeypatch.setattr(state, "VFS_MAX_ENTRY_BYTES", 1_048_576)
    monkeypatch.setattr(state, "VFS_MAX_WORKSPACES", 20)
    monkeypatch.setattr(state, "VFS_ROLLBACK_MAX", 100)
    monkeypatch.setattr(state, "VFS_ROLLBACK_TIMEOUT", 60)


class TestCrossAgentCacheSharing:
    """Test that sub-agents share the parent's workspace cache."""

    def test_shared_workspace(self, vfs_enabled):
        """Parent reads files, child sees them via shared workspace."""
        ws = "parent_session:default"

        _cache_file(APP, "app content", ws, "parent_session")
        _cache_file(STATE_PY, "state content", ws, "parent_session")
        _cache_file(PIPELINE, "pipeline content", ws, "parent_session")

        with state.vfs_lock:
            cache = state.vfs_caches.get(ws)
        assert cache is not None
        assert cache.size == 3

        with state.vfs_lock:
            entry = cache.get(APP)
        assert entry is not None
        assert entry.content == "app content"

    def test_write_through_visible_to_siblings(self, vfs_enabled):
        """Agent A writes, Agent B sees the update via shared cache."""
        ws = "parent:default"

        _cache_file(APP, "original", ws, "agent_a")
        _cache_file_write_through(APP, "agent_a_update", ws, "agent_a", "call_a1")

        with state.vfs_lock:
            cache = state.vfs_caches.get(ws)
            entry = cache.get(APP)
        assert entry.content == "agent_a_update"
        assert entry.verified is False

    def test_rollback_visible_to_siblings(self, vfs_enabled):
        """Agent A's failed write is rolled back, Agent B sees original."""
        ws = "parent:default"

        _cache_file(APP, "original", ws, "agent_a")
        _cache_file_write_through(APP, "bad_update", ws, "agent_a", "call_a1")

        with state.vfs_lock:
            snap = state.vfs_rollback_snapshots.pop("call_a1")
        _rollback_snapshot(snap)

        with state.vfs_lock:
            entry = state.vfs_caches[ws].get(APP)
        assert entry.content == "original"


class TestEditChainIntegrity:
    """Test that edit chains maintain cache consistency."""

    def test_sequential_edits(self, vfs_enabled):
        """Multiple sequential edits produce correct final content."""
        ws = "ws1"
        _cache_file(APP, "line1\nline2\nline3\n", ws, "s1")

        _cache_file_edit_through(APP, "line2", "line2_modified", ws, "s1", tool_call_id="e1")
        with state.vfs_lock:
            entry = state.vfs_caches[ws].get(APP)
        assert entry.content == "line1\nline2_modified\nline3\n"

        with state.vfs_lock:
            state.vfs_rollback_snapshots.pop("e1", None)

        _cache_file_edit_through(APP, "line3", "line3_modified", ws, "s1", tool_call_id="e2")
        with state.vfs_lock:
            entry = state.vfs_caches[ws].get(APP)
        assert entry.content == "line1\nline2_modified\nline3_modified\n"

    def test_edit_then_write(self, vfs_enabled):
        """Edit followed by write to same file."""
        ws = "ws1"
        _cache_file(APP, "original", ws, "s1")

        _cache_file_edit_through(APP, "original", "edited", ws, "s1", tool_call_id="e1")

        _cache_file_write_through(APP, "completely new", ws, "s1", "w1")

        with state.vfs_lock:
            entry = state.vfs_caches[ws].get(APP)
        assert entry.content == "completely new"

    def test_concurrent_edit_superseded(self, vfs_enabled):
        """Two writes where the second supersedes the first; rollback of first is no-op."""
        ws = "ws1"
        _cache_file(APP, "base content", ws, "s1")

        _cache_file_write_through(APP, "write1 result", ws, "s1", "w1")
        _cache_file_write_through(APP, "write2 result", ws, "s1", "w2")

        with state.vfs_lock:
            snap = state.vfs_rollback_snapshots.pop("w1", None)
        if snap:
            _rollback_snapshot(snap)

        with state.vfs_lock:
            entry = state.vfs_caches[ws].get(APP)
        assert entry.content == "write2 result"


class TestScopeGuardIntegration:
    """Test that scope guards work correctly across the VFS pipeline."""

    def test_subs_only_blocks_main(self, vfs_enabled, monkeypatch):
        monkeypatch.setattr(state, "VFS_SCOPE", "subs")
        assert _vfs_active_for_context(is_subagent=False) is False
        assert _vfs_active_for_context(is_subagent=True) is True

    def test_main_only_blocks_subs(self, vfs_enabled, monkeypatch):
        monkeypatch.setattr(state, "VFS_SCOPE", "main")
        assert _vfs_active_for_context(is_subagent=False) is True
        assert _vfs_active_for_context(is_subagent=True) is False

    def test_disabled_blocks_all(self, vfs_enabled, monkeypatch):
        monkeypatch.setattr(state, "VFS_ENABLED", False)
        assert _vfs_active_for_context(is_subagent=False) is False
        assert _vfs_active_for_context(is_subagent=True) is False


class TestWorkspaceEviction:
    """Test workspace-level LRU eviction."""

    def test_evict_oldest_workspace(self, vfs_enabled, monkeypatch):
        monkeypatch.setattr(state, "VFS_MAX_WORKSPACES", 3)

        with state.vfs_lock:
            _get_or_create_workspace_cache("ws1")
            _get_or_create_workspace_cache("ws2")
            _get_or_create_workspace_cache("ws3")

        assert len(state.vfs_caches) == 3

        with state.vfs_lock:
            _get_or_create_workspace_cache("ws1")

        with state.vfs_lock:
            _get_or_create_workspace_cache("ws4")

        assert "ws2" not in state.vfs_caches
        assert "ws1" in state.vfs_caches
        assert "ws3" in state.vfs_caches
        assert "ws4" in state.vfs_caches

    def test_zero_max_workspaces(self, vfs_enabled, monkeypatch):
        """VFS_MAX_WORKSPACES=0 should still create caches (no eviction)."""
        monkeypatch.setattr(state, "VFS_MAX_WORKSPACES", 0)

        with state.vfs_lock:
            cache = _get_or_create_workspace_cache("ws1")
        assert cache is not None
        assert "ws1" in state.vfs_caches


class TestTTLandExpiry:
    """Test TTL-based expiry across the VFS pipeline."""

    def test_expired_entry_not_served(self, vfs_enabled):
        """Expired entries should not be served from cache."""
        ws = "ws1"
        _cache_file(APP, "content", ws, "s1")

        with state.vfs_lock:
            cache = state.vfs_caches[ws]
            entry = cache.get(APP)
            expired = VFSEntry(
                path=entry.path, content=entry.content,
                content_hash=entry.content_hash, size_bytes=entry.size_bytes,
                cached_at=time.time() - 120,
                cached_by=entry.cached_by, workspace_id=entry.workspace_id,
            )
            cache.put(expired)

        with state.vfs_lock:
            entry = state.vfs_caches[ws].get(APP)
        assert _is_expired(entry, 60) is True

    def test_ttl_boundary(self, vfs_enabled):
        """Entry just within TTL should not be expired."""
        entry = VFSEntry(
            path="test.py", content="x", content_hash="h",
            size_bytes=1, cached_at=time.time() - 59,
            cached_by="s1", workspace_id="ws1",
        )
        assert _is_expired(entry, 60) is False


class TestWriteLogIntegration:
    """Test write log across multiple operations."""

    def test_write_and_edit_logged(self, vfs_enabled):
        ws = "ws1"
        _cache_file(APP, "original", ws, "s1")
        _cache_file_write_through(APP, "written", ws, "s1", "w1")
        _cache_file_edit_through(APP, "written", "edited", ws, "s1", tool_call_id="e1")

        assert len(state.vfs_write_log) == 2
        assert state.vfs_write_log[0].operation == "write"
        assert state.vfs_write_log[1].operation == "edit"

    def test_write_log_bounded(self, vfs_enabled, monkeypatch):
        monkeypatch.setattr(state, "VFS_WRITE_LOG_MAX", 5)
        ws = "ws1"
        for i in range(10):
            _log_write(f"file{i}.py", "s1", ws, "write", None, f"h{i}")
        assert len(state.vfs_write_log) == 5


class TestRollbackSnapshotCapacity:
    """Test rollback snapshot capacity management."""

    def test_oldest_discarded_at_capacity(self, vfs_enabled, monkeypatch):
        monkeypatch.setattr(state, "VFS_ROLLBACK_MAX", 3)
        ws = "ws1"

        _cache_file_write_through("a.py", "a", ws, "s1", "c1")
        time.sleep(0.01)
        _cache_file_write_through("b.py", "b", ws, "s1", "c2")
        time.sleep(0.01)
        _cache_file_write_through("c.py", "c", ws, "s1", "c3")

        with state.vfs_lock:
            assert len(state.vfs_rollback_snapshots) == 3

        _cache_file_write_through("d.py", "d", ws, "s1", "c4")

        with state.vfs_lock:
            assert len(state.vfs_rollback_snapshots) == 3
            assert "c1" not in state.vfs_rollback_snapshots
            assert "c4" in state.vfs_rollback_snapshots
