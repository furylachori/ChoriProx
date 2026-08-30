"""Unit tests for VFS write-through caching, rollback, and helper functions."""

import os
import time
import hashlib
import pytest
import proxy.state as state
from proxy.vfs import (
    VFSEntry, VFSWriteRecord, VFSRollbackSnapshot, VFSCache,
    _is_expired, _vfs_active_for_context, _get_or_create_workspace_cache,
    _cache_file, _cache_file_write_through, _cache_file_edit_through,
    _log_write, _rollback_snapshot, _rollback_stale_snapshots,
    _rollback_oldest_snapshot, _looks_like_failure,
)

NORM_APP_PY = os.path.normpath(os.path.realpath("src/app.py"))


@pytest.fixture(autouse=True)
def clean_vfs_state():
    """Reset VFS state before each test."""
    with state.vfs_lock:
        state.vfs_caches.clear()
        state.vfs_write_log.clear()
        state.vfs_rollback_snapshots.clear()
    yield
    with state.vfs_lock:
        state.vfs_caches.clear()
        state.vfs_write_log.clear()
        state.vfs_rollback_snapshots.clear()


@pytest.fixture
def vfs_enabled(monkeypatch):
    """Ensure VFS is enabled with default scope."""
    monkeypatch.setattr(state, "VFS_ENABLED", True)
    monkeypatch.setattr(state, "VFS_SCOPE", "all")
    monkeypatch.setattr(state, "VFS_TTL_SECONDS", 60)
    monkeypatch.setattr(state, "VFS_MAX_ENTRIES", 500)
    monkeypatch.setattr(state, "VFS_MAX_ENTRY_BYTES", 1_048_576)
    monkeypatch.setattr(state, "VFS_MAX_WORKSPACES", 20)
    monkeypatch.setattr(state, "VFS_ROLLBACK_MAX", 100)
    monkeypatch.setattr(state, "VFS_ROLLBACK_TIMEOUT", 60)


def _make_entry(path="src/app.py", content="hello world", session="s1",
                workspace="ws1", verified=True, cached_at=None):
    return VFSEntry(
        path=path,
        content=content,
        content_hash=hashlib.sha256(content.encode()).hexdigest(),
        size_bytes=len(content),
        cached_at=cached_at or time.time(),
        cached_by=session,
        workspace_id=workspace,
        verified=verified,
    )


class TestIsExpired:
    def test_not_expired(self, vfs_enabled):
        entry = _make_entry(cached_at=time.time())
        assert not _is_expired(entry, 60)

    def test_expired(self, vfs_enabled):
        entry = _make_entry(cached_at=time.time() - 120)
        assert _is_expired(entry, 60)

    def test_zero_ttl(self, vfs_enabled):
        entry = _make_entry(cached_at=time.time() - 1)
        assert _is_expired(entry, 0)


class TestVfsActiveForContext:
    """US-057: VFS Scope Guard — _vfs_active_for_context branch coverage.

    Exercises all scope × subagent × enabled combinations.
    """
    def test_all_scope_main(self, vfs_enabled):
        assert _vfs_active_for_context(is_subagent=False) is True

    def test_all_scope_sub(self, vfs_enabled):
        assert _vfs_active_for_context(is_subagent=True) is True

    def test_subs_scope_main(self, vfs_enabled, monkeypatch):
        monkeypatch.setattr(state, "VFS_SCOPE", "subs")
        assert _vfs_active_for_context(is_subagent=False) is False

    def test_subs_scope_sub(self, vfs_enabled, monkeypatch):
        monkeypatch.setattr(state, "VFS_SCOPE", "subs")
        assert _vfs_active_for_context(is_subagent=True) is True

    def test_main_scope_main(self, vfs_enabled, monkeypatch):
        monkeypatch.setattr(state, "VFS_SCOPE", "main")
        assert _vfs_active_for_context(is_subagent=False) is True

    def test_main_scope_sub(self, vfs_enabled, monkeypatch):
        monkeypatch.setattr(state, "VFS_SCOPE", "main")
        assert _vfs_active_for_context(is_subagent=True) is False

    def test_disabled(self, vfs_enabled, monkeypatch):
        monkeypatch.setattr(state, "VFS_ENABLED", False)
        assert _vfs_active_for_context(is_subagent=False) is False
        assert _vfs_active_for_context(is_subagent=True) is False


class TestGetOrCreateWorkspaceCache:
    def test_create_new(self, vfs_enabled):
        with state.vfs_lock:
            cache = _get_or_create_workspace_cache("ws1")
        assert cache is not None
        assert isinstance(cache, VFSCache)
        assert "ws1" in state.vfs_caches

    def test_reuse_existing(self, vfs_enabled):
        with state.vfs_lock:
            c1 = _get_or_create_workspace_cache("ws1")
            c2 = _get_or_create_workspace_cache("ws1")
        assert c1 is c2

    def test_lru_eviction(self, vfs_enabled, monkeypatch):
        monkeypatch.setattr(state, "VFS_MAX_WORKSPACES", 2)
        with state.vfs_lock:
            _get_or_create_workspace_cache("ws1")
            _get_or_create_workspace_cache("ws2")
            _get_or_create_workspace_cache("ws3")
        assert "ws1" not in state.vfs_caches
        assert "ws2" in state.vfs_caches
        assert "ws3" in state.vfs_caches


class TestCacheFile:
    def test_basic(self, vfs_enabled):
        _cache_file("src/app.py", "hello world", "ws1", "s1")
        with state.vfs_lock:
            cache = state.vfs_caches.get("ws1")
        assert cache is not None
        entry = cache.get(NORM_APP_PY)
        assert entry is not None
        assert entry.content == "hello world"
        assert entry.verified is True

    def test_skip_duplicate(self, vfs_enabled):
        _cache_file("src/app.py", "hello", "ws1", "s1")
        _cache_file("src/app.py", "hello", "ws1", "s1")
        with state.vfs_lock:
            cache = state.vfs_caches.get("ws1")
        assert cache.size == 1

    def test_update_changed(self, vfs_enabled):
        _cache_file("src/app.py", "v1", "ws1", "s1")
        _cache_file("src/app.py", "v2", "ws1", "s1")
        with state.vfs_lock:
            cache = state.vfs_caches.get("ws1")
        entry = cache.get(NORM_APP_PY)
        assert entry.content == "v2"


class TestCacheFileWriteThrough:
    def test_basic_write(self, vfs_enabled):
        _cache_file_write_through("src/app.py", "new content", "ws1", "s1", "call_123")
        with state.vfs_lock:
            cache = state.vfs_caches.get("ws1")
            entry = cache.get("src/app.py")
            snap = state.vfs_rollback_snapshots.get("call_123")
        assert entry is not None
        assert entry.content == "new content"
        assert entry.verified is False
        assert snap is not None
        assert snap.old_entry is None

    def test_write_over_existing(self, vfs_enabled):
        with state.vfs_lock:
            cache = _get_or_create_workspace_cache("ws1")
            cache.put(_make_entry(path="src/app.py", content="old content"))
        _cache_file_write_through("src/app.py", "new content", "ws1", "s1", "call_456")
        with state.vfs_lock:
            snap = state.vfs_rollback_snapshots.get("call_456")
        assert snap is not None
        assert snap.old_entry is not None
        assert snap.old_entry.content == "old content"

    def test_write_log_populated(self, vfs_enabled):
        _cache_file_write_through("src/app.py", "content", "ws1", "s1")
        assert len(state.vfs_write_log) == 1
        assert state.vfs_write_log[0].operation == "write"


class TestCacheFileEditThrough:
    def test_basic_edit(self, vfs_enabled):
        with state.vfs_lock:
            cache = _get_or_create_workspace_cache("ws1")
            cache.put(_make_entry(path="src/app.py", content="hello world"))
        _cache_file_edit_through("src/app.py", "world", "universe", "ws1", "s1", tool_call_id="call_1")
        with state.vfs_lock:
            cache = state.vfs_caches.get("ws1")
            entry = cache.get("src/app.py")
        assert entry.content == "hello universe"
        assert entry.verified is False

    def test_edit_not_cached(self, vfs_enabled):
        _cache_file_edit_through("src/app.py", "old", "new", "ws1", "s1")
        with state.vfs_lock:
            assert "ws1" not in state.vfs_caches

    def test_edit_old_str_not_found(self, vfs_enabled):
        with state.vfs_lock:
            cache = _get_or_create_workspace_cache("ws1")
            cache.put(_make_entry(path="src/app.py", content="hello world"))
        _cache_file_edit_through("src/app.py", "nonexistent", "new", "ws1", "s1")
        with state.vfs_lock:
            entry = state.vfs_caches["ws1"].get("src/app.py")
        assert entry.content == "hello world"

    def test_replace_all(self, vfs_enabled):
        with state.vfs_lock:
            cache = _get_or_create_workspace_cache("ws1")
            cache.put(_make_entry(path="src/app.py", content="aaa bbb aaa"))
        _cache_file_edit_through("src/app.py", "aaa", "xxx", "ws1", "s1", replace_all=True)
        with state.vfs_lock:
            entry = state.vfs_caches["ws1"].get("src/app.py")
        assert entry.content == "xxx bbb xxx"


class TestRollback:
    def test_rollback_restores_old(self, vfs_enabled):
        with state.vfs_lock:
            cache = _get_or_create_workspace_cache("ws1")
            cache.put(_make_entry(path="src/app.py", content="old content"))
        _cache_file_write_through("src/app.py", "new content", "ws1", "s1", "call_1")

        with state.vfs_lock:
            snap = state.vfs_rollback_snapshots.pop("call_1")
        _rollback_snapshot(snap)

        with state.vfs_lock:
            entry = state.vfs_caches["ws1"].get("src/app.py")
        assert entry.content == "old content"

    def test_rollback_new_file(self, vfs_enabled):
        _cache_file_write_through("new.py", "content", "ws1", "s1", "call_1")
        with state.vfs_lock:
            snap = state.vfs_rollback_snapshots.pop("call_1")
        _rollback_snapshot(snap)
        with state.vfs_lock:
            entry = state.vfs_caches["ws1"].get("new.py")
        assert entry is None

    def test_rollback_skipped_when_superseded(self, vfs_enabled):
        _cache_file_write_through("src/app.py", "v1", "ws1", "s1", "call_1")
        _cache_file_write_through("src/app.py", "v2", "ws1", "s1", "call_2")

        with state.vfs_lock:
            snap = state.vfs_rollback_snapshots.pop("call_1")
        _rollback_snapshot(snap)

        with state.vfs_lock:
            entry = state.vfs_caches["ws1"].get("src/app.py")
        assert entry.content == "v2"


class TestRollbackStaleSnapshots:
    def test_stale_rollback(self, vfs_enabled, monkeypatch):
        monkeypatch.setattr(state, "VFS_ROLLBACK_TIMEOUT", 1)
        with state.vfs_lock:
            cache = _get_or_create_workspace_cache("ws1")
            cache.put(_make_entry(path="src/app.py", content="old"))
        _cache_file_write_through("src/app.py", "new", "ws1", "s1", "call_1")

        with state.vfs_lock:
            for snap in state.vfs_rollback_snapshots.values():
                object.__setattr__(snap, 'created_at', time.time() - 10)

        _rollback_stale_snapshots("ws1")

        with state.vfs_lock:
            assert len(state.vfs_rollback_snapshots) == 0
            entry = state.vfs_caches["ws1"].get("src/app.py")
        assert entry.content == "old"


class TestRollbackOldestSnapshot:
    def test_discard_oldest(self, vfs_enabled, monkeypatch):
        monkeypatch.setattr(state, "VFS_ROLLBACK_MAX", 2)
        _cache_file_write_through("a.py", "a", "ws1", "s1", "call_1")
        time.sleep(0.01)
        _cache_file_write_through("b.py", "b", "ws1", "s1", "call_2")

        with state.vfs_lock:
            assert len(state.vfs_rollback_snapshots) == 2
            _rollback_oldest_snapshot()
            assert len(state.vfs_rollback_snapshots) == 1
            assert "call_1" not in state.vfs_rollback_snapshots
            assert "call_2" in state.vfs_rollback_snapshots


class TestLooksLikeFailure:
    def test_success(self):
        assert _looks_like_failure("File written successfully") is False
        assert _looks_like_failure("Successfully updated foo.py") is False
        assert _looks_like_failure("Created new file") is False

    def test_failure(self):
        assert _looks_like_failure("Error: permission denied") is True
        assert _looks_like_failure("Failed to write file") is True
        assert _looks_like_failure("No such file or directory") is True

    def test_empty(self):
        assert _looks_like_failure("") is False
        assert _looks_like_failure(None) is False

    def test_leading_whitespace(self):
        assert _looks_like_failure("  File written successfully") is False

    def test_error_in_path(self):
        assert _looks_like_failure("Wrote error_handler.py successfully") is False


class TestLogWrite:
    def test_basic(self, vfs_enabled):
        _log_write("src/app.py", "s1", "ws1", "write", None, "abc123")
        assert len(state.vfs_write_log) == 1
        assert state.vfs_write_log[0].path == "src/app.py"
        assert state.vfs_write_log[0].operation == "write"

    def test_bounded(self, vfs_enabled, monkeypatch):
        monkeypatch.setattr(state, "VFS_WRITE_LOG_MAX", 5)
        for i in range(10):
            _log_write(f"file{i}.py", "s1", "ws1", "write", None, f"hash{i}")
        assert len(state.vfs_write_log) == 5
        assert state.vfs_write_log[0].path == "file5.py"
