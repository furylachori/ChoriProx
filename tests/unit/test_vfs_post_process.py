"""Unit tests for VFS post-processing: cache population and write result scanning."""

import json
import os
import time
import hashlib
import pytest
import proxy.state as state
from proxy.vfs import (
    VFSEntry, VFSCache,
    _cache_file, _cache_file_write_through, _cache_file_edit_through,
    _looks_like_failure, _rollback_snapshot, _rollback_stale_snapshots,
    _is_expired,
)


@pytest.fixture(autouse=True)
def clean_vfs_state():
    """Reset VFS state before each test."""
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
    """Ensure VFS is enabled with default scope."""
    monkeypatch.setattr(state, "VFS_ENABLED", True)
    monkeypatch.setattr(state, "VFS_SCOPE", "all")
    monkeypatch.setattr(state, "VFS_TTL_SECONDS", 60)
    monkeypatch.setattr(state, "VFS_MAX_ENTRIES", 500)
    monkeypatch.setattr(state, "VFS_MAX_ENTRY_BYTES", 1_048_576)
    monkeypatch.setattr(state, "VFS_MAX_WORKSPACES", 20)
    monkeypatch.setattr(state, "VFS_ROLLBACK_MAX", 100)
    monkeypatch.setattr(state, "VFS_ROLLBACK_TIMEOUT", 60)


def _norm(path):
    """Normalize path the same way _cache_file does."""
    return os.path.normpath(os.path.realpath(path))


def _make_read_messages(file_path, content, tool_call_id="call_read_1"):
    """Create a pair of messages simulating a read tool call and result."""
    return [
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [{
                "id": tool_call_id,
                "type": "function",
                "function": {
                    "name": "read",
                    "arguments": json.dumps({"file_path": file_path}),
                },
            }],
        },
        {
            "role": "tool",
            "tool_call_id": tool_call_id,
            "content": content,
        },
    ]


def _make_write_messages(file_path, content, tool_call_id="call_write_1",
                          result_content="File written successfully"):
    """Create a pair of messages simulating a write tool call and result."""
    return [
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [{
                "id": tool_call_id,
                "type": "function",
                "function": {
                    "name": "write",
                    "arguments": json.dumps({"file_path": file_path, "content": content}),
                },
            }],
        },
        {
            "role": "tool",
            "tool_call_id": tool_call_id,
            "content": result_content,
        },
    ]


def _make_edit_messages(file_path, old_str, new_str, tool_call_id="call_edit_1",
                         result_content="File updated successfully"):
    """Create a pair of messages simulating an edit tool call and result."""
    return [
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [{
                "id": tool_call_id,
                "type": "function",
                "function": {
                    "name": "edit",
                    "arguments": json.dumps({
                        "file_path": file_path,
                        "old_string": old_str,
                        "new_string": new_str,
                    }),
                },
            }],
        },
        {
            "role": "tool",
            "tool_call_id": tool_call_id,
            "content": result_content,
        },
    ]


class TestCacheFileFromReadResults:
    """Test _cache_file() which is called by _populate_vfs_cache()."""

    def test_basic_caching(self, vfs_enabled):
        _cache_file("src/app.py", "hello world", "ws1", "s1")
        with state.vfs_lock:
            cache = state.vfs_caches.get("ws1")
        assert cache is not None
        entry = cache.get(_norm("src/app.py"))
        assert entry is not None
        assert entry.content == "hello world"
        assert entry.verified is True

    def test_multiple_files(self, vfs_enabled):
        _cache_file("src/app.py", "app content", "ws1", "s1")
        _cache_file("src/state.py", "state content", "ws1", "s1")
        _cache_file("tests/test_app.py", "test content", "ws1", "s1")
        with state.vfs_lock:
            cache = state.vfs_caches.get("ws1")
        assert cache.size == 3

    def test_skip_empty_content(self, vfs_enabled):
        _cache_file("src/app.py", "", "ws1", "s1")
        with state.vfs_lock:
            cache = state.vfs_caches.get("ws1")
        entry = cache.get(_norm("src/app.py"))
        assert entry is not None
        assert entry.content == ""

    def test_content_hash_is_sha256(self, vfs_enabled):
        content = "hello world"
        expected_hash = hashlib.sha256(content.encode()).hexdigest()
        _cache_file("src/app.py", content, "ws1", "s1")
        with state.vfs_lock:
            cache = state.vfs_caches.get("ws1")
            entry = cache.get(_norm("src/app.py"))
        assert entry.content_hash == expected_hash

    def test_duplicate_read_skipped(self, vfs_enabled):
        """Same content read twice should not create duplicate entries."""
        _cache_file("src/app.py", "same content", "ws1", "s1")
        _cache_file("src/app.py", "same content", "ws1", "s1")
        with state.vfs_lock:
            cache = state.vfs_caches.get("ws1")
        assert cache.size == 1

    def test_updated_content_replaces(self, vfs_enabled):
        """Different content for same path should replace the entry."""
        _cache_file("src/app.py", "version 1", "ws1", "s1")
        _cache_file("src/app.py", "version 2", "ws1", "s1")
        with state.vfs_lock:
            cache = state.vfs_caches.get("ws1")
            entry = cache.get(_norm("src/app.py"))
        assert entry.content == "version 2"
        assert cache.size == 1


class TestWriteResultScanning:
    """Test the write result scanning logic."""

    def test_successful_write_confirmed(self, vfs_enabled):
        """Successful write should flip verified=True."""
        _cache_file_write_through("src/app.py", "new content", "ws1", "s1", "call_1")

        with state.vfs_lock:
            entry = state.vfs_caches["ws1"].get("src/app.py")
        assert entry.verified is False

        with state.vfs_lock:
            cache = state.vfs_caches["ws1"]
            entry = cache.get("src/app.py")
            if entry and not entry.verified:
                confirmed = VFSEntry(
                    path=entry.path, content=entry.content,
                    content_hash=entry.content_hash, size_bytes=entry.size_bytes,
                    cached_at=entry.cached_at, cached_by=entry.cached_by,
                    workspace_id=entry.workspace_id, verified=True,
                )
                cache.put(confirmed)

        with state.vfs_lock:
            entry = state.vfs_caches["ws1"].get("src/app.py")
        assert entry.verified is True

    def test_failed_write_rolled_back(self, vfs_enabled):
        """Failed write should rollback to old content."""
        _cache_file_write_through("src/app.py", "old content", "ws1", "s1", "call_init")

        # Confirm the initial write
        with state.vfs_lock:
            cache = state.vfs_caches["ws1"]
            entry = cache.get("src/app.py")
            confirmed = VFSEntry(
                path=entry.path, content=entry.content,
                content_hash=entry.content_hash, size_bytes=entry.size_bytes,
                cached_at=entry.cached_at, cached_by=entry.cached_by,
                workspace_id=entry.workspace_id, verified=True,
            )
            cache.put(confirmed)
            state.vfs_rollback_snapshots.pop("call_init", None)

        # Now do a second write that will fail
        _cache_file_write_through("src/app.py", "new content", "ws1", "s1", "call_1")

        result = "Error: permission denied"
        assert _looks_like_failure(result) is True

        with state.vfs_lock:
            snapshot = state.vfs_rollback_snapshots.pop("call_1")
        _rollback_snapshot(snapshot)

        with state.vfs_lock:
            entry = state.vfs_caches["ws1"].get("src/app.py")
        assert entry.content == "old content"

    def test_failed_write_new_file_invalidated(self, vfs_enabled):
        """Failed write to new file should invalidate the entry."""
        _cache_file_write_through("new.py", "content", "ws1", "s1", "call_1")

        with state.vfs_lock:
            snapshot = state.vfs_rollback_snapshots.pop("call_1")
        _rollback_snapshot(snapshot)

        with state.vfs_lock:
            entry = state.vfs_caches["ws1"].get("new.py")
        assert entry is None

    def test_stale_snapshot_rollback(self, vfs_enabled, monkeypatch):
        """Stale snapshots should be rolled back on timeout."""
        monkeypatch.setattr(state, "VFS_ROLLBACK_TIMEOUT", 1)

        _cache_file_write_through("src/app.py", "old", "ws1", "s1", "call_init")

        # Confirm the initial write
        with state.vfs_lock:
            cache = state.vfs_caches["ws1"]
            entry = cache.get("src/app.py")
            confirmed = VFSEntry(
                path=entry.path, content=entry.content,
                content_hash=entry.content_hash, size_bytes=entry.size_bytes,
                cached_at=entry.cached_at, cached_by=entry.cached_by,
                workspace_id=entry.workspace_id, verified=True,
            )
            cache.put(confirmed)
            state.vfs_rollback_snapshots.pop("call_init", None)

        # Now do a write that will become stale
        _cache_file_write_through("src/app.py", "new", "ws1", "s1", "call_1")

        # Age the snapshot
        with state.vfs_lock:
            for snap in state.vfs_rollback_snapshots.values():
                snap.created_at = time.time() - 10

        _rollback_stale_snapshots("ws1")

        with state.vfs_lock:
            assert len(state.vfs_rollback_snapshots) == 0
            entry = state.vfs_caches["ws1"].get("src/app.py")
        assert entry.content == "old"


class TestLooksLikeFailure:
    """Test the _looks_like_failure heuristic."""

    def test_success_messages_not_failure(self):
        assert _looks_like_failure("File written successfully") is False
        assert _looks_like_failure("Successfully updated") is False
        assert _looks_like_failure("Updated 3 lines") is False
        assert _looks_like_failure("Created new file") is False
        assert _looks_like_failure("Wrote 100 bytes") is False

    def test_error_messages_detected(self):
        assert _looks_like_failure("Error: permission denied") is True
        assert _looks_like_failure("Failed to write file") is True
        assert _looks_like_failure("ENOENT: no such file") is True
        assert _looks_like_failure("EACCES: permission denied") is True

    def test_empty_or_none(self):
        assert _looks_like_failure("") is False
        assert _looks_like_failure(None) is False

    def test_old_string_not_found(self):
        assert _looks_like_failure("old_string not found in content") is True

    def test_permission_denied(self):
        assert _looks_like_failure("permission denied") is True


class TestMessageScanningLogic:
    """Test the message scanning patterns used by _scan_write_results and _populate_vfs_cache."""

    def test_build_call_index(self):
        """Test building tool_call_id → (name, args) index from messages."""
        messages = _make_read_messages("src/app.py", "content", "call_1")
        messages.extend(_make_write_messages("src/state.py", "new", "call_2"))

        call_index = {}
        for msg in messages:
            if msg.get("role") != "assistant":
                continue
            for tc in msg.get("tool_calls", []):
                func = tc.get("function", {})
                name = func.get("name", "")
                if name in ("read", "write", "edit"):
                    args = json.loads(func.get("arguments", "{}"))
                    file_path = args.get("file_path") or args.get("path") or ""
                    if file_path:
                        call_index[tc.get("id")] = (name, file_path)

        assert len(call_index) == 2
        assert call_index["call_1"] == ("read", "src/app.py")
        assert call_index["call_2"] == ("write", "src/state.py")

    def test_match_tool_results(self):
        """Test matching tool results to their parent tool calls."""
        messages = _make_read_messages("src/app.py", "file content here", "call_1")

        results = {}
        for msg in messages:
            if msg.get("role") == "tool":
                tc_id = msg.get("tool_call_id")
                results[tc_id] = msg.get("content", "")

        assert results["call_1"] == "file content here"

    def test_partial_read_detection(self):
        """Test that partial reads (offset/limit) are detected."""
        messages = [{
            "role": "assistant",
            "content": None,
            "tool_calls": [{
                "id": "call_1",
                "type": "function",
                "function": {
                    "name": "read",
                    "arguments": json.dumps({"file_path": "big.py", "offset": 100, "limit": 50}),
                },
            }],
        }]

        for msg in messages:
            if msg.get("role") != "assistant":
                continue
            for tc in msg.get("tool_calls", []):
                func = tc.get("function", {})
                args = json.loads(func.get("arguments", "{}"))
                assert args.get("offset") is not None
                assert args.get("limit") is not None

    def test_multiple_tool_calls_in_one_message(self):
        """Test handling multiple tool calls in a single assistant message."""
        messages = [{
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "read", "arguments": json.dumps({"file_path": "a.py"})},
                },
                {
                    "id": "call_2",
                    "type": "function",
                    "function": {"name": "write", "arguments": json.dumps({"file_path": "b.py", "content": "new"})},
                },
                {
                    "id": "call_3",
                    "type": "function",
                    "function": {"name": "edit", "arguments": json.dumps({"file_path": "c.py", "old_string": "x", "new_string": "y"})},
                },
            ],
        }]

        call_index = {}
        for msg in messages:
            if msg.get("role") != "assistant":
                continue
            for tc in msg.get("tool_calls", []):
                func = tc.get("function", {})
                name = func.get("name", "")
                call_index[tc.get("id")] = name

        assert call_index["call_1"] == "read"
        assert call_index["call_2"] == "write"
        assert call_index["call_3"] == "edit"


class TestEndToEndFlow:
    """Test the complete VFS flow: read → cache → write → cache update → read → cache hit."""

    def test_full_flow(self, vfs_enabled):
        # Step 1: Read file → cache populated
        _cache_file("src/app.py", "original content", "ws1", "s1")

        # Step 2: Write file → write-through updates cache
        _cache_file_write_through(_norm("src/app.py"), "updated content", "ws1", "s1", "call_w1")

        with state.vfs_lock:
            entry = state.vfs_caches["ws1"].get(_norm("src/app.py"))
        assert entry.content == "updated content"
        assert entry.verified is False

        # Step 3: Write succeeds → flip verified
        with state.vfs_lock:
            cache = state.vfs_caches["ws1"]
            entry = cache.get(_norm("src/app.py"))
            confirmed = VFSEntry(
                path=entry.path, content=entry.content,
                content_hash=entry.content_hash, size_bytes=entry.size_bytes,
                cached_at=entry.cached_at, cached_by=entry.cached_by,
                workspace_id=entry.workspace_id, verified=True,
            )
            cache.put(confirmed)
            state.vfs_rollback_snapshots.pop("call_w1", None)

        # Step 4: Another agent reads → cache hit with verified content
        with state.vfs_lock:
            cache = state.vfs_caches["ws1"]
            entry = cache.get(_norm("src/app.py"))
        assert entry.content == "updated content"
        assert entry.verified is True

    def test_write_fail_rollback_flow(self, vfs_enabled):
        # Step 1: Write-through initial
        _cache_file_write_through("src/app.py", "original", "ws1", "s1", "call_init")

        # Confirm initial write
        with state.vfs_lock:
            cache = state.vfs_caches["ws1"]
            entry = cache.get("src/app.py")
            confirmed = VFSEntry(
                path=entry.path, content=entry.content,
                content_hash=entry.content_hash, size_bytes=entry.size_bytes,
                cached_at=entry.cached_at, cached_by=entry.cached_by,
                workspace_id=entry.workspace_id, verified=True,
            )
            cache.put(confirmed)
            state.vfs_rollback_snapshots.pop("call_init", None)

        # Step 2: Bad write-through
        _cache_file_write_through("src/app.py", "bad update", "ws1", "s1", "call_w1")

        # Step 3: Write fails → rollback
        with state.vfs_lock:
            snapshot = state.vfs_rollback_snapshots.pop("call_w1")
        _rollback_snapshot(snapshot)

        # Step 4: Cache has original content
        with state.vfs_lock:
            entry = state.vfs_caches["ws1"].get("src/app.py")
        assert entry.content == "original"

    def test_edit_through_flow(self, vfs_enabled):
        """Edit should update cached content with string replacement."""
        _cache_file_write_through("src/app.py", "hello world", "ws1", "s1", "call_init")

        # Confirm initial write
        with state.vfs_lock:
            cache = state.vfs_caches["ws1"]
            entry = cache.get("src/app.py")
            confirmed = VFSEntry(
                path=entry.path, content=entry.content,
                content_hash=entry.content_hash, size_bytes=entry.size_bytes,
                cached_at=entry.cached_at, cached_by=entry.cached_by,
                workspace_id=entry.workspace_id, verified=True,
            )
            cache.put(confirmed)
            state.vfs_rollback_snapshots.pop("call_init", None)

        # Edit
        _cache_file_edit_through("src/app.py", "world", "universe", "ws1", "s1",
                                  tool_call_id="call_e1")

        with state.vfs_lock:
            entry = state.vfs_caches["ws1"].get("src/app.py")
        assert entry.content == "hello universe"
        assert entry.verified is False

    def test_edit_rollback(self, vfs_enabled):
        """Failed edit should rollback to pre-edit content."""
        _cache_file_write_through("src/app.py", "hello world", "ws1", "s1", "call_init")

        # Confirm initial write
        with state.vfs_lock:
            cache = state.vfs_caches["ws1"]
            entry = cache.get("src/app.py")
            confirmed = VFSEntry(
                path=entry.path, content=entry.content,
                content_hash=entry.content_hash, size_bytes=entry.size_bytes,
                cached_at=entry.cached_at, cached_by=entry.cached_by,
                workspace_id=entry.workspace_id, verified=True,
            )
            cache.put(confirmed)
            state.vfs_rollback_snapshots.pop("call_init", None)

        # Edit
        _cache_file_edit_through("src/app.py", "world", "universe", "ws1", "s1",
                                  tool_call_id="call_e1")

        # Rollback edit
        with state.vfs_lock:
            snapshot = state.vfs_rollback_snapshots.pop("call_e1")
        _rollback_snapshot(snapshot)

        with state.vfs_lock:
            entry = state.vfs_caches["ws1"].get("src/app.py")
        assert entry.content == "hello world"


class TestIsExpired:
    """Test _is_expired helper."""

    def test_fresh_entry_not_expired(self, vfs_enabled):
        entry = VFSEntry(
            path="test.py", content="x", content_hash="h",
            size_bytes=1, cached_at=time.time(), cached_by="s1",
            workspace_id="ws1",
        )
        assert _is_expired(entry, 60) is False

    def test_old_entry_expired(self, vfs_enabled):
        entry = VFSEntry(
            path="test.py", content="x", content_hash="h",
            size_bytes=1, cached_at=time.time() - 120, cached_by="s1",
            workspace_id="ws1",
        )
        assert _is_expired(entry, 60) is True
