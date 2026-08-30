"""Integration tests for the VFS triple contract:

1. Write-through caching  — ToolPipeline._write_through_cache()
2. Read cache hits        — ToolPipeline._check_vfs_read_cache()
3. Rollback on failure    — PostProcessStage._scan_write_results()

Tests exercise the BOUNDARY between tool execution and cache management.
"""

import json
import os
import time
import hashlib
import pytest
from unittest.mock import MagicMock, patch

import proxy.state as state
from proxy.vfs import (
    VFSEntry, VFSCache,
    _cache_file, _cache_file_write_through, _cache_file_edit_through,
    _rollback_snapshot, _vfs_active_for_context,
    _get_or_create_workspace_cache, _is_expired,
)
from proxy.tool_pipeline.tool_pipeline import ToolPipeline, ToolContext
from proxy.tool_pipeline.tool_call import ToolCall
from proxy.tool_pipeline.tool_limits import FILE_BLOCKLIST_PATTERNS


NORM_APP_PY = os.path.normpath(os.path.realpath("src/app.py"))
WS_ID = "test-session:default"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def clean_vfs_state():
    """Reset VFS state before and after each test."""
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
    state.vfs_cache_hits = 0
    state.vfs_cache_misses = 0


@pytest.fixture
def vfs_enabled(monkeypatch):
    """Enable VFS with default settings."""
    monkeypatch.setattr(state, "VFS_ENABLED", True)
    monkeypatch.setattr(state, "VFS_SCOPE", "all")
    monkeypatch.setattr(state, "VFS_TTL_SECONDS", 60)
    monkeypatch.setattr(state, "VFS_MAX_ENTRIES", 500)
    monkeypatch.setattr(state, "VFS_MAX_ENTRY_BYTES", 1_048_576)
    monkeypatch.setattr(state, "VFS_MAX_WORKSPACES", 20)
    monkeypatch.setattr(state, "VFS_ROLLBACK_MAX", 100)
    monkeypatch.setattr(state, "VFS_ROLLBACK_TIMEOUT", 60)


@pytest.fixture
def pipeline():
    """Create a fresh ToolPipeline instance."""
    return ToolPipeline()


@pytest.fixture
def mock_workspace_id(monkeypatch):
    """Patch _resolve_workspace_id to return deterministic workspace."""
    monkeypatch.setattr(
        ToolPipeline, "_resolve_workspace_id",
        lambda self, session_id, client_hint="": WS_ID,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_tc(name, arguments, tc_id=None):
    """Create a ToolCall with given name and arguments."""
    if tc_id is None:
        tc_id = f"call_{name}_{hashlib.md5(json.dumps(arguments, sort_keys=True).encode()).hexdigest()[:8]}"
    return ToolCall(id=tc_id, name=name, arguments=arguments, source="llm")


def _make_response(tool_calls=None, content=""):
    """Build a minimal OpenAI-format response dict."""
    msg = {"role": "assistant", "content": content}
    if tool_calls:
        msg["tool_calls"] = [
            {
                "id": tc.id,
                "type": "function",
                "function": {
                    "name": tc.name,
                    "arguments": json.dumps(tc.arguments),
                },
            }
            for tc in tool_calls
        ]
    return {
        "choices": [{"message": msg, "finish_reason": "tool_calls" if tool_calls else "stop"}],
        "model": "test-model",
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    }


def _make_ctx(session_id="test-session", is_subagent=False, tool_calls_list=None):
    """Create a minimal ToolContext for testing."""
    body = {"model": "test-model", "messages": [{"role": "user", "content": "test"}]}
    return ToolContext(
        session_id=session_id,
        body=body,
        llama_body=body.copy(),
        auth_headers={},
        original_model="test-model",
        upstream_url="http://localhost:9999/v1",
        execute_fn=lambda name, args: {"content": "executed"},
        parse_tools_fn=lambda text: [],
        is_subagent=is_subagent,
        tool_calls_list=tool_calls_list,
    )


def _get_cache(workspace_id=WS_ID):
    """Get the VFSCache for a workspace (under lock)."""
    with state.vfs_lock:
        return state.vfs_caches.get(workspace_id)


def _get_entry(path, workspace_id=WS_ID):
    """Get a VFSEntry from the cache (under lock)."""
    with state.vfs_lock:
        cache = state.vfs_caches.get(workspace_id)
        if cache is None:
            return None
        return cache.get(path)


# ---------------------------------------------------------------------------
# Test 1: Write → VFS cache populated
# ---------------------------------------------------------------------------


class TestWriteThroughCache:
    """US-054: VFS Write-Through Cache Update — boundary contract.

    Test that write tool calls populate the VFS cache.
    """

    def test_write_populates_cache(self, vfs_enabled, pipeline, mock_workspace_id):
        """A write tool call should populate the VFS cache with the written content."""
        write_tc = _make_tc("write", {"file_path": "src/app.py", "content": "new content"})
        tool_calls = [write_tc]

        pipeline._write_through_cache(tool_calls, "test-session")

        entry = _get_entry(NORM_APP_PY)
        assert entry is not None
        assert entry.content == "new content"
        assert entry.verified is False  # write-through entries are unconfirmed

    def test_write_cache_key_matches_path(self, vfs_enabled, pipeline, mock_workspace_id):
        """The cache key should be the normalized absolute path."""
        write_tc = _make_tc("write", {"file_path": "src/app.py", "content": "data"})
        pipeline._write_through_cache([write_tc], "test-session")

        with state.vfs_lock:
            cache = state.vfs_caches[WS_ID]
            paths = [e.path for e in cache.snapshot_entries()]
        assert NORM_APP_PY in paths

    def test_write_creates_rollback_snapshot(self, vfs_enabled, pipeline, mock_workspace_id):
        """A write-through should create a rollback snapshot for the tool_call_id."""
        write_tc = _make_tc("write", {"file_path": "src/app.py", "content": "v1"},
                            tc_id="call_wt_001")
        pipeline._write_through_cache([write_tc], "test-session")

        with state.vfs_lock:
            snap = state.vfs_rollback_snapshots.get("call_wt_001")
        assert snap is not None
        assert snap.old_entry is None  # new file
        assert snap.path == NORM_APP_PY

    def test_write_over_existing_saves_old_entry(self, vfs_enabled, pipeline, mock_workspace_id):
        """Write over existing file should save old entry in rollback snapshot."""
        _cache_file("src/app.py", "old content", WS_ID, "test-session")

        write_tc = _make_tc("write", {"file_path": "src/app.py", "content": "new content"},
                            tc_id="call_wt_002")
        pipeline._write_through_cache([write_tc], "test-session")

        with state.vfs_lock:
            snap = state.vfs_rollback_snapshots.get("call_wt_002")
        assert snap is not None
        assert snap.old_entry is not None
        assert snap.old_entry.content == "old content"

    def test_write_multiple_files(self, vfs_enabled, pipeline, mock_workspace_id):
        """Multiple write tool calls should all populate the cache."""
        tc1 = _make_tc("write", {"file_path": "src/a.py", "content": "aaa"})
        tc2 = _make_tc("write", {"file_path": "src/b.py", "content": "bbb"})
        pipeline._write_through_cache([tc1, tc2], "test-session")

        norm_a = os.path.normpath(os.path.realpath("src/a.py"))
        norm_b = os.path.normpath(os.path.realpath("src/b.py"))
        assert _get_entry(norm_a).content == "aaa"
        assert _get_entry(norm_b).content == "bbb"

    def test_edit_populates_cache_on_existing(self, vfs_enabled, pipeline, mock_workspace_id):
        """An edit tool call should update the cache if the file is already cached."""
        _cache_file("src/app.py", "hello world", WS_ID, "test-session")

        edit_tc = _make_tc("edit", {
            "file_path": "src/app.py",
            "old_string": "world",
            "new_string": "universe",
        }, tc_id="call_et_001")
        pipeline._write_through_cache([edit_tc], "test-session")

        entry = _get_entry(NORM_APP_PY)
        assert entry.content == "hello universe"
        assert entry.verified is False


# ---------------------------------------------------------------------------
# Test 2: Read → VFS cache hit
# ---------------------------------------------------------------------------


class TestReadCacheHit:
    """US-053: VFS Read Cache Hit — No Disk I/O.

    Test that read tool calls hit the VFS cache when content is available.
    """

    def test_read_hits_cache_after_write(self, vfs_enabled, pipeline, mock_workspace_id):
        """After a write populates the cache, a read of the same file should hit the cache."""
        write_tc = _make_tc("write", {"file_path": "src/app.py", "content": "cached data"})
        pipeline._write_through_cache([write_tc], "test-session")

        read_tc = _make_tc("read", {"file_path": "src/app.py"})
        result = pipeline._check_vfs_read_cache(read_tc, "test-session")

        assert result is not None
        assert result.startswith("cached data")  # unverified entries get VFS WARNING appended

    def test_read_returns_none_on_miss(self, vfs_enabled, pipeline, mock_workspace_id):
        """Read of an uncached file should return None."""
        read_tc = _make_tc("read", {"file_path": "src/nonexistent.py"})
        result = pipeline._check_vfs_read_cache(read_tc, "test-session")
        assert result is None

    def test_read_returns_none_for_non_read_tool(self, vfs_enabled, pipeline, mock_workspace_id):
        """Non-read tools should return None from read cache check."""
        write_tc = _make_tc("write", {"file_path": "src/app.py", "content": "data"})
        pipeline._write_through_cache([write_tc], "test-session")

        bash_tc = _make_tc("bash", {"command": "cat src/app.py"})
        result = pipeline._check_vfs_read_cache(bash_tc, "test-session")
        assert result is None

    def test_read_returns_none_for_partial_read(self, vfs_enabled, pipeline, mock_workspace_id):
        """Partial reads (offset/limit) should not be served from cache."""
        _cache_file("src/app.py", "line1\nline2\nline3\n", WS_ID, "test-session")

        read_tc = _make_tc("read", {"file_path": "src/app.py", "offset": 1, "limit": 2})
        result = pipeline._check_vfs_read_cache(read_tc, "test-session")
        assert result is None

    def test_read_returns_none_for_expired_entry(self, vfs_enabled, pipeline, mock_workspace_id,
                                                  monkeypatch):
        """Expired entries should not be served from cache."""
        monkeypatch.setattr(state, "VFS_TTL_SECONDS", 1)
        _cache_file("src/app.py", "content", WS_ID, "test-session")

        # Age the entry past TTL
        with state.vfs_lock:
            cache = state.vfs_caches[WS_ID]
            entry = cache.get(NORM_APP_PY)
            aged = VFSEntry(
                path=entry.path, content=entry.content,
                content_hash=entry.content_hash, size_bytes=entry.size_bytes,
                cached_at=time.time() - 5, cached_by=entry.cached_by,
                workspace_id=entry.workspace_id, verified=entry.verified,
            )
            cache.put(aged)

        read_tc = _make_tc("read", {"file_path": "src/app.py"})
        result = pipeline._check_vfs_read_cache(read_tc, "test-session")
        assert result is None

    def test_read_cache_hit_increments_counter(self, vfs_enabled, pipeline, mock_workspace_id):
        """A cache hit should increment vfs_cache_hits."""
        _cache_file("src/app.py", "data", WS_ID, "test-session")

        read_tc = _make_tc("read", {"file_path": "src/app.py"})
        pipeline._check_vfs_read_cache(read_tc, "test-session")

        assert state.vfs_cache_hits == 1

    def test_read_cache_miss_increments_counter(self, vfs_enabled, pipeline, mock_workspace_id):
        """A cache miss should increment vfs_cache_misses."""
        read_tc = _make_tc("read", {"file_path": "src/missing.py"})
        pipeline._check_vfs_read_cache(read_tc, "test-session")

        assert state.vfs_cache_misses >= 1

    def test_read_content_matches_cached(self, vfs_enabled, pipeline, mock_workspace_id):
        """Read content must contain what was written (plus VFS WARNING for unconfirmed entries)."""
        content = "line1\nline2\nline3\n"
        write_tc = _make_tc("write", {"file_path": "src/app.py", "content": content})
        pipeline._write_through_cache([write_tc], "test-session")

        read_tc = _make_tc("read", {"file_path": "src/app.py"})
        result = pipeline._check_vfs_read_cache(read_tc, "test-session")

        assert content in result

    def test_read_from_prepopulated_cache(self, vfs_enabled, pipeline, mock_workspace_id):
        """Read cache hit works for entries populated via _cache_file (read results)."""
        _cache_file("src/app.py", "disk content", WS_ID, "test-session")

        read_tc = _make_tc("read", {"file_path": "src/app.py"})
        result = pipeline._check_vfs_read_cache(read_tc, "test-session")

        assert result == "disk content"

    def test_read_unverified_entry_includes_warning(self, vfs_enabled, pipeline, mock_workspace_id):
        """Unverified entries should include a VFS WARNING in the returned content."""
        write_tc = _make_tc("write", {"file_path": "src/app.py", "content": "pending"})
        pipeline._write_through_cache([write_tc], "test-session")

        read_tc = _make_tc("read", {"file_path": "src/app.py"})
        result = pipeline._check_vfs_read_cache(read_tc, "test-session")

        assert "pending" in result
        assert "[VFS WARNING" in result


# ---------------------------------------------------------------------------
# Test 3: Write failure → VFS rollback
# ---------------------------------------------------------------------------


class TestWriteFailureRollback:
    """US-055: VFS Rollback on Write Failure — boundary contract.

    Test that failed writes trigger VFS rollback.
    """

    def test_rollback_restores_old_content(self, vfs_enabled, pipeline, mock_workspace_id):
        """A failed write should roll back the cache to the previous content."""
        _cache_file("src/app.py", "original", WS_ID, "test-session")

        write_tc = _make_tc("write", {"file_path": "src/app.py", "content": "bad update"},
                            tc_id="call_rb_001")
        pipeline._write_through_cache([write_tc], "test-session")

        # Verify cache has the new (unconfirmed) content
        assert _get_entry(NORM_APP_PY).content == "bad update"

        # Simulate failure detection by _scan_write_results
        with state.vfs_lock:
            snap = state.vfs_rollback_snapshots.pop("call_rb_001")
        _rollback_snapshot(snap)

        entry = _get_entry(NORM_APP_PY)
        assert entry.content == "original"

    def test_rollback_new_file_removes_entry(self, vfs_enabled, pipeline, mock_workspace_id):
        """Failed write of a new file should remove the entry entirely."""
        write_tc = _make_tc("write", {"file_path": "src/new_file.py", "content": "should not stay"},
                            tc_id="call_rb_002")
        pipeline._write_through_cache([write_tc], "test-session")

        norm = os.path.normpath(os.path.realpath("src/new_file.py"))
        assert _get_entry(norm) is not None

        with state.vfs_lock:
            snap = state.vfs_rollback_snapshots.pop("call_rb_002")
        _rollback_snapshot(snap)

        assert _get_entry(norm) is None

    def test_rollback_no_stale_data(self, vfs_enabled, pipeline, mock_workspace_id):
        """After rollback, the cache should have no stale data from the failed write."""
        _cache_file("src/app.py", "clean state", WS_ID, "test-session")

        write_tc = _make_tc("write", {"file_path": "src/app.py", "content": "contamination"},
                            tc_id="call_rb_003")
        pipeline._write_through_cache([write_tc], "test-session")

        with state.vfs_lock:
            snap = state.vfs_rollback_snapshots.pop("call_rb_003")
        _rollback_snapshot(snap)

        entry = _get_entry(NORM_APP_PY)
        assert entry.content == "clean state"
        assert "contamination" not in entry.content

    def test_rollback_cascades_dependent_edits(self, vfs_enabled, pipeline, mock_workspace_id):
        """Rolling back a write should cascade to dependent edit snapshots."""
        _cache_file("src/app.py", "base", WS_ID, "test-session")

        write_tc = _make_tc("write", {"file_path": "src/app.py", "content": "write1"},
                            tc_id="call_chain_w1")
        pipeline._write_through_cache([write_tc], "test-session")

        edit_tc = _make_tc("edit", {
            "file_path": "src/app.py",
            "old_string": "write1",
            "new_string": "edit1",
        }, tc_id="call_chain_e1")
        pipeline._write_through_cache([edit_tc], "test-session")

        # Set parent for cascade tracking
        with state.vfs_lock:
            snap = state.vfs_rollback_snapshots.get("call_chain_e1")
            if snap:
                from proxy.vfs import VFSRollbackSnapshot
                state.vfs_rollback_snapshots["call_chain_e1"] = VFSRollbackSnapshot(
                    path=snap.path, workspace_id=snap.workspace_id,
                    old_entry=snap.old_entry, new_hash=snap.new_hash,
                    tool_call_id=snap.tool_call_id, created_at=snap.created_at,
                    parent_snapshot_id="call_chain_w1",
                )

        # Rollback the write
        with state.vfs_lock:
            snap = state.vfs_rollback_snapshots.pop("call_chain_w1")
        _rollback_snapshot(snap)

        # Dependent edit should also be removed
        with state.vfs_lock:
            assert "call_chain_e1" not in state.vfs_rollback_snapshots

    def test_post_process_rollback_on_error_result(self, vfs_enabled, mock_workspace_id):
        """PostProcessStage._scan_write_results rolls back when tool result is an error."""
        from proxy.pipeline.post_process import PostProcessStage

        _cache_file("src/app.py", "safe content", WS_ID, "test-session")

        _cache_file_write_through(NORM_APP_PY, "dangerous content", WS_ID, "test-session",
                                   tool_call_id="call_pp_001")

        messages = [
            {"role": "user", "content": "write the file"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [{
                    "id": "call_pp_001",
                    "type": "function",
                    "function": {
                        "name": "write",
                        "arguments": json.dumps({"file_path": "src/app.py", "content": "dangerous content"}),
                    },
                }],
            },
            {
                "role": "tool",
                "tool_call_id": "call_pp_001",
                "content": "Error: permission denied",
            },
        ]

        ctx = MagicMock()
        ctx.session_id = "test-session"
        ctx.messages_original = messages
        ctx.body = {"messages": messages}
        ctx.is_subagent = False

        stage = PostProcessStage()
        with patch.object(stage, '_resolve_vfs_workspace', return_value=WS_ID):
            stage._scan_write_results(ctx)

        entry = _get_entry(NORM_APP_PY)
        assert entry.content == "safe content"

    def test_post_process_confirms_on_success(self, vfs_enabled, mock_workspace_id):
        """PostProcessStage._scan_write_results confirms entry on successful write."""
        from proxy.pipeline.post_process import PostProcessStage

        _cache_file_write_through(NORM_APP_PY, "new content", WS_ID, "test-session",
                                   tool_call_id="call_pp_002")

        entry = _get_entry(NORM_APP_PY)
        assert entry.verified is False

        messages = [
            {"role": "user", "content": "write the file"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [{
                    "id": "call_pp_002",
                    "type": "function",
                    "function": {
                        "name": "write",
                        "arguments": json.dumps({"file_path": "src/app.py", "content": "new content"}),
                    },
                }],
            },
            {
                "role": "tool",
                "tool_call_id": "call_pp_002",
                "content": "File written successfully",
            },
        ]

        ctx = MagicMock()
        ctx.session_id = "test-session"
        ctx.messages_original = messages
        ctx.body = {"messages": messages}
        ctx.is_subagent = False

        stage = PostProcessStage()
        with patch.object(stage, '_resolve_vfs_workspace', return_value=WS_ID):
            stage._scan_write_results(ctx)

        entry = _get_entry(NORM_APP_PY)
        assert entry is not None
        assert entry.content == "new content"
        assert entry.verified is True

    def test_post_process_rolls_back_opencode_camelcase_write(self, vfs_enabled, mock_workspace_id):
        """PostProcessStage must recognize opencode's camelCase ``filePath`` argument.

        Without extract_file_path, the VFS rollback on failure silently misses
        the path, leaving a phantom write in the cache. Regression test for the
        "Tool-call argument normalization" landmine.
        """
        from proxy.pipeline.post_process import PostProcessStage

        _cache_file("src/app.py", "safe content", WS_ID, "test-session")

        _cache_file_write_through(NORM_APP_PY, "dangerous content", WS_ID, "test-session",
                                   tool_call_id="call_pp_opencode")

        messages = [
            {"role": "user", "content": "write the file"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [{
                    "id": "call_pp_opencode",
                    "type": "function",
                    "function": {
                        "name": "write",
                        # opencode emits camelCase filePath, NOT snake_case file_path
                        "arguments": json.dumps({"filePath": "src/app.py",
                                                 "content": "dangerous content"}),
                    },
                }],
            },
            {
                "role": "tool",
                "tool_call_id": "call_pp_opencode",
                "content": "Error: permission denied",
            },
        ]

        ctx = MagicMock()
        ctx.session_id = "test-session"
        ctx.messages_original = messages
        ctx.body = {"messages": messages}
        ctx.is_subagent = False

        stage = PostProcessStage()
        with patch.object(stage, '_resolve_vfs_workspace', return_value=WS_ID):
            stage._scan_write_results(ctx)

        # If extract_file_path wasn't used, the path would be "" and rollback
        # would silently skip, leaving "dangerous content" cached.
        entry = _get_entry(NORM_APP_PY)
        assert entry.content == "safe content", (
            "opencode's camelCase filePath MUST be recognized for VFS rollback"
        )


# ---------------------------------------------------------------------------
# Test 4: Bash tool → VFS cache invalidated
# ---------------------------------------------------------------------------


class TestBashInvalidation:
    """Test that bash tool calls invalidate the VFS cache."""

    def test_bash_invalidates_entire_cache(self, vfs_enabled, pipeline, mock_workspace_id):
        """A bash tool call should invalidate all cached entries."""
        _cache_file("src/app.py", "content1", WS_ID, "test-session")
        _cache_file("src/state.py", "content2", WS_ID, "test-session")

        bash_tc = _make_tc("bash", {"command": "rm -rf /"})
        pipeline._write_through_cache([bash_tc], "test-session")

        cache = _get_cache()
        assert cache is not None
        assert cache.size == 0

    def test_read_miss_after_bash(self, vfs_enabled, pipeline, mock_workspace_id):
        """After bash invalidation, a read should miss the cache."""
        write_tc = _make_tc("write", {"file_path": "src/app.py", "content": "data"})
        pipeline._write_through_cache([write_tc], "test-session")

        # Verify read hits
        read_tc = _make_tc("read", {"file_path": "src/app.py"})
        assert pipeline._check_vfs_read_cache(read_tc, "test-session") is not None

        # Bash invalidates
        bash_tc = _make_tc("bash", {"command": "echo hello"})
        pipeline._write_through_cache([bash_tc], "test-session")

        # Read should miss
        result = pipeline._check_vfs_read_cache(read_tc, "test-session")
        assert result is None

    def test_bash_before_write_in_batch_preserves_write(self, vfs_enabled, pipeline,
                                                         mock_workspace_id):
        """When bash and write are in the same batch, bash processes first (invalidate_all),
        then write re-populates. The write should survive."""
        bash_tc = _make_tc("bash", {"command": "echo hi"})
        write_tc = _make_tc("write", {"file_path": "src/app.py", "content": "survived"})
        pipeline._write_through_cache([bash_tc, write_tc], "test-session")

        entry = _get_entry(NORM_APP_PY)
        assert entry is not None
        assert entry.content == "survived"


# ---------------------------------------------------------------------------
# Test 5: Blocklisted paths not cached
# ---------------------------------------------------------------------------


class TestBlocklistedPaths:
    """US-056: VFS Blocklist Enforcement — boundary contract.

    Test that blocklisted file paths are never cached.
    """

    @pytest.mark.parametrize("path", [
        ".env",
        ".env.local",
        "config/.env",
        "certs/server.pem",
        "keys/id_rsa.pem",
        "credentials.json",
        "secret_key.txt",
        "auth_token.json",
        ".git/config",
        ".git/objects/abc",
        "master.password",
        "my.password",
    ])
    def test_blocklisted_write_not_cached(self, vfs_enabled, pipeline, mock_workspace_id, path):
        """Write to blocklisted paths should NOT populate the cache."""
        write_tc = _make_tc("write", {"file_path": path, "content": "SENSITIVE_DATA"})
        pipeline._write_through_cache([write_tc], "test-session")

        norm = os.path.normpath(os.path.realpath(path))
        entry = _get_entry(norm)
        assert entry is None, f"Blocklisted path '{path}' should not be cached"

    @pytest.mark.parametrize("path", [
        ".env",
        ".pem",
        "credentials",
        "secrets.yaml",
        "auth_token",
        ".git/HEAD",
        "master.password",
    ])
    def test_blocklisted_read_not_served(self, vfs_enabled, pipeline, mock_workspace_id, path):
        """Read of blocklisted paths should NOT be served from cache, even if somehow cached."""
        # Manually inject into cache (bypassing blocklist in _cache_file)
        norm = os.path.normpath(os.path.realpath(path))
        with state.vfs_lock:
            cache = _get_or_create_workspace_cache(WS_ID)
            cache.put(VFSEntry(
                path=norm, content="leaked", content_hash="h",
                size_bytes=6, cached_at=time.time(), cached_by="s1",
                workspace_id=WS_ID, verified=True,
            ))

        read_tc = _make_tc("read", {"file_path": path})
        result = pipeline._check_vfs_read_cache(read_tc, "test-session")
        assert result is None, f"Blocklisted path '{path}' should not be served from cache"

    def test_env_file_not_cached_via_write_through(self, vfs_enabled, pipeline, mock_workspace_id):
        """Write-through should skip .env files."""
        write_tc = _make_tc("write", {"file_path": ".env", "content": "SECRET=abc123"})
        pipeline._write_through_cache([write_tc], "test-session")

        cache = _get_cache()
        if cache is not None:
            norm_env = os.path.normpath(os.path.realpath(".env"))
            assert cache.get(norm_env) is None

    def test_pem_file_not_cached_via_write_through(self, vfs_enabled, pipeline, mock_workspace_id):
        """Write-through should skip .pem files."""
        write_tc = _make_tc("write", {"file_path": "server.pem", "content": "-----BEGIN CERTIFICATE-----"})
        pipeline._write_through_cache([write_tc], "test-session")

        cache = _get_cache()
        if cache is not None:
            norm_pem = os.path.normpath(os.path.realpath("server.pem"))
            assert cache.get(norm_pem) is None

    def test_credential_path_not_cached(self, vfs_enabled, pipeline, mock_workspace_id):
        """Write-through should skip paths containing 'credential'."""
        write_tc = _make_tc("write", {"file_path": "db_credentials", "content": "user:pass"})
        pipeline._write_through_cache([write_tc], "test-session")

        cache = _get_cache()
        if cache is not None:
            norm = os.path.normpath(os.path.realpath("db_credentials"))
            assert cache.get(norm) is None

    def test_edit_skips_blocklisted(self, vfs_enabled, pipeline, mock_workspace_id):
        """Edit-through should skip blocklisted paths."""
        # Manually cache a .env file (bypassing blocklist)
        norm_env = os.path.normpath(os.path.realpath(".env"))
        with state.vfs_lock:
            cache = _get_or_create_workspace_cache(WS_ID)
            cache.put(VFSEntry(
                path=norm_env, content="SECRET=old", content_hash="h1",
                size_bytes=10, cached_at=time.time(), cached_by="s1",
                workspace_id=WS_ID, verified=True,
            ))

        edit_tc = _make_tc("edit", {
            "file_path": ".env",
            "old_string": "old",
            "new_string": "new",
        })
        pipeline._write_through_cache([edit_tc], "test-session")

        # Should remain unchanged
        with state.vfs_lock:
            entry = state.vfs_caches[WS_ID].get(norm_env)
        assert entry.content == "SECRET=old"

    def test_blocklist_patterns_match_case_insensitive(self, vfs_enabled, pipeline,
                                                        mock_workspace_id):
        """Blocklist patterns should match case-insensitively."""
        write_tc = _make_tc("write", {"file_path": "MY.SECRET", "content": "data"})
        pipeline._write_through_cache([write_tc], "test-session")

        cache = _get_cache()
        if cache is not None:
            norm = os.path.normpath(os.path.realpath("MY.SECRET"))
            assert cache.get(norm) is None

    def test_write_content_not_in_read_cache(self, vfs_enabled, pipeline, mock_workspace_id):
        """Read cache for write-through entry contains the written content (not stale)."""
        write_tc = _make_tc("write", {"file_path": "src/app.py", "content": "fresh data"})
        pipeline._write_through_cache([write_tc], "test-session")

        read_tc = _make_tc("read", {"file_path": "src/app.py"})
        result = pipeline._check_vfs_read_cache(read_tc, "test-session")
        assert "fresh data" in result
        assert "stale" not in result
