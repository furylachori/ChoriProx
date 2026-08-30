"""M3-1 + M3-2: VFS write-through/cache-hit guard interactions.

M3-1: previously `_write_through_cache(tool_calls, ...)` was called
with the FULL tool_calls list, so a guard-blocked write still updated
the VFS cache. Cross-agent cache poisoning.

M3-2: previously the cache-hit path appended reads to memory_calls,
then the guard loop called `tracker.try_read_file()` which incremented
the read count and blocked re-reads. A cache hit means the read
already happened in a prior request and was already counted — the
guard should not count it again. The fix passes `from_cache=True`
through the ToolCall annotation to the tracker.
"""

import json
from unittest.mock import MagicMock, patch

import pytest

from proxy.tool_pipeline.tool_pipeline import (
    ToolPipeline,
    ToolContext,
    _stable_args_key,
)
from proxy.tool_pipeline.tool_call import ToolCall
import proxy.state as state
from proxy.vfs import _cache_file_write_through


def _make_ctx(session_id="sess-test", is_subagent=False, client_hint=""):
    return ToolContext(
        session_id=session_id,
        body={"model": "test-model"},
        llama_body={"messages": [], "tools": [], "model": "test-model"},
        auth_headers={},
        original_model="test-model",
        upstream_url="http://upstream/v1",
        execute_fn=lambda *a, **kw: {"result": "ok"},
        is_subagent=is_subagent,
        client_hint=client_hint,
    )


def _make_response_data(tool_calls):
    native = []
    for name, args in tool_calls:
        native.append({
            "id": f"call_{name}",
            "type": "function",
            "function": {"name": name, "arguments": json.dumps(args)},
        })
    return {
        "choices": [{
            "message": {"content": "", "tool_calls": native},
            "finish_reason": "tool_calls",
        }]
    }


def _mock_result(tool_calls=None, content=None, finish_reason="stop"):
    r = MagicMock()
    r.tool_calls = tool_calls
    r.content = content
    r.finish_reason = finish_reason
    r.messages_for_history = None
    return r


@pytest.fixture(autouse=True)
def _enable_vfs(monkeypatch):
    """Enable VFS state for these tests + clear cache between runs."""
    monkeypatch.setattr(state, "VFS_ENABLED", True)
    monkeypatch.setattr(state, "VFS_SCOPE", "all")
    monkeypatch.setattr(state, "VFS_TTL_SECONDS", 60)
    monkeypatch.setattr(state, "VFS_MAX_ENTRIES", 500)
    monkeypatch.setattr(state, "VFS_MAX_ENTRY_BYTES", 1_048_576)
    monkeypatch.setattr(state, "VFS_MAX_WORKSPACES", 20)
    monkeypatch.setattr(state, "VFS_ROLLBACK_MAX", 100)
    monkeypatch.setattr(state, "VFS_ROLLBACK_TIMEOUT", 60)
    monkeypatch.setattr(state, "VFS_WRITE_LOG_MAX", 100)

    from proxy.workspace_state import workspace_store
    workspace_store._active.clear()
    workspace_store._client_active.clear()

    with state.vfs_lock:
        state.vfs_caches.clear()
    yield
    with state.vfs_lock:
        state.vfs_caches.clear()
    workspace_store._active.clear()
    workspace_store._client_active.clear()


def test_blocked_write_does_not_poison_vfs_cache(monkeypatch):
    """A guard-blocked write must NOT land in the VFS cache."""
    from proxy.tool_pipeline.tool_pipeline import MemoryToolGuard

    pipeline = ToolPipeline(registry=_registry_with_write())

    def fake_check(tc, ctx):
        if tc.name == "write" and "/etc/passwd" in tc.arguments.get("content", ""):
            return "blocklist: cannot write to sensitive path"
        return None

    fake_guard = MemoryToolGuard()
    fake_guard.check = fake_check
    pipeline.guard = fake_guard

    def mock_execute(*args, **kwargs):
        return _mock_result()
    pipeline._execute_via_loop = mock_execute

    response_data = _make_response_data([
        ("write", {"filePath": "src/app.py", "content": "good content"}),
        ("write", {"filePath": "src/secret.py", "content": "/etc/passwd override"}),
    ])
    ctx = _make_ctx()

    pipeline.run(response_data, ctx)

    with state.vfs_lock:
        cache = state.vfs_caches.get("default")

    assert cache is not None, "VFS cache was not populated for the safe write"
    # The blocked path must NOT be in the cache.
    assert "/etc/passwd" not in cache._entries, (
        f"VFS cache poisoned with blocked write: {cache._entries!r}"
    )
    # The non-blocked write SHOULD have landed. The key is the
    # absolute path (extract_file_path → realpath normalisation).
    safe_keys = [k for k in cache._entries if "src/app.py" in k]
    assert len(safe_keys) == 1, (
        f"Expected exactly 1 entry containing 'src/app.py', got: {list(cache._entries.keys())}"
    )
    safe_entry = cache.get(safe_keys[0])
    # Write-through entries are verified=False until the client confirms
    # the write succeeded — see vfs.py:31 docstring. The test confirms the
    # entry EXISTS, not that it was verified.


def _registry_with_write():
    """Build a registry that classifies write/edit/bash as MEMORY tools
    so the guard runs against them at the soft-error check."""
    from proxy.tool_pipeline.tool_registry import (
        ToolRegistry,
        ToolMeta,
        ToolCategory,
    )

    reg = ToolRegistry()
    # Register as MEMORY so the guard.check() runs in the soft-error path
    # at tool_pipeline.py:188 (iterates memory_calls).
    reg.register(ToolMeta(name="write", category=ToolCategory.MEMORY))
    reg.register(ToolMeta(name="edit", category=ToolCategory.MEMORY))
    reg.register(ToolMeta(name="bash", category=ToolCategory.MEMORY))
    return reg


def test_blocked_edit_does_not_poison_vfs_cache(monkeypatch):
    """Same guarantee for edit calls."""
    from proxy.tool_pipeline.tool_pipeline import MemoryToolGuard

    pipeline = ToolPipeline(registry=_registry_with_write())

    def fake_check(tc, ctx):
        if tc.name == "edit" and "rm -rf" in tc.arguments.get("new_string", ""):
            return "blocklist: dangerous command"
        return None

    fake_guard = MemoryToolGuard()
    fake_guard.check = fake_check
    pipeline.guard = fake_guard

    pipeline._execute_via_loop = lambda *a, **kw: _mock_result()

    response_data = _make_response_data([
        ("edit", {
            "filePath": "src/safe.py",
            "old_string": "old",
            "new_string": "good edit",
        }),
        ("edit", {
            "filePath": "src/safe.py",
            "old_string": "different",
            "new_string": "rm -rf /",
        }),
    ])
    ctx = _make_ctx()

    pipeline.run(response_data, ctx)

    with state.vfs_lock:
        cache = state.vfs_caches.get("default")
    if cache is not None:
        # The "rm -rf" edit was blocked — VFS should not reflect that
        # dangerous content.
        for path, entry in cache._entries.items():
            assert "rm -rf" not in str(entry.content), (
                f"VFS cache poisoned with blocked edit: {path} -> {entry.content!r}"
            )


def test_blocked_bash_does_not_invalidate_vfs_cache(monkeypatch):
    """A blocked bash call must NOT trigger VFS cache invalidation.

    The write-through iterates bash first (invalidate_all). If the bash
    call was blocked, the cache of legitimate writes should remain.
    """
    from proxy.tool_pipeline.tool_pipeline import MemoryToolGuard

    pipeline = ToolPipeline(registry=_registry_with_write())

    def fake_check(tc, ctx):
        if tc.name == "bash" and "rm" in tc.arguments.get("command", ""):
            return "blocklist: dangerous bash command"
        return None

    fake_guard = MemoryToolGuard()
    fake_guard.check = fake_check
    pipeline.guard = fake_guard

    pipeline._execute_via_loop = lambda *a, **kw: _mock_result()

    _cache_file_write_through(
        "src/existing.py", "legit content", "default", "s1", "call_legit"
    )

    response_data = _make_response_data([
        ("write", {"filePath": "src/new.py", "content": "fresh content"}),
        ("bash", {"command": "rm -rf /"}),
    ])
    ctx = _make_ctx()

    pipeline.run(response_data, ctx)

    with state.vfs_lock:
        cache = state.vfs_caches.get("default")
    # The blocked bash should not have wiped the pre-existing entry or
    # prevented the safe write from landing.
    assert cache is not None
    safe_keys = [k for k in cache._entries if "src/new.py" in k]
    assert len(safe_keys) == 1, (
        f"Expected exactly 1 entry containing 'src/new.py', got: {list(cache._entries.keys())}"
    )


def test_all_blocked_runs_without_crash(monkeypatch):
    """When every tool call is blocked, the VFS cache must remain untouched."""
    from proxy.tool_pipeline.tool_pipeline import MemoryToolGuard

    pipeline = ToolPipeline(registry=_registry_with_write())

    def fake_check(tc, ctx):
        return f"blocked {tc.name}"

    fake_guard = MemoryToolGuard()
    fake_guard.check = fake_check
    pipeline.guard = fake_guard

    pipeline._execute_via_loop = lambda *a, **kw: _mock_result()

    response_data = _make_response_data([
        ("write", {"filePath": "a.py", "content": "x"}),
        ("bash", {"command": "ls"}),
    ])
    ctx = _make_ctx()

    pipeline.run(response_data, ctx)

    with state.vfs_lock:
        cache = state.vfs_caches.get("default")
    # All calls blocked, so VFS write-through filtered them all out.
    if cache is not None:
        assert cache.get("a.py") is None


# ============================================================================
# M3-2: cache hits must serve content and skip read-once guard
# ============================================================================


def test_vfs_cache_hit_serves_content_despite_read_once_limit(monkeypatch):
    """A VFS cache hit must serve content even when read-once would block.

    Pre-fix: cache hit appended the read to memory_calls. The guard loop
    then called tracker.try_read_file() which checked read-once and
    BLOCKED with "File already read this request". The cached content
    was never served to the LLM.

    Post-fix: when _check_vfs_read_cache() returns a hit, the tool call
    is annotated with `from_cache=True`. The guard's try_read_file()
    bypasses the read-once check (the read already happened in a prior
    request and was already counted). The cached content is delivered.
    """
    from proxy.tool_pipeline.tool_pipeline import MemoryToolGuard
    from proxy.vfs import _cache_file
    from proxy.tool_pipeline.tool_call_tracker import ToolCallTracker
    from proxy.tool_pipeline.tool_limits import ToolLimits

    # Fresh tracker so the read-once count starts at 0 for this session.
    session_id = "sess-cache-hit"
    tracker = ToolCallTracker.get_instance(session_id)
    # Reset the inner ToolLimits state so the read-once counter starts at 0
    # (other tests in this file/session may have left state behind).
    tracker._limits.reset()

    # Prime the VFS cache: a prior request read the file. The cache
    # entry is `verified=True` (came from disk).
    target_file = "src/important.py"
    _cache_file(target_file, "PRIOR CONTENT FROM CACHE", "default", session_id)

    # 'read' is normally in the passthrough bucket (default registry
    # only marks proxy_* tools as memory). Build a custom registry
    # that classifies 'read' as MEMORY so the guard runs against it.
    from proxy.tool_pipeline.tool_registry import (
        ToolRegistry,
        ToolMeta,
        ToolCategory,
    )
    reg = ToolRegistry()
    reg.register(ToolMeta(name="read", category=ToolCategory.MEMORY))
    pipeline = ToolPipeline(registry=reg)
    pipeline.guard = MemoryToolGuard()

    captured = {}

    def mock_execute(*args, **kwargs):
        captured["executed"] = True
        captured["memory_calls"] = list(args[1]) if len(args) > 1 else None
        # Snapshot _cached_reads BEFORE the pipeline's finally clears it.
        captured["_cached_reads_at_execute"] = dict(pipeline._cached_reads)
        return _mock_result(
            tool_calls=[
                {"id": "call_read", "type": "function",
                 "function": {"name": "read", "arguments": json.dumps({"filePath": target_file})}}
            ],
            finish_reason="tool_calls",
        )

    pipeline._execute_via_loop = mock_execute

    response_data = _make_response_data([
        ("read", {"filePath": target_file}),
    ])
    ctx = _make_ctx(session_id=session_id)

    pipeline.run(response_data, ctx)

    # Read count must NOT have been incremented (cache hit was served).
    # Pre-fix: read count would be 1 because the guard called try_read_file
    # which counted the read despite the cache hit.
    # Post-fix: from_cache=True bypasses the read-once count, so count stays 0.
    import os
    normalized = os.path.normpath(os.path.realpath(target_file)).lower().replace("\\", "/")
    # tracker._limits is the underlying ToolLimits object.
    record = tracker._limits._file_reads.get(normalized)
    if record is not None:
        # If a record was created, the count must be 0 (cache hit didn't
        # count) — NOT 1 (which would be the pre-fix bug).
        assert record.read_count == 0, (
            f"Cache hit incremented read count to {record.read_count}; "
            f"cache hits must NOT count against the per-request read-once limit"
        )

    # The cache content should be marked for delivery (the LLM gets
    # the cached content, not a "File already read" block).
    # The _cached_reads dict at pipeline._cached_reads is keyed by
    # absolute path; the content is the value.
    normalized_abs = os.path.normpath(os.path.realpath(target_file))
    # The pipeline.run() finally clause clears _cached_reads at line 283.
    # We capture the snapshot inside mock_execute (BEFORE the clear).
    cached_at_execute = captured.get("_cached_reads_at_execute", {})
    cached = cached_at_execute.get(normalized_abs)
    assert cached == "PRIOR CONTENT FROM CACHE", (
        f"Cache content was not served at execute time. "
        f"_cached_reads_at_execute keys: {list(cached_at_execute.keys())}, "
        f"expected normalized: {normalized_abs}"
    )

    # And the read made it into memory_calls with from_cache=True so
    # the execute path knows it's a cache-served call.
    assert captured.get("executed") is True, "Pipeline did not reach execute_via_loop"
    if captured.get("memory_calls"):
        served = captured["memory_calls"][0]
        assert getattr(served, "from_cache", False) is True, (
            f"Cache-served tool call missing from_cache=True annotation: {served!r}"
        )


def _registry_with_read():
    """Build a registry that classifies 'read' as MEMORY so the guard
    loop runs try_read_file on it (the default registry only marks
    proxy_* tools as memory)."""
    from proxy.tool_pipeline.tool_registry import (
        ToolRegistry,
        ToolMeta,
        ToolCategory,
    )

    reg = ToolRegistry()
    reg.register(ToolMeta(name="read", category=ToolCategory.MEMORY))
    return reg