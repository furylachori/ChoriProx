"""Integration tests for C5 — passthrough early return removed (F5).

After C5, small conversations flow through the normal path. The removed
passthrough short-circuited all tier injection when estimated_tokens was
below 30% of safe. The new fast-exit only fires when chain-state miss AND
no `conversation_summaries` / `message_summaries` rows exist (one indexed
query per request — NO caching of the negative verdict).

Tests:
  1. sub-threshold + existing ancient → ancient is now surfaced (F5).
  2. fresh tiny session → fast-exit returns cleaned; seed a summary
     mid-test → next call sees it (proves no negative-verdict cache).
  3. sub-threshold + VFS cache warm → VFS block surfaces (F5).
  4. Tier-2 capture + pin emission still runs on the flowing path.
  5. sub-threshold session that HAS a conversation_summaries row →
     normal path runs → ctx.v2_boundary_idx is a real value (was None
     under the removed passthrough, breaking post_process raw_start).
"""
import hashlib
from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest


pytestmark = pytest.mark.integration


SESSION = "test-c5-passthrough-removed"


@pytest.fixture(autouse=True)
def _isolate_db(monkeypatch):
    """Per-test fresh DB (in-memory, like the C4 test file).

    Wipes the shared SESSION between tests so each test starts with no
    DB rows (no ancient, no summaries). Tests that want pre-seeded rows
    seed them explicitly inside the test.
    """
    monkeypatch.setenv("PROXY_TEST_DB", ":memory:")
    from memory.db import init_db
    from memory.db_pool import connection
    init_db(":memory:")
    with connection(pooled=True) as conn:
        conn.execute("DELETE FROM message_summaries WHERE session_id = ?", (SESSION,))
        conn.execute("DELETE FROM messages WHERE session_id = ?", (SESSION,))
        conn.execute("DELETE FROM conversation_summaries WHERE session_id = ?", (SESSION,))
        conn.commit()
    yield ":memory:"


def _content_hash(content: str) -> str:
    return hashlib.sha256(content.encode()).hexdigest()[:16]


def _seed_messages(session_id: str, n_exchanges: int, chars_per_msg: int = 100):
    from memory.db import store_messages_batch
    msgs = []
    for i in range(n_exchanges):
        msgs.append({"role": "user", "content": f"user-{i} " + "U" * chars_per_msg})
        msgs.append({"role": "assistant", "content": f"asst-{i} " + "A" * chars_per_msg})
    store_messages_batch(session_id, msgs, start_index=0)
    return msgs


def _make_request(n_exchanges: int, chars_per_msg: int = 100):
    msgs = []
    for i in range(n_exchanges):
        msgs.append({"role": "user", "content": f"user-{i} " + "U" * chars_per_msg})
        msgs.append({"role": "assistant", "content": f"asst-{i} " + "A" * chars_per_msg})
    return msgs


def _make_cfg(safe_tokens: int = 100_000):
    """Build a SimpleNamespace that the V2 path accepts as model_cfg."""
    return SimpleNamespace(backend_model="m")


def _patch_context_config(safe_tokens: int = 100_000):
    cfg = {
        "max_context_tokens": int(safe_tokens / 0.75),
        "safe_context_tokens": safe_tokens,
        "optimal_context_tokens": int(safe_tokens * 0.47),
        "tokens_per_char": 0.25,
        "chars_per_token": 4,
        "context_reduction_ratio": 0.75,
        "raw_exchange_count": max(5, int(safe_tokens / 0.75) // 4000),
    }
    return patch("proxy.context.get_context_config", return_value=cfg)


def _patch_vfs():
    return (
        patch("proxy.vfs._build_vfs_context", return_value=None),
        patch("proxy.vfs._build_vfs_inherited_context", return_value=None),
    )


# ============================================================================
# Test 1: sub-threshold + existing ancient is now included
# ============================================================================


class TestSubThresholdAncientSurfaced:
    """F5: small conversations now see ancient_history if present."""

    def test_subthreshold_session_with_existing_ancient_includes_ancient(self):
        from memory.db_summaries import update_ancient_history
        from proxy.context_v2 import _select_context_v2

        _seed_messages(SESSION, 3, chars_per_msg=100)
        update_ancient_history(SESSION, "ANCIENT_MARKER_PASSTHROUGH_TEST")

        req = _make_request(3, chars_per_msg=100)

        with ExitStack() as ctx_stack:
            ctx_stack.enter_context(_patch_context_config(100_000))
            for p in _patch_vfs():
                ctx_stack.enter_context(p)
            result = _select_context_v2(
                SESSION, req, model_name="m",
                model_cfg=_make_cfg(100_000),
            )

        all_text = " ".join(
            m.get("content", "") for m in result
            if isinstance(m.get("content"), str)
        )
        assert "ANCIENT_MARKER_PASSTHROUGH_TEST" in all_text, (
            "After passthrough removal, ancient must surface even for "
            "sub-threshold conversations. "
            f"Got: {all_text[:500]!r}"
        )
        assert "[TIMELINE" in all_text, "Ancient must be in [TIMELINE] block"


# ============================================================================
# Test 2: fast-exit on fresh session
# ============================================================================


class TestFastExitOnFreshSession:
    """Fast-exit: chain-state miss + no DB rows → return cleaned.
    NO caching — re-query every request (architect finding C5-4).
    """

    def test_fresh_tiny_session_no_db_rows_takes_fast_exit(self):
        from memory.db_summaries import store_message_summary
        from proxy.context_v2 import _select_context_v2

        _seed_messages(SESSION, 2, chars_per_msg=100)
        req = _make_request(2, chars_per_msg=100)

        with ExitStack() as ctx_stack:
            ctx_stack.enter_context(_patch_context_config(100_000))
            for p in _patch_vfs():
                ctx_stack.enter_context(p)
            first = _select_context_v2(
                SESSION, req, model_name="m",
                model_cfg=_make_cfg(100_000),
            )

        first_text = " ".join(
            m.get("content", "") for m in first
            if isinstance(m.get("content"), str)
        )
        assert "[PAST CONTEXT" not in first_text, (
            "Fresh session with no summaries: must NOT inject summaries"
        )
        assert "[TIMELINE" not in first_text, (
            "Fresh session with no ancient: must NOT inject ancient"
        )

        store_message_summary(
            session_id=SESSION,
            exchange_index=0,
            role="user",
            summary="FRESH_SEEDED_SUMMARY_FOR_FAST_EXIT",
            tool_summary="",
            content_hash=_content_hash("seed-fast-exit"),
            model_used="seed",
            token_estimate=50,
            end_message_index=1,
            message_index=0,
        )

        with ExitStack() as ctx_stack:
            ctx_stack.enter_context(_patch_context_config(100_000))
            for p in _patch_vfs():
                ctx_stack.enter_context(p)
            second = _select_context_v2(
                SESSION, req, model_name="m",
                model_cfg=_make_cfg(100_000),
            )

        second_text = " ".join(
            m.get("content", "") for m in second
            if isinstance(m.get("content"), str)
        )
        assert "FRESH_SEEDED_SUMMARY_FOR_FAST_EXIT" in second_text, (
            "After seeding a summary, the next request must see it. "
            "Proves the fast-exit does NOT cache the negative verdict. "
            f"Got: {second_text[:500]!r}"
        )

    def test_fresh_large_session_exceeds_budget_normal_path_evicts(self):
        from proxy.context_v2 import _select_context_v2

        big_safe = 10_000
        # Slice 4D restoration (2026-07-18): bumped from 30→50 exchanges
        # because at 30 exchanges (~9115 tokens) the conversation fit
        # safe=10000 and no eviction was required, failing the test's
        # premise. 50 exchanges (~15195 tokens) overflows safe, so the
        # cold path MUST evict some history for the candidate to fit.
        big_req = _make_request(50, chars_per_msg=600)

        with ExitStack() as ctx_stack:
            ctx_stack.enter_context(_patch_context_config(big_safe))
            for p in _patch_vfs():
                ctx_stack.enter_context(p)
            result = _select_context_v2(
                SESSION, big_req, model_name="m",
                model_cfg=_make_cfg(big_safe),
            )

        all_text = " ".join(
            m.get("content", "") for m in result
            if isinstance(m.get("content"), str)
        )
        # Slice 4D restoration: bumped range from 30→50 to match the
        # new exchange count. With 50 exchanges (~15195 tokens) and
        # safe=10000, the cold path must evict at least one pair.
        for i in range(50):
            if f"user-{i}" in all_text and f"asst-{i}" in all_text:
                continue
            break
        else:
            pytest.fail(
                "Over-budget fresh session should be evicted by the normal path. "
                "Fast-exit is only for conversations that fit in safe_tokens."
            )


# ============================================================================
# Test 3: sub-threshold + VFS cache warm
# ============================================================================


class TestSubThresholdVfsSurfaced:
    """F5: small conversations now see VFS block if VFS cache is warm.

    Pre-C5: the removed passthrough short-circuited sub-threshold
    conversations and did NOT call `_build_vfs_context` (VFS build
    happened AFTER the passthrough early-return).
    Post-C5: when the fast-exit does NOT fire (e.g. session has prior
    rows in conversation_summaries), the normal path runs and VFS is
    built and injected.
    """

    @pytest.fixture(autouse=True)
    def _c5b_vfs_enabled(self, monkeypatch):
        import proxy.state as state
        monkeypatch.setattr(state, "VFS_ENABLED", True)
        monkeypatch.setattr(state, "VFS_SCOPE", "all")
        monkeypatch.setattr(state, "VFS_TTL_SECONDS", 3600)
        monkeypatch.setattr(state, "VFS_MAX_ENTRIES", 500)
        monkeypatch.setattr(state, "VFS_MAX_ENTRY_BYTES", 1_048_576)
        monkeypatch.setattr(state, "VFS_MAX_WORKSPACES", 20)
        yield

    def test_subthreshold_session_vfs_cache_warm_includes_vfs(self):
        """Wave 5b: small conversations see real VFS on the normal tiered path.

        The prior canned-mock variant asserted on a `VFS_MARKER_PASSTHROUGH_REMOVED`
        constant that the mock itself returned, which made the test a tautology.
        The corrected variant exercises the real builder, real workspace
        resolver, real cache, real splitter, real cleanup, and a real
        `RequestContext` carrying a `client_hint`.
        """
        import uuid

        from memory.db_summaries import update_ancient_history
        from proxy.context_v2 import _select_context_v2
        from proxy.types import RequestContext
        from proxy.vfs import _cache_file
        import proxy.state as state

        run_id = uuid.uuid4().hex[:8]
        SESSION = f"test-c5b-{run_id}"
        CLIENT_HINT = f"c5b-client-hint-{run_id}"
        WORKSPACE = f"c5b-alpha-{run_id}"
        SEED_SESSION = f"test-c5b-seed-{run_id}"
        MODEL_NAME = f"c5b-model-{run_id}"
        ANCIENT_MARKER = f"c5b ancient marker {run_id}"
        CURRENT_USER_SENTINEL = f"c5b current user {run_id}"
        CACHE_PATH = f"/c5b/alpha-only-{run_id}.py"
        CACHE_CONTENT = f"alpha cache content for {run_id}"

        # Workspace: activate named-workspace mapping under CLIENT_HINT.
        state.workspace_store.activate(
            SEED_SESSION, WORKSPACE, client_hint=CLIENT_HINT,
        )
        state.workspace_store.pop_session(SEED_SESSION)

        # Real VFS cache: populate the named workspace.
        _cache_file(
            CACHE_PATH, CACHE_CONTENT, WORKSPACE, SEED_SESSION,
        )
        # Also populate a `default` workspace with a different sentinel
        # so the routing test can prove isolation.
        _cache_file(
            f"/c5b/default-only-{run_id}.py",
            f"default cache content for {run_id}",
            "default",
            SEED_SESSION,
        )

        # Seed messages and ancient in real SQLite so the normal tiered path
        # runs (deflects the fresh fast return).
        _seed_messages(SESSION, 3, chars_per_msg=100)
        update_ancient_history(SESSION, ANCIENT_MARKER)

        # 3 completed exchanges + trailing current user.
        completed_req = _make_request(3, chars_per_msg=100)
        trailing_user_msg = {
            "role": "user", "content": CURRENT_USER_SENTINEL,
        }
        req = completed_req + [trailing_user_msg]
        req_before = [dict(m) for m in req]

        ctx = RequestContext(
            body={"messages": req},
            model_name=MODEL_NAME,
            wants_stream=False,
            session_id=SESSION,
            client_hint=CLIENT_HINT,
            messages=req,
            remote_addr="127.0.0.1",
            workspace_name=WORKSPACE,
        )

        result = _select_context_v2(
            SESSION, req, model_name=MODEL_NAME,
            model_cfg=_make_cfg(100_000), ctx=ctx,
        )

        def _idx(predicate):
            for i, m in enumerate(result):
                if predicate(m):
                    return i
            return -1

        vfs_indices = [
            i for i, m in enumerate(result)
            if "[VFS CONTEXT" in (m.get("content") or "")
        ]
        assert len(vfs_indices) == 1, (
            f"Exactly one [VFS CONTEXT expected; got positions={vfs_indices}"
        )
        vfs_idx = vfs_indices[0]
        vfs_content = result[vfs_idx].get("content", "")

        # The real cached filename appears in the rendered block.
        assert CACHE_PATH in vfs_content, (
            f"Real cached filename {CACHE_PATH!r} must appear in VFS block; "
            f"got first 300 chars: {vfs_content[:300]!r}"
        )

        # The default-cache sentinel MUST NOT appear (proves routing).
        assert f"default-only-{run_id}.py" not in vfs_content, (
            f"Default-workspace sentinel leaked into named-workspace VFS"
        )

        # The old canned VFS marker is gone.
        all_text = " ".join(
            m.get("content", "") for m in result
            if isinstance(m.get("content"), str)
        )
        assert "VFS_MARKER_PASSTHROUGH_REMOVED" not in all_text, (
            "Canned VFS_MARKER_PASSTHROUGH_REMOVED marker must NOT appear"
        )

        # Canonical semantic order: ancient/completed -> VFS -> current user.
        ancient_idx = _idx(lambda m: ANCIENT_MARKER in (m.get("content") or ""))
        assert ancient_idx >= 0, (
            f"Ancient context with marker {ANCIENT_MARKER!r} must surface"
        )
        assert ancient_idx < vfs_idx, (
            f"Ancient (idx={ancient_idx}) must precede VFS (idx={vfs_idx})"
        )

        completed_user_indices = [
            i for i, m in enumerate(result[:vfs_idx])
            if m.get("role") == "user" and "user-0" in (m.get("content") or "")
        ]
        assert completed_user_indices, (
            "At least one completed-exchange user must precede VFS"
        )

        current_user_idx = _idx(
            lambda m: m.get("role") == "user"
            and CURRENT_USER_SENTINEL in (m.get("content") or "")
        )
        assert current_user_idx >= 0
        assert vfs_idx < current_user_idx, (
            f"VFS (idx={vfs_idx}) must precede current user "
            f"(idx={current_user_idx})"
        )

        assert result[current_user_idx]["content"] == CURRENT_USER_SENTINEL, (
            "Current user must be unchanged"
        )

        # No single-space context-layer assistant barrier.
        assert not any(
            m.get("role") == "assistant" and m.get("content") == " "
            for m in result
        ), "Wave 4 invariant: no context-layer barrier"

        # Input request unchanged.
        assert req == req_before, (
            f"Input request must be unchanged. Diff: "
            f"{[(i, e, r) for i, (e, r) in enumerate(zip(req_before, req)) if e != r]}"
        )

        # Teardown.
        self._c5b_teardown(SEED_SESSION, SESSION, CLIENT_HINT)

    @staticmethod
    def _c5b_teardown(seed_session, target_session, client_hint):
        """Clean test-owned VFS and workspace state."""
        import proxy.state as state
        for sid in (seed_session, target_session):
            if sid:
                try:
                    state.workspace_store.deactivate(sid)
                    state.workspace_store.pop_session(sid)
                except Exception:
                    pass
        if seed_session and client_hint:
            try:
                state.workspace_store.deactivate(seed_session, client_hint=client_hint)
            except Exception:
                pass
        with state.vfs_lock:
            state.vfs_caches.clear()
            state.vfs_write_log.clear()
            state.vfs_rollback_snapshots.clear()
            state.instruction_files_by_workspace.clear()
            state._last_scanned_by_tool_call.clear()


# ============================================================================
# Test 4: Tier-2 preservation
# ============================================================================


class TestTier2Preservation:
    """Tier-2 capture + pin emission must still run on the flowing path."""

    def test_tier2_capture_runs_for_subthreshold_session(self):
        from proxy.context_v2 import _select_context_v2

        _seed_messages(SESSION, 3, chars_per_msg=100)
        req = _make_request(3, chars_per_msg=100)

        with ExitStack() as ctx_stack:
            ctx_stack.enter_context(_patch_context_config(100_000))
            for p in _patch_vfs():
                ctx_stack.enter_context(p)
            mock_capture = ctx_stack.enter_context(
                patch(
                    "proxy.context_v2._capture_tier2_instructions",
                    side_effect=lambda *a, **kw: None,
                )
            )
            result = _select_context_v2(
                SESSION, req, model_name="m",
                model_cfg=_make_cfg(100_000),
            )

        assert mock_capture.called, (
            "_capture_tier2_instructions must run (Tier-2 preservation)"
        )
        assert result is not None, "_select_context_v2 must return messages"
        assert isinstance(result, list), "Result must be a list of messages"


# ============================================================================
# Test 5: ctx.v2_boundary_idx consistency
# ============================================================================


class TestBoundaryIdxConsistency:
    """With passthrough removed, boundary is a real value (was None before)."""

    def test_subthreshold_session_persists_real_boundary(self):
        from memory.db_summaries import update_ancient_history
        from proxy.context_v2 import _select_context_v2

        _seed_messages(SESSION, 3, chars_per_msg=100)
        req = _make_request(3, chars_per_msg=100)
        update_ancient_history(SESSION, "boundary-persist-test")

        ctx = SimpleNamespace(
            request_id="test",
            workspace_name="default",
            is_subagent=False,
            messages_original=req,
            body={"messages": req},
        )

        with ExitStack() as ctx_stack:
            ctx_stack.enter_context(_patch_context_config(100_000))
            for p in _patch_vfs():
                ctx_stack.enter_context(p)
            _select_context_v2(
                SESSION, req, model_name="m",
                model_cfg=_make_cfg(100_000),
                ctx=ctx,
            )

        assert ctx.v2_boundary_idx is not None, (
            "With passthrough removed, subthreshold sessions that have a "
            "conversation_summaries row must produce a real boundary_idx "
            "for post_process to persist raw_start_message_index. "
            f"Got v2_boundary_idx={getattr(ctx, 'v2_boundary_idx', None)!r}"
        )
