"""Unit tests for V2 concurrency safety: CAS, crash recovery, performance.

Covers §11 "Concurrency tests" of docs/v2-chain-hash-redesign.md:
- Concurrent same-key requests (M9 CAS)
- Concurrent different-key requests (no contention)
- Crash mid-request recovery
- Large session performance (10K messages)
"""
import threading
import time
from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import patch, MagicMock

import pytest

from proxy.context_v2 import _select_context_v2
from proxy.chain_hash import compute_chain, find_fork_point
from tests.concurrency_harness import run_concurrently

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _system(content: str) -> dict:
    return {"role": "system", "content": content}


def _user(content: str) -> dict:
    return {"role": "user", "content": content}


def _budget_config(safe=10_000):
    return {
        "max_context_tokens": 40_000,
        "tokens_per_char": 0.25,
        "safe_context_tokens": safe,
        "raw_exchange_count": 100,
    }


def _msgs(count, chars=400, start=0):
    pad = "X" * chars
    return [_user(f"msg {i} {pad}") for i in range(start, start + count)]


def _make_ctx(request_id="test-req"):
    return SimpleNamespace(
        request_id=request_id,
        v2_boundary_idx=None,
        v2_boundary_chain=None,
    )


def _mock_conn(min_mi=None):
    def execute_side_effect(query, *args, **kwargs):
        q = str(query).upper()
        result = MagicMock()
        if "SELECT 1 FROM CONVERSATION_SUMMARIES" in q or "SELECT 1 FROM MESSAGE_SUMMARIES" in q:
            result.fetchone.return_value = (1,)
        elif "MIN(MESSAGE_INDEX)" in q:
            result.fetchone.return_value = {"mi": min_mi}
        elif "RAW_START_MESSAGE_INDEX" in q:
            result.fetchone.return_value = None
        elif "CHAIN_HASH" in q:
            result.fetchall.return_value = []
        else:
            result.fetchone.return_value = None
            result.fetchall.return_value = []
        return result

    conn = MagicMock()
    conn.execute.side_effect = execute_side_effect
    cm = MagicMock()
    cm.__enter__.return_value = conn
    cm.__exit__.return_value = False
    return cm


def _common_patches(cfg, *, rich_summaries=None, ancient_history="", vfs_return=None,
                    last_exchange_index=-1, min_mi=None):
    """Return a list of context managers for the standard V2 mock surface."""
    conn_cm = _mock_conn(min_mi=min_mi)
    return [
        patch("proxy.context.get_context_config", return_value=cfg),
        patch("memory.db.get_rich_summaries", return_value=rich_summaries or []),
        patch("memory.db.get_ancient_history", return_value=ancient_history),
        patch("proxy.vfs._build_vfs_context", return_value=vfs_return),
        patch("proxy.vfs._build_vfs_inherited_context", return_value=vfs_return),
        patch("memory.db.connection", return_value=conn_cm),
        patch(
            "memory.db_summaries.get_last_summarized_exchange_index",
            return_value=last_exchange_index,
        ),
    ]


_SUMMARIES = [{"message_index": 5, "role": "user", "summary": "sum", "tool_summary": ""}]


# ---------------------------------------------------------------------------
# Concurrent same-key requests (M9 CAS)
# ---------------------------------------------------------------------------

class TestConcurrentSameKey:
    """Two threads hitting the same (session_id, model_name) simultaneously.
    The CAS at step 10 ensures exactly one wins; the other falls back."""

    def test_no_deadlock_same_key(self):
        """2 threads, same key: no deadlock, both complete."""
        cfg = _budget_config()
        msgs = [_system("sys")] + _msgs(100)

        with ExitStack() as stack:
            for p in _common_patches(cfg, rich_summaries=_SUMMARIES):
                stack.enter_context(p)

            ctx1 = _make_ctx("warmup")
            _select_context_v2("sess1", msgs, model_name="test", ctx=ctx1)

            ctx_a = _make_ctx("a")
            ctx_b = _make_ctx("b")

            def call_a():
                return _select_context_v2("sess1", msgs, model_name="test", ctx=ctx_a)

            def call_b():
                return _select_context_v2("sess1", msgs, model_name="test", ctx=ctx_b)

            run_concurrently([call_a, call_b], timeout=15)

        assert True, "Both threads completed without deadlock"

    def test_cas_one_wins_one_falls_back(self):
        """Both threads read the same state snapshot (version). One CAS-succeeds
        (ctx.v2_boundary_idx set); the other detects version mismatch and falls
        back (ctx.v2_boundary_idx = None).

        A threading.Barrier inside find_fork_point (step 5, after state read at
        step 4) guarantees both threads read state before either writes.
        """
        cfg = _budget_config()
        msgs = [_system("sys")] + _msgs(100)

        barrier = threading.Barrier(2)

        def sync_find(new, old):
            barrier.wait(timeout=10)
            return find_fork_point(new, old)

        with ExitStack() as stack:
            for p in _common_patches(cfg, rich_summaries=_SUMMARIES):
                stack.enter_context(p)
            stack.enter_context(
                patch("proxy.chain_hash.find_fork_point", side_effect=sync_find)
            )

            ctx_warmup = _make_ctx("warmup")
            _select_context_v2("sess1", msgs, model_name="test", ctx=ctx_warmup)

            ctx_a = _make_ctx("a")
            ctx_b = _make_ctx("b")

            def call_a():
                _select_context_v2("sess1", msgs, model_name="test", ctx=ctx_a)

            def call_b():
                _select_context_v2("sess1", msgs, model_name="test", ctx=ctx_b)

            run_concurrently([call_a, call_b], timeout=15)

        boundaries = [ctx_a.v2_boundary_idx, ctx_b.v2_boundary_idx]
        none_count = sum(1 for b in boundaries if b is None)
        set_count = sum(1 for b in boundaries if b is not None)
        assert none_count == 1, (
            f"Exactly one thread must lose CAS (boundary=None); got {boundaries}"
        )
        assert set_count == 1, (
            f"Exactly one thread must win CAS (boundary set); got {boundaries}"
        )


# ---------------------------------------------------------------------------
# Concurrent different-key requests (no contention)
# ---------------------------------------------------------------------------

class TestConcurrentDifferentKey:
    """Two threads hitting different (session_id, model_name) pairs: no
    contention, both succeed with their own boundary."""

    def test_different_keys_both_succeed(self):
        cfg = _budget_config()
        msgs = [_system("sys")] + _msgs(100)

        with ExitStack() as stack:
            for p in _common_patches(cfg, rich_summaries=_SUMMARIES):
                stack.enter_context(p)

            ctx_a = _make_ctx("a")
            ctx_b = _make_ctx("b")

            def call_a():
                _select_context_v2("sess_a", msgs, model_name="modelA", ctx=ctx_a)

            def call_b():
                _select_context_v2("sess_b", msgs, model_name="modelB", ctx=ctx_b)

            run_concurrently([call_a, call_b], timeout=15)

        assert ctx_a.v2_boundary_idx is not None, "Thread A must set boundary"
        assert ctx_b.v2_boundary_idx is not None, "Thread B must set boundary"


# ---------------------------------------------------------------------------
# Crash mid-request recovery
# ---------------------------------------------------------------------------

class TestCrashMidRequest:
    """If the proxy crashes (exception raised) between step 4 (state read) and
    step 10 (CAS write), the next request handles gracefully. State is not
    corrupted because step 4 only reads."""

    def test_crash_between_snapshot_and_cas_next_request_ok(self):
        cfg = _budget_config()
        msgs = [_system("sys")] + _msgs(100)

        with ExitStack() as stack:
            for p in _common_patches(cfg, rich_summaries=_SUMMARIES):
                stack.enter_context(p)

            ctx1 = _make_ctx("turn1")
            _select_context_v2("sess1", msgs, model_name="test", ctx=ctx1)
            boundary1 = ctx1.v2_boundary_idx
            assert boundary1 is not None

        with ExitStack() as stack:
            for p in _common_patches(cfg, rich_summaries=_SUMMARIES):
                stack.enter_context(p)

            def crash_find(new, old):
                raise RuntimeError("simulated crash mid-request")

            stack.enter_context(
                patch("proxy.chain_hash.find_fork_point", side_effect=crash_find)
            )

            ctx2 = _make_ctx("turn2")
            with pytest.raises(RuntimeError, match="simulated crash"):
                _select_context_v2("sess1", msgs, model_name="test", ctx=ctx2)

        with ExitStack() as stack:
            for p in _common_patches(cfg, rich_summaries=_SUMMARIES):
                stack.enter_context(p)

            ctx3 = _make_ctx("turn3")
            result = _select_context_v2("sess1", msgs, model_name="test", ctx=ctx3)

            assert result and all(isinstance(m, dict) and "role" in m for m in result), (
                f"Post-crash request must return a non-empty list of message "
                f"dicts; got {result!r}"
            )
            assert ctx3.v2_boundary_idx is not None, (
                "Post-crash request must set boundary (warm path from turn 1 state)"
            )
            assert ctx3.v2_boundary_idx == boundary1, (
                "Boundary from turn 1 must still be valid after crash"
            )


# ---------------------------------------------------------------------------
# Large session performance
# ---------------------------------------------------------------------------

class TestLargeSession:
    """10K-message session: chain computation < 100ms, V2 end-to-end < 200ms."""

    def test_10k_messages_chain_and_v2_under_thresholds(self):
        cfg = _budget_config(safe=10_000)
        msgs = [_system("sys")] + _msgs(10_000, chars=40)

        conversation = msgs[1:]
        start = time.perf_counter()
        compute_chain(conversation)
        chain_ms = (time.perf_counter() - start) * 1000
        assert chain_ms < 100, f"compute_chain took {chain_ms:.1f}ms (threshold 100ms)"

        ctx = _make_ctx("perf")
        with ExitStack() as stack:
            for p in _common_patches(cfg, rich_summaries=_SUMMARIES):
                stack.enter_context(p)
            start = time.perf_counter()
            result = _select_context_v2("sess1", msgs, model_name="test", ctx=ctx)
            v2_ms = (time.perf_counter() - start) * 1000

        assert v2_ms < 200, f"V2 end-to-end took {v2_ms:.1f}ms (threshold 200ms)"
        assert result and isinstance(result[0], dict) and "role" in result[0], (
            f"V2 must return a non-empty list of message dicts for 10K input; "
            f"got first item {result[0] if result else result!r}"
        )
        assert ctx.v2_boundary_idx is not None
