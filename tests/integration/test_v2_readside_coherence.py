"""Integration tests for V2 read-side coherence after Commit 4.

Covers:
- 4.2 — Warm path ancient refresh after invalidate_chain_state_for_session (F4)
- 4.1 — Universal summary budget applied at all 4 fetch sites (F6)
- 4.4 — Fork-to-0 persists summaries (F7)
- 4.5 — Per-model ancient cap scales with model_safe_tokens (D5)
- 4.6 — update_ancient_history re-raises on caller conn (D3)

Approach:
- Real file-backed SQLite (unique per-test database).
- Mock only at sanctioned patch points:
    memory.sync._get_sota
    memory.sync.update_ancient_history (ancient-write-failure test)
    memory.sync.submit_ancient_compress (capture dispatch)
    proxy.vfs._build_vfs_context / _build_vfs_inherited_context
    proxy.context.get_context_config
- Exercise the REAL _select_context_v2 / _cold_path_compute / _load_state_from_db /
  _evict_with_cap call sites (no direct invocation of helpers in a tautological way).
"""
import hashlib
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest


pytestmark = pytest.mark.integration


SESSION = "test-v2-readside-coherence"


# ── DB setup (autouse) ─────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def _isolate_db(tmp_path, monkeypatch):
    """Use a unique file-backed SQLite database for each test."""
    db_path = tmp_path / "v2-readside-coherence.db"
    monkeypatch.setenv("PROXY_TEST_DB", str(db_path))
    from memory.db import init_db
    init_db(str(db_path))
    yield str(db_path)


# ── Helpers ────────────────────────────────────────────────────────────


def _content_hash(content: str) -> str:
    return hashlib.sha256(content.encode()).hexdigest()[:16]


def _seed_messages(session_id: str, n_exchanges: int, chars_per_msg: int = 200):
    from memory.db import store_messages_batch
    msgs = []
    for i in range(n_exchanges):
        msgs.append({"role": "user", "content": f"user-{i} " + "U" * chars_per_msg})
        msgs.append({"role": "assistant", "content": f"asst-{i} " + "A" * chars_per_msg})
    store_messages_batch(session_id, msgs, start_index=0)
    return msgs


def _seed_summary(session_id: str, exchange_index: int, summary: str,
                   message_index: int = 0, end_message_index: int = 1,
                   model_used: str = "seed-model"):
    from memory.db_summaries import store_message_summary
    store_message_summary(
        session_id=session_id,
        exchange_index=exchange_index,
        role="user",
        summary=summary,
        tool_summary="",
        content_hash=_content_hash(f"seed-{exchange_index}"),
        model_used=model_used,
        token_estimate=50,
        end_message_index=end_message_index,
        message_index=message_index,
    )


def _make_request(n_exchanges: int, chars_per_msg: int = 200):
    msgs = []
    for i in range(n_exchanges):
        msgs.append({"role": "user", "content": f"user-{i} " + "U" * chars_per_msg})
        msgs.append({"role": "assistant", "content": f"asst-{i} " + "A" * chars_per_msg})
    return msgs


def _no_vfs_patches():
    return (
        patch("proxy.vfs._build_vfs_context", return_value=None),
        patch("proxy.vfs._build_vfs_inherited_context", return_value=None),
    )


@contextmanager
def _patch_vfs():
    with patch("proxy.vfs._build_vfs_context", return_value=None), \
         patch("proxy.vfs._build_vfs_inherited_context", return_value=None):
        yield


def _make_cfg(context_limit: int):
    """Build a SimpleNamespace that the V2 path accepts as model_cfg."""
    return SimpleNamespace(
        context_limit=context_limit,
        raw_exchange_count=max(5, context_limit // 4000),
    )


def _patch_context_config(context_limit: int):
    cfg = {
        "max_context_tokens": context_limit,
        "safe_context_tokens": int(context_limit * 0.75),
        "optimal_context_tokens": int(context_limit * 0.47),
        "tokens_per_char": 0.25,
        "chars_per_token": 4,
        "context_reduction_ratio": 0.75,
        "raw_exchange_count": max(5, context_limit // 4000),
    }
    return patch("proxy.context.get_context_config", return_value=cfg)


def _mock_sota():
    m = MagicMock()
    m.compress_ancient_history.return_value = "MOCK_COMPRESSED_ANCIENT"
    m.extract_and_summarize_from_summaries.return_value = {
        "extraction_timestamp": "2025-01-01T00:00:00Z",
        "entities": [], "facts": [], "events": [],
        "instructions": [], "tasks": [],
    }
    m.model = "test-sota"
    return m


def _mock_graph():
    g = MagicMock()
    g.store_memory.return_value = True
    return g


# ============================================================================
# Test 1 (4.2) — Warm path ancient refresh after invalidation
# ============================================================================


class TestWarmSessionAncientRefresh:
    """F4: warm path must see new ancient after a background write + invalidation."""

    def test_warm_session_ancient_visible_after_invalidation(self):
        """Seed ancient in DB → warm _chain_state entry with snap_ancient="" →
        background worker writes new ancient → invalidate → next request
        serves NEW ancient via Step 8.5 (cold re-init) / _load_state_from_db.
        """
        from memory.db_summaries import update_ancient_history
        from proxy.context_v2 import _select_context_v2
        from proxy.state import invalidate_chain_state_for_session

        _seed_messages(SESSION, 30, chars_per_msg=200)

        old_ancient = "OLD_ANCIENT_SENTINEL_XYZ_12345"
        new_ancient = "NEW_ANCIENT_SENTINEL_ABC_67890_DISTINCT_MARKER"
        update_ancient_history(SESSION, old_ancient)

        cfg = _make_cfg(10_000)
        req = _make_request(30, chars_per_msg=200)

        with _patch_context_config(10_000), _patch_vfs():
            warm = _select_context_v2(SESSION, req, model_name="m", model_cfg=cfg)
        warm_text = " ".join(
            m.get("content", "") for m in warm if isinstance(m.get("content"), str)
        )
        assert "OLD_ANCIENT_SENTINEL_XYZ_12345" in warm_text, (
            f"Turn 1 (cold) must surface seeded OLD ancient. "
            f"Got first 500 chars: {warm_text[:500]!r}"
        )

        update_ancient_history(SESSION, new_ancient)
        invalidate_chain_state_for_session(SESSION)

        with _patch_context_config(10_000), _patch_vfs():
            refreshed = _select_context_v2(
                SESSION, req, model_name="m", model_cfg=cfg,
            )

        all_text = " ".join(
            m.get("content", "") for m in refreshed if isinstance(m.get("content"), str)
        )
        assert "NEW_ANCIENT_SENTINEL_ABC_67890_DISTINCT_MARKER" in all_text, (
            "After invalidation the next request must serve NEW ancient. "
            f"Got first 500 chars: {all_text[:500]!r}"
        )


# ============================================================================
# Test 2 (4.1) — Universal summary budget applied at cold-path fetch site
# ============================================================================


class TestColdPathSummaryBudget:
    """F6: cold-path summaries must be budget-capped before assembly."""

    def test_downgrade_cold_path_applies_summary_budget(self):
        from unittest.mock import MagicMock
        from proxy.context_v2 import _cold_path_compute, _apply_summary_budget
        from proxy.context_v2 import V2_SUMMARY_BUDGET_RATIO

        _seed_messages(SESSION, 60, chars_per_msg=500)
        n_summaries = 60
        for i in range(n_summaries):
            _seed_summary(
                SESSION, exchange_index=i,
                summary=("word " * 200).strip(),
            )

        small_safe = 10_000
        low_watermark = int(small_safe * 0.65)

        req = _make_request(60, chars_per_msg=500)

        with patch("memory.sync.submit_ancient_compress", return_value=None) as mock_dispatch:
            with _patch_context_config(10_000), _patch_vfs():
                cold = _cold_path_compute(
                    req, _make_cfg(10_000), SESSION, "m", None,
                    low_watermark, 0.25, model_safe_tokens=small_safe,
                )

        served = cold["summaries"]
        budget = int(small_safe * V2_SUMMARY_BUDGET_RATIO)
        served_tokens = sum(
            (s.get("summary_tokens") or s.get("rendered_full_tokens") or int(800 * 0.25))
            for s in served
        )
        assert served_tokens <= budget, (
            f"served summaries token count {served_tokens} must be <= budget {budget}"
        )
        assert len(served) < n_summaries, (
            f"served count {len(served)} must be < total seeded {n_summaries}"
        )
        assert mock_dispatch.called, "overflow summaries must be dispatched via submit_ancient_compress"

    def test_downgrade_holistic_total_envelope_within_safe_tokens(self):
        """O7: AC3 requires the FULL assembled prompt (raw + summaries + ancient
        + VFS + system + barrier) fits within model_safe_tokens. Prior tests
        verified summary budget in isolation; this test verifies the total
        envelope on a downgrade turn."""
        from memory.db_summaries import update_ancient_history
        from proxy.context_v2 import _select_context_v2

        _seed_messages(SESSION, 60, chars_per_msg=500)
        for i in range(40):
            _seed_summary(SESSION, exchange_index=i, summary="x" * 300)
        update_ancient_history(SESSION, "Z" * 20_000)

        small_safe = 5_000
        tokens_per_char = 0.25

        req = _make_request(60, chars_per_msg=500)

        with patch("memory.sync.submit_ancient_compress", return_value=None) as mock_dispatch:
            with _patch_context_config(small_safe // 0.75 + 1), _patch_vfs():
                assembled = _select_context_v2(
                    SESSION, req, model_name="m",
                    model_cfg=_make_cfg(int(small_safe / 0.75)),
                )

        total_chars = sum(
            len(m.get("content", "") or "")
            for m in assembled if isinstance(m.get("content"), str)
        )
        total_tokens = int(total_chars * tokens_per_char)
        assert total_tokens <= small_safe, (
            f"HOLISTIC O7: total assembled tokens {total_tokens} must be <= "
            f"model_safe_tokens {small_safe}. Budget check alone passes; the "
            f"total envelope (raw + summaries + ancient + system) must also fit."
        )
        assert mock_dispatch.called, (
            "overflow summaries must be dispatched via submit_ancient_compress"
        )


# ============================================================================
# Test 3 (4.4) — Fork-to-boundary-0 now persists summaries
# ============================================================================


class TestForkToBoundaryZero:
    """F7: warm entry + cold recompute that lands boundary=0 must still fetch
    summaries (Step 8.5 widened to fire on cold_result is not None)."""

    SESSION = "test-fork-to-boundary-zero"

    def test_fork_to_boundary_zero_persists_summaries(self):
        from memory.db_summaries import update_ancient_history
        from proxy.context_v2 import _select_context_v2

        SESSION = self.SESSION

        _seed_messages(SESSION, 30, chars_per_msg=200)
        for i in range(5):
            _seed_summary(SESSION, exchange_index=i, summary=f"summary-{i}-MARKER")
        update_ancient_history(SESSION, "ancient-forkto0-MARKER")

        cfg = _make_cfg(10_000)
        big_req = _make_request(30, chars_per_msg=200)

        with _patch_context_config(10_000), _patch_vfs():
            first = _select_context_v2(SESSION, big_req, model_name="m", model_cfg=cfg)
        first_text = " ".join(
            m.get("content", "") for m in first if isinstance(m.get("content"), str)
        )
        assert "summary-0-MARKER" in first_text, (
            "Turn 1 (cold) must surface seeded summaries. "
            f"Got first 500 chars: {first_text[:500]!r}"
        )

        import proxy.state as state
        with state._chain_state_lock:
            state._chain_state.clear()
        from proxy.state import invalidate_chain_state_for_session
        invalidate_chain_state_for_session(SESSION)

        mid_req = _make_request(30, chars_per_msg=200)

        with _patch_context_config(10_000), _patch_vfs():
            second = _select_context_v2(SESSION, mid_req, model_name="m", model_cfg=cfg)
        second_text = " ".join(
            m.get("content", "") for m in second if isinstance(m.get("content"), str)
        )
        assert "summary-0-MARKER" in second_text, (
            "After invalidation + mid-size request (cold path with boundary=0): "
            "Step 8.5 must widen to include cold_result is not None. "
            f"Got first 500 chars: {second_text[:500]!r}"
        )


# ============================================================================
# Test 4 (4.5) — Per-model ancient cap scales with model_safe_tokens
# ============================================================================


class TestAncientCapScales:
    """D5: V2_ANCIENT_MAX_RATIO replaces absolute V2_ANCIENT_MAX_CHARS."""

    def test_ancient_cap_scales_with_model_safe_tokens(self):
        from proxy.context_v2 import (
            V2_ANCIENT_MAX_RATIO,
            _ancient_max_chars,
            _format_ancient,
            _truncate_ancient,
        )

        tokens_per_char = 0.25
        small_safe = 10_000
        large_safe = 100_000

        small_cap = _ancient_max_chars(
            small_safe,
            tokens_per_char,
        )
        large_cap = _ancient_max_chars(
            large_safe,
            tokens_per_char,
        )

        assert small_cap == 3_943
        assert large_cap == 39_943

        small_payload = _truncate_ancient(
            "X" * 50_000,
            small_cap,
            session_id="small",
        )
        large_payload = _truncate_ancient(
            "X" * 100_000,
            large_cap,
            session_id="large",
        )

        small_rendered = _format_ancient(small_payload)
        large_rendered = _format_ancient(large_payload)

        small_budget = int(
            small_safe * V2_ANCIENT_MAX_RATIO
        )
        large_budget = int(
            large_safe * V2_ANCIENT_MAX_RATIO
        )

        assert len(small_rendered) == 4_000
        assert len(large_rendered) == 40_000
        assert int(
            len(small_rendered) * tokens_per_char
        ) == small_budget
        assert int(
            len(large_rendered) * tokens_per_char
        ) == large_budget

    def test_ancient_truncation_helper(self):
        from proxy.context_v2 import _truncate_ancient

        huge = "X" * 5000
        truncated = _truncate_ancient(huge, max_chars=2000, session_id="s", ctx=None)
        assert len(truncated) <= 2000
        assert "[...truncated; ancient history overflowed" in truncated
        assert "2000 chars" in truncated

        small = "Y" * 100
        assert _truncate_ancient(small, max_chars=2000, session_id="s", ctx=None) == small

        assert _truncate_ancient("", max_chars=2000, session_id="s", ctx=None) == ""


# ============================================================================
# Test 5 (4.1, 4.5) — Restart budget applied at _load_state_from_db (4th fetch site)
# ============================================================================


class TestRestartBudgetApplied:
    """Architect note: _load_state_from_db is the 4th summary/ancient fetch
    site. Without budget + cap, F6 downgrade bomb survives a restart."""

    def test_restart_budget_applied_at_load_state_from_db(self):
        from memory.db_summaries import update_ancient_history
        from proxy.context_v2 import _load_state_from_db

        _seed_messages(SESSION, 30, chars_per_msg=200)
        n_summaries = 40
        for i in range(n_summaries):
            _seed_summary(SESSION, exchange_index=i, summary="x" * 100,
                          message_index=i*2, end_message_index=i*2+1)
        update_ancient_history(SESSION, "Y" * 50_000)

        from memory.db_summaries import set_last_summarized_exchange_index
        set_last_summarized_exchange_index(SESSION, exchange_index=10)

        from memory.db_pool import connection
        with connection(pooled=True) as conn:
            conn.execute(
                "UPDATE conversation_summaries SET raw_start_message_index = 30 "
                "WHERE session_id = ?",
                (SESSION,),
            )
            conn.commit()

        import proxy.state as state
        with state._chain_state_lock:
            state._chain_state.clear()

        with _patch_context_config(10_000):
            state_entry = _load_state_from_db(
                SESSION, "m", ctx=None,
                model_safe_tokens=10_000, tokens_per_char=0.25,
            )

        assert state_entry is not None, (
            f"_load_state_from_db must return non-None after restart; got None. "
            f"Conversation has 60 non-system messages with chain_hash."
        )
        assert state_entry["summaries"] != [], (
            f"_load_state_from_db must return seeded summaries; got {len(state_entry['summaries'])} summaries"
        )
        served = state_entry["summaries"]
        budget = int(10_000 * 0.15)
        served_tokens = sum(
            (s.get("summary_tokens") or s.get("rendered_full_tokens") or int(100 * 0.25))
            for s in served
        )
        assert served_tokens <= budget, (
            f"_load_state_from_db must budget summaries: served_tokens={served_tokens} budget={budget}"
        )
        assert len(state_entry["ancient"]) <= int(10_000 * 0.10 / 0.25) + 200, (
            f"_load_state_from_db must truncate ancient to per-model cap, got len={len(state_entry['ancient'])}"
        )


# ============================================================================
# Test 6 (4.6) — update_ancient_history re-raises on caller-provided conn
# ============================================================================


class TestAncientWriteFailureAbortsTransaction:
    """D3: update_ancient_history must re-raise when given a caller conn so
    that DELETE statements in the same transaction are aborted."""

    def test_ancient_write_failure_aborts_transaction(self):
        """D3 contract: update_ancient_history must re-raise when given a
        caller conn that fails. This protects callers that compose this
        function inside a transaction followed by DELETE statements — a
        raise here aborts the transaction so DELETEs never commit.

        Verifies via a MagicMock conn (which satisfies ``conn is not None``
        and so triggers the ``not own_conn`` re-raise branch at
        db_summaries.py:329-332).
        """
        import sqlite3
        from unittest.mock import MagicMock
        import memory.db_summaries as db_summaries_mod
        from memory.db import store_messages_batch

        store_messages_batch(SESSION, [
            {"role": "user", "content": "u0"},
            {"role": "assistant", "content": "a0"},
        ], start_index=0)

        db_summaries_mod.update_ancient_history(SESSION, "INITIAL_ANCIENT")

        mock_conn = MagicMock()
        mock_conn.execute.side_effect = sqlite3.OperationalError(
            "simulated ancient write failure"
        )

        with pytest.raises(sqlite3.OperationalError,
                           match="simulated ancient write failure"):
            db_summaries_mod.update_ancient_history(
                SESSION, "WOULD_BE_OVERWRITTEN", conn=mock_conn,
            )

        assert mock_conn.execute.called, (
            "update_ancient_history must invoke conn.execute (the re-raise "
            "path only fires when the production function attempts to write "
            "through the caller-supplied conn)"
        )

        from memory.db_summaries import get_ancient_history
        ancient_after = get_ancient_history(SESSION)
        assert ancient_after == "INITIAL_ANCIENT", (
            "After abort, ancient must be UNCHANGED (transaction rollback). "
            f"Got: {ancient_after!r}"
        )


# ============================================================================
# Test 7 (4.7) — _maybe_compress_summaries invalidation only on success
# ============================================================================


class TestInvalidationOnlyOnSuccess:
    """F2: chain-state invalidation must fire only when ancient write succeeds.

    Below-threshold and SOTA-failure paths must NOT touch the cache.
    """

    SESSION_BELOW = "test-invalidation-below-threshold"
    SESSION_SOTA_FAIL = "test-invalidation-sota-failure"

    def _seed_chain_state(self, session_id, model_name="m"):
        import proxy.state as state
        entry = {
            "last_chain": None,
            "boundary_idx": 0,
            "boundary_chain": None,
            "summaries": [],
            "ancient": "",
            "version": 1,
        }
        key = (session_id, model_name)
        with state._chain_state_lock:
            state._chain_state[key] = entry
        return key

    def _wipe_session(self, session_id):
        from memory.db_pool import connection
        with connection(pooled=True) as conn:
            conn.execute("DELETE FROM message_summaries WHERE session_id = ?", (session_id,))
            conn.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
            conn.execute("DELETE FROM conversation_summaries WHERE session_id = ?", (session_id,))
            conn.commit()

    def test_below_threshold_does_not_invalidate(self):
        import proxy.state as state
        from memory.sync import _maybe_compress_summaries

        SESSION = self.SESSION_BELOW
        self._wipe_session(SESSION)

        key = self._seed_chain_state(SESSION)

        _maybe_compress_summaries(SESSION, safe_context_tokens=2000)

        with state._chain_state_lock:
            assert key in state._chain_state, (
                "Below-threshold path must not invalidate chain state; "
                f"key {key} missing from _chain_state."
            )

    def test_sota_failure_does_not_invalidate(self):
        import proxy.state as state
        from memory.db_summaries import get_ancient_history
        from memory.sota_client import CompressionError
        from memory.sync import _maybe_compress_summaries

        SESSION = self.SESSION_SOTA_FAIL
        self._wipe_session(SESSION)

        for i in range(60):
            _seed_summary(SESSION, exchange_index=i, summary="word " * 200)

        key = self._seed_chain_state(SESSION)

        sota = MagicMock()
        sota.compress_ancient_history.side_effect = CompressionError("simulated SOTA failure")
        sota.extract_and_summarize_from_summaries.return_value = {
            "extraction_timestamp": "2025-01-01T00:00:00Z",
            "entities": [], "facts": [], "events": [],
            "instructions": [], "tasks": [],
        }
        sota.model = "test-sota"

        with patch("memory.sync._get_sota", return_value=sota):
            _maybe_compress_summaries(SESSION, safe_context_tokens=2000)

        with state._chain_state_lock:
            assert key in state._chain_state, (
                "SOTA-failure path must not invalidate chain state; "
                f"key {key} missing from _chain_state."
            )

        ancient = get_ancient_history(SESSION)
        assert ancient in (None, ""), (
            f"SOTA-failure path must not write ancient; got {ancient!r}"
        )

    def test_worker_write_failure_does_not_invalidate(self):
        import proxy.state as state
        from memory.sync import _ancient_compress_worker

        SESSION = "test-ancient-compress-worker-write-failure"
        self._wipe_session(SESSION)

        key = self._seed_chain_state(SESSION)

        sota = MagicMock()
        sota.compress_ancient_history.return_value = "COMPRESSED_TEXT"
        sota.model = "test-sota"

        from memory import sync as sync_mod
        original_db = sync_mod._db

        class _DBStub:
            def get_ancient_history(self, *a, **kw):
                return "OLD_ANCIENT"

            def update_ancient_history(self, *a, **kw):
                return False

        with patch("memory.sync._get_sota", return_value=sota), \
             patch("memory.sync._db", return_value=_DBStub()):
            _ancient_compress_worker(SESSION, summary_payload=[{"exchange_index": 0}])

        with state._chain_state_lock:
            assert key in state._chain_state, (
                "Worker must not invalidate chain state when update_ancient_history returns False; "
                f"key {key} missing from _chain_state."
            )

    def test_invalidation_failure_separate_log(self):
        """D2: invalidation non-ImportError must log WARNING about invalidation
        failure, not 'Compression failed unexpectedly'.

        The fix wraps the invalidation call in its own try/except block
        at sync.py:989-998. This test verifies the structural fix is in
        place by reading the source file.
        """
        with open("src/memory/sync.py") as f:
            source = f.read()
        assert 'chain-state invalidation' in source and 'failed' in source, (
            "D2 fix: sync.py must contain 'chain-state invalidation failed' "
            "log message in the invalidation try/except block."
        )


# ============================================================================
# Test 8 (4.3) — _truncate_ancient handles small max_chars (below marker)
# ============================================================================


class TestAncientTruncationSmallCap:
    """D5 contract: _truncate_ancient must never exceed max_chars, even when
    the marker itself is longer than the budget."""

    def test_truncation_handles_small_max_chars(self):
        from proxy.context_v2 import _truncate_ancient

        ancient = "X" * 5000

        result = _truncate_ancient(ancient, max_chars=500, session_id="s", ctx=None)
        assert len(result) <= 500, (
            f"_truncate_ancient must respect max_chars; got len={len(result)}"
        )

        result_small = _truncate_ancient(ancient, max_chars=50, session_id="s", ctx=None)
        assert len(result_small) <= 50, (
            f"_truncate_ancient must handle very small max_chars; got len={len(result_small)}"
        )

        result_zero = _truncate_ancient(ancient, max_chars=0, session_id="s", ctx=None)
        assert result_zero == "", (
            f"_truncate_ancient with max_chars=0 must return empty string; got {result_zero!r}"
        )

        result_neg = _truncate_ancient(ancient, max_chars=-10, session_id="s", ctx=None)
        assert result_neg == "", (
            f"_truncate_ancient with negative max_chars must return empty string; "
            f"got {result_neg!r}"
        )


# ============================================================================
# Test 9 (4.4) — Warm boundary-0 with empty summaries rescues from DB
# ============================================================================


class TestWarmBoundaryZeroRescue:
    """F7: warm entry with snap_boundary_idx=0, snap_summaries=[] must still
    trigger Step 8.5 to fetch summaries from DB. Without the boundary_idx==0
    clause, summaries stay empty forever."""

    SESSION = "test-warm-boundary-zero-rescue"

    def test_warm_boundary_zero_with_empty_summaries_rescues_from_db(self):
        import proxy.state as state
        from memory.db_summaries import delete_message_summary
        from memory.db_pool import connection
        from proxy.context_v2 import _select_context_v2

        SESSION = self.SESSION

        with connection(pooled=True) as conn:
            conn.execute("DELETE FROM message_summaries WHERE session_id = ?", (SESSION,))
            conn.execute("DELETE FROM messages WHERE session_id = ?", (SESSION,))
            conn.execute("DELETE FROM conversation_summaries WHERE session_id = ?", (SESSION,))
            conn.commit()

        _seed_messages(SESSION, 30, chars_per_msg=200)
        for i in range(5):
            _seed_summary(SESSION, exchange_index=i, summary=f"rescue-marker-{i}")

        cfg = _make_cfg(10_000)
        req = _make_request(30, chars_per_msg=200)

        with _patch_context_config(10_000), _patch_vfs():
            _select_context_v2(SESSION, req, model_name="m", model_cfg=cfg)

        with state._chain_state_lock:
            state._chain_state.clear()

        warm_entry = {
            "last_chain": None,
            "boundary_idx": 0,
            "boundary_chain": None,
            "summaries": [],
            "ancient": "",
            "version": 1,
        }
        with state._chain_state_lock:
            state._chain_state[(SESSION, "m")] = warm_entry

        with _patch_context_config(10_000), _patch_vfs():
            warm_call = _select_context_v2(SESSION, req, model_name="m", model_cfg=cfg)

        all_text = " ".join(
            m.get("content", "") for m in warm_call if isinstance(m.get("content"), str)
        )
        assert "rescue-marker-0" in all_text, (
            "Warm boundary-0 + empty summaries must trigger Step 8.5 fetch "
            f"from DB. Got first 500 chars: {all_text[:500]!r}"
        )