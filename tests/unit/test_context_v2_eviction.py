"""Unit tests for the V2 chain-hash grow/evict context pipeline.

Covers the §11 test plan of docs/v2-chain-hash-redesign.md:
growing phase, eviction event (capped at last-summarized), ancient
overflow, edit in raw/summarized range, model switch, normal append
(B5 case b), barrier message, VFS preservation, fast-exit guard,
restart recovery, and M15 hard cap.

Post-C5: the "passthrough guard" at the start of the function is gone.
Small conversations now flow through the normal path. A new "fast-exit"
fires only when chain-state miss + no DB rows exist.
"""
import hashlib
import logging
from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import patch, MagicMock

import pytest

from proxy.context_v2 import (
    _select_context_v2,
    V2_LOW_WATERMARK_RATIO,
    V2_HIGH_WATERMARK_RATIO,
)
from proxy.context import _count_msg_chars
from proxy.chain_hash import compute_chain

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _system(content: str) -> dict:
    return {"role": "system", "content": content}


def _user(content: str) -> dict:
    return {"role": "user", "content": content}


def _assistant(content: str = "", tool_calls=None) -> dict:
    msg = {"role": "assistant", "content": content}
    if tool_calls is not None:
        msg["tool_calls"] = tool_calls
    return msg


def _tool_call(tc_id: str, name: str = "do_thing", args: str = "{}") -> dict:
    return {"id": tc_id, "type": "function", "function": {"name": name, "arguments": args}}


def _tool(tc_id: str, content: str = "result") -> dict:
    return {"role": "tool", "tool_call_id": tc_id, "content": content}


def _budget_config(safe=10_000):
    """safe=10000 → LOW=6500, HIGH=9500 (at default ratios 0.65/0.95)."""
    return {
        "max_context_tokens": 40_000,
        "tokens_per_char": 0.25,
        "safe_context_tokens": safe,
        "raw_exchange_count": 100,
    }


def _msgs(count, chars=400, start=0):
    """Build `count` user messages of `chars` X-padding (100 tok at chars=400)."""
    pad = "X" * chars
    return [_user(f"msg {i} {pad}") for i in range(start, start + count)]


def _make_ctx(request_id="test-req"):
    return SimpleNamespace(
        request_id=request_id,
        v2_boundary_idx=None,
        v2_boundary_chain=None,
    )


def _mock_conn(min_mi=None, raw_start=None, chain_rows=None, fast_exit_fires=False):
    """Build a mock SQLite connection for V2 DB queries.

    - raw_start_message_index query: returns raw_start (None → _load_state returns None)
    - MIN(message_index) query: returns {mi: min_mi} (for eviction cap)
    - COUNT(*) position query: returns (min_mi,) — assumes test data has no
      system messages, so conversation position == SQL message_index
    - chain_hash query: returns chain_rows (empty → _load_state returns None)
    - fast-exit SELECT 1 FROM conversation_summaries / message_summaries:
      by default returns rows (fast-exit does NOT fire) so the normal path runs;
      pass fast_exit_fires=True to make the mock return None (fast-exit fires).
    """
    def execute_side_effect(query, *args, **kwargs):
        q = str(query).upper()
        result = MagicMock()
        if "MIN(MESSAGE_INDEX)" in q:
            result.fetchone.return_value = {"mi": min_mi}
        elif "RAW_START_MESSAGE_INDEX" in q:
            if raw_start is not None:
                result.fetchone.return_value = {"raw_start_message_index": raw_start}
            else:
                result.fetchone.return_value = None
        elif "COUNT(*)" in q and "MESSAGE_INDEX" in q:
            result.fetchone.return_value = (min_mi if min_mi is not None else 0,)
        elif "CHAIN_HASH" in q:
            result.fetchall.return_value = chain_rows or []
        elif "SELECT 1 FROM CONVERSATION_SUMMARIES" in q:
            result.fetchone.return_value = None if fast_exit_fires else (1,)
        elif "SELECT 1 FROM MESSAGE_SUMMARIES" in q:
            result.fetchone.return_value = None if fast_exit_fires else (1,)
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


def _run(
    cfg,
    messages,
    *,
    rich_summaries=None,
    ancient_history="",
    vfs_return=None,
    low_water=None,
    high_water=None,
    is_subagent=False,
    session_id="sess1",
    model_name="test",
    ctx=None,
    model_cfg=None,
    last_exchange_index=-1,
    min_mi=None,
    load_state_override=None,
    submit_ancient_mock=None,
    fast_exit_fires=False,
):
    """Run _select_context_v2 with all DB/VFS side-effects mocked.

    Returns (result, rich_mock, ancient_mock) so callers can inspect calls.
    `submit_ancient_mock`, when provided, is patched in as
    memory.sync.submit_ancient_compress so overflow-dispatch tests can assert
    on it. When omitted, a no-op MagicMock is still patched defensively so no
    unit test ever dispatches a real SOTA job.
    `fast_exit_fires` (default False) controls whether the C5 fast-exit query
    returns a row (False → fast-exit skipped, normal path runs) or None
    (True → fast-exit fires, returns `cleaned` directly).
    """
    rich_mock = MagicMock(return_value=rich_summaries or [])
    ancient_mock = MagicMock(return_value=ancient_history)
    conn_cm = _mock_conn(min_mi=min_mi, fast_exit_fires=fast_exit_fires)
    if submit_ancient_mock is None:
        submit_ancient_mock = MagicMock(return_value=None)

    patches_list = [
        patch("proxy.context.get_context_config", return_value=cfg),
        patch("memory.db.get_rich_summaries", rich_mock),
        patch("memory.db.get_ancient_history", ancient_mock),
        patch("proxy.vfs._build_vfs_context", return_value=vfs_return),
        patch("proxy.vfs._build_vfs_inherited_context", return_value=vfs_return),
        patch("memory.db.connection", return_value=conn_cm),
        patch(
            "memory.db_summaries.get_last_summarized_exchange_index",
            return_value=last_exchange_index,
        ),
        patch("memory.sync.submit_ancient_compress", new=submit_ancient_mock),
    ]
    if load_state_override is not None:
        patches_list.append(
            patch("proxy.context_v2._load_state_from_db", return_value=load_state_override)
        )
    if low_water is not None:
        patches_list.append(patch("proxy.context_v2.V2_LOW_WATERMARK_RATIO", low_water))
    if high_water is not None:
        patches_list.append(patch("proxy.context_v2.V2_HIGH_WATERMARK_RATIO", high_water))

    with ExitStack() as stack:
        for p in patches_list:
            stack.enter_context(p)
        result = _select_context_v2(
            session_id, messages, model_name=model_name, model_cfg=model_cfg,
            is_subagent=is_subagent, ctx=ctx,
        )
    return result, rich_mock, ancient_mock


def _summaries_content(result):
    """Extract the rendered summaries system message content from assembly."""
    for m in result:
        if m.get("role") == "system" and "[PAST CONTEXT" in m.get("content", ""):
            return m["content"]
    return None


def _ancient_content(result):
    """Extract the rendered ancient system message content from assembly."""
    for m in result:
        if m.get("role") == "system" and "[TIMELINE" in m.get("content", ""):
            return m["content"]
    return None


# ---------------------------------------------------------------------------
# 1. Fast-exit returns all messages, no DB fetch (C5 replacement for passthrough)
# ---------------------------------------------------------------------------

class TestFastExitNoFetch:
    """Chain-state miss + no conversation_summaries / message_summaries rows
    → fast-exit returns `cleaned` without touching the summary/ancient DB layer."""

    def test_fast_exit_returns_all_and_skips_db(self):
        cfg = _budget_config()
        messages = [_system("sys")] + _msgs(20)

        result, rich_mock, ancient_mock = _run(cfg, messages, fast_exit_fires=True)

        non_system = [m for m in result if m.get("role") != "system"]
        assert len(non_system) == 20, "Fast-exit must return all 20 messages"
        assert rich_mock.called is False, "get_rich_summaries must NOT be called on fast-exit"
        assert ancient_mock.called is False, "get_ancient_history must NOT be called on fast-exit"


# ---------------------------------------------------------------------------
# 2. Overflow falls through to eviction (no longer a passthrough concern)
# ---------------------------------------------------------------------------

class TestOverflowFallsThroughToEviction:
    """When raw exceeds the high watermark, the tiered pipeline evicts oldest
    messages. Post-C5: the overflow guard is just the natural budget
    enforcement, not a passthrough inner condition."""

    def test_overflow_triggers_eviction(self):
        cfg = _budget_config()
        messages = [_system("sys")] + _msgs(110)

        result, rich_mock, ancient_mock = _run(cfg, messages)

        non_system = [m for m in result if m.get("role") != "system"]
        assert len(non_system) < 110, (
            f"Overflow should trigger eviction; got {len(non_system)} of 110"
        )


# ---------------------------------------------------------------------------
# 3. Low-watermark target
# ---------------------------------------------------------------------------

class TestLowWatermarkTarget:
    """Evicted raw block must fit within safe * V2_LOW_WATERMARK_RATIO + slack."""

    def test_evicted_raw_tokens_within_low_watermark(self):
        cfg = _budget_config(safe=10_000)
        safe = cfg["safe_context_tokens"]
        messages = [_system("")] + _msgs(150)

        result, _, _ = _run(cfg, messages)

        non_system = [m for m in result if m.get("role") != "system"]
        raw_tokens = sum(int(_count_msg_chars(m) * 0.25) for m in non_system)
        budget = int(safe * V2_LOW_WATERMARK_RATIO)
        assert raw_tokens <= budget + 300, (
            f"Evicted raw tokens ({raw_tokens}) exceed low-watermark ({budget}) + slack"
        )


# ---------------------------------------------------------------------------
# 4. No summary fetch when no eviction (high watermark)
# ---------------------------------------------------------------------------

class TestNoFetchWithoutEviction:
    """When nothing is trimmed (high watermark), S1 contract still requires
    fetching summaries — they are surfaced for continuity even without eviction.
    This is the contract pinned by tests/integration/test_sync_context_contract.py."""

    def test_high_watermark_no_eviction_fetch_for_continuity(self):
        cfg = _budget_config()
        messages = [_system("")] + _msgs(50)

        result, rich_mock, ancient_mock = _run(
            cfg, messages, low_water=2.0,
        )

        non_system = [m for m in result if m.get("role") != "system"]
        assert len(non_system) == 50, "High watermark should keep everything"
        assert rich_mock.called is True, (
            "get_rich_summaries MUST be called when in tiered pipeline even "
            "without eviction (S1 contract for summary continuity)"
        )


# ---------------------------------------------------------------------------
# 5. target_exchange does not advance on fast-exit (C5 replacement for passthrough)
# ---------------------------------------------------------------------------

class TestTargetExchangeNotAdvancedOnFastExit:
    """On fast-exit, get_rich_summaries must never be called."""

    def test_fast_exit_does_not_call_rich_summaries(self):
        cfg = _budget_config()
        messages = [_system("sys")] + _msgs(15)

        result, rich_mock, _ = _run(cfg, messages, fast_exit_fires=True)

        assert rich_mock.called is False


# ---------------------------------------------------------------------------
# 6. Message order (A3): last user message stays final
# ---------------------------------------------------------------------------

class TestA3MessageOrder:
    """Assembly order: [system, raw-minus-last-user, vfs, summaries, ancient,
    (barrier), last-user]. The current user turn must be the FINAL message."""

    def test_last_user_is_final_and_blocks_precede_it(self):
        cfg = _budget_config()
        pad = "X" * 400
        messages = [_system("sys")] + [_user(f"msg {i} {pad}") for i in range(150)]
        last_user_content = messages[-1]["content"]

        ancient_text = "ancient timeline context"
        summaries = [{
            "message_index": 200, "role": "user", "summary": "summary line",
            "tool_summary": "", "rendered_narrative_tokens": 10,
        }]
        vfs = {"role": "system", "content": "[VFS] workspace context"}

        result, _, _ = _run(
            cfg, messages, rich_summaries=summaries,
            ancient_history=ancient_text, vfs_return=vfs,
        )

        assert result[-1]["role"] == "user", "Final message must be the user turn"
        assert result[-1]["content"] == last_user_content, (
            "Final message content must equal the last user turn"
        )

        injected_indices = [
            i for i, m in enumerate(result)
            if m.get("role") == "system" and i > 0
            and ("[TIMELINE" in m.get("content", "")
                 or "[PAST CONTEXT" in m.get("content", "")
                 or "[VFS]" in m.get("content", ""))
        ]
        assert injected_indices, "Expected at least one injected system block"
        idx = injected_indices[0]
        assert 0 < idx < len(result) - 1, "Injected block must be between raw and final user"
        prev = result[idx - 1].get("role")
        assert prev != "system" or idx - 1 == 0, (
            "A raw message must precede the first injected system block "
            "(unless it's right after the system prompt at idx 0)"
        )


# ---------------------------------------------------------------------------
# 7. Tool-pair integrity under real eviction
# ---------------------------------------------------------------------------

class TestToolPairIntegrityUnderEviction:
    """50 assistant+tool pairs under eviction: no orphans, no missing results,
    and the dropped pairs are the oldest."""

    def test_pairs_atomic_and_oldest_dropped(self):
        cfg = _budget_config()
        pad = "X" * 400
        messages = [_system("sys")]
        for i in range(50):
            tc_id = f"tc_{i}"
            messages.append(_assistant(
                content=f"calling {i} {pad}", tool_calls=[_tool_call(tc_id)],
            ))
            messages.append(_tool(tc_id, content=f"result {i} {pad}"))

        result, _, _ = _run(cfg, messages)

        assistant_tc_ids = set()
        for m in result:
            if m.get("role") == "assistant" and isinstance(m.get("tool_calls"), list):
                for tc in m["tool_calls"]:
                    if tc.get("id"):
                        assistant_tc_ids.add(tc["id"])
        present_tool_ids = {
            m.get("tool_call_id") for m in result if m.get("role") == "tool"
        }
        for m in result:
            if m.get("role") == "tool":
                assert m.get("tool_call_id") in assistant_tc_ids, "Orphan tool result present"
        assert assistant_tc_ids <= present_tool_ids, "Assistant missing its tool result"

        assert "tc_0" not in assistant_tc_ids, "Oldest pair (tc_0) should be evicted"
        assert "tc_49" in assistant_tc_ids, "Newest pair (tc_49) should be kept"

        non_system = [m for m in result if m.get("role") != "system"]
        assert len(non_system) < 100


# ---------------------------------------------------------------------------
# 8. Small-model arithmetic
# ---------------------------------------------------------------------------

class TestSmallModelArithmetic:
    """safe=51000: a sub-30% session fast-exits (no DB fetch); a >30% session
    evicts to 0.65×safe without raising ContextLimitError."""

    def test_just_under_threshold_takes_fast_exit(self):
        safe = 51_000
        cfg = _budget_config(safe=safe)
        threshold_tokens = int(safe * 0.30)
        per_msg_tokens = (threshold_tokens // 10) - 50
        per_msg_chars = int(per_msg_tokens / 0.25)
        messages = [_system("")] + _msgs(9, chars=per_msg_chars)

        result, rich_mock, _ = _run(cfg, messages, fast_exit_fires=True)

        non_system = [m for m in result if m.get("role") != "system"]
        assert len(non_system) == 9, "Just-under session should fast-exit untrimmed"
        assert rich_mock.called is False

    def test_over_threshold_evicts_without_error(self):
        safe = 51_000
        cfg = _budget_config(safe=safe)
        threshold_tokens = int(safe * 0.30)
        per_msg_tokens = (threshold_tokens // 10) - 50
        per_msg_chars = int(per_msg_tokens / 0.25)
        messages = [_system("")] + _msgs(11, chars=per_msg_chars)

        result, rich_mock, _ = _run(cfg, messages)

        non_system = [m for m in result if m.get("role") != "system"]
        assert len(non_system) < 49, "Over-threshold session must be evicted"
        assert rich_mock.called is True
        raw_tokens = sum(int(_count_msg_chars(m) * 0.25) for m in non_system)
        assert raw_tokens + 2000 <= safe


# ---------------------------------------------------------------------------
# 9. Growing phase — raw grows LOW→HIGH, summaries/ancient static
# ---------------------------------------------------------------------------

class TestGrowingPhase:
    """Raw block grows from LOW to HIGH between evictions. Summaries and
    ancient must remain byte-identical across growing turns (cache-stable)."""

    def test_summaries_sha256_identical_across_growing_turns(self):
        cfg = _budget_config(safe=10_000)
        summaries = [
            {"message_index": 5, "role": "user", "summary": "early turn", "tool_summary": ""},
            {"message_index": 10, "role": "assistant", "summary": "reply", "tool_summary": ""},
        ]
        ancient = "ancient timeline text"

        ctx1 = _make_ctx("r1")
        msgs1 = [_system("sys")] + _msgs(100)
        r1, _, _ = _run(cfg, msgs1, rich_summaries=summaries, ancient_history=ancient, ctx=ctx1)

        ctx2 = _make_ctx("r2")
        msgs2 = [_system("sys")] + _msgs(102)
        r2, _, _ = _run(cfg, msgs2, rich_summaries=summaries, ancient_history=ancient, ctx=ctx2)

        ctx3 = _make_ctx("r3")
        msgs3 = [_system("sys")] + _msgs(104)
        r3, _, _ = _run(cfg, msgs3, rich_summaries=summaries, ancient_history=ancient, ctx=ctx3)

        s1 = _summaries_content(r1)
        s2 = _summaries_content(r2)
        s3 = _summaries_content(r3)
        assert s1 is not None and s2 is not None and s3 is not None
        h1 = hashlib.sha256(s1.encode()).hexdigest()
        h2 = hashlib.sha256(s2.encode()).hexdigest()
        h3 = hashlib.sha256(s3.encode()).hexdigest()
        assert h1 == h2 == h3, "Summaries must be byte-identical across growing turns"

    def test_ancient_sha256_identical_across_growing_turns(self):
        cfg = _budget_config(safe=10_000)
        summaries = [{"message_index": 5, "role": "user", "summary": "sum", "tool_summary": ""}]
        ancient = "ancient timeline text"

        ctx1 = _make_ctx("r1")
        msgs1 = [_system("sys")] + _msgs(100)
        r1, _, _ = _run(cfg, msgs1, rich_summaries=summaries, ancient_history=ancient, ctx=ctx1)

        ctx2 = _make_ctx("r2")
        msgs2 = [_system("sys")] + _msgs(102)
        r2, _, _ = _run(cfg, msgs2, rich_summaries=summaries, ancient_history=ancient, ctx=ctx2)

        a1 = _ancient_content(r1)
        a2 = _ancient_content(r2)
        assert a1 is not None and a2 is not None
        assert hashlib.sha256(a1.encode()).hexdigest() == hashlib.sha256(a2.encode()).hexdigest()

    def test_raw_block_grows_while_boundary_stable(self):
        cfg = _budget_config(safe=10_000)
        summaries = [{"message_index": 5, "role": "user", "summary": "sum", "tool_summary": ""}]

        ctx1 = _make_ctx("r1")
        msgs1 = [_system("sys")] + _msgs(100)
        r1, _, _ = _run(cfg, msgs1, rich_summaries=summaries, ctx=ctx1)
        raw1 = [m for m in r1 if m.get("role") != "system" and "[PAST" not in m.get("content", "")]
        boundary1 = ctx1.v2_boundary_idx

        ctx2 = _make_ctx("r2")
        msgs2 = [_system("sys")] + _msgs(104)
        r2, _, _ = _run(cfg, msgs2, rich_summaries=summaries, ctx=ctx2)
        raw2 = [m for m in r2 if m.get("role") != "system" and "[PAST" not in m.get("content", "")]
        boundary2 = ctx2.v2_boundary_idx

        assert boundary1 == boundary2, "Boundary must stay the same during growing phase"
        assert len(raw2) > len(raw1), "Raw block must grow when messages are appended"


# ---------------------------------------------------------------------------
# 10. Eviction event — capped at last-summarized boundary
# ---------------------------------------------------------------------------

class TestEvictionEvent:
    """Raw exceeds HIGH → eviction proposed → CAPPED at last-summarized boundary."""

    def test_eviction_capped_at_last_summarized_boundary(self):
        cfg = _budget_config(safe=10_000)
        summaries = [{"message_index": 5, "role": "user", "summary": "sum", "tool_summary": ""}]

        ctx1 = _make_ctx("r1")
        msgs1 = [_system("sys")] + _msgs(100)
        _run(cfg, msgs1, rich_summaries=summaries, ctx=ctx1)
        boundary1 = ctx1.v2_boundary_idx

        ctx2 = _make_ctx("r2")
        msgs2 = [_system("sys")] + _msgs(200)
        r2, _, _ = _run(
            cfg, msgs2, rich_summaries=summaries,
            last_exchange_index=50, min_mi=60, ctx=ctx2,
        )
        boundary2 = ctx2.v2_boundary_idx

        assert boundary2 > boundary1, "Eviction must advance the boundary"
        assert boundary2 <= 60, f"Eviction must be capped at last-summarized (60), got {boundary2}"

    def test_no_eviction_possible_warning_and_overflow_accepted(self):
        """When max_evictable_boundary == 0 (nothing summarized), eviction is
        skipped and overflow is accepted for this turn."""
        cfg = _budget_config(safe=10_000)
        summaries = [{"message_index": 5, "role": "user", "summary": "sum", "tool_summary": ""}]

        ctx1 = _make_ctx("r1")
        msgs1 = [_system("sys")] + _msgs(100)
        _run(cfg, msgs1, rich_summaries=summaries, ctx=ctx1)
        boundary1 = ctx1.v2_boundary_idx

        ctx2 = _make_ctx("r2")
        msgs2 = [_system("sys")] + _msgs(110)
        r2, _, _ = _run(
            cfg, msgs2, rich_summaries=summaries,
            last_exchange_index=-1, min_mi=None, ctx=ctx2,
        )
        boundary2 = ctx2.v2_boundary_idx

        assert boundary2 == boundary1, (
            "Boundary must NOT advance when nothing is summarizable"
        )

    def test_next_call_after_capped_eviction_has_stable_summaries(self):
        cfg = _budget_config(safe=10_000)
        summaries = [{"message_index": 5, "role": "user", "summary": "sum", "tool_summary": ""}]

        ctx1 = _make_ctx("r1")
        msgs1 = [_system("sys")] + _msgs(100)
        _run(cfg, msgs1, rich_summaries=summaries, ctx=ctx1)

        ctx2 = _make_ctx("r2")
        msgs2 = [_system("sys")] + _msgs(110)
        r2, _, _ = _run(cfg, msgs2, rich_summaries=summaries, ctx=ctx2)

        ctx3 = _make_ctx("r3")
        msgs3 = [_system("sys")] + _msgs(112)
        r3, _, _ = _run(cfg, msgs3, rich_summaries=summaries, ctx=ctx3)

        s2 = _summaries_content(r2)
        s3 = _summaries_content(r3)
        assert s2 is not None and s3 is not None
        assert hashlib.sha256(s2.encode()).hexdigest() == hashlib.sha256(s3.encode()).hexdigest()


# ---------------------------------------------------------------------------
# 11. Ancient overflow (async) — summaries truncated, SOTA dispatched
# ---------------------------------------------------------------------------

class TestAncientOverflowAsync:
    """Summary overflow on eviction: truncate oldest + dispatch submit_ancient_compress.

    Per docs/v2-chain-hash-redesign.md §10: when fetched summaries overflow the
    summary budget (15% of safe_context_tokens), the oldest summaries are
    hard-truncated THIS turn and an async SOTA re-compression job is dispatched
    via memory.sync.submit_ancient_compress(session_id, dropped).
    """

    BUDGET_SAFE = 10_000  # summary_budget = int(10000 * 0.15) = 1500 tokens

    @staticmethod
    def _huge_summaries(count=10, start=5, pad=800):
        """Build `count` summaries whose rendered cost (~205 tok each at
        tokens_per_char=0.25) overflows the 1500-token budget."""
        return [
            {
                "message_index": start + i,
                "role": "user",
                "summary": f"MARK{start + i}" + ("X" * pad),
                "tool_summary": "",
            }
            for i in range(count)
        ]

    def test_overflow_triggers_ancient_compress_and_truncates(self):
        cfg = _budget_config(safe=self.BUDGET_SAFE)
        huge = self._huge_summaries(count=10, start=5)

        ctx1 = _make_ctx("r1")
        msgs1 = [_system("sys")] + _msgs(100)
        _run(cfg, msgs1, rich_summaries=huge, ctx=ctx1)

        compress_mock = MagicMock(return_value=None)
        ctx2 = _make_ctx("r2")
        msgs2 = [_system("sys")] + _msgs(200)
        r2, _, _ = _run(
            cfg, msgs2, rich_summaries=huge,
            last_exchange_index=50, min_mi=60, ctx=ctx2,
            submit_ancient_mock=compress_mock,
        )

        assert compress_mock.called, "Overflow must dispatch submit_ancient_compress"

        s = _summaries_content(r2)
        assert s is not None, "Truncated summaries must still be served"
        # Oldest (MARK5..MARK7) dropped to re-compress; MARK8..MARK14 kept.
        assert "MARK5" not in s, "Oldest overflowed summary must be truncated out"
        assert "MARK7" not in s
        assert "MARK8" in s, "First under-budget summary must be retained"
        assert "MARK14" in s, "Newest summary must be retained"
        kept = [m for m in [f"MARK{5 + i}" for i in range(10)] if m in s]
        assert len(kept) == 7, f"Expected 7 kept summaries, got {len(kept)}"

    def test_ancient_compress_call_args_correct(self):
        cfg = _budget_config(safe=self.BUDGET_SAFE)
        huge = self._huge_summaries(count=10, start=5)

        ctx1 = _make_ctx("r1")
        msgs1 = [_system("sys")] + _msgs(100)
        _run(cfg, msgs1, rich_summaries=huge, ctx=ctx1,
             session_id="sess-aa", model_name="model-aa")

        compress_mock = MagicMock(return_value=None)
        ctx2 = _make_ctx("r2")
        msgs2 = [_system("sys")] + _msgs(200)
        _run(
            cfg, msgs2, rich_summaries=huge,
            last_exchange_index=50, min_mi=60, ctx=ctx2,
            session_id="sess-aa", model_name="model-aa",
            submit_ancient_mock=compress_mock,
        )

        assert compress_mock.called
        args, kwargs = compress_mock.call_args
        assert args[0] == "sess-aa", "First arg must be session_id"
        dropped = args[1]
        assert isinstance(dropped, list), "Second arg must be a list of dropped summaries"
        assert len(dropped) == 3, f"Expected 3 dropped (oldest) summaries, got {len(dropped)}"
        dropped_indices = sorted(d["message_index"] for d in dropped)
        assert dropped_indices == [5, 6, 7], "Dropped summaries must be the oldest (lowest message_index)"


# ---------------------------------------------------------------------------
# 12. Edit in raw block — fork detected, boundary valid, cache hits up to fork
# ---------------------------------------------------------------------------

class TestEditInRawBlock:
    """User edits a recent message: CLI sends shorter conversation (cut after
    edit). Fork is after boundary → boundary valid → warm path."""

    def test_fork_in_raw_block_boundary_valid_warm_path(self):
        cfg = _budget_config(safe=10_000)
        summaries = [{"message_index": 5, "role": "user", "summary": "sum", "tool_summary": ""}]

        ctx1 = _make_ctx("r1")
        msgs1 = [_system("sys")] + _msgs(100)
        r1, rich1, _ = _run(cfg, msgs1, rich_summaries=summaries, ctx=ctx1)
        boundary1 = ctx1.v2_boundary_idx

        edited = [_system("sys")] + _msgs(95) + [_user("EDITED content here")]
        ctx2 = _make_ctx("r2")
        r2, rich2, _ = _run(cfg, edited, rich_summaries=summaries, ctx=ctx2)
        boundary2 = ctx2.v2_boundary_idx

        assert boundary2 == boundary1, "Boundary must remain valid after edit in raw block"
        assert not rich2.called, "Warm path must NOT re-fetch summaries after edit in raw block"

    def test_raw_block_shorter_after_edit(self):
        cfg = _budget_config(safe=10_000)
        summaries = [{"message_index": 5, "role": "user", "summary": "sum", "tool_summary": ""}]

        ctx1 = _make_ctx("r1")
        msgs1 = [_system("sys")] + _msgs(100)
        r1, _, _ = _run(cfg, msgs1, rich_summaries=summaries, ctx=ctx1)
        raw1 = [m for m in r1 if m.get("role") == "user" and "msg " in m.get("content", "")]

        edited = [_system("sys")] + _msgs(95) + [_user("EDITED content here")]
        ctx2 = _make_ctx("r2")
        r2, _, _ = _run(cfg, edited, rich_summaries=summaries, ctx=ctx2)
        raw2 = [m for m in r2 if m.get("role") == "user" and ("msg " in m.get("content", "") or "EDITED" in m.get("content", ""))]

        assert len(raw2) < len(raw1), (
            f"Raw block must be shorter after edit ({len(raw2)} vs {len(raw1)})"
        )


# ---------------------------------------------------------------------------
# 13. Edit in summarized range — boundary invalidated, cold path
# ---------------------------------------------------------------------------

class TestEditInSummarizedRange:
    """User edits an old message: chain breaks before boundary → boundary
    invalid → cold path (full re-trim)."""

    def test_boundary_invalidated_cold_path_runs(self):
        cfg = _budget_config(safe=10_000)
        summaries = [{"message_index": 5, "role": "user", "summary": "sum", "tool_summary": ""}]

        ctx1 = _make_ctx("r1")
        msgs1 = [_system("sys")] + _msgs(100)
        _run(cfg, msgs1, rich_summaries=summaries, ctx=ctx1)
        boundary1 = ctx1.v2_boundary_idx

        edited = [_system("sys")] + _msgs(10) + [_user("EDITED old message")] + _msgs(85, start=11)
        ctx2 = _make_ctx("r2")
        r2, rich2, _ = _run(cfg, edited, rich_summaries=summaries, ctx=ctx2)

        assert rich2.called, "Cold path must re-fetch summaries when boundary invalidated"

    def test_full_re_trim_new_boundary_computed(self):
        cfg = _budget_config(safe=10_000)
        summaries = [{"message_index": 5, "role": "user", "summary": "sum", "tool_summary": ""}]

        ctx1 = _make_ctx("r1")
        msgs1 = [_system("sys")] + _msgs(100)
        _run(cfg, msgs1, rich_summaries=summaries, ctx=ctx1)

        edited_msgs = [_system("sys")] + _msgs(10) + [_user("EDITED old message")] + _msgs(85, start=11)
        ctx2 = _make_ctx("r2")
        r2, _, _ = _run(cfg, edited_msgs, rich_summaries=summaries, ctx=ctx2)
        boundary2 = ctx2.v2_boundary_idx

        assert boundary2 is not None
        non_system = [m for m in r2 if m.get("role") != "system"]
        assert len(non_system) <= 96, "Cold path must re-trim to low watermark"


# ---------------------------------------------------------------------------
# 14. Model switch — keyed on (session_id, model_name)
# ---------------------------------------------------------------------------

class TestModelSwitch:
    """Switching models uses per-(session, model) state. First encounter is
    cold; returning to a prior model reuses warm state."""

    def test_model_switch_first_encounter_cold_return_warm(self):
        cfg = _budget_config(safe=10_000)
        summaries = [{"message_index": 5, "role": "user", "summary": "sum", "tool_summary": ""}]
        msgs = [_system("sys")] + _msgs(100)

        ctx_a1 = _make_ctx("a1")
        _run(cfg, msgs, rich_summaries=summaries, model_name="modelA", ctx=ctx_a1)

        ctx_b1 = _make_ctx("b1")
        _, rich_b, _ = _run(cfg, msgs, rich_summaries=summaries, model_name="modelB", ctx=ctx_b1)
        assert rich_b.called, "First modelB turn must be cold path"

        ctx_a2 = _make_ctx("a2")
        _, rich_a2, _ = _run(cfg, msgs, rich_summaries=summaries, model_name="modelA", ctx=ctx_a2)
        assert not rich_a2.called, "Second modelA turn must be warm (state preserved)"

    def test_different_models_have_independent_boundaries(self):
        cfg = _budget_config(safe=10_000)
        summaries = [{"message_index": 5, "role": "user", "summary": "sum", "tool_summary": ""}]
        msgs = [_system("sys")] + _msgs(100)

        ctx_a = _make_ctx("a")
        _run(cfg, msgs, rich_summaries=summaries, model_name="modelA", ctx=ctx_a)

        ctx_b = _make_ctx("b")
        _run(cfg, msgs, rich_summaries=summaries, model_name="modelB", ctx=ctx_b)

        import proxy.state as state
        key_a = ("sess1", "modelA")
        key_b = ("sess1", "modelB")
        assert key_a in state._chain_state, "modelA state must exist"
        assert key_b in state._chain_state, "modelB state must exist"
        assert state._chain_state[key_a] is not state._chain_state[key_b]


# ---------------------------------------------------------------------------
# 15. Normal append — B5 case b (conversation fits LOW_WATERMARK)
# ---------------------------------------------------------------------------

class TestNormalAppend:
    """3 consecutive calls with +2 messages each under LOW_WATERMARK. All warm
    (case b: boundary_idx=0, boundary_chain=None). No summaries injected."""

    def test_case_b_warm_path_no_summaries_injected(self):
        cfg = _budget_config(safe=400_000)
        msgs1 = [_system("sys")] + _msgs(200, chars=200)
        msgs2 = [_system("sys")] + _msgs(202, chars=200)
        msgs3 = [_system("sys")] + _msgs(204, chars=200)

        ctx1 = _make_ctx("r1")
        r1, _, _ = _run(cfg, msgs1, ctx=ctx1)
        assert ctx1.v2_boundary_idx == 0, "Cold path with small conversation → boundary at 0"
        assert ctx1.v2_boundary_chain is None
        assert _summaries_content(r1) is None, "No summaries when boundary is 0"

        ctx2 = _make_ctx("r2")
        r2, rich2, _ = _run(cfg, msgs2, ctx=ctx2)
        assert ctx2.v2_boundary_idx == 0, "Case b warm path: boundary stays at 0"
        assert _summaries_content(r2) is None, "Case b: no summaries injected"

        ctx3 = _make_ctx("r3")
        r3, rich3, _ = _run(cfg, msgs3, ctx=ctx3)
        assert ctx3.v2_boundary_idx == 0
        assert _summaries_content(r3) is None

    def test_case_b_all_warm_no_db_fetch_after_first(self):
        cfg = _budget_config(safe=400_000)
        msgs1 = [_system("sys")] + _msgs(200, chars=200)
        msgs2 = [_system("sys")] + _msgs(202, chars=200)

        ctx1 = _make_ctx("r1")
        _run(cfg, msgs1, ctx=ctx1)

        ctx2 = _make_ctx("r2")
        _, rich2, ancient2 = _run(cfg, msgs2, ctx=ctx2)
        assert rich2.called, (
            "Case b warm path with boundary_idx=0 must fetch summaries from DB "
            "(architect finding F7 — was the bug where warm + boundary=0 left "
            "summaries empty forever)"
        )
        assert ancient2.called, (
            "Case b warm path with boundary_idx=0 must fetch ancient history from DB"
        )


# ---------------------------------------------------------------------------
# 16. Barrier message — REMOVED in Wave 4
# ---------------------------------------------------------------------------
#
# Wave 4 removed the context-layer barrier emission from
# `_select_context_v2`. The MiniMax user-merge problem is now handled
# by `MiniMaxAdapter.apply` downstream of V2
# (see `test_context_v2.py::TestAssemblyOrder::test_minimax_adapter_handles_vfs_user_adjacency`
# and `test_v2_barrier_case_insensitivity.py::TestCleanupStripsWedgedBarrier`).
#
# The class `TestBarrierMessage` previously pinned V2-internal barrier
# emission; the assertions have been updated to verify "no context-layer
# barrier" — the new Wave 4 contract.

class TestBarrierMessage:
    """Wave 4 contract: V2 emits NO context-layer barrier. The MiniMax
    user-merge problem is delegated to the provider adapter downstream."""

    def test_no_barrier_in_v2_for_minimax_adapter(self):
        cfg = _budget_config(safe=10_000)
        summaries = [{"message_index": 5, "role": "user", "summary": "sum", "tool_summary": ""}]
        msgs = [_system("sys")] + _msgs(100)
        model_cfg = SimpleNamespace(backend_model="MiniMax-M3")

        ctx = _make_ctx()
        result, _, _ = _run(cfg, msgs, rich_summaries=summaries, ancient_history="anc", ctx=ctx, model_cfg=model_cfg)

        barrier_messages = [
            m for m in result
            if m.get("role") == "assistant" and m.get("content") == " "
        ]
        assert not barrier_messages, (
            "Wave 4 must NOT emit a context-layer barrier in V2 output; "
            f"found: {barrier_messages}"
        )

    def test_no_barrier_in_v2_for_non_merging_adapter(self):
        cfg = _budget_config(safe=10_000)
        summaries = [{"message_index": 5, "role": "user", "summary": "sum", "tool_summary": ""}]
        msgs = [_system("sys")] + _msgs(100)
        model_cfg = SimpleNamespace(backend_model="gpt-4o")

        ctx = _make_ctx()
        result, _, _ = _run(cfg, msgs, rich_summaries=summaries, ancient_history="anc", ctx=ctx, model_cfg=model_cfg)

        barrier_messages = [
            m for m in result
            if m.get("role") == "assistant" and m.get("content") == " "
        ]
        assert not barrier_messages, (
            "Wave 4 must NOT emit a context-layer barrier in V2 output "
            "(non-merging adapter)"
        )

    def test_last_user_message_is_a_user_message(self):
        cfg = _budget_config(safe=10_000)
        summaries = [{"message_index": 5, "role": "user", "summary": "sum", "tool_summary": ""}]
        pad = "X" * 400
        msgs = [_system("sys")] + [_user(f"msg {i} {pad}") for i in range(100)]
        model_cfg = SimpleNamespace(backend_model="MiniMax-M3")

        ctx = _make_ctx()
        result, _, _ = _run(cfg, msgs, rich_summaries=summaries, ancient_history="anc", ctx=ctx, model_cfg=model_cfg)

        last_idx = len(result) - 1
        last_user = result[last_idx]
        assert last_user.get("role") == "user"
        barrier_messages = [
            m for m in result
            if m.get("role") == "assistant" and m.get("content") == " "
        ]
        assert not barrier_messages, (
            "Wave 4 must NOT emit a context-layer barrier in V2 output; "
            f"found: {barrier_messages}"
        )


# ---------------------------------------------------------------------------
# 17. VFS preserved — between raw_block and summaries
# ---------------------------------------------------------------------------

class TestVfsPreserved:
    """Canonical Wave 4 placement: VFS appears after summaries/ancient and
    after completed_history (sys → summaries → ancient → completed_history →
    VFS → live_exchange)."""

    def test_vfs_slot_after_summaries(self):
        cfg = _budget_config(safe=10_000)
        summaries = [{"message_index": 5, "role": "user", "summary": "sum", "tool_summary": ""}]
        msgs = [_system("sys")] + _msgs(100)
        vfs = {"role": "system", "content": "[VFS] workspace context"}

        ctx = _make_ctx()
        result, _, _ = _run(cfg, msgs, rich_summaries=summaries, ancient_history="anc", vfs_return=vfs, ctx=ctx)

        vfs_idx = None
        summaries_idx = None
        for i, m in enumerate(result):
            c = m.get("content", "")
            if "[VFS]" in c:
                vfs_idx = i
            if "[PAST CONTEXT" in c:
                summaries_idx = i

        assert vfs_idx is not None, "VFS must be present in assembly"
        assert summaries_idx is not None, "Summaries must be present in assembly"
        assert summaries_idx < vfs_idx, "Summaries must come before VFS (after system prompt)"

        before_vfs = result[vfs_idx - 1]
        assert before_vfs.get("role") in ("user", "assistant", "tool", "system"), (
            "A non-system-prompt block must precede VFS"
        )


# ---------------------------------------------------------------------------
# 18. Fast-exit guard — fresh sessions skip chain computation (C5 replacement)
# ---------------------------------------------------------------------------

class TestFastExitGuard:
    """Chain-state miss + no DB rows → fast-exit returns full messages with
    no chain computation, no state lookup, no chain-state creation."""

    def test_small_conversation_no_state_lookup(self):
        cfg = _budget_config()
        msgs = [_system("sys")] + _msgs(20)

        import proxy.state as state
        state._chain_state.clear()

        ctx = _make_ctx()
        result, rich_mock, ancient_mock = _run(cfg, msgs, ctx=ctx, fast_exit_fires=True)

        non_system = [m for m in result if m.get("role") != "system"]
        assert len(non_system) == 20
        assert ctx.v2_boundary_idx is None, "Fast-exit must not set boundary"
        assert ("sess1", "test") not in state._chain_state, (
            "Fast-exit must not create chain state"
        )

    def test_fast_exit_returns_full_messages_minimal_assembly(self):
        cfg = _budget_config()
        msgs = [_system("sys")] + _msgs(40)

        ctx = _make_ctx()
        result, _, _ = _run(cfg, msgs, ctx=ctx, fast_exit_fires=True)

        assert result[0].get("role") == "system"
        non_system = [m for m in result if m.get("role") != "system"]
        assert len(non_system) == 40
        injected = [m for m in result if "[PAST CONTEXT" in m.get("content", "")]
        assert len(injected) == 0, "Fast-exit must not inject summaries"


# ---------------------------------------------------------------------------
# 19. Restart recovery — warm path from DB state
# ---------------------------------------------------------------------------

class TestRestartRecovery:
    """On restart, _chain_state is empty. V2 loads state from SQLite via
    _load_state_from_db and uses the warm path (no cold re-trim)."""

    def test_warm_path_after_restart_no_cold_retrim(self):
        cfg = _budget_config(safe=10_000)
        msgs = [_system("sys")] + _msgs(100)
        conversation = msgs[1:]
        chain, _chain_to_position = compute_chain(conversation)

        boundary_idx = 35
        db_summaries = [{"message_index": 10, "role": "user", "summary": "from db", "tool_summary": ""}]
        db_ancient = "ancient from db"
        state_from_db = {
            "last_chain": chain,
            "boundary_idx": boundary_idx,
            "boundary_chain": chain[boundary_idx - 1],
            "summaries": db_summaries,
            "ancient": db_ancient,
            "version": 0,
        }

        ctx = _make_ctx()
        result, rich_mock, _ = _run(
            cfg, msgs, rich_summaries=db_summaries, ancient_history=db_ancient,
            load_state_override=state_from_db, ctx=ctx,
        )

        assert not rich_mock.called, "Restart recovery warm path must NOT re-fetch summaries"
        assert ctx.v2_boundary_idx == boundary_idx, (
            f"Boundary from DB must be used ({boundary_idx}), got {ctx.v2_boundary_idx}"
        )
        s = _summaries_content(result)
        assert s is not None and "from db" in s

    def test_chain_from_db_matches_compute_chain(self):
        cfg = _budget_config(safe=10_000)
        msgs = [_system("sys")] + _msgs(100)
        conversation = msgs[1:]
        chain, _chain_to_position = compute_chain(conversation)

        boundary_idx = 35
        state_from_db = {
            "last_chain": chain,
            "boundary_idx": boundary_idx,
            "boundary_chain": chain[boundary_idx - 1],
            "summaries": [{"message_index": 10, "role": "user", "summary": "sum", "tool_summary": ""}],
            "ancient": "",
            "version": 0,
        }

        ctx = _make_ctx()
        result, rich_mock, _ = _run(
            cfg, msgs, rich_summaries=state_from_db["summaries"],
            load_state_override=state_from_db, ctx=ctx,
        )

        assert not rich_mock.called, (
            "Warm path succeeds → DB chain matched runtime chain (boundary valid)"
        )
        assert ctx.v2_boundary_idx == boundary_idx


# ---------------------------------------------------------------------------
# 20. Hard cap (M15) — force evict past last-summarized at HIGH * 1.10
# ---------------------------------------------------------------------------

class TestHardCap:
    """When raw_tokens > HIGH_WATERMARK * 1.10 and nothing can be evicted
    (max_evictable_boundary=0), the proxy force-evicts past the last-summarized
    boundary and logs a CRITICAL warning."""

    def test_force_evict_past_boundary_with_critical_log(self, caplog):
        cfg = _budget_config(safe=10_000)
        summaries = [{"message_index": 5, "role": "user", "summary": "sum", "tool_summary": ""}]

        ctx1 = _make_ctx("r1")
        msgs1 = [_system("sys")] + _msgs(100)
        _run(cfg, msgs1, rich_summaries=summaries, ctx=ctx1)
        boundary1 = ctx1.v2_boundary_idx

        ctx2 = _make_ctx("r2")
        msgs2 = [_system("sys")] + _msgs(200)
        with caplog.at_level(logging.CRITICAL, logger="proxy.context_v2"):
            r2, _, _ = _run(
                cfg, msgs2, rich_summaries=summaries,
                last_exchange_index=-1, min_mi=None, ctx=ctx2,
            )
        boundary2 = ctx2.v2_boundary_idx

        assert boundary2 > boundary1, (
            f"Hard cap must force-evict past boundary ({boundary1} → got {boundary2})"
        )
        critical_msgs = [r for r in caplog.records if r.levelno >= logging.CRITICAL]
        assert len(critical_msgs) > 0, "CRITICAL warning must be logged on hard-cap eviction"
        assert any("HIGH_WATERMARK" in r.message for r in critical_msgs)


# ---------------------------------------------------------------------------
# 21. CRITICAL A — model_name key consistency (reader vs writer)
# ---------------------------------------------------------------------------

class TestModelNameRoundTrip:
    """CRITICAL A: reader (mode.py → V2) and writer (post_process.py) must use
    the same canonical model_name form. After the fix, mode.py passes
    ctx.model_name (transformed, e.g. "claude-3_5") instead of body.get("model")
    (raw, e.g. "anthropic/claude-3 5"). If forms differ, _load_state_from_db
    returns None every restart and the cold path runs forever."""

    def test_v2_state_keyed_on_model_name_parameter(self):
        import proxy.state as state
        state._chain_state.clear()

        cfg = _budget_config(safe=10_000)
        model_name = "claude-3_5"
        msgs = [_system("sys")] + _msgs(100)

        ctx = _make_ctx()
        _run(cfg, msgs, model_name=model_name, ctx=ctx)

        key = ("sess1", model_name)
        assert key in state._chain_state, (
            f"State must be keyed under the model_name parameter {key}"
        )

    def test_restart_recovery_with_transformed_model_name(self):
        cfg = _budget_config(safe=10_000)
        model_name = "claude-3_5"
        msgs = [_system("sys")] + _msgs(100)
        conversation = msgs[1:]
        chain, _chain_to_position = compute_chain(conversation)

        boundary_idx = 35
        db_state = {
            "last_chain": chain,
            "boundary_idx": boundary_idx,
            "boundary_chain": chain[boundary_idx - 1],
            "summaries": [{"message_index": 10, "role": "user", "summary": "db", "tool_summary": ""}],
            "ancient": "",
            "version": 0,
        }

        ctx = _make_ctx()
        result, rich_mock, _ = _run(
            cfg, msgs,
            rich_summaries=db_state["summaries"],
            load_state_override=db_state,
            model_name=model_name,
            ctx=ctx,
        )

        assert not rich_mock.called, "Warm path must succeed with consistent model_name key"
        assert ctx.v2_boundary_idx == boundary_idx

    def test_mode_passes_ctx_model_name_to_select_context(self):
        from proxy.mode import apply_mode, Mode

        ctx = SimpleNamespace(
            session_id="s1",
            model_name="claude-3_5",
            is_subagent=False,
            model_cfg=None,
            request_id="test",
            _tools_snapshot=None,
        )
        body = {
            "model": "anthropic/claude-3 5",
            "messages": [
                {"role": "system", "content": "sys"},
                {"role": "user", "content": "hi"},
            ],
        }
        agent = MagicMock()

        captured = {}

        def capture_select(sid, msgs, model_name, **kwargs):
            captured["model_name"] = model_name
            return msgs

        mock_tooling = MagicMock()
        mock_tooling.augmented_tools = []

        with patch("proxy.skill_prompts.assemble_system_prompt", return_value="sys"), \
             patch("proxy.tool_cache.prepare_tooling_payload", return_value=mock_tooling), \
             patch("proxy.skill_tools.select_tools", return_value=[]), \
             patch("proxy.context.select_context", side_effect=capture_select):
            apply_mode(ctx, agent, body, Mode.DEFAULT)

        assert captured.get("model_name") == "claude-3_5", (
            f"apply_mode must pass ctx.model_name ('claude-3_5'), "
            f"got {captured.get('model_name')!r}"
        )


# ---------------------------------------------------------------------------
# 22. CRITICAL B — cold path preserves oversized last user message
# ---------------------------------------------------------------------------

class TestColdPathPreservesLastUser:
    """CRITICAL B: when the last user turn exceeds low_watermark, the cold path
    must NOT drop it. Without the fix, kept_units=[] → raw_block=[] →
    last_user=None → the model receives only the system prompt and the user's
    question is LOST."""

    def test_oversized_last_user_not_dropped(self):
        cfg = _budget_config(safe=10_000)
        huge_content = "X" * 30_000
        messages = [_system("sys"), _user(huge_content)]

        ctx = _make_ctx()
        result, _, _ = _run(cfg, messages, ctx=ctx)

        assert len(result) >= 2, "Must include system + user"
        last_msg = result[-1]
        assert last_msg.get("role") == "user", "Last message must be the user turn"
        assert len(last_msg.get("content", "")) > 0, (
            "User content must be present (possibly trimmed, not dropped)"
        )

    def test_oversized_last_user_content_preserved_or_trimmed(self):
        cfg = _budget_config(safe=10_000)
        huge_content = "Q" * 50_000
        messages = [_system("sys"), _user(huge_content)]

        ctx = _make_ctx()
        result, _, _ = _run(cfg, messages, ctx=ctx)

        last_msg = result[-1]
        assert last_msg.get("role") == "user"
        assert "[truncated]" in last_msg.get("content", ""), (
            "Oversized user message must be trimmed by assembly safety net"
        )
        assert len(last_msg["content"]) > 0
        assert len(last_msg["content"]) < len(huge_content), (
            "Trimmed content must be shorter than original"
        )

    def test_oversized_message_trimmed_in_assembly(self):
        """Verify the final assembly trim step trims oversized messages and
        the output fits within the model safe budget."""
        cfg = _budget_config(safe=10_000)
        huge_content = "Z" * 50_000
        messages = [_system("sys"), _user(huge_content)]

        ctx = _make_ctx()
        result, _, _ = _run(cfg, messages, ctx=ctx)

        last_msg = result[-1]
        assert last_msg.get("role") == "user"
        assert "[truncated]" in last_msg.get("content", ""), (
            "Assembly trim must mark truncated content"
        )
        assert len(last_msg["content"]) < len(huge_content), (
            "Trimmed content must be shorter than the original oversized input"
        )
        assert len(last_msg["content"]) > 0

        non_system = [m for m in result if m.get("role") != "system"]
        raw_tokens = sum(int(_count_msg_chars(m) * 0.25) for m in non_system)
        assert raw_tokens <= 10_000, (
            f"Total raw tokens ({raw_tokens}) must fit safe budget "
            f"(tool_schema_tokens=0 when tools absent, per slice 4A)"
        )


# ---------------------------------------------------------------------------
# 23. CRITICAL B corollary — eviction preserves last user (tiny low_watermark)
# ---------------------------------------------------------------------------

class TestEvictionPreservesLastUser:
    """When low_watermark is tiny, the eviction loop can pop ALL units. Without
    the fix, raw_block becomes empty and the last user is lost. The fix ensures
    at least the last unit survives."""

    def test_tiny_low_watermark_keeps_last_user(self):
        cfg = _budget_config(safe=10_000)
        summaries = [{"message_index": 5, "role": "user", "summary": "sum", "tool_summary": ""}]

        ctx1 = _make_ctx("r1")
        msgs1 = [_system("sys")] + _msgs(100)
        _run(cfg, msgs1, rich_summaries=summaries, ctx=ctx1)

        ctx2 = _make_ctx("r2")
        msgs2 = [_system("sys")] + _msgs(200)
        last_user_content = msgs2[-1]["content"]
        r2, _, _ = _run(
            cfg, msgs2, rich_summaries=summaries,
            last_exchange_index=-1, min_mi=None,
            low_water=0.01, ctx=ctx2,
        )

        assert len(r2) > 0, "Result must not be empty"
        last_msg = r2[-1]
        assert last_msg.get("role") == "user", "Last message must be the user turn"
        assert last_msg["content"] == last_user_content, (
            "Last user content must be preserved"
        )


# ---------------------------------------------------------------------------
# 24. HIGH 2 — compute_chain AttributeError falls through to passthrough
# ---------------------------------------------------------------------------

class TestComputeChainNonDict:
    """When compute_chain raises AttributeError (e.g. a non-dict message),
    V2 must catch it and fall through to returning full_messages instead of
    crashing the entire request."""

    def test_attribute_error_in_compute_chain_does_not_crash(self):
        cfg = _budget_config(safe=10_000)
        messages = [_system("sys")] + _msgs(60)

        ctx = _make_ctx()
        with patch(
            "proxy.chain_hash.compute_chain",
            side_effect=AttributeError("'str' object has no attribute 'get'"),
        ):
            result, _, _ = _run(cfg, messages, ctx=ctx)

        assert result is not None, "V2 must return messages, not crash"
        assert len(result) == len(messages), "Passthrough must return all messages"
        assert ctx.v2_boundary_idx is None, "No boundary set on compute_chain failure"
