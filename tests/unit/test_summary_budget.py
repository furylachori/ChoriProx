"""Unit tests for summary token budget.

Covers the tiered-pipeline budget enforcement (C4/C5 post-passthrough-removal):
  - Small messages → no eviction → raw returned intact
  - Large messages → budget pipeline runs → trimming

Budget math (tokens_per_char=0.25, safe_context_tokens=10000, system=0, tools=2000):
  total_available = 8000
  ancient_budget = int(8000 * 0.05) = 400
  recent_raw_budget = int(10000 * 0.65) = 6500
  rich_summary_budget = 8000 - 400 - 6500 = 1100
"""
from contextlib import ExitStack
from unittest.mock import MagicMock, patch

import pytest

from proxy.context_v2 import (
    _apply_summary_budget,
    _format_summaries,
    _select_context_v2,
)

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _system(content: str) -> dict:
    return {"role": "system", "content": content}


def _user(content: str) -> dict:
    return {"role": "user", "content": content}


def _assistant(content: str) -> dict:
    return {"role": "assistant", "content": content}


def _conn_mock_with_rows():
    """Return a mock memory.db.connection context manager that returns
    rows for the C5 fast-exit SELECT 1 queries (so the fast-exit does NOT
    fire and the normal path runs). Other queries return shape-correct
    empty results so _load_state_from_db and the cold path don't crash
    on the mock.
    """
    def execute_side_effect(query, *args, **kwargs):
        q = str(query).upper()
        result = MagicMock()
        if "SELECT 1 FROM CONVERSATION_SUMMARIES" in q or "SELECT 1 FROM MESSAGE_SUMMARIES" in q:
            result.fetchone.return_value = (1,)
        elif "RAW_START_MESSAGE_INDEX" in q:
            result.fetchone.return_value = None
        elif "MIN(MESSAGE_INDEX)" in q:
            result.fetchone.return_value = None
        elif "CHAIN_HASH" in q:
            result.fetchall.return_value = []
        else:
            result.fetchone.return_value = None
            result.fetchall.return_value = []
        return result

    mock_conn = MagicMock()
    mock_conn.execute.side_effect = execute_side_effect
    mock_conn_cm = MagicMock()
    mock_conn_cm.__enter__ = MagicMock(return_value=mock_conn)
    mock_conn_cm.__exit__ = MagicMock(return_value=False)
    return mock_conn_cm


def _no_db_mocks(cfg, rich_summaries=None, ancient_history=""):
    """Return a list of patches that suppress DB and VFS side-effects.

    Returned as a list (not a tuple) so callers can unwind via ExitStack
    without index fragility when patches are added/removed.
    """
    return [
        patch("proxy.context.get_context_config", return_value=cfg),
        patch("memory.db.get_ancient_history", return_value=ancient_history),
        patch("memory.db.get_rich_summaries", return_value=rich_summaries or []),
        patch("proxy.vfs._build_vfs_inherited_context", return_value=None),
        patch("proxy.vfs._build_vfs_context", return_value=None),
        patch("memory.db.connection", return_value=_conn_mock_with_rows()),
    ]


def _budget_config():
    """Standard config for budget tests.

    tokens_per_char=0.25 → 400 chars = 100 tokens.
    safe_context_tokens=10000 → total_available=8000 (after tools=2000, system=0).
    ancient=400, raw=6500, rich_summary_budget = 8000-400-6500 = 1100.
    """
    return {
        "max_context_tokens": 40_000,
        "tokens_per_char": 0.25,
        "safe_context_tokens": 10_000,
        "raw_exchange_count": 100,
    }


def _large_raw(count=150, chars=400):
    """Build `count` user messages of `chars` each (100 tokens at chars=400)."""
    pad = "X" * chars
    return [_user(f"msg {i} {pad}") for i in range(count)]


# ---------------------------------------------------------------------------
# Test 1: Tiered pipeline behavior (post passthrough removal)
# ---------------------------------------------------------------------------

class TestTieredPipelineBehavior:
    """The tiered pipeline runs for every conversation; small inputs return
    intact, large inputs engage eviction/budget.

    The removed passthrough (F5) used to short-circuit small conversations.
    After C5, all conversations flow through the normal path; the fast-exit
    only fires when the session has no DB rows at all.
    """

    def test_small_messages_returned_intact(self):
        """Small conversation fits under low_watermark → returned intact, no trim.

        safe_context_tokens=10000, tokens_per_char=0.25
        10 messages × 400 chars × 0.25 = 1000 tokens < 6500 low_watermark.
        """
        cfg = _budget_config()
        pad = "X" * 400
        messages = [_system("sys")] + [_user(f"msg {i} {pad}") for i in range(10)]

        with ExitStack() as stack:
            for p in _no_db_mocks(cfg):
                stack.enter_context(p)
            result = _select_context_v2(
                "sess1", messages, model_name="test",
            )

        assert isinstance(result, list)
        non_system = [m for m in result if m.get("role") != "system"]
        assert len(non_system) == 10, (
            f"Small conversation should pass through the tiered pipeline intact, "
            f"got {len(non_system)} of 10 non-system messages."
        )

    def test_large_messages_trimmed(self):
        """Large messages well above low_watermark → eviction trims.

        safe_context_tokens=10000, tokens_per_char=0.25
        low_watermark = 6500 tokens.
        110 messages × 400 chars × 0.25 = 11000 tokens > 6500 → eviction runs.
        """
        cfg = _budget_config()
        pad = "X" * 400
        messages = [_system("sys")] + [_user(f"msg {i} {pad}") for i in range(110)]

        with ExitStack() as stack:
            for p in _no_db_mocks(cfg):
                stack.enter_context(p)
            result = _select_context_v2(
                "sess1", messages, model_name="test",
            )

        assert isinstance(result, list)
        non_system = [m for m in result if m.get("role") != "system"]
        assert len(non_system) < 110, (
            f"Budget pipeline should trim 110 messages, got {len(non_system)} of 110."
        )

    def test_below_low_watermark(self):
        """Below the low_watermark the conversation is returned intact, no
        eviction. The strict-inverse of passthrough: the removed guard
        used to short-circuit at 30% of safe. Now the tiered pipeline runs
        but the budget gate (low_watermark=0.65) decides whether to trim.
        """
        cfg = _budget_config()
        pad = "X" * 400
        messages = [_system("sys")] + [_user(f"msg {i} {pad}") for i in range(60)]

        with ExitStack() as stack:
            for p in _no_db_mocks(cfg):
                stack.enter_context(p)
            result = _select_context_v2(
                "sess1", messages, model_name="test",
            )

        assert isinstance(result, list)
        non_system = [m for m in result if m.get("role") != "system"]
        assert len(non_system) == 60, (
            "Sub-low-watermark conversation should be returned intact; "
            f"got {len(non_system)} of 60 non-system messages."
        )
        assert result[0]["role"] == "system"


# ---------------------------------------------------------------------------
# Test 2: Summary token budget
# ---------------------------------------------------------------------------

class TestSummaryTokenBudget:
    """Verify summary content is counted against the token budget.

    Inputs are sized (150 messages × 400 chars × 0.25 = 15000 tokens) to force
    eviction past the 6500 low-watermark so summaries are actually fetched.
    rich_summary_budget = 1100.
    """

    def _make_summaries(self, count, token_cost=200, summary_text=None):
        """Create summary dicts for get_rich_summaries mock.

        Each summary claims `token_cost` rendered tokens AND its actual text
        is sized to match (tokens_per_char=0.25 → 4 chars/token), so the
        post-assembly overflow check (which uses actual content length)
        agrees with the budget loop (which uses rendered_*_tokens).
        """
        if summary_text is None:
            summary_text = "x" * int(token_cost / 0.25)

        summaries = []
        for i in range(count):
            summaries.append({
                "message_index": i + 50,  # above ZONE1_FULL_MIN to hit narrative zone
                "role": "user" if i % 2 == 0 else "assistant",
                "summary": summary_text,
                "tool_summary": "",
                "rendered_full_tokens": token_cost,
                "rendered_action_tokens": token_cost,
                "rendered_narrative_tokens": token_cost,
            })
        return summaries

    def test_small_summaries_not_truncated(self):
        """Summaries that fit within the budget should all be included.

        5 summaries × 100 tokens = 500 tokens < 1100 budget → all included.
        """
        cfg = _budget_config()
        summaries = self._make_summaries(5, token_cost=100)
        messages = [_system("sys")] + _large_raw(150)

        with ExitStack() as stack:
            for p in _no_db_mocks(cfg, rich_summaries=summaries):
                stack.enter_context(p)
            result = _select_context_v2(
                "sess1", messages, model_name="test",
            )

        assert isinstance(result, list)

        summary_messages = [
            m for m in result
            if m.get("role") == "system" and "PAST CONTEXT" in m.get("content", "")
        ]

        assert summary_messages, "Expected a PAST CONTEXT summary block under eviction"
        content = summary_messages[0]["content"]
        line_count = content.count("[msg#")
        assert line_count == 5, (
            f"All 5 summaries should fit in budget (500 < 1100), got {line_count}"
        )

    def test_slice_4e_full_rendered_summary_block_respects_cap(self):
        """Slice 4E counts the main wrapper and row separators."""
        summaries = [
            {
                "message_index": index,
                "role": "user",
                "summary": "S" * 400,
                "tool_summary": "",
            }
            for index in range(200, 220)
        ]

        with patch(
            "memory.sync.submit_ancient_compress",
            return_value=None,
        ) as dispatch:
            kept = _apply_summary_budget(
                summaries,
                model_safe_tokens=10_000,
                tokens_per_char=0.25,
                session_id="slice-4e-main-summary",
                ctx=None,
            )

        assert [
            row["message_index"]
            for row in kept
        ] == list(range(207, 220))
        assert [
            row["message_index"]
            for row in dispatch.call_args.args[1]
        ] == list(range(200, 207))

        rendered = _format_summaries(kept)
        assert len(rendered) == 5_643
        assert int(len(rendered) * 0.25) == 1_410
        assert int(
            len(_format_summaries(summaries[6:])) * 0.25
        ) == 1_515
        assert int(len(rendered) * 0.25) <= 1_500
