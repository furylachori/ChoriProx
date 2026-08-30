"""
Unit tests for pair-aware context trimming with synthetic tool result injection.

The context selection pipeline (v1 and v2) must guarantee:
  - Every `role: tool` in the output has its parent `assistant tool_calls` in the output.
  - Every `assistant tool_calls` in the output has ALL its `tool` results in the output.

A "unit" is a logical group of messages (assistant + its tool results) that
must never be split by trimming. These tests cover the unit grouping, the
pair-aware trim, the orphan-dropping cleanup, and the synthetic tool result
injection.
"""

import copy
import os
from unittest.mock import patch

import pytest

from proxy.context import _group_message_units, select_context
from proxy.context_v2 import _split_completed_and_live_exchange

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _user(content: str) -> dict:
    return {"role": "user", "content": content}


def _assistant(content: str = "", tool_calls: list = None) -> dict:
    msg = {"role": "assistant", "content": content}
    if tool_calls is not None:
        msg["tool_calls"] = tool_calls
    return msg


def _tool_call(tc_id: str, name: str = "do_thing", args: str = "{}") -> dict:
    return {
        "id": tc_id,
        "type": "function",
        "function": {"name": name, "arguments": args},
    }


def _tool(tc_id: str, content: str = "result") -> dict:
    return {"role": "tool", "tool_call_id": tc_id, "content": content}


def _pair(prefix: str) -> tuple:
    """Build a (assistant, tool) pair with matching tc_id."""
    tc_id = f"{prefix}_tc"
    assistant = _assistant(content="calling", tool_calls=[_tool_call(tc_id)])
    tool = _tool(tc_id, content=f"result of {prefix}")
    return assistant, tool


def _assert_pair_integrity(messages: list) -> None:
    """Verify the tool-call chain invariant: every tool has its parent
    assistant, and every assistant tool_call has its result tool."""
    # 1. Every tool must have a parent assistant with matching tc_id
    assistant_tc_ids = set()
    for m in messages:
        if m.get("role") == "assistant" and isinstance(m.get("tool_calls"), list):
            for tc in m["tool_calls"]:
                if tc.get("id"):
                    assistant_tc_ids.add(tc["id"])
    for m in messages:
        if m.get("role") == "tool":
            tc_id = m.get("tool_call_id")
            assert tc_id in assistant_tc_ids, (
                f"Orphan tool message (tool_call_id={tc_id}) — "
                f"parent assistant is missing from output"
            )

    # 2. Every assistant tool_call must have a matching tool result
    present_tool_ids = {
        m.get("tool_call_id") for m in messages if m.get("role") == "tool"
    }
    for m in messages:
        if m.get("role") == "assistant" and isinstance(m.get("tool_calls"), list):
            for tc in m["tool_calls"]:
                tc_id = tc.get("id")
                assert tc_id in present_tool_ids, (
                    f"Assistant tool_call (id={tc_id}) has no matching tool result"
                )


# ---------------------------------------------------------------------------
# v2 base config — small safe_context_tokens so the trim path runs
# ---------------------------------------------------------------------------

def _v2_config():
    """A config that forces the v2 trim path. safe_context_tokens must be
    large enough that total_available = safe - 0 - 2000 (tool_schema_tokens)
    stays positive."""
    return {
        "safe_context_tokens": 10000,
        "max_context_tokens": 40000,
        "tokens_per_char": 0.25,
        "raw_exchange_count": 100,  # large enough that recent_raw captures everything
    }


def _safe_limit():
    return {"max_tokens": 4000}


# ---------------------------------------------------------------------------
# TestGroupMessageUnits
# ---------------------------------------------------------------------------

class TestGroupMessageUnits:
    """Direct unit tests for _group_message_units."""

    def test_empty_list_returns_empty(self):
        assert _group_message_units([]) == []

    def test_single_user_message(self):
        m = _user("hi")
        assert _group_message_units([m]) == [[m]]

    def test_single_assistant_no_tools(self):
        m = _assistant("hello")
        assert _group_message_units([m]) == [[m]]

    def test_assistant_with_tools_groups_with_results(self):
        a, t = _pair("p1")
        units = _group_message_units([a, t])
        assert units == [[a, t]]

    def test_assistant_with_multiple_tool_results(self):
        a = _assistant(
            content="calling",
            tool_calls=[_tool_call("tc1"), _tool_call("tc2")],
        )
        t1 = _tool("tc1")
        t2 = _tool("tc2")
        units = _group_message_units([a, t1, t2])
        assert units == [[a, t1, t2]]

    def test_user_assistant_pair_user(self):
        """Three separate units: user, pair, user."""
        u1 = _user("a")
        a, t = _pair("p1")
        u2 = _user("b")
        units = _group_message_units([u1, a, t, u2])
        assert units == [[u1], [a, t], [u2]]

    def test_pair_then_user_then_pair(self):
        """Two pairs separated by a user message."""
        a1, t1 = _pair("p1")
        u = _user("between")
        a2, t2 = _pair("p2")
        units = _group_message_units([a1, t1, u, a2, t2])
        assert units == [[a1, t1], [u], [a2, t2]]

    def test_standalone_tool_with_no_parent_is_own_unit(self):
        """Malformed input: a tool with no preceding assistant-with-tool-calls
        is a unit of length 1 (orphan, to be dropped by cleanup)."""
        t = _tool("orphan_tc")
        assert _group_message_units([t]) == [[t]]

    def test_standalone_tool_followed_by_user(self):
        t = _tool("orphan_tc")
        u = _user("hi")
        units = _group_message_units([t, u])
        assert units == [[t], [u]]

    def test_assistant_with_empty_tool_calls_list_is_own_unit(self):
        """tool_calls=[] is NOT a pair head; the assistant is its own unit."""
        a = _assistant("hello", tool_calls=[])
        t = _tool("tc1")
        units = _group_message_units([a, t])
        assert units == [[a], [t]]

    def test_preserves_original_order(self):
        u1 = _user("a")
        u2 = _user("b")
        a, t = _pair("p1")
        units = _group_message_units([u1, u2, a, t])
        # Flatten and compare
        flat = [m for unit in units for m in unit]
        assert flat == [u1, u2, a, t]

    def test_unknown_role_is_own_unit(self):
        m = {"role": "developer", "content": "weird"}
        assert _group_message_units([m]) == [[m]]


# ---------------------------------------------------------------------------
# TestV2PairAwareTrim — pair-aware trimming for v2
# ---------------------------------------------------------------------------

class TestV2PairAwareTrim:
    """V2 trim tests using small budgets to force the cut path."""

    def _run_v2(self, messages):
        """Run select_context with v2 forced, with mocks for memory/db."""
        with patch("proxy.context.get_context_config", return_value=_v2_config()), \
             patch("proxy.context.get_safe_context_limit", return_value=_safe_limit()), \
             patch("memory.db.get_ancient_history", return_value=""), \
             patch("memory.db.get_rich_summaries", return_value=[]):
            return select_context("test_session", messages)

    def test_token_budget_eviction_keeps_pairs_atomic_v2(self):
        """Token-budget eviction must cut at unit boundaries, never mid-pair.

        With safe=10000 (total_available=8000, recent_raw_budget=6500) and
        enough raw tokens to exceed the 6500 low-watermark, the trim keeps
        complete assistant+tool pairs and drops the oldest. No pair is split.
        """
        # 35 user messages + 35 (assistant + tool) pairs = 105 messages.
        # Each message ~400 chars (~100 tok) → ~10500 raw tokens > 6500 budget.
        # Trim keeps the newest ~65 messages worth of complete units.
        pad = "x" * 400
        messages = []
        for i in range(35):
            messages.append(_user(f"u{i} {pad}"))
        for i in range(35):
            tc_id = f"tc_{i}"
            messages.append(_assistant(
                content=f"calling {i} {pad}",
                tool_calls=[_tool_call(tc_id)],
            ))
            messages.append(_tool(tc_id, content=f"result {i} {pad}"))

        assert len(messages) == 105

        result = self._run_v2(messages)

        # The tool-call chain invariant must hold (no split pairs).
        _assert_pair_integrity(result)

        # Eviction must have actually occurred (token budget < input).
        non_system = [m for m in result if m.get("role") != "system"]
        assert len(non_system) < 105, (
            f"Token-budget eviction should trim below 105 messages, got {len(non_system)}"
        )

        # Sanity: result should contain tool messages (at least one full pair kept).
        tool_count = sum(1 for m in result if m.get("role") == "tool")
        assert tool_count > 0, "Expected at least one tool message in output"

    def test_pair_straddling_token_cap_kept_together_v2(self):
        """Token-budget cap should also cut at unit boundaries, never mid-pair."""
        # Build 20 user/assistant/tool triples. Force a budget that only fits
        # a portion so the token cap cuts aggressively at unit boundaries.
        messages = []
        for i in range(20):
            tc_id = f"tc_{i}"
            # Use a long content to push past pass-through threshold
            pad = "x" * 400
            messages.append(_user(f"u{i} {pad}"))
            messages.append(_assistant(
                content=f"calling {i} {pad}",
                tool_calls=[_tool_call(tc_id)],
            ))
            messages.append(_tool(tc_id, content=f"result {i} {pad}"))

        # safe must be large enough that recent_raw_budget + tool_schema <= safe.
        # recent_raw_budget = int(safe*0.65); need 0.65*safe + 2000 <= safe → safe >= 5715.
        # With safe=8000: total_available = 6000; recent_raw_budget = 5200 tokens.
        # 20 triples × ~307 tok = ~6140 tok > 5200 → eviction cuts at pair boundaries.
        tiny_config = {
            "safe_context_tokens": 8000,
            "max_context_tokens": 40000,
            "tokens_per_char": 0.25,
            "raw_exchange_count": 100,
        }
        with patch("proxy.context.get_context_config", return_value=tiny_config), \
             patch("proxy.context.get_safe_context_limit", return_value=_safe_limit()), \
             patch("memory.db.get_ancient_history", return_value=""), \
             patch("memory.db.get_rich_summaries", return_value=[]):
            result = select_context("test_session", messages)

        # Invariant: no split pairs in the output
        _assert_pair_integrity(result)
        # Eviction must have trimmed something
        non_system = [m for m in result if m.get("role") != "system"]
        assert len(non_system) < len(messages), "Expected token-budget eviction to trim"

    def test_orphan_tool_dropped_v2(self):
        """A tool message whose parent assistant was pruned should be
        dropped (not left as an orphan in the output).
        """
        # Build messages such that after trim, the only assistant-with-
        # tool_calls is dropped, leaving an orphan tool.
        # Construct full_messages with only an orphan tool at the end
        # (no parent). The grouping code treats it as a unit of length 1;
        # cleanup then drops it.
        messages = []
        for i in range(3):
            messages.append(_user(f"u{i}"))
            messages.append(_assistant(content=f"hi {i}"))
        # Add a malformed orphan tool at the end
        messages.append(_tool("orphan_tc", content="no parent"))

        result = self._run_v2(messages)

        # The orphan tool should not appear in the output
        tool_tc_ids = [m.get("tool_call_id") for m in result if m.get("role") == "tool"]
        assert "orphan_tc" not in tool_tc_ids, (
            "Orphan tool result (no parent assistant) should be dropped"
        )

    def test_assistant_with_missing_results_gets_synthetic_results_v2(self):
        """When an assistant with tool_calls survives the trim but its
        tool result was cut, a synthetic tool result is injected with
        the matching tool_call_id and a clear marker.
        """
        # Build: user, assistant(tc1, tc2), tool(tc1)
        # The assistant has 2 tool_calls but only 1 result. The cleanup
        # must inject a synthetic result for tc2.
        messages = [
            _user("do two things"),
            _assistant(
                content="calling",
                tool_calls=[_tool_call("tc1"), _tool_call("tc2")],
            ),
            _tool("tc1", content="result of tc1"),
            # tc2's result is missing — this is the malformed case
        ]

        result = self._run_v2(messages)

        # The synthetic injection runs only if a kept assistant has a
        # tool_call without a matching tool in the output. The orphan
        # tool (tc1 only) cleanup first drops any orphan tools, then
        # pass 2 injects synthetic for tc2.
        _assert_pair_integrity(result)

        # Find any synthetic tool result
        synthetic = [m for m in result
                     if m.get("role") == "tool"
                     and m.get("tool_call_id") == "tc2"
                     and "pruned" in m.get("content", "").lower()]
        assert len(synthetic) == 1, (
            f"Expected exactly one synthetic tool result for tc2, got {len(synthetic)}. "
            f"Result: {result}"
        )
        assert "Result pruned from earlier context" in synthetic[0]["content"]


# ---------------------------------------------------------------------------
# TestSplitCompletedAndLiveExchange
# ---------------------------------------------------------------------------

class TestSplitCompletedAndLiveExchange:
    """US: Wave 4 splitter — pure, deep-copying, message-unit-aware split."""

    def test_empty_returns_two_empty_lists(self):
        completed, live = _split_completed_and_live_exchange([])
        assert completed == []
        assert live == []

    def test_single_current_user_is_entire_live_exchange(self):
        messages = [{"role": "user", "content": "hi"}]
        completed, live = _split_completed_and_live_exchange(messages)
        assert completed == []
        assert len(live) == 1
        assert live[0] == {"role": "user", "content": "hi"}

    def test_completed_exchange_precedes_current_user(self):
        messages = [
            {"role": "user", "content": "old"},
            {"role": "assistant", "content": "old answer"},
            {"role": "user", "content": "current"},
        ]
        completed, live = _split_completed_and_live_exchange(messages)
        assert len(completed) == 2
        assert completed[0]["role"] == "user"
        assert completed[1]["role"] == "assistant"
        assert len(live) == 1
        assert live[0]["content"] == "current"

    def test_walks_backward_over_consecutive_current_users(self):
        messages = [
            {"role": "assistant", "content": "completed"},
            {"role": "user", "content": "current one"},
            {"role": "user", "content": "current two"},
        ]
        completed, live = _split_completed_and_live_exchange(messages)
        assert len(completed) == 1
        assert completed[0]["content"] == "completed"
        assert len(live) == 2
        assert live[0]["content"] == "current one"
        assert live[1]["content"] == "current two"

    def test_consecutive_users_with_tool_continuation_share_live_suffix(self):
        """Wave 4a: consecutive users preceding a tool continuation all stay
        in the live suffix; VFS cannot be inserted between them."""
        messages = [
            {"role": "user", "content": "fragment one"},
            {"role": "user", "content": "fragment two"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {"id": "A", "type": "function",
                     "function": {"name": "read", "arguments": "{}"}},
                ],
            },
            {"role": "tool", "tool_call_id": "A", "content": "result"},
        ]
        completed, live = _split_completed_and_live_exchange(messages)
        assert completed == [], (
            "No preceding message → completed must be empty; the entire input "
            "is the live user-led exchange."
        )
        assert len(live) == 4
        assert live[0]["content"] == "fragment one"
        assert live[1]["content"] == "fragment two"
        assert live[2]["role"] == "assistant"
        assert live[3]["role"] == "tool"
        assert live[3]["tool_call_id"] == "A"

    def test_consecutive_users_with_multi_tool_continuation_share_live_suffix(self):
        """Wave 4a: consecutive users preceding a multi-tool continuation all
        stay in the live suffix; tool atomicity preserved."""
        messages = [
            {"role": "user", "content": "fragment one"},
            {"role": "user", "content": "fragment two"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {"id": "A", "type": "function",
                     "function": {"name": "read", "arguments": "{}"}},
                    {"id": "B", "type": "function",
                     "function": {"name": "read", "arguments": "{}"}},
                ],
            },
            {"role": "tool", "tool_call_id": "A", "content": "result A"},
            {"role": "tool", "tool_call_id": "B", "content": "result B"},
        ]
        completed, live = _split_completed_and_live_exchange(messages)
        assert completed == []
        assert len(live) == 5
        assert live[0]["content"] == "fragment one"
        assert live[1]["content"] == "fragment two"
        assert live[2]["role"] == "assistant"
        assert live[3]["tool_call_id"] == "A"
        assert live[4]["tool_call_id"] == "B"

    def test_keeps_older_completed_tool_unit_atomic(self):
        messages = [
            {"role": "user", "content": "old one"},
            {"role": "assistant", "tool_calls": [
                {"id": "X", "type": "function",
                 "function": {"name": "r", "arguments": "{}"}},
            ]},
            {"role": "tool", "tool_call_id": "X", "content": "result"},
            {"role": "user", "content": "current"},
        ]
        completed, live = _split_completed_and_live_exchange(messages)
        assert len(completed) == 3
        assert completed[0]["content"] == "old one"
        assert completed[1]["role"] == "assistant"
        assert completed[1].get("tool_calls")
        assert completed[2]["tool_call_id"] == "X"
        assert len(live) == 1
        assert live[0]["content"] == "current"

    def test_missing_tool_result_remains_in_live_suffix_without_synthesis(self):
        messages = [
            {"role": "user", "content": "current"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {"id": "Z", "type": "function",
                     "function": {"name": "r", "arguments": "{}"}},
                ],
            },
        ]
        completed, live = _split_completed_and_live_exchange(messages)
        assert completed == []
        assert len(live) == 2
        assert live[0]["content"] == "current"
        assert live[1]["role"] == "assistant"

    def test_orphan_tool_after_current_user_remains_live_until_cleanup(self):
        messages = [
            {"role": "user", "content": "current"},
            {"role": "tool", "tool_call_id": "orphan", "content": "stray"},
        ]
        completed, live = _split_completed_and_live_exchange(messages)
        assert completed == []
        assert len(live) == 2
        assert live[1]["tool_call_id"] == "orphan"

    def test_orphan_tool_without_any_user_does_not_create_live_suffix(self):
        messages = [
            {"role": "assistant", "content": "no user at all"},
            {"role": "tool", "tool_call_id": "orphan", "content": "stray"},
        ]
        completed, live = _split_completed_and_live_exchange(messages)
        assert len(completed) == 2
        assert completed[0]["role"] == "assistant"
        assert completed[1]["tool_call_id"] == "orphan"
        assert live == []

    def test_assistant_only_returns_no_live_exchange(self):
        messages = [{"role": "assistant", "content": "alone"}]
        completed, live = _split_completed_and_live_exchange(messages)
        assert len(completed) == 1
        assert live == []

    def test_unknown_tail_after_latest_user_remains_live(self):
        messages = [
            {"role": "user", "content": "current"},
            {"role": "unknown", "content": "weird"},
        ]
        completed, live = _split_completed_and_live_exchange(messages)
        assert completed == []
        assert len(live) == 2
        assert live[1]["role"] == "unknown"

    def test_does_not_mutate_or_alias_nested_input(self):
        messages = [
            {"role": "user", "content": "old"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "X",
                        "type": "function",
                        "function": {"name": "r", "arguments": '{"k":"v"}'},
                    },
                ],
            },
            {"role": "tool", "tool_call_id": "X", "content": "res"},
        ]
        before = copy.deepcopy(messages)
        completed, live = _split_completed_and_live_exchange(messages)

        assert live[1] is not messages[1]
        assert live[1]["tool_calls"] is not messages[1]["tool_calls"]
        assert (
            live[1]["tool_calls"][0]["function"]
            is not messages[1]["tool_calls"][0]["function"]
        )

        live[1]["tool_calls"][0]["function"]["name"] = "mutated"
        assert messages[1]["tool_calls"][0]["function"]["name"] == "r"

        assert messages == before
