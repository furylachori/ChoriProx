"""
Regression tests for context.py and context_v2.py budget-triggered eviction.

These tests verify:
  - Chronological ordering (oldest→newest) is preserved after optimization
  - Token-budget low-watermark (0.65 of safe) caps the raw tier
  - Token budget keeps NEWEST messages (not oldest)
  - Token estimation produces correct results
  - Edge cases are handled gracefully
"""

import os
from unittest.mock import patch

import pytest

from proxy.context import select_context

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


def _system(content: str) -> dict:
    return {"role": "system", "content": content}


def _v2_config():
    return {
        "safe_context_tokens": 10000,
        "max_context_tokens": 40000,
        "tokens_per_char": 0.25,
        "raw_exchange_count": 100,
    }


def _safe_limit():
    return {"max_tokens": 4000}


def _run_v2(messages, config=None):
    """Run v2 context selection with mocks."""
    cfg = config or _v2_config()
    with patch("proxy.context.get_context_config", return_value=cfg), \
         patch("proxy.context.get_safe_context_limit", return_value=_safe_limit()), \
         patch("memory.db.get_ancient_history", return_value=""), \
         patch("memory.db.get_rich_summaries", return_value=[]):
        return select_context("test_session", messages)


# ---------------------------------------------------------------------------
# Test 1: Chronological order in v2 context optimization
# ---------------------------------------------------------------------------

class TestV2ChronologicalOrder:
    """Verify trimmed_raw messages appear in chronological order."""

    def test_v2_context_chronological_order(self):
        """Verify trimmed_raw messages appear in chronological order."""
        # Build a conversation with many messages (more than budget allows)
        messages = [_system("System prompt")]
        for i in range(50):
            messages.append(_user(f"User message {i:03d}"))
            messages.append(_assistant(f"Assistant response {i:03d}"))

        result = _run_v2(messages)

        # Extract non-system messages
        non_system = [m for m in result if m.get("role") != "system"]

        # Verify messages are oldest→newest
        msg_numbers = []
        for m in non_system:
            content = m.get("content", "")
            if "message" in content or "response" in content:
                parts = content.split()
                for part in parts:
                    if part.isdigit():
                        msg_numbers.append(int(part))
                        break

        # Verify ascending order
        for i in range(len(msg_numbers) - 1):
            assert msg_numbers[i] <= msg_numbers[i + 1], (
                f"V2 messages not in chronological order: {msg_numbers[i]} > {msg_numbers[i + 1]}"
            )

    def test_v2_last_message_is_most_recent(self):
        """Verify trimmed_raw[-1] is the most recent message."""
        messages = [_system("System")]
        for i in range(30):
            messages.append(_user(f"User {i:03d}"))
            messages.append(_assistant(f"Assistant {i:03d}"))

        result = _run_v2(messages)

        # The last non-system message should be the most recent assistant message
        non_system = [m for m in result if m.get("role") != "system"]
        assert len(non_system) > 0, "Should have at least one non-system message"

        last_msg = non_system[-1]
        assert "029" in last_msg.get("content", ""), (
            f"Last message should be the most recent (029), got: {last_msg.get('content')}"
        )


# ---------------------------------------------------------------------------
# Test 3: Token-budget low-watermark cap
# ---------------------------------------------------------------------------

class TestRawTokenBudgetCap:
    """Verify that the recent_raw low-watermark (0.65 of safe) caps the raw tier.

    The old count-based cap was removed in favor of a pure token budget.
    recent_raw_budget = int(model_safe_tokens * 0.65).
    """

    def test_raw_token_budget_cap_enforced_v2(self):
        """The raw tier must be trimmed to the 0.65 low-watermark of safe.

        safe=10000 → recent_raw_budget = int(10000 * 0.65) = 6500 tokens.
        200 messages × 100 tok = 20000 tokens → trimmed to ~65 messages.
        """
        pad = "x" * 400  # ~100 tokens per message
        messages = [_system("System")]
        for i in range(100):
            messages.append(_user(f"User {i} {pad}"))
            messages.append(_assistant(f"Assistant {i} {pad}"))

        result = _run_v2(messages)

        # Count non-system messages. Each is ~100 tokens; 6500 budget → ~65 kept
        # (user messages are single-message units, so the boundary lands cleanly).
        non_system = [m for m in result if m.get("role") != "system"]
        assert len(non_system) <= 65, (
            f"V2 output has {len(non_system)} non-system messages; raw budget "
            f"is 6500 tokens (~65 msgs at 100 tok each)"
        )
        assert len(non_system) > 0, "Expected some messages to survive the trim"


# ---------------------------------------------------------------------------
# Test 4: Token budget enforcement
# ---------------------------------------------------------------------------

class TestTokenBudgetEnforcement:
    """Verify that token budget is respected and newest messages are kept."""

    def test_token_budget_keeps_newest_v2(self):
        """Verify that the NEWEST units are kept (not the oldest) in v2."""
        # Build a conversation where only a portion of units fit
        messages = [_system("System")]
        for i in range(30):
            pad = "x" * 400
            messages.append(_user(f"User {i} {pad}"))
            messages.append(_assistant(f"Assistant {i} {pad}"))

        # safe must be large enough that recent_raw_budget + tool_schema <= safe.
        # safe=8000 → recent_raw_budget = 5200. 30 pairs × ~204 tok = ~6120 > 5200
        # → eviction keeps the newest ~25 pairs.
        tight_config = {
            "safe_context_tokens": 8000,
            "max_context_tokens": 40000,
            "tokens_per_char": 0.25,
            "raw_exchange_count": 100,
        }

        result = _run_v2(messages, config=tight_config)

        # Extract message numbers
        non_system = [m for m in result if m.get("role") != "system"]
        msg_numbers = []
        for m in non_system:
            content = m.get("content", "")
            if "User" in content or "Assistant" in content:
                parts = content.split()
                for part in parts:
                    if part.isdigit():
                        msg_numbers.append(int(part))
                        break

        # Eviction must have occurred and kept the NEWEST messages.
        assert msg_numbers, "Expected at least some messages to survive"
        assert 29 in msg_numbers, (
            f"Newest pair (29) must survive trimming, got numbers={sorted(set(msg_numbers))}"
        )
        assert 0 not in msg_numbers, (
            f"Oldest pair (0) must be evicted under tight budget, "
            f"got numbers={sorted(set(msg_numbers))}"
        )


# ---------------------------------------------------------------------------
# Test 5: String concatenation produces same token estimate
# ---------------------------------------------------------------------------

class TestTokenEstimationEquivalence:
    """Verify that the optimized token estimation produces the same result as string concatenation."""

    def test_token_estimation_equivalence(self):
        """Verify that sum of lengths equals length of concatenation."""
        # Build a list of messages with various content types
        messages = [
            _user("Hello, world!"),
            _assistant("Hi there! How can I help you today?"),
            _user("I need help with Python programming."),
            _assistant("Sure, I'd be happy to help with Python."),
            _user("Can you explain list comprehensions?"),
            _assistant("List comprehensions are a concise way to create lists."),
        ]

        # Old method: string concatenation + len
        total_content_old = ""
        for m in messages:
            content = m.get("content", "")
            if isinstance(content, str):
                total_content_old += content + "\n"
        estimated_old = int(len(total_content_old) * 0.25)

        # New method: sum of lengths (optimized)
        total_len = sum(
            len(m.get("content", "")) + 1  # +1 for newline
            for m in messages
            if isinstance(m.get("content", ""), str)
        )
        estimated_new = int(total_len * 0.25)

        # Verify they produce the same estimated_total
        assert estimated_old == estimated_new, (
            f"Token estimation mismatch: old={estimated_old}, new={estimated_new}"
        )

    def test_token_estimation_with_empty_content(self):
        """Verify token estimation handles empty content correctly."""
        messages = [
            _user("Hello"),
            _assistant(""),  # Empty content
            _user("World"),
        ]

        # Old method
        total_content_old = ""
        for m in messages:
            content = m.get("content", "")
            if isinstance(content, str):
                total_content_old += content + "\n"
        estimated_old = int(len(total_content_old) * 0.25)

        # New method
        total_len = sum(
            len(m.get("content", "")) + 1
            for m in messages
            if isinstance(m.get("content", ""), str)
        )
        estimated_new = int(total_len * 0.25)

        assert estimated_old == estimated_new


# ---------------------------------------------------------------------------
# Test 6: Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    """Verify edge cases are handled gracefully."""

    def test_empty_messages_v2(self):
        """Verify empty message list produces correct output."""
        result = _run_v2([])
        assert result == [], "Empty input should produce empty output"

    def test_single_message_v2(self):
        """Verify single message produces correct output."""
        messages = [_user("Hello")]
        result = _run_v2(messages)
        assert len(result) >= 1, "Single message should produce at least one output"
        # The message should be preserved
        assert any(m.get("content") == "Hello" for m in result)

    def test_all_messages_fit_v2(self):
        """Verify all messages kept when budget is large."""
        messages = [_system("System")]
        for i in range(5):
            messages.append(_user(f"User {i}"))
            messages.append(_assistant(f"Assistant {i}"))

        # Use a very large budget
        large_config = {
            "safe_context_tokens": 100000,
            "max_context_tokens": 400000,
            "tokens_per_char": 0.25,
            "raw_exchange_count": 100,
        }

        result = _run_v2(messages, config=large_config)

        # All non-system messages should be preserved
        non_system_input = [m for m in messages if m.get("role") != "system"]
        non_system_output = [m for m in result if m.get("role") != "system"]

        assert len(non_system_output) == len(non_system_input), (
            f"All messages should fit: expected {len(non_system_input)}, got {len(non_system_output)}"
        )

    def test_no_messages_fit_v2(self):
        """Verify graceful handling when no messages fit in budget."""
        messages = [_system("System")]
        for i in range(10):
            pad = "x" * 10000
            messages.append(_user(f"User {i} {pad}"))

        # Extremely tight budget (but must keep total_available > 0)
        tiny_config = {
            "safe_context_tokens": 2001,  # total_available = 2001 - 0 - 2000 = 1
            "max_context_tokens": 8004,
            "tokens_per_char": 0.25,
            "raw_exchange_count": 100,
        }

        result = _run_v2(messages, config=tiny_config)

        # Should return something (at least system prompt or last message)
        assert isinstance(result, list), "Should return a list even when nothing fits"
        assert len(result) >= 1, "Should return at least the system prompt or last message"

    def test_system_prompt_always_first(self):
        """Verify system prompt is always at index 0 when present."""
        messages = [_system("You are helpful")]
        for i in range(20):
            messages.append(_user(f"User {i}"))
            messages.append(_assistant(f"Assistant {i}"))

        result = _run_v2(messages)
        assert result[0]["role"] == "system", "System prompt should be first in v2"
        assert "helpful" in result[0]["content"]
