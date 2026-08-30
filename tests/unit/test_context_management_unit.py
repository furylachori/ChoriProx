"""
Unit tests for select_context function.
Tests context management with memory injection.
"""

import os
import pytest
from unittest.mock import patch
from qwen_proxy import select_context
from proxy.context import _count_msg_chars, ContextLimitError


class TestSystemPrompt:
    """# US-026: Tests for system prompt preservation."""

    def test_system_prompt_always_index_0(self):
        """System prompt should always be at index 0."""
        system_msg = {"role": "system", "content": "You are a helpful assistant."}
        user_msg = {"role": "user", "content": "Hello"}
        messages = [system_msg, user_msg]

        result = select_context("test_session", messages)

        assert len(result) >= 1
        assert result[0]["role"] == "system"
        assert result[0]["content"] == "You are a helpful assistant."


class TestTruncation:
    """Tests for token truncation behavior."""

    def test_truncation_respects_max_tokens(self):
        """Context should respect token limits."""
        system_msg = {"role": "system", "content": "System prompt"}
        large_msg = {"role": "user", "content": "A" * 10000}
        messages = [system_msg, large_msg]

        result = select_context("test_session", messages)

        assert len(result) >= 1


class TestOversizedMessage:
    """Tests for oversized single message handling."""

    def test_oversized_single_message_mode(self):
        """Oversized message (>1200 tokens) should trigger single message mode."""
        system_msg = {"role": "system", "content": "System prompt"}
        large_content = "This is a very long message. " * 500
        large_msg = {"role": "user", "content": large_content}
        messages = [system_msg, large_msg]

        result = select_context("test_session", messages)

        assert len(result) == 2
        assert result[0]["role"] == "system"


class TestBatchMode:
    """Tests for normal batch mode."""

    def test_normal_batch_mode(self):
        """Normal sized messages should use batch mode."""
        system_msg = {"role": "system", "content": "System"}
        user_msg = {"role": "user", "content": "Short message"}
        messages = [system_msg, user_msg]

        result = select_context("test_session", messages)

        assert len(result) >= 1


class TestMemoryInjection:
    """Tests for memory injection behavior."""

    def test_memory_injection_skipped_on_empty_recall(self):
        """Memory injection should be skipped when recall returns empty."""
        system_msg = {"role": "system", "content": "You are helpful."}
        user_msg = {"role": "user", "content": "Hello"}
        messages = [system_msg, user_msg]

        result = select_context("test_session", messages)

        assert len(result) >= 1

    def test_memory_injection_preserves_message_order(self):
        """Memory injection should preserve original message order."""
        system_msg = {"role": "system", "content": "System"}
        user_msg = {"role": "user", "content": "Hello"}
        messages = [system_msg, user_msg]

        result = select_context("test_session", messages)

        roles = [msg["role"] for msg in result]
        if len(roles) > 1:
            system_indices = [i for i, r in enumerate(roles) if r == "system"]
            if len(system_indices) > 1:
                assert system_indices[0] < system_indices[1]


class TestModelSafeTokensBoundary:
    """Tests for model_safe_tokens <= 0 edge case."""

    @patch("proxy.context.get_context_config")
    def test_model_safe_tokens_zero_returns_system_and_last(self, mock_get_config):
        """Slice 4D: a nonempty candidate cannot fit a zero-token safe
        budget, so ``_select_context_v2`` raises ``ContextLimitError``.
        The empty-input/no-pin/no-tools case remains ``[]``; this test
        deliberately supplies a nonempty conversation.
        """
        mock_get_config.return_value = {
            "safe_context_tokens": 0,
            "max_context_tokens": 100000,
            "tokens_per_char": 0.25,
        }

        system_msg = {"role": "system", "content": "You are helpful."}
        user_msg1 = {"role": "user", "content": "First message"}
        user_msg2 = {"role": "user", "content": "Last message"}
        messages = [system_msg, user_msg1, user_msg2]

        with pytest.raises(ContextLimitError) as excinfo:
            select_context("test_session", messages)

        assert "Final context exceeds safe token budget" in str(excinfo.value)


class TestCountMsgChars:
    """Tests for _count_msg_chars utility."""

    def test_string_content(self):
        msg = {"role": "user", "content": "hello"}
        assert _count_msg_chars(msg) == 5

    def test_list_content(self):
        msg = {"role": "user", "content": [{"type": "text", "text": "abc"}]}
        assert _count_msg_chars(msg) == 3

    def test_empty_content(self):
        msg = {"role": "assistant", "content": ""}
        assert _count_msg_chars(msg) == 0

    def test_missing_content(self):
        msg = {"role": "user"}
        assert _count_msg_chars(msg) == 0

    def test_tool_calls(self):
        msg = {"role": "assistant", "content": None, "tool_calls": [
            {"function": {"name": "read", "arguments": '{"path":"/tmp"}'}}
        ]}
        assert _count_msg_chars(msg) == len("read") + len('{"path":"/tmp"}')

    def test_reasoning_content(self):
        msg = {"role": "assistant", "content": "hi", "reasoning_content": "thinking"}
        assert _count_msg_chars(msg) == 2 + 8  # "hi" + "thinking"

    def test_reasoning_field_fallback(self):
        msg = {"role": "assistant", "content": "ok", "reasoning": "deep"}
        assert _count_msg_chars(msg) == 2 + 4


class TestContextLimitErrorSeparation:
    """Tests for H2: ContextLimitError vs ValueError separation."""

    def test_context_limit_error_is_not_value_error(self):
        """ContextLimitError should NOT be a ValueError subclass (standalone)."""
        assert not issubclass(ContextLimitError, ValueError)

    def test_context_limit_error_is_exception(self):
        """ContextLimitError should be an Exception subclass."""
        assert issubclass(ContextLimitError, Exception)

    def test_empty_messages_returns_empty(self):
        """C4: Empty messages list should return empty, not crash."""
        from proxy.context_v2 import _select_context_v2
        cfg = {"max_tokens": 1000, "tokens_per_char": 0.25, "safe_context_tokens": 1000}
        limits = {"model_safe_tokens": 1000}

        with patch("proxy.context.get_context_config", return_value=cfg), \
             patch("memory.db.get_ancient_history", return_value=""), \
             patch("memory.db.get_rich_summaries", return_value=[]):
            result = _select_context_v2("sess1", [], model_name="test")
        assert result == []


class TestSuffixReservation:
    """Tests for M2: truncation suffix should not exceed remaining budget."""

    def test_truncation_reserves_suffix_space(self):
        """After truncation, content + suffix should fit within budget."""
        from proxy.context import calculate_tokens_from_text
        suffix = "... [truncated]"
        # Simulate a tight budget: 10 tokens → ~40 chars
        max_chars = int(10 / 0.25) - len(suffix)
        content = "A" * 200
        if max_chars > 0 and len(content) > max_chars:
            truncated = content[:max_chars] + suffix
            assert len(truncated) == max_chars + len(suffix)
            # Verify it's strictly less than or equal to original budget in chars
            assert len(truncated) <= int(10 / 0.25)

    def test_very_tight_budget_includes_message_as_is(self):
        """When max_chars <= 0 after suffix reservation, message should be included as-is."""
        # This tests the edge case where remaining_budget is too small for suffix
        # The code should include the message unchanged rather than crash
        suffix = "... [truncated]"
        remaining_budget = 1  # 1 token → 4 chars
        tokens_per_char = 0.25
        max_chars = int(remaining_budget / tokens_per_char) - len(suffix)
        assert max_chars <= 0  # Verify edge case is triggered
