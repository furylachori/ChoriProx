import pytest
from unittest.mock import MagicMock, patch

import memory.sync as sync_mod
from memory.sync import (
    _build_previous_context,
    _cache_reasoning,
    _pop_reasoning,
    _reasoning_cache,
    _reasoning_cache_lock,
    _MAX_REASONING_ENTRIES,
)
from memory.sota_client import SOTAClient
from memory.prompts.summarize_message import SUMMARIZE_MESSAGE_SYSTEM_PROMPT, SUMMARIZE_MESSAGE_INSTRUCTIONS

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _clear_reasoning_cache():
    """Reset the in-memory reasoning cache before every test."""
    with _reasoning_cache_lock:
        _reasoning_cache.clear()
    yield
    with _reasoning_cache_lock:
        _reasoning_cache.clear()


# ---------------------------------------------------------------------------
# TestBuildPreviousContext
# ---------------------------------------------------------------------------

class TestBuildPreviousContext:
    """Tests for _build_previous_context in memory.sync."""

    # ── helpers ──────────────────────────────────────────────────────────

    @staticmethod
    def _rows(*indices, session_id="sess-1"):
        """Build a minimal rows list with user messages at given indices."""
        return [
            {
                "message_index": i,
                "role": "user",
                "content": f"message {i}",
                "session_id": session_id,
            }
            for i in indices
        ]

    # ── tests ────────────────────────────────────────────────────────────

    def test_empty_rows_returns_empty_string(self):
        result = _build_previous_context([], 5)
        assert result == ""

    def test_no_preceding_messages_returns_empty_string(self):
        rows = self._rows(5, 6, 7)
        result = _build_previous_context(rows, 5)
        assert result == ""

    def test_single_preceding_message_formatted(self):
        rows = self._rows(0, 1)
        result = _build_previous_context(rows, 1)
        assert result == "[MSG #0 | user]: message 0"

    def test_multiple_preceding_messages_in_chronological_order(self):
        rows = self._rows(0, 1, 2)
        result = _build_previous_context(rows, 3)
        lines = result.split("\n\n")
        assert len(lines) == 3
        assert "[MSG #0 " in lines[0]
        assert "[MSG #1 " in lines[1]
        assert "[MSG #2 " in lines[2]
        # Verify oldest first by checking index order in the string
        idx_positions = [result.index(f"[MSG #{i}") for i in range(3)]
        assert idx_positions == sorted(idx_positions)

    def test_system_messages_skipped(self):
        rows = [
            {
                "message_index": 0,
                "role": "user",
                "content": "hello",
                "session_id": "sess-1",
            },
            {
                "message_index": 1,
                "role": "system",
                "content": "you are a bot",
                "session_id": "sess-1",
            },
            {
                "message_index": 2,
                "role": "user",
                "content": "world",
                "session_id": "sess-1",
            },
        ]
        result = _build_previous_context(rows, 3)
        # System message at index 1 should not appear
        assert "system" not in result
        assert "bot" not in result
        # Both user messages should appear
        assert "[MSG #0" in result
        assert "[MSG #2" in result

    def test_respects_max_chars_budget(self):
        # Use a very small budget so truncation is obvious
        rows = self._rows(0, 1, 2, 3, 4)
        result = _build_previous_context(rows, 5, max_chars=50)
        # The function walks backwards; the oldest surviving entry gets truncated
        # mid-string. The "..." appears somewhere, not necessarily at the end.
        assert "..." in result
        # Result should not contain every message (indices 0-2 are dropped)
        assert "[MSG #0" not in result
        assert "[MSG #1" not in result
        assert "[MSG #2" not in result
        # But the most recent preceding messages (3, 4) should be present
        assert "[MSG #3" in result
        assert "[MSG #4" in result

    def test_includes_reasoning_when_cached(self):
        """When _pop_reasoning returns reasoning, output includes thinking block."""
        rows = [
            {
                "message_index": 0,
                "role": "user",
                "content": "question",
                "session_id": "sess-1",
            },
            {
                "message_index": 1,
                "role": "assistant",
                "content": "answer",
                "session_id": "sess-1",
            },
        ]
        with patch("memory.sync._pop_reasoning", return_value="deep analysis"):
            result = _build_previous_context(rows, 2)
        # Assistant message should show reasoning
        expected = (
            "[MSG #0 | user]: question\n\n"
            "[MSG #1 | assistant]:  thinking deep analysis... response answer"
        )
        assert result == expected

    def test_no_reasoning_when_not_cached(self):
        """When _pop_reasoning returns empty, plain format is used."""
        rows = [
            {
                "message_index": 0,
                "role": "user",
                "content": "question",
                "session_id": "sess-1",
            },
            {
                "message_index": 1,
                "role": "assistant",
                "content": "answer",
                "session_id": "sess-1",
            },
        ]
        with patch("memory.sync._pop_reasoning", return_value=""):
            result = _build_previous_context(rows, 2)
        expected = (
            "[MSG #0 | user]: question\n\n"
            "[MSG #1 | assistant]: answer"
        )
        assert result == expected

    def test_empty_content_message_handled(self):
        """A message with None content uses empty string for content."""
        rows = [
            {
                "message_index": 0,
                "role": "user",
                "content": "hi",
                "session_id": "sess-1",
            },
            {
                "message_index": 1,
                "role": "assistant",
                "content": None,
                "session_id": "sess-1",
            },
        ]
        with patch("memory.sync._pop_reasoning", return_value=""):
            result = _build_previous_context(rows, 2)
        assert "[MSG #1 | assistant]: " in result


# ---------------------------------------------------------------------------
# TestReasoningCache
# ---------------------------------------------------------------------------

class TestReasoningCache:
    """Tests for _cache_reasoning and _pop_reasoning in memory.sync."""

    def test_cache_and_pop_reasoning(self):
        _cache_reasoning("sess-a", 5, "step-by-step thinking")
        result = _pop_reasoning("sess-a", 5)
        assert result == "step-by-step thinking"

    def test_pop_nonexistent_returns_empty(self):
        result = _pop_reasoning("sess-a", 99)
        assert result == ""

    def test_cache_empty_string_skipped(self):
        _cache_reasoning("sess-a", 5, "")
        result = _pop_reasoning("sess-a", 5)
        assert result == ""

    def test_cache_whitespace_only_skipped(self):
        _cache_reasoning("sess-a", 5, "   ")
        result = _pop_reasoning("sess-a", 5)
        assert result == ""

    def test_cross_session_isolation(self):
        _cache_reasoning("sess-a", 1, "reasoning-a")
        # Pop from a different session should return ""
        result = _pop_reasoning("sess-b", 1)
        assert result == ""
        # The original entry should still be available for sess-a
        result_a = _pop_reasoning("sess-a", 1)
        assert result_a == "reasoning-a"

    def test_pop_removes_entry(self):
        _cache_reasoning("sess-a", 3, "some reasoning")
        first = _pop_reasoning("sess-a", 3)
        assert first == "some reasoning"
        second = _pop_reasoning("sess-a", 3)
        assert second == ""

    def test_lru_eviction_when_over_cap(self):
        """With _MAX_REASONING_ENTRIES=2, caching 3 entries evicts oldest."""
        saved_cap = _MAX_REASONING_ENTRIES
        try:
            sync_mod._MAX_REASONING_ENTRIES = 2

            _cache_reasoning("sess-x", 1, "first")
            _cache_reasoning("sess-x", 2, "second")
            _cache_reasoning("sess-x", 3, "third")  # should evict idx 1 (oldest)

            # Oldest should be gone
            assert _pop_reasoning("sess-x", 1) == ""
            # The two remaining should survive
            assert _pop_reasoning("sess-x", 2) == "second"
            assert _pop_reasoning("sess-x", 3) == "third"
        finally:
            sync_mod._MAX_REASONING_ENTRIES = saved_cap


# ---------------------------------------------------------------------------
# TestSummarizeMessagePromptNewContent
# ---------------------------------------------------------------------------

class TestSummarizeMessagePromptNewContent:
    """Verify the updated prompts contain the expected sections."""

    def test_system_prompt_mentions_previous_context(self):
        assert "previous conversation history" in SUMMARIZE_MESSAGE_SYSTEM_PROMPT.lower()

    def test_instructions_mention_dimensions(self):
        for term in ("WHAT", "DECISIONS", "FILES", "TOOLS", "OUTCOME"):
            assert term in SUMMARIZE_MESSAGE_INSTRUCTIONS, f"'{term}' not in instructions"

    def test_instructions_mention_3_to_6_sentences(self):
        assert "3-6 sentence" in SUMMARIZE_MESSAGE_INSTRUCTIONS


# ---------------------------------------------------------------------------
# TestSummarizeMessageWithContext
# ---------------------------------------------------------------------------

class TestSummarizeMessageWithContext:
    """Mock-based tests verifying previous_messages_text is injected into the API call."""

    def test_context_passed_to_prompt(self):
        """When previous_messages_text is provided, it appears in the user message."""
        client = SOTAClient()

        mock_response = MagicMock()
        mock_response.choices = [
            MagicMock(
                message=MagicMock(
                    role="assistant",
                    content='{"summary": "test", "tool_summary": "", "token_estimate": 1}',
                    tool_calls=None,
                )
            )
        ]

        # Ensure SOTAChannel.get_sota_client() returns the same client instance
        # that we've patched (the module-level singleton is a separate instance).
        with patch.object(
            client.client.chat.completions, "create", return_value=mock_response
        ) as mock_create, patch(
            "memory._sota_http.get_sota_client", return_value=client.client
        ), patch(
            "time.monotonic", side_effect=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        ), patch("time.sleep"):
            result = client.summarize_message(
                "sess-1", 0, "Hello", "user",
                previous_messages_text="some context",
            )

        assert result["summary"] == "test"

        # Verify the user message sent to the API contains the context
        call_args = mock_create.call_args
        messages = call_args.kwargs["messages"]
        user_message = [m for m in messages if m["role"] == "user"][0]
        assert "[Conversation History" in user_message["content"]
        assert "some context" in user_message["content"]
