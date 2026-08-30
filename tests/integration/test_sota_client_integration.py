import json
import pytest
from unittest.mock import MagicMock, patch

from memory.sota_client import (
    SOTAClient,
    SOTAParseError,
    strip_code_blocks,
    resolve_temporal_reference,
)
from memory.sota_channel import CircuitBreakerOpen
from datetime import datetime


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _reset_sota_breaker():
    """Force circuit breaker closed before/after each test."""
    from memory.sota_channel import _sota_breaker
    _sota_breaker.force_closed()
    yield
    _sota_breaker.force_closed()


@pytest.fixture
def mock_sota_client():
    """Patch get_sota_client() so SOTAChannel uses a mock OpenAI client."""
    client_mock = MagicMock()
    with patch("memory._sota_http.get_sota_client", return_value=client_mock):
        yield client_mock


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_tool_response(tool_args: str):
    """Build a mock OpenAI response with a single tool call.

    Simulates what SOTAChannel.ask() processes in tool-call mode.
    """
    mock_fn = MagicMock()
    mock_fn.arguments = tool_args

    mock_tc = MagicMock()
    mock_tc.function = mock_fn

    mock_msg = MagicMock()
    mock_msg.content = None
    mock_msg.tool_calls = [mock_tc]  # single tool call

    mock_choice = MagicMock()
    mock_choice.message = mock_msg

    mock_resp = MagicMock()
    mock_resp.choices = [mock_choice]
    mock_resp.usage = MagicMock()
    return mock_resp


def _make_api_error(status_code: int, message="API error"):
    """Build a mock OpenAI APIError for SOTAChannel's retry handling."""
    from openai import APIError
    mock_response = MagicMock()
    mock_response.status_code = status_code
    mock_response.text = message
    return APIError(message=message, response=mock_response, body=None)


# ---------------------------------------------------------------------------
# Existing tests (preserved)
# ---------------------------------------------------------------------------

class TestSOTAClientIntegration:
    def test_extract_strips_code_blocks(self):
        text = "Some text with ```python\nprint('hello')\n``` code blocks"
        result = strip_code_blocks(text)
        assert "[CODE BLOCK REMOVED]" in result
        assert "print('hello')" not in result

    def test_extract_resolves_temporal_references(self):
        now = datetime(2025, 4, 20, 12, 0, 0)
        text = "I worked on this yesterday and also last week"
        result = resolve_temporal_reference(text, now)
        assert "yesterday" not in result
        assert "2025-04-19" in result


# ---------------------------------------------------------------------------
# Channel-wired integration tests
# ---------------------------------------------------------------------------

class TestSummarizeMessageChannelWired:
    """SOTAClient.summarize_message wired through SOTAChannel → mock OpenAI."""

    def test_returns_parsed_dict(self, mock_sota_client):
        client = SOTAClient(model="test-model")
        tool_args = json.dumps({
            "summary": "User asked about file structure.",
            "tool_summary": "[tools: read]",
            "token_estimate": 14,
        })
        mock_sota_client.chat.completions.create.return_value = (
            _make_tool_response(tool_args)
        )

        result = client.summarize_message("sess-1", 0, "What files are here?", "user")

        assert result["summary"] == "User asked about file structure."
        assert result["tool_summary"] == "[tools: read]"
        assert result["token_estimate"] == 14

    def test_empty_content_returns_default(self, mock_sota_client):
        """Empty raw_content short-circuits without calling SOTA."""
        client = SOTAClient(model="test-model")
        result = client.summarize_message("sess-1", 0, "", "user")
        assert result["summary"] == "(empty message)"
        assert result["tool_summary"] == ""

    def test_missing_keys_in_response_get_defaults(self, mock_sota_client):
        """When tool-call JSON is missing keys, defaults apply."""
        client = SOTAClient(model="test-model")
        tool_args = json.dumps({"summary": "Just summary"})
        mock_sota_client.chat.completions.create.return_value = (
            _make_tool_response(tool_args)
        )

        result = client.summarize_message("sess-1", 0, "hello", "user")

        assert result["summary"] == "Just summary"
        assert result["tool_summary"] == ""  # default
        assert result["token_estimate"] == 0  # default

    def test_long_summary_capped_at_512_tokens(self, mock_sota_client):
        """Summaries exceeding ~384 tokens are truncated with '...'."""
        client = SOTAClient(model="test-model")
        long_summary = " ".join(["word"] * 500)
        tool_args = json.dumps({
            "summary": long_summary,
            "tool_summary": "",
            "token_estimate": 500,
        })
        mock_sota_client.chat.completions.create.return_value = (
            _make_tool_response(tool_args)
        )

        result = client.summarize_message("sess-1", 0, "long text", "user")

        words = result["summary"].split()
        assert len(words) <= 385  # ~512 tokens * 0.75 = 384 words + "..."
        assert result["summary"].endswith("...")


class TestExtractAndSummarizeChannelWired:
    """SOTAClient.extract_and_summarize wired through SOTAChannel → mock OpenAI."""

    def test_returns_sync_result(self, mock_sota_client):
        client = SOTAClient(model="test-model")
        tool_args = json.dumps({
            "extraction_timestamp": "2025-05-10T12:00:00Z",
            "entities": [],
            "facts": [
                {
                    "source_entity": "user",
                    "target_entity": "python",
                    "relationship_type": "USES",
                    "fact": "User is learning Python.",
                    "confidence": "high",
                }
            ],
            "events": [],
            "instructions": [],
            "tasks": [
                {
                    "content": "Review Python basics",
                    "status": "pending",
                }
            ],
            "summary": "User learning Python.",
        })

        mock_sota_client.chat.completions.create.return_value = (
            _make_tool_response(tool_args)
        )

        result = client.extract_and_summarize(
            [{"role": "user", "content": "I'm learning Python!"}],
            current_summary="",
        )

        assert result.summary == "User learning Python."
        assert len(result.facts) == 1
        assert result.facts[0]["source_entity"] == "user"
        assert result.facts[0]["relationship_type"] == "USES"
        assert len(result.tasks) == 1
        assert result.tasks[0]["status"] == "pending"

    def test_missing_keys_default_to_empty_lists(self, mock_sota_client):
        """When response dict is missing extraction keys, defaults apply."""
        client = SOTAClient(model="test-model")
        tool_args = json.dumps({
            "extraction_timestamp": "2025-05-10T12:00:00Z",
            "entities": [],
            "summary": "Minimal response",
            # facts, events, instructions, tasks deliberately omitted
        })
        mock_sota_client.chat.completions.create.return_value = (
            _make_tool_response(tool_args)
        )

        result = client.extract_and_summarize(
            [{"role": "user", "content": "hello"}],
            current_summary="",
        )

        assert result.summary == "Minimal response"
        assert result.facts == []
        assert result.events == []
        assert result.instructions == []
        assert result.tasks == []

    def test_invalid_tool_args_raises_parse_error(self, mock_sota_client):
        """Malformed JSON in tool-call args raises SOTAParseError (after retries)."""
        client = SOTAClient(model="test-model")
        tool_args = "this is not valid json"
        mock_sota_client.chat.completions.create.return_value = (
            _make_tool_response(tool_args)
        )

        with pytest.raises(SOTAParseError, match="Failed to parse tool-call"):
            client.extract_and_summarize(
                [{"role": "user", "content": "hello"}],
                current_summary="",
            )


class TestChannelErrorPropagation:
    """SOTAClient error propagation through SOTAChannel."""

    def test_parse_error_from_malformed_tool_call(self, mock_sota_client):
        """Malformed tool-call JSON raises SOTAParseError in summarize_message."""
        client = SOTAClient(model="test-model")
        tool_args = "{{{broken"
        mock_sota_client.chat.completions.create.return_value = (
            _make_tool_response(tool_args)
        )

        with pytest.raises(SOTAParseError, match="Failed to parse tool-call"):
            client.summarize_message("sess-1", 0, "hello", "user")
