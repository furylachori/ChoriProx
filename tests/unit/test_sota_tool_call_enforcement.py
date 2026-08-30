"""
***** DO NOT REWRITE THIS TEST *****

This test exists because StepFun (the SOTA provider) does NOT support
response_format={"type": "json_object"}. The API returns 400 Bad Request:
  "structured_outputs is not supported"

ALL SOTA calls MUST use forced tool calls instead:
  tools=[{...tool schema...}]
  tool_choice={"type": "function", "function": {"name": "..."}}

If any SOTA method is rewritten to use response_format, this test catches it.
DO NOT fix this test by adding response_format to the assertions --
remove response_format from the SOTA method and use tool calls.
"""

import pytest
from unittest.mock import MagicMock, patch

pytestmark = pytest.mark.unit


def _build_mock_sota_result(raw_json):
    """Build a mock SOTAResult with parsed data."""
    mock = MagicMock()
    mock.raw = raw_json
    mock.parsed = raw_json
    mock.elapsed_ms = 50.0
    mock.attempts = 1
    mock.usage = None
    return mock


class TestSummarizeMessageUsesToolCalls:

    def test_no_response_format(self):
        from memory.sota_client import SOTAClient

        mock_result = _build_mock_sota_result(
            {"summary": "test", "tool_summary": "", "token_estimate": 1}
        )
        with patch("memory.sota_channel.SOTAChannel.ask", return_value=mock_result) as mock_ask:
            sota = SOTAClient(api_key="k", endpoint="https://x/v1", model="m", timeout=10)
            sota.summarize_message("s", 0, "hello world", "user")
            call_kwargs = mock_ask.call_args.kwargs
            assert "response_format" not in call_kwargs, (
                "\n"
                "  ==========================================================\n"
                "  *** SOTA CRITICAL: response_format DETECTED ***\n"
                "  Method: summarize_message()\n"
                "  StepFun returns 400: structured_outputs is not supported.\n"
                "  FIX: Remove response_format. Use tools + tool_choice.\n"
                "  ==========================================================\n"
            )


class TestExtractAndSummarizeUsesToolCalls:

    def test_no_response_format(self):
        from memory.sota_client import SOTAClient

        mock_result = _build_mock_sota_result({
            "extraction_timestamp": "2026-01-01T00:00:00Z",
            "entities": [],
            "facts": [],
            "events": [],
            "instructions": [],
            "tasks": [],
            "summary": "",
        })
        with patch("memory.sota_channel.SOTAChannel.ask", return_value=mock_result) as mock_ask:
            sota = SOTAClient(api_key="k", endpoint="https://x/v1", model="m", timeout=10)
            sota.extract_and_summarize([{"role": "user", "content": "x"}])
            call_kwargs = mock_ask.call_args.kwargs
            assert "response_format" not in call_kwargs, (
                "\n"
                "  ==========================================================\n"
                "  *** SOTA CRITICAL: response_format DETECTED ***\n"
                "  Method: extract_and_summarize()\n"
                "  StepFun returns 400: structured_outputs is not supported.\n"
                "  FIX: Remove response_format. Use tools + tool_choice.\n"
                "  ==========================================================\n"
            )


class TestExtractFromSummariesUsesToolCalls:

    def test_no_response_format(self):
        from memory.sota_client import SOTAClient

        mock_result = _build_mock_sota_result({
            "extraction_timestamp": "2026-01-01T00:00:00Z",
            "entities": [{"name": "XX", "type": "Concept", "summary": "Test entity"}],
            "facts": [],
        })
        with patch("memory.sota_channel.SOTAChannel.ask", return_value=mock_result) as mock_ask:
            sota = SOTAClient(api_key="k", endpoint="https://x/v1", model="m", timeout=10)
            sota.extract_and_summarize_from_summaries("s", [
                {"message_index": 1, "exchange_index": 1, "role": "user", "summary": "a"},
                {"message_index": 2, "exchange_index": 2, "role": "assistant", "summary": "b"},
                {"message_index": 3, "exchange_index": 3, "role": "user", "summary": "c"},
            ])
            call_kwargs = mock_ask.call_args.kwargs
            assert "response_format" not in call_kwargs, (
                "\n"
                "  ==========================================================\n"
                "  *** SOTA CRITICAL: response_format DETECTED ***\n"
                "  Method: extract_and_summarize_from_summaries()\n"
                "  StepFun returns 400: structured_outputs is not supported.\n"
                "  FIX: Remove response_format. Use tools + tool_choice.\n"
                "  ==========================================================\n"
            )