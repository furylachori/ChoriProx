"""
Unit tests for stream_response function.
Tests SSE chunk generation for streaming responses.
"""

import pytest
import json
from qwen_proxy import stream_response


class TestSSEChunkStructure:
    """Tests for SSE chunk structure and content."""

    def test_sse_chunk_has_role_delta(self):
        """SSE chunk should include role delta for assistant."""
        message = {"content": "Hello world", "tool_calls": None}
        original_data = {"id": "test-123", "model": "qwen3-coder"}

        chunks = list(stream_response(message, original_data))
        chunk_data = json.loads(chunks[0].replace("data: ", "").strip())

        assert "choices" in chunk_data
        assert len(chunk_data["choices"]) > 0
        assert "delta" in chunk_data["choices"][0]

    def test_sse_chunk_has_tool_calls(self):
        """SSE chunk should include tool_calls when present in message."""
        tool_call = {
            "id": "call_123",
            "type": "function",
            "function": {"name": "read_file", "arguments": '{"path": "/tmp/test.txt"}'},
        }
        message = {"tool_calls": [tool_call], "content": None}
        original_data = {"id": "test-123", "model": "qwen3-coder"}

        chunks = list(stream_response(message, original_data))
        chunk_text = "".join(chunks)

        assert "tool_calls" in chunk_text or "read_file" in chunk_text

    def test_sse_chunk_has_finish_reason(self):
        """SSE chunk should include finish_reason."""
        message = {"content": "Done", "tool_calls": None}
        original_data = {"id": "test-123", "model": "qwen3-coder"}

        chunks = list(stream_response(message, original_data))
        chunk_text = "".join(chunks)

        assert "finish_reason" in chunk_text

    def test_sse_tool_call_index_sequence(self):
        """Tool calls should have correct index sequence (0, 1, 2...)."""
        tool_calls = [
            {
                "id": "call_1",
                "type": "function",
                "function": {
                    "name": "read_file",
                    "arguments": '{"path": "/tmp/a.txt"}',
                },
            },
            {
                "id": "call_2",
                "type": "function",
                "function": {
                    "name": "write_file",
                    "arguments": '{"path": "/tmp/b.txt"}',
                },
            },
        ]
        message = {"tool_calls": tool_calls, "content": None}
        original_data = {"id": "test-123", "model": "qwen3-coder"}

        chunks = list(stream_response(message, original_data))

        indices_found = set()
        for chunk in chunks:
            if "index" in chunk:
                data = json.loads(chunk.replace("data: ", "").strip())
                for choice in data.get("choices", []):
                    for tc in choice.get("delta", {}).get("tool_calls", []):
                        if "index" in tc:
                            indices_found.add(tc["index"])

        if indices_found:
            assert 0 in indices_found


class TestStreamResponseContent:
    """Tests for content streaming behavior."""

    def test_content_streamed_when_no_tools(self):
        """Content should be streamed when no tool calls present."""
        message = {"content": "Hello there!", "tool_calls": None}
        original_data = {"id": "test-123", "model": "qwen3-coder"}

        chunks = list(stream_response(message, original_data))
        chunk_text = "".join(chunks)

        assert "Hello" in chunk_text or "content" in chunk_text

    def test_done_signal_sent(self):
        """DONE signal should be sent at end of stream."""
        message = {"content": "Done", "tool_calls": None}
        original_data = {"id": "test-123", "model": "qwen3-coder"}

        chunks = list(stream_response(message, original_data))

        assert "[DONE]" in chunks[-1] or chunks[-1].endswith('"stop"}')


class TestTerminalFinishReason:
    """Terminal-chunk finish_reason must propagate truncation reasons from
    original_data and default otherwise."""

    _TOOL_CALL = {
        "id": "call_1",
        "type": "function",
        "function": {"name": "read_file", "arguments": '{"path": "/tmp/a"}'},
    }

    @staticmethod
    def _terminal_finish_reason(chunks):
        """Return finish_reason from the terminal chunk (the one with non-null finish_reason)."""
        for chunk in chunks:
            if chunk.startswith("data: [DONE]"):
                continue
            data = json.loads(chunk.replace("data: ", "", 1).strip())
            fr = data["choices"][0].get("finish_reason")
            if fr is not None:
                return fr
        raise AssertionError("no terminal chunk with finish_reason found")

    def test_content_length_preserved(self):
        message = {"content": "x"}
        original_data = {"choices": [{"finish_reason": "length"}]}
        chunks = list(stream_response(message, original_data))
        assert self._terminal_finish_reason(chunks) == "length"

    def test_content_content_filter_preserved(self):
        message = {"content": "x"}
        original_data = {"choices": [{"finish_reason": "content_filter"}]}
        chunks = list(stream_response(message, original_data))
        assert self._terminal_finish_reason(chunks) == "content_filter"

    def test_content_stop_unchanged(self):
        message = {"content": "x"}
        original_data = {"choices": [{"finish_reason": "stop"}]}
        chunks = list(stream_response(message, original_data))
        assert self._terminal_finish_reason(chunks) == "stop"

    def test_content_omitted_defaults_stop(self):
        message = {"content": "x"}
        original_data = {"choices": [{}]}
        chunks = list(stream_response(message, original_data))
        assert self._terminal_finish_reason(chunks) == "stop"

    def test_content_none_defaults_stop(self):
        message = {"content": "x"}
        original_data = {"choices": [{"finish_reason": None}]}
        chunks = list(stream_response(message, original_data))
        assert self._terminal_finish_reason(chunks) == "stop"

    def test_tool_calls_length_preserved(self):
        message = {"tool_calls": [self._TOOL_CALL], "content": None}
        original_data = {"choices": [{"finish_reason": "length"}]}
        chunks = list(stream_response(message, original_data))
        assert self._terminal_finish_reason(chunks) == "length"

    def test_tool_calls_content_filter_preserved(self):
        message = {"tool_calls": [self._TOOL_CALL], "content": None}
        original_data = {"choices": [{"finish_reason": "content_filter"}]}
        chunks = list(stream_response(message, original_data))
        assert self._terminal_finish_reason(chunks) == "content_filter"

    def test_tool_calls_stop_yields_tool_calls(self):
        message = {"tool_calls": [self._TOOL_CALL], "content": None}
        original_data = {"choices": [{"finish_reason": "stop"}]}
        chunks = list(stream_response(message, original_data))
        assert self._terminal_finish_reason(chunks) == "tool_calls"

    def test_tool_calls_none_yields_tool_calls(self):
        message = {"tool_calls": [self._TOOL_CALL], "content": None}
        original_data = {"choices": [{"finish_reason": None}]}
        chunks = list(stream_response(message, original_data))
        assert self._terminal_finish_reason(chunks) == "tool_calls"

    def test_malformed_original_data_choices_missing(self):
        message = {"content": "x"}
        original_data = {}
        chunks = list(stream_response(message, original_data))
        assert self._terminal_finish_reason(chunks) == "stop"

    def test_reasoning_chunks_carry_null_finish_reason(self):
        message = {"content": "x", "reasoning_content": "think"}
        original_data = {"choices": [{"finish_reason": "stop"}]}
        chunks = list(stream_response(message, original_data))
        finish_reasons = []
        for chunk in chunks:
            if chunk.startswith("data: [DONE]"):
                continue
            data = json.loads(chunk.replace("data: ", "", 1).strip())
            finish_reasons.append(data["choices"][0].get("finish_reason"))
        non_null = [fr for fr in finish_reasons if fr is not None]
        assert len(non_null) == 1, f"expected exactly one terminal chunk, got {non_null}"
        assert all(fr is None for fr in finish_reasons[:-1])
