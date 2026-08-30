"""Comprehensive SSE/SSE streaming tests for the ChoriProx proxy.

Covers:
- SSE format validation (data: {...}\\n\\n)
- Streaming with tool calls and correct index sequencing
- Streaming error handling
- MiniMax <think> block extraction in streaming context
- Chunk assembly and partial JSON buffering
- Streaming timeout behavior
- SSE headers (Cache-Control, Connection, X-Accel-Buffering)
- Session token injection in streaming responses
- Edge cases (empty content, null fields, multiple tool calls)
# US-059: SSE Streaming Format
"""

import json
import re
import time
from unittest.mock import MagicMock, patch

import pytest

from proxy.streaming import make_sse_chunk, stream_response


def parse_sse_events(sse_text):
    """Parse raw SSE text into structured events.

    Returns list of dicts: {"type": "data"|"done"|"unknown", "payload": dict|str}
    """
    events = []
    for block in sse_text.strip().split("\n\n"):
        block = block.strip()
        if not block:
            continue
        if block == "data: [DONE]":
            events.append({"type": "done", "payload": "[DONE]"})
        elif block.startswith("data: "):
            try:
                events.append({"type": "data", "payload": json.loads(block[6:])})
            except json.JSONDecodeError:
                events.append({"type": "unknown", "payload": block})
        else:
            events.append({"type": "unknown", "payload": block})
    return events


def parse_sse_chunks(sse_text):
    """Parse SSE text into list of JSON dicts (excluding [DONE])."""
    return [e["payload"] for e in parse_sse_events(sse_text) if e["type"] == "data"]


# ============================================================================
# 1. Basic SSE Format Validation
# ============================================================================


class TestSSEFormatValidation:
    """Verify strict SSE format: every chunk must be 'data: {json}\\n\\n'."""

    def test_make_sse_chunk_format(self):
        chunk = make_sse_chunk({"hello": "world"})
        assert chunk == 'data: {"hello": "world"}\n\n'

    def test_make_sse_chunk_ends_with_double_newline(self):
        chunk = make_sse_chunk({"key": "value"})
        assert chunk.endswith("\n\n")

    def test_make_sse_chunk_starts_with_data_prefix(self):
        chunk = make_sse_chunk({"a": 1})
        assert chunk.startswith("data: ")

    def test_make_sse_chunk_valid_json_payload(self):
        data = {"choices": [{"delta": {"content": "test"}, "index": 0}]}
        chunk = make_sse_chunk(data)
        payload = chunk[6:-2]
        parsed = json.loads(payload)
        assert parsed == data

    def test_stream_response_all_chunks_valid_sse_format(self):
        message = {"role": "assistant", "content": "Hello world"}
        original_data = {"id": "test-1", "created": 1000, "model": "test-model"}
        sse = "".join(stream_response(message, original_data, "test-model"))

        for block in sse.strip().split("\n\n"):
            block = block.strip()
            if not block:
                continue
            assert block.startswith("data: "), f"Invalid SSE block: {block!r}"

    def test_stream_response_ends_with_done(self):
        message = {"role": "assistant", "content": "Hello"}
        original_data = {"id": "test-1", "created": 1000, "model": "m"}
        sse = "".join(stream_response(message, original_data, "m"))
        assert sse.strip().endswith("data: [DONE]")

    def test_stream_response_object_field_is_chunk(self):
        message = {"role": "assistant", "content": "test"}
        original_data = {"id": "test-1", "created": 1000, "model": "m"}
        chunks = parse_sse_chunks("".join(stream_response(message, original_data, "m")))
        for chunk in chunks:
            assert chunk["object"] == "chat.completion.chunk"

    def test_stream_response_preserves_id(self):
        message = {"role": "assistant", "content": "test"}
        original_data = {"id": "chatcmpl-abc123", "created": 1000, "model": "m"}
        chunks = parse_sse_chunks(
            "".join(stream_response(message, original_data, "m"))
        )
        for chunk in chunks:
            assert chunk["id"] == "chatcmpl-abc123"

    def test_stream_response_preserves_created(self):
        message = {"role": "assistant", "content": "test"}
        original_data = {"id": "test-1", "created": 1700000000, "model": "m"}
        chunks = parse_sse_chunks(
            "".join(stream_response(message, original_data, "m"))
        )
        for chunk in chunks:
            assert chunk["created"] == 1700000000

    def test_stream_response_uses_original_model(self):
        message = {"role": "assistant", "content": "test"}
        original_data = {"id": "test-1", "created": 1000, "model": "backend-model"}
        chunks = parse_sse_chunks(
            "".join(stream_response(message, original_data, "user-facing-model"))
        )
        for chunk in chunks:
            assert chunk["model"] == "user-facing-model"

    def test_stream_response_fallback_model(self):
        message = {"role": "assistant", "content": "test"}
        original_data = {"id": "test-1", "created": 1000, "model": "fallback"}
        chunks = parse_sse_chunks(
            "".join(stream_response(message, original_data))
        )
        for chunk in chunks:
            assert chunk["model"] == "fallback"


# ============================================================================
# 2. Content Streaming
# ============================================================================


class TestContentStreaming:
    """Verify content is streamed correctly in various scenarios."""

    def test_content_stream_chunk_sequence(self):
        message = {"role": "assistant", "content": "Hello world"}
        original_data = {"id": "t", "created": 0, "model": "m"}
        events = parse_sse_events(
            "".join(stream_response(message, original_data, "m"))
        )

        assert events[0]["type"] == "data"
        assert events[0]["payload"]["choices"][0]["delta"].get("role") == "assistant"

        content_events = [
            e for e in events
            if e["type"] == "data"
            and e["payload"]["choices"][0]["delta"].get("content")
        ]
        assert len(content_events) == 1
        assert content_events[0]["payload"]["choices"][0]["delta"]["content"] == "Hello world"

        assert events[-1]["type"] == "done"

    def test_finish_reason_stop_for_content(self):
        message = {"role": "assistant", "content": "Done"}
        original_data = {"id": "t", "created": 0, "model": "m"}
        chunks = parse_sse_chunks(
            "".join(stream_response(message, original_data, "m"))
        )
        last_chunk = chunks[-1]
        assert last_chunk["choices"][0]["finish_reason"] == "stop"

    def test_empty_content_still_emits_role_and_stop(self):
        message = {"role": "assistant", "content": ""}
        original_data = {"id": "t", "created": 0, "model": "m"}
        chunks = parse_sse_chunks(
            "".join(stream_response(message, original_data, "m"))
        )
        assert len(chunks) >= 2
        assert chunks[0]["choices"][0]["delta"].get("role") == "assistant"
        assert chunks[-1]["choices"][0]["finish_reason"] == "stop"

    def test_none_content_still_emits_role_and_stop(self):
        message = {"role": "assistant", "content": None}
        original_data = {"id": "t", "created": 0, "model": "m"}
        chunks = parse_sse_chunks(
            "".join(stream_response(message, original_data, "m"))
        )
        assert len(chunks) >= 2
        assert chunks[0]["choices"][0]["delta"].get("role") == "assistant"
        assert chunks[-1]["choices"][0]["finish_reason"] == "stop"

    def test_content_chunk_has_index_zero(self):
        message = {"role": "assistant", "content": "test"}
        original_data = {"id": "t", "created": 0, "model": "m"}
        chunks = parse_sse_chunks(
            "".join(stream_response(message, original_data, "m"))
        )
        for chunk in chunks:
            assert chunk["choices"][0]["index"] == 0

    def test_large_content_streamed_as_single_chunk(self):
        large_content = "x" * 10000
        message = {"role": "assistant", "content": large_content}
        original_data = {"id": "t", "created": 0, "model": "m"}
        chunks = parse_sse_chunks(
            "".join(stream_response(message, original_data, "m"))
        )
        content_chunks = [
            c for c in chunks
            if c["choices"][0]["delta"].get("content")
        ]
        assert len(content_chunks) == 1
        assert content_chunks[0]["choices"][0]["delta"]["content"] == large_content


# ============================================================================
# 3. Streaming with Tool Calls
# ============================================================================


class TestStreamingToolCalls:
    """Verify tool calls are streamed correctly with correct structure."""

    def test_single_tool_call_structure(self):
        tool_call = {
            "id": "call_abc",
            "type": "function",
            "function": {"name": "read_file", "arguments": '{"path": "/tmp/x"}'},
        }
        message = {"role": "assistant", "tool_calls": [tool_call]}
        original_data = {"id": "t", "created": 0, "model": "m"}
        chunks = parse_sse_chunks(
            "".join(stream_response(message, original_data, "m"))
        )

        tool_chunks = [
            c for c in chunks
            if "tool_calls" in c["choices"][0].get("delta", {})
        ]
        assert len(tool_chunks) == 1
        tc = tool_chunks[0]["choices"][0]["delta"]["tool_calls"][0]
        assert tc["index"] == 0
        assert tc["id"] == "call_abc"
        assert tc["type"] == "function"
        assert tc["function"]["name"] == "read_file"
        assert tc["function"]["arguments"] == '{"path": "/tmp/x"}'

    def test_multiple_tool_calls_index_sequence(self):
        tool_calls = [
            {
                "id": f"call_{i}",
                "type": "function",
                "function": {"name": f"tool_{i}", "arguments": "{}"},
            }
            for i in range(5)
        ]
        message = {"role": "assistant", "tool_calls": tool_calls}
        original_data = {"id": "t", "created": 0, "model": "m"}
        chunks = parse_sse_chunks(
            "".join(stream_response(message, original_data, "m"))
        )

        tool_chunks = [
            c for c in chunks
            if "tool_calls" in c["choices"][0].get("delta", {})
        ]
        assert len(tool_chunks) == 5

        indices = [
            tc["choices"][0]["delta"]["tool_calls"][0]["index"]
            for tc in tool_chunks
        ]
        assert indices == [0, 1, 2, 3, 4]

    def test_tool_call_finish_reason(self):
        tool_call = {
            "id": "call_1",
            "type": "function",
            "function": {"name": "bash", "arguments": "{}"},
        }
        message = {"role": "assistant", "tool_calls": [tool_call]}
        original_data = {"id": "t", "created": 0, "model": "m"}
        chunks = parse_sse_chunks(
            "".join(stream_response(message, original_data, "m"))
        )
        assert chunks[-1]["choices"][0]["finish_reason"] == "tool_calls"

    def test_tool_call_stream_emits_role_delta_first(self):
        tool_call = {
            "id": "call_1",
            "type": "function",
            "function": {"name": "bash", "arguments": "{}"},
        }
        message = {"role": "assistant", "tool_calls": [tool_call]}
        original_data = {"id": "t", "created": 0, "model": "m"}
        chunks = parse_sse_chunks(
            "".join(stream_response(message, original_data, "m"))
        )
        assert chunks[0]["choices"][0]["delta"].get("role") == "assistant"
        assert chunks[0]["choices"][0]["delta"].get("content") == ""

    def test_tool_call_no_content_chunk(self):
        tool_call = {
            "id": "call_1",
            "type": "function",
            "function": {"name": "bash", "arguments": "{}"},
        }
        message = {"role": "assistant", "tool_calls": [tool_call]}
        original_data = {"id": "t", "created": 0, "model": "m"}
        chunks = parse_sse_chunks(
            "".join(stream_response(message, original_data, "m"))
        )
        content_chunks = [
            c for c in chunks
            if c["choices"][0]["delta"].get("content")
        ]
        assert len(content_chunks) == 0

    def test_tool_call_with_reasoning_before_tools(self):
        tool_call = {
            "id": "call_1",
            "type": "function",
            "function": {"name": "bash", "arguments": '{"cmd": "ls"}'},
        }
        message = {
            "role": "assistant",
            "reasoning_content": "I need to list files",
            "tool_calls": [tool_call],
        }
        original_data = {"id": "t", "created": 0, "model": "m"}
        chunks = parse_sse_chunks(
            "".join(stream_response(message, original_data, "m"))
        )

        reasoning_idxs = [
            i for i, c in enumerate(chunks)
            if c["choices"][0]["delta"].get("reasoning_content")
        ]
        tool_idxs = [
            i for i, c in enumerate(chunks)
            if "tool_calls" in c["choices"][0].get("delta", {})
        ]
        assert len(reasoning_idxs) == 1
        assert len(tool_idxs) == 1
        assert reasoning_idxs[0] < tool_idxs[0]

    def test_tool_call_chunk_count(self):
        """role_delta + N tool_calls + finish = N+2 chunks (no reasoning)."""
        tool_calls = [
            {
                "id": f"call_{i}",
                "type": "function",
                "function": {"name": f"t{i}", "arguments": "{}"},
            }
            for i in range(3)
        ]
        message = {"role": "assistant", "tool_calls": tool_calls}
        original_data = {"id": "t", "created": 0, "model": "m"}
        chunks = parse_sse_chunks(
            "".join(stream_response(message, original_data, "m"))
        )
        assert len(chunks) == 5


# ============================================================================
# 4. MiniMax Reasoning (<think> blocks) in Streaming Context
# ============================================================================


class TestMiniMaxReasoningStreaming:
    """Verify <think> blocks are extracted and streamed correctly for MiniMax."""

    def test_reasoning_content_emitted_as_separate_chunk(self):
        message = {
            "role": "assistant",
            "reasoning_content": "Let me think about this...",
            "content": "The answer is 42.",
        }
        original_data = {"id": "t", "created": 0, "model": "m"}
        chunks = parse_sse_chunks(
            "".join(stream_response(message, original_data, "minimax-m1"))
        )

        reasoning_chunks = [
            c for c in chunks
            if c["choices"][0]["delta"].get("reasoning_content")
        ]
        assert len(reasoning_chunks) == 1
        assert reasoning_chunks[0]["choices"][0]["delta"]["reasoning_content"] == "Let me think about this..."

    def test_reasoning_before_content_ordering(self):
        message = {
            "role": "assistant",
            "reasoning_content": "thinking...",
            "content": "result",
        }
        original_data = {"id": "t", "created": 0, "model": "m"}
        events = parse_sse_events(
            "".join(stream_response(message, original_data, "m"))
        )
        data_events = [e for e in events if e["type"] == "data"]

        reasoning_idx = None
        content_idx = None
        for i, e in enumerate(data_events):
            delta = e["payload"]["choices"][0]["delta"]
            if delta.get("reasoning_content"):
                reasoning_idx = i
            if delta.get("content"):
                content_idx = i

        assert reasoning_idx is not None
        assert content_idx is not None
        assert reasoning_idx < content_idx

    def test_reasoning_field_alias(self):
        message = {
            "role": "assistant",
            "reasoning": "Thinking via reasoning field",
            "content": "Result",
        }
        original_data = {"id": "t", "created": 0, "model": "m"}
        chunks = parse_sse_chunks(
            "".join(stream_response(message, original_data, "m"))
        )
        reasoning_chunks = [
            c for c in chunks
            if c["choices"][0]["delta"].get("reasoning_content") == "Thinking via reasoning field"
        ]
        assert len(reasoning_chunks) == 1

    def test_no_reasoning_chunk_when_absent(self):
        message = {"role": "assistant", "content": "Just content"}
        original_data = {"id": "t", "created": 0, "model": "m"}
        chunks = parse_sse_chunks(
            "".join(stream_response(message, original_data, "m"))
        )
        reasoning_chunks = [
            c for c in chunks
            if "reasoning_content" in c["choices"][0].get("delta", {})
        ]
        assert len(reasoning_chunks) == 0

    def test_empty_reasoning_not_emitted(self):
        message = {
            "role": "assistant",
            "reasoning_content": "",
            "content": "Result",
        }
        original_data = {"id": "t", "created": 0, "model": "m"}
        chunks = parse_sse_chunks(
            "".join(stream_response(message, original_data, "m"))
        )
        reasoning_chunks = [
            c for c in chunks
            if c["choices"][0]["delta"].get("reasoning_content")
        ]
        assert len(reasoning_chunks) == 0

    def test_minimax_think_extraction_via_post_process(self):
        """Verify PostProcessStage._extract_minimax_reasoning extracts <think> blocks."""
        from proxy.pipeline.post_process import PostProcessStage

        stage = PostProcessStage()
        ctx = MagicMock()
        ctx.model_route = MagicMock()
        ctx.model_route.model_cfg = MagicMock()
        ctx.model_route.model_cfg.backend_model = "minimax/MiniMax-M1"
        ctx.response_data = {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "<think>I am reasoning</think>The answer is yes.",
                    }
                }
            ]
        }
        ctx.session_id = "test-session"

        with patch("proxy.utils.extract_text_content", side_effect=lambda x: x):
            with patch("memory.sync._cache_reasoning"):
                stage._extract_minimax_reasoning(ctx)

        msg = ctx.response_data["choices"][0]["message"]
        assert msg["reasoning_content"] == "I am reasoning"
        assert msg["content"] == "The answer is yes."

    def test_minimax_think_not_extracted_for_non_minimax(self):
        """Verify <think> blocks are NOT extracted for non-MiniMax models."""
        from proxy.pipeline.post_process import PostProcessStage

        stage = PostProcessStage()
        ctx = MagicMock()
        ctx.model_route = MagicMock()
        ctx.model_route.model_cfg = MagicMock()
        ctx.model_route.model_cfg.backend_model = "openai/gpt-4"
        ctx.response_data = {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "<think>Should not extract</think>The answer.",
                    }
                }
            ]
        }

        stage._extract_minimax_reasoning(ctx)

        msg = ctx.response_data["choices"][0]["message"]
        assert "reasoning_content" not in msg
        assert msg["content"] == "<think>Should not extract</think>The answer."


# ============================================================================
# 5. Streaming Chunk Assembly / Partial JSON Buffering
# ============================================================================


class TestChunkAssembly:
    """Verify the proxy correctly buffers upstream and re-streams as SSE.

    The proxy calls upstream with stream=False, buffers the full JSON response,
    then re-streams it as SSE chunks via stream_response(). These tests verify
    the re-streaming assembly is correct.
    """

    def test_full_response_assembled_from_chunks(self):
        message = {"role": "assistant", "content": "Hello world, this is a test."}
        original_data = {"id": "chatcmpl-1", "created": 1000, "model": "m"}
        sse = "".join(stream_response(message, original_data, "m"))
        chunks = parse_sse_chunks(sse)

        content_parts = []
        for c in chunks:
            delta = c["choices"][0]["delta"]
            if delta.get("content"):
                content_parts.append(delta["content"])

        assembled = "".join(content_parts)
        assert assembled == "Hello world, this is a test."

    def test_tool_calls_assembled_from_chunks(self):
        tool_calls = [
            {
                "id": "call_1",
                "type": "function",
                "function": {"name": "read", "arguments": '{"f": "a"}'},
            },
            {
                "id": "call_2",
                "type": "function",
                "function": {"name": "write", "arguments": '{"f": "b"}'},
            },
        ]
        message = {"role": "assistant", "tool_calls": tool_calls}
        original_data = {"id": "t", "created": 0, "model": "m"}
        chunks = parse_sse_chunks(
            "".join(stream_response(message, original_data, "m"))
        )

        assembled_tools = {}
        for c in chunks:
            for tc in c["choices"][0]["delta"].get("tool_calls", []):
                idx = tc["index"]
                assembled_tools[idx] = {
                    "id": tc["id"],
                    "name": tc["function"]["name"],
                    "arguments": tc["function"]["arguments"],
                }

        assert len(assembled_tools) == 2
        assert assembled_tools[0]["name"] == "read"
        assert assembled_tools[1]["name"] == "write"

    def test_reasoning_and_content_assembled_separately(self):
        message = {
            "role": "assistant",
            "reasoning_content": "Thinking step by step",
            "content": "Final answer",
        }
        original_data = {"id": "t", "created": 0, "model": "m"}
        chunks = parse_sse_chunks(
            "".join(stream_response(message, original_data, "m"))
        )

        reasoning_parts = []
        content_parts = []
        for c in chunks:
            delta = c["choices"][0]["delta"]
            if delta.get("reasoning_content"):
                reasoning_parts.append(delta["reasoning_content"])
            if delta.get("content"):
                content_parts.append(delta["content"])

        assert "".join(reasoning_parts) == "Thinking step by step"
        assert "".join(content_parts) == "Final answer"

    def test_sse_from_dict_in_app(self):
        """Verify _sse_from_response_data correctly converts response dict to SSE stream."""
        from proxy.app import _sse_from_response_data

        data = {
            "id": "chatcmpl-test",
            "created": 1000,
            "model": "backend-model",
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "Test content",
                    }
                }
            ],
        }
        sse = "".join(_sse_from_response_data(data, "user-model"))
        chunks = parse_sse_chunks(sse)

        assert len(chunks) >= 2
        content_chunks = [
            c for c in chunks if c["choices"][0]["delta"].get("content")
        ]
        assert len(content_chunks) == 1
        assert content_chunks[0]["choices"][0]["delta"]["content"] == "Test content"
        for chunk in chunks:
            assert chunk["model"] == "user-model"


# ============================================================================
# 6. Streaming via Flask Test Client (End-to-End SSE)
# ============================================================================


class TestFlaskStreamingE2E:
    """End-to-end streaming tests through the Flask test client."""

    def test_stream_response_content_type(self, client, mock_llama_response):
        mock_llama_response(
            data={
                "choices": [
                    {
                        "message": {"role": "assistant", "content": "streamed"},
                        "finish_reason": "stop",
                    }
                ]
            }
        )
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "test",
                "messages": [{"role": "user", "content": "hi"}],
                "stream": True,
            },
        )
        assert response.status_code == 200
        assert response.content_type == "text/event-stream"

    def test_stream_response_headers(self, client, mock_llama_response):
        mock_llama_response(
            data={
                "choices": [
                    {
                        "message": {"role": "assistant", "content": "test"},
                        "finish_reason": "stop",
                    }
                ]
            }
        )
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "test",
                "messages": [{"role": "user", "content": "hi"}],
                "stream": True,
            },
        )
        assert response.headers.get("Cache-Control") == "no-cache"
        assert response.headers.get("Connection") == "keep-alive"
        assert response.headers.get("X-Accel-Buffering") == "no"

    def test_stream_response_iter_chunks(self, client, mock_llama_response):
        mock_llama_response(
            data={
                "choices": [
                    {
                        "message": {"role": "assistant", "content": "chunked"},
                        "finish_reason": "stop",
                    }
                ]
            }
        )
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "test",
                "messages": [{"role": "user", "content": "hi"}],
                "stream": True,
            },
        )
        assert response.status_code == 200
        raw_chunks = list(response.response)
        assert len(raw_chunks) > 0

        full_sse = b"".join(raw_chunks).decode("utf-8")
        assert "data:" in full_sse
        assert "[DONE]" in full_sse

    def test_stream_false_returns_json(self, client, mock_llama_response):
        mock_llama_response(
            data={
                "choices": [
                    {
                        "message": {"role": "assistant", "content": "json response"},
                        "finish_reason": "stop",
                    }
                ]
            }
        )
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "test",
                "messages": [{"role": "user", "content": "hi"}],
                "stream": False,
            },
        )
        assert response.status_code == 200
        assert response.content_type == "application/json"

    def test_stream_with_tool_calls_e2e(self, client, mock_llama_response):
        mock_llama_response(
            data={
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "tool_calls": [
                                {
                                    "id": "call_e2e",
                                    "type": "function",
                                    "function": {
                                        "name": "search",
                                        "arguments": '{"q": "test"}',
                                    },
                                }
                            ],
                        },
                        "finish_reason": "tool_calls",
                    }
                ]
            }
        )
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "test",
                "messages": [{"role": "user", "content": "search"}],
                "stream": True,
            },
        )
        assert response.status_code == 200
        assert response.content_type == "text/event-stream"

        full_sse = b"".join(response.response).decode("utf-8")
        events = parse_sse_events(full_sse)
        data_events = [e for e in events if e["type"] == "data"]
        assert len(data_events) >= 2
        assert events[-1]["type"] == "done"

    def test_stream_with_reasoning_e2e(self, client, mock_llama_response):
        mock_llama_response(
            data={
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "reasoning_content": "deep thought",
                            "content": "the answer",
                        },
                        "finish_reason": "stop",
                    }
                ]
            }
        )
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "test",
                "messages": [{"role": "user", "content": "think"}],
                "stream": True,
            },
        )
        assert response.status_code == 200
        full_sse = b"".join(response.response).decode("utf-8")
        assert "deep thought" in full_sse
        assert "the answer" in full_sse


# ============================================================================
# 7. Streaming Error Handling
# ============================================================================


class TestStreamingErrorHandling:
    """Verify error behavior in streaming context."""

    def test_error_response_not_streamed(self, client, mock_llama_response):
        """Backend errors are now streamed as SSE error chunks (200 OK)
        rather than JSON error responses (500/502) when using keepalive streams.

        US-060: Streaming Keepalive."""
        import requests as _requests
        from qwen_proxy import _HTTP_SESSION

        # Build a fake upstream response: status 500, body with the contract
        # message, and raise_for_status that raises HTTPError carrying this
        # response on its `.response` attribute. call_upstream's
        # `except requests.exceptions.HTTPError` branch then extracts
        # `e.response.status_code` / `e.response.json()` and raises
        # BackendError(500, body), which the completion thread's
        # `except BackendError` handler converts to an `upstream_error`
        # SSE chunk — that's the contract this test pins.
        error_response = MagicMock()
        error_response.status_code = 500
        error_response.json.return_value = {"error": "Internal server error"}
        error_response.text = "Internal server error"

        http_error = _requests.exceptions.HTTPError("500 Server Error")
        http_error.response = error_response
        error_response.raise_for_status.side_effect = http_error

        # The streaming completion thread runs asynchronously: `client.post`
        # returns after `: connected` is yielded. To assert the patched 500
        # was actually used, the body MUST be drained (response.get_data)
        # while the `_HTTP_SESSION.post` patch is still active — otherwise
        # the worker sees the autouse `mock_llama_response` default after
        # the patch is removed.
        with patch.object(_HTTP_SESSION, "post", return_value=error_response):
            response = client.post(
                "/v1/chat/completions",
                json={
                    "model": "test",
                    "messages": [{"role": "user", "content": "hi"}],
                    "stream": True,
                },
            )
            assert response.status_code == 200
            data = response.get_data(as_text=True)

        # The patched 500 upstream must reach the client as a sanitized
        # SSE error chunk (upstream_error envelope). Asserting only on
        # generic "error" presence would accept any error-shaped body —
        # pin the actual contract.
        assert '"type": "upstream_error"' in data, (
            f"expected upstream_error envelope from patched 500; got: {data!r}"
        )
        assert "Internal server error" in data, (
            f"expected sanitized upstream message 'Internal server error' in SSE body; "
            f"got: {data!r}"
        )
        assert data.count("data: [DONE]\n\n") == 1, (
            "expected exactly one [DONE] terminator; got "
            + str(data.count("data: [DONE]\n\n"))
            + " in: "
            + repr(data)
        )
        # The autouse mock response (a default OpenAI-style payload) MUST
        # NOT have leaked into the body — that would mean the patch was
        # removed before the worker ran.
        assert "mock response" not in data, (
            f"the autouse mock body leaked into the stream; the patch was "
            f"removed before the completion thread ran. Body: {data!r}"
        )

    def test_empty_choices_handled(self):
        """stream_response handles message with no content and no tools."""
        message = {"role": "assistant"}
        original_data = {"id": "t", "created": 0, "model": "m"}
        sse = "".join(stream_response(message, original_data, "m"))
        assert "data: [DONE]" in sse

    def test_missing_id_generates_default(self):
        message = {"role": "assistant", "content": "test"}
        original_data = {"created": 0, "model": "m"}
        chunks = parse_sse_chunks(
            "".join(stream_response(message, original_data, "m"))
        )
        for chunk in chunks:
            assert "id" in chunk
            assert chunk["id"].startswith("chatcmpl-")

    def test_missing_created_generates_default(self):
        message = {"role": "assistant", "content": "test"}
        original_data = {"id": "t", "model": "m"}
        chunks = parse_sse_chunks(
            "".join(stream_response(message, original_data, "m"))
        )
        for chunk in chunks:
            assert "created" in chunk
            assert isinstance(chunk["created"], int)


# ============================================================================
# 8. Streaming Timeout Behavior
# ============================================================================


class TestStreamingTimeout:
    """Verify timeout behavior when upstream is slow.

    The proxy buffers the full upstream response (stream=False), so timeout
    is enforced at the HTTP client level, not during SSE re-streaming.
    """

    def test_upstream_timeout_config_default(self):
        from proxy.http_client import _resolve_upstream_timeout

        with patch("proxy.http_client.CONFIG", {}):
            timeout = _resolve_upstream_timeout({"messages": []})
        assert timeout == 1800

    def test_upstream_timeout_config_custom(self):
        from proxy.http_client import _resolve_upstream_timeout

        with patch("proxy.http_client.CONFIG", {"upstream": {"timeout": 60}}):
            timeout = _resolve_upstream_timeout({"messages": []})
        assert timeout == 60

    def test_upstream_timeout_scales_for_large_requests(self):
        from proxy.http_client import _resolve_upstream_timeout

        large_body = {
            "messages": [{"role": "user", "content": "x" * 1000} for _ in range(25)]
        }
        with patch(
            "proxy.http_client.CONFIG",
            {"upstream": {"timeout": 60, "timeout_multiplier": 2.0}},
        ):
            timeout = _resolve_upstream_timeout(large_body)
        assert timeout == 120

    def test_upstream_timeout_scales_for_large_body_few_messages(self):
        from proxy.http_client import _resolve_upstream_timeout

        large_body = {
            "messages": [
                {"role": "user", "content": "x" * 25000},
                {"role": "assistant", "content": "y" * 25000},
            ]
        }
        with patch(
            "proxy.http_client.CONFIG",
            {"upstream": {"timeout": 60, "timeout_multiplier": 2.0}},
        ):
            timeout = _resolve_upstream_timeout(large_body)
        assert timeout == 120

    def test_upstream_timeout_cap(self):
        from proxy.http_client import call_upstream

        valid_response = {
            "choices": [{
                "message": {"role": "assistant", "content": "ok"},
                "finish_reason": "stop",
            }]
        }
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = valid_response
        mock_response.raise_for_status = MagicMock()

        with patch("proxy.http_client._HTTP_SESSION") as mock_session:
            mock_session.post.return_value = mock_response
            with patch("proxy.http_client.CONFIG", {"upstream": {"timeout": 300}}):
                with patch("proxy.http_client._resolve_upstream_timeout", return_value=300):
                    call_upstream(
                        "http://test/v1", {}, {}, False, timeout_cap=30
                    )

        call_kwargs = mock_session.post.call_args
        timeout_used = call_kwargs[1].get("timeout") or call_kwargs.kwargs.get("timeout")
        assert timeout_used == (30, 30)

    def test_upstream_retry_on_connection_error(self):
        """# US-038"""
        import requests as req
        from proxy.http_client import call_upstream

        valid_response = {
            "choices": [{
                "message": {"role": "assistant", "content": "ok"},
                "finish_reason": "stop",
            }]
        }

        with patch("proxy.http_client._HTTP_SESSION") as mock_session:
            mock_session.post.side_effect = [
                req.exceptions.ConnectionError("connection reset"),
                MagicMock(
                    status_code=200,
                    json=MagicMock(return_value=valid_response),
                    raise_for_status=MagicMock(),
                ),
            ]
            with patch("proxy.http_client.CONFIG", {
                "upstream": {
                    "timeout": 30,
                    "retry_max_attempts": 3,
                    "retry_base_delay": 0.01,
                    "retry_max_delay": 0.1,
                }
            }):
                result = call_upstream("http://test/v1", {}, {}, False)

        assert result == valid_response
        assert mock_session.post.call_count == 2

    def test_upstream_retry_on_429_with_retry_after_header(self):
        """US-039: call_upstream retries after 429 with Retry-After header.

        Patches _HTTP_SESSION.post to return 429 with Retry-After: 5,
        then 200 on retry. Verifies call_upstream sleeps for the delay
        and retries successfully.
        """
        from proxy.http_client import call_upstream

        mock_429 = MagicMock()
        mock_429.status_code = 429
        mock_429.headers = {"Retry-After": "5"}
        mock_429.json.return_value = {"error": "rate limited"}
        mock_429.raise_for_status = MagicMock()

        mock_200 = MagicMock()
        mock_200.status_code = 200
        mock_200.json.return_value = {"choices": [{"message": {"content": "ok"}}]}
        mock_200.raise_for_status = MagicMock()

        with patch("proxy.http_client._HTTP_SESSION") as mock_session:
            mock_session.post.side_effect = [mock_429, mock_200]
            with patch("proxy.http_client.CONFIG", {
                "upstream": {
                    "timeout": 30,
                    "retry_max_attempts": 3,
                    "retry_base_delay": 0.01,
                    "retry_max_delay": 0.1,
                }
            }):
                with patch("proxy.http_client.time.sleep") as mock_sleep:
                    result = call_upstream("http://test/v1", {}, {}, False)

        assert result == {"choices": [{"message": {"content": "ok"}}]}
        mock_sleep.assert_called_once_with(5)
        assert mock_session.post.call_count == 2
        mock_429.close.assert_called_once_with()
        mock_200.close.assert_called_once_with()


# ============================================================================
# 9. Edge Cases and Boundary Conditions
# ============================================================================


class TestStreamingEdgeCases:
    """Edge cases and boundary conditions for SSE streaming."""

    def test_unicode_content(self):
        message = {"role": "assistant", "content": "Hello 世界 🌍"}
        original_data = {"id": "t", "created": 0, "model": "m"}
        sse = "".join(stream_response(message, original_data, "m"))
        chunks = parse_sse_chunks(sse)
        content_chunks = [
            c for c in chunks if c["choices"][0]["delta"].get("content")
        ]
        assert content_chunks[0]["choices"][0]["delta"]["content"] == "Hello 世界 🌍"

    def test_special_characters_in_content(self):
        message = {"role": "assistant", "content": 'data: {"hack": true}\n\n'}
        original_data = {"id": "t", "created": 0, "model": "m"}
        sse = "".join(stream_response(message, original_data, "m"))
        chunks = parse_sse_chunks(sse)
        content_chunks = [
            c for c in chunks if c["choices"][0]["delta"].get("content")
        ]
        assert content_chunks[0]["choices"][0]["delta"]["content"] == 'data: {"hack": true}\n\n'

    def test_newlines_in_content(self):
        message = {"role": "assistant", "content": "line1\nline2\nline3"}
        original_data = {"id": "t", "created": 0, "model": "m"}
        sse = "".join(stream_response(message, original_data, "m"))
        chunks = parse_sse_chunks(sse)
        content_chunks = [
            c for c in chunks if c["choices"][0]["delta"].get("content")
        ]
        assert content_chunks[0]["choices"][0]["delta"]["content"] == "line1\nline2\nline3"

    def test_tool_call_with_complex_arguments(self):
        complex_args = json.dumps({
            "path": "/tmp/test.txt",
            "content": "multi\nline\ncontent",
            "options": {"encoding": "utf-8", "flag": True},
        })
        tool_call = {
            "id": "call_complex",
            "type": "function",
            "function": {"name": "write_file", "arguments": complex_args},
        }
        message = {"role": "assistant", "tool_calls": [tool_call]}
        original_data = {"id": "t", "created": 0, "model": "m"}
        chunks = parse_sse_chunks(
            "".join(stream_response(message, original_data, "m"))
        )
        tool_chunks = [
            c for c in chunks
            if "tool_calls" in c["choices"][0].get("delta", {})
        ]
        assert tool_chunks[0]["choices"][0]["delta"]["tool_calls"][0]["function"]["arguments"] == complex_args

    def test_concurrent_stream_generators_independent(self):
        msg1 = {"role": "assistant", "content": "stream A"}
        msg2 = {"role": "assistant", "content": "stream B"}
        data = {"id": "t", "created": 0, "model": "m"}

        gen1 = stream_response(msg1, data, "m")
        gen2 = stream_response(msg2, data, "m")

        chunks1 = list(gen1)
        chunks2 = list(gen2)

        assert "stream A" in "".join(chunks1)
        assert "stream B" in "".join(chunks2)
        assert "stream B" not in "".join(chunks1)
        assert "stream A" not in "".join(chunks2)

    def test_make_sse_chunk_with_nested_objects(self):
        data = {
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "call_1",
                                "function": {"name": "test", "arguments": "{}"},
                            }
                        ]
                    },
                }
            ]
        }
        chunk = make_sse_chunk(data)
        parsed = json.loads(chunk[6:-2])
        assert parsed == data

    def test_stream_response_generator_is_lazy(self):
        """Verify stream_response is a generator (lazy evaluation)."""
        message = {"role": "assistant", "content": "test"}
        original_data = {"id": "t", "created": 0, "model": "m"}
        gen = stream_response(message, original_data, "m")
        assert hasattr(gen, "__next__")
        assert hasattr(gen, "__iter__")

        first = next(gen)
        assert first.startswith("data: ")
