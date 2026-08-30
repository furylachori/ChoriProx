import pytest
import json


class TestFinishReasonConsistencyIntegration:
    def test_tool_calls_with_tool_calls_reason(self, client, mock_llama_response):
        mock_llama_response(
            data={
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "tool_calls": [
                                {
                                    "id": "call_1",
                                    "type": "function",
                                    "function": {
                                        "name": "read_file",
                                        "arguments": '{"path": "/tmp/test.txt"}',
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
                "model": "qwen3-coder",
                "messages": [{"role": "user", "content": "read file"}],
                "tools": [
                    {
                        "type": "function",
                        "function": {
                            "name": "read_file",
                            "description": "Read file",
                            "parameters": {
                                "type": "object",
                                "properties": {"path": {"type": "string"}},
                            },
                        },
                    }
                ],
            },
        )

        result = response.get_json()
        choices = result["choices"]
        assert len(choices) > 0
        assert choices[0].get("finish_reason") == "tool_calls"

    def test_content_with_stop_reason(self, client, mock_llama_response):
        mock_llama_response(
            data={
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": "Hello world",
                        },
                        "finish_reason": "stop",
                    }
                ]
            }
        )

        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "qwen3-coder",
                "messages": [{"role": "user", "content": "say hello"}],
            },
        )

        result = response.get_json()
        choices = result["choices"]
        assert len(choices) > 0
        assert choices[0].get("finish_reason") == "stop"

    def test_mismatch_detected(self, client, mock_llama_response):
        mock_llama_response(
            data={
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": "content but wrong reason",
                            "tool_calls": [
                                {
                                    "id": "call_1",
                                    "type": "function",
                                    "function": {
                                        "name": "read_file",
                                        "arguments": '{"path": "/tmp/test.txt"}',
                                    },
                                }
                            ],
                        },
                        "finish_reason": "content",
                    }
                ]
            }
        )

        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "qwen3-coder",
                "messages": [{"role": "user", "content": "test"}],
                "tools": [
                    {
                        "type": "function",
                        "function": {
                            "name": "read_file",
                            "description": "Read file",
                            "parameters": {
                                "type": "object",
                                "properties": {"path": {"type": "string"}},
                            },
                        },
                    }
                ],
            },
        )

        result = response.get_json()
        choices = result["choices"]
        has_tool_calls = len(choices[0].get("message", {}).get("tool_calls", [])) > 0
        has_mismatch = choices[0].get("finish_reason") != "tool_calls"
        assert has_tool_calls and choices[0].get("finish_reason") == "tool_calls"

    def test_length_with_tool_calls_preserved(self, client, mock_llama_response):
        """Upstream finish_reason='length' + tool_calls → preserve 'length' (truncation)."""
        mock_llama_response(
            data={
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": "call_1",
                                    "type": "function",
                                    "function": {
                                        "name": "read_file",
                                        "arguments": '{"path": "/tmp/test.txt"}',
                                    },
                                }
                            ],
                        },
                        "finish_reason": "length",
                    }
                ]
            }
        )

        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "qwen3-coder",
                "messages": [{"role": "user", "content": "read"}],
                "tools": [
                    {
                        "type": "function",
                        "function": {
                            "name": "read_file",
                            "parameters": {
                                "type": "object",
                                "properties": {"path": {"type": "string"}},
                            },
                        },
                    }
                ],
            },
        )

        choices = response.get_json()["choices"]
        assert choices[0].get("finish_reason") == "length", (
            "truncation signal must NOT be overridden to 'tool_calls' — "
            "the client uses it to decide whether to continue"
        )

    def test_content_filter_with_tool_calls_preserved(self, client, mock_llama_response):
        """Upstream finish_reason='content_filter' + tool_calls → preserve 'content_filter'."""
        mock_llama_response(
            data={
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": "refused",
                            "tool_calls": [
                                {
                                    "id": "call_1",
                                    "type": "function",
                                    "function": {
                                        "name": "read_file",
                                        "arguments": '{"path": "/tmp/test.txt"}',
                                    },
                                }
                            ],
                        },
                        "finish_reason": "content_filter",
                    }
                ]
            }
        )

        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "qwen3-coder",
                "messages": [{"role": "user", "content": "read"}],
                "tools": [
                    {
                        "type": "function",
                        "function": {
                            "name": "read_file",
                            "parameters": {
                                "type": "object",
                                "properties": {"path": {"type": "string"}},
                            },
                        },
                    }
                ],
            },
        )

        choices = response.get_json()["choices"]
        assert choices[0].get("finish_reason") == "content_filter", (
            "content_filter refusal signal must be preserved when tool_calls present"
        )

    def test_stop_with_tool_calls_overrides_to_tool_calls(self, client, mock_llama_response):
        """Upstream finish_reason='stop' + tool_calls → 'tool_calls' (the spec happy path)."""
        mock_llama_response(
            data={
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": "call_1",
                                    "type": "function",
                                    "function": {
                                        "name": "read_file",
                                        "arguments": '{"path": "/tmp/test.txt"}',
                                    },
                                }
                            ],
                        },
                        "finish_reason": "stop",
                    }
                ]
            }
        )

        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "qwen3-coder",
                "messages": [{"role": "user", "content": "read"}],
                "tools": [
                    {
                        "type": "function",
                        "function": {
                            "name": "read_file",
                            "parameters": {
                                "type": "object",
                                "properties": {"path": {"type": "string"}},
                            },
                        },
                    }
                ],
            },
        )

        choices = response.get_json()["choices"]
        assert choices[0].get("finish_reason") == "tool_calls"
