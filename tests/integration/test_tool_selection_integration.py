import pytest
import json


def _make_tool_call(name, arguments, idx=0):
    return {
        "id": f"call_{idx}",
        "type": "function",
        "function": {
            "name": name,
            "arguments": json.dumps(arguments)
            if isinstance(arguments, dict)
            else arguments,
        },
    }


class TestToolSelectionIntegration:
    def test_correct_tool_selected_for_read(self, client, mock_llama_response):
        mock_llama_response(
            data={
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "tool_calls": [
                                _make_tool_call("read", {"path": "/tmp/test.txt"})
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
                "messages": [
                    {"role": "user", "content": "read the file /tmp/test.txt"}
                ],
                "tools": [
                    {
                        "type": "function",
                        "function": {
                            "name": "read",
                            "description": "Read file contents",
                            "parameters": {
                                "type": "object",
                                "properties": {"path": {"type": "string"}},
                            },
                        },
                    },
                    {
                        "type": "function",
                        "function": {
                            "name": "write",
                            "description": "Write file contents",
                            "parameters": {
                                "type": "object",
                                "properties": {
                                    "path": {"type": "string"},
                                    "content": {"type": "string"},
                                },
                            },
                        },
                    },
                ],
            },
        )

        result = response.get_json()
        tool_calls = result["choices"][0]["message"]["tool_calls"]
        assert len(tool_calls) > 0
        assert tool_calls[0]["function"]["name"] == "read"

    def test_correct_tool_selected_for_write(self, client, mock_llama_response):
        mock_llama_response(
            data={
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "tool_calls": [
                                _make_tool_call(
                                    "write",
                                    {
                                        "path": "/tmp/output.txt",
                                        "content": "hello world",
                                    },
                                )
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
                "messages": [
                    {"role": "user", "content": "write hello world to /tmp/output.txt"}
                ],
                "tools": [
                    {
                        "type": "function",
                        "function": {
                            "name": "read",
                            "description": "Read file",
                            "parameters": {
                                "type": "object",
                                "properties": {"path": {"type": "string"}},
                            },
                        },
                    },
                    {
                        "type": "function",
                        "function": {
                            "name": "write",
                            "description": "Write file",
                            "parameters": {
                                "type": "object",
                                "properties": {
                                    "path": {"type": "string"},
                                    "content": {"type": "string"},
                                },
                            },
                        },
                    },
                ],
            },
        )

        result = response.get_json()
        tool_calls = result["choices"][0]["message"]["tool_calls"]
        assert len(tool_calls) > 0
        assert tool_calls[0]["function"]["name"] == "write"

    def test_correct_tool_selected_for_bash(self, client, mock_llama_response):
        mock_llama_response(
            data={
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "tool_calls": [
                                _make_tool_call("bash", {"command": "ls -la"})
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
                "messages": [
                    {"role": "user", "content": "list files in current directory"}
                ],
                "tools": [
                    {
                        "type": "function",
                        "function": {
                            "name": "bash",
                            "description": "Run shell command",
                            "parameters": {
                                "type": "object",
                                "properties": {"command": {"type": "string"}},
                            },
                        },
                    }
                ],
            },
        )

        result = response.get_json()
        tool_calls = result["choices"][0]["message"]["tool_calls"]
        assert len(tool_calls) > 0
        assert tool_calls[0]["function"]["name"] == "bash"
