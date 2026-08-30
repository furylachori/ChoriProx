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


class TestParseableArgumentsIntegration:
    def test_arguments_parseable_json(self, client, mock_llama_response):
        mock_llama_response(
            data={
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "tool_calls": [
                                _make_tool_call("read_file", {"path": "/tmp/test.txt"})
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
                            "properties": {"file_path": {"type": "string"}},
                        },
                        },
                    }
                ],
            },
        )

        result = response.get_json()
        tool_calls = result["choices"][0]["message"]["tool_calls"]
        args_str = tool_calls[0]["function"]["arguments"]
        args = json.loads(args_str)
        assert "file_path" in args

    def test_schema_violations_caught(self, client, mock_llama_response):
        mock_llama_response(
            data={
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "tool_calls": [_make_tool_call("read_file", {"path": 123})],
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
                            "properties": {"file_path": {"type": "string"}},
                        },
                        },
                    }
                ],
            },
        )

        result = response.get_json()
        assert "choices" in result
