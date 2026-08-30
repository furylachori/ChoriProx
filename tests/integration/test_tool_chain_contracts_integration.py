import pytest
import json


def _make_tool_call(name, arguments, idx=0):
    return {
        "id": f"call_{idx}",
        "type": "function",
        "function": {
            "name": name,
            "arguments": json.dumps({"name": name, "arguments": arguments})
            if isinstance(arguments, dict)
            else arguments,
        },
    }


class TestToolChainContractsIntegration:
    def test_output_feeds_next_input(self, client, mock_llama_response):
        mock_llama_response(
            data={
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "tool_calls": [
                                _make_tool_call(
                                    "read_file", {"path": "/tmp/output.txt"}
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
                    {"role": "user", "content": "read the output file"},
                ],
                "tools": [
                    {
                        "type": "function",
                        "function": {
                            "name": "read_file",
                            "description": "Read file",
                            "parameters": {
                                "type": "object",
                                "properties": {"path": {"type": "string"}},
                                "required": ["path"],
                            },
                        },
                    },
                    {
                        "type": "function",
                        "function": {
                            "name": "edit_file",
                            "description": "Edit file",
                            "parameters": {
                                "type": "object",
                                "properties": {
                                    "path": {"type": "string"},
                                    "old_string": {"type": "string"},
                                    "new_string": {"type": "string"},
                                },
                                "required": ["path", "old_string", "new_string"],
                            },
                        },
                    },
                ],
            },
        )

        result = response.get_json()
        tool_calls = result["choices"][0]["message"]["tool_calls"]
        assert len(tool_calls) > 0
        assert tool_calls[0]["function"]["name"] == "read_file"
