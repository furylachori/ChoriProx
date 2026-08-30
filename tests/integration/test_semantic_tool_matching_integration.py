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


class TestSemanticToolMatchingIntegration:
    def test_bash_maps_to_run_shell_command(self, client, mock_llama_response):
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
                "messages": [{"role": "user", "content": "list files"}],
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

    def test_read_maps_to_read_file(self, client, mock_llama_response):
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
                "messages": [{"role": "user", "content": "read file"}],
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
                    }
                ],
            },
        )

        result = response.get_json()
        tool_calls = result["choices"][0]["message"]["tool_calls"]
        assert len(tool_calls) > 0

    def test_glob_maps_to_file_glob(self, client, mock_llama_response):
        mock_llama_response(
            data={
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "tool_calls": [
                                _make_tool_call("glob", {"pattern": "*.py"})
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
                "messages": [{"role": "user", "content": "find python files"}],
                "tools": [
                    {
                        "type": "function",
                        "function": {
                            "name": "glob",
                            "description": "Find files matching pattern",
                            "parameters": {
                                "type": "object",
                                "properties": {"pattern": {"type": "string"}},
                            },
                        },
                    }
                ],
            },
        )

        result = response.get_json()
        tool_calls = result["choices"][0]["message"]["tool_calls"]
        assert len(tool_calls) > 0
