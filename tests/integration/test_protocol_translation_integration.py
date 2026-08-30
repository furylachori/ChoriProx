import pytest
import json


class TestProtocolTranslationIntegration:
    def test_native_tool_calls_pass_through_yaml_parser(self, client, mock_llama_response):
        mock_llama_response(
            data={
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "tool_calls": [
                                {
                                    "id": "call_0",
                                    "type": "function",
                                    "function": {
                                        "name": "read_file",
                                        "arguments": json.dumps(
                                            {"path": "/tmp/test.txt"}
                                        ),
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
                "messages": [
                    {"role": "user", "content": "read the file at /tmp/test.txt"}
                ],
                "tools": [
                    {
                        "type": "function",
                        "function": {
                            "name": "read_file",
                            "description": "Read file contents",
                            "parameters": {
                                "type": "object",
                                "properties": {
                                    "file_path": {
                                        "type": "string",
                                        "description": "File path",
                                    }
                                },
                                "required": ["file_path"],
                            },
                        },
                    }
                ],
            },
        )

        assert response.status_code == 200
        result = response.get_json()
        assert "choices" in result
        choices = result["choices"]
        assert len(choices) > 0

        tool_calls = choices[0].get("message", {}).get("tool_calls", [])
        assert len(tool_calls) > 0
        assert tool_calls[0]["function"]["name"] == "read_file"

        args = json.loads(tool_calls[0]["function"]["arguments"])
        assert args["file_path"] == "/tmp/test.txt"

    def test_native_tool_calls_pass_through_xml_parser(self, client, mock_llama_response):
        mock_llama_response(
            data={
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "tool_calls": [
                                {
                                    "id": "call_0",
                                    "type": "function",
                                    "function": {
                                        "name": "write_file",
                                        "arguments": json.dumps(
                                            {"path": "/tmp/output.txt", "content": ""}
                                        ),
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
                "messages": [{"role": "user", "content": "write file"}],
                "tools": [
                    {
                        "type": "function",
                        "function": {
                            "name": "write_file",
                            "description": "write file",
                            "parameters": {
                                "type": "object",
                                "properties": {
                                    "file_path": {"type": "string"},
                                    "content": {"type": "string"},
                                },
                                "required": ["file_path", "content"],
                            },
                        },
                    }
                ],
            },
        )

        assert response.status_code == 200
        result = response.get_json()
        assert "choices" in result
        choices = result["choices"]
        assert len(choices) > 0

        tool_calls = choices[0].get("message", {}).get("tool_calls", [])
        assert len(tool_calls) > 0
        assert tool_calls[0]["function"]["name"] == "write_file"

        args = json.loads(tool_calls[0]["function"]["arguments"])
        assert args["file_path"] == "/tmp/output.txt"

    def test_native_tool_calls_pass_through_json_parser(self, client, mock_llama_response):
        mock_llama_response(
            data={
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "tool_calls": [
                                {
                                    "id": "call_0",
                                    "type": "function",
                                    "function": {
                                        "name": "read_file",
                                        "arguments": json.dumps(
                                            {"path": "/tmp/test.txt"}
                                        ),
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
                                "properties": {"file_path": {"type": "string"}},
                            },
                        },
                    }
                ],
            },
        )

        assert response.status_code == 200
        result = response.get_json()
        assert "choices" in result
        choices = result["choices"]
        assert len(choices) > 0

        tool_calls = choices[0].get("message", {}).get("tool_calls", [])
        assert len(tool_calls) > 0
        assert tool_calls[0]["function"]["name"] == "read_file"

        args = json.loads(tool_calls[0]["function"]["arguments"])
        assert "file_path" in args
