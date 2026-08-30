import pytest
from qwen_proxy import validate_tool_schema


class TestNegativeInvalidSchemasIntegration:
    def test_malformed_json_arguments_caught(self, client, mock_llama_response):
        bad_json_content = '{"name": "read_file", "arguments": {invalid}}'
        mock_llama_response(
            data={
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": bad_json_content,
                        }
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

        assert response.status_code == 200, (
            f"proxy must tolerate malformed JSON in LLM content without crashing; "
            f"got status {response.status_code} body={response.get_data(as_text=True)[:200]!r}"
        )
        result = response.get_json()
        assert "choices" in result, (
            f"response must conform to OpenAI shape (have 'choices'); got keys={list(result.keys())}"
        )
        assert "error" not in result, (
            f"malformed LLM content must not be promoted to a proxy-level error; got {result!r}"
        )
        choice = result["choices"][0]
        assert choice["message"]["role"] == "assistant"
        assert "content" in choice["message"] or "tool_calls" in choice["message"]

    def test_missing_properties_caught(self, client, mock_llama_response):
        tools_invalid = [
            {
                "type": "function",
                "function": {
                    "name": "read_file",
                    "description": "Read file",
                    "parameters": {
                        "type": "object",
                    },
                },
            }
        ]

        is_valid = validate_tool_schema(tools_invalid)
        assert is_valid is False
