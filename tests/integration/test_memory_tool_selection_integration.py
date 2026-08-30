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


class TestMemoryToolSelectionIntegration:
    def test_recall_selected_when_semantic_targets_present(
        self, client, mock_llama_response
    ):
        mock_llama_response(
            data={
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "tool_calls": [
                                _make_tool_call(
                                    "recall",
                                    {"semantic_targets": ["previous work on project"], "keywords": ["project"]},
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
                    {"role": "user", "content": "what was I working on last time?"}
                ],
                "tools": [
                    {
                        "type": "function",
                        "function": {
                            "name": "recall",
                            "description": "Search memory",
                            "parameters": {
                                "type": "object",
                                "properties": {
                                    "semantic_targets": {"type": "array"},
                                    "keywords": {"type": "array"}
                                },
                                "required": ["semantic_targets", "keywords"],
                            },
                        },
                    }
                ],
            },
        )

        result = response.get_json()
        message = result["choices"][0]["message"]
        assert "tool_calls" not in message, (
            "recall must be intercepted by the proxy's tool pipeline, not "
            "returned to the client as raw tool_calls; "
            f"got message={message!r}"
        )
        assert "content" in message, (
            f"after recall interception, response must have a content field "
            f"carrying the recall result; got message keys: {sorted(message.keys())}"
        )
        assert result["choices"][0]["finish_reason"] in ("stop", "tool_calls")

    def test_search_selected_when_keywords_present(
        self, client, mock_llama_response
    ):
        mock_llama_response(
            data={
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "tool_calls": [
                                _make_tool_call(
                                    "search", {"keywords": ["project architecture"]}
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
                    {
                        "role": "user",
                        "content": "search for project architecture",
                    }
                ],
                "tools": [
                    {
                        "type": "function",
                        "function": {
                            "name": "search",
                            "description": "Search chat history",
                            "parameters": {
                                "type": "object",
                                "properties": {"keywords": {"type": "array"}},
                                "required": ["keywords"],
                            },
                        },
                    }
                ],
            },
        )

        result = response.get_json()
        message = result["choices"][0]["message"]
        assert "tool_calls" not in message, (
            "search must be intercepted by the proxy's tool pipeline, not "
            "returned to the client as raw tool_calls; "
            f"got message={message!r}"
        )
        assert "content" in message, (
            f"after search interception, response must have a content field "
            f"carrying the search result; got message keys: {sorted(message.keys())}"
        )
        assert result["choices"][0]["finish_reason"] in ("stop", "tool_calls")

    def test_memory_tools_intercepted_before_kilocode(
        self, client, mock_llama_response
    ):
        mock_llama_response(
            data={
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "tool_calls": [
                                _make_tool_call(
                                    "recall", {"semantic_targets": ["test query"], "keywords": ["test"]}
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
                "messages": [{"role": "user", "content": "query memory"}],
                "tools": [
                    {
                        "type": "function",
                        "function": {
                            "name": "recall",
                            "description": "Memory recall",
                            "parameters": {
                                "type": "object",
                                "properties": {"semantic_targets": {"type": "array"}},
                            },
                        },
                    }
                ],
            },
        )

        result = response.get_json()
        message = result["choices"][0]["message"]
        assert "tool_calls" not in message, (
            "memory tool (recall) must be intercepted before reaching "
            "kilocode/opencode clients, not returned as raw tool_calls; "
            f"got message={message!r}"
        )
        assert "content" in message, (
            f"after memory tool interception, response must have a content "
            f"field carrying the recall result; got message keys: {sorted(message.keys())}"
        )
        assert result["choices"][0]["finish_reason"] in ("stop", "tool_calls")

    def test_non_memory_tools_passthrough(self, client, mock_llama_response):
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
                            "name": "read",
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
        tool_calls = result["choices"][0]["message"]["tool_calls"]
        assert tool_calls[0]["function"]["name"] != "recall"
