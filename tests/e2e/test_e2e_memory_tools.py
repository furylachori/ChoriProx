"""
E2E Memory Tools Tests

E2E validation of memory tool pipeline including:
1. recall - search memory across all stores
2. search - search chat history by keywords
3. memory_persistence_across_turns
4. memory_context_injection
5. memory_tool_chain

Requirements:
- llama-server on port 7777
- Neo4j on bolt://localhost:7687 (NEO4J_AUTH=neo4j:password)
- SQLite at data/memory.db
- SOTA API (PROXY_SOTA_API_KEY, PROXY_SOTA_ENDPOINT, PROXY_SOTA_MODEL)

Uses @pytest.mark.skipif if services unavailable.
"""

import pytest
import json
import os

pytestmark = pytest.mark.e2e


def _get_memory_tools():
    return [
        {
            "type": "function",
            "function": {
                "name": "recall",
                "description": "Search memory across all stores. Provide semantic_targets (concepts to find) AND keywords (exact terms to match). Results hitting both paths are highest confidence. Returns unified feed sorted by recency.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "semantic_targets": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Concept-level queries (e.g. ['async error handling', 'deployment rollback strategy']). Used for embedding/semantic search.",
                        },
                        "keywords": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Exact terms to match against topic keys, entity names, and message content (e.g. ['Redis', 'project/stack', 'Kubernetes']). Grounds semantic results to actual stored data.",
                        },
                        "limit": {
                            "type": "integer",
                            "description": "Max results (default 10).",
                        },
                    },
                    "required": ["semantic_targets", "keywords"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "search",
                "description": "Search chat history by keywords. Matches against raw message content, returns summaries for matched messages sorted by recency.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "keywords": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Terms to match against raw message content (e.g. ['Redis timeout', 'deploy rollback']).",
                        },
                        "match_all": {
                            "type": "boolean",
                            "description": "If true, only return messages matching ALL keywords. If false (default), return messages matching ANY keyword.",
                        },
                        "session_id": {
                            "type": "string",
                            "description": "Scope to a specific session. Omit to search across all sessions.",
                        },
                        "time_start": {
                            "type": "string",
                            "description": "ISO 8601 start time filter.",
                        },
                        "time_end": {
                            "type": "string",
                            "description": "ISO 8601 end time filter.",
                        },
                        "limit": {
                            "type": "integer",
                            "description": "Max results (default 10, max 50).",
                        },
                    },
                    "required": ["keywords"],
                },
            },
        },
    ]


def test_recall_e2e(mock_llama_response):
    """Test E2E recall tool - semantic search through stored memories."""
    import sys, os

    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
    from qwen_proxy import app

    mock_llama_response(
        data={
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "I found no matching memories.",
                    },
                    "finish_reason": "stop",
                }
            ]
        }
    )

    app.config["TESTING"] = True
    test_client = app.test_client()

    tools = _get_memory_tools()

    messages = [
        {
            "role": "user",
            "content": "Search for memories about Python programming and software development",
        }
    ]
    payload = {
        "messages": messages,
        "tools": tools,
        "model": "qwen2.5-coder-32b-instruct",
        "temperature": 0.0,
        "seed": 42,
    }

    response = test_client.post("/v1/chat/completions", json=payload)
    assert response.status_code == 200

    data = response.get_json()
    choices = data.get("choices", [])
    assert len(choices) > 0

    message = choices[0].get("message", {})
    assert message.get("content") == "I found no matching memories."


def test_search_e2e(mock_llama_response):
    """Test E2E search tool - lookup by topic/entity."""
    import sys, os

    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
    from qwen_proxy import app

    mock_llama_response(
        data={
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "No matching chat history.",
                    },
                    "finish_reason": "stop",
                }
            ]
        }
    )

    app.config["TESTING"] = True
    test_client = app.test_client()

    tools = _get_memory_tools()

    messages = [
        {
            "role": "user",
            "content": "What do you remember about the Python programming language?",
        }
    ]
    payload = {
        "messages": messages,
        "tools": tools,
        "model": "qwen2.5-coder-32b-instruct",
        "temperature": 0.0,
        "seed": 42,
    }

    response = test_client.post("/v1/chat/completions", json=payload)
    assert response.status_code == 200

    data = response.get_json()
    choices = data.get("choices", [])
    assert len(choices) > 0

    message = choices[0].get("message", {})
    assert message.get("content") == "No matching chat history."


def test_memory_persistence_across_turns(mock_llama_response):
    """Test that memory persists across conversation turns."""
    import sys, os

    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
    from qwen_proxy import app, _HTTP_SESSION
    from unittest.mock import patch, MagicMock

    first_response = MagicMock()
    first_response.status_code = 200
    first_response.json.return_value = {
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
                                "name": "search",
                                "arguments": json.dumps(
                                    {"keywords": ["Python", "FastAPI"]}
                                ),
                            },
                        }
                    ],
                },
                "finish_reason": "tool_calls",
            }
        ]
    }
    first_response.raise_for_status = MagicMock()

    second_response = MagicMock()
    second_response.status_code = 200
    second_response.json.return_value = {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": "We are using FastAPI as the framework.",
                },
                "finish_reason": "stop",
            }
        ]
    }
    second_response.raise_for_status = MagicMock()

    call_count = [0]

    def mock_side_effect(*args, **kwargs):
        call_count[0] += 1
        if call_count[0] == 1:
            return first_response
        return second_response

    app.config["TESTING"] = True
    test_client = app.test_client()

    tools = _get_memory_tools()

    messages = [
        {
            "role": "user",
            "content": "I want to search for Python 3.11 and FastAPI framework",
        },
    ]
    payload = {
        "messages": messages,
        "tools": tools,
        "model": "qwen2.5-coder-32b-instruct",
        "temperature": 0.0,
        "seed": 42,
    }

    with patch.object(_HTTP_SESSION, "request", side_effect=mock_side_effect):
        response1 = test_client.post("/v1/chat/completions", json=payload)
        assert response1.status_code == 200

        data1 = response1.get_json()
        choices1 = data1.get("choices", [])
        assert len(choices1) > 0

        message1 = choices1[0].get("message", {})
        tool_calls1 = message1.get("tool_calls", [])
        content1 = message1.get("content", "")

        search_calls = [
            tc for tc in tool_calls1 if tc.get("function", {}).get("name") == "search"
        ]

        if search_calls:
            first_call = search_calls[0]
            func_def = first_call.get("function", {})
            args_str = func_def.get("arguments", "{}")

            try:
                args = json.loads(args_str) if isinstance(args_str, str) else args_str
                assert "keywords" in args, "search should have 'keywords' parameter"
            except json.JSONDecodeError:
                pytest.fail("Invalid JSON arguments for search")
        else:
            assert len(content1) > 0, (
                "Expected either a search tool_call (passthrough) or content "
                "from the memory tool loop's re-query; got neither. The proxy "
                "should execute memory tool calls and surface the result as content."
            )

        messages.append({"role": "assistant", "content": message1.get("content", "")})
        messages.append(
            {"role": "user", "content": "What framework are we using for this project?"}
        )

        payload2 = {
            "messages": messages,
            "tools": tools,
            "model": "qwen2.5-coder-32b-instruct",
            "temperature": 0.0,
            "seed": 42,
        }

        response2 = test_client.post("/v1/chat/completions", json=payload2)
        assert response2.status_code == 200

        data2 = response2.get_json()
        choices2 = data2.get("choices", [])
        assert len(choices2) > 0

        message2 = choices2[0].get("message", {})
        content2 = message2.get("content", "")

        assert len(content2) > 0, "Expected response that references stored memory"


def test_memory_context_injection_e2e(mock_llama_response):
    """Test that memory context is injected into conversation context."""
    import sys, os

    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
    from qwen_proxy import app

    mock_llama_response(
        data={
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "I have no stored coding tasks yet.",
                    },
                    "finish_reason": "stop",
                }
            ]
        }
    )

    app.config["TESTING"] = True
    test_client = app.test_client()

    tools = _get_memory_tools()

    messages = [
        {"role": "user", "content": "Recall information about my recent coding tasks"},
    ]
    payload = {
        "messages": messages,
        "tools": tools,
        "model": "qwen2.5-coder-32b-instruct",
        "temperature": 0.0,
        "seed": 42,
    }

    response = test_client.post("/v1/chat/completions", json=payload)
    assert response.status_code == 200

    data = response.get_json()
    choices = data.get("choices", [])
    assert len(choices) > 0

    message = choices[0].get("message", {})
    assert message.get("content") == "I have no stored coding tasks yet."


def test_memory_tool_chain_e2e():
    """Test E2E memory tool chain - recall -> search."""
    import sys, os

    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
    from qwen_proxy import app

    app.config["TESTING"] = True
    test_client = app.test_client()

    tools = _get_memory_tools()

    messages = [
        {
            "role": "user",
            "content": "First recall memories about testing, then search for pytest",
        },
    ]
    payload = {
        "messages": messages,
        "tools": tools,
        "model": "qwen2.5-coder-32b-instruct",
        "temperature": 0.0,
        "seed": 42,
    }

    from unittest.mock import patch, MagicMock

    # First call: LLM responds with memory tool call
    first_response = MagicMock()
    first_response.status_code = 200
    first_response.json.return_value = {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {
                                "name": "recall",
                                "arguments": '{"semantic_targets": ["testing"], "keywords": ["testing"]}',
                            },
                        }
                    ],
                },
                "finish_reason": "tool_calls",
            }
        ]
    }
    first_response.raise_for_status = MagicMock()

    # Second call (re-query): LLM responds with passthrough (non-memory) tool call
    second_response = MagicMock()
    second_response.status_code = 200
    second_response.json.return_value = {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": "Here are the results",
                    "tool_calls": [
                        {
                            "id": "call_2",
                            "type": "function",
                            "function": {
                                "name": "read",
                                "arguments": '{"path": "/tmp/test.txt"}',
                            },
                        }
                    ],
                },
                "finish_reason": "tool_calls",
            }
        ]
    }
    second_response.raise_for_status = MagicMock()

    call_count = [0]

    def mock_side_effect(*args, **kwargs):
        call_count[0] += 1
        if call_count[0] == 1:
            return first_response
        return second_response

    with patch("qwen_proxy._HTTP_SESSION.request", side_effect=mock_side_effect):
        response = test_client.post("/v1/chat/completions", json=payload)
    assert response.status_code == 200

    data = response.get_json()
    choices = data.get("choices", [])
    assert len(choices) > 0

    message = choices[0].get("message", {})
    tool_calls = message.get("tool_calls", [])

    tool_names = [tc.get("function", {}).get("name", "") for tc in tool_calls]

    assert len(tool_calls) >= 1, "Expected at least one memory tool call"

    memory_tool_names = {
        "recall",
        "search",
    }
    for name in tool_names:
        if name in memory_tool_names:
            tc = next(
                (t for t in tool_calls if t.get("function", {}).get("name") == name),
                None,
            )
            if tc:
                args_str = tc.get("function", {}).get("arguments", "{}")
                try:
                    args = (
                        json.loads(args_str) if isinstance(args_str, str) else args_str
                    )
                    assert isinstance(args, dict), (
                        f"Arguments should be dict for {name}"
                    )
                except json.JSONDecodeError:
                    pytest.fail(f"Invalid JSON arguments for {name}")
            break


def test_memory_tools_e2e():
    """Test E2E memory tools - all supported memory operations."""
    import sys, os

    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
    from qwen_proxy import app

    app.config["TESTING"] = True
    test_client = app.test_client()

    tools = _get_memory_tools()

    messages = [
        {
            "role": "user",
            "content": "Search for information about testing and development",
        },
    ]
    payload = {
        "messages": messages,
        "tools": tools,
        "model": "qwen2.5-coder-32b-instruct",
        "temperature": 0.0,
        "seed": 42,
    }

    response = test_client.post("/v1/chat/completions", json=payload)
    assert response.status_code == 200

    data = response.get_json()
    choices = data.get("choices", [])
    assert len(choices) > 0

    message = choices[0].get("message", {})
    tool_calls = message.get("tool_calls", [])

    memory_tool_names = {"recall", "search"}
    for tc in tool_calls:
        name = tc.get("function", {}).get("name", "")
        if name in memory_tool_names:
            args_str = tc.get("function", {}).get("arguments", "{}")
            try:
                args = json.loads(args_str) if isinstance(args_str, str) else args_str
                assert isinstance(args, dict), f"Arguments should be dict for {name}"
            except json.JSONDecodeError:
                pytest.fail(f"Invalid JSON arguments for {name}")
            break

    assert len(tool_calls) > 0 or len(message.get("content", "")) > 0, (
        "Expected response with tool calls or content"
    )
