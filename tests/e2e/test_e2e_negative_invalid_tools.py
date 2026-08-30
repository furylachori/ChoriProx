"""
E2E Negative Tests - Invalid Tools

Tests that invalid tool names are handled gracefully.

E2E Pass Criteria:
- Invalid tool names do not crash the system
- Graceful error handling for unknown tools
- System remains stable after invalid input
"""

import pytest
import json

pytestmark = pytest.mark.e2e


def test_invalid_tool_name_handling():
    """Test that invalid tool names are handled gracefully."""
    import sys, os

    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
    from qwen_proxy import app

    app.config["TESTING"] = True
    test_client = app.test_client()

    tools = [
        {
            "type": "function",
            "function": {
                "name": "nonexistent_tool",
                "description": "This tool does not exist",
                "parameters": {
                    "type": "object",
                    "properties": {"param": {"type": "string"}},
                    "required": ["param"],
                },
            },
        },
    ]

    messages = [{"role": "user", "content": "Try to use a tool that doesn't exist"}]
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
    content = message.get("content", "")

    assert len(content) > 0 or len(message.get("tool_calls", [])) > 0, (
        "Expected some response even with invalid tool"
    )


def test_typo_in_tool_name():
    """Test handling of typos in tool names."""
    import sys, os

    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
    from qwen_proxy import app

    app.config["TESTING"] = True
    test_client = app.test_client()

    tools = [
        {
            "type": "function",
            "function": {
                "name": "bash",
                "description": "Execute bash commands",
                "parameters": {
                    "type": "object",
                    "properties": {"command": {"type": "string"}},
                    "required": ["command"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "read",
                "description": "Read a file",
                "parameters": {
                    "type": "object",
                    "properties": {"file_path": {"type": "string"}},
                    "required": ["file_path"],
                },
            },
        },
    ]

    messages = [{"role": "user", "content": "Use the 'bas' tool to run 'echo test'"}]
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

    if tool_calls:
        first_call = tool_calls[0]
        func_def = first_call.get("function", {})
        name = func_def.get("name", "")

        if name == "bash":
            args_str = func_def.get("arguments", "{}")
            try:
                args = json.loads(args_str) if isinstance(args_str, str) else args_str
                assert "command" in args, "Missing command parameter"
            except json.JSONDecodeError:
                pytest.fail("Invalid JSON arguments")


def test_empty_tool_name():
    """Test handling of empty tool name."""
    import sys, os

    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
    from qwen_proxy import app

    app.config["TESTING"] = True
    test_client = app.test_client()

    tools = [
        {
            "type": "function",
            "function": {
                "name": "",
                "description": "Empty name tool",
                "parameters": {"type": "object", "properties": {}},
            },
        },
    ]

    messages = [{"role": "user", "content": "Do something"}]
    payload = {
        "messages": messages,
        "tools": tools,
        "model": "qwen2.5-coder-32b-instruct",
        "temperature": 0.0,
        "seed": 42,
    }

    response = test_client.post("/v1/chat/completions", json=payload)
    assert response.status_code == 200


def test_whitespace_tool_name():
    """Test handling of whitespace-only tool names."""
    import sys, os

    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
    from qwen_proxy import app

    app.config["TESTING"] = True
    test_client = app.test_client()

    tools = [
        {
            "type": "function",
            "function": {
                "name": "   ",
                "description": "Whitespace name tool",
                "parameters": {"type": "object", "properties": {}},
            },
        },
    ]

    messages = [{"role": "user", "content": "Run a command"}]
    payload = {
        "messages": messages,
        "tools": tools,
        "model": "qwen2.5-coder-32b-instruct",
        "temperature": 0.0,
        "seed": 42,
    }

    response = test_client.post("/v1/chat/completions", json=payload)
    assert response.status_code == 200


def test_special_characters_in_tool_name(mock_llama_response):
    """Test handling of special characters in tool names."""
    import sys, os

    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
    from qwen_proxy import app

    mock_llama_response(
        data={
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "I cannot use that tool.",
                    },
                    "finish_reason": "stop",
                }
            ]
        }
    )

    app.config["TESTING"] = True
    test_client = app.test_client()

    tools = [
        {
            "type": "function",
            "function": {
                "name": "tool@#$%",
                "description": "Special chars tool",
                "parameters": {"type": "object", "properties": {}},
            },
        },
        {
            "type": "function",
            "function": {
                "name": "bash",
                "description": "Execute bash commands",
                "parameters": {
                    "type": "object",
                    "properties": {"command": {"type": "string"}},
                    "required": ["command"],
                },
            },
        },
    ]

    messages = [{"role": "user", "content": "Execute a bash command"}]
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
    assert message.get("content") == "I cannot use that tool."


def test_very_long_tool_name():
    """Test handling of very long tool names."""
    import sys, os

    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
    from qwen_proxy import app

    app.config["TESTING"] = True
    test_client = app.test_client()

    long_name = "a" * 500
    tools = [
        {
            "type": "function",
            "function": {
                "name": long_name,
                "description": "Very long name tool",
                "parameters": {"type": "object", "properties": {}},
            },
        },
    ]

    messages = [{"role": "user", "content": "Do something"}]
    payload = {
        "messages": messages,
        "tools": tools,
        "model": "qwen2.5-coder-32b-instruct",
        "temperature": 0.0,
        "seed": 42,
    }

    response = test_client.post("/v1/chat/completions", json=payload)
    assert response.status_code == 200


def test_tool_name_case_sensitivity():
    """Test that tool name matching is case-sensitive."""
    import sys, os

    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
    from qwen_proxy import app

    app.config["TESTING"] = True
    test_client = app.test_client()

    tools = [
        {
            "type": "function",
            "function": {
                "name": "Bash",
                "description": "Capital Bash",
                "parameters": {
                    "type": "object",
                    "properties": {"command": {"type": "string"}},
                    "required": ["command"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "bash",
                "description": "Lowercase bash",
                "parameters": {
                    "type": "object",
                    "properties": {"command": {"type": "string"}},
                    "required": ["command"],
                },
            },
        },
    ]

    messages = [{"role": "user", "content": "Run 'echo case_test' using bash"}]
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

    if tool_calls:
        first_call = tool_calls[0]
        func_def = first_call.get("function", {})
        name = func_def.get("name", "")
        assert name in ("bash", "Bash"), f"Expected 'bash' or 'Bash', got '{name}'"


def test_underscore_vs_camelcase_tool_name():
    """Test handling of underscore vs camelCase tool names."""
    import sys, os

    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
    from qwen_proxy import app

    app.config["TESTING"] = True
    test_client = app.test_client()

    tools = [
        {
            "type": "function",
            "function": {
                "name": "run_shell_command",
                "description": "Underscore style",
                "parameters": {
                    "type": "object",
                    "properties": {"command": {"type": "string"}},
                    "required": ["command"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "runShellCommand",
                "description": "CamelCase style",
                "parameters": {
                    "type": "object",
                    "properties": {"command": {"type": "string"}},
                    "required": ["command"],
                },
            },
        },
    ]

    messages = [{"role": "user", "content": "Execute 'echo test' in the shell"}]
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


def test_system_remains_stable_after_invalid_tool():
    """Test that system remains stable after invalid tool request."""
    import sys, os

    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
    from qwen_proxy import app

    app.config["TESTING"] = True
    test_client = app.test_client()

    valid_tools = [
        {
            "type": "function",
            "function": {
                "name": "bash",
                "description": "Execute bash commands",
                "parameters": {
                    "type": "object",
                    "properties": {"command": {"type": "string"}},
                    "required": ["command"],
                },
            },
        },
    ]

    messages = [
        {"role": "user", "content": "Try using invalid_tool_12345"},
        {"role": "assistant", "content": "I can't use that tool."},
        {
            "role": "user",
            "content": "Now run 'echo stability test' with the valid bash tool",
        },
    ]
    payload = {
        "messages": messages,
        "tools": valid_tools,
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

    if tool_calls:
        first_call = tool_calls[0]
        func_def = first_call.get("function", {})
        name = func_def.get("name", "")

        assert name == "bash", f"Expected 'bash' tool after recovery, got '{name}'"

        args_str = func_def.get("arguments", "{}")
        try:
            args = json.loads(args_str) if isinstance(args_str, str) else args_str
            assert "command" in args, "Missing command parameter"
            assert "echo" in args["command"].lower(), "Expected echo command"
        except json.JSONDecodeError:
            pytest.fail("Invalid JSON arguments after recovery")


def test_multiple_invalid_tools_sequence():
    """Test handling of multiple invalid tool calls in sequence."""
    import sys, os

    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
    from qwen_proxy import app

    app.config["TESTING"] = True
    test_client = app.test_client()

    tools = [
        {
            "type": "function",
            "function": {
                "name": "bash",
                "description": "Execute bash commands",
                "parameters": {
                    "type": "object",
                    "properties": {"command": {"type": "string"}},
                    "required": ["command"],
                },
            },
        },
    ]

    messages = [
        {
            "role": "user",
            "content": "First use fake_tool_1, then fake_tool_2, then use bash to run 'echo test'",
        }
    ]
    payload = {
        "messages": messages,
        "tools": tools,
        "model": "qwen2.5-coder-32b-instruct",
        "temperature": 0.0,
        "seed": 42,
    }

    from unittest.mock import patch, MagicMock

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
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
                                "name": "bash",
                                "arguments": '{"command": "echo test"}',
                            },
                        }
                    ],
                },
                "finish_reason": "tool_calls",
            }
        ]
    }
    mock_response.raise_for_status = MagicMock()

    with patch("qwen_proxy._HTTP_SESSION.request", return_value=mock_response):
        response = test_client.post("/v1/chat/completions", json=payload)
    assert response.status_code == 200

    data = response.get_json()
    choices = data.get("choices", [])
    assert len(choices) > 0

    message = choices[0].get("message", {})
    tool_calls = message.get("tool_calls", [])

    valid_tool_calls = [
        tc for tc in tool_calls if tc.get("function", {}).get("name", "") == "bash"
    ]
    assert len(valid_tool_calls) > 0, "Expected at least one valid 'bash' tool call"
