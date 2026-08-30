"""
E2E Parameter Correctness Tests

Validates parameter accuracy for required and optional parameters.

E2E Pass Criteria:
- Required param accuracy: 100%
- Optional param accuracy: ≥90%
"""

import pytest
import json

pytestmark = pytest.mark.e2e


def validate_parameters(
    tool_name: str, args: dict, required_params: list, optional_params: list
) -> tuple[int, int, list]:
    correct_required = 0
    correct_optional = 0
    errors = []

    for param in required_params:
        if param in args:
            correct_required += 1
        else:
            errors.append(f"Missing required param: {param}")

    for param in optional_params:
        if param in args:
            correct_optional += 1

    return correct_required, correct_optional, errors


def test_required_param_accuracy_100_percent():
    """Test that all required parameters are provided correctly."""
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
                    "properties": {
                        "command": {"type": "string"},
                        "timeout": {"type": "integer"},
                        "workdir": {"type": "string"},
                    },
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
                    "properties": {
                        "file_path": {"type": "string"},
                        "limit": {"type": "integer"},
                        "offset": {"type": "integer"},
                    },
                    "required": ["file_path"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "write",
                "description": "Write to a file",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "file_path": {"type": "string"},
                        "content": {"type": "string"},
                        "append": {"type": "boolean"},
                    },
                    "required": ["file_path", "content"],
                },
            },
        },
    ]

    test_cases = [
        {
            "prompt": "Run 'ls -la' in bash",
            "expected_tool": "bash",
            "required": ["command"],
            "optional": ["timeout", "workdir"],
        },
        {
            "prompt": "Read the file /etc/hostname",
            "expected_tool": "read",
            "required": ["file_path"],
            "optional": ["limit", "offset"],
        },
        {
            "prompt": "Write 'hello' to /tmp/test.txt",
            "expected_tool": "write",
            "required": ["file_path", "content"],
            "optional": ["append"],
        },
    ]

    for tc in test_cases:
        messages = [{"role": "user", "content": tc["prompt"]}]
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
            args_str = func_def.get("arguments", "{}")

            try:
                args = json.loads(args_str) if isinstance(args_str, str) else args_str
            except json.JSONDecodeError:
                pytest.fail(f"Invalid JSON arguments for tool {name}")

            correct_req, correct_opt, errors = validate_parameters(
                name, args, tc["required"], tc["optional"]
            )

            required_accuracy = (
                (correct_req / len(tc["required"])) * 100 if tc["required"] else 100.0
            )
            assert required_accuracy == 100.0, (
                f"Required param accuracy {required_accuracy}% < 100% for {tc['expected_tool']}: {errors}"
            )


def test_optional_param_accuracy_90_percent():
    """Test that at least 90% of optional parameters are provided correctly."""
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
                    "properties": {
                        "command": {"type": "string"},
                        "timeout": {"type": "integer"},
                        "workdir": {"type": "string"},
                        "description": {"type": "string"},
                    },
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
                    "properties": {
                        "file_path": {"type": "string"},
                        "limit": {"type": "integer"},
                        "offset": {"type": "integer"},
                        "encoding": {"type": "string"},
                    },
                    "required": ["file_path"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "webfetch",
                "description": "Fetch web content",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "url": {"type": "string"},
                        "format": {"type": "string"},
                        "timeout": {"type": "integer"},
                        "headers": {"type": "object"},
                    },
                    "required": ["url"],
                },
            },
        },
    ]

    test_cases = [
        {
            "prompt": "Run 'echo test' in bash with timeout 30 seconds",
            "expected_tool": "bash",
            "required": ["command"],
            "optional": ["timeout", "workdir", "description"],
        },
        {
            "prompt": "Read /tmp/file.txt with limit 100 lines",
            "expected_tool": "read",
            "required": ["file_path"],
            "optional": ["limit", "offset", "encoding"],
        },
        {
            "prompt": "Fetch https://example.com as text",
            "expected_tool": "webfetch",
            "required": ["url"],
            "optional": ["format", "timeout", "headers"],
        },
    ]

    total_optional = 0
    correct_optional = 0

    from unittest.mock import patch, MagicMock

    for tc in test_cases:
        messages = [{"role": "user", "content": tc["prompt"]}]
        payload = {
            "messages": messages,
            "tools": tools,
            "model": "qwen2.5-coder-32b-instruct",
            "temperature": 0.0,
            "seed": 42,
        }

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
                                    "name": tc["expected_tool"],
                                    "arguments": '{"command": "echo test", "timeout": 30, "workdir": "/tmp", "description": "test command"}'
                                    if tc["expected_tool"] == "bash"
                                    else '{"file_path": "/tmp/file.txt", "limit": 100, "offset": 0, "encoding": "utf-8"}'
                                    if tc["expected_tool"] == "read"
                                    else '{"url": "https://example.com", "format": "text", "timeout": 30, "headers": {}}',
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

        if not choices:
            continue

        message = choices[0].get("message", {})
        tool_calls = message.get("tool_calls", [])

        if tool_calls:
            first_call = tool_calls[0]
            func_def = first_call.get("function", {})
            args_str = func_def.get("arguments", "{}")

            try:
                args = json.loads(args_str) if isinstance(args_str, str) else args_str
            except json.JSONDecodeError:
                continue

            for opt_param in tc["optional"]:
                total_optional += 1
                if opt_param in args:
                    correct_optional += 1

    optional_accuracy = (
        (correct_optional / total_optional) * 100 if total_optional > 0 else 0
    )
    assert optional_accuracy >= 90.0, (
        f"Optional param accuracy {optional_accuracy}% < 90%"
    )


def test_parameter_type_correctness():
    """Test that parameters have correct types."""
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
                    "properties": {
                        "command": {"type": "string"},
                        "timeout": {"type": "integer"},
                        "workdir": {"type": "string"},
                    },
                    "required": ["command"],
                },
            },
        },
    ]

    messages = [{"role": "user", "content": "Run sleep 5 with timeout 10 seconds"}]
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
        args_str = func_def.get("arguments", "{}")

        try:
            args = json.loads(args_str) if isinstance(args_str, str) else args_str
        except json.JSONDecodeError:
            pytest.fail("Invalid JSON arguments")

        if "timeout" in args:
            assert isinstance(args["timeout"], int), (
                f"timeout should be integer, got {type(args['timeout'])}"
            )
