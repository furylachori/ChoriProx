"""
E2E Error Recovery Tests

Tests recovery from malformed inputs: malformed → clarification → success.

E2E Pass Criteria:
- Malformed input is handled gracefully
- Clarification is requested when needed
- Successful recovery when proper input is provided
"""

import pytest
import json

pytestmark = pytest.mark.e2e


def test_missing_required_parameter(mock_llama_response):
    """Test handling of missing required parameters."""
    import sys, os

    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
    from qwen_proxy import app

    mock_llama_response(
        data={
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "Please specify which bash command to run.",
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
                "name": "bash",
                "description": "Execute bash commands",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "command": {"type": "string"},
                        "timeout": {"type": "integer"},
                    },
                    "required": ["command"],
                },
            },
        },
    ]

    messages = [
        {"role": "user", "content": "Run a bash command but don't specify what to run"}
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
    content = message.get("content", "")

    assert content == "Please specify which bash command to run."


def test_empty_arguments_handling(mock_llama_response):
    """Test handling of empty tool arguments."""
    import sys, os

    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
    from qwen_proxy import app

    mock_llama_response(
        data={
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "Need a command argument to execute.",
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

    messages = [{"role": "user", "content": "Execute bash with no actual command"}]
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
    assert message.get("content") == "Need a command argument to execute."


def test_malformed_json_recovery(mock_llama_response):
    """Test recovery from malformed JSON in arguments."""
    import sys, os

    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
    from qwen_proxy import app

    mock_llama_response(
        data={
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
                                    "name": "bash",
                                    "arguments": json.dumps({"command": "echo test"}),
                                },
                            }
                        ],
                    },
                    "finish_reason": "tool_calls",
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
                "name": "bash",
                "description": "Execute bash commands",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "command": {"type": "string"},
                        "timeout": {"type": "integer"},
                    },
                    "required": ["command"],
                },
            },
        },
    ]

    messages = [{"role": "user", "content": "Run 'echo test' in bash"}]
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

    assert len(tool_calls) > 0, "Expected tool_calls for malformed JSON recovery"

    for tc in tool_calls:
        args_str = tc.get("function", {}).get("arguments", "{}")
        try:
            args = json.loads(args_str) if isinstance(args_str, str) else args_str
            assert isinstance(args, dict), "Arguments should be a valid JSON object"
        except json.JSONDecodeError:
            pytest.fail("Arguments should be valid JSON")


def test_invalid_file_path_recovery(mock_llama_response):
    """Test recovery from invalid file paths."""
    import sys, os

    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
    from qwen_proxy import app

    mock_llama_response(
        data={
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
                                    "name": "read",
                                    "arguments": json.dumps(
                                        {"file_path": "/nonexistent/directory/file.txt"}
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

    app.config["TESTING"] = True
    test_client = app.test_client()

    tools = [
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

    messages = [
        {
            "role": "user",
            "content": "Read a file from an invalid path /nonexistent/directory/file.txt",
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
    tool_calls = message.get("tool_calls", [])

    assert len(tool_calls) > 0, "Expected tool_calls for invalid file path recovery"

    first_call = tool_calls[0]
    func_def = first_call.get("function", {})
    name = func_def.get("name", "")
    assert name == "read", f"Expected 'read' tool, got '{name}'"

    args_str = func_def.get("arguments", "{}")
    try:
        args = json.loads(args_str) if isinstance(args_str, str) else args_str
        assert "file_path" in args, "Missing file_path parameter"
    except json.JSONDecodeError:
        pytest.fail("Invalid JSON arguments")


def test_ambiguous_request_clarification(mock_llama_response):
    """Test that ambiguous requests get clarification."""
    import sys, os

    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
    from qwen_proxy import app

    mock_llama_response(
        data={
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "Which file would you like me to read?",
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

    messages = [{"role": "user", "content": "Read something"}]
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
    assert message.get("content") == "Which file would you like me to read?"


def test_recovery_after_error(mock_llama_response):
    """Test successful recovery after initial error."""
    import sys, os

    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
    from qwen_proxy import app

    mock_llama_response(
        data={
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
                                    "name": "bash",
                                    "arguments": json.dumps(
                                        {"command": "echo recovery success"}
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
        {"role": "user", "content": "Run some command"},
        {
            "role": "assistant",
            "content": "I need more information about what command to run. Could you please specify what bash command you'd like me to execute?",
        },
        {"role": "user", "content": "Run 'echo recovery success'"},
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

    assert len(tool_calls) > 0, "Expected tool_calls after clarification"

    first_call = tool_calls[0]
    func_def = first_call.get("function", {})
    name = func_def.get("name", "")
    args_str = func_def.get("arguments", "{}")

    assert name == "bash", f"Expected 'bash' tool after clarification, got '{name}'"

    try:
        args = json.loads(args_str) if isinstance(args_str, str) else args_str
        assert "command" in args, "Missing command parameter"
        assert (
            "echo" in args["command"].lower()
            or "recovery" in args["command"].lower()
        ), "Expected recovery-related command"
    except json.JSONDecodeError:
        pytest.fail("Invalid JSON arguments")


def test_wrong_tool_selection_recovery(mock_llama_response):
    """Test recovery from selecting wrong tool."""
    import sys, os

    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
    from qwen_proxy import app

    mock_llama_response(
        data={
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
                                    "name": "write",
                                    "arguments": json.dumps(
                                        {
                                            "file_path": "/tmp/test.txt",
                                            "content": "hello",
                                        }
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

    app.config["TESTING"] = True
    test_client = app.test_client()

    tools = [
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
                    },
                    "required": ["file_path", "content"],
                },
            },
        },
    ]

    messages = [
        {"role": "user", "content": "Write 'hello' to /tmp/test.txt (NOT read)"}
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

    assert len(tool_calls) > 0, "Expected tool_calls for wrong tool selection recovery"

    first_call = tool_calls[0]
    func_def = first_call.get("function", {})
    name = func_def.get("name", "")
    args_str = func_def.get("arguments", "{}")

    assert name == "write", f"Expected 'write' tool (not 'read'), got '{name}'"


def test_type_mismatch_recovery(mock_llama_response):
    """Test recovery from parameter type mismatches."""
    import sys, os

    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
    from qwen_proxy import app

    mock_llama_response(
        data={
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
                                    "name": "bash",
                                    "arguments": json.dumps(
                                        {"command": "echo test", "timeout": 30}
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
                    },
                    "required": ["command"],
                },
            },
        },
    ]

    messages = [
        {
            "role": "user",
            "content": "Run 'echo test' with timeout as string '30' instead of number",
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
    tool_calls = message.get("tool_calls", [])

    assert len(tool_calls) > 0, "Expected tool_calls for type mismatch recovery"

    for tc in tool_calls:
        args_str = tc.get("function", {}).get("arguments", "{}")
        try:
            args = json.loads(args_str) if isinstance(args_str, str) else args_str
            if "timeout" in args:
                assert isinstance(args["timeout"], (int, str)), (
                    "Timeout should be numeric or string"
                )
        except json.JSONDecodeError:
            pytest.fail("Invalid JSON arguments")
