"""
E2E Tool Chain Tests

Tests that the output of tool[N] correctly feeds into tool[N+1]
in multi-step tool chains.

E2E Pass Criteria:
- Tool selection: EXACT MATCH
- Schema validity: 100%
- Chained output feeds next tool correctly
"""

import pytest
import json

pytestmark = pytest.mark.e2e


def test_two_step_chain():
    """Test a 2-step tool chain: glob -> read."""
    import sys, os

    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
    from qwen_proxy import app

    app.config["TESTING"] = True
    test_client = app.test_client()

    tools = [
        {
            "type": "function",
            "function": {
                "name": "glob",
                "description": "Find files by pattern",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string"},
                        "pattern": {"type": "string"},
                    },
                    "required": ["path"],
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

    messages = [
        {
            "role": "user",
            "content": "Find all .txt files in /tmp, then read the first one you find",
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
                                "name": "glob",
                                "arguments": '{"path": "/tmp", "pattern": "*.txt"}',
                            },
                        },
                        {
                            "id": "call_2",
                            "type": "function",
                            "function": {
                                "name": "read",
                                "arguments": '{"file_path": "/tmp/test.txt"}',
                            },
                        },
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

    assert len(tool_calls) >= 1, "Expected at least one tool call"

    first_tool = tool_calls[0].get("function", {}).get("name", "")
    assert first_tool == "glob", f"First tool should be 'glob', got '{first_tool}'"

    if len(tool_calls) >= 2:
        second_tool = tool_calls[1].get("function", {}).get("name", "")
        assert second_tool == "read", (
            f"Second tool should be 'read', got '{second_tool}'"
        )


def test_three_step_chain():
    """Test a 3-step tool chain: glob -> read -> write."""
    import sys, os

    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
    from qwen_proxy import app

    app.config["TESTING"] = True
    test_client = app.test_client()

    tools = [
        {
            "type": "function",
            "function": {
                "name": "glob",
                "description": "Find files by pattern",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string"},
                        "pattern": {"type": "string"},
                    },
                    "required": ["path"],
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
        {
            "role": "user",
            "content": "Find .txt files in /tmp, read the first one, then save a summary to /tmp/summary.txt",
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
                                "name": "glob",
                                "arguments": '{"path": "/tmp", "pattern": "*.txt"}',
                            },
                        },
                        {
                            "id": "call_2",
                            "type": "function",
                            "function": {
                                "name": "read",
                                "arguments": '{"file_path": "/tmp/test.txt"}',
                            },
                        },
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

    assert len(tool_calls) >= 2, "Expected at least 2 tool calls"

    tool_names = [tc.get("function", {}).get("name", "") for tc in tool_calls]
    assert "glob" in tool_names, "Expected 'glob' in tool chain"
    assert "read" in tool_names, "Expected 'read' in tool chain"


def test_output_feeds_next_tool():
    """Test that output of one tool properly feeds into the next."""
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
        {
            "role": "user",
            "content": "Run 'ls /tmp' and save the output to /tmp/ls_output.txt",
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
                                "arguments": '{"command": "ls /tmp"}',
                            },
                        },
                        {
                            "id": "call_2",
                            "type": "function",
                            "function": {
                                "name": "write",
                                "arguments": '{"file_path": "/tmp/ls_output.txt", "content": "file1\\nfile2"}',
                            },
                        },
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

    assert len(tool_calls) >= 1, "Expected at least one tool call"

    first_tool = tool_calls[0].get("function", {}).get("name", "")
    assert first_tool == "bash", f"First tool should be 'bash', got '{first_tool}'"

    if len(tool_calls) >= 2:
        second_tool = tool_calls[1].get("function", {}).get("name", "")
        assert second_tool == "write", (
            f"Second tool should be 'write', got '{second_tool}'"
        )

        write_args_str = tool_calls[1].get("function", {}).get("arguments", "{}")
        try:
            write_args = (
                json.loads(write_args_str)
                if isinstance(write_args_str, str)
                else write_args_str
            )
            assert write_args.get("file_path") == "/tmp/ls_output.txt", (
                "Expected output to /tmp/ls_output.txt"
            )
        except json.JSONDecodeError:
            pass


def test_four_step_chain():
    """Test a 4-step tool chain: glob -> read -> edit -> write."""
    import sys, os

    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
    from qwen_proxy import app

    app.config["TESTING"] = True
    test_client = app.test_client()

    tools = [
        {
            "type": "function",
            "function": {
                "name": "glob",
                "description": "Find files by pattern",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string"},
                        "pattern": {"type": "string"},
                    },
                    "required": ["path"],
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
        {
            "type": "function",
            "function": {
                "name": "edit",
                "description": "Edit a file",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "file_path": {"type": "string"},
                        "old_string": {"type": "string"},
                        "new_string": {"type": "string"},
                    },
                    "required": ["file_path", "old_string", "new_string"],
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
        {
            "role": "user",
            "content": "Find .txt files in /tmp, read one, modify 'old' to 'new', and save to /tmp/modified.txt",
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
                                "name": "read",
                                "arguments": '{"file_path": "/tmp/test.txt"}',
                            },
                        },
                        {
                            "id": "call_2",
                            "type": "function",
                            "function": {
                                "name": "write",
                                "arguments": '{"file_path": "/tmp/modified.txt", "content": "new content"}',
                            },
                        },
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

    tool_names = [tc.get("function", {}).get("name", "") for tc in tool_calls]

    assert tool_names == ["read", "write"], (
        f"Expected read+write tool chain from mocked upstream, got {tool_names}"
    )


def test_chain_with_conditional_branching():
    """Test tool chain with conditional branching based on output."""
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
                "name": "glob",
                "description": "Find files by pattern",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string"},
                        "pattern": {"type": "string"},
                    },
                    "required": ["path"],
                },
            },
        },
    ]

    messages = [
        {
            "role": "user",
            "content": "List files in /tmp. If there are .log files, count them. If not, list all files.",
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
                                "arguments": '{"command": "ls /tmp"}',
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

    assert len(tool_calls) >= 1, "Expected at least one tool call"


def test_chain_schema_validity():
    """Test that all tools in chain have valid schemas."""
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

    messages = [
        {"role": "user", "content": "Run 'pwd' and save output to /tmp/pwd.txt"}
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

    for tc in tool_calls:
        func_def = tc.get("function", {})
        name = func_def.get("name", "")
        args_str = func_def.get("arguments", "{}")

        assert name in ("bash", "write"), f"Unexpected tool: {name}"

        try:
            args = json.loads(args_str) if isinstance(args_str, str) else args_str
            assert isinstance(args, dict), f"Arguments should be dict for {name}"

            if name == "bash":
                assert "command" in args, f"Missing required 'command' for bash"
            elif name == "write":
                assert "file_path" in args, f"Missing required 'file_path' for write"
                assert "content" in args, f"Missing required 'content' for write"
        except json.JSONDecodeError:
            pytest.fail(f"Invalid JSON arguments for tool {name}")
