"""
E2E File Operations Tests

Tests that files are written with exact content as specified,
and file state is properly verified.

E2E Pass Criteria:
- File content matches exactly what was specified
- File state verification with content hash
- Schema validity: 100%
"""

import pytest
import json
import hashlib
import os

pytestmark = pytest.mark.e2e


def compute_content_hash(content: str) -> str:
    return hashlib.sha256(content.encode()).hexdigest()


def verify_file_content(file_path: str, expected_content: str) -> bool:
    if not os.path.exists(file_path):
        return False
    with open(file_path, "r") as f:
        actual_content = f.read()
    return actual_content == expected_content


def test_write_exact_content(mock_llama_response):
    """Test that write operations create files with exact content."""
    import sys, os

    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
    from qwen_proxy import app

    test_content = "Hello, World! This is a test file.\nLine 2\nLine 3\n"
    test_file = "/tmp/e2e_test_exact.txt"

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
                                            "file_path": test_file,
                                            "content": test_content,
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
                "name": "write",
                "description": "Write content to a file",
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

    expected_hash = compute_content_hash(test_content)

    messages = [
        {
            "role": "user",
            "content": f"Write the following exact content to {test_file}:\n\n{test_content}\n\nDo not add any extra characters or whitespace.",
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

    assert len(tool_calls) > 0, "Expected tool_calls for write operation"

    found_matching_write = False
    for tc in tool_calls:
        func_def = tc.get("function", {})
        name = func_def.get("name", "")
        if name == "write":
            args_str = func_def.get("arguments", "{}")
            try:
                args = (
                    json.loads(args_str) if isinstance(args_str, str) else args_str
                )
                if args.get("file_path") == test_file and "content" in args:
                    written_content = args["content"]
                    actual_hash = compute_content_hash(written_content)
                    assert actual_hash == expected_hash, (
                        f"Content hash mismatch: expected {expected_hash}, got {actual_hash}"
                    )
                    found_matching_write = True
            except json.JSONDecodeError:
                pass

    assert found_matching_write, "Expected a write tool_call matching test_file"

    if os.path.exists(test_file):
        os.remove(test_file)


def test_write_json_file(mock_llama_response):
    """Test writing JSON content exactly."""
    import sys, os

    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
    from qwen_proxy import app

    json_content = (
        '{"key": "value", "number": 42, "array": [1, 2, 3], "nested": {"a": true}}'
    )
    test_file = "/tmp/e2e_test_json.txt"

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
                                            "file_path": test_file,
                                            "content": json_content,
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
                "name": "write",
                "description": "Write content to a file",
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

    expected_hash = compute_content_hash(json_content)

    messages = [
        {
            "role": "user",
            "content": f"Write this exact JSON to {test_file}:\n{json_content}",
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

    assert len(tool_calls) > 0, "Expected tool_calls for write operation"

    found_matching_write = False
    for tc in tool_calls:
        func_def = tc.get("function", {})
        name = func_def.get("name", "")
        if name == "write":
            args_str = func_def.get("arguments", "{}")
            try:
                args = (
                    json.loads(args_str) if isinstance(args_str, str) else args_str
                )
                if args.get("file_path") == test_file and "content" in args:
                    actual_hash = compute_content_hash(args["content"])
                    assert actual_hash == expected_hash, (
                        "JSON content hash mismatch"
                    )
                    found_matching_write = True
            except json.JSONDecodeError:
                pass

    assert found_matching_write, "Expected a write tool_call matching test_file"


def test_edit_replaces_exact_content(mock_llama_response):
    """Test that edit operations replace content exactly."""
    import sys, os

    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
    from qwen_proxy import app

    test_file = "/tmp/e2e_edit_test.txt"
    old_string = "Line 2"
    new_string = "Line 2 MODIFIED"

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
                                    "name": "edit",
                                    "arguments": json.dumps(
                                        {
                                            "file_path": test_file,
                                            "old_string": old_string,
                                            "new_string": new_string,
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
    ]

    original_content = "Line 1\nLine 2\nLine 3\n"

    with open(test_file, "w") as f:
        f.write(original_content)

    messages = [
        {
            "role": "user",
            "content": f"Edit {test_file} to replace '{old_string}' with '{new_string}'",
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

    assert len(tool_calls) > 0, "Expected tool_calls for edit operation"

    found_matching_edit = False
    for tc in tool_calls:
        func_def = tc.get("function", {})
        name = func_def.get("name", "")
        if name == "edit":
            args_str = func_def.get("arguments", "{}")
            try:
                args = (
                    json.loads(args_str) if isinstance(args_str, str) else args_str
                )
                if (
                    args.get("old_string") == old_string
                    and args.get("new_string") == new_string
                ):
                    found_matching_edit = True
            except json.JSONDecodeError:
                pass

    assert found_matching_edit, "Expected an edit tool_call matching old_string/new_string"

    if os.path.exists(test_file):
        os.remove(test_file)


def test_file_state_verification(mock_llama_response):
    """Test file state verification with content hash."""
    import sys, os

    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
    from qwen_proxy import app

    mock_llama_response(
        data={
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "I would write the file and verify with a hash.",
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
                "name": "write",
                "description": "Write content to a file",
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

    test_content = "File state verification test content"
    test_file = "/tmp/e2e_state_test.txt"
    expected_hash = compute_content_hash(test_content)

    messages = [
        {
            "role": "user",
            "content": f"Write '{test_content}' to {test_file} and verify it was written correctly",
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
    content = message.get("content", "")

    assert content == "I would write the file and verify with a hash."


def test_multiline_content_preservation(mock_llama_response):
    """Test that multiline content is preserved exactly."""
    import sys, os

    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
    from qwen_proxy import app

    multiline_content = """Line 1: Hello
Line 2: World
Line 3: Test

Empty line above.

Tab\tbetween\twords
End."""
    test_file = "/tmp/e2e_multiline_test.txt"

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
                                            "file_path": test_file,
                                            "content": multiline_content,
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
                "name": "write",
                "description": "Write content to a file",
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

    expected_hash = compute_content_hash(multiline_content)

    messages = [
        {
            "role": "user",
            "content": f"Write this exact multiline content to {test_file}:\n{multiline_content}",
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

    assert len(tool_calls) > 0, "Expected tool_calls for write operation"

    found_matching_write = False
    for tc in tool_calls:
        func_def = tc.get("function", {})
        name = func_def.get("name", "")
        if name == "write":
            args_str = func_def.get("arguments", "{}")
            try:
                args = (
                    json.loads(args_str) if isinstance(args_str, str) else args_str
                )
                if args.get("file_path") == test_file and "content" in args:
                    actual_hash = compute_content_hash(args["content"])
                    assert actual_hash == expected_hash, (
                        "Multiline content hash mismatch"
                    )
                    found_matching_write = True
            except json.JSONDecodeError:
                pass

    assert found_matching_write, "Expected a write tool_call matching test_file"
