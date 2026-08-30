MOCK_TOOL_SELECTION = {
    "tool_calls": [
        {"name": "bash", "arguments": {"command": "echo test", "timeout": 30}},
        {"name": "read", "arguments": {"file_path": "/tmp/test.txt"}},
    ],
    "finish_reason": "tool_calls",
}

MOCK_PARAMETER_VALIDATION = {
    "tool_calls": [{"name": "bash", "arguments": {"command": "ls", "timeout": -1}}],
    "finish_reason": "tool_calls",
}

MOCK_ERROR_HANDLING = {
    "tool_calls": [{"name": "invalid_tool", "arguments": {}}],
    "finish_reason": "tool_calls",
}

MOCK_EMPTY_RESPONSE = {"tool_calls": [], "finish_reason": "stop"}

MOCK_MULTI_TOOL = {
    "tool_calls": [
        {"name": "bash", "arguments": {"command": "pwd"}},
        {"name": "glob", "arguments": {"path": "/tmp", "pattern": "*.txt"}},
        {"name": "read", "arguments": {"file_path": "/tmp/data.txt"}},
    ],
    "finish_reason": "tool_calls",
}

MOCK_FILE_STATE = {
    "tool_calls": [
        {"name": "write", "arguments": {"file_path": "/tmp/out.txt", "content": "data"}}
    ],
    "finish_reason": "tool_calls",
    "file_state": {"modified": ["/tmp/out.txt"], "created": [], "deleted": []},
}
