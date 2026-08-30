TOOL_NAMES = [
    "bash",
    "read",
    "glob",
    "edit",
    "write",
    "webfetch",
    "recall",
    "search",
]

TOOL_CALL_EXAMPLES = {
    "bash": {
        "yaml": "tool_call:\n  name: bash\n  arguments:\n    command: ls -la\n    timeout: 30",
        "xml": "<tool_call>\n  <name>bash</name>\n  <arguments>\n    <command>ls -la</command>\n    <timeout>30</timeout>\n  </arguments>\n</tool_call>",
        "json": '{"name": "bash", "arguments": {"command": "ls -la", "timeout": 30}}',
        "normalized": {"command": "ls -la", "timeout": 30},
    },
    "read": {
        "yaml": "tool_call:\n  name: read\n  arguments:\n    file_path: /tmp/test.txt\n    offset: 0\n    limit: 100",
        "xml": "<tool_call>\n  <name>read</name>\n  <arguments>\n    <file_path>/tmp/test.txt</file_path>\n    <offset>0</offset>\n    <limit>100</limit>\n  </arguments>\n</tool_call>",
        "json": '{"name": "read", "arguments": {"file_path": "/tmp/test.txt", "offset": 0, "limit": 100}}',
        "normalized": {"file_path": "/tmp/test.txt", "offset": 0, "limit": 100},
    },
    "glob": {
        "yaml": 'tool_call:\n  name: glob\n  arguments:\n    path: /path/to/documents\n    pattern: "*.py"',
        "xml": "<tool_call>\n  <name>glob</name>\n  <arguments>\n    <path>/path/to/documents</path>\n    <pattern>*.py</pattern>\n  </arguments>\n</tool_call>",
        "json": '{"name": "glob", "arguments": {"path": "/path/to/documents", "pattern": "*.py"}}',
        "normalized": {"path": "/path/to/documents", "pattern": "*.py"},
    },
    "edit": {
        "yaml": "tool_call:\n  name: edit\n  arguments:\n    file_path: /tmp/test.py\n    old_string: def foo():\n    new_string: def bar():",
        "xml": "<tool_call>\n  <name>edit</name>\n  <arguments>\n    <file_path>/tmp/test.py</file_path>\n    <old_string>def foo():</old_string>\n    <new_string>def bar():</new_string>\n  </arguments>\n</tool_call>",
        "json": '{"name": "edit", "arguments": {"file_path": "/tmp/test.py", "old_string": "def foo():", "new_string": "def bar():"}}',
        "normalized": {
            "file_path": "/tmp/test.py",
            "old_string": "def foo():",
            "new_string": "def bar():",
        },
    },
    "write": {
        "yaml": 'tool_call:\n  name: write\n  arguments:\n    file_path: /tmp/new.py\n    content: print("hello")',
        "xml": '<tool_call>\n  <name>write</name>\n  <arguments>\n    <file_path>/tmp/new.py</file_path>\n    <content>print("hello")</content>\n  </arguments>\n</tool_call>',
        "json": '{"name": "write", "arguments": {"file_path": "/tmp/new.py", "content": "print(\\"hello\\")"}}',
        "normalized": {"file_path": "/tmp/new.py", "content": 'print("hello")'},
    },
    "webfetch": {
        "yaml": "tool_call:\n  name: webfetch\n  arguments:\n    url: https://example.com\n    format: markdown",
        "xml": "<tool_call>\n  <name>webfetch</name>\n  <arguments>\n    <url>https://example.com</url>\n    <format>markdown</format>\n  </arguments>\n</tool_call>",
        "json": '{"name": "webfetch", "arguments": {"url": "https://example.com", "format": "markdown"}}',
        "normalized": {"url": "https://example.com", "format": "markdown"},
    },
    "recall": {
        "yaml": "tool_call:\n  name: recall\n  arguments:\n    semantic_targets:\n      - recent tasks\n    keywords:\n      - task\n    limit: 5",
        "xml": "<tool_call>\n  <name>recall</name>\n  <arguments>\n    <semantic_targets>recent tasks</semantic_targets>\n    <keywords>task</keywords>\n    <limit>5</limit>\n  </arguments>\n</tool_call>",
        "json": '{"name": "recall", "arguments": {"semantic_targets": ["recent tasks"], "keywords": ["task"], "limit": 5}}',
        "normalized": {"semantic_targets": ["recent tasks"], "keywords": ["task"], "limit": 5},
    },
    "search": {
        "yaml": "tool_call:\n  name: search\n  arguments:\n    keywords:\n      - previous conversation\n    limit: 10",
        "xml": "<tool_call>\n  <name>search</name>\n  <arguments>\n    <keywords>previous conversation</keywords>\n    <limit>10</limit>\n  </arguments>\n</tool_call>",
        "json": '{"name": "search", "arguments": {"keywords": ["previous conversation"], "limit": 10}}',
        "normalized": {"keywords": ["previous conversation"], "limit": 10},
    },
}
