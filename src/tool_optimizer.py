"""Tool description optimizer — produces optimized OpenAI tool schemas.

Single live function: `generate_optimized_tool_description(tool_name)` which
returns a pre-built schema (≈80-90% smaller than the verbose upstream
schema) for the canonical tools that the opencode/kilocode clients use.
The `OPENTOOLS_TO_CANONICAL` table maps short client-side names (e.g.
`read`) to the canonical names (`read_file`) whose schemas live below.

Invoked from `proxy_recovery_system.apply_proxy_recovery_system`, which
walks `body["tools"]` and rewrites any tool missing a `parameters` block.
"""

import copy

OPENTOOLS_TO_CANONICAL = {
    "bash": "run_shell_command",
    "read": "read_file",
    "write": "write_file",
    "edit": "edit_file",
    "glob": "file_glob",
    "webfetch": "web_fetch",
}


def generate_optimized_tool_description(tool_name: str) -> dict:
    """Generate LLM-optimized tool description in OpenAI format (80-90% reduction)"""

    optimized_descriptions = {
        "read_file": {
            "type": "function",
            "function": {
                "name": "read_file",
                "description": "Read file contents",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "filePath": {
                            "type": "string",
                            "description": "File path to read",
                        },
                        "offset": {
                            "type": "integer",
                            "description": "Line offset (optional)",
                        },
                        "limit": {
                            "type": "integer",
                            "description": "Max lines (optional)",
                        },
                    },
                    "required": ["filePath"],
                },
            },
        },
        "write_file": {
            "type": "function",
            "function": {
                "name": "write_file",
                "description": "Write content to a file (overwrites existing)",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "filePath": {
                            "type": "string",
                            "description": "File path to write",
                        },
                        "content": {
                            "type": "string",
                            "description": "Content to write",
                        },
                    },
                    "required": ["filePath", "content"],
                },
            },
        },
        "edit_file": {
            "type": "function",
            "function": {
                "name": "edit_file",
                "description": "Replace text in a file",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "filePath": {
                            "type": "string",
                            "description": "File path to edit",
                        },
                        "oldString": {
                            "type": "string",
                            "description": "Text to replace",
                        },
                        "newString": {
                            "type": "string",
                            "description": "Replacement text",
                        },
                        "replaceAll": {
                            "type": "boolean",
                            "description": "Replace all occurrences (optional)",
                        },
                    },
                    "required": ["filePath", "oldString", "newString"],
                },
            },
        },
        "file_glob": {
            "type": "function",
            "function": {
                "name": "file_glob",
                "description": "Find files matching a pattern",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "pattern": {
                            "type": "string",
                            "description": "Glob pattern",
                        },
                        "path": {
                            "type": "string",
                            "description": "Root directory (optional)",
                        },
                    },
                    "required": ["pattern"],
                },
            },
        },
        "run_shell_command": {
            "type": "function",
            "function": {
                "name": "run_shell_command",
                "description": "Execute a shell command. Provide the exact command you want to run in the 'command' parameter (e.g., 'ls -la', 'grep \"pattern\" file.txt', 'python3 script.py').",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "command": {
                            "type": "string",
                            "description": "Command to execute",
                        },
                        "description": {
                            "type": "string",
                            "description": "Clear description of command purpose (optional)",
                        },
                    },
                    "required": ["command"],
                },
            },
        },
        "web_fetch": {
            "type": "function",
            "function": {
                "name": "web_fetch",
                "description": "Fetch content from a URL",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "url": {"type": "string", "description": "URL to fetch"},
                        "format": {
                            "type": "string",
                            "enum": ["text", "markdown", "html"],
                            "description": "Output format (optional)",
                        },
                        "timeout": {
                            "type": "integer",
                            "description": "Timeout in seconds (optional)",
                        },
                    },
                    "required": ["url"],
                },
            },
        },
        "grep": {
            "type": "function",
            "function": {
                "name": "grep",
                "description": "Search file contents using a regular expression pattern",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "pattern": {"type": "string", "description": "Regex pattern to search for"},
                        "path": {"type": "string", "description": "Directory or file path to search (default: current directory)"},
                        "include": {"type": "string", "description": "Glob pattern to filter files (e.g., '*.py')"},
                    },
                    "required": ["pattern"],
                },
            },
        },
    }

    canonical_name = OPENTOOLS_TO_CANONICAL.get(tool_name, tool_name)

    if canonical_name in optimized_descriptions:
        result = copy.deepcopy(optimized_descriptions[canonical_name])
        result["function"]["name"] = tool_name
        return result

    return {
        "type": "function",
        "function": {"name": tool_name, "description": "Tool not optimized"},
    }



