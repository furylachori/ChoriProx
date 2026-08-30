"""
Unit tests for validate_tool_schema function.
Tests schema validation for tool definitions.
"""

import pytest
from qwen_proxy import validate_tool_schema


class TestValidateToolSchema:
    """Tests for validate_tool_schema function."""

    def test_validate_tool_schema_valid(self):
        """Valid tool schema should pass validation."""
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "read_file",
                    "description": "Read file contents",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {"type": "string"},
                        },
                        "required": ["path"],
                    },
                },
            }
        ]
        result = validate_tool_schema(tools)
        assert result is True

    def test_validate_tool_schema_missing_required(self):
        """Schema with required field not in properties should fail validation."""
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "read_file",
                    "description": "Read file contents",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {"type": "string"},
                        },
                        "required": ["path", "nonexistent"],
                    },
                },
            }
        ]
        result = validate_tool_schema(tools)
        assert result is False

    def test_validate_tool_schema_wrong_type(self):
        """Schema with wrong type should fail validation."""
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "read_file",
                    "description": "Read file contents",
                    "parameters": {
                        "type": "string",
                        "properties": {},
                    },
                },
            }
        ]
        result = validate_tool_schema(tools)
        assert result is False

    def test_canonical_tool_names(self):
        """All canonical tool names should have valid schemas."""
        canonical_tools = [
            {
                "type": "function",
                "function": {
                    "name": "read_file",
                    "description": "Read file contents",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {"type": "string"},
                        },
                        "required": ["path"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "write_file",
                    "description": "Write content to file",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {"type": "string"},
                            "content": {"type": "string"},
                        },
                        "required": ["path", "content"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "edit_file",
                    "description": "Replace text in file",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {"type": "string"},
                            "old_string": {"type": "string"},
                            "new_string": {"type": "string"},
                        },
                        "required": ["path", "old_string", "new_string"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "file_glob",
                    "description": "Find files matching pattern",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "pattern": {"type": "string"},
                        },
                        "required": ["pattern"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "run_shell_command",
                    "description": "Execute shell command",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "command": {"type": "string"},
                        },
                        "required": ["command"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "web_fetch",
                    "description": "Fetch content from URL",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "url": {"type": "string"},
                        },
                        "required": ["url"],
                    },
                },
            },
        ]
        for tool in canonical_tools:
            result = validate_tool_schema([tool])
            assert result is True, f"Tool {tool['function']['name']} should be valid"
