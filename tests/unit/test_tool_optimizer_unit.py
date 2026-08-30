"""
Unit tests for tool_optimizer module.
Tests generate_optimized_tool_description output format and
apply_proxy_recovery_system orchestration.
"""

import pytest
from tool_optimizer import generate_optimized_tool_description
from proxy_recovery_system import apply_proxy_recovery_system


class TestToolOptimizerOutput:
    """Tests for generate_optimized_tool_description output."""

    def test_output_has_required_fields(self):
        """Output should have type, function with name, description, and parameters."""
        result = generate_optimized_tool_description("read_file")

        assert "type" in result
        assert result["type"] == "function"
        assert "function" in result
        func = result["function"]
        assert "name" in func
        assert "description" in func
        assert "parameters" in func

    def test_parameters_schema_valid(self):
        """Output parameters schema should be valid object type."""
        result = generate_optimized_tool_description("read_file")

        params = result["function"]["parameters"]
        assert params["type"] == "object"
        assert "properties" in params
        assert isinstance(params["properties"], dict)

    def test_write_file_output(self):
        """write_file optimization should produce valid schema."""
        result = generate_optimized_tool_description("write_file")
        assert result["function"]["name"] == "write_file"
        props = result["function"]["parameters"]["properties"]
        assert "filePath" in props or "path" in props

    def test_edit_file_output(self):
        """edit_file optimization should produce valid schema."""
        result = generate_optimized_tool_description("edit_file")
        assert result["function"]["name"] == "edit_file"

    def test_file_glob_output(self):
        """file_glob optimization should produce valid schema."""
        result = generate_optimized_tool_description("file_glob")
        assert result["function"]["name"] == "file_glob"

    def test_run_shell_command_output(self):
        """run_shell_command optimization should produce valid schema."""
        result = generate_optimized_tool_description("run_shell_command")
        assert result["function"]["name"] == "run_shell_command"
        props = result["function"]["parameters"]["properties"]
        assert "command" in props

    def test_web_fetch_output(self):
        """web_fetch optimization should produce valid schema."""
        result = generate_optimized_tool_description("web_fetch")
        assert result["function"]["name"] == "web_fetch"

    def test_unknown_tool_returns_fallback(self):
        """Unknown tool should return fallback schema."""
        result = generate_optimized_tool_description("nonexistent_tool")
        assert result["function"]["name"] == "nonexistent_tool"
        assert "description" in result["function"]


class TestApplyProxyRecoverySystem:
    """Tests for apply_proxy_recovery_system orchestration.

    Restored after Tier 2.1 deleted tests/unit/test_proxy_recovery_unit.py.
    Covers proxy_recovery_system.py:29 — the branch that decides whether
    to optimize a tool (no parameters) or pass it through unchanged.
    """

    def test_apply_proxy_recovery_system_returns_dict(self):
        """Tools with parameters should pass through unchanged; result is a dict."""
        body = {
            "messages": [{"role": "user", "content": "Hello"}],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "read_file",
                        "description": "Read file",
                        "parameters": {
                            "type": "object",
                            "properties": {"path": {"type": "string"}},
                            "required": ["path"],
                        },
                    },
                }
            ],
        }

        result = apply_proxy_recovery_system(body, "test_session")

        assert isinstance(result, dict)
        assert "tools" in result
        assert len(result["tools"]) == 1

    def test_apply_proxy_recovery_system_optimizes_tools(self):
        """Tools missing parameters should be replaced with optimized schemas."""
        body = {
            "messages": [{"role": "user", "content": "Hello"}],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "read_file",
                        "description": "Read file contents",
                    },
                }
            ],
        }

        result = apply_proxy_recovery_system(body, "test_session")

        assert len(result["tools"]) >= 1
        optimized = result["tools"][0]
        assert optimized["function"]["name"] == "read_file"
        assert "parameters" in optimized["function"]
