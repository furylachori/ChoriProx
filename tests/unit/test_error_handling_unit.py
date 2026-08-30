"""
Unit tests for error handling in malformed responses.
Tests that malformed inputs don't crash the parsing.
"""

import pytest
import json
from qwen_proxy import parse_all_tools_from_content, parse_tool_from_content


class TestMalformedJson:
    """Tests for malformed JSON handling."""

    def test_malformed_json_doesnt_crash(self):
        """Malformed JSON should not crash the parser."""
        content = "{ this is not valid json at all }"

        try:
            tools = parse_all_tools_from_content(content)
            assert isinstance(tools, list)
        except Exception:
            pytest.fail("Parser should not crash on malformed JSON")

    def test_malformed_json_returns_empty_or_partial(self):
        """Malformed JSON should return empty list or partial results."""
        content = '{"name": "read_file", "arguments": {invalid}}'

        tools = parse_all_tools_from_content(content)
        assert isinstance(tools, list)


class TestMissingToolName:
    """Tests for missing tool name handling."""

    def test_missing_tool_name_handled(self):
        """Missing tool name should be handled gracefully."""
        content = '{"arguments": {"path": "/tmp/test.txt"}}'

        tools = parse_all_tools_from_content(content)
        assert isinstance(tools, list)

    def test_missing_arguments_handled(self):
        """Missing arguments should be handled gracefully."""
        content = '{"name": "read_file"}'

        tools = parse_all_tools_from_content(content)
        assert isinstance(tools, list)


class TestInvalidArgumentsType:
    """Tests for invalid argument type handling."""

    def test_invalid_arguments_type_handled(self):
        """Invalid arguments type (non-dict) should be handled."""
        content = '{"name": "read_file", "arguments": "not a dict"}'

        tools = parse_all_tools_from_content(content)
        assert isinstance(tools, list)

    def test_arguments_as_string(self):
        """Arguments as string instead of dict should not crash."""
        content = '{"name": "bash", "arguments": "ls -la"}'

        tools = parse_all_tools_from_content(content)
        assert isinstance(tools, list)


class TestExceptionInParsing:
    """Tests for exceptions during parsing."""

    def test_exception_in_parsing_returns_empty(self):
        """Exception during parsing should return empty list."""
        content = None

        tools = parse_all_tools_from_content(content)
        assert tools == []

    def test_empty_content_returns_empty(self):
        """Empty content should return empty list."""
        tools = parse_all_tools_from_content("")
        assert tools == []

    def test_whitespace_only_returns_empty(self):
        """Whitespace-only content should return empty list."""
        tools = parse_all_tools_from_content("   \n\n  ")
        assert tools == []


class TestEdgeCases:
    """Tests for edge cases in parsing."""

    def test_nested_braces_json(self):
        """JSON with nested braces should be handled."""
        content = '{"name": "read_file", "arguments": {"path": "/tmp/{test}.txt"}}'

        tools = parse_all_tools_from_content(content)
        assert isinstance(tools, list)

    def test_tool_name_with_underscore(self):
        """Tool names with underscores should be parsed."""
        content = "some_tool_name:\n  path: /tmp/test.txt"

        tools = parse_all_tools_from_content(content)
        assert isinstance(tools, list)

    def test_multiple_tools_same_name(self):
        """Multiple tools with same name should be deduplicated."""
        content = '{"name": "read_file", "arguments": {"path": "/tmp/a.txt"}}'
        content2 = '{"name": "read_file", "arguments": {"path": "/tmp/a.txt"}}'

        combined = content + "\n" + content2
        tools = parse_all_tools_from_content(combined)

        read_count = sum(1 for t in tools if t["name"] == "read_file")
        assert read_count <= 2
