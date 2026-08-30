"""
Unit tests for memory tool parsing.
Tests parsing of all memory tool call formats.
"""

import pytest
import json
from qwen_proxy import parse_all_tools_from_content, parse_tool_from_content


MEMORY_TOOL_NAMES = [
    "recall",
    "search",
]


class TestRecallParsing:
    """Tests for recall tool parsing."""

    def test_parse_recall_tool(self):
        """recall tool should be parseable in JSON format."""
        content = '{"name": "recall", "arguments": {"semantic_targets": ["recent tasks"], "keywords": ["task"], "limit": 5}}'

        tools = parse_all_tools_from_content(content)
        recall_tool = None
        for t in tools:
            if t["name"] == "recall":
                recall_tool = t
                break

        assert recall_tool is not None, "recall tool not found"
        assert "semantic_targets" in recall_tool["arguments"]

    def test_parse_recall_json(self):
        """recall tool should be parseable in JSON format."""
        content = '{"name": "recall", "arguments": {"semantic_targets": ["find tasks"], "keywords": ["task"], "limit": 10}}'

        tools = parse_all_tools_from_content(content)
        recall_tool = None
        for t in tools:
            if t["name"] == "recall":
                recall_tool = t
                break

        assert recall_tool is not None
        assert "semantic_targets" in recall_tool["arguments"]


class TestSearchParsing:
    """Tests for search tool parsing."""

    def test_parse_search_tool(self):
        """search tool should be parseable in JSON format."""
        content = '{"name": "search", "arguments": {"keywords": ["previous conversation"], "limit": 10}}'

        tools = parse_all_tools_from_content(content)
        search_tool = None
        for t in tools:
            if t["name"] == "search":
                search_tool = t
                break

        assert search_tool is not None, "search tool not found"
        assert search_tool["arguments"]["keywords"] == ["previous conversation"]

    def test_parse_search_json(self):
        """search tool should be parseable in JSON format."""
        content = '{"name": "search", "arguments": {"keywords": ["test"]}}'

        tools = parse_all_tools_from_content(content)
        search_tool = None
        for t in tools:
            if t["name"] == "search":
                search_tool = t
                break

        assert search_tool is not None


class TestMemoryToolInvalidFormat:
    """Tests for invalid memory tool format handling."""

    def test_memory_tool_invalid_format_returns_empty(self):
        """Invalid format for memory tools should return empty or partial."""
        content = "This is not a tool call at all."

        tools = parse_all_tools_from_content(content)
        memory_tools = [t for t in tools if t["name"] in MEMORY_TOOL_NAMES]

        assert len(memory_tools) == 0

    def test_memory_tool_malformed_json(self):
        """Malformed JSON for memory tools should not crash."""
        content = '{"name": "recall", "arguments": {invalid}}'

        tools = parse_all_tools_from_content(content)
        assert tools == []

    def test_memory_tool_missing_name(self):
        """Missing name in memory tool should be handled gracefully."""
        content = '{"arguments": {"query": "test"}}'

        tools = parse_all_tools_from_content(content)
        assert isinstance(tools, list)
