"""
Unit tests for YAML/XML/JSON parsing functions.
All tests use mocked dependencies - NO real network calls or real LLM.
"""

import pytest
import json
from unittest.mock import patch, MagicMock


class TestYamlToolBlocks:
    """Tests for parse_yaml_tool_blocks function."""

    def test_parse_yaml_tool_blocks_single(self):
        """Test parsing a single YAML tool call."""
        from qwen_proxy import parse_yaml_tool_blocks

        content = """Here is a tool call:
read_file:
  path: /tmp/test.txt
  offset: 1
  limit: 100

And some more text after."""

        tools = parse_yaml_tool_blocks(content)
        assert len(tools) == 1
        assert tools[0]["name"] == "read_file"
        args = tools[0]["arguments"]
        assert args["path"] == "/tmp/test.txt"
        assert args["offset"] == "1"
        assert args["limit"] == "100"

    def test_parse_yaml_tool_blocks_multiple(self):
        """Test parsing multiple YAML tool calls."""
        from qwen_proxy import parse_yaml_tool_blocks

        content = """write_file:
  path: /tmp/output.txt
  content: hello world

read_file:
  path: /tmp/input.txt
  offset: 0"""

        tools = parse_yaml_tool_blocks(content)
        assert len(tools) == 2
        assert tools[0]["name"] == "write_file"
        assert tools[1]["name"] == "read_file"

    def test_parse_xml_tool_format(self):
        """Test parsing XML format tool calls."""
        from qwen_proxy import parse_all_tools_from_content

        content = """<function=bash>
  <parameter=command>ls -la</parameter>
</function>"""

        tools = parse_all_tools_from_content(content)
        assert len(tools) == 1
        assert tools[0]["name"] == "bash"
        assert tools[0]["arguments"]["command"] == "ls -la"

    def test_parse_all_tools_from_content_yaml_matches_isolated(self):
        """parse_all_tools_from_content must dispatch to parse_yaml_tool_blocks
        for YAML inputs and return the same result as the isolated call.
        """
        from qwen_proxy import (
            parse_all_tools_from_content,
            parse_yaml_tool_blocks,
        )

        content = """read_file:
  path: /tmp/test.txt
  offset: 1
  limit: 100"""

        isolated = parse_yaml_tool_blocks(content)
        dispatched = parse_all_tools_from_content(content)
        assert isolated == dispatched
        assert len(dispatched) == 1
        assert dispatched[0]["name"] == "read_file"
