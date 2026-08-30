# tests/unit/clients/test_opencode.py
"""Unit tests for clients.opencode."""

import logging
from unittest.mock import MagicMock, patch

import pytest
import yaml

from clients.opencode import OpenCodeAdapter
from clients.interface import ClientConfig, ConfigLoader, CapabilitySet
from clients.capabilities import OPENCODE_CAPABILITIES


class TestOpenCodeAdapterInstantiation:
    """Tests for OpenCodeAdapter basic properties."""

    def test_name(self):
        """OpenCodeAdapter has correct name."""
        adapter = OpenCodeAdapter()
        assert adapter.name == "opencode"

    def test_get_capabilities(self):
        """get_capabilities returns OPENCODE_CAPABILITIES."""
        adapter = OpenCodeAdapter()
        caps = adapter.get_capabilities()
        assert caps is OPENCODE_CAPABILITIES
        assert isinstance(caps, CapabilitySet)

    def test_get_tool_names(self):
        """get_tool_names returns expected tool set."""
        adapter = OpenCodeAdapter()
        tools = adapter.get_tool_names()
        expected = {
            "bash",
            "read",
            "glob",
            "grep",
            "edit",
            "write",
            "webfetch",
            "MiniMax_web_search",
            "recall",
            "search",
            "suggest",
            "task",
        }
        assert tools == expected


class TestOpenCodeAdapterNormalizeToolArgs:
    """Tests for normalize_tool_args and denormalize_tool_args."""

    def test_normalize_returns_copy(self):
        """normalize_tool_args returns a shallow copy."""
        adapter = OpenCodeAdapter()
        original = {"key": "val"}
        result = adapter.normalize_tool_args("any", original)
        assert result == original
        assert result is not original

    def test_denormalize_returns_copy(self):
        """denormalize_tool_args returns a shallow copy."""
        adapter = OpenCodeAdapter()
        original = {"key": "val"}
        result = adapter.denormalize_tool_args("any", original)
        assert result == original
        assert result is not original


class TestOpenCodeAdapterFormatResponse:
    """Tests for format_response and format_sse_chunk."""

    def test_format_response_returns_data(self):
        """format_response returns data unchanged."""
        adapter = OpenCodeAdapter()
        data = {"role": "assistant"}
        assert adapter.format_response(data, True) is data

    def test_format_response_with_wants_stream_false(self):
        """format_response returns data unchanged when wants_stream is False."""
        adapter = OpenCodeAdapter()
        data = {"role": "assistant"}
        assert adapter.format_response(data, False) is data

    def test_format_sse_chunk(self):
        """format_sse_chunk produces correct SSE line."""
        adapter = OpenCodeAdapter()
        result = adapter.format_sse_chunk({"delta": "hi"})
        assert result == 'data: {"delta": "hi"}\n\n'


class TestOpenCodeAdapterLoadConfig:
    """Tests for load_config with mocked ConfigLoader and yaml."""

    def test_load_config_no_config_dir(self):
        """load_config returns default when .opencode dir does not exist."""
        adapter = OpenCodeAdapter()
        loader = MagicMock(spec=ConfigLoader)
        loader.path_join.side_effect = lambda *parts: "/".join(parts)
        loader.is_dir.return_value = False
        result = adapter.load_config("/project", loader=loader)
        assert isinstance(result, ClientConfig)
        assert result.name == "opencode"

    def test_load_config_reads_markdown_frontmatter(self):
        """load_config reads YAML frontmatter from .md file."""
        adapter = OpenCodeAdapter()
        loader = MagicMock(spec=ConfigLoader)
        loader.path_join.side_effect = lambda *parts: "/".join(parts)
        loader.is_dir.return_value = True
        loader.is_file.return_value = True
        loader.list_dir.return_value = ["agent.md"]
        loader.read_file.return_value = (
            "---\nname: my-agent\nmodel: gpt-4\ntemperature: 0.7\n---\n# Agent\n"
        )
        result = adapter.load_config("/project", loader=loader)
        assert result.name == "my-agent"
        assert result.model == "gpt-4"
        assert result.temperature == 0.7

    def test_load_config_skips_non_md_files(self):
        """load_config skips files not ending in .md."""
        adapter = OpenCodeAdapter()
        loader = MagicMock(spec=ConfigLoader)
        loader.path_join.side_effect = lambda *parts: "/".join(parts)
        loader.is_dir.return_value = True
        loader.is_file.return_value = True
        loader.list_dir.return_value = ["README.txt"]
        result = adapter.load_config("/project", loader=loader)
        assert result.name == "opencode"

    def test_load_config_skips_directories(self):
        """load_config skips directories inside .opencode."""
        adapter = OpenCodeAdapter()
        loader = MagicMock(spec=ConfigLoader)
        loader.path_join.side_effect = lambda *parts: "/".join(parts)
        loader.is_dir.return_value = True
        loader.is_file.return_value = False
        loader.list_dir.return_value = ["subdir"]
        result = adapter.load_config("/project", loader=loader)
        assert result.name == "opencode"

    def test_load_config_no_frontmatter(self):
        """load_config skips .md files without frontmatter."""
        adapter = OpenCodeAdapter()
        loader = MagicMock(spec=ConfigLoader)
        loader.path_join.side_effect = lambda *parts: "/".join(parts)
        loader.is_dir.return_value = True
        loader.is_file.return_value = True
        loader.list_dir.return_value = ["agent.md"]
        loader.read_file.return_value = "# No frontmatter\n"
        result = adapter.load_config("/project", loader=loader)
        assert result.name == "opencode"

    def test_load_config_invalid_yaml_logs_warning(self, caplog):
        """load_config logs warning for invalid YAML frontmatter."""
        adapter = OpenCodeAdapter()
        loader = MagicMock(spec=ConfigLoader)
        loader.path_join.side_effect = lambda *parts: "/".join(parts)
        loader.is_dir.return_value = True
        loader.is_file.return_value = True
        loader.list_dir.return_value = ["bad.md"]
        loader.read_file.return_value = "---\n[invalid yaml\n---\n"
        with caplog.at_level(logging.WARNING):
            result = adapter.load_config("/project", loader=loader)
        assert "Failed to load OpenCode config" in caplog.text
        assert result.name == "opencode"

    def test_load_config_non_dict_frontmatter(self):
        """load_config skips frontmatter that parses to non-dict."""
        adapter = OpenCodeAdapter()
        loader = MagicMock(spec=ConfigLoader)
        loader.path_join.side_effect = lambda *parts: "/".join(parts)
        loader.is_dir.return_value = True
        loader.is_file.return_value = True
        loader.list_dir.return_value = ["agent.md"]
        loader.read_file.return_value = "---\n- list_item\n---\n"
        result = adapter.load_config("/project", loader=loader)
        assert result.name == "opencode"

    def test_load_config_uses_default_loader(self):
        """load_config uses ConfigLoader() when loader is None."""
        adapter = OpenCodeAdapter()
        with patch.object(ConfigLoader, "is_dir", return_value=False):
            result = adapter.load_config("/project", loader=None)
        assert result.name == "opencode"

    def test_load_config_missing_optional_fields(self):
        """load_config uses defaults for missing frontmatter fields."""
        adapter = OpenCodeAdapter()
        loader = MagicMock(spec=ConfigLoader)
        loader.path_join.side_effect = lambda *parts: "/".join(parts)
        loader.is_dir.return_value = True
        loader.is_file.return_value = True
        loader.list_dir.return_value = ["agent.md"]
        loader.read_file.return_value = "---\nmodel: gpt-3\n---\n"
        result = adapter.load_config("/project", loader=loader)
        assert result.name == "agent"  # defaults to filename without .md
        assert result.model == "gpt-3"
        assert result.temperature == 0.1

    def test_load_config_oserror_logs_warning(self, caplog):
        """load_config logs warning on OSError from read_file."""
        adapter = OpenCodeAdapter()
        loader = MagicMock(spec=ConfigLoader)
        loader.path_join.side_effect = lambda *parts: "/".join(parts)
        loader.is_dir.return_value = True
        loader.is_file.return_value = True
        loader.list_dir.return_value = ["agent.md"]
        loader.read_file.side_effect = OSError("read failed")
        with caplog.at_level("WARNING"):
            result = adapter.load_config("/project", loader=loader)
        assert result.name == "opencode"
        assert "read failed" in caplog.text
