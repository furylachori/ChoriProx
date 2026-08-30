# tests/unit/clients/test_kilocode.py
"""Unit tests for clients.kilocode."""

import json
import logging
from unittest.mock import MagicMock, patch

import pytest

from clients.kilocode import KiloCodeAdapter
from clients.interface import ClientConfig, ConfigLoader, CapabilitySet
from clients.capabilities import KILOCODE_CAPABILITIES


class TestKiloCodeAdapterInstantiation:
    """Tests for KiloCodeAdapter basic properties."""

    def test_name(self):
        """KiloCodeAdapter has correct name."""
        adapter = KiloCodeAdapter()
        assert adapter.name == "kilocode"

    def test_get_capabilities(self):
        """get_capabilities returns KILOCODE_CAPABILITIES."""
        adapter = KiloCodeAdapter()
        caps = adapter.get_capabilities()
        assert caps is KILOCODE_CAPABILITIES
        assert isinstance(caps, CapabilitySet)

    def test_get_tool_names(self):
        """get_tool_names returns expected tool set."""
        adapter = KiloCodeAdapter()
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


class TestKiloCodeAdapterNormalizeToolArgs:
    """Tests for normalize_tool_args."""

    def test_normalize_read_filePath_to_path(self):
        """normalize_tool_args maps filePath to path for read."""
        adapter = KiloCodeAdapter()
        result = adapter.normalize_tool_args("read", {"filePath": "/tmp"})
        assert result == {"path": "/tmp"}
        assert "filePath" not in result

    def test_normalize_write_filePath_to_path(self):
        """normalize_tool_args maps filePath to path for write."""
        adapter = KiloCodeAdapter()
        result = adapter.normalize_tool_args("write", {"filePath": "/tmp"})
        assert result == {"path": "/tmp"}

    def test_normalize_edit_all_mappings(self):
        """normalize_tool_args maps all edit fields correctly."""
        adapter = KiloCodeAdapter()
        result = adapter.normalize_tool_args(
            "edit",
            {
                "filePath": "/tmp",
                "oldString": "old",
                "newString": "new",
            },
        )
        assert result == {
            "path": "/tmp",
            "old_string": "old",
            "new_string": "new",
        }

    def test_normalize_bash_no_change(self):
        """normalize_tool_args leaves bash command unchanged."""
        adapter = KiloCodeAdapter()
        result = adapter.normalize_tool_args("bash", {"command": "ls"})
        assert result == {"command": "ls"}

    def test_normalize_unknown_tool(self):
        """normalize_tool_args returns copy for unknown tool."""
        adapter = KiloCodeAdapter()
        result = adapter.normalize_tool_args("unknown", {"key": "val"})
        assert result == {"key": "val"}

    def test_normalize_does_not_mutate_input(self):
        """normalize_tool_args does not modify the input dict."""
        adapter = KiloCodeAdapter()
        original = {"filePath": "/tmp"}
        adapter.normalize_tool_args("read", original)
        assert original == {"filePath": "/tmp"}


class TestKiloCodeAdapterDenormalizeToolArgs:
    """Tests for denormalize_tool_args."""

    def test_denormalize_read_path_to_filePath(self):
        """denormalize_tool_args maps path to filePath for read."""
        adapter = KiloCodeAdapter()
        result = adapter.denormalize_tool_args("read", {"path": "/tmp"})
        assert result == {"filePath": "/tmp"}

    def test_denormalize_write_path_to_filePath(self):
        """denormalize_tool_args maps path to filePath for write."""
        adapter = KiloCodeAdapter()
        result = adapter.denormalize_tool_args("write", {"path": "/tmp"})
        assert result == {"filePath": "/tmp"}

    def test_denormalize_edit_all_mappings(self):
        """denormalize_tool_args maps all edit fields correctly."""
        adapter = KiloCodeAdapter()
        result = adapter.denormalize_tool_args(
            "edit",
            {
                "path": "/tmp",
                "old_string": "old",
                "new_string": "new",
            },
        )
        assert result == {
            "filePath": "/tmp",
            "oldString": "old",
            "newString": "new",
        }

    def test_denormalize_bash_no_change(self):
        """denormalize_tool_args leaves bash command unchanged."""
        adapter = KiloCodeAdapter()
        result = adapter.denormalize_tool_args("bash", {"command": "ls"})
        assert result == {"command": "ls"}

    def test_denormalize_unknown_tool(self):
        """denormalize_tool_args returns copy for unknown tool."""
        adapter = KiloCodeAdapter()
        result = adapter.denormalize_tool_args("unknown", {"key": "val"})
        assert result == {"key": "val"}

    def test_denormalize_does_not_mutate_input(self):
        """denormalize_tool_args does not modify the input dict."""
        adapter = KiloCodeAdapter()
        original = {"path": "/tmp"}
        adapter.denormalize_tool_args("read", original)
        assert original == {"path": "/tmp"}


class TestKiloCodeAdapterFormatResponse:
    """Tests for format_response and format_sse_chunk."""

    def test_format_response_returns_data(self):
        """format_response returns data unchanged."""
        adapter = KiloCodeAdapter()
        data = {"role": "assistant"}
        assert adapter.format_response(data, True) is data

    def test_format_response_with_wants_stream_false(self):
        """format_response returns data unchanged when wants_stream is False."""
        adapter = KiloCodeAdapter()
        data = {"role": "assistant"}
        assert adapter.format_response(data, False) is data

    def test_format_sse_chunk(self):
        """format_sse_chunk produces correct SSE line."""
        adapter = KiloCodeAdapter()
        result = adapter.format_sse_chunk({"delta": "hi"})
        assert result == 'data: {"delta": "hi"}\n\n'


class TestKiloCodeAdapterLoadConfig:
    """Tests for load_config with mocked ConfigLoader."""

    def test_load_config_no_config_dir(self):
        """load_config returns default when agent dir does not exist."""
        adapter = KiloCodeAdapter()
        loader = MagicMock(spec=ConfigLoader)
        loader.path_join.return_value = "/project/.kilo/agent"
        loader.is_dir.return_value = False
        result = adapter.load_config("/project", loader=loader)
        assert isinstance(result, ClientConfig)
        assert result.name == "kilocode"

    def test_load_config_reads_json(self):
        """load_config reads and parses JSON config file."""
        adapter = KiloCodeAdapter()
        loader = MagicMock(spec=ConfigLoader)
        loader.path_join.side_effect = lambda *parts: "/".join(parts)
        loader.is_dir.return_value = True
        loader.list_dir.return_value = ["agent.json"]
        loader.read_file.return_value = json.dumps(
            {
                "name": "my-agent",
                "model": "gpt-4",
                "temperature": 0.5,
            }
        )
        result = adapter.load_config("/project", loader=loader)
        assert result.name == "my-agent"
        assert result.model == "gpt-4"
        assert result.temperature == 0.5

    def test_load_config_skips_non_json(self):
        """load_config skips files not ending in .json."""
        adapter = KiloCodeAdapter()
        loader = MagicMock(spec=ConfigLoader)
        loader.path_join.side_effect = lambda *parts: "/".join(parts)
        loader.is_dir.return_value = True
        loader.list_dir.return_value = ["README.md", "agent.json"]
        loader.read_file.return_value = json.dumps({"name": "agent"})

        def side_effect(path):
            if path.endswith("agent.json"):
                return json.dumps({"name": "agent"})
            return ""

        loader.read_file.side_effect = side_effect

        result = adapter.load_config("/project", loader=loader)
        assert result.name == "agent"

    def test_load_config_invalid_json_logs_warning(self, caplog):
        """load_config logs warning for invalid JSON and returns default."""
        adapter = KiloCodeAdapter()
        loader = MagicMock(spec=ConfigLoader)
        loader.path_join.side_effect = lambda *parts: "/".join(parts)
        loader.is_dir.return_value = True
        loader.list_dir.return_value = ["bad.json"]
        loader.read_file.return_value = "not json"
        with caplog.at_level(logging.WARNING):
            result = adapter.load_config("/project", loader=loader)
        assert "Failed to load KiloCode config" in caplog.text
        assert result.name == "kilocode"

    def test_load_config_type_error_logs_warning(self, caplog):
        """load_config logs warning when read_file returns non-string."""
        adapter = KiloCodeAdapter()
        loader = MagicMock(spec=ConfigLoader)
        loader.path_join.side_effect = lambda *parts: "/".join(parts)
        loader.is_dir.return_value = True
        loader.list_dir.return_value = ["bad.json"]
        loader.read_file.return_value = None  # json.loads(None) raises TypeError
        with caplog.at_level(logging.WARNING):
            result = adapter.load_config("/project", loader=loader)
        assert "Failed to load KiloCode config" in caplog.text
        assert result.name == "kilocode"

    def test_load_config_uses_default_loader(self):
        """load_config uses ConfigLoader() when loader is None."""
        adapter = KiloCodeAdapter()
        with patch.object(ConfigLoader, "is_dir", return_value=False):
            result = adapter.load_config("/project", loader=None)
        assert result.name == "kilocode"

    def test_load_config_missing_optional_fields_use_defaults(self):
        """load_config uses defaults for missing fields in JSON."""
        adapter = KiloCodeAdapter()
        loader = MagicMock(spec=ConfigLoader)
        loader.path_join.side_effect = lambda *parts: "/".join(parts)
        loader.is_dir.return_value = True
        loader.list_dir.return_value = ["partial.json"]
        loader.read_file.return_value = json.dumps({"model": "gpt-3"})
        result = adapter.load_config("/project", loader=loader)
        assert result.name == "unknown"
        assert result.model == "gpt-3"
        assert result.temperature == 0.1


class TestKiloCodeAdapterSystemPrompt:
    """Tests for get_system_prompt."""

    def test_get_system_prompt(self):
        """get_system_prompt returns kilocode-specific prompt."""
        adapter = KiloCodeAdapter()
        prompt = adapter.get_system_prompt({})
        assert "Kilo" not in prompt  # It's generic in spec
        assert "AI assistant" in prompt

    def test_get_system_prompt_with_context(self):
        """get_system_prompt accepts context dict without error."""
        adapter = KiloCodeAdapter()
        prompt = adapter.get_system_prompt({"project": "test"})
        assert "AI assistant" in prompt


class TestKiloCodeAdapterOtherMethods:
    """Tests for remaining concrete methods."""

    def test_get_plugin_adapter_returns_none(self):
        """get_plugin_adapter returns None."""
        adapter = KiloCodeAdapter()
        assert adapter.get_plugin_adapter() is None

    def test_inherits_generate_tool_call_id(self):
        """KiloCodeAdapter inherits generate_tool_call_id from ClientAdapter."""
        adapter = KiloCodeAdapter()
        tid = adapter.generate_tool_call_id("tool")
        assert tid.startswith("call_")
