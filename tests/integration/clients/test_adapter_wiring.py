"""Adapter wiring smoke tests.

Verify adapters produce output compatible with the proxy's request/response flow.
These tests ensure adapters can slot into qwen_proxy.py without format mismatches.
"""

import json
import pytest

from clients.kilocode import KiloCodeAdapter
from clients.opencode import OpenCodeAdapter
from clients.interface import FormattedRequest
from clients.capabilities import KILOCODE_CAPABILITIES, OPENCODE_CAPABILITIES


class TestAdapterWiringSmokeTests:
    def test_kilocode_tool_names_cover_proxy_opencode_set(self):
        """The proxy hardcodes OPENCODE_TOOL_NAMES (line 1279-1292).
        KiloCodeAdapter must expose a superset so wiring doesn't drop tools.
        """
        proxy_tool_names = {
            "bash",
            "read",
            "glob",
            "edit",
            "write",
            "webfetch",
            "recall",
            "search",
        }
        adapter = KiloCodeAdapter()
        assert proxy_tool_names <= adapter.get_tool_names()

    def test_opencode_tool_names_cover_proxy_opencode_set(self):
        proxy_tool_names = {
            "bash",
            "read",
            "glob",
            "edit",
            "write",
            "webfetch",
            "recall",
            "search",
        }
        adapter = OpenCodeAdapter()
        assert proxy_tool_names <= adapter.get_tool_names()

    def test_kilocode_system_prompt_matches_proxy_canonical(self):
        """The proxy replaces system prompt with OUR_SYSTEM_PROMPT (line 1385-1391).
        KiloCodeAdapter must return the same text so wiring is transparent.
        """
        proxy_prompt = (
            "You are an AI assistant with access to tools.\n"
            "Use tools when needed. Keep responses concise."
        )
        adapter = KiloCodeAdapter()
        assert adapter.get_system_prompt({}) == proxy_prompt

    def test_opencode_system_prompt_matches_proxy_canonical(self):
        proxy_prompt = (
            "You are an AI coding assistant with access to tools.\n"
            "Use tools when needed. Keep responses concise."
        )
        adapter = OpenCodeAdapter()
        assert adapter.get_system_prompt({}) == proxy_prompt

    def test_kilocode_normalize_maps_filePath_to_path(self):
        adapter = KiloCodeAdapter()
        normalized = adapter.normalize_tool_args("read", {"filePath": "/tmp/test.py"})
        assert normalized["path"] == "/tmp/test.py"
        assert "filePath" not in normalized

    def test_kilocode_normalize_maps_edit_params(self):
        adapter = KiloCodeAdapter()
        normalized = adapter.normalize_tool_args(
            "edit",
            {"filePath": "/tmp/a.py", "oldString": "foo", "newString": "bar"},
        )
        assert normalized["path"] == "/tmp/a.py"
        assert normalized["old_string"] == "foo"
        assert normalized["new_string"] == "bar"

    def test_kilocode_denormalize_is_inverse(self):
        adapter = KiloCodeAdapter()
        original = {"filePath": "/tmp/x.py", "oldString": "a", "newString": "b"}
        normalized = adapter.normalize_tool_args("edit", original)
        denormalized = adapter.denormalize_tool_args("edit", normalized)
        assert denormalized == original

    def test_opencode_normalize_is_identity(self):
        adapter = OpenCodeAdapter()
        args = {"path": "/tmp/test.py", "command": "ls"}
        assert adapter.normalize_tool_args("read", args) == args
        assert adapter.denormalize_tool_args("read", args) == args

    def test_kilocode_sse_chunk_format(self):
        adapter = KiloCodeAdapter()
        chunk = {"choices": [{"delta": {"content": "hi"}}]}
        line = adapter.format_sse_chunk(chunk)
        assert line.startswith("data: ")
        assert line.endswith("\n\n")
        parsed = json.loads(line[6:])
        assert parsed == chunk

    def test_opencode_sse_chunk_format(self):
        adapter = OpenCodeAdapter()
        chunk = {"choices": [{"delta": {"content": "hi"}}]}
        line = adapter.format_sse_chunk(chunk)
        assert line.startswith("data: ")
        assert line.endswith("\n\n")

    def test_formatted_request_validates_messages_type(self):
        with pytest.raises(TypeError):
            FormattedRequest(messages="not a list")

    def test_kilocode_generate_tool_call_id(self):
        adapter = KiloCodeAdapter()
        tid = adapter.generate_tool_call_id("bash")
        assert tid.startswith("call_")
        assert len(tid) == 17  # call_ + 12 hex chars

    def test_kilocode_format_tool_result(self):
        adapter = KiloCodeAdapter()
        result = adapter.format_tool_result("bash", "output")
        assert result["role"] == "tool"
        assert result["content"] == "output"

    def test_kilocode_format_error(self):
        adapter = KiloCodeAdapter()
        err = adapter.format_error(ValueError("boom"))
        assert err["role"] == "tool"
        assert "boom" in err["content"]

    def test_kilocode_capabilities_match_expected(self):
        adapter = KiloCodeAdapter()
        caps = adapter.get_capabilities()
        assert caps == KILOCODE_CAPABILITIES
        assert caps.has_plugin_system is True
        assert caps.auth_mechanism == "bearer"

    def test_opencode_capabilities_match_expected(self):
        adapter = OpenCodeAdapter()
        caps = adapter.get_capabilities()
        assert caps == OPENCODE_CAPABILITIES
        assert caps.has_plugin_system is False
        assert caps.config_format == "markdown_frontmatter"
