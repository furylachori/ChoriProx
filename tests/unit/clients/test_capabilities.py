# tests/unit/clients/test_capabilities.py
"""Unit tests for clients.capabilities."""

from clients.capabilities import (
    KILOCODE_CAPABILITIES,
    OPENCODE_CAPABILITIES,
)
from clients.interface import (
    CapabilitySet,
    ToolFormat,
    StreamingFormat,
    MessageFormat,
)
from clients.orchestrator_config import HookCapabilities


class TestKiloCodeCapabilities:
    """Tests for KILOCODE_CAPABILITIES constant."""

    def test_is_capability_set(self):
        """KILOCODE_CAPABILITIES is a CapabilitySet instance."""
        assert isinstance(KILOCODE_CAPABILITIES, CapabilitySet)

    def test_tool_format(self):
        """KILOCODE_CAPABILITIES uses OPENAI tool format."""
        assert KILOCODE_CAPABILITIES.tool_format == ToolFormat.OPENAI

    def test_streaming_format(self):
        """KILOCODE_CAPABILITIES uses SSE_OPENAI streaming."""
        assert KILOCODE_CAPABILITIES.streaming_format == StreamingFormat.SSE_OPENAI

    def test_message_format(self):
        """KILOCODE_CAPABILITIES uses OPENAI message format."""
        assert KILOCODE_CAPABILITIES.message_format == MessageFormat.OPENAI

    def test_plugin_system(self):
        """KILOCODE_CAPABILITIES has plugin system enabled."""
        assert KILOCODE_CAPABILITIES.has_plugin_system is True

    def test_hooks_enabled(self):
        """KILOCODE_CAPABILITIES has all hooks enabled."""
        hooks = KILOCODE_CAPABILITIES.hooks
        assert isinstance(hooks, HookCapabilities)
        assert hooks.provider is True
        assert hooks.auth is True
        assert hooks.tool_definition is True
        assert hooks.command_execute is True
        assert hooks.text_complete is True

    def test_config_dirs(self):
        """KILOCODE_CAPABILITIES uses .kilo directories."""
        assert KILOCODE_CAPABILITIES.agent_config_dir == ".kilo"
        assert KILOCODE_CAPABILITIES.command_config_dir == ".kilo"

    def test_auth_mechanism(self):
        """KILOCODE_CAPABILITIES uses bearer auth."""
        assert KILOCODE_CAPABILITIES.auth_mechanism == "bearer"

    def test_immutability(self):
        """KILOCODE_CAPABILITIES is frozen (cannot mutate)."""
        import pytest
        from dataclasses import FrozenInstanceError

        with pytest.raises(FrozenInstanceError):
            KILOCODE_CAPABILITIES.has_plugin_system = False


class TestOpenCodeCapabilities:
    """Tests for OPENCODE_CAPABILITIES constant."""

    def test_is_capability_set(self):
        """OPENCODE_CAPABILITIES is a CapabilitySet instance."""
        assert isinstance(OPENCODE_CAPABILITIES, CapabilitySet)

    def test_tool_format(self):
        """OPENCODE_CAPABILITIES uses OPENAI tool format."""
        assert OPENCODE_CAPABILITIES.tool_format == ToolFormat.OPENAI

    def test_config_format(self):
        """OPENCODE_CAPABILITIES uses markdown_frontmatter config."""
        assert OPENCODE_CAPABILITIES.config_format == "markdown_frontmatter"

    def test_hooks_disabled(self):
        """OPENCODE_CAPABILITIES has empty hooks."""
        hooks = OPENCODE_CAPABILITIES.hooks
        assert isinstance(hooks, HookCapabilities)
        assert hooks.provider is False
        assert hooks.auth is False

    def test_config_dirs(self):
        """OPENCODE_CAPABILITIES uses .opencode directories."""
        assert OPENCODE_CAPABILITIES.agent_config_dir == ".opencode"
        assert OPENCODE_CAPABILITIES.command_config_dir == ".opencode"

    def test_plugin_system(self):
        """OPENCODE_CAPABILITIES does not have plugin system."""
        assert OPENCODE_CAPABILITIES.has_plugin_system is False
