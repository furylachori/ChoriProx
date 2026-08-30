# tests/unit/clients/test_orchestrator_config.py
"""Unit tests for clients.orchestrator_config."""

import pytest
from dataclasses import FrozenInstanceError

from clients.orchestrator_config import HookCapabilities, ProxyFeatures


class TestHookCapabilities:
    """Tests for HookCapabilities frozen dataclass."""

    def test_instantiation_defaults(self):
        """HookCapabilities instantiates with all False defaults."""
        hooks = HookCapabilities()
        assert hooks.provider is False
        assert hooks.auth is False
        assert hooks.tool_definition is False
        assert hooks.tool_execute_before is False
        assert hooks.tool_execute_after is False
        assert hooks.chat_params is False
        assert hooks.chat_headers is False
        assert hooks.chat_messages_transform is False
        assert hooks.system_prompt_transform is False
        assert hooks.permission_ask is False
        assert hooks.shell_env is False
        assert hooks.session_compacting is False
        assert hooks.event is False
        assert hooks.config is False
        assert hooks.command_execute is False
        assert hooks.text_complete is False

    def test_instantiation_with_values(self):
        """HookCapabilities accepts explicit values."""
        hooks = HookCapabilities(provider=True, auth=True)
        assert hooks.provider is True
        assert hooks.auth is True
        assert hooks.tool_definition is False

    def test_equality(self):
        """Two identical HookCapabilities are equal."""
        a = HookCapabilities(provider=True)
        b = HookCapabilities(provider=True)
        c = HookCapabilities(provider=False)
        assert a == b
        assert a != c

    def test_immutability(self):
        """HookCapabilities is frozen and raises on mutation."""
        hooks = HookCapabilities()
        with pytest.raises(FrozenInstanceError):
            hooks.provider = True

    def test_hashable(self):
        """Frozen dataclass can be used as dict key."""
        hooks = HookCapabilities()
        d = {hooks: 1}
        assert d[hooks] == 1


class TestProxyFeatures:
    """Tests for ProxyFeatures frozen dataclass."""

    def test_instantiation_defaults(self):
        """ProxyFeatures instantiates with correct defaults."""
        pf = ProxyFeatures()
        assert pf.version == 1
        assert pf.intercepts_memory is True
        assert pf.runs_tool_loop is True
        assert pf.injects_context is True
        assert pf.replaces_system_prompt is True
        assert pf.memory_tool_names == frozenset(
            {
                "recall",
                "search",
                "proxy_ticket_list_tasks",
                "proxy_ticket_get_task",
                "proxy_ticket_create_tasks",
                "proxy_ticket_update_task",
                "proxy_ticket_edit",
                "proxy_ticket_set_parent",
                "proxy_ticket_clear_parent",
                "proxy_ticket_clear_blocked_by",
                "proxy_ticket_set_priority",
                "proxy_ticket_append_review_note",
                "proxy_ticket",
                "proxy_work",
            }
        )
        assert pf.max_tool_iterations == 15
        assert pf.tool_loop_timeout == 90

    def test_instantiation_with_values(self):
        """ProxyFeatures accepts explicit values."""
        pf = ProxyFeatures(version=2, max_tool_iterations=5)
        assert pf.version == 2
        assert pf.max_tool_iterations == 5
        assert pf.intercepts_memory is True  # default unchanged

    def test_equality(self):
        """Two identical ProxyFeatures are equal."""
        a = ProxyFeatures(version=2)
        b = ProxyFeatures(version=2)
        c = ProxyFeatures(version=1)
        assert a == b
        assert a != c

    def test_immutability(self):
        """ProxyFeatures is frozen and raises on mutation."""
        pf = ProxyFeatures()
        with pytest.raises(FrozenInstanceError):
            pf.version = 2

    def test_custom_memory_tool_names(self):
        """ProxyFeatures accepts custom memory_tool_names."""
        custom = frozenset({"recall"})
        pf = ProxyFeatures(memory_tool_names=custom)
        assert pf.memory_tool_names == custom

    def test_hashable(self):
        """Frozen dataclass can be used as dict key."""
        pf = ProxyFeatures()
        d = {pf: 1}
        assert d[pf] == 1
