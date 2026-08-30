# tests/unit/clients/test_registry.py
"""Unit tests for clients.registry."""

import os
from unittest.mock import MagicMock, patch

import pytest

from clients.registry import ClientRegistry
from clients.interface import (
    ClientAdapter,
    CapabilitySet,
    RegistryError,
    DetectionError,
)


class DummyAdapter(ClientAdapter):
    """Minimal concrete adapter for registry tests."""

    name = "dummy"

    def get_capabilities(self):
        return CapabilitySet(agent_config_dir=".dummy")

    def get_tool_names(self):
        return {"tool"}

    def normalize_tool_args(self, tool_name, arguments):
        return dict(arguments)


class TestClientRegistrySingleton:
    """Tests for singleton behavior."""

    def setup_method(self):
        """Reset singleton before each test."""
        ClientRegistry.reset()

    def teardown_method(self):
        """Reset singleton after each test."""
        ClientRegistry.reset()

    def test_instance_returns_same_object(self):
        """instance() returns the same singleton object."""
        a = ClientRegistry.instance()
        b = ClientRegistry.instance()
        assert a is b

    def test_instance_is_client_registry(self):
        """instance() returns a ClientRegistry."""
        reg = ClientRegistry.instance()
        assert isinstance(reg, ClientRegistry)

    def test_reset_clears_instance(self):
        """reset() clears the singleton so next instance() creates new one."""
        a = ClientRegistry.instance()
        ClientRegistry.reset()
        b = ClientRegistry.instance()
        assert a is not b

    def test_for_testing_returns_new_instance(self):
        """_for_testing returns a fresh isolated instance."""
        singleton = ClientRegistry.instance()
        isolated = ClientRegistry._for_testing()
        assert isinstance(isolated, ClientRegistry)
        assert isolated is not singleton

    def test_for_testing_does_not_affect_singleton(self):
        """_for_testing does not modify the singleton."""
        singleton = ClientRegistry.instance()
        singleton.register(DummyAdapter())
        isolated = ClientRegistry._for_testing()
        isolated.register(DummyAdapter())
        assert "dummy" in singleton.list_clients()
        # isolated has its own _clients
        assert isolated.list_clients() == ["dummy"]


class TestClientRegistryRegister:
    """Tests for register method."""

    def setup_method(self):
        ClientRegistry.reset()

    def teardown_method(self):
        ClientRegistry.reset()

    def test_register_adds_adapter(self):
        """register stores adapter by name."""
        reg = ClientRegistry._for_testing()
        adapter = DummyAdapter()
        reg.register(adapter)
        assert reg.list_clients() == ["dummy"]

    def test_register_overwrites_same_name(self):
        """register overwrites adapter with same name."""
        reg = ClientRegistry._for_testing()
        a = DummyAdapter()
        b = DummyAdapter()
        reg.register(a)
        reg.register(b)
        assert reg.get("dummy") is b


class TestClientRegistryGet:
    """Tests for get method."""

    def setup_method(self):
        ClientRegistry.reset()

    def teardown_method(self):
        ClientRegistry.reset()

    def test_get_returns_adapter(self):
        """get returns the registered adapter."""
        reg = ClientRegistry._for_testing()
        adapter = DummyAdapter()
        reg.register(adapter)
        assert reg.get("dummy") is adapter

    def test_get_raises_when_not_found(self):
        """get raises RegistryError for unknown client."""
        reg = ClientRegistry._for_testing()
        with pytest.raises(RegistryError, match="Client 'missing' not registered"):
            reg.get("missing")

    def test_get_error_includes_available(self):
        """RegistryError includes available client names."""
        reg = ClientRegistry._for_testing()
        reg.register(DummyAdapter())
        with pytest.raises(RegistryError, match=r"Available: \['dummy'\]"):
            reg.get("missing")


class TestClientRegistryActive:
    """Tests for active and set_active methods."""

    def setup_method(self):
        ClientRegistry.reset()

    def teardown_method(self):
        ClientRegistry.reset()

    def test_active_raises_when_none_set(self):
        """active raises RegistryError when no active client."""
        reg = ClientRegistry._for_testing()
        with pytest.raises(RegistryError, match="No active client set"):
            reg.active()

    def test_set_active_then_active_returns_adapter(self):
        """active returns adapter after set_active."""
        reg = ClientRegistry._for_testing()
        adapter = DummyAdapter()
        reg.register(adapter)
        reg.set_active("dummy")
        assert reg.active() is adapter

    def test_set_active_raises_for_unregistered(self):
        """set_active raises RegistryError for unregistered client."""
        reg = ClientRegistry._for_testing()
        with pytest.raises(RegistryError, match="Client 'missing' not registered"):
            reg.set_active("missing")


class TestClientRegistryDetect:
    """Tests for detect method."""

    def setup_method(self):
        ClientRegistry.reset()

    def teardown_method(self):
        ClientRegistry.reset()

    def test_detect_empty_project_root_raises(self):
        """detect raises DetectionError for empty project_root."""
        reg = ClientRegistry._for_testing()
        with pytest.raises(
            DetectionError, match="project_root must be a non-empty string"
        ):
            reg.detect("")

    def test_detect_finds_config_dir(self, tmp_path):
        """detect returns adapter when config dir exists."""
        reg = ClientRegistry._for_testing()
        adapter = DummyAdapter()
        reg.register(adapter)
        (tmp_path / ".dummy").mkdir()
        result = reg.detect(str(tmp_path))
        assert result is adapter
        assert reg.active() is adapter

    def test_detect_no_config_dir_raises(self, tmp_path):
        """detect raises DetectionError when no config dir found."""
        reg = ClientRegistry._for_testing()
        adapter = DummyAdapter()
        reg.register(adapter)
        with pytest.raises(DetectionError, match="No client config directory found"):
            reg.detect(str(tmp_path))

    def test_detect_sets_active(self, tmp_path):
        """detect sets _active to matched adapter name."""
        reg = ClientRegistry._for_testing()
        adapter = DummyAdapter()
        reg.register(adapter)
        (tmp_path / ".dummy").mkdir()
        reg.detect(str(tmp_path))
        assert reg.active() is adapter

    def test_detect_multiple_adapters_first_match_wins(self, tmp_path):
        """detect returns first adapter whose config dir exists."""

        class SecondAdapter(ClientAdapter):
            name = "second"

            def get_capabilities(self):
                return CapabilitySet(agent_config_dir=".second")

            def get_tool_names(self):
                return set()

            def normalize_tool_args(self, tool_name, arguments):
                return dict(arguments)

        reg = ClientRegistry._for_testing()
        first = DummyAdapter()
        second = SecondAdapter()
        reg.register(first)
        reg.register(second)
        (tmp_path / ".second").mkdir()
        result = reg.detect(str(tmp_path))
        assert result is second  # .second exists, .dummy does not
        # Now create .dummy and verify first wins (registration order)
        (tmp_path / ".dummy").mkdir()
        result2 = reg.detect(str(tmp_path))
        assert result2 is first


class TestClientRegistryListClients:
    """Tests for list_clients method."""

    def setup_method(self):
        ClientRegistry.reset()

    def teardown_method(self):
        ClientRegistry.reset()

    def test_list_clients_empty(self):
        """list_clients returns empty list when no clients."""
        reg = ClientRegistry._for_testing()
        assert reg.list_clients() == []

    def test_list_clients_returns_names(self):
        """list_clients returns registered client names."""
        reg = ClientRegistry._for_testing()
        reg.register(DummyAdapter())
        assert reg.list_clients() == ["dummy"]

    def test_list_clients_isolated_from_internal_dict(self):
        """list_clients returns a copy, not the internal dict."""
        reg = ClientRegistry._for_testing()
        reg.register(DummyAdapter())
        result = reg.list_clients()
        result.append("tampered")
        assert reg.list_clients() == ["dummy"]
