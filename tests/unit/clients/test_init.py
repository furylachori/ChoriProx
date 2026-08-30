"""Unit tests for clients.__init__ module-level functions."""

import pytest

from clients import (
    register_client,
    get_client,
    detect_client,
    active_client,
    list_clients,
    ClientAdapter,
    CapabilitySet,
    RegistryError,
    DetectionError,
)
from clients.registry import ClientRegistry


class DummyAdapter(ClientAdapter):
    """Minimal concrete adapter for init tests."""

    name = "dummy_init"

    def get_capabilities(self):
        return CapabilitySet(agent_config_dir=".dummy_init")

    def get_tool_names(self):
        return {"tool"}

    def normalize_tool_args(self, tool_name, arguments):
        return dict(arguments)


@pytest.fixture(autouse=True)
def reset_registry():
    """Reset the singleton before and after each test."""
    ClientRegistry.reset()
    yield
    ClientRegistry.reset()


class TestRegisterClient:
    """Tests for register_client."""

    def test_register_client(self):
        """register_client adds adapter to singleton registry."""
        adapter = DummyAdapter()
        register_client(adapter)
        assert list_clients() == ["dummy_init"]

    def test_register_overwrites(self):
        """register_client overwrites adapter with same name."""
        a = DummyAdapter()
        b = DummyAdapter()
        register_client(a)
        register_client(b)
        assert get_client("dummy_init") is b


class TestGetClient:
    """Tests for get_client."""

    def test_get_existing_client(self):
        """get_client returns registered adapter."""
        adapter = DummyAdapter()
        register_client(adapter)
        assert get_client("dummy_init") is adapter

    def test_get_missing_raises(self):
        """get_client raises RegistryError for missing client."""
        with pytest.raises(RegistryError, match="not registered"):
            get_client("missing")


class TestDetectClient:
    """Tests for detect_client."""

    def test_detect_finds_config(self, tmp_path):
        """detect_client finds adapter when config dir exists."""
        adapter = DummyAdapter()
        register_client(adapter)
        (tmp_path / ".dummy_init").mkdir()
        result = detect_client(str(tmp_path))
        assert result is adapter

    def test_detect_raises_when_no_match(self, tmp_path):
        """detect_client raises DetectionError when no config dir found."""
        adapter = DummyAdapter()
        register_client(adapter)
        with pytest.raises(DetectionError, match="No client config directory"):
            detect_client(str(tmp_path))

    def test_detect_empty_project_root(self):
        """detect_client raises DetectionError for empty project_root."""
        with pytest.raises(
            DetectionError, match="project_root must be a non-empty string"
        ):
            detect_client("")


class TestActiveClient:
    """Tests for active_client."""

    def test_active_after_detect(self, tmp_path):
        """active_client returns adapter after detect_client."""
        adapter = DummyAdapter()
        register_client(adapter)
        (tmp_path / ".dummy_init").mkdir()
        detect_client(str(tmp_path))
        assert active_client() is adapter

    def test_active_raises_when_not_set(self):
        """active_client raises RegistryError when no active client."""
        with pytest.raises(RegistryError, match="No active client set"):
            active_client()


class TestListClients:
    """Tests for list_clients."""

    def test_list_empty(self):
        """list_clients returns empty list when nothing registered."""
        assert list_clients() == []

    def test_list_after_register(self):
        """list_clients returns names after registration."""
        register_client(DummyAdapter())
        assert list_clients() == ["dummy_init"]


class TestModuleExports:
    """Tests that __all__ exports are importable."""

    def test_import_client_adapter(self):
        """ClientAdapter is importable from clients."""
        from clients import ClientAdapter as _

        assert _ is ClientAdapter

    def test_import_capability_set(self):
        """CapabilitySet is importable from clients."""
        from clients import CapabilitySet as _

        assert _ is CapabilitySet
