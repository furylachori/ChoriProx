# tests/unit/clients/test_interface.py
"""Unit tests for clients.interface."""

import json
import uuid
from abc import ABC
from dataclasses import FrozenInstanceError
from unittest.mock import MagicMock, patch

import pytest

from clients.interface import (
    ToolFormat,
    StreamingFormat,
    MessageFormat,
    ClientsError,
    RegistryError,
    ConfigError,
    DetectionError,
    CapabilitySet,
    ClientConfig,
    ConfigLoader,
    ClientAdapter,
    ToolCall,
    FormattedRequest,
)
from clients.orchestrator_config import HookCapabilities


class TestEnums:
    """Tests for interface enums."""

    def test_tool_format_members(self):
        """ToolFormat has expected members."""
        assert ToolFormat.OPENAI is not None
        assert ToolFormat.ANTHROPIC is not None
        assert ToolFormat.CUSTOM is not None

    def test_streaming_format_members(self):
        """StreamingFormat has expected members."""
        assert StreamingFormat.SSE_OPENAI is not None
        assert StreamingFormat.SSE_ANTHROPIC is not None
        assert StreamingFormat.JSON is not None
        assert StreamingFormat.CUSTOM is not None

    def test_message_format_members(self):
        """MessageFormat has expected members."""
        assert MessageFormat.OPENAI is not None
        assert MessageFormat.ANTHROPIC is not None
        assert MessageFormat.CUSTOM is not None


class TestExceptionHierarchy:
    """Tests for exception hierarchy."""

    def test_clients_error_is_exception(self):
        """ClientsError is an Exception subclass."""
        assert issubclass(ClientsError, Exception)

    def test_registry_error_is_clients_error(self):
        """RegistryError is a ClientsError subclass."""
        assert issubclass(RegistryError, ClientsError)

    def test_config_error_is_clients_error(self):
        """ConfigError is a ClientsError subclass."""
        assert issubclass(ConfigError, ClientsError)

    def test_detection_error_is_clients_error(self):
        """DetectionError is a ClientsError subclass."""
        assert issubclass(DetectionError, ClientsError)

    def test_registry_error_message(self):
        """RegistryError carries message correctly."""
        err = RegistryError("not found")
        assert str(err) == "not found"

    def test_detection_error_message(self):
        """DetectionError carries message correctly."""
        err = DetectionError("no client")
        assert str(err) == "no client"


class TestCapabilitySet:
    """Tests for CapabilitySet frozen dataclass."""

    def test_default_instantiation(self):
        """CapabilitySet instantiates with all defaults."""
        caps = CapabilitySet()
        assert caps.tool_format == ToolFormat.OPENAI
        assert caps.streaming_format == StreamingFormat.SSE_OPENAI
        assert caps.message_format == MessageFormat.OPENAI
        assert caps.supports_parallel_tool_calls is True
        assert caps.supports_streaming is True
        assert caps.supports_streaming_events is False
        assert caps.max_tools_per_request is None
        assert caps.supports_vision is False
        assert caps.has_plugin_system is False
        assert isinstance(caps.hooks, HookCapabilities)
        assert caps.config_format == "json"
        assert caps.agent_config_dir == ".opencode"
        assert caps.command_config_dir == ".opencode"
        assert caps.auth_mechanism == "bearer"

    def test_custom_instantiation(self):
        """CapabilitySet accepts custom values."""
        caps = CapabilitySet(
            tool_format=ToolFormat.ANTHROPIC,
            supports_vision=True,
        )
        assert caps.tool_format == ToolFormat.ANTHROPIC
        assert caps.supports_vision is True

    def test_equality(self):
        """Two identical CapabilitySets are equal."""
        a = CapabilitySet(supports_vision=True)
        b = CapabilitySet(supports_vision=True)
        c = CapabilitySet(supports_vision=False)
        assert a == b
        assert a != c

    def test_immutability(self):
        """CapabilitySet is frozen and raises on mutation."""
        caps = CapabilitySet()
        with pytest.raises(FrozenInstanceError):
            caps.supports_vision = True

    def test_hooks_default_factory(self):
        """Default hooks factory produces new HookCapabilities each time."""
        a = CapabilitySet()
        b = CapabilitySet()
        assert a.hooks == b.hooks
        assert a.hooks is not b.hooks


class TestClientConfig:
    """Tests for ClientConfig dataclass."""

    def test_default_instantiation(self):
        """ClientConfig instantiates with correct defaults."""
        cfg = ClientConfig()
        assert cfg.name == "unknown"
        assert cfg.provider_id == "openai"
        assert cfg.base_url == ""
        assert cfg.model == ""
        assert cfg.api_key is None
        assert cfg.temperature == 0.1
        assert cfg.extra == {}

    def test_custom_instantiation(self):
        """ClientConfig accepts custom values."""
        cfg = ClientConfig(name="test", model="gpt-4")
        assert cfg.name == "test"
        assert cfg.model == "gpt-4"

    def test_api_key_not_in_repr(self):
        """api_key is excluded from repr."""
        cfg = ClientConfig(api_key="secret")
        assert "secret" not in repr(cfg)

    def test_mutability(self):
        """ClientConfig is mutable (not frozen)."""
        cfg = ClientConfig()
        cfg.name = "changed"
        assert cfg.name == "changed"

    def test_equality(self):
        """Two identical ClientConfigs are equal."""
        a = ClientConfig(name="test")
        b = ClientConfig(name="test")
        c = ClientConfig(name="other")
        assert a == b
        assert a != c


class TestConfigLoader:
    """Tests for ConfigLoader filesystem abstraction."""

    def test_instantiation(self):
        """ConfigLoader instantiates without arguments."""
        loader = ConfigLoader()
        assert isinstance(loader, ConfigLoader)

    def test_list_dir(self, tmp_path):
        """list_dir returns directory contents."""
        (tmp_path / "a.txt").touch()
        (tmp_path / "b.txt").touch()
        loader = ConfigLoader()
        result = loader.list_dir(str(tmp_path))
        assert "a.txt" in result
        assert "b.txt" in result

    def test_list_dir_error(self, tmp_path):
        """list_dir raises on non-existent path."""
        loader = ConfigLoader()
        with pytest.raises(OSError):
            loader.list_dir(str(tmp_path / "does_not_exist"))

    def test_read_file(self, tmp_path):
        """read_file returns file contents."""
        path = tmp_path / "test.txt"
        path.write_text("hello")
        loader = ConfigLoader()
        assert loader.read_file(str(path)) == "hello"

    def test_read_file_not_found(self, tmp_path):
        """read_file raises on missing file."""
        loader = ConfigLoader()
        with pytest.raises(OSError):
            loader.read_file(str(tmp_path / "missing.txt"))

    def test_is_dir_true(self, tmp_path):
        """is_dir returns True for directories."""
        loader = ConfigLoader()
        assert loader.is_dir(str(tmp_path)) is True

    def test_is_dir_false(self, tmp_path):
        """is_dir returns False for files."""
        path = tmp_path / "file.txt"
        path.touch()
        loader = ConfigLoader()
        assert loader.is_dir(str(path)) is False

    def test_is_file_true(self, tmp_path):
        """is_file returns True for files."""
        path = tmp_path / "file.txt"
        path.touch()
        loader = ConfigLoader()
        assert loader.is_file(str(path)) is True

    def test_is_file_false(self, tmp_path):
        """is_file returns False for directories."""
        loader = ConfigLoader()
        assert loader.is_file(str(tmp_path)) is False

    def test_path_join(self):
        """path_join joins path parts."""
        loader = ConfigLoader()
        result = loader.path_join("a", "b", "c")
        assert result == "a/b/c" or result == "a\\b\\c"


class ConcreteClientAdapter(ClientAdapter):
    """Minimal concrete implementation for testing ABC."""

    name = "concrete"

    def get_capabilities(self):
        return CapabilitySet()

    def get_tool_names(self):
        return {"tool_a"}

    def normalize_tool_args(self, tool_name, arguments):
        return dict(arguments)


class TestClientAdapterABC:
    """Tests for ClientAdapter abstract base class."""

    def test_cannot_instantiate_abc(self):
        """ClientAdapter cannot be instantiated directly."""
        with pytest.raises(TypeError):
            ClientAdapter()

    def test_missing_abstract_methods_raises(self):
        """Subclass missing abstract methods cannot be instantiated."""

        class Partial(ClientAdapter):
            name = "partial"

            def get_capabilities(self):
                return CapabilitySet()

            def get_tool_names(self):
                return set()

            # Missing normalize_tool_args

        with pytest.raises(TypeError):
            Partial()

    def test_concrete_methods_exist(self):
        """Concrete adapter has all expected methods."""
        adapter = ConcreteClientAdapter()
        assert adapter.name == "concrete"
        assert adapter.denormalize_tool_args("x", {"a": 1}) == {"a": 1}
        assert adapter.format_response({"key": "val"}, True) == {"key": "val"}
        assert adapter.format_sse_chunk({"delta": "hi"}) == 'data: {"delta": "hi"}\n\n'
        assert (
            adapter.get_system_prompt({})
            == "You are an AI assistant with access to tools."
        )
        assert adapter.format_tool_result("tool", "ok") == {
            "role": "tool",
            "content": "ok",
        }
        assert adapter.get_plugin_adapter() is None
        assert adapter.validate_tool_result("tool", "ok") is True
        assert adapter.get_auth_headers() == {"Authorization": "Bearer not-needed"}

    def test_generate_tool_call_id_format(self):
        """generate_tool_call_id produces correct prefix and length."""
        adapter = ConcreteClientAdapter()
        tid = adapter.generate_tool_call_id("tool")
        assert tid.startswith("call_")
        assert len(tid) == 17  # "call_" + 12 hex chars

    def test_generate_tool_call_id_unique(self):
        """generate_tool_call_id produces unique IDs."""
        adapter = ConcreteClientAdapter()
        ids = {adapter.generate_tool_call_id("t") for _ in range(100)}
        assert len(ids) == 100

    def test_format_error(self):
        """format_error wraps exception in dict."""
        adapter = ConcreteClientAdapter()
        err = ValueError("boom")
        result = adapter.format_error(err)
        assert result["role"] == "tool"
        assert "boom" in result["content"]

    def test_load_config_default(self):
        """load_config returns default ClientConfig."""
        adapter = ConcreteClientAdapter()
        cfg = adapter.load_config("/fake/path")
        assert isinstance(cfg, ClientConfig)
        assert cfg.name == "concrete"

    def test_load_config_with_mock_loader(self):
        """load_config can accept a mocked ConfigLoader."""
        adapter = ConcreteClientAdapter()
        loader = MagicMock(spec=ConfigLoader)
        cfg = adapter.load_config("/fake/path", loader=loader)
        assert cfg.name == "concrete"

    def test_format_sse_chunk_empty_dict(self):
        """format_sse_chunk serializes empty dict correctly."""
        adapter = ConcreteClientAdapter()
        result = adapter.format_sse_chunk({})
        assert result == "data: {}\n\n"

    def test_format_tool_result_converts_object_to_string(self):
        """format_tool_result stringifies non-string results."""
        adapter = ConcreteClientAdapter()
        result = adapter.format_tool_result("tool", {"key": "val"})
        assert result["role"] == "tool"
        assert "key" in result["content"]

    def test_validate_tool_result_any_value(self):
        """validate_tool_result always returns True."""
        adapter = ConcreteClientAdapter()
        assert adapter.validate_tool_result("tool", None) is True
        assert adapter.validate_tool_result("tool", {}) is True

    def test_denormalize_tool_args_returns_copy(self):
        """denormalize_tool_args returns a shallow copy."""
        adapter = ConcreteClientAdapter()
        original = {"a": 1}
        result = adapter.denormalize_tool_args("x", original)
        assert result == original
        assert result is not original


class TestToolCall:
    """Tests for ToolCall frozen dataclass."""

    def test_instantiation(self):
        """ToolCall instantiates with name and arguments."""
        tc = ToolCall(name="read", arguments={"path": "/tmp"})
        assert tc.name == "read"
        assert tc.arguments == {"path": "/tmp"}

    def test_equality(self):
        """Two identical ToolCalls are equal."""
        a = ToolCall(name="read", arguments={"path": "/tmp"})
        b = ToolCall(name="read", arguments={"path": "/tmp"})
        c = ToolCall(name="write", arguments={"path": "/tmp"})
        assert a == b
        assert a != c

    def test_immutability(self):
        """ToolCall is frozen and raises on mutation."""
        tc = ToolCall(name="read", arguments={})
        with pytest.raises(FrozenInstanceError):
            tc.name = "write"


class TestFormattedRequest:
    """Tests for FormattedRequest dataclass."""

    def test_instantiation_defaults(self):
        """FormattedRequest instantiates with defaults."""
        fr = FormattedRequest(messages=[])
        assert fr.messages == []
        assert fr.system == ""
        assert fr.extra_headers == {}

    def test_instantiation_with_values(self):
        """FormattedRequest accepts all fields."""
        fr = FormattedRequest(
            messages=[{"role": "user"}],
            system="sys",
            extra_headers={"x": 1},
        )
        assert fr.messages == [{"role": "user"}]
        assert fr.system == "sys"
        assert fr.extra_headers == {"x": 1}

    def test_post_init_raises_on_non_list(self):
        """__post_init__ raises TypeError when messages is not a list."""
        with pytest.raises(TypeError, match="messages must be a list"):
            FormattedRequest(messages="not a list")

    def test_post_init_accepts_list(self):
        """__post_init__ accepts list for messages."""
        fr = FormattedRequest(messages=[1, 2, 3])
        assert fr.messages == [1, 2, 3]

    def test_mutability(self):
        """FormattedRequest is mutable."""
        fr = FormattedRequest(messages=[])
        fr.system = "changed"
        assert fr.system == "changed"
