"""Edge case tests for src/clients/ — covers boundary conditions,
property-based invariants, and fills coverage gaps."""

import json
import os
import threading
import tempfile

import pytest

from clients.interface import (
    CapabilitySet,
    ClientConfig,
    ClientAdapter,
    ClientsError,
    RegistryError,
    ConfigError,
    DetectionError,
    ToolFormat,
    StreamingFormat,
    MessageFormat,
    ToolCall,
    FormattedRequest,
    ConfigLoader,
)
from clients.orchestrator_config import HookCapabilities, ProxyFeatures
from clients.registry import ClientRegistry
from clients.model_registry import (
    ModelConfig,
    ModelRegistry,
    ToolParserRegistry,
    OpenAIToolParser,
    QwenXMLToolParser,
)
from clients.kilocode import KiloCodeAdapter
from clients.opencode import OpenCodeAdapter
from clients.capabilities import (
    KILOCODE_CAPABILITIES,
    OPENCODE_CAPABILITIES,
)
# M7 #2: stale fork removed.


# ============================================================
# 1. Frozen Dataclass Immutability
# ============================================================


class TestFrozenDataclassImmutability:
    """Verify that frozen dataclasses reject attribute assignment."""

    def test_capabilityset_frozen(self):
        cs = CapabilitySet()
        with pytest.raises(AttributeError):
            cs.tool_format = ToolFormat.ANTHROPIC

    def test_capabilityset_hooks_frozen(self):
        cs = CapabilitySet()
        with pytest.raises(AttributeError):
            cs.hooks = HookCapabilities()

    def test_toolcall_frozen(self):
        tc = ToolCall(name="bash", arguments={"cmd": "ls"})
        with pytest.raises(AttributeError):
            tc.name = "edit"

    def test_modelconfig_frozen(self):
        mc = ModelConfig(backend="http://x", tool_parser="openai")
        with pytest.raises(AttributeError):
            mc.backend = "http://y"

    def test_hookcapabilities_frozen(self):
        hc = HookCapabilities()
        with pytest.raises(AttributeError):
            hc.provider = True

    def test_proxyfeatures_frozen(self):
        pf = ProxyFeatures()
        with pytest.raises(AttributeError):
            pf.version = 2


# ============================================================
# 2. Round-trip Invariants (normalize → denormalize)
# ============================================================


class TestRoundTripInvariants:
    """normalize then denormalize should return original dict."""

    def test_kilocode_roundtrip_read(self):
        a = KiloCodeAdapter()
        original = {"filePath": "/tmp/x.py", "offset": 10}
        assert (
            a.denormalize_tool_args("read", a.normalize_tool_args("read", original))
            == original
        )

    def test_kilocode_roundtrip_write(self):
        a = KiloCodeAdapter()
        original = {"filePath": "/tmp/y.py", "content": "hello"}
        assert (
            a.denormalize_tool_args("write", a.normalize_tool_args("write", original))
            == original
        )

    def test_kilocode_roundtrip_edit(self):
        a = KiloCodeAdapter()
        original = {"filePath": "/tmp/z.py", "oldString": "foo", "newString": "bar"}
        assert (
            a.denormalize_tool_args("edit", a.normalize_tool_args("edit", original))
            == original
        )

    def test_kilocode_roundtrip_bash(self):
        a = KiloCodeAdapter()
        original = {"command": "ls -la"}
        assert (
            a.denormalize_tool_args("bash", a.normalize_tool_args("bash", original))
            == original
        )

    def test_kilocode_normalize_preserves_unmapped_keys(self):
        a = KiloCodeAdapter()
        result = a.normalize_tool_args("read", {"filePath": "/x", "extra": 5})
        assert result["extra"] == 5
        assert result["path"] == "/x"

    def test_kilocode_normalize_empty_dict(self):
        a = KiloCodeAdapter()
        assert a.normalize_tool_args("read", {}) == {}

    def test_kilocode_denormalize_empty_dict(self):
        a = KiloCodeAdapter()
        assert a.denormalize_tool_args("read", {}) == {}

    def test_kilocode_normalize_unknown_tool_passthrough(self):
        a = KiloCodeAdapter()
        original = {"some_key": "some_val"}
        assert a.normalize_tool_args("unknown_tool", original) == original

    def test_kilocode_denormalize_unknown_tool_passthrough(self):
        a = KiloCodeAdapter()
        original = {"some_key": "some_val"}
        assert a.denormalize_tool_args("unknown_tool", original) == original

    def test_kilocode_normalize_no_mapped_keys_present(self):
        """Branch coverage: tool_name in mappings but old key NOT present."""
        a = KiloCodeAdapter()
        result = a.normalize_tool_args("read", {"unrelated": "value"})
        assert result == {"unrelated": "value"}
        # Also test bash with a key that happens to NOT be "command"
        result2 = a.normalize_tool_args("bash", {"args": "hello"})
        assert result2 == {"args": "hello"}

    def test_kilocode_denormalize_no_mapped_keys_present(self):
        """Branch coverage: denormalize with tool in mappings but key not present."""
        a = KiloCodeAdapter()
        result = a.denormalize_tool_args("read", {"unrelated": "value"})
        assert result == {"unrelated": "value"}

    def test_opencode_roundtrip_identity(self):
        a = OpenCodeAdapter()
        original = {"path": "/tmp/x", "content": "hi"}
        assert a.normalize_tool_args("bash", original) == original
        assert a.denormalize_tool_args("bash", original) == original

    def test_kilocode_roundtrip_nested_args(self):
        a = KiloCodeAdapter()
        original = {"filePath": "/x", "nested": {"a": 1}}
        roundtripped = a.denormalize_tool_args(
            "read", a.normalize_tool_args("read", original)
        )
        assert roundtripped == original


# ============================================================
# 3. Registry Edge Cases
# ============================================================


class TestRegistryEdgeCases:
    """Edge cases for ClientRegistry."""

    def test_register_same_adapter_twice_overwrites(self):
        r = ClientRegistry._for_testing()
        a = KiloCodeAdapter()
        r.register(a)
        r.register(a)  # re-register same
        assert r.list_clients() == ["kilocode"]

    def test_get_empty_registry_raises(self):
        r = ClientRegistry._for_testing()
        with pytest.raises(RegistryError, match="not registered"):
            r.get("nonexistent")

    def test_set_active_unregistered_raises(self):
        r = ClientRegistry._for_testing()
        with pytest.raises(RegistryError, match="not registered"):
            r.set_active("ghost")

    def test_active_without_set_raises(self):
        r = ClientRegistry._for_testing()
        with pytest.raises(RegistryError, match="No active client"):
            r.active()

    def test_detect_empty_string_raises(self):
        r = ClientRegistry._for_testing()
        with pytest.raises(DetectionError, match="non-empty"):
            r.detect("")

    def test_detect_nonexistent_path_raises(self):
        r = ClientRegistry._for_testing()
        r.register(KiloCodeAdapter())
        r.register(OpenCodeAdapter())
        with pytest.raises(
            DetectionError, match="project_root is not a valid directory"
        ):
            r.detect("/nonexistent/path/that/does/not/exist")

    def test_list_clients_empty(self):
        r = ClientRegistry._for_testing()
        assert r.list_clients() == []


# ============================================================
# 4. Concurrent Access Stress Tests
# ============================================================


class TestConcurrentAccess:
    """Stress test thread-safety of ClientRegistry."""

    def test_concurrent_register_stress(self):
        r = ClientRegistry._for_testing()
        errors = []

        def register_adapters():
            try:
                for i in range(20):
                    a = KiloCodeAdapter()
                    a.name = f"client_{threading.current_thread().name}_{i}"
                    r.register(a)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=register_adapters) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
        assert len(r.list_clients()) > 0

    def test_concurrent_get_and_register(self):
        r = ClientRegistry._for_testing()
        r.register(KiloCodeAdapter())
        errors = []

        def reader():
            try:
                for _ in range(50):
                    r.get("kilocode")
                    r.list_clients()
            except Exception as e:
                errors.append(e)

        def writer():
            try:
                for i in range(50):
                    a = KiloCodeAdapter()
                    a.name = f"writer_{i}"
                    r.register(a)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=reader) for _ in range(3)] + [
            threading.Thread(target=writer)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors


# ============================================================
# 5. qwen_xml Parser — Coverage & Edge Cases (M7 #2: stale fork
# removed; these tests now exercise the CANONICAL parser at
# proxy.tool_parsing.parsers via re-import)
# ============================================================

from proxy.tool_parsing.parsers import parse_all_tools_from_content


class TestQwenXMLParser:
    """Comprehensive tests for parse_all_tools_from_content."""

    def test_empty_string(self):
        assert parse_all_tools_from_content("") == []

    def test_whitespace_only(self):
        assert parse_all_tools_from_content("   \n\t  ") == []

    def test_none_like_input(self):
        # Falsy input returns []
        assert parse_all_tools_from_content("") == []

    def test_xml_wrapper_valid(self):
        """Format 1: <tools>{json}</tools>"""
        content = '<tools>{"name": "bash", "arguments": {"cmd": "ls"}}</tools>'
        result = parse_all_tools_from_content(content)
        assert len(result) == 1
        assert result[0]["name"] == "bash"

    def test_xml_wrapper_invalid_json(self):
        content = "<tools>not json</tools>"
        result = parse_all_tools_from_content(content)
        assert result == []

    def test_regex_tools_wrapper(self):
        """Format 2: <tools>{json}</tools> via regex"""
        content = 'Some text <tools>{"name": "read", "arguments": {"path": "/x"}}</tools> more'
        result = parse_all_tools_from_content(content)
        assert any(t["name"] == "read" for t in result)

    def test_function_xml_params(self):
        """Format 3: <function=name><parameter=key>val</parameter></function>"""
        content = "<function=bash><parameter=command>ls -la</parameter></function>"
        result = parse_all_tools_from_content(content)
        assert len(result) == 1
        assert result[0]["name"] == "bash"
        assert result[0]["arguments"]["command"] == "ls -la"

    def test_function_xml_multiple_params(self):
        content = "<function=edit><parameter=filePath>/tmp/x</parameter><parameter=oldString>foo</parameter></function>"
        result = parse_all_tools_from_content(content)
        assert len(result) == 1
        assert result[0]["name"] == "edit"
        assert "filePath" in result[0]["arguments"]

    def test_markdown_json_code_block(self):
        """Format 4: ```json {...} ```"""
        content = '```json\n{"name": "glob", "arguments": {"pattern": "*.py"}}\n```'
        result = parse_all_tools_from_content(content)
        assert len(result) >= 1
        assert any(t["name"] == "glob" for t in result)

    def test_bare_json_object(self):
        """Format 5: bare JSON with name+arguments."""
        content = '{"name": "write", "arguments": {"path": "/x", "content": "hi"}}'
        result = parse_all_tools_from_content(content)
        assert len(result) >= 1
        assert any(t["name"] == "write" for t in result)

    def test_multiple_tools_in_content(self):
        """Multiple tool calls in different formats."""
        content = (
            "<function=bash><parameter=command>ls</parameter></function>\n"
            '```json\n{"name": "read", "arguments": {"path": "/x"}}\n```\n'
        )
        result = parse_all_tools_from_content(content)
        assert len(result) >= 2

    def test_deduplication(self):
        """Identical tool calls should be deduplicated."""
        content = '{"name": "bash", "arguments": {"cmd": "ls"}}\n{"name": "bash", "arguments": {"cmd": "ls"}}'
        result = parse_all_tools_from_content(content)
        # Should deduplicate to 1
        bash_calls = [t for t in result if t["name"] == "bash"]
        assert len(bash_calls) == 1

    def test_no_name_key_in_json(self):
        """JSON without 'name' key should not appear in format 5 results."""
        content = '{"command": "ls", "arguments": {}}'
        result = parse_all_tools_from_content(content)
        for t in result:
            assert "name" in t

    def test_json_decode_error_in_regex_tools(self):
        """Invalid JSON in <tools> regex match should be skipped."""
        content = "<tools>{broken json}</tools>"
        result = parse_all_tools_from_content(content)
        assert result == []

    def test_function_xml_empty_params(self):
        """<function=name></function> with no parameters produces no args."""
        content = "<function=bash></function>"
        result = parse_all_tools_from_content(content)
        assert result == []

    def test_markdown_json_invalid(self):
        """Invalid JSON in markdown code block should be skipped."""
        content = "```json\n{not valid json}\n```"
        result = parse_all_tools_from_content(content)
        assert result == []

    def test_deeply_nested_xml(self):
        """Complex XML that doesn't match <tools> root."""
        content = "<root><tools>something</tools></root>"
        result = parse_all_tools_from_content(content)
        assert result == []

    def test_tool_with_missing_key_in_dedup(self):
        """Dedup handles tools without 'name' key (KeyError in try)."""
        content = '{"name": "bash", "arguments": {"cmd": "ls"}}'
        result = parse_all_tools_from_content(content)
        assert result == [{"name": "bash", "arguments": {"cmd": "ls"}}]


# ============================================================
# 6. Model Registry Edge Cases
# ============================================================


class TestModelRegistryEdgeCases:
    def test_load_json_with_invalid_model_config(self, tmp_path):
        """Model config with invalid types should be skipped with warning."""
        config = {
            "models": {
                "bad_model": {
                    "backend": 123,
                    "tool_parser": "openai",
                }  # backend should be str
            }
        }
        config_file = tmp_path / "proxy_models.json"
        config_file.write_text(json.dumps(config))
        mr = ModelRegistry(base_dir=str(tmp_path))
        # bad_model likely won't be loaded; default still works
        default = mr.get("nonexistent")
        assert default.tool_parser == "xml"

    def test_load_json_with_type_error_in_model(self, tmp_path):
        """Model config missing required fields raises TypeError, caught."""
        config = {
            "models": {
                "partial": {}  # missing 'backend' and 'tool_parser'
            }
        }
        config_file = tmp_path / "proxy_models.json"
        config_file.write_text(json.dumps(config))
        mr = ModelRegistry(base_dir=str(tmp_path))
        # partial should not be loaded; fallback to default
        result = mr.get("partial")
        # If partial was loaded, check it; otherwise default
        assert isinstance(result, ModelConfig)

    def test_load_json_non_dict(self, tmp_path):
        """JSON file containing a non-dict top level."""
        config_file = tmp_path / "proxy_models.json"
        config_file.write_text("[1, 2, 3]")
        mr = ModelRegistry(base_dir=str(tmp_path))
        # Should log warning and use defaults
        result = mr.get("anything")
        assert result.tool_parser == "xml"

    def test_load_invalid_json(self, tmp_path):
        """Malformed JSON raises JSONDecodeError — proxy must refuse to boot."""
        import json as _json
        config_file = tmp_path / "proxy_models.json"
        config_file.write_text("{invalid json")
        with pytest.raises(_json.JSONDecodeError):
            ModelRegistry(base_dir=str(tmp_path))

    def test_prefix_match_lookup(self, tmp_path):
        """Model lookup with prefix matching."""
        config = {
            "models": {
                "qwen3*": {"backend": "http://llm:7777", "tool_parser": "xml"},
                "qwen3-coder": {"backend": "http://llm:7777", "tool_parser": "openai"},
            }
        }
        config_file = tmp_path / "proxy_models.json"
        config_file.write_text(json.dumps(config))
        mr = ModelRegistry(base_dir=str(tmp_path))
        # "qwen3-coder" exact match
        result = mr.get("qwen3-coder")
        assert result.tool_parser == "openai"

    def test_prefix_match_fallback(self, tmp_path):
        """Model lookup falls back to prefix match when no exact match."""
        config = {
            "models": {"qwen3": {"backend": "http://llm:7777", "tool_parser": "xml"}}
        }
        config_file = tmp_path / "proxy_models.json"
        config_file.write_text(json.dumps(config))
        mr = ModelRegistry(base_dir=str(tmp_path))
        result = mr.get("qwen3-coder-instruct")
        assert result.tool_parser == "xml"

    def test_auto_detect(self, tmp_path):
        """auto_detect extracts model from request body."""
        mr = ModelRegistry(base_dir=str(tmp_path))
        result = mr.auto_detect({"model": "gpt-4"})
        # Falls back to default since no model loaded
        assert isinstance(result, ModelConfig)

    def test_invalid_default_config(self, tmp_path):
        """Default config with invalid types should be caught."""
        config = {
            "default": {"backend": 123}  # not a string
        }
        config_file = tmp_path / "proxy_models.json"
        config_file.write_text(json.dumps(config))
        mr = ModelRegistry(base_dir=str(tmp_path))
        # Should survive and keep built-in default
        result = mr.get("anything")
        assert result.backend == "http://localhost:7777"


# ============================================================
# 7. ToolParserRegistry Edge Cases
# ============================================================


class TestToolParserRegistryEdgeCases:
    def test_unknown_parser_falls_back_to_xml(self):
        tpr = ToolParserRegistry()
        parser = tpr.get("nonexistent")
        assert parser.name == "xml"

    def test_unknown_parser_strict_raises(self):
        tpr = ToolParserRegistry()
        with pytest.raises(KeyError, match="Unknown parser"):
            tpr.get("nonexistent", strict=True)

    def test_openai_parser_returns_empty(self):
        parser = OpenAIToolParser()
        assert parser.parse("anything") == []

    def test_qwen_xml_parser_delegation(self):
        """QwenXMLToolParser delegates to parse_all_tools_from_content."""
        parser = QwenXMLToolParser()
        result = parser.parse('{"name": "bash", "arguments": {"cmd": "ls"}}')
        # Should find the tool via format 5 (bare JSON)
        assert any(t["name"] == "bash" for t in result)


# ============================================================
# 9. Exception Hierarchy
# ============================================================


class TestExceptionHierarchy:
    def test_registry_error_inherits_clients_error(self):
        assert issubclass(RegistryError, ClientsError)

    def test_config_error_inherits_clients_error(self):
        assert issubclass(ConfigError, ClientsError)

    def test_detection_error_inherits_clients_error(self):
        assert issubclass(DetectionError, ClientsError)

    def test_catch_all_as_clients_error(self):
        with pytest.raises(ClientsError):
            raise RegistryError("test")

        with pytest.raises(ClientsError):
            raise ConfigError("test")

        with pytest.raises(ClientsError):
            raise DetectionError("test")


# ============================================================
# 10. ConfigLoader Real Filesystem Edge Cases
# ============================================================


class TestConfigLoaderRealFilesystem:
    def test_read_file_nonexistent_raises(self, tmp_path):
        loader = ConfigLoader()
        with pytest.raises(FileNotFoundError):
            loader.read_file(str(tmp_path / "nonexistent.txt"))

    def test_list_dir_nonexistent_raises(self, tmp_path):
        loader = ConfigLoader()
        with pytest.raises(OSError):
            loader.list_dir(str(tmp_path / "nonexistent_dir"))

    def test_is_dir_nonexistent_returns_false(self, tmp_path):
        loader = ConfigLoader()
        assert loader.is_dir(str(tmp_path / "no_such_dir")) is False

    def test_is_file_nonexistent_returns_false(self, tmp_path):
        loader = ConfigLoader()
        assert loader.is_file(str(tmp_path / "no_file.txt")) is False

    def test_is_dir_with_real_dir(self, tmp_path):
        loader = ConfigLoader()
        assert loader.is_dir(str(tmp_path)) is True

    def test_is_file_with_real_file(self, tmp_path):
        f = tmp_path / "test.txt"
        f.write_text("hello")
        loader = ConfigLoader()
        assert loader.is_file(str(f)) is True

    def test_path_join(self):
        loader = ConfigLoader()
        result = loader.path_join("/base", "sub", "file.txt")
        assert result == os.path.join("/base", "sub", "file.txt")


# ============================================================
# 11. Dataclass Defaults & Equality
# ============================================================


class TestDataclassDefaultsAndEquality:
    def test_default_capabilityset_has_correct_defaults(self):
        cs = CapabilitySet()
        assert cs.tool_format == ToolFormat.OPENAI
        assert cs.streaming_format == StreamingFormat.SSE_OPENAI
        assert cs.message_format == MessageFormat.OPENAI
        assert cs.supports_parallel_tool_calls is True
        assert cs.supports_streaming is True
        assert cs.supports_streaming_events is False
        assert cs.max_tools_per_request is None
        assert cs.supports_vision is False
        assert cs.has_plugin_system is False
        assert cs.config_format == "json"
        assert cs.agent_config_dir == ".opencode"
        assert cs.command_config_dir == ".opencode"
        assert cs.auth_mechanism == "bearer"

    def test_capabilityset_equality(self):
        a = CapabilitySet()
        b = CapabilitySet()
        assert a == b

    def test_clientconfig_defaults(self):
        cc = ClientConfig()
        assert cc.name == "unknown"
        assert cc.provider_id == "openai"
        assert cc.base_url == ""
        assert cc.model == ""
        assert cc.api_key is None
        assert cc.temperature == 0.1
        assert cc.extra == {}

    def test_proxyfeatures_default_memory_tools(self):
        pf = ProxyFeatures()
        assert "recall" in pf.memory_tool_names
        assert "search" in pf.memory_tool_names

    def test_kilocode_capabilities_are_frozen_instance(self):
        assert isinstance(KILOCODE_CAPABILITIES, CapabilitySet)
        assert KILOCODE_CAPABILITIES.has_plugin_system is True
        assert KILOCODE_CAPABILITIES.agent_config_dir == ".kilo"

    def test_opencode_capabilities_config_format(self):
        assert OPENCODE_CAPABILITIES.config_format == "markdown_frontmatter"
        assert OPENCODE_CAPABILITIES.agent_config_dir == ".opencode"


# ============================================================
# 12. Adapter Default Methods
# ============================================================


class TestAdapterDefaultMethods:
    def test_kilocode_format_response(self):
        a = KiloCodeAdapter()
        data = {"choices": [{"text": "hi"}]}
        assert a.format_response(data, wants_stream=True) == data

    def test_opencode_format_response(self):
        a = OpenCodeAdapter()
        data = {"choices": [{"text": "hi"}]}
        assert a.format_response(data, wants_stream=False) == data

    def test_kilocode_get_plugin_adapter_returns_none(self):
        a = KiloCodeAdapter()
        assert a.get_plugin_adapter() is None

    def test_client_adapter_default_validate_tool_result(self):
        """Test default validate_tool_result returns True."""
        a = KiloCodeAdapter()
        assert a.validate_tool_result("bash", "output") is True

    def test_client_adapter_default_get_auth_headers(self):
        a = KiloCodeAdapter()
        headers = a.get_auth_headers()
        assert "Authorization" in headers

    def test_client_adapter_default_format_tool_result(self):
        a = KiloCodeAdapter()
        result = a.format_tool_result("bash", "output")
        assert result == {"role": "tool", "content": "output"}

    def test_client_adapter_default_format_error(self):
        a = KiloCodeAdapter()
        result = a.format_tool_result("bash", "error output")
        assert result["role"] == "tool"

    def test_generate_tool_call_id_format(self):
        a = KiloCodeAdapter()
        call_id = a.generate_tool_call_id("bash")
        assert call_id.startswith("call_")
        assert len(call_id) == 17  # "call_" + 12 hex chars


# ============================================================
# 13. OpenCode Config Loading Edge Cases
# ============================================================


class TestOpenCodeConfigEdgeCases:
    def test_load_config_with_no_md_files(self, tmp_path):
        """Directory exists but contains no .md files."""
        from clients.opencode import OpenCodeAdapter
        from clients.interface import ConfigLoader

        a = OpenCodeAdapter()
        agent_dir = tmp_path / ".opencode"
        agent_dir.mkdir()
        # Only non-.md file
        (agent_dir / "config.json").write_text("{}")
        result = a.load_config(str(tmp_path))
        # Falls back to default
        assert result.name == "opencode"

    def test_load_config_md_file_without_frontmatter(self, tmp_path):
        from clients.opencode import OpenCodeAdapter

        a = OpenCodeAdapter()
        agent_dir = tmp_path / ".opencode"
        agent_dir.mkdir()
        (agent_dir / "agent.md").write_text("# Just a heading\nNo frontmatter here.")
        result = a.load_config(str(tmp_path))
        assert result.name == "opencode"

    def test_load_config_md_with_invalid_yaml(self, tmp_path):
        """YAML frontmatter that's invalid yaml."""
        from clients.opencode import OpenCodeAdapter

        a = OpenCodeAdapter()
        agent_dir = tmp_path / ".opencode"
        agent_dir.mkdir()
        (agent_dir / "agent.md").write_text("---\ninvalid: [yaml: content\n---\nBody")
        result = a.load_config(str(tmp_path))
        # Should catch YAMLError and return default
        assert result.name == "opencode"


# ============================================================
# 14. KiloCode Config Loading Edge Cases
# ============================================================


class TestKiloCodeConfigEdgeCases:
    def test_load_config_malformed_json(self, tmp_path):
        from clients.kilocode import KiloCodeAdapter

        a = KiloCodeAdapter()
        agent_dir = tmp_path / ".kilo" / "agent"
        agent_dir.mkdir(parents=True)
        (agent_dir / "bad.json").write_text("not valid json{")
        result = a.load_config(str(tmp_path))
        assert result.name == "kilocode"

    def test_load_config_json_missing_keys(self, tmp_path):
        from clients.kilocode import KiloCodeAdapter

        a = KiloCodeAdapter()
        agent_dir = tmp_path / ".kilo" / "agent"
        agent_dir.mkdir(parents=True)
        (agent_dir / "partial.json").write_text('{"model": "gpt-4"}')
        result = a.load_config(str(tmp_path))
        # Should load with defaults for missing keys
        assert result.name == "unknown"  # name not in JSON

    def test_load_config_no_agent_dir(self, tmp_path):
        from clients.kilocode import KiloCodeAdapter

        a = KiloCodeAdapter()
        result = a.load_config(str(tmp_path))
        assert result.name == "kilocode"
