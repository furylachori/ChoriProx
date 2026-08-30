# tests/unit/clients/test_model_registry.py
"""Unit tests for clients.model_registry."""

import json
import logging
from abc import ABC
from dataclasses import FrozenInstanceError
from unittest.mock import MagicMock, patch

import pytest

from clients.model_registry import (
    ModelConfig,
    ToolParser,
    OpenAIToolParser,
    QwenXMLToolParser,
    ToolParserRegistry,
    ModelRegistry,
)


class TestModelConfig:
    """Tests for ModelConfig frozen dataclass."""

    def test_instantiation_required_fields(self):
        """ModelConfig requires backend and tool_parser."""
        cfg = ModelConfig(backend="http://localhost", tool_parser="openai")
        assert cfg.backend == "http://localhost"
        assert cfg.tool_parser == "openai"

    def test_default_optional_fields(self):
        """ModelConfig has correct defaults for optional fields."""
        cfg = ModelConfig(backend="http://localhost", tool_parser="openai")
        assert cfg.name == ""
        assert cfg.context_limit == 32000
        assert cfg.supports_vision is False
        assert cfg.supports_parallel_tools is True

    def test_custom_fields(self):
        """ModelConfig accepts all fields."""
        cfg = ModelConfig(
            backend="http://localhost",
            tool_parser="xml",
            name="My Custom Model",
            context_limit=64000,
            supports_vision=True,
            supports_parallel_tools=False,
        )
        assert cfg.name == "My Custom Model"
        assert cfg.context_limit == 64000
        assert cfg.supports_vision is True
        assert cfg.supports_parallel_tools is False

    def test_equality(self):
        """Two identical ModelConfigs are equal."""
        a = ModelConfig(backend="a", tool_parser="t")
        b = ModelConfig(backend="a", tool_parser="t")
        c = ModelConfig(backend="b", tool_parser="t")
        assert a == b
        assert a != c

    def test_immutability(self):
        """ModelConfig is frozen and raises on mutation."""
        cfg = ModelConfig(backend="a", tool_parser="t")
        with pytest.raises(FrozenInstanceError):
            cfg.backend = "b"

    def test_api_key_env_default(self):
        """ModelConfig defaults api_key_env to empty string."""
        cfg = ModelConfig(backend="http://test", tool_parser="openai")
        assert cfg.api_key_env == ""
        assert cfg.backend_model == ""

    def test_api_key_env_set(self):
        """ModelConfig accepts api_key_env and backend_model."""
        cfg = ModelConfig(
            backend="http://test",
            tool_parser="openai",
            name="test",
            api_key_env="OPENCODE_API_KEY",
            backend_model="opencode-go/kimi-k2.6",
        )
        assert cfg.api_key_env == "OPENCODE_API_KEY"
        assert cfg.backend_model == "opencode-go/kimi-k2.6"

    def test_registry_loads_new_fields_from_json(self, tmp_path):
        """ModelRegistry loads api_key_env and backend_model from JSON."""
        config = {
            "models": {
                "kimi": {
                    "backend": "https://opencode.ai/zen/go/v1",
                    "tool_parser": "openai",
                    "context_limit": 262144,
                    "name": "ChoriProx Kimi K2.6",
                    "api_key_env": "OPENCODE_API_KEY",
                    "backend_model": "opencode-go/kimi-k2.6",
                }
            },
            "default": {
                "backend": "http://localhost:7777",
                "tool_parser": "xml",
                "context_limit": 32000,
            },
        }
        path = tmp_path / "proxy_models.json"
        path.write_text(json.dumps(config))
        reg = ModelRegistry(config_path="proxy_models.json", base_dir=str(tmp_path))
        cfg = reg.get("kimi")
        assert cfg.api_key_env == "OPENCODE_API_KEY"
        assert cfg.backend_model == "opencode-go/kimi-k2.6"
        assert cfg.context_limit == 262144

    def test_registry_unknown_fields_filtered(self, tmp_path):
        """ModelRegistry filters unknown fields, new fields are known."""
        config = {
            "models": {
                "kimi": {
                    "backend": "https://opencode.ai/zen/go/v1",
                    "tool_parser": "openai",
                    "context_limit": 262144,
                    "name": "Kimi",
                    "api_key_env": "OPENCODE_API_KEY",
                    "backend_model": "opencode-go/kimi-k2.6",
                    "junk_field": "ignored",
                    "another_unknown": 42,
                }
            },
            "default": {
                "backend": "http://localhost:7777",
                "tool_parser": "xml",
                "context_limit": 32000,
            },
        }
        path = tmp_path / "proxy_models.json"
        path.write_text(json.dumps(config))
        reg = ModelRegistry(config_path="proxy_models.json", base_dir=str(tmp_path))
        cfg = reg.get("kimi")
        assert cfg.backend == "https://opencode.ai/zen/go/v1"
        assert cfg.api_key_env == "OPENCODE_API_KEY"
        assert cfg.backend_model == "opencode-go/kimi-k2.6"


class TestToolParserABC:
    """Tests for ToolParser abstract base class."""

    def test_cannot_instantiate(self):
        """ToolParser cannot be instantiated directly."""
        with pytest.raises(TypeError):
            ToolParser()

    def test_missing_methods_raises(self):
        """Subclass missing abstract methods cannot be instantiated."""

        class Partial(ToolParser):
            name = "partial"

        with pytest.raises(TypeError):
            Partial()

    def test_is_abc(self):
        """ToolParser is an ABC subclass."""
        assert issubclass(ToolParser, ABC)


class TestOpenAIToolParser:
    """Tests for OpenAIToolParser."""

    def test_name(self):
        """OpenAIToolParser has correct name."""
        parser = OpenAIToolParser()
        assert parser.name == "openai"

    def test_parse_returns_empty(self):
        """OpenAIToolParser.parse always returns empty list."""
        parser = OpenAIToolParser()
        assert parser.parse("") == []
        assert parser.parse("anything") == []


class TestQwenXMLToolParser:
    """Tests for QwenXMLToolParser with mocked canonical parser."""

    def test_name(self):
        """QwenXMLToolParser has correct name."""
        parser = QwenXMLToolParser()
        assert parser.name == "xml"

    def test_parse_delegates_to_canonical_parser(self):
        """parse delegates to proxy.tool_parsing.parsers.parse_all_tools_from_content (M7 #2: stale qwen_xml fork removed)."""
        parser = QwenXMLToolParser()
        with patch(
            "proxy.tool_parsing.parsers.parse_all_tools_from_content",
            return_value=[{"name": "tool"}],
        ) as mock:
            result = parser.parse("content")
        mock.assert_called_once_with("content")
        assert result == [{"name": "tool"}]


class TestToolParserRegistry:
    """Tests for ToolParserRegistry."""

    def test_get_openai(self):
        """get returns OpenAIToolParser for 'openai'."""
        registry = ToolParserRegistry()
        parser = registry.get("openai")
        assert isinstance(parser, OpenAIToolParser)

    def test_get_xml(self):
        """get returns QwenXMLToolParser for 'xml'."""
        registry = ToolParserRegistry()
        parser = registry.get("xml")
        assert isinstance(parser, QwenXMLToolParser)

    def test_get_unknown_fallback(self):
        """get falls back to xml for unknown parser name."""
        registry = ToolParserRegistry()
        parser = registry.get("unknown")
        assert isinstance(parser, QwenXMLToolParser)

    def test_get_unknown_strict_raises(self):
        """get raises KeyError in strict mode for unknown parser."""
        registry = ToolParserRegistry()
        with pytest.raises(KeyError, match="Unknown parser: missing"):
            registry.get("missing", strict=True)

    def test_get_unknown_logs_warning(self, caplog):
        """get logs warning when falling back for unknown parser."""
        registry = ToolParserRegistry()
        with caplog.at_level(logging.WARNING):
            registry.get("unknown")
        assert "Unknown parser 'unknown'" in caplog.text


class TestModelRegistry:
    """Tests for ModelRegistry."""

    def test_init_no_config_file(self, tmp_path):
        """ModelRegistry initializes with defaults when config missing."""
        reg = ModelRegistry(base_dir=str(tmp_path))
        assert reg._default.backend == "http://localhost:7777"
        assert reg._default.tool_parser == "xml"
        assert reg._models == {}

    def test_init_loads_config(self, tmp_path):
        """ModelRegistry loads model configs from JSON file."""
        config = {
            "models": {
                "gpt-4": {
                    "backend": "http://api.example.com",
                    "tool_parser": "openai",
                    "context_limit": 8192,
                }
            }
        }
        path = tmp_path / "proxy_models.json"
        path.write_text(json.dumps(config))
        reg = ModelRegistry(config_path="proxy_models.json", base_dir=str(tmp_path))
        cfg = reg.get("gpt-4")
        assert cfg.backend == "http://api.example.com"
        assert cfg.context_limit == 8192

    def test_init_loads_default_config(self, tmp_path):
        """ModelRegistry loads default model config."""
        config = {
            "default": {
                "backend": "http://default.com",
                "tool_parser": "json",
            }
        }
        path = tmp_path / "proxy_models.json"
        path.write_text(json.dumps(config))
        reg = ModelRegistry(base_dir=str(tmp_path))
        assert reg._default.backend == "http://default.com"
        assert reg._default.tool_parser == "json"

    def test_init_invalid_json_raises(self, tmp_path):
        """ModelRegistry raises json.JSONDecodeError on malformed JSON.

        A broken proxy_models.json is a fatal config error — silently
        booting with zero models routes every request to the default
        backend, which fails. The proxy must refuse to start instead.
        """
        import json as _json
        path = tmp_path / "proxy_models.json"
        path.write_text("not json")
        with pytest.raises(_json.JSONDecodeError):
            ModelRegistry(base_dir=str(tmp_path))

    def test_init_non_dict_json_logs_warning(self, tmp_path, caplog):
        """ModelRegistry logs warning when JSON is not a dict."""
        path = tmp_path / "proxy_models.json"
        path.write_text("[1, 2, 3]")
        with caplog.at_level(logging.WARNING):
            reg = ModelRegistry(base_dir=str(tmp_path))
        assert "must be a JSON object" in caplog.text

    def test_init_invalid_model_config_logs_warning(self, tmp_path, caplog):
        """ModelRegistry logs warning for invalid model config fields."""
        config = {
            "models": {
                "bad": {
                    "backend": "http://bad",
                    "tool_parser": "openai",
                    "unknown_field": "ignored",
                }
            }
        }
        path = tmp_path / "proxy_models.json"
        path.write_text(json.dumps(config))
        with caplog.at_level(logging.WARNING):
            reg = ModelRegistry(base_dir=str(tmp_path))
        # Should load successfully since unknown fields are filtered
        cfg = reg.get("bad")
        assert cfg.backend == "http://bad"

    def test_init_invalid_default_config_logs_warning(self, tmp_path, caplog):
        """ModelRegistry logs warning for invalid default config."""
        config = {"default": {"backend": 123}}  # missing required tool_parser
        path = tmp_path / "proxy_models.json"
        path.write_text(json.dumps(config))
        with caplog.at_level(logging.WARNING):
            reg = ModelRegistry(base_dir=str(tmp_path))
        assert "Invalid default model config" in caplog.text

    def test_get_exact_match(self, tmp_path):
        """get returns exact match model config.
        # US-061
        """
        config = {
            "models": {"gpt-4": {"backend": "http://gpt4", "tool_parser": "openai"}}
        }
        path = tmp_path / "proxy_models.json"
        path.write_text(json.dumps(config))
        reg = ModelRegistry(base_dir=str(tmp_path))
        cfg = reg.get("gpt-4")
        assert cfg.backend == "http://gpt4"

    def test_get_prefix_match(self, tmp_path):
        """get returns longest prefix match for keys ending with '*'."""
        config = {
            "models": {
                "gpt-*": {"backend": "http://gpt", "tool_parser": "openai"},
                "gpt-4*": {"backend": "http://gpt4", "tool_parser": "openai"},
            }
        }
        path = tmp_path / "proxy_models.json"
        path.write_text(json.dumps(config))
        reg = ModelRegistry(base_dir=str(tmp_path))
        cfg = reg.get("gpt-4-turbo")
        assert cfg.backend == "http://gpt4"

    def test_get_no_prefix_match_without_star(self, tmp_path):
        """get does NOT prefix-match keys without '*' suffix."""
        config = {
            "models": {
                "gpt-": {"backend": "http://gpt", "tool_parser": "openai"},
            }
        }
        path = tmp_path / "proxy_models.json"
        path.write_text(json.dumps(config))
        reg = ModelRegistry(base_dir=str(tmp_path))
        # "gpt-" without * requires exact match, so "gpt-4-turbo" falls to default
        cfg = reg.get("gpt-4-turbo")
        assert cfg.backend == "http://localhost:7777"  # default

    def test_get_no_match_returns_default(self, tmp_path):
        """get returns default when no match found."""
        config = {
            "models": {"gpt-4": {"backend": "http://gpt4", "tool_parser": "openai"}}
        }
        path = tmp_path / "proxy_models.json"
        path.write_text(json.dumps(config))
        reg = ModelRegistry(base_dir=str(tmp_path))
        cfg = reg.get("unknown-model")
        assert cfg.backend == "http://localhost:7777"

    def test_auto_detect(self, tmp_path):
        """auto_detect extracts model name from request body."""
        config = {
            "models": {"gpt-4": {"backend": "http://gpt4", "tool_parser": "openai"}}
        }
        path = tmp_path / "proxy_models.json"
        path.write_text(json.dumps(config))
        reg = ModelRegistry(base_dir=str(tmp_path))
        cfg = reg.auto_detect({"model": "gpt-4"})
        assert cfg.backend == "http://gpt4"

    def test_auto_detect_empty_model_returns_default(self, tmp_path):
        """auto_detect returns default when model field is empty."""
        reg = ModelRegistry(base_dir=str(tmp_path))
        cfg = reg.auto_detect({})
        assert cfg.backend == "http://localhost:7777"

    def test_list_models_empty(self, tmp_path):
        """list_models returns empty list when no models registered."""
        reg = ModelRegistry(base_dir=str(tmp_path))
        assert reg.list_models() == []

    def test_list_models_returns_registered(self, tmp_path):
        """list_models returns list of (key, config) pairs."""
        config = {
            "models": {"gpt-4": {"backend": "http://gpt4", "tool_parser": "openai"}}
        }
        path = tmp_path / "proxy_models.json"
        path.write_text(json.dumps(config))
        reg = ModelRegistry(base_dir=str(tmp_path))
        models = reg.list_models()
        assert len(models) == 1
        key, cfg = models[0]
        assert key == "gpt-4"
        assert cfg.backend == "http://gpt4"

    def test_get_none_returns_default(self):
        """ModelRegistry.get(None) should return default, not crash."""
        reg = ModelRegistry()
        result = reg.get(None)
        assert result is not None
        assert result.backend == "http://localhost:7777"

    def test_get_default_returns_config(self, tmp_path):
        """get_default returns the default model config."""
        reg = ModelRegistry(base_dir=str(tmp_path))
        default = reg.get_default()
        assert default.backend == "http://localhost:7777"
        assert default.tool_parser == "xml"

    def test_get_default_overridden(self, tmp_path):
        """get_default returns custom default when configured."""
        config = {
            "default": {
                "backend": "http://custom-default.com",
                "tool_parser": "json",
            }
        }
        path = tmp_path / "proxy_models.json"
        path.write_text(json.dumps(config))
        reg = ModelRegistry(base_dir=str(tmp_path))
        default = reg.get_default()
        assert default.backend == "http://custom-default.com"
        assert default.tool_parser == "json"

    def test_invalid_scheme_rejected(self, tmp_path, caplog):
        """ModelRegistry rejects backends with invalid schemes."""
        config = {
            "models": {
                "bad-scheme": {"backend": "ftp://example.com", "tool_parser": "openai"}
            }
        }
        path = tmp_path / "proxy_models.json"
        path.write_text(json.dumps(config))
        with caplog.at_level(logging.WARNING):
            reg = ModelRegistry(base_dir=str(tmp_path))
        assert "Invalid scheme" in caplog.text
        assert "bad-scheme" not in [k for k, _ in reg.list_models()]

    def test_models_list_rejected(self, tmp_path, caplog):
        """ModelRegistry handles 'models' as list gracefully."""
        config = {"models": []}
        path = tmp_path / "proxy_models.json"
        path.write_text(json.dumps(config))
        with caplog.at_level(logging.WARNING):
            reg = ModelRegistry(base_dir=str(tmp_path))
        assert "'models' must be an object" in caplog.text
        assert reg.list_models() == []

    def test_name_field_validated(self, tmp_path, caplog):
        """ModelRegistry validates name field length and pattern."""
        config = {
            "models": {
                "test-model": {
                    "backend": "http://localhost",
                    "tool_parser": "xml",
                    "name": "a" * 129,  # Too long
                }
            }
        }
        path = tmp_path / "proxy_models.json"
        path.write_text(json.dumps(config))
        with caplog.at_level(logging.WARNING):
            reg = ModelRegistry(base_dir=str(tmp_path))
        assert "Invalid name" in caplog.text


def test_name_with_slash_allowed(tmp_path):
    """ModelRegistry allows '/' in name field."""
    config = {
        "models": {
            "model/v2": {
                "backend": "http://localhost",
                "tool_parser": "xml",
                "name": "Model v2",
            }
        }
    }
    path = tmp_path / "proxy_models.json"
    path.write_text(json.dumps(config))
    reg = ModelRegistry(config_path="proxy_models.json", base_dir=str(tmp_path))
    cfg = reg.get("model/v2")
    assert cfg.backend == "http://localhost"


SIX_MODEL_CONFIG = {
    "models": {
        "qwen3-coder": {
            "backend": "http://localhost:7777",
            "tool_parser": "xml",
            "context_limit": 32000,
            "name": "ChoriProx Qwen3 Coder",
        },
        "kimi": {
            "backend": "https://opencode.ai/zen/go/v1",
            "tool_parser": "openai",
            "context_limit": 262144,
            "name": "ChoriProx Kimi K2.6",
            "api_key_env": "OPENCODE_API_KEY",
            "backend_model": "opencode-go/kimi-k2.6",
        },
        "mimo": {
            "backend": "https://token-plan-sgp.xiaomimimo.com/v1",
            "tool_parser": "openai",
            "context_limit": 1048576,
            "name": "ChoriProx MiMo V2.5 Pro",
            "api_key_env": "MIMO_API_KEY",
            "backend_model": "mimo/mimo-v2.5-pro",
        },
        "mini": {
            "backend": "https://api.minimax.io/v1",
            "tool_parser": "openai",
            "context_limit": 204800,
            "name": "ChoriProx MiniMax M2.7",
            "api_key_env": "MINIMAX_API_KEY",
            "backend_model": "minimax/MiniMax-M2.7",
        },
        "glm": {
            "backend": "https://opencode.ai/zen/go/v1",
            "tool_parser": "openai",
            "context_limit": 204800,
            "name": "ChoriProx GLM 5.1",
            "api_key_env": "OPENCODE_API_KEY",
            "backend_model": "opencode-go/glm-5.1",
        },
        "qwen": {
            "backend": "https://opencode.ai/zen/go/v1",
            "tool_parser": "openai",
            "context_limit": 1048576,
            "name": "ChoriProx Qwen 3.6 Plus",
            "api_key_env": "OPENCODE_API_KEY",
            "backend_model": "opencode-go/qwen3.6-plus",
        },
    },
    "default": {
        "backend": "http://localhost:7777",
        "tool_parser": "xml",
        "context_limit": 32000,
    },
}


class TestSixModelLoading:
    """Tests for loading all 6 models from proxy_models.json."""

    def test_all_6_models_loaded(self, tmp_path):
        """All 6 models (qwen3-coder + 5 remote) load correctly."""
        path = tmp_path / "proxy_models.json"
        path.write_text(json.dumps(SIX_MODEL_CONFIG))
        reg = ModelRegistry(config_path="proxy_models.json", base_dir=str(tmp_path))
        models = dict(reg.list_models())
        assert len(models) == 6
        for alias in ("qwen3-coder", "kimi", "mimo", "mini", "glm", "qwen"):
            assert alias in models

    def test_api_key_env_values(self, tmp_path):
        """Verify api_key_env deserialized correctly for all models."""
        path = tmp_path / "proxy_models.json"
        path.write_text(json.dumps(SIX_MODEL_CONFIG))
        reg = ModelRegistry(config_path="proxy_models.json", base_dir=str(tmp_path))
        assert reg.get("kimi").api_key_env == "OPENCODE_API_KEY"
        assert reg.get("mimo").api_key_env == "MIMO_API_KEY"
        assert reg.get("mini").api_key_env == "MINIMAX_API_KEY"
        assert reg.get("glm").api_key_env == "OPENCODE_API_KEY"
        assert reg.get("qwen").api_key_env == "OPENCODE_API_KEY"
        assert reg.get("qwen3-coder").api_key_env == ""

    def test_backend_model_values(self, tmp_path):
        """Verify backend_model deserialized correctly."""
        path = tmp_path / "proxy_models.json"
        path.write_text(json.dumps(SIX_MODEL_CONFIG))
        reg = ModelRegistry(config_path="proxy_models.json", base_dir=str(tmp_path))
        assert reg.get("kimi").backend_model == "opencode-go/kimi-k2.6"
        assert reg.get("mimo").backend_model == "mimo/mimo-v2.5-pro"
        assert reg.get("mini").backend_model == "minimax/MiniMax-M2.7"
        assert reg.get("glm").backend_model == "opencode-go/glm-5.1"
        assert reg.get("qwen").backend_model == "opencode-go/qwen3.6-plus"
        assert reg.get("qwen3-coder").backend_model == ""

    def test_context_limits(self, tmp_path):
        """Verify context limits match plan."""
        path = tmp_path / "proxy_models.json"
        path.write_text(json.dumps(SIX_MODEL_CONFIG))
        reg = ModelRegistry(config_path="proxy_models.json", base_dir=str(tmp_path))
        assert reg.get("kimi").context_limit == 262144
        assert reg.get("mimo").context_limit == 1048576
        assert reg.get("mini").context_limit == 204800
        assert reg.get("glm").context_limit == 204800
        assert reg.get("qwen").context_limit == 1048576
        assert reg.get("qwen3-coder").context_limit == 32000

    def test_backends_correct(self, tmp_path):
        """Verify backend URLs are correct."""
        path = tmp_path / "proxy_models.json"
        path.write_text(json.dumps(SIX_MODEL_CONFIG))
        reg = ModelRegistry(config_path="proxy_models.json", base_dir=str(tmp_path))
        assert reg.get("qwen3-coder").backend == "http://localhost:7777"
        assert reg.get("kimi").backend == "https://opencode.ai/zen/go/v1"
        assert reg.get("mimo").backend == "https://token-plan-sgp.xiaomimimo.com/v1"
        assert reg.get("mini").backend == "https://api.minimax.io/v1"
        assert reg.get("glm").backend == "https://opencode.ai/zen/go/v1"
        assert reg.get("qwen").backend == "https://opencode.ai/zen/go/v1"

    def test_tool_parsers_correct(self, tmp_path):
        """Verify tool parsers are correct."""
        path = tmp_path / "proxy_models.json"
        path.write_text(json.dumps(SIX_MODEL_CONFIG))
        reg = ModelRegistry(config_path="proxy_models.json", base_dir=str(tmp_path))
        assert reg.get("qwen3-coder").tool_parser == "xml"
        assert reg.get("kimi").tool_parser == "openai"
        assert reg.get("mimo").tool_parser == "openai"
        assert reg.get("mini").tool_parser == "openai"
        assert reg.get("glm").tool_parser == "openai"
        assert reg.get("qwen").tool_parser == "openai"


class TestLoadTimeValidation:
    """Tests for load-time validation warnings."""

    def test_remote_model_without_backend_model_warns(self, tmp_path, caplog):
        """Remote model without backend_model set logs a warning."""
        config = {
            "models": {
                "bad": {
                    "backend": "https://api.example.com/v1",
                    "tool_parser": "openai",
                    "api_key_env": "TEST_KEY",
                },
            },
        }
        path = tmp_path / "proxy_models.json"
        path.write_text(json.dumps(config))
        with caplog.at_level(logging.WARNING):
            reg = ModelRegistry(config_path="proxy_models.json", base_dir=str(tmp_path))
        assert "Remote model 'bad' has no backend_model set" in caplog.text

    def test_missing_api_key_env_warns(self, tmp_path, caplog):
        """Model referencing unset env var logs a warning."""
        config = {
            "models": {
                "bad": {
                    "backend": "https://api.example.com/v1",
                    "tool_parser": "openai",
                    "api_key_env": "NOT_SET_ENV_VAR_12345",
                },
            },
        }
        path = tmp_path / "proxy_models.json"
        path.write_text(json.dumps(config))
        with caplog.at_level(logging.WARNING):
            reg = ModelRegistry(config_path="proxy_models.json", base_dir=str(tmp_path))
        assert "NOT_SET_ENV_VAR_12345" in caplog.text


class TestToolParserRegistryPreInstantiation:
    """Tests for ToolParserRegistry pre-instantiation behavior."""

    def test_get_returns_same_instance(self):
        """get returns the same parser instance on repeated calls."""
        registry = ToolParserRegistry()
        parser1 = registry.get("openai")
        parser2 = registry.get("openai")
        assert parser1 is parser2

    def test_all_parsers_preinstantiated(self):
        """All registered parsers are pre-instantiated."""
        registry = ToolParserRegistry()
        for name in ["openai", "xml", "anthropic", "json"]:
            parser = registry.get(name)
            assert parser is not None
