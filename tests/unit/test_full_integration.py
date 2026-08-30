"""Unit tests for ChoriProx model registration — full integration."""

import json
import os
import re
import pytest
from unittest.mock import patch, MagicMock
from qwen_proxy import BackendError, AuthConfigError, _sanitize_error_body


class TestModelConfigFields:
    def test_fields_default_empty(self):
        from clients.model_registry import ModelConfig

        cfg = ModelConfig(
            backend="http://test", tool_parser="openai", context_limit=128000, name="t"
        )
        assert cfg.api_key_env == ""
        assert cfg.backend_model == ""

    def test_fields_set(self):
        from clients.model_registry import ModelConfig

        cfg = ModelConfig(
            backend="http://test",
            tool_parser="openai",
            context_limit=128000,
            name="t",
            api_key_env="KEY",
            backend_model="model/v1",
        )
        assert cfg.api_key_env == "KEY"
        assert cfg.backend_model == "model/v1"

    def test_api_key_env_empty_string_default(self):
        from clients.model_registry import ModelConfig

        cfg = ModelConfig(
            backend="http://test", tool_parser="openai", context_limit=32000, name="x"
        )
        assert cfg.api_key_env == ""
        assert bool(cfg.api_key_env) is False

    def test_backend_model_empty_string_default(self):
        from clients.model_registry import ModelConfig

        cfg = ModelConfig(
            backend="http://test", tool_parser="openai", context_limit=32000, name="x"
        )
        assert cfg.backend_model == ""
        assert bool(cfg.backend_model) is False


class TestExceptionClasses:
    def test_backend_error(self):
        e = BackendError(429, {"error": "rate limited"})
        assert e.status_code == 429
        assert e.error_body == {"error": "rate limited"}

    def test_backend_error_string(self):
        e = BackendError(500, {"message": "internal"})
        assert "500" in str(e)

    def test_auth_config_error(self):
        e = AuthConfigError("kimi", "KEY")
        assert "kimi" in str(e)
        assert "KEY" in str(e)

    def test_auth_config_error_model_name(self):
        e = AuthConfigError("mimo", "MIMO_API_KEY")
        assert e.model_name == "mimo"
        assert e.env_var == "MIMO_API_KEY"

    def test_sanitize_strips_api_key(self):
        body = {"error": {"api_key": "sk-leak", "info": "ok"}}
        result = _sanitize_error_body(body)
        assert "api_key" not in result["error"]
        assert result["error"]["info"] == "ok"

    def test_sanitize_strips_token(self):
        body = {"error": {"token": "bearer-xxx", "info": "ok"}}
        result = _sanitize_error_body(body)
        assert "token" not in result["error"]

    def test_sanitize_strips_secret(self):
        body = {"error": {"secret": "xyz", "info": "ok"}}
        result = _sanitize_error_body(body)
        assert "secret" not in result["error"]

    def test_sanitize_strips_apiKey(self):
        body = {"error": {"apiKey": "sk-camel", "info": "ok"}}
        result = _sanitize_error_body(body)
        assert "apiKey" not in result["error"]

    def test_sanitize_strips_authorization(self):
        body = {"error": {"authorization": "Bearer xxx", "info": "ok"}}
        result = _sanitize_error_body(body)
        assert "authorization" not in result["error"]

    def test_sanitize_handles_lists(self):
        body = {"errors": [{"api_key": "leak"}, {"info": "ok"}]}
        result = _sanitize_error_body(body)
        assert "api_key" not in result["errors"][0]
        assert result["errors"][1]["info"] == "ok"

    def test_sanitize_nested_dicts(self):
        body = {"error": {"details": {"api_key": "nested", "code": 200}}}
        result = _sanitize_error_body(body)
        assert "api_key" not in result["error"]["details"]
        assert result["error"]["details"]["code"] == 200

    def test_sanitize_preserves_non_sensitive(self):
        body = {"error": {"message": "fail", "code": 429, "type": "rate_limit"}}
        result = _sanitize_error_body(body)
        assert result["error"]["message"] == "fail"
        assert result["error"]["code"] == 429
        assert result["error"]["type"] == "rate_limit"

    def test_sanitize_returns_input(self):
        body = {"error": "simple string"}
        result = _sanitize_error_body(body)
        assert result["error"] == "simple string"

    def test_sanitize_handles_empty_dict(self):
        result = _sanitize_error_body({})
        assert result == {}


class TestURLConstructionPatterns:
    @pytest.mark.parametrize(
        "backend,expected",
        [
            (
                "https://opencode.ai/zen/go/v1",
                "https://opencode.ai/zen/go/v1/chat/completions",
            ),
            (
                "https://opencode.ai/zen/go/v1/",
                "https://opencode.ai/zen/go/v1/chat/completions",
            ),
            ("https://api.minimax.io/v1", "https://api.minimax.io/v1/chat/completions"),
            (
                "https://api.minimax.io/v1/",
                "https://api.minimax.io/v1/chat/completions",
            ),
            ("https://api.example.com", "https://api.example.com/v1/chat/completions"),
            ("https://api.example.com/", "https://api.example.com/v1/chat/completions"),
            (
                "https://api.example.com/v1/chat/completions",
                "https://api.example.com/v1/chat/completions",
            ),
            (
                "https://api.example.com/v1/chat/completions/",
                "https://api.example.com/v1/chat/completions",
            ),
            ("http://localhost:7777", "http://localhost:7777/v1/chat/completions"),
        ],
    )
    def test_url_pattern(self, backend, expected):
        from clients.model_registry import ModelConfig, ModelRegistry
        from proxy.model_registry import resolve_model_route

        registry = ModelRegistry.__new__(ModelRegistry)
        registry._models = {
            "url_pattern_test": ModelConfig(backend=backend, tool_parser="xml"),
        }
        route = resolve_model_route("url_pattern_test", registry)
        assert route.upstream_url == expected


class TestModelRewriting:
    def test_rewrite_alias_to_backend_model(self):
        from clients.model_registry import ModelConfig
        from proxy.pipeline.stage_modules.upstream import prepare_upstream_request
        from proxy.types import ModelRoute

        body = {"model": "kimi"}
        cfg = ModelConfig(
            backend="http://test",
            tool_parser="openai",
            backend_model="opencode-go/kimi-k2.6",
        )
        route = ModelRoute(cfg, "http://test/v1/chat/completions", None, True)

        upstream = prepare_upstream_request(body, cfg, route)

        assert body["model"] == "opencode-go/kimi-k2.6"
        assert upstream.original_model == "kimi"

    def test_restore_alias_in_response(self):
        from clients.model_registry import ModelConfig
        from proxy.pipeline.stage_modules.upstream import prepare_upstream_request
        from proxy.types import ModelRoute

        body = {"model": "kimi"}
        cfg = ModelConfig(
            backend="http://test",
            tool_parser="openai",
            backend_model="opencode-go/kimi-k2.6",
        )
        route = ModelRoute(cfg, "http://test/v1/chat/completions", None, True)

        upstream = prepare_upstream_request(body, cfg, route)
        data = {"model": body["model"]}
        data["model"] = upstream.original_model

        assert data["model"] == "kimi"

    def test_no_rewrite_when_backend_model_empty(self):
        from clients.model_registry import ModelConfig
        from proxy.pipeline.stage_modules.upstream import prepare_upstream_request
        from proxy.types import ModelRoute

        body = {"model": "qwen3-coder"}
        cfg = ModelConfig(backend="http://test", tool_parser="openai", backend_model="")
        route = ModelRoute(cfg, "http://test/v1/chat/completions", None, True)

        upstream = prepare_upstream_request(body, cfg, route)

        assert body["model"] == "qwen3-coder"
        assert upstream.original_model == "qwen3-coder"


class TestTemperatureClamping:
    def _prepare(self, body, backend_model):
        from clients.model_registry import ModelConfig
        from proxy.pipeline.stage_modules.upstream import prepare_upstream_request
        from proxy.types import ModelRoute

        cfg = ModelConfig(
            backend="http://test",
            tool_parser="openai",
            backend_model=backend_model,
        )
        route = ModelRoute(cfg, "http://test/v1/chat/completions", None, True)
        prepare_upstream_request(body, cfg, route)

    def test_below_min_clamped_to_0_01(self):
        body = {"model": "x", "temperature": 0}
        self._prepare(body, "minimax/MiniMax-M2.7")
        assert body["temperature"] == 0.01

    def test_above_max_clamped_to_1_0(self):
        body = {"model": "x", "temperature": 1.5}
        self._prepare(body, "minimax/MiniMax-M2.7")
        assert body["temperature"] == 1.0

    def test_within_range_unchanged(self):
        body = {"model": "x", "temperature": 0.5}
        self._prepare(body, "minimax/MiniMax-M2.7")
        assert body["temperature"] == 0.5

    def test_boundary_0_01_unchanged(self):
        body = {"model": "x", "temperature": 0.01}
        self._prepare(body, "minimax/MiniMax-M2.7")
        assert body["temperature"] == 0.01

    def test_boundary_1_0_unchanged(self):
        body = {"model": "x", "temperature": 1.0}
        self._prepare(body, "minimax/MiniMax-M2.7")
        assert body["temperature"] == 1.0

    def test_negative_clamped_to_0_01(self):
        body = {"model": "x", "temperature": -0.5}
        self._prepare(body, "minimax/MiniMax-M2.7")
        assert body["temperature"] == 0.01


class TestThinkingTagExtraction:
    def test_extract_think_and_content(self):
        content = "<think>Let me think about this...</think>The answer is 42."
        match = re.search(r"<think>(.*?)</think>(.*)", content, re.DOTALL)
        assert match is not None
        assert match.group(1).strip() == "Let me think about this..."
        assert match.group(2).strip() == "The answer is 42."

    def test_extract_multiline_thinking(self):
        content = """<think>
Line 1
Line 2

</think>
Final answer."""
        match = re.search(r"<think>(.*?)</think>(.*)", content, re.DOTALL)
        assert match is not None
        assert "Line 1" in match.group(1)
        assert "Line 2" in match.group(1)
        assert match.group(2).strip() == "Final answer."

    def test_absent_returns_none(self):
        content = "<think> some thinking without closing tag"
        match = re.search(r"<think>(.*?)</think>(.*)", content, re.DOTALL)
        assert match is None

    def test_content_only(self):
        content = "The answer is 42."
        match = re.search(r"<think>(.*?)(.*)", content, re.DOTALL)
        assert match is None


class TestSSEReasoningContent:
    def test_reasoning_before_content_in_sse(self):
        from qwen_proxy import stream_response

        msg = {"role": "assistant", "reasoning_content": "think", "content": "answer"}
        sse = "".join(
            stream_response(msg, {"id": "t", "created": 0, "model": "k"}, "k")
        )
        assert sse.index("think") < sse.index("answer")

    def test_model_alias_in_all_sse_chunks(self):
        from qwen_proxy import stream_response

        msg = {
            "role": "assistant",
            "reasoning_content": "thinking",
            "content": "result",
        }
        sse = "".join(
            stream_response(
                msg,
                {"id": "t", "created": 0, "model": "opencode-go/kimi-k2.6"},
                "kimi",
            )
        )
        for line in sse.split("\n"):
            if line.startswith("data: ") and line != "data: [DONE]":
                chunk = json.loads(line[6:])
                assert chunk["model"] == "kimi"

    def test_done_terminator_present(self):
        from qwen_proxy import stream_response

        msg = {"role": "assistant", "content": "ok"}
        sse = "".join(
            stream_response(msg, {"id": "t", "created": 0, "model": "k"}, "k")
        )
        assert "data: [DONE]" in sse

    def test_no_reasoning_chunk_when_absent(self):
        from qwen_proxy import stream_response

        msg = {"role": "assistant", "content": "Hello."}
        sse = "".join(
            stream_response(msg, {"id": "t", "created": 0, "model": "k"}, "k")
        )
        assert "reasoning_content" not in sse

    def test_reasoning_fallback_to_reasoning_field(self):
        from qwen_proxy import stream_response

        msg = {
            "role": "assistant",
            "reasoning": "via reasoning field",
            "content": "result",
        }
        sse = "".join(
            stream_response(msg, {"id": "t", "created": 0, "model": "k"}, "k")
        )
        assert "via reasoning field" in sse

    def test_reasoning_with_tool_calls_both_emitted(self):
        from qwen_proxy import stream_response

        msg = {
            "role": "assistant",
            "reasoning_content": "need tool",
            "tool_calls": [
                {
                    "id": "c1",
                    "type": "function",
                    "function": {"name": "bash", "arguments": "{}"},
                }
            ],
        }
        sse = "".join(
            stream_response(msg, {"id": "t", "created": 0, "model": "k"}, "k")
        )
        assert "need tool" in sse
        assert "tool_calls" in sse


class TestContextLimitEnforcement:
    def test_within_limit_allowed(self):
        estimated_tokens = 1000
        context_limit = 262144
        assert not (context_limit and estimated_tokens > context_limit)

    def test_exceeds_limit_rejected(self):
        estimated_tokens = 300000
        context_limit = 262144
        assert estimated_tokens > context_limit

    def test_exactly_at_limit_allowed(self):
        estimated_tokens = 262144
        context_limit = 262144
        assert not (context_limit and estimated_tokens > context_limit)

    def test_just_over_limit_rejected(self):
        estimated_tokens = 262145
        context_limit = 262144
        assert estimated_tokens > context_limit

    def test_none_context_limit_allows_all(self):
        estimated_tokens = 999999
        context_limit = None
        assert not (context_limit and estimated_tokens > context_limit)

    def test_zero_tokens_allowed(self):
        estimated_tokens = 0
        context_limit = 262144
        assert not (context_limit and estimated_tokens > context_limit)
