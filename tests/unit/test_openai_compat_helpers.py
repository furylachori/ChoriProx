"""Tests for Tier 3 Batch 1 helpers."""

import re
import time
from datetime import datetime, timezone

import pytest

from proxy.utils import _now_iso_sqlite, _now_iso
from proxy.openai_compat import (
    build_openai_response,
    build_upstream_protocol_error,
    chat_completion_shape_error,
)
from proxy.rate_limit_response import rate_limited_response, rate_limited_error_response


class TestTimestampHelpers:
    def test_now_iso_sqlite_format(self):
        result = _now_iso_sqlite()
        assert re.match(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z", result)

    def test_now_iso_sqlite_no_microseconds(self):
        result = _now_iso_sqlite()
        assert "." not in result

    def test_now_iso_format(self):
        result = _now_iso()
        assert re.match(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d+\+\d{2}:\d{2}", result)

    def test_now_iso_has_timezone(self):
        result = _now_iso()
        assert result.endswith("+00:00")


class TestBuildOpenaiResponse:
    def test_default_fields(self):
        resp = build_openai_response("hello", model="test-model")
        assert resp["id"].startswith("chatcmpl-")
        assert resp["object"] == "chat.completion"
        assert isinstance(resp["created"], int)
        assert resp["model"] == "test-model"
        assert resp["choices"][0]["message"]["content"] == "hello"
        assert resp["choices"][0]["message"]["role"] == "assistant"
        assert resp["choices"][0]["finish_reason"] == "stop"
        assert resp["usage"]["prompt_tokens"] == 0

    def test_with_usage(self):
        usage = {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150}
        resp = build_openai_response("hi", model="m", usage=usage)
        assert resp["usage"] == usage

    def test_custom_finish_reason(self):
        resp = build_openai_response("hi", model="m", finish_reason="length")
        assert resp["choices"][0]["finish_reason"] == "length"

    def test_custom_id_and_created(self):
        resp = build_openai_response("hi", model="m", response_id="chatcmpl-custom", created=1234567890)
        assert resp["id"] == "chatcmpl-custom"
        assert resp["created"] == 1234567890


class TestRateLimitedResponse:
    def test_flat_shape(self):
        body, status = rate_limited_response(42)
        assert body == {"error": "rate_limited", "retry_after": 42}
        assert status == 429

    def test_nested_shape(self):
        body, status = rate_limited_error_response("Rate limit exceeded for ^^imp", 30)
        assert body == {
            "error": {
                "message": "Rate limit exceeded for ^^imp",
                "type": "rate_limit_exceeded",
                "retry_after": 30,
            }
        }
        assert status == 429


# US-060a: shared malformed-completion validation contract.
MALFORMED_SHAPE_CASES = [
    (None, "response is not an object"),
    ([], "response is not an object"),
    ("invalid", "response is not an object"),
    ({}, "choices is not an array"),
    ({"choices": None}, "choices is not an array"),
    ({"choices": {}}, "choices is not an array"),
    ({"choices": []}, "choices is empty"),
    ({"choices": [None]}, "choices[0] is not an object"),
    ({"choices": ["invalid"]}, "choices[0] is not an object"),
    ({"choices": [{"message": None}]}, "choices[0].message is not an object"),
    ({"choices": [{"finish_reason": "stop"}]}, "choices[0].message is not an object"),
    ({"error": "provider failed"}, "choices is not an array"),
    ({"error": None}, "choices is not an array"),
    ({"error": []}, "choices is not an array"),
]


class TestChatCompletionShapeError:
    """US-060a: chat_completion_shape_error returns a stable reason for malformed upstream payloads."""

    @pytest.mark.parametrize("data,expected", MALFORMED_SHAPE_CASES)
    def test_malformed_shape_returns_reason(self, data, expected):
        # US-060a
        assert chat_completion_shape_error(data) == expected

    def test_valid_text_completion(self):
        # US-060a
        data = {"choices": [{"message": {"role": "assistant", "content": "ok"}}]}
        assert chat_completion_shape_error(data) is None

    def test_valid_tool_call_completion(self):
        # US-060a
        data = {
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [{
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "do_thing", "arguments": "{}"},
                    }],
                }
            }]
        }
        assert chat_completion_shape_error(data) is None

    def test_valid_reasoning_completion(self):
        # US-060a
        data = {
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": "ok",
                    "reasoning_content": "thought process",
                }
            }]
        }
        assert chat_completion_shape_error(data) is None

    def test_explicit_error_envelope(self):
        # US-060a
        data = {"error": {"message": "provider failed", "type": "provider_error"}}
        assert chat_completion_shape_error(data) is None

    def test_explicit_error_envelope_with_empty_choices(self):
        # US-060a
        data = {
            "error": {"message": "provider failed", "type": "provider_error"},
            "choices": [],
        }
        assert chat_completion_shape_error(data) is None


class TestBuildUpstreamProtocolError:
    def test_exact_shape(self):
        # US-060a
        assert build_upstream_protocol_error() == {
            "error": {
                "message": "Upstream returned an invalid chat completion response",
                "type": "upstream_protocol_error",
                "code": "invalid_chat_completion",
            }
        }
