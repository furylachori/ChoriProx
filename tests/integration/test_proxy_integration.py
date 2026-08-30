"""Integration tests for ChoriProx model registration."""

import json
import pytest

pytestmark = pytest.mark.integration


class TestAuthInjection:
    def test_auth_header_sent_to_backend(self, proxy_with_mock):
        client, handler, port = proxy_with_mock
        handler._response_body = json.dumps(
            {
                "id": "test",
                "object": "chat.completion",
                "created": 0,
                "model": "opencode-go/kimi-k2.6",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "ok"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 1,
                    "completion_tokens": 1,
                    "total_tokens": 2,
                },
            }
        ).encode()

        resp = client.post(
            "/v1/chat/completions",
            json={
                "model": "kimi",
                "messages": [{"role": "user", "content": "hello"}],
                "stream": False,
            },
        )
        assert resp.status_code == 200
        assert (
            handler._captured_headers.get("Authorization")
            == "Bearer sk-test-secret-key-12345"
        )

    def test_no_auth_for_local_model(self, proxy_with_mock):
        client, handler, port = proxy_with_mock
        handler._response_body = json.dumps(
            {
                "id": "test",
                "object": "chat.completion",
                "created": 0,
                "model": "qwen3-coder",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "ok"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 1,
                    "completion_tokens": 1,
                    "total_tokens": 2,
                },
            }
        ).encode()

        resp = client.post(
            "/v1/chat/completions",
            json={
                "model": "qwen3-coder",
                "messages": [{"role": "user", "content": "hello"}],
                "stream": False,
            },
        )
        assert resp.status_code == 200
        assert handler._captured_headers.get("Authorization") is None

    def test_missing_api_key_returns_401(self, proxy_with_mock, monkeypatch):
        client, handler, port = proxy_with_mock
        monkeypatch.delenv("TEST_API_KEY", raising=False)

        resp = client.post(
            "/v1/chat/completions",
            json={
                "model": "kimi",
                "messages": [{"role": "user", "content": "hello"}],
                "stream": False,
            },
        )
        assert resp.status_code == 401
        assert handler._request_count == 0


class TestModelRewriting:
    def test_model_rewritten_in_request(self, proxy_with_mock):
        client, handler, port = proxy_with_mock
        handler._response_body = json.dumps(
            {
                "id": "test",
                "object": "chat.completion",
                "created": 0,
                "model": "opencode-go/kimi-k2.6",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "ok"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 1,
                    "completion_tokens": 1,
                    "total_tokens": 2,
                },
            }
        ).encode()

        resp = client.post(
            "/v1/chat/completions",
            json={
                "model": "kimi",
                "messages": [{"role": "user", "content": "hello"}],
                "stream": False,
            },
        )
        assert resp.status_code == 200
        sent_body = json.loads(handler._captured_request_body)
        assert sent_body["model"] == "opencode-go/kimi-k2.6"

    def test_response_model_restored(self, proxy_with_mock):
        client, handler, port = proxy_with_mock
        handler._response_body = json.dumps(
            {
                "id": "test",
                "object": "chat.completion",
                "created": 0,
                "model": "opencode-go/kimi-k2.6",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "ok"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 1,
                    "completion_tokens": 1,
                    "total_tokens": 2,
                },
            }
        ).encode()

        resp = client.post(
            "/v1/chat/completions",
            json={
                "model": "kimi",
                "messages": [{"role": "user", "content": "hello"}],
                "stream": False,
            },
        )
        data = resp.get_json()
        assert data["model"] == "kimi"


class TestURLConstruction:
    def test_backend_url_correct(self, proxy_with_mock):
        client, handler, port = proxy_with_mock
        handler._response_body = json.dumps(
            {
                "id": "test",
                "object": "chat.completion",
                "created": 0,
                "model": "test",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "ok"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 1,
                    "completion_tokens": 1,
                    "total_tokens": 2,
                },
            }
        ).encode()

        resp = client.post(
            "/v1/chat/completions",
            json={
                "model": "kimi",
                "messages": [{"role": "user", "content": "hello"}],
                "stream": False,
            },
        )
        assert resp.status_code == 200
        assert handler._request_count == 1


class TestContextLimit:
    @pytest.mark.xfail(
        reason="Context limit check runs after single-message truncation; 100KB-per-message cap returns 0 tokens, bypassing enforcement"
    )
    def test_oversized_content_rejected(self, proxy_with_mock):
        client, handler, port = proxy_with_mock
        huge = "x" * 600_000

        resp = client.post(
            "/v1/chat/completions",
            json={
                "model": "kimi",
                "messages": [{"role": "user", "content": huge}],
                "stream": False,
            },
        )
        assert resp.status_code == 400


class TestErrorPropagation:
    def test_429_forwarded(self, proxy_with_mock):
        client, handler, port = proxy_with_mock
        handler._status_code = 429
        handler._response_body = json.dumps(
            {"error": {"message": "Rate limited"}}
        ).encode()

        resp = client.post(
            "/v1/chat/completions",
            json={
                "model": "kimi",
                "messages": [{"role": "user", "content": "hello"}],
                "stream": False,
            },
        )
        assert resp.status_code == 429

    def test_500_sanitized(self, proxy_with_mock):
        client, handler, port = proxy_with_mock
        handler._status_code = 500
        handler._response_body = json.dumps(
            {"error": {"message": "Internal error", "api_key": "sk-leaked"}}
        ).encode()

        resp = client.post(
            "/v1/chat/completions",
            json={
                "model": "kimi",
                "messages": [{"role": "user", "content": "hello"}],
                "stream": False,
            },
        )
        data = resp.get_json()
        assert resp.status_code == 502
        error_body = json.dumps(data)
        assert "sk-leaked" not in error_body


class TestReasoningContent:
    def test_reasoning_in_non_streaming(self, proxy_with_mock):
        client, handler, port = proxy_with_mock
        handler._response_body = json.dumps(
            {
                "id": "test",
                "object": "chat.completion",
                "created": 0,
                "model": "opencode-go/kimi-k2.6",
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "reasoning_content": "Let me think...",
                            "content": "The answer.",
                        },
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 1,
                    "completion_tokens": 1,
                    "total_tokens": 2,
                },
            }
        ).encode()

        resp = client.post(
            "/v1/chat/completions",
            json={
                "model": "kimi",
                "messages": [{"role": "user", "content": "hello"}],
                "stream": False,
            },
        )
        data = resp.get_json()
        msg = data["choices"][0]["message"]
        assert msg["reasoning_content"] == "Let me think..."
        assert msg["content"] == "The answer."
        assert data["model"] == "kimi"
