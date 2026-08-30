import pytest
from unittest.mock import patch, MagicMock
from qwen_proxy import app, AuthConfigError


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setenv("PROXY_API_KEY", "test-proxy-key")
    app.config["TESTING"] = True
    with app.test_client() as c:
        yield c


def test_auth_header_sent_when_env_var_set(client, monkeypatch):
    """When api_key_env is set and env var exists, Authorization header is injected."""
    monkeypatch.setenv("OLLAMA_API_KEY", "sk-test-123")

    captured_requests = []

    def capture_post(url, **kwargs):
        captured_requests.append({"url": url, "kwargs": kwargs})
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
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
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        }
        mock_resp.raise_for_status = MagicMock()
        return mock_resp

    with patch("qwen_proxy._HTTP_SESSION.post", side_effect=capture_post):
        resp = client.post(
            "/v1/chat/completions",
            json={
                "model": "glm",
                "messages": [{"role": "user", "content": "hello"}],
            },
            headers={"Authorization": "Bearer test-proxy-key"},
        )

    assert resp.status_code == 200
    assert len(captured_requests) == 1
    assert "headers" in captured_requests[0]["kwargs"]
    assert (
        captured_requests[0]["kwargs"]["headers"].get("Authorization")
        == "Bearer sk-test-123"
    )


def test_auth_config_error_when_env_var_missing(client, monkeypatch):
    """When api_key_env is set but env var is missing, AuthConfigError is raised (401)."""
    monkeypatch.delenv("OLLAMA_API_KEY", raising=False)

    with patch("qwen_proxy._HTTP_SESSION.post") as mock_post:
        mock_post.return_value = MagicMock()
        resp = client.post(
            "/v1/chat/completions",
            json={
                "model": "glm",
                "messages": [{"role": "user", "content": "hello"}],
            },
            headers={"Authorization": "Bearer test-proxy-key"},
        )

    assert resp.status_code == 401
    data = resp.get_json()
    assert "error" in data
    assert (
        "auth_error" in data.get("type", "").lower()
        or "Model configuration error" in data["error"]
    )


def test_auth_config_error_when_env_var_empty_string(client, monkeypatch):
    """When api_key_env is set but env var is empty string, AuthConfigError is raised (401)."""
    monkeypatch.setenv("OLLAMA_API_KEY", "")

    with patch("qwen_proxy._HTTP_SESSION.post") as mock_post:
        mock_post.return_value = MagicMock()
        resp = client.post(
            "/v1/chat/completions",
            json={
                "model": "glm",
                "messages": [{"role": "user", "content": "hello"}],
            },
            headers={"Authorization": "Bearer test-proxy-key"},
        )

    assert resp.status_code == 401


def test_no_auth_header_for_local_model(client, monkeypatch):
    """When model has no api_key_env configured, no Authorization header is sent."""
    monkeypatch.delenv("OPENCODE_API_KEY", raising=False)

    captured_requests = []

    def capture_post(url, **kwargs):
        captured_requests.append({"url": url, "kwargs": kwargs})
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
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
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        }
        mock_resp.raise_for_status = MagicMock()
        return mock_resp

    with patch("qwen_proxy._HTTP_SESSION.post", side_effect=capture_post):
        resp = client.post(
            "/v1/chat/completions",
            json={
                "model": "qwen3-coder",
                "messages": [{"role": "user", "content": "hello"}],
            },
            headers={"Authorization": "Bearer test-proxy-key"},
        )

    assert resp.status_code == 200
    assert len(captured_requests) == 1
    headers = captured_requests[0]["kwargs"].get("headers", {})
    assert "Authorization" not in headers or headers.get("Authorization") is None


def test_auth_headers_empty_dict_when_no_api_key_env(client, monkeypatch):
    """When model has no api_key_env, _auth_headers is empty dict (no auth injected)."""
    monkeypatch.delenv("OPENCODE_API_KEY", raising=False)

    captured_kwargs = []

    def capture_post(url, **kwargs):
        captured_kwargs.append(kwargs)
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "choices": [
                {
                    "message": {"role": "assistant", "content": "ok"},
                    "finish_reason": "stop",
                }
            ]
        }
        mock_resp.raise_for_status = MagicMock()
        return mock_resp

    with patch("qwen_proxy._HTTP_SESSION.post", side_effect=capture_post):
        resp = client.post(
            "/v1/chat/completions",
            json={
                "model": "qwen3-coder",
                "messages": [{"role": "user", "content": "hello"}],
            },
            headers={"Authorization": "Bearer test-proxy-key"},
        )

    assert resp.status_code == 200
    assert len(captured_kwargs) == 1
    headers = captured_kwargs[0].get("headers", {})
    assert headers == {} or "Authorization" not in headers


def test_fail_closed_no_api_key_returns_500(monkeypatch):
    """US-051: When PROXY_API_KEY is not set, /v1/chat/completions returns 500 (fail-closed)."""
    monkeypatch.delenv("PROXY_API_KEY", raising=False)
    app.config["TESTING"] = True
    with app.test_client() as c:
        resp = c.post(
            "/v1/chat/completions",
            json={
                "model": "glm",
                "messages": [{"role": "user", "content": "hello"}],
            },
            environ_overrides={"REMOTE_ADDR": "10.0.0.1"},
        )
    assert resp.status_code == 500
    data = resp.get_json()
    assert data["type"] == "auth_config_error"


def test_fail_closed_models_endpoint_still_works(monkeypatch):
    """US-051/US-047: /v1/models remains accessible without PROXY_API_KEY."""
    monkeypatch.delenv("PROXY_API_KEY", raising=False)
    app.config["TESTING"] = True
    with app.test_client() as c:
        resp = c.get("/v1/models")
    assert resp.status_code == 200


def test_fail_closed_wrong_key_returns_401(monkeypatch):
    """When PROXY_API_KEY is set, wrong key from non-localhost returns 401."""
    monkeypatch.setenv("PROXY_API_KEY", "correct-key")
    app.config["TESTING"] = True
    with app.test_client() as c:
        resp = c.post(
            "/v1/chat/completions",
            json={
                "model": "glm",
                "messages": [{"role": "user", "content": "hello"}],
            },
            headers={"Authorization": "Bearer wrong-key"},
            environ_overrides={"REMOTE_ADDR": "10.0.0.1"},
        )
    assert resp.status_code == 401
