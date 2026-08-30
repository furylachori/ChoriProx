import json
import pytest
from qwen_proxy import app, BackendError, AuthConfigError, _sanitize_error_body


@pytest.fixture
def client():
    app.config["TESTING"] = True
    with app.test_client() as c:
        yield c


def test_sanitize_error_body_strips_sensitive_keys():
    """# US-035: Sensitive key sanitization in error bodies."""
    body = {
        "error": {
            "message": "fail",
            "api_key": "sk-secret",
            "token": "bearer-xxx",
            "info": "ok",
        }
    }
    result = _sanitize_error_body(body)
    assert "api_key" not in result["error"]
    assert "token" not in result["error"]
    assert result["error"]["info"] == "ok"
    assert result["error"]["message"] == "fail"


def test_sanitize_error_body_handles_nested():
    body = {"error": {"details": {"api_key": "nested-secret", "code": 42}}}
    result = _sanitize_error_body(body)
    assert "api_key" not in result["error"]["details"]
    assert result["error"]["details"]["code"] == 42


def test_sanitize_error_body_handles_lists():
    body = {"errors": [{"api_key": "leaked"}, {"info": "ok"}]}
    result = _sanitize_error_body(body)
    assert "api_key" not in result["errors"][0]
    assert result["errors"][1]["info"] == "ok"


def test_backend_error_class():
    e = BackendError(429, {"error": "rate limited"})
    assert e.status_code == 429
    assert e.error_body == {"error": "rate limited"}


def test_auth_config_error_class():
    e = AuthConfigError("kimi", "OPENCODE_API_KEY")
    assert "kimi" in str(e)
    assert "OPENCODE_API_KEY" in str(e)
