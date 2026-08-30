import json
from unittest.mock import patch

import proxy.session_tokens as st


SECRET = "test-secret-key-12345"
SID = "sess-xyz"


def _with_secret(fn):
    """Run fn with _SESSION_SECRET patched to SECRET."""
    with patch.object(st, "_SESSION_SECRET", SECRET):
        return fn()


def test_create_and_validate_round_trip():
    with patch.object(st, "_SESSION_SECRET", SECRET):
        with patch("proxy.session_tokens.time.time", return_value=1000.0):
            token = st.create_session_token("client-abc", SID)
        assert token != ""

        with patch("proxy.session_tokens.time.time", return_value=1000.0):
            payload = st.validate_session_token(token)

    assert payload is not None
    assert payload["client_id"] == "client-abc"
    assert payload["session_id"] == SID
    assert payload["created_at"] == 1000.0
    assert "nonce" in payload


def test_invalid_signature_returns_none():
    with patch.object(st, "_SESSION_SECRET", SECRET):
        with patch("proxy.session_tokens.time.time", return_value=1000.0):
            token = st.create_session_token("client-abc", SID)

    parts = token.rsplit(":", 1)
    tampered = parts[0] + ":0000000000000000"

    with patch.object(st, "_SESSION_SECRET", SECRET):
        with patch("proxy.session_tokens.time.time", return_value=1000.0):
            result = st.validate_session_token(tampered)
    assert result is None


def test_malformed_token_returns_none():
    with patch.object(st, "_SESSION_SECRET", SECRET):
        assert st.validate_session_token("garbage-no-colon") is None
        assert st.validate_session_token("") is None
        assert st.validate_session_token("a:b:c:d") is None


def test_empty_secret_fails_closed():
    with patch.object(st, "_SESSION_SECRET", ""):
        token = st.create_session_token("client-abc", SID)
        assert token == ""

        result = st.validate_session_token("some:token")
        assert result is None


def test_token_expiry():
    with patch.object(st, "_SESSION_SECRET", SECRET):
        with patch("proxy.session_tokens.time.time", return_value=1000.0):
            token = st.create_session_token("client-abc", SID)

        with patch("proxy.session_tokens.time.time", return_value=1000.0 + 3600 + 1):
            result = st.validate_session_token(token, max_age_seconds=3600)
    assert result is None


def test_token_within_expiry_accepted():
    with patch.object(st, "_SESSION_SECRET", SECRET):
        with patch("proxy.session_tokens.time.time", return_value=1000.0):
            token = st.create_session_token("client-abc", SID)

        with patch("proxy.session_tokens.time.time", return_value=1000.0 + 1800):
            result = st.validate_session_token(token, max_age_seconds=3600)
    assert result is not None
    assert result["client_id"] == "client-abc"


def test_nonce_uniqueness():
    tokens = []
    with patch.object(st, "_SESSION_SECRET", SECRET):
        with patch("proxy.session_tokens.time.time", return_value=1000.0):
            for _ in range(20):
                tokens.append(st.create_session_token("client-abc", SID))

    nonces = set()
    for token in tokens:
        payload_str = token.rsplit(":", 1)[0]
        payload = json.loads(payload_str)
        nonces.add(payload["nonce"])

    assert len(nonces) == 20


def test_missing_fields_returns_none():
    with patch.object(st, "_SESSION_SECRET", SECRET):
        payload_no_client = json.dumps(
            {"created_at": 1000.0, "nonce": "abc"}, separators=(",", ":")
        )
        import hmac as _hmac
        import hashlib
        sig = _hmac.new(SECRET.encode(), payload_no_client.encode(), hashlib.sha256).hexdigest()[:16]
        token_no_client = f"{payload_no_client}:{sig}"

        with patch("proxy.session_tokens.time.time", return_value=1000.0):
            result = st.validate_session_token(token_no_client)
        assert result is not None
        assert result.get("client_id") is None

        payload_no_created = json.dumps(
            {"client_id": "x", "nonce": "abc"}, separators=(",", ":")
        )
        sig2 = _hmac.new(SECRET.encode(), payload_no_created.encode(), hashlib.sha256).hexdigest()[:16]
        token_no_created = f"{payload_no_created}:{sig2}"

        with patch("proxy.session_tokens.time.time", return_value=1000.0):
            result2 = st.validate_session_token(token_no_created, max_age_seconds=500)
        assert result2 is None
