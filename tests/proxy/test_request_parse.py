"""US-W3-REQVAL: Wave 3 — Request validation (spec §7.1).

Per spec §7.1, ``parse_request`` MUST validate the structural shape of the
incoming chat-completions body before dispatch. Malformed requests return a
structured 400 envelope rendered by the chat handler when it catches
``InvalidRequestError``. These tests pin the validators at the
``parse_request`` call site (not at the Flask handler) so a regression in
either layer surfaces clearly.
"""
import json
from types import SimpleNamespace
from typing import Optional
from unittest.mock import patch

import pytest

from proxy.pipeline.stage_modules.request_parse import parse_request
from proxy.types import InvalidRequestError


def _make_fake_request(body: dict, headers: Optional[dict] = None) -> SimpleNamespace:
    """Build a ``parse_request``-compatible fake request.

    Mirrors the SimpleNamespace pattern used by ``tests/unit/test_subagent_workspace_inheritance.py``
    and ``tests/unit/test_security_boundaries.py`` so the validators are
    exercised through the REAL production call site — no mocking of the
    function under test.
    """
    merged_headers = {"X-Session-Client": "test-client"}
    if headers:
        merged_headers.update(headers)
    return SimpleNamespace(
        method="POST",
        path="/v1/chat/completions",
        remote_addr="127.0.0.1",
        headers=merged_headers,
        get_json=lambda silent=False: body,
    )


@pytest.fixture(autouse=True)
def _no_proxy_session_secret(monkeypatch):
    """Disable PROXY_SESSION_SECRET so parse_request takes the no-token branch
    and skips HMAC validation. The token-validation tests live elsewhere
    (test_session_tokens.py)."""
    monkeypatch.delenv("PROXY_SESSION_SECRET", raising=False)


class TestValidRequests:
    def test_valid_user_message_passes(self):
        req = _make_fake_request({
            "model": "test-model",
            "messages": [{"role": "user", "content": "hello"}],
        })
        ctx = parse_request(req)
        assert ctx.session_id
        assert ctx.model_name == "test-model"
        assert len(ctx.messages) == 1
        assert ctx.messages[0]["role"] == "user"

    def test_valid_multimodal_content_list_passes(self):
        req = _make_fake_request({
            "model": "test-model",
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": "describe"},
                    {"type": "image_url", "image_url": {"url": "http://x/y.png"}},
                ],
            }],
        })
        ctx = parse_request(req)
        assert ctx.messages[0]["content"][0]["text"] == "describe"
        assert ctx.messages[0]["content"][1]["image_url"]["url"] == "http://x/y.png"

    def test_stream_true_passes(self):
        """A boolean ``stream: true`` is the canonical SSE flow — must parse."""
        req = _make_fake_request({
            "model": "test-model",
            "stream": True,
            "messages": [{"role": "user", "content": "go"}],
        })
        ctx = parse_request(req)
        assert ctx.wants_stream is True

    def test_stream_omitted_passes(self):
        req = _make_fake_request({
            "model": "test-model",
            "messages": [{"role": "user", "content": "go"}],
        })
        ctx = parse_request(req)
        assert ctx.wants_stream is False

    def test_tool_message_with_tool_call_id_passes(self):
        req = _make_fake_request({
            "model": "test",
            "messages": [{
                "role": "tool",
                "tool_call_id": "call_abc",
                "content": "result text",
            }],
        })
        ctx = parse_request(req)
        assert ctx.messages[0]["tool_call_id"] == "call_abc"

    def test_content_null_passes(self):
        """A None content is valid for assistant messages (the model may emit
        tool_calls without text)."""
        req = _make_fake_request({
            "model": "test",
            "messages": [{"role": "assistant", "content": None}],
        })
        ctx = parse_request(req)
        assert ctx.messages[0]["content"] is None


class TestStreamValidation:
    def test_stream_must_be_boolean_raises(self):
        req = _make_fake_request({
            "model": "test-model",
            "stream": "true",
            "messages": [{"role": "user", "content": "hi"}],
        })
        with pytest.raises(InvalidRequestError) as exc_info:
            parse_request(req)
        assert exc_info.value.code == "invalid_stream"
        assert exc_info.value.param == "stream"
        assert "boolean" in exc_info.value.message.lower()

    def test_stream_int_raises(self):
        """Integers are NOT booleans, even though Python treats them loosely.
        The validation requires strict bool type."""
        req = _make_fake_request({
            "model": "test",
            "stream": 1,
            "messages": [{"role": "user", "content": "hi"}],
        })
        with pytest.raises(InvalidRequestError) as exc:
            parse_request(req)
        assert exc.value.code == "invalid_stream"


class TestMessagesStructure:
    def test_messages_missing_raises(self):
        """Omitting messages entirely is malformed."""
        req = _make_fake_request({"model": "test"})
        with pytest.raises(InvalidRequestError) as exc:
            parse_request(req)
        assert exc.value.code == "invalid_messages"
        assert exc.value.param == "messages"

    def test_messages_not_list_raises(self):
        req = _make_fake_request({"model": "test", "messages": "not a list"})
        with pytest.raises(InvalidRequestError) as exc:
            parse_request(req)
        assert exc.value.code == "invalid_messages"

    def test_empty_messages_raises(self):
        req = _make_fake_request({"model": "test", "messages": []})
        with pytest.raises(InvalidRequestError) as exc:
            parse_request(req)
        assert exc.value.code == "invalid_messages"
        assert "non-empty" in exc.value.message.lower()

    def test_message_not_object_raises(self):
        req = _make_fake_request({
            "model": "test",
            "messages": [{"role": "user", "content": "hi"}, "not a dict"],
        })
        with pytest.raises(InvalidRequestError) as exc:
            parse_request(req)
        assert exc.value.code == "invalid_message"
        assert exc.value.param == "messages[1]"


class TestRoleValidation:
    def test_role_missing_raises(self):
        req = _make_fake_request({
            "model": "test",
            "messages": [{"content": "hi"}],
        })
        with pytest.raises(InvalidRequestError) as exc:
            parse_request(req)
        assert exc.value.code == "invalid_role"
        assert exc.value.param == "messages[0].role"

    def test_role_not_string_raises(self):
        req = _make_fake_request({
            "model": "test",
            "messages": [{"role": 123, "content": "hi"}],
        })
        with pytest.raises(InvalidRequestError) as exc:
            parse_request(req)
        assert exc.value.code == "invalid_role"
        assert exc.value.param == "messages[0].role"

    def test_role_unsupported_value_raises(self):
        req = _make_fake_request({
            "model": "test",
            "messages": [{"role": "moderator", "content": "hi"}],
        })
        with pytest.raises(InvalidRequestError) as exc:
            parse_request(req)
        assert exc.value.code == "invalid_role_value"
        assert exc.value.param == "messages[0].role"
        assert "moderator" in exc.value.message


class TestContentValidation:
    def test_content_int_raises(self):
        req = _make_fake_request({
            "model": "test",
            "messages": [{"role": "user", "content": 42}],
        })
        with pytest.raises(InvalidRequestError) as exc:
            parse_request(req)
        assert exc.value.code == "invalid_content"

    def test_content_list_item_not_object_raises(self):
        req = _make_fake_request({
            "model": "test",
            "messages": [{"role": "user", "content": ["just a string"]}],
        })
        with pytest.raises(InvalidRequestError) as exc:
            parse_request(req)
        assert exc.value.code == "invalid_content"
        assert exc.value.param == "messages[0].content[0]"

    def test_content_list_item_missing_type_raises(self):
        req = _make_fake_request({
            "model": "test",
            "messages": [{"role": "user", "content": [{"text": "hi"}]}],
        })
        with pytest.raises(InvalidRequestError) as exc:
            parse_request(req)
        assert exc.value.code == "invalid_content"

    def test_content_list_item_unknown_type_raises(self):
        req = _make_fake_request({
            "model": "test",
            "messages": [{"role": "user", "content": [{"type": "audio"}]}],
        })
        with pytest.raises(InvalidRequestError) as exc:
            parse_request(req)
        assert exc.value.code == "invalid_content"

    def test_content_string_passes(self):
        req = _make_fake_request({
            "model": "test",
            "messages": [{"role": "user", "content": "just text"}],
        })
        ctx = parse_request(req)
        assert ctx.messages[0]["content"] == "just text"


class TestToolCallsValidation:
    def test_tool_calls_must_be_list_raises(self):
        req = _make_fake_request({
            "model": "test",
            "messages": [{
                "role": "assistant",
                "content": "ok",
                "tool_calls": {"not": "a list"},
            }],
        })
        with pytest.raises(InvalidRequestError) as exc:
            parse_request(req)
        assert exc.value.code == "invalid_tool_calls"
        assert exc.value.param == "messages[0].tool_calls"

    def test_tool_calls_list_passes(self):
        req = _make_fake_request({
            "model": "test",
            "messages": [{
                "role": "assistant",
                "content": None,
                "tool_calls": [{
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "noop", "arguments": "{}"},
                }],
            }],
        })
        ctx = parse_request(req)
        assert isinstance(ctx.messages[0]["tool_calls"], list)


class TestToolCallIdValidation:
    def test_tool_message_missing_tool_call_id_raises(self):
        req = _make_fake_request({
            "model": "test",
            "messages": [{"role": "tool", "content": "result"}],
        })
        with pytest.raises(InvalidRequestError) as exc:
            parse_request(req)
        assert exc.value.code == "missing_tool_call_id"
        assert exc.value.param == "messages[0].tool_call_id"

    def test_tool_message_empty_tool_call_id_raises(self):
        req = _make_fake_request({
            "model": "test",
            "messages": [{"role": "tool", "tool_call_id": "", "content": "r"}],
        })
        with pytest.raises(InvalidRequestError) as exc:
            parse_request(req)
        assert exc.value.code == "missing_tool_call_id"

    def test_tool_message_non_string_tool_call_id_raises(self):
        req = _make_fake_request({
            "model": "test",
            "messages": [{"role": "tool", "tool_call_id": 99, "content": "r"}],
        })
        with pytest.raises(InvalidRequestError) as exc:
            parse_request(req)
        assert exc.value.code == "missing_tool_call_id"


class TestErrorAggregation:
    def test_multiple_errors_are_aggregated_into_single_exception(self):
        """Spec §7.1 — collect errors rather than aborting at the first one.
        The summary message must mention at least 2 distinct field paths."""
        req = _make_fake_request({
            "model": "test",
            "stream": "true",
            "messages": [
                {"role": "user", "content": "ok"},
                {"role": "moderator", "content": 42},
            ],
        })
        with pytest.raises(InvalidRequestError) as exc:
            parse_request(req)
        assert exc.value.code == "invalid_stream"
        assert "stream" in exc.value.message
        assert "messages[1].role" in exc.value.message
        assert "messages[1].content" in exc.value.message

    def test_many_errors_truncated_with_more_marker(self):
        """When there are >3 errors, the summary includes a '+N more' marker."""
        bad_messages = [
            {"role": "bad", "content": 1},
            {"role": "bad", "content": 2},
            {"role": "bad", "content": 3},
            {"role": "bad", "content": 4},
        ]
        req = _make_fake_request({
            "model": "test",
            "messages": bad_messages,
        })
        with pytest.raises(InvalidRequestError) as exc:
            parse_request(req)
        assert "+" in exc.value.message
        assert "more" in exc.value.message

    def test_first_error_param_is_returned(self):
        """The exception's .param/.code attributes reflect the FIRST failure
        encountered, so the chat handler can render a stable error pointer
        without re-inspecting the summary message."""
        req = _make_fake_request({
            "model": "test",
            "messages": [{"role": 42, "content": "x"}],
        })
        with pytest.raises(InvalidRequestError) as exc:
            parse_request(req)
        assert exc.value.param == "messages[0].role"
        assert exc.value.code == "invalid_role"


class TestErrorEnvelopeShape:
    """The chat handler in app.py renders the InvalidRequestError into the
    spec §7.1 envelope. Pin the envelope shape here at the type level so a
    future rename surfaces."""

    def test_invalidrequesterror_carries_param_code_message(self):
        err = InvalidRequestError(param="messages[2].role", code="invalid_role", message="x")
        assert err.param == "messages[2].role"
        assert err.code == "invalid_role"
        assert err.message == "x"
        assert isinstance(err, ValueError)

    def test_invalidrequesterror_is_valueerror_subclass(self):
        """InvalidRequestError must extend ValueError so the legacy
        ``except ValueError`` block in app.py remains a valid safety net
        for non-InvalidRequestError ValueErrors."""
        err = InvalidRequestError(param="x", code="y", message="z")
        with pytest.raises(ValueError):
            raise err


class TestTopLevelBodyValidation:
    """MED-2 (Wave 3): the three top-level structural checks at the head
    of `parse_request` MUST raise the structured `InvalidRequestError`
    (spec §7.1) so the chat handler renders them as a 400 envelope. The
    pre-Wave-3 code raised a plain `ValueError`, which the chat handler
    caught in its generic `except ValueError` branch — bypassing the
    structured envelope and losing the param/code path."""

    def test_body_none_raises_invalid_body(self):
        """Body=None (empty body / non-JSON) → ``InvalidRequestError(
        param="body", code="invalid_body")``."""
        req = _make_fake_request(None)  # get_json(silent=True) returns None
        # Override get_json for the None case.
        req.get_json = lambda silent=False: None
        with pytest.raises(InvalidRequestError) as exc:
            parse_request(req)
        assert exc.value.code == "invalid_body"
        assert exc.value.param == "body"
        assert "JSON object" in exc.value.message

    def test_body_not_dict_raises_invalid_body(self):
        """Body is not a dict (e.g. a list, a string) → ``InvalidRequestError(
        param="body", code="invalid_body")``."""
        req = _make_fake_request(["not a dict"])
        req.get_json = lambda silent=False: ["not a dict"]
        with pytest.raises(InvalidRequestError) as exc:
            parse_request(req)
        assert exc.value.code == "invalid_body"
        assert exc.value.param == "body"
        assert "JSON object" in exc.value.message

    def test_model_not_string_raises_invalid_model(self):
        """`body["model"]` is not a string → ``InvalidRequestError(
        param="model", code="invalid_model")``."""
        req = _make_fake_request({
            "model": 123,  # int, not string
            "messages": [{"role": "user", "content": "hi"}],
        })
        with pytest.raises(InvalidRequestError) as exc:
            parse_request(req)
        assert exc.value.code == "invalid_model"
        assert exc.value.param == "model"
        assert "string" in exc.value.message.lower()


class TestStreamReturnsJsonNotSse:
    """Spec §7.1 — when a request with ``stream=true`` carries a malformed
    body, the proxy MUST respond with a plain JSON 400 envelope, NOT a
    Server-Sent Events stream. This pins the contract that the request
    parser runs BEFORE the streaming branch in the chat handler, so an
    ``InvalidRequestError`` raised from ``parse_request`` falls into the
    JSON envelope handler rather than the SSE keepalive generator.

    Regression target: if the streaming branch is taken before
    ``parse_request`` (or if ``InvalidRequestError`` is allowed to bubble
    into the SSE generator), the response content-type becomes
    ``text/event-stream`` and clients cannot parse the error.
    """

    def test_stream_true_with_invalid_body_returns_plain_json_400(self, client):
        """Per spec §7.1, a stream=true request with malformed body must return
        plain JSON 400, NOT SSE. parse_request runs BEFORE the streaming branch
        (app.py:703 vs :749), so the InvalidRequestError raise hits the JSON
        envelope handler (app.py:813) before any SSE chunk is emitted.
        """
        response = client.post("/v1/chat/completions", json={
            "model": "test",
            "stream": True,
            "messages": [],  # spec §7.1: non-empty list required
        })
        assert response.status_code == 400
        assert response.content_type == "application/json"
        body = response.get_json()
        assert body["error"]["type"] == "invalid_request_error"
        assert body["error"]["code"] == "invalid_messages"


class TestSessionTokenIdentityCP2A:
    """CP2A: the session token embeds a continuity session_id (HMAC-signed)
    that ``parse_request`` restores on the next request, and the auth check
    compares the token's client_id against the X-Session-Client hint.

    These tests exercise the REAL ``parse_request`` call site (no mocking of
    ``validate_session_token``). They mint tokens directly; the app.py embed
    side (create_session_token(client_id=..., session_id=...)) is exercised by
    the integration round-trip in tests/integration/test_vfs_cache_safe_tail.py.

    Each test OVERRIDES the module autouse ``_no_proxy_session_secret`` fixture
    by re-setting PROXY_SESSION_SECRET (env) AND patching the module-level
    ``proxy.session_tokens._SESSION_SECRET`` (which both create_session_token
    and validate_session_token read).
    """

    SECRET = "cp2a-test-secret-key-98765"
    SID_A = "a1b2c3d4e5f60718"

    def _enable_secret(self, monkeypatch):
        import proxy.session_tokens as st
        monkeypatch.setenv("PROXY_SESSION_SECRET", self.SECRET)
        monkeypatch.setattr(st, "_SESSION_SECRET", self.SECRET)
        return st

    def test_token_restores_session_id_across_differing_content(self, monkeypatch):
        """A valid token restores its embedded session_id verbatim, giving
        continuity even when the request content (which would otherwise drive
        session-id derivation) differs between turns.

        Revert-check: reverting Change 3 (the restore block) breaks this — the
        session_id would fall back to content-derived and diverge for A vs B.
        """
        from proxy.model_registry import resolve_session_id

        st = self._enable_secret(monkeypatch)
        token = st.create_session_token("test-client", self.SID_A)
        assert token

        body_a = {
            "model": "test-model",
            "messages": [{"role": "user", "content": "alpha content"}],
        }
        body_b = {
            "model": "test-model",
            "messages": [{"role": "user", "content": "completely different beta content"}],
        }
        headers = {"X-Session-Client": "test-client", "X-Session-Token": token}

        ctx_a = parse_request(_make_fake_request(body_a, headers))
        ctx_b = parse_request(_make_fake_request(body_b, headers))

        assert ctx_a.session_id == self.SID_A
        assert ctx_b.session_id == self.SID_A

        # Non-tautology guard: content-derivation WOULD diverge for A vs B.
        assert resolve_session_id(body_a, "test-client") != resolve_session_id(body_b, "test-client")

    def test_client_echoes_own_token_not_403(self, monkeypatch):
        """A client that echoes back its own token (matching X-Session-Client)
        is NOT rejected with 403, and its session_id is restored."""
        st = self._enable_secret(monkeypatch)
        token = st.create_session_token("test-client", self.SID_A)
        assert token

        body = {
            "model": "test-model",
            "messages": [{"role": "user", "content": "hello again"}],
        }
        headers = {"X-Session-Client": "test-client", "X-Session-Token": token}

        ctx = parse_request(_make_fake_request(body, headers))
        assert ctx.session_id == self.SID_A

    def test_token_client_mismatch_still_403(self, monkeypatch):
        """A token minted for client-A but presented with X-Session-Client
        client-B must still be rejected with a 403 client mismatch."""
        from proxy.pipeline.exceptions import StageAbort

        st = self._enable_secret(monkeypatch)
        token = st.create_session_token("client-A", self.SID_A)
        assert token

        body = {
            "model": "test-model",
            "messages": [{"role": "user", "content": "cross-client attempt"}],
        }
        headers = {"X-Session-Client": "client-B", "X-Session-Token": token}

        with pytest.raises(StageAbort) as exc:
            parse_request(_make_fake_request(body, headers))
        assert exc.value.response_data[1] == 403

    def test_subagent_does_not_restore_parent_session(self, monkeypatch):
        """A sub-agent that echoes the parent's token must NOT collapse onto
        the parent's session_id — the restore is gated on ``not is_subagent``.

        The first user message carries a bare sub-agent canary so
        ``detect_subagent_ingress`` sets is_subagent=True.
        """
        PARENT_SID = self.SID_A
        st = self._enable_secret(monkeypatch)
        token = st.create_session_token("test-client", PARENT_SID)
        assert token

        body = {
            "model": "test-model",
            "messages": [{"role": "user", "content": "[[__SUBAGENT__]] real content"}],
        }
        headers = {"X-Session-Client": "test-client", "X-Session-Token": token}

        ctx = parse_request(_make_fake_request(body, headers))
        assert ctx.is_subagent is True
        assert ctx.session_id != PARENT_SID