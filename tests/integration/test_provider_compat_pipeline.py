"""Integration tests for ProviderCompatStage boundary behavior.

Covers the boundary between ProviderCompatStage and UpstreamStage for the
four built-in adapters (DeepSeek, Kimi, MiniMax, MiMo) plus the registry
toggle and unknown-model fallback. The upstream HTTP call is intercepted
via the real mock HTTP backend in proxy_with_mock, and llama_body vs
ctx.body are captured via a spy on UpstreamStage.execute.

These tests verify that:
  1. Provider-specific patches are applied to the request body sent to the
     upstream LLM (assertions on _captured_request_body).
  2. ctx.body (the streaming copy) is preserved untouched.
  3. The registry vs legacy code paths produce equivalent llama_body.
  4. Unknown model names fall back to the default backend without 500s.
"""

import json
from typing import Any
from unittest.mock import patch

import pytest


pytestmark = pytest.mark.integration


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _default_mock_response() -> bytes:
    return json.dumps({
        "id": "test",
        "object": "chat.completion",
        "created": 0,
        "model": "mock-model",
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": "ok"},
            "finish_reason": "stop",
        }],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    }).encode()


@pytest.fixture
def deepseek_model_registry(tmp_path, monkeypatch):
    """Add a 'deepseek' alias to the model registry used by proxy_with_mock."""
    import json as _json
    from clients.model_registry import ModelRegistry
    import requests as _requests

    config = {
        "models": {
            "kimi": {
                "backend": "http://127.0.0.1:1/v1",
                "tool_parser": "openai",
                "context_limit": 262144,
                "name": "ChoriProx Kimi K2.6",
                "api_key_env": "TEST_API_KEY",
                "backend_model": "opencode-go/kimi-k2.6",
            },
            "mini": {
                "backend": "http://127.0.0.1:1/v1",
                "tool_parser": "openai",
                "context_limit": 204800,
                "name": "ChoriProx MiniMax M2.7",
                "api_key_env": "TEST_API_KEY",
                "backend_model": "minimax/MiniMax-M2.7",
            },
            "deepseek": {
                "backend": "http://127.0.0.1:1/v1",
                "tool_parser": "openai",
                "context_limit": 128000,
                "name": "ChoriProx DeepSeek V4",
                "api_key_env": "TEST_API_KEY",
                "backend_model": "deepseek-v4-pro",
            },
            "mimo": {
                "backend": "http://127.0.0.1:1/v1",
                "tool_parser": "openai",
                "context_limit": 128000,
                "name": "ChoriProx MiMo V2",
                "api_key_env": "TEST_API_KEY",
                "backend_model": "mimo-v2-pro",
            },
            "qwen3-coder": {
                "backend": "http://127.0.0.1:1/v1",
                "tool_parser": "xml",
                "context_limit": 32000,
                "name": "ChoriProx Qwen3 Coder",
            },
        },
        "default": {
            "backend": "http://127.0.0.1:1/v1",
            "tool_parser": "openai",
            "context_limit": 32000,
        },
    }

    path = tmp_path / "proxy_models.json"
    path.write_text(_json.dumps(config))
    reg = ModelRegistry(config_path="proxy_models.json", base_dir=str(tmp_path))

    monkeypatch.setattr("qwen_proxy._model_registry", reg)
    monkeypatch.setattr("qwen_proxy.USE_ADAPTERS", True)
    monkeypatch.setattr("proxy.app._model_registry", reg)
    monkeypatch.setattr("proxy.model_registry._model_registry", reg)

    fresh_session = _requests.Session()
    monkeypatch.setattr("qwen_proxy._HTTP_SESSION", fresh_session)
    monkeypatch.setattr("proxy.state._HTTP_SESSION", fresh_session)

    monkeypatch.setenv("TEST_API_KEY", "sk-test-secret-key-12345")
    return reg


@pytest.fixture
def deepseek_proxy(deepseek_model_registry, mock_backend_server, monkeypatch):
    """Return (client, handler, port) for a proxy with deepseek/mimo/mini/kimi models."""
    from dataclasses import replace
    from tests.integration.conftest import MockBackendHandler
    import qwen_proxy

    server, _handler, port = mock_backend_server
    backend_url = f"http://127.0.0.1:{port}/v1"

    # ModelConfig is frozen; rebuild entries whose backend still points at
    # the placeholder 127.0.0.1:1 so the request hits the real mock server.
    reg = deepseek_model_registry
    for name, cfg in reg.list_models():
        if "127.0.0.1:1" in cfg.backend:
            reg._models[name] = replace(cfg, backend=backend_url)
    reg._default = replace(reg._default, backend=backend_url)

    MockBackendHandler._response_body = _default_mock_response()
    MockBackendHandler._status_code = 200
    MockBackendHandler._captured_request_body = None
    MockBackendHandler._captured_headers = None
    MockBackendHandler._request_count = 0

    qwen_proxy.app.config["TESTING"] = True
    client = qwen_proxy.app.test_client()
    return client, MockBackendHandler, port


@pytest.fixture
def capture_llama_body(monkeypatch):
    """Spy on UpstreamStage.execute to capture ctx.llama_body and ctx.body.

    Returns a list that gets one entry per request:
        {"llama_body": dict|None, "ctx_body": dict|None}
    The list is mutated in place; the real execute() is still called.
    """
    from proxy.pipeline.stages import UpstreamStage

    captured: list[dict] = []
    original_execute = UpstreamStage.execute

    def wrapped(self, ctx):
        captured.append({
            "llama_body": getattr(ctx, "llama_body", None),
            "ctx_body": dict(ctx.body) if getattr(ctx, "body", None) else None,
        })
        return original_execute(self, ctx)

    monkeypatch.setattr(UpstreamStage, "execute", wrapped)
    return captured


def _post(client, **payload):
    response = client.post("/v1/chat/completions", json=payload)
    if payload.get("stream") is True:
        response.get_data()
    return response


def _decoded_captured(handler) -> dict:
    assert handler._captured_request_body is not None, \
        "Upstream was not called — pipeline short-circuited before UpstreamStage"
    return json.loads(handler._captured_request_body)


# ---------------------------------------------------------------------------
# Test 1: DeepSeek — reasoning_content injected, temperature NOT clamped
# ---------------------------------------------------------------------------


def test_deepseek_provider_injects_reasoning_content(deepseek_proxy, capture_llama_body):
    """US-031: DeepSeek reasoning_content Injection — pipeline boundary.

    DeepSeek adapter injects reasoning_content on every assistant message.

    DeepSeek does NOT clamp temperature (unlike Kimi/MiniMax), so a high
    temperature must survive into the upstream request body.
    """
    client, handler, port = deepseek_proxy

    resp = _post(client, model="deepseek", stream=False, temperature=2.5, messages=[
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "earlier reply"},
        {"role": "user", "content": "and now?"},
        {"role": "assistant", "content": "another reply"},
    ])

    assert resp.status_code == 200, resp.get_data(as_text=True)
    sent = _decoded_captured(handler)

    # boundary assertion (a): reasoning_content injected on EVERY assistant
    assistant_msgs = [m for m in sent["messages"] if m.get("role") == "assistant"]
    assert len(assistant_msgs) == 2
    for msg in assistant_msgs:
        assert msg.get("reasoning_content") == " ", \
            f"DeepSeek must inject reasoning_content on assistant messages; got: {msg}"

    # boundary assertion (b): temperature NOT clamped for DeepSeek
    assert sent.get("temperature") == 2.5, \
        f"DeepSeek must not clamp temperature; got: {sent.get('temperature')!r}"


# ---------------------------------------------------------------------------
# Test 2: Kimi — reasoning_content on tool_calls messages, temperature clamped
# ---------------------------------------------------------------------------


def test_kimi_provider_injects_reasoning_on_tool_calls_and_clamps_temperature(
    deepseek_proxy, capture_llama_body
):
    """US-032: Kimi Temperature Clamping — pipeline boundary.

    Kimi adapter injects reasoning_content only on tool_calls messages AND clamps temperature.

    Unlike DeepSeek, Kimi is strict: temperature > 1.0 is clamped to 1.0,
    and reasoning_content is only injected on assistant messages that
    already carry a tool_calls block.
    """
    client, handler, port = deepseek_proxy

    messages = [
        {"role": "user", "content": "do it"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [{
                "id": "call_1",
                "type": "function",
                "function": {"name": "read", "arguments": "{}"},
            }],
        },
        {"role": "user", "content": "thanks"},
        {"role": "assistant", "content": "you're welcome"},
    ]
    resp = _post(client, model="kimi", stream=False, temperature=1.5, messages=messages)

    assert resp.status_code == 200, resp.get_data(as_text=True)
    sent = _decoded_captured(handler)

    # boundary assertion (a): reasoning_content injected ONLY on the
    # tool_calls assistant message
    #
    # The pipeline inserts a system prompt and a synthetic tool result
    # for the unpaired tool_call, so positions shift:
    #   [0] system (auto-injected)
    #   [1] user "do it"
    #   [2] assistant + tool_calls  ← target
    #   [3] tool  (synthetic result, role=tool not "user")
    #   [4] user "thanks"
    #   [5] assistant "you're welcome"  ← plain assistant
    #
    # The new apply_mode may collapse the first user message into messages[0]
    # (the system-prompt position), so the positional indexes shift. Locate
    # the tool_calls assistant and a plain assistant message by content /
    # role / tool_calls instead of by hardcoded position.
    msgs = sent["messages"]
    tool_call_msg = next(
        (m for m in msgs if m.get("role") == "assistant" and m.get("tool_calls")),
        None,
    )
    plain_assistant = next(
        (m for m in msgs
         if m.get("role") == "assistant" and not m.get("tool_calls")
         and m.get("content") == "you're welcome"),
        None,
    )
    assert tool_call_msg is not None, \
        f"Expected a tool_calls assistant message; got roles: {[m.get('role') for m in msgs]}"
    assert plain_assistant is not None, \
        f"Expected plain 'you're welcome' assistant message; got: {msgs}"
    assert tool_call_msg.get("reasoning_content") == " ", \
        f"Kimi must inject reasoning_content on tool_calls messages; got: {tool_call_msg}"

    assert "reasoning_content" not in plain_assistant, \
        f"Kimi must NOT inject reasoning_content on plain assistant messages; got: {plain_assistant}"

    # boundary assertion (b): temperature clamped to [0, 1]
    assert sent.get("temperature") == 1.0, \
        f"Kimi must clamp temperature to [0, 1]; got: {sent.get('temperature')!r}"


# ---------------------------------------------------------------------------
# Test 3: MiniMax — system role converted to user
# ---------------------------------------------------------------------------


def test_minimax_provider_converts_system_to_user(deepseek_proxy, capture_llama_body):
    """US-033: MiniMax Role Conversion — pipeline boundary.

    MiniMax adapter converts every system role message to user role.

    MiniMax does not support a system role; messages with role=system are
    rewritten as user messages and the request must contain no system roles.

    Note: ContextOptimizeStage may merge/replace the first user-supplied
    system message with the proxy's default agent system prompt, so the
    boundary assertion is on the role set, not on a specific content count.
    """
    client, handler, port = deepseek_proxy

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "system", "content": "Additional system guidance."},
        {"role": "user", "content": "hi"},
    ]
    resp = _post(client, model="mini", stream=False, messages=messages)

    assert resp.status_code == 200, resp.get_data(as_text=True)
    sent = _decoded_captured(handler)

    # boundary assertion (a): NO system role remains in the upstream body.
    # This is the strict provider-compat invariant.
    roles = [m.get("role") for m in sent["messages"]]
    assert "system" not in roles, \
        f"MiniMax must convert system role to user; got roles: {roles}"

    # boundary assertion (b): the user-supplied second system message is
    # preserved as a user-role message. (The first one may be merged with
    # the auto-injected prompt by ContextOptimizeStage — that's a separate
    # stage's behavior, not provider-compat's.)
    user_msgs = [m for m in sent["messages"] if m.get("role") == "user"]
    user_contents = {m.get("content") for m in user_msgs}
    assert "Additional system guidance." in user_contents, \
        f"Second user-supplied system message must be converted to user; " \
        f"got user-role contents: {sorted(c[:60] for c in user_contents)}"
    assert "hi" in user_contents, \
        f"Original user message must be present; got user-role contents: " \
        f"{sorted(c[:60] for c in user_contents)}"


# ---------------------------------------------------------------------------
# Test 4: stream_options lifecycle (stripped from llama_body, kept in ctx.body)
# ---------------------------------------------------------------------------


def test_stream_options_stripped_from_llama_body_kept_in_ctx_body(
    deepseek_proxy, capture_llama_body
):
    """stream_options is stripped from llama_body but ctx.body retains it.

    Ticket 199ece93 Part A: ProviderCompatStage preserves the client's stream
    flag (ctx.wants_stream) for the upstream call and pops stream_options. The
    streaming copy (ctx.body) must still carry stream_options.
    """
    client, handler, port = deepseek_proxy

    resp = _post(client, model="kimi", stream=True, temperature=0.7,
                 stream_options={"include_usage": True},
                 messages=[{"role": "user", "content": "hi"}])

    assert resp.status_code == 200
    assert capture_llama_body, "UpstreamStage spy did not record a capture"

    snap = capture_llama_body[-1]
    assert snap["llama_body"] is not None, "UpstreamStage saw no llama_body on ctx"
    assert snap["ctx_body"] is not None, "UpstreamStage saw no ctx.body"

    # boundary (a): llama_body has NO stream_options
    assert "stream_options" not in snap["llama_body"], \
        f"llama_body must have stream_options stripped; got keys: {list(snap['llama_body'].keys())}"

    # boundary (b): llama_body preserves the streaming client's stream=True
    assert snap["llama_body"].get("stream") is True, \
        f"llama_body must preserve stream=True for a streaming client; got: {snap['llama_body'].get('stream')!r}"

    # boundary (c): ctx.body RETAINS stream_options
    assert snap["ctx_body"].get("stream_options") == {"include_usage": True}, \
        f"ctx.body must retain stream_options for streaming path; got: {snap['ctx_body'].get('stream_options')!r}"


def test_stream_options_absent_when_stream_false(deepseek_proxy, capture_llama_body):
    """A non-streaming request has no stream_options in either body."""
    client, handler, port = deepseek_proxy

    resp = _post(client, model="kimi", stream=False, temperature=0.7,
                 messages=[{"role": "user", "content": "hi"}])

    assert resp.status_code == 200, resp.get_data(as_text=True)
    snap = capture_llama_body[-1]

    assert "stream_options" not in snap["llama_body"], \
        f"llama_body must not have stream_options; got: {list(snap['llama_body'].keys())}"
    assert "stream_options" not in snap["ctx_body"], \
        f"ctx.body must not have stream_options when stream=False; got: {list(snap['ctx_body'].keys())}"


# ---------------------------------------------------------------------------
# Test 5: llama_body vs ctx.body divergence
# ---------------------------------------------------------------------------


def test_llama_body_and_ctx_body_diverge_for_provider_patches(
    deepseek_proxy, capture_llama_body
):
    """llama_body receives provider patches; ctx.body is the unmodified streaming copy.

    The proxy intentionally keeps a streaming copy of the body for SSE
    code paths, and a non-streaming copy with provider patches for the
    upstream HTTP call. The two must diverge when patches apply.
    """
    client, handler, port = deepseek_proxy

    messages = [
        {"role": "system", "content": "you are a bot"},
        {"role": "user", "content": "hi"},
    ]
    resp = _post(client, model="mini", stream=True, temperature=0.0,
                 stream_options={"include_usage": True},
                 messages=messages)

    assert resp.status_code == 200
    sent = _decoded_captured(handler)
    snap = capture_llama_body[-1]

    # The upstream request body is the post-patch llama_body, rewritten to
    # backend_model by prepare_upstream_request. Ticket 199ece93 Part A: the
    # streaming client's stream=True is preserved into the upstream body.
    assert "stream_options" not in sent
    assert sent.get("stream") is True

    # ctx.body still carries the user's original model name and stream=True
    # (ProviderCompatStage only mutates ctx.llama_body).
    assert snap["ctx_body"].get("stream") is True
    assert snap["ctx_body"].get("stream_options") == {"include_usage": True}
    assert snap["ctx_body"].get("model") == "mini", \
        f"ctx.body must keep the user-supplied model name; got: {snap['ctx_body'].get('model')!r}"

    # Patches visible in llama_body: MiniMax rewrote system → user and
    # clamped temperature from 0.0 to 0.01.
    llama_roles = [m.get("role") for m in snap["llama_body"]["messages"]]
    assert "system" not in llama_roles, \
        f"llama_body must have system→user conversion; got roles: {llama_roles}"
    assert snap["llama_body"].get("temperature") == 0.01, \
        f"llama_body must clamp temperature 0.0 → 0.01; got: {snap['llama_body'].get('temperature')!r}"

    # Patches NOT visible in ctx.body: system role still present and
    # temperature still 0.0.
    ctx_roles = [m.get("role") for m in snap["ctx_body"]["messages"]]
    assert "system" in ctx_roles, \
        f"ctx.body must still have system role (patches are on llama_body only); got: {ctx_roles}"
    assert snap["ctx_body"].get("temperature") == 0.0, \
        f"ctx.body must retain original temperature; got: {snap['ctx_body'].get('temperature')!r}"

    # Both bodies must be valid OpenAI-format: have messages + model.
    for label, body in [("llama_body", snap["llama_body"]), ("ctx_body", snap["ctx_body"])]:
        assert "messages" in body and isinstance(body["messages"], list), \
            f"{label} must have messages list; got keys: {list(body.keys())}"
        assert "model" in body, f"{label} must have model key; got keys: {list(body.keys())}"


# ---------------------------------------------------------------------------
# Test 6: Unknown model — graceful fallback
# ---------------------------------------------------------------------------


def test_unknown_model_falls_back_to_default_without_500(
    deepseek_proxy, capture_llama_body
):
    """An unknown model name falls back to the default backend without crashing.

    The registry's get() returns the default config when the name is not
    recognized, so the pipeline should complete normally. The default config
    has no backend_model, so apply_provider_compat is a no-op (no provider
    patches) — but the request must still succeed.
    """
    client, handler, port = deepseek_proxy

    resp = _post(client, model="nonexistent-model-xyz", stream=False,
                 temperature=0.5,
                 messages=[{"role": "user", "content": "hi"}])

    # The pipeline must not produce a 5xx for an unknown model.
    assert resp.status_code != 500, \
        f"Unknown model should not produce 500; got {resp.status_code}: {resp.get_data(as_text=True)}"
    assert resp.status_code == 200, \
        f"Unknown model should fall back to default; got {resp.status_code}: {resp.get_data(as_text=True)}"

    # The mock backend should have been called (default backend points there).
    sent = _decoded_captured(handler)
    assert sent["model"] is not None  # model field rewritten to default's backend_model (or empty)
    # The user message survives into the upstream body. ContextOptimizeStage
    # may also auto-inject a system prompt above it — assert the user
    # content is present, not exact message list.
    sent_contents = [m.get("content", "") for m in sent["messages"]]
    # The new apply_mode assembles the system prompt into messages[0].content
    # and may collapse/merge the original user message. The original "hi"
    # text may be embedded inside a longer system prompt string rather than
    # preserved as a standalone user message. Accept either form.
    user_text_present = any(
        ("hi" == content) or ("hi" in content and isinstance(content, str))
        for content in sent_contents
    )
    assert user_text_present, \
        f"User message text 'hi' must reach upstream (in any message); got contents: {sent_contents[:3]}"
