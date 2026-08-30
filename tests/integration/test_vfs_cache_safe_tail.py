"""Integration tests for Wave 7 + Wave 8 — provider-bound role contracts
and cache-prefix lifecycle for tail-placed main VFS (US-127).

End-to-end through the real Flask route, the real VFS builder, the real
provider compatibility stage, the real MiniMaxAdapter, the real Kimi
adapter, and the real session-token validation flow.

Assertions inspect actual upstream request bodies captured from
`MockBackendHandler._captured_request_body` after the full production
pipeline. Tests use a real signed session token (production HMAC) for
stable session identity across replays and growing conversations.
"""
import copy
import hashlib
import hmac
import json
import time
import uuid
from dataclasses import replace

import pytest

import proxy.canary
import proxy.session_tokens
import proxy.state as state


pytestmark = pytest.mark.integration


# ---------------------------------------------------------------------------
# Mock backend canned response
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


# ---------------------------------------------------------------------------
# Message helpers
# ---------------------------------------------------------------------------


def _vfs_indices(messages):
    return [
        i for i, m in enumerate(messages)
        if "[VFS CONTEXT" in str(m.get("content") or "")
    ]


MAIN_VFS_MARKER = "[VFS CONTEXT — Cached file state]"
INHERITED_VFS_MARKER = "[VFS INHERITED CONTEXT —"


def _main_vfs_indices(messages):
    return [
        i for i, m in enumerate(messages)
        if MAIN_VFS_MARKER in str(m.get("content") or "")
    ]


def _inherited_vfs_indices(messages):
    return [
        i for i, m in enumerate(messages)
        if INHERITED_VFS_MARKER in str(m.get("content") or "")
    ]


def _has_exact_barrier(messages):
    return any(
        m.get("role") == "assistant" and m.get("content") == " "
        for m in messages
    )


def _has_no_adjacent_users(messages):
    roles = [m.get("role") for m in messages]
    return all(
        not (roles[i] == "user" and roles[i + 1] == "user")
        for i in range(len(roles) - 1)
    )


def _idx(predicate, messages):
    for i, m in enumerate(messages):
        if predicate(m):
            return i
    return -1


def _get_cache_entries(workspace_name):
    """Snapshot the list of VFSEntry for a workspace under vfs_lock."""
    with state.vfs_lock:
        cache = state.vfs_caches.get(workspace_name)
        if not cache:
            return []
        return list(cache.snapshot_entries())


def _replace_entry_timestamp(workspace_name, path, new_cached_at):
    """Replace a single entry's cached_at via dataclasses.replace under lock."""
    with state.vfs_lock:
        cache = state.vfs_caches.get(workspace_name)
        if cache is None:
            return
        entries = list(cache.snapshot_entries())
        for entry in entries:
            if entry.path == path:
                cache.put(replace(entry, cached_at=new_cached_at))
                return


def _cache_get_for_lru(workspace_name, path):
    """Touch an entry via cache.get(path) to update LRU order, under lock."""
    with state.vfs_lock:
        cache = state.vfs_caches.get(workspace_name)
        if cache is None:
            return None
        return cache.get(path)


# ---------------------------------------------------------------------------
# Per-test VFS state isolation
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _wave7_vfs_state(monkeypatch):
    """Enable VFS via monkeypatch; track owned workspaces/client_hints for teardown.

    Session-level cleanup is handled by the broader `reset_shared_state`
    autouse fixture in `tests/integration/conftest.py:130-148` because the
    actual target session_id is derived by `resolve_session_id(body, client_hint)`
    at request time and is not known at setup time.
    """
    run_id = uuid.uuid4().hex[:8]
    owned_client_hints: list[tuple[str, str]] = []
    owned_workspaces: list[str] = []

    monkeypatch.setattr(state, "VFS_ENABLED", True)
    monkeypatch.setattr(state, "VFS_SCOPE", "all")
    monkeypatch.setattr(state, "VFS_TTL_SECONDS", 3600)
    monkeypatch.setattr(state, "VFS_MAX_ENTRIES", 500)
    monkeypatch.setattr(state, "VFS_MAX_ENTRY_BYTES", 1_048_576)
    monkeypatch.setattr(state, "VFS_MAX_WORKSPACES", 20)

    yield {
        "run_id": run_id,
        "owned_client_hints": owned_client_hints,
        "owned_workspaces": owned_workspaces,
    }

    for seed_sid, ch in owned_client_hints:
        state.workspace_store.deactivate(seed_sid, client_hint=ch)
    with state.vfs_lock:
        for ws in owned_workspaces:
            state.vfs_caches.pop(ws, None)
        state.vfs_write_log.clear()
        state.vfs_rollback_snapshots.clear()


def _activate_client_workspace(ctx_state, run_id, workspace_name, client_hint):
    """Activate a client-level named workspace via the seed-then-pop pattern.

    Activates on a seed session under the given client_hint, then pops the
    seed session so the session-level mapping is cleared. The client-level
    mapping in workspace_store._client_active[client_hint] survives and
    resolves the target session via client_hint.
    """
    seed_session = f"vfs-tail-seed-{run_id}-{workspace_name}"
    state.workspace_store.activate(
        seed_session, workspace_name, client_hint=client_hint,
    )
    state.workspace_store.pop_session(seed_session)
    ctx_state["owned_client_hints"].append((seed_session, client_hint))
    if workspace_name not in ctx_state["owned_workspaces"]:
        ctx_state["owned_workspaces"].append(workspace_name)
    return seed_session


def _cache_workspace_file(workspace_name, cache_path, cache_content, cached_by):
    """Cache a file via real _cache_file API (no lock held by caller)."""
    from proxy.vfs import _cache_file
    _cache_file(cache_path, cache_content, workspace_name, cached_by)


def _seed_workspace_and_cache(
    ctx_state, run_id, workspace_name, client_hint, cache_path, cache_content
):
    """Backwards-compatible Wave 7 helper: activate workspace + cache file."""
    seed_session = _activate_client_workspace(
        ctx_state, run_id, workspace_name, client_hint,
    )
    _cache_workspace_file(
        workspace_name, cache_path, cache_content, seed_session,
    )
    return seed_session


def _mint_stable_token(monkeypatch, run_id, client_hint):
    """Generate a test secret, sign a session token, patch both env and
    module caches for session_tokens and canary.

    Returns (token, secret, payload). The proxy restores the embedded
    session_id verbatim on the next request (CP2A stable identity contract).
    """
    secret = f"wave8-{run_id}-secret"
    monkeypatch.setenv("PROXY_SESSION_SECRET", secret)
    monkeypatch.setattr(proxy.session_tokens, "_SESSION_SECRET", secret)
    monkeypatch.setattr(proxy.canary, "_SESSION_SECRET", secret)
    # CP2A: session_id is now an INPUT to the token (embedded + HMAC-signed),
    # restored verbatim on the next request. Derive a stable id from client_hint.
    import hashlib as _hl
    session_id = _hl.sha256(f"{client_hint}:{run_id}".encode()).hexdigest()[:16]
    token = proxy.session_tokens.create_session_token(client_hint, session_id)
    assert token, "create_session_token must return non-empty string"
    payload = proxy.session_tokens.validate_session_token(token)
    assert payload is not None, "freshly minted token must validate"
    assert payload["client_id"] == client_hint
    assert payload["session_id"] == session_id
    return token, secret, payload


# ---------------------------------------------------------------------------
# Flask + mock backend proxy fixture
# ---------------------------------------------------------------------------


@pytest.fixture
def wave7_proxy(tmp_path, monkeypatch, mock_backend_server):
    """Build a Flask test client + mock backend with a multi-model registry."""
    from dataclasses import replace as _dc_replace
    from tests.integration.conftest import MockBackendHandler
    from clients.model_registry import ModelRegistry
    import requests as _requests
    import qwen_proxy

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
        },
        "default": {
            "backend": "http://127.0.0.1:1/v1",
            "tool_parser": "openai",
            "context_limit": 32000,
        },
    }
    path = tmp_path / "proxy_models.json"
    path.write_text(json.dumps(config))
    reg = ModelRegistry(config_path="proxy_models.json", base_dir=str(tmp_path))

    monkeypatch.setattr("qwen_proxy._model_registry", reg)
    monkeypatch.setattr("qwen_proxy.USE_ADAPTERS", True)
    monkeypatch.setattr("proxy.app._model_registry", reg)
    monkeypatch.setattr("proxy.model_registry._model_registry", reg)

    fresh_session = _requests.Session()
    monkeypatch.setattr("qwen_proxy._HTTP_SESSION", fresh_session)
    monkeypatch.setattr("proxy.state._HTTP_SESSION", fresh_session)

    monkeypatch.setenv("TEST_API_KEY", "sk-test-secret-key-12345")

    server, _handler_cls, port = mock_backend_server
    backend_url = f"http://127.0.0.1:{port}/v1"
    for name, cfg in reg.list_models():
        if "127.0.0.1:1" in cfg.backend:
            reg._models[name] = _dc_replace(cfg, backend=backend_url)
    reg._default = _dc_replace(reg._default, backend=backend_url)

    MockBackendHandler._response_body = _default_mock_response()
    MockBackendHandler._status_code = 200
    MockBackendHandler._captured_request_body = None
    MockBackendHandler._captured_headers = None
    MockBackendHandler._request_count = 0

    qwen_proxy.app.config["TESTING"] = True
    client = qwen_proxy.app.test_client()
    return client, MockBackendHandler, port


# ---------------------------------------------------------------------------
# Request helpers
# ---------------------------------------------------------------------------


def _post_and_capture_bytes(
    client, handler, body, client_hint, session_token=None
):
    """POST and return captured upstream bytes (Wave 8 sequential capture).

    Records the request_count BEFORE the call so a sequence of two calls
    always has `second._request_count == first._request_count + 1` and the
    caller can copy first_bytes immediately, before state mutation or
    the second request.
    """
    # CP4-4: clear dedup store so each request hits upstream (this helper's
    # contract requires capturing the upstream request body).
    import proxy.state as _state
    _state.dedup_store.clear()

    before_count = handler._request_count
    headers = {"X-Session-Client": client_hint}
    if session_token is not None:
        headers["X-Session-Token"] = session_token
    response = client.post(
        "/v1/chat/completions",
        json=body,
        headers=headers,
    )
    assert response.status_code == 200, (
        response.get_data(as_text=True)
    )
    assert handler._request_count == before_count + 1, (
        f"Expected exactly one request increment; "
        f"before={before_count}, after={handler._request_count}"
    )
    assert handler._captured_request_body is not None, (
        "Upstream was not called; pipeline short-circuited before UpstreamStage"
    )
    return bytes(handler._captured_request_body)


def _post_and_capture(client, handler, body, client_hint, session_token=None):
    """Backwards-compatible decoded wrapper (Wave 7 tests)."""
    captured = _post_and_capture_bytes(
        client, handler, body, client_hint, session_token=session_token
    )
    return json.loads(captured)


# ---------------------------------------------------------------------------
# Canonical input shapes
# ---------------------------------------------------------------------------


def _ordinary_messages(rid: str):
    return [
        {"role": "user", "content": f"wave7 completed user {rid}"},
        {"role": "assistant", "content": f"wave7 completed assistant {rid}"},
        {"role": "user", "content": f"wave7 live user {rid}"},
    ]


def _multi_tool_messages(rid: str):
    return [
        {"role": "user", "content": f"wave7 completed user {rid}"},
        {"role": "assistant", "content": f"wave7 completed assistant {rid}"},
        {"role": "user", "content": f"wave7 live tool user {rid}"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": f"{rid}-call-a",
                    "type": "function",
                    "function": {
                        "name": "read",
                        "arguments": '{"file_path":"/wave7/a.py"}',
                    },
                },
                {
                    "id": f"{rid}-call-b",
                    "type": "function",
                    "function": {
                        "name": "read",
                        "arguments": '{"file_path":"/wave7/b.py"}',
                    },
                },
            ],
        },
        {"role": "tool", "tool_call_id": f"{rid}-call-a", "content": f"wave7 tool result a {rid}"},
        {"role": "tool", "tool_call_id": f"{rid}-call-b", "content": f"wave7 tool result b {rid}"},
    ]


def _lifecycle_messages(rid: str):
    """3-message shape: completed user, completed assistant, current user."""
    return [
        {"role": "user", "content": f"wave8 completed user {rid}"},
        {"role": "assistant", "content": f"wave8 completed assistant {rid}"},
        {"role": "user", "content": f"wave8 live user {rid}"},
    ]


# ---------------------------------------------------------------------------
# Test 1: MiniMax ordinary provider body
# ---------------------------------------------------------------------------


def test_minimax_provider_body_keeps_vfs_distinct_with_exact_barrier(
    wave7_proxy, _wave7_vfs_state, monkeypatch
):
    """MiniMax ordinary completed/live exchange: VFS distinct, exact barrier."""
    client, handler, _port = wave7_proxy
    rid = _wave7_vfs_state["run_id"]
    workspace_name = f"wave7-mini-ord-{rid}"
    client_hint = f"wave7-mini-ord-client-{rid}"
    cache_path = f"/wave7/mini-ord-{rid}.py"
    cache_content = f"alpha cache content mini-ord {rid}"

    _seed_workspace_and_cache(
        _wave7_vfs_state, rid,
        workspace_name, client_hint, cache_path, cache_content,
    )

    body = {
        "model": "mini",
        "stream": False,
        "messages": _ordinary_messages(rid),
    }
    body_before = copy.deepcopy(body)

    sent = _post_and_capture(client, handler, body, client_hint)

    # Backend model rewrite to MiniMax.
    assert sent.get("model") == "minimax/MiniMax-M2.7", (
        f"Backend model must be rewritten to MiniMax; got {sent.get('model')!r}"
    )

    msgs = sent["messages"]
    # No system role reaches MiniMax.
    assert not any(m.get("role") == "system" for m in msgs), (
        f"No system role should remain in MiniMax output; got "
        f"{[m.get('role') for m in msgs]}"
    )
    # Exactly one VFS block.
    vfs_indices = _vfs_indices(msgs)
    assert len(vfs_indices) == 1, (
        f"Exactly one VFS block expected; got {len(vfs_indices)} at {vfs_indices}"
    )
    vfs_idx = vfs_indices[0]
    vfs_msg = msgs[vfs_idx]
    # VFS has role=user and contains the cached path.
    assert vfs_msg.get("role") == "user", (
        f"VFS message must have role=user after MiniMax conversion; got {vfs_msg}"
    )
    assert cache_path in str(vfs_msg.get("content") or ""), (
        f"Cached path {cache_path!r} must appear in VFS message"
    )
    # VFS does not contain the live-user content.
    assert f"wave7 live user {rid}" not in str(vfs_msg.get("content") or ""), (
        "VFS message must not contain live-user content"
    )
    # Current user remains a separate exact message.
    current_user_idx = _idx(
        lambda m: m.get("role") == "user"
        and (m.get("content") or "") == f"wave7 live user {rid}",
        msgs,
    )
    assert current_user_idx >= 0, (
        f"Live user {f'wave7 live user {rid}'!r} must appear as a separate user message"
    )
    assert vfs_idx < current_user_idx, (
        f"VFS (idx={vfs_idx}) must precede live user (idx={current_user_idx})"
    )
    # Exact barrier immediately after VFS.
    assert msgs[vfs_idx + 1] == {
        "role": "assistant",
        "content": " ",
    }, f"Exact single-space barrier must follow VFS; got msgs[{vfs_idx + 1}]={msgs[vfs_idx + 1]!r}"
    # Current user immediately after barrier.
    assert msgs[vfs_idx + 2] == {
        "role": "user",
        "content": f"wave7 live user {rid}",
    }, f"Current user must follow barrier; got msgs[{vfs_idx + 2}]={msgs[vfs_idx + 2]!r}"
    # No adjacent users.
    assert _has_no_adjacent_users(msgs), (
        f"No adjacent user roles allowed; got roles={[m.get('role') for m in msgs]}"
    )
    # Input body unchanged.
    assert body == body_before, "Input body must be unchanged"


# ---------------------------------------------------------------------------
# Test 2: MiniMax multi-tool provider body
# ---------------------------------------------------------------------------


def test_minimax_provider_body_preserves_multi_tool_unit_after_vfs_barrier(
    wave7_proxy, _wave7_vfs_state, monkeypatch
):
    """MiniMax multi-tool continuation: VFS distinct, exact barrier, live unit contiguous.

    The proxy's V2 path may inject a system prompt at the start (auto-prompt),
    which after MiniMax's system→user conversion produces a system-converted
    user message that itself requires a barrier before the next non-converted
    user. So we cannot assume the VFS barrier is the ONLY assistant(' ') in
    the final list. We instead verify:

    1. Exactly one VFS block, distinct from the live user.
    2. The live exchange immediately follows the VFS with an
       assistant(' ') barrier between VFS and the live user.
    3. Within the live unit (live user → assistant(tool_calls) → tools),
       the assistant and tool messages stay contiguous (no extra barrier,
       no VFS bleed-through).
    """
    client, handler, _port = wave7_proxy
    rid = _wave7_vfs_state["run_id"]
    workspace_name = f"wave7-mini-tool-{rid}"
    client_hint = f"wave7-mini-tool-client-{rid}"
    cache_path = f"/wave7/mini-tool-{rid}.py"
    cache_content = f"alpha cache content mini-tool {rid}"

    _seed_workspace_and_cache(
        _wave7_vfs_state, rid,
        workspace_name, client_hint, cache_path, cache_content,
    )

    body = {
        "model": "mini",
        "stream": False,
        "messages": _multi_tool_messages(rid),
    }

    sent = _post_and_capture(client, handler, body, client_hint)

    # Backend model rewrite to MiniMax.
    assert sent.get("model") == "minimax/MiniMax-M2.7", (
        f"Backend model must be rewritten to MiniMax; got {sent.get('model')!r}"
    )

    msgs = sent["messages"]
    assert not any(m.get("role") == "system" for m in msgs)
    assert _has_no_adjacent_users(msgs)

    vfs_indices = _vfs_indices(msgs)
    assert len(vfs_indices) == 1
    vfs_idx = vfs_indices[0]

    # Find positions in the live exchange (relative to VFS, not absolute).
    live_idx = _idx(
        lambda m: m.get("role") == "user"
        and (m.get("content") or "") == f"wave7 live tool user {rid}",
        msgs,
    )
    assistant_idx = _idx(
        lambda m: m.get("role") == "assistant"
        and isinstance(m.get("tool_calls"), list)
        and len(m["tool_calls"]) == 2,
        msgs,
    )
    tool_a_idx = _idx(
        lambda m: m.get("role") == "tool"
        and m.get("tool_call_id") == f"{rid}-call-a",
        msgs,
    )
    tool_b_idx = _idx(
        lambda m: m.get("role") == "tool"
        and m.get("tool_call_id") == f"{rid}-call-b",
        msgs,
    )
    for label, val in [
        ("live_idx", live_idx),
        ("assistant_idx", assistant_idx),
        ("tool_a_idx", tool_a_idx),
        ("tool_b_idx", tool_b_idx),
    ]:
        assert val >= 0, f"{label} must be present in upstream body"

    # Live exchange immediately follows VFS (with a barrier slot).
    assert vfs_idx < live_idx, (
        f"VFS (idx={vfs_idx}) must precede live user (idx={live_idx})"
    )
    assert vfs_idx + 1 < live_idx, (
        f"At least one message must sit between VFS and live user (VFS idx={vfs_idx}, live_idx={live_idx})"
    )

    # The message immediately after VFS is the barrier, immediately before live user.
    barrier_slot_idx = vfs_idx + 1
    assert barrier_slot_idx == live_idx - 1, (
        f"barrier slot (vfs_idx+1={barrier_slot_idx}) must be immediately before live user (live_idx={live_idx})"
    )
    assert msgs[barrier_slot_idx] == {
        "role": "assistant",
        "content": " ",
    }, (
        f"Exact barrier must occupy the vfs→live slot; got msgs[{barrier_slot_idx}]={msgs[barrier_slot_idx]!r}"
    )

    # Live unit is contiguous: live user, asst(tc), tool_a, tool_b in order.
    assert live_idx + 1 == assistant_idx, (
        f"assistant(tc) must be live_idx + 1 (got assistant_idx={assistant_idx})"
    )
    assert assistant_idx + 1 == tool_a_idx, (
        f"tool_a must be assistant_idx + 1 (got tool_a_idx={tool_a_idx})"
    )
    assert assistant_idx + 2 == tool_b_idx, (
        f"tool_b must be assistant_idx + 2 (got tool_b_idx={tool_b_idx})"
    )

    # Tool-call IDs in correct order.
    asst_msg = msgs[assistant_idx]
    assert [tc.get("id") for tc in asst_msg["tool_calls"]] == [
        f"{rid}-call-a",
        f"{rid}-call-b",
    ], f"Tool-call IDs in wrong order: {[tc.get('id') for tc in asst_msg['tool_calls']]}"
    # Tool-result IDs in same order.
    assert msgs[tool_a_idx].get("tool_call_id") == f"{rid}-call-a"
    assert msgs[tool_b_idx].get("tool_call_id") == f"{rid}-call-b"

    # No barrier inside the assistant/tool unit.
    for i in range(assistant_idx, tool_b_idx + 1):
        assert not (msgs[i].get("role") == "assistant"
                    and msgs[i].get("content") == " "), (
            f"Unexpected barrier inside assistant/tool unit at idx {i}"
        )
    # No VFS marker inside tool results.
    for i in range(tool_a_idx, tool_b_idx + 1):
        assert "[VFS CONTEXT" not in str(msgs[i].get("content") or ""), (
            f"Unexpected VFS marker in tool result at idx {i}"
        )


# ---------------------------------------------------------------------------
# Test 3: Non-MiniMax (Kimi) provider body
# ---------------------------------------------------------------------------


def test_non_minimax_provider_body_keeps_system_vfs_without_barrier(
    wave7_proxy, _wave7_vfs_state, monkeypatch
):
    """Kimi (non-MiniMax) provider: VFS retains role=system, no synthetic barrier."""
    client, handler, _port = wave7_proxy
    rid = _wave7_vfs_state["run_id"]
    workspace_name = f"wave7-kimi-{rid}"
    client_hint = f"wave7-kimi-client-{rid}"
    cache_path = f"/wave7/kimi-{rid}.py"
    cache_content = f"alpha cache content kimi {rid}"

    _seed_workspace_and_cache(
        _wave7_vfs_state, rid,
        workspace_name, client_hint, cache_path, cache_content,
    )

    body = {
        "model": "kimi",
        "stream": False,
        "messages": _multi_tool_messages(rid),
    }

    sent = _post_and_capture(client, handler, body, client_hint)

    # Backend model rewrite to Kimi.
    assert sent.get("model") == "opencode-go/kimi-k2.6", (
        f"Backend model must be rewritten to Kimi; got {sent.get('model')!r}"
    )

    msgs = sent["messages"]
    vfs_indices = _vfs_indices(msgs)
    assert len(vfs_indices) == 1, (
        f"Exactly one VFS block expected; got {len(vfs_indices)}"
    )
    vfs_idx = vfs_indices[0]
    vfs_msg = msgs[vfs_idx]
    # VFS retains role=system for non-MiniMax.
    assert vfs_msg.get("role") == "system", (
        f"Kimi must retain VFS role=system; got {vfs_msg}"
    )
    assert cache_path in str(vfs_msg.get("content") or ""), (
        f"Cached path {cache_path!r} must appear in VFS message"
    )
    # No synthetic barrier.
    assert not _has_exact_barrier(msgs), (
        "Kimi must not receive a synthetic assistant(' ') barrier"
    )
    # Required semantic order.
    completed_user_idx = _idx(
        lambda m: m.get("role") == "user"
        and (m.get("content") or "") == f"wave7 completed user {rid}",
        msgs,
    )
    completed_asst_idx = _idx(
        lambda m: m.get("role") == "assistant"
        and (m.get("content") or "") == f"wave7 completed assistant {rid}",
        msgs,
    )
    live_user_idx = _idx(
        lambda m: m.get("role") == "user"
        and (m.get("content") or "") == f"wave7 live tool user {rid}",
        msgs,
    )
    asst_tc_idx = _idx(
        lambda m: m.get("role") == "assistant"
        and isinstance(m.get("tool_calls"), list)
        and len(m["tool_calls"]) == 2,
        msgs,
    )
    tool_a_idx = _idx(
        lambda m: m.get("role") == "tool"
        and m.get("tool_call_id") == f"{rid}-call-a",
        msgs,
    )
    tool_b_idx = _idx(
        lambda m: m.get("role") == "tool"
        and m.get("tool_call_id") == f"{rid}-call-b",
        msgs,
    )
    assert completed_user_idx < completed_asst_idx < vfs_idx < live_user_idx < asst_tc_idx
    assert tool_a_idx == asst_tc_idx + 1
    assert tool_b_idx == asst_tc_idx + 2

    # Current-user content distinct from VFS.
    assert f"wave7 live tool user {rid}" not in str(vfs_msg.get("content") or ""), (
        "Live user content must not appear in VFS message"
    )
    # Cached path only in VFS.
    cache_occurrences = sum(
        1 for m in msgs if cache_path in str(m.get("content") or "")
    )
    assert cache_occurrences == 1, (
        f"Cached path must appear in exactly one message (VFS); got {cache_occurrences}"
    )


# ===========================================================================
# Wave 8 — provider-bound VFS lifecycle and prefix equality
# ===========================================================================


def test_absent_to_present_preserves_full_completed_prefix(
    wave7_proxy, _wave7_vfs_state, monkeypatch
):
    """No cache → cache one entry. Absent prefix must equal present
    prefix-before-VFS (placement vs front-placement discriminator)."""
    client, handler, _port = wave7_proxy
    rid = _wave7_vfs_state["run_id"]
    workspace_name = f"wave8-absent-{rid}"
    client_hint = f"wave8-absent-client-{rid}"
    cache_path = f"/wave8/absent-{rid}.py"
    cache_content = f"alpha cache content absent {rid}"

    _activate_client_workspace(
        _wave7_vfs_state, rid, workspace_name, client_hint,
    )
    token, _secret, _payload = _mint_stable_token(
        monkeypatch, rid, client_hint,
    )

    body = {"model": "kimi", "stream": False, "messages": _lifecycle_messages(rid)}
    body_before = copy.deepcopy(body)

    # Request 1: no VFS yet.
    first_bytes = _post_and_capture_bytes(
        client, handler, body, client_hint, session_token=token,
    )
    first = json.loads(first_bytes)
    first_msgs = first["messages"]
    assert first.get("model") == "opencode-go/kimi-k2.6"
    assert not any(
        MAIN_VFS_MARKER in str(m.get("content") or "")
        for m in first_msgs
    ), f"Request 1 must have no VFS; got roles={[m.get('role') for m in first_msgs]}"
    abs_completed_user_idx = next(
        i for i, m in enumerate(first_msgs)
        if m.get("role") == "user"
        and (m.get("content") or "") == f"wave8 completed user {rid}"
    )
    abs_completed_asst_idx = next(
        i for i, m in enumerate(first_msgs)
        if m.get("role") == "assistant"
        and (m.get("content") or "") == f"wave8 completed assistant {rid}"
    )
    abs_live_idx = next(
        i for i, m in enumerate(first_msgs)
        if m.get("role") == "user"
        and (m.get("content") or "") == f"wave8 live user {rid}"
    )
    abs_prefix = first_msgs[:abs_completed_asst_idx + 1]
    assert any(
        m.get("role") == "system" and (m.get("content") or "") != ""
        for m in abs_prefix
    ), "Absent prefix must contain a primary system message"
    assert abs_completed_user_idx < abs_completed_asst_idx < abs_live_idx

    # Mutation: cache one entry via real _cache_file.
    seed_session = f"vfs-tail-seed-{rid}-{workspace_name}"
    _cache_workspace_file(
        workspace_name, cache_path, cache_content, seed_session,
    )

    # Request 2: VFS now appears.
    second_bytes = _post_and_capture_bytes(
        client, handler, body, client_hint, session_token=token,
    )
    second = json.loads(second_bytes)
    second_msgs = second["messages"]
    vfs_indices = _main_vfs_indices(second_msgs)
    assert len(vfs_indices) == 1, (
        f"Exactly one VFS block expected; got {len(vfs_indices)}"
    )
    vfs_idx = vfs_indices[0]
    vfs_msg = second_msgs[vfs_idx]
    assert cache_path in str(vfs_msg.get("content") or ""), (
        f"Cached path {cache_path!r} must appear in VFS message"
    )
    expected_hash = hashlib.sha256(cache_content.encode()).hexdigest()[:8]
    assert expected_hash in str(vfs_msg.get("content") or ""), (
        f"Hash prefix {expected_hash!r} must appear in VFS"
    )
    sec_live_idx = next(
        i for i, m in enumerate(second_msgs)
        if m.get("role") == "user"
        and (m.get("content") or "") == f"wave8 live user {rid}"
    )
    assert vfs_idx < sec_live_idx, "VFS must precede live user"

    # Prefix before VFS equals absent prefix.
    sec_prefix = second_msgs[:vfs_idx]
    assert sec_prefix == abs_prefix, (
        f"Present prefix before VFS must equal absent completed prefix; "
        f"abs_prefix roles={[m.get('role') for m in abs_prefix]} vs "
        f"sec_prefix roles={[m.get('role') for m in sec_prefix]}"
    )
    # Incoming body unchanged.
    assert body == body_before, "Input body must be unchanged"


@pytest.mark.parametrize(
    "provider,model_alias,backend_model",
    [
        pytest.param("kimi", "kimi", "opencode-go/kimi-k2.6", id="kimi"),
        pytest.param("mini", "mini", "minimax/MiniMax-M2.7", id="minimax"),
    ],
)
def test_present_to_changed_preserves_prefix_before_each_vfs(
    wave7_proxy, _wave7_vfs_state, monkeypatch, provider, model_alias, backend_model
):
    """Cache one entry → change content. Pre-VFS prefix must remain
    structurally equal; VFS content and hash must change."""
    client, handler, _port = wave7_proxy
    rid = _wave7_vfs_state["run_id"]
    workspace_name = f"wave8-changed-{provider}-{rid}"
    client_hint = f"wave8-changed-client-{provider}-{rid}"
    cache_path = f"/wave8/changed-{provider}-{rid}.py"
    initial_content = f"initial cache content {provider} {rid}"
    changed_content = f"CHANGED cache content {provider} {rid}"

    _activate_client_workspace(
        _wave7_vfs_state, rid, workspace_name, client_hint,
    )
    token, _secret, _payload = _mint_stable_token(
        monkeypatch, rid, client_hint,
    )

    initial_hash = hashlib.sha256(initial_content.encode()).hexdigest()[:8]
    changed_hash = hashlib.sha256(changed_content.encode()).hexdigest()[:8]

    seed_session = f"vfs-tail-seed-{rid}-{workspace_name}"
    _cache_workspace_file(
        workspace_name, cache_path, initial_content, seed_session,
    )

    body = {
        "model": model_alias, "stream": False,
        "messages": _lifecycle_messages(rid),
    }
    body_before = copy.deepcopy(body)

    first_bytes = _post_and_capture_bytes(
        client, handler, body, client_hint, session_token=token,
    )
    first = json.loads(first_bytes)
    first_msgs = first["messages"]
    assert first.get("model") == backend_model
    first_vfs_indices = _main_vfs_indices(first_msgs)
    assert len(first_vfs_indices) == 1
    first_vfs_idx = first_vfs_indices[0]
    first_vfs_content = first_msgs[first_vfs_idx].get("content", "")
    assert cache_path in first_vfs_content
    assert initial_hash in first_vfs_content
    assert changed_hash not in first_vfs_content
    first_prefix = first_msgs[:first_vfs_idx]

    # Mutation: change content via real _cache_file.
    _cache_workspace_file(
        workspace_name, cache_path, changed_content, seed_session,
    )

    second_bytes = _post_and_capture_bytes(
        client, handler, body, client_hint, session_token=token,
    )
    second = json.loads(second_bytes)
    second_msgs = second["messages"]
    second_vfs_indices = _main_vfs_indices(second_msgs)
    assert len(second_vfs_indices) == 1
    second_vfs_idx = second_vfs_indices[0]
    second_vfs_content = second_msgs[second_vfs_idx].get("content", "")
    assert cache_path in second_vfs_content
    assert changed_hash in second_vfs_content, "Changed hash must appear"
    assert initial_hash not in second_vfs_content, "Initial hash must be gone"
    second_prefix = second_msgs[:second_vfs_idx]
    assert second_prefix == first_prefix, (
        "Prefix before VFS must remain equal across content change"
    )
    live_idx = next(
        i for i, m in enumerate(second_msgs)
        if m.get("role") == "user"
        and (m.get("content") or "") == f"wave8 live user {rid}"
    )
    assert second_vfs_idx < live_idx
    assert body == body_before


def test_stable_vfs_is_byte_identical_across_controlled_age_and_lru_access(
    wave7_proxy, _wave7_vfs_state, monkeypatch
):
    """Kimi. Seed 3 paths (zeta, alpha, middle). Restamp to alphabetic MRU
    order, then perform LRU accesses so request 2 enters the renderer
    with non-alphabetic LRU order. VFS message must be byte-identical
    and remain alphabetically sorted across the two requests. AC-10
    deliberately supplies non-alphabetic LRU order to the renderer."""
    client, handler, _port = wave7_proxy
    rid = _wave7_vfs_state["run_id"]
    workspace_name = f"wave8-stable-{rid}"
    client_hint = f"wave8-stable-client-{rid}"

    paths = [
        f"/wave8/zeta-{rid}.py",
        f"/wave8/alpha-{rid}.py",
        f"/wave8/middle-{rid}.py",
    ]
    _activate_client_workspace(
        _wave7_vfs_state, rid, workspace_name, client_hint,
    )
    token, _secret, _payload = _mint_stable_token(
        monkeypatch, rid, client_hint,
    )

    seed_session = f"vfs-tail-seed-{rid}-{workspace_name}"
    for p in paths:
        _cache_workspace_file(
            workspace_name, p, f"content of {p}", seed_session,
        )

    anchor = time.time()
    ttl = state.VFS_TTL_SECONDS
    margin = 60.0

    alphabetic_order = sorted(paths)
    for i, p in enumerate(alphabetic_order):
        _replace_entry_timestamp(
            workspace_name, p, anchor - margin - i,
        )

    body = {"model": "kimi", "stream": False, "messages": _lifecycle_messages(rid)}
    body_before = copy.deepcopy(body)

    first_bytes = _post_and_capture_bytes(
        client, handler, body, client_hint, session_token=token,
    )
    first = json.loads(first_bytes)
    first_msgs = first["messages"]
    first_vfs_indices = _main_vfs_indices(first_msgs)
    assert len(first_vfs_indices) == 1
    first_vfs_idx = first_vfs_indices[0]
    first_vfs_msg = first_msgs[first_vfs_idx]
    first_vfs_content = first_vfs_msg.get("content", "")

    pos_zeta = first_vfs_content.find(paths[0])
    pos_alpha = first_vfs_content.find(paths[1])
    pos_middle = first_vfs_content.find(paths[2])
    assert 0 <= pos_alpha < pos_middle < pos_zeta, (
        f"Paths must be alphabetically ordered: "
        f"alpha@{pos_alpha} middle@{pos_middle} zeta@{pos_zeta}"
    )

    first_prefix = first_msgs[:first_vfs_idx]
    first_vfs_dict_copy = {
        "role": first_vfs_msg.get("role"),
        "content": first_vfs_content,
    }

    # Mutation: restamp first (alphabetic) to normalize MRU order, then
    # perform LRU accesses so request 2 enters the renderer with
    # non-alphabetic LRU order. After alphabetical restamping, cache
    # order is [alpha, middle, zeta]; get(zeta) is a no-op (already MRU);
    # get(middle) moves middle to MRU → final order [alpha, zeta, middle].
    for i, p in enumerate(alphabetic_order):
        _replace_entry_timestamp(
            workspace_name, p, anchor - margin - 100 - i,
        )
    _cache_get_for_lru(workspace_name, paths[0])
    _cache_get_for_lru(workspace_name, paths[2])

    # Fixture precondition: request 2 must enter rendering with
    # non-alphabetic LRU order. AC-10 deliberately supplies
    # non-alphabetic LRU order to the renderer.
    mutated_cache_order = [
        entry.path
        for entry in _get_cache_entries(workspace_name)
    ]
    assert mutated_cache_order == [paths[1], paths[0], paths[2]], (
        f"Request 2 must enter rendering with non-alphabetic LRU order; "
        f"got {mutated_cache_order}"
    )
    assert mutated_cache_order != alphabetic_order

    second_bytes = _post_and_capture_bytes(
        client, handler, body, client_hint, session_token=token,
    )
    second = json.loads(second_bytes)
    second_msgs = second["messages"]
    second_vfs_indices = _main_vfs_indices(second_msgs)
    assert len(second_vfs_indices) == 1
    second_vfs_idx = second_vfs_indices[0]
    second_vfs_msg = second_msgs[second_vfs_idx]
    second_vfs_content = second_vfs_msg.get("content", "")
    assert second_vfs_content == first_vfs_content, (
        "VFS content must be byte-identical across LRU + timestamp mutation"
    )
    second_vfs_dict_copy = {
        "role": second_vfs_msg.get("role"),
        "content": second_vfs_content,
    }
    assert second_vfs_dict_copy == first_vfs_dict_copy
    pos_alpha_2 = second_vfs_content.find(paths[1])
    pos_middle_2 = second_vfs_content.find(paths[2])
    pos_zeta_2 = second_vfs_content.find(paths[0])
    assert 0 <= pos_alpha_2 < pos_middle_2 < pos_zeta_2
    second_prefix = second_msgs[:second_vfs_idx]
    assert second_prefix == first_prefix
    assert body == body_before


def test_present_to_expired_removes_vfs_and_preserves_completed_prefix(
    wave7_proxy, _wave7_vfs_state, monkeypatch
):
    """Kimi. Set entry timestamp → mutate timestamp to before anchor-ttl-margin.
    VFS must disappear; completed prefix preserved; real V2 boundary tuple
    captured; no chain state. AC-11 compares the completed provider-bound
    prefix and real V2 boundary tuple across expiration."""
    client, handler, _port = wave7_proxy
    rid = _wave7_vfs_state["run_id"]
    workspace_name = f"wave8-expired-{rid}"
    client_hint = f"wave8-expired-client-{rid}"
    cache_path = f"/wave8/expired-{rid}.py"
    cache_content = f"expired cache content {rid}"

    cached_hash = hashlib.sha256(cache_content.encode()).hexdigest()[:8]

    _activate_client_workspace(
        _wave7_vfs_state, rid, workspace_name, client_hint,
    )
    token, _secret, _payload = _mint_stable_token(
        monkeypatch, rid, client_hint,
    )

    # Wrap RequestPipeline.run to record V2 boundary tuple after each request.
    from proxy.pipeline.request_pipeline import RequestPipeline
    _boundary_snapshots = []
    _original_run = RequestPipeline.run

    def _wrapped_run(self, ctx):
        result = _original_run(self, ctx)
        _boundary_snapshots.append(
            (
                getattr(ctx, "v2_boundary_idx", None),
                getattr(ctx, "v2_boundary_chain", None),
            )
        )
        return result

    monkeypatch.setattr(RequestPipeline, "run", _wrapped_run)

    seed_session = f"vfs-tail-seed-{rid}-{workspace_name}"
    _cache_workspace_file(
        workspace_name, cache_path, cache_content, seed_session,
    )

    initial_ts = time.time() - 10.0
    _replace_entry_timestamp(workspace_name, cache_path, initial_ts)

    body = {"model": "kimi", "stream": False, "messages": _lifecycle_messages(rid)}
    body_before = copy.deepcopy(body)

    first_bytes = _post_and_capture_bytes(
        client, handler, body, client_hint, session_token=token,
    )
    first = json.loads(first_bytes)
    first_msgs = first["messages"]
    first_vfs_indices = _main_vfs_indices(first_msgs)
    assert len(first_vfs_indices) == 1
    first_vfs_idx = first_vfs_indices[0]
    first_vfs_content = first_msgs[first_vfs_idx].get("content", "")
    assert cache_path in first_vfs_content
    assert cached_hash in first_vfs_content
    first_prefix = first_msgs[:first_vfs_idx]

    # Mutation: replace timestamp so entry is older than TTL+margin.
    ttl = state.VFS_TTL_SECONDS
    new_ts = initial_ts - ttl - 60.0
    _replace_entry_timestamp(workspace_name, cache_path, new_ts)

    second_bytes = _post_and_capture_bytes(
        client, handler, body, client_hint, session_token=token,
    )
    second = json.loads(second_bytes)
    second_msgs = second["messages"]
    assert not any(
        MAIN_VFS_MARKER in str(m.get("content") or "")
        for m in second_msgs
    ), f"Expired VFS must be removed; got roles={[m.get('role') for m in second_msgs]}"
    for m in second_msgs:
        assert cache_path not in str(m.get("content") or ""), (
            f"Expired path {cache_path!r} must be absent"
        )
        assert cached_hash not in str(m.get("content") or ""), (
            f"Expired hash {cached_hash!r} must be absent"
        )
    sec_live_idx = next(
        i for i, m in enumerate(second_msgs)
        if m.get("role") == "user"
        and (m.get("content") or "") == f"wave8 live user {rid}"
    )
    assert sec_live_idx >= 0, "Live user must remain"
    assert not any(
        INHERITED_VFS_MARKER in str(m.get("content") or "")
        for m in second_msgs
    ), "No inherited VFS expected on main path"
    assert body == body_before

    # AC-11: completed-prefix comparison. Request 1 prefix-before-VFS must
    # equal request 2 prefix through the completed assistant anchor.
    completed_assistant_content = f"wave8 completed assistant {rid}"
    second_completed_assistant_idx = next(
        i for i, m in enumerate(second_msgs)
        if m.get("role") == "assistant"
        and (m.get("content") or "") == completed_assistant_content
    )
    second_completed_prefix = second_msgs[:second_completed_assistant_idx + 1]
    assert second_completed_prefix == first_prefix, (
        f"Expired request completed prefix must equal first prefix; "
        f"first len={len(first_prefix)}, second len={len(second_completed_prefix)}"
    )
    assert any(
        m.get("role") == "system" and (m.get("content") or "") != ""
        for m in second_completed_prefix
    ), "Completed prefix must contain primary system"
    assert any(
        m.get("role") == "user"
        and (m.get("content") or "") == f"wave8 completed user {rid}"
        for m in second_completed_prefix
    )
    assert any(
        m.get("role") == "assistant"
        and (m.get("content") or "") == completed_assistant_content
        for m in second_completed_prefix
    )
    assert not any(
        "wave8 live user" in str(m.get("content") or "")
        for m in second_completed_prefix
    ), "Completed prefix must not contain live user"
    assert not any(
        MAIN_VFS_MARKER in str(m.get("content") or "")
        for m in second_completed_prefix
    ), "Completed prefix must not contain main VFS"
    assert not any(
        INHERITED_VFS_MARKER in str(m.get("content") or "")
        for m in second_completed_prefix
    ), "Completed prefix must not contain inherited VFS"

    # AC-11: real V2 boundary tuple across both lifecycle requests.
    # The normal-path assembly sets ctx.v2_boundary_idx = boundary_idx (=0
    # for a fresh session whose first user message is at position 0) and
    # ctx.v2_boundary_chain = None. This pins that the production pipeline
    # does not write None to the boundary field on a normal-path render.
    assert _boundary_snapshots == [(0, None), (0, None)], (
        f"Expected normal-path boundary tuple (0, None) for both renders; "
        f"got {_boundary_snapshots}"
    )


@pytest.mark.parametrize(
    "provider,model_alias,backend_model",
    [
        pytest.param("kimi", "kimi", "opencode-go/kimi-k2.6", id="kimi"),
        pytest.param("mini", "mini", "minimax/MiniMax-M2.7", id="minimax"),
    ],
)
def test_workspace_alpha_to_beta_changes_only_vfs_suffix(
    wave7_proxy, _wave7_vfs_state, monkeypatch, provider, model_alias, backend_model
):
    """Seed alpha/beta/default/unrelated. Switch session from alpha to beta.
    Only VFS suffix changes; prefix preserved; default + unrelated absent."""
    client, handler, _port = wave7_proxy
    rid = _wave7_vfs_state["run_id"]
    alpha = f"wave8-{provider}-alpha-{rid}"
    beta = f"wave8-{provider}-beta-{rid}"
    default = f"wave8-{provider}-default-{rid}"
    unrelated = f"wave8-{provider}-unrelated-{rid}"
    client_hint = f"wave8-ws-client-{provider}-{rid}"
    alpha_path = f"/wave8/{provider}/alpha-only-{rid}.py"
    beta_path = f"/wave8/{provider}/beta-only-{rid}.py"
    default_path = f"/wave8/{provider}/default-only-{rid}.py"
    unrelated_path = f"/wave8/{provider}/unrelated-only-{rid}.py"

    token, _secret, _payload = _mint_stable_token(
        monkeypatch, rid, client_hint,
    )
    session_id = _payload["session_id"]

    # Seed alpha for our session.
    state.workspace_store.set_active(
        session_id, alpha, client_hint=client_hint,
    )
    _wave7_vfs_state["owned_workspaces"].extend([alpha, beta, default, unrelated])
    _cache_workspace_file(
        alpha, alpha_path, f"alpha content {provider} {rid}",
        cached_by=f"vfs-tail-seed-{rid}-{alpha}",
    )
    # Seed beta cache (no session mapping — pure client-hint territory).
    _activate_client_workspace(
        _wave7_vfs_state, rid, beta, client_hint,
    )
    _cache_workspace_file(
        beta, beta_path, f"beta content {provider} {rid}",
        cached_by=f"vfs-tail-seed-{rid}-{beta}",
    )
    # Seed default cache.
    _cache_workspace_file(
        "default", default_path, f"default content {provider} {rid}",
        cached_by=f"vfs-tail-seed-{rid}-{default}",
    )
    # Seed unrelated cache.
    _cache_workspace_file(
        unrelated, unrelated_path, f"unrelated content {provider} {rid}",
        cached_by=f"vfs-tail-seed-{rid}-{unrelated}",
    )
    # Re-activate alpha for the actual session.
    state.workspace_store.set_active(
        session_id, alpha, client_hint=client_hint,
    )

    body = {
        "model": model_alias, "stream": False,
        "messages": _lifecycle_messages(rid),
    }
    body_before = copy.deepcopy(body)

    # Request 1: alpha workspace.
    first_bytes = _post_and_capture_bytes(
        client, handler, body, client_hint, session_token=token,
    )
    first = json.loads(first_bytes)
    first_msgs = first["messages"]
    assert first.get("model") == backend_model
    first_vfs_indices = _main_vfs_indices(first_msgs)
    assert len(first_vfs_indices) == 1
    first_vfs_idx = first_vfs_indices[0]
    first_vfs = first_msgs[first_vfs_idx].get("content", "")
    assert alpha in first_vfs, f"VFS must name alpha; got {first_vfs[:200]}"
    assert alpha_path in first_vfs
    assert beta_path not in first_vfs
    assert default_path not in first_vfs
    assert unrelated_path not in first_vfs
    first_prefix = first_msgs[:first_vfs_idx]

    # Mutation: switch the same session to beta.
    old_ws = state.workspace_store.set_active(
        session_id, beta, client_hint=client_hint,
    )
    assert old_ws == alpha, (
        f"set_active must return previous alpha; got {old_ws!r}"
    )
    resolved = state.workspace_store.get_active(
        session_id, client_hint=client_hint,
    )
    assert resolved == beta, f"Expected beta; got {resolved!r}"

    # Request 2: beta workspace.
    second_bytes = _post_and_capture_bytes(
        client, handler, body, client_hint, session_token=token,
    )
    second = json.loads(second_bytes)
    second_msgs = second["messages"]
    second_vfs_indices = _main_vfs_indices(second_msgs)
    assert len(second_vfs_indices) == 1
    second_vfs_idx = second_vfs_indices[0]
    second_vfs = second_msgs[second_vfs_idx].get("content", "")
    assert beta in second_vfs, f"VFS must name beta; got {second_vfs[:200]}"
    assert beta_path in second_vfs
    assert alpha_path not in second_vfs
    assert default_path not in second_vfs
    assert unrelated_path not in second_vfs
    second_prefix = second_msgs[:second_vfs_idx]
    assert second_prefix == first_prefix
    live_idx = next(
        i for i, m in enumerate(second_msgs)
        if m.get("role") == "user"
        and (m.get("content") or "") == f"wave8 live user {rid}"
    )
    assert second_vfs_idx < live_idx
    assert second_vfs != first_vfs
    assert body == body_before


def test_growing_conversation_preserves_region_before_prior_vfs_boundary(
    wave7_proxy, _wave7_vfs_state, monkeypatch
):
    """Kimi. Body grows from 3 → 5 messages. The leading region (length of
    prior prefix) must equal the request-1 prefix exactly."""
    client, handler, _port = wave7_proxy
    rid = _wave7_vfs_state["run_id"]
    workspace_name = f"wave8-grow-{rid}"
    client_hint = f"wave8-grow-client-{rid}"

    _activate_client_workspace(
        _wave7_vfs_state, rid, workspace_name, client_hint,
    )
    token, _secret, _payload = _mint_stable_token(
        monkeypatch, rid, client_hint,
    )

    seed_session = f"vfs-tail-seed-{rid}-{workspace_name}"
    _cache_workspace_file(
        workspace_name, f"/wave8/grow-{rid}.py", f"grow content {rid}",
        cached_by=seed_session,
    )

    body1_messages = [
        {"role": "user", "content": f"wave8 grow completed user 0 {rid}"},
        {"role": "assistant", "content": f"wave8 grow completed assistant 0 {rid}"},
        {"role": "user", "content": f"wave8 grow live user 1 {rid}"},
    ]
    body1 = {"model": "kimi", "stream": False, "messages": body1_messages}
    body1_before = copy.deepcopy(body1)

    first_bytes = _post_and_capture_bytes(
        client, handler, body1, client_hint, session_token=token,
    )
    first = json.loads(first_bytes)
    first_msgs = first["messages"]
    first_vfs_indices = _main_vfs_indices(first_msgs)
    assert len(first_vfs_indices) == 1
    first_vfs_idx = first_vfs_indices[0]
    first_prefix = first_msgs[:first_vfs_idx]
    assert any(
        m.get("role") == "system" and (m.get("content") or "") != ""
        for m in first_prefix
    ), "Prefix must contain primary system"
    assert any(
        m.get("role") == "user"
        and (m.get("content") or "") == f"wave8 grow completed user 0 {rid}"
        for m in first_prefix
    ), "Prefix must contain completed user 0"
    assert any(
        m.get("role") == "assistant"
        and (m.get("content") or "") == f"wave8 grow completed assistant 0 {rid}"
        for m in first_prefix
    ), "Prefix must contain completed assistant 0"
    assert not any(
        "wave8 grow live user 1" in str(m.get("content") or "")
        for m in first_prefix
    ), "Prefix must not contain live user"

    body2_messages = [
        {"role": "user", "content": f"wave8 grow completed user 0 {rid}"},
        {"role": "assistant", "content": f"wave8 grow completed assistant 0 {rid}"},
        {"role": "user", "content": f"wave8 grow user 1 {rid}"},
        {"role": "assistant", "content": f"wave8 grow assistant 1 {rid}"},
        {"role": "user", "content": f"wave8 grow live user 2 {rid}"},
    ]
    body2 = {"model": "kimi", "stream": False, "messages": body2_messages}
    body2_before = copy.deepcopy(body2)

    second_bytes = _post_and_capture_bytes(
        client, handler, body2, client_hint, session_token=token,
    )
    second = json.loads(second_bytes)
    second_msgs = second["messages"]
    second_vfs_indices = _main_vfs_indices(second_msgs)
    assert len(second_vfs_indices) == 1
    second_vfs_idx = second_vfs_indices[0]
    second_leading = second_msgs[:len(first_prefix)]
    assert second_leading == first_prefix, (
        f"Second request's leading region must equal first request's "
        f"prior prefix; first len={len(first_prefix)}, "
        f"second len={len(second_leading)}"
    )
    middle = second_msgs[len(first_prefix):second_vfs_idx]
    assert any(
        "wave8 grow user 1" in str(m.get("content") or "") for m in middle
    ), f"Middle region must contain user 1; got middle={[m.get('content','')[:50] for m in middle]}"
    assert any(
        "wave8 grow assistant 1" in str(m.get("content") or "") for m in middle
    ), "Middle region must contain assistant 1"
    live_idx = next(
        i for i, m in enumerate(second_msgs)
        if m.get("role") == "user"
        and (m.get("content") or "") == f"wave8 grow live user 2 {rid}"
    )
    assert second_vfs_idx < live_idx, "VFS must precede live user 2"
    assert body1 == body1_before
    assert body2 == body2_before


def test_main_request_receives_only_main_vfs_marker(
    wave7_proxy, _wave7_vfs_state, monkeypatch
):
    """Kimi. Warm main VFS. Exactly one main marker, zero inherited."""
    client, handler, _port = wave7_proxy
    rid = _wave7_vfs_state["run_id"]
    workspace_name = f"wave8-main-isol-{rid}"
    client_hint = f"wave8-main-isol-client-{rid}"

    _activate_client_workspace(
        _wave7_vfs_state, rid, workspace_name, client_hint,
    )
    token, _secret, _payload = _mint_stable_token(
        monkeypatch, rid, client_hint,
    )
    seed_session = f"vfs-tail-seed-{rid}-{workspace_name}"
    _cache_workspace_file(
        workspace_name, f"/wave8/main-isol-{rid}.py",
        f"main isol content {rid}",
        cached_by=seed_session,
    )

    body = {"model": "kimi", "stream": False, "messages": _lifecycle_messages(rid)}
    body_before = copy.deepcopy(body)

    captured = _post_and_capture_bytes(
        client, handler, body, client_hint, session_token=token,
    )
    msgs = json.loads(captured)["messages"]
    main_vfs = _main_vfs_indices(msgs)
    inherited_vfs = _inherited_vfs_indices(msgs)
    assert len(main_vfs) == 1, (
        f"Exactly one main VFS expected; got {len(main_vfs)}"
    )
    assert len(inherited_vfs) == 0, (
        f"Zero inherited VFS expected; got {len(inherited_vfs)}"
    )
    vfs_msg = msgs[main_vfs[0]]
    assert vfs_msg.get("role") == "system", (
        f"Kimi must keep VFS as system; got {vfs_msg.get('role')}"
    )
    assert f"/wave8/main-isol-{rid}.py" in str(vfs_msg.get("content") or ""), (
        "Cached path must appear in VFS message"
    )
    live_idx = next(
        i for i, m in enumerate(msgs)
        if m.get("role") == "user"
        and (m.get("content") or "") == f"wave8 live user {rid}"
    )
    assert main_vfs[0] < live_idx
    assert body == body_before


def test_subagent_request_receives_only_inherited_vfs_marker(
    wave7_proxy, _wave7_vfs_state, monkeypatch
):
    """Kimi. Real WORKSPACE canary → sub-agent. Exactly one inherited
    VFS, zero main VFS. Raw canary must be stripped from upstream body."""
    client, handler, _port = wave7_proxy
    rid = _wave7_vfs_state["run_id"]
    workspace_name = f"wave8-sub-{rid}"
    client_hint = f"wave8-sub-client-{rid}"
    canary_workspace = workspace_name

    _activate_client_workspace(
        _wave7_vfs_state, rid, workspace_name, client_hint,
    )
    token, secret, _payload = _mint_stable_token(
        monkeypatch, rid, client_hint,
    )
    seed_session = f"vfs-tail-seed-{rid}-{workspace_name}"
    _cache_workspace_file(
        workspace_name, f"/wave8/sub-{rid}.py",
        f"sub cache content {rid}",
        cached_by=seed_session,
    )

    ts = int(time.time())
    message = f"workspace:{canary_workspace}::{ts}"
    sig = hmac.new(secret.encode(), message.encode(), hashlib.sha256).hexdigest()
    canary = f"[[__SUBAGENT__:WORKSPACE:{canary_workspace}:{sig}:{ts}]]"

    messages = [
        {"role": "user", "content": f"{canary}\n\nDo something useful {rid}"},
    ]
    body = {"model": "kimi", "stream": False, "messages": messages}
    body_before = copy.deepcopy(body)

    captured = _post_and_capture_bytes(
        client, handler, body, client_hint, session_token=token,
    )
    msgs = json.loads(captured)["messages"]
    main_vfs = _main_vfs_indices(msgs)
    inherited_vfs = _inherited_vfs_indices(msgs)
    assert len(main_vfs) == 0, (
        f"Zero main VFS expected for sub-agent; got {len(main_vfs)}"
    )
    assert len(inherited_vfs) == 1, (
        f"Exactly one inherited VFS expected; got {len(inherited_vfs)}"
    )
    inherited_msg = msgs[inherited_vfs[0]]
    assert inherited_msg.get("role") == "system", (
        f"Kimi must keep inherited VFS as system; got {inherited_msg.get('role')}"
    )
    inherited_content = inherited_msg.get("content", "")
    assert workspace_name in inherited_content, (
        f"Inherited VFS must name workspace {workspace_name!r}"
    )
    assert f"/wave8/sub-{rid}.py" in inherited_content, (
        "Inherited VFS must contain cached path"
    )
    for m in msgs:
        assert "[[__SUBAGENT__" not in str(m.get("content") or ""), (
            f"Raw canary must be stripped; found in message {m}"
        )
    assert not any(
        INHERITED_VFS_MARKER in str(m.get("content") or "")
        and MAIN_VFS_MARKER in str(m.get("content") or "")
        for m in msgs
    ), "No message may contain both main and inherited markers"
    assert body == body_before
