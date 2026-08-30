"""CP4-2/CP4-4: Integration tests for concurrent dedup coalescing.

Pins the behavior that two (or more) concurrent identical requests are
coalesced into a single upstream call, with followers receiving the same
artifact as the owner. Also pins the follower-cap 503 path.

Uses a GatedBackendHandler that blocks until a threading.Event is set,
ensuring deterministic ordering: the owner hits the backend (creating a
PENDING dedup entry) BEFORE the follower's lookup_or_insert runs.
"""

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Optional

import pytest

pytestmark = pytest.mark.integration


# ---------------------------------------------------------------------------
# SSE helper
# ---------------------------------------------------------------------------


def _response_dict_to_sse_frames(payload: dict) -> list:
    """Convert a complete OpenAI chat.completion dict into genuine SSE delta
    frames (role -> content fragments -> terminal finish -> [DONE]).
    """
    choice = payload["choices"][0]
    message = choice.get("message", {}) or {}
    finish_reason = choice.get("finish_reason")
    model = payload.get("model", "mock-llm")
    usage = payload.get("usage")

    def _chunk(delta: dict, finish=None, usage_obj=None) -> bytes:
        ch = {"index": 0, "delta": delta, "finish_reason": finish}
        obj: dict = {
            "object": "chat.completion.chunk",
            "model": model,
            "choices": [ch],
        }
        if usage_obj is not None:
            obj["usage"] = usage_obj
        return f"data: {json.dumps(obj)}\n\n".encode()

    frames: list = [_chunk({"role": "assistant"})]

    content = message.get("content")
    if isinstance(content, str) and content:
        if len(content) >= 2:
            mid = len(content) // 2
            frames.append(_chunk({"content": content[:mid]}))
            frames.append(_chunk({"content": content[mid:]}))
        else:
            frames.append(_chunk({"content": content}))

    reasoning = message.get("reasoning_content")
    if isinstance(reasoning, str) and reasoning:
        frames.append(_chunk({"reasoning_content": reasoning}))

    for i, tc in enumerate(message.get("tool_calls") or []):
        fn = tc.get("function", {}) or {}
        frames.append(_chunk({"tool_calls": [{
            "index": i,
            "id": tc.get("id"),
            "type": tc.get("type", "function"),
            "function": {"name": fn.get("name"), "arguments": fn.get("arguments", "")},
        }]}))

    # Terminal chunk carries the finish reason (and usage, if any).
    frames.append(_chunk({}, finish=finish_reason, usage_obj=usage))
    frames.append(b"data: [DONE]\n\n")
    return frames


# ---------------------------------------------------------------------------
# GatedBackendHandler
# ---------------------------------------------------------------------------


class GatedBackendHandler(BaseHTTPRequestHandler):
    """Mock backend that blocks on a gate event before responding.

    Class-level state is reset by the gated_backend fixture before each test.
    """

    _responses: list = []
    _request_count: int = 0
    _captured_bodies: list = []
    _gate: Optional[threading.Event] = None       # handler waits on this before responding
    _backend_hit: Optional[threading.Event] = None  # handler sets this when request arrives

    def do_POST(self):
        content_length = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(content_length)
        body = json.loads(raw) if raw else {}
        GatedBackendHandler._captured_bodies.append(body)
        GatedBackendHandler._request_count += 1

        # Signal that the backend was hit, then wait for the gate
        if GatedBackendHandler._backend_hit is not None:
            GatedBackendHandler._backend_hit.set()
        if GatedBackendHandler._gate is not None:
            GatedBackendHandler._gate.wait(timeout=15)

        # Respond based on request
        idx = GatedBackendHandler._request_count - 1
        if idx < len(GatedBackendHandler._responses):
            payload = GatedBackendHandler._responses[idx]
        else:
            payload = {
                "choices": [{"message": {"role": "assistant", "content": "coalesced-response"}, "finish_reason": "stop"}],
                "model": "mock-llm",
                "id": "chatcmpl-test-123",
                "created": 1700000000,
            }

        wants_stream = body.get("stream") is True
        if wants_stream and isinstance(payload, dict) and payload.get("choices"):
            # Emit genuine SSE
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            for frame in _response_dict_to_sse_frames(payload):
                self.wfile.write(frame)
                self.wfile.flush()
        else:
            body_bytes = json.dumps(payload).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(body_bytes)

    def log_message(self, format, *args):
        pass


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def gated_backend():
    """Start a gated mock HTTP backend."""
    server = HTTPServer(("127.0.0.1", 0), GatedBackendHandler)
    port = server.server_address[1]
    # Reset class state
    GatedBackendHandler._responses = []
    GatedBackendHandler._request_count = 0
    GatedBackendHandler._captured_bodies = []
    GatedBackendHandler._gate = threading.Event()  # NOT set — handler will block
    GatedBackendHandler._backend_hit = threading.Event()  # NOT set — handler sets it
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield server, port
    # Ensure gate is released so handler threads don't hang
    GatedBackendHandler._gate.set()
    server.shutdown()


@pytest.fixture
def dedup_app(gated_backend, tmp_path, monkeypatch):
    """Wire Flask app to the gated mock backend; return the app."""
    from clients.model_registry import ModelRegistry
    import requests as _requests

    server, port = gated_backend
    backend_url = f"http://127.0.0.1:{port}/v1"

    config = {
        "models": {
            "test-model": {
                "backend": backend_url,
                "tool_parser": "openai",
                "context_limit": 262144,
                "name": "Test Model",
                "api_key_env": "TEST_API_KEY",
                "backend_model": "test/llm-v1",
            },
        },
        "default": {"backend": backend_url, "tool_parser": "openai", "context_limit": 32000},
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
    monkeypatch.delenv("PROXY_SESSION_SECRET", raising=False)

    import qwen_proxy
    qwen_proxy.app.config["TESTING"] = True
    return qwen_proxy.app


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_streaming_concurrent_coalescing(dedup_app, gated_backend):
    """Two concurrent identical streaming requests -> one upstream call, same artifact."""
    import proxy.state as state
    from proxy.completion_coordinator import AdmissionKind
    import proxy.completion_coordinator as cc

    baseline_pending = cc.pending_count()

    # Configure mock response
    GatedBackendHandler._responses = [{
        "choices": [{"message": {"role": "assistant", "content": "coalesced-response"}, "finish_reason": "stop"}],
        "model": "mock-llm",
        "id": "chatcmpl-test-123",
        "created": 1700000000,
    }]

    # Tracking wrapper: 2nd lookup waits on backend_hit (ensures PENDING entry exists)
    follower_attached = threading.Event()
    _real_lookup = state.dedup_store.lookup_or_insert
    _lookup_count = [0]
    _lookup_lock = threading.Lock()

    def _tracking_lookup(fingerprint, record_factory, is_stream):
        with _lookup_lock:
            _lookup_count[0] += 1
            count = _lookup_count[0]
        if count >= 2:
            # Wait until the owner has hit the backend (entry is PENDING)
            assert GatedBackendHandler._backend_hit.wait(timeout=10), "owner never hit backend"
        entry, kind = _real_lookup(fingerprint, record_factory, is_stream)
        if kind == AdmissionKind.FOLLOWER:
            follower_attached.set()
        return entry, kind

    state.dedup_store.lookup_or_insert = _tracking_lookup

    try:
        # Request body (identical for both threads)
        body = json.dumps({
            "model": "test-model",
            "messages": [{"role": "user", "content": "hello dedup test"}],
            "stream": True,
        })

        results = [None, None]
        errors = [None, None]
        barrier = threading.Barrier(2, timeout=10)

        def _send_request(idx):
            try:
                barrier.wait()
                client = dedup_app.test_client()
                resp = client.post(
                    "/v1/chat/completions",
                    data=body,
                    content_type="application/json",
                )
                results[idx] = resp
            except Exception as e:
                errors[idx] = e

        threads = [threading.Thread(target=_send_request, args=(i,)) for i in range(2)]
        for t in threads:
            t.start()

        # Wait for follower to attach, then release the gate
        try:
            assert follower_attached.wait(timeout=10), "follower never attached"
            GatedBackendHandler._gate.set()
        finally:
            GatedBackendHandler._gate.set()  # ensure release even on assertion failure

        for t in threads:
            t.join(timeout=15)
    finally:
        state.dedup_store.lookup_or_insert = _real_lookup

    # Assertions
    assert errors == [None, None], f"Thread errors: {errors}"
    assert GatedBackendHandler._request_count == 1, (
        f"Expected 1 upstream call, got {GatedBackendHandler._request_count}"
    )

    # Parse SSE data lines (strip comments like ": connected", ": keepalive")
    def _extract_data_lines(resp):
        text = resp.data.decode("utf-8")
        return [line for line in text.split("\n") if line.startswith("data:")]

    data_0 = _extract_data_lines(results[0])
    data_1 = _extract_data_lines(results[1])
    assert data_0 == data_1, f"SSE data lines differ:\n{data_0}\nvs\n{data_1}"
    assert any("[DONE]" in line for line in data_0), "Missing [DONE] in response 0"
    assert any("[DONE]" in line for line in data_1), "Missing [DONE] in response 1"
    # Content correctness
    assert any("coalesced-response" in line for line in data_0), "Expected content not found"

    # Coordinator state check
    assert cc.pending_count() <= baseline_pending + 1  # allow brief race


def test_nonstream_concurrent_coalescing(dedup_app, gated_backend):
    """Two concurrent identical non-stream requests -> one upstream call, same JSON body."""
    import proxy.state as state
    from proxy.completion_coordinator import AdmissionKind

    # Configure mock response
    GatedBackendHandler._responses = [{
        "choices": [{"message": {"role": "assistant", "content": "coalesced-response"}, "finish_reason": "stop"}],
        "model": "mock-llm",
        "id": "chatcmpl-test-456",
        "created": 1700000000,
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    }]

    # Tracking wrapper: 2nd lookup waits on backend_hit (ensures PENDING entry exists)
    follower_attached = threading.Event()
    _real_lookup = state.dedup_store.lookup_or_insert
    _lookup_count = [0]
    _lookup_lock = threading.Lock()

    def _tracking_lookup(fingerprint, record_factory, is_stream):
        with _lookup_lock:
            _lookup_count[0] += 1
            count = _lookup_count[0]
        if count >= 2:
            # Wait until the owner has hit the backend (entry is PENDING)
            assert GatedBackendHandler._backend_hit.wait(timeout=10), "owner never hit backend"
        entry, kind = _real_lookup(fingerprint, record_factory, is_stream)
        if kind == AdmissionKind.FOLLOWER:
            follower_attached.set()
        return entry, kind

    state.dedup_store.lookup_or_insert = _tracking_lookup

    try:
        # Request body (identical for both threads)
        body = json.dumps({
            "model": "test-model",
            "messages": [{"role": "user", "content": "hello nonstream dedup"}],
            "stream": False,
        })

        results = [None, None]
        errors = [None, None]
        barrier = threading.Barrier(2, timeout=10)

        def _send_request(idx):
            try:
                barrier.wait()
                client = dedup_app.test_client()
                resp = client.post(
                    "/v1/chat/completions",
                    data=body,
                    content_type="application/json",
                )
                results[idx] = resp
            except Exception as e:
                errors[idx] = e

        threads = [threading.Thread(target=_send_request, args=(i,)) for i in range(2)]
        for t in threads:
            t.start()

        # Wait for follower to attach, then release the gate
        try:
            assert follower_attached.wait(timeout=10), "follower never attached"
            GatedBackendHandler._gate.set()
        finally:
            GatedBackendHandler._gate.set()

        for t in threads:
            t.join(timeout=15)
    finally:
        state.dedup_store.lookup_or_insert = _real_lookup

    # Assertions
    assert errors == [None, None], f"Thread errors: {errors}"
    assert GatedBackendHandler._request_count == 1, (
        f"Expected 1 upstream call, got {GatedBackendHandler._request_count}"
    )

    # Both responses should be identical JSON
    json_0 = results[0].get_json()
    json_1 = results[1].get_json()
    assert json_0 == json_1, f"JSON responses differ:\n{json_0}\nvs\n{json_1}"

    # Content correctness
    content_0 = json_0["choices"][0]["message"]["content"]
    assert content_0 == "coalesced-response", f"Unexpected content: {content_0}"


def test_follower_cap_exceeded(dedup_app, gated_backend, monkeypatch):
    """With max_followers=1, 3 concurrent requests -> one gets 503 dedup_follower_cap."""
    import proxy.state as state
    from proxy.completion_coordinator import AdmissionKind

    # Set cap directly on the instance (env var is read at construction time)
    original_cap = state.dedup_store.max_followers_per_key
    state.dedup_store.max_followers_per_key = 1

    try:
        # Configure mock response
        GatedBackendHandler._responses = [{
            "choices": [{"message": {"role": "assistant", "content": "coalesced-response"}, "finish_reason": "stop"}],
            "model": "mock-llm",
            "id": "chatcmpl-test-789",
            "created": 1700000000,
        }]

        # Tracking wrapper: calls 2+ wait on backend_hit; signal when all 3 lookups done
        all_lookups_done = threading.Event()
        _real_lookup = state.dedup_store.lookup_or_insert
        _lookup_count = [0]
        _lookup_lock = threading.Lock()

        def _tracking_lookup(fingerprint, record_factory, is_stream):
            with _lookup_lock:
                _lookup_count[0] += 1
                count = _lookup_count[0]
            if count >= 2:
                # Wait until the owner has hit the backend (entry is PENDING)
                assert GatedBackendHandler._backend_hit.wait(timeout=10), "owner never hit backend"
            result = _real_lookup(fingerprint, record_factory, is_stream)
            with _lookup_lock:
                if _lookup_count[0] >= 3:
                    all_lookups_done.set()
            return result

        state.dedup_store.lookup_or_insert = _tracking_lookup

        # Request body (identical for all 3 threads)
        body = json.dumps({
            "model": "test-model",
            "messages": [{"role": "user", "content": "hello cap test"}],
            "stream": True,
        })

        results = [None, None, None]
        errors = [None, None, None]
        barrier = threading.Barrier(3, timeout=10)

        def _send_request(idx):
            try:
                barrier.wait()
                client = dedup_app.test_client()
                resp = client.post(
                    "/v1/chat/completions",
                    data=body,
                    content_type="application/json",
                )
                results[idx] = resp
            except Exception as e:
                errors[idx] = e

        threads = [threading.Thread(target=_send_request, args=(i,)) for i in range(3)]
        for t in threads:
            t.start()

        # Wait for all lookups to complete, then release the gate
        try:
            assert all_lookups_done.wait(timeout=10), "not all lookups completed"
            GatedBackendHandler._gate.set()
        finally:
            GatedBackendHandler._gate.set()

        for t in threads:
            t.join(timeout=15)

        # Assertions
        assert errors == [None, None, None], f"Thread errors: {errors}"
        assert GatedBackendHandler._request_count == 1, (
            f"Expected 1 upstream call, got {GatedBackendHandler._request_count}"
        )

        # Exactly one response should be 503 with code "dedup_follower_cap"
        status_codes = [r.status_code for r in results]
        cap_exceeded_count = sum(
            1 for r in results
            if r.status_code == 503
            and r.get_json().get("error", {}).get("code") == "dedup_follower_cap"
        )
        assert cap_exceeded_count == 1, (
            f"Expected exactly 1 dedup_follower_cap 503, got {cap_exceeded_count}. "
            f"Status codes: {status_codes}"
        )

        # The other two should be 200 (owner + 1 follower within cap)
        ok_count = sum(1 for r in results if r.status_code == 200)
        assert ok_count == 2, f"Expected 2 OK responses, got {ok_count}. Status codes: {status_codes}"

    finally:
        state.dedup_store.lookup_or_insert = _real_lookup
        state.dedup_store.max_followers_per_key = original_cap
        GatedBackendHandler._gate.set()


def test_nonstream_follower_timeout(dedup_app, gated_backend, monkeypatch):
    """Non-stream follower returns 504 when owner doesn't complete within timeout."""
    import proxy.state as state
    from proxy.completion_coordinator import AdmissionKind

    # Set a very short follower timeout (1s) so the test completes quickly
    monkeypatch.setenv("PROXY_DEDUP_NONSTREAM_FOLLOWER_TIMEOUT", "1")

    GatedBackendHandler._responses = [{
        "choices": [{"message": {"role": "assistant", "content": "slow-response"}, "finish_reason": "stop"}],
        "model": "mock-llm",
        "id": "chatcmpl-test-timeout",
        "created": 1700000000,
    }]

    follower_attached = threading.Event()
    follower_done = threading.Event()
    _real_lookup = state.dedup_store.lookup_or_insert
    _lookup_count = [0]
    _lookup_lock = threading.Lock()

    def _tracking_lookup(fingerprint, record_factory, is_stream):
        with _lookup_lock:
            _lookup_count[0] += 1
            count = _lookup_count[0]
        if count >= 2:
            assert GatedBackendHandler._backend_hit.wait(timeout=10), "owner never hit backend"
        entry, kind = _real_lookup(fingerprint, record_factory, is_stream)
        if kind == AdmissionKind.FOLLOWER:
            follower_attached.set()
        return entry, kind

    state.dedup_store.lookup_or_insert = _tracking_lookup

    body = json.dumps({
        "model": "test-model",
        "messages": [{"role": "user", "content": "hello timeout test"}],
        "stream": False,
    })

    results = [None, None]
    errors = [None, None]
    barrier = threading.Barrier(2, timeout=10)

    def _send_request(idx):
        try:
            barrier.wait()
            client = dedup_app.test_client()
            resp = client.post(
                "/v1/chat/completions",
                data=body,
                content_type="application/json",
            )
            results[idx] = resp
        except Exception as e:
            errors[idx] = e
        finally:
            if results[idx] is not None and results[idx].status_code == 504:
                follower_done.set()

    threads = [threading.Thread(target=_send_request, args=(i,)) for i in range(2)]
    for t in threads:
        t.start()

    try:
        assert follower_attached.wait(timeout=10), "follower never attached"
        # Follower will time out after 1s (gate not released yet)
        assert follower_done.wait(timeout=10), "follower never timed out"
        # Release the gate so the owner can complete
        GatedBackendHandler._gate.set()
        for t in threads:
            t.join(timeout=10)
    finally:
        state.dedup_store.lookup_or_insert = _real_lookup
        GatedBackendHandler._gate.set()

    assert errors == [None, None], f"Thread errors: {errors}"

    # One response should be 504 (follower timeout), one should be 200 (owner)
    status_codes = sorted(r.status_code for r in results if r is not None)
    assert 504 in status_codes, f"Expected a 504 in responses, got {status_codes}"
    assert 200 in status_codes, f"Expected a 200 in responses, got {status_codes}"

    # Verify the 504 has the correct error code
    timeout_resp = next(r for r in results if r is not None and r.status_code == 504)
    err = timeout_resp.get_json()["error"]
    assert err["code"] == "dedup_follower_timeout"
    assert timeout_resp.headers.get("Retry-After") == "5"

    # Owner got the expected content
    ok_resp = next(r for r in results if r is not None and r.status_code == 200)
    assert ok_resp.get_json()["choices"][0]["message"]["content"] == "slow-response"
