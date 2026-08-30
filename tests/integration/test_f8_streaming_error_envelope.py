"""US-xxx / F8 regression: streaming worker exceptions map to typed SSE
envelopes, NOT the opaque "Internal stream error".

Pinned so reverting the F8 mapping (back to the old collapsed envelope)
fails this test. Exercises the real production call site — the
``outcome["error"]`` branch in ``app._keepalive_stream`` — by forcing a
RuntimeError inside the completion worker via a side_effect injected on
``pipeline.run``.
"""

import json
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler
from unittest.mock import patch

import pytest

from proxy.agent.agent import Agent as _RealAgent
from proxy.agent.agent_registry import registry as _agent_registry
from proxy.model_registry import resolve_session_id
from proxy.skills.skill_definition import ModeState as _ModeState


class _FastBackendHandler(BaseHTTPRequestHandler):
    """Returns valid text immediately; we inject the worker error separately."""

    _responses = []

    def do_POST(self):
        content_length = int(self.headers.get("Content-Length", 0))
        self.rfile.read(content_length)
        idx = _FastBackendHandler._request_idx
        _FastBackendHandler._request_idx += 1
        if idx < len(_FastBackendHandler._responses):
            payload = _FastBackendHandler._responses[idx]
        else:
            payload = {
                "choices": [{"message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}],
                "model": "test/llm-v1",
            }
        body_bytes = json.dumps(payload).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(body_bytes)

    def log_message(self, *args):
        pass


def _split_sse(body):
    return [c for c in body.split("\n\n") if c]


def _sse_data_payloads(body):
    done_seen = False
    payloads = []
    events = _split_sse(body)
    for idx, event in enumerate(events):
        non_empty = [l for l in event.split("\n") if l]
        if not non_empty:
            continue
        if all(l.startswith(":") for l in non_empty):
            continue
        for line in non_empty:
            assert line.startswith("data: "), f"non-data line: {line!r}"
        data_vals = [l[len("data: "):] for l in non_empty]
        joined = "\n".join(data_vals)
        if joined == "[DONE]":
            done_seen = True
            assert idx == len(events) - 1, "[DONE] not last"
            continue
        assert not done_seen, "event after [DONE]"
        payloads.append(json.loads(joined))
    assert done_seen, "no [DONE]"
    return payloads


@pytest.fixture
def f8_backend(tmp_path, monkeypatch):
    server = HTTPServer(("127.0.0.1", 0), _FastBackendHandler)
    port = server.server_address[1]
    _FastBackendHandler._responses = []
    _FastBackendHandler._request_idx = 0
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()

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
    path = tmp_path / "proxy_models_f8.json"
    path.write_text(json.dumps(config))
    from clients.model_registry import ModelRegistry
    import requests as _r

    reg = ModelRegistry(config_path="proxy_models_f8.json", base_dir=str(tmp_path))
    monkeypatch.setattr("qwen_proxy._model_registry", reg)
    monkeypatch.setattr("qwen_proxy.USE_ADAPTERS", True)
    monkeypatch.setattr("proxy.app._model_registry", reg)
    monkeypatch.setattr("proxy.model_registry._model_registry", reg)
    monkeypatch.setattr("proxy.state._HTTP_SESSION", _r.Session())
    monkeypatch.setenv("TEST_API_KEY", "sk-test-secret-key-12345")
    import qwen_proxy
    qwen_proxy.app.config["TESTING"] = True
    yield qwen_proxy.app.test_client(), server
    server.shutdown()


@pytest.mark.integration
def test_f8_runtime_error_maps_to_502_envelope(f8_backend):
    """A RuntimeError in the completion worker MUST produce a 502-shaped
    SSE envelope (code=backend_error, type=runtime_error), NOT the opaque
    'Internal stream error' / proxy_error that the old code emitted.
    """
    client, _server = f8_backend
    _FastBackendHandler._responses = [
        {"choices": [{"message": {"role": "assistant", "content": "hi"}, "finish_reason": "stop"}],
         "model": "test/llm-v1"},
    ]

    def _raise(self, ctx):
        raise RuntimeError("injected upstream-processing failure")

    from proxy.pipeline.request_pipeline import RequestPipeline
    with patch.object(RequestPipeline, "run", _raise):
        response = client.post(
            "/v1/chat/completions",
            json={"model": "test-model",
                  "messages": [{"role": "user", "content": "hello"}],
                  "stream": True},
            headers={"X-Session-ID": "sess-f8-runtime-1"},
        )
        # Consume the stream while the patch is active: the completion worker runs
        # pipeline.run(ctx) async in the coordinator pool; future.result() inside
        # get_data() blocks until it finishes. Reading outside the patch races the pool.
        assert response.status_code == 200
        assert response.content_type == "text/event-stream"
        body = response.get_data(as_text=True)

    payloads = _sse_data_payloads(body)
    error_payloads = [p for p in payloads if isinstance(p, dict) and "error" in p]
    assert len(error_payloads) == 1, f"expected 1 error payload, got {payloads!r}"
    err = error_payloads[0]["error"]
    # F8: typed envelope, not the old opaque proxy_error.
    assert err["type"] == "runtime_error", f"expected runtime_error type, got {err!r}"
    assert err["code"] == "backend_error", f"expected backend_error code, got {err!r}"
    assert "Backend processing error" in err["message"]


@pytest.mark.integration
def test_f8_value_error_maps_to_400_envelope(f8_backend):
    """A ValueError in the completion worker maps to a 400-shaped envelope."""
    client, _server = f8_backend
    _FastBackendHandler._responses = [
        {"choices": [{"message": {"role": "assistant", "content": "hi"}, "finish_reason": "stop"}],
         "model": "test/llm-v1"},
    ]

    def _raise(self, ctx):
        raise ValueError("bad params injected")

    from proxy.pipeline.request_pipeline import RequestPipeline
    with patch.object(RequestPipeline, "run", _raise):
        response = client.post(
            "/v1/chat/completions",
            json={"model": "test-model",
                  "messages": [{"role": "user", "content": "hello"}],
                  "stream": True},
            headers={"X-Session-ID": "sess-f8-value-1"},
        )
        # Consume the stream while the patch is active: the completion worker runs
        # pipeline.run(ctx) async in the coordinator pool; future.result() inside
        # get_data() blocks until it finishes. Reading outside the patch races the pool.
        assert response.status_code == 200  # streaming wire status stays 200
        body = response.get_data(as_text=True)

    payloads = _sse_data_payloads(body)
    error_payloads = [p for p in payloads if isinstance(p, dict) and "error" in p]
    assert len(error_payloads) == 1
    err = error_payloads[0]["error"]
    assert err["type"] == "validation_error", f"expected validation_error, got {err!r}"
    assert err["code"] == "invalid_request"


@pytest.mark.integration
def test_f8_unmapped_exception_maps_to_generic_proxy_error(f8_backend):
    """An unexpected exception type falls back to proxy_error (not a crash)."""
    client, _server = f8_backend
    _FastBackendHandler._responses = [
        {"choices": [{"message": {"role": "assistant", "content": "hi"}, "finish_reason": "stop"}],
         "model": "test/llm-v1"},
    ]

    class _WeirdError(Exception):
        pass

    def _raise(self, ctx):
        raise _WeirdError("exotic worker failure")

    from proxy.pipeline.request_pipeline import RequestPipeline
    with patch.object(RequestPipeline, "run", _raise):
        response = client.post(
            "/v1/chat/completions",
            json={"model": "test-model",
                  "messages": [{"role": "user", "content": "hello"}],
                  "stream": True},
            headers={"X-Session-ID": "sess-f8-weird-1"},
        )
        # Consume the stream while the patch is active: the completion worker runs
        # pipeline.run(ctx) async in the coordinator pool; future.result() inside
        # get_data() blocks until it finishes. Reading outside the patch races the pool.
        assert response.status_code == 200
        body = response.get_data(as_text=True)

    payloads = _sse_data_payloads(body)
    error_payloads = [p for p in payloads if isinstance(p, dict) and "error" in p]
    assert len(error_payloads) == 1
    err = error_payloads[0]["error"]
    assert err["type"] == "proxy_error", f"expected proxy_error fallback, got {err!r}"
    assert err["code"] == "internal_error"
