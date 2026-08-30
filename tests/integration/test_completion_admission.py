"""F1: streaming admission — synchronous 503 at capacity + no slot leak on disconnect."""
import json
import threading
import time

import pytest

import proxy.completion_coordinator as cc
from tests.integration.test_f8_streaming_error_envelope import (
    f8_backend,
    _FastBackendHandler,
)


@pytest.mark.integration
def test_over_capacity_returns_503_synchronously(f8_backend, monkeypatch):
    """Zero-permit semaphore -> admit() fails its non-blocking acquire -> the
    route returns 503 + Retry-After BEFORE any stream status is committed."""
    client, _server = f8_backend
    monkeypatch.setattr(cc, "_semaphore", threading.Semaphore(0))
    resp = client.post(
        "/v1/chat/completions",
        json={"model": "test-model",
              "messages": [{"role": "user", "content": "hi"}],
              "stream": True},
        headers={"X-Session-ID": "sess-cap-1"},
    )
    assert resp.status_code == 503
    assert resp.headers.get("Retry-After") == "1"
    body = json.loads(resp.get_data(as_text=True))
    assert body["error"]["code"] == "over_capacity"
    assert body["error"]["type"] == "rate_limit"


@pytest.mark.integration
def test_disconnect_does_not_leak_admission_slot(f8_backend):
    """Client disconnect (GeneratorExit) must NOT leak the admission slot."""
    client, _server = f8_backend
    _FastBackendHandler._responses = [
        {"choices": [{"message": {"role": "assistant", "content": "ok"},
                      "finish_reason": "stop"}], "model": "test/llm-v1"},
    ]
    start_pending = cc.pending_count()
    resp = client.post(
        "/v1/chat/completions",
        json={"model": "test-model",
              "messages": [{"role": "user", "content": "bye"}],
              "stream": True},
        headers={"X-Session-ID": "sess-disc-leak-1"},
    )
    assert resp.status_code == 200
    chunk_iter = iter(resp.response)
    assert b": connected" in next(chunk_iter)
    resp.close()  # GeneratorExit -> client disconnect
    deadline = time.time() + 10
    while cc.pending_count() > start_pending and time.time() < deadline:
        time.sleep(0.02)
    assert cc.pending_count() == start_pending, "admission slot leaked after disconnect"
    assert cc._semaphore._value == cc.MAX_CONCURRENT_COMPLETIONS
