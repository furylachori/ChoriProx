import json

from clients.model_registry import ModelConfig
from proxy.pipeline.stage_modules.upstream import prepare_upstream_request
from proxy.streaming import stream_response
from proxy.types import ModelRoute


def _make_route(backend_model: str) -> ModelRoute:
    cfg = ModelConfig(
        backend="http://test",
        tool_parser="openai",
        backend_model=backend_model,
    )
    return ModelRoute(cfg, "http://test/v1/chat/completions", None, True)


def _first_sse_chunk_model(message: dict, original_data: dict, original_model=None):
    chunks = list(stream_response(message, original_data, original_model))
    return json.loads(chunks[0][len("data: "):])


def test_model_rewritten_to_backend_model():
    body = {"model": "kimi"}
    route = _make_route("opencode-go/kimi-k2.6")

    upstream = prepare_upstream_request(body, route.model_cfg, route)

    assert body["model"] == "opencode-go/kimi-k2.6"
    assert upstream.original_model == "kimi"


def test_model_not_rewritten_when_backend_model_empty():
    body = {"model": "qwen3-coder"}
    route = _make_route("")

    upstream = prepare_upstream_request(body, route.model_cfg, route)

    assert body["model"] == "qwen3-coder"
    assert upstream.original_model == "qwen3-coder"


def test_response_model_restored():
    body = {"model": "kimi"}
    route = _make_route("opencode-go/kimi-k2.6")

    upstream = prepare_upstream_request(body, route.model_cfg, route)

    data = {"model": body["model"], "choices": []}
    data["model"] = upstream.original_model

    assert data["model"] == "kimi"


def test_stream_response_model_from_original():
    message = {"content": "hello"}
    original_data = {"model": "opencode-go/kimi-k2.6", "id": "test-id"}
    original_model = "kimi"

    base = _first_sse_chunk_model(message, original_data, original_model)

    assert base["model"] == "kimi"


def test_stream_response_model_fallback():
    message = {"content": "hello"}
    original_data = {"model": "backend-model", "id": "test-id"}

    base = _first_sse_chunk_model(message, original_data)

    assert base["model"] == "backend-model"
