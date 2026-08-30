import pytest

from clients.model_registry import ModelConfig, ModelRegistry
from proxy.model_registry import resolve_model_route


def _registry_with_backend(backend: str) -> ModelRegistry:
    registry = ModelRegistry.__new__(ModelRegistry)
    registry._models = {
        "url_construction_test": ModelConfig(backend=backend, tool_parser="xml"),
    }
    return registry


@pytest.mark.parametrize(
    "backend,expected",
    [
        (
            "https://opencode.ai/zen/go/v1",
            "https://opencode.ai/zen/go/v1/chat/completions",
        ),
        (
            "https://opencode.ai/zen/go/v1/",
            "https://opencode.ai/zen/go/v1/chat/completions",
        ),
        ("https://api.minimax.io/v1", "https://api.minimax.io/v1/chat/completions"),
        ("https://api.minimax.io/v1/", "https://api.minimax.io/v1/chat/completions"),
        (
            "https://api.example.com/v1/chat/completions",
            "https://api.example.com/v1/chat/completions",
        ),
        (
            "https://api.example.com/v1/chat/completions/",
            "https://api.example.com/v1/chat/completions",
        ),
        ("https://api.example.com", "https://api.example.com/v1/chat/completions"),
        ("https://api.example.com/", "https://api.example.com/v1/chat/completions"),
        ("http://localhost:7777", "http://localhost:7777/v1/chat/completions"),
    ],
)
def test_url_construction(backend, expected):
    registry = _registry_with_backend(backend)
    route = resolve_model_route("url_construction_test", registry)
    assert route.upstream_url == expected
