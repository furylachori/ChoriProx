"""Flask test client harness for ChoriProx proxy.

Provides structured helpers for sending chat completion and tool call
requests through the proxy's Flask test client. Works with the existing
conftest.py fixtures (block_all_http, mock_sota_client, mock_llama_response).

Usage:
    from tests.flask_harness import create_test_proxy_app, make_chat_request

    app, client = create_test_proxy_app()
    resp = make_chat_request(client, "sess-1", [{"role": "user", "content": "hi"}])
    assert resp.status_code == 200
    assert "choices" in resp.json
"""

import json
import os
from dataclasses import dataclass, field
from typing import Any, Optional
from unittest.mock import patch, MagicMock


@dataclass
class ProxyResponse:
    """Structured response from a proxy test request."""

    status_code: int
    json: dict
    headers: dict = field(default_factory=dict)
    raw: Any = None

    @property
    def content(self) -> str:
        """Extract assistant message content from a chat completion response."""
        choices = self.json.get("choices", [])
        if choices:
            return choices[0].get("message", {}).get("content", "")
        return ""

    @property
    def error(self) -> Optional[dict]:
        """Return the error block if present, else None."""
        return self.json.get("error")

    @property
    def is_error(self) -> bool:
        return self.status_code >= 400 or "error" in self.json

    @property
    def is_stream(self) -> bool:
        ct = self.headers.get("Content-Type", "")
        return "text/event-stream" in ct

    @property
    def stream_chunks(self) -> list[str]:
        """Parse SSE body into individual data lines."""
        if not self.raw:
            return []
        body = self.raw.data if hasattr(self.raw, "data") else b""
        if isinstance(body, bytes):
            body = body.decode("utf-8", errors="replace")
        chunks = []
        for line in body.split("\n"):
            line = line.strip()
            if line.startswith("data: ") and line != "data: [DONE]":
                chunks.append(line[6:])
        return chunks

    @property
    def stream_json(self) -> list[dict]:
        """Parse SSE data lines into JSON objects."""
        result = []
        for chunk in self.stream_chunks:
            try:
                result.append(json.loads(chunk))
            except json.JSONDecodeError:
                pass
        return result


def create_test_proxy_app(config_overrides: Optional[dict] = None):
    """Create a Flask test app and client for the proxy.

    Returns (app, client) where client is a Flask test client.
    The app is configured for testing with TESTING=True.

    This imports the existing app singleton from proxy.app — there is no
    create_app() factory. Config overrides are applied to app.config.
    """
    from proxy.app import app as flask_app

    flask_app.config["TESTING"] = True
    if config_overrides:
        flask_app.config.update(config_overrides)

    client = flask_app.test_client()
    return flask_app, client


def make_chat_request(
    client,
    session_id: str,
    messages: list[dict],
    tools: Optional[list[dict]] = None,
    model: str = "test-model",
    stream: bool = False,
    extra_headers: Optional[dict] = None,
) -> ProxyResponse:
    """Send a chat completion request and return a structured response.

    Args:
        client: Flask test client.
        session_id: Session identifier (sent via X-Session-ID header).
        messages: List of message dicts (role, content).
        tools: Optional tool definitions.
        model: Model name.
        stream: Whether to request streaming SSE.
        extra_headers: Additional headers to include.

    Returns:
        ProxyResponse with parsed JSON, status code, and content helpers.
    """
    payload = {
        "model": model,
        "messages": messages,
        "stream": stream,
    }
    if tools:
        payload["tools"] = tools

    headers = {"X-Session-ID": session_id}
    if extra_headers:
        headers.update(extra_headers)

    resp = client.post(
        "/v1/chat/completions",
        json=payload,
        headers=headers,
        content_type="application/json",
    )

    resp_headers = dict(resp.headers)
    resp_json = resp.get_json(silent=True) or {}

    return ProxyResponse(
        status_code=resp.status_code,
        json=resp_json,
        headers=resp_headers,
        raw=resp,
    )


def make_tool_request(
    client,
    session_id: str,
    tool_name: str,
    arguments: dict,
    mode: Optional[str] = None,
    model: str = "test-model",
    extra_headers: Optional[dict] = None,
) -> ProxyResponse:
    """Send a tool call request through the chat completions endpoint.

    Constructs a messages list that simulates a tool call response cycle:
    - user message with the tool intent
    - assistant message with a tool_calls block
    - tool message with the result placeholder

    Args:
        client: Flask test client.
        session_id: Session identifier.
        tool_name: Name of the tool to invoke.
        arguments: Tool call arguments dict.
        mode: Optional mode hint (e.g., "ticket", "workspace").
        model: Model name.
        extra_headers: Additional headers.

    Returns:
        ProxyResponse.
    """
    messages = [
        {"role": "user", "content": f"Use tool {tool_name}"},
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": "call_test_001",
                    "type": "function",
                    "function": {
                        "name": tool_name,
                        "arguments": json.dumps(arguments),
                    },
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "call_test_001",
            "content": "",
        },
    ]

    if mode:
        messages[0]["content"] = f"^^{mode} {messages[0]['content']}"

    headers = {"X-Session-ID": session_id}
    if extra_headers:
        headers.update(extra_headers)

    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
    }

    resp = client.post(
        "/v1/chat/completions",
        json=payload,
        headers=headers,
        content_type="application/json",
    )

    resp_headers = dict(resp.headers)
    resp_json = resp.get_json(silent=True) or {}

    return ProxyResponse(
        status_code=resp.status_code,
        json=resp_json,
        headers=resp_headers,
        raw=resp,
    )
