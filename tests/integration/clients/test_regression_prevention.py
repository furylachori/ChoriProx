"""Regression prevention integration tests.

Verify that importing and using the clients layer does not break existing proxy behavior.
"""

import json
import pytest

from clients import (
    register_client,
    get_client,
    detect_client,
    active_client,
    list_clients,
)
from clients.kilocode import KiloCodeAdapter
from clients.opencode import OpenCodeAdapter
from clients.registry import ClientRegistry


class TestRegressionPrevention:
    @pytest.fixture(autouse=True)
    def reset_singleton(self):
        """Ensure no singleton leakage between tests."""
        yield
        ClientRegistry.reset()

    def test_proxy_chat_completions_still_works_after_clients_import(
        self, client, mock_llama_response, monkeypatch
    ):
        """Importing clients must not mutate proxy globals or break the endpoint."""
        monkeypatch.setattr("qwen_proxy.TEST_MODE", True)

        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "qwen3-coder",
                "messages": [{"role": "user", "content": "hello"}],
            },
        )
        assert response.status_code == 200
        data = response.get_json()
        assert "choices" in data
        assert data["choices"][0]["message"]["role"] == "assistant"

    def test_proxy_system_prompt_replacement_unchanged(
        self, client, mock_llama_response, monkeypatch
    ):
        """The proxy still replaces system prompt even after clients layer is imported."""
        monkeypatch.setattr("qwen_proxy.TEST_MODE", True)

        from qwen_proxy import _HTTP_SESSION
        _HTTP_SESSION.request.reset_mock()

        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "qwen3-coder",
                "messages": [
                    {"role": "system", "content": "Old custom prompt"},
                    {"role": "user", "content": "hi"},
                ],
            },
        )
        assert response.status_code == 200

        sent_body = _HTTP_SESSION.request.call_args.kwargs.get("json") or {}
        sent_messages = sent_body.get("messages", [])
        assert sent_messages, (
            "expected upstream body to contain messages"
        )
        system_msgs = [m for m in sent_messages if m.get("role") == "system"]
        assert system_msgs, (
            "expected the upstream body to contain at least one system message "
            "(the proxy's own orchestration system prompt) — the user's supplied "
            "system message must have been replaced, not dropped"
        )
        first_system_content = str(system_msgs[0].get("content", ""))
        assert first_system_content.startswith("You are operating"), (
            "upstream system message must start with the proxy's own system prompt, "
            "not the user's 'Old custom prompt'. The user's content may legitimately "
            "appear later as <TASK_PAYLOAD> DATA, but the system instructions must "
            f"be the proxy's. Got first 200 chars: {first_system_content[:200]!r}"
        )
        assert not first_system_content.startswith("Old custom prompt"), (
            f"user-supplied system message must not be used verbatim as upstream "
            f"system prompt; got {first_system_content[:200]!r}"
        )

    def test_proxy_tool_filtering_unchanged(
        self, client, mock_llama_response, monkeypatch
    ):
        """The proxy still filters tools to the hardcoded set after clients import."""
        monkeypatch.setattr("qwen_proxy.TEST_MODE", True)

        from qwen_proxy import _HTTP_SESSION
        _HTTP_SESSION.request.reset_mock()

        tools = [
            {"type": "function", "function": {"name": "bash", "parameters": {}}},
            {
                "type": "function",
                "function": {"name": "unknown_tool", "parameters": {}},
            },
        ]
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "qwen3-coder",
                "messages": [{"role": "user", "content": "run ls"}],
                "tools": tools,
            },
        )
        assert response.status_code == 200

        sent_body = _HTTP_SESSION.request.call_args.kwargs.get("json") or {}
        sent_tools = sent_body.get("tools", [])
        sent_tool_names = [
            (t.get("function") or {}).get("name") for t in sent_tools
        ]
        assert "unknown_tool" not in sent_tool_names, (
            f"unknown_tool must be filtered out before upstream dispatch; "
            f"got tool names {sent_tool_names}"
        )
        assert "bash" in sent_tool_names, (
            f"bash must be preserved in upstream tool list; "
            f"got tool names {sent_tool_names}"
        )

    def test_proxy_models_endpoint_unchanged(self, client):
        """The /v1/models endpoint must still respond (mocked in conftest)."""
        response = client.get("/v1/models")
        # conftest.block_all_http mocks requests.get to return empty JSON
        assert response.status_code == 200

    def test_proxy_sse_streaming_unchanged(
        self, client, mock_llama_response, monkeypatch
    ):
        monkeypatch.setattr("qwen_proxy.TEST_MODE", True)

        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "qwen3-coder",
                "messages": [{"role": "user", "content": "hello"}],
                "stream": True,
            },
        )
        assert response.status_code == 200
        assert response.content_type == "text/event-stream"

    def test_clients_public_api_does_not_mutate_proxy(self):
        """Registering clients via public API must not touch proxy globals."""
        import qwen_proxy

        original_tool_names = (
            set(qwen_proxy.OPENCODE_TOOL_NAMES)
            if hasattr(qwen_proxy, "OPENCODE_TOOL_NAMES")
            else None
        )
        original_llama_url = (
            qwen_proxy.LLAMA_URL if hasattr(qwen_proxy, "LLAMA_URL") else None
        )

        register_client(KiloCodeAdapter())
        register_client(OpenCodeAdapter())

        if original_tool_names is not None:
            assert qwen_proxy.OPENCODE_TOOL_NAMES == original_tool_names
        if original_llama_url is not None:
            assert qwen_proxy.LLAMA_URL == original_llama_url

    def test_client_registry_reset_leaves_proxy_untouched(self):
        """Resetting the client singleton must not affect proxy state."""
        import qwen_proxy

        request_counter_before = qwen_proxy.request_counter
        ClientRegistry.reset()
        assert qwen_proxy.request_counter == request_counter_before

    def test_tool_loop_mock_response_still_works(
        self, client, mock_llama_response, monkeypatch
    ):
        """TEST_MODE mock response path in proxy must remain intact."""
        monkeypatch.setattr("qwen_proxy.TEST_MODE", True)

        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "qwen3-coder",
                "messages": [{"role": "user", "content": "test"}],
            },
        )
        data = response.get_json()
        assert data["model"] == "qwen3-coder"
        assert data["choices"][0]["finish_reason"] == "stop"

    def test_memory_tool_names_unchanged_after_clients_import(self):
        """The proxy's LLM-callable memory tools must remain in the adapter tool set.

        Post-Phase 3: MEMORY_TOOL_NAMES literal was deleted. All consumers
        go through ToolRegistry.get_memory_tool_names(). This test pins
        that the LLM-callable subset (recall, search) is still in the
        adapter's LLM-facing tool set after the clients layer is imported.

        Post-Slice 8b: the monolithic ``proxy_ticket`` was decomposed into
        10 ``proxy_ticket_*`` tools. The decomposed tools are registered
        as MEMORY but they are also PROXY_ADMIN_TOOLS — they require skill
        context to be safe and bare sub-agents must not see them. Only
        ``recall`` and ``search`` remain in the LLM-callable subset
        (visible in default mode without a skill context).

        The runtime tool guard would block admin tools at execution time
        if a bare sub-agent saw them in its tool list."""
        from proxy.tool_pipeline.tool_registry import registry, _PROXY_ADMIN_TOOLS

        all_memory = registry.get_memory_tool_names()
        llm_callable_memory = all_memory - _PROXY_ADMIN_TOOLS
        adapter = KiloCodeAdapter()
        client_names = adapter.get_tool_names()

        # LLM-callable memory tools must be present in the adapter tool set
        assert llm_callable_memory <= client_names, (
            f"LLM-callable memory tools {llm_callable_memory} not subset of adapter tools {client_names}"
        )
        # And ALL proxy admin tools (monolithic + decomposed) must NOT leak
        # into the LLM-facing set
        assert _PROXY_ADMIN_TOOLS.isdisjoint(client_names), (
            f"Proxy admin tools should not be in LLM-facing adapter tools {client_names}"
        )

    def test_no_import_side_effects_on_sys_modules(self):
        """Re-importing clients must not reload or alter proxy module state."""
        import sys
        import clients

        mod_id_before = id(sys.modules.get("clients"))
        import clients as clients2

        assert id(clients2) == mod_id_before
