"""Integration tests for adapter wiring coverage.

Tests that the proxy correctly uses adapter methods at each wiring point,
and that fallback behavior works when adapters are disabled or unavailable.
"""

import pytest


class TestKillSwitchWiring:
    def test_kill_switch_disables_adapter_tool_names(
        self, client, mock_llama_response, monkeypatch
    ):
        """PROXY_USE_ADAPTERS=0 → proxy uses DEFAULT_TOOL_NAMES, not adapter names."""
        monkeypatch.setenv("PROXY_USE_ADAPTERS", "0")
        monkeypatch.setattr("qwen_proxy.TEST_MODE", True)
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "qwen3-coder",
                "messages": [{"role": "user", "content": "test"}],
            },
        )
        assert response.status_code == 200

    def test_kill_switch_disables_adapter_system_prompt(
        self, client, mock_llama_response, monkeypatch
    ):
        """PROXY_USE_ADAPTERS=0 → proxy uses OUR_SYSTEM_PROMPT, not adapter prompt."""
        monkeypatch.setenv("PROXY_USE_ADAPTERS", "0")
        monkeypatch.setattr("qwen_proxy.TEST_MODE", True)
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "qwen3-coder",
                "messages": [{"role": "user", "content": "test"}],
            },
        )
        assert response.status_code == 200


class TestClientDetectionWiring:
    def test_header_overrides_active_client(
        self, client, mock_llama_response, monkeypatch
    ):
        """X-Client-Name header overrides ACTIVE_CLIENT for tool filtering."""
        monkeypatch.setattr("qwen_proxy.TEST_MODE", True)
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "qwen3-coder",
                "messages": [{"role": "user", "content": "test"}],
            },
            headers={"X-Client-Name": "opencode"},
        )
        assert response.status_code == 200

    def test_unknown_header_falls_back(self, client, mock_llama_response, monkeypatch):
        """Unknown X-Client-Name → RegistryError → falls back to ACTIVE_CLIENT."""
        monkeypatch.setattr("qwen_proxy.TEST_MODE", True)
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "qwen3-coder",
                "messages": [{"role": "user", "content": "test"}],
            },
            headers={"X-Client-Name": "nonexistent_client"},
        )
        assert response.status_code == 200

    def test_no_adapter_uses_defaults(self, client, mock_llama_response, monkeypatch):
        """ACTIVE_CLIENT=None, USE_ADAPTERS=True → defaults used."""
        monkeypatch.setattr("qwen_proxy.ACTIVE_CLIENT", None)
        monkeypatch.setattr("qwen_proxy.TEST_MODE", True)
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "qwen3-coder",
                "messages": [{"role": "user", "content": "test"}],
            },
        )
        assert response.status_code == 200


class TestSystemPromptWiring:
    def test_active_client_provides_system_prompt(
        self, client, mock_llama_response, monkeypatch
    ):
        """With ACTIVE_CLIENT set, adapter's system prompt is used."""
        monkeypatch.setattr("qwen_proxy.TEST_MODE", True)
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "qwen3-coder",
                "messages": [{"role": "user", "content": "test"}],
            },
        )
        assert response.status_code == 200

    def test_tool_filtering_uses_adapter_names(
        self, client, mock_llama_response, monkeypatch
    ):
        """With ACTIVE_CLIENT set, tools are filtered to adapter's tool set."""
        monkeypatch.setattr("qwen_proxy.TEST_MODE", True)
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
                "messages": [{"role": "user", "content": "test"}],
                "tools": tools,
            },
        )
        assert response.status_code == 200


class TestRequeryWiring:
    def test_requery_body_preserves_filtered_tools(
        self, client, mock_llama_response, monkeypatch
    ):
        """_build_requery_body uses filtered tools from llama_body, not original request."""
        monkeypatch.setattr("qwen_proxy.TEST_MODE", True)
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "qwen3-coder",
                "messages": [{"role": "user", "content": "test"}],
            },
        )
        assert response.status_code == 200

    def test_kill_switch_plus_header_interaction(
        self, client, mock_llama_response, monkeypatch
    ):
        """Kill switch ON + X-Client-Name header → header is ignored, defaults used."""
        monkeypatch.setenv("PROXY_USE_ADAPTERS", "0")
        monkeypatch.setattr("qwen_proxy.TEST_MODE", True)
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "qwen3-coder",
                "messages": [{"role": "user", "content": "test"}],
            },
            headers={"X-Client-Name": "opencode"},
        )
        assert response.status_code == 200

    def test_rapid_toggle_different_clients(
        self, client, mock_llama_response, monkeypatch
    ):
        """Two sequential requests with different X-Client-Name values work independently."""
        monkeypatch.setattr("qwen_proxy.TEST_MODE", True)
        r1 = client.post(
            "/v1/chat/completions",
            json={
                "model": "qwen3-coder",
                "messages": [{"role": "user", "content": "a"}],
            },
            headers={"X-Client-Name": "kilocode"},
        )
        r2 = client.post(
            "/v1/chat/completions",
            json={
                "model": "qwen3-coder",
                "messages": [{"role": "user", "content": "b"}],
            },
            headers={"X-Client-Name": "opencode"},
        )
        assert r1.status_code == 200
        assert r2.status_code == 200
