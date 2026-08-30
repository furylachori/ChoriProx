"""Full Flask POST pipeline proof for a JSON-defined custom skill."""
from __future__ import annotations

import json

import pytest


@pytest.fixture
def disk_skill_dir(tmp_path, monkeypatch):
    skills_dir = tmp_path / "skills"
    skills_dir.mkdir()
    (skills_dir / "diskproof.json").write_text(json.dumps({
        "name": "diskproof",
        "trigger": "^^diskproof",
        "description": "End-to-end pipeline proof skill.",
        "prompt": "CUSTOM_SKILL_PROMPT_SENTINEL_7F3A",
        "tool_whitelist": ["read", "grep"],
        "aliases": ["^^dp"],
        "ttl_seconds": 300,
    }))
    monkeypatch.setenv("PROXY_SKILLS_DIR", str(skills_dir))
    from proxy.skills.skill_registry import reset_default_registry
    reset_default_registry()
    yield skills_dir
    reset_default_registry()


def _decode_captured(handler_class):
    captured = handler_class._captured_request_body
    assert captured is not None, "upstream was not called"
    if isinstance(captured, bytes):
        return json.loads(captured)
    return captured


def _register_real_agent_for_body(body, client_hint):
    from proxy.agent.agent import Agent
    from proxy.agent.agent_registry import registry as agent_registry
    from proxy.model_registry import resolve_session_id

    session_id = resolve_session_id(body, client_hint)
    agent_registry.register(Agent(session_id=session_id, agent_id=f"agent_{session_id}"))
    return session_id


@pytest.mark.integration
@pytest.mark.real_agent_registry
def test_disk_skill_flask_post_reaches_upstream_with_rewrite_prompt_and_filtered_tools(
    disk_skill_dir, proxy_with_mock, monkeypatch
):
    client, handler_class, _port = proxy_with_mock
    body = {
        "model": "test-model",
        "messages": [{"role": "user", "content": "^^diskproof USER_ARGUMENT_SENTINEL_91C2"}],
        "tools": [
            {"type": "function", "function": {"name": "read", "description": "Read", "parameters": {"type": "object", "properties": {"path": {"type": "string"}}}}},
            {"type": "function", "function": {"name": "grep", "description": "Grep", "parameters": {"type": "object", "properties": {"pattern": {"type": "string"}}}}},
            {"type": "function", "function": {"name": "bash", "description": "Bash", "parameters": {"type": "object", "properties": {"cmd": {"type": "string"}}}}},
        ],
    }
    client_hint = "custom-skill-pipeline"
    _register_real_agent_for_body(body, client_hint)

    response = client.post("/v1/chat/completions", json=body, headers={"X-Session-Client": client_hint})

    assert response.status_code == 200
    captured = _decode_captured(handler_class)
    last_msg = captured["messages"][-1]
    assert "^^diskproof" not in last_msg["content"]
    assert "User request: USER_ARGUMENT_SENTINEL_91C2" in last_msg["content"]
    system_msg = captured["messages"][0]
    assert system_msg["role"] == "system"
    assert "CUSTOM_SKILL_PROMPT_SENTINEL_7F3A" in system_msg["content"]
    upstream_tool_names = {t["function"]["name"] for t in captured.get("tools", [])}
    assert "read" in upstream_tool_names
    assert "grep" in upstream_tool_names
    assert "bash" not in upstream_tool_names


@pytest.mark.integration
@pytest.mark.real_agent_registry
def test_disk_skill_alias_takes_same_pipeline_and_strips_alias(disk_skill_dir, proxy_with_mock):
    client, handler_class, _port = proxy_with_mock
    body = {
        "model": "test-model",
        "messages": [{"role": "user", "content": "^^dp ALIAS_ARGUMENT_SENTINEL"}],
    }
    client_hint = "custom-skill-alias-pipeline"
    _register_real_agent_for_body(body, client_hint)

    response = client.post("/v1/chat/completions", json=body, headers={"X-Session-Client": client_hint})

    assert response.status_code == 200
    captured = _decode_captured(handler_class)
    last_msg = captured["messages"][-1]
    assert "^^dp" not in last_msg["content"]
    assert "^^diskproof" not in last_msg["content"]
    assert "ALIAS_ARGUMENT_SENTINEL" in last_msg["content"]
