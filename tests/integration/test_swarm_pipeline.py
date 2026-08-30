"""Phase 2 Agent 5 — full Flask pipeline tests for swarm canary injection.

End-to-end coverage for `docs/specs/custom-skill-research-swarms.md` §11 / §13
through the real Flask client → RequestPipeline → mock upstream loop.

Scenarios:
  1. Valid MAIN swarm parent: task emitted → SKILL canary → 200 OK.
  2. Over-budget response: HTTP 400 → swarm_max_calls_exceeded.
  3. Sub-agent nesting: HTTP 403 → swarm_subagent_cannot_spawn.
  4. Signer failure: HTTP 500 → swarm_*_failed with no prompt/secret leak.
  5. Child ingress round-trip: emit canary → second request → inherited skill.
"""
from __future__ import annotations

import json
import os
import re
from unittest.mock import patch

import pytest


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def disk_swarm_skill_dir(tmp_path, monkeypatch):
    """Create a swarm-capable custom JSON skill on disk and reset registry."""
    skills_dir = tmp_path / "skills"
    skills_dir.mkdir()
    (skills_dir / "research.json").write_text(json.dumps({
        "name": "research",
        "trigger": "^^research",
        "description": "End-to-end swarm proof skill.",
        "prompt": "RESEARCH_PROMPT_SENTINEL_AAA",
        "tool_whitelist": ["read", "grep", "task"],
        "subagents": {
            "tool_whitelist": ["read", "grep"],
            "max_task_calls_per_request": 4,
        },
    }))
    monkeypatch.setenv("PROXY_SKILLS_DIR", str(skills_dir))
    monkeypatch.setenv("PROXY_ENABLE_CUSTOM_SKILL_SUBAGENTS", "1")
    monkeypatch.setenv("PROXY_SESSION_SECRET", "x" * 32)
    from proxy.skills.skill_registry import reset_default_registry
    reset_default_registry()
    yield skills_dir
    reset_default_registry()


@pytest.fixture
def disk_swarm_skill_with_tiny_budget(tmp_path, monkeypatch):
    """Same as disk_swarm_skill_dir but max_task_calls_per_request=1 (forces over-budget)."""
    skills_dir = tmp_path / "skills"
    skills_dir.mkdir()
    (skills_dir / "research.json").write_text(json.dumps({
        "name": "research",
        "trigger": "^^research",
        "description": "End-to-end swarm proof skill (tiny budget).",
        "prompt": "RESEARCH_PROMPT_SENTINEL_AAA",
        "tool_whitelist": ["read", "grep", "task"],
        "subagents": {
            "tool_whitelist": ["read", "grep"],
            "max_task_calls_per_request": 1,
        },
    }))
    monkeypatch.setenv("PROXY_SKILLS_DIR", str(skills_dir))
    monkeypatch.setenv("PROXY_ENABLE_CUSTOM_SKILL_SUBAGENTS", "1")
    monkeypatch.setenv("PROXY_SESSION_SECRET", "x" * 32)
    from proxy.skills.skill_registry import reset_default_registry
    reset_default_registry()
    yield skills_dir
    reset_default_registry()


def _set_mock_response(handler_class, payload):
    """Configure the mock backend handler to return the given JSON payload."""
    handler_class._response_body = json.dumps(payload).encode()
    handler_class._status_code = 200


def _decode_response_data(response):
    """Decode JSON response body."""
    return json.loads(response.get_data(as_text=True))


def _register_real_agent_for_body(body, client_hint):
    """Pre-register a real Agent for the session derived from body."""
    from proxy.agent.agent import Agent
    from proxy.agent.agent_registry import registry as agent_registry
    from proxy.model_registry import resolve_session_id

    session_id = resolve_session_id(body, client_hint)
    agent_registry.register(Agent(session_id=session_id, agent_id=f"agent_{session_id}"))
    return session_id


def _emit_swarm_canary_for_test(skill_name: str, workspace: str = "default") -> str:
    """Build a swarm canary using the production builder, no signature substitution."""
    from proxy.canary import build_swarm_skill_canary
    state = {"workspace_name": workspace} if workspace != "default" else {}
    return build_swarm_skill_canary(skill_name, state, session_id="test-session")


# ===========================================================================
# 1. Valid MAIN swarm parent
# ===========================================================================


@pytest.mark.integration
@pytest.mark.real_agent_registry
def test_swarm_parent_emits_signed_canary_through_full_flask_pipeline(
    disk_swarm_skill_dir, proxy_with_mock
):
    """Upstream returns a task call → response carries a valid SKILL canary."""
    from proxy.canary import parse_canary

    client, handler_class, _port = proxy_with_mock
    # Mock upstream returns a single native `task` tool call.
    task_tc = {
        "id": "call_swarm_1",
        "type": "function",
        "function": {
            "name": "task",
            "arguments": json.dumps({
                "description": "investigate",
                "prompt": "Look at the auth bug.",
            }),
        },
    }
    _set_mock_response(handler_class, {
        "choices": [{
            "message": {"role": "assistant", "content": "", "tool_calls": [task_tc]},
            "finish_reason": "tool_calls",
        }],
        "model": "test/llm-v1",
    })

    body = {
        "model": "test-model",
        "messages": [{"role": "user", "content": "^^research investigate auth bug"}],
    }
    client_hint = "swarm-flask-ok"
    _register_real_agent_for_body(body, client_hint)

    response = client.post(
        "/v1/chat/completions", json=body,
        headers={"X-Session-Client": client_hint},
    )

    assert response.status_code == 200, (
        f"Expected 200, got {response.status_code}: "
        f"{response.get_data(as_text=True)[:200]!r}"
    )
    data = _decode_response_data(response)
    args = json.loads(
        data["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"]
    )
    prompt = args["prompt"]
    assert prompt.startswith("[[__SUBAGENT__:SKILL:research:"), (
        f"Expected SKILL canary for 'research', got: {prompt[:80]!r}"
    )
    # Round-trip parse to verify signature.
    parsed = parse_canary(prompt)
    assert parsed is not None
    assert parsed.kind == "skill"
    assert parsed.skill_name == "research"


# ===========================================================================
# 2. Over-budget response
# ===========================================================================


@pytest.mark.integration
@pytest.mark.real_agent_registry
def test_swarm_parent_over_budget_returns_400_no_tool_calls(
    disk_swarm_skill_with_tiny_budget, proxy_with_mock
):
    """max_task_calls_per_request=1 but upstream emits 2 → HTTP 400, no tool_calls."""
    client, handler_class, _port = proxy_with_mock
    # Two task calls (exceeds tiny budget of 1).
    task_calls = [
        {
            "id": f"call_over_{i}",
            "type": "function",
            "function": {
                "name": "task",
                "arguments": json.dumps({"description": "x", "prompt": f"p{i}"}),
            },
        }
        for i in range(2)
    ]
    _set_mock_response(handler_class, {
        "choices": [{
            "message": {"role": "assistant", "content": "", "tool_calls": task_calls},
            "finish_reason": "tool_calls",
        }],
        "model": "test/llm-v1",
    })

    body = {
        "model": "test-model",
        "messages": [{"role": "user", "content": "^^research over budget"}],
    }
    client_hint = "swarm-flask-overbudget"
    _register_real_agent_for_body(body, client_hint)

    response = client.post(
        "/v1/chat/completions", json=body,
        headers={"X-Session-Client": client_hint},
    )

    assert response.status_code == 400, (
        f"Expected HTTP 400, got {response.status_code}"
    )
    data = _decode_response_data(response)
    # Top-level envelope present.
    assert "error" in data, f"Expected top-level error envelope, got: {data!r}"
    err = data["error"]
    assert err["code"] == "swarm_max_calls_exceeded"
    # No choices/tool_calls leaked.
    assert "choices" not in data or not data.get("choices"), (
        f"Over-budget response must NOT include choices: {data!r}"
    )


# ===========================================================================
# 3. Sub-agent nesting
# ===========================================================================


@pytest.mark.integration
@pytest.mark.real_agent_registry
def test_swarm_subagent_nesting_returns_403(
    disk_swarm_skill_dir, proxy_with_mock
):
    """A sub-agent (parent's task prompt already has a canary) emits task → 403."""
    client, handler_class, _port = proxy_with_mock
    # The sub-agent's request body already contains a SKILL canary at position 0,
    # so the proxy treats it as a sub-agent. We then have it emit task.
    canary = _emit_swarm_canary_for_test("research")
    task_tc = {
        "id": "call_subagent_nest",
        "type": "function",
        "function": {
            "name": "task",
            "arguments": json.dumps({"description": "nested", "prompt": "x"}),
        },
    }
    _set_mock_response(handler_class, {
        "choices": [{
            "message": {"role": "assistant", "content": "", "tool_calls": [task_tc]},
            "finish_reason": "tool_calls",
        }],
        "model": "test/llm-v1",
    })

    body = {
        "model": "test-model",
        "messages": [{"role": "user", "content": f"{canary} investigate"}],
    }
    client_hint = "swarm-flask-nesting"
    _register_real_agent_for_body(body, client_hint)

    response = client.post(
        "/v1/chat/completions", json=body,
        headers={"X-Session-Client": client_hint},
    )

    assert response.status_code == 403, (
        f"Expected HTTP 403, got {response.status_code}"
    )
    data = _decode_response_data(response)
    assert data["error"]["code"] == "swarm_subagent_cannot_spawn"


# ===========================================================================
# 4. Signer / internal failure
# ===========================================================================


@pytest.mark.integration
@pytest.mark.real_agent_registry
def test_swarm_signer_failure_returns_500_no_prompt_leak(
    disk_swarm_skill_dir, proxy_with_mock, monkeypatch
):
    """Forced sign_canary_strict failure → 500 swarm_signing_failed, no prompt leak."""
    client, handler_class, _port = proxy_with_mock
    task_tc = {
        "id": "call_signer_fail",
        "type": "function",
        "function": {
            "name": "task",
            "arguments": json.dumps({
                "description": "investigate",
                "prompt": "SECRET-LEAK-MARKER-12345",
            }),
        },
    }
    _set_mock_response(handler_class, {
        "choices": [{
            "message": {"role": "assistant", "content": "", "tool_calls": [task_tc]},
            "finish_reason": "tool_calls",
        }],
        "model": "test/llm-v1",
    })

    # Patch build_swarm_skill_canary to raise SwarmCanaryError.
    from proxy.canary import SwarmCanaryError

    def _raise(*args, **kwargs):
        raise SwarmCanaryError("swarm_signing_failed", "forced for test")

    monkeypatch.setattr("proxy.canary.build_swarm_skill_canary", _raise)

    body = {
        "model": "test-model",
        "messages": [{"role": "user", "content": "^^research trigger"}],
    }
    client_hint = "swarm-flask-signerfail"
    _register_real_agent_for_body(body, client_hint)

    response = client.post(
        "/v1/chat/completions", json=body,
        headers={"X-Session-Client": client_hint},
    )

    assert response.status_code == 500, (
        f"Expected HTTP 500, got {response.status_code}"
    )
    body_text = response.get_data(as_text=True)
    data = _decode_response_data(response)
    assert data["error"]["code"] == "swarm_signing_failed"
    # No prompt leak.
    assert "SECRET-LEAK-MARKER-12345" not in body_text, (
        f"Prompt content leaked into error response: {body_text[:300]!r}"
    )


# ===========================================================================
# 5. Child ingress round-trip
# ===========================================================================


@pytest.mark.integration
@pytest.mark.real_agent_registry
def test_swarm_child_ingress_round_trip(
    disk_swarm_skill_dir, proxy_with_mock
):
    """Emit canary from upstream → second request with that canary → inherited skill."""
    from proxy.canary import build_swarm_skill_canary, parse_canary

    client, handler_class, _port = proxy_with_mock

    # Step 1: parent emits a task call → response carries a canary.
    task_tc = {
        "id": "call_parent_emit",
        "type": "function",
        "function": {
            "name": "task",
            "arguments": json.dumps({
                "description": "investigate",
                "prompt": "look at bug",
            }),
        },
    }
    _set_mock_response(handler_class, {
        "choices": [{
            "message": {"role": "assistant", "content": "", "tool_calls": [task_tc]},
            "finish_reason": "tool_calls",
        }],
        "model": "test/llm-v1",
    })

    body = {
        "model": "test-model",
        "messages": [{"role": "user", "content": "^^research start"}],
    }
    client_hint = "swarm-flask-roundtrip"
    _register_real_agent_for_body(body, client_hint)

    parent_response = client.post(
        "/v1/chat/completions", json=body,
        headers={"X-Session-Client": client_hint},
    )
    assert parent_response.status_code == 200
    parent_data = _decode_response_data(parent_response)
    parent_prompt = json.loads(
        parent_data["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"]
    )["prompt"]
    parsed = parse_canary(parent_prompt)
    assert parsed is not None and parsed.kind == "skill"
    assert parsed.skill_name == "research"

    # Step 2: feed the emitted canary to a second request (sub-agent ingress).
    canary = parent_prompt.split("]]")[0] + "]]"
    # Configure a non-task response for the child.
    _set_mock_response(handler_class, {
        "choices": [{
            "message": {"role": "assistant", "content": "OK done", "tool_calls": []},
            "finish_reason": "stop",
        }],
        "model": "test/llm-v1",
    })
    child_body = {
        "model": "test-model",
        "messages": [{"role": "user", "content": f"{canary} look at bug"}],
    }
    child_hint = "swarm-flask-roundtrip-child"
    _register_real_agent_for_body(child_body, child_hint)

    child_response = client.post(
        "/v1/chat/completions", json=child_body,
        headers={"X-Session-Client": child_hint},
    )
    # The child request should succeed (text-only response, no task calls).
    assert child_response.status_code == 200
    child_data = _decode_response_data(child_response)
    assert child_data["choices"][0]["message"]["content"] == "OK done"
    # Inherited skill identity is observable: the upstream body should now have
    # the research prompt installed.
    captured = handler_class._captured_request_body
    assert captured is not None
    if isinstance(captured, bytes):
        captured = json.loads(captured)
    # The system message should include the research custom prompt.
    system_msg = captured["messages"][0]
    assert system_msg["role"] == "system"
    assert "RESEARCH_PROMPT_SENTINEL_AAA" in system_msg["content"]
