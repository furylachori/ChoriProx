"""Stage-boundary localization for custom JSON skills."""
from __future__ import annotations

import json
import time
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from proxy.agent.agent import Agent
from proxy.skills.skill_definition import ModeState
from proxy.skills.skill_registry import get_default_registry, reset_default_registry
from proxy.types import RequestContext


@pytest.fixture
def fresh_disk_skill(tmp_path, monkeypatch):
    skills_dir = tmp_path / "skills"
    skills_dir.mkdir()
    (skills_dir / "stagect.json").write_text(json.dumps({
        "name": "stagect",
        "trigger": "^^stagect",
        "description": "Stage contract skill.",
        "prompt": "STAGECT_PROMPT_SENTINEL",
        "tool_whitelist": ["read"],
    }))
    monkeypatch.setenv("PROXY_SKILLS_DIR", str(skills_dir))
    reset_default_registry()
    yield skills_dir
    reset_default_registry()


@pytest.mark.real_agent_registry
def test_dispatch_stage_synchronizes_ctx_messages_and_ctx_body(fresh_disk_skill):
    from proxy.pipeline.dispatch import DispatchResult
    from proxy.pipeline.stage_modules.dispatch_stage import DispatchStage

    reg = get_default_registry()
    skill = reg.get_by_trigger("^^stagect")
    assert skill is not None

    ctx = MagicMock(spec=RequestContext)
    ctx.messages = [{"role": "user", "content": "^^stagect original_text"}]
    ctx.body = {"messages": [{"role": "user", "content": "^^stagect original_text"}]}
    ctx.session_id = "sess_stagect"
    ctx.client_hint = "test"
    ctx._features = object()
    result = DispatchResult(
        rewritten_text="User request: original_text",
        handled=False,
        skill=skill,
    )

    from proxy.pipeline import dispatch as dispatch_mod
    with patch.object(dispatch_mod, "run_dispatch", return_value=result):
        DispatchStage().execute(ctx)

    assert ctx.messages[-1]["content"] == "User request: original_text"
    assert ctx.body["messages"][-1]["content"] == "User request: original_text"


@pytest.mark.real_agent_registry
def test_context_optimize_uses_active_disk_skill_for_prompt_and_tools(fresh_disk_skill):
    from proxy.agent.agent_registry import registry as agent_registry
    from proxy.pipeline.stage_modules.context_optimize import ContextOptimizeStage

    reg = get_default_registry()
    skill = reg.get_by_trigger("^^stagect")
    assert skill is not None

    agent = Agent(session_id="sess_stagecontract", agent_id="agent_stagecontract", is_subagent=False)
    mode_state = ModeState(name="stagect", activated_at=time.time(), ttl_seconds=300)
    agent.set_active_mode(mode_state)
    agent_registry.register(agent)

    body = {
        "messages": [{"role": "user", "content": "hello"}],
        "tools": [
            {"type": "function", "function": {"name": "read", "description": "Read", "parameters": {"type": "object"}}},
            {"type": "function", "function": {"name": "bash", "description": "Bash", "parameters": {"type": "object"}}},
        ],
    }
    ctx = MagicMock(spec=RequestContext)
    ctx.body = body
    ctx.messages = body["messages"]
    ctx.session_id = "sess_stagecontract"
    ctx.client_hint = "test"
    ctx.is_subagent = False
    ctx.inherited_skill = None
    ctx.subagent_ticket_mode = False
    ctx.model_name = "test-model"
    ctx.model_cfg = SimpleNamespace(context_limit=32000, raw_exchange_count=8, budget_ratio=None)

    with patch.object(agent_registry, "get_or_load", return_value=agent):
        ContextOptimizeStage().execute(ctx)

    system_msg = ctx.body["messages"][0]
    assert system_msg["role"] == "system"
    assert "STAGECT_PROMPT_SENTINEL" in system_msg["content"]
    tool_names = {t["function"]["name"] for t in ctx.body.get("tools", [])}
    assert "read" in tool_names
    assert "bash" not in tool_names
