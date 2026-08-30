"""Tests for sub-agent skill inheritance (Commit 1).

Verifies that sub-agents INHERIT skills via _inherit_skill (not _activate_skill),
that the canary carries task_hash end-to-end, and that on_activate is NOT called
for sub-agents.

Bug: ^^work <hash> main agent dispatches correctly, but spawned sub-agents
failed to inherit the work skill. They ran with a default prompt, no task_hash
context, and produced garbage output.
"""
import json
import logging
import os
import time
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from proxy.agent.agent import _inherit_skill
from proxy.canary import detect_subagent_ingress, sign_canary
from proxy.skills.skill_registry import (
    SkillContext,
    SkillRegistry,
    _skill_registry_new__,
    reset_default_registry,
)


@pytest.fixture(autouse=True)
def _dev_env(monkeypatch):
    """Ensure dev mode for canary signing/verification (no secret needed)."""
    monkeypatch.setenv("PROXY_ENV", "dev")
    monkeypatch.delenv("PROXY_SESSION_SECRET", raising=False)
    yield


@pytest.fixture(autouse=True)
def _fresh_registry():
    """Each test gets a fresh registry populated with builtin skills."""
    reset_default_registry()
    yield
    reset_default_registry()


def _get_work_skill_definition():
    """Return the real work SkillDefinition from the registry (with trust token)."""
    reg = _skill_registry_new__(SkillRegistry)
    reg.populate_builtins()
    skill = reg.get("work")
    assert skill is not None, "work skill not registered"
    return skill


# ---------------------------------------------------------------------------
# 1. _inherit_skill installs correct ModeState
# ---------------------------------------------------------------------------


def test_inherit_skill_installs_correct_mode_state():
    """_inherit_skill installs ModeState with name, skill_state, and filters."""
    skill = _get_work_skill_definition()

    installed = []
    agent = SimpleNamespace(
        set_active_mode=lambda ms: installed.append(ms) or True,
    )
    skill_ctx = SkillContext(
        session_id="test_session",
        agent_id="test_agent",
        messages=[],
        skill_state={"arg": "abc123"},
        is_subagent=True,
    )

    _inherit_skill(agent, skill, skill_ctx)

    assert len(installed) == 1, "set_active_mode must be called exactly once"
    ms = installed[0]
    assert ms.name == "work"
    assert ms.skill_state["arg"] == "abc123"
    assert ms.allowed_tool_names == skill.sub_agent_tool_filter
    assert ms.sub_agent_tool_filter == skill.sub_agent_tool_filter


# ---------------------------------------------------------------------------
# 2. on_activate NOT called
# ---------------------------------------------------------------------------


def test_inherit_skill_does_not_call_on_activate():
    """_inherit_skill must NOT call skill.on_activate (top-level only hook)."""
    on_activate_calls = []

    class SpySkill:
        name = "work"
        ttl_seconds = 300
        sub_agent_tool_filter = frozenset({"read", "write", "edit", "bash"})

        def on_activate(self, ctx, mode_state=None):
            on_activate_calls.append((ctx, mode_state))

    skill = SpySkill()
    agent = SimpleNamespace(set_active_mode=lambda ms: True)
    skill_ctx = SkillContext(
        session_id="test",
        agent_id="a",
        messages=[],
        skill_state={"arg": "abc"},
        is_subagent=True,
    )

    _inherit_skill(agent, skill, skill_ctx)

    assert len(on_activate_calls) == 0, (
        "on_activate must not be called by _inherit_skill"
    )


def test_inherit_skill_does_not_set_ctx_skill_activated():
    """_inherit_skill must NOT set ctx._skill_activated (DispatchStage overwrites it)."""
    skill = _get_work_skill_definition()
    agent = SimpleNamespace(set_active_mode=lambda ms: True)
    ctx = SimpleNamespace(_skill_activated=False)
    skill_ctx = SkillContext(
        session_id="test",
        agent_id="a",
        messages=[],
        skill_state={"arg": "abc"},
        is_subagent=True,
    )

    _inherit_skill(agent, skill, skill_ctx, ctx)

    assert ctx._skill_activated is False, (
        "_inherit_skill must not set ctx._skill_activated"
    )


# ---------------------------------------------------------------------------
# 3. detect_subagent_ingress extracts task_hash from SKILL canary (branch 1)
# ---------------------------------------------------------------------------


def test_detect_subagent_ingress_extracts_task_hash_skill_branch():
    """Branch 1: direct SKILL canary with JSON state → task_hash extracted."""
    sig = sign_canary("work")
    state = json.dumps({"task_hash": "abc123", "workspace_name": "ws"})
    canary = f"[[__SUBAGENT__:SKILL:work:{sig}|{state}]]"

    body = {"messages": [{"role": "user", "content": canary + " do the work"}]}

    result = detect_subagent_ingress(body)

    assert result.is_subagent is True
    assert result.inherited_skill == "work"
    assert result.task_hash == "abc123"
    assert result.workspace_name == "ws"


def test_detect_subagent_ingress_task_hash_none_without_state():
    """SKILL canary without JSON state → task_hash is None."""
    sig = sign_canary("work")
    canary = f"[[__SUBAGENT__:SKILL:work:{sig}]]"

    body = {"messages": [{"role": "user", "content": canary + " do the work"}]}

    result = detect_subagent_ingress(body)

    assert result.is_subagent is True
    assert result.inherited_skill == "work"
    assert result.task_hash is None


# ---------------------------------------------------------------------------
# 4. Bare [[__SUBAGENT__]] + SKILL canary → task_hash extracted (branch 2)
# ---------------------------------------------------------------------------


def test_detect_subagent_ingress_bare_then_skill_extracts_task_hash():
    """Branch 2: bare [[__SUBAGENT__]] followed by SKILL canary."""
    sig = sign_canary("work")
    state = json.dumps({"task_hash": "def456"})
    skill_canary = f"[[__SUBAGENT__:SKILL:work:{sig}|{state}]]"
    content = f"[[__SUBAGENT__]]\n{skill_canary} do the work"

    body = {"messages": [{"role": "user", "content": content}]}
    result = detect_subagent_ingress(body)

    assert result.is_subagent is True
    assert result.inherited_skill == "work"
    assert result.task_hash == "def456"


# ---------------------------------------------------------------------------
# 5. Empty task_hash → warning logged
# ---------------------------------------------------------------------------


def test_resolve_mode_warns_when_task_hash_empty(caplog):
    """resolve_mode logs a warning when inherited_task_hash is None/empty."""
    from proxy.skill_resolver import resolve_mode

    reg = _skill_registry_new__(SkillRegistry)
    reg.populate_builtins()

    # SimpleNamespace (NOT MagicMock) so `or ""` guard works correctly.
    ctx = SimpleNamespace(
        is_subagent=True,
        inherited_skill="work",
        inherited_task_hash=None,
        body={},
        session_id="s1",
        messages=[],
    )
    agent = SimpleNamespace(
        snapshot_mode_and_ticket_ts=lambda: None,
        snapshot_mode=lambda: None,
        clear_active_mode=lambda expected_mode=None: None,
        set_active_mode=lambda ms: True,
        agent_id="a1",
    )

    with caplog.at_level(logging.WARNING, logger="proxy.skill_resolver"):
        resolve_mode(ctx, agent, reg)

    assert any("no task_hash" in r.getMessage() for r in caplog.records), (
        "expected warning about missing task_hash"
    )


# ---------------------------------------------------------------------------
# 6. End-to-end: detect_subagent_ingress → resolve_mode → ModeState installed
# ---------------------------------------------------------------------------


def test_end_to_end_canary_to_mode_state():
    """Full flow: canary with task_hash → detect → resolve_mode → ModeState.

    This is the user-reported bug scenario: a sub-agent spawned with a work
    canary must inherit the work skill with the task_hash preserved in
    skill_state["arg"].
    """
    from proxy.mode import Mode
    from proxy.skill_resolver import resolve_mode

    reg = _skill_registry_new__(SkillRegistry)
    reg.populate_builtins()

    # Build a work canary carrying task_hash
    sig = sign_canary("work")
    state = json.dumps({"task_hash": "abc123"})
    canary = f"[[__SUBAGENT__:SKILL:work:{sig}|{state}]]"
    body = {"messages": [{"role": "user", "content": canary + " do work"}]}

    # Step 1: detect_subagent_ingress extracts task_hash
    ingress = detect_subagent_ingress(body)
    assert ingress.task_hash == "abc123"
    assert ingress.inherited_skill == "work"

    # Step 2: resolve_mode inherits the skill
    installed = []
    ctx = SimpleNamespace(
        is_subagent=True,
        inherited_skill=ingress.inherited_skill,
        inherited_task_hash=ingress.task_hash,
        body={},
        session_id="s1",
        messages=[],
    )
    agent = SimpleNamespace(
        snapshot_mode_and_ticket_ts=lambda: None,
        snapshot_mode=lambda: None,
        clear_active_mode=lambda expected_mode=None: None,
        set_active_mode=lambda ms: installed.append(ms) or True,
        agent_id="a1",
    )

    mode = resolve_mode(ctx, agent, reg)

    assert mode == str(Mode.SKILL_SUBAGENT.value)
    assert len(installed) == 1, "set_active_mode must be called once"
    ms = installed[0]
    assert ms.name == "work"
    assert ms.skill_state["arg"] == "abc123", (
        "task_hash must be translated to skill_state['arg']"
    )
    assert ms.allowed_tool_names == ms.sub_agent_tool_filter
    assert "task" not in ms.sub_agent_tool_filter, (
        "sub-agent filter must exclude 'task' (no sub-sub-agents)"
    )


# ---------------------------------------------------------------------------
# 7. resolve_mode with real work skill: ModeState filter correctness
# ---------------------------------------------------------------------------


def test_resolve_mode_inherits_work_skill_with_correct_filter():
    """resolve_mode for work sub-agent installs ModeState with correct filter.

    Verifies the ModeState's allowed_tool_names matches the WorkSkill's
    sub_agent_tool_filter (which excludes 'task').
    """
    from proxy.skill_resolver import resolve_mode

    skill_def = _get_work_skill_definition()
    expected_filter = skill_def.sub_agent_tool_filter

    reg = _skill_registry_new__(SkillRegistry)
    reg.populate_builtins()

    installed = []
    ctx = SimpleNamespace(
        is_subagent=True,
        inherited_skill="work",
        inherited_task_hash="abc123",
        body={},
        session_id="s1",
        messages=[],
    )
    agent = SimpleNamespace(
        snapshot_mode_and_ticket_ts=lambda: None,
        snapshot_mode=lambda: None,
        clear_active_mode=lambda expected_mode=None: None,
        set_active_mode=lambda ms: installed.append(ms) or True,
        agent_id="a1",
    )

    resolve_mode(ctx, agent, reg)

    assert len(installed) == 1
    ms = installed[0]
    assert ms.allowed_tool_names == expected_filter
    assert ms.sub_agent_tool_filter == expected_filter
    assert "task" not in expected_filter
