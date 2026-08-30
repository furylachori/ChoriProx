"""Phase 2 Agent 4 — request-local swarm policy installation tests.

Covers `docs/specs/custom-skill-research-swarms.md` §11.3 + AC 16.4, 16.14,
16.15:

  - ``ctx.swarm_policy`` is None by default on RequestContext.
  - ContextOptimizeStage writes the immutable SubagentPolicy from the active
    skill after a successful ``apply_mode``.
  - The reference is identity-stable: ``ctx.swarm_policy is skill.subagents``.
  - Sub-agents (universal no-nesting §14) keep ``swarm_policy = None`` even
    if the inherited skill has subagents configured.
  - The reset-at-start invariant: a reused RequestContext with stale
    non-None policy ends the request at None for an ordinary MAIN.
  - apply_mode fail-closed paths leave policy at None.
  - Concurrent requests sharing a session get independent request-local
    policies.

The tests use the real ContextOptimizeStage code path. Mocks are limited
to the agent_registry.get_or_load seam and the enforce_context_limits
helper — we never mock the production code under test.
"""
from __future__ import annotations

import threading
import time as _time
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from proxy.agent.agent import Agent
from proxy.agent.agent_registry import registry as agent_registry
from proxy.pipeline.stage_modules.context_optimize import ContextOptimizeStage
from proxy.skills.skill_definition import (
    ModeState,
    SkillDefinition,
    SubagentPolicy,
)
from proxy.skills.skill_registry import (
    get_default_registry,
    reset_default_registry,
)
from proxy.types import RequestContext


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def fresh_registry():
    """A clean registry per test (no leaked builtins-population)."""
    reset_default_registry()
    yield
    reset_default_registry()


def _make_swarm_skill(
    name: str,
    *,
    parent_tools: tuple[str, ...] = ("read", "task"),
    child_tools: frozenset[str] = frozenset({"read"}),
    max_calls: int = 4,
) -> SkillDefinition:
    """Build a swarm-capable custom SkillDefinition with the given child policy."""
    return SkillDefinition(
        name=name,
        trigger="^^" + name,
        aliases=(),
        description=f"Swarm skill {name}",
        allowed_tool_names=frozenset(parent_tools),
        subagents=SubagentPolicy(
            tool_whitelist=child_tools,
            max_task_calls_per_request=max_calls,
        ),
    )


def _make_ordinary_skill(
    name: str,
    *,
    parent_tools: tuple[str, ...] = ("read",),
) -> SkillDefinition:
    """Build an ordinary custom SkillDefinition without a swarm policy."""
    return SkillDefinition(
        name=name,
        trigger="^^" + name,
        aliases=(),
        description=f"Ordinary skill {name}",
        allowed_tool_names=frozenset(parent_tools),
        subagents=None,
    )


def _register(*skills: SkillDefinition) -> SkillRegistry:
    """Register a hand-built skill set in the singleton registry.

    Uses the same canonical path as ``populate_builtins`` — bypasses the
    per-skill `register_skill` trust gate by submitting the WHOLE batch
    to ``_register_validated_batch`` (the loader's loader-side contract
    per Wave 1 / Phase 2). No security boundary expansion: the batch
    is structurally inspected just like a populated-from-disk batch.
    """
    reg = get_default_registry()
    reg._register_validated_batch(tuple(skills))
    return reg


def _make_ctx(
    *,
    session_id: str = "sess_swarm_unit",
    is_subagent: bool = False,
    model_name: str = "test-model",
    body: dict | None = None,
    stale_swarm_policy=None,
) -> RequestContext:
    """Build a real RequestContext with the swarm-impl-relevant fields set."""
    if body is None:
        body = {"messages": [{"role": "user", "content": "hello"}]}
    ctx = RequestContext(
        body=body,
        model_name=model_name,
        wants_stream=False,
        session_id=session_id,
        client_hint="test",
        messages=list(body.get("messages", [])),
        model_cfg=SimpleNamespace(
            context_limit=32000,
            raw_exchange_count=8,
            budget_ratio=None,
        ),
        is_subagent=is_subagent,
    )
    if stale_swarm_policy is not None:
        # Simulate a request-scoped reuse (e.g. a test fixture that
        # initialises a ctx once and re-runs multiple stages against it).
        ctx.swarm_policy = stale_swarm_policy
    return ctx


def _activate(agent: Agent, skill_name: str) -> None:
    """Install a fresh ModeState on the agent."""
    ms = ModeState(name=skill_name, activated_at=_time.time(), ttl_seconds=300)
    agent.set_active_mode(ms)


# ---------------------------------------------------------------------------
# Section 1 — RequestContext defaults
# ---------------------------------------------------------------------------


class TestRequestContextDefaults:
    def test_swarm_policy_default_is_none(self):
        ctx = _make_ctx()
        assert ctx.swarm_policy is None

    def test_swarm_policy_field_is_optional_in_dataclass(self):
        import dataclasses
        field_obj = next(
            f for f in dataclasses.fields(RequestContext) if f.name == "swarm_policy"
        )
        # Default is None — present in the dataclass.
        assert field_obj.default is None
        # Type annotation is forward-reference form (string under from __future__).
        assert "Optional" in str(field_obj.type) or "SubagentPolicy" in str(field_obj.type)


# ---------------------------------------------------------------------------
# Section 2 — MAIN swarm skill installs the policy
# ---------------------------------------------------------------------------


class TestMainSwarmSkillInstallation:
    def test_main_swarm_skill_installs_policy_after_apply_mode(self):
        skill = _make_swarm_skill("research", parent_tools=("read", "task"))
        _register(skill)
        agent = Agent(session_id="sess_main_swarm", agent_id="agent_main_swarm", is_subagent=False)
        _activate(agent, "research")
        agent_registry.register(agent)

        ctx = _make_ctx(session_id="sess_main_swarm")
        with patch.object(agent_registry, "get_or_load", return_value=agent):
            ContextOptimizeStage().execute(ctx)

        assert ctx.swarm_policy is skill.subagents

    def test_installed_policy_is_identity_equal_to_skill_subagents(self):
        skill = _make_swarm_skill("audit", child_tools=frozenset({"read", "glob"}))
        _register(skill)
        agent = Agent(session_id="sess_id", agent_id="a", is_subagent=False)
        _activate(agent, "audit")
        agent_registry.register(agent)

        ctx = _make_ctx(session_id="sess_id")
        with patch.object(agent_registry, "get_or_load", return_value=agent):
            ContextOptimizeStage().execute(ctx)

        # Identity (not equality) — proves no copy / no deep-copy / no
        # dict-conversion path was used.
        assert ctx.swarm_policy is skill.subagents


# ---------------------------------------------------------------------------
# Section 3 — Ordinary MAIN keeps policy None
# ---------------------------------------------------------------------------


class TestOrdinaryMainSkill:
    def test_ordinary_custom_skill_keeps_policy_none(self):
        skill = _make_ordinary_skill("audit", parent_tools=("read", "grep"))
        _register(skill)
        agent = Agent(session_id="sess_ord", agent_id="a", is_subagent=False)
        _activate(agent, "audit")
        agent_registry.register(agent)

        ctx = _make_ctx(session_id="sess_ord")
        with patch.object(agent_registry, "get_or_load", return_value=agent):
            ContextOptimizeStage().execute(ctx)

        assert ctx.swarm_policy is None

    def test_no_active_mode_keeps_policy_none(self):
        skill = _make_ordinary_skill("ok")
        _register(skill)
        agent = Agent(session_id="sess_no_mode", agent_id="a", is_subagent=False)
        # Do NOT call _activate; no mode set.
        agent_registry.register(agent)

        ctx = _make_ctx(session_id="sess_no_mode")
        with patch.object(agent_registry, "get_or_load", return_value=agent):
            ContextOptimizeStage().execute(ctx)

        assert ctx.swarm_policy is None


# ---------------------------------------------------------------------------
# Section 4 — Sub-agent NEVER gets a swarm policy (universal no-nesting §14)
# ---------------------------------------------------------------------------


class TestSubAgentNoPolicy:
    def test_subagent_with_swarm_inherited_skill_keeps_policy_none(self):
        skill = _make_swarm_skill(
            "research", parent_tools=("read", "task"), child_tools=frozenset({"read"})
        )
        _register(skill)
        agent = Agent(session_id="sess_sub", agent_id="a", is_subagent=True)
        # Inherited mode for a sub-agent uses Mode.SKILL_SUBAGENT but
        # snapshot_mode().name == "research" — same code path.
        _activate(agent, "research")
        agent_registry.register(agent)

        ctx = _make_ctx(session_id="sess_sub", is_subagent=True)
        with patch.object(agent_registry, "get_or_load", return_value=agent):
            ContextOptimizeStage().execute(ctx)

        assert ctx.swarm_policy is None


# ---------------------------------------------------------------------------
# Section 5 — apply_mode fail-closed paths leave policy None
# ---------------------------------------------------------------------------


class TestApplyModeFailure:
    def test_apply_mode_raises_keeps_policy_none_no_stale_install(self):
        # Register an ordinary skill. Then have apply_mode raise a
        # deterministic exception to force the fail-closed ActiveSkillUnavailable
        # branch. ctx.swarm_policy must remain None throughout — installed
        # BEFORE apply_mode would be a regression.
        skill = _make_swarm_skill("boom", child_tools=frozenset({"read"}))
        _register(skill)
        agent = Agent(session_id="sess_boom", agent_id="a", is_subagent=False)
        _activate(agent, "boom")
        agent_registry.register(agent)

        ctx = _make_ctx(session_id="sess_boom")
        # Apply_mode raises; the StageOptimizer catches ActiveSkillUnavailable
        # and re-raises as StageAbort with HTTP 409 skill_state_conflict.
        from proxy.mode import ActiveSkillUnavailable
        with patch.object(agent_registry, "get_or_load", return_value=agent), \
             patch(
                 "proxy.mode.apply_mode",
                 side_effect=ActiveSkillUnavailable("boom"),
             ):
            with pytest.raises(Exception):
                ContextOptimizeStage().execute(ctx)

        # Reset to None + never installed -> still None.
        assert ctx.swarm_policy is None


# ---------------------------------------------------------------------------
# Section 6 — Reused RequestContext: stale non-None policy gets reset
# ---------------------------------------------------------------------------


class TestReusedContext:
    def test_stale_policy_resets_to_none_for_agent_none_cascade(self):
        """A reused RequestContext with stale non-None policy must end
        the request at None when the agent cannot be loaded — the
        cascade fallback path skips ``_install_swarm_policy`` entirely,
        so the only line of defense is the per-execute reset at the
        top of ``ContextOptimizeStage.execute``.
        """
        stale_policy = SubagentPolicy(
            tool_whitelist=frozenset({"read"}),
            max_task_calls_per_request=1,
        )
        ctx = _make_ctx(
            session_id="sess_no_agent",
            stale_swarm_policy=stale_policy,
        )
        # Sanity: pre-stage state had stale non-None.
        assert ctx.swarm_policy is stale_policy

        # Force agent=None — the cascade fallback path runs.
        with patch.object(
            agent_registry, "get_or_load", return_value=None,
        ), patch(
            "proxy.context.enforce_context_limits", lambda *_a, **_kw: None,
        ):
            ContextOptimizeStage().execute(ctx)

        # After stage: policy is None. Without the per-execute reset,
        # the stale SubagentPolicy would persist — no other code in
        # the cascade path writes to ctx.swarm_policy.
        assert ctx.swarm_policy is None


# ---------------------------------------------------------------------------
# Section 7 — Concurrent requests share one Agent/session, get independent
# request-local policies (deterministic via threading.Barrier).
# ---------------------------------------------------------------------------


class TestConcurrentRequestIsolation:
    def test_concurrent_contexts_do_not_share_swarm_policy(self):
        """Two concurrent requests on DIFFERENT sessions, each with
        its own Agent, produce independent request-local policies.

        Proves the production invariant: ``ctx.swarm_policy`` is stored
        solely on each ``RequestContext`` instance. There is no global
        counter, no session-level mutable policy, and no shared lock
        for policy storage.

        We exercise the canonical ``_install_swarm_policy`` helper
        directly (no DB I/O, no shared ``unittest.mock.patch.object``
        globals). The barrier synchronises both threads at the install
        boundary.
        """
        from proxy.pipeline.stage_modules.context_optimize import (
            _install_swarm_policy,
        )

        swarm_skill = _make_swarm_skill(
            "swarm_one", child_tools=frozenset({"read"}),
        )
        _register(swarm_skill)

        # Two distinct sessions, each with its own Agent (matches
        # production: one Agent per session_id).
        swarm_agent = Agent(session_id="sess_swarm", agent_id="a_swarm")
        plain_agent = Agent(session_id="sess_plain", agent_id="a_plain")
        agent_registry.register(swarm_agent)
        agent_registry.register(plain_agent)
        _activate(swarm_agent, "swarm_one")
        # plain_agent has no active mode — the helper falls through to None.

        ctx_swarm = _make_ctx(session_id="sess_swarm")
        ctx_plain = _make_ctx(session_id="sess_plain")

        policy_sentinel = swarm_skill.subagents
        barrier = threading.Barrier(2)
        errors: list[BaseException] = []

        def run_swarm() -> None:
            try:
                barrier.wait()
                _install_swarm_policy(ctx_swarm, swarm_agent)
            except BaseException as e:  # noqa: BLE001
                errors.append(e)

        def run_plain() -> None:
            try:
                barrier.wait()
                _install_swarm_policy(ctx_plain, plain_agent)
            except BaseException as e:  # noqa: BLE001
                errors.append(e)

        t1 = threading.Thread(target=run_swarm)
        t2 = threading.Thread(target=run_plain)
        t1.start()
        t2.start()
        t1.join(timeout=10)
        t2.join(timeout=10)

        assert not errors, f"thread errors: {errors}"

        # Independent request-local storage — neither context leaks
        # into the other.
        assert ctx_swarm.swarm_policy is policy_sentinel
        assert ctx_plain.swarm_policy is None
        # The two contexts themselves are distinct objects.
        assert ctx_swarm is not ctx_plain

    def test_concurrent_requests_sharing_session_get_same_immutable_policy(self):
        """Two concurrent requests on the SAME session with the SAME
        active mode both receive the same immutable ``SubagentPolicy``
        reference (identity-equal). Each RequestContext is a fresh
        object; the Agent is the shared anchor; the policy is shared by
        identity, not by mutable state.
        """
        from proxy.pipeline.stage_modules.context_optimize import (
            _install_swarm_policy,
        )

        swarm_skill = _make_swarm_skill(
            "shared_swarm", child_tools=frozenset({"read"}),
        )
        _register(swarm_skill)

        # Production invariant: one Agent per session_id.
        shared_agent = Agent(session_id="sess_shared", agent_id="a_shared")
        agent_registry.register(shared_agent)
        _activate(shared_agent, "shared_swarm")

        policy_sentinel = swarm_skill.subagents
        ctx_a = _make_ctx(session_id="sess_shared")
        ctx_b = _make_ctx(session_id="sess_shared")

        barrier = threading.Barrier(2)
        errors: list[BaseException] = []

        def run_a() -> None:
            try:
                barrier.wait()
                _install_swarm_policy(ctx_a, shared_agent)
            except BaseException as e:  # noqa: BLE001
                errors.append(e)

        def run_b() -> None:
            try:
                barrier.wait()
                _install_swarm_policy(ctx_b, shared_agent)
            except BaseException as e:  # noqa: BLE001
                errors.append(e)

        t1 = threading.Thread(target=run_a)
        t2 = threading.Thread(target=run_b)
        t1.start()
        t2.start()
        t1.join(timeout=10)
        t2.join(timeout=10)

        assert not errors, f"thread errors: {errors}"

        # Both contexts receive the SAME immutable object. Identity
        # equality (not value equality) is the strict invariant the
        # PostProcess boundary uses when comparing references.
        assert ctx_a.swarm_policy is policy_sentinel
        assert ctx_b.swarm_policy is policy_sentinel
        assert ctx_a.swarm_policy is ctx_b.swarm_policy
        # Yet the contexts are distinct.
        assert ctx_a is not ctx_b
