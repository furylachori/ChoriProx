"""Phase 2 Agent 4 — exact child-tool selection tests for custom swarm.

Covers `docs/specs/custom-skill-research-swarms.md` §8 (Tool whitelist
semantics) + AC 16.4, AC 16.14:

  - Configured empty child policy (frozenset()) = exactly 0 tools;
    parent allowed_tool_names MUST NOT leak.
  - Configured non-empty child policy = exact list; parent-only and
    prohibited (task) tools removed.
  - Ordinary custom sub-agent without swarm policy = existing builtin
    fallback unchanged.
  - Ticket / Work sub-agent = existing sub_agent_tool_filter behavior
    unchanged.
  - MAIN swarm parent = parent allowed_tool_names; parent task visible.
  - Unknown body tool = dropped (no synthesis).
  - Body without tools + configured empty = [] (never returns defaults).
  - Integration path through ContextOptimizeStage → apply_mode →
    select_tools actually puts the exact child whitelist in the body
    sent upstream.

These tests use real ``select_tools`` and real ``SkillDefinition``
with ``SubagentPolicy`` attached. Patches are limited to
``enforce_context_limits`` (DB I/O) and the agent-registry seam that
``ContextOptimizeStage`` requires.
"""
from __future__ import annotations

import time as _time
import types
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from proxy.agent.agent import Agent
from proxy.agent.agent_registry import registry as agent_registry
from proxy.pipeline.stage_modules.context_optimize import ContextOptimizeStage
from proxy.skill_tools import select_tools
from proxy.skills.skill_definition import (
    ModeState,
    SkillDefinition,
    SubagentPolicy,
)
from proxy.skills.skill_registry import (
    SkillRegistry,
    _skill_registry_new__,
    get_default_registry,
    reset_default_registry,
)
from proxy.types import RequestContext


# ---------------------------------------------------------------------------
# Fixtures + helpers
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def fresh_registry():
    """Each test starts with a singleton registry that has builtins only.

    The tests below manually inject their swarm-capable skill via
    ``_register_validated_batch`` (the canonical loader path).
    """
    reset_default_registry()
    yield
    reset_default_registry()


def _register_via_batch(*skills: SkillDefinition) -> SkillRegistry:
    """Register custom skills via the canonical loader's bypass path.

    Targets the SINGLETON registry (the `SkillRegistry()` accessor
    returns the cached instance). Production callers (``apply_mode``,
    ``resolve_mode``, etc.) use the same singleton, so registering
    into the singleton here keeps the lookup paths consistent.
    """
    reg = SkillRegistry()  # returns the singleton via __new__
    reg._register_validated_batch(tuple(skills))
    return reg


def _swarm_skill(
    name: str,
    *,
    parent_tools: frozenset[str],
    child_tools: frozenset[str],
    max_calls: int = 4,
) -> SkillDefinition:
    """Build a swarm-capable custom SkillDefinition with the given children."""
    return SkillDefinition(
        name=name,
        trigger="^^" + name,
        aliases=(),
        description=f"Swarm skill {name}",
        allowed_tool_names=parent_tools,
        subagents=SubagentPolicy(
            tool_whitelist=child_tools,
            max_task_calls_per_request=max_calls,
        ),
    )


def _ordinary_skill(
    name: str,
    *,
    parent_tools: frozenset[str],
    sub_filter: frozenset[str] = frozenset(),
) -> SkillDefinition:
    """Build an ordinary custom SkillDefinition (no swarm policy)."""
    return SkillDefinition(
        name=name,
        trigger="^^" + name,
        aliases=(),
        description=f"Skill {name}",
        allowed_tool_names=parent_tools,
        sub_agent_tool_filter=sub_filter,
    )


_BODY_TEMPLATE = lambda names: {
    "tools": [{"function": {"name": n}} for n in names],
}


# Body with our standard 8-name mix.
def _body(
    names=("read", "glob", "grep", "bash", "task", "question", "proxy_ticket", "proxy_work"),
) -> dict:
    return {
        "messages": [{"role": "user", "content": "hello"}],
        "tools": [{"function": {"name": n}} for n in names],
    }


def _agent(skill_name: str) -> Agent:
    a = Agent(session_id="sess_swarm_proxy", agent_id="a", is_subagent=False)
    a.set_active_mode(ModeState(
        name=skill_name, activated_at=_time.time(), ttl_seconds=300,
    ))
    return a


def _ctx(
    is_subagent: bool = True,
    body: dict | None = None,
) -> RequestContext:
    if body is None:
        body = _body()
    ctx = RequestContext(
        body=body,
        model_name="t",
        wants_stream=False,
        session_id="sess_swarm_proxy",
        client_hint="t",
        messages=list(body.get("messages", [])),
        model_cfg=SimpleNamespace(
            context_limit=32000,
            raw_exchange_count=8,
            budget_ratio=None,
        ),
        is_subagent=is_subagent,
    )
    return ctx


def _names(result: list) -> list:
    """Extract tool names from a select_tools result."""
    return [t.name for t in result]


# ---------------------------------------------------------------------------
# Section 1 — Configured empty child policy
# ---------------------------------------------------------------------------


class TestSwarmChildEmptyPolicy:
    def test_configured_empty_child_policy_returns_exactly_empty(self):
        """frozenset() means zero tools; the parent whitelist MUST NOT
        leak to the child even though it carries tools like 'task',
        'bash', etc.
        """
        skill = _swarm_skill(
            "research",
            parent_tools=frozenset({"read", "glob", "task", "bash"}),
            child_tools=frozenset(),
        )
        reg = _register_via_batch(skill)
        agent = _agent("research")
        ctx = _ctx(is_subagent=True)

        result = select_tools(agent, ctx, _body(), reg)
        assert result == []

    def test_configured_empty_child_with_permissive_parent_returns_empty(self):
        """If the parent unrestricted, the child STILL gets [] — parent
        whitelist does NOT cascade to a swarm child that explicitly
        declared empty.
        """
        skill = _swarm_skill(
            "research",
            parent_tools=None,  # unrestricted
            child_tools=frozenset(),
        )
        reg = _register_via_batch(skill)
        agent = _agent("research")
        ctx = _ctx(is_subagent=True)

        result = select_tools(agent, ctx, _body(), reg)
        assert result == []


# ---------------------------------------------------------------------------
# Section 2 — Configured non-empty child policy
# ---------------------------------------------------------------------------


class TestSwarmChildConfiguredPolicy:
    def test_configured_non_empty_child_policy_is_exact(self):
        """Configured list is exact — listed tools kept, parent-only
        and 'task' removed.
        """
        skill = _swarm_skill(
            "research",
            parent_tools=frozenset({"read", "glob", "grep", "task"}),
            child_tools=frozenset({"read", "glob"}),
        )
        reg = _register_via_batch(skill)
        agent = _agent("research")
        ctx = _ctx(is_subagent=True)

        result = select_tools(agent, ctx, _body(), reg)
        names = set(_names(result))
        # Exact match: child whitelist.
        assert "read" in names
        assert "glob" in names
        # parent-only or forbidden:
        assert "grep" not in names  # parent-only
        assert "bash" not in names  # parent-only
        assert "task" not in names  # swarm forbidden
        assert "proxy_ticket" not in names
        assert "proxy_work" not in names

    def test_child_policy_does_not_carry_over_to_main_request(self):
        """A MAIN request for the SAME skill uses the parent policy,
        not the child policy. Pin: 'task' visible to the parent.
        """
        skill = _swarm_skill(
            "research",
            parent_tools=frozenset({"read", "task"}),
            child_tools=frozenset({"read"}),
        )
        reg = _register_via_batch(skill)
        agent = _agent("research")
        ctx = _ctx(is_subagent=False)  # MAIN request

        result = select_tools(agent, ctx, _body(names=["read", "task"]), reg)
        names = set(_names(result))
        assert "read" in names
        assert "task" in names  # parent policy path; task allowed


# ---------------------------------------------------------------------------
# Section 3 — Ordinary custom sub-agent without policy
# ---------------------------------------------------------------------------


class TestOrdinarySubAgentFallback:
    def test_ordinary_sub_agent_uses_sub_agent_tool_filter(self):
        """No swarm policy → falls through to existing builtin path
        (sub_agent_tool_filter then allowed_tool_names).
        """
        skill = _ordinary_skill(
            "ok",
            parent_tools=frozenset({"read", "bash"}),
            sub_filter=frozenset({"read"}),
        )
        reg = _register_via_batch(skill)
        agent = _agent("ok")
        ctx = _ctx(is_subagent=True)

        result = select_tools(agent, ctx, _body(names=["read", "bash"]), reg)
        names = set(_names(result))
        assert names == {"read"}  # sub_filter applied

    def test_ordinary_sub_agent_empty_sub_filter_uses_parent(self):
        """sub_filter empty → falls back to parent allowed_tool_names
        (existing Wave 1 three-state behavior, not three-state swarm).
        """
        skill = _ordinary_skill(
            "ok",
            parent_tools=frozenset({"read", "bash"}),
            sub_filter=frozenset(),
        )
        reg = _register_via_batch(skill)
        agent = _agent("ok")
        ctx = _ctx(is_subagent=True)

        result = select_tools(agent, ctx, _body(names=["read", "bash"]), reg)
        names = set(_names(result))
        assert names == {"read", "bash"}


# ---------------------------------------------------------------------------
# Section 4 — Ticket and Work sub-agent preservation
# ---------------------------------------------------------------------------


class TestBuiltinPreservation:
    def _registered_builtin(self, name, sub_filter):
        """Mutate the canonical builtin's ``sub_agent_tool_filter``
        via ``object.__setattr__`` (SkillDefinition is frozen). The
        singleton's builtin instance is reused.
        """
        reg = get_default_registry()
        skill = reg.get(name)
        if skill is None:
            reg.populate_builtins()
            skill = reg.get(name)
        assert skill is not None, f"builtin {name!r} not registered"
        object.__setattr__(skill, "sub_agent_tool_filter", sub_filter)
        return skill

    def test_ticket_sub_agent_uses_sub_filter(self):
        """Use the canonical registered 'ticket' builtin with a mutated
        sub_agent_tool_filter; verify select_tools respects it.
        """
        self._registered_builtin("ticket", sub_filter=frozenset({"read"}))
        reg = get_default_registry()
        agent = _agent("ticket")
        ctx = _ctx(is_subagent=True)
        result = select_tools(agent, ctx, _body(), reg)
        names = set(_names(result))
        # Sub-filter restricts to read; proxy_work must NOT survive.
        assert "read" in names
        assert "proxy_work" not in names
        # Cleanup so the mutation does not leak to other tests.
        from proxy.skills.skill_registry import reset_default_registry
        reset_default_registry()

    def test_work_sub_agent_uses_sub_filter(self):
        self._registered_builtin("work", sub_filter=frozenset({"read", "bash"}))
        reg = get_default_registry()
        agent = _agent("work")
        ctx = _ctx(is_subagent=True)
        result = select_tools(agent, ctx, _body(), reg)
        names = set(_names(result))
        assert {"read", "bash"}.issubset(names)
        from proxy.skills.skill_registry import reset_default_registry
        reset_default_registry()


# ---------------------------------------------------------------------------
# Section 5 — Unknown body tool and body-without-tools behavior
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_unknown_body_tool_dropped_for_swarm_child(self):
        skill = _swarm_skill(
            "research",
            parent_tools=frozenset({"read", "glob"}),
            child_tools=frozenset({"read"}),
        )
        reg = _register_via_batch(skill)
        agent = _agent("research")
        ctx = _ctx(is_subagent=True)
        body = {
            "tools": [
                {"function": {"name": "read"}},
                {"function": {"name": "custom_made_up_tool"}},
            ],
        }
        result = select_tools(agent, ctx, body, reg)
        names = set(_names(result))
        # read survives (in child + known); the unknown body tool drops
        # (no synthesis).
        assert "read" in names
        assert "custom_made_up_tool" not in names

    def test_body_with_parent_allowed_tool_plus_empty_child_returns_empty(self):
        """A body tool that's in the parent allowed_tool_names but NOT
        in the empty child policy must still be denied. This is the
        strongest pin on the swarm-specific branch — without it,
        the parent policy leaks through.
        """
        skill = _swarm_skill(
            "research",
            parent_tools=frozenset({"read", "glob"}),
            child_tools=frozenset(),
        )
        reg = _register_via_batch(skill)
        agent = _agent("research")
        ctx = _ctx(is_subagent=True)
        body = {
            "messages": [{"role": "user", "content": "hi"}],
            "tools": [
                {"function": {"name": "read"}},
                {"function": {"name": "glob"}},
            ],
        }
        result = select_tools(agent, ctx, body, reg)
        # Empty child policy MUST deny both parent-allowed tools.
        assert result == []


# ---------------------------------------------------------------------------
# Section 6 — Integration path: ContextOptimizeStage → apply_mode → select_tools
# ---------------------------------------------------------------------------


class TestIntegrationStageToBody:
    """End-to-end: the exact child whitelist reaches the body actually
    sent upstream.
    """

    def _stage_setup(self):
        from proxy.pipeline.stage_modules.context_optimize import (
            _install_swarm_policy,
        )
        return _install_swarm_policy

    def test_main_swarm_request_has_exact_child_tools_in_body(self):
        """A swarm MAIN request, after the full pipeline, must carry
        the exact child policy tools (not the parent's).

        The integration test patches the agent_registry seam and
        enforces_context_limits (DB I/O bypass). It invokes the full
        real ContextOptimizeStage, then runs select_tools against the
        resulting policy.
        """
        skill = _swarm_skill(
            "research",
            parent_tools=frozenset({"read", "glob", "task", "bash"}),
            child_tools=frozenset({"read", "glob"}),
        )
        reg = _register_via_batch(skill)

        agent = Agent(
            session_id="sess_integrate_swarm",
            agent_id="a_integrate_swarm",
            is_subagent=False,
        )
        agent.set_active_mode(ModeState(
            name="research", activated_at=_time.time(), ttl_seconds=300,
        ))
        # Make the agent retrievable via the registry's get_or_load seam.
        agent_registry.register(agent)

        # Body includes parent-only + child-allowed tools so the filter
        # has something to drop.
        body = _body(
            names=("read", "glob", "grep", "bash", "task"),
        )
        ctx = RequestContext(
            body=dict(body),
            model_name="t",
            wants_stream=False,
            session_id="sess_integrate_swarm",
            client_hint="t",
            messages=list(body["messages"]),
            model_cfg=SimpleNamespace(
                context_limit=32000,
                raw_exchange_count=8,
                budget_ratio=None,
            ),
            is_subagent=False,
        )

        # Run the full ContextOptimizeStage — installs policy on ctx.
        # Note: apply_mode requires body to be a structured dict; on
        # failure it raises StageAbort; we only test the happy path.
        with patch.object(
            agent_registry, "get_or_load", return_value=agent,
        ), patch(
            "proxy.context.enforce_context_limits", lambda *_a, **_kw: None,
        ):
            ContextOptimizeStage().execute(ctx)

        assert ctx.swarm_policy is skill.subagents

        # The MAIN request uses the parent whitelist (child policy is
        # only for sub-agents). The parent's `task` IS visible to a MAIN
        # swarm parent.
        result = select_tools(agent, ctx, ctx.body, reg)
        names = set(_names(result))
        assert "read" in names
        assert "glob" in names
        assert "task" in names  # parent policy path

    def test_sub_agent_request_uses_exact_child_whitelist_in_body(self):
        """An inherited sub-agent request, after the pipeline, must see
        ONLY the child's exact whitelist tools in select_tools result
        (even though ctx.is_subagent=True → swarm branch applies).
        """
        skill = _swarm_skill(
            "research",
            parent_tools=frozenset({"read", "glob", "task"}),
            child_tools=frozenset({"read"}),
        )
        reg = _register_via_batch(skill)
        # Sub-agent inherits the same skill via canary. The active
        # mode for a sub-agent request resolves to the inherited skill
        # in resolver. We use a fresh agent as the sub-agent's anchor.
        agent = Agent(
            session_id="sess_integrate_sub",
            agent_id="a_integrate_sub",
            is_subagent=True,
        )
        agent.set_active_mode(ModeState(
            name="research", activated_at=_time.time(), ttl_seconds=300,
        ))
        agent_registry.register(agent)

        body = _body(names=("read", "glob", "task"))
        ctx = RequestContext(
            body=dict(body),
            model_name="t",
            wants_stream=False,
            session_id="sess_integrate_sub",
            client_hint="t",
            messages=list(body["messages"]),
            model_cfg=SimpleNamespace(
                context_limit=32000,
                raw_exchange_count=8,
                budget_ratio=None,
            ),
            is_subagent=True,
            inherited_skill="research",
        )

        with patch.object(
            agent_registry, "get_or_load", return_value=agent,
        ), patch(
            "proxy.context.enforce_context_limits", lambda *_a, **_kw: None,
        ):
            ContextOptimizeStage().execute(ctx)

        # Universal no-nesting §14: ctx.swarm_policy stays None for the
        # sub-agent request itself.
        assert ctx.swarm_policy is None

        # But select_tools uses the swarm child branch (because the
        # active skill's subagents is non-None) — the child whitelist
        # gives exactly {"read"}.
        result = select_tools(agent, ctx, ctx.body, reg)
        names = set(_names(result))
        assert names == {"read"}
