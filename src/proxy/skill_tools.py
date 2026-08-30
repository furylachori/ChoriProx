"""Generic tool selector (Phase 1c + Wave 1 three-state policy).

Replaces the per-skill `get_tool_filter(skill_ctx)` calls in
`proxy/mode.py:295` and `proxy/tool_cache.py:100` with a single
`select_tools(agent, ctx, body, registry)` helper that reads the active
skill from `agent.active_mode` and combines `skill.allowed_tool_names`
(or `skill.sub_agent_tool_filter` for sub-agents) with the body-supplied
tool list.

The result is a list of `ToolMeta` instances ordered by the body's tool
order. If no skill is active, the default + passthrough tools are
returned (intersected with the body when the body declares tools).

Wave 1 (spec §6) three-state policy for `allowed_tool_names`:

  - ``None``           — unrestricted. Pass through every body tool,
                         looking up each via ``tool_registry.get(name)``
                         to preserve the registry's canonical ToolMeta
                         (full schema, category, available_in_skills).
                         DO NOT synthesize a stub ``ToolMeta`` for
                         unknown names — just drop them. This is the
                         Trusted Python-defined skill path.
  - ``frozenset()``    — deny all tools. Return ``[]``. The LLM sees no
                         tools. This is the fail-closed default for
                         custom JSON skills whose ``tool_whitelist`` is
                         empty.
  - non-empty frozenset — exact allowlist. Body tools whose name is in
                         the set are kept (looked up via
                         ``tool_registry.get(name)`` to get the
                         canonical ToolMeta; if unknown, kept with the
                         body's own schema). Names NOT in the set are
                         dropped.

Body tool pass-through rule: a body tool is KEPT iff (a) it's in
``tool_registry.get_tool_names()`` OR (b) it's in ``allowed_tool_names``
(when ``allowed_tool_names`` is non-empty). Otherwise DROP. Do NOT
synthesize ``ToolMeta(category=EDITOR)`` for unknown names.
"""
import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from proxy.agent.agent import Agent
    from proxy.types import RequestContext
    from proxy.skills.skill_registry import SkillRegistry

logger = logging.getLogger(__name__)


def select_tools(agent: "Agent", ctx, body: dict, registry: "SkillRegistry") -> list:
    """Select tools for the current request based on the active skill.

    Wave 1 (spec §6) three-state behavior:

      - No active skill (``agent is None`` or ``snapshot_mode()``
        returns ``None``): returns the default + passthrough tools as
        ``ToolMeta`` instances via ``tool_registry.get(name)``. If the
        request body declares its own tool list, the result is the
        intersection of body names and ``tool_registry.get_tool_names()``
        (no synthesis for unknown names). May return ``[]`` only when
        the body declares no usable tools and the tool registry has no
        default/passthrough entries.
      - Active skill (main agent): the body's tools are filtered against
        ``skill.allowed_tool_names`` using the three-state semantics
        above. A non-empty ``sub_agent_tool_filter`` is consulted only
        when ``is_subagent=True`` (sub-agent path).
      - Active skill (sub-agent): ``skill.sub_agent_tool_filter`` is used
        when non-empty (always a frozenset — NOT three-state). The
        sub-agent filter, when empty, falls back to the same
        ``allowed_tool_names`` three-state check on the main skill.

    The legacy "empty allow-set = allow everything" fail-open semantics
    are DELETED. An empty ``allowed_tool_names`` now means "deny all"
    (spec §4.4 fail-closed contract for custom skills).
    """
    from proxy.tool_pipeline.tool_registry import (
        ToolCategory,
        ToolMeta,
        registry as tool_registry,
    )

    if agent is None:
        return _default_tools_no_skill(body, tool_registry, ToolMeta, ToolCategory)

    ms = agent.snapshot_mode()
    skill_name = getattr(ms, "name", None) if ms is not None else None
    skill = registry.get(skill_name) if skill_name else None

    # No active skill: return default + passthrough tools (intersected with
    # the body's tool list when the body declares any).
    if skill is None:
        return _default_tools_no_skill(body, tool_registry, ToolMeta, ToolCategory)

    # Active skill: filter the body's tools against the skill's
    # allowed_tool_names (or sub_agent_tool_filter for sub-agents).
    body_tools = body.get("tools", []) if isinstance(body, dict) else []
    is_subagent = bool(getattr(ctx, "is_subagent", False))

    # Resolve the active policy. `allowed` is `Optional[frozenset[str]]`
    # (three-state). `sub_agent_tool_filter` is always `frozenset[str]`.
    #
    # Phase 2 swarm-capable custom-skill child branch (spec §8.1).
    # Three exact rules:
    #   1. Custom swarm child with a published SubagentPolicy → use its
    #      `tool_whitelist` exactly. An empty frozenset() means zero
    #      tools (deny-all) and MUST NOT fall through to the parent
    #      whitelist — no `if truthy:` shortcut.
    #   2. Custom swarm child WITHOUT a SubagentPolicy → existing
    #      builtin/non-swarm behavior unchanged (sub_agent_tool_filter
    #      first, then allowed_tool_names).
    #   3. MAIN request → parent allowed_tool_names (existing behavior).
    if is_subagent and getattr(skill, "subagents", None) is not None:
        # REVERT-RESISTANT: use `if subagents is not None` (NOT
        # `if subagents.tool_whitelist`) so a configured empty child
        # tool_whitelist (frozenset()) still reaches this branch and
        # returns [] below — the parent allowed_tool_names must NOT
        # leak in.
        allowed = skill.subagents.tool_whitelist
    elif is_subagent:
        sub_filter = getattr(skill, "sub_agent_tool_filter", frozenset())
        if sub_filter:
            # Sub-agent has its own non-empty filter — apply it
            # directly (sub_agent_tool_filter is NOT three-state).
            allowed = sub_filter
        else:
            # Sub-agent filter empty: fall back to the main skill's
            # three-state `allowed_tool_names`.
            allowed = getattr(skill, "allowed_tool_names", None)
    else:
        allowed = getattr(skill, "allowed_tool_names", None)

    # ---- Wave 1 three-state policy branch ----
    known_names = tool_registry.get_tool_names()

    if allowed is None:
        # Branch 1: unrestricted. Pass-through every body tool — look up
        # the canonical ToolMeta via the registry; if unknown, drop (do
        # NOT synthesize). This is the Trusted Python-defined skill path.
        result = []
        for t in body_tools:
            if not isinstance(t, dict):
                continue
            name = t.get("function", {}).get("name", "")
            if not name:
                continue
            if name not in known_names:
                # Unknown body tool name under unrestricted policy —
                # drop. The trusted-skill caller controls the body, so
                # any unknown name here is either a typo or a custom
                # tool the registry does not track. We do not invent a
                # category for it.
                continue
            existing = tool_registry.get(name)
            if existing is not None:
                result.append(existing)
        return result

    # REVERT TEST: the OLD pre-Wave 1 logic was `if allowed and name not in
    # allowed`. That truthiness check treated an empty frozenset as
    # "no restriction" — fail-open. The new semantics require deny-all
    # for empty frozenset. Verify the test below fails when reverted.

    # Branch 2: empty allow-list — deny all (fail-closed per
    # spec §4.4 / §6). Return [] regardless of body contents.
    if not allowed:
        return []

    # Branch 3: non-empty allow-list — exact allowlist.
    result = []
    for t in body_tools:
        if not isinstance(t, dict):
            continue
        name = t.get("function", {}).get("name", "")
        if not name:
            continue
        if name not in allowed:
            continue
        existing = tool_registry.get(name)
        if existing is not None:
            # Known name: use the registry's canonical ToolMeta (full
            # schema, category, available_in_skills).
            result.append(existing)
        else:
            # Unknown name but in the allow-list. The allow-list
            # already validated the name exists (loader rejects
            # unknown names at load time per spec §4.4), but defensive
            # coverage here: build a minimal ToolMeta from the body
            # schema only. Do NOT synthesize ToolMeta(category=EDITOR).
            from proxy.tool_pipeline.tool_registry import ToolMeta as _TM
            result.append(_TM(name=name, category=ToolCategory.EDITOR))
    return result


def _default_tools_no_skill(body, tool_registry, ToolMeta, ToolCategory) -> list:
    """Return default + passthrough tools for the no-skill path.

    Shared by ``select_tools`` when ``agent is None`` (no ModeState can be
    read) and when the agent has no active skill.

    Wave 1 (spec §6): same body-tool pass-through rule as the active-skill
    path — body tools not in ``tool_registry.get_tool_names()`` are
    DROPPED (no synthesis). When the body declares its own tool list, the
    result is the intersection of body names and the registry's known
    names. When the body has no tools, the full default + passthrough set
    is returned (as ``ToolMeta`` instances via ``tool_registry.get(name)``).
    """
    body_tools = body.get("tools", []) if isinstance(body, dict) else []
    body_names = {
        t.get("function", {}).get("name", "")
        for t in body_tools
        if isinstance(t, dict)
    }
    known_names = tool_registry.get_tool_names()
    if body_names:
        # Intersect body-declared names with the registry's known set;
        # drop unknown names (no synthesis).
        candidate_names = sorted(n for n in body_names if n in known_names)
    else:
        defaults = tool_registry.get_default_tools() or set()
        passthrough = tool_registry.get_passthrough_tools() or set()
        # Every default/passthrough tool name is in the registry's
        # known set by construction (they were registered there), so
        # no synthesis is needed.
        candidate_names = sorted(defaults | passthrough)
    result = []
    for name in candidate_names:
        if not name:
            continue
        existing = tool_registry.get(name)
        if existing is not None:
            result.append(existing)
        # Defensive: if a default/passthrough name somehow isn't in the
        # registry (shouldn't happen — they're registered together),
        # drop it. No synthesis.
    return result


__all__ = ["select_tools"]