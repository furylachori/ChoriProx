"""Tests for select_tools — Wave 1 three-state policy (spec §6).

Covers the three-state ``allowed_tool_names`` semantics:

  - ``None``           — unrestricted. Pass through every body tool that
                          is known to the tool registry.
  - ``frozenset()``    — deny all. Returns ``[]``.
  - non-empty frozenset — exact allowlist. Body tools whose name is in
                          the set are kept (looked up via the registry).
                          Names NOT in the set are dropped.

Body tool pass-through rule: a body tool is KEPT iff (a) it's in
``tool_registry.get_tool_names()`` OR (b) it's in
``allowed_tool_names`` (when non-empty). Otherwise DROP. No synthetic
``ToolMeta`` for unknown names.
"""
import pytest, time
from proxy.skill_tools import select_tools
from proxy.skills.skill_registry import (
    SkillRegistry, reset_default_registry, _skill_registry_new__,
)
from proxy.skills.skill_definition import ModeState, SkillDefinition


@pytest.fixture(autouse=True)
def fresh_registry():
    """Each test gets a fresh registry (autouse)."""
    reset_default_registry()
    yield
    reset_default_registry()


def _make_skill(name, allowed=None, sub_filter=frozenset()) -> SkillDefinition:
    """Build a SkillDefinition with the given tool filters.

    ``allowed`` is the three-state policy field (``None`` / ``frozenset()``
    / non-empty frozenset). Default ``None`` = unrestricted (matches the
    Wave 1 BaseSkill default after this wave's changes).

    SkillDefinition is @dataclass(frozen=True) so we cannot mutate it
    after construction; tests pass the desired filters up front.
    """
    return SkillDefinition(
        name=name,
        trigger="^^" + name,
        aliases=(),
        allowed_tool_names=allowed,
        sub_agent_tool_filter=sub_filter,
    )


# ---------------------------------------------------------------------------
# DENY ALL: empty frozenset → []
# ---------------------------------------------------------------------------

def test_select_tools_returns_empty_for_skill_with_no_tools():
    """§5 + Wave 1: ``allowed_tool_names=frozenset()`` means DENY ALL.

    The previous (pre-Wave 1) test relied on the legacy
    ``if allowed and name not in allowed`` truthiness check — an empty
    ``allowed_tool_names`` was treated as "no restriction" (fail-open).
    Under the Wave 1 three-state semantics, an empty frozenset is the
    EXPLICIT deny-all sentinel, so the result is ``[]`` regardless of
    the body's declared tools.
    """
    reg = _skill_registry_new__(SkillRegistry)
    skill = _make_skill("deny_all", allowed=frozenset())
    object.__setattr__(reg, "_inner", {"deny_all": skill})
    object.__setattr__(reg, "_trigger_map", {"^^deny_all": "deny_all"})
    ms = ModeState(name="deny_all", activated_at=time.time(), ttl_seconds=300)
    agent = type("Agent", (), {
        "snapshot_mode": lambda self: ms,
    })()
    ctx = type("Ctx", (), {"is_subagent": False})()
    body = {"tools": [{"function": {"name": "bash"}}, {"function": {"name": "read"}}]}
    result = select_tools(agent, ctx, body, reg)
    assert result == []


def test_select_tools_deny_all_drops_every_tool_even_known_ones():
    """§5 + Wave 1: ``allowed_tool_names=frozenset()`` overrides body tools.

    Even when the body declares ``bash`` (which IS in the registry),
    the deny-all policy wins — the result is ``[]``.
    """
    reg = _skill_registry_new__(SkillRegistry)
    skill = _make_skill("deny_all_2", allowed=frozenset())
    object.__setattr__(reg, "_inner", {"deny_all_2": skill})
    object.__setattr__(reg, "_trigger_map", {"^^deny_all_2": "deny_all_2"})
    ms = ModeState(name="deny_all_2", activated_at=time.time(), ttl_seconds=300)
    agent = type("Agent", (), {
        "snapshot_mode": lambda self: ms,
    })()
    ctx = type("Ctx", (), {"is_subagent": False})()
    body = {"tools": [{"function": {"name": "bash"}}]}
    result = select_tools(agent, ctx, body, reg)
    assert result == []


# ---------------------------------------------------------------------------
# UNRESTRICTED (None): pass-through body tools
# ---------------------------------------------------------------------------

def test_select_tools_with_none_allowed_passes_through_body_tools():
    """§5 + Wave 1: ``allowed_tool_names=None`` → pass-through.

    Every body tool whose name is known to ``tool_registry`` is kept.
    The result preserves the registry's canonical ``ToolMeta`` (full
    schema, category, available_in_skills).
    """
    reg = _skill_registry_new__(SkillRegistry)
    skill = _make_skill("unrestricted", allowed=None)
    object.__setattr__(reg, "_inner", {"unrestricted": skill})
    object.__setattr__(reg, "_trigger_map", {"^^unrestricted": "unrestricted"})
    ms = ModeState(name="unrestricted", activated_at=time.time(), ttl_seconds=300)
    agent = type("Agent", (), {
        "snapshot_mode": lambda self: ms,
    })()
    ctx = type("Ctx", (), {"is_subagent": False})()
    body = {"tools": [
        {"function": {"name": "read"}},
        {"function": {"name": "bash"}},
    ]}
    result = select_tools(agent, ctx, body, reg)
    names = sorted(t.name for t in result)
    # Both are known tools — both pass through.
    assert "read" in names
    assert "bash" in names


def test_select_tools_drops_unknown_body_tool_not_in_whitelist():
    """§5 + Wave 1: unknown body tool name is DROPPED (no synthesis).

    Under the unrestricted (``None``) policy, the body's
    ``not_a_real_tool`` is NOT known to ``tool_registry`` and is
    DROPPED — the function does NOT synthesize a stub
    ``ToolMeta(category=EDITOR)`` for unknown names (spec §6.1).
    """
    reg = _skill_registry_new__(SkillRegistry)
    skill = _make_skill("unrestricted2", allowed=None)
    object.__setattr__(reg, "_inner", {"unrestricted2": skill})
    object.__setattr__(reg, "_trigger_map", {"^^unrestricted2": "unrestricted2"})
    ms = ModeState(name="unrestricted2", activated_at=time.time(), ttl_seconds=300)
    agent = type("Agent", (), {
        "snapshot_mode": lambda self: ms,
    })()
    ctx = type("Ctx", (), {"is_subagent": False})()
    body = {"tools": [
        {"function": {"name": "read"}},
        {"function": {"name": "not_a_real_tool"}},
    ]}
    result = select_tools(agent, ctx, body, reg)
    names = [t.name for t in result]
    assert "read" in names
    # Unknown name dropped — no stub ToolMeta synthesized.
    assert "not_a_real_tool" not in names


def test_select_tools_keeps_unknown_body_tool_in_whitelist():
    """§5 + Wave 1: unknown-but-allowed body tool kept (pass-through).

    When the body declares a tool name that is in the (non-empty)
    allowlist but NOT in ``tool_registry``, the function keeps the
    tool with a minimal ToolMeta so the LLM still sees it. This is the
    body-schema pass-through rule — the allowlist trusts the author's
    intent (spec §6.1 "Body tool schemas that match are kept verbatim").
    """
    reg = _skill_registry_new__(SkillRegistry)
    skill = _make_skill("allow_unknown", allowed=frozenset({"custom_tool"}))
    object.__setattr__(reg, "_inner", {"allow_unknown": skill})
    object.__setattr__(reg, "_trigger_map", {"^^allow_unknown": "allow_unknown"})
    ms = ModeState(name="allow_unknown", activated_at=time.time(), ttl_seconds=300)
    agent = type("Agent", (), {
        "snapshot_mode": lambda self: ms,
    })()
    ctx = type("Ctx", (), {"is_subagent": False})()
    body = {"tools": [{"function": {"name": "custom_tool"}}]}
    result = select_tools(agent, ctx, body, reg)
    names = [t.name for t in result]
    assert "custom_tool" in names


def test_select_tools_keeps_known_body_tool_full_schema():
    """§5 + Wave 1: known body tool kept with registry's full ToolMeta.

    When the body declares a tool name that is in the allowlist AND in
    ``tool_registry``, the function uses the registry's canonical
    ToolMeta (full schema, category, available_in_skills) — NOT a
    stub. This preserves the registry's category information for the
    upstream LLM call.
    """
    from proxy.tool_pipeline.tool_registry import registry as tool_registry
    reg = _skill_registry_new__(SkillRegistry)
    skill = _make_skill("allow_known", allowed=frozenset({"read"}))
    object.__setattr__(reg, "_inner", {"allow_known": skill})
    object.__setattr__(reg, "_trigger_map", {"^^allow_known": "allow_known"})
    ms = ModeState(name="allow_known", activated_at=time.time(), ttl_seconds=300)
    agent = type("Agent", (), {
        "snapshot_mode": lambda self: ms,
    })()
    ctx = type("Ctx", (), {"is_subagent": False})()
    body = {"tools": [{"function": {"name": "read"}}]}
    result = select_tools(agent, ctx, body, reg)
    assert len(result) == 1
    # The canonical ToolMeta is identity-equal to the registry's entry
    # (the function uses ``tool_registry.get(name)`` to look it up).
    canonical = tool_registry.get("read")
    assert canonical is not None
    assert result[0] is canonical


# ---------------------------------------------------------------------------
# EXACT ALLOWLIST (non-empty frozenset)
# ---------------------------------------------------------------------------

def test_select_tools_returns_allowed_tool_names():
    """§5: tools in skill.allowed_tool_names (non-empty frozenset) that
    are also in ToolMeta.available_in_skills are returned as ToolMeta
    instances. The custom skill allows only ``read``; ``bash`` in the
    body is filtered out.
    """
    reg = _skill_registry_new__(SkillRegistry)
    skill = _make_skill("onlyread", allowed=frozenset({"read"}))
    object.__setattr__(reg, "_inner", {"onlyread": skill})
    object.__setattr__(reg, "_trigger_map", {"^^onlyread": "onlyread"})
    ms = ModeState(name="onlyread", activated_at=time.time(), ttl_seconds=300)
    agent = type("Agent", (), {"snapshot_mode": lambda self: ms})()
    ctx = type("Ctx", (), {"is_subagent": False})()
    body = {"tools": [
        {"function": {"name": "read"}},
        {"function": {"name": "bash"}},
    ]}
    result = select_tools(agent, ctx, body, reg)
    names = [t.name for t in result]
    assert "read" in names
    assert "bash" not in names


# ---------------------------------------------------------------------------
# sub-agent filter
# ---------------------------------------------------------------------------

def test_select_tools_uses_sub_agent_filter_when_is_subagent():
    """R3 CRIT-1: ctx.is_subagent=True → uses sub_agent_tool_filter.

    The source falls back to ``allowed_tool_names`` when
    ``sub_agent_tool_filter`` is empty, so the test must provide a
    NON-empty sub-agent filter. The sub-agent filter is a strict subset
    of the main filter (``read`` only) and the body's ``proxy_work``
    is therefore filtered out for sub-agents.
    """
    reg = _skill_registry_new__(SkillRegistry)
    skill = _make_skill(
        "subskill",
        # Main agent: full set (unrestricted). Sub-agent filter does the work.
        allowed=None,
        sub_filter=frozenset({"read"}),  # sub-agent: read only
    )
    object.__setattr__(reg, "_inner", {"subskill": skill})
    object.__setattr__(reg, "_trigger_map", {"^^subskill": "subskill"})
    ms = ModeState(name="subskill", activated_at=time.time(), ttl_seconds=300)
    agent = type("Agent", (), {"snapshot_mode": lambda self: ms})()
    ctx = type("Ctx", (), {"is_subagent": True})()
    body = {"tools": [{"function": {"name": "proxy_work"}}]}
    result = select_tools(agent, ctx, body, reg)
    # Sub-agent filter ({"read"}) excludes proxy_work.
    assert "proxy_work" not in [t.name for t in result]


# ---------------------------------------------------------------------------
# no active mode
# ---------------------------------------------------------------------------

def test_select_tools_returns_default_for_no_active_mode():
    """No active mode → default + passthrough (returns a list, possibly empty)."""
    reg = _skill_registry_new__(SkillRegistry)
    agent = type("Agent", (), {
        "snapshot_mode": lambda self: None,
    })()
    ctx = type("Ctx", (), {"is_subagent": False})()
    result = select_tools(agent, ctx, {"tools": []}, reg)
    assert len(result) > 0
    assert any(t.name == "bash" for t in result)


# ---------------------------------------------------------------------------
# agent=None guard (Commit 2)
# ---------------------------------------------------------------------------

def test_select_tools_handles_none_agent_without_crashing():
    """Commit 2: select_tools(agent=None, ...) must not crash.

    With agent=None there is no ModeState to read — the function must
    follow the same no-skill path it uses when ``snapshot_mode()``
    returns None: return the default + passthrough tools as a list.
    """
    reg = _skill_registry_new__(SkillRegistry)
    ctx = type("Ctx", (), {"is_subagent": False})()
    result = select_tools(None, ctx, {"tools": []}, reg)
    assert len(result) > 0
    assert any(t.name == "bash" for t in result)


def test_select_tools_none_agent_intersects_body_tools():
    """Commit 2: agent=None still restricts to body-declared tools.

    When the body declares its own tool list, the no-skill path returns
    only those names (intersected with the tool registry). The agent=None
    path must behave identically to the no-active-mode path. Wave 1:
    unknown body tool names are dropped (no synthesis).
    """
    reg = _skill_registry_new__(SkillRegistry)
    ctx = type("Ctx", (), {"is_subagent": False})()

    agent_with_none_mode = type("Agent", (), {
        "snapshot_mode": lambda self: None,
    })()
    body = {"tools": [
        {"function": {"name": "read"}},
        {"function": {"name": "bash"}},
    ]}

    none_agent_result = select_tools(None, ctx, body, reg)
    no_mode_result = select_tools(agent_with_none_mode, ctx, body, reg)

    none_names = sorted(t.name for t in none_agent_result)
    no_mode_names = sorted(t.name for t in no_mode_result)
    assert none_names == no_mode_names, (
        f"agent=None path {none_names} must match no-active-mode path "
        f"{no_mode_names}"
    )
    assert "read" in none_names
    assert "bash" in none_names


# ---------------------------------------------------------------------------
# cross-skill filtering
# ---------------------------------------------------------------------------

def test_select_tools_filters_by_available_in_skills():
    """Tools not in skill.allowed_tool_names are filtered out.

    The custom skill only allows ``read``; the body's ``proxy_ticket``
    is filtered out. The result is a list (possibly empty after
    filtering).
    """
    reg = _skill_registry_new__(SkillRegistry)
    skill = _make_skill("workskill", allowed=frozenset({"read"}))
    object.__setattr__(reg, "_inner", {"workskill": skill})
    object.__setattr__(reg, "_trigger_map", {"^^workskill": "workskill"})
    ms = ModeState(name="workskill", activated_at=time.time(), ttl_seconds=300)
    agent = type("Agent", (), {"snapshot_mode": lambda self: ms})()
    ctx = type("Ctx", (), {"is_subagent": False})()
    body = {"tools": [
        {"function": {"name": "read"}},
        {"function": {"name": "proxy_ticket"}},
    ]}
    result = select_tools(agent, ctx, body, reg)
    names = [t.name for t in result]
    # Only "read" is allowed by the custom skill.
    assert "read" in names
    assert "proxy_ticket" not in names