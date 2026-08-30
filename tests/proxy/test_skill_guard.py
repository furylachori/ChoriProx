"""Tests for sub-agent guard (Phase 1c).

8 tests covering: sub-agent proxy_ticket/work access, READ_ONLY/NEVER matrix,
main-agent exemption, bare sub-agent default-deny, check_action callback.
"""
import pytest
from proxy.skill_guard import check_sub_agent_proxy_access
from proxy.skills.skill_registry import SkillRegistry, reset_default_registry, _skill_registry_new__
from proxy.skills.skill_definition import ModeState, ProxyAccessLevel
import time


@pytest.fixture(autouse=True)
def fresh_registry():
    """Each test gets a fresh registry (autouse)."""
    reset_default_registry()
    yield
    reset_default_registry()


def _make_tool_call(name, args=None):
    """Build a minimal ToolCall-like object with .name and .arguments."""
    tc = type("ToolCall", (), {})()
    tc.name = name
    tc.arguments = args or {}
    return tc


# ---------------------------------------------------------------------------
# proxy_ticket access
# ---------------------------------------------------------------------------

def test_check_sub_agent_proxy_ticket_access_allowed_for_list_tasks():
    """§7: sub-agent + ticket skill + proxy_ticket_list_tasks → allowed (None).

    R3 CRIT-2 fix: the previous version used a bare sub-agent (no
    `snapshot_mode`); that exception is removed — bare sub-agents are
    now default-deny for ALL `proxy_*`. The test now uses an active
    ticket skill mode so it exercises the per-skill access-level gate
    (FULL → list_tasks allowed).

    Slice 8b: the legacy monolithic `proxy_ticket` is decomposed into
    fine-grained tools (proxy_ticket_list_tasks, proxy_ticket_get_task, ...).
    The guard resolves action from the tool name (TICKET_TOOL_OPERATIONS).
    """
    ctx = type("Ctx", (), {"is_subagent": True})()
    ms = ModeState(name="ticket", activated_at=time.time(), ttl_seconds=300)
    agent = type("Agent", (), {"snapshot_mode": lambda self: ms})()
    reg = _skill_registry_new__(SkillRegistry)
    reg.populate_builtins()
    tc = _make_tool_call("proxy_ticket_list_tasks", {})
    assert check_sub_agent_proxy_access(tc, ctx, agent, reg) is None


def test_check_sub_agent_proxy_ticket_access_denied_for_create():
    """§7: bare sub-agent (no active mode) + proxy_ticket create_task → denied.

    R3 CRIT-2 fix: the bare-sub-agent exception that previously allowed
    `proxy_ticket list_tasks` / `append_review_note` is removed.
    All `proxy_*` tools now require an active trusted builtin skill.
    """
    ctx = type("Ctx", (), {"is_subagent": True})()
    agent = type("Agent", (), {"snapshot_mode": lambda self: None})()
    reg = _skill_registry_new__(SkillRegistry)
    reg.populate_builtins()
    tc = _make_tool_call("proxy_ticket", {"action": "create_task"})
    result = check_sub_agent_proxy_access(tc, ctx, agent, reg)
    assert result is not None
    assert "Sub-agents without an inherited trusted skill" in result


def test_check_sub_agent_proxy_work_access_denied_for_bare_subagent():
    """§7: bare sub-agent (no active mode) + proxy_work → denied.

    R3 CRIT-2 fix: the bare-sub-agent exception that previously
    emitted 'Ticket sub-agents cannot invoke proxy_work' is replaced
    with a generic 'no active skill' message that applies to ALL
    `proxy_*` tools.
    """
    ctx = type("Ctx", (), {"is_subagent": True})()
    agent = type("Agent", (), {"snapshot_mode": lambda self: None})()
    reg = _skill_registry_new__(SkillRegistry)
    reg.populate_builtins()
    tc = _make_tool_call("proxy_work", {})
    result = check_sub_agent_proxy_access(tc, ctx, agent, reg)
    assert result is not None
    # New error message (R3 CRIT-2): bare sub-agent cannot invoke
    # ANY proxy_* tool. The error string is generic ("proxy tools")
    # rather than naming the specific tool.
    assert "cannot call proxy tools" in result
    assert "inherited trusted skill" in result


# ---------------------------------------------------------------------------
# READ_ONLY / NEVER access matrix
# ---------------------------------------------------------------------------

def test_check_sub_agent_read_only_blocks_write_actions(monkeypatch):
    """§7: ProxyAccessLevel.READ_ONLY + delete_task → denied.

    The skill MUST be registered in the registry before the access
    check runs (otherwise the function returns the generic
    "not registered" string and the test passes for the wrong reason).
    The CAT-3 trust gate (`is_trusted_skill`) is bypassed via
    `monkeypatch` so the access-level logic is what produces the
    denial, not the trust gate.
    """
    monkeypatch.setattr("proxy.skill_guard.is_trusted_skill", lambda name: True)
    reg = _skill_registry_new__(SkillRegistry)
    sd = type("SD", (), {
        "name": "myplugin",
        "trigger": "^^myplugin",
        "aliases": (),
        "is_builtin": False,
        "sub_agent_proxy_ticket_access": ProxyAccessLevel.READ_ONLY,
        "sub_agent_proxy_work_access": ProxyAccessLevel.NEVER,
    })()
    reg._inner[sd.name] = sd
    ms = ModeState(name="myplugin", activated_at=time.time(), ttl_seconds=300)
    agent = type("Agent", (), {"snapshot_mode": lambda self: ms})()
    ctx = type("Ctx", (), {"is_subagent": True})()
    tc = _make_tool_call("proxy_ticket", {"action": "delete_task"})
    result = check_sub_agent_proxy_access(tc, ctx, agent, reg)
    assert result is not None
    # READ_ONLY should deny "delete_task" with a read-only / list_tasks message
    assert "read-only" in result.lower() or "list_tasks" in result


def test_check_sub_agent_never_blocks_all_actions(monkeypatch):
    """§7: ProxyAccessLevel.NEVER + any action → denied (no actions allowed).

    The skill MUST be registered in the registry before the access
    check runs (otherwise the function returns the generic
    "not registered" string and the test passes for the wrong reason).
    The CAT-3 trust gate (`is_trusted_skill`) is bypassed via
    `monkeypatch` so the NEVER access level is what produces the
    denial, not the trust gate.
    """
    monkeypatch.setattr("proxy.skill_guard.is_trusted_skill", lambda name: True)
    reg = _skill_registry_new__(SkillRegistry)
    sd = type("SD", (), {
        "name": "myplugin",
        "trigger": "^^myplugin",
        "aliases": (),
        "is_builtin": False,
        "sub_agent_proxy_ticket_access": ProxyAccessLevel.NEVER,
        "sub_agent_proxy_work_access": ProxyAccessLevel.NEVER,
    })()
    reg._inner[sd.name] = sd
    ms = ModeState(name="myplugin", activated_at=time.time(), ttl_seconds=300)
    agent = type("Agent", (), {"snapshot_mode": lambda self: ms})()
    ctx = type("Ctx", (), {"is_subagent": True})()
    tc = _make_tool_call("proxy_ticket", {"action": "list_tasks"})
    result = check_sub_agent_proxy_access(tc, ctx, agent, reg)
    assert result is not None
    # NEVER should deny any action with a "forbid" / "never" message
    assert "forbid" in result.lower() or "never" in result.lower()


# ---------------------------------------------------------------------------
# main-agent exemption
# ---------------------------------------------------------------------------

def test_check_sub_agent_returns_none_for_main_agent():
    """§7: main agent (not sub-agent) → guard returns None (no check)."""
    ctx = type("Ctx", (), {"is_subagent": False})()
    agent = type("Agent", (), {"snapshot_mode": lambda self: None})()
    reg = _skill_registry_new__(SkillRegistry)
    reg.populate_builtins()
    tc = _make_tool_call("proxy_ticket", {"action": "create_task"})
    assert check_sub_agent_proxy_access(tc, ctx, agent, reg) is None


# ---------------------------------------------------------------------------
# check_action callback wiring
# ---------------------------------------------------------------------------

def test_check_sub_agent_check_action_callback_blocks():
    """When skill.check_action returns non-None, it becomes the denial message."""
    from proxy.skills.skill_definition import ProxyAccessLevel
    from proxy.skills.skill_registry import SkillRegistry, _skill_registry_new__
    import time

    # NOTE: name "ticket" is in TRUSTED_BUILTIN_SKILLS, which is required
    # to bypass the CAT-3 trust gate (user-defined skills cannot grant
    # proxy_* to sub-agents — checked BEFORE check_action). The duck-typed
    # SkillDefinition bypasses BaseSkill.to_skill_definition()'s
    # TrustToken check, so we don't need a real TrustToken.
    def _my_check_action(tool_call, ctx):
        return f"check_action blocked {getattr(tool_call, 'name', '')}"

    sd = type("SD", (), {
        "name": "ticket",
        "trigger": "^^custom",
        "aliases": (),
        "is_builtin": True,
        "sub_agent_proxy_ticket_access": ProxyAccessLevel.FULL,
        "sub_agent_proxy_work_access": ProxyAccessLevel.FULL,
    })()
    # Set as instance attribute (not class attribute) so Python does NOT
    # bind it as a method. Same pattern as BaseSkill.check_action = staticmethod(...).
    sd.check_action = _my_check_action

    reg = _skill_registry_new__(SkillRegistry)
    reg._inner[sd.name] = sd
    ms = ModeState(name=sd.name, activated_at=time.time(), ttl_seconds=300)
    agent = type("Agent", (), {"snapshot_mode": lambda self: ms})()
    ctx = type("Ctx", (), {"is_subagent": True})()
    tc = type("ToolCall", (), {"name": "proxy_ticket", "arguments": {"action": "list_tasks"}})()

    result = check_sub_agent_proxy_access(tc, ctx, agent, reg)
    assert result is not None
    assert "check_action blocked" in result


def test_check_sub_agent_check_action_exception_returns_error():
    """R3-HIGH-4: If check_action raises, the exception is returned as an error (fail-closed)."""
    from proxy.skills.skill_definition import ProxyAccessLevel
    from proxy.skills.skill_registry import SkillRegistry, _skill_registry_new__
    import time

    # NOTE: name "ticket" is in TRUSTED_BUILTIN_SKILLS, which is required
    # to bypass the CAT-3 trust gate (user-defined skills cannot grant
    # proxy_* to sub-agents — checked BEFORE check_action). The duck-typed
    # SkillDefinition bypasses BaseSkill.to_skill_definition()'s
    # TrustToken check, so we don't need a real TrustToken.
    def _broken_check_action(tool_call, ctx):
        raise RuntimeError("skill is broken")

    sd = type("SD", (), {
        "name": "ticket",
        "trigger": "^^broken",
        "aliases": (),
        "is_builtin": True,
        "sub_agent_proxy_ticket_access": ProxyAccessLevel.FULL,
        "sub_agent_proxy_work_access": ProxyAccessLevel.FULL,
    })()
    # Set as instance attribute (not class attribute) so Python does NOT
    # bind it as a method. Same pattern as BaseSkill.check_action = staticmethod(...).
    sd.check_action = _broken_check_action

    reg = _skill_registry_new__(SkillRegistry)
    reg._inner[sd.name] = sd
    ms = ModeState(name=sd.name, activated_at=time.time(), ttl_seconds=300)
    agent = type("Agent", (), {"snapshot_mode": lambda self: ms})()
    ctx = type("Ctx", (), {"is_subagent": True})()
    tc = type("ToolCall", (), {"name": "proxy_ticket", "arguments": {"action": "list_tasks"}})()

    # R3-HIGH-4: exception is returned as an error (fail-closed), not swallowed
    result = check_sub_agent_proxy_access(tc, ctx, agent, reg)
    assert result is not None
    assert "skill is broken" in result


# ---------------------------------------------------------------------------
# R3 CRIT-2 fix: ALL `proxy_*` tools must be gated, not just proxy_ticket/work
# ---------------------------------------------------------------------------

def test_check_sub_agent_bare_subagent_denies_all_proxy_tools():
    """§7 (R3 CRIT-2): bare sub-agent is default-deny for EVERY `proxy_*` tool.

    Previously the guard only handled `proxy_ticket` (with a list_tasks
    exception) and `proxy_work`. Any other `proxy_*` tool (proxy_admin,
    proxy_config, proxy_session, ...) passed through and was allowed.
    Now all `proxy_*` tools are denied for bare sub-agents.
    """
    reg = _skill_registry_new__(SkillRegistry)
    reg.populate_builtins()
    ctx = type("Ctx", (), {"is_subagent": True})()
    agent = type("Agent", (), {"snapshot_mode": lambda self: None})()
    for tool in ("proxy_ticket", "proxy_work", "proxy_admin", "proxy_config", "proxy_session"):
        tc = _make_tool_call(tool, {})
        result = check_sub_agent_proxy_access(tc, ctx, agent, reg)
        assert result is not None, f"bare sub-agent should be denied for {tool}"
        assert "inherited trusted skill" in result, (
            f"unexpected error message for {tool}: {result!r}"
        )


def test_check_sub_agent_untrusted_skill_denies_all_proxy_tools(monkeypatch):
    """§7 (R3 CRIT-2): untrusted (user-defined) skill → default-deny ALL `proxy_*`.

    Previously the guard only checked the untrusted-skill branch for
    `proxy_ticket` and `proxy_work`. Other `proxy_*` tools (proxy_admin,
    ...) passed through to `skill.check_action`, which a user-defined
    skill could trivially allow (`check_action = lambda *_: None`).
    Now the untrusted-skill branch applies to ALL `proxy_*` tools.
    """
    # Force `is_trusted_skill` to return False for every name in this test.
    # (Bypasses the monkeypatch via the real TRUSTED_BUILTIN_SKILLS set.)
    monkeypatch.setattr("proxy.skill_guard.is_trusted_skill", lambda name: False)

    reg = _skill_registry_new__(SkillRegistry)
    sd = type("SD", (), {
        "name": "userplugin",
        "trigger": "^^userplugin",
        "aliases": (),
        "is_builtin": False,
        "sub_agent_proxy_ticket_access": ProxyAccessLevel.FULL,
        "sub_agent_proxy_work_access": ProxyAccessLevel.FULL,
        # No `check_action` defined — fall-through to None would otherwise
        # allow the call. The trust gate must catch this first.
    })()
    reg._inner[sd.name] = sd
    ms = ModeState(name="userplugin", activated_at=time.time(), ttl_seconds=300)
    agent = type("Agent", (), {"snapshot_mode": lambda self: ms})()
    ctx = type("Ctx", (), {"is_subagent": True})()
    for tool in ("proxy_ticket", "proxy_work", "proxy_admin", "proxy_config", "proxy_session"):
        tc = _make_tool_call(tool, {"action": "list_tasks"})
        result = check_sub_agent_proxy_access(tc, ctx, agent, reg)
        assert result is not None, (
            f"untrusted skill should be denied for {tool} "
            f"(R3 CRIT-2: ALL proxy_* must be gated, not just ticket/work)"
        )
        assert "user-defined skill" in result.lower(), (
            f"unexpected error message for {tool}: {result!r}"
        )


@pytest.mark.xfail(strict=True, reason="Pending strict-mode enforcement: any proxy_* not in explicit allowlist must be denied.")
def test_check_sub_agent_unconfigured_proxy_tool_xfail_strict_mode_pending():
    """Strict-mode contract: any proxy_* not in the explicit allowlist is denied.

    The CURRENT (post-CR2) behavior for a TRUSTED builtin skill with no
    `check_action` and an unconfigured `proxy_*` tool (e.g. `proxy_admin`)
    is to ALLOW the call (returns None). That is a security gap: closing it
    requires a per-tool allowlist on `SkillDefinition` and a strict deny
    for tools outside the allowlist. This test pins the DESIRED behavior
    so the gap is visible; once the strict-mode enforcement ships, the
    assertion flips to ``is not None`` and this xfail becomes an XPASS →
    strict failure (xfail marker enforces the expected outcome).
    """
    from proxy.skills.skill_definition import ProxyAccessLevel
    from proxy.skills.skill_registry import SkillRegistry, _skill_registry_new__
    import time

    class CustomSkill:
        name = "ticket"
        trigger = "^^ticket"
        aliases = ()
        is_builtin = True
        sub_agent_proxy_ticket_access = ProxyAccessLevel.FULL
        sub_agent_proxy_work_access = ProxyAccessLevel.FULL
        # No explicit allowlist for proxy_admin

    reg = _skill_registry_new__(SkillRegistry)
    sd = CustomSkill()
    reg._inner[sd.name] = sd
    ms = ModeState(name="ticket", activated_at=time.time(), ttl_seconds=300)
    agent = type("Agent", (), {"snapshot_mode": lambda self: ms})()
    ctx = type("Ctx", (), {"is_subagent": True})()
    tc = type("ToolCall", (), {"name": "proxy_admin", "arguments": {}})()

    # proxy_admin is not in the trusted builtin tools, so should be denied.
    # NOTE: ticket is in TRUSTED_BUILTIN_SKILLS, so is_trusted_skill returns
    # True. The guard falls through to skill.check_action. Without check_action
    # defined, the CURRENT behavior is to return None (allow). The DESIRED
    # strict-mode behavior is to deny (return a non-None error string).
    result = check_sub_agent_proxy_access(tc, ctx, agent, reg)
    assert result is not None, (
        "strict-mode contract violated: unconfigured proxy_admin must be denied "
        "for a trusted builtin skill (R3 hardening: per-tool allowlist required)"
    )


# ---------------------------------------------------------------------------
# Sub-agent `task` tool blocking (sub-agents cannot spawn sub-sub-agents)
#
# CRITICAL bug fix: sub-agents could previously see AND execute the `task`
# tool, spawning sub-sub-agents that current CLIs cannot represent. This
# section exercises MemoryToolGuard._check_blocked_tools_for_subagent, which
# reads the denylist from the active skill's SkillDefinition.blocked_tools_for_subagent.
# ---------------------------------------------------------------------------

def _make_guard_tool_call(name, args=None):
    """Build a minimal ToolCall-like object with .name and .arguments."""
    tc = type("ToolCall", (), {})()
    tc.name = name
    tc.arguments = args or {}
    return tc


def _make_guard_ctx(session_id="sess-test", is_subagent=True, subagent_ticket_mode=False):
    """Build a minimal guard ctx with the attributes _check_blocked_tools_for_subagent reads."""
    ctx = type("Ctx", (), {})()
    ctx.session_id = session_id
    ctx.is_subagent = is_subagent
    ctx.subagent_ticket_mode = subagent_ticket_mode
    return ctx


def _register_skill_mode_agent(monkeypatch, mode_name, session_id="sess-test"):
    """Patch agent_registry.get_or_load to return an agent with the given active mode.

    The agent's snapshot_mode() returns a ModeState whose `.name` is the skill
    name; the guard resolves the SkillDefinition via get_default_registry().
    """
    from proxy.agent.agent_registry import registry as agent_registry
    ms = ModeState(name=mode_name, activated_at=time.time(), ttl_seconds=300)
    agent = type("Agent", (), {"snapshot_mode": lambda self: ms})()
    monkeypatch.setattr(agent_registry, "get_or_load", lambda sid: agent)
    return agent


def test_task_tool_blocked_for_subagent_in_ticket_mode(monkeypatch):
    """Sub-agent in ticket mode calling `task` is blocked (no sub-sub-agent nesting)."""
    from proxy.tool_pipeline.tool_guard import MemoryToolGuard
    _register_skill_mode_agent(monkeypatch, "ticket")
    guard = MemoryToolGuard()
    tc = _make_guard_tool_call("task", {"description": "spawn another"})
    ctx = _make_guard_ctx(is_subagent=True, subagent_ticket_mode=True)
    err = guard._check_blocked_tools_for_subagent(tc, ctx)
    assert err is not None, "Sub-agent `task` call must be blocked in ticket mode"
    assert "sub-agents cannot spawn sub-sub-agents" in err.lower()
    assert "task" in err


def test_task_tool_blocked_for_subagent_in_work_mode(monkeypatch):
    """Sub-agent in work mode calling `task` is blocked (no sub-sub-agent nesting)."""
    from proxy.tool_pipeline.tool_guard import MemoryToolGuard
    _register_skill_mode_agent(monkeypatch, "work")
    guard = MemoryToolGuard()
    tc = _make_guard_tool_call("task", {"description": "spawn another"})
    ctx = _make_guard_ctx(is_subagent=True)
    err = guard._check_blocked_tools_for_subagent(tc, ctx)
    assert err is not None, "Sub-agent `task` call must be blocked in work mode"
    assert "sub-agents cannot spawn sub-sub-agents" in err.lower()
    assert "task" in err


def test_task_tool_allowed_for_main_agent(monkeypatch):
    """MAIN agent (is_subagent=False) calling `task` is allowed in both modes."""
    from proxy.tool_pipeline.tool_guard import MemoryToolGuard
    for mode_name in ("ticket", "work"):
        _register_skill_mode_agent(monkeypatch, mode_name)
        guard = MemoryToolGuard()
        tc = _make_guard_tool_call("task", {"description": "spawn sub-agent"})
        ctx = _make_guard_ctx(is_subagent=False)
        err = guard._check_blocked_tools_for_subagent(tc, ctx)
        assert err is None, (
            f"MAIN agent `task` call must be allowed in {mode_name} mode, got: {err}"
        )


def test_question_still_blocked_for_subagent(monkeypatch):
    """Regression: sub-agent `question` call is still blocked via the new general check."""
    from proxy.tool_pipeline.tool_guard import MemoryToolGuard
    _register_skill_mode_agent(monkeypatch, "ticket")
    guard = MemoryToolGuard()
    tc = _make_guard_tool_call("question", {"question": "what?", "header": "x"})
    ctx = _make_guard_ctx(is_subagent=True, subagent_ticket_mode=True)
    # The new general check blocks `question` (it is in blocked_tools_for_subagent).
    err = guard._check_blocked_tools_for_subagent(tc, ctx)
    assert err is not None, "Sub-agent `question` call must be blocked"
    assert "sub-agents cannot spawn sub-sub-agents" in err.lower()


def test_question_blocked_in_full_check_pipeline(monkeypatch):
    """Full guard.check() still blocks `question` for sub-agents (wiring intact)."""
    from proxy.tool_pipeline.tool_guard import MemoryToolGuard
    _register_skill_mode_agent(monkeypatch, "ticket")
    guard = MemoryToolGuard()
    tc = _make_guard_tool_call("question", {"question": "what?"})
    ctx = _make_guard_ctx(is_subagent=True, subagent_ticket_mode=True)
    err = guard.check(tc, ctx)
    assert err is not None
    assert "sub-agents cannot spawn sub-sub-agents" in err.lower()


def test_blocked_tools_for_subagent_enforced_from_skill_definition(monkeypatch):
    """The guard reads blocked_tools_for_subagent from the SkillDefinition, not a hardcoded list.

    Registers a custom skill with a custom blocked set (blocking `task` and a
    custom tool `weird_tool`) and asserts the guard honors the skill-defined
    denylist rather than the hardcoded `{"task", "question"}` fallback. This
    proves the guard consults the registry, not a constant.
    """
    from proxy.tool_pipeline.tool_guard import MemoryToolGuard
    from proxy.skills.skill_registry import (
        SkillRegistry, _skill_registry_new__, reset_default_registry,
    )
    import proxy.skills.skill_registry as skill_registry_mod

    # Build a fresh registry with a custom skill whose blocked set blocks
    # `weird_tool` (NOT in the hardcoded fallback set).
    reset_default_registry()
    custom_reg = _skill_registry_new__(SkillRegistry)
    custom_reg.populate_builtins()

    custom_sd = type("SD", (), {
        "name": "customtest",
        "trigger": "^^customtest",
        "aliases": (),
        "is_builtin": True,
        "sub_agent_proxy_ticket_access": ProxyAccessLevel.NEVER,
        "sub_agent_proxy_work_access": ProxyAccessLevel.NEVER,
        "blocked_tools_for_subagent": frozenset({"task", "weird_tool"}),
    })()
    custom_reg._inner[custom_sd.name] = custom_sd
    # Point the module-level default registry at our custom registry so the
    # guard's get_default_registry() call resolves to it.
    monkeypatch.setattr(skill_registry_mod, "_default_registry", custom_reg)

    _register_skill_mode_agent(monkeypatch, "customtest")
    guard = MemoryToolGuard()
    # `weird_tool` is in the skill-defined denylist but NOT in the hardcoded
    # {"task", "question"} fallback. If the guard reads the SkillDefinition,
    # this is blocked; if it uses the hardcoded list, it is allowed.
    tc = _make_guard_tool_call("weird_tool", {})
    ctx = _make_guard_ctx(is_subagent=True)
    err = guard._check_blocked_tools_for_subagent(tc, ctx)
    assert err is not None, (
        "Guard must read blocked_tools_for_subagent from the SkillDefinition "
        "(weird_tool is only blocked via the custom skill, not the hardcoded fallback)"
    )
    assert "weird_tool" in err
    # `task` is also blocked by the custom skill (and the fallback).
    tc2 = _make_guard_tool_call("task", {})
    err2 = guard._check_blocked_tools_for_subagent(tc2, ctx)
    assert err2 is not None
    # A tool NOT in the custom denylist is allowed (proves the set is the
    # custom one, not a broader constant).
    tc3 = _make_guard_tool_call("question", {})
    err3 = guard._check_blocked_tools_for_subagent(tc3, ctx)
    assert err3 is None, (
        "question is NOT in the custom skill's blocked set, so it must be allowed "
        "by _check_blocked_tools_for_subagent (the hardcoded fallback is NOT used "
        "when the skill is resolvable)"
    )

    reset_default_registry()


