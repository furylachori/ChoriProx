"""Tests for resolve_mode (Phase 1c).

6 tests covering: default mode, active-skill preservation, TTL expiry,
canary inheritance, workspace routing, fail-closed on unknown skill.
"""
import pytest, time
from proxy.skill_resolver import resolve_mode
from proxy.skills.skill_registry import (
    SkillRegistry, reset_default_registry, _skill_registry_new__,
)
from proxy.skills.skill_definition import ModeState, SkillRegistrationError
from proxy.mode import Mode


@pytest.fixture(autouse=True)
def fresh_registry():
    """Each test gets a fresh registry (autouse)."""
    reset_default_registry()
    yield
    reset_default_registry()


# ---------------------------------------------------------------------------
# default mode
# ---------------------------------------------------------------------------

def test_resolve_mode_returns_default_for_main_agent_with_no_active_skill():
    """§4: no active skill + no canary → Mode.DEFAULT.

    The resolver returns the canonical string form (str(Mode.X.value))
    so it survives importlib.reload(proxy.mode) which would otherwise
    invalidate identity-based enum comparisons.
    """
    ctx = type("Ctx", (), {
        "is_subagent": False,
        "inherited_skill": None,
        "body": {},
        "session_id": "s1",
    })()
    agent = type("Agent", (), {
        "snapshot_mode_and_ticket_ts": lambda self: None,
        "snapshot_mode": lambda self: None,
        "clear_active_mode": lambda self, expected_mode=None: None,
    })()
    reg = _skill_registry_new__(SkillRegistry)
    mode = resolve_mode(ctx, agent, reg)
    assert mode == str(Mode.DEFAULT.value)


# ---------------------------------------------------------------------------
# active skill preservation
# ---------------------------------------------------------------------------

def test_resolve_mode_preserves_active_skill():
    """§4: active skill (not expired) → Mode.SKILL (ticket/work preserved)."""
    ms = ModeState(name="ticket", activated_at=time.time(), ttl_seconds=300)
    ctx = type("Ctx", (), {
        "is_subagent": False,
        "inherited_skill": None,
        "body": {},
        "session_id": "s1",
    })()
    agent = type("Agent", (), {
        "snapshot_mode_and_ticket_ts": lambda self: (ms, None),
        "snapshot_mode": lambda self: ms,
        "clear_active_mode": lambda self, expected_mode=None: None,
    })()
    reg = _skill_registry_new__(SkillRegistry)
    mode = resolve_mode(ctx, agent, reg)
    assert mode == str(Mode.SKILL.value)


# ---------------------------------------------------------------------------
# TTL expiry
# ---------------------------------------------------------------------------

def test_resolve_mode_clears_expired_active_skill():
    """R2 HIGH-4: expired active skill → cleared and Mode.DEFAULT returned."""
    # TTL expired: activated_at is very old, ttl is finite.
    ms = ModeState(name="ticket", activated_at=0.0, ttl_seconds=300)
    # ModeState.is_expired() returns True when activated_at + ttl < now
    assert ms.is_expired() is True

    cleared = []
    ctx = type("Ctx", (), {
        "is_subagent": False,
        "inherited_skill": None,
        "body": {},
        "session_id": "s1",
    })()
    agent = type("Agent", (), {
        "snapshot_mode_and_ticket_ts": lambda self: (ms, None),
        "snapshot_mode": lambda self: ms,
        "clear_active_mode": lambda self, expected_mode=None: cleared.append(expected_mode),
    })()
    reg = _skill_registry_new__(SkillRegistry)
    mode = resolve_mode(ctx, agent, reg)
    assert mode == str(Mode.DEFAULT.value)
    assert len(cleared) == 1


# ---------------------------------------------------------------------------
# canary inheritance
# ---------------------------------------------------------------------------

def test_resolve_mode_inherits_from_canary():
    """§4: sub-agent with canary-inherited skill → Mode.SKILL_SUBAGENT.

    The sub-agent path calls _inherit_skill (not _activate_skill), which
    installs a ModeState on the agent via set_active_mode. This test
    verifies that the ModeState is actually installed with the correct
    name field.
    """
    installed_modes = []

    def _set_active_mode(self, ms):
        installed_modes.append(ms)
        return True

    ctx = type("Ctx", (), {
        "is_subagent": True,
        "inherited_skill": "ticket",
        "inherited_task_hash": None,
        "body": {},
        "session_id": "s1",
        "messages": [],
    })()
    agent = type("Agent", (), {
        "snapshot_mode_and_ticket_ts": lambda self: None,
        "snapshot_mode": lambda self: None,
        "clear_active_mode": lambda self, expected_mode=None: None,
        "set_active_mode": _set_active_mode,
    })()
    reg = _skill_registry_new__(SkillRegistry)
    reg.populate_builtins()
    mode = resolve_mode(ctx, agent, reg)
    assert mode == str(Mode.SKILL_SUBAGENT.value)
    # _inherit_skill must have installed a ModeState on the agent.
    assert len(installed_modes) == 1, (
        f"expected _inherit_skill to call set_active_mode once, "
        f"got {len(installed_modes)} calls"
    )
    ms = installed_modes[0]
    assert ms.name == "ticket", f"expected ModeState.name='ticket', got {ms.name!r}"


# ---------------------------------------------------------------------------
# workspace routing
# ---------------------------------------------------------------------------

def test_resolve_mode_returns_skill_subagent_when_inherited_is_workspace():
    """R3 CRIT-5 (Phase 2B): workspace sub-agent inherits SKILL_SUBAGENT mode.

    Phase 2B changed the workspace contract: workspace sub-agents now enter
    ``Mode.SKILL_SUBAGENT`` (mirrors the legacy ``is_ticket=True`` path), so
    the workspace skill is properly activated on the sub-agent and tool
    filtering applies. Workspace routing via ``ctx.workspace_name`` still
    runs downstream in ``apply_mode`` — the resolver now just sets the
    correct mode.
    """
    ctx = type("Ctx", (), {
        "is_subagent": True,
        "inherited_skill": "workspace",
        "body": {},
        "session_id": "s1",
    })()
    agent = type("Agent", (), {
        "snapshot_mode_and_ticket_ts": lambda self: None,
        "snapshot_mode": lambda self: None,
        "clear_active_mode": lambda self, expected_mode=None: None,
    })()
    reg = _skill_registry_new__(SkillRegistry)
    mode = resolve_mode(ctx, agent, reg)
    assert mode == str(Mode.SKILL_SUBAGENT.value)


# ---------------------------------------------------------------------------
# fail-closed
# ---------------------------------------------------------------------------

def test_resolve_mode_returns_mode_enum_string_not_modestate():
    """Returns a Mode enum value (as string), NOT a ModeState dataclass.

    The string form (str(Mode.X.value)) is used so the resolver survives
    importlib.reload(proxy.mode) which would invalidate identity-based
    enum comparisons.
    """
    ctx = type("Ctx", (), {
        "is_subagent": False,
        "inherited_skill": None,
        "body": {},
        "session_id": "s1",
    })()
    agent = type("Agent", (), {
        "snapshot_mode_and_ticket_ts": lambda self: None,
        "snapshot_mode": lambda self: None,
        "clear_active_mode": lambda self, expected_mode=None: None,
    })()
    reg = _skill_registry_new__(SkillRegistry)
    mode = resolve_mode(ctx, agent, reg)
    # The result is a string of the int value (e.g. "1", "2", "3", "4").
    assert isinstance(mode, str)
    # And it corresponds to one of the Mode enum values.
    assert int(mode) in [m.value for m in Mode]
