"""Tests for SkillDispatcher (Phase 1c).

10 tests covering: detect_command (4), dispatch (4), activate (1), error handling (1).
"""
import os, pytest, time
from proxy.skill_dispatcher import SkillDispatcher, DispatchResult
from proxy.skills.skill_registry import (
    SkillRegistry, get_default_registry, reset_default_registry, _skill_registry_new__,
)
from proxy.skills.skill_definition import (
    SkillDefinition, BaseSkill, TrustToken, _held_trust_token,
    SkillRegistrationError, ModeState,
)


@pytest.fixture(autouse=True)
def fresh_registry():
    """Each test gets a fresh registry (autouse, runs before/after every test)."""
    reset_default_registry()
    yield
    reset_default_registry()


# ---------------------------------------------------------------------------
# Helpers — build a SkillDefinition with the callables we need.
# SkillDefinition is @dataclass(frozen=True) so we cannot mutate it after
# construction; the test must provide a fully-configured instance up front.
# ---------------------------------------------------------------------------

def _make_test_skill(
    name: str = "test",
    trigger: str = "^^test",
    *,
    parse_subcommand=None,
    handle_direct=None,
    on_activate=None,
    get_system_prompt=None,
    allowed_tool_names=frozenset(),
    sub_agent_tool_filter=frozenset(),
    ttl_seconds: int = 300,
    **extra_fields,
) -> SkillDefinition:
    """Build a `SkillDefinition` with the provided callables wired in.

    `parse_subcommand(text)` MUST return `(action, arg)` — `action=None` triggers
    skill activation in the dispatcher. `handle_direct(ctx, action, arg)` should
    return a dict (handled=True) or `None` (fall through to activation).
    """
    return SkillDefinition(
        name=name,
        trigger=trigger,
        aliases=(),
        allowed_tool_names=allowed_tool_names,
        sub_agent_tool_filter=sub_agent_tool_filter,
        ttl_seconds=ttl_seconds,
        parse_subcommand=parse_subcommand,
        handle_direct=handle_direct,
        on_activate=on_activate,
        get_system_prompt=get_system_prompt,
        **extra_fields,
    )


def _make_ctx():
    """Build a minimal ctx-like object with the attributes the dispatcher reads."""
    return type("Ctx", (), {
        "session_id": "s1",
        "is_subagent": False,
        "inherited_skill": None,
        "body": {},
        "workspace_name": "",
        "subagent_ticket_mode": False,
        "seeded_tasks": [],
        "client_hint": None,
        "messages": [],
    })()


def _make_agent():
    """Build a minimal agent-like object with the methods the dispatcher calls."""
    return type("Agent", (), {
        "snapshot_mode": lambda self: None,
        "set_active_mode": lambda self, ms: None,
    })()


# ---------------------------------------------------------------------------
# detect_command
# ---------------------------------------------------------------------------

def test_detect_command_returns_none_when_no_trigger_match():
    """R1-CRIT-1: no trigger match → detect_command returns None."""
    reg = _skill_registry_new__(SkillRegistry)
    reg.populate_builtins()
    dispatcher = SkillDispatcher(reg)
    result = dispatcher.detect_command("hello world")
    assert result is None


def test_detect_command_finds_trigger_match():
    """detect_command returns the SkillDefinition for a matching trigger."""
    reg = _skill_registry_new__(SkillRegistry)
    reg.populate_builtins()
    dispatcher = SkillDispatcher(reg)
    result = dispatcher.detect_command("^^ticket list")
    assert result is not None
    assert result.name == "ticket"


def test_detect_command_sorts_by_trigger_length_desc():
    """R1 HIGH-10: longer triggers win when multiple overlap.

    Register two skills with overlapping triggers where the longer
    trigger is a strict prefix-extension of the shorter one. When the
    input matches both, the dispatcher must return the longer match.
    SkillDefinition trigger regex requires lowercase + ``[a-z0-9_-]``.
    """
    reg = _skill_registry_new__(SkillRegistry)
    long_skill = _make_test_skill(name="longer", trigger="^^abc-long")
    short_skill = _make_test_skill(name="shorter", trigger="^^abc")
    object.__setattr__(reg, "_inner", {
        "longer": long_skill,
        "shorter": short_skill,
    })
    object.__setattr__(reg, "_trigger_map", {
        "^^abc-long": "longer",
        "^^abc": "shorter",
    })
    dispatcher = SkillDispatcher(reg)
    # "^^abc-long" is the longer trigger and matches first (sorted DESC).
    result = dispatcher.detect_command("^^abc-long")
    assert result is not None
    assert result.name == "longer"


def test_detect_command_normalizes_zero_width_chars():
    """R4 MED-1: U+200B inside the trigger is stripped before match.

    A user can copy/paste a trigger that contains invisible zero-width
    characters (e.g., from a clipboard that encoded them). detect_command
    must normalize these away so the trigger still matches.
    """
    reg = _skill_registry_new__(SkillRegistry)
    reg.populate_builtins()
    dispatcher = SkillDispatcher(reg)
    # Insert U+200B between the ^'s and "ticket" — must still match.
    result = dispatcher.detect_command("^^\u200bticket list")
    assert result is not None
    assert result.name == "ticket"


# ---------------------------------------------------------------------------
# dispatch
# ---------------------------------------------------------------------------

def test_dispatch_returns_dispatch_result_with_handled_false_when_no_trigger():
    """R1-CRIT-1, R3-CRIT-2: dispatch() returns DispatchResult(handled=False) (NOT None)
    when no trigger matches. The caller passes through to the LLM by
    checking ``result.handled``.
    """
    reg = _skill_registry_new__(SkillRegistry)
    reg.populate_builtins()
    dispatcher = SkillDispatcher(reg)
    result = dispatcher.dispatch("hello world", _make_ctx(), _make_agent())
    assert isinstance(result, DispatchResult)
    assert result.handled is False


def test_dispatch_returns_direct_handle_result():
    """§3: when handle_direct returns a response, dispatch returns handled=True."""
    def _parse(text):
        return ("list_tasks", "")

    def _handle(ctx, action, arg):
        return {"choices": [{"message": {"role": "assistant", "content": "OK"}}]}

    reg = _skill_registry_new__(SkillRegistry)
    skill = _make_test_skill(
        name="test", trigger="^^test",
        parse_subcommand=_parse,
        handle_direct=_handle,
    )
    object.__setattr__(reg, "_inner", {"test": skill})
    object.__setattr__(reg, "_trigger_map", {"^^test": "test"})
    dispatcher = SkillDispatcher(reg)
    result = dispatcher.dispatch("^^test list_tasks", _make_ctx(), _make_agent())
    assert result.handled is True
    assert result.response is not None


def test_dispatch_returns_skill_activation_result():
    """§3: skill match + handle_direct returns None → has_rewrite=True.

    Phase 2B contract (SkillDispatcher): ``parse_subcommand`` returning a
    real ``(action, arg)`` AND ``handle_direct`` returning ``None`` triggers
    skill activation via ``dispatcher.activate(...)``, producing
    ``has_rewrite=True`` with the matched skill attached. (Note: the legacy
    Phase 1c behaviour where ``(None, "")`` from parse_subcommand triggered
    activation is no longer the contract — None action falls through to
    ``handled=False`` instead.)
    """
    def _parse(text):
        return ("activate", "do something")  # real action → reaches activation path

    reg = _skill_registry_new__(SkillRegistry)
    skill = _make_test_skill(
        name="test", trigger="^^test",
        parse_subcommand=_parse,
        handle_direct=lambda ctx, action, arg: None,
    )
    object.__setattr__(reg, "_inner", {"test": skill})
    object.__setattr__(reg, "_trigger_map", {"^^test": "test"})
    dispatcher = SkillDispatcher(reg)
    result = dispatcher.dispatch("^^test", _make_ctx(), _make_agent())
    assert result.has_rewrite is True
    assert result.skill is not None


def test_dispatch_re_raises_permission_error_from_handle_direct():
    """§3: PermissionError from handle_direct propagates up (re-raised)."""
    def _parse(text):
        return ("act", "")

    def _handle(ctx, action, arg):
        raise PermissionError("blocked")

    reg = _skill_registry_new__(SkillRegistry)
    skill = _make_test_skill(
        name="test", trigger="^^test",
        parse_subcommand=_parse,
        handle_direct=_handle,
    )
    object.__setattr__(reg, "_inner", {"test": skill})
    object.__setattr__(reg, "_trigger_map", {"^^test": "test"})

    class FailingAgent:
        def snapshot_mode(self): return None
        def set_active_mode(self, ms): pass

    dispatcher = SkillDispatcher(reg)
    with pytest.raises(PermissionError):
        dispatcher.dispatch("^^test", _make_ctx(), FailingAgent())


def test_dispatch_re_raises_value_error_from_parse_subcommand():
    """§3: ValueError from parse_subcommand propagates up (re-raised)."""
    def _parse(text):
        raise ValueError("bad subcommand")

    reg = _skill_registry_new__(SkillRegistry)
    skill = _make_test_skill(
        name="test", trigger="^^test",
        parse_subcommand=_parse,
        handle_direct=lambda ctx, action, arg: None,
    )
    object.__setattr__(reg, "_inner", {"test": skill})
    object.__setattr__(reg, "_trigger_map", {"^^test": "test"})
    dispatcher = SkillDispatcher(reg)
    with pytest.raises(ValueError):
        dispatcher.dispatch("^^test", _make_ctx(), _make_agent())


# ---------------------------------------------------------------------------
# activate
# ---------------------------------------------------------------------------

def test_activate_calls_on_activate_before_set_active_mode():
    """§3: on_activate is called BEFORE set_active_mode (ordering matters).

    Uses a non-trusted skill name (e.g. ``test``) to avoid the trust-gate
    on builtin names (which would require a held TRUSTED TrustToken).
    """
    order = []
    def _on_activate(ctx, ms):
        order.append("on_activate")

    reg = _skill_registry_new__(SkillRegistry)
    skill = _make_test_skill(
        name="test", trigger="^^test",
        on_activate=_on_activate,
    )
    object.__setattr__(reg, "_inner", {"test": skill})
    object.__setattr__(reg, "_trigger_map", {"^^test": "test"})
    dispatcher = SkillDispatcher(reg)

    class FakeAgent:
        def snapshot_mode(self): return None
        def set_active_mode(self, ms):
            order.append("set_active_mode")

    dispatcher.activate(skill, _make_ctx(), FakeAgent())
    assert order == ["on_activate", "set_active_mode"]


# ---------------------------------------------------------------------------
# activate - spec §7.3 fail-closed ordering
# ---------------------------------------------------------------------------

def test_activate_skips_set_active_mode_when_on_activate_raises_non_security():
    """§7.3: when on_activate raises a non-security exception,
    set_active_mode is NOT called and no mode is installed.

    Security/policy exceptions (ValueError, PermissionError) from
    on_activate are re-raised (test below). Any other exception must
    block activation entirely - this is the fail-closed guard.
    """
    set_calls = []

    def _on_activate_raises(ctx, ms):
        raise RuntimeError("on_activate failed (non-security)")

    reg = _skill_registry_new__(SkillRegistry)
    skill = _make_test_skill(
        name="test", trigger="^^test",
        on_activate=_on_activate_raises,
    )
    object.__setattr__(reg, "_inner", {"test": skill})
    object.__setattr__(reg, "_trigger_map", {"^^test": "test"})
    dispatcher = SkillDispatcher(reg)

    class FakeAgent:
        def snapshot_mode(self): return None
        def set_active_mode(self, ms):
            set_calls.append(ms)

    result = dispatcher.activate(skill, _make_ctx(), FakeAgent())
    assert result is False, "activate must return False when on_activate raises"
    assert set_calls == [], (
        "set_active_mode must NOT be called when on_activate raises a "
        "non-security exception (spec §7.3 fail-closed ordering)"
    )


def test_activate_installs_mode_when_on_activate_succeeds():
    """Spec §7.3 success path: when on_activate returns cleanly,
    set_active_mode IS called and the mode is installed.
    """
    set_calls = []

    def _on_activate_ok(ctx, ms):
        return None

    reg = _skill_registry_new__(SkillRegistry)
    skill = _make_test_skill(
        name="test", trigger="^^test",
        on_activate=_on_activate_ok,
    )
    object.__setattr__(reg, "_inner", {"test": skill})
    object.__setattr__(reg, "_trigger_map", {"^^test": "test"})
    dispatcher = SkillDispatcher(reg)

    class FakeAgent:
        def snapshot_mode(self): return None
        def set_active_mode(self, ms):
            set_calls.append(ms)

    result = dispatcher.activate(skill, _make_ctx(), FakeAgent())
    assert result is True, "activate must return True on clean success"
    assert len(set_calls) == 1, "set_active_mode must be called exactly once"


def test_activate_skips_set_active_mode_when_set_active_mode_raises():
    """Spec §7.3 fail-closed: when set_active_mode itself raises
    (e.g. lock contention), no mode is installed and activate
    returns False. Caller is responsible for rolling back any
    previously-snapshotted expected_mode (see dispatch.py / dispatch_stage.py).
    """
    set_calls = []

    reg = _skill_registry_new__(SkillRegistry)
    skill = _make_test_skill(
        name="test", trigger="^^test",
        on_activate=lambda ctx, ms: None,
    )
    object.__setattr__(reg, "_inner", {"test": skill})
    object.__setattr__(reg, "_trigger_map", {"^^test": "test"})
    dispatcher = SkillDispatcher(reg)

    class FakeAgent:
        def snapshot_mode(self): return None
        def set_active_mode(self, ms):
            set_calls.append(ms)
            raise RuntimeError("set_active_mode failed (lock contention)")

    result = dispatcher.activate(skill, _make_ctx(), FakeAgent())
    assert result is False, "activate must return False when set_active_mode raises"
    assert len(set_calls) == 1, (
        "set_active_mode was called (we recorded the attempt), but the "
        "exception was logged and the dispatcher reports activation failed"
    )


def test_activate_re_raises_security_exception_from_on_activate():
    """Spec §7.3: ValueError and PermissionError from
    on_activate propagate up unchanged so callers can surface them.
    """
    for exc_type in (ValueError, PermissionError):

        def _on_activate_raises_security(ctx, ms, _exc=exc_type):
            raise _exc("policy violation")

        reg = _skill_registry_new__(SkillRegistry)
        skill = _make_test_skill(
            name="test", trigger="^^test",
            on_activate=_on_activate_raises_security,
        )
        object.__setattr__(reg, "_inner", {"test": skill})
        object.__setattr__(reg, "_trigger_map", {"^^test": "test"})
        dispatcher = SkillDispatcher(reg)

        class FakeAgent:
            def snapshot_mode(self): return None
            def set_active_mode(self, ms):  # pragma: no cover
                raise AssertionError("set_active_mode must not be called when on_activate raised a security exception")

        with pytest.raises(exc_type):
            dispatcher.activate(skill, _make_ctx(), FakeAgent())


# ---------------------------------------------------------------------------
# dispatch - expected_mode snapshot (spec §7.3 rollback contract)
# ---------------------------------------------------------------------------

def test_dispatch_attaches_expected_mode_snapshot_on_activation_path():
    """Spec §7.3: the rollback anchor is captured AFTER activation so the
    compare-and-clear CAS in `agent.clear_active_mode(expected_mode=...)`
    matches the just-installed mode and the rollback actually clears it.
    """
    reg = _skill_registry_new__(SkillRegistry)
    skill = _make_test_skill(
        name="test", trigger="^^test",
        parse_subcommand=lambda text: ("act", ""),
        handle_direct=lambda ctx, action, arg: None,
        on_activate=lambda ctx, ms: None,
    )
    object.__setattr__(reg, "_inner", {"test": skill})
    object.__setattr__(reg, "_trigger_map", {"^^test": "test"})

    class FakeAgent:
        def __init__(self):
            self.active_mode = None  # sentinel — no pre-existing mode
        def snapshot_mode(self):
            return self.active_mode
        def set_active_mode(self, ms):
            self.active_mode = ms
            return True
        def clear_active_mode(self, expected_mode=None):
            # Faithful CAS — only clear if expected_mode IS active_mode.
            if expected_mode is not None and self.active_mode is not expected_mode:
                return False
            self.active_mode = None
            return True

    dispatcher = SkillDispatcher(reg)
    agent = FakeAgent()
    result = dispatcher.dispatch("^^test", _make_ctx(), agent)
    assert result.has_rewrite is True
    # The expected_mode must be the JUST-INSTALLED mode (the rollback anchor
    # that makes the downstream CAS succeed). Pre-fix this was the
    # pre-activation mode, which always mismatched the post-activation
    # active_mode and left the NEW mode to leak.
    assert result.expected_mode is not None
    assert getattr(result.expected_mode, "name", None) == "test"
    # Agent's active_mode was actually overwritten by activation.
    assert getattr(agent.active_mode, "name", None) == "test"
    # The CAS rollback against the expected_mode snapshot must succeed and
    # clear the just-installed mode. This is the regression-pinning check:
    # if the anchor were the pre-activation mode, the CAS would mismatch and
    # the rollback would leave the new mode on the agent.
    cleared = agent.clear_active_mode(expected_mode=result.expected_mode)
    assert cleared is True, (
        "rollback CAS must match: expected_mode (just-installed) IS active_mode"
    )
    assert agent.active_mode is None, (
        "rollback must clear the just-installed mode — pre-fix this left the "
        "new mode to leak to subsequent requests"
    )


def test_dispatch_expected_mode_none_on_no_trigger_match():
    """When no skill triggers, no snapshot is needed: expected_mode=None."""
    reg = _skill_registry_new__(SkillRegistry)
    reg.populate_builtins()
    dispatcher = SkillDispatcher(reg)

    class FakeAgent:
        def snapshot_mode(self): return "anything"
        def set_active_mode(self, ms): return True

    result = dispatcher.dispatch("hello world", _make_ctx(), FakeAgent())
    assert result.handled is False
    assert result.expected_mode is None


def test_dispatch_expected_mode_on_failed_activation_path():
    """When activation fails (on_activate raises non-security), the
    snapshot is still attached (so the caller has the rollback anchor)
    and has_rewrite=False signals the failure."""
    reg = _skill_registry_new__(SkillRegistry)
    skill = _make_test_skill(
        name="test", trigger="^^test",
        parse_subcommand=lambda text: ("act", ""),
        handle_direct=lambda ctx, action, arg: None,
        on_activate=lambda ctx, ms: (_ for _ in ()).throw(RuntimeError("boom")),
    )
    object.__setattr__(reg, "_inner", {"test": skill})
    object.__setattr__(reg, "_trigger_map", {"^^test": "test"})

    class FakeAgent:
        def __init__(self):
            self.active_mode = "previous-mode"
        def snapshot_mode(self): return self.active_mode
        def set_active_mode(self, ms):  # pragma: no cover
            raise AssertionError("set_active_mode must not be called when on_activate raised")
        def clear_active_mode(self, expected_mode=None): return True

    dispatcher = SkillDispatcher(reg)
    agent = FakeAgent()
    result = dispatcher.dispatch("^^test", _make_ctx(), agent)
    assert result.has_rewrite is False, "has_rewrite=False when activation failed"
    # Snapshot must still be attached even on failure path - caller can
    # pass it to clear_active_mode(expected_mode=...) if needed.
    assert result.expected_mode == "previous-mode"
    # Agent's state must NOT have been mutated.
    assert agent.active_mode == "previous-mode"


def test_rollback_clears_installed_mode_after_rewrite_failure():
    """Spec §7.3 (B1 fix): when `get_activation_rewrite` raises AFTER
    activation succeeded, the rollback path (``dispatch.py:481` /
    `dispatch_stage.py:96`) must compare-and-clear the just-installed
    mode. The anchor captured by `SkillDispatcher.dispatch()` IS the
    just-installed mode, so the CAS in `clear_active_mode(expected_mode=...)`
    matches and the rollback actually clears the mode — instead of leaving
    the new mode to leak to subsequent requests.

    This test exercises the full snapshot → CAS → rollback pipeline end-to-end
    at the `SkillDispatcher` level: dispatch installs a mode, the snapshot
    is captured, the downstream "rewrite failed" handler invokes the
    rollback, and we assert the agent's active_mode is cleared.
    """
    reg = _skill_registry_new__(SkillRegistry)
    # NOTE: use a non-trusted name like ``"test"`` because ``"ticket"``
    # and ``"work"`` are trusted builtins (TRUSTED_BUILTIN_SKILLS) and
    # require a held TRUSTED TrustToken to construct. The skill semantics
    # under test (activation → snapshot → rollback CAS) are independent
    # of whether the skill is a builtin.
    skill = _make_test_skill(
        name="test",
        trigger="^^test",
        parse_subcommand=lambda text: ("act", ""),
        handle_direct=lambda ctx, action, arg: None,
        on_activate=lambda ctx, ms: None,
        get_activation_rewrite=lambda ctx: (_ for _ in ()).throw(
            RuntimeError("rewrite boom")
        ),
    )
    object.__setattr__(reg, "_inner", {"test": skill})
    object.__setattr__(reg, "_trigger_map", {"^^test": "test"})

    class FakeAgent:
        def __init__(self):
            self.active_mode = None  # no pre-existing mode
            self.clear_calls = []    # record clear_active_mode invocations

        def snapshot_mode(self):
            return self.active_mode

        def set_active_mode(self, ms):
            self.active_mode = ms
            return True

        def clear_active_mode(self, expected_mode=None):
            self.clear_calls.append(expected_mode)
            # Real CAS: only clear when `expected_mode IS active_mode`.
            if expected_mode is not None and self.active_mode is not expected_mode:
                return False
            self.active_mode = None
            return True

    dispatcher = SkillDispatcher(reg)
    agent = FakeAgent()
    result = dispatcher.dispatch("^^test", _make_ctx(), agent)

    # (1) Activation succeeded → has_rewrite=True.
    assert result.has_rewrite is True, "activation should succeed (on_activate is benign)"

    # (2) The rollback anchor must be the JUST-INSTALLED mode (the new mode
    # is what we need to clear, not the pre-existing one — there is none).
    assert result.expected_mode is not None
    assert getattr(result.expected_mode, "name", None) == "test"

    # (3) Simulate the downstream rewrite-failure path: dispatch.py /
    # dispatch_stage.py catches the get_activation_rewrite exception and
    # invokes `agent.clear_active_mode(expected_mode=result.expected_mode)`.
    # The CAS must match (expected_mode IS active_mode) so the rollback
    # actually clears the mode. Pre-fix, `expected_mode` was the
    # pre-activation mode (None here) so the CAS could not match and the
    # NEW mode leaked.
    cleared = agent.clear_active_mode(expected_mode=result.expected_mode)
    assert cleared is True, (
        "rollback CAS must match — expected_mode is the just-installed mode"
    )
    assert agent.active_mode is None, (
        "after rollback, the just-installed mode must be cleared; pre-fix "
        "this left the new mode on the agent and it leaked to subsequent "
        "requests (B1 critical mode-leak)"
    )
    # The rollback was invoked exactly once with the expected snapshot.
    assert len(agent.clear_calls) == 1
    assert agent.clear_calls[0] is result.expected_mode
