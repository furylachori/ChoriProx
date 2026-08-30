"""tests/test_skill_definition.py — Phase 1a minimum-bar test suite.

13 minimum-bar tests (R3-HIGH-1 / R3-HIGH-8 / R3-HIGH-9 fix):
  1.  test_skill_definition_rejects_is_builtin_kwarg
  2.  test_skill_definition_is_builtin_readonly
  3.  test_skill_definition_public_surface_matches_helper
  4.  test_trusted_skill_filter_only_includes_builtins (parametrized)
      + test_is_builtin_recomputes_on_trusted_set_mutation (monkeypatch half)
  5.  test_trusted_builtin_skills_constant_value
  6.  test_skill_definition_rejects_invalid_trigger
  7.  test_skill_definition_rejects_invalid_name
  8.  test_skill_definition_rejects_duplicate_alias_trigger
  9.  test_mode_state_is_frozen
  10. test_mode_state_skill_state_is_readonly_mapping
  14. test_register_skill_rejects_untrusted_builtin
  15. test_mode_state_skill_state_mappingproxy_enforced
  16. test_base_skill_to_skill_definition_round_trip_all_23_fields

Plus recommended:
  17. test_base_skill_to_skill_definition_cache_is_builtin_immutable
  13. test_mode_state_keeps_is_expired

Plus 23 new regression tests added in sub-agent 5 (R1/R2/R3 followups):
  18. test_skill_definition_object_setattr_raises
  19. test_modestate_object_setattr_in_post_init_succeeds
  20. test_is_trusted_skill_public_name_contract
  21. test_skill_request_context_from_request_forwards_mode_state
  22. test_skill_definition_rejects_empty_trigger_post_init
  23. test_skill_definition_rejects_empty_name_post_init
  24. test_skill_definition_rejects_alias_dup_of_name
  25. test_skill_definition_rejects_invalid_alias_chars
  26. test_base_skill_to_skill_definition_handles_instance_level_callable
  27. test_mode_state_is_expired_returns_false_for_fresh
  28. test_mode_state_is_expired_persistent_zero_ttl
  29. test_mode_state_rejects_negative_ttl
  30. test_mode_state_rejects_future_activated_at
  31. test_mode_state_has_no_refresh_method
  32. test_base_skill_setattr_blocks_definition_reassignment
  33. test_base_skill_setattr_blocks_name_mutation_after_conversion
  34. test_base_skill_definition_cache_returns_same_object_across_mutation
  35. test_skill_definition_rejects_untrusted_builtin_at_construction
  36. test_register_skill_rejects_non_skill_definition_input
  37. test_register_skill_allows_skilldefinition_overwrite  (renamed Wave 2.1, was ..._rejects_duplicate_builtin_name)
  38. test_register_skill_rejects_untrusted_via_trust_token  (Wave 2.1: vacuous TRUSTED branch fixed)
  38b. test_trust_token_for_trusted_caller_gate  (NEW Wave 2.1)
  39. test_register_legacy_rejects_reserved_name_via_trust_gate
  40. test_register_legacy_raises_skill_registration_error
  41. test_skill_registration_error_is_permission_error_subclass
   42. test_reset_default_registry_is_unconditional
"""
import time
import pytest
from dataclasses import FrozenInstanceError
from types import MappingProxyType

from proxy.skills.skill_definition import (
    BaseSkill,
    ModeState,
    ProxyAccessLevel,
    SkillDefinition,
    SkillRegistrationError,
    SkillRequestContext,
    TRUSTED_BUILTIN_SKILLS,
    is_trusted_skill,
    _trust_token_local,
)
from proxy.skills.skill_registry import SkillRegistry, TrustToken, reset_default_registry


# ---------------------------------------------------------------------------
# Helper: hold a TRUSTED TrustToken on the current thread
# ---------------------------------------------------------------------------
class _HeldTrustToken:
    """Context manager + decorator that holds a TRUSTED TrustToken on
    `_trust_token_local` so that `SkillDefinition(name=<builtin>, ...)`
    construction passes the R2-6 trust gate.

    The R2-6 (HIGH) fix at construction time reads
    `_trust_token_local.token` and raises `SkillRegistrationError` if
    no TRUSTED token is held. Tests that construct builtin
    `SkillDefinition` instances MUST wrap the construction (or the
    whole test) in this helper.

    Usage::

        with _HeldTrustToken():
            sd = SkillDefinition(name="ticket", trigger="^^ticket")
    """

    def __enter__(self):
        _trust_token_local.token = TrustToken.for_trusted_caller()
        return self

    def __exit__(self, *exc):
        # Clear the thread-local so the token does not leak across tests.
        if hasattr(_trust_token_local, "token"):
            del _trust_token_local.token


@pytest.fixture
def _held_trust_token():
    """Pytest fixture: hold a TRUSTED TrustToken for the duration of a test."""
    with _HeldTrustToken():
        yield


# ---------------------------------------------------------------------------
# Test 1 (T1 MED): is_builtin kwarg rejected
# ---------------------------------------------------------------------------
def test_skill_definition_rejects_is_builtin_kwarg():
    """SkillDefinition(is_builtin=True) raises TypeError (CAT-3)."""
    with pytest.raises(TypeError):
        SkillDefinition(name="ticket", trigger="^^ticket", is_builtin=True)  # noqa
    with pytest.raises(TypeError):
        SkillDefinition(name="ticket", trigger="^^ticket", is_builtin=False)  # noqa


# ---------------------------------------------------------------------------
# Test 2 (N3 MED): is_builtin is read-only
# ---------------------------------------------------------------------------
def test_skill_definition_is_builtin_readonly(_held_trust_token):
    """Assignment to sd.is_builtin raises AttributeError."""
    sd = SkillDefinition(name="ticket", trigger="^^ticket")
    with pytest.raises(AttributeError):
        sd.is_builtin = True  # noqa


# ---------------------------------------------------------------------------
# Test 3 (N3 MED): property matches helper for every name
# ---------------------------------------------------------------------------
def test_skill_definition_public_surface_matches_helper(_held_trust_token):
    """sd.is_builtin == is_trusted_skill(sd.name) for every name."""
    for name in ["ticket", "work", "attack", "review", "myplugin", "workspace"]:
        sd = SkillDefinition(name=name, trigger=f"^^{name}")
        assert sd.is_builtin == is_trusted_skill(name), (
            f"is_builtin/is_trusted_skill mismatch for {name!r}"
        )


# ---------------------------------------------------------------------------
# Test 4 (T2 MED, R3-HIGH-4 rewrite + R3-HIGH-6 restored live-recompute)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("name,expected", [
    ("ticket", True), ("work", True),
    ("attack", False), ("review", False), ("myplugin", False),
])
def test_trusted_skill_filter_only_includes_builtins(name, expected, _held_trust_token):
    sd = SkillDefinition(name=name, trigger=f"^^{name}")
    assert sd.is_builtin is expected


def test_is_builtin_recomputes_on_trusted_set_mutation(_held_trust_token):
    """is_builtin reads TRUSTED_BUILTIN_SKILLS at access time (R3-HIGH-6)."""
    from proxy.skills import skill_definition as sd_module
    sd = SkillDefinition(name="work", trigger="^^work")
    assert sd.is_builtin is True
    original = sd_module.TRUSTED_BUILTIN_SKILLS
    sd_module.TRUSTED_BUILTIN_SKILLS = frozenset({"ticket"})
    try:
        assert sd.is_builtin is False
    finally:
        sd_module.TRUSTED_BUILTIN_SKILLS = original


# ---------------------------------------------------------------------------
# Test 5: TRUSTED_BUILTIN_SKILLS constant value
# ---------------------------------------------------------------------------
def test_trusted_builtin_skills_constant_value():
    assert TRUSTED_BUILTIN_SKILLS == frozenset({"ticket", "work"})
    assert "workspace" not in TRUSTED_BUILTIN_SKILLS


# ---------------------------------------------------------------------------
# Test 6 (R4 CRIT-4, R3-HIGH-8 PROMOTED): invalid trigger
# ---------------------------------------------------------------------------
def test_skill_definition_rejects_invalid_trigger():
    """Trigger without ^^ prefix is rejected; __post_init__ uses ValueError."""
    with pytest.raises(ValueError):
        SkillDefinition(name="ticket", trigger="ticket")
    with pytest.raises(ValueError):
        SkillDefinition(name="ticket", trigger="^^TICKET")  # uppercase
    with pytest.raises(ValueError):
        SkillDefinition(name="ticket", trigger="")


# ---------------------------------------------------------------------------
# Test 7 (R1 LOW-3, R3-HIGH-8 PROMOTED): invalid name
# ---------------------------------------------------------------------------
def test_skill_definition_rejects_invalid_name():
    """Name regex requires leading lowercase."""
    with pytest.raises(ValueError):
        SkillDefinition(name="Ticket", trigger="^^ticket")
    with pytest.raises(ValueError):
        SkillDefinition(name="", trigger="^^ticket")
    with pytest.raises(ValueError):
        SkillDefinition(name="123start", trigger="^^123start")


# ---------------------------------------------------------------------------
# Test 8 (R2 LOW-2, R3-HIGH-8 PROMOTED): duplicate alias
# ---------------------------------------------------------------------------
def test_skill_definition_rejects_duplicate_alias_trigger(_held_trust_token):
    """Alias equal to name raises ValueError (R1-4 alias completeness).

    R1-4 fix: the alias regex `^[a-z][a-z0-9_-]*$` does NOT allow the
    `^^` trigger prefix, so a literal `alias == trigger` match is
    unreachable. The reachable duplicate check is `alias == name`.
    """
    with pytest.raises(ValueError, match="[Aa]lias.*duplicates name"):
        SkillDefinition(name="ticket", trigger="^^ticket", aliases=("ticket",))


# ---------------------------------------------------------------------------
# Test 9 (CRIT-S.1 / R4 CRIT-5): ModeState is frozen
# ---------------------------------------------------------------------------
def test_mode_state_is_frozen():
    """Field reassignment on ModeState raises FrozenInstanceError."""
    from dataclasses import FrozenInstanceError
    ms = ModeState(name="ticket", activated_at=0.0, ttl_seconds=300)
    with pytest.raises(FrozenInstanceError):
        ms.name = "work"  # noqa
    with pytest.raises(FrozenInstanceError):
        ms.ttl_seconds = 600  # noqa


# ---------------------------------------------------------------------------
# Test 10 (CRIT-S.1): skill_state is read-only mapping
# ---------------------------------------------------------------------------
def test_mode_state_skill_state_is_readonly_mapping():
    """MappingProxyType blocks item assignment."""
    ms = ModeState(name="ticket", activated_at=0.0, ttl_seconds=300)
    with pytest.raises(TypeError):
        ms.skill_state["x"] = 1  # noqa


# ---------------------------------------------------------------------------
# Test 13 (R3-HIGH-2 extended): is_expired actually works
# ---------------------------------------------------------------------------
def test_mode_state_keeps_is_expired():
    """is_expired() returns True for an expired ModeState."""
    ms = ModeState(name="ticket", activated_at=0.0, ttl_seconds=300)
    assert ms.is_expired() is True


# ---------------------------------------------------------------------------
# Test 14 (R1 HIGH-4, R3-HIGH-1 rewrite, R3-CRIT-2 narrow exception)
# ---------------------------------------------------------------------------
@pytest.fixture
def _isolated_registry(monkeypatch):
    """Save and restore the SkillRegistry singleton around the test.

    Phase 1b rewrite: uses module-level _default_registry instead of the
    deleted SkillRegistry._instance class attribute. Snapshots the
    mutable state (_inner, _trigger_map) and restores it after the test.

    The snapshot copies the `_inner` dict (skills) and `_trigger_map`
    dict (trigger/alias -> name) into plain dicts. The
    `MappingProxyType` snapshot is reconstructed from the saved
    `_inner` because `MappingProxyType` itself is not picklable
    (a deep copy raises `TypeError: cannot pickle 'mappingproxy'
    object`).

    test_register_skill_rejects_untrusted_builtin writes a SkillDefinition
    into the shared singleton's _inner dict (named "ticket"). Without
    restoration, subsequent tests that look up "ticket" via the singleton
    get the SkillDefinition instead of the real TicketSkill instance,
    breaking tests like test_byte_identical_prompt which call
    skill.get_tool_filter() on the resolved skill.
    """
    from proxy.skills import skill_definition as sd_module
    from proxy.skills import skill_registry as sr_module
    from types import MappingProxyType

    # Force singleton creation + builtin registration so we capture the
    # initial state.
    SkillRegistry()

    # Snapshot the mutable state. We copy the dicts and rebuild the
    # MappingProxyType on restore (MappingProxyType is not picklable
    # via deepcopy).
    saved_inner = dict(sr_module._default_registry._inner)
    saved_trigger_map = dict(sr_module._default_registry._trigger_map)
    saved_trusted = sd_module.TRUSTED_BUILTIN_SKILLS

    try:
        yield sr_module._default_registry
    finally:
        # Rebuild a fresh instance and rehydrate its state from the
        # snapshot, then install it via monkeypatch so the restoration
        # is automatically undone by pytest at session teardown.
        # ``SkillRegistry`` is ``@dataclass(frozen=True)`` so plain
        # ``restored._inner = ...`` raises ``FrozenInstanceError``; use
        # ``object.__setattr__`` to bypass the frozen guard.
        restored = sr_module._skill_registry_new__(SkillRegistry)
        object.__setattr__(restored, "_inner", saved_inner)
        object.__setattr__(restored, "_skills", MappingProxyType(saved_inner))
        object.__setattr__(restored, "_trigger_map", saved_trigger_map)
        monkeypatch.setattr(sr_module, "_default_registry", restored)
        # Also re-isolate TRUSTED_BUILTIN_SKILLS for tests that mutate it.
        monkeypatch.setattr(sd_module, "TRUSTED_BUILTIN_SKILLS", saved_trusted)


def test_register_skill_rejects_untrusted_builtin(_isolated_registry, _held_trust_token):
    """register_skill raises SkillRegistrationError for reserved names.

    R2-6 fix: a TRUSTED TrustToken must be HELD on the current thread
    to construct a builtin SkillDefinition. The token is held for the
    construction; the register_skill call uses an UNTRUSTED token to
    demonstrate the registry-level trust gate.
    """
    reg = _isolated_registry
    sd = SkillDefinition(name="ticket", trigger="^^ticket")
    assert sd.is_builtin is True
    # Register with the registry-level UNTRUSTED token: the registry's
    # R2-2 trust gate fires because "ticket" is in
    # TRUSTED_BUILTIN_SKILLS.
    with pytest.raises(SkillRegistrationError) as excinfo:
        reg.register_skill(sd, trusted_registrant=TrustToken.UNTRUSTED)
    assert "ticket" in str(excinfo.value)


# ---------------------------------------------------------------------------
# Test 15 (R3 CRIT-9): plain dict at construction is upgraded
# ---------------------------------------------------------------------------
def test_mode_state_skill_state_mappingproxy_enforced():
    """__post_init__ upgrades a plain dict to MappingProxyType."""
    ms = ModeState(
        name="ticket",
        activated_at=0.0,
        ttl_seconds=300,
        skill_state={"k": "v"},
    )
    assert isinstance(ms.skill_state, MappingProxyType)
    with pytest.raises(TypeError):
        ms.skill_state["k"] = "x"  # noqa


# ---------------------------------------------------------------------------
# Test 16 (R1 CRIT-1 fix): all 23 fields round-trip
# ---------------------------------------------------------------------------
def test_base_skill_to_skill_definition_round_trip_all_23_fields():
    """All 23 SkillDefinition fields are copied by to_skill_definition()."""
    sentinel_prompt = lambda ctx: "sentinel"
    sentinel_subagent = lambda ctx: "subagent"
    sentinel_activate = lambda ctx, ms: None
    sentinel_deactivate = lambda ctx, ms: None
    sentinel_canary = lambda ctx: {"v": 1}
    sentinel_parse = lambda text, sid: (None, "")
    sentinel_direct = lambda ctx, action, arg: None
    sentinel_post = lambda ctx, resp: None
    sentinel_deact = lambda ctx, fr: False
    sentinel_rewrite = lambda ctx: ""
    sentinel_check_action = lambda ctx, action, agent=None: None

    class Full(BaseSkill):
        name = "attack"
        trigger = "^^attack"
        aliases = ("a1",)
        sub_agent_proxy_ticket_access = ProxyAccessLevel.FULL
        sub_agent_proxy_work_access = ProxyAccessLevel.READ_ONLY
        ttl_seconds = 600
        fallback_skill = None
        fire_and_forget = True
        blocked_tools_for_subagent = frozenset({"question", "flag"})
        # Define behavior fields as raw class attributes (plain lambdas,
        # not methods, not staticmethods) so vars(type(self))['field']
        # returns the raw lambda. The to_skill_definition code uses
        # vars(type(self)).get('field', None) to bypass the descriptor
        # protocol, storing the raw lambda in the SkillDefinition.
        # This pins identity stability for the round-trip test.
        check_action = sentinel_check_action
        call_limits = {"proxy_ticket": 50}
        get_system_prompt = sentinel_prompt
        get_subagent_prompt = sentinel_subagent
        on_activate = sentinel_activate
        on_deactivate = sentinel_deactivate
        get_canary = sentinel_canary
        parse_subcommand = sentinel_parse
        handle_direct = sentinel_direct
        post_process = sentinel_post
        should_deactivate = sentinel_deact
        get_activation_rewrite = sentinel_rewrite

        def get_tool_filter(self, ctx):
            return frozenset({"read", "glob"})

        def get_sub_agent_tool_filter(self, ctx):
            return frozenset({"read", "grep"})

    skill = Full()
    sd = skill.to_skill_definition()

    # ---- Identity (3) ----
    assert sd.name == "attack"
    assert sd.trigger == "^^attack"
    assert sd.aliases == ("a1",)
    # ---- Tool filters (2) ----
    assert sd.allowed_tool_names == frozenset({"read", "glob"})
    assert sd.sub_agent_tool_filter == frozenset({"read", "grep"})
    # ---- Sub-agent permissions (2) ----
    assert sd.sub_agent_proxy_ticket_access is ProxyAccessLevel.FULL
    assert sd.sub_agent_proxy_work_access is ProxyAccessLevel.READ_ONLY
    # ---- TTL and displacement (2) ----
    assert sd.ttl_seconds == 600
    assert sd.fallback_skill is None
    # ---- Cross-cutting config (2) ----
    assert sd.fire_and_forget is True
    assert sd.blocked_tools_for_subagent == frozenset({"question", "flag"})
    # ---- Action-level restrictions (1) ----
    assert sd.check_action is sentinel_check_action
    # ---- Call limits (1) ----
    assert sd.call_limits == {"proxy_ticket": 50}
    # ---- Behavior (10) ----
    assert sd.get_system_prompt is sentinel_prompt
    assert sd.get_subagent_prompt is sentinel_subagent
    assert sd.on_activate is sentinel_activate
    assert sd.on_deactivate is sentinel_deactivate
    assert sd.get_canary is sentinel_canary
    assert sd.parse_subcommand is sentinel_parse
    assert sd.handle_direct is sentinel_direct
    assert sd.post_process is sentinel_post
    assert sd.should_deactivate is sentinel_deact
    assert sd.get_activation_rewrite is sentinel_rewrite
    # 3 + 2 + 2 + 2 + 2 + 1 + 1 + 10 = 23 fields covered.
    # Caching sanity: second call returns the SAME object.
    assert skill.to_skill_definition() is sd


# ---------------------------------------------------------------------------
# Test 17 (R3-HIGH-10 NEW): cache frozen at first call
# ---------------------------------------------------------------------------
def test_base_skill_to_skill_definition_cache_is_builtin_immutable():
    """Cached SkillDefinition is frozen at first call (R3-HIGH-10)."""
    class S(BaseSkill):
        name = "attack"
        trigger = "^^attack"

    skill = S()
    sd1 = skill.to_skill_definition()
    assert sd1.is_builtin is False
    # Mutate the class-level attribute AFTER the first call. The
    # cached definition is frozen; the second call returns the
    # SAME object and its is_builtin is still computed from the
    # ORIGINAL name ("attack"), not the mutated "ticket".
    S.name = "ticket"
    try:
        sd2 = skill.to_skill_definition()
        assert sd2 is sd1, "cache must return the same object"
        assert sd1.is_builtin is False, (
            "cached is_builtin must reflect the original name, "
            "not the post-mutation name"
        )
    finally:
        S.name = "attack"


# ===========================================================================
# Sub-agent 5 — 23 new regression tests (R1/R2/R3 followups)
# ===========================================================================


# ---------------------------------------------------------------------------
# Test 18 (R2-3): SkillDefinition object-level setattr raises
# ---------------------------------------------------------------------------
def test_skill_definition_object_setattr_raises():
    """R2-3 SkillDefinition frozenness: direct attribute assignment
    on a SkillDefinition object raises FrozenInstanceError.

    The @dataclass(frozen=True) decorator blocks field reassignment
    (`sd.name = "x"` raises). This pins the contract that callers
    cannot mutate a SkillDefinition after construction.
    """
    sd = SkillDefinition(name="attack", trigger="^^attack")
    with pytest.raises(FrozenInstanceError):
        sd.name = "x"  # noqa
    with pytest.raises(FrozenInstanceError):
        sd.trigger = "^^other"  # noqa
    with pytest.raises(FrozenInstanceError):
        sd.aliases = ("new",)  # noqa


# ---------------------------------------------------------------------------
# Test 19 (R3-2 agent.py:53 fix): ModeState object.__setattr__ in __post_init__
# ---------------------------------------------------------------------------
def test_modestate_object_setattr_in_post_init_succeeds():
    """R3-2 fix: `ModeState.__post_init__` uses `object.__setattr__` to
    upgrade a plain dict skill_state to a MappingProxyType. The upgrade
    must succeed even when the `__setattr__` thread-local guard is active.

    Pin: a fresh ModeState constructed with a plain dict produces a
    MappingProxyType view (not the original dict), proving the post-init
    mutation path works.
    """
    ms = ModeState(
        name="ticket",
        activated_at=time.time(),
        ttl_seconds=300,
        skill_state={"k": "v"},
    )
    assert isinstance(ms.skill_state, MappingProxyType)
    # And the read-only contract still holds after the upgrade.
    with pytest.raises(TypeError):
        ms.skill_state["k"] = "x"  # noqa


# ---------------------------------------------------------------------------
# Test 20 (HIGH): is_trusted_skill is a public (no leading underscore) name
# ---------------------------------------------------------------------------
def test_is_trusted_skill_public_name_contract():
    """is_trusted_skill is exported as a public name (no leading underscore).

    CRIT-S.2 / N4 CRIT fix: a previous draft left this helper as
    `_is_trusted_skill`, which surfaced as a NameError at first call
    rather than at module-import time. Pin: the function is importable
    by its plain name and is callable with a single string argument.
    """
    from proxy.skills import skill_definition as sd_module
    # The function is in the module's public namespace.
    assert "is_trusted_skill" in dir(sd_module)
    # The function's own __name__ has no leading underscore.
    assert not is_trusted_skill.__name__.startswith("_")
    # And it behaves correctly for known builtins and non-builtins.
    assert is_trusted_skill("ticket") is True
    assert is_trusted_skill("work") is True
    assert is_trusted_skill("attack") is False
    assert is_trusted_skill("myplugin") is False


# ---------------------------------------------------------------------------
# Test 21 (HIGH): SkillRequestContext.from_request_context forwards mode_state
# ---------------------------------------------------------------------------
def test_skill_request_context_from_request_forwards_mode_state():
    """R1-HIGH-2 fix: SkillRequestContext.from_request_context forwards
    `mode_state` from the pipeline RequestContext via getattr default.

    A skill that needs the active ModeState reads it from
    SkillRequestContext.mode_state without reaching back into the
    pipeline's RequestContext.
    """
    sentinel_ms = ModeState(
        name="ticket", activated_at=time.time(), ttl_seconds=300
    )

    class MockRequestContext:
        session_id = "sess-1"
        is_subagent = True
        inherited_skill = "ticket"
        subagent_ticket_mode = False
        workspace_name = "ws-1"
        client_hint = "cli"
        body = {"k": "v"}
        seeded_tasks = []
        mode_state = sentinel_ms

    ctx = MockRequestContext()
    out = SkillRequestContext.from_request_context(ctx)
    assert out.mode_state is sentinel_ms
    # Other fields are forwarded as well (sanity).
    assert out.session_id == "sess-1"
    assert out.is_subagent is True
    assert out.inherited_skill == "ticket"


# ---------------------------------------------------------------------------
# Test 22 (HIGH): SkillDefinition rejects empty trigger in __post_init__
# ---------------------------------------------------------------------------
def test_skill_definition_rejects_empty_trigger_post_init():
    """Empty trigger string is rejected.

    HIGH fix 7: the empty case (and the prefix-only "^^" case) produce
    "trigger must not be empty" (clearer than the regex error). A
    non-empty but invalid trigger (e.g., "ticket" without the "^^"
    prefix) is rejected by the trigger regex itself.
    """
    with pytest.raises(ValueError, match="trigger must not be empty"):
        SkillDefinition(name="ticket", trigger="")
    # The "^^" prefix with no name is also caught by the empty-trigger
    # guard (per source code: it cannot match the trigger regex and
    # the dedicated empty-message is clearer than the generic regex
    # error).
    with pytest.raises(ValueError, match="trigger must not be empty"):
        SkillDefinition(name="ticket", trigger="^^")
    # A non-empty trigger that doesn't match the regex is rejected
    # by the regex check itself.
    with pytest.raises(ValueError, match="Invalid trigger"):
        SkillDefinition(name="ticket", trigger="ticket")


# ---------------------------------------------------------------------------
# Test 23 (HIGH): SkillDefinition rejects empty name in __post_init__
# ---------------------------------------------------------------------------
def test_skill_definition_rejects_empty_name_post_init():
    """Empty name string is rejected by the name regex
    `^[a-z][a-z0-9_-]*$` (HIGH empty-name guard).
    """
    with pytest.raises(ValueError, match="name must not be empty"):
        SkillDefinition(name="", trigger="^^ticket")


# ---------------------------------------------------------------------------
# Test 24 (R1-4): SkillDefinition rejects alias that duplicates the name
# ---------------------------------------------------------------------------
def test_skill_definition_rejects_alias_dup_of_name():
    """R1-4 alias completeness: an alias that equals the canonical name
    is a logical duplicate (same key, same name) and must be rejected.
    """
    with pytest.raises(ValueError, match="[Aa]lias"):
        SkillDefinition(name="myname", trigger="^^trigger", aliases=("myname",))


# ---------------------------------------------------------------------------
# Test 25 (R1-4): SkillDefinition rejects aliases with invalid characters
# ---------------------------------------------------------------------------
def test_skill_definition_rejects_invalid_alias_chars():
    """R1-4 alias regex: aliases must match the same regex as names
    `^[a-z][a-z0-9_-]*$`. Uppercase letters, special characters,
    and empty strings are all rejected.
    """
    with pytest.raises(ValueError):
        SkillDefinition(name="myname", trigger="^^trigger", aliases=("BAD@CHAR",))
    with pytest.raises(ValueError):
        SkillDefinition(name="myname", trigger="^^trigger", aliases=("UPPER",))
    with pytest.raises(ValueError):
        SkillDefinition(name="myname", trigger="^^trigger", aliases=("123start",))


# ---------------------------------------------------------------------------
# Test 26 (R1-1): BaseSkill.to_skill_definition handles instance-level callable
# ---------------------------------------------------------------------------
def test_base_skill_to_skill_definition_handles_instance_level_callable():
    """R1-1 vars() deviation: `to_skill_definition` currently uses
    `vars(type(self)).get('field', None)` which does NOT see
    instance-level callables — only class-level ones. The desired
    behavior is that an instance-level override IS picked up.

    Pins the desired behavior (the deviation to fix). A subclass
    that sets `inst.get_system_prompt = some_callable` should have
    that callable stored in the resulting SkillDefinition.
    """
    def my_callable(ctx):  # noqa
        return "hi"

    class S(BaseSkill):
        name = "attack"
        trigger = "^^attack"

    inst = S()
    inst.get_system_prompt = my_callable
    sd = inst.to_skill_definition()
    # Desired (post-fix) behavior: instance-level callable is picked up.
    assert sd.get_system_prompt is my_callable, (
        "R1-1 deviation: to_skill_definition should fall back from "
        "vars(type(self)) to getattr(self, ...) so instance-level "
        "callable overrides are honored"
    )


# ---------------------------------------------------------------------------
# Test 27 (HIGH): is_expired returns False for a fresh ModeState
# ---------------------------------------------------------------------------
def test_mode_state_is_expired_returns_false_for_fresh():
    """A fresh ModeState (activated_at=now, ttl>0) is NOT expired."""
    ms = ModeState(name="ticket", activated_at=time.time(), ttl_seconds=300)
    assert ms.is_expired() is False


# ---------------------------------------------------------------------------
# Test 28 (HIGH): is_expired returns False for persistent (ttl_seconds==0)
# ---------------------------------------------------------------------------
def test_mode_state_is_expired_persistent_zero_ttl():
    """Persistent ModeState (ttl_seconds==0) is never expired, even
    when activated_at is in the distant past.
    """
    ms = ModeState(name="ticket", activated_at=0.0, ttl_seconds=0)
    assert ms.is_expired() is False


# ---------------------------------------------------------------------------
# Test 29 (HIGH): ModeState rejects negative ttl_seconds at construction
# ---------------------------------------------------------------------------
def test_mode_state_rejects_negative_ttl():
    """R2-HIGH-4 fix: negative ttl_seconds is invalid. `ttl_seconds == 0`
    remains the persistent (no-expiry) sentinel. Rejecting negative
    values surfaces caller bugs at construction time.
    """
    with pytest.raises(ValueError, match="ttl_seconds"):
        ModeState(name="ticket", activated_at=time.time(), ttl_seconds=-1)
    with pytest.raises(ValueError, match="ttl_seconds"):
        ModeState(name="ticket", activated_at=time.time(), ttl_seconds=-100)


# ---------------------------------------------------------------------------
# Test 30 (HIGH): ModeState rejects activated_at in the far future
# ---------------------------------------------------------------------------
def test_mode_state_rejects_future_activated_at():
    """R2-HIGH-5 fix: activated_at in the far future is almost always a
    caller bug (clock skew, units confusion between seconds and
    milliseconds). A 60s clock-skew tolerance is allowed; anything
    beyond is rejected.
    """
    too_far = time.time() + 1000  # ~16 minutes in the future
    with pytest.raises(ValueError, match="activated_at"):
        ModeState(name="ticket", activated_at=too_far, ttl_seconds=300)


# ---------------------------------------------------------------------------
# Test 31 (HIGH): ModeState has no refresh() method
# ---------------------------------------------------------------------------
def test_mode_state_has_no_refresh_method():
    """R1-HIGH-4 fix: `ModeState.refresh()` was REMOVED in 1a (YAGNI).
    Re-add only if a real production consumer is introduced in 1b/1c.
    """
    ms = ModeState(name="ticket", activated_at=time.time(), ttl_seconds=300)
    assert not hasattr(ms, "refresh"), (
        "ModeState.refresh() must remain removed (R1-HIGH-4 YAGNI). "
        "Re-add only with a regression test that exercises the new "
        "consumer path."
    )


# ---------------------------------------------------------------------------
# Test 32 (HIGH): BaseSkill setattr blocks _definition reassignment
# ---------------------------------------------------------------------------
def test_base_skill_setattr_blocks_definition_reassignment():
    """HIGH BaseSkill frozen: a caller cannot overwrite `BaseSkill._definition`
    (the cached SkillDefinition) once it has been set.

    HIGH fix 5/6: the first assignment to `_definition` is allowed (this
    is how `to_skill_definition()` initializes the cache); any
    subsequent assignment raises `AttributeError`. The pin here is the
    *cache immutability* — the second write must fail.
    """
    class S(BaseSkill):
        name = "attack"
        trigger = "^^attack"

    inst = S()
    # First write is allowed (this is how the cache is initialized).
    inst._definition = "first"
    # Second write is blocked — the cache is frozen at first call.
    with pytest.raises(AttributeError):
        inst._definition = "fake"  # noqa


# ---------------------------------------------------------------------------
# Test 33 (HIGH): BaseSkill setattr blocks name mutation after conversion
# ---------------------------------------------------------------------------
def test_base_skill_setattr_blocks_name_mutation_after_conversion():
    """HIGH BaseSkill frozen: a caller cannot mutate `BaseSkill.name`
    after `to_skill_definition()` has been called and cached the
    SkillDefinition.
    """
    class S(BaseSkill):
        name = "attack"
        trigger = "^^attack"

    inst = S()
    sd = inst.to_skill_definition()
    # The cached definition uses the original name.
    assert sd.name == "attack"
    with pytest.raises((AttributeError, TypeError)):
        inst.name = "work"  # noqa


# ---------------------------------------------------------------------------
# Test 34 (HIGH): _definition cache returns the same object across mutation
# ---------------------------------------------------------------------------
def test_base_skill_definition_cache_returns_same_object_across_mutation():
    """HIGH _definition cache: the cached SkillDefinition is identity-stable.
    Mutating the class-level `name` attribute AFTER the first
    `to_skill_definition()` call does NOT produce a new SkillDefinition;
    the cache returns the SAME object with the ORIGINAL name.

    Differs from T17 in that the mutation is the OTHER direction
    (name -> reserved builtin), exercising the cache invariant.
    """
    class S(BaseSkill):
        name = "attack"
        trigger = "^^attack"

    inst = S()
    sd1 = inst.to_skill_definition()
    assert sd1.name == "attack"
    # Mutate the class-level name to a reserved builtin.
    S.name = "ticket"
    try:
        sd2 = inst.to_skill_definition()
        # Same cached object, NOT a freshly-constructed one.
        assert sd2 is sd1, "cache must return the same object"
        assert sd1.name == "attack", "cached name must not be mutated"
        # is_builtin is computed from the ORIGINAL name, not the mutated one.
        assert sd1.is_builtin is False
    finally:
        S.name = "attack"


# ---------------------------------------------------------------------------
# Test 35 (R2-6): SkillDefinition rejects untrusted builtin at construction
# ---------------------------------------------------------------------------
def test_skill_definition_rejects_untrusted_builtin_at_construction():
    """R2-6 construction gate: constructing a `SkillDefinition` for a
    name in `TRUSTED_BUILTIN_SKILLS` (e.g., "ticket", "work") WITHOUT a
    held TRUSTED `TrustToken` raises `SkillRegistrationError`.

    The construction-time gate is the first line of defense — it stops
    a privileged name from ever being instantiated in a context that
    doesn't hold a token. The registry-level gate is a separate
    mechanism that inspects the call stack.
    """
    # No trust token held: construction of a builtin raises.
    with pytest.raises(SkillRegistrationError, match="Cannot construct"):
        SkillDefinition(name="ticket", trigger="^^ticket")
    with pytest.raises(SkillRegistrationError, match="Cannot construct"):
        SkillDefinition(name="work", trigger="^^work")
    # A non-builtin name does NOT need a trust token.
    sd = SkillDefinition(name="myplugin", trigger="^^myplugin")
    assert sd.is_builtin is False
    # And holding a TRUSTED token unblocks builtin construction.
    with _HeldTrustToken():
        sd2 = SkillDefinition(name="ticket", trigger="^^ticket")
    assert sd2.is_builtin is True


# ---------------------------------------------------------------------------
# Test 36 (HIGH): register_skill rejects non-SkillDefinition input
# ---------------------------------------------------------------------------
def test_register_skill_rejects_non_skill_definition_input(_isolated_registry):
    """HIGH subclass validation: `register_skill` accepts only
    `SkillDefinition` (or subclasses). Strings, dicts, and other
    types are rejected with `SkillRegistrationError` so a caller bug
    surfaces immediately rather than at attribute access time.
    """
    reg = _isolated_registry
    for bad in ["not a SkillDefinition", 42, {"name": "x"}, object()]:
        with pytest.raises(SkillRegistrationError):
            reg.register_skill(bad)  # noqa


# ---------------------------------------------------------------------------
# Test 37 (R1-3): register_skill allows SkillDefinition-over-SkillDefinition overwrite
# ---------------------------------------------------------------------------
def test_register_skill_allows_skilldefinition_overwrite():
    """R1-3 contract: overwriting an existing `SkillDefinition` with a
    new `SkillDefinition` (same name) is ALLOWED.

    After the fix to `BaseSkill.to_skill_definition()`, builtins are now
    `SkillDefinition` objects (not raw `BaseSkill`/`TicketSkill`). The
    R1-3 duplicate check at `register_skill` time (skill_registry.py
    lines 694-701) fires only when the existing entry is NOT a
    `SkillDefinition` — it intentionally allows overwriting an existing
    `SkillDefinition` (e.g., during a hot-reload of plugin config).

    This test pins the new contract: registering a different
    `SkillDefinition` under an already-registered name (where the
    existing entry is itself a `SkillDefinition`) succeeds and replaces
    the prior entry. This is the ALLOW path, not a rejection — hence
    the renamed test function (the previous name said "rejects", which
    contradicted the assertions).
    """
    from proxy.skills.skill_registry import _skill_registry_new__
    reg = _skill_registry_new__(SkillRegistry)  # bare registry, no builtins
    sd1 = SkillDefinition(name="foo", trigger="^^foo")
    reg.register_skill(sd1, trusted_registrant=TrustToken.UNTRUSTED)
    # Overwrite with a new SkillDefinition (same name) — ALLOWED (R1-3 contract)
    sd2 = SkillDefinition(name="foo", trigger="^^foo2")
    reg.register_skill(sd2, trusted_registrant=TrustToken.UNTRUSTED)
    assert reg.get("foo") is sd2


# ---------------------------------------------------------------------------
# Test 38 (R2-2): register_skill rejects untrusted via trust token
# ---------------------------------------------------------------------------
def test_register_skill_rejects_untrusted_via_trust_token(_held_trust_token):
    """R2-2 trust gate: UNTRUSTED + reserved name → rejected;
    TRUSTED + reserved name → succeeds (the gate allows trusted callers).

    After the fix to `BaseSkill.to_skill_definition()`, builtins are now
    `SkillDefinition` objects. The R1-3 duplicate check protects only
    non-SkillDefinition entries; overwriting a `SkillDefinition` is
    allowed. The R2-2 trust gate (UNTRUSTED token + reserved name) is
    a separate, independent check that still fires.

    The construction-time R2-6 gate for `name="ticket"` is passed via
    the `_held_trust_token` fixture, which holds a TRUSTED `TrustToken`
    on the current thread for the lifetime of the test.

    CRIT-2 R2 (Wave 2.1) — fix vacuous TRUSTED assertion: the previous
    version of this test used `name="newskill"` (non-reserved), so the
    trust gate never fired and the assertion was vacuous. It also used
    `TrustToken(_verified=True)` directly, which is now blocked by the
    constructor guard added in Wave 1.2. The fix uses `name="ticket"`
    (reserved) and `TrustToken.for_trusted_caller()`, exercising the
    actual gate path: TRUSTED callers are allowed to register a
    reserved name.
    """
    from proxy.skills.skill_registry import _skill_registry_new__
    reg = _skill_registry_new__(SkillRegistry)  # bare registry, no builtins
    # UNTRUSTED + reserved name → rejected (R2-2 trust gate fires)
    sd = SkillDefinition(name="ticket", trigger="^^ticket")
    with pytest.raises(SkillRegistrationError):
        reg.register_skill(sd, trusted_registrant=TrustToken.UNTRUSTED)
    # TRUSTED + reserved name → succeeds (trust gate allows trusted callers).
    # The fixture already holds a TRUSTED token on the thread, so both
    # the construction-time R2-6 gate and the register-time R2-2 gate pass.
    sd2 = SkillDefinition(name="ticket", trigger="^^ticket2")
    reg.register_skill(sd2, trusted_registrant=TrustToken.for_trusted_caller())
    assert reg.get("ticket") is sd2


# ---------------------------------------------------------------------------
# Test 38b (Wave 2.1): TrustToken.for_trusted_caller() gate
# ---------------------------------------------------------------------------
def test_trust_token_for_trusted_caller_gate():
    """Wave 2.1 (Fix 3): pin the `TrustToken` security boundary.

    - `TrustToken.for_trusted_caller()` succeeds from a module listed in
      `TRUSTED_TRUST_CALLERS` (this test module, `tests.test_skill_definition`,
      is in that set per Wave 1.2). The returned token has
      `is_trusted is True`.
    - `TrustToken(_verified=True)` direct mint is blocked with
      `PermissionError` from any module — the constructor guard
      (Wave 1.2) only allows the class itself to mint TRUSTED tokens.

    Together these guarantee that the only public path to a TRUSTED
    token is `for_trusted_caller()`, which itself enforces the
    `TRUSTED_TRUST_CALLERS` allowlist via `inspect.stack()`. A test
    cannot bypass the trust boundary by importing the private
    constructor.
    """
    # From a trusted module (this test module IS in TRUSTED_TRUST_CALLERS).
    token = TrustToken.for_trusted_caller()
    assert token.is_trusted is True

    # Direct mint from any module is blocked by the constructor guard.
    with pytest.raises(PermissionError) as exc_info:
        TrustToken(_verified=True)
    msg = str(exc_info.value)
    assert "_verified" in msg or "for_trusted_caller" in msg


# ---------------------------------------------------------------------------
# Test 39 (R1-2 / R2-1): register_legacy rejects reserved name via trust gate
# ---------------------------------------------------------------------------
def test_register_legacy_rejects_reserved_name_via_trust_gate(
    _isolated_registry,
):
    """R1-2 / R2-1 trust gate: the legacy `register(skill: Skill)` method
    blocks re-registration of any name in `TRUSTED_BUILTIN_SKILLS` from
    an untrusted caller. The check uses `inspect.stack()` to verify the
    caller's module is in `TRUSTED_TRUST_CALLERS`; this test runs in
    a pytest module that is NOT in that set, so the gate fires.
    """
    from proxy.skills.skill_registry import SkillRegistry

    class FakeBuiltin:
        name = "ticket"
        trigger = "^^ticket"
        aliases: tuple = ()

    reg = _isolated_registry
    with pytest.raises(SkillRegistrationError):
        reg.register(FakeBuiltin())


# ---------------------------------------------------------------------------
# Test 40 (HIGH): register_legacy raises SkillRegistrationError
# ---------------------------------------------------------------------------
def test_register_legacy_raises_skill_registration_error(_isolated_registry):
    """HIGH: the legacy `register(skill: Skill)` raises
    `SkillRegistrationError` (a `PermissionError` subclass) for both
    trust-gate denials and duplicate-name denials, so callers can
    catch a single exception type.
    """
    from proxy.skills.skill_registry import SkillRegistry

    class FakeSkill:
        name = "unique_skill_for_register_legacy"
        trigger = "^^unique"
        aliases: tuple = ()

    reg = _isolated_registry
    # Successful registration (untrusted, name not reserved).
    reg.register(FakeSkill())
    # Re-registering the same name raises SkillRegistrationError.
    with pytest.raises(SkillRegistrationError):
        reg.register(FakeSkill())


# ---------------------------------------------------------------------------
# Test 41 (CRIT R3-5): SkillRegistrationError is a PermissionError subclass
# ---------------------------------------------------------------------------
def test_skill_registration_error_is_permission_error_subclass():
    """CRIT R3-5: `SkillRegistrationError` inherits from `PermissionError`
    so existing `except PermissionError` blocks continue to work, AND
    narrow catches like `except SkillRegistrationError` can target
    registration-only denials.
    """
    assert issubclass(SkillRegistrationError, PermissionError)
    # And it can be raised/caught as a PermissionError.
    with pytest.raises(PermissionError):
        raise SkillRegistrationError("narrow subclass test")


# ---------------------------------------------------------------------------
# Test 42 (HIGH): reset_singleton requires the env var
# ---------------------------------------------------------------------------
def test_reset_default_registry_is_unconditional(monkeypatch):
    """1b: `reset_default_registry` is unconditional — no env var required.

    R2-3 (HIGH) fix: the old `SkillRegistry.reset_singleton()` was gated by
    `CHORI_SKILL_REGISTRY_RESET=1`. The new `reset_default_registry()` has
    no gate — tests may call it freely.
    """
    # Ensure the env var is not set (should not matter).
    monkeypatch.delenv("CHORI_SKILL_REGISTRY_RESET", raising=False)
    reset_default_registry()  # must not raise
    # Calling twice is also fine.
    reset_default_registry()  # must not raise
