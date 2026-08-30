"""Tests for TrustToken, _held_trust_token, and SkillDefinition trust gate.

R5 CRIT-1: TrustToken construction/mutation/forgery paths had ZERO test
coverage. These tests pin the security boundary so future refactors cannot
silently weaken it (e.g., by removing the inspect.stack() check in the
constructor or the __setattr__ guard on _verified).

The trust gate has THREE independent components, each tested here:
  1. TrustToken.__init__ blocks external callers (only proxy.skills.skill_definition
     may mint tokens directly).
  2. TrustToken._verified is immutable post-construction (no attribute
     mutation can forge a TRUSTED token from UNTRUSTED).
  3. TrustToken.for_trusted_caller() enforces the TRUSTED_TRUST_CALLERS
     allowlist (only trusted modules can mint TRUSTED tokens at all).

Plus integration tests for the construction-time gate (R2-6 HIGH) and
the registry-level gate (R1 HIGH-4).
"""
import pytest
from unittest.mock import patch
from proxy.skills.skill_definition import (
    TrustToken,
    SkillDefinition,
    _held_trust_token,
    _trust_token_local,
    is_trusted_skill,
    TRUSTED_BUILTIN_SKILLS,
    SkillRegistrationError,
)
from proxy.skills.skill_registry import (
    SkillRegistry,
    _skill_registry_new__,
)


# ---------------------------------------------------------------------------
# TrustToken constructor gate (HIGH-1 R1 fix)
# ---------------------------------------------------------------------------

def test_trust_token_constructor_blocks_external_caller():
    """TrustToken(_verified=True) from outside proxy.skills.skill_definition raises.

    The constructor inspects inspect.stack()[1] and rejects any caller whose
    module is not 'proxy.skills.skill_definition'. This is the only line of
    defense against a user-defined plugin calling `TrustToken(_verified=True)`
    to forge a TRUSTED token and bypass the registry gate.
    """
    with pytest.raises(PermissionError):
        TrustToken(_verified=True)


def test_trust_token_constructor_blocks_unverified_external_caller():
    """TrustToken(_verified=False) from outside proxy.skills.skill_definition raises.

    Same gate as the _verified=True case — the constructor is fully private,
    not just for TRUSTED tokens.
    """
    with pytest.raises(PermissionError):
        TrustToken(_verified=False)


# ---------------------------------------------------------------------------
# TrustToken._verified immutability (HIGH-1 R1 fix)
# ---------------------------------------------------------------------------

def test_trust_token_verified_is_immutable_post_construction():
    """token._verified = True after construction raises AttributeError.

    Without this guard, an attacker could do:
        token = TrustToken.UNTRUSTED
        token._verified = True   # would forge a TRUSTED token
    The __setattr__ override blocks ALL writes to _verified post-construction.
    """
    token = TrustToken.UNTRUSTED
    assert token.is_trusted is False
    with pytest.raises(AttributeError):
        token._verified = True


def test_trust_token_verified_immutable_on_untrusted_after_untrusted_read():
    """Second mutation attempt on the same UNTRUSTED token also raises.

    Pins that the __setattr__ guard is per-instance, not per-call (no caching
    that would let the second write succeed).
    """
    token = TrustToken.UNTRUSTED
    with pytest.raises(AttributeError):
        token._verified = True
    with pytest.raises(AttributeError):
        token._verified = False


# ---------------------------------------------------------------------------
# TrustToken.for_trusted_caller() allowlist (CRIT R2-1)
# ---------------------------------------------------------------------------

def test_trust_token_for_trusted_caller_rejects_untrusted_module():
    """for_trusted_caller() raises when the caller's module is not in the allowlist.

    tests.proxy.test_trust_gate IS in TRUSTED_TRUST_CALLERS (so other tests
    can mint TRUSTED tokens), but the rejection path is verified here by
    patching the allowlist down to an empty set. This simulates a caller
    module that is not on the allowlist and pins the PermissionError path.

    An earlier draft popped `proxy.skills.skill_definition` from
    sys.modules to attempt the same effect, but that approach caused test
    isolation issues (subsequent tests imported the reloaded module which
    had a fresh _trust_token_local decoupled from skill_registry's
    cached reference). The mock.patch approach is robust.
    """
    from unittest.mock import patch
    from proxy.skills import skill_registry as _sr
    # Clear any leaked TRUSTED token so the test starts clean.
    if hasattr(_trust_token_local, "token"):
        del _trust_token_local.token
    # Patch the allowlist to an empty set so this module is no longer trusted.
    with patch.object(_sr, "TRUSTED_TRUST_CALLERS", frozenset()):
        with pytest.raises(PermissionError):
            TrustToken.for_trusted_caller()


def test_trust_token_for_trusted_caller_succeeds_for_trusted_module():
    """for_trusted_caller() succeeds from a module in TRUSTED_TRUST_CALLERS.

    tests.proxy.test_trust_gate is added to TRUSTED_TRUST_CALLERS by the
    Wave 2.3 fixes (so this test can mint TRUSTED tokens). The
    add-to-allowlist change is security-relevant and is pinned here.
    """
    token = TrustToken.for_trusted_caller()
    assert token.is_trusted is True


# ---------------------------------------------------------------------------
# TrustToken.UNTRUSTED sentinel (basic invariants)
# ---------------------------------------------------------------------------

def test_trust_token_untrusted_is_trusted_false():
    """UNTRUSTED.is_trusted is False.

    The UNTRUSTED sentinel is the only public way to obtain a token from
    outside proxy.skills.skill_definition. It must always report
    is_trusted=False.
    """
    assert TrustToken.UNTRUSTED.is_trusted is False


def test_trust_token_untrusted_is_singleton():
    """TrustToken.UNTRUSTED is a single module-level instance.

    Multiple accesses return the same object — identity equality holds.
    """
    assert TrustToken.UNTRUSTED is TrustToken.UNTRUSTED


def test_trust_token_repr_does_not_leak_internal_state():
    """repr() does not expose whether the token is forged or genuine.

    Defensive: the repr includes is_trusted=<bool>, which is fine (the
    bool is what the rest of the code consumes), but it MUST NOT include
    a flag that distinguishes 'constructed internally' from 'forged'.
    """
    r = repr(TrustToken.UNTRUSTED)
    assert "TrustToken" in r
    # Must contain is_trusted=False (informational only).
    assert "is_trusted=False" in r
    # Must NOT contain any 'forged' or 'caller_module' markers.
    assert "forged" not in r.lower()
    assert "caller_module" not in r


# ---------------------------------------------------------------------------
# SkillDefinition construction-time trust gate (HIGH R2-6)
# ---------------------------------------------------------------------------

def test_skill_definition_trusted_name_requires_held_token():
    """Constructing SkillDefinition(name='ticket') without a held TRUSTED token raises.

    The __post_init__ gate reads `_trust_token_local.token` and raises
    SkillRegistrationError if the new SkillDefinition's name is in
    TRUSTED_BUILTIN_SKILLS and no TRUSTED TrustToken is held on the
    current thread.
    """
    # Ensure no leaked token from a prior test.
    if hasattr(_trust_token_local, "token"):
        del _trust_token_local.token
    with pytest.raises(SkillRegistrationError):
        SkillDefinition(name="ticket", trigger="^^ticket")


def test_skill_definition_untrusted_name_does_not_require_token():
    """Constructing SkillDefinition(name='myplugin') without a held token works.

    The construction-time gate is name-based: only names in
    TRUSTED_BUILTIN_SKILLS ('ticket', 'work') are gated. User-defined
    skill names do not need a held token.
    """
    if hasattr(_trust_token_local, "token"):
        del _trust_token_local.token
    sd = SkillDefinition(name="myplugin", trigger="^^myplugin")
    assert sd.name == "myplugin"
    assert sd.is_builtin is False


def test_skill_definition_trusted_name_succeeds_with_held_token():
    """With _held_trust_token(TRUSTED), SkillDefinition(name='ticket') succeeds.

    Pins the happy path: when the construction-time gate is satisfied,
    the SkillDefinition is created normally.
    """
    if hasattr(_trust_token_local, "token"):
        del _trust_token_local.token
    with _held_trust_token(TrustToken.for_trusted_caller()):
        sd = SkillDefinition(name="ticket", trigger="^^ticket")
    assert sd.name == "ticket"
    assert sd.is_builtin is True


# ---------------------------------------------------------------------------
# SkillRegistry.register_skill trust gate (R1 HIGH-4)
# ---------------------------------------------------------------------------

def test_register_skill_blocks_untrusted_registrant_for_trusted_name():
    """register_skill with name='ticket' and UNTRUSTED token raises.

    Workflow: construct the SkillDefinition with a HELD TRUSTED token
    (so the construction gate passes), then try to register with
    trusted_registrant=TrustToken.UNTRUSTED. The registry must reject
    the registration even though the SkillDefinition is well-formed.
    """
    reg = _skill_registry_new__(SkillRegistry)
    # Hold a TRUSTED token on the thread so the construction-time gate passes.
    with _held_trust_token(TrustToken.for_trusted_caller()):
        sd = SkillDefinition(name="ticket", trigger="^^ticket")
    # Now attempt registration with UNTRUSTED — must raise.
    with pytest.raises(SkillRegistrationError):
        reg.register_skill(sd, trusted_registrant=TrustToken.UNTRUSTED)


def test_register_skill_succeeds_for_trusted_name_with_trusted_registrant():
    """register_skill with name='ticket' and TRUSTED token succeeds.

    Happy-path: a TRUSTED token bypasses the registry gate, and the
    SkillDefinition lands in _inner.
    """
    reg = _skill_registry_new__(SkillRegistry)
    with _held_trust_token(TrustToken.for_trusted_caller()):
        sd = SkillDefinition(name="ticket", trigger="^^ticket")
    reg.register_skill(sd, trusted_registrant=TrustToken.for_trusted_caller())
    assert "ticket" in reg
    assert reg.get("ticket") is sd


def test_register_skill_allows_untrusted_name_with_untrusted_registrant():
    """register_skill with name='myplugin' and UNTRUSTED token succeeds.

    The registry gate is name-based: only TRUSTED_BUILTIN_SKILLS names
    are gated. A user-defined name like 'myplugin' is allowed regardless
    of the token.
    """
    reg = _skill_registry_new__(SkillRegistry)
    sd = SkillDefinition(name="myplugin", trigger="^^myplugin")
    reg.register_skill(sd, trusted_registrant=TrustToken.UNTRUSTED)
    assert "myplugin" in reg


# ---------------------------------------------------------------------------
# is_trusted_skill / TRUSTED_BUILTIN_SKILLS public surface
# ---------------------------------------------------------------------------

def test_is_trusted_skill_returns_true_for_builtins():
    """is_trusted_skill('ticket') and is_trusted_skill('work') are True."""
    assert is_trusted_skill("ticket") is True
    assert is_trusted_skill("work") is True


def test_is_trusted_skill_returns_false_for_user_names():
    """is_trusted_skill returns False for non-builtin names."""
    assert is_trusted_skill("myplugin") is False
    assert is_trusted_skill("attack") is False
    assert is_trusted_skill("") is False


def test_trusted_builtin_skills_is_frozenset():
    """TRUSTED_BUILTIN_SKILLS is a frozenset (immutable)."""
    assert isinstance(TRUSTED_BUILTIN_SKILLS, frozenset)
    assert TRUSTED_BUILTIN_SKILLS == frozenset({"ticket", "work"})


def test_skill_definition_is_builtin_uses_helper():
    """sd.is_builtin agrees with is_trusted_skill(sd.name) for every test name.

    Pins the contract that the read-only `is_builtin` property and the
    public `is_trusted_skill` helper never drift apart.
    """
    with _held_trust_token(TrustToken.for_trusted_caller()):
        for name in ("ticket", "work", "myplugin", "attack"):
            sd = SkillDefinition(name=name, trigger=f"^^{name}")
            assert sd.is_builtin is is_trusted_skill(name)


# ---------------------------------------------------------------------------
# TrustToken synthetic-frame defense (R3-HIGH-1 fix)
# ---------------------------------------------------------------------------

def test_trust_token_rejects_synthetic_frame_via_exec():
    """Direct synthetic-frame forgery via exec() with spoofed __name__ is rejected.

    The strongest test of the R3-HIGH-1 fix: an attacker who constructs
    a globals dict with `__name__ = 'proxy.skills.skill_definition'` and
    exec's a string that calls `TrustToken(_verified=True)` STILL cannot
    mint a TRUSTED token, because the synthetic frame's `f_code.co_filename`
    is the test file, NOT the real skill_definition module file.
    """
    # Build a synthetic globals dict that spoofs __name__ and provides
    # the TrustToken symbol (via the real module). Then exec a string
    # that attempts TrustToken(_verified=True) inside this forged frame.
    forged_globals = {
        '__name__': 'proxy.skills.skill_definition',
        '__file__': '/some/fake/path.py',  # irrelevant — attacker controls this
    }
    forged_globals['TrustToken'] = TrustToken  # import through real module
    code = "TrustToken(_verified=True)"
    with pytest.raises(PermissionError):
        exec(code, forged_globals)


def test_object_setattr_bypass_is_documented():
    """documented: object.__setattr__ can bypass TrustToken's __setattr__ guard.

    This is a known defense-in-depth limitation. Python's __setattr__ override
    does NOT block object.__setattr__ from modifying attributes. The real
    defense is the constructor gate which checks f_globals['__name__'].
    """
    from proxy.skills.skill_definition import TrustToken
    t = TrustToken.UNTRUSTED
    # Snapshot the original state so we can restore after the bypass demo
    saved_verified = object.__getattribute__(t, "_verified")
    try:
        object.__setattr__(t, "_verified", True)
        assert t.is_trusted is True
    finally:
        # CRITICAL: restore the singleton's state so subsequent tests
        # that rely on TrustToken.UNTRUSTED.is_trusted == False don't
        # get false TRUSTED tokens via the bypass.
        object.__setattr__(t, "_verified", saved_verified)


# ---------------------------------------------------------------------------
# Confused-deputy defense (R3-HIGH-2 fix)
# ---------------------------------------------------------------------------
# A wrapper function in a TRUSTED PRODUCTION module (e.g.
# `proxy.skills.skill_registry`) has the same `__name__` AND the same
# `co_filename` as the legit entry point. The R3-HIGH-1 synthetic-frame
# check does NOT catch it. The R3-HIGH-2 fix adds a `_TRUSTED_CALL_SITES`
# allowlist on `co_name` so a wrapper with a different function name is
# rejected.
#
# These tests use REAL stack inspection (no mocking of inspect.stack),
# because the vulnerability is in the stack logic itself. The attack is
# simulated by `exec()`-ing a function definition into a real trusted
# module's `__dict__`, with the real module's `__file__` as `co_filename`
# — exactly what an attacker who can plant a wrapper in the trusted
# module would produce.

def test_for_trusted_caller_rejects_wrapper_in_trusted_module():
    """A wrapper fn in a trusted module (NOT in _TRUSTED_CALL_SITES) is rejected.

    Confused-deputy scenario: an attacker plants a helper function
    `evil_wrapper` inside `proxy.skills.skill_registry` (the trusted
    production module). An untrusted caller invokes
    `skill_registry.evil_wrapper()`, which calls
    `TrustToken.for_trusted_caller()` on the attacker's behalf.

    Without R3-HIGH-2: the wrapper's frame has `__name__ ==
    'proxy.skills.skill_registry'` (trusted ✓) and `co_filename ==
    skill_registry.py` (passes the synthetic-frame check ✓), so BOTH
    existing gates pass and a TRUSTED token is minted for the untrusted
    caller.

    With R3-HIGH-2: the wrapper's `co_name == 'evil_wrapper'` is NOT in
    `_TRUSTED_CALL_SITES`, so the call is rejected with PermissionError.
    """
    import sys
    from proxy.skills import skill_definition as _sd
    from proxy.skills import skill_registry as _sr

    mod = sys.modules["proxy.skills.skill_registry"]
    real_file = mod.__file__
    # Compile a wrapper function whose frame will report:
    #   __name__   = 'proxy.skills.skill_registry' (from mod.__dict__)
    #   co_filename = real_file (we pass it to compile())
    #   co_name    = 'evil_wrapper' (NOT in _TRUSTED_CALL_SITES)
    code = compile(
        "def evil_wrapper():\n"
        "    return TrustToken.for_trusted_caller()\n",
        real_file,
        "exec",
    )
    # Execute into the real module's namespace so the function's
    # __globals__ is the module's __dict__ (i.e. __name__ is trusted).
    exec(code, mod.__dict__)
    injected = mod.__dict__["evil_wrapper"]
    try:
        assert injected.__globals__["__name__"] == "proxy.skills.skill_registry"
        assert injected.__code__.co_filename == real_file
        assert injected.__code__.co_name == "evil_wrapper"
        with pytest.raises(PermissionError) as exc_info:
            injected()
        msg = str(exc_info.value)
        assert "confused-deputy" in msg
        assert "evil_wrapper" in msg
    finally:
        # Clean up the injected function so the production module is
        # not polluted for subsequent tests.
        mod.__dict__.pop("evil_wrapper", None)


def test_for_trusted_caller_accepts_known_safe_entry_point_name():
    """A production trusted module fn whose co_name IS in _TRUSTED_CALL_SITES succeeds.

    Pins the happy path for the production caller `populate_builtins`
    in `proxy.skills.skill_registry`. We simulate the production call
    site by exec()-ing a function named `populate_builtins` into the
    real module's __dict__ (same trick as the wrapper test, but with
    the allowlisted function name). The call MUST succeed — this is
    the legit entry point the hardening is designed to permit.
    """
    import sys
    from proxy.skills import skill_definition as _sd

    mod = sys.modules["proxy.skills.skill_registry"]
    real_file = mod.__file__
    code = compile(
        "def populate_builtins():\n"
        "    return TrustToken.for_trusted_caller()\n",
        real_file,
        "exec",
    )
    # Save and restore the real populate_builtins (the bound method on
    # the class is distinct from this module-level function, but be
    # safe and restore anything we shadow).
    saved = mod.__dict__.get("populate_builtins")
    try:
        exec(code, mod.__dict__)
        injected = mod.__dict__["populate_builtins"]
        assert injected.__globals__["__name__"] == "proxy.skills.skill_registry"
        assert injected.__code__.co_name == "populate_builtins"
        token = injected()
        assert token.is_trusted is True
    finally:
        if saved is None:
            mod.__dict__.pop("populate_builtins", None)
        else:
            mod.__dict__["populate_builtins"] = saved


def test_populate_builtins_still_mints_tokens_after_hardening():
    """Integration: the real populate_builtins() still mints TRUSTED tokens.

    The real production call site (`SkillRegistry.populate_builtins`)
    has `co_name == 'populate_builtins'`, which is in
    `_TRUSTED_CALL_SITES`. After R3-HIGH-2, this path must still work
    end-to-end (both `for_trusted_caller()` calls inside it succeed).
    """
    from proxy.skills.skill_registry import (
        SkillRegistry,
        _skill_registry_new__,
    )
    reg = _skill_registry_new__(SkillRegistry)
    count = reg.populate_builtins(trusted=True)
    # Ticket + work => up to 2 new registrations (may be 0 if a prior
    # test left the registry populated, but populate_builtins is
    # idempotent on the singleton; we use a FRESH registry here so it
    # should be 2).
    assert count == 2
    assert "ticket" in reg
    assert "work" in reg


def test_trusted_call_sites_is_frozenset_and_contains_populate_builtins():
    """_TRUSTED_CALL_SITES is a frozenset and lists the known entry point.

    Adding a new production call site is a security-relevant change that
    MUST add the function name here. This test pins the current set so a
    refactor that drops the allowlist is caught.
    """
    from proxy.skills.skill_definition import _TRUSTED_CALL_SITES
    assert isinstance(_TRUSTED_CALL_SITES, frozenset)
    assert "populate_builtins" in _TRUSTED_CALL_SITES


def test_is_exempt_from_callsite_check_exempts_test_modules():
    """Test modules (tests.*) are exempt from the call-site check.

    Production modules (proxy.*, __main__) are subject to it. This
    asymmetry is intentional: tests need to mint tokens from arbitrary
    test function names; production callers must use a known-safe
    entry point.
    """
    from proxy.skills.skill_definition import _is_exempt_from_callsite_check
    assert _is_exempt_from_callsite_check("tests.proxy.test_trust_gate") is True
    assert _is_exempt_from_callsite_check("tests.test_skill_definition") is True
    assert _is_exempt_from_callsite_check("proxy.skills.skill_registry") is False
    assert _is_exempt_from_callsite_check("__main__") is False
    assert _is_exempt_from_callsite_check("proxy.agent.agent_registry") is False
