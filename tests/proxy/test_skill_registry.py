# tests/proxy/test_skill_registry.py
# Phase 1b — SkillRegistry full surface area tests

import os, pytest, concurrent.futures
from proxy.skills.skill_registry import (
    SkillRegistry, get_default_registry, reset_default_registry, _default_registry,
)
from proxy.skills.skill_definition import (
    BaseSkill,
    SkillDefinition,
    SkillRegistrationError,
    TrustToken,
    _held_trust_token,
)


class DummySkill(BaseSkill):
    """Minimal skill for testing (Phase 1d: NEW BaseSkill path)."""
    name: str = "test"
    trigger: str = "^^test"
    aliases: tuple = ()

    def __init__(self, name: str = "test", trigger: str = "^^test"):
        # Use object.__setattr__ to bypass NEW BaseSkill's __setattr__ override.
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "trigger", trigger)


@pytest.fixture(autouse=True)
def fresh_registry():
    """Each test gets a fresh registry."""
    reset_default_registry()
    yield
    reset_default_registry()


@pytest.fixture(autouse=True)
def isolate_custom_skills_dir(tmp_path, monkeypatch):
    """Disable custom-skill loading so tests see only the two builtins.

    Under Wave 1's strict-schema loader (spec §5), setting PROXY_SKILLS_DIR
    to a non-existent path is a fatal startup error. We instead point
    PROXY_SKILLS_DIR at a real but empty tmp_path subdirectory so the
    loader runs (no JSON files match) and registers only the two
    builtins.
    """
    empty_dir = tmp_path / "empty_skills"
    empty_dir.mkdir()
    monkeypatch.setenv("PROXY_SKILLS_DIR", str(empty_dir))


# === 1a-deferred tests ===

def test_user_registration_rejects_reserved_names():
    """register_skill with name='ticket' from untrusted caller raises."""
    reg = SkillRegistry()
    # SkillDefinition.__post_init__ rejects trusted names without a held token,
    # so hold a TRUSTED token during construction.  The register_skill call
    # still receives UNTRUSTED and must raise.
    trusted_token = TrustToken.for_trusted_caller()
    print('DEBUG: TrustToken.for_trusted_caller succeeded')
    with _held_trust_token(trusted_token):
        sd = SkillDefinition(name="ticket", trigger="^^ticket", aliases=())
    with pytest.raises(SkillRegistrationError):
        reg.register_skill(sd, trusted_registrant=TrustToken.UNTRUSTED)


def test_skill_registry_inner_dict_isolation():
    """registry._skills is read-only MappingProxyType."""
    reg = SkillRegistry()
    sd = SkillDefinition(name="new", trigger="^^new", aliases=())
    with pytest.raises(TypeError):
        reg._skills["new"] = sd


# === New surface: get / get_skill ===

def test_get_returns_skill_definition():
    reg = SkillRegistry()
    reg.populate_builtins()
    sd = reg.get("ticket")
    assert sd is not None
    assert sd.name == "ticket"


def test_get_skill_aliases_get():
    reg = SkillRegistry()
    reg.populate_builtins()
    assert reg.get_skill("ticket") == reg.get("ticket")


def test_get_unknown_returns_none():
    reg = SkillRegistry()
    assert reg.get("nonexistent") is None


# === get_by_trigger ===

def test_get_by_trigger_finds_builtin():
    reg = SkillRegistry()
    reg.populate_builtins()
    sd = reg.get_by_trigger("^^ticket")
    assert sd is not None
    assert sd.name == "ticket"


# === list_skills / all ===

def test_list_skills_returns_all_builtins():
    reg = SkillRegistry()
    reg.populate_builtins()
    skills = reg.list_skills()
    assert len(skills) == 2
    names = {s.name for s in skills}
    assert names == {"ticket", "work"}


def test_all_returns_same_as_list_skills():
    reg = SkillRegistry()
    reg.populate_builtins()
    assert reg.all() == reg.list_skills()
    # Both should return the same list objects (not copies that diverge)
    assert reg.all() is reg.list_skills() or reg.all() == reg.list_skills()


# === __contains__ ===

def test_contains_returns_true_for_builtin():
    reg = SkillRegistry()
    reg.populate_builtins()
    assert "ticket" in reg


def test_contains_returns_false_for_unknown():
    reg = SkillRegistry()
    assert "nonexistent" not in reg


# === populate_builtins ===

def test_populate_builtins_is_idempotent():
    reg = SkillRegistry()
    first = reg.populate_builtins()
    second = reg.populate_builtins()
    assert second == 0  # nothing new registered


def test_populate_builtins_returns_two_on_fresh_registry():
    """First call returns count of new registrations (ticket + work, not workspace per R2-Issue 3).

    NOTE: SkillRegistry() goes through the __new__ shim which auto-calls
    get_default_registry() → populate_builtins(). So a plain SkillRegistry()
    is always already populated and a second populate_builtins() returns 0.
    To test the "first call returns 2" scenario we must construct a bare
    instance via _skill_registry_new__ to bypass the shim.
    """
    from proxy.skills.skill_registry import _skill_registry_new__
    fresh = _skill_registry_new__(SkillRegistry)
    count = fresh.populate_builtins()
    assert count == 2  # ticket + work
    # Verify workspace is NOT included
    assert "workspace" not in fresh


def test_workspace_not_registered_after_populate_builtins():
    reg = SkillRegistry()
    reg.populate_builtins()
    assert "workspace" not in reg


# === Module-level functions ===

def test_get_default_registry_returns_same_instance():
    assert get_default_registry() is get_default_registry()


def test_reset_default_registry_clears_singleton():
    r1 = get_default_registry()
    reset_default_registry()
    r2 = get_default_registry()
    assert r1 is not r2


# === Thread-safety ===

def test_concurrent_get_default_registry_constructs_once():
    """N=16 threads racing on get_default_registry() — exactly one instance created.

    NOTE: We must access _default_registry via the module (sr_module._default_registry)
    rather than the `from ... import _default_registry` binding in this test file.
    The local binding is captured at import time and does NOT reflect later
    reassignments to the module global (Python name-binding semantics for
    immutable reassignments).
    """
    import proxy.skills.skill_registry as sr_module
    sr_module.reset_default_registry()
    # Verify we start with no singleton
    assert sr_module._default_registry is None

    def get_reg():
        r = sr_module.get_default_registry()
        return id(r)

    with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:
        futures = [executor.submit(get_reg) for _ in range(16)]
        ids = [f.result() for f in concurrent.futures.as_completed(futures)]

    # All 16 threads see the same instance
    assert len(set(ids)) == 1, f"Expected 1 instance, got {len(set(ids))}"
    # The singleton is now set
    assert sr_module._default_registry is not None


def test_concurrent_get_default_registry_returns_one_complete_snapshot(monkeypatch, tmp_path):
    """Wave 2: 16 threads call get_default_registry() simultaneously, gated
    by a threading.Barrier (per tests/concurrency_harness.py convention).

    Asserts the atomic-publication contract: every caller receives the SAME
    object identity AND the SAME immutable snapshot. A partial-publication
    bug (some skills registered, others not) would manifest as diverging
    snapshots across threads even when object identity matches.
    """
    import json
    import threading

    # Set a clean env so the singleton constructs successfully with a
    # known skills directory containing one custom skill.
    skills_dir = tmp_path / "concurrent_skills"
    skills_dir.mkdir()
    (skills_dir / "concurrent_a.json").write_text(json.dumps({
        "name": "concurrent_a",
        "trigger": "^^concurrent_a",
        "description": "Concurrent test skill A.",
        "prompt": "Prompt A.",
        "tool_whitelist": ["read"],
    }))
    monkeypatch.setenv("PROXY_SKILLS_DIR", str(skills_dir))
    reset_default_registry()

    barrier = threading.Barrier(parties=16, timeout=10)
    results: list = []
    errors: list = []

    def worker() -> None:
        try:
            barrier.wait()
            reg = get_default_registry()
            # Snapshot = (registry id, sorted skill-name tuple). Sort by
            # name attribute since SkillDefinition is frozen and does not
            # define __lt__ — sorted(list_skills()) would raise TypeError.
            names = sorted(sd.name for sd in reg.list_skills())
            results.append((id(reg), tuple(names)))
        except Exception as exc:
            errors.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(16)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=10)

    assert not errors, f"Worker threads raised: {errors}"
    assert len(results) == 16

    # All threads got the same registry object identity.
    first_id = results[0][0]
    assert all(rid == first_id for rid, _ in results), (
        f"Threads got different registry objects: "
        f"{set(rid for rid, _ in results)}"
    )
    # All threads saw the same immutable snapshot. Builtins (ticket, work)
    # plus the one custom skill we placed under PROXY_SKILLS_DIR.
    first_snapshot = results[0][1]
    assert all(snap == first_snapshot for _, snap in results), (
        f"Threads saw different snapshots. First: {first_snapshot}. "
        f"Other divergent: "
        f"{[snap for _, snap in results if snap != first_snapshot]}"
    )


# === register trust gate ===

def test_register_method_rejects_reserved_name():
    """register() with name='ticket' from outside TRUSTED_TRUST_CALLERS raises.

    Wave 1.2 adaptation: this test module is now in TRUSTED_TRUST_CALLERS
    (so it can mint TRUSTED tokens via `TrustToken.for_trusted_caller()`).
    To still exercise the trust-gate rejection path, we invoke
    `register()` from a synthetic module's globals whose `__name__` is
    NOT in `TRUSTED_TRUST_CALLERS`. The `inspect.stack()[1]` check in
    `register()` reads the immediate caller's `f_globals['__name__']`,
    which is the synthetic module's `__name__` because `exec()` runs
    the code in a new frame whose globals are the dict we pass.
    """
    import types
    from proxy.skills.skill_registry import _skill_registry_new__
    reg = _skill_registry_new__(SkillRegistry)
    dummy = DummySkill(name="ticket", trigger="^^ticket")
    # Synthesize a caller that is NOT in TRUSTED_TRUST_CALLERS.
    fake_globals = {
        "reg": reg,
        "dummy": dummy,
        "__name__": "fake.untrusted.caller",
    }
    with pytest.raises(SkillRegistrationError):
        exec("reg.register(dummy)", fake_globals)


# === Negative tests (Reviewer 3 HIGH-2, HIGH-3, HIGH-4, HIGH-5) ===

def test_register_rejects_invalid_name_format():
    """register() rejects names not matching the regex ^[a-z][a-z0-9_-]*$."""
    reg = SkillRegistry()
    dummy = DummySkill(name="InvalidName", trigger="^^invalid")
    with pytest.raises(SkillRegistrationError):
        reg.register(dummy)


def test_get_by_trigger_returns_none_for_unknown():
    """get_by_trigger() returns None for a non-existent trigger."""
    reg = SkillRegistry()
    reg.populate_builtins()
    assert reg.get_by_trigger("^^nonexistent") is None


def test_register_rejects_duplicate_trigger():
    """register() rejects a skill whose trigger is already taken by another skill."""
    reg = SkillRegistry()
    reg.populate_builtins()
    dummy = DummySkill(name="newskill", trigger="^^ticket")  # duplicate trigger
    with pytest.raises(SkillRegistrationError):
        reg.register(dummy)


def test_register_skill_rejects_non_skill_definition():
    """register_skill() rejects objects that are not SkillDefinition instances."""
    reg = SkillRegistry()
    with pytest.raises(SkillRegistrationError):
        reg.register_skill("not a SkillDefinition", trusted_registrant=TrustToken.UNTRUSTED)


def test_register_skill_rejects_overwrite_of_legacy_baseskill():
    """R1-3: register_skill() rejects overwriting a legacy BaseSkill in _inner.

    If a legacy BaseSkill is already registered via register(), a subsequent
    register_skill(SkillDefinition(...)) with the same name must raise
    SkillRegistrationError. This prevents user-defined SkillDefinitions from
    silently clobbering builtins loaded via the legacy path.
    """
    from proxy.skills.skill_registry import _skill_registry_new__
    reg = _skill_registry_new__(SkillRegistry)
    legacy = DummySkill(name="foo", trigger="^^foo")
    reg.register(legacy)  # legacy BaseSkill in _inner
    assert reg.get("foo") is legacy  # legacy object stored

    # Try to overwrite with a SkillDefinition via register_skill
    sd = SkillDefinition(name="foo", trigger="^^foo")
    from proxy.skills.skill_definition import _held_trust_token
    with _held_trust_token(TrustToken.for_trusted_caller()):
        with pytest.raises(SkillRegistrationError) as exc_info:
            reg.register_skill(sd, trusted_registrant=TrustToken.for_trusted_caller())
    assert "refusing to overwrite" in str(exc_info.value) or "existing entry" in str(exc_info.value)
    # Legacy object is still in the registry
    assert reg.get("foo") is legacy
