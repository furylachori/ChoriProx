"""Startup-contract tests for the strict-schema custom-skill loader.

These tests verify the integration between `get_default_registry()` (the
module-level singleton) and the strict-schema custom-skill loader. Per
spec `docs/specs/custom-skills-remediation.md` §5.1, the singleton:

- is initialized exactly once via `populate_builtins()`;
- never publishes a partial custom-skill snapshot;
- raises `SkillRegistryConfigurationError` on any single-file failure;
- treats `PROXY_SKILLS_DIR=""` as an explicit disabled flag;
- treats unset `PROXY_SKILLS_DIR` with no default directory as healthy
  builtins-only state;
- has no hot-reload — restart-only.

The tests MUST go through `get_default_registry()` and the
`reset_default_registry()` test-only helper — NOT direct private loader
calls. This is the production startup path.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from proxy.skills.custom_skill_loader import (
    CustomSkillsStatus,
    SkillRegistryConfigurationError,
)
from proxy.skills.skill_registry import (
    SkillRegistry,
    _default_registry,
    get_default_registry,
    reset_default_registry,
)


@pytest.fixture(autouse=True)
def reset_singleton():
    """Reset the singleton before AND after every test."""
    reset_default_registry()
    yield
    reset_default_registry()


def _write(root: Path, name: str, payload) -> Path:
    p = root / f"{name}.json"
    if isinstance(payload, str):
        p.write_text(payload)
    else:
        p.write_text(json.dumps(payload))
    return p


def _valid(name: str = "ok", **overrides) -> dict:
    base = {
        "name": name, "trigger": f"^^{name}",
        "description": f"{name} skill", "prompt": f"do {name}",
        "tool_whitelist": ["read"],
    }
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


def test_startup_loads_valid_custom_skill_from_disk(tmp_path, monkeypatch):
    """`get_default_registry()` registers a JSON-defined skill."""
    skills_dir = tmp_path / "skills"
    skills_dir.mkdir()
    _write(skills_dir, "ok", _valid("ok"))
    monkeypatch.setenv("PROXY_SKILLS_DIR", str(skills_dir))
    reset_default_registry()
    reg = get_default_registry()
    assert reg.get_by_trigger("^^ok") is not None
    assert reg.get("ok") is not None
    status = reg.custom_skills_status()
    assert status.state == "ok"
    assert status.enabled is True
    assert status.loaded_names == ("ok",)
    assert status.directory == str(skills_dir)


# ---------------------------------------------------------------------------
# Atomic failure
# ---------------------------------------------------------------------------


def test_startup_atomic_failure_no_partial_publish(tmp_path, monkeypatch):
    """A second invalid file fails the entire startup — no partial singleton."""
    skills_dir = tmp_path / "skills"
    skills_dir.mkdir()
    _write(skills_dir, "ok", _valid("ok"))
    _write(skills_dir, "bad", _valid("bad", description=""))
    monkeypatch.setenv("PROXY_SKILLS_DIR", str(skills_dir))
    reset_default_registry()
    with pytest.raises(SkillRegistryConfigurationError) as exc_info:
        get_default_registry()
    assert len(exc_info.value.issues) >= 1
    assert _default_registry is None
    with pytest.raises(SkillRegistryConfigurationError):
        get_default_registry()


# ---------------------------------------------------------------------------
# Env semantics
# ---------------------------------------------------------------------------


def test_startup_proxies_skills_dir_empty_string_disables(monkeypatch):
    """Empty env value → explicit disable → healthy builtins-only."""
    monkeypatch.setenv("PROXY_SKILLS_DIR", "")
    reset_default_registry()
    reg = get_default_registry()
    status = reg.custom_skills_status()
    assert status.state == "disabled"
    assert status.enabled is False
    assert status.source == "disabled"


def test_startup_default_dir_is_healthy(monkeypatch):
    """Env unset + default exists (or missing) → healthy.

    The default `<repo>/skills` directory MAY exist (with example skills)
    or MAY be missing (Agent 4 relocation pending). Either state is
    healthy — the loader does not raise.
    """
    monkeypatch.delenv("PROXY_SKILLS_DIR", raising=False)
    reset_default_registry()
    reg = get_default_registry()
    status = reg.custom_skills_status()
    # Healthy outcomes: 'ok' (loaded some/all) or 'disabled' (missing dir).
    assert status.state in ("ok", "disabled")
    assert status.enabled in (True, False)
    # restart_required is False (nothing to fix)
    assert status.restart_required is False


def test_startup_explicit_missing_dir_is_fatal(monkeypatch):
    """Env set to a non-existent path → fatal SkillRegistryConfigurationError."""
    monkeypatch.setenv("PROXY_SKILLS_DIR", "/no/such/dir/at/all")
    reset_default_registry()
    with pytest.raises(SkillRegistryConfigurationError):
        get_default_registry()
    assert _default_registry is None


# ---------------------------------------------------------------------------
# No hot reload
# ---------------------------------------------------------------------------


def test_no_hot_reload_after_publication(tmp_path, monkeypatch):
    """Adding a JSON after get_default_registry() must not be visible."""
    skills_dir = tmp_path / "skills"
    skills_dir.mkdir()
    _write(skills_dir, "ok", _valid("ok"))
    monkeypatch.setenv("PROXY_SKILLS_DIR", str(skills_dir))
    reset_default_registry()
    reg = get_default_registry()
    assert reg.get_by_trigger("^^ok") is not None

    # Add another file AFTER startup
    _write(skills_dir, "late", _valid("late"))
    assert reg.get_by_trigger("^^late") is None
    # A fresh construction picks it up
    reset_default_registry()
    reg2 = get_default_registry()
    assert reg2.get_by_trigger("^^late") is not None


# ---------------------------------------------------------------------------
# reset_default_registry
# ---------------------------------------------------------------------------


def test_reset_clears_old_registry_state(tmp_path, monkeypatch):
    """The OLD registry must be empty after reset (FIX-4 split-brain defense)."""
    skills_dir = tmp_path / "skills"
    skills_dir.mkdir()
    _write(skills_dir, "ok", _valid("ok"))
    monkeypatch.setenv("PROXY_SKILLS_DIR", str(skills_dir))
    reset_default_registry()
    reg1 = get_default_registry()
    assert "ok" in reg1
    reset_default_registry()
    assert "ok" not in reg1
    assert reg1.get_by_trigger("^^ok") is None
    reg2 = get_default_registry()
    assert "ok" in reg2


# ---------------------------------------------------------------------------
# Status snapshot
# ---------------------------------------------------------------------------


def test_custom_skills_status_returns_snapshot(tmp_path, monkeypatch):
    """custom_skills_status returns a CustomSkillsStatus with all fields."""
    skills_dir = tmp_path / "skills"
    skills_dir.mkdir()
    _write(skills_dir, "ok", _valid("ok"))
    monkeypatch.setenv("PROXY_SKILLS_DIR", str(skills_dir))
    reset_default_registry()
    reg = get_default_registry()
    status = reg.custom_skills_status()
    assert isinstance(status, CustomSkillsStatus)
    for field in (
        "enabled", "directory", "source", "state",
        "discovered", "loaded_names", "issues", "restart_required",
        "swarm_enabled", "swarm_capable_skills",
    ):
        assert hasattr(status, field)


def test_builtins_always_registered(monkeypatch):
    """ticket and work are always present regardless of custom config."""
    monkeypatch.delenv("PROXY_SKILLS_DIR", raising=False)
    reset_default_registry()
    reg = get_default_registry()
    assert "ticket" in reg
    assert "work" in reg
    assert reg.get_by_trigger("^^ticket") is not None
    assert reg.get_by_trigger("^^work") is not None


def test_builtins_and_customs_coexist(tmp_path, monkeypatch):
    """Custom skills add to (don't replace) the builtin set."""
    skills_dir = tmp_path / "skills"
    skills_dir.mkdir()
    _write(skills_dir, "audit", {
        "name": "audit", "trigger": "^^audit",
        "description": "audit code", "prompt": "audit",
        "tool_whitelist": ["read", "grep"],
    })
    monkeypatch.setenv("PROXY_SKILLS_DIR", str(skills_dir))
    reset_default_registry()
    reg = get_default_registry()
    assert "ticket" in reg
    assert "work" in reg
    assert "audit" in reg
    assert reg.get_by_trigger("^^audit") is not None
    assert len(reg.list_skills()) == 3


def test_status_restart_required_on_failure(tmp_path, monkeypatch):
    """A failed startup leaves restart_required=True on the status snapshot."""
    skills_dir = tmp_path / "skills"
    skills_dir.mkdir()
    _write(skills_dir, "bad", _valid("bad", description=""))
    monkeypatch.setenv("PROXY_SKILLS_DIR", str(skills_dir))
    reset_default_registry()
    from proxy.skills.skill_registry import _skill_registry_new__
    reg = _skill_registry_new__(SkillRegistry)
    try:
        reg.populate_builtins()
    except SkillRegistryConfigurationError:
        pass
    status = reg.custom_skills_status()
    assert status.restart_required is True
    assert status.state == "degraded"


def test_status_restart_required_false_on_success(monkeypatch):
    """A successful startup has restart_required=False."""
    monkeypatch.delenv("PROXY_SKILLS_DIR", raising=False)
    reset_default_registry()
    reg = get_default_registry()
    status = reg.custom_skills_status()
    assert status.restart_required is False


def test_status_empty_env_is_disabled_not_degraded(monkeypatch):
    """Empty env value is a clean disable, not a degraded state."""
    monkeypatch.setenv("PROXY_SKILLS_DIR", "")
    reset_default_registry()
    reg = get_default_registry()
    status = reg.custom_skills_status()
    assert status.state == "disabled"
    assert status.enabled is False
    assert status.restart_required is False


def test_singleton_identity_preserved_across_calls():
    """get_default_registry() returns the same instance every call."""
    reset_default_registry()
    r1 = get_default_registry()
    r2 = get_default_registry()
    r3 = get_default_registry()
    assert r1 is r2 is r3


def test_atomic_failure_builtins_loaded_even_if_customs_fail(tmp_path, monkeypatch):
    """Builtins are registered FIRST; a custom load failure doesn't roll them back."""
    skills_dir = tmp_path / "skills"
    skills_dir.mkdir()
    _write(skills_dir, "bad", _valid("bad", description=""))
    monkeypatch.setenv("PROXY_SKILLS_DIR", str(skills_dir))
    reset_default_registry()
    from proxy.skills.skill_registry import _skill_registry_new__
    reg = _skill_registry_new__(SkillRegistry)
    try:
        reg.populate_builtins()
        assert False, "should have raised"
    except SkillRegistryConfigurationError:
        pass
    # Builtins present
    assert "ticket" in reg._inner
    assert "work" in reg._inner
    # Custom absent
    assert "bad" not in reg._inner


# ---------------------------------------------------------------------------
# Swarm failure aggregation (spec §4, AC 16.16)
# ---------------------------------------------------------------------------


def test_startup_swarm_failure_aggregates_atomically(tmp_path, monkeypatch):
    """3 files with different swarm failures → startup raises with all issue codes.

    Uses only validation paths that don't require SubagentPolicy
    instantiation, so the test runs without Agent 2:
      bad1 — task in parent without `subagents` → swarm_subagents_missing
      bad2 — `subagents: {}` → swarm_tool_whitelist_missing + swarm_max_calls_missing
      bad3 — child whitelist contains forbidden tool → swarm_child_forbidden_tool
    """
    skills_dir = tmp_path / "skills"
    skills_dir.mkdir()
    _write(skills_dir, "bad1", {
        "name": "bad1", "trigger": "^^bad1",
        "description": "d", "prompt": "p",
        "tool_whitelist": ["read", "task"],
    })
    _write(skills_dir, "bad2", {
        "name": "bad2", "trigger": "^^bad2",
        "description": "d", "prompt": "p",
        "tool_whitelist": ["read", "task"],
        "subagents": {},
    })
    _write(skills_dir, "bad3", {
        "name": "bad3", "trigger": "^^bad3",
        "description": "d", "prompt": "p",
        "tool_whitelist": ["read", "task"],
        "subagents": {
            "tool_whitelist": ["task"],
            "max_task_calls_per_request": 4,
        },
    })
    monkeypatch.setenv("PROXY_SKILLS_DIR", str(skills_dir))
    reset_default_registry()
    with pytest.raises(SkillRegistryConfigurationError) as exc_info:
        get_default_registry()
    codes = sorted([i.code for i in exc_info.value.issues])
    assert "swarm_subagents_missing" in codes
    assert "swarm_tool_whitelist_missing" in codes
    assert "swarm_max_calls_missing" in codes
    assert "swarm_child_forbidden_tool" in codes