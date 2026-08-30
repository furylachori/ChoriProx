"""Unit tests for BaseSkill.build_skill_canary.

build_skill_canary wraps the signed canary in the marker format
``[[__SUBAGENT__:SKILL:<name>:<sig>:<state>]]``. The wrapped format is
4-part (``SKILL:<name>:<sig>:<state>``), but ``verify_canary`` expects
either the bare 3-part token (``<name>:<sig>:<ts>``) or the 4-part
``workspace:<name>:<sig>:<ts>`` form. The wrapped marker is therefore
unparseable by ``verify_canary`` — the round-trip must go through
``parse_canary``, which understands the marker wrapper.

These tests cover:

  - the non-serializable-state branch in
    ``build_skill_canary`` (``except (TypeError, ValueError): pass``)
    — the canary must still produce a parseable wrapped marker with an
    empty state suffix (HMAC over an empty state_str);
  - the no-inheritance-state branch (``state`` falsy) — same contract:
    a parseable wrapped marker.

We assert the round-trip via ``parse_canary``, NOT via
``verify_canary`` (the latter cannot read the wrapped marker). The
assertion that proves the production code path was exercised is the
``parsed.skill_name == "test"`` check on the parsed payload — which
ONLY succeeds when ``parse_canary`` accepts the marker AND HMAC
verification passes.
"""
from __future__ import annotations

import importlib
from unittest.mock import MagicMock

import pytest


_SECRET = "test-secret-for-build-skill-canary"


@pytest.fixture
def with_canary_secret(monkeypatch):
    """Set PROXY_SESSION_SECRET and reload proxy.canary so its
    ``_SESSION_SECRET`` cache picks up the new value.

    Without the reload, the module-level cache (set at first import)
    keeps the OLD secret, and ``_get_session_secret()`` falls back to
    it when the env var is empty — which is exactly what pollutes
    sibling tests that expect the dev (no-secret) canary path.
    """
    monkeypatch.setenv("PROXY_SESSION_SECRET", _SECRET)
    import proxy.canary as canary_mod
    importlib.reload(canary_mod)
    yield canary_mod
    # monkeypatch restores the env var; the tearDown then reloads
    # proxy.canary to capture the original (empty / absent) secret so
    # sibling tests that expect the dev path see what they expect.


@pytest.fixture(autouse=True)
def _restore_canary_module():
    """Reload proxy.canary AFTER each test in this module so the
    ``_SESSION_SECRET`` cache is re-captured from the post-test env
    (which monkeypatch has restored). This prevents the secret we set
    via the fixture from leaking into sibling test modules.
    """
    yield
    import proxy.canary as canary_mod
    importlib.reload(canary_mod)


class _NonSerializable:
    """A value that json.dumps cannot serialise (TypeError)."""

    def __repr__(self) -> str:
        return "<non-serializable>"


def test_serializable_state_round_trips_via_parse_canary(with_canary_secret):
    from proxy.skills.skill_definition import BaseSkill

    class S(BaseSkill):
        name = "ticket"

        def get_inheritance_state(self, ctx):
            return {"workspace_name": "choriprox", "tasks": []}

    ctx = MagicMock()
    ctx.session_id = "sess-test"

    canary = S().build_skill_canary(ctx)

    assert canary.startswith("[[__SUBAGENT__:SKILL:"), canary
    assert canary.endswith("]]"), canary

    parsed = with_canary_secret.parse_canary(canary)
    assert parsed is not None, f"parse_canary returned None for {canary!r}"
    assert parsed.kind == "skill"
    assert parsed.skill_name == "ticket"
    assert parsed.verified is True
    assert parsed.state == {"workspace_name": "choriprox", "tasks": []}


def test_non_serializable_state_branch_does_not_break_canary(with_canary_secret):
    """The except (TypeError, ValueError): pass path in build_skill_canary
    must still produce a parseable canary (state_str simply stays empty,
    HMAC is computed over an empty state_str)."""
    from proxy.skills.skill_definition import BaseSkill

    class S(BaseSkill):
        name = "ticket"

        def get_inheritance_state(self, ctx):
            return {"bad": _NonSerializable()}

    ctx = MagicMock()
    ctx.session_id = "sess-test"

    canary = S().build_skill_canary(ctx)

    assert canary.startswith("[[__SUBAGENT__:SKILL:"), canary
    assert canary.endswith("]]"), canary

    parsed = with_canary_secret.parse_canary(canary)
    assert parsed is not None, (
        f"build_skill_canary produced unparseable canary on "
        f"non-serializable state: {canary!r}"
    )
    assert parsed.kind == "skill"
    assert parsed.skill_name == "ticket"
    assert parsed.verified is True


def test_no_inheritance_state_branch_produces_parseable_canary(with_canary_secret):
    """When get_inheritance_state returns None/empty, state_str stays
    empty and the canary must still be parseable."""
    from proxy.skills.skill_definition import BaseSkill

    class S(BaseSkill):
        name = "ticket"

        def get_inheritance_state(self, ctx):
            return None

    ctx = MagicMock()
    ctx.session_id = "sess-test"

    canary = S().build_skill_canary(ctx)

    assert canary.startswith("[[__SUBAGENT__:SKILL:"), canary
    assert canary.endswith("]]"), canary

    parsed = with_canary_secret.parse_canary(canary)
    assert parsed is not None, (
        f"build_skill_canary produced unparseable canary with no state: {canary!r}"
    )
    assert parsed.kind == "skill"
    assert parsed.skill_name == "ticket"
    assert parsed.verified is True
    assert parsed.state is None