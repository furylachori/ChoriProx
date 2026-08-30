"""tests/test_mocks.py — coverage for tests/mocks.py (R3-3 fix).

These tests pin the contracts that the §5 test suite relies on but
which the 1a plan did not exercise directly:

  1.  (REMOVED in Phase 3) `set_ticket_mode` was deleted from both the
      real `Agent` and the mock `MockAgent` (zero production callers —
      see docs/specs/consolidation-research/18-phase-3-final.md §H1).
      Ticket mode is now installed via
      `set_active_mode(ModeState(name="ticket", ...))`.
  2.  (REMOVED in Phase 3) `set_ticket_mode(False)` clearing path is
      covered by `clear_active_mode()` directly.
  3.  test_mock_agent_set_active_mode_cas_returns_false_on_mismatch
      — the `expected_mode` CAS branch returns False WITHOUT mutating
      `active_mode` (R3-CRIT-5 fix contract).
  4.  test_mock_agent_is_mode_expired_none_returns_false
      — `is_mode_expired()` returns False when `active_mode is None`
        (no-mode-active → not-expired).
  5.  test_mock_agent_is_ticket_mode_handles_workspace_mode
      — `is_ticket_mode` returns False when `active_mode.name` is not
        "ticket" (work/workspace modes are NOT ticket modes).
  6.  test_mock_agent_snapshot_mode_and_ticket_ts_shape
      — `snapshot_mode_and_ticket_ts()` returns the
        `(active_mode, _ticket_mode_ts)` pair.
  7.  test_mock_request_context_default_body_is_callable
      — `MockRequestContext().body` defaults to a `MagicMock` (which
        is callable for `ctx.body["messages"]` access patterns).
  8.  test_mocks_module_raises_import_error_outside_test_mode
      — verifies the module-level guard at the top of `tests/mocks.py`
        raises `ImportError` when pytest is not loaded AND the
        `PYTEST_CURRENT_TEST` / `CHORI_TEST_MODE` env vars are unset.
        This is the R3-HIGH-1 / R3-HIGH-8 guard that keeps
        `tests/mocks.py` out of production.
  9.  (REMOVED in Phase 3) `set_ticket_mode` signature pinning is no
      longer meaningful — both sides of the lock-step were deleted.

The test file is self-contained: it imports only `tests.mocks` and
`proxy.skills.skill_definition.ModeState`. It does NOT mutate any other
file in the repo.
"""
import importlib
import os
import sys
import time

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _fresh_mock_agent():
    """Construct a MockAgent with no mode active and no legacy ticket ts."""
    from tests.mocks import MockAgent  # noqa: F401  (import may trigger guard)
    return MockAgent()


# ---------------------------------------------------------------------------
# Test 1 (REMOVED): set_ticket_mode(True) constructs the canonical
# ticket ModeState. The mock's set_ticket_mode was deleted in Phase 3
# alongside the real Agent.set_ticket_mode (zero production callers —
# see docs/specs/consolidation-research/18-phase-3-final.md §H1). The
# canonical ticket ModeState is now constructed directly in tests via
# ModeState(name="ticket", ttl_seconds=300, allowed_tool_names=frozenset({...}))
# and installed via set_active_mode(). This is exercised by
# test_mock_agent_is_ticket_mode_handles_workspace_mode below.
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Test 2 (REMOVED): set_ticket_mode(False) clears active_mode. Same
# rationale as Test 1 above. clear_active_mode() is the supported
# API for clearing the active mode and is covered by other tests.
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Test 3: set_active_mode CAS returns False on identity mismatch
# ---------------------------------------------------------------------------
def test_mock_agent_set_active_mode_cas_returns_false_on_mismatch():
    """`set_active_mode(new, expected_mode=other)` returns False
    WITHOUT mutating `active_mode` when `expected_mode` is not the
    same object as `self.active_mode` (R3-CRIT-5 fix contract).

    Sets up an initial mode, then attempts to swap with a stale
    `expected_mode`. The swap must fail and the initial mode must
    remain in place.
    """
    from proxy.skills.skill_definition import ModeState

    agent = _fresh_mock_agent()
    initial = ModeState(
        name="ticket",
        activated_at=time.time(),
        ttl_seconds=300,
        allowed_tool_names=frozenset({"read"}),
    )
    assert agent.set_active_mode(initial) is True
    assert agent.active_mode is initial

    # Build a different ModeState and a stale expected_mode that does
    # NOT match the current `active_mode`. The CAS should fail.
    stale_expected = ModeState(
        name="work",
        activated_at=time.time(),
        ttl_seconds=60,
        allowed_tool_names=frozenset(),
    )
    new_mode = ModeState(
        name="work",
        activated_at=time.time() + 1,
        ttl_seconds=60,
        allowed_tool_names=frozenset(),
    )

    result = agent.set_active_mode(new_mode, expected_mode=stale_expected)

    assert result is False
    # `active_mode` MUST still be the original ticket mode (not new_mode).
    assert agent.active_mode is initial
    assert agent.active_mode.name == "ticket"


# ---------------------------------------------------------------------------
# Test 4: is_mode_expired returns False when active_mode is None
# ---------------------------------------------------------------------------
def test_mock_agent_is_mode_expired_none_returns_false():
    """`is_mode_expired()` returns False when no mode is active.

    A fresh MockAgent has `active_mode = None`; the contract is that
    `is_mode_expired()` returns False (no mode → not expired). The
    method must not raise AttributeError.
    """
    agent = _fresh_mock_agent()
    assert agent.active_mode is None  # pre-condition
    # Must not raise; must return False.
    assert agent.is_mode_expired() is False


# ---------------------------------------------------------------------------
# Test 5: is_ticket_mode returns False for workspace mode
# ---------------------------------------------------------------------------
def test_mock_agent_is_ticket_mode_handles_workspace_mode():
    """`is_ticket_mode` returns False when `active_mode.name` is
    "workspace" (work/workspace modes are NOT ticket modes).

    The pre-1a `Agent.is_ticket_mode` body returns False for any
    non-"ticket" mode. The mock mirrors that contract.
    """
    from proxy.skills.skill_definition import ModeState

    agent = _fresh_mock_agent()
    agent.set_active_mode(ModeState(
        name="workspace",
        activated_at=time.time(),
        ttl_seconds=0,  # persistent
        allowed_tool_names=None,
    ))
    # Sanity: the active mode is workspace, not ticket.
    assert agent.active_mode.name == "workspace"
    # The contract: is_ticket_mode must be False.
    assert agent.is_ticket_mode is False


# ---------------------------------------------------------------------------
# Test 6: snapshot_mode_and_ticket_ts returns the (mode, ts) pair
# ---------------------------------------------------------------------------
def test_mock_agent_snapshot_mode_and_ticket_ts_shape():
    """`snapshot_mode_and_ticket_ts()` returns the
    `(active_mode, _ticket_mode_ts)` pair in the pre-1a shape.

    After setting a ticket mode (via set_active_mode), both the
    ModeState and the `_ticket_mode_ts` should be present in the pair.
    After clearing, both should be None.

    Phase 3: ticket mode is installed via set_active_mode(ModeState)
    instead of the deleted set_ticket_mode wrapper.
    """
    from proxy.skills.skill_definition import ModeState

    agent = _fresh_mock_agent()

    # No mode active → (None, None)
    mode, ts = agent.snapshot_mode_and_ticket_ts()
    assert mode is None
    assert ts is None

    # Set ticket mode → (ModeState, float)
    agent.set_active_mode(ModeState(
        name="ticket",
        activated_at=time.time(),
        ttl_seconds=300,
        allowed_tool_names=frozenset({"read", "glob", "grep", "search", "proxy_ticket"}),
    ))
    mode, ts = agent.snapshot_mode_and_ticket_ts()
    assert mode is not None
    assert mode.name == "ticket"
    assert isinstance(ts, float)
    assert ts > 0  # the wall-clock at activation time

    # Clear → (None, None)
    agent.clear_active_mode()
    mode, ts = agent.snapshot_mode_and_ticket_ts()
    assert mode is None
    assert ts is None


# ---------------------------------------------------------------------------
# Test 7: MockRequestContext default body is a MagicMock (callable)
# ---------------------------------------------------------------------------
def test_mock_request_context_default_body_is_callable():
    """`MockRequestContext().body` defaults to a `MagicMock`.

    The default body MUST be callable so that downstream code that
    uses `ctx.body["messages"]` (or any other indexing/access pattern)
    can rely on `body` behaving like a dict-like MagicMock without
    raising TypeError. The `MagicMock` supports `__getitem__`,
    `__setitem__`, etc. via the magic-method protocol.
    """
    from unittest.mock import MagicMock
    from tests.mocks import MockRequestContext

    ctx = MockRequestContext()

    # Default body should be a MagicMock instance.
    assert isinstance(ctx.body, MagicMock)
    # And it should be callable (MagicMock is callable by default).
    assert callable(ctx.body)
    # And it should support attribute access (the downstream pattern
    # is `ctx.body["messages"]` etc.; MagicMock's magic-method
    # protocol makes `body[...]` return child mocks without raising).
    assert ctx.body["messages"] is not None  # MagicMock returns a child mock

    # Other defaults should be the documented safe fallbacks.
    assert ctx.session_id == "sess_test"
    assert ctx.is_subagent is False
    assert ctx.inherited_skill is None
    assert ctx.subagent_ticket_mode is False
    assert ctx.workspace_name == ""
    assert ctx.client_hint is None
    assert ctx.seeded_tasks == []


# ---------------------------------------------------------------------------
# Test 8: the module-level guard raises ImportError outside test mode
# ---------------------------------------------------------------------------
def test_mocks_module_raises_import_error_outside_test_mode(
    monkeypatch, request,
):
    """The module-level guard at the top of `tests/mocks.py` raises
    `ImportError` when:
      - `"pytest"` is NOT in `sys.modules`
      - `PYTEST_CURRENT_TEST` env var is unset
      - `CHORI_TEST_MODE` env var is not "1"

    This test simulates a "production" import (none of the three
    guards satisfied) by removing the env vars AND evicting
    `sys.modules["pytest"]` (if present), then re-importing
    `tests.mocks`. The guard at the top of the module must fire.

    The eviction is necessary because Python caches imported modules
    in `sys.modules`; a second `import tests.mocks` would return the
    cached module without re-executing the module-level guard.

    At test teardown, monkeypatch restores the original state of
    `sys.modules` and the env vars, so subsequent tests see the
    normal pytest-loaded environment.
    """
    # 1. Remove the env vars. `raising=False` so monkeypatch does not
    #    raise if the var was already unset.
    monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)
    monkeypatch.delenv("CHORI_TEST_MODE", raising=False)

    # 2. Evict "pytest" from sys.modules (if present) so the
    #    `"pytest" in sys.modules` check returns False.
    monkeypatch.delitem(sys.modules, "pytest", raising=False)

    # 3. Evict "tests.mocks" so the re-import actually re-runs the
    #    module-level guard (otherwise Python returns the cached
    #    module without re-executing it).
    monkeypatch.delitem(sys.modules, "tests.mocks", raising=False)

    # 4. Also evict any already-loaded submodules of `tests.mocks`
    #    (e.g., the test_mocks module itself) so a re-import sees a
    #    clean state. Skip this if `tests.mocks` is not in sys.modules
    #    at all (clean run).
    for mod_name in list(sys.modules):
        if mod_name == "tests.mocks" or mod_name.startswith("tests.mocks."):
            monkeypatch.delitem(sys.modules, mod_name, raising=False)

    # 5. Sanity-check the guard preconditions are NOT satisfied.
    assert "pytest" not in sys.modules
    assert not os.environ.get("PYTEST_CURRENT_TEST")
    assert os.environ.get("CHORI_TEST_MODE") != "1"

    # 6. Attempt to re-import `tests.mocks` and expect ImportError.
    with pytest.raises(ImportError) as exc_info:
        importlib.import_module("tests.mocks")

    # 7. Verify the error message is the expected one (defends against
    #    accidental changes to the guard message).
    assert "test-only" in str(exc_info.value).lower() or "do not import" in str(
        exc_info.value
    ).lower(), (
        f"Unexpected ImportError message: {exc_info.value!r}"
    )


# ---------------------------------------------------------------------------
# Test 9 (REMOVED): set_ticket_mode signature matches Agent.set_ticket_mode.
# Both `MockAgent.set_ticket_mode` and `Agent.set_ticket_mode` were deleted
# in Phase 3 (zero production callers — see
# docs/specs/consolidation-research/18-phase-3-final.md §H1). The lock-step
# contract is no longer meaningful; ticket mode is installed via
# `set_active_mode(ModeState(name="ticket", ...))` instead and is covered
# by test_mock_agent_is_ticket_mode_handles_workspace_mode and
# test_mock_agent_snapshot_mode_and_ticket_ts_shape above.
# ---------------------------------------------------------------------------

