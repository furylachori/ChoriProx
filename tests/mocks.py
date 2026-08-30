# tests/mocks.py — shared test doubles for the skill system.
#
# These doubles intentionally mirror the pre-1a production classes
# (`proxy.agent.agent.Agent`, `proxy.types.RequestContext`) at the
# field-and-method level. They are NOT drop-in replacements for the
# real classes (e.g., `MockAgent.is_ticket_mode` is a simple attribute
# read, not a property with side-effects), but they expose the same
# surface for tests that need a stand-in.
#
# Per the 1a plan: this module ships in the same PR as
# `src/proxy/skills/skill_definition.py` and `tests/test_skill_definition.py`
# (R4-Issue 15, CRIT co-shipping rule). Without it, the §5 tests cannot
# instantiate a `MockAgent` to pass to `SkillRequestContext.from_request_context`.
#
# R3 HIGH-13 fix: the MockAgent shapes below were verified against the
# pre-1a `proxy.agent.agent.Agent` (file: src/proxy/agent/agent.py):
#   - `Agent.snapshot_mode_and_ticket_ts() -> tuple[Optional[ModeState], Optional[float]]`
#   - `Agent.set_active_mode(mode_state, expected_mode=None) -> bool`
#   - `Agent.is_mode_expired() -> bool`
#   - `Agent.is_ticket_mode -> bool` (property)
#   - `Agent.is_skill_mode -> bool` (property)
# All five signatures match the MockAgent methods below.
#
# Phase 3: `Agent.set_ticket_mode` was deleted from the real Agent (zero
# production callers). The mock therefore also drops `set_ticket_mode`;
# tests must use `set_active_mode(ticket_mode_state)` instead.
#
# R1-HIGH-3 fix: `ModeState` is imported at the top of this module so
# the import is hoisted to module load time. This eliminates a per-call
# import cost and keeps the mock importable before
# `proxy.skills.skill_definition` has been fully wired (defensive: the
# import would resolve anyway because `skill_definition` is co-shipped
# in the same PR per R4-Issue 15).

import os
import sys
import time
from typing import Optional
from unittest.mock import MagicMock

# Test-only guard: prevent tests/mocks.py from being importable in production.
# The guard fires ONLY when (a) pytest is not in sys.modules, (b) PYTEST_CURRENT_TEST
# is unset, and (c) CHORI_TEST_MODE is not "1". The first condition is met as soon
# as pytest is imported anywhere in the process (which happens at pytest's own
# startup). The other two are explicit overrides for non-pytest test runners
# (e.g., direct python -m invocation) and manual test-mode opt-in.
_GUARD_OK = (
    "pytest" in sys.modules
    or os.environ.get("PYTEST_CURRENT_TEST")
    or os.environ.get("CHORI_TEST_MODE") == "1"
)
if not _GUARD_OK:
    raise ImportError("tests.mocks is test-only; do not import from production code")

from proxy.skills.skill_definition import ModeState  # R1-HIGH-3: hoisted import


class MockAgent:
    """Drop-in test double for `proxy.agent.agent.Agent`.

    Carries the 8 fields/methods that 1a tests need:
      - `agent_id`, `session_id`: identity
      - `active_mode`: the `ModeState` currently set (or None)
      - `snapshot_mode()`: returns `active_mode` (read-only view)
      - `set_active_mode(mode_state)`: assigns `active_mode` AND
        manages `_ticket_mode_ts` for non-ticket modes (R2-HIGH-9 fix)
      - `clear_active_mode()`: assigns None
      - `is_mode_expired()`: wraps `active_mode.is_expired()` with a
        None-guard (mirrors the pre-1a `Agent.is_mode_expired` body
        at agent.py:254-264)
      - `is_ticket_mode`, `is_skill_mode`: derived from `active_mode`

    Phase 3: `set_ticket_mode` was removed from the real Agent and is
    likewise removed from this mock. Tests must build a ticket
    `ModeState` and pass it to `set_active_mode()` instead.
    """

    def __init__(
        self,
        agent_id: str = "agent_test",
        session_id: str = "sess_test",
        active_mode=None,
    ):
        self.agent_id = agent_id
        self.session_id = session_id
        self.active_mode = active_mode
        self._ticket_mode_ts: Optional[float] = None
        self.context_limit_override: Optional[int] = None

    def set_context_limit_override(self, tokens):
        self.context_limit_override = tokens

    def get_context_limit_override(self):
        return self.context_limit_override

    def snapshot_mode(self):
        """Return the current `ModeState` (or None). Mirrors the pre-1a `Agent.snapshot_mode`."""
        return self.active_mode

    def snapshot_mode_and_ticket_ts(self):
        """Return (ModeState, ticket_mode_ts) — pre-1a pair shape."""
        return self.active_mode, self._ticket_mode_ts

    def set_active_mode(self, mode_state, expected_mode=None) -> bool:
        """Assign `active_mode`. Returns True (pre-1a `Agent.set_active_mode` shape).

        R2-HIGH-9 fix: when `mode_state` is non-ticket (or None),
        `_ticket_mode_ts` is cleared to None.

        R3-CRIT-4 fix: `_ticket_mode_ts` is set to `time.time()` (the
        wall-clock at the moment of the call), NOT `mode_state.activated_at`.

        R3-CRIT-5 fix: the `expected_mode` CAS is honored. If
        `expected_mode is not None and self.active_mode is not
        expected_mode`, the call returns `False` WITHOUT mutating
        state. Returns `True` on successful assignment.
        """
        if expected_mode is not None and self.active_mode is not expected_mode:
            return False
        self.active_mode = mode_state
        if mode_state is not None and getattr(mode_state, "name", None) == "ticket":
            self._ticket_mode_ts = time.time()
        else:
            self._ticket_mode_ts = None
        return True

    def clear_active_mode(self, expected_mode=None) -> bool:
        """Clear `active_mode`. Returns True (pre-1a `Agent.clear_active_mode` shape).

        R3-CRIT-5 fix: the `expected_mode` CAS is honored.
        """
        if expected_mode is not None and self.active_mode is not expected_mode:
            return False
        self.active_mode = None
        self._ticket_mode_ts = None
        return True

    def is_mode_expired(self) -> bool:
        """Return True iff `active_mode` is set and its TTL has expired.

        Returns False if `active_mode` is None (no mode active → not
        expired). Otherwise, returns `active_mode.is_expired()`.
        """
        ms = self.active_mode
        if ms is None:
            return False
        return ms.is_expired()

    @property
    def is_ticket_mode(self) -> bool:
        """Mirrors `Agent.is_ticket_mode` at agent.py:188-200 EXACTLY (R3-CRIT-6 fix).

        The pre-1a `Agent.is_ticket_mode` body has FOUR branches:
          1. `active_mode is None` → fall through to the
             `_ticket_mode_ts` legacy-timestamp fallback.
          2. `active_mode.name != "ticket"` → return False
             (work/workspace modes are NOT ticket modes).
          3. `active_mode is None and _ticket_mode_ts is None` →
             return False (no mode and no legacy timestamp).
          4. `active_mode.name == "ticket" and not is_expired()` →
             return True.
        """
        if self.active_mode is None:
            return self._ticket_mode_ts is not None
        if self.active_mode.name != "ticket":
            return False
        return not self.active_mode.is_expired()

    @property
    def is_skill_mode(self) -> bool:
        """Returns True iff a non-None, non-expired mode is active.

        R3-CRIT-7 fix: this is the NEW-DESIGN shape. The 1c design
        considers `workspace` to be a parser-only mode (not a
        registered skill), so `is_skill_mode` in the 1c+ design is
        "any non-None, non-expired mode".

        # WARNING: Mirrors pre-1a Agent.is_skill_mode (returns True
        # for ticket AND workspace). The 1c design considers
        # workspace parser-only, so the post-1c shape would be
        # `active_mode is not None and not active_mode.is_expired()
        # and active_mode.name != "workspace"`. The mock keeps the
        # pre-1a body (treating both ticket and workspace as skill
        # modes) until 1d migrates the test suite to the new
        # contract. This is a deliberate divergence from the new
        # design — see the 1d plan row in §10.
        """
        return self.active_mode is not None and not self.active_mode.is_expired()

class MockRequestContext:
    """Drop-in test double for the pipeline `RequestContext`.

    Carries the 8 fields that `SkillRequestContext.from_request_context`
    reads. `body` is a `MagicMock` by default so tests can configure
    `ctx.body["messages"]` etc. without building a full OpenAI-shape
    request dict.
    """

    def __init__(
        self,
        session_id: str = "sess_test",
        is_subagent: bool = False,
        inherited_skill: Optional[str] = None,
        subagent_ticket_mode: bool = False,
        workspace_name: str = "",
        client_hint: Optional[str] = None,
        body: Optional[dict] = None,
        seeded_tasks: Optional[list] = None,
    ):
        self.session_id = session_id
        self.is_subagent = is_subagent
        self.inherited_skill = inherited_skill
        self.subagent_ticket_mode = subagent_ticket_mode
        self.workspace_name = workspace_name
        self.client_hint = client_hint
        self.body = body if body is not None else MagicMock()
        self.seeded_tasks = seeded_tasks if seeded_tasks is not None else []
