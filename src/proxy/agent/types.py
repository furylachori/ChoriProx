"""Agent types — shared dataclasses for task proposals.

==============================================================================
R1-HIGH-4 (1a) — `refresh()` intentionally REMOVED from ModeState
==============================================================================
`ModeState.refresh()` was REMOVED in Phase 1a as YAGNI. Reasoning:

  * Pre-1a `refresh()` was called only from a test-side path that 1a
    rewrites (the 11-test Phase 1 minimum-bar suite uses
    `time.sleep(N)` to force expiry instead of calling `refresh()`).
  * The 7 production callers in §3.3.1 of the spec consume
    `is_expired()` only; they never mutate `activated_at` to "extend"
    a mode's lifetime.
  * Removing `refresh()` keeps the frozen contract tight: a caller
    cannot `ms.refresh()` to silently shift `activated_at` after the
    ModeState has been published to other threads (CON-007).

If a real production consumer of `refresh()` is introduced in 1b/1c,
re-add the method using a single `object.__setattr__` write to
`activated_at` (the only field that needs to move). Pin a regression
test in `test_mode_state.py` that exercises the new consumer path.

==============================================================================
"""
import threading
import time
from types import MappingProxyType
from dataclasses import dataclass, field, FrozenInstanceError
from typing import Optional


# R2-HIGH-3 fix: thread-local flag used by ModeState.__setattr__ to
# permit writes ONLY during ModeState.__post_init__. The dataclass
# `frozen=True` machinery bypasses __setattr__ for its own field
# assignments in __init__ (it uses `object.__setattr__` directly), so
# the flag does not need to be set during the auto-generated __init__
# — only when __post_init__ is about to perform post-construction
# mutations (e.g., the MappingProxyType upgrade for `skill_state`).
_in_mode_state_post_init = threading.local()


@dataclass(frozen=True)
class ModeState:
    """Tracks the active operating mode for an agent.

    CRIT-S.1 fix (N1 CRIT): `skill_state` is a `MappingProxyType` (a
    read-only view of a dict), NOT a plain dict. This blocks in-place
    mutation of skill state at the dataclass level: a caller cannot
    write `ms.skill_state["x"] = 1`. The `frozen=True` decorator blocks
    whole-attribute reassignment (`ms.skill_state = {}`); the
    `MappingProxyType` blocks item-level mutation. Both must hold for
    the immutability contract.

    R3 CRIT-9 fix: `__post_init__` enforces the `MappingProxyType`
    invariant at construction time — even when the caller passes a
    plain `dict`, the field is silently upgraded to a
    `MappingProxyType` view. Without this, `ModeState(name="x",
    skill_state={"k": "v"})` would leave the field as a mutable dict
    and the CRIT-S.1 contract would be bypassable.

    R1 CRIT-2 fix: `is_expired()` is KEPT on `ModeState` as a non-data
    method. The `frozen=True` decorator blocks field reassignment
    (`ms.name = "work"` raises `FrozenInstanceError`); it does NOT
    block method definitions. The 7 production callers (§3.3.1) MUST
    continue to work without modification.

    R1-HIGH-4 fix: `refresh()` is REMOVED in 1a (YAGNI — no current
    production consumers; pre-1a `refresh()` was only called from a
    test-side path that 1a rewrites). Re-add in 1b/1c if a real
    consumer is introduced.

    R2-HIGH-8 fix: `allowed_tool_names` keeps its pre-1a type
    `Optional[frozenset[str]] = None` (NOT `frozenset[str] = field(...)`).
    `None` means "no restriction" — the 7 production callers and the
    pre-1a no-restriction sentinel both pass `None` for the unrestricted
    case.

    Wave 1 (spec §6) three-state semantics for `allowed_tool_names`:

      - ``None`` — unrestricted. Reserved for explicitly trusted
        Python-defined skills only (TicketSkill, WorkSkill). The
        custom-skill loader NEVER passes ``None``; it passes
        ``frozenset()`` for empty whitelists (deny all).
      - ``frozenset()`` — deny all tools. The LLM receives no tools.
      - Non-empty ``frozenset`` — exact allowlist.

    All callers (LLM-facing tool visibility, intercepted memory-tool
    guards, main-agent mode, mode snapshots, tests, docs) MUST use this
    same three-state semantics. No truthiness check may conflate
    ``None`` with ``frozenset()`` — they are distinct sentinels with
    opposite intent (unrestricted vs. deny-all).

    Attributes:
        name:                Mode identifier (e.g., "ticket", "workspace").
        activated_at:        time.time() when mode was activated.
        ttl_seconds:         Time-to-live in seconds (0 = persistent/no expiry).
        allowed_tool_names:  Optional immutable frozenset of permitted tool
                             names. None means no restriction (all tools
                             allowed). frozenset() means deny all. Non-empty
                             frozenset is an exact allowlist. See class
                             docstring + SkillDefinition class docstring.
        sub_agent_tool_filter: Tool filter for sub-agents when this mode is
                             active. Always a frozenset (NOT three-state) —
                             the sub-agent filter is only consulted when
                             `allowed_tool_names` is non-empty.
        skill_state:         Arbitrary skill-specific state (read-only MappingProxyType).
    """
    name: str
    activated_at: float
    ttl_seconds: int
    # R2-HIGH-8 fix: keep pre-1a Optional[frozenset[str]] = None type.
    # None means "no restriction" — the 7 production callers and the
    # pre-1a no-restriction sentinel both pass None for the unrestricted case.
    allowed_tool_names: Optional[frozenset[str]] = None
    sub_agent_tool_filter: frozenset[str] = field(default_factory=frozenset)
    # CRIT-S.1 fix: MappingProxyType blocks in-place mutation.
    # R3 CRIT-9 fix: __post_init__ enforces this even if a plain dict is passed.
    skill_state: MappingProxyType = field(
        default_factory=lambda: MappingProxyType({})
    )

    def __post_init__(self):
        # R2-HIGH-3 fix: enable the thread-local bypass so the
        # MappingProxyType upgrade and any future post-construction
        # mutations can write to fields without tripping the new
        # __setattr__ guard. The flag is per-thread so concurrent
        # ModeState construction on other threads is unaffected.
        # The try/finally guarantees the flag is cleared even on
        # ValueError below, preventing leakage across calls.
        _in_mode_state_post_init.active = True
        try:
            # R3 CRIT-9 fix: enforce MappingProxyType invariant. A
            # plain dict passed by the caller is silently upgraded to
            # a read-only view so the CRIT-S.1 read-only contract
            # cannot be bypassed by callers passing mutable dicts.
            if not isinstance(self.skill_state, MappingProxyType):
                self.skill_state = MappingProxyType(dict(self.skill_state))

            # R2-HIGH-4 fix: negative TTL is invalid. `ttl_seconds == 0`
            # remains the persistent (no-expiry) sentinel (R1 CRIT-2
            # contract). Rejecting negative values surfaces caller bugs
            # at construction time rather than as silent "always
            # expired" ModeStates that nobody notices in production.
            if self.ttl_seconds < 0:
                raise ValueError(
                    f"ttl_seconds must be >= 0 (got {self.ttl_seconds!r}); "
                    "use 0 for persistent (no expiry)"
                )

            # R2-HIGH-5 fix: activated_at in the far future is almost
            # always a caller bug (wrong clock, off-by-one, units
            # confusion between seconds and milliseconds). Allow a 60s
            # clock-skew tolerance for systems with mildly unsynced
            # clocks; reject anything beyond that. Pinned by
            # `test_mode_state_rejects_future_activated_at` in
            # `test_mode_state.py` (added in sub-agent 5).
            if self.activated_at > time.time() + 60:
                raise ValueError(
                    f"activated_at is too far in the future "
                    f"({self.activated_at!r} > time.time() + 60); "
                    "check for clock skew or a caller bug"
                )
        finally:
            _in_mode_state_post_init.active = False

    def is_expired(self) -> bool:
        """Check TTL expiry. Returns False for persistent skills (ttl_seconds == 0)."""
        if self.ttl_seconds == 0:
            return False
        return time.time() - self.activated_at > self.ttl_seconds

    # NO refresh() method — removed in 1a (R1-HIGH-4 YAGNI).


# R2-HIGH-3 fix: install a custom __setattr__ on ModeState that
# permits writes only when the _in_mode_state_post_init thread-local
# flag is set. We set it programmatically (not in the class body)
# because the `@dataclass(frozen=True)` decorator on Python 3.9/3.10
# refuses to overwrite a class-body __setattr__ (`TypeError: Cannot
# overwrite attribute __setattr__`). Setting it after decoration
# replaces the frozen guard with our thread-gated variant. The
# auto-generated __init__ still uses `object.__setattr__` for its
# own field assignments (it never goes through this hook), so the
# thread-local flag is only required when __post_init__ performs
# post-construction mutations.
def _mode_state_setattr(self, name: str, value) -> None:
    if getattr(_in_mode_state_post_init, 'active', False):
        object.__setattr__(self, name, value)
    else:
        raise FrozenInstanceError(
            f"cannot assign to field {name!r}"
        )


ModeState.__setattr__ = _mode_state_setattr  # type: ignore[assignment]
