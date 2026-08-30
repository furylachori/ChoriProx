"""Agent class - full ownership model.

Phase 5: Thin wrapper over existing chat_histories.
Phase 7: Full ownership with MessageBuffer, AgentConfig, parent link, lifecycle methods.
"""
from __future__ import annotations

import hashlib
import threading
import time
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Optional, List, ClassVar

from proxy.agent.agent_config import AgentConfig
from proxy.agent.message_buffer import MessageBuffer
from proxy.lock_ordering import make_lock
import logging as _logging

# Phase 3: set_ticket_mode was deleted (zero production callers). Callers
# must use set_active_mode() directly with a ticket ModeState.
# Type annotations referencing ModeState are strings (from __future__
# import annotations at line 6) and resolve at type-check time only.

_logger = _logging.getLogger("proxy.agent")


def _make_telemetry():
    from proxy.pipeline.agent_telemetry import AgentTelemetry
    return AgentTelemetry()


def _make_threshold_engine():
    from proxy.pipeline.threshold_engine import ThresholdEngine
    return ThresholdEngine()


def _activate_skill(agent, skill, skill_ctx, ctx=None) -> None:
    """Activate a skill on the agent (Phase B skill system).

    Activation contract:
      - `on_activate` runs first; its result is validated.
      - Security/policy exceptions (`ValueError`, `PermissionError`) from
        `on_activate` are re-raised so the caller can surface them.
      - Any other exception sets `activation_failed=True` and BLOCKS
        `agent.set_active_mode(...)`. Previously, the mode was installed
        even when `on_activate` raised — a fail-open privilege-escalation
        window. The `if not activation_failed:` guard below closes it.
      - A failed `set_active_mode` itself (e.g. lock contention) also
        blocks the install; the caller can roll back the previously
        snapshotted `expected_mode` if it already captured one.

    Note: the caller at `proxy/skill_resolver.py:107-115` already wraps
    this with a `(ValueError, PermissionError)` re-raise + generic
    `Exception` log swallow, so any non-security exception propagated
    from here is a violation of the spec - but we still defend in depth
    because activation-running-code paths bypass the resolver wrapper.
    """
    import time as _time
    from proxy.skills.skill_definition import ModeState
    mode_state = ModeState(
        name=skill.name,
        activated_at=_time.time(),
        ttl_seconds=skill.ttl_seconds,
        allowed_tool_names=skill.get_tool_filter(skill_ctx),
        skill_state={},
    )
    activation_failed = False
    if getattr(skill, "on_activate", None) is not None:
        try:
            skill.on_activate(skill_ctx, mode_state)
        except (ValueError, PermissionError):
            # Re-raise: these signal security/policy violations.
            raise
        except Exception as e:
            # Non-security exception: log + block installation entirely
            # (spec §7.3). Do NOT raise - let the caller continue with
            # the existing (unset) active_mode.
            _logger.warning(
                "skill.on_activate raised for %s: %s", skill.name, e
            )
            activation_failed = True
    # FAIL-CLOSED: only install the mode if on_activate did not fail AND
    # the skill_state finalization completes. The frozen ModeState guard
    # below also lives behind the `activation_failed` check.
    if not activation_failed:
        # R3-CRIT-8 fix: pre-1a `mode_state.skill_state = ...` raises
        # FrozenInstanceError after the 1a `frozen=True` rewrite. The
        # `object.__setattr__` bypass preserves the frozen guard for all
        # other writes; the `MappingProxyType` wrap preserves the CRIT-S.1
        # read-only invariant on the new value.
        object.__setattr__(mode_state, 'skill_state', MappingProxyType(dict(skill_ctx.skill_state)))
        try:
            agent.set_active_mode(mode_state)
        except Exception as e:
            _logger.warning(
                "agent.set_active_mode failed for %s: %s", skill.name, e
            )
            # Mode was not installed; nothing to roll back here. The
            # expected_mode snapshot (caller's responsibility) is still
            # pointing at the previous agent state, so no spurious
            # compare-and-clear would have been triggered.
            return None
        _logger.info("Skill activated: %s session=%s", skill.name, skill_ctx.session_id[:8])
        if ctx is not None:
            ctx._skill_activated = True


def _inherit_skill(agent, skill, skill_ctx, ctx=None) -> None:
    """Inherit a skill for a sub-agent WITHOUT calling on_activate.

    Builds ModeState, freezes skill_state, calls set_active_mode.
    Skips `on_activate` (top-level hook for validation/claiming).
    Pin: `tests/unit/test_skill_inheritance.py::test_inherit_skill_does_not_call_on_activate`.

    Per spec §7.3 the `set_active_mode` call is fail-closed: if it
    raises (e.g. lock contention), the call is logged and the mode
    is NOT installed. The pre-spec behavior of letting the exception
    propagate was fail-open at the caller (`proxy/skill_resolver.py:155`
    catches `Exception` and only logs), which left the agent in an
    indeterminate state. The guard below closes that window.
    """
    import time as _time
    from types import MappingProxyType
    from proxy.skills.skill_definition import ModeState

    sub_filter = getattr(skill, "sub_agent_tool_filter", frozenset())
    mode_state = ModeState(
        name=skill.name,
        activated_at=_time.time(),
        ttl_seconds=skill.ttl_seconds,
        allowed_tool_names=sub_filter,
        sub_agent_tool_filter=sub_filter,
        skill_state={},
    )
    object.__setattr__(
        mode_state, 'skill_state',
        MappingProxyType(dict(skill_ctx.skill_state))
    )
    try:
        agent.set_active_mode(mode_state)
    except Exception as e:
        _logger.warning(
            "agent.set_active_mode failed for inherited %s: %s", skill.name, e
        )
        # Sub-agent inheritance is best-effort; do NOT propagate. The
        # sub-agent continues with whatever mode the parent had set.
        return None
    _logger.info("Skill inherited: %s session=%s",
                 skill.name, skill_ctx.session_id[:8])


@dataclass
class Agent:
    """Owns messages, config, tools, parent link.
    
    Phase 7: Full Agent ownership model.
    
    Attributes:
        agent_id:        Deterministic identifier derived from session_id (sha256[:12]).
                         Spawned sub-agents may override with an explicit value.
        session_id:      The session this agent is associated with.
        parent:          Parent agent if this is a sub-agent (None for root).
        config:          AgentConfig with provider, tools, budget, etc.
        _buffer:         MessageBuffer for message storage (replaces chat_histories).
        active_mode:     Active ModeState from Phase B skill system (or None).
        _mode_lock:      Lock for thread-safe active_mode access.
        is_subagent:     Whether this agent was spawned as a sub-agent.
        _metadata:       Optional AgentMetadata dict.
    """
    # Per-instance buffer-replace lock. M4-1: was ClassVar[threading.Lock]
    # (shared across all agents — 100 agents serialized on one mutex).
    # Now field(default_factory=threading.Lock) gives each instance its
    # own lock. Intentionally outside state.py lock-ordering hierarchy —
    # this lock never interacts with any other lock in the system.
    _buffer_replace_lock: threading.Lock = field(
        default_factory=threading.Lock, repr=False, compare=False
    )
    
    agent_id: str = ""
    session_id: str = ""
    parent: Optional["Agent"] = None
    config: AgentConfig = field(default_factory=AgentConfig)
    is_subagent: bool = False
    _metadata: Optional[dict] = None
    # Reduce-only per-session context budget override (raw tokens). None =
    # model default. Set by `^^context <N>` command; read by ContextOptimizeStage
    # and resolved against the real model window in context.resolve_effective_context_limit.
    # Plain field, no lock: single Optional[int] scalar, atomic assignment;
    # worst-case race = one turn on a just-superseded budget, self-correcting.
    context_limit_override: Optional[int] = None

    _buffer: MessageBuffer = field(default_factory=MessageBuffer)
    # Phase B skill system fields
    active_mode: Optional[Any] = None  # ModeState or None
    _mode_lock: Optional[threading.Lock] = None
    _ticket_mode_ts: Optional[float] = None  # Legacy timestamp fallback
    # Prompt Injection Layer fields
    telemetry: Any = field(default_factory=_make_telemetry)
    threshold_engine: Any = field(default_factory=_make_threshold_engine)

    def __post_init__(self):
        """Initialize fields that need to be set after dataclass construction."""
        # Derive agent_id from session_id when not explicitly provided.
        # This ensures agent_id is deterministic and stable across LRU
        # eviction/reload cycles, fixing the assignment model.
        if not self.agent_id and self.session_id:
            self.agent_id = hashlib.sha256(self.session_id.encode()).hexdigest()[:12]
        if self._mode_lock is None:
            self._mode_lock = make_lock(7, "Agent._mode_lock", threading.Lock)
        self._deactivation_callbacks: List[Any] = []

    def clear_active_mode(self, expected_mode: Any = None) -> bool:
        """Clear the active skill mode (Phase B skill system).

        Uses Pattern A: snapshot state under lock, clear under lock,
        call on_deactivate() outside the lock. Safe for concurrent calls.

        Args:
            expected_mode: If provided, only clear if agent.active_mode
                is the same object (identity check). This prevents a race
                where a new mode is set between a staleness check and the
                clear. Pass the mode object snapshotted earlier.

        Returns:
            True if the mode was successfully cleared (CAS succeeded),
            False if another thread set a new mode (CAS failed).
        """
        with self._mode_lock:
            if expected_mode is not None and self.active_mode is not expected_mode:
                return False
            mode_state = self.active_mode
            self.active_mode = None
            self._ticket_mode_ts = None

        if mode_state is not None:
            self._fire_deactivation_callbacks(mode_state)

        return True

    @property
    def parent_agent_id(self) -> Optional[str]:
        """Get parent agent ID.

        Returns parent.agent_id if parent exists, otherwise None.
        Sub-agents always have ``self.parent`` set; the None fallback
        covers only the trivial "this agent has no parent" case.
        """
        if self.parent is not None:
            return self.parent.agent_id
        return None

    @property
    def messages(self) -> List[dict]:
        """Read messages from MessageBuffer. Thread-safe."""
        return self._buffer.to_list()

    @messages.setter
    def messages(self, value: List[dict]) -> None:
        """Write messages to MessageBuffer. Thread-safe.

        Acquires the per-instance _buffer_replace_lock to serialize
        buffer replacement with concurrent append_message()/
        extend_messages() calls, preventing messages from being
        appended to the old buffer after replacement.
        """
        with self._buffer_replace_lock:
            self._buffer = MessageBuffer(messages=list(value))

    def append_message(self, message: dict) -> None:
        """Append a single message to the buffer."""
        with self._buffer_replace_lock:
            self._buffer.append(message)

    def extend_messages(self, messages: List[dict]) -> None:
        """Extend buffer with multiple messages."""
        with self._buffer_replace_lock:
            self._buffer.extend(messages)

    @property
    def is_ticket_mode(self) -> bool:
        """Check if ticket mode is active (Phase B skill system).
        Returns True when active_mode is set, name is 'ticket',
        and TTL has not expired.
        Uses _mode_lock to prevent TOCTOU (CON-007).
        """
        with self._mode_lock:
            mode = self.active_mode
            if mode is None:
                # Phase B fallback: check legacy timestamp for edge cases
                if self._ticket_mode_ts is not None:
                    return True
                return False
            if getattr(mode, 'name', None) != 'ticket':
                return False
            # SEC-001: Check TTL
            if mode.is_expired():
                return False
            return True

    @property
    def is_skill_mode(self) -> bool:
        """Check if any skill mode is active and not expired.
        Generic version of is_ticket_mode that works for all skills.
        """
        with self._mode_lock:
            mode = self.active_mode
            if mode is None:
                return False
            if mode.is_expired():
                return False
            return True

    @property
    def is_workspace_mode(self) -> bool:
        """Check if workspace mode is active (Phase B skill system).
        Workspace mode is persistent (ttl_seconds=0), so no TTL expiry check needed.
        """
        with self._mode_lock:
            mode = self.active_mode
            if mode is None:
                return False
            return getattr(mode, 'name', None) == 'workspace'

    def snapshot_mode(self) -> Optional[ModeState]:
        """Atomic read of active_mode under _mode_lock.

        Returns the current ModeState or None. Use this instead of
        directly accessing active_mode to ensure thread-safe reads.
        """
        with self._mode_lock:
            return self.active_mode

    def snapshot_mode_and_ticket_ts(self) -> tuple[Optional[ModeState], Optional[float]]:
        """Atomic multi-field read of mode and ticket timestamp.

        Eliminates TOCTOU between reading active_mode and _ticket_mode_ts.
        Returns (ModeState | None, float | None).
        """
        with self._mode_lock:
            return self.active_mode, getattr(self, '_ticket_mode_ts', None)

    def get_mode_name(self) -> Optional[str]:
        """Convenience: returns the active mode name or None.

        Replaces the common pattern of locking, reading active_mode,
        then extracting .name with getattr chains.
        """
        with self._mode_lock:
            ms = self.active_mode
            return ms.name if ms else None

    def is_mode_expired(self) -> bool:
        """Check if the active mode's TTL has expired.

        Returns False if no mode is active. Checks is_expired()
        on the ModeState under the lock for thread safety.
        """
        with self._mode_lock:
            ms = self.active_mode
            if ms is None:
                return False
            return ms.is_expired()

    def register_deactivation_callback(self, cb: Any) -> None:
        """Register a callback fired when the active mode is cleared or replaced.

        Enables Inversion of Control: callers can react to mode changes
        without the Agent needing to import their modules. Callbacks are
        fired outside the _mode_lock to prevent deadlocks.

        Args:
            cb: Callable that accepts a single ModeState argument.
        """
        if not hasattr(self, '_deactivation_callbacks'):
            self._deactivation_callbacks = []
        self._deactivation_callbacks.append(cb)

    def _fire_deactivation_callbacks(self, old_mode):
        """Fire all registered deactivation callbacks for the given mode.

        Called outside _mode_lock to prevent deadlocks. Callbacks are
        registered at agent creation time via register_lifecycle_callbacks().
        """
        if old_mode is None:
            return
        for cb in getattr(self, '_deactivation_callbacks', []):
            try:
                cb(old_mode)
            except Exception:
                _logger.exception("Deactivation callback failed for %s",
                                  getattr(old_mode, 'name', '?'))

    # ---- Phase B skill system methods ----
    def set_active_mode(self, mode_state: Any, expected_mode: Any = None) -> bool:
        """Set active_mode under lock. Centralized setter for Phase B skill system.

        Atomically swaps old mode → new mode under a single lock acquisition.
        on_deactivate for the old mode runs outside the lock to prevent deadlock.

        Args:
            mode_state: The new ModeState to install (or None to clear).
            expected_mode: If provided, only swap if agent.active_mode is the
                same object (identity check). This prevents a race where two
                concurrent threads overwrite each other's mode. Returns False
                if the identity check fails. If not provided, always swaps.

        Returns:
            True if the swap succeeded, False if expected_mode was provided
            and did not match (stale).
        """
        import time as _time

        # Single atomic swap: snapshot old, set new, all under one lock.
        with self._mode_lock:
            if expected_mode is not None and self.active_mode is not expected_mode:
                return False
            old_mode = self.active_mode
            self.active_mode = mode_state
            if mode_state is not None and getattr(mode_state, 'name', None) == 'ticket':
                self._ticket_mode_ts = _time.time()
            else:
                self._ticket_mode_ts = None

        # Deactivate old mode outside lock to prevent deadlock.
        if old_mode is not None:
            self._fire_deactivation_callbacks(old_mode)

        return True

    def clear_ticket_mode(self) -> None:
        """Legacy wrapper. Calls clear_active_mode()."""
        self.clear_active_mode()

    def set_context_limit_override(self, tokens: Optional[int]) -> None:
        """Set the per-session context budget override (raw tokens)."""
        self.context_limit_override = tokens

    def get_context_limit_override(self) -> Optional[int]:
        """Return the per-session context budget override (raw tokens) or None."""
        return self.context_limit_override


