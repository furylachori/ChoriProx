"""Generic skill dispatcher (Phase 1c).

Provides the `SkillDispatcher` class — the generic per-skill dispatcher that
replaces the per-skill `if/elif` chain in `pipeline/dispatch.py`. The
dispatcher resolves the skill from a text trigger, then delegates to the
skill's `handle_direct` / `parse_subcommand` / `on_activate` methods.

The companion dataclass `DispatchResult` describes what the caller should do:
either return the handled response, or fall through to the LLM with an
activation rewrite attached.

Side-effect-free: the dispatcher does NOT call `agent.set_active_mode` itself;
that is the responsibility of the separate `activate()` method (separation of
concerns per Phase 1c plan §3.1).
"""
import logging
import time
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from proxy.agent.agent import Agent
    from proxy.types import RequestContext
    from proxy.skills.skill_definition import SkillDefinition
    from proxy.skills.skill_registry import SkillRegistry

logger = logging.getLogger(__name__)


@dataclass
class DispatchResult:
    """Result of a dispatcher attempt.

    Attributes:
        handled:    True iff `skill.handle_direct` produced a response.
        response:   The dict to return when `handled=True` (an OpenAI-style
                    response shape).
        has_rewrite: True iff the dispatcher activated the skill and the
                     caller should run the activation rewrite.
        skill:      The matched `SkillDefinition` (set when `has_rewrite=True`
                     so the caller can apply `skill.get_activation_rewrite`).
        action:     The parsed subcommand action (from `parse_subcommand`).
        arg:        The parsed subcommand argument (from `parse_subcommand`).
        expected_mode:  The rollback anchor used by the downstream
                        rewrite-failure path (`dispatch.py:481`,
                        `dispatch_stage.py:96`) to compare-and-clear the
                        just-installed mode via
                        `agent.clear_active_mode(expected_mode=...)`.
                        Populated on every non-None result.

                        Semantics (spec §7.3):
                          - When activation SUCCEEDED (`has_rewrite=True`):
                            `expected_mode` is the **just-installed** mode
                            (snapshot captured AFTER `agent.set_active_mode`).
                            This is the value the rollback CAS needs to
                            match — the NEW mode IS the current mode, so
                            the CAS succeeds and the rollback clears it.
                          - When activation FAILED (`has_rewrite=False`,
                            activation branch did not install a mode):
                            `expected_mode` is the **pre-activation**
                            snapshot (captured BEFORE the activate call)
                            for diagnostic consistency; no rollback should
                            be needed because no mode was installed.
                          - `None` if no agent was provided or the
                            activation branch did not run.
    """
    handled: bool
    response: Optional[dict] = None
    has_rewrite: bool = False
    skill: Optional["SkillDefinition"] = None
    action: Optional[str] = None
    arg: Optional[str] = None
    expected_mode: Optional[Any] = None


class SkillDispatcher:
    """Generic skill dispatcher — consults the registry, not per-skill branches."""

    def __init__(self, registry: "SkillRegistry") -> None:
        self.registry = registry

    def detect_command(self, text: str) -> Optional["SkillDefinition"]:
        """Match `text` against all skill triggers, return the matched skill.

        R1-HIGH-10: sort candidates by `len(trigger)` DESC so longer triggers win.
        R4 MED-1: normalize zero-width characters via `_normalize_text`.
        R2-Issue 3 / HIGH fix: only match on EXACT trigger equality OR
        `trigger + " "` (trigger followed by a space/arg). Bare `startswith`
        would let `"^^workspace foo"` match `^^work`.
        """
        from proxy.skills.skill_registry import _normalize_text

        normalized = _normalize_text(text)
        candidates = self.registry.list_skills()
        candidates = sorted(
            candidates,
            key=lambda s: len(getattr(s, "trigger", "") or ""),
            reverse=True,
        )
        for skill in candidates:
            trigger = getattr(skill, "trigger", None)
            if trigger:
                if normalized == trigger or normalized.startswith(trigger + " "):
                    return skill
            for alias in getattr(skill, "aliases", ()) or ():
                if alias and (normalized == alias or normalized.startswith(alias + " ")):
                    return skill
        return None

    def dispatch(self, text: str, ctx, agent: "Agent") -> DispatchResult:
        """Dispatch a text message to the matching skill.

        Returns `DispatchResult(handled=False)` (NEVER `None`) when:
          - no trigger matches the text, OR
          - `parse_subcommand` returns `action=None`, OR
          - `handle_direct` returns `None`.

        On match + `handle_direct` returning a response: returns
        `DispatchResult(handled=True, response=...)`.

        On match + no direct handle: invokes `self.activate(...)` and returns
        `DispatchResult(handled=False, has_rewrite=True, skill=...)`.

        Per spec §7.3 the current `agent.active_mode` is snapshotted BEFORE
        activation so a downstream rewrite failure can compare-and-clear the
        snapshot precisely. The snapshot is attached to every returned
        `DispatchResult.expected_mode` (None if `agent is None` or no skill
        matched).

        Re-raises `PermissionError` and `ValueError` from skill internals
        so the caller can decide whether to surface them.
        """
        from proxy.skills.skill_definition import SkillRequestContext

        skill = self.detect_command(text)
        if skill is None:
            return DispatchResult(handled=False)

        skill_ctx = SkillRequestContext.from_request_context(ctx)

        # Try direct handle first
        try:
            action, arg = skill.parse_subcommand(text)
        except ValueError:
            raise

        # If parse_subcommand returns None (skill recognized but command not valid),
        # fall through with handled=False so the caller's run_dispatch loop can try
        # the next handler.
        if action is None:
            return DispatchResult(
                handled=False,
                action=action,
                arg=arg,
                expected_mode=None,
            )

        try:
            result = skill.handle_direct(skill_ctx, action, arg)
        except PermissionError:
            raise
        if result is not None:
            return DispatchResult(
                handled=True,
                response=result,
                skill=skill,
                action=action,
                arg=arg,
                expected_mode=None,
            )

        # Set skill_state on skill_ctx so get_activation_rewrite can access arg.
        # The attribute is set dynamically (SkillRequestContext has no field for it;
        # individual skills use getattr with defaults, so missing attr is safe).
        skill_ctx.skill_state = {"action": action, "arg": arg if arg is not None else ""}
        skill_ctx.raw_user_text = text

        # Spec §7.3: capture the PRE-ACTIVATION snapshot for diagnostic
        # purposes. The actual rollback anchor (attached to
        # DispatchResult.expected_mode) is captured AFTER activation below.
        # Pre-activation is kept only for the failed-activation path so the
        # caller has a stable reference; it is NOT the rollback CAS anchor.
        pre_activation_mode = agent.snapshot_mode() if agent is not None else None

        # Fall back to skill activation
        activated = self.activate(skill, ctx, agent, skill_state=skill_ctx.skill_state, raw_user_text=text)
        if not activated:
            return DispatchResult(
                handled=False,
                has_rewrite=False,
                skill=skill,
                action=action,
                arg=arg,
                expected_mode=pre_activation_mode,
            )
        # Spec §7.3: the rollback anchor IS the just-installed mode.
        # clear_active_mode's CAS compares `expected_mode` against the
        # current `agent.active_mode`. When `expected_mode` is the
        # just-installed mode (the NEW mode), the CAS matches and the
        # rollback clears it. Pre-fix, the anchor was the OLD (pre-activation)
        # mode, so the CAS always mismatched and the NEW mode leaked across
        # requests.
        installed_mode = agent.snapshot_mode() if agent is not None else None
        return DispatchResult(
            handled=False,
            has_rewrite=True,
            skill=skill,
            action=action,
            arg=arg,
            expected_mode=installed_mode,
        )

    def activate(self, skill: "SkillDefinition", ctx, agent: "Agent", skill_state: Optional[dict] = None, raw_user_text: str = "") -> bool:
        """Activate a skill. Returns True on success, False if activation failed.

        Per spec §7.3 the activation has a strict two-step ordering:
          1. ``skill.on_activate`` runs first; its result is validated.
             - Security/policy exceptions (``ValueError``, ``PermissionError``)
               are re-raised — they are the only path that propagates.
             - Any other exception is logged and sets ``activation_failed``.
          2. ``agent.set_active_mode(mode_state)`` runs ONLY IF step 1 did
             not fail. If it raises, the mode is NOT installed and the
             dispatcher returns ``has_rewrite=False``. Earlier code
             (pre-spec) called ``set_active_mode`` unconditionally; that
             "fail-open privilege-escalation window" is closed here.

        Separated from ``dispatch()`` so the pipeline can call them
        independently (Phase 1c plan §3.1: separation of concerns).
        """
        from proxy.skills.skill_definition import ModeState, SkillRequestContext

        skill_ctx = SkillRequestContext.from_request_context(ctx)
        if skill_state is not None:
            skill_ctx.skill_state = skill_state
        else:
            skill_ctx.skill_state = {}
        skill_ctx.raw_user_text = raw_user_text
        mode_state = ModeState(
            name=skill.name,
            activated_at=time.time(),
            ttl_seconds=getattr(skill, "ttl_seconds", 300),
            allowed_tool_names=getattr(skill, "allowed_tool_names", None),
            sub_agent_tool_filter=getattr(skill, "sub_agent_tool_filter", frozenset()),
        )
        activation_failed = False
        if hasattr(skill, "on_activate") and skill.on_activate is not None:
            try:
                skill.on_activate(skill_ctx, mode_state)
            except (ValueError, PermissionError):
                # Re-raise: these signal security/policy violations.
                raise
            except Exception as e:
                # Non-security exceptions from on_activate block activation
                # entirely (spec §7.3). Do NOT install the mode.
                logger.warning(
                    "skill.on_activate raised for %s: %s", skill.name, e
                )
                activation_failed = True
        # Thread skill_state into the frozen ModeState (after on_activate, which
        # may populate skill_ctx.skill_state further). Mirrors the pattern in
        # proxy.agent.agent._activate_skill — see R3-CRIT-8.
        # Only thread when activation did not fail; a ModeState whose
        # populate step is skipped here is discarded (not passed to
        # set_active_mode below).
        if not activation_failed:
            object.__setattr__(
                mode_state, 'skill_state', MappingProxyType(dict(skill_ctx.skill_state))
            )
            try:
                agent.set_active_mode(mode_state)
            except Exception as e:
                logger.warning(
                    "agent.set_active_mode failed for %s: %s", skill.name, e
                )
                # Mode was not installed; nothing to roll back here.
                # Caller (dispatch.py / dispatch_stage.py) is responsible
                # for comparing/clearing the expected_mode snapshot if it
                # already captured one before this call.
                activation_failed = True
        return not activation_failed


__all__ = ["SkillDispatcher", "DispatchResult"]