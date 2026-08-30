"""Generic mode resolver (Phase 1c).

Replaces the per-mode ticket/work branches in `proxy/mode.py:resolve_mode`
with one generic helper:

  - `resolve_mode(ctx, agent, registry)` — returns a `Mode` enum value.

The resolver is side-effect-aware: it clears stale modes (with the skill's
`on_deactivate` callback). It does NOT build `ModeState` objects for the
caller — that is the caller's responsibility (the resolver returns the
new `Mode` and the caller wraps it in a `ModeState` via
`ModeState.from_mode(new_mode, ts=...)`).
"""
import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from proxy.agent.agent import Agent
    from proxy.types import RequestContext
    from proxy.skills.skill_registry import SkillRegistry

logger = logging.getLogger(__name__)


def _mode_value(mode_enum) -> str:
    """Return the string value of a Mode enum member, robust to reloads.

    `enum.auto()` produces values that are NOT stable strings, so identity
    comparisons break under `importlib.reload(proxy.mode)`. Always compare
    by `.value` (the int), and serialize to a canonical string for return.
    """
    return str(mode_enum.value)


def resolve_mode(ctx: "RequestContext", agent: "Agent", registry: "SkillRegistry") -> str:
    """Resolve the current mode based on agent state + canary inheritance.

    Returns the canonical string form of the resolved `Mode` enum value
    (`SKILL`, `SKILL_SUBAGENT`, `SUBAGENT`, or `DEFAULT`). The string form
    is used (instead of the enum directly) so the function survives
    `importlib.reload(proxy.mode)` (which recreates the enum class and
    invalidates identity-based comparisons; see Phase 1c plan §3.2).
    """
    from proxy.mode import Mode

    ms_tuple = agent.snapshot_mode_and_ticket_ts()
    if ms_tuple is None:
        ms, _ = None, None
    else:
        ms, _ = ms_tuple

    # Stale-mode clearing (R2 HIGH-4: TOCTOU logic preserved).
    # If `ms` has expired, clear it and consult the registry for `on_deactivate`.
    if ms is not None and hasattr(ms, "is_expired") and ms.is_expired():
        # Capture the snapshot before clearing so we can consult the
        # skill's `on_deactivate` callback with the EXPIRED mode (not None).
        expired_ms = ms
        skill = registry.get(expired_ms.name) if expired_ms else None
        # R2 CRIT-3: split into two try blocks so a failing `on_deactivate`
        # does NOT skip `clear_active_mode`. The clear must always run to
        # avoid leaving the agent in an inconsistent state.
        if (
            skill is not None
            and hasattr(skill, "on_deactivate")
            and skill.on_deactivate is not None
        ):
            try:
                skill.on_deactivate(ctx, expired_ms)
            except Exception as e:
                logger.warning(
                    "resolve_mode: on_deactivate failed for %s: %s",
                    getattr(expired_ms, "name", "?"), e,
                )
        # Always clear, even if on_deactivate raised above.
        try:
            agent.clear_active_mode(expected_mode=ms)
        except Exception as e:
            logger.warning(
                "resolve_mode: clear_active_mode failed for %s: %s",
                getattr(ms, "name", "?"), e,
            )
        ms = None

    is_subagent = bool(getattr(ctx, "is_subagent", False))

    if not is_subagent:
        # UNREACHABLE in production: every canary branch that sets
        # inherited_skill also sets is_subagent=True (see canary.py
        # detect_subagent_ingress). The non-subagent inherited-skill
        # path is preserved for defensive completeness only.
        # Phase 2B: non-subagent inherited_skill must also activate
        # the skill (mirrors legacy resolve_mode _has_inherited_skill branch).
        inherited = getattr(ctx, "inherited_skill", None)
        if isinstance(inherited, str) and bool(inherited) and inherited != "workspace":
            skill = registry.get(inherited)
            if skill is not None:
                from proxy.agent.agent import _activate_skill
                from proxy.skills.skill_registry import SkillContext
                messages_snapshot = getattr(ctx, 'messages', [])
                skill_state = {}
                skill_ctx = SkillContext(
                    session_id=getattr(ctx, 'session_id', ''),
                    agent_id=getattr(agent, 'agent_id', ''),
                    messages=messages_snapshot,
                    skill_state=skill_state,
                )
                try:
                    _activate_skill(agent, skill, skill_ctx, ctx)
                except (ValueError, PermissionError):
                    raise
                except Exception as e:
                    logger.warning(
                        "resolve_mode: non-subagent _activate_skill failed for %s: %s",
                        inherited, e,
                    )
                return _mode_value(Mode.SKILL)
        if ms is not None:
            return _mode_value(Mode.SKILL)
        return _mode_value(Mode.DEFAULT)

    # Sub-agent path
    inherited = getattr(ctx, "inherited_skill", None)
    if inherited:
        if inherited == "workspace":
            # R3 CRIT-5: workspace is parser-only; routing goes via
            # `ctx.workspace_name`, NOT via skill activation.
            # Workspace is parser-only; routing goes via ctx.workspace_name.
            # Return SKILL_SUBAGENT so the sub-agent inherits the skill-mode prompt layer.
            return _mode_value(Mode.SKILL_SUBAGENT)
        skill = registry.get(inherited)
        if skill is not None:
            from proxy.agent.agent import _inherit_skill
            from proxy.skills.skill_registry import SkillContext

            task_hash = getattr(ctx, "inherited_task_hash", None) or ""
            if not task_hash:
                logger.warning(
                    "resolve_mode: sub-agent inherited skill '%s' but "
                    "canary carried no task_hash — inheriting without "
                    "task context",
                    inherited,
                )

            messages_snapshot = getattr(ctx, 'messages', [])
            skill_ctx = SkillContext(
                session_id=getattr(ctx, 'session_id', ''),
                agent_id=getattr(agent, 'agent_id', ''),
                messages=messages_snapshot,
                # KEY TRANSLATION: canary "task_hash" → ModeState "arg"
                # (matches dispatch.py:445 and skill_dispatcher.py:135).
                skill_state={"arg": task_hash},
                is_subagent=True,
            )
            try:
                _inherit_skill(agent, skill, skill_ctx, ctx)
            except Exception as e:
                logger.warning(
                    "resolve_mode: sub-agent _inherit_skill failed for %s: %s",
                    inherited, e,
                )
            return _mode_value(Mode.SKILL_SUBAGENT)
        # Unknown inherited skill — fail-closed (R4 CRIT-2). Bare sub-agent.
        return _mode_value(Mode.SUBAGENT)

    return _mode_value(Mode.SUBAGENT)


__all__ = ["resolve_mode"]