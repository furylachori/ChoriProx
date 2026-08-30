"""Skill lifecycle management — IoC bridge between Agent and SkillRegistry.

Eliminates the Agent → SkillRegistry circular import by using callbacks.
"""
import logging

logger = logging.getLogger(__name__)


def create_deactivation_handler(agent):
    """Create a deactivation callback that notifies SkillRegistry.

    Returns a callable that accepts a ModeState and calls on_deactivate
    on the corresponding skill. The agent reference is captured in the
    closure to provide session_id and agent_id for SkillContext.
    """
    def _on_mode_deactivated(mode_state):
        try:
            from proxy.skills.skill_registry import SkillRegistry, SkillContext
            registry = SkillRegistry()
            skill = registry.get(mode_state.name)
            if skill is not None:
                # Wave 1 (spec §6): `allowed_tool_names` is three-state
                # (`None` | `frozenset()` | non-empty frozenset). The
                # deactivation snapshot must coerce `None` (unrestricted)
                # to an empty frozenset so the `_tools_snapshot` field —
                # which downstream code (e.g., sub-agent lifecycle hooks)
                # treats as a concrete frozenset — does not break. Under
                # the old semantics `None` meant "no restriction" and the
                # `frozenset()` default was the "empty filter" sentinel;
                # both are now distinct.
                _allowed = getattr(mode_state, 'allowed_tool_names', None)
                _tools_snapshot = (
                    _allowed if isinstance(_allowed, frozenset) else frozenset()
                )
                ctx = SkillContext(
                    session_id=agent.session_id,
                    agent_id=agent.agent_id,
                    messages=[],
                    skill_state=dict(mode_state.skill_state),
                    _tools_snapshot=_tools_snapshot,
                    _skill_activated=False,
                    raw_user_text="",
                )
                skill.on_deactivate(ctx)
        except Exception:
            logger.exception("Deactivation handler failed for %s",
                             getattr(mode_state, 'name', '?'))

    return _on_mode_deactivated


def register_lifecycle_callbacks(agent):
    """Register skill lifecycle callbacks on an Agent instance.

    Call this when an Agent is created to wire up deactivation handling.
    """
    agent.register_deactivation_callback(create_deactivation_handler(agent))
