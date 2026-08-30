"""Generic sub-agent guard (Phase 1c).

Replaces the `_check_hierarchy` skill-name branches in
`proxy/tool_pipeline/tool_guard.py:381` with a generic helper:

  - `check_sub_agent_proxy_access(tool_call, ctx, agent, registry)` —
    decides whether a sub-agent in the active skill can invoke a
    `proxy_*` tool (or any tool that requires a skill-level action check).

The function returns `Optional[str]` — `None` when allowed, a non-None
error string when denied. Returning a string (NOT raising) lets the
caller attach the error to the tool-call result without exception
plumbing (R1-CRIT-3).

CAT-3 trust gate (N4 CRIT): `is_trusted_skill` and `ProxyAccessLevel` are
imported at the TOP of the module as PUBLIC names so a missing import is
caught at module-load time, not at request time. User-defined skills
(name NOT in `TRUSTED_BUILTIN_SKILLS`) cannot grant `proxy_*` to sub-agents.
"""
import logging
from typing import Optional, TYPE_CHECKING

# R4 N4 CRIT: `is_trusted_skill` MUST be a direct public import, not
# transitive. Missing-import is caught at module load, not at first call.
from proxy.skills.skill_definition import is_trusted_skill, ProxyAccessLevel

if TYPE_CHECKING:
    from proxy.agent.agent import Agent
    from proxy.types import RequestContext
    from proxy.skills.skill_registry import SkillRegistry

logger = logging.getLogger(__name__)


def check_sub_agent_proxy_access(
    tool_call, ctx, agent: "Agent", registry: "SkillRegistry"
) -> Optional[str]:
    """Check if a sub-agent can invoke a `proxy_*` tool.

    Returns `None` if allowed, error string if denied. The error string
    is suitable for attaching to the tool-call result (no exception
    plumbing needed).

    Sole gate for ALL `proxy_*` tools on sub-agents:

      - Main agent (not a sub-agent) → allowed (other guards handle it).
      - Bare sub-agent (no active mode) → default-deny ALL `proxy_*`.
      - Active sub-agent in an untrusted (user-defined) skill →
        default-deny ALL `proxy_*` (CAT-3 trust gate).
      - Active sub-agent in a trusted builtin skill → per-tool access
        level check (proxy_ticket READ_ONLY/NEVER, proxy_work NEVER),
        then the skill's `check_action` callback for any further gating.
    """
    name = getattr(tool_call, "name", "")
    args = getattr(tool_call, "arguments", {}) or {}
    action = args.get("action", "") if isinstance(args, dict) else ""
    # Slice 8a: derive action for decomposed ticket tools
    if action == "" and name.startswith("proxy_ticket_") and name != "proxy_ticket":
        from memory.tools.dispatch import TICKET_TOOL_OPERATIONS
        action = TICKET_TOOL_OPERATIONS.get(name, "")

    is_subagent = bool(getattr(ctx, "is_subagent", False))
    if not is_subagent:
        return None

    if agent is None:
        return "Authorization failed: agent not found for session"

    is_proxy_tool = name.startswith("proxy_")
    if not is_proxy_tool:
        return None

    ms = agent.snapshot_mode()
    if ms is None:
        # Bare sub-agent (no inherited skill) — default-deny ALL
        # `proxy_*`. The proxy_* namespace requires an active skill.
        return (
            f"Sub-agents without an inherited trusted skill cannot call proxy tools. "
            f"Complete your task with your available tools "
            f"and return your findings to the orchestrator."
        )

    skill = registry.get(ms.name)
    if skill is None:
        return f"Skill '{ms.name}' is not registered."

    # CAT-3 trust gate: user-defined skills cannot grant ANY `proxy_*`
    # to sub-agents. Only built-in trusted skills may grant `proxy_*`.
    if not is_trusted_skill(skill.name):
        return (
            f"User-defined skill '{skill.name}' cannot invoke "
            f"proxy_* tools."
        )

    # C7b: delegate to capability_table when available (behavior-neutral —
    # proven by tests/unit/test_guard_table_parity.py).
    if name in ("proxy_ticket", "proxy_work") or name.startswith("proxy_ticket_"):
        from proxy.skills.skill_definition import CapabilityTable
        table = getattr(skill, 'capability_table', None)
        if isinstance(table, CapabilityTable):
            return table.check_action(name, action, "subagent")
        # Legacy fallback: skills without a table (custom skills).
        if name == "proxy_ticket" or name.startswith("proxy_ticket_"):
            access = getattr(skill, "sub_agent_proxy_ticket_access", None)
            if access == ProxyAccessLevel.NEVER or access == "never":
                return (
                    f"Skill '{skill.name}' forbids proxy_ticket access for "
                    f"sub-agents."
                )
            if access == ProxyAccessLevel.READ_ONLY or access == "read_only":
                if action not in {"list_tasks", "get_task", "append_review_note"}:
                    return (
                        f"Skill '{skill.name}' grants read-only proxy_ticket; "
                        f"action '{action}' not allowed."
                    )
        if name == "proxy_work":
            access = getattr(skill, "sub_agent_proxy_work_access", None)
            if access == ProxyAccessLevel.NEVER or access == "never":
                return (
                    f"Skill '{skill.name}' forbids proxy_work access for "
                    f"sub-agents."
                )
        if hasattr(skill, "check_action") and skill.check_action is not None:
            try:
                result = skill.check_action(tool_call, ctx)
                if result is not None:
                    return result
            except Exception as e:
                return f"Skill '{skill.name}' check_action raised: {e}"

    return None


__all__ = ["check_sub_agent_proxy_access"]