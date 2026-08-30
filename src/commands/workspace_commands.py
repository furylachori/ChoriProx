"""Workspace command handlers — `^^workspace` / `^^ws` (alias).

R2-Issue 3 restored the `^^workspace` / `^^ws` commands that were lost
during the skill consolidation. Rather than re-registering WorkspaceSkill
in the SkillRegistry (which would re-introduce the deliberately-removed
skill-mode activation path), these commands are registered directly in
COMMAND_REGISTRY and delegate to `WorkspaceSkill.handle_direct`.

WorkspaceSkill.handle_direct is a pure local-state mutation: it only
reads `ctx.session_id` and `ctx.client_hint` via `workspace_store` and
never touches the agent registry. The command dispatch path
(`try_command`, priority 4) runs in DispatchStage BEFORE
ContextOptimizeStage registers the agent, but that is safe here because
no agent is required.
"""
from __future__ import annotations

import logging

from commands import register
from proxy.skills.skill_definition import SkillRequestContext
from skills.workspace import WorkspaceSkill

logger = logging.getLogger("commands.workspace")

# Reuse a single WorkspaceSkill instance — it is stateless aside from
# the class-level trigger/alias metadata and the shared workspace_store.
_WORKSPACE_SKILL = WorkspaceSkill()

_HELP_TEXT = (
    "Manage workspaces: switch, create, forget, recent, list, status, leave, help. "
    "Usage: ^^workspace [list|status|recent|leave|help] | "
    "^^workspace <name> | ^^workspace create <name> | ^^workspace forget <name>"
)


def _build_skill_ctx(session_id: str, client_hint: str = "") -> SkillRequestContext:
    """Construct a minimal SkillRequestContext for WorkspaceSkill.handle_direct.

    WorkspaceSkill reads `session_id` and `client_hint` from the context
    (via workspace_store). Bug #7: ``client_hint`` is now threaded from
    CommandContext (set at dispatch.py build time from RequestContext), so
    ``^^ws <name>`` populates ``workspace_store._client_active`` rather
    than only session-level state. ``client_hint`` defaults to "" — the
    falsy-but-string value workspace_store.activate expects.
    """
    return SkillRequestContext(
        session_id=session_id,
        is_subagent=False,
        inherited_skill=None,
        subagent_ticket_mode=False,
        workspace_name="",
        client_hint=client_hint,
        body={},
    )


def _dispatch_workspace(ctx, args) -> dict:
    """Shared handler for ^^workspace and ^^ws (alias).

    `args` is the list of tokens after the command name (e.g.
    `["create", "myproject"]` for `^^workspace create myproject`). We
    re-join them and let WorkspaceSkill.parse_subcommand normalize the
    action/arg. Using parse_subcommand (instead of hand-parsing) keeps a
    single source of truth for subcommand dispatch, including the bare
    trigger → "status" default and the switch fallback.
    """
    # Reconstruct the trigger text so parse_subcommand can strip it.
    # parse_subcommand lowercases via _normalize_text and checks both the
    # primary trigger ("^^workspace") and the alias ("^^ws").
    remainder = " ".join(args)
    skill_ctx = _build_skill_ctx(ctx.session_id, ctx.client_hint)

    # parse_subcommand expects the full trigger text. Rebuild it from the
    # raw_text (which preserves the original alias used) rather than
    # guessing the trigger — this keeps the alias path identical to the
    # skill-registered path.
    action, arg = _WORKSPACE_SKILL.parse_subcommand(ctx.raw_text, ctx.session_id)
    if action is None:
        # parse_subcommand could not match (e.g. oversized/unsupported text).
        from commands import make_command_response
        return make_command_response(
            "Could not parse workspace command. Type `^^workspace help` for usage.",
            model="^^command",
        )

    resp = _WORKSPACE_SKILL.handle_direct(skill_ctx, action, arg)
    if resp is None:
        # handle_direct returns None only for the defunct "switch →
        # activate" backwards-compat path, which no longer fires because
        # _dispatch_subcommand maps bare names to ("switch", name) and
        # handle_direct handles switch directly. Guard anyway.
        from commands import make_command_response
        return make_command_response(
            f"Workspace command '{action}' was not handled. Type `^^workspace help`.",
            model="^^command",
        )
    return resp


@register("^^workspace", _HELP_TEXT)
def cmd_workspace(ctx, args):
    """`^^workspace` — manage workspaces (list/create/switch/forget/recent/status/leave/help)."""
    return _dispatch_workspace(ctx, args)


@register("^^ws", _HELP_TEXT)
def cmd_workspace_alias(ctx, args):
    """`^^ws` — alias for ^^workspace."""
    return _dispatch_workspace(ctx, args)