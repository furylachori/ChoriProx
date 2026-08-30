"""Workspace command set — atomic one-shot direct commands, no LLM, no mode.

Per US-066: `^^workspace <name>` is a local state mutation handled in
`handle_direct()`. There is NO workspace mode — activating it was a
scaffolded feature that was never built. The LLM never sees a system
prompt for workspace mode; the skill system prompt is never read.

This module keeps:
- All `^^workspace <subcommand>` direct commands (switch, create,
  forget, recent, list, status, leave, help) — US-066 through US-072.
- `parse_subcommand` and `handle_direct` (the actual code path).
- `get_inheritance_state` for sub-agent workspace propagation via
  the canary.
- Workspace name validation.

This module removes (the mode scaffold):
- `get_tool_filter`, `get_system_prompt` — never called.
- `on_activate`, `on_deactivate`, `should_deactivate`,
  `_deactivate_workspace` — mode-activation paths.
- `get_canary` — only used in conjunction with mode activation;
  the canary parsing in `canary.py` is retained for backwards
  compat with any client that injects `[[__SUBAGENT__:WORKSPACE:...]]`
  directly.

If a future feature needs an actual workspace mode (e.g. multi-step
workspace setup with tool calls), it should be built fresh — not
resurrected from this scaffold.
"""
from __future__ import annotations

import logging
import re
from difflib import get_close_matches
from typing import Optional

from proxy import state
from proxy.workspace_state import workspace_store, resolve_workspace, DEFAULT_WORKSPACE
from proxy.skills.command_text_utils import (
    extract_command_text,
    strip_environment_details,
)
from proxy.skills.skill_definition import BaseSkill, SkillRequestContext
from proxy.skills.skill_registry import (
    SkillActivation,
    SkillMeta,
    _normalize_text,
)

logger = logging.getLogger("skills.workspace")

# Valid workspace name pattern (alphanumeric, hyphens only)
_WORKSPACE_NAME_PATTERN = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9-]*$")
_MAX_WORKSPACE_NAME_LENGTH = 64
_RESERVED_WORKSPACE_NAMES = {"leave", "default", "recent", "help", "list", "status", "admin", "system", "all", "create", "forget", "cleanup"}

# Subcommand constants for parse_subcommand
_SUBCMD_LEAVE = "leave"
_SUBCMD_RECENT = "recent"
_SUBCMD_HELP = "help"
_SUBCMD_LIST = "list"
_SUBCMD_STATUS = "status"
_SUBCMD_SWITCH = "switch"
_SUBCMD_CREATE = "create"
_SUBCMD_FORGET = "forget"
_SUBCMD_CLEANUP = "cleanup"
# Typo-guard threshold. SequenceMatcher.ratio() returns 2*M/T where M is matches
# and T is total chars. 0.7 catches single-char typos (insert/delete/substitute)
# and transpositions for workspace names of length 4-15. Higher = stricter.
_FUZZY_MATCH_CUTOFF = 0.7


def _is_valid_workspace_name(name: str) -> bool:
    """Validate workspace name format."""
    if not name or len(name) > _MAX_WORKSPACE_NAME_LENGTH:
        return False
    if name in _RESERVED_WORKSPACE_NAMES:
        return False  # Reserved keyword
    return bool(_WORKSPACE_NAME_PATTERN.match(name))


class WorkspaceSkill(BaseSkill):
    """Implements the `^^workspace` direct-command set (no LLM, no mode).

    Per US-066: every subcommand is handled in `handle_direct()` and
    returns immediately. There is no skill activation, no system prompt,
    no tool filter. This class exists to register the trigger and
    dispatch to local state mutations.

    Phase 1d: extends NEW `BaseSkill` from `proxy.skills.skill_definition`.
    Workspace is parser-only (never registered via `populate_builtins()`),
    so the 23-field `SkillDefinition` inventory is NOT populated here.
    The class body provides the minimum (name, trigger, aliases) needed
    for the NEW `BaseSkill.to_skill_definition()` validation; downstream
    callers (dispatcher, canary parser) use the parse/handle_direct path,
    not the SkillDefinition registry path.
    """

    # ---- Identity (1-3) — minimum needed for NEW BaseSkill validation ----
    name: str = "workspace"
    trigger: str = "^^workspace"
    # Wave 1: short human-readable description (≤160 chars). Surfaced by
    # `^^help`. WorkspaceSkill is NOT registered via populate_builtins so
    # this description is currently metadata-only — kept for forward
    # compatibility with future workspace-mode work.
    description: str = "Switch between workspace contexts (^^workspace — direct commands, no LLM)."
    # R1-HIGH-2 fix: the `aliases` field is validated against `^[a-z][a-z0-9_-]*$`
    # in SkillDefinition.__post_init__ (skill_definition.py:497). The trigger-prefix
    # form ("^^ws") does NOT match that regex (it starts with `^^`), so it would
    # raise ValueError when WorkspaceSkill.to_skill_definition() is called. Use a
    # SEPARATE field name (`_parser_aliases`) so the SkillDefinition-level
    # validator sees `aliases=()` (the BaseSkill default) while the parser-only
    # code paths below (get_meta, parse_subcommand) still resolve the trigger
    # prefix aliases correctly.
    _parser_aliases: tuple = ("^^ws",)

    # ttl_seconds: 0 — workspace mode is never activated (US-066/073).

    # -------------------------------------------------------------------
    # 1a compat: `get_meta()` is read by code that does
    # `getattr(skill, 'get_meta', lambda: None)()`. Kept for backwards
    # compat even though `populate_builtins()` does NOT register workspace.
    # -------------------------------------------------------------------
    def get_meta(self) -> SkillMeta:
        return SkillMeta(
            name=self.name,
            trigger=self.trigger,
            aliases=self._parser_aliases,
            activation=SkillActivation.COMMAND,
            ttl_seconds=0,
            description="Switch between workspace contexts (direct command, no LLM)",
            help_text="^^workspace <name> switches or creates a workspace. Use 'create'/'forget' for explicit ops. See 'help' for details.",
        )

    def parse_subcommand(self, text: str, session_id: str = "") -> tuple[Optional[str], str]:
        """Parse workspace command from text.

        Returns (action, arg) where action is one of the direct subcommands.
        For 'switch', returns (None, "") to defer to handle_direct (which
        also handles 'switch' directly).
        """
        text = _normalize_text(text)

        # --- Fast path ---
        # Check primary trigger and all aliases
        matched_trigger = None
        if text.startswith(self.trigger):
            matched_trigger = self.trigger
        else:
            for alias in self._parser_aliases:
                if text.startswith(alias):
                    matched_trigger = alias
                    break

        if matched_trigger:
            stripped = text[len(matched_trigger):].strip()
            stripped = strip_environment_details(stripped) if stripped else stripped
            if not stripped:
                return ("status", "")
            return self._dispatch_subcommand(stripped)

        # --- Slow path: XML-stripped fallback (size-guarded) ---
        if len(text) > 2048:
            return (None, "")

        cleaned_text = extract_command_text(text)

        if cleaned_text == text:
            return (None, "")  # Stripping didn't change anything

        # Re-check trigger/aliases on cleaned text
        if cleaned_text.startswith(self.trigger):
            matched_trigger = self.trigger
        else:
            for alias in self._parser_aliases:
                if cleaned_text.startswith(alias):
                    matched_trigger = alias
                    break
            else:
                return (None, "")

        stripped = cleaned_text[len(matched_trigger):].strip()
        return self._dispatch_subcommand(stripped)

    def _dispatch_subcommand(self, stripped: str) -> tuple[str, str]:
        """Parse the stripped remainder into (action, arg)."""
        if not stripped:
            return ("status", "")

        parts = stripped.split(None, 1)
        subcmd = parts[0].lower()
        arg = parts[1] if len(parts) > 1 else ""

        if subcmd == _SUBCMD_LEAVE:
            return (_SUBCMD_LEAVE, "")
        if subcmd == _SUBCMD_RECENT:
            return (_SUBCMD_RECENT, "")
        if subcmd == _SUBCMD_HELP:
            return (_SUBCMD_HELP, "")
        if subcmd == _SUBCMD_LIST:
            return (_SUBCMD_LIST, "")
        if subcmd == _SUBCMD_STATUS:
            return (_SUBCMD_STATUS, "")

        if subcmd == _SUBCMD_CREATE:
            return (_SUBCMD_CREATE, arg)
        if subcmd == _SUBCMD_FORGET:
            return (_SUBCMD_FORGET, arg)
        if subcmd == _SUBCMD_CLEANUP:
            return (_SUBCMD_CLEANUP, arg)

        return (_SUBCMD_SWITCH, stripped)

    def handle_direct(self, ctx: SkillRequestContext, action: str, arg: str) -> Optional[dict]:
        """Handle workspace subcommands without activating skill mode.

        All subcommands are atomic one-shot state mutations. No LLM call.
        """
        from proxy.skills.skill_registry import error_response

        try:
            if action == "status":
                return self._show_status(ctx)
            if action == "recent":
                return self._show_recent(ctx)
            if action == "help":
                return self._show_help(ctx)
            if action == "list":
                return self._show_list(ctx)
            if action == "leave":
                return self._handle_leave(ctx)
            if action == _SUBCMD_SWITCH:
                return self._handle_switch(ctx, arg)
            if action == _SUBCMD_CREATE:
                return self._handle_create(ctx, arg)
            if action == _SUBCMD_FORGET:
                return self._handle_forget(ctx, arg)
            if action == _SUBCMD_CLEANUP:
                return self._handle_cleanup(ctx)
        except Exception:
            logger.exception("handle_direct(%s) failed for session %s",
                            action, str(ctx.session_id or "unknown")[:8])
            return error_response(f"Workspace command '{action}' failed. Please try again.", 500)

        # switch → defer to activation (kept for backwards compat with dispatch flow)
        return None

    def _handle_leave(self, ctx: SkillRequestContext) -> dict:
        """Handle leave: clear active workspace, return to default.

        TOCTOU fix: the entire read-check-write sequence is protected by
        workspace_store.lock to prevent a race where another thread
        activates a different workspace between the read and the write.
        """
        from commands import make_command_response
        from proxy.skills.skill_registry import error_response

        # TOCTOU fix: wrap read-check-write in lock
        with workspace_store.lock:
            current = workspace_store.get_active(
                ctx.session_id, client_hint=ctx.client_hint
            )
            if current == DEFAULT_WORKSPACE:
                return make_command_response("Already at default workspace.")

            # Just clear the workspace. No mode to deactivate (no mode is ever active).
            workspace_store.deactivate(ctx.session_id, client_hint=ctx.client_hint)
            logger.info(
                "Workspace deactivated: session=%s (was: %s)",
                ctx.session_id[:8], current,
            )
            return make_command_response("Returned to default workspace.")

    def _handle_switch(self, ctx: SkillRequestContext, workspace_name: str) -> dict:
        """Switch to an existing workspace. Does NOT auto-create — use create subcommand for that.
        Local state mutation, no LLM, no mode activation.

        TOCTOU fix: the read-check-write sequence (get_active, get_global_catalog,
        get_recent, then activate) is protected by workspace_store.lock to prevent
        a race where another thread changes the active workspace or recent list
        between the read and the activate() call.
        """
        from commands import make_command_response
        from proxy.skills.skill_registry import error_response

        if not workspace_name:
            return make_command_response(f"Current workspace: {workspace_store.get_active(ctx.session_id, client_hint=ctx.client_hint)}")

        normalized = workspace_name.lower()
        if not _is_valid_workspace_name(normalized):
            return error_response(
                f"Invalid workspace name: '{workspace_name}'. "
                "Use letters, numbers, and hyphens (max 64 chars, not reserved).",
                400,
            )

        # v5: check against global catalog + per-session MRU.
        # Bug #12 / F2: fetch the global catalog BEFORE acquiring
        # workspace_store.lock. get_global_catalog performs the Neo4j
        # round-trip outside its own lock internally, but that release is a
        # no-op when the caller already holds this RLock — so fetching under
        # the outer lock would still block ALL sessions. Pulling the fetch
        # out of the critical section keeps the lock held only for the
        # in-memory read-check-write. No lock-ordering risk: get_global_catalog
        # acquires only this same #6 lock internally.
        catalog = workspace_store.get_global_catalog()
        # TOCTOU fix: wrap the in-memory read-check-write in lock
        with workspace_store.lock:
            active = workspace_store.get_active(ctx.session_id, client_hint=ctx.client_hint)
            recent = workspace_store.get_recent(ctx.session_id)
            # Build candidates: catalog + recent + active (in case catalog is stale)
            existing = set(catalog) | set(recent) | {active} - {"default"}

            if normalized in existing:
                if normalized == active:
                    return make_command_response(f"Already in workspace: {normalized}")
                old, _ = workspace_store.activate(ctx.session_id, normalized, client_hint=ctx.client_hint)
                return make_command_response(f"Switched to workspace: {normalized} (was: {old})")

        # Not found — check for typos, then suggest creation
        close = get_close_matches(normalized, list(existing), n=3, cutoff=_FUZZY_MATCH_CUTOFF)

        if close:
            hint = f"Did you mean: {', '.join(close)}? "
        else:
            recent_list = [w for w in catalog if w != active][:5] if catalog else []
            if recent_list:
                hint = f"Available workspaces: {', '.join(recent_list)}. "
            else:
                hint = ""

        return error_response(
            f"Workspace '{normalized}' does not exist. "
            f"{hint}"
            f"Use `^^workspace create {normalized}` to create it.",
            404,
        )

    def _handle_create(self, ctx: SkillRequestContext, workspace_name: str) -> dict:
        """Explicitly create a workspace. Bypasses typo guard. Local, no LLM."""
        from commands import make_command_response
        from proxy.skills.skill_registry import error_response

        if not workspace_name:
            return error_response("Usage: ^^workspace create <name>", 400)

        normalized = workspace_name.lower()
        if not _is_valid_workspace_name(normalized):
            return error_response(
                f"Invalid workspace name: '{workspace_name}'.",
                400,
            )

        old, _ = workspace_store.activate(ctx.session_id, normalized, client_hint=ctx.client_hint)
        if old == normalized:
            return make_command_response(f"Already in workspace: {normalized}")
        return make_command_response(f"Created and switched to workspace: {normalized} (was: {old})")

    def _handle_forget(self, ctx: SkillRequestContext, workspace_name: str) -> dict:
        """Remove a workspace from MRU. Refuses if it's the active workspace. Local, no LLM.

        TOCTOU fix: the read-check-write sequence (get_active then forget) is
        protected by workspace_store.lock to prevent a race where another thread
        activates the workspace between the check and the forget() call.
        """
        from commands import make_command_response
        from proxy.skills.skill_registry import error_response

        if not workspace_name:
            return error_response("Usage: ^^workspace forget <name>", 400)

        normalized = workspace_name.lower()
        if not _is_valid_workspace_name(normalized):
            return error_response(
                f"Invalid workspace name: '{workspace_name}'.",
                400,
            )

        # TOCTOU fix: wrap read-check-write in lock
        with workspace_store.lock:
            if workspace_store.get_active(ctx.session_id, client_hint=ctx.client_hint) == normalized:
                return error_response(
                    f"Cannot forget the active workspace '{normalized}'. "
                    f"Use '^^workspace leave' or '^^workspace <other>' first.",
                    400,
                )

            removed = workspace_store.forget(ctx.session_id, normalized)
        if removed:
            return make_command_response(f"Forgot workspace: {normalized}")
        return make_command_response(f"Workspace '{normalized}' was not in your recent list.")

    def _handle_cleanup(self, ctx: SkillRequestContext) -> dict:
        """Delete empty workspaces (no live tasks, no bugs) from the global catalog.

        Protects 'default' and the caller's active workspace. Removes the
        :Workspace nodes and their :Session/:LAST_USED_WORKSPACE edges.
        Invalidates the in-memory catalog cache so the next list reflects
        the cleanup.
        """
        from commands import make_command_response
        from proxy.skills.skill_registry import error_response
        from memory.workspace_persistence import cleanup_empty_workspaces

        active = resolve_workspace(ctx)
        protect = [active] if active and active != DEFAULT_WORKSPACE else []
        deleted_count, deleted_names = cleanup_empty_workspaces(protect)

        # Invalidate catalog cache so the next `^^workspace list` is fresh.
        with workspace_store.lock:
            workspace_store._catalog_cache = []
            workspace_store._catalog_cache_ts = 0.0

        if deleted_count == 0:
            return make_command_response("No empty workspaces to clean up.")
        return make_command_response(
            f"Cleaned up {deleted_count} empty workspace(s): {', '.join(deleted_names)}"
        )

    def _show_status(self, ctx: SkillRequestContext) -> dict:
        """Show current workspace status."""
        from commands import make_command_response
        current = workspace_store.get_active(ctx.session_id, client_hint=ctx.client_hint)
        return make_command_response(f"Current workspace: {current}")

    def _show_recent(self, ctx: SkillRequestContext) -> dict:
        """Show recent workspaces for this session."""
        from commands import make_command_response
        recent = workspace_store.get_recent(ctx.session_id)
        active = workspace_store.get_active(ctx.session_id, client_hint=ctx.client_hint)
        if not recent:
            return make_command_response("No recent workspaces.")
        lines = ["Recent workspaces:"]
        for i, name in enumerate(recent, 1):
            marker = " ← current" if name == active else ""
            lines.append(f"  {i}. {name}{marker}")
        return make_command_response("\n".join(lines))

    def _show_help(self, ctx: SkillRequestContext) -> dict:
        """Show workspace help."""
        from commands import make_command_response
        help_text = """Workspace Commands:
  ^^workspace <name>        Switch to an existing workspace
  ^^workspace create <name> Create and switch to a new workspace
  ^^workspace forget <name> Remove a workspace from your recent list
  ^^workspace cleanup       Delete empty workspaces (no tasks/bugs) from the catalog
  ^^workspace recent        List recent workspaces (MRU)
  ^^workspace list          List all available workspaces (global catalog)
  ^^workspace status        Show current workspace
  ^^workspace leave         Return to default workspace
  ^^workspace help          Show this help

Tip: typo guard prevents creating workspaces similar to existing ones.
     Use '^^workspace create <name>' to bypass."""
        return make_command_response(help_text)

    def _show_list(self, ctx: SkillRequestContext) -> dict:
        """List all workspaces for this session."""
        from commands import make_command_response
        # v5: show global catalog
        workspaces = workspace_store.get_global_catalog()
        current = workspace_store.get_active(ctx.session_id, client_hint=ctx.client_hint)
        lines = ["Workspaces:"]
        for name in workspaces:
            marker = " ← current" if name == current else ""
            lines.append(f"  • {name}{marker}")
        return make_command_response("\n".join(lines))

    def get_inheritance_state(self, ctx: SkillRequestContext) -> Optional[dict]:
        """Return workspace state for sub-agents.

        Used by the canary flow to propagate the active workspace to
        spawned sub-agents. Pure data — no mode activation.
        """
        ws = workspace_store.get_active(ctx.session_id, client_hint=ctx.client_hint)
        return {
            "workspace_name": ws,
            "workspace_id": ws,
        }

    # ---- No-op stubs for the BaseSkill protocol ----
    # These methods are part of the BaseSkill protocol but are never called
    # for this skill because no LLM-driven mode activation occurs.

    def on_activate(self, ctx: SkillRequestContext, mode_state=None) -> None:
        """No-op. Workspace mode is never activated (US-066/073).

        NOTE: ctx type annotation is `SkillRequestContext` per Phase 1d
        plan §3.4. Runtime ctx is still OLD `SkillContext` until Phase 2.
        """
        logger.debug(
            "WorkspaceSkill.on_activate called — no-op (workspace mode is not "
            "a real mode; this should not happen in production)."
        )

    def on_deactivate(self, ctx: SkillRequestContext, mode_state=None) -> None:
        """No-op. Workspace mode is never deactivated (US-066/073)."""
        pass

    def should_deactivate(self, ctx: SkillRequestContext, finish_reason: Optional[str]) -> bool:
        """Always False. Workspace mode never activates."""
        return False
