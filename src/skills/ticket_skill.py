"""Ticket skill — implements Skill protocol for task management via ^^ticket.

Phase 1d: extends NEW `BaseSkill` from `proxy.skills.skill_definition`.
Provides the 23 `SkillDefinition` fields as class attributes.
"""
from __future__ import annotations

import logging
import re
from typing import Any, Optional

from proxy.skills.skill_definition import BaseSkill, ModeState, ProxyAccessLevel, SkillRequestContext, SkillValidationError, TICKET_CAPABILITY_TABLE
from proxy.skills.command_text_utils import (
    extract_command_text,
    strip_environment_details,
)
from proxy.context import _get_task_graph_singleton
from proxy.workspace_state import workspace_store, resolve_workspace, DEFAULT_WORKSPACE
from proxy.skills.skill_registry import (
    SkillActivation,
    SkillMeta,
    _normalize_text,
    error_response,
)

logger = logging.getLogger("skills.ticket")

MAX_INHERITANCE_TASKS = 100


class TicketSkill(BaseSkill):
    """Skill implementing the ^^ticket command set (Phase 1d: NEW BaseSkill).

    Class attributes below populate the 23 `SkillDefinition` fields when
    `to_skill_definition()` is called (inherited from NEW `BaseSkill`).
    """

    # ---- Identity (1-3) ----
    name: str = "ticket"
    trigger: str = "^^ticket"
    aliases: tuple = ()
    # Wave 1: short human-readable description (≤160 chars). Surfaced by
    # `^^help` and the custom-JSON loader (Agent 2 parses it from JSON).
    description: str = "Create and manage tasks (^^ticket — per-turn planner with proxy_ticket_* tool access)."
    capability_table = TICKET_CAPABILITY_TABLE

    # ---- Sub-agent permissions (6, 7) — R1 CRIT-3 ----
    sub_agent_proxy_ticket_access: ProxyAccessLevel = ProxyAccessLevel.FULL
    sub_agent_proxy_work_access: ProxyAccessLevel = ProxyAccessLevel.NEVER

    # ---- TTL (8) ----
    ttl_seconds: int = 300

    # ---- Cross-cutting config (10, 11) ----
    # US-017a: ticket mode is non-interactive (fire-and-forget).
    fire_and_forget: bool = True
    # Default: block `question` tool for sub-agents (prevents ping-pong).
    blocked_tools_for_subagent: frozenset = frozenset({"question", "task"})

    # ---- Call limits (13) — R3 MED-16 ----
    call_limits: dict = {"proxy_ticket": 50}

    # `fallback_skill` (9): None — ticket has no fallback.
    # `check_action` (12): None — handled via tool_guard layer.

    # -------------------------------------------------------------------
    # 1a compat: `get_meta()` is read by code that does
    # `getattr(skill, 'get_meta', lambda: None)()`. The NEW BaseSkill does
    # not require it, but tests and downstream consumers still expect it.
    # -------------------------------------------------------------------
    def get_meta(self) -> SkillMeta:
        return SkillMeta(
            name=self.name,
            trigger=self.trigger,
            aliases=self.aliases,
            activation=SkillActivation.COMMAND,
            ttl_seconds=self.ttl_seconds,
            description="Create and manage tasks",
            help_text="Use ^^ticket <description> to plan tasks, ^^ticket to list tasks",
        )

    @staticmethod
    def _looks_like_hash_list(hashes_str: str) -> bool:
        """Return True if `hashes_str` contains at least one valid short hash token.

        A short hash is exactly 8 hex characters. If none of the comma-separated
        tokens match, the input is natural language (e.g. "the #4", "task 3") and
        the parser should fall through to plan mode so the LLM can interpret it.
        """
        from memory.task_graph import TaskGraph
        for token in hashes_str.split(","):
            token = token.strip().lstrip("#")
            if not token:
                continue
            if TaskGraph.SHORT_HASH_RE.fullmatch(token):
                return True
        return False

    @staticmethod
    def _plan_fallback(user_args: str) -> tuple[str, str]:
        """Build a (action, arg) tuple for the LLM plan-mode path.

        Preserves the FULL user text so the LLM sees the user's complete
        intent (e.g. "option 1" stays "option 1", "close the #4" stays
        "close the #4"). Only nested ^^command tokens are stripped.

        Stripping a leading word is lossy and was the root cause of the
        "option 1 -> 1" bug: the parser assumed the first word was always a
        planning verb, but for content like "option 1" the first word is the
        payload. The LLM interprets natural language from the complete input.
        """
        plan_arg = re.sub(r"\^\^\w+\s*", "", user_args)
        return ("plan", plan_arg)

    # -------------------------------------------------------------------
    # 19 — parse_subcommand
    # -------------------------------------------------------------------
    def parse_subcommand(self, text: str, session_id: str = "") -> tuple[Optional[str], str]:
        """Parse subcommand and argument from full trigger text.

        Returns (action, arg) where action is:
        - "list": bare trigger or bare "done"/"invalidate" → list tasks
        - "show": bare hash (exactly 8 hex chars) → show task info
        - "done": "done <hashes>" → batch complete
        - "invalidate": "invalidate <hashes>" → batch invalidate
        - "plan": everything else → LLM plan mode
        - None: not a ticket command

        Checks raw text first, falls back to XML-stripped detection
        only for short messages (CLI-wrapped trigger case).
        """
        from commands.task_commands import is_ticket_trigger

        text = _normalize_text(text)

        # --- Fast path ---
        if is_ticket_trigger(text):
            stripped = text[len(self.trigger):].strip()
            return self._parse_ticket_args(stripped)

        # --- Slow path: XML-stripped fallback (size-guarded) ---
        if len(text) > 2048:
            return (None, "")

        cleaned = extract_command_text(text)

        if cleaned == text:
            return (None, "")

        if not is_ticket_trigger(cleaned):
            return (None, "")

        stripped = cleaned[len(self.trigger):].strip()
        return self._parse_ticket_args(stripped)

    def _parse_ticket_args(self, user_args: str) -> tuple[str, str]:
        """Parse ticket subcommands from stripped argument text.

        Both fast and slow paths delegate here — ensures consistent parsing.
        """
        from memory.task_graph import TaskGraph

        if not user_args:
            return ("list", "")

        # XML stripping — user_args may still have trailing XML
        user_args = strip_environment_details(user_args)
        if not user_args:
            return ("list", "")

        user_lower = user_args.lower()

        # Bug subcommands (Phase 3) — check BEFORE bare-hash detection so
        # `bug <8hex>` is treated as bug_show, not show.
        if user_lower == "bug" or user_lower.startswith("bug "):
            bug_arg = user_args[4:].strip() if user_lower.startswith("bug ") else ""
            if not bug_arg:
                return ("bug_list", "")
            bug_tokens = bug_arg.split()
            bug_first = bug_tokens[0].lower()
            if bug_first == "list":
                return ("bug_list", "")
            if bug_first == "show" and len(bug_tokens) > 1:
                return ("bug_show", bug_tokens[1])
            if bug_first == "resolve" and len(bug_tokens) > 1:
                # rest = reason (may be empty)
                rest = " ".join(bug_tokens[2:])
                return ("bug_resolve", bug_tokens[1] + (("|" + rest) if rest else ""))
            if bug_first == "dismiss" and len(bug_tokens) > 1:
                rest = " ".join(bug_tokens[2:])
                return ("bug_dismiss", bug_tokens[1] + (("|" + rest) if rest else ""))
            return ("bug_list", "")

        # Exact-match bare subcommands (no args) → list
        if user_lower in ("done", "close", "invalidate", "reopen"):
            return ("list", "")

        if user_lower.startswith("done ") or user_lower.startswith("close "):
            prefix_len = 5 if user_lower.startswith("done ") else 6
            hashes = user_args[prefix_len:].strip()
            if not hashes:
                return ("list", "")
            if not self._looks_like_hash_list(hashes):
                return self._plan_fallback(user_args)
            return ("done", hashes)

        if user_lower.startswith("reopen "):
            hashes = user_args[7:].strip()
            if not hashes:
                return ("list", "")
            if not self._looks_like_hash_list(hashes):
                return self._plan_fallback(user_args)
            return ("reopen", hashes)

        if user_lower.startswith("invalidate "):
            hashes = user_args[11:].strip()
            if not hashes:
                return ("list", "")
            if not self._looks_like_hash_list(hashes):
                return self._plan_fallback(user_args)
            return ("invalidate", hashes)

        # Bare hash (exactly 8 hex chars) → show
        if user_args.strip() and TaskGraph.SHORT_HASH_RE.fullmatch(user_args.strip()):
            return ("show", user_args.strip())

        # Everything else → plan mode
        # Preserve the FULL user text so the LLM sees complete intent.
        # Stripping a leading word is lossy: "option 1" -> "1", "fix the
        # login bug" -> "the login bug". The LLM interprets natural language
        # from the complete input.
        return self._plan_fallback(user_args)

    # -------------------------------------------------------------------
    # 20 — handle_direct
    # -------------------------------------------------------------------
    def handle_direct(self, ctx: SkillRequestContext, action: str, arg: str) -> Optional[dict]:
        """Handle ticket subcommands without activating skill mode.

        Phase 2: uses action/arg parameters, handles list/show/done/invalidate.
        Phase 3: adds bug_list/bug_show/bug_resolve/bug_dismiss.
        Returns None for plan → triggers activation.

        NOTE: ctx type annotation is `SkillRequestContext` per Phase 1d
        plan §3.4. At runtime, callers (e.g. dispatch.py:try_skill_activation)
        still pass OLD `SkillContext` until Phase 2 cuts over. The field
        accesses below work for both types because they use `getattr` or
        access fields present on both (`session_id`, `client_hint`).
        """
        try:
            workspace = resolve_workspace(ctx)
            if workspace == DEFAULT_WORKSPACE:
                return error_response(
                    "Tickets live in a workspace — disabled on the default workspace. "
                    "Create one with `^^ws create <name>`, list existing with `^^ws list`, "
                    "or learn more with `^^ws help`.", 409)
            if action == "list":
                return self._list_tasks(ctx)
            if action == "show":
                return self._show_task(ctx, arg)
            if action == "done":
                return self._batch_status_update(ctx, arg, "complete")
            if action == "reopen":
                return self._batch_status_update(ctx, arg, "in_progress")
            if action == "invalidate":
                return self._batch_status_update(ctx, arg, "invalid")
            if action == "bug_list":
                return self._list_bugs(ctx)
            if action == "bug_show":
                return self._show_bug(ctx, arg)
            if action == "bug_resolve":
                return self._resolve_bug(ctx, arg, resolution="approved")
            if action == "bug_dismiss":
                return self._resolve_bug(ctx, arg, resolution="dismissed")

            # plan → defer to activation
            return None
        except Exception as e:
            logger.exception("handle_direct failed: action=%s error=%s", action, e)
            return error_response("An unexpected error occurred. Please try again.", 500)

    def _list_tasks(self, ctx: SkillRequestContext) -> dict:
        """List actionable tasks for this session."""
        from commands import make_command_response

        # F4: Use singleton to avoid creating new Neo4j driver/connection pool per call
        tg = _get_task_graph_singleton()
        if not tg.is_available():
            return error_response("Task store is unavailable. Cannot list tasks.", 503)

        # Derive workspace scope from current state
        workspace = resolve_workspace(ctx)
        workspace_id = workspace

        try:
            tasks = tg.get_actionable_tasks(workspace_id=workspace_id, limit=500)
        except Exception:
            logger.exception("get_actionable_tasks failed for session %s", ctx.session_id[:8])
            return error_response("Task store error. Cannot list tasks.", 500)

        MAX_LIST_TASKS = 500
        if len(tasks) > MAX_LIST_TASKS:
            logger.warning("_list_tasks: %d tasks exceeds cap %d — truncating", len(tasks), MAX_LIST_TASKS)
            tasks = tasks[:MAX_LIST_TASKS]

        if not tasks:
            return make_command_response("No active tasks for this session.")

        lines = ["Active Tasks:"]
        task_map = {t["short_hash"]: t for t in tasks}

        def _fmt_node(t: dict, number: int, depth: int = 0) -> str:
            tag = t.get("tag", "")
            tag_str = f" [{tag}]" if tag else ""
            b = " ⛔" if t.get("is_blocked") else ""
            indent = "  " * depth
            return (
                f"{indent}#{number} [{t.get('short_hash', '?')}] "
                f"{t.get('priority', '?').upper()} — "
                f"{t.get('name', '?')}{tag_str}{b}"
            )

        # children_map: O(n) linear scan over all tasks. Bounded by MAX_LIST_TASKS
        # cap above; without the cap this would be a per-request memory spike for
        # large workspaces. Tradeoff: tasks beyond the cap are silently dropped from
        # the tree rather than paginated, which is acceptable for a listing endpoint.
        children_map: dict = {}
        for t in tasks:
            parent = t.get("parent_short_hash")
            children_map.setdefault(parent, []).append(t)

        roots = [
            t for t in tasks
            if not t.get("parent_short_hash") or t.get("parent_short_hash") not in task_map
        ]

        _counter = [0]
        def _walk(tasks_list: list, depth: int = 0) -> None:
            for t in tasks_list:
                _counter[0] += 1
                lines.append(_fmt_node(t, _counter[0], depth))
                childs = children_map.get(t["short_hash"], [])
                childs.sort(
                    key=lambda x: (
                        {"high": 1, "medium": 2, "low": 3}.get(x.get("priority", "medium"), 2),
                        1 if x.get("is_blocked") else 0,
                        (x.get("name") or "").lower(),
                    ),
                )
                _walk(childs, depth + 1)

        roots.sort(
            key=lambda x: (
                {"high": 1, "medium": 2, "low": 3}.get(x.get("priority", "medium"), 2),
                1 if x.get("is_blocked") else 0,
                (x.get("name") or "").lower(),
            ),
        )
        _walk(roots)

        return make_command_response("\n".join(lines))

    def _show_task(self, ctx: SkillRequestContext, task_hash: str) -> dict:
        """Show full task info for a short hash.

        WARNING: get_task_by_short_hash returns RAW Neo4j properties without
        the enrichment pass (blockers, parent) that get_actionable_tasks applies.
        This handler manually computes blockers and parent after retrieval.
        """
        from commands import make_command_response
        from memory.task_graph import AmbiguousHashError

        hash = task_hash  # avoid shadowing Python builtin; internal alias for readability

        # Derive workspace scope
        workspace = resolve_workspace(ctx)
        workspace_id = workspace

        # F4: Use singleton to avoid creating new Neo4j driver/connection pool per call
        tg = _get_task_graph_singleton()

        if not tg.is_available():
            return error_response("Task store is unavailable. Cannot show task.", 503)

        # F3: Call get_task_enriched directly — it internally calls get_task_by_short_hash,
        # avoiding a redundant Neo4j query. AmbiguousHashError propagates from the internal call.
        try:
            task = tg.get_task_enriched(hash, workspace_id=workspace_id)
        except AmbiguousHashError as e:
            logger.warning("Ambiguous hash '%s' for workspace %s: %d tasks match",
                           hash, workspace_id, e.count)
            return error_response(
                f"Ambiguous hash '{hash}': {e.count} tasks match. "
                f"Use ^^ticket to find the task.", 409,
            )
        except Exception:
            logger.exception("get_task_enriched failed for hash %s", hash)
            return error_response("Task store error. Cannot show task.", 500)

        if task is None:
            return error_response(f"No task found with hash {hash}.", 200)

        # Use enriched data
        blockers = task.get("blockers", [])
        parent_short = task.get("parent_short_hash")
        parent_name = task.get("parent_name")

        # Fetch children via get_child_tasks
        children = []
        try:
            short_hash = task.get("short_hash")
            if short_hash:
                children = tg.get_child_tasks(short_hash, workspace_id=workspace_id)
            else:
                children = []
        except AmbiguousHashError as e:
            logger.warning("Ambiguous parent hash: %d tasks match", e.count)
            children = []  # show task without children
        except Exception:
            logger.exception("get_child_tasks failed for %s", task.get("uuid", "unknown"))
            children = []

        child_lines = []
        if children:
            for i, c in enumerate(children, 1):
                child_lines.append(
                    f"  #[{i}] [{c.get('short_hash', '?')}] — "
                    f"{c.get('priority', '?').upper()} — "
                    f"{c.get('status', '?')} — {c.get('name', '?')}",
                )

        # Format blockers per spec §6.2: "Blocked by: #[index] [{hash}] — {priority}"
        if blockers:
            blocker_strs = [
                f"#[{i+1}] [{b['short_hash']}] — {b.get('priority', '?').upper()}"
                for i, b in enumerate(blockers)
            ]
            blocker_text = ", ".join(blocker_strs)
        else:
            blocker_text = "None"

        lines = [
            f"Task [{hash}] — {task.get('priority', '?').upper()} — {task.get('status', '?')}",
            "─" * 40,
            f"Name    : {task.get('name', '(none)')}",
            f"Summary : {task.get('summary') or '(none)'}",
            f"Blocked by: {blocker_text}",
            f"Parent   : [{parent_short or 'None'}] — {parent_name or 'None'}",
            f"Created  : {task.get('created_at', '?')}",
            "─" * 40,
            "Attributes:",
            f"  planning_info: {task.get('planning_info') or '(none)'}",
            f"  review_notes : {task.get('review_notes') or '(none)'}",
        ]

        if child_lines:
            lines.append("─" * 40)
            lines.append("Children:")
            lines.extend(child_lines)
        else:
            lines.append("─" * 40)
            lines.append("Children: (none)")

        return make_command_response("\n".join(lines))

    def _batch_status_update(self, ctx: SkillRequestContext, hashes_str: str, new_status: str) -> dict:
        """Batch update status for done/invalidate commands."""
        from commands import make_command_response
        from memory.task_graph import AmbiguousHashError

        # Parse hashes: split by comma, strip, filter empty, lowercase, DEDUPLICATE
        hashes = [h.strip().lower() for h in hashes_str.split(",") if h.strip()]
        # Contract: Caller MUST deduplicate. Use dict.fromkeys() to preserve order.
        hashes = list(dict.fromkeys(hashes))

        if not hashes:
            return error_response("No valid hashes provided.", 400)

        # Derive workspace scope
        workspace = resolve_workspace(ctx)
        workspace_id = workspace

        # F4: Use singleton to avoid creating new Neo4j driver/connection pool per call
        tg = _get_task_graph_singleton()

        if not tg.is_available():
            return error_response("Task store is unavailable.", 503)

        try:
            results = tg.update_tasks_batch_status(hashes, new_status, workspace_id=workspace_id)
        except AmbiguousHashError as e:
            logger.warning("Ambiguous hash in batch for session %s: %d tasks match",
                           str(ctx.session_id or "unknown")[:8], e.count)
            return error_response(
                f"Ambiguous hash: {e.count} tasks match one or more hashes. "
                f"Use ^^ticket list to find exact hashes.", 409,
            )
        except Exception:
            logger.exception("update_tasks_batch_status failed for session %s", ctx.session_id[:8])
            return error_response("Task store error. Cannot update tasks.", 500)

        # Build response lines
        status_label = {"complete": "complete", "invalid": "invalid", "in_progress": "in_progress"}.get(new_status, new_status)
        lines = []
        for r in results:
            h = r["hash"]
            err = r.get("error_reason", "")
            if r["ok"] and r["changed"]:
                lines.append(f'→ "{h}" marked {status_label}.')
            elif r["ok"] and not r["changed"]:
                lines.append(f'→ "{h}" already at {new_status} — no change.')
            elif err == "invalid_transition":
                lines.append(f"→ \"{h}\" FAILED — invalid_transition (current status: {r['status']}).")
            elif err == "concurrent_modification":
                lines.append(f'→ "{h}" FAILED — concurrent_modification (task was modified by another request).')
            elif err == "no_workspace":
                lines.append(f'→ "{h}" FAILED — no workspace scope.')
            elif err == "db_error":
                lines.append(f'→ "{h}" FAILED — database error (status could not be updated).')
            else:
                lines.append(f'→ "{h}" FAILED — not found.')

        return make_command_response("\n".join(lines))

    # -------------------------------------------------------------------
    # 14 — get_system_prompt
    # -------------------------------------------------------------------
    def get_system_prompt(self, ctx: SkillRequestContext) -> str:
        """Return MAIN ticket orchestrator system prompt.

        M1-M3 refactor: the MAIN vs SUB role dispatch is performed by the
        framework — `get_system_prompt` returns MAIN's prompt and
        `get_subagent_prompt` returns SUB's. No `is_subagent` branching
        lives in the skill — the hooks do the dispatch.

        MAIN sees `TICKET_MAIN_PROMPT` directly (MAIN identity block +
        shared reference). The shared reference documents the 10-action
        proxy_ticket surface; the identity block tells MAIN that tool_guard
        blocks proxy_work and proxy_ticket write actions (see
        `tool_guard._check_work_mode_actions` for the proxy_ticket R/O
        allowlist and `tool_guard._check_ticket_mode_orchestrator` for
        proxy_work) and must delegate via the `task` tool.
        """
        from proxy.tool_parsing.ticket_prompts import TICKET_MAIN_PROMPT
        return TICKET_MAIN_PROMPT

    # -----------------------------------------------------------------------
    # Phase 3: Bug subcommand handlers
    # -----------------------------------------------------------------------

    def _list_bugs(self, ctx: SkillRequestContext) -> dict:
        from commands import make_command_response
        from memory.bug_tracker import list_bugs

        workspace = resolve_workspace(ctx)
        workspace_id = workspace
        tg = _get_task_graph_singleton()
        if not tg.is_available():
            return error_response("Task store is unavailable.", 503)
        try:
            bugs = list_bugs(tg, workspace_id=workspace_id, status_filter=["open"], limit=100)
        except Exception:
            logger.exception("list_bugs failed")
            return error_response("Task store error. Cannot list bugs.", 500)
        if not bugs:
            return make_command_response("No open bugs for this workspace.")
        lines = ["Open Bugs:"]
        for i, b in enumerate(bugs, 1):
            lines.append(
                f"  #[{i}] [{b.get('short_hash', '?')}] {b.get('severity', '?').upper()} — "
                f"{b.get('title', '?')} (reviews: {b.get('review_count', 0)}/{b.get('max_reviews', 0)})"
            )
        return make_command_response("\n".join(lines))

    def _show_bug(self, ctx: SkillRequestContext, bug_hash: str) -> dict:
        from commands import make_command_response
        from memory.bug_tracker import get_bug_by_short_hash
        from memory.task_graph import AmbiguousHashError

        workspace = resolve_workspace(ctx)
        workspace_id = workspace
        tg = _get_task_graph_singleton()
        if not tg.is_available():
            return error_response("Task store is unavailable.", 503)
        try:
            bug = get_bug_by_short_hash(tg, bug_hash, workspace_id)
        except AmbiguousHashError as e:
            return error_response(
                f"Ambiguous bug hash '{bug_hash}': {e.count} bugs match.", 409,
            )
        if bug is None:
            return error_response(f"No bug found with hash {bug_hash}.", 404)
        if bug.get("error") == "infrastructure_error":
            return error_response(
                bug.get("message") or "Database temporarily unavailable", 503,
            )
        reviews = bug.get("reviews", [])
        lines = [
            f"Bug [{bug.get('short_hash', '?')}] — {bug.get('status', '?')}",
            "─" * 40,
            f"Title   : {bug.get('title', '?')}",
            f"Severity: {bug.get('severity', '?')}",
            f"Reporter: {bug.get('reported_by_agent', '?')}",
            f"Reviews : {bug.get('review_count', 0)}/{bug.get('max_reviews', 0)}",
            f"Validated: {bool(bug.get('validated_at'))}",
        ]
        if bug.get("dismissal_reason"):
            lines.append(f"Dismiss reason: {bug.get('dismissal_reason')}")
        lines.append("─" * 40)
        lines.append(f"Description: {(bug.get('description') or '(none)')[:500]}")
        if reviews:
            lines.append("─" * 40)
            lines.append("Reviews:")
            for r in reviews:
                lines.append(
                    f"  • {r.get('reviewer_agent_id', '?')} → {r.get('verdict', '?')} "
                    f"(confidence={float(r.get('confidence_score') or 0):.2f})"
                )
                lines.append(f"    {(r.get('proposal') or '(no proposal)')[:200]}")
        return make_command_response("\n".join(lines))

    def _resolve_bug(self, ctx: SkillRequestContext, args_str: str, resolution: str) -> dict:
        """Resolve a bug (approved|dismissed). Called via ^^ticket bug resolve|dismiss.

        `args_str` may contain `hash|reason` (the `|` is our internal separator
        to keep the hash and reason together as a single arg). Backwards compat:
        plain `hash [reason]` (space-separated) is also supported.
        """
        import secrets
        import uuid as uuid_mod

        from commands import make_command_response
        from memory.bug_tracker import get_bug_by_short_hash, resolve_bug
        from memory.task_graph import AmbiguousHashError

        if "|" in args_str:
            bug_hash, _, reason = args_str.partition("|")
            bug_hash = bug_hash.strip()
            reason = reason.strip() or None
        else:
            parts = args_str.split(None, 1)
            bug_hash = parts[0].strip() if parts else ""
            reason = parts[1].strip() if len(parts) > 1 and parts[1].strip() else None

        if not bug_hash:
            return error_response("Bug short_hash is required.", 400)

        workspace = resolve_workspace(ctx)
        workspace_id = workspace
        tg = _get_task_graph_singleton()
        if not tg.is_available():
            return error_response("Task store is unavailable.", 503)

        try:
            bug = get_bug_by_short_hash(tg, bug_hash, workspace_id)
        except AmbiguousHashError as e:
            return error_response(
                f"Ambiguous bug hash '{bug_hash}': {e.count} bugs match.", 409,
            )
        if bug is None:
            return error_response(f"No bug found with hash {bug_hash}.", 404)
        if bug.get("error") == "infrastructure_error":
            return error_response(
                bug.get("message") or "Database temporarily unavailable", 503,
            )
        if bug.get("status") != "open":
            return error_response(f"Bug is {bug.get('status')}, not open.", 409)
        if not bug.get("validated_at"):
            return error_response(
                "Bug must be validated first. Run ^^ticket bug show for status.", 400,
            )

        task_uuid = f"entity-{uuid_mod.uuid4().hex}"
        task_short_hash = secrets.token_hex(4)
        try:
            result = resolve_bug(
                tg=tg,
                bug_uuid=bug["uuid"],
                resolution=resolution,
                reason=reason,
                caller_agent_id=None,  # Human caller — no agent_id
                workspace_id=workspace_id,
                task_uuid=task_uuid,
                task_short_hash=task_short_hash,
            )
            if isinstance(result, dict) and "error" in result:
                return error_response(result["error"], 400)
            response = f"Bug [{bug_hash}] {resolution}."
            if resolution == "approved" and result.get("task_short_hash"):
                response += f" Task created: [{result['task_short_hash']}] {bug.get('title', '?')}"
            return make_command_response(response)
        except Exception:
            logger.exception("resolve_bug failed")
            return error_response("Task store error. Cannot resolve bug.", 500)

    # -------------------------------------------------------------------
    # 15 — get_subagent_prompt
    # -------------------------------------------------------------------
    def get_subagent_prompt(self, ctx: SkillRequestContext) -> str:
        """Return ticket sub-agent system prompt.

        M1-M3 refactor: SUB agents see `TICKET_SUB_PROMPT` (SUB identity
        block + the same shared reference MAIN sees). The role dispatch
        is performed by the framework — `get_system_prompt` returns
        MAIN's prompt, `get_subagent_prompt` returns SUB's. This
        preserves the unified-prompt design — single shared reference
        for both roles, identity blocks are the only difference.
        """
        from proxy.tool_parsing.ticket_prompts import TICKET_SUB_PROMPT
        return TICKET_SUB_PROMPT

    # -------------------------------------------------------------------
    # 4 — get_tool_filter (allowed_tool_names derived)
    # 5 — sub_agent_tool_filter derived from get_sub_agent_tool_filter;
    #     TicketSkill overrides it to exclude `task` (sub-agents cannot
    #     spawn sub-sub-agents — current CLIs do not support nesting).
    # -------------------------------------------------------------------
    def get_tool_filter(self, ctx: SkillRequestContext) -> frozenset[str]:
        """Return allowed tools when ticket mode is active (C7b: derived from capability_table)."""
        return self.capability_table.visible_tools("main")

    # -------------------------------------------------------------------
    # 5 — get_sub_agent_tool_filter (override: exclude task — no sub-sub-agent nesting)
    # -------------------------------------------------------------------
    def get_sub_agent_tool_filter(self, ctx: SkillRequestContext) -> frozenset[str]:
        """Return allowed tools for sub-agents in ticket mode (C7b: derived from capability_table)."""
        return self.capability_table.visible_tools("subagent")

    # -------------------------------------------------------------------
    # 18 — get_canary
    # -------------------------------------------------------------------
    def get_canary(self, ctx: SkillRequestContext) -> str:
        return self.build_skill_canary(ctx)

    # -------------------------------------------------------------------
    # Phase 1a compat: get_inheritance_state is read by code that does
    # `getattr(skill, 'get_inheritance_state', lambda: None)(ctx)`. Not in
    # the 23 fields, but kept for backwards compat.
    # -------------------------------------------------------------------
    def get_inheritance_state(self, ctx: SkillRequestContext) -> Optional[dict]:
        """Return state to pass to sub-agents."""
        import logging as _logging
        _logger = _logging.getLogger(__name__)

        try:
            # F4: Use singleton to avoid creating new Neo4j driver/connection pool per call
            tg = _get_task_graph_singleton()

            # Resolve workspace_name from in-memory store BEFORE the Neo4j
            # availability check so all return paths (including the
            # Neo4j-unavailable early return) include it.
            workspace_name = resolve_workspace(ctx)

            if not tg.is_available():
                _logger.warning("get_inheritance_state: TaskGraph unavailable")
                from proxy.workspace_state import DEFAULT_WORKSPACE
                result = {"tasks": [], "store_status": "unavailable"}
                if workspace_name and workspace_name != DEFAULT_WORKSPACE:
                    result["workspace_name"] = workspace_name
                return result

            workspace_id = workspace_name
            tasks = tg.get_actionable_tasks(workspace_id=workspace_id, limit=500)

            # F3: Truncate to prevent memory spikes and Neo4j timeouts for sub-agents
            tasks = tasks[:MAX_INHERITANCE_TASKS]

            from proxy.workspace_state import DEFAULT_WORKSPACE
            if not tasks:
                result = {"tasks": [], "store_status": tg.get_store_status()}
            else:
                result = {
                    "tasks": [
                        {
                            "short_hash": t.get("short_hash", ""),
                            "name": t.get("name", ""),
                            "priority": t.get("priority", "medium"),
                            "status": t.get("status", "pending"),
                            "summary": t.get("summary", ""),
                            "tag": t.get("tag", ""),
                            "blocked_by": t.get("blockers", []),
                            "is_blocked": t.get("is_blocked", False),
                            "blocked_count": t.get("blocked_count", 0),
                            "parent_short_hash": t.get("parent_short_hash"),
                            "parent_name": t.get("parent_name"),
                        }
                        for t in tasks
                    ],
                    "store_status": tg.get_store_status(),
                }

            if workspace_name and workspace_name != DEFAULT_WORKSPACE:
                result["workspace_name"] = workspace_name
            return result
        except Exception as e:
            _logger.exception("get_inheritance_state failed: %s", e)
            return {"tasks": [], "store_status": "error"}

    # -------------------------------------------------------------------
    # 16 — on_activate
    # -------------------------------------------------------------------
    def on_activate(self, ctx: SkillRequestContext, mode_state: Optional[ModeState] = None) -> None:
        """Called after skill mode is activated by _activate_skill.
        Ticket-specific setup: logging only.
        """
        workspace = resolve_workspace(ctx)
        if workspace == DEFAULT_WORKSPACE:
            raise SkillValidationError(
                "Tickets live in a workspace — disabled on the default workspace.",
                http_status=409,
            )
        import logging as _logging
        _logger = _logging.getLogger("skills.ticket")
        _logger.info("Ticket skill activated: session=%s", ctx.session_id[:8])

    # -------------------------------------------------------------------
    # 17 — on_deactivate
    # -------------------------------------------------------------------
    def on_deactivate(self, ctx: SkillRequestContext, mode_state: Optional[ModeState] = None) -> None:
        """Called when ticket mode is being deactivated.
        Ticket-specific cleanup only. Setting/clearing the active mode
        is the responsibility of Agent.set_active_mode / clear_active_mode,
        which call this method as part of their own lifecycle.
        """
        pass

    # -------------------------------------------------------------------
    # 17b — should_deactivate
    # -------------------------------------------------------------------
    def should_deactivate(self, ctx: SkillRequestContext, finish_reason: str) -> bool:
        """Ticket mode is PER-TURN: it deactivates on stop for BOTH main and sub-agents.

        Ticket mode is scoped to the turn activated by the user's `^^ticket`
        trigger. When the model emits `finish_reason="stop"` the mode clears
        for BOTH main agents and sub-agents. PostProcessStage is the sole
        authority on the keep/clear decision (per ticket 199ece93 §4); a
        downstream disconnect does NOT clear mode. To re-enter ticket mode
        the user re-types `^^ticket`.

        This is safe for multi-sub-agent ticket orchestration: a turn that
        spawns a sub-agent carries `finish_reason="tool_calls"`, never
        "stop"; the main agent only emits "stop" at the true end of an
        orchestration, which is exactly when the mode should clear.

        Clearing on stop prevents `apply_post_process`
        (`src/proxy/mode.py`) from rebuilding a ticket canary and attaching
        it to every `task` tool call on later plain prompts — which would
        leak read-only ticket tools into bare sub-agents.
        """
        return finish_reason == "stop"

    # -------------------------------------------------------------------
    # 23 — get_activation_rewrite
    # -------------------------------------------------------------------
    def get_activation_rewrite(self, ctx: SkillRequestContext) -> str:
        """Rewrite user text for LLM consumption in ticket mode.

        NOTE: ctx type annotation is `SkillRequestContext` per Phase 1d
        plan §3.4. Runtime ctx is still OLD `SkillContext` until Phase 2.
        Uses `getattr` for fields (`skill_state`, `raw_user_text`) that
        exist on OLD `SkillContext` but not on `SkillRequestContext`.
        """
        skill_state = (
            getattr(ctx.mode_state, 'skill_state', None)
            if getattr(ctx, 'mode_state', None) is not None
            else getattr(ctx, 'skill_state', None)
        ) or {}
        arg = skill_state.get("arg", "")
        if not arg:
            arg = getattr(ctx, 'raw_user_text', '')
        # Sanitize: apply same prompt injection stripping as BaseSkill
        if arg:
            arg = _normalize_text(arg)
            arg = strip_environment_details(arg)
            arg = re.sub(
                r"```(?:system|user)\n.*?\n```",
                "",
                arg,
                flags=re.IGNORECASE | re.DOTALL,
            )
            arg = re.sub(
                r"```(?:system|user)\s*",
                "",
                arg,
                flags=re.IGNORECASE,
            )
            arg = re.sub(r"\[\[(?:system|user)\]\]", "", arg, flags=re.IGNORECASE)
        return f"Delegate the following ticket instruction to a sub-agent via the task tool (sub-agents have proxy_ticket access):\n\n{arg}"