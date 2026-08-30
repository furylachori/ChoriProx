"""Work skill — implements Skill protocol for work execution via ^^work.

Phase 1d: extends NEW `BaseSkill` from `proxy.skills.skill_definition`.
Provides the 23 `SkillDefinition` fields as class attributes (per
docs/specs/consolidation-research/14-phase-1d-plan.md §3.5).
"""
from __future__ import annotations

import logging
from typing import Any, Optional

from proxy.skills.skill_definition import (
    BaseSkill,
    ModeState,
    ProxyAccessLevel,
    SkillRequestContext,
    SkillValidationError,
    WORK_CAPABILITY_TABLE,
)
from proxy.skills.skill_registry import (
    SkillActivation,
    SkillMeta,
    _normalize_text,
    error_response,
)
from proxy.tool_parsing.work_prompts import WORK_MANAGER_PROMPT
from proxy.workspace_state import workspace_store, resolve_workspace, DEFAULT_WORKSPACE

logger = logging.getLogger("skills.work")


class WorkSkill(BaseSkill):
    """Skill implementing the ^^work command set (Phase 1d: NEW BaseSkill).

    Class attributes below populate the 23 `SkillDefinition` fields when
    `to_skill_definition()` is called (inherited from NEW `BaseSkill`).
    """

    # ---- Identity (1-3) ----
    name: str = "work"
    trigger: str = "^^work"
    aliases: tuple = ()
    # Wave 1: short human-readable description (≤160 chars). Surfaced by
    # `^^help` and the custom-JSON loader (Agent 2 parses it from JSON).
    description: str = "Execute work on a task (^^work — full coding tools, interactive session)."
    capability_table = WORK_CAPABILITY_TABLE

    # ---- Sub-agent permissions (6, 7) — C4.5: aligned with plan locked decision #3 ----
    # Work sub-agents have READ-ONLY proxy_ticket access
    # (list_tasks, get_task, append_review_note) per plan locked decision #3.
    # The MAIN orchestrator owns ticket writes; subs execute code and report
    # findings via append_review_note.
    sub_agent_proxy_ticket_access: ProxyAccessLevel = ProxyAccessLevel.READ_ONLY
    sub_agent_proxy_work_access: ProxyAccessLevel = ProxyAccessLevel.FULL

    # ---- TTL (8) ----
    ttl_seconds: int = 300

    # ---- Fallback (9) — DELETED in C3: no builtin has a fallback after the work→ticket TTL deletion ----
    fallback_skill: Optional[str] = None

    # ---- Cross-cutting config (10, 11) ----
    # Work mode IS interactive (agent needs tool calls to do work).
    fire_and_forget: bool = False
    blocked_tools_for_subagent: frozenset = frozenset({"question", "task"})

    # ---- Call limits (13) — R3 MED-16 ----
    call_limits: dict = {"proxy_work": 50, "task": 3}

    # `check_action` (12): C7b set to None — US-095 lifecycle exclusion is now
    # encoded in WORK_CAPABILITY_TABLE.actions_allowed for role='subagent'
    # (C7b folded the callback into the table). The capability table is the
    # single source of truth for BUILTIN skills; custom skills use legacy
    # ProxyAccessLevel + check_action fallback paths.

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
            description="Execute work on a task (full tool access, read-only ticket)",
            help_text="Use ^^work <task_hash> to begin working on a task",
        )

    # -------------------------------------------------------------------
    # 19 — parse_subcommand
    # -------------------------------------------------------------------
    def parse_subcommand(self, text: str, session_id: str = "") -> tuple[Optional[str], str]:
        """Parse work subcommand.

        Returns:
            ("task", task_hash) — activate work mode for task
            ("list", "") — list tasks (alias for ^^ticket list)
            (None, "") — not a work command
        """
        from commands.task_commands import is_work_trigger
        text = _normalize_text(text)
        if is_work_trigger(text):
            stripped = text[len(self.trigger):].strip()
            if not stripped:
                return ("list", "")
            from memory.task_graph import TaskGraph
            if TaskGraph.SHORT_HASH_RE.fullmatch(stripped):
                return ("task", stripped)
            return ("list", "")
        return (None, "")

    # -------------------------------------------------------------------
    # 20 — handle_direct
    # -------------------------------------------------------------------
    def handle_direct(self, ctx: SkillRequestContext, action: str, arg: str) -> Optional[dict]:
        """Handle ^^work list (alias for ^^ticket list)."""
        workspace = resolve_workspace(ctx)
        if workspace == DEFAULT_WORKSPACE:
            return error_response(
                "Tickets live in a workspace — disabled on the default workspace. "
                "Create one with `^^ws create <name>`, list existing with `^^ws list`, "
                "or learn more with `^^ws help`.", 409)
        if action == "list":
            from skills.ticket_skill import TicketSkill
            ticket_skill = TicketSkill()
            return ticket_skill.handle_direct(ctx, "list", arg)
        return None

    # -------------------------------------------------------------------
    # 16 — on_activate
    # -------------------------------------------------------------------
    def on_activate(self, ctx: SkillRequestContext, mode_state: Optional[ModeState] = None) -> None:
        """Activate work mode. Reject if sub-agent (prevent recursion).

        US-092: ^^work <task_hash> activates work mode with task hash captured.
        Sub-agents MUST NOT invoke ^^work (per SPEC line 188-191).

        R4 CRIT-4 fix: do NOT mutate the request `ctx` with the captured
        task hash. The captured hash lives in `skill_state["arg"]` (set by
        skill_dispatcher.SkillDispatcher.dispatch / pipeline.dispatch.
        try_skill_activation) and is persisted onto the ModeState by
        SkillDispatcher.activate. Downstream consumers (get_system_prompt)
        read it back from ModeState.skill_state["arg"] via snapshot_mode().

        Bug #10: validate the hash exists in the task graph before
        activating. A nonexistent hash would otherwise activate silently
        and the agent would only discover the error when start_work
        fails upstream. Validation is fail-open if the task store is
        unavailable (so a Neo4j outage does not block activation — the
        agent will hit the error at start_work, same as today).
        """
        if getattr(ctx, "is_subagent", False):
            raise ValueError(
                "Sub-agents cannot activate work mode. "
                "Work mode is only for top-level agents spawned via the task tool."
            )
        skill_state = getattr(ctx, "skill_state", None) or {}
        task_hash = skill_state.get("arg", "") if skill_state else ""
        if not task_hash:
            raise ValueError("Work mode requires a task hash (US-092).")

        # Bug #10: validate the hash exists. Fail-open on infra/store
        # unavailability so a Neo4j outage does not block activation.
        try:
            from proxy.context import _get_task_graph_singleton
            from proxy.workspace_state import workspace_store
            from memory.task_graph import AmbiguousHashError

            tg = _get_task_graph_singleton()
            if tg.is_available():
                workspace = resolve_workspace(ctx)
                try:
                    task = tg.get_task_enriched(task_hash, workspace_id=workspace)
                except AmbiguousHashError:
                    # Multiple matches — the hash is not nonexistent, just
                    # ambiguous. Activation proceeds; start_work will surface
                    # the ambiguity with a usable error when the agent acts.
                    task = True
                if task is None:
                    raise SkillValidationError(
                        f"No task found with hash '{task_hash}'. "
                        f"Use ^^ticket list to find a valid task hash.",
                        http_status=404,
                    )
        except ValueError:
            raise
        except Exception as e:
            logger.warning(
                "Work skill on_activate: task hash validation skipped for %s: %s",
                task_hash, e,
            )

        logger.info(
            "Work skill activated: session=%s task_hash=%s",
            ctx.session_id[:8] if ctx.session_id else "",
            task_hash,
        )

    # -------------------------------------------------------------------
    # 17 — on_deactivate
    # -------------------------------------------------------------------
    def on_deactivate(self, ctx: SkillRequestContext, mode_state: Optional[ModeState] = None) -> None:
        """Work mode deactivation. Work sessions survive in Neo4j — no cleanup needed."""
        pass

    # -------------------------------------------------------------------
    # 14 — get_system_prompt
    # -------------------------------------------------------------------
    def get_system_prompt(self, ctx: SkillRequestContext) -> str:
        """Return work execution system prompt with task context anchored.

        SPEC lines 145-176: the prompt is anchored with task hash, name, priority, summary.

        Bug #2 fix: the three previous hash sources were all broken:
          - ctx.work_task_hash: SkillRequestContext has no such field (and
            R4 CRIT-4 prohibits on_activate from setting it).
          - ctx.skill_state["task_hash"]: wrong key. The dispatcher stores
            the hash under ``skill_state["arg"]`` (see pipeline/dispatch.py
            and skill_dispatcher.SkillDispatcher.dispatch — both write
            ``{"action": ..., "arg": <hash>}``).
          - ctx.mode_state: always None on the SkillRequestContext built by
            skill_prompts.assemble_system_prompt (RequestContext has no
            mode_state field).

        The correct source is the active ModeState's skill_state, persisted
        by skill_dispatcher.SkillDispatcher.activate via
        ``object.__setattr__(mode_state, 'skill_state',
        MappingProxyType({"action": ..., "arg": <hash>}))``. Read it via
        agent.snapshot_mode() (src/skills/AGENTS.md: no TOCTOU on active_mode).
        ctx.skill_state["arg"] is a fallback for tests that build a ctx
        directly without installing a ModeState.
        """
        task_hash = self._resolve_task_hash(ctx)
        if not task_hash:
            return (
                WORK_MANAGER_PROMPT
                .replace("<short_hash>", "(ad-hoc edit session)")
                .replace("<task_name>", "(no specific task)")
                .replace("<priority>", "MEDIUM")
                .replace("<summary>", "(no summary)")
            )

        from proxy.context import _get_task_graph_singleton
        from proxy.workspace_state import workspace_store
        task = None
        try:
            workspace = resolve_workspace(ctx)
            tg = _get_task_graph_singleton()
            task = tg.get_task_enriched(task_hash, workspace_id=workspace)
        except Exception as e:
            logger.warning(
                "get_system_prompt: failed to fetch task %s for session=%s: %s",
                task_hash,
                getattr(ctx, "session_id", "")[:8],
                e,
            )

        # Always start with the template containing task_hash so we never lose it
        block = WORK_MANAGER_PROMPT.replace("<short_hash>", str(task_hash))
        if task:
            block = (
                block.replace("<task_name>", str(task.get("name") or "?"))
                .replace("<priority>", (str(task.get("priority") or "medium")).upper())
                .replace("<summary>", str(task.get("summary") or "(no summary)"))
            )
        else:
            block = (
                block.replace("<task_name>", "?")
                .replace("<priority>", "MEDIUM")
                .replace("<summary>", "(no summary)")
            )
        return block

    def _resolve_task_hash(self, ctx: SkillRequestContext) -> str:
        """Resolve the work task hash from the active ModeState, falling back to ctx.

        Primary path: read ``ModeState.skill_state["arg"]`` via
        ``agent.snapshot_mode()`` — this is the value persisted by the
        dispatcher at activation. The ``arg`` key matches
        ``dispatch.py:try_skill_activation`` and
        ``skill_dispatcher.SkillDispatcher.dispatch`` (both write
        ``skill_ctx.skill_state = {"action": ..., "arg": <hash>}``).

        Fallback: ``ctx.skill_state["arg"]`` — exercised by tests that
        construct a SkillRequestContext directly without going through the
        dispatcher / agent registry.
        """
        session_id = getattr(ctx, "session_id", "")
        if session_id:
            try:
                from proxy.agent.agent_registry import registry as agent_registry
                agent = agent_registry.get_or_load(session_id)
                if agent is not None:
                    ms = agent.snapshot_mode()
                    if ms is not None and getattr(ms, "name", None) == "work":
                        skill_state = dict(getattr(ms, "skill_state", {}) or {})
                        arg = skill_state.get("arg", "")
                        if arg:
                            return arg
            except Exception as e:
                logger.warning(
                    "get_system_prompt: could not read skill_state from ModeState for session=%s: %s",
                    session_id[:8], e,
                )
        skill_state = getattr(ctx, "skill_state", None) or {}
        return skill_state.get("arg", "")

    # -------------------------------------------------------------------
    # 15 — get_subagent_prompt
    # -------------------------------------------------------------------
    def get_subagent_prompt(self, ctx: SkillRequestContext) -> Optional[str]:
        """Return work SUB-AGENT prompt (no lifecycle actions, distinct from MAIN).

        Sub-agents in work mode are the workers — the orchestrator owns the
        work-session lifecycle (start_work/stop_work) and these calls are
        excluded from `WORK_CAPABILITY_TABLE.actions_allowed` for role='subagent'
        (enforced via `proxy.skill_guard.check_sub_agent_proxy_access`). The
        sub-agent prompt must not instruct the sub-agent to make those calls.
        """
        from proxy.tool_parsing.work_prompts import WORK_SUB_PROMPT
        task_hash = self._resolve_task_hash(ctx)
        if not task_hash:
            return (
                WORK_SUB_PROMPT
                .replace("<short_hash>", "(ad-hoc edit session)")
                .replace("<task_name>", "(no specific task)")
                .replace("<priority>", "MEDIUM")
                .replace("<summary>", "(no summary)")
            )
        # Reuse the same task-fetch logic as get_system_prompt for placeholder substitution.
        from proxy.context import _get_task_graph_singleton
        from proxy.workspace_state import workspace_store
        task = None
        try:
            from proxy.workspace_state import resolve_workspace
            workspace = resolve_workspace(ctx)
            tg = _get_task_graph_singleton()
            task = tg.get_task_enriched(task_hash, workspace_id=workspace)
        except Exception as e:
            import logging as _logging
            _logging.getLogger(__name__).warning(
                "get_subagent_prompt: failed to fetch task %s for session=%s: %s",
                task_hash, getattr(ctx, "session_id", "")[:8], e,
            )
        block = WORK_SUB_PROMPT.replace("<short_hash>", str(task_hash))
        if task:
            block = (
                block.replace("<task_name>", str(task.get("name") or "?"))
                .replace("<priority>", (str(task.get("priority") or "medium")).upper())
                .replace("<summary>", str(task.get("summary") or "(no summary)"))
            )
        else:
            block = (
                block.replace("<task_name>", "?")
                .replace("<priority>", "MEDIUM")
                .replace("<summary>", "(no summary)")
            )
        return block

    # -------------------------------------------------------------------
    # 23 — get_activation_rewrite
    # -------------------------------------------------------------------
    def get_activation_rewrite(self, ctx: SkillRequestContext) -> str:
        """Rewrite user text for LLM consumption in work mode.

        NOTE: ctx type annotation is `SkillRequestContext` per Phase 1d
        plan §3.4. Runtime ctx is still OLD `SkillContext` until Phase 2.
        Uses `getattr` for `skill_state` and `raw_user_text`.
        """
        skill_state = (
            getattr(ctx.mode_state, 'skill_state', None)
            if getattr(ctx, 'mode_state', None) is not None
            else getattr(ctx, 'skill_state', None)
        ) or {}
        arg = skill_state.get("arg", "")
        if not arg:
            arg = getattr(ctx, 'raw_user_text', '')
        return f"Execute the following task using proxy_work tool calls. Claim it, do the work, report progress, and stop when done.\n\n{arg}"

    # -------------------------------------------------------------------
    # 4 — get_tool_filter (allowed_tool_names derived)
    # -------------------------------------------------------------------
    def get_tool_filter(self, ctx: SkillRequestContext) -> frozenset[str]:
        """Return allowed tools when work mode is active (C7b: derived from capability_table)."""
        return self.capability_table.visible_tools("main")

    # -------------------------------------------------------------------
    # 5 — get_sub_agent_tool_filter (separate from main per R1 HIGH-2)
    # -------------------------------------------------------------------
    def get_sub_agent_tool_filter(self, ctx: SkillRequestContext) -> frozenset[str]:
        """Return allowed tools for sub-agents in work mode (C7b: derived from capability_table)."""
        return self.capability_table.visible_tools("subagent")

    # -------------------------------------------------------------------
    # Phase 1a compat: get_inheritance_state is read by code that does
    # `getattr(skill, 'get_inheritance_state', lambda: None)(ctx)`. Not in
    # the 23 fields, but kept for backwards compat.
    # -------------------------------------------------------------------
    def get_inheritance_state(self, ctx: SkillRequestContext) -> Optional[dict]:
        skill_state = getattr(ctx, 'skill_state', None) or {}
        task_hash = skill_state.get("arg", "") if skill_state else ""
        if not task_hash:
            task_hash = getattr(ctx, "work_task_hash", "") or ""
        result: Optional[dict] = None
        if task_hash:
            result = {"task_hash": task_hash}
        try:
            from proxy.workspace_state import resolve_workspace, DEFAULT_WORKSPACE
            workspace_name = resolve_workspace(ctx)
            if workspace_name and workspace_name != DEFAULT_WORKSPACE:
                if result is None:
                    result = {}
                result["workspace_name"] = workspace_name
        except Exception as exc:
            logger.debug(
                "get_inheritance_state: workspace lookup failed for session %s: %s",
                ctx.session_id, exc,
            )
        return result

    # -------------------------------------------------------------------
    # 18 — get_canary
    # -------------------------------------------------------------------
    def get_canary(self, ctx: SkillRequestContext) -> Optional[str]:
        return self.build_skill_canary(ctx)

    # -------------------------------------------------------------------
    # 12 — check_action (R3 CRIT-4 / US-095)
    # -------------------------------------------------------------------
    # C7b: US-095 work-mode lifecycle gating (start_work/stop_work/complete_work
    # blocked for sub-agents) is now enforced by the WorkSkill capability_table
    # via proxy.skill_guard.check_sub_agent_proxy_access →
    # table.check_action. The legacy _check_work_mode_action callback is dead
    # code — leaving check_action = None so to_skill_definition() exposes an
    # empty slot. The parity tests in tests/unit/test_guard_table_parity.py
    # prove the table produces the same verdicts the callback used to.
    check_action = None