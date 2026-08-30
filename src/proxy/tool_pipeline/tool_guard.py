"""Pre-execution guard pipeline for memory tools.

Performs 4 checks before allowing memory tool execution:
1. Self-reference check — agent can't modify its own history
2. Mode allowlist — ticket_mode restricts tools
3. Call limits check (proxy_ticket, file reads, sub-agent spawns)
4. Hash provenance check — MODIFY actions must reference known hashes
5. Hierarchy check — sub-agent permissions
"""
import logging
from typing import Optional, Any

from memory.query_executor import InfrastructureError
from memory.task_graph import AmbiguousHashError
from proxy.tool_pipeline.tool_call import ToolCall
from proxy.tool_pipeline.tool_call_tracker import ToolCallTracker
from proxy.tool_pipeline.tool_limits import MAX_REPORT_BUG_CALLS
from proxy.tool_parsing.normalize import extract_file_path

logger = logging.getLogger("proxy.tool_pipeline.guard")


class MemoryToolGuard:
    """Pre-execution guard pipeline for memory tools."""

    # Phase 4: proxy_work action -> short_hash argument name
    WORK_HASH_ACTIONS = {
        "start_work": "short_hash",
        "stop_work": "short_hash",
        "handoff_task": "short_hash",
        "report_bug": "task_short_hash",
    }

    # Slice 8a: decomposed ticket tool names
    _TICKET_TOOL_NAMES = frozenset({
        "proxy_ticket_list_tasks", "proxy_ticket_get_task",
        "proxy_ticket_create_tasks", "proxy_ticket_update_task",
        "proxy_ticket_edit", "proxy_ticket_set_parent",
        "proxy_ticket_clear_parent", "proxy_ticket_clear_blocked_by",
        "proxy_ticket_set_priority", "proxy_ticket_append_review_note",
    })

    # Slice 8a: hash field mapping for decomposed tools
    _TICKET_HASH_FIELDS = {
        "proxy_ticket_get_task": ("short_hash",),
        "proxy_ticket_update_task": ("short_hash",),
        "proxy_ticket_edit": ("short_hash", "parent_hash", "blocked_by"),
        "proxy_ticket_set_parent": ("short_hash", "parent_hash"),
        "proxy_ticket_clear_parent": ("short_hash",),
        "proxy_ticket_clear_blocked_by": ("short_hash",),
        "proxy_ticket_set_priority": ("short_hash",),
        "proxy_ticket_append_review_note": ("short_hash",),
    }

    def check(self, tool_call: ToolCall, ctx: Any) -> Optional[str]:
        """Returns error message if guard rejects, None if OK.

        Args:
            tool_call: The ToolCall to validate
            ctx: ToolContext with session_id, is_ticket_mode, etc.

        Returns:
            Error string if rejected, None if allowed.
        """
        if not tool_call.name or not tool_call.name.strip():
            return "Tool call rejected: empty tool name"

        error = self._check_self_reference(tool_call, ctx)
        if error:
            return error

        error = self._check_mode_allowlist(tool_call, ctx)
        if error:
            return error

        error = self._check_work_mode_actions(tool_call, ctx)
        if error:
            return error

        error = self._check_ticket_mode_orchestrator(tool_call, ctx)
        if error:
            return error

        error = self._check_blocked_tools_for_subagent(tool_call, ctx)
        if error:
            return error

        error = self._check_hash_provenance(tool_call, ctx)
        if error:
            return error

        error = self._check_hierarchy(tool_call, ctx)
        if error:
            return error

        # Call limits LAST — only count if all other checks pass
        error = self._check_call_limits(tool_call, ctx)
        if error:
            return error

        return None

    def _check_self_reference(self, tool_call: ToolCall, ctx: Any) -> Optional[str]:
        """Prevent agent from invoking tools that would modify its own state.

        Mirrors tool_loop.py:297-301 ToolCallGuardException.

        Blocks: agent:* task:* ticket:* tool names.
        """
        agent_name = tool_call.name.split(":")[0].lower() if ":" in tool_call.name else tool_call.name.lower()

        if agent_name in ("agent", "task", "ticket"):
            return (
                f"Tool call self-reference blocked: {tool_call.name}. "
                f"Agent attempted to invoke itself."
            )

        return None

    # NOTE: Intentional TOCTOU trade-off. allowed_tools is read under _mode_lock as a
    # frozenset snapshot (immutable). The membership check on line 107 runs outside the
    # lock for performance. If the mode deactivates during this window, the stale
    # frozenset is used. Harmless in practice — worst case, a single tool call slips
    # through before deactivation takes effect. The frozenset snapshot pattern makes
    # this race safe without holding the lock across the membership check.
    def _check_mode_allowlist(self, tool_call: ToolCall, ctx: Any) -> Optional[str]:
        """Check if tool is allowed in the current mode.

        Dual authorization formula (spec §15):
            skill.get_tool_filter(ctx) ∩ (available_in_skills ∪ default_tools)
        
        Phase B path: Check agent.active_mode → skill.get_tool_filter()
        Priority chain: ctx._tools_snapshot > active_mode.allowed_tool_names > full registry
        """
        from proxy.agent.agent_registry import registry as agent_registry
        from proxy.skills.skill_registry import SkillRegistry

        session_id = getattr(ctx, "session_id", None)

        # Phase B: skill-aware filtering via agent.active_mode
        # Priority chain: ctx._tools_snapshot > active_mode.allowed_tool_names > full registry
        if session_id:
            try:
                _agent = agent_registry.get_or_load(session_id)
                if _agent is None:
                    logger.warning("_check_mode_allowlist: agent registry returned None for session %s", session_id)
                    return "Authorization failed: agent not found for session"
                # SEC-008: Read active_mode via snapshot (TOCTOU fix)
                mode_state = _agent.snapshot_mode()
                if mode_state is None:
                    return None
                # Phase B: Use skill's tool filter from active_mode
                skill_name = getattr(mode_state, "name", None)
                allowed_tools = getattr(mode_state, "allowed_tool_names", None)
                # R4-C4: allowed_tools is a frozenset reference snapshotted under lock.
                # frozenset is immutable, so the snapshot is safe even after lock release.
                # SEC-005: Use `is not None` — None means no restriction (all tools allowed).
                # Empty frozenset means NO tools allowed.
                if allowed_tools is not None and tool_call.name not in allowed_tools:
                    return (
                        f"Tool '{tool_call.name}' is not allowed. "
                        f"Active skill '{skill_name}' restricts tools to: {sorted(allowed_tools)}"
                    )
                return None
            except Exception as e:
                logger.error("Tool guard authorization check failed: %s", e, exc_info=True)
                return f"Authorization check failed for tool '{tool_call.name}'. Denying for safety."

        return None

    def _resolve_call_limits_for_skill(self, ctx: Any) -> dict:
        """Resolve the call_limits dict from the active skill.
        
        Returns empty dict on any failure or if not set.
        """
        try:
            from proxy.agent.agent_registry import registry as agent_registry
            from proxy.skills.skill_registry import get_default_registry

            session_id = getattr(ctx, "session_id", None)
            if not session_id:
                return {}

            agent = agent_registry.get_or_load(session_id)
            if agent is None:
                return {}
            mode_state = agent.snapshot_mode()
            if mode_state is None:
                return {}
            mode_name = getattr(mode_state, "name", None)
            if not mode_name:
                return {}

            registry = get_default_registry()
            skill_def = registry.get_skill(mode_name)
            if skill_def is None:
                return {}
            return getattr(skill_def, "call_limits", {}) or {}
        except InfrastructureError:
            raise  # let caller handle retry
        except Exception as e:
            logger.error(
                "Failed to resolve call_limits from skill definition: %s",
                e, exc_info=True,
            )
            return {}

    def _check_call_limits(self, tool_call: ToolCall, ctx: Any) -> Optional[str]:
        """Check per-request call limits via ToolCallTracker.
        
        Phase 5.5: Uses ToolCallTracker for:
        - proxy_ticket call count (50 max)
        - file read blocklist + read-once
        - sub-agent spawn count (3 max)
        """

        # grep path sandbox — block searches over sensitive paths
        if tool_call.name == "grep":
            search_path = tool_call.arguments.get("path") or tool_call.arguments.get("dir") or ""
            if search_path:
                reason = self._check_grep_sandbox(search_path, ctx)
                if reason:
                    return reason

        session_id = getattr(ctx, "session_id", None)
        if not session_id:
            return None

        tracker = ToolCallTracker.get_instance(session_id)
        call_limits = self._resolve_call_limits_for_skill(ctx)

        # proxy_ticket call count.
        # Slice 8a decomposed proxy_ticket_* names are the only live callers in
        # an active skill mode. The bare `== "proxy_ticket"` disjunct is RETAINED
        # (not dead): _check_mode_allowlist runs earlier but only blocks bare
        # proxy_ticket when an active mode restricts allowed_tool_names. In
        # DEFAULT mode (snapshot_mode() is None) the allowlist returns None and
        # a hallucinated bare `proxy_ticket` call reaches this counter. Dropping
        # the disjunct would stop counting those calls (an unbounded-call hole).
        if tool_call.name == "proxy_ticket" or tool_call.name in MemoryToolGuard._TICKET_TOOL_NAMES:
            max_limit = call_limits.get("proxy_ticket")
            if not tracker.try_count_proxy_ticket(tool_call.id, max_limit=max_limit):
                status = tracker.status_summary()
                effective_max = max_limit if max_limit is not None else status['proxy_ticket_max']
                return (
                    f"proxy_ticket call limit reached ({status['proxy_ticket_calls']}/{effective_max}). "
                    "Present what you have and summarize remaining work."
                )

        # US-088: report_bug call limit (max 5 per request) — runs AFTER the
        # total proxy_work counter below. The total counter is the broader
        # check (all proxy_work calls), report_bug is a per-action sub-limit.
        if tool_call.name == "proxy_work":
            # Total proxy_work call counter (per-skill via call_limits)
            max_limit = call_limits.get("proxy_work")
            if not tracker.try_count_proxy_work(tool_call.id, max_limit=max_limit):
                status = tracker.status_summary()
                effective_max = max_limit if max_limit is not None else status['proxy_work_max']
                return (
                    f"proxy_work call limit reached ({status['proxy_work_calls']}/{effective_max}). "
                    "Present what you have and summarize remaining work."
                )

            # Per-action sub-limit: report_bug (global MAX_REPORT_BUG_CALLS)
            action = tool_call.arguments.get("action", "")
            if action == "report_bug":
                if not tracker.try_count_report_bug(tool_call.id):
                    status = tracker.status_summary()
                    return (
                        f"report_bug call limit reached ({status.get('report_bug_calls', 0)}/{MAX_REPORT_BUG_CALLS}). "
                        "Present what you have and continue with the task."
                    )

        # read file blocklist + read-once (atomic check-and-record)
        if tool_call.name == "read":
            file_path = extract_file_path(tool_call.arguments)
            if file_path:
                # M3-2: cache hits bypass read-once. The cache means the
                # read already happened (and was already counted) in a
                # prior request. Counting it again would waste the per-
                # request read limit on a no-op.
                from_cache = bool(getattr(tool_call, "from_cache", False))
                can_read, reason = tracker.try_read_file(file_path, from_cache=from_cache)
                if not can_read:
                    return reason

        # Sub-agent spawn limit (atomic check-and-record)
        if tool_call.name == "task":
            max_limit = call_limits.get("task")
            if not tracker.try_spawn_subagent(session_id, tool_call.id, max_limit=max_limit):
                status = tracker.status_summary()
                effective_max = max_limit if max_limit is not None else status['subagent_max']
                return (
                    f"Sub-agent limit reached ({status['subagents_spawned']}/{effective_max}). "
                    "Present what you have and stop."
                )

        return None

    def _check_grep_sandbox(self, path: str, ctx: Any = None) -> Optional[str]:
        """Reject grep searches over sensitive file paths.

        grep defaults to "." (entire workspace) with no path guards like read has.
        This blocks search paths that directly target or traverse credential/sensitive files.
        """
        import os

        # H3: Path traversal protection - reject .. segments before resolution
        if ".." in path.split(os.sep) or ".." in path.split("/"):
            return (
                f"grep path blocked: '<redacted>' contains path traversal ('..'). "
                "Use absolute paths or paths relative to workspace root without '..'."
            )

        # H3: Resolve to absolute path and verify within workspace root
        # R1-HIGH-1: ctx.active_workspace contains workspace NAME, not path
        # Use cwd as the actual workspace root for filesystem sandboxing
        workspace_root = os.getcwd()
        workspace_root = os.path.realpath(workspace_root)
        resolved_path = os.path.realpath(path)

        if not resolved_path.startswith(workspace_root + os.sep) and resolved_path != workspace_root:
            return (
                f"grep path blocked: '<redacted>' resolves outside workspace root. "
                f"Path must be within {workspace_root}."
            )

        normalized = os.path.normpath(path).replace("\\", "/").rstrip("/")
        lower = normalized.lower()

        # Block exact or parent-dir matches for sensitive patterns
        SENSITIVE_SEGMENTS = (
            ".env", ".pem", ".git",
        )

        segments = lower.split("/")
        for seg in segments:
            for pattern in SENSITIVE_SEGMENTS:
                if seg == pattern:
                    return (
                        f"grep path blocked: '<redacted>' references sensitive item '{pattern}'. "
                        "Use a more specific path that excludes credential/secret/token files."
                    )
            for suffix in ('.env', '.pem', '.key', '.token', '.password', 'credentials', 'credential'):
                if seg.endswith(suffix):
                    return (
                        f"grep path blocked: '<redacted>' references sensitive file pattern '{suffix}'. "
                        "Use a more specific path that excludes credential/secret/token files."
                    )

        return None

    def _check_hash_provenance(self, tool_call: ToolCall, ctx: Any) -> Optional[str]:
        """Validate that MODIFY actions reference hashes the agent has seen.

        Phase 5.5: edit/update_task/set_parent/clear_parent/clear_blocked_by/set_priority
        must target hashes returned by a prior list_tasks call.
        Phase 4: proxy_work actions (start_work, stop_work, handoff_task, report_bug)
        must also target known hashes.
        """
        session_id = getattr(ctx, "session_id", None)
        if not session_id:
            return None

        if tool_call.name in MemoryToolGuard._TICKET_HASH_FIELDS:
            fields = MemoryToolGuard._TICKET_HASH_FIELDS[tool_call.name]
            tracker = ToolCallTracker.get_instance(session_id)
            for field in fields:
                val = (tool_call.arguments.get(field) or "")
                if isinstance(val, list):
                    for item in val:
                        item = (item or "").strip()
                        if item and not self._hash_is_known(tracker, item):
                            return (
                                f"Hash '{item}' was not returned by a list_tasks call in this request. "
                                "Call list_tasks first to discover valid hashes."
                            )
                elif isinstance(val, str):
                    val = val.strip()
                    if val and not self._hash_is_known(tracker, val):
                        return (
                            f"Hash '{val}' was not returned by a list_tasks call in this request. "
                            "Call list_tasks first to discover valid hashes."
                        )

        if tool_call.name == "proxy_work":
            action = tool_call.arguments.get("action", "")
            param_name = self.WORK_HASH_ACTIONS.get(action)
            if param_name is None:
                return None  # review_bug, validate_bug, resolve_bug — no hash
            tracker = ToolCallTracker.get_instance(session_id)
            hash_val = (tool_call.arguments.get(param_name) or "").strip()
            if not hash_val:
                return None  # Missing hash will be caught by action-level validation
            if not self._hash_is_known(tracker, hash_val):
                return (
                    f"Hash '{hash_val}' was not returned by a list_tasks call in this request. "
                    "Call list_tasks first to discover valid hashes."
                )
            return None

        return None

    def _hash_is_known(self, tracker: ToolCallTracker, short_hash: str) -> bool:
        """Check if a hash is known — tracker first, then TaskGraph fallback.

        Returns True if the hash exists (in tracker or TaskGraph) or on infra
        failure (fail-open). Returns False only when the hash is confirmed absent
        from both sources.
        """
        if tracker.has_seen_hash(short_hash):
            return True

        try:
            from proxy.context import _get_task_graph_singleton
            tg = _get_task_graph_singleton()
            task = tg.get_task_by_short_hash(short_hash)
            if task:
                tracker.add_seen_hashes([short_hash])
                return True
            return False
        except InfrastructureError as e:
            logger.error(
                "Hash provenance fail-closed for '%s': Neo4j unavailable, "
                "only in-memory tracker hashes accepted: %s",
                short_hash, e
            )
            return False  # FAIL-CLOSED: only tracker-known hashes accepted
        except AmbiguousHashError:
            # Hash matched multiple tasks — treat as known (it exists)
            tracker.add_seen_hashes([short_hash])
            return True

    def _check_hierarchy(self, tool_call: ToolCall, ctx: Any) -> Optional[str]:
        """Check sub-agent permissions for `proxy_*` tools.

        Sole gate: delegates to ``proxy.skill_guard.check_sub_agent_proxy_access``,
        which enforces the CAT-3 trust gate (only built-in trusted skills may
        grant `proxy_*` to sub-agents), per-skill access levels
        (proxy_ticket READ_ONLY/NEVER, proxy_work NEVER), and the
        skill-defined ``check_action`` callback for any other `proxy_*` tool.

        Mirrors: app.py:767-777 canary injection; Agent.is_subagent + skill
        registry.
        """
        from proxy.skill_guard import check_sub_agent_proxy_access
        from proxy.agent.agent_registry import registry as agent_registry
        from proxy.skills.skill_registry import SkillRegistry
        agent = agent_registry.get_or_load(ctx.session_id)
        return check_sub_agent_proxy_access(tool_call, ctx, agent, SkillRegistry())

    def _check_work_mode_actions(self, tool_call: ToolCall, ctx: Any) -> Optional[str]:
        """Restrict proxy_ticket actions for the MAIN orchestrator in work AND ticket modes.

        US-095: The MAIN agent in work mode is the orchestrator — it should
        only use list_tasks, get_task, and append_review_note on proxy_ticket.
        Sub-agents bypass this guard (access is enforced by skill_guard per
        the skill's access level — FULL for ticket subs, READ_ONLY for
        work subs per C4.5).

        US-096: The MAIN agent in ticket mode receives proxy_ticket read-only
        (list_tasks, get_task only). Write actions (create_tasks, update_task,
        edit, set_parent, clear_parent, clear_blocked_by, set_priority,
        append_review_note) are blocked for MAIN — delegate via task sub-agents.
        Sub-agents in ticket mode have FULL proxy_ticket access.
        (Work sub-agents have READ_ONLY per C4.5.)
        """
        if tool_call.name != "proxy_ticket" and tool_call.name not in MemoryToolGuard._TICKET_TOOL_NAMES:
            return None

        is_subagent = getattr(ctx, "is_subagent", False)
        if is_subagent:
            return None  # sub-agents bypass this check; skill_guard enforces access level

        session_id = getattr(ctx, "session_id", None)
        if not session_id:
            return None

        from proxy.agent.agent_registry import registry as agent_registry
        agent = agent_registry.get_or_load(session_id)
        if agent is None:
            logger.warning("_check_work_mode_actions: agent registry returned None for session %s", session_id)
            return "Authorization failed: agent not found for session"

        mode_state = agent.snapshot_mode()
        if mode_state is None:
            return None
        mode_name = getattr(mode_state, "name", None)

        # C7b: delegate to capability_table when available (proven behavior-neutral
        # by tests/unit/test_guard_table_parity.py).
        from proxy.skills.skill_registry import SkillRegistry as _SkillReg
        from proxy.skills.skill_definition import CapabilityTable
        _reg = _SkillReg()
        _skill = _reg.get(mode_name) if mode_name else None
        _table = getattr(_skill, 'capability_table', None) if _skill else None
        if isinstance(_table, CapabilityTable):
            # Decomposed ticket tools encode the action in the tool NAME; the
            # capability table keys on (tool_name, role), so no action argument
            # is needed. A stray bare `proxy_ticket` has no table entry and is
            # denied fail-closed by check_action.
            return _table.check_action(tool_call.name, None, "main")

        # Fail-closed fallback: a skill without a capability_table (custom
        # skill) has no configured proxy_ticket* policy. Both builtins (ticket,
        # work) define a capability_table and never reach here; custom skills
        # are denied proxy_ticket* at load. Deny rather than silently allow.
        return (
            f"Tool '{tool_call.name}' is not available in this skill mode."
        )
    def _check_ticket_mode_orchestrator(self, tool_call: ToolCall, ctx: Any) -> Optional[str]:
        """Block proxy_work for the MAIN agent in ticket mode, and block both
        proxy_ticket and proxy_work for bare sub-agents (no canary skill) when
        the orchestrator is in ticket mode.

        Sub-agents spawned during ticket mode (subagent_ticket_mode=True) are
        allowed. The MAIN agent must delegate proxy_work operations to
        sub-agents via the task tool.

        proxy_ticket for MAIN in ticket mode is read-only and is enforced by
        ``_check_work_mode_actions`` (runs earlier in the guard chain). This
        guard does NOT block proxy_ticket for MAIN in ticket mode — only for
        bare sub-agents (subagent_ticket_mode=False), which is the canary-cache
        defense: a bare sub-agent without a verified ticket canary must not
        reach proxy_ticket even when the orchestrator's persistent mode is
        ticket.

        MAIN agent in ticket mode is unconditionally blocked from
        ``proxy_work``: ticket mode is one-shot — only work-mode agents
        run work sessions, so the orchestrator has no use for the work
        tool surface in ticket mode.
        """
        # Slice 8a decomposed proxy_ticket_* names are the live callers. The
        # bare `== "proxy_ticket"` disjunct is RETAINED (not provably dead):
        # dropping it would make a bare `proxy_ticket` call fall through the
        # `if not (...)` guard below and early-return None, bypassing the
        # sub-agent / capability_table branches that this guard still applies to
        # non-MAIN and non-ticket-mode paths. Keeping it preserves behavior.
        _is_proxy_ticket = tool_call.name == "proxy_ticket" or tool_call.name in MemoryToolGuard._TICKET_TOOL_NAMES
        _is_proxy_work = tool_call.name == "proxy_work"
        if not (_is_proxy_ticket or _is_proxy_work):
            return None

        is_subagent = getattr(ctx, 'is_subagent', False)
        subagent_ticket_mode = getattr(ctx, 'subagent_ticket_mode', False)
        if is_subagent and subagent_ticket_mode:
            return None  # ticket sub-agents are allowed

        # proxy_ticket for MAIN in ticket mode is read-only, enforced by
        # _check_work_mode_actions (which runs earlier in the guard chain
        # and rejects write actions). MAIN read-only proxy_ticket calls reach
        # here and must NOT be blocked — only bare sub-agents are blocked.
        if _is_proxy_ticket and not is_subagent:
            return None

        # C7b: delegate proxy_work blocking for ticket MAIN to capability_table.
        from proxy.skills.skill_registry import SkillRegistry as _SkillReg
        from proxy.skills.skill_definition import CapabilityTable
        from proxy.agent.agent_registry import registry as agent_registry
        _reg = _SkillReg()
        session_id = getattr(ctx, "session_id", None)
        agent = agent_registry.get_or_load(session_id) if session_id else None
        mode_state = agent.snapshot_mode() if agent is not None else None
        mode_name = getattr(mode_state, 'name', None) if mode_state else None
        if mode_name:
            _skill = _reg.get(mode_name)
            _table = getattr(_skill, 'capability_table', None) if _skill else None
            if isinstance(_table, CapabilityTable):
                action = tool_call.arguments.get("action", "")
                _result = _table.check_action("proxy_work", action, "main")
                if _result is not None:
                    return _result
        # Legacy fallback: ticket MAIN blocked unconditionally from proxy_work
        if mode_name == "ticket":
            return (
                f"Tool '{tool_call.name}' is blocked for the MAIN agent in ticket mode. "
                f"Delegate to sub-agents via the `task` tool."
            )
        return None

    def _check_blocked_tools_for_subagent(self, tool_call: ToolCall, ctx: Any) -> Optional[str]:
        """Block tools listed in the active skill's `blocked_tools_for_subagent`.

        Reads the denylist from the skill's `SkillDefinition.blocked_tools_for_subagent`
        set (which defaults to `{"question"}` and is overridden by
        TicketSkill/WorkSkill to also block `task`). Sub-agents cannot
        spawn sub-sub-agents (current CLIs do not support nesting), and
        sub-agents cannot call `question` (would deadlock).

        Defense in depth: if no skill is active, or the skill definition
        cannot be resolved, falls back to a hardcoded denylist of
        `{"task", "question"}`.
        """
        is_subagent = getattr(ctx, "is_subagent", False)
        if not is_subagent:
            return None

        blocked = self._resolve_blocked_tools_for_subagent(ctx)
        if tool_call.name in blocked:
            return (
                f"Tool '{tool_call.name}' is not available to sub-agents. "
                "Sub-agents cannot spawn sub-sub-agents (current CLIs do not "
                "support nesting)."
            )
        return None

    def _resolve_blocked_tools_for_subagent(self, ctx: Any) -> frozenset[str]:
        """Resolve the blocked-tools set from the active skill, with hardcoded fallback.

        Reads the agent's active mode name via `snapshot_mode()`, looks up
        the matching `SkillDefinition` in the default registry, and returns
        its `blocked_tools_for_subagent`. On any failure (no active mode,
        skill not registered, missing attribute, registry error) falls
        back to `{"task", "question"}` (defense in depth).
        """
        HARDCODED_FALLBACK = frozenset({"task", "question"})
        try:
            from proxy.agent.agent_registry import registry as agent_registry
            from proxy.skills.skill_registry import get_default_registry

            session_id = getattr(ctx, "session_id", None)
            if not session_id:
                return HARDCODED_FALLBACK

            agent = agent_registry.get_or_load(session_id)
            if agent is None:
                return HARDCODED_FALLBACK
            mode_state = agent.snapshot_mode()
            if mode_state is None:
                return HARDCODED_FALLBACK
            mode_name = getattr(mode_state, "name", None)
            if not mode_name:
                return HARDCODED_FALLBACK

            registry = get_default_registry()
            skill_def = registry.get_skill(mode_name)
            if skill_def is None:
                return HARDCODED_FALLBACK
            blocked = getattr(skill_def, "blocked_tools_for_subagent", None)
            if blocked is None:
                return HARDCODED_FALLBACK
            return frozenset(blocked)
        except Exception as e:
            logger.error(
                "Failed to resolve blocked_tools_for_subagent from skill "
                "definition; falling back to hardcoded set: %s",
                e, exc_info=True,
            )
            return HARDCODED_FALLBACK
