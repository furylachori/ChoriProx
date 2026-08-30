# `src/skills/AGENTS.md` — Concrete Skill Classes

> The actual skill implementations: TicketSkill (888 LOC), WorkSkill (439 LOC),
> WorkspaceSkill (501 LOC). All three extend `BaseSkill` from
> `proxy/skills/skill_definition.py`. Skill infrastructure (registry, trust gate,
> custom JSON loader) lives in `src/proxy/skills/AGENTS.md`.

## Module Purpose

Owns the three concrete skill classes:
- **TicketSkill** — per-turn planner, fire-and-forget, planner-only tools
- **WorkSkill** — interactive worker, full coding tools
- **WorkspaceSkill** — parser-only direct commands (NO mode — workspace mode was never built)

## Files in this module

| File | LOC | Purpose |
|---|---|---|
| `__init__.py` | | Package marker |
| `ticket_skill.py` | 888 | `TicketSkill` — `^^ticket` planner |
| `work_skill.py` | 439 | `WorkSkill` — `^^work` interactive executor; sub-agent capability gating via `WORK_CAPABILITY_TABLE` / `proxy.skill_guard` |
| `workspace.py` | 501 | `WorkspaceSkill` — `^^workspace` parser-only direct commands |

## TicketSkill (ticket_skill.py)

```python
class TicketSkill(BaseSkill):
    name: str = "ticket"
    trigger: str = "^^ticket"
    aliases: tuple = ()
    sub_agent_proxy_ticket_access: ProxyAccessLevel = ProxyAccessLevel.FULL
    sub_agent_proxy_work_access: ProxyAccessLevel = ProxyAccessLevel.NEVER
    ttl_seconds: int = 300
    fire_and_forget: bool = True  # non-interactive
    blocked_tools_for_subagent: frozenset = frozenset({"question", "task"})
    call_limits: dict = {"proxy_ticket": 50}
    fallback_skill: None  # no fallback
    check_action: None  # handled via tool_guard layer
```

### Lifecycle (per-turn contract)

- **`should_deactivate(ctx, finish_reason) -> finish_reason == "stop"`** (line 831).
  Ticket mode clears on `finish_reason='stop'` via PostProcessStage. A downstream
  disconnect does not clear mode (completion is the sole authority — per ticket
  199ece93 §4). To continue in ticket mode, the user re-invokes `^^ticket`.
- **`fire_and_forget=True`** — TTL=300s, but the per-turn deactivation fires
  first on `finish_reason="stop"`.

### Tool filters

**`get_tool_filter(ctx)`** (line 719) — main agent:
```python
frozenset({
    "read", "glob", "grep", "search",
    "proxy_ticket",
    "task", "webfetch",
})
```

**`get_sub_agent_tool_filter(ctx)`** (line 726) — sub-agent (no `task`, `proxy_work` per C4.5):
```python
frozenset({
    "read", "glob", "grep", "search",
    "proxy_ticket",
    "webfetch",
})
```

Planners don't write code directly — no `write`/`edit`/`bash` in either filter.

### `parse_subcommand(text, session_id)` (line 119)

Returns:
- `("list", "")` — list tasks (no args or `^^ticket list`)
- `("create", text)` — plan tasks from natural language
- `("complete", task_hash)` — close a task
- `("invalid", task_hash)` — mark task invalid
- `("review", task_hash)` — add review note
- `("approve", task_hash)` — approve a bug
- `(None, "")` — not a ticket command

### `handle_direct(ctx, action, arg)` (line 238)

Routes parsed actions to:
- `proxy_ticket` actions (`list_tasks`, `get_task`, `update_task`, `append_review_note`, `complete_task`, `approve_bug`)
- `_plan_fallback` (line 101) — preserves FULL user text for LLM plan-mode when input is natural language

### `get_system_prompt(ctx)` (line 529)

Returns `proxy.tool_parsing.ticket_prompts.TICKET_MAIN_PROMPT`.

### `get_subagent_prompt(ctx)` (line 700)

Returns `TICKET_SUB_PROMPT` (additive preamble).

### `get_inheritance_state(ctx)` (line 741)

Returns state for sub-agents:
```python
{
    "tasks": [{short_hash, name, priority, status, summary, tag,
               blocked_by, is_blocked, blocked_count, parent_short_hash, parent_name}],
    "store_status": tg.get_store_status(),
    "workspace_name": workspace_name  # if not DEFAULT_WORKSPACE
}
```

Truncated to `MAX_INHERITANCE_TASKS=100`. Uses singleton `TaskGraph` via
`_get_task_graph_singleton()`.

### `get_canary(ctx)` (line 733)

Calls `self.build_skill_canary(ctx)` — produces the HMAC-signed canary marker
for sub-agent ingress. See `proxy/canary.py` for parser + verifier.

## WorkSkill (work_skill.py)

```python
class WorkSkill(BaseSkill):
    name: str = "work"
    trigger: str = "^^work"
    aliases: tuple = ()
    sub_agent_proxy_ticket_access: ProxyAccessLevel = ProxyAccessLevel.READ_ONLY
    sub_agent_proxy_work_access: ProxyAccessLevel = ProxyAccessLevel.FULL
    ttl_seconds: int = 300
    fallback_skill: Optional[str] = None  # no fallback (deleted in C3)
    fire_and_forget: bool = False  # interactive
    blocked_tools_for_subagent: frozenset = frozenset({"question", "task"})
    call_limits: dict = {"proxy_work": 50, "task": 3}
    capability_table = WORK_CAPABILITY_TABLE  # enforced via proxy.skill_guard
```

### Lifecycle (interactive contract)

- **`should_deactivate(ctx, finish_reason) -> finish_reason == "stop"`** — same per-turn contract as ticket. Completion is the sole mode-clear authority (PostProcessStage); a downstream disconnect does not clear work mode either.
- **`fire_and_forget=False`** — agent actively uses tools.
- **TTL=300s.** Backstop only — the per-turn `stop` deactivation fires first under the one-shot contract. Orphaned work sessions are handled by the 24h stale sweep (`work_sessions.py:cleanup_stale_sessions`). The work→ticket fallback machinery and Bug #3 stop_work exemption were deleted in PROXY_MODES_FIX_PLAN C3.

### Tool filters

**`get_tool_filter(ctx)`** (line 386) — main agent (full coding):
```python
frozenset({
    "read", "write", "edit", "bash", "glob", "grep", "search",
    "task", "proxy_work", "proxy_ticket",
})
```

**`get_sub_agent_tool_filter(ctx)`** (line 393) — sub-agent (no `task`):
```python
frozenset({
    "read", "write", "edit", "bash", "glob", "grep", "search",
    "proxy_work", "proxy_ticket",
})
```

### `parse_subcommand(text, session_id)` (line 95)

Returns:
- `("task", task_hash)` — activate work mode for task
- `("list", "")` — list tasks (alias for `^^ticket list`)
- `(None, "")` — not a work command

### `handle_direct(ctx, action, arg)` (line 118)

Routes to:
- `action == "list"` — delegates to `TicketSkill.handle_direct(ctx, "list", arg)`
- Returns `None` for other actions (activation happens via `on_activate`)

### US-095 enforcement (work sub-agents)

C7b: the legacy `_check_work_mode_action` callback has been DELETED. The
US-095 lifecycle exclusion (start_work/stop_work/complete_work blocked for
work sub-agents; report_bug/review_bug/etc. allowed) is now encoded in
`WORK_CAPABILITY_TABLE.actions_allowed` for role='subagent' (the table's
subagent row for `proxy_work`). Enforcement happens at
`proxy.skill_guard.check_sub_agent_proxy_access` → `table.check_action()`.

**Table-driven rules (work sub-agent role):**
- `proxy_ticket` — READ-ONLY: `list_tasks`, `get_task`, `append_review_note`.
- `proxy_work` — Lifecycle actions EXCLUDED from `actions_allowed`
  (start_work, stop_work, complete_work return deny_message). Bug actions
  ALLOWED: `report_bug`, `review_bug`, `validate_bug`, `resolve_bug`,
  `handoff_task`.

The capability table is the single source of truth for BUILTIN skills
(ticket, work). Custom skills use legacy ProxyAccessLevel + check_action
fallback paths.

### `get_inheritance_state(ctx)` (line 402)

Returns state for sub-agents:
```python
{"task_hash": task_hash}  # if available
# plus workspace_name if not DEFAULT_WORKSPACE
```

`task_hash` comes from `skill_state["arg"]` (set by `handle_direct`) or
`ctx.work_task_hash` (populated by `WorkSkill.on_activate`).

### `get_canary(ctx)` (line 427)

Calls `self.build_skill_canary(ctx)` — produces the work-skill canary marker.

### `on_activate(ctx, mode_state)` (line 135)

Captures `task_hash` from skill_state for the system prompt to anchor the
agent to a specific task. Stored in `ctx.work_task_hash`.

## WorkspaceSkill (workspace.py)

**Workspace mode was NEVER BUILT** (US-066). `^^workspace` is parser-only
direct commands. The LLM never sees a workspace system prompt.

This module keeps:
- All `^^workspace <subcommand>` direct commands (switch, create, forget,
  recent, list, status, leave, help) — US-066 through US-072.
- `parse_subcommand` and `handle_direct`.
- `get_inheritance_state` for sub-agent workspace propagation via the canary.
- Workspace name validation.

This module REMOVES (the mode scaffold):
- `get_tool_filter`, `get_system_prompt` — never called.
- `on_activate`, `on_deactivate`, `should_deactivate`, `_deactivate_workspace` — mode-activation paths.
- `get_canary` — only used in conjunction with mode activation.

If a future feature needs an actual workspace mode, build it fresh — do not
resurrect from this scaffold.

### Subcommands (`parse_subcommand`, line 134)

| Subcommand | Action | Handler |
|---|---|---|
| `^^workspace` (bare) | "status" | show current workspace |
| `^^workspace <name>` | "switch" | activate workspace |
| `^^workspace create <name>` | "create" | new + switch |
| `^^workspace forget <name>` | "forget" | remove from MRU |
| `^^workspace recent` | "recent" | list MRU |
| `^^workspace list` | "list" | global catalog |
| `^^workspace status` | "status" | show current |
| `^^workspace leave` | "leave" | deactivate |
| `^^workspace help` | "help" | usage text |

### Workspace name validation (line 52)

```python
_WORKSPACE_NAME_PATTERN = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9-]*$")
_MAX_WORKSPACE_NAME_LENGTH = 64
_RESERVED_WORKSPACE_NAMES = {"leave", "default", "recent", "help", "list",
                              "status", "admin", "system", "all", "create",
                              "forget", "cleanup"}
```

Typo-guard: `_FUZZY_MATCH_CUTOFF = 0.7` catches single-char typos for
workspace names of length 4-15.

### `get_inheritance_state(ctx)` (line 468)

Returns `{"workspace_name": ws, "workspace_id": ws}` — pure data, no mode activation.
Used by the canary flow to propagate the active workspace to spawned sub-agents.

### `on_activate` / `on_deactivate` / `should_deactivate` (line 484-499)

All no-op stubs. Workspace mode never activates.

## Key Invariants

1. **TicketSkill is fire-and-forget, WorkSkill is interactive.** Both have `should_deactivate -> finish_reason == "stop"`.
2. **WorkSkill fallback is None** — no work→ticket TTL fallback. Modes are one-shot (PROXY_MODES_FIX_PLAN C3).
3. **WorkSkill sub-agents are BLOCKED from `start_work`/`stop_work`/`complete_work`** via `WORK_CAPABILITY_TABLE.actions_allowed` for role='subagent' (lifecycle actions excluded from the subagent row).
4. **WorkSkill sub-agents have READ-ONLY proxy_ticket access (list_tasks, get_task, append_review_note) per C4.5** — they're the workers.
5. **TicketSkill main agent is read-only on proxy_ticket** — only `list_tasks`/`get_task` (enforced by `tool_guard._check_work_mode_actions`). Ticket sub-agents have FULL proxy_ticket access; work sub-agents have READ_ONLY.
6. **WorkspaceSkill has NO mode.** Workspace mode was never built (US-066). Parser-only direct commands.
7. **Custom JSON skills** (per `src/proxy/skills/AGENTS.md`) have direct-dispatch behavior callbacks — the same per-turn `should_deactivate -> finish_reason == "stop"` contract.
8. **`on_activate` runs `set_active_mode(mode_state)` then sets `ctx._skill_activated = True`** (in `_activate_skill`, `src/proxy/agent/agent.py:39`).
9. **`_inherit_skill` skips `on_activate`** (line 110 of agent.py) — sub-agent inheritance just sets the mode without top-level hooks.

## Landmines

- **Workspace mode was NEVER built.** Don't add mode-related code to `WorkspaceSkill` (no `get_tool_filter`, no `get_system_prompt`).
- **Custom skills use `should_deactivate -> finish_reason == "stop"`** — same per-turn contract. Don't override with TTL-based deactivation.
- **`build_skill_canary` is the canonical canary builder** — sub-classes call `self.build_skill_canary(ctx)`. Don't hand-construct canary strings.
- **WorkSkill sub-agent bug actions are NOT blocked by the capability table** — `WORK_CAPABILITY_TABLE.actions_allowed` for role='subagent' allows `proxy_work` bug actions (`report_bug`, `review_bug`, `validate_bug`, `resolve_bug`, `handoff_task`); only lifecycle actions (`start_work`/`stop_work`/`complete_work`) are excluded from the subagent row.
- **TicketSkill sub-agent `task` is blocked** (in `blocked_tools_for_subagent = frozenset({"question", "task"})`) — sub-agents cannot spawn sub-sub-agents.
- **`call_limits` differ** — TicketSkill: `{"proxy_ticket": 50}`. WorkSkill: `{"proxy_work": 50, "task": 3}`. Used by `MemoryToolGuard._check_call_limits`.
- **`MAX_INHERITANCE_TASKS=100`** (ticket_skill.py:29) — truncate `get_inheritance_state` to prevent memory spikes for sub-agents.
- **`fire_and_forget` semantics** — `True` for ticket (TTL=300s but per-turn deactivation fires first); `False` for work (interactive).
- **`fallback_skill` is None for WorkSkill** — no TTL fallback; one-shot modes clear on `stop`. TTL=300s is a backstop for sessions that never reach `stop`.

## Code Conventions

```python
# CORRECT — activate a skill
from proxy.skills.skill_definition import ModeState
from proxy.agent.agent import _activate_skill
mode_state = ModeState(
    name=skill.name,
    activated_at=time.time(),
    ttl_seconds=skill.ttl_seconds,
    allowed_tool_names=skill.get_tool_filter(skill_ctx),
    skill_state={},
)
agent.set_active_mode(mode_state)

# CORRECT — inherit a skill for a sub-agent (no on_activate)
from proxy.agent.agent import _inherit_skill
sub_mode_state = ModeState(
    name=skill.name,
    activated_at=time.time(),
    ttl_seconds=skill.ttl_seconds,
    allowed_tool_names=skill.sub_agent_tool_filter,
    sub_agent_tool_filter=skill.sub_agent_tool_filter,
    skill_state={},
)
agent.set_active_mode(sub_mode_state)

# WRONG — direct mode.set without _activate_skill
agent.active_mode = mode_state  # bypasses deactivation callbacks; use set_active_mode
```

## External Dependencies

**Imports from:**
- `proxy.skills.skill_definition` — `BaseSkill`, `ModeState`, `ProxyAccessLevel`, `SkillRequestContext`, `SkillValidationError`.
- `proxy.skills.skill_registry` — `SkillMeta`, `error_response`, `_normalize_text`.
- `proxy.skills.command_text_utils` — `extract_command_text`, `strip_environment_details`.
- `proxy.context._get_task_graph_singleton` — singleton access.
- `proxy.workspace_state` — `workspace_store`, `resolve_workspace`, `DEFAULT_WORKSPACE`.
- `proxy.tool_parsing.ticket_prompts.TICKET_MAIN_PROMPT` (MAIN) / `TICKET_SUB_PROMPT` (sub-agent).
- `proxy.tool_parsing.work_prompts.WORK_MANAGER_PROMPT`.
- `commands.task_commands.is_work_trigger` — predicate.
- `memory.task_graph.TaskGraph.SHORT_HASH_RE` — short-hash regex.

**Imported by:**
- `src/proxy/skills/skill_registry.py` — `populate_builtins` instantiates TicketSkill/WorkSkill.
- `src/proxy/pipeline/dispatch.py` — `SkillDispatcher` routes triggers.
- `src/proxy/pipeline/post_process.py` — `_apply_skill_response_transform` / `_inject_task_canaries` / `_finalize_skill_lifecycle` invoke the split post-process helpers from `proxy.mode`.
- Tests under `tests/skills/`, `tests/proxy/`, `tests/integration/`.

## Cross-References

- `src/proxy/skills/AGENTS.md` — `SkillRegistry` + `BaseSkill` + `SkillDefinition`.
- `src/proxy/agent/AGENTS.md` — `Agent.set_active_mode` + `_activate_skill` / `_inherit_skill`.
- `src/proxy/canary.py` (module docstring) — `parse_canary` + `_locate_canary` + `_ingress_from_parse`.
- `src/commands/AGENTS.md` — `^^workspace` is dispatched as a command (NOT a skill) via `commands/workspace_commands.py`. `WorkspaceSkill.handle_direct` is called from there.
- `src/memory/tools/AGENTS.md` — `execute_ticket_tool` / `execute_proxy_work` are the actions invoked from ticket/work skills.