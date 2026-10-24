# `src/memory/tools/AGENTS.md` — Memory Tools (4)

> The 4 memory tools that the proxy intercepts locally. Action registration
> happens at import time via `proxy.tool_pipeline.action_registry.register_action()`.
> See `src/proxy/tool_pipeline/AGENTS.md` for the 17 TaskAction classes.

## Module Purpose

Local execution handlers for:
- `recall` (memory tool, STUB)
- `search` (memory tool, STUB)
- `proxy_ticket` (10 ticket actions + dispatcher)
- `proxy_work` (7 work actions + dispatcher + mode gate)

Plus the top-level dispatcher `execute_memory_tool(name, arguments)`.

## Files in this module

| File | Purpose |
|---|---|
| `__init__.py` | Re-exports `execute_memory_tool`, `execute_ticket_tool`, `execute_proxy_work`, `recall`, `search`, schema definitions |
| `dispatch.py` | `execute_memory_tool(name, arguments)` — top-level dispatcher routing to the right handler |
| `proxy_ticket.py` | `execute_ticket_tool(operation, arguments, session_id, client_hint)` — mode gate + `_find_parent_workspace` for sub-agent inheritance + registers 10 ticket actions at import time |
| `proxy_work.py` | `execute_proxy_work(arguments, session_id, client_hint)` — mode gate + workspace_id extraction + registers 7 work actions at import time |
| `recall.py` | `recall(queries, keywords, limit, session_id)` — **STUB** (always returns empty) |
| `search.py` | `search(keywords, match_all, session_id, time_start, time_end, limit)` — **STUB** (always returns empty) |
| `schemas.py` | Tool definitions (`MEMORY_TOOL_DEFINITIONS`, `PROXY_TICKET_TOOL_DEFINITION`, `PROXY_WORK_TOOL_DEFINITION`) |

## `execute_memory_tool(name, arguments)` (dispatch.py)

The single entry point called from `proxy/tool_pipeline/tool_pipeline.py`
when a memory tool is detected. Routes:

```python
if name == "recall":       → memory.tools.recall.recall
elif name == "search":     → memory.tools.search.search
elif name in TICKET_TOOL_OPERATIONS: → execute_ticket_tool(operation, ...)  # 10 decomposed proxy_ticket_* names
elif name == "proxy_work": → execute_proxy_work
else:                       → {"error": f"Unknown memory tool: {name}"}
```

Returns the result dict (or `{error: ...}` envelope on unknown tool).
`InfrastructureError` is re-raised (let caller handle retry). Other
exceptions are caught + returned as `{error: str(e)}`.

## `recall` and `search` — STUBS

Both return empty results. The Neo4j read path for memory recall is
deliberately disabled.

```python
def recall(queries, keywords, limit=10, session_id=None):
    return {"results": [], "total": 0, "truncated": False}

def search(keywords, match_all=False, session_id=None, time_start=None, time_end=None, limit=10):
    return {"results": [], "total": 0}
```

The `memory/recall.py` (top-level, NOT `memory/tools/recall.py`) provides
`filter_memories_for_context` + `format_memories_for_injection` which are
used for V2 context-injection, not for the memory-tool handlers.

## `execute_ticket_tool` (memory/tools/proxy_ticket.py:110)

The ticket-mode tool dispatcher. Dispatches by the `action` field to one
of the 10 registered TaskAction classes:

| Action | Class |
|---|---|
| `create_tasks` | `CreateTasksAction` |
| `update_task` | `UpdateTaskAction` |
| `list_tasks` | `ListTasksAction` |
| `get_task` | `GetTaskAction` |
| `edit` | `EditAction` |
| `set_parent` | `SetParentAction` |
| `clear_parent` | `ClearParentAction` |
| `clear_blocked_by` | `ClearBlockedByAction` |
| `set_priority` | `SetPriorityAction` |
| `append_review_note` | `AppendReviewNoteAction` |

### `_find_parent_workspace(subagent_session_id)` (line 34)

Scans the agent registry for a non-subagent agent with an active ticket
mode and returns its workspace_name. Used for sub-agent workspace
inheritance when the SKILL canary format doesn't carry `workspace_name`.

Returns the parent's `workspace_name` (most recent by `_ticket_mode_ts`),
or `None`.

### Action registration (line 98-107)

```python
from proxy.tool_pipeline.action_registry import register_action
from proxy.tool_pipeline.actions import (
    CreateTasksAction, UpdateTaskAction, ListTasksAction, GetTaskAction,
    EditAction, SetParentAction, ClearParentAction, ClearBlockedByAction,
    SetPriorityAction, AppendReviewNoteAction,
)
register_action("create_tasks", CreateTasksAction)
register_action("update_task", UpdateTaskAction)
# ... (10 total)
```

Runs at module load. The `ACTION_REGISTRY` global dict in
`proxy/tool_pipeline/action_registry.py` is mutated as a side-effect.

## `execute_proxy_work` (memory/tools/proxy_work.py:33)

The work-mode tool dispatcher. Mode-gated to `work` only (C3 tightened
from work-or-ticket to work-only):

```python
WORK_ACTIONS = {
    "start_work", "stop_work", "report_bug", "review_bug",
    "validate_bug", "resolve_bug", "handoff_task",
}
```

### Mode gate (line 51-80)

```python
# Read mode + ticket_ts atomically
_mode, _ticket_ts = _agent.snapshot_mode_and_ticket_ts()
if _mode is None:
    in_work_mode = False  # C3: removed work-or-ticket fallback path
else:
    _mode_name = getattr(_mode, "name", None)
    in_work_mode = _mode_name == "work" and not _mode.is_expired()

if not in_work_mode:
    from proxy.tool_pipeline.action_envelope import error_envelope
    return error_envelope(
        "proxy_work", None, "mode_inactive",
        "This session is not in work mode. ...",
        session_id,
    )
```

Default mode and workspace mode are rejected with `mode_inactive`.

### Workspace resolution (line 120-137)

```python
# Try active_mode.skill_state.workspace_name, fall back to workspace_store
mode_state = _agent.snapshot_mode()
workspace_name = skill_state.get("workspace_name") if mode_state else None
if not workspace_name:
    workspace_name = workspace_store.get_active(session_id, client_hint)
if workspace_name:
    workspace_id = workspace_name
```

### Action registration (line 24-30)

```python
register_action("start_work", StartWorkAction)
register_action("stop_work", StopWorkAction)
register_action("report_bug", ReportBugAction)
register_action("review_bug", ReviewBugAction)
register_action("validate_bug", ValidateBugAction)
register_action("resolve_bug", ResolveBugAction)
register_action("handoff_task", HandoffTaskAction)
```

## `dispatch_action` error envelopes (action_registry.py:44)

`dispatch_action` returns canonical spec §11 envelopes built by
`action_envelope.error_envelope()` — never a bare `{"error": ...}` dict:

```python
{
  "ok": False, "tool": ..., "operation": ..., "status": "error",
  "data": None, "session_id": ...,
  "error": {"code": <code>, "message": ..., "retryable": ..., "category": ...},
}
```

The six error `code` values:

```python
"invalid_action"        # unknown action
"ambiguous_hash"        # AmbiguousHashError
"infrastructure_error"  # InfrastructureError
"query_timeout"         # QueryTimeoutError
"constraint_violation"  # ConstraintViolation
"action_failed"         # generic Exception
```

Authorization is enforced by `execute_ticket_tool` / `execute_proxy_work`
BEFORE `dispatch_action` is called. The guard layer
(`proxy/tool_pipeline/tool_guard.py`) gates access by mode + skill.

## Action classes (17 — see `proxy/tool_pipeline/actions/`)

Each action is a small class implementing the `TaskAction` Protocol:

```python
class TaskAction(Protocol):
    def execute(self, arguments: dict, session_id: str, tg: Any,
                workspace_id: str = None, caller_agent_id: str = None) -> dict:
        ...
```

### `proxy_ticket` actions (10)

| Action | File | Purpose |
|---|---|---|
| `create_tasks` | `actions/create_tasks.py` | Batch create tasks (UNWIND/MERGE) |
| `update_task` | `actions/update_task.py` | Status transition + multi-field update (assignment guard) |
| `list_tasks` | `actions/list_tasks.py` | `get_actionable_tasks` with workspace filter |
| `get_task` | `actions/get_task.py` | `get_task_enriched` (with ReviewNote) |
| `edit` | `actions/edit.py` | Editing fields (name, summary, tag, planning_info) |
| `set_parent` | `actions/set_parent.py` | `[:HAS_PARENT]` relationship (cycle detection via `[:HAS_PARENT*1..20]`) |
| `clear_parent` | `actions/clear_parent.py` | Drop `[:HAS_PARENT]` |
| `clear_blocked_by` | `actions/clear_blocked_by.py` | Drop `[:BLOCKED_BY]` |
| `set_priority` | `actions/set_priority.py` | low/medium/high/critical |
| `append_review_note` | `actions/append_review_note.py` | Create `(:ReviewNote)` + `[:HAS_NOTE]` |

### `proxy_work` actions (7)

| Action | File | Purpose |
|---|---|---|
| `start_work` | `actions/start_work.py` | Create `(:WorkSession)` (active) |
| `stop_work` | `actions/stop_work.py` | Close `(:WorkSession)` (completed/aborted) |
| `report_bug` | `actions/report_bug.py` | Create `(:Bug)` (open) |
| `review_bug` | `actions/review_bug.py` | Create `(:Review)` on `(:Bug)` |
| `validate_bug` | `actions/validate_bug.py` | `Bug` transition open → approved |
| `resolve_bug` | `actions/resolve_bug.py` | `Bug` transition approved → resolved |
| `handoff_task` | `actions/handoff_task.py` | Cross-agent task transfer |

## Key Invariants

1. **`recall` / `search` are STUBS.** Always return empty results. Don't try to "fix" them — they're deliberately disabled.
2. **`execute_proxy_work` mode gate is strict.** Default mode + workspace mode → `mode_inactive`.
3. **`execute_ticket_tool` has NO mode gate** (the guard layer handles it via `MemoryToolGuard._check_mode_allowlist`).
4. **Action registration at import time.** `register_action(...)` runs at module load. If you add a new action class, also import + register in the appropriate `proxy_ticket.py` / `proxy_work.py`.
5. **`_find_parent_workspace` walks `agent_registry._agents`** looking for a non-subagent agent in ticket mode. Picks the most recent by `_ticket_mode_ts`.
6. **`dispatch_action` handles 6 error categories.** Authorization is upstream; this layer just translates exceptions to envelopes.

## Landmines

- **Mode gate on `proxy_work`.** The MAIN agent in ticket mode is blocked from `proxy_work` (C3 deleted the Bug #3 stop_work exemption — no legitimate `proxy_work` call exists for ticket-mode agents under the one-shot contract). Orphaned work sessions are handled by the 24h stale sweep (`work_sessions.py:cleanup_stale_sessions`).
- **`_find_parent_workspace` race.** The agent registry lock (#0) is NOT held during the scan. Snapshot inconsistency is possible during heavy spawning.
- **`WORK_ACTIONS` set size.** Adding a new work action requires adding to this set AND registering in `proxy_work.py:24-30`.
- **Action registration order.** Registration is a dict mutation; later registrations overwrite earlier. New action names MUST be unique.
- **`AmbiguousHashError` is caught at `dispatch_action` level.** Hash uniqueness check happens before `action.execute()`. Multiple matching hashes → `ambiguous` envelope.
- **`caller_agent_id` resolution.** `dispatch_action` resolves `caller_agent_id` from `session_id` if not provided (line 59-63 of action_registry.py).
- **`workspace_id`** is resolved in `execute_proxy_work` (lines 120-133) and passed to `dispatch_action` (line 142), not set on the return. The `execute_ticket_tool` flow does NOT add workspace_id (it comes from the action's `update_task_status` arg).

## Code Conventions

```python
# CORRECT — use the schema-driven normalizer for client args
from proxy.tool_parsing import build_valid_params_by_tool, normalize_tool_arguments, extract_file_path

# CORRECT — error envelope from action
try:
    return handler.execute(arguments, session_id, tg)
except AmbiguousHashError:
    return {"error": "ambiguous", "message": "Multiple tasks match the given hash. Please be more specific.", "session_id": session_id}
except InfrastructureError:
    return {"error": "infrastructure_error", "message": "Database temporarily unavailable", "session_id": session_id}

# CORRECT — use the shared helpers from actions/_errors.py
from proxy.tool_pipeline.actions._errors import log_and_emit_infra_error
return log_and_emit_infra_error(logger, "resolve_parent_hash", exc, session_id)
```

## External Dependencies

**Imports from:**
- `memory.query_executor` — `InfrastructureError`.
- `memory.task_graph` — `AmbiguousHashError`, `TaskGraph`.
- `memory.graph` — `Neo4jClient` (lazy).
- `proxy.tool_pipeline.action_registry` — `register_action`, `dispatch_action`.
- `proxy.tool_pipeline.actions.*` — 17 action classes.
- `proxy.context._get_task_graph_singleton` — singleton access.
- `proxy.agent.agent_registry` — `registry.get_or_load` for caller_agent_id.
- `proxy.workspace_state` — `workspace_store`, `DEFAULT_WORKSPACE`.
- `proxy.tool_parsing` — `extract_file_path`.

**Imported by:**
- `proxy.tool_pipeline.tool_pipeline` — `execute_memory_tool` after local execution.
- `src/skills/ticket_skill.py` — `proxy_ticket` action calls from planning mode.
- `src/skills/work_skill.py` — `proxy_work` action calls from work mode.
- Tests under `tests/integration/`, `tests/unit/`, `tests/proxy/`.

## Cross-References

- `src/memory/AGENTS.md` — Neo4j exception hierarchy + QueryExecutor + TaskGraph.
- `src/proxy/tool_pipeline/AGENTS.md` — ToolRegistry classifies tools into MEMORY vs PASSTHROUGH; `MemoryToolGuard` enforces mode gates.
- `src/proxy/tool_parsing/AGENTS.md` — `extract_file_path()` over `.get("file_path")` for write/edit arg paths.
- `src/skills/AGENTS.md` — `TicketSkill` / `WorkSkill` are the primary callers of `proxy_ticket` / `proxy_work`.

## Repository artifact policy

Reviewers MUST flag and request rework when a PR commits a non-code artifact without an explicit justification in that originating PR identifying the file and its required purpose. Passing tests or calling a file "evidence" does not waive this rule.

Generated evidence, logs, traces, screenshots, recordings, dumps, and reports belong in the originating PR or issue, NEVER in the repository. Store bulky artifacts in Google Drive and link them from that PR or issue. Do not commit those artifacts as documentation or fixtures to evade this policy.

Source code and tests belong in the repository. Any minimal fixture, documentation, or configuration genuinely required by the implementation must have an explicit file-and-purpose justification in the originating PR; retain only what is required. If that justification is absent, request rework rather than accepting the artifact.

This AGENTS.md policy update is explicitly requested by the repository owner.
