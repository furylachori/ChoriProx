# `src/proxy/tool_pipeline/AGENTS.md` — Tool Subsystem

> The biggest non-pipeline module (~29 .py files, 4800 LOC). Three coordinated
> layers: registry → pipeline → actions. Tool argument normalization lives
> separately in `tool_parsing/AGENTS.md` (THE LANDMINE).

## Module Purpose

Owns:
1. **ToolRegistry** — the canonical source of truth for tool names + categories + skill availability.
2. **ToolPipeline** — unified parse → classify → execute → format lifecycle.
3. **MemoryToolGuard** — 8-check pre-execution guard chain (self-ref, mode allowlist, work-mode actions, ticket-mode orchestrator, blocked tools, hash provenance, hierarchy, call limits; the grep sandbox runs inside call limits).
4. **ToolCallTracker** — per-session singleton for call counts + read-once + hash provenance.
5. **ToolLimits** — per-request hard caps (50/50/5 proxy_ticket/proxy_work/report_bug).
6. **Actions** — 17 TaskAction classes (10 `proxy_ticket` + 7 `proxy_work`) registered at import time.

## Files in this module

| File | Purpose |
|---|---|
| `tool_registry.py` | `ToolRegistry` (199 LOC) — 30 tools across 5 categories (the 20 base tools plus 10 decomposed `proxy_ticket_*` MEMORY tools, Slice 8a). `registry = ToolRegistry()` module-level singleton at line 154. |
| `tool_pipeline.py` | `ToolPipeline` (843 LOC) — `run()` entry + `_parse` + `_classify` + `_execute_via_loop` + `_format` + `_check_vfs_read_cache` + `_write_through_cache` + `ToolContext` dataclass |
| `tool_guard.py` | `MemoryToolGuard` (649 LOC) — 8-check guard chain (+ `_check_grep_sandbox` helper invoked inside `_check_call_limits`) |
| `tool_call_tracker.py` | `ToolCallTracker` (215 LOC) — per-session singleton (LRU evict at 3600s) |
| `tool_limits.py` | `ToolLimits` (481 LOC) — per-instance limits with `_overrides` |
| `action_registry.py` | `TaskAction` Protocol + `ACTION_REGISTRY` + `register_action` + `dispatch_action` with error envelopes |
| `action_envelope.py` | `success_envelope` + `partial_envelope` + `error_envelope` — the ONLY sanctioned spec §11 result shapes (no bare `{"error": ...}` dicts) |
| `error_taxonomy.py` | `classify_error(code)` — maps an error code to `(category, retryable)` for the canonical error envelope |
| `tool_call.py` | `ToolCall` frozen dataclass — `id`, `name`, `arguments`, `source`, `from_cache` |
| `actions/__init__.py` | Re-exports all 17 action classes |
| `actions/_errors.py` | `infra_error_envelope` + `log_and_emit_infra_error` shared helpers |
| `actions/*.py` | 17 action classes (one per file) — see "Actions" section below |

## ToolRegistry — canonical source

30 tools registered at module load (`tool_registry.py:157-199`):

| Category | Tools | Notes |
|---|---|---|
| **EDITOR** | `bash`, `read`, `glob`, `edit`, `write`, `grep`, `suggest` | `read`/`glob`/`grep` have `available_in_skills=frozenset({"ticket","work"})` |
| **WEB** | `webfetch`, `MiniMax_web_search` | `webfetch` available in ticket only |
| **MEMORY** | `recall`, `search`, `proxy_ticket`, `proxy_work` | `search` available in ticket only; `proxy_work` available in `{"ticket","work"}`; `proxy_ticket`/`proxy_work` are **admin tools** — excluded from `get_default_tools()` |
| **MEMORY (decomposed)** | `proxy_ticket_list_tasks`, `proxy_ticket_get_task`, `proxy_ticket_create_tasks`, `proxy_ticket_update_task`, `proxy_ticket_edit`, `proxy_ticket_set_parent`, `proxy_ticket_clear_parent`, `proxy_ticket_clear_blocked_by`, `proxy_ticket_set_priority`, `proxy_ticket_append_review_note` | Slice 8a (dormant until 8b); `available_in_skills=frozenset({"ticket","work"})`; also in `_PROXY_ADMIN_TOOLS` |
| **TASK** | `task` | |
| **PASSTHROUGH** | `todowrite`, `ask`, `question`, `new_task`, `TaskCreateTool`, `TaskUpdateTool` | `todowrite` available in ticket only; `question` has `available_in_skills=frozenset()` (forbidden in all skills) |

- **`get_default_tools()`** = 12 tools (excludes the 12 `_PROXY_ADMIN_TOOLS`: `proxy_ticket`, `proxy_work`, and the 10 decomposed `proxy_ticket_*` tools).
- **`get_passthrough_tools()`** = 6 tools.
- **`get_memory_tool_names()`** = `{recall, search, proxy_ticket, proxy_work}` plus the 10 decomposed `proxy_ticket_*` tools (14 total — every MEMORY-category tool).
- **`_PROXY_ADMIN_TOOLS`** (line 136) — 12 entries: `proxy_ticket`, `proxy_work`, and the 10 decomposed `proxy_ticket_*` tools. Bare sub-agents (no canary skill) MUST NOT see these in their LLM-facing tool list. The tool guard blocks them at execution time; excluding them from the default tool list prevents the LLM from even trying.

## ToolPipeline.run() flow

`ToolPipeline().run(response_data, ctx)` per call (line 95):

1. **`_parse(response_data, valid_params_by_tool=vpbt)`** — extracts native `tool_calls` from response; falls back to `parse_all_tools_from_content()` for DSML/YAML/XML/Markdown embedded formats. Calls `canonicalize_tool_name` + `normalize_tool_arguments` per call.
2. **`_classify(tool_calls)`** — splits into `memory_calls` (MEMORY category) vs `passthrough_calls` via `registry.get_memory_tool_names()`.
3. **VFS cache check** (two passes) — for each `read` tool call, checks VFS cache. On hit, sets `from_cache=True` and routes through `memory_calls` (cache means the read already happened in a prior request, so the read-once guard is bypassed).
4. **`question` unattended check** — if `unattended` or `is_subagent`, replace tool result with "no human partner" + `finish_reason="stop"`.
5. **Guard pass** — `self.guard.check(tc, ctx)` per memory call. Blocked calls go into `_blocked_errors` keyed by `(name, stable_args)` for soft-error path.
6. **Filter + write-through** — non-blocked calls go through `_write_through_cache` (VFS write-result scan). Blocked calls are filtered out before the write to prevent cache poisoning (M3-1).
7. **`_execute_via_loop(...)`** — delegates to `run_memory_tool_loop()` in `memory/tool_loop.py`. Builds `original_tool_calls_list` (un-normalized) for the loop's `make_tool_calls()` call.
8. **`_format(...)`** — assemble the final `response_data` with `tool_calls` and tool results.

The `try/except/finally` in `run()` is the catch-all boundary — no
exceptions escape to the caller. Self._cached_reads is reset in `finally`.

### `_stable_args_key` (line 29)

Strips proxy-injected keys (`session_id`, `client_hint`) before keying by
`(name, args)`. Guard-time and execution-time keys match.

### `TRUNCATION_REASONS = frozenset({"length", "content_filter"})` (line 26)

When `finish_reason` is set and not in this set, override to `"tool_calls"`.

## MemoryToolGuard.check() chain

Returns an error string if rejected, `None` if allowed. Order (line 55):

```python
1. _check_self_reference        — reject agent:* task:* ticket:*
2. _check_mode_allowlist        — skill.get_tool_filter() ∩ tool availability
3. _check_work_mode_actions     — MAIN orchestrator in work AND ticket mode: work mode allows list_tasks/get_task/append_review_note on proxy_ticket; ticket mode allows list_tasks/get_task only
4. _check_ticket_mode_orchestrator — MAIN orchestrator in ticket mode: unconditionally blocked from proxy_work; blocks both proxy_ticket/proxy_work for bare sub-agents (canary-cache defense)
5. _check_blocked_tools_for_subagent — sub-agent in skill: blocked from `task` + `question` (skill-configurable)
6. _check_hash_provenance       — MODIFY actions must reference hashes from prior list_tasks
7. _check_hierarchy             — delegates to proxy.skill_guard.check_sub_agent_proxy_access (CAT-3 trust gate)
8. _check_call_limits           — proxy_ticket/proxy_work counts, file read-once, sub-agent spawn, grep sandbox
```

### `_check_ticket_mode_orchestrator` (line 511)

MAIN in ticket mode is **unconditionally blocked from `proxy_work`** — under
the one-shot contract (PROXY_MODES_FIX_PLAN C3, locked decision #1), only
work-mode agents run work sessions, so MAIN in ticket mode has no legitimate
`proxy_work` call to make. The previous Bug #3 stop_work exemption was
deleted because the work→ticket TTL fallback machinery it served has zero
production callers; orphaned work sessions rely on the 24h stale sweep
(`work_sessions.py:cleanup_stale_sessions`). `proxy_ticket` for MAIN in
ticket mode is read-only, enforced by `_check_work_mode_actions` (runs
earlier in the guard chain); this guard blocks `proxy_ticket` only for bare
sub-agents (canary-cache defense).

### `_check_work_mode_actions` (line 452)

Restricts `proxy_ticket` actions for the MAIN orchestrator in work AND ticket
mode. In work mode, allows only `list_tasks`, `get_task`,
`append_review_note`. In ticket mode, allows only `list_tasks`, `get_task`
(write actions blocked — delegate via task sub-agents). Sub-agents bypass
this check; access is enforced by skill_guard per skill access level
(FULL for ticket subs, READ_ONLY for work subs per C4.5).

### `_check_call_limits` (line 208)

Per-session tracking via `ToolCallTracker.get_instance(session_id)`:

- **`proxy_ticket` count** — `call_limits.get("proxy_ticket")` or `MAX_PROXY_TICKET_CALLS=50`
- **`proxy_work` count** — `call_limits.get("proxy_work")` or `MAX_PROXY_WORK_CALLS=50`
- **`report_bug` sub-limit** — `MAX_REPORT_BUG_CALLS=5` (global hard cap)
- **`read` blocklist + read-once** — `extract_file_path()` + `tracker.try_read_file(file_path, from_cache=...)`
- **`task` sub-agent spawn** — `call_limits.get("task")` or `MAX_SUBAGENTS=3`

### `_check_grep_sandbox` (line 300)

Blocks searches over sensitive paths: `.env`, `.pem`, `.git`, anything
ending in `.key`/`.token`/`.password`/`credentials`/`credential`. Rejects
`..` traversal and any path that realpath-resolves outside `cwd`.

### `_check_hash_provenance` (line 353)

MODIFY actions on `proxy_ticket` (`edit`, `update_task`, `set_parent`,
`clear_parent`, `clear_blocked_by`, `set_priority`, `append_review_note`)
must reference `short_hash` / `parent_hash` / `blocked_by` entries that
were returned by a prior `list_tasks` call. (The ticket-side check is keyed
on the decomposed `proxy_ticket_*` tool names via `_TICKET_HASH_FIELDS`,
`tool_guard.py:44-53`.) WORK_HASH_ACTIONS (`start_work`,
`stop_work`, `handoff_task`, `report_bug`) require `short_hash` or
`task_short_hash` provenance.

`_hash_is_known` checks the tracker first, then `TaskGraph.get_task_by_short_hash`
as a fallback. Neo4j infrastructure failures fail-CLOSED
(`return False` — only tracker-known hashes accepted).

## ToolCallTracker (per-session singleton)

`tool_call_tracker.py:49-50` — `_instances: dict[str, "ToolCallTracker"]`
with `_instances_lock = threading.RLock()`. `get_instance(session_id)`
lazy-creates; `evict_stale(max_age_seconds=3600)` prunes untouched entries.

`reset_instance(session_id)` resets the tracker for a new user message from
the same session.

## ToolLimits — per-instance with overrides

`tool_limits.py:67-78` — `ToolLimits()` allocates its own `threading.Lock()`
+ `_overrides: dict[str, int]` and calls `reset()`.

```python
MAX_PROXY_TICKET_CALLS = 50
MAX_PROXY_WORK_CALLS = 50
MAX_FILE_READS = 30
MAX_SUBAGENTS = 3
MAX_REPORT_BUG_CALLS = 5

FILE_BLOCKLIST_PATTERNS = (
    ".env", ".pem", "credential", "secret", "token", ".git/", ".password",
)
```

`set_limit(field, n)` overrides the module-level constant for one field
(`"proxy_ticket"`, `"proxy_work"`, `"report_bug"`, `"file_reads"`,
`"subagents"`). Empty `_overrides` preserves the original behavior.

`try_read_file(file_path, from_cache=False)` — atomic check-and-record.
M3-2: when `from_cache=True`, the read-once check is BYPASSED (cache means
the read already happened in a prior request); the blocklist still
applies; the count is NOT incremented for cache hits.

## Actions (17 — 10 proxy_ticket + 7 proxy_work)

Each action is a small class implementing the `TaskAction` Protocol
(`execute(arguments, session_id, tg, workspace_id=None, caller_agent_id=None) -> dict`).
Registered at import time in `memory/tools/proxy_ticket.py:98-107` and
`memory/tools/proxy_work.py:24-30`.

### `proxy_ticket` actions (10)

| Action | File | Purpose |
|---|---|---|
| `create_tasks` | `actions/create_tasks.py` | Batch create tasks (UNWIND/MERGE) |
| `update_task` | `actions/update_task.py` | Status transition + multi-field update (assignment guard) |
| `list_tasks` | `actions/list_tasks.py` | `get_actionable_tasks` with workspace filter |
| `get_task` | `actions/get_task.py` | `get_task_enriched` (with ReviewNote) |
| `edit` | `actions/edit.py` | Editing fields (name, summary, tag, planning_info) |
| `set_parent` | `actions/set_parent.py` | `[:HAS_PARENT]` relationship (cycle detection) |
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

### `dispatch_action` error envelopes (action_registry.py:44)

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

Authorization is enforced by `execute_ticket_tool()` and `execute_proxy_work()`
in `memory/tools/` BEFORE `dispatch_action()` is called.

## ToolContext (dataclass, tool_pipeline.py:40)

Per-request context passed to `ToolPipeline.run()`:

```python
@dataclass
class ToolContext:
    session_id: str
    body: dict
    llama_body: dict
    auth_headers: dict
    original_model: str
    upstream_url: str
    model_cfg: Any = None
    execute_fn: Callable = None
    parse_tools_fn: Callable = None
    max_iterations: int = 5
    max_total_time: float = 90.0
    max_result_size: int = 16 * 1024
    max_calls_per_iteration: int = 50
    tool_parser: Any = None
    is_subagent: bool = False
    subagent_ticket_mode: bool = False
    unattended: bool = False
    tool_calls_list: Optional[List[dict]] = None
    client_hint: str = ""
    incoming_tool_names: Optional[set] = None
```

## Key Invariants

1. **ToolRegistry is the single source of truth** for tool names + categories + skill availability. No separate allowlists.
2. **`_PROXY_ADMIN_TOOLS` must NEVER appear in `get_default_tools()`.** Bare sub-agents (no canary skill) must not see `proxy_ticket`/`proxy_work`.
3. **All guard checks happen in order.** Call limits LAST — only count if all other checks pass.
4. **`extract_file_path()` over `.get("file_path")`** — `tool_guard.py:276`, `tool_pipeline.py:184`. See `src/proxy/tool_parsing/AGENTS.md`.
5. **`make_tool_calls(..., valid_params_by_tool=...)`** for every client-facing response (line 257).
6. **M3-2 cache-hit bypass** — `from_cache=True` on a tool call bypasses read-once but not the blocklist.
7. **Action registration at import time** — `register_action(...)` runs at module load in `memory/tools/proxy_ticket.py:98-107` and `proxy_work.py:24-30`. Reordering changes dispatch priority but doesn't change correctness.
8. **`ToolCallTracker` is per-session** — `get_instance(session_id)` is the only constructor.

## Landmines

- **Tool name casing in the LLM response.** LLMs emit `read_file`/`filePath`. `canonicalize_tool_name` + `normalize_tool_arguments(valid_params=...)` heal them. **Never hand-construct the client-facing `tool_calls` array** — always go through `make_tool_calls(...)`.
- **Patch-routing trap.** Patching `proxy.tool_pipeline.X` doesn't always intercept the in-leaf call. Always verify which module the function lives in.
- **`_check_work_mode_actions` is MAIN-only.** Sub-agents bypass it (line 472). Adding sub-agent restrictions to this guard is a logic error.
- **Soft-error path vs hard block.** Guard rejections go into `_blocked_errors` and are converted to `{error: ...}` tool results so the LLM sees the rejection. **Do NOT return early** — the LLM must see the rejection to recover.
- **Cache poisoning via guard-blocked writes** (M3-1). `non_blocked_calls` filter MUST run before `_write_through_cache` so blocked writes never hit VFS.
- **`WORK_HASH_ACTIONS` hash param name varies** by action: `short_hash` for `start_work`/`stop_work`/`handoff_task`, `task_short_hash` for `report_bug`. Map is in `tool_guard.py:27-32`.
- **`_check_ticket_mode_orchestrator` unconditional proxy_work block (post-C3).** MAIN in ticket mode is blocked from `proxy_work` with no exemption; orphan work sessions are recovered by `work_sessions.py:cleanup_stale_sessions`.
- **`_PROXY_ADMIN_TOOLS` exclusion.** Adding `proxy_*` to `get_default_tools()` lets bare sub-agents see them — guaranteed runtime rejection by `_check_hierarchy`.
- **Custom-skill `tool_whitelist` is REQUIRED.** Omitting it is a hard load reject (`tool_whitelist_missing` in `custom_skill_loader._check_tool_whitelist`, `custom_skill_loader.py:491`); an empty list `[]` means **deny-all** (zero tools, `frozenset()`); a non-empty list is the exact allowlist. The `None`/unrestricted sentinel is reserved for trusted Python builtins (TicketSkill/WorkSkill) and is never produced by the loader.

## Code Conventions

```python
# Always import via the leaf to avoid circular imports
from proxy.tool_parsing.normalize import (
    extract_file_path, normalize_tool_arguments, make_tool_calls, build_valid_params_by_tool,
)

# Action implementation — register the CLASS at import time (execute is a @staticmethod)
class MyAction:
    @staticmethod
    def execute(arguments, session_id, tg, workspace_id=None, caller_agent_id=None):
        ...

register_action("my_action", MyAction)

# Error envelope for infrastructure failures
from proxy.tool_pipeline.actions._errors import log_and_emit_infra_error
return log_and_emit_infra_error(logger, "proxy_ticket", "resolve_parent_hash", exc, session_id)
```

## External Dependencies

**Imports from:**
- `proxy.tool_parsing.normalize` — `extract_file_path`, `normalize_tool_arguments`, `make_tool_calls`, `build_valid_params_by_tool`, `canonicalize_tool_name`, `parse_all_tools_from_content`. **Always import from the leaf**, not `proxy.tool_parsing` (the package), to avoid circular imports.
- `proxy.agent.agent_registry` — `registry.get_or_load(session_id)`.
- `proxy.skills.skill_registry` — `get_default_registry()`, `SkillRegistry()`.
- `proxy.skill_guard` — `check_sub_agent_proxy_access` (CAT-3 trust gate).
- `proxy.workspace_state` — `resolve_inherited_workspace`.
- `proxy.chat_history` — `chat_histories`, `chat_histories_lock`.
- `proxy.state` — `_HTTP_SESSION`, counters.
- `memory.tool_loop` — `run_memory_tool_loop` (the LIVE re-query loop).
- `memory.query_executor` — `InfrastructureError`, `ConstraintViolation`, `QueryTimeoutError`.
- `memory.task_graph` — `AmbiguousHashError`, `TaskGraph`.
- `memory.tools.proxy_ticket` / `memory.tools.proxy_work` — dispatchers.

**Imported by:**
- `src/proxy/pipeline/tool_interception.py` — primary caller via `handle_tool_interception(...)`.
- `src/proxy/pipeline/post_process.py` — secondary.
- `src/proxy/pipeline/motivator_stage.py` — telemetry.
- Tests under `tests/proxy/`, `tests/unit/`.

## Cross-References

- `src/proxy/tool_parsing/AGENTS.md` — schema-driven argument normalization (THE LANDMINE).
- `src/proxy/AGENTS.md` — proxy shell + lock hierarchy.
- `src/proxy/pipeline/AGENTS.md` — `ToolPipeline` is invoked from `post_process.py` after the upstream response.
- `src/proxy/skills/AGENTS.md` — `SkillDefinition.allowed_tool_names` + `blocked_tools_for_subagent` drive the guard.
- `src/memory/tools/AGENTS.md` — the 4 memory tools + action dispatch (where actions are registered).
- `src/memory/tool_loop.py` (module docstring) — `run_memory_tool_loop` signature.