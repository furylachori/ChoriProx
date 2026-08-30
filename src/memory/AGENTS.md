# `src/memory/AGENTS.md` — Memory Layer (SQLite + Neo4j + SOTA)

> The second-largest source subtree (~58 .py files, 18916 LOC). Strict
> module layout: façade + pure leaves for SQLite; `QueryExecutor` singleton
> for Neo4j; `sync.py` for SOTA background extraction; `tool_loop.py` is the
> LIVE re-query path.

## Module Purpose

Memory subsystem owns:
- **SQLite CRUD** for messages, summaries, batches, sessions
- **Neo4j graph** for tasks, bugs, work sessions, workspaces, review notes
- **SOTA background extraction** via `sync.py` (1404 LOC)
- **Memory tool loop** — `run_memory_tool_loop` (1461 LOC, the LIVE re-query path)
- **11 Neo4j schema migrations** (001-011)
- **Connection pool** with WAL mode

## Files in this module

### DB module layout (façade + leaves)

| File | LOC | Purpose |
|---|---|---|
| `db.py` | 618 | **Public façade** — message/session CRUD + re-export surface for `db_pool`/`db_migrations` leaves. Installs `_DBModule(_types.ModuleType)` on `sys.modules[__name__].__class__` so `_pool` is a LIVE view of `db_pool._pool` (reassigned by `_ensure_pool`). |
| `_db.py` | 19 | Lazy-import cache `get_db()` — breaks circular import at module load |
| `db_pool.py` | 2747 | Connections use deferred SQLite transactions (`isolation_level=""`); checkin rolls back residual work. Configured capacity is 48. `_created` counts physical connections, `_creating` counts reserved opens, and `_inflight` counts admitted checkout calls. `pool_size` rejects bool, non-integer, zero, and negative values. Utilization is `len(_checked_out) / configured capacity`. Warnings fire above 0.80 and re-arm at or below 0.70. Every checkout returns a generation-bearing connection lease. Every cursor is also lease-bound; stale leases and cursors cannot touch a later borrower. Cursor registration happens under the operation lock so a concurrent checkin cannot miss it. Corruption is reported to a global coordinator via `_RecoveryRequired` (raised when an existing retiring source generation must be recovered) and via inline `_quarantine_operation_corruption` on leases/cursors (sets poisoning + retiring marker, never invokes recovery under the lease operation lock). The next global checkout performs recovery and retries through the replacement generation. A superseded source (one whose path is now bound to a different, newer generation) is retired but does NOT rewrite the file; the newer generation must independently observe corruption to repair. Recovery publication is strict identity compare-and-swap. Pool selection and checkout admission are atomic for the requested canonical path. Raw migration connections participate in recovery quiescence. Path maintenance (`begin_path_maintenance`/`end_path_maintenance`) uses a separate drain state, NOT corruption retirement: it rejects new checkouts but does not force-fail existing leases. The drain waits for `_inflight`, `_creating`, `_checked_out`, and `_checking_in` to reach zero with a bounded deadline, then closes idle conns and shuts the pool. The module registers one fixed fallback atexit callback. Terminal manager shutdown uses a single monotonic deadline; it waits for active recovery leaders, raw conns, and maintenance paths under that deadline and skips the WAL checkpoint if the deadline has passed. The module never imports `memory.db`; recovery dependencies remain lazy. |
| `db_migrations.py` | 1283 | `init_db` + `check_integrity` + `SCHEMA_SQL` + migration runner. **Pure leaf** — imports from `db_pool`, NEVER from `db`. |
| `db_recovery.py` | 881 | **C7+ SQLite corruption auto-heal.** `ensure_db_integrity(db_path)` (magic-byte sniff + `PRAGMA integrity_check`) + `recover_corrupt_db(db_path)` (dump/recover → heal → atomic swap, serialized via `proxy.state.recovery_lock` #13). **Pure leaf** — stdlib only (sqlite3, subprocess, os, shutil, re, datetime). |
| `db_summaries.py` | 566 | Summary persistence + cursor management (`get_last_summarized_exchange_index` / `set_last_summarized_exchange_index`) |
| `db_entities.py` | 249 | Entity/fact persistence (companion to Neo4j graph) |
| `persistence.py` | 427 | `_store_entities_and_facts` + `_store_memories` + `_get_graph` |
| `executors.py` | 432 | `_sync_pool` (4 workers) + `_query_pool` (2) + `_summary_pool` (4) + `_timeout_pool` (4) — bounded `ThreadPoolExecutor`s with `atexit` shutdown + watchdog thread; `_hung_futures` tracking dict + `_hung_futures_lock` record in-flight futures that exceed the timeout for hung-worker observability |
| `recall.py` | 70 | `recall()` memory tool + `filter_memories_for_context` + `format_memories_for_injection` |
| `_sota_http.py` | 64 | `SOTAMode` enum + `get_sota_client()` (lazy `OpenAI` singleton) |
| `_sota_rate_limiter.py` | 31 | |

### Neo4j layer

| File | LOC | Purpose |
|---|---|---|
| `query_executor.py` | 424 | **The single Neo4j driver** via `get_shared_driver` + `get_shared_executor` + `get_shared_executor_from_env`. Owns `InfrastructureError` / `ConstraintViolation` / `QueryTimeoutError` exception types. `execute_read` / `execute_write` / `execute_write_no_retry` / `execute_transaction` / `execute_transaction_no_retry`. |
| `graph.py` | 771 | `Neo4jClient` wrapper; memory + entity + topic operations. `init_schema` creates the `entity_name_type_unique` composite uniqueness constraint on `(e.name, e.entity_type)` (also created by migration 011) |
| `task_graph.py` | 903 | `TaskGraph(TaskGraphBatchMixin, TaskGraphQueryMixin)` — canonical ops: `create_task`, `create_tasks_batch_atomic`, `update_task_status` (with assignment guard), `update_task` (multi-field), `append_review_note`, `resolve_blocked_by_hashes`, `resolve_parent_hash` (cycle detection via `[:HAS_PARENT*1..20]`) |
| `task_graph_batch.py` | 486 | Atomic batch ops mixin — chunks of 100, UNWIND + MERGE pattern |
| `task_graph_queries.py` | 281 | Read mixin — `get_actionable_tasks`, `get_child_tasks`, `get_task_enriched` (with ReviewNote schema check), `get_recent_review_notes` |
| `task_graph_common.py` | 15 | `AmbiguousHashError`, `MAX_*`, `SHORT_HASH_RE` |
| `bug_tracker.py` | 611 | 4-state bug FSM (open → approved/dismissed → resolved); `create_bug`/`submit_review`/`validate_bug`/`resolve_bug` |
| `work_sessions.py` | 398 | `create_work_session` + `close_work_session` + `get_active_sessions` + `cleanup_stale_sessions` (24h stale) |
| `workspace_persistence.py` | 276 | `:Workspace` global catalog + `:Session` per-session + `[:LAST_USED_WORKSPACE]` edge; persists to Neo4j via callback wired into `proxy.state.workspace_store` |
| `circuit_breaker.py` | 183 | `CircuitBreaker(failure_threshold=3, recovery_timeout=30s)` — used as `state.neo4j_circuit_breaker`; isolated proxy continues when graph DB down |
| `security.py` | 73 | Cypher injection guard |
| `extraction_validator.py` | 201 | Validates SOTA extraction output schema |
| `utils.py` | 20 | `strip_args` |
| `prompts/` | (dir) | SOTA prompt templates |

### SOTA + tool loop

| File | LOC | Purpose |
|---|---|---|
| `sync.py` | 1404 | **THE SOTA background pipeline.** `MAX_SYNC_BACKLOG=100` (line 77) + `_cache_reasoning` (line 126, LRU/TTL 500/600s) + `submit_sync(session_id)` (line 1251) + `submit_ancient_compress` (line 1256) + `init_db` / `reconcile_sessions` |
| `sota_channel.py` | 431 | `SOTAChannel` + `SyncResult` + validation + extraction schemas |
| `sota_client.py` | 593 | `SOTAClient` — high-level OpenAI-compatible wrapper + `_call_compressor` + `extract_and_summarize` |
| `tool_loop.py` | 1461 | **`run_memory_tool_loop` — the LIVE re-query loop.** Token-trim + count-cap with sentinels preserved (H1+H2 invariants). `max_iterations=5`, `max_total_time=900.0`, `max_calls_per_iteration=50`, `max_messages=69`. |

### Memory tools

| File | LOC | Purpose |
|---|---|---|
| `tools/__init__.py` | 46 | Re-exports `execute_memory_tool`, `execute_ticket_tool`, `execute_proxy_work`, `recall`, `search`, schema definitions |
| `tools/dispatch.py` | 76 | `execute_memory_tool(name, arguments)` — top-level dispatcher |
| `tools/proxy_ticket.py` | 232 | `execute_ticket_tool` — action dispatcher + `_find_parent_workspace` (sub-agent inheritance) + registers 10 actions at import time |
| `tools/proxy_work.py` | 152 | `execute_proxy_work` — mode gate (work/ticket only) + registers 7 actions at import time |
| `tools/recall.py` | 6 | **`recall` STUB** — always returns `{"results": [], "total": 0, "truncated": False}` |
| `tools/search.py` | 6 | **`search` STUB** — always returns `{"results": [], "total": 0}` |
| `tools/schemas.py` | 347 | Tool definitions (`MEMORY_TOOL_DEFINITIONS`, `PROXY_TICKET_TOOL_DEFINITION`, `PROXY_WORK_TOOL_DEFINITION`) |

### Migrations

See `src/memory/migrations/AGENTS.md` — 14 files (__init__.py + base.py + runner.py + 001-011).

## DB Module Layout (the only sanctioned compat re-export)

**`memory.db` is the public façade.** It re-exports `connection` /
`init_db` / `SCHEMA_SQL` / `get_connection` / etc. from `db_pool` and
`db_migrations`. This is the **ONE sanctioned compat re-export** in the
codebase. `from memory.db import connection` stays valid for public callers
and tests that exercise façade-importing paths.

**Rules:**

- Submodules needing pool/transaction access (`db_summaries`, `batch_manager`,
  `sync`) import from `memory.db_pool` **directly**, not the façade
  (avoids the re-export cycle). CRUD access goes through `memory.db`.
- **`memory.db._pool` is a LIVE view of `db_pool._pool`** via a
  `_DBModule(_types.ModuleType)` class installed with
  `sys.modules[__name__].__class__` (line 84-85 of db.py). It is reassigned
  by `_ensure_pool`, so it must NOT be a real module attribute.
- **`isinstance(memory.db, types.ModuleType)` and `importlib.reload(memory.db)`
  both work** because the install is on the real module's `__class__`, not
  an object replacement.

**Façade patches do NOT intercept direct `memory.db_pool` callers.** Tests
for `db_summaries` / `batch_manager` / `sync` paths (which import pool /
transaction symbols directly from `memory.db_pool`) must patch
`memory.db_pool.*`, NOT `memory.db.*`.

## QueryExecutor — the single Neo4j driver

```python
# CORRECT
from memory.query_executor import (
    get_shared_executor, get_shared_executor_from_env,
    InfrastructureError, ConstraintViolation, QueryTimeoutError,
)

executor = get_shared_executor_from_env()
try:
    rows = executor.execute_read(query, params)
except InfrastructureError:
    raise  # let caller handle retry
except ConstraintViolation as e:
    return {"error": "duplicate", "message": str(e)}
except QueryTimeoutError:
    # ...
```

### Exception types (query_executor.py)

- `InfrastructureError(Exception)` — driver / network / service unavailable. Re-raise for caller retry.
- `ConstraintViolation(Exception)` — uniqueness constraint. Return 4xx-style response.
- `QueryTimeoutError(Exception)` — query exceeded timeout. Return 5xx-style response.

### Write methods

- `execute_read(query, params=None, timeout=30)` — idempotent read.
- `execute_write(query, params=None, timeout=30)` — idempotent write (auto-retried).
- `execute_write_no_retry(query, params=None, timeout=30)` — non-idempotent writes; pre-generate UUIDs.
- `execute_transaction(fn)` — transactional with retry.
- `execute_transaction_no_retry(fn)` — for relationship mutations (per AGENTS.md).

**Always pre-generate UUIDs before non-idempotent writes.** Never
`uuid.uuid4()` inside a transaction.

## Neo4j One-Driver Rule

There must be exactly ONE Neo4j driver. Creating additional drivers
(found in 2+ bugs) causes resource leaks. Use `get_shared_driver()` or
`get_shared_executor_from_env()`. The driver is closed by
`close_shared_driver()` at shutdown.

## SOTA Background Sync (`sync.py`)

`submit_sync(session_id)` (line 1251) is the public entry —
delegates to `_exec_submit_sync(sync_exchange, session_id, ...)`.

| Component | Where | What |
|---|---|---|
| `MAX_SYNC_BACKLOG` | sync.py:77 | 100 — caps processing to newest N when uncached |
| `_cache_reasoning` | sync.py:126 | LRU/TTL cache (max 500 entries, 600s TTL) |
| `_ancient_locks` / `_get_ancient_lock` / `_cleanup_ancient_locks` | sync.py:97 / 101 / 1394 | Per-session ancient-compress lock (background pool only — not in #0-#12 hierarchy) |
| `_ANCIENT_COMPRESS_MAX_PENDING` | sync.py:44 | 100 — bounded queue for ancient-compress (module constant, not env) |
| `reconcile_sessions` | sync.py:1108 | SQLite↔Neo4j reconciliation |
| `_sota_client` | sync.py:618 | Lazy singleton `SOTAClient` |

### Env vars

`PROXY_SOTA_API_KEY` / `PROXY_SOTA_ENDPOINT` (default
`https://api.stepfun.ai/step_plan/v1`) / `PROXY_SOTA_MODEL` (default
`step-3.7-flash`) / `PROXY_SOTA_TIMEOUT` (default 180s) /
`PROXY_SOTA_RETRY_AFTER_CAP` (default 60s, clamped to `[1, 300]` — caps the
server-supplied Retry-After delay in `sota_channel.py`).

### Channel retry

Fixed 2.5s delay (no exponential backoff) up to 5 attempts; circuit breaker
opens after 3 consecutive failures (`memory/circuit_breaker.py`).

### Background executors (`executors.py`)

| Pool | Workers | Purpose |
|---|---|---|
| `_sync_pool` | 4 | SOTA sync_exchange jobs |
| `_query_pool` | 2 | Memory queries |
| `_summary_pool` | 4 | Ancient-compress + summarization |
| `_timeout_pool` | 4 | Per-session lock + 600s timeout guard |

All registered for `atexit` shutdown + SIGTERM/SIGINT handlers.

Hung-worker observability: futures still running past the timeout are recorded
in the module-level `_hung_futures` dict (future-id → `(session_id, start_time)`),
guarded by `_hung_futures_lock`; the watchdog logs the accumulated count and
session ids, and entries are popped when their future completes.

## `tool_loop.py` — the LIVE tool re-query path

`run_memory_tool_loop(...)` (line 839) — defaults `max_iterations=5`,
`max_total_time=900.0` (Phase 2C, raised from 90s), `max_result_size=16*1024`,
`max_calls_per_iteration=50`.

**Production trim path:**
- `max_messages=69` (US-023), token cap `CONTEXT_TOKEN_LIMIT=26000`.
- `USABLE_CONTEXT_RATIO` from `proxy.state.CONTEXT_BUDGET_RATIO` (default 0.75).
- **H2 invariant** (line 116-118): `_sub_agent_return` sentinels ALWAYS kept — slight budget overrun accepted over lost sub-agent state.
- **H1 invariant** (line 148): sentinels appended LAST — survive hard cap.

**`MessageBuffer.trim_to_budget` / `enforce_hard_cap` are DEAD CODE** —
deleted in commit `7049942` after wave 4 caught the tautological-test bug
class. The 28 sentinel tests were pinned to these dead methods while
production kept using inline trim in `tool_loop.py:1220-1245`. Pin
`tests/unit/test_tool_loop_trim_sentinels.py` instead.

## Circuit Breaker (`circuit_breaker.py`)

`CircuitBreaker(failure_threshold=3, recovery_timeout_seconds=30.0, name="neo4j")`.

States: CLOSED → OPEN (after 3 failures) → HALF_OPEN (after 30s cooldown,
one probe) → CLOSED (probe success) or OPEN (probe failure).

Used as `proxy.state.neo4j_circuit_breaker`. Isolated proxy continues when
Neo4j is down.

## Task Graph Schema (graph-native, no JSON blobs)

| Relationship | Direction | Replaces |
|---|---|---|
| `[:BLOCKED_BY {created_at}]` | `(blocked)-[:BLOCKED_BY]->(blocker)` | `attributes.blocked_by` JSON array |
| `[:HAS_PARENT {created_at}]` | `(child)-[:HAS_PARENT]->(parent)` | `parent_task_uuid` property |
| `[:HAS_NOTE]` | `(task)-[:HAS_NOTE]->(review_note)` | `review_notes` property on Task nodes |

### Top-level Task properties

`uuid`, `short_hash`, `name`, `priority`, `status`, `summary`, `tag`,
`planning_info`, `created_at`, `deleted_at`, `status_changed_at`,
`workspace_id`, `session_id`.

### What does NOT exist

- NO `attributes` JSON blob on Task nodes
- NO `entity_type` property on Task nodes
- NO `parent_task_uuid` property on Task nodes
- NO `blocked_by` JSON array on Task nodes

**Migration 004 transformed all existing data.** Any code expecting the
old schema will fail.

## Key Invariants

1. **`memory.db` is the public façade.** `_pool` is a live view of `db_pool._pool`.
2. **Pure leaves (`db_pool`, `db_migrations`, `db_recovery`) never import `memory.db`.** They import `db_pool` from `db_pool` to avoid the re-export cycle.
3. **One Neo4j driver.** Use `QueryExecutor.get_shared_executor_from_env()`.
4. **Pre-generate UUIDs before non-idempotent writes.** Never inside a transaction.
5. **`MessageBuffer.trim_to_budget` is DEAD.** Use `_trim_working_messages_by_tokens` / `_enforce_working_messages_count_cap` in `tool_loop.py`.
6. **`recall` / `search` are STUBS.** Always return empty results.
7. **Action registration at import time.** `proxy_ticket.py:98-107` and `proxy_work.py:24-30` register via `register_action()`.
8. **`CONTEXT_BUDGET_RATIO` default 0.75.** Env `PROXY_CONTEXT_BUDGET_RATIO`, clamped to `[0.1, 0.95]`.
9. **All task mutations that modify relationships MUST use `execute_transaction_no_retry()` for atomicity.**
10. **Graph-native task schema.** No JSON blobs. Use `MATCH (t:Task)` directly.
11. **Entity `(name, entity_type)` uniqueness.** The `entity_name_type_unique` composite constraint is created by `graph.py` `init_schema` and backfilled/deduped by migration 011; duplicate `(name, entity_type)` Entity nodes are merged before the constraint lands.

## Landmines

- **Memory.db `_pool` reassignment.** `_pool` MUST NOT be a real attribute on `memory.db` — that would shadow `__getattr__` and go stale after `_ensure_pool` rebinds.
- **Façade patches don't intercept direct `db_pool` callers.** Tests for `db_summaries` / `batch_manager` / `sync` must patch `db_pool.*`, NOT `db.*`.
- **Two drivers = resource leak.** Use the shared driver.
- **SOTA circuit breaker open path** — sync jobs return without dispatching; proxy continues with truncated summaries (non-fatal).
- **SOTA channel retry is fixed 2.5s, not exponential.** Up to 5 attempts.
- **Concurrency tests need `threading.Barrier`.** Sequential tests on `_set_batch_processing` race never exercise the race window. See `tests/integration/test_sync_batch_toctou.py`.
- **`submit_sync` per-session lock** — concurrent sync for the same session_id returns None (debounce).
- **Graceful shutdown is async.** `atexit` runs `_graceful_db_shutdown` -> `_shutdown_thread_pool` (bounded wait) to drain queued tasks. `_signal_handler` only sets the flag and re-raises; calling `pool.shutdown()` from a signal handler is unsafe.
- **`_ANCIENT_COMPRESS_MAX_PENDING` is in-process only.** Restart drops the queue.

## Code Conventions

```python
# CORRECT — schema-driven read
from memory.query_executor import (
    get_shared_executor_from_env,
    InfrastructureError, ConstraintViolation, QueryTimeoutError,
)
executor = get_shared_executor_from_env()

# CORRECT — atomic batch write with pre-generated UUIDs
import uuid
batch = [{"uuid": f"task-{uuid.uuid4().hex}", "name": t["name"], ...} for t in tasks]
try:
    result = executor.execute_write_no_retry(batch_create_query, {"batch": batch})
except InfrastructureError:
    raise

# CORRECT — graph-native relationship traversal
rows = executor.execute_read(
    "MATCH (t:Task {short_hash: $hash})-[:BLOCKED_BY]->(blocker:Task) "
    "RETURN blocker.short_hash AS hash",
    {"hash": "abc12345"},
)

# WRONG — JSON blob parsing
attrs = json.loads(task.get("attributes", "{}"))
blocked_by = attrs.get("blocked_by", [])

# WRONG — label-checking pattern
"MATCH (e:Entity) WHERE 'Task' IN labels(e)"
# CORRECT: "MATCH (t:Task)"

# CORRECT — pool/transaction direct import (NOT through façade)
from memory.db_pool import transaction, get_connection

# CORRECT — error envelopes (action_registry style)
try:
    return handler.execute(arguments, session_id, tg)
except InfrastructureError:
    return {"error": "infrastructure_error", ...}
except QueryTimeoutError:
    return {"error": "query_timeout", ...}
except ConstraintViolation as e:
    return {"error": "constraint_violation", "message": str(e)}
```

## External Dependencies

**Imports from:**
- (Neo4j driver only) `neo4j` + `neo4j.exceptions` (ClientError, ServiceUnavailable, SessionExpired, TransientError) — try/except import in `query_executor.py`.
- `proxy.state` (counters + Neo4j circuit breaker).
- `proxy.tool_pipeline.action_registry` — `register_action`, `dispatch_action`.
- `proxy.tool_pipeline.actions.*` — 17 TaskAction classes.
- `proxy.agent.agent_registry` — `registry.get_or_load` for caller_agent_id resolution.
- `proxy.workspace_state` — `workspace_store`, `DEFAULT_WORKSPACE`, `resolve_inherited_workspace`.
- `proxy.context._get_task_graph_singleton` — singleton access.
- `proxy.tool_parsing.normalize` — `extract_file_path` for write/edit tracking.
- (tests) `tests/integration/neo4j_harness.py`, `tests/integration/neo4j_harness_utils.py`.

**Imported by:**
- `src/proxy/app.py` (via re-exports).
- `src/proxy/pipeline/*` — persistence operations.
- `src/proxy/tool_pipeline/*` — hash provenance.
- `src/skills/ticket_skill.py` / `work_skill.py` — task graph access.
- `src/devtools/debug_session.py` — for testing.
- Tests under `tests/integration/`, `tests/unit/`, `tests/proxy/`.

## Cross-References

- `src/memory/tools/AGENTS.md` — the 4 memory tools + action dispatch.
- `src/memory/migrations/AGENTS.md` — 11 Neo4j schema migrations + runner.
- `src/proxy/AGENTS.md` — `proxy.state.neo4j_circuit_breaker` is the wired singleton.
- `src/proxy/pipeline/AGENTS.md` — post_process runs `submit_sync` (SOTA, gated on `is_final_response and not subagent`).
- `src/proxy/tool_pipeline/AGENTS.md` — `MemoryToolGuard._hash_provenance` uses `TaskGraph.get_task_by_short_hash` as fallback.
- `tests/AGENTS.md` (planned) — `safe_db_name` invariants, `_clear_all_data` unconditional no-wipe refusal, `neo4j_test_db` / `real_or_mock_neo4j` fixtures.