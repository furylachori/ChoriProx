# `src/proxy/AGENTS.md` — Proxy Core

> The largest source subtree (~106 .py files). Holds the Flask shell, the
> 15-position lock hierarchy, the singleton-state rules, and ownership of the
> 9-stage request pipeline. Module-specific conventions for the pipeline
> itself live in `pipeline/AGENTS.md`; for tool handling in
> `tool_pipeline/AGENTS.md`; for tool argument normalization in
> `tool_parsing/AGENTS.md`; for canary in `canary.py`'s module docstring.

## Module Purpose

The proxy shell owns:
- Flask app + routes (`app.py`) and the admin blueprint (`admin.py`)
- The single mutable-state module `proxy.state` (locks, counters, rate limiters, singletons)
- The 9-stage request pipeline (`pipeline/`)
- The canary HMAC subsystem (`canary.py`)
- Skill infrastructure (`skills/`) + concrete skill classes (`skills/` in `src/skills/`)
- Tool subsystem (`tool_pipeline/`) + tool argument normalization (`tool_parsing/`)
- Agent + AgentRegistry (`agent/`)
- Session + workspace state (`session.py`, `chat_history.py`, `workspace_state.py`, `eviction_coordinator.py`)
- Event log + telemetry (`event_log.py`)
- VFS virtual filesystem layer (`vfs.py`)

## Files in this module

| File | Purpose |
|---|---|
| `app.py` | Flask routes — `_require_api_key`, `_enforce_fail_closed`, `parse_request` (pre-pipeline), `InvalidRequestError` structured 400 handling, admin blueprint, full-lifecycle SSE heartbeat (`_keepalive_stream`, completion-thread + `: connected`/`: keepalive`/`: [DONE]` per ticket 199ece93), lifecycle hooks; owns `_eager_init_registry()` (Wave 2) |
| `admin.py` | Admin blueprint — `/v1/models`, `/health` (with `custom_skills` block per Wave 2 spec §8.3), debug endpoints |
| `state.py` | Single source of truth for ALL shared mutable state (locks, counters, rate limiters, vfs_caches, canary cache) — see **Lock hierarchy** below |
| `lock_ordering.py` | `make_lock(position, name, lock_class)` factory with the 15-position hierarchy encoded |
| `types.py` | `RequestContext` dataclass, `BackendError`, `InvalidRequestError` (spec §7.1 structured 400 envelope), `SubagentIngress` |
| `session.py` | Session ID resolution + `cleanup_old_sessions` |
| `chat_history.py` | `ChatHistoryCache` LRU dict + read-through to SQLite (lock #2 owner) |
| `workspace_state.py` | `WorkspaceStateStore` — per-session active workspace + Neo4j persistence |
| `eviction_coordinator.py` | `build_and_process_deferred_clears` — releases chat_histories_lock then acquires AgentRegistry._lock (#0) |
| `event_log.py` | SQLite `logging.Handler` + `BoundedQueueHandler` + contextvar request/session binding; autocommit (`isolation_level=None`) |
| `mode.py` | `Mode` enum + `resolve_mode` + `apply_mode` (V2 entry) + `apply_skill_response_transform` + `inject_task_canaries` + `classify_turn_response` (spec §7.4 single-source classifier) + `TurnDisposition` + `finalize_skill_mode` (CAS compare-and-clear invoked from PostProcessStage.execute()'s `finally` per spec §7.5); raises `ActiveSkillUnavailable` / `SkillRuntimeError` (fail-closed exceptions per spec §7.2) |
| `skill_dispatcher.py` | `SkillDispatcher` + `DispatchResult` — priority dispatch for command/enhancer/task |
| `agent/agent.py` | `Agent` dataclass + public mode API + unified activation helper (`on_activate` before mode install) |
| `skill_lifecycle.py` | `register_lifecycle_callbacks(agent)` — wires deactivation handler |
| `skill_guard.py` | `check_sub_agent_proxy_access(tool_call, ctx, agent, registry)` — CAT-3 trust gate |
| `skill_resolver.py` | `resolve_mode` — sub-agent inheritance |
| `model_registry.py` | `_model_registry`, `_initialize_adapters()`, `resolve_model_route`, `resolve_session_id` |
| `openai_compat.py` | `build_openai_response(content, model, response_id, created)` — synthetic OpenAI response shape; `chat_completion_shape_error(data)` — single-source semantic validator; `build_upstream_protocol_error()` — canonical `upstream_protocol_error` envelope (code `invalid_chat_completion`) |
| `http_client.py` | `call_upstream(...)` — unified HTTP 200 path; semantically validates every parsed JSON/SSE completion; permits at most one malformed-response retry, only when reported `usage.total_tokens` is absent or integer zero. When `wants_stream=True` it invokes Requests with `stream=True` and consumes upstream SSE via `iter_lines()` into `OpenAIStreamAccumulator`, bounded by a wall-clock budget equal to the effective read timeout (raises non-retryable `UpstreamStreamError(code="stream_timeout")` on exceedance) |
| `upstream_sse.py` | `OpenAIStreamAccumulator` + `reconstruct_openai_sse` + `UpstreamStreamError`/`UpstreamStreamProtocolError`/`UpstreamStreamLimitError` — upstream OpenAI-compatible SSE reconstruction leaf (stdlib-only). WIRED and load-bearing (ticket 199ece93 Part A): reconstructs one complete `chat.completion` dict from incremental deltas before PostProcess |
| `streaming.py` | `make_sse_chunk`, `stream_response` |
| `requery.py` | `requery_llm(...)` — re-query loop after memory tool execution |
| `vfs.py` | Virtual filesystem layer — caches, write-through, rollback snapshots |
| `context.py` + `context_v2.py` | Context V2 chain-hash pipeline (V1 was deleted) |
| `utils.py` | Shared utilities — `_now_iso`, `safe_content`, `extract_text_content` |
| `requery.py` | Re-query after memory tool execution |
| `canary.py` | **HMAC-signed canary parser + verifier** — single-source after `a69078f`/`9f3829f`. See module docstring for the parser pipeline. |
| `session_tokens.py` | HMAC-signed client identity |
| `rate_limiter.py` | `RateLimiter(max_requests, window_seconds, persist=)` — sliding window |

### Pipeline + tool + skill + agent subdirectories

See the linked sub-AGENTS.md files: `pipeline/`, `tool_pipeline/`, `tool_parsing/`, `skills/`, `agent/` (including `agent.py`'s unified activation helper for ordered `on_activate` → mode install).

## Lock Hierarchy (15 positions, `state.py:294-315`)

```
#0  AgentRegistry._lock        (agent/agent_registry.py:46)  ← coarsest
#1  context_config_lock        (state.py:103)  RLock
#2  chat_histories_lock        (state.py via chat_history.py:13)  RLock
#3  chat_summaries_lock        (state.py:105)
#4  tool_cache_lock            (state.py:104)
#5  request_counter_lock       (state.py:102)
#6  active_workspace_lock      (state.py:512)  RLock
#7  Agent._mode_lock           (agent/agent.py:197)  PER-INSTANCE
#8  AgentTelemetry._lock       (pipeline/agent_telemetry.py)  PER-INSTANCE
#9  vfs_lock                   (state.py:108)
#10 _chain_state_lock          (state.py:109)
#11 canary_verify_cache_lock   (state.py:110)
#12 log_overflow_count_lock    (state.py:211)  leaf (never acquires other locks)
#13 recovery_lock              (state.py:220)  RLock — leaf
                                    memory.db_recovery corruption-recovery serialization
#14 dedup_lock                 (state.py:233)  leaf — DedupStore keyed completion map (CP4)
```

**Rule:** acquire in ascending order; never hold a lower-numbered lock while
acquiring a higher one. Violations caused 8+ adversarial-review-round deadlocks.
Position #7 and #8 are per-instance — do NOT add a `_MODE_LOCK` global to `state.py`.
Use `proxy.lock_ordering.make_lock` to construct new locks with their number.

## Transparent Request Deduplication (CP4)

The proxy transparently deduplicates retried requests (same body re-sent by
the client) without requiring any client change, header, or plugin.

### Fingerprint
`compute_fingerprint(ctx, mode_name)` in `src/proxy/dedup.py` computes an
HMAC-SHA256 fingerprint of the validated request, scoped by: session_id,
client_hint, model, workspace_name, is_subagent, active mode name, and the
full body (with `stream` stripped — stream mode is captured by `is_stream`).
Secret fallback: `PROXY_IDEMPOTENCY_SECRET` → `PROXY_SESSION_SECRET` →
process-local random. Truncated to 32 hex chars. Never logged in full
(only first 8 chars in debug logs).

### DedupStore
`src/proxy/dedup.py` — OrderedDict keyed by fingerprint, protected by
`state.dedup_lock` (position #14, leaf). LRU eviction via `move_to_end` +
`popitem(last=False)`. Caps (env-tunable):
- `PROXY_DEDUP_MAX_COMPLETED` (default 256): max COMPLETED entries.
- `PROXY_DEDUP_MAX_BYTES` (default 16MB): total stored bytes.
- `PROXY_DEDUP_MAX_FOLLOWERS` (default 16): max followers per PENDING key.
- `PROXY_DEDUP_COMPLETED_TTL` (default 120s): TTL for COMPLETED entries.
- `PROXY_DEDUP_PENDING_MAX_AGE` (default 600s): max age for PENDING entries.
- `PROXY_DEDUP_NONSTREAM_FOLLOWER_TIMEOUT` (default 60s): non-stream follower wait timeout. On timeout, 504 + Retry-After: 5.

`release_follower(fingerprint)` (dedup.py:326) releases a follower slot
previously allocated by `lookup_or_insert` — idempotent, only decrements
while the entry is still PENDING, and bumps the `releases` observability
counter exposed via `stats()`.

### AdmissionKind
`src/proxy/completion_coordinator.py` — enum: OWNER | FOLLOWER | REPLAY | CONFLICT.
- OWNER: new request, runs the full pipeline.
- FOLLOWER: same fingerprint as a PENDING owner, attaches to the same
  CompletionRecord, waits on the same completion_event, gets the same artifact.
- REPLAY: same fingerprint as a COMPLETED entry within TTL whose artifact
  passes the safety gate (logical_status=="ok", no tool_calls). Serves the
  cached artifact directly — zero upstream, zero pipeline, zero owner_setup.
- CONFLICT: same fingerprint as a PENDING entry with mismatched `is_stream`.
  Returns 409 `stream_conflict`.

### Safety gate
`is_safe_for_replay(entry)` in `src/proxy/dedup.py` — pure, total, fail-closed.
Returns True only if: COMPLETED + artifact not None + logical_status=="ok" +
has_tool_calls==False. Any uncertainty → False. Tool calls (tool_calls,
function_call, tool_calls/function_call finish_reason) in ANY choice are
detected by `_materialize_artifact` (app.py) and stored as
`CompletionArtifact.has_tool_calls`.

### CompletionArtifact
Dual-purpose frozen dataclass: `terminal_chunks` (SSE) for streaming, `body`
(JSON dict) for non-streaming. `has_tool_calls` gates replay. `logical_status`
("ok"/"stage_abort"/"backend_error"/"error") gates replay + mark_completed/mark_failed.

### Invariants
- Transparent: no client change, no new required header, no plugin.
- Owner-only mutations: `owner_setup` fires ONLY for OWNER. Never for
  FOLLOWER/REPLAY/CONFLICT.
- One worker, one pipeline: FOLLOWER/REPLAY never call `admit()` or
  `pipeline.run()`.
- Pending never evicted: TTL starts at COMPLETED. PENDING entries survive
  until the owner completes or fails (max_age cap is a crash guard).
- Exact replay = stored artifact: never re-render via `stream_response`.
- Safety gate: tool_call and error responses are NEVER replayed.
- Retry storm break: failed owner → mark_failed → evicted on next lookup_or_insert.
- Lock #14 is a leaf: never held while waiting on completion_event, running
  pipeline stages, or yielding SSE.
- Fingerprint never logged in full: only first 8 chars in debug logs.

### Observability events (Phase 5)
Canonical lifecycle events logged with request_id + session_id:
stream_opened, completion_owner_started, completion_follower_attached,
completion_replayed, completion_conflict, delivery_detached,
completion_finalized, delivery_finished, completion_capacity_rejected.

## `proxy.state` mutation pattern

```python
# CORRECT — module-level mutation via proxy.state reference
import proxy.state as state
state.tool_cache_size += 1

# WRONG — creates LOCAL variable, module-level unchanged
from proxy.state import tool_cache_size
tool_cache_size += 1
```

This applies to ALL integer counters and mutable collections:
`request_counter`, `tool_cache_size`, `tool_cache_hits`, `tool_cache_misses`,
`chat_summaries`, `param_modifications`, `param_drops`, `param_clamps`,
`param_injects`, `param_mod_details`, `vfs_cache_hits`, `vfs_cache_misses`,
`vfs_write_conflicts`, `log_overflow_count`.

`AppState` (state.py:671) holds ONLY immutable config (`llama_url`,
`cleanup_interval=100`, `session_ttl=28800`, etc.) — mutable state lives at
module level, NOT in `AppState`.

## Rate Limiters (`state.py:131-143`)

| Limiter | Limit | Window | Persist | Notes |
|---|---|---|---|---|
| `command_rate_limiter` | 500 | 60s | yes | |
| `enhancer_rate_limiter` | 500 | 60s | yes | ^^imp gate (DEAD path) |
| `escalation_rate_limiter` | 5 | 3600s | **no** | ticket→work sub-agent escalation (B-5 mitigation) |
| `skill_rate_limiter` | 1000 | 60s | yes | |
| `skill_ip_rate_limiter` | 3000 | 60s | yes | |
| `task_rate_limiter` | 2000 | 60s | yes | |
| `chat_rate_limiter` | 1000 | 60s | yes | `/v1/chat/completions` |
| `admin_rate_limiter` | 1000 | 60s | yes | `/v1/models`, `/health`, debug |

All auto-discovered + reset by `reset_all_rate_limiters()` (state.py:146).
**Interim limits** (commit `872964c` raised 50-100× from HANDOVER_2026_06_04
values) — do NOT tighten back to 60/min without first implementing
per-account rate limiting.

## Singletons (single-instance rule)

| Singleton | Location | Notes |
|---|---|---|
| `_HTTP_SESSION` | `state.py:667` | One `requests.Session()` shared; per-request auth via `headers=` |
| `_chain_state` | `state.py:470` | OrderedDict keyed on `(session_id, model_name)`, max 100 entries (env `PROXY_V2_CHAIN_STATE_MAX_ENTRIES`) |
| `invalidate_chain_state_for_session` | `state.py:499` | Drop all chain-state entries for a session (multi-model safe); acquired under `#10` lock |
| `canary_verify_cache` | `state.py:701` | OrderedDict, max 4096, TTL 3600s — keeps long-running sub-agents past HMAC freshness expiry |
| `Neo4j circuit breaker` | `state.py:657` | `CircuitBreaker(3, 30.0)` |
| `ToolRegistry` | `tool_pipeline/tool_registry.py:154` | Module-level singleton |
| `SkillRegistry` | `skills/skill_registry.py` | Double-checked locking singleton; constructed eagerly at CLI boot (Wave 2); discoverability surface: `custom_skills_status()` (skill_registry.py:329) + `list_skills()` (skill_registry.py:386); read-only under `_lock`, never mutates |
| `AgentRegistry` | `agent/agent_registry.py:29` | Singleton via `agent_registry.registry` |
| `workspace_store` | `state.py:523` | `WorkspaceStateStore` instance |

**Always acquire the singleton's lock through `proxy.state`/`proxy.X.lock` — never `Lock()` locally.**

## Custom-skill operator surface (Wave 5 / spec §8.1, §8.4)

Two operator-facing read-only paths expose the live registry without
mutating it. Both share the same `custom_skill_loader` parser — no
duplicated schema logic:

| Surface | Entry point | Used by | Detail doc |
|---|---|---|---|
| `^^help` Skills section | `SkillRegistry.list_skills()` (skill_registry.py:386) | `commands/__init__.py:cmd_help` | `src/proxy/skills/AGENTS.md` §"^^help registry discovery" |
| `^^status` Custom Skills block + `/health` `custom_skills` block | `SkillRegistry.custom_skills_status()` (skill_registry.py:329) | `commands/__init__.py:cmd_status`, `admin.py:/health` | `src/proxy/skills/AGENTS.md` §"Public status surface" + `src/commands/AGENTS.md` §"^^status" |
| Standalone validator CLI | `proxy.skills.validate_custom_skills` (`python -m proxy.skills.validate_custom_skills`) | CI, pre-deploy hooks, ops triage | `src/proxy/skills/AGENTS.md` §"Standalone CLI validator" |

The validator and the runtime loader return identical accept/reject
decisions — they share `custom_skill_loader.load_custom_skills` (and
`path_utils.resolve_skills_dir`) so config-resolution precedence is
identical (flag → env → default → disabled). See `src/proxy/skills/AGENTS.md`
§"Standalone CLI validator" for exit codes (`0`/`1`/`2`) and the JSON
output schema.

## Key Invariants

1. **Single mutable state owner.** `proxy.state` is the only module that defines shared locks/dicts/counters. Other modules import references.
2. **Lock ordering.** 15 positions (#0–#14); ascending order only.
3. **Agent mode lock is per-instance (#7).** Defined on the `Agent` instance in `agent.py:197`; NOT a global in `state.py`.
4. **`AppState` is immutable config.** All mutable state lives at module level.
5. **`_HTTP_SESSION` is the only HTTP session.** Per-request auth is via `headers=` param, not a fresh session.
6. **`parse_request` is pre-pipeline.** Called from `app.py` BEFORE `RequestPipeline.run()`; raises `StageAbort((dict, 401))` for invalid session token.
7. **VFS scope is env-driven.** `PROXY_VFS_SCOPE` = `"all"` (default) | `"subs"` | `"main"`. Invalid values log WARNING and fall back to `"all"`.
8. **Malformed-completion boundaries.** A malformed upstream HTTP 200 must never cross the client boundary.
   - `call_upstream()` semantically validates every parsed HTTP-200 completion through the single-source `chat_completion_shape_error` validator.
   - It permits at most one malformed-response retry, only when reported `usage.total_tokens` is absent or integer zero.
   - `PostProcessStage` is the fail-closed lifecycle and side-effect boundary — see `src/proxy/pipeline/AGENTS.md` §"Malformed-completion boundaries".
   - SSE formatting is defense in depth and never emits malformed completion data.
   - Non-stream failure is canonical HTTP 502; stream failure is one canonical error event plus one `[DONE]`.
   - Ordinary logs contain stable shape diagnostics only, never the malformed body.

## Swarm runtime (custom-skill research swarms spec)

Owns the swarm-capable custom-skill contract: `SubagentPolicy` (frozen
dataclass, `proxy/skills/skill_definition.py:249`), `SkillDefinition.subagents`
(`proxy/skills/skill_definition.py:775`), `RequestContext.swarm_policy`,
`proxy/mode.inject_task_canaries` (the swarm boundary, invoked from
`PostProcessStage._inject_task_canaries` at `pipeline/post_process.py:494`),
and `SWARM_RUNTIME_STATUS` (`proxy/mode.py:24`).

| `RequestContext.swarm_policy` field | Owner → Readers |
|---|---|
| `swarm_policy` | `ContextOptimizeStage` (write, after `apply_mode` succeeds via `_install_swarm_policy` at `pipeline/stage_modules/context_optimize.py:336`); reset to `None` at the start of every `ContextOptimizeStage.execute()` (`pipeline/stage_modules/context_optimize.py:32`); read by `PostProcessStage._inject_task_canaries` (`pipeline/post_process.py:494`). |

`SWARM_RUNTIME_STATUS` maps error codes to HTTP status:

| Code | HTTP |
|---|---|
| `swarm_max_calls_exceeded` | 400 |
| `swarm_subagent_cannot_spawn` | 403 |
| `swarm_missing_secret` | 500 |
| `swarm_signing_failed` | 500 |
| `swarm_policy_unavailable` | 500 |
| `swarm_malformed_task_call` | 502 |
| `swarm_state_too_large` | 500 |

The `StageAbort((swarm_error(code, message), status))` envelope is the
canonical wire form; `_inject_task_canaries` re-raises `StageAbort` so the
Flask shell emits it directly. See `src/proxy/pipeline/AGENTS.md` §"Swarm
boundary" and `src/proxy/skills/AGENTS.md` §"`SubagentPolicy` and
`SkillDefinition.subagents`" for the full invariant set.
See the internal documentation for the swarm-capable custom-skill contract.

## Landmines

- **Lock ordering** (above) — the #1 cause of deadlocks across 8+ adversarial rounds.
- **TOCTOU on `active_mode`.** Reading `agent.active_mode` without holding `_mode_lock` returns stale data. Always use `agent.snapshot_mode()`.
- **Shallow copies in pipeline stages.** Always `copy.deepcopy()`, never `list()`, when duplicating message lists.
- **Proxy state import pattern.** Named imports (`from proxy.state import counter`) create local shadows.
- **Neo4j driver multiplicity.** There must be exactly ONE Neo4j driver. Creating additional drivers (found in 2+ bugs) causes resource leaks. Use the shared driver from `QueryExecutor`.
- **SQLite parameter limit.** SQLite has a ~999 parameter limit. Large IN clauses must be chunked. Fixed in Round 26.
- **Error handling gaps.** Never use bare `except:` or `except Exception: pass`. Always handle `InfrastructureError`, `ConstraintViolation`, and `QueryTimeoutError` explicitly.
- **`NEO4J_AUTH` separator.** Use `:` not `/`. Mismatch between `app.py` and `sync.py` was a bug.
- **Canary degradation.** When `PROXY_SESSION_SECRET` is unset, canary signing returns unsigned and verification accepts any signature. Intentional fail-safe but should emit a startup warning.
- **Session token empty string.** When `PROXY_SESSION_SECRET` is unset, `create_session_token()` returns `""`. Callers must handle this.
- **Rate-limiter self-DoS.** Per-session / per-IP bucketing conflates all local sub-agent traffic into one bucket. Do NOT tighten limits.

## Code Conventions

### Logging
```python
logger.info("message", extra={"request_id": ctx.request_id})
```

### `proxy.state` re-import pattern for mutable state
```python
def get_config():
    from proxy.module import _config  # re-import to get current value
    return _config
```
Use when another module reassigns the symbol (e.g., `_pool` on `db_pool`).

### Pipeline stage pre-request reset
`app.py` resets `DispatchStage._features` per request. Adding new mutable
stage-instance state requires a per-request reset to avoid cross-request
leaks. See `pipeline/AGENTS.md`.

### Sub-agent delegation rule
Main agents delegate non-trivial edits to sub-agents. Read-only investigation
is fine on the main agent. See `src/AGENTS.md` for the full rule.

## External Dependencies

**Imports from:**
- `memory/` — for `db.py` CRUD, `db_pool` connections, `query_executor` for Neo4j, `task_graph` for tasks, `tools/` for memory tools, `sync` for SOTA.
- `clients/` — for `OpenCodeAdapter`, `KiloCodeAdapter`, `ClientRegistry`, `ModelRegistry`. Note: `clients.sota_client` is the ONLY legal path for non-`memory` code to reach SOTA.
- `skills/` — for `TicketSkill`, `WorkSkill`, `WorkspaceSkill` (BaseSkill subclasses).
- `commands/` — for `is_command`, `parse_command`, `dispatch_command`.
- `enhancer/` — for `try_enhancer` (DEAD — always returns None).
- `devtools/` — only in tests/debug harness.

**Imported by:**
- `src/qwen_proxy.py` (the canonical entry shim, re-exports curated public symbols).
- All `src/proxy/*` submodules.
- Tests under `tests/proxy/`, `tests/integration/`, `tests/unit/`.

## Cross-References

- `src/proxy/pipeline/AGENTS.md` — 9-stage pipeline lifecycle.
- `src/proxy/tool_pipeline/AGENTS.md` — ToolRegistry + ToolGuard + 17 actions.
- `src/proxy/tool_parsing/AGENTS.md` — schema-driven argument normalization (THE LANDMINE).
- `src/proxy/skills/AGENTS.md` — SkillRegistry + TrustToken + custom JSON loader.
- `src/proxy/agent/AGENTS.md` — Agent dataclass + AgentRegistry.
- `src/proxy/canary.py` (module docstring) — HMAC-signed canary parser + verifier.
- `src/memory/AGENTS.md` — SQLite + Neo4j + SOTA.
- `src/memory/migrations/AGENTS.md` — 11 Neo4j schema migrations.
- `src/clients/AGENTS.md` — ClientAdapter + adapters + registries.
- `src/skills/AGENTS.md` — concrete skill classes.
- `src/AGENTS.md` — cross-cutting conventions and the top-10 landmines table.

## Repository artifact policy

Reviewers MUST flag and request rework when a PR commits a non-code artifact without an explicit justification in that originating PR identifying the file and its required purpose. Passing tests or calling a file "evidence" does not waive this rule.

Generated evidence, logs, traces, screenshots, recordings, dumps, and reports belong in the originating PR or issue, NEVER in the repository. Store bulky artifacts in Google Drive and link them from that PR or issue. Do not commit those artifacts as documentation or fixtures to evade this policy.

Source code and tests belong in the repository. Any minimal fixture, documentation, or configuration genuinely required by the implementation must have an explicit file-and-purpose justification in the originating PR; retain only what is required. If that justification is absent, request rework rather than accepting the artifact.

This AGENTS.md policy update is explicitly requested by the repository owner.
