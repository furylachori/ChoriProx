# `src/proxy/pipeline/AGENTS.md` — Request Pipeline (9 stages)

> Owns the request lifecycle. The 9 stages are ordered, share singleton
> instances, and never block. See `src/proxy/AGENTS.md` for the 15-position
> lock hierarchy and `src/proxy/tool_pipeline/AGENTS.md` for tool handling.

## Module Purpose

`RequestPipeline.run(ctx)` is the orchestrator. Each request flows through
9 stages, each implemented as a thin object with an `execute(ctx)` method.
Critical stages re-raise on exception; non-critical stages log + continue.

## Files in this module

| File | Purpose | Key lines |
|---|---|---|
| `request_pipeline.py` | `RequestPipeline` orchestrator + `CRITICAL_STAGES` + `_run_stages` loop + `request_scope` wrapping | full file (218 LOC) |
| `stages.py` | Public façade — re-exports 9 stages + `DEFAULT_STAGES` + `parse_request` + `prepare_upstream_request` | `DEFAULT_STAGES` = list of 9 stage instances at line 18 |
| `exceptions.py` | `StageAbort(response_data)` — early exit with response (not an error) | line 4 |
| `protocols.py` | `PipelineStage` Protocol with `execute(ctx)` | line 6 |
| `conditional_stage.py` | `ConditionalStage` base class — `should_run()` + `_execute()` with safety guard | line 7 |
| `dispatch.py` | `run_dispatch(ctx, features)` + `try_command` + `try_enhancer` (DEAD) | 548 LOC |
| `motivator_stage.py` | `MotivatorStage` — sub-agent context-usage reminders at 50/75/90% thresholds | 173 LOC |
| `post_process.py` | `PostProcessStage` — 8 ordered sub-methods (MiniMax reasoning extract, VFS write scan, chat history sync, SQLite persist, VFS cache populate, cache metrics, tool interception, model name assign) + `submit_sync` + 3 split post-process ops (`_apply_skill_response_transform`, `_inject_task_canaries`, `_finalize_skill_lifecycle`); the third op runs unconditionally in `execute()`'s `finally` block and sets `ctx.skill_lifecycle_finalized=True` for observability (the flag is no longer consumed by any abort-clear gate — the completion-thread + heartbeat model in `_keepalive_stream` makes PostProcessStage the sole authority on the keep/clear decision; per ticket 199ece93 §4 a downstream disconnect does NOT clear the mode); pre-method-list shape check + `StageAbort((body, 502))` enforces the malformed-completion boundary (see §"Malformed-completion boundaries" below) | 843 LOC |
| `provider_compat.py` | `apply_provider_compat` — DeepSeek/Kimi/MiniMax/MiMo patches; `resolve_upstream_route` | 366 LOC |
| `prompt_injector.py` | `PromptInjector.prepend_to_last_user_message` | 154 LOC |
| `agent_telemetry.py` | `AgentTelemetry` per-agent telemetry (lock #8) | 102 LOC |
| `threshold_engine.py` | `ThresholdEngine` — emits motivator triggers | 56 LOC |
| `tool_interception.py` | `handle_tool_interception(...)` — delegates to `ToolPipeline.run()` | 125 LOC |

### Stage modules (`stage_modules/`)

| Leaf file | Stage | LOC |
|---|---|---|
| `cleanup.py` | `CleanupStage` — increments `request_counter`, periodic cleanup every 100 requests | |
| `dispatch_stage.py` | `DispatchStage` — calls `run_dispatch(ctx, features)` | |
| `route_resolution.py` | `RouteResolutionStage` — `resolve_model_route` (sets `ctx.model_route`, `ctx.model_cfg`) | |
| `security.py` | `SecurityStage` — `detect_malicious_content` on user-role messages | |
| `context_optimize.py` | `ContextOptimizeStage` — `enforce_context_limits` → `resolve_mode` → `apply_mode` (V2); fail-closed active-skill errors map to 409/500 without DEFAULT fallback | |
| `provider_compat_stage.py` | `ProviderCompatStage` — `apply_proxy_recovery_system` (if tools) → `apply_provider_compat` → sets `ctx.llama_body` | |
| `request_parse.py` | `parse_request` — pre-pipeline helper (NOT a stage; called from `app.py:896`); raises `InvalidRequestError` for structured 400 envelopes | |
| `upstream.py` | `UpstreamStage` + `prepare_upstream_request` — model rewrite, MiniMax temperature clamp `[0.01, 1.0]`, auth header build → `call_upstream(..., wants_stream=bool(ctx.wants_stream))` (ticket 199ece93 Part A: client stream=true → upstream SSE reconstructed to a complete dict via `proxy.upstream_sse`; stream=false → JSON as before; requeries stay non-stream); the legacy `_defer_upstream` flag is gone (removed by ticket 199ece93 — the full pipeline runs in the completion thread) | |

(`MotivatorStage` and `PostProcessStage` are top-level in `pipeline/`, not in `stage_modules/`.)

## The 9 Stages (in `DEFAULT_STAGES` order)

```
1. CleanupStage         — request_counter, periodic session/workspace cleanup, stale ticket-mode cleanup
2. DispatchStage        — priority: enhancer (^^imp) → commands (^^help, ^^status) → skills (^^ticket, ^^work, ^^workspace)
3. RouteResolutionStage — resolve_model_route → ctx.model_route + ctx.model_cfg (CRITICAL)
4. SecurityStage        — detect_malicious_content → StageAbort(400) on threat (CRITICAL)
5. ContextOptimizeStage — enforce_context_limits → resolve_mode → apply_mode V2 (CRITICAL)
6. MotivatorStage       — sub-agent-only context-usage reminders (50/75/90%) via PromptInjector
7. ProviderCompatStage  — apply_proxy_recovery_system (if tools) → apply_provider_compat → ctx.llama_body
8. UpstreamStage        — prepare_upstream_request → call_upstream(..., wants_stream=bool(ctx.wants_stream)) (CRITICAL)
9. PostProcessStage     — 8 sub-methods → tool interception → submit_sync (SOTA) → 3 split skill ops (`apply_skill_response_transform` → `inject_task_canaries` → `finalize_skill_lifecycle` in `finally`)
```

### `CRITICAL_STAGES` (`request_pipeline.py:51-56`)

```python
CRITICAL_STAGES = {
    "RouteResolutionStage",
    "SecurityStage",
    "ContextOptimizeStage",
    "UpstreamStage",
}
```

Exceptions in these re-raise. Non-critical exceptions log + continue.

### `ContextOptimizeStage` active-skill fail-closed contract

Active-skill failures are not recoverable DEFAULT-mode fallbacks: a missing
registered skill raises/returns `skill_state_conflict` (409), and prompt/tool
policy callback failures raise/return `skill_runtime_error` (500). The active
mode is cleared via compare-and-clear when the runtime contract requires it,
and no upstream request is sent for these failures.

## Key Invariants

### `DEFAULT_STAGES` singleton-sharing + per-request reset

`stages.py:18-28` constructs 9 stage instances **once at module load**:

```python
DEFAULT_STAGES = [
    CleanupStage(),
    DispatchStage(),
    RouteResolutionStage(),
    SecurityStage(),
    ContextOptimizeStage(),
    MotivatorStage(),
    ProviderCompatStage(),
    UpstreamStage(),
    PostProcessStage(),
]
```

`RequestPipeline.__init__` does a **shallow `list(DEFAULT_STAGES)` copy** — the
list is per-pipeline but the 9 stage **objects are shared** across all
requests. Stages are effectively stateless except `DispatchStage._features`,
which `owner_setup.py:41-49` resets per request (owner path, immediately
before `pipeline.run`).

**Adding new mutable stage-instance state requires a per-request reset to avoid cross-request leaks.**

### Leaf-leaves-cycle rule

`stages.py` imports every stage leaf at module level. Any leaf importing
back from `proxy.pipeline.stages` creates a cycle. Leaves import only from
sibling leaves / source modules — never from the façade.
`motivator_stage.py` and `post_process.py` are top-level and already isolated.

All leaves use `logging.getLogger("proxy.pipeline.stages")` for log continuity.

### Patch-routing rule (HIGH — read before moving any stage helper)

`patch("proxy.pipeline.stages.prepare_upstream_request")` does **NOT**
intercept `UpstreamStage.execute()`'s in-leaf call — it resolves against the
LEAF. Tests patching a moved module-global MUST target
`proxy.pipeline.stage_modules.<leaf>.<symbol>`.

Always grep `patch("proxy.pipeline.stages.<symbol>")` before moving any
stage helper. The single-line grep misses multi-line `patch(...)` blocks;
use `grep -A` / AST / `git grep`.

### Test-injection corollary

Migrating a call site to a helper/leaf breaks tests that inject mocks at the
old call site to observe behavior — lock-scope, call-order, call-count. Same
class as the patch-routing trap, on the test-injection axis.

Before moving a helper, grep `monkeypatch.setattr`, `patch.object`, and
`@patch` decorators targeting the migrated symbol; retarget those tests to
the new call site to preserve their behavioral coverage.

### `StageAbort` semantics

`StageAbort` is **not an error**. It's the dispatch path saying "I've handled
this request — return my response and short-circuit the pipeline." Carries
`response_data`; raised by `parse_request` (401/403), `DispatchStage` (handler
emits a response), and any other stage that fully handles a request.

```python
class StageAbort(Exception):
    def __init__(self, response_data):
        self.response_data = response_data
```

`RequestPipeline._run_stages` always re-raises `StageAbort`.

### Swarm boundary — `PostProcessStage._inject_task_canaries` (post_process.py:494)

The swarm canary-injection step inside `PostProcessStage.execute()` is the
**swarm boundary**: it is where `ctx.swarm_policy` (set by
`ContextOptimizeStage` after `apply_mode` succeeds) is consulted at response
time. `_inject_task_canaries` calls `proxy.mode.inject_task_canaries(...)`
and treats `StageAbort` specially:

- `StageAbort` raised from `inject_task_canaries` (swarm rejection paths —
  `swarm_max_calls_exceeded`, `swarm_subagent_cannot_spawn`,
  `swarm_missing_secret`, `swarm_signing_failed`, `swarm_policy_unavailable`,
  `swarm_malformed_task_call`, `swarm_state_too_large` — see
  `SWARM_RUNTIME_STATUS` in `proxy/mode.py:24`) is **re-raised** so the Flask
  shell emits the canonical `custom_skill_swarm_error` envelope.
- Legacy unexpected exceptions are logged + swallowed per existing behaviour.

This is the only post-process op with a swarm-aware `StageAbort` re-raise;
`_apply_skill_response_transform` and `_finalize_skill_lifecycle` do not
propagate swarm aborts. Runtime never reparses JSON at this boundary; if
`SkillDefinition.subagents` is unexpectedly absent or structurally invalid
the response is failed-closed with `code="swarm_policy_unavailable"` (HTTP
500) per spec §10/§13.3.

### Malformed-completion boundaries

A malformed upstream HTTP 200 (`choices: null` / empty / non-object / wrong
type) must never cross the client boundary. The contract is enforced at
three layers, in order:

1. **HTTP client** (`src/proxy/http_client.py`) — `call_upstream()` parses
   all successful JSON and SSE responses through one validator
   (`proxy.openai_compat.chat_completion_shape_error`) and performs at most
   one safe semantic-empty retry. The retry is permitted only when reported
   `usage.total_tokens` is absent or integer zero; nonzero, negative,
   boolean, null, string, float, or otherwise untrustworthy token values
   skip the retry and return the canonical protocol error immediately.
2. **PostProcess** (`src/proxy/pipeline/post_process.py`) —
   `PostProcessStage.execute()` validates the response **before** invoking
   any success-side-effect method (reasoning extract, persistence, history
   sync, VFS mutation, cache metrics, tool interception, model assignment,
   SOTA sync). A malformed completion is replaced with
   `build_upstream_protocol_error()` and aborts with
   `StageAbort((build_upstream_protocol_error(), 502))`. An explicit
   dictionary-valued provider error preserves its original body and aborts
   through the existing `StageAbort((provider_error_body, 502))` path. The
   `finally` block still runs `_finalize_skill_lifecycle_with_flag(ctx,
   expected_mode)` so the completed malformed turn is classified as FAILED
   and the expected active mode is cleared.
3. **SSE** (`src/proxy/app.py`) — `_sse_from_response_data` is the final
   defense-in-depth boundary. It preserves the object-shaped provider
   error branch (`error` object + no `choices`), validates the rest through
   the shared `chat_completion_shape_error`, converts every malformed
   response to the canonical upstream protocol error, and emits exactly
   one error data event and one `data: [DONE]\n\n` terminator. There is no
   raw malformed-object SSE fallback.

`response_data = None` remains the existing **no-completed-turn / preflight
lifecycle sentinel** (e.g. command/enhancer/skills short-circuits): the
lifecycle `finally` block keys off it to decide whether the turn ever
reached a real completion. Actual upstream JSON `null` (and any other
non-dict payload) is rejected in the HTTP client before PostProcess — it
never reaches the `response_data` field as a literal `None` from upstream.

### `RequestPipeline.run` wraps in `request_scope`

`run()` opens `agent_registry.request_scope(session_id)` before iterating
stages. This prevents LRU eviction / TTL expiry mid-request (R3-H8 / R3-H9).
Even when session_id is empty (anonymous), the pipeline still runs.

### `parse_request` is pre-pipeline

`stage_modules/request_parse.py` is a **pre-pipeline helper** (not a stage).
Called from `app.py:896` BEFORE `RequestPipeline.run()`. Returns
`RequestContext`. Raises `StageAbort((dict, 401))` for invalid session
token and `(dict, 403)` for client_id mismatch. Structural body validation
raises `InvalidRequestError`; `app.py` renders it as HTTP 400 with
`{"error": {"message", "type": "invalid_request_error", "param", "code"}}`.
Malformed requests with `stream=true` still return ordinary JSON 400; no SSE
or upstream call starts.

### `apply_provider_compat` has no feature flag

`ProviderCompatStage.execute()` calls `apply_provider_compat` unconditionally
inside a try block — no fallback branch. Adding a feature flag here would
re-introduce the dead-fallback pattern. If you ever need to disable provider
compat temporarily, do it at the call site with a comment, not via a flag.

## Landmines

- **Singleton leak via new mutable instance state.** Only `DispatchStage._features` is mutable; everything else must be per-request reset.
- **Patch-routing trap.** Patches against `proxy.pipeline.stages.<symbol>` don't intercept in-leaf calls.
- **Test-injection trap.** Migrations to leaves break `monkeypatch.setattr` / `@patch` decorators at the old site.
- **Multi-line grep artifact.** Single-line `grep "from X import.*Y"` misses parenthesized imports + multi-line `patch(…)`. Use `grep -A` / AST.
- **Re-run every gate independently.** A reported-green gate was once red.
- **`DispatchStage._features` reset.** Forgetting to reset per request causes cross-request leak of `_features`.
- **`ConditionalStage` exception leak.** Subclass + override `_execute()`; the base class wraps in try/except + logs WARNING. NOT a silent pass — review the log.

### Test-debt (flag, don't fix)

`test_upstream_stage_test_mode_returns_mock` is tautological w.r.t. the
`prepare_upstream_request` patch; the other 3 sites are discriminating.

## Code Conventions

```python
# Logging in a stage
logger = logging.getLogger("proxy.pipeline.stages")
logger.info("stage completed", extra={"request_id": getattr(ctx, "request_id", "?")})

# Use copy.deepcopy(), never list(), when duplicating message lists
import copy
new_messages = copy.deepcopy(ctx.messages)

# Stage base class for non-critical work
from proxy.pipeline.conditional_stage import ConditionalStage
class MyStage(ConditionalStage):
    def should_run(self, ctx):
        return getattr(ctx, "some_flag", False)
    def _execute(self, ctx):
        ...
```

## External Dependencies

**Imports from:**
- `proxy.state` (counters, locks, `tool_cache_size`, `request_counter`, `*_rate_limiter`).
- `proxy.tool_parsing` — for `normalize_tool_arguments`, `make_tool_calls`, `extract_file_path`, `parse_all_tools_from_content`. Watch the circular import: `tool_parsing.__init__` → `prompts` → `tool_pipeline` → `tool_guard`. Always import from the leaf, e.g. `from proxy.tool_parsing.normalize import ...`, not the package.
- `proxy.tool_pipeline` — for `ToolPipeline.run()` (post-process delegates), `ToolRegistry`, `MemoryToolGuard`.
- `proxy.canary` — for canary detection in `parse_request` / `app.py`.
- `proxy.agent` — for `AgentRegistry`, `Agent`, `ModeState`.
- `proxy.skill_resolver` — `resolve_mode`.
- `proxy.mode` — `apply_mode`, `apply_skill_response_transform`, `inject_task_canaries`, `finalize_skill_mode`.
- `proxy.model_registry` — `resolve_model_route`, `_initialize_adapters`.
- `memory.db`, `memory.db_pool`, `memory.task_graph`, `memory.tools.*` — persistence + actions.
- `memory.tool_loop` — `run_memory_tool_loop` (the LIVE re-query loop, 691 LOC).
- `memory.sync` — `submit_sync`, `submit_ancient_compress`.

**Imported by:**
- `src/proxy/app.py` — constructs the pipeline in the chat-completions handler.
- `src/devtools/debug_session.py` — drives via Flask test client.
- Tests under `tests/proxy/`, `tests/integration/`, `tests/unit/`.

## Cross-References

- `src/proxy/AGENTS.md` — Flask shell, lock hierarchy, AppState.
- `src/proxy/tool_pipeline/AGENTS.md` — `ToolPipeline`, `MemoryToolGuard`, 17 actions.
- `src/proxy/tool_parsing/AGENTS.md` — schema-driven argument normalization (THE LANDMINE).
- `src/proxy/canary.py` (module docstring) — HMAC-signed canary parser.
- `src/memory/tool_loop.py` (module docstring) — the LIVE tool re-query loop.
- `src/skills/ticket_skill.py` + `work_skill.py` — `should_deactivate` / `get_inheritance_state` / `build_skill_canary` referenced from PostProcessStage's split ops (`_apply_skill_response_transform` / `_inject_task_canaries` / `_finalize_skill_lifecycle`).