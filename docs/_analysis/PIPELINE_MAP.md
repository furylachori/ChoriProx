# ChoriProx Sub-Agent Request Pipeline Map

> **Status**: Complete
> **Generated for**: Sub-agent handling, HTTP entry → LLM delivery
> **Working dir**: `/path/to/choriprox`

This document maps the full request pipeline from Flask HTTP entry through the
9-stage pipeline to the LLM call, with special attention to the three
sub-agent-related fields on `RequestContext`:

- `is_subagent` (bool)
- `inherited_skill` (Optional[str])
- `subagent_ticket_mode` (bool)

Plus the related `workspace_name` (str).

---

## 1. Executive Summary

ChoriProx exposes an OpenAI-compatible `/v1/chat/completions` (and `/v2/...`)
endpoint in `src/proxy/app.py:573-685`. Every request is converted to a
`RequestContext` by `parse_request()` in `src/proxy/pipeline/stages.py:73-185`,
then run through a 9-stage `RequestPipeline` (`request_pipeline.py`).

**Sub-agent identification happens in-band**: the first user message must start
with a `[[__SUBAGENT__:...]]` canary marker. `detect_subagent_ingress()` in
`canary.py:271-367` scans `messages[0].content` for that prefix, strips it, and
populates a `SubagentIngress` dataclass (`types.py:19-34`) that is folded into
the `RequestContext`.

**Three modes matter for sub-agents** (defined in `mode.py:23-33`):

| Mode            | When                                              |
|-----------------|---------------------------------------------------|
| `DEFAULT`       | Main session, no skill                            |
| `SKILL`         | Main session, skill active (alias of TICKET)      |
| `SUBAGENT`      | Sub-agent, no inherited skill                     |
| `SKILL_SUBAGENT`| Sub-agent with inherited skill from parent canary |

The mode is *resolved* (`mode.py:70-191`) and *applied* (`mode.py:194-404`)
inside `ContextOptimizeStage`. The cascade fallback `_cascade_apply_default()`
(`stages.py:742-795`) runs when `get_or_load()` returns `None` — a frequent
case for brand-new sub-agent sessions.

**The canary is generated and re-injected** on the *response* side by
`apply_post_process()` (`mode.py:407-535`), which calls
`_inject_canary_into_task_calls()` (`canary.py:467-508`) to prepend the right
marker onto the `prompt` argument of any `task` tool call. That is how the
sub-agent identification is re-established for the next-hop LLM call.

---

## 2. Data Structures (`src/proxy/types.py`)

### 2.1 `BackendError` (lines 7-16)
Plain exception class, not a data structure. `app.py` catches it; `http_client`
raises it.

### 2.2 `SubagentIngress` (lines 19-34)
**Purpose**: Carries the result of canary detection.

| Field                    | Type   | Default | Meaning                                              |
|--------------------------|--------|---------|------------------------------------------------------|
| `is_subagent`            | bool   | `False` | Canary was found in messages[0].content              |
| `subagent_ticket_mode`   | bool   | `False` | True only when canary said "ticket" (or legacy TICKET_MODE) |
| `inherited_skill`        | Optional[str] | `None` | Skill name from canary (`"ticket"`, `"workspace"`, ...) |
| `workspace_name`         | str    | `""`    | Workspace name from `[[__SUBAGENT__:WORKSPACE:name]]` |

### 2.3 `RequestContext` (lines 37-101)
**Purpose**: Carries request state through the pipeline. Field ownership is
documented inline in the source (lines 41-71). Key sub-agent fields:

| Field                    | Type      | Default | Set by (writer)        | Read by (reader)                          |
|--------------------------|-----------|---------|------------------------|-------------------------------------------|
| `is_subagent`            | bool      | `False` | `parse_request` (line 172) via `canary.detect_subagent_ingress` | `mode.resolve_mode`, `mode.apply_mode`, `tool_cache.prepare_tooling_payload`, `tool_pipeline.ToolContext`, `vfs`, `post_process`, `motivator`, `dispatch.try_skill_activation` |
| `subagent_ticket_mode`   | bool      | `False` | `parse_request` (line 174) | `mode.resolve_mode` (line 161), `tool_guard` (line 377) |
| `inherited_skill`        | Optional[str] | `None` | `parse_request` (line 175) | `mode.resolve_mode` (line 85), `tool_guard` (line 378), `skill_registry` (line 115) |
| `workspace_name`         | str       | `""`    | `parse_request` (line 176) | `mode.resolve_mode` (line 145) |
| `unattended`             | bool      | `False` | `parse_request` (line 168-173): forced True if `is_subagent` | `tool_pipeline`, `tool_guard` |
| `_mode`                  | Optional[Any] | `None` | `mode.resolve_mode` (line 188) or `stages.ContextOptimizeStage` (line 723) | `post_process._skill_post_process` (line 346), `post_process.apply_post_process` |
| `_skill_activated`       | bool      | `False` | `agent._activate_skill` (line 57) | `mode.resolve_mode` (line 112) |
| `agent`                  | Any       | `None`  | `dispatch.DispatchStage` (not set in stages.py) | `post_process._skill_post_process`, `mode.apply_mode` |
| `model_route`            | Any       | `None`  | `RouteResolutionStage` (line 497) | `UpstreamStage`, `ProviderCompatStage` |
| `response_data`          | dict      | `{}`    | `UpstreamStage` (line 945) | `PostProcessStage` |
| `messages_original`      | Optional[list] | `None` | `ContextOptimizeStage` (lines 694, 724) | `MotivatorStage.should_run` (line 44) |

### 2.4 Other dataclasses
- `ModelRoute` (104-110): model_cfg, upstream_url, tool_parser, is_remote.
- `TokenEstimate` (113-117): estimated_tokens, limit, within_limit.
- `UpstreamRequest` (121-127): url, body, headers, original_model.
- `ToolingPayload` (130-136): tools, tool_hash, cached, augmented_tools.
- `RequeryResult` (139-149): requery_failed, requery_data, ...
- `TaskResult` (152-159): action, session_id, success, error, message, data.
- `MessageProvenance` (163-180): agent_id, message_type, stage, provider, ts.

### 2.5 `ModeState` (`src/proxy/agent/types.py`, lines 8-30)
**Purpose**: Tracks active skill mode for an Agent. Fields: `name`,
`activated_at`, `ttl_seconds`, `allowed_tool_names` (frozenset|None),
`skill_state` (dict). `is_expired()` returns False for `ttl_seconds=0` (workspace).

---

## 3. HTTP Entry Point (`src/proxy/app.py`)

### 3.1 Route registration
- Lines 573-575: `@app.route("/v1/chat/completions", methods=["POST"])` and
  `/v2/chat/completions`, both protected by `@_require_api_key`.
- Handler name: `proxy_chat_completions()` (line 576).

### 3.2 Non-streaming path (lines 576-685)

| Line | Action                                                                 |
|------|------------------------------------------------------------------------|
| 580  | `chat_rate_limiter.check(get_client_ip())` — reject with 429 if over    |
| 585  | Imports `agent_registry` (not directly used here)                     |
| 587  | **`ctx = parse_request(request)`** — first sub-agent detection happens |
| 588  | `ctx._features = _FEATURES` — feature flags for dispatch               |
| 589  | `ToolCallTracker.reset_instance(ctx.session_id)`                       |
| 590-604 | Logs request size, message count, model name                        |
| 606-614 | Logs `ctx.is_subagent` and incoming tool list (line 612-613)        |
| 616  | `pipeline = RequestPipeline()`                                        |
| 617-620 | Injects `_FEATURES` into the `DispatchStage` instance               |
| 623-644 | If `ctx.wants_stream`, calls `_keepalive_stream(ctx, pipeline)`     |
| 647-654 | Non-streaming: `final_response = pipeline.run(ctx)`                 |
| 656-676 | If `_PROXY_TRACE`, logs the full upstream request                   |
| 678-685 | Returns `jsonify(final_response)`                                   |
| 687-719 | Exception handling (ValueError, StageAbort, AuthConfigError, BackendError, RuntimeError, generic) |

### 3.3 Streaming path (`_keepalive_stream`, lines 466-...; completion-thread model)
- `_defer_upstream` was REMOVED (ticket 199ece93, commit `c44e9d4`): the full
  pipeline — including `UpstreamStage` — now runs **synchronously inside a
  bounded completion worker**. There is no deferred-upstream flag and no
  two-phase preflight + daemon-thread split; that pre-199ece93 architecture is
  fully retired.
- Architecture (per `_keepalive_stream` docstring at `app.py:466-502` +
  `proxy.completion_coordinator`):
  1. `app.py:801` — `completion_future = admit(pipeline, ctx)` submits the
     request to the bounded `completion_coordinator` executor pool. Admission
     is **synchronous**: `admit()` runs `Phase 1 DECIDE` (read `_pending` +
     `_shutdown` under `_lock`) before any mutation, then `Phase 2 MUTATE`
     (acquire semaphore + increment counter). A `CompletionCapacityError`
     surfaces as a 503 BEFORE any SSE headers are committed (F1 fix, R2-A #3).
  2. The pool worker (`_run_completion` at `completion_coordinator.py:79-113`)
     runs `pipeline.run(ctx)` — **the full 9-stage pipeline**, including
     `UpstreamStage`'s `call_upstream(..., streaming=False)` (line 100).
     Event-log ContextVars do NOT propagate to new threads; the worker
     installs its own via `set_request_context(...)` then clears in `finally`.
  3. `app.py:818` — `final_response = _keepalive_stream(ctx, completion_future)`
     yields the SSE artifact: `: connected\n\n` immediately, then
     `: keepalive\n\n` every `heartbeat_interval` seconds (per
     `_resolve_heartbeat_interval`, default 5.0s) until the future resolves.
     On completion the generator yields exactly one terminal `data:` chunk +
     one `data: [DONE]\n\n`. The single SSE formatter is the canonical
     `[DONE]` emitter.
  4. Disconnect semantics (`app.py:496-501`): `GeneratorExit` is logged and
     re-raised; the completion worker continues in the pool, the `Future` is
     left intact, and mode-clear is the responsibility of `PostProcessStage`
     (which has either already run or is still running). The previous
     disconnect-cancels-mode contract is gone.
- Non-streaming path: `app.py:843-845` — `final_response = pipeline.run(ctx)`
  directly. Same `request_scope` (`request_pipeline.py:86`) wraps the run;
  no daemon thread, no admission queue, no completion future.
- Result is then streamed as SSE through `_handle_tool_interception` /
  `PostProcessStage`.

### 3.4 Sub-agent detection at HTTP entry
- `parse_request(request)` (stages.py:73) is the only place where sub-agent
  state is **parsed from the HTTP body**. It:
  1. Reads `body` JSON (line 78).
  2. Computes `session_id` from `body` + `client_hint` (line 116).
  3. Calls `detect_subagent_ingress(body)` (line 151) which scans
     `messages[0].content` for the canary prefix.
  4. Reads `unattended` flag (line 166) and forces it True for sub-agents (line 167-168).
  5. Constructs `RequestContext` (lines 170-177).
  6. Sets `ctx.request_id = str(uuid.uuid4())[:12]` (line 178).

**No HTTP header** is used to mark sub-agent requests. Detection is purely
text-based in the first user message.

---

## 4. Pipeline Orchestrator (`src/proxy/pipeline/request_pipeline.py`)

### 4.1 `RequestPipeline` class
- Line 40-138. `stages` is a list of `PipelineStage` instances (Protocol, stages.py:47-65).
- `CRITICAL_STAGES` (lines 50-55): `{"RouteResolutionStage", "SecurityStage",
  "ContextOptimizeStage", "UpstreamStage"}` — exceptions in these abort the
  request. Note: `MotivatorStage` and `ProviderCompatStage` are NOT critical.

### 4.2 `run(ctx)` (lines 66-89)
```python
def run(self, ctx):
    session_id = getattr(ctx, "session_id", None)
    if session_id:
        with agent_registry.request_scope(session_id):
            return self._run_stages(ctx)
    else:
        return self._run_stages(ctx)
```
- Wraps in `request_scope` (R3-H8/R3-H9): prevents LRU eviction during the
  tool loop.
- Returns the final response, or raises.

### 4.3 `_run_stages(ctx)` (lines 91-137)
```python
for stage in self.stages:
    result = stage.execute(ctx)
    if result is not None:
        return result        # short-circuit
    try: ...
    except StageAbort: raise
    except Exception:
        if stage_name in CRITICAL_STAGES: raise
        else: continue        # degraded mode
return ctx.response_data
```

### 4.4 `create_pipeline()` (lines 140-145)
Default factory — builds a `RequestPipeline()` with `DEFAULT_STAGES`.

### 4.5 `is_critical_stage(stage)` (lines 148-150)
Helper for callers to check stage criticality.

---

## 5. Stages (`src/proxy/pipeline/stages.py`)

The default order (`stages.py:961-971`):
```
CleanupStage → DispatchStage → RouteResolutionStage → SecurityStage →
ContextOptimizeStage → MotivatorStage → ProviderCompatStage → UpstreamStage →
PostProcessStage
```

### 5.1 Pre-pipeline: `parse_request()` (lines 73-185)
**Input**: Flask `request` object.
**Output**: `RequestContext`.
**Side effects**: None on global state. Mutates `body["messages"][0].content`
via `canary.detect_subagent_ingress` (strips canary prefix in place).
**Sub-agent handling**:
- Line 149-155: `ingress = detect_subagent_ingress(body)` then unpack
  `is_subagent`, `subagent_ticket_mode`, `inherited_skill`, `workspace_name`.
- Line 167-168: `if is_subagent: unattended = True`.
- Line 170-177: Constructs `RequestContext` with all four fields.
- Line 178: Sets `ctx.request_id = uuid.uuid4()[:12]`.
**Failure modes**:
- Body not JSON: `ValueError` (line 79-80).
- Body not a dict: `ValueError` (line 81-82).
- Model not a string: `ValueError` (line 102-103).
- Invalid session token with `PROXY_SESSION_SECRET` set: `StageAbort((401,))`
  (line 133).
- Mismatched client_id: `StageAbort((403,))` (line 137).
- Invalid canary signature: silently returns SubagentIngress with all False/None
  (canary.py:202-205). The canary is just not detected.

### 5.2 Pre-pipeline: `handle_tool_interception()` (lines 234-347)
**Input**: response `data`, plus many kwargs including `is_subagent: bool = False`
(line 250).
**Output**: Modified `data` after running `ToolPipeline.run()`.
**Side effects**: Builds a `ToolContext` (line 324) and delegates to
`ToolPipeline`.
**Sub-agent handling**:
- Line 313: `is_subagent = is_subagent or body.get("is_subagent", False)` —
  the parameter can be OR'd from the body. This is the only direct read of
  `body["is_subagent"]` in the pipeline.
- Line 339: `is_subagent=is_subagent` is passed to `ToolContext`.
**Failure modes**: None fatal. If `body["tools"]` is malformed, the function
returns `data` unchanged.

### 5.3 Stage 1: `CleanupStage` (lines 372-418)
**Input**: `ctx`.
**Output**: Always `None` (does not short-circuit).
**Side effects**:
- `state.request_counter += 1` under `state.request_counter_lock` (line 386).
- Every 100 calls (configurable via `state.CLEANUP_INTERVAL`): calls
  `cleanup_old_sessions()`, `cleanup_expired_messages()`, `cleanup_stale_workspaces()`.
- Always calls `agent_registry.cleanup_stale_ticket_modes()` (line 414).
**Sub-agent handling**: **None**. This stage does not look at `is_subagent`,
`inherited_skill`, or `subagent_ticket_mode`.
**Failure modes**: All cleanups are wrapped in `try/except ImportError` and
`try/except` to log and continue.

### 5.4 Stage 2: `DispatchStage` (lines 426-476)
**Input**: `ctx` (with `ctx._features`).
**Output**: `None` (or raises `StageAbort` if a handler returned a response).
**Side effects**: Mutates `ctx.messages[-1]["content"]` and
`ctx.body["messages"][-1]["content"]` in place if a handler rewrote text
(lines 466-475).
**Sub-agent handling**:
- `app.py:617-620` injects `_FEATURES` into this stage's `_features` attribute.
- `dispatch.py:271-273`: `try_skill_activation` returns `None` early for
  sub-agents ("Sub-agents must not trigger skill activation").
**Failure modes**: `ImportError` is caught and dispatch is disabled (line 447).

### 5.5 Stage 3: `RouteResolutionStage` (lines 484-508)
**Input**: `ctx.model_name`.
**Output**: `None`.
**Side effects**:
- `ctx.model_route = route` (line 497).
- `ctx.model_cfg = route.model_cfg` (line 498).
**Sub-agent handling**: **None direct**. Routes the same way regardless of
`is_subagent`.
**Failure modes**: Wraps `resolve_model_route()` in `try/except Exception` →
raises `StageAbort((502, route_error))` (line 504). **CRITICAL stage**.

### 5.6 Stage 4: `SecurityStage` (lines 516-552)
**Input**: `ctx`.
**Output**: `None` (or raises `StageAbort` on threat).
**Side effects**: None.
**Sub-agent handling**: **None**. Scans the same `body["messages"]` for all
requests. Does not branch on `is_subagent`.
**Failure modes**: `ImportError` is caught and security is disabled (line 526).
Threat detected → `StageAbort(400, security_violation)`. **CRITICAL stage**.

### 5.7 Stage 5: `ContextOptimizeStage` (lines 560-727) — the centerpiece for sub-agents

**Input**: `ctx`.
**Output**: `None`.
**Side effects**:
- Mutates `ctx.body["messages"]` (via `apply_mode`).
- Sets `ctx.messages` (line 700, 726).
- Sets `ctx.messages_original` (lines 694, 724).
- Sets `ctx._mode` (via `resolve_mode` line 188, or line 723 `Mode.DEFAULT` in fallback).
- **Auto-registers a new Agent** for new sub-agent sessions (lines 622-674).

**Sub-agent handling — detailed flow**:

1. **Lines 572-577**: Diagnostic log of `is_subagent`, `inherited_skill`,
   `subagent_ticket_mode`.

2. **Lines 579-593**: Calls `enforce_context_limits(ctx.body, model_cfg)`.
   - On `ValueError` (over limit) → `StageAbort((413,))`.
   - Other exceptions → log and continue (degraded).

3. **Line 594**: `agent = agent_registry.get_or_load(ctx.session_id)`.
   - **Sub-agent fallback (lines 622-674)**: If `agent is None AND
     ctx.is_subagent == True`, auto-constructs a fresh `Agent(is_subagent=True)`
     and registers it. Trust assumption: `ctx.is_subagent` was set by the
     canary parser from parent-controlled metadata.
   - This is the **critical bug path** documented in the source comments
     (lines 612-621, 747-750): for a brand-new sub-agent with no SQLite
     history, `get_or_load` returns `None`, and the `if agent is not None`
     primary path (line 691) is skipped, falling into `_cascade_apply_default`
     (line 725).

4. **Lines 681-688**: `body_original = copy.deepcopy(ctx.body)` — snapshot
   before mutation.

5. **Primary path (lines 691-713)**: `if agent is not None`:
   - Line 693: `mode = resolve_mode(ctx, agent)`.
   - Line 694: `ctx.messages_original = copy.deepcopy(body_original.get("messages", []))`.
   - Line 699: `ctx.body = apply_mode(ctx, agent, ctx.body, mode)`.
   - Line 700: `ctx.messages = copy.deepcopy(ctx.body.get("messages", []))`.
   - On exception: calls `_raise_413_if_context_limit(e, ctx)` (line 709); if
     that doesn't fire, falls through to cascade.

6. **Cascade fallback path (lines 715-727)**: `if agent is None` (or primary
   raised):
   - Line 717-722: Diagnostic log.
   - Line 723: `ctx._mode = Mode.DEFAULT`.
   - Line 724: `ctx.messages_original = copy.deepcopy(body_original.get("messages", []))`.
   - Line 725: `_cascade_apply_default(ctx, body_original, agent_was_none=(agent is None))`.
   - Line 726: `ctx.messages = copy.deepcopy(ctx.body.get("messages", []))`.

**Failure modes**:
- `apply_mode` with `agent=None, mode != DEFAULT` raises `ValueError` (mode.py:225).
  `_cascade_apply_default` catches this and re-raises as `StageAbort((413,))` only
  for `agent_was_none=True` (stages.py:773-778). Otherwise logs and continues.
- `_cascade_apply_default` has a final fallback to minimal `[system, last]`
  messages (stages.py:781-795) and then to `[]` empty.
- `deepcopy` failure → `StageAbort((500,))` (lines 683-688).
- **CRITICAL stage** — failure aborts the request.

### 5.8 `_cascade_apply_default` (lines 742-795)
**Signature**: `def _cascade_apply_default(ctx, body_original, agent_was_none=False) -> None`.

**Purpose**: Fallback when `agent` is `None` or `resolve_mode`/`apply_mode`
raised. Always uses `Mode.DEFAULT` (line 764:
`ctx.body = apply_mode(ctx, None, body_original, Mode.DEFAULT)`).

**Failure cascade**:
1. `apply_mode(...)` raises `ContextLimitError` → `StageAbort((413,))`
   (lines 766-771).
2. `apply_mode(...)` raises `ValueError` AND `agent_was_none=True` →
   `StageAbort((413,))` (lines 773-778). P1-1 fix.
3. Any other exception: tries `minimal = [system_using_OUR_SYSTEM_PROMPT, last_msg]`
   (lines 781-789). On failure of that: `ctx.body = {"messages": []}` (line 793).

**Sub-agent bug path**: Diagnostic comment at lines 747-750 explicitly notes
that this is the "BUG PATH" — if `agent_was_none=True` and
`ctx.is_subagent=True`, the sub-agent gets `Mode.DEFAULT` instead of
`Mode.SKILL_SUBAGENT`. The auto-registration at lines 622-674 is the
mitigation.

### 5.9 Stage 6: `MotivatorStage` (imported from `motivator_stage.py`)
**Source**: `src/proxy/pipeline/motivator_stage.py`.

**Input**: `ctx`.
**Output**: `None`.
**Side effects**:
- Mutates `ctx.body["messages"]` and `ctx.messages` (lines 91-92, 119-121).
- Mutates `agent.telemetry` and `agent.threshold_engine`.

**Sub-agent handling**:
- `should_run()` (lines 40-45) gates on:
  - `MOTIVATOR_ENABLED` env var (default True).
  - `getattr(ctx, "is_subagent", False)` — sub-agents ONLY.
  - `ctx.messages_original is not None` — must have been set by
    `ContextOptimizeStage`.
- Line 50: `agent = registry.get_or_load(ctx.session_id)` — may return None
  for new sub-agent sessions, in which case stage exits silently (line 51-52).
- Lines 70-71: First-turn reset of `agent.threshold_engine`.
- Lines 88-92: Injects threshold reminder into system message.
- Lines 107-126: Turn-2 early reminder for tasks > 100 chars.

**Failure modes**: All wrapped; returns None on no-agent.

### 5.10 Stage 7: `ProviderCompatStage` (lines 810-858)
**Input**: `ctx`.
**Output**: `None`.
**Side effects**: `ctx.llama_body = llama_body` (line 856).
**Sub-agent handling**: **None direct**. Uses `ctx.body` regardless of
`is_subagent`.
**Failure modes**: Both `apply_provider_compat` and `_apply_provider_compat` are
wrapped in try/except.

### 5.11 Stage 8: `UpstreamStage` (lines 866-947)
**Input**: `ctx` (with `model_route`, `llama_body`).
**Output**: `None` (or sets `ctx.response_data`).
**Side effects**:
- `ctx.upstream = upstream` (line 886).
- `ctx.response_data = data` (lines 931, 945).

**Sub-agent handling**: **None direct**. Calls `prepare_upstream_request` and
`call_upstream`.

**Failure modes**:
- `ctx.model_route` not set → `RuntimeError` (line 876). **CRITICAL stage** → abort.
- `AuthConfigError` re-raised.
- Other → `StageAbort((502,))`.

### 5.12 Stage 9: `PostProcessStage` (imported from `post_process.py`)
**Source**: `src/proxy/pipeline/post_process.py` (661 lines).

**Input**: `ctx` (with `response_data`).
**Output**: `data` (modified `ctx.response_data`) or `None`.

**Step-by-step decomposition** (lines 47-56):
1. `_extract_minimax_reasoning` — strips <think> blocks.
2. `_scan_write_results` — VFS write snapshot confirmation (line 351+).
3. `_sync_chat_history` — sync messages to ChatHistoryCache.
4. `_persist_messages` — store to SQLite.
5. `_populate_vfs_cache` — cache file contents.
6. `_capture_cache_metrics` — metrics.
7. `_handle_tool_interception` — **the canary-re-injection point** via
   `apply_post_process` (see below).
8. `_assign_model_name` — set `data["model"]`.

After the methods (line 65-99):
- Lines 78-87: For `is_final_response` AND `not (ctx.is_subagent and
  ignore_subagents)` AND `ctx.session_id` AND `memory_enabled`: call
  `submit_sync(ctx.session_id, model_name=ctx.model_name)`. This is where
  sub-agents may be **excluded** from chat-history persistence.
- Line 97: `self._skill_post_process(ctx)`.

**Sub-agent handling**:
- Line 84: `ctx.is_subagent` gates memory sync.
- Line 321: `is_subagent=getattr(ctx, "is_subagent", False)` passed to
  `handle_tool_interception()`.
- Line 369, 463: `_vfs_active_for_context(getattr(ctx, 'is_subagent', False))` —
  VFS is scoped differently for sub-agents (`vfs.py:171-183`).

**`_skill_post_process` (lines 332-349)**:
```python
def _skill_post_process(self, ctx):
    try:
        from proxy.agent.agent_registry import registry as agent_registry
        from proxy.mode import apply_post_process
        agent = agent_registry.get_or_load(ctx.session_id)
        if agent is None:
            logger.warning(...)
        else:
            from proxy.mode import Mode
            mode = getattr(ctx, "_mode", Mode.DEFAULT)
            ctx.response_data = apply_post_process(ctx, agent, ctx.response_data, mode)
    except Exception as e:
        logger.warning("PostProcessStage: apply_post_process failed: %s", e)
```

**Failure modes**: `get_or_load` returning None is **logged but not fatal**:
`apply_post_process` is **not called** for the sub-agent. The canary is
therefore not re-injected. This is a critical failure mode for the
`task`-tool-spawn sub-agent chain.

---

## 6. Canary System (`src/proxy/canary.py`)

### 6.1 Canary marker formats (lines 87-91)
```python
_CANARY_SKILL_PREFIX = "[[__SUBAGENT__:SKILL:"
_CANARY_WORKSPACE_PREFIX = "[[__SUBAGENT__:WORKSPACE:"
_CANARY_TICKET_MODE_LEGACY = "[[__SUBAGENT__:TICKET_MODE]]"
_CANARY_BASE = "[[__SUBAGENT__]]"
```

### 6.2 HMAC signing (lines 36-85)
- `_sign_canary(skill_name)` (lines 36-47): returns `f"{sig}:{timestamp}"`
  where `sig` is HMAC-SHA256 truncated to 12 hex chars. Returns `""` if
  `PROXY_SESSION_SECRET` not set.
- `_verify_canary(skill_name, signature)` (lines 50-85): Validates the
  signature AND checks that `abs(now - timestamp) <= 300` (5-minute
  freshness window). **Dev-mode fail-OPEN** if `PROXY_SESSION_SECRET` is
  unset (lines 65-67).

### 6.3 `parse_canary(text)` (parser-collapse refactor)
**Input**: raw text whose start may carry a canary prefix.
**Output**: `Optional[CanaryParse]`. Returns `None` ONLY when no canary prefix
is present. A present-but-unverified canary returns `CanaryParse(verified=False)`
(graceful degradation — caller still marks `is_subagent=True`, grants no privilege).

Pipeline (single source of truth): `_locate_canary` (the sole position-based
walker, anchored on `_RE_SKILL_SIG_ANCHOR = :64hex:digits`) → `parse_canary` →
`_verify_canary` (HMAC + optional skill-registry check) → `_ingress_from_parse`
(the SINGLE field-mapping site → `SubagentIngress` fields).

`CanaryParse` fields: `kind` (`"bare"`|`"skill"`|`"workspace"`), `remaining`,
`verified`, `skill_name`, `workspace_name`, `state`.

- `[[__SUBAGENT__:SKILL:name:sig:ts]]` → `CanaryParse(kind="skill", verified=True, skill_name=name)`.
- `[[__SUBAGENT__:SKILL:name:sig:ts|{json}]]` → same, plus `state=json_state`.
- `[[__SUBAGENT__:WORKSPACE:name:sig:ts]]` → `CanaryParse(kind="workspace", verified=True, workspace_name=name)`.
- Bare `[[__SUBAGENT__]]` → `CanaryParse(kind="bare", verified=True)` (no inheritance).
- **Skill must be registered** (skill only, AFTER HMAC): unregistered skills
  yield `verified=False`.
- **Workspace name length cap**: `_MAX_WORKSPACE_NAME_LENGTH = 64`.
- HMAC-failed / structurally-malformed → `verified=False` (NOT `None`); the
  caller still marks `is_subagent=True` but grants no privilege.

> Note: the old unsigned formats (`[[__SUBAGENT__:SKILL:name]]` with no HMAC,
> and legacy `[[__SUBAGENT__:TICKET_MODE]]`) are no longer accepted — the walker
> requires the `:64hex:digits` anchor. The pre-refactor function was
> `parse_skill_canary`.

### 6.4 `detect_subagent_ingress(body)` (lines 271-367) — **THE detector**
**Input**: parsed request body dict.
**Output**: `SubagentIngress`.
**Side effects**: Mutates `body["messages"][i].content` in place to **strip
the canary prefix** (lines 308-310, 326-329, 338-340, 347-349).

**The exact assignments** (this is where `is_subagent`, `inherited_skill`,
`subagent_ticket_mode` are first set):

| Line | Code | Result |
|------|------|--------|
| 311  | `result.is_subagent = True` | SKILL branch with parseable skill_name |
| 312  | `result.inherited_skill = skill_name` | |
| 313  | `result.subagent_ticket_mode = (skill_name == "ticket")` | |
| 321  | `result.is_subagent = True` | SKILL branch with unparseable skill_name (fallback) |
| 329  | `result.is_subagent = True` | WORKSPACE branch with valid state |
| 330  | `result.inherited_skill = "workspace"` | |
| 341  | `result.is_subagent = True` | WORKSPACE branch with invalid state (fallback) |
| 344  | `result.is_subagent = True` | Bare `[[__SUBAGENT__]]` branch |

**`workspace_name` is only set** at line 331: `result.workspace_name = state.get("workspace_name", "")`.

**Failure modes**:
- No canary at all in `messages[0].content` → returns default
  `SubagentIngress(is_subagent=False, ...)`.
- `parse_canary` returns `CanaryParse(..., verified=False)` (e.g. unregistered skill,
  bad HMAC) → still sets `is_subagent = True` and strips the prefix
  (canary.py:315-321), but `inherited_skill` and `subagent_ticket_mode`
  remain at defaults. This is a security-aware degradation: the request is
  treated as a sub-agent but with no skill inheritance.

### 6.5 `_find_canary_close(text, start)` (lines 370-437)
Position-based parser for arbitrary canary states containing `]]` or `|`.
Used by `_strip_existing_canaries`.

### 6.6 `_strip_existing_canaries(text)` (lines 440-464)
Removes all canary markers from text to prevent stacking. Iterates with
`_find_canary_close`.

### 6.7 `_inject_canary_into_task_calls(data, canary)` (lines 467-508) — **THE injector**
**Input**: LLM response `data`, canary string.
**Output**: Modified `data` (same dict, mutated in place).
**Side effects**:
- For each `tool_call` in `data["choices"][0].message.tool_calls` with
  `function.name == "task"`:
  - Parses `arguments` as JSON.
  - Strips existing canaries from `prompt` via `_strip_existing_canaries`.
  - Prepends the new canary: `args["prompt"] = canary + prompt`.
  - Re-serializes `arguments`.

**Where the canary comes from**: Called from `apply_post_process` in
`mode.py:489-509`:
```python
canary = None
ms = agent.snapshot_mode()
if ms:
    skill = registry.get(ms.name)
    if skill:
        canary = skill.get_canary(skill_ctx)   # e.g. "[[__SUBAGENT__:SKILL:ticket|sig:ts]]"
if canary is None:
    canary = "[[__SUBAGENT__]]\n"             # bare fallback
data = _inject_canary_into_task_calls(data, canary)
```

**The canary is NOT injected** if `apply_post_process` is never called —
which happens in `_skill_post_process` (post_process.py:332-349) when
`get_or_load` returns None.

---

## 7. Mode System (`src/proxy/mode.py`)

### 7.1 `Mode` enum (lines 23-33)
```python
class Mode(enum.Enum):
    DEFAULT = enum.auto()           # Standard coding assistant
    SKILL = enum.auto()             # Main session, skill active (alias: TICKET)
    TICKET = SKILL                  # Alias for SKILL
    SUBAGENT = enum.auto()          # Sub-agent, no skill inheritance
    SKILL_SUBAGENT = enum.auto()    # Sub-agent, inherited skill mode
```

### 7.2 `resolve_mode(ctx, agent)` (lines 70-191) — sub-agent branching

**Input**: `ctx`, `agent`.
**Output**: `Mode` enum.
**Side effects**: Sets `ctx._mode = mode` (line 188). May set
`ctx._skill_activated = True` (via `_activate_skill`, agent.py:57).

**Sub-agent flow** (numbered):

1. **Lines 75-81**: Diagnostic log of entry state.

2. **Lines 85-86**: Read `inherited_skill` from ctx.
   ```python
   inherited_skill = getattr(ctx, 'inherited_skill', None)
   _has_inherited_skill = isinstance(inherited_skill, str) and bool(inherited_skill)
   ```

3. **Lines 92-104**: Atomic snapshot of agent's mode + ticket timestamp
   (`agent.snapshot_mode_and_ticket_ts()`). Maps to `is_ticket` and
   `is_workspace` locals.

4. **Lines 110-132**: Stale-clear: if agent has ticket/workspace mode but
   `not ctx.is_subagent` and no `_skill_activated`, calls
   `agent.clear_active_mode()` (CAS-guarded). Re-snapshots on failure.

5. **Lines 134-172**: Skill activation branch:
   - **If `_has_inherited_skill`**: Looks up `registry.get(inherited_skill)`,
     builds `SkillContext(workspace_name=...)` for workspace case, calls
     `_activate_skill(agent, skill, skill_ctx, ctx)`. Sets `is_ticket = True`.
   - **Elif `ctx.is_subagent and ctx.subagent_ticket_mode`**: Same
     activation path for the legacy "ticket canary" branch.

6. **Line 175**: `is_skill_active = is_ticket or is_workspace`.

7. **Lines 178-185**: Mode decision:
   ```python
   if getattr(ctx, 'is_subagent', False):
       mode = Mode.SKILL_SUBAGENT if is_skill_active else Mode.SUBAGENT
   else:
       mode = Mode.SKILL if is_skill_active else Mode.DEFAULT
   ```

8. **Line 188**: `ctx._mode = mode`.

**Failure modes**:
- `inherited_skill` not in registry → activation skipped (line 141 `if skill:`).
- Sub-agent with no inherited skill and no ticket_mode → `Mode.SUBAGENT`.
- Sub-agent with `subagent_ticket_mode=True` → `Mode.SKILL_SUBAGENT` (via the
  `elif` branch in step 5).

### 7.3 `apply_mode(ctx, agent, body, mode)` (lines 194-404)
**Input**: ctx, agent, body, mode.
**Output**: modified body dict.
**Side effects**: Mutates `body["messages"]`, `body["tools"]`, may set
`ctx._tools_snapshot`, `ctx.seeded_tasks`.

**Sub-agent branches**:
- Lines 210-215: Computed flags `_is_skill`, `_is_ticket`, `_is_subagent`,
  `_is_skill_subagent`, `_is_skill_or_ticket`,
  `_is_subagent_or_skill_subagent` from `mode.value` (lines 207-215 — **uses
  `mode.value` not identity**, H1 fix to survive `importlib.reload`).
- Line 224: **Guard**: `if agent is None and mode.value != Mode.DEFAULT.value:
  raise ValueError(...)`.
- Line 270: `tooling = prepare_tooling_payload(body, model_cfg,
  ticket_mode=is_ticket_mode, is_subagent=is_subagent,
  skill_name=_snapshot_skill_name or None, session_id=ctx.session_id,
  agent_id=agent.agent_id if agent else "")`.
- Lines 339-341: For `_is_skill_subagent` and `skill is not None`: prepends
  `build_skill_subagent_prompt(skill, skill_ctx)` to system prompt.
- Lines 375-378: For `_is_subagent_or_skill_subagent`: appends
  `YOUR TASK:\n{task}\n` (extracted from first substantive user message).
- Line 395: `is_subagent=getattr(ctx, 'is_subagent', False)` passed to
  `select_context()`.

**Failure modes**:
- `ValueError("apply_mode: agent is required for mode {mode}")` for
  non-DEFAULT mode with `agent is None` (line 225).
- `ContextLimitError` from `select_context` (line 399) is re-raised.
- Other exceptions in `select_context` are caught and logged (line 401-402).

### 7.4 `apply_post_process(ctx, agent, data, mode)` (lines 407-535)
**Input**: ctx, agent, response data, mode.
**Output**: modified data dict.
**Side effects**: May call `agent.clear_active_mode()` (lines 433, 530).

**Sub-agent-relevant steps**:
1. **Lines 415-434**: Defensive guard. If `data` is malformed (no choices):
   - Snapshots agent's mode.
   - Asks skill's `should_deactivate()`. If True → `agent.clear_active_mode()`.
2. **Lines 448-467**: For ticket-mode requests: snapshots mode, looks up
   skill, calls `skill.post_process(skill_ctx, data)`.
3. **Lines 469-509** (the canary injection — **CRITICAL for sub-agents**):
   ```python
   from proxy.canary import _inject_canary_into_task_calls
   canary = None
   ms = agent.snapshot_mode()
   if ms:
       skill = registry.get(ms.name)
       if skill:
           canary = skill.get_canary(skill_ctx)  # may return None
   if canary is None:
       canary = "[[__SUBAGENT__]]\n"
   data = _inject_canary_into_task_calls(data, canary)
   ```
   **This runs for ALL requests, not just sub-agents** (lines 469-509 — see
   the long comment block explaining the BUG fix that removed the
   `_pp_ticket_mode` gate).
4. **Lines 512-530**: If `finish_reason == "stop"`, calls
   `skill.should_deactivate()` then `agent.clear_active_mode()`.
5. **Lines 533-534**: If `finish_reason is None` (tool-call continuation),
   logs but keeps mode alive.

**When does `apply_post_process` run?**
- Only from `PostProcessStage._skill_post_process()` (post_process.py:347).
- Which only runs if `agent_registry.get_or_load(ctx.session_id)` returns a
  non-None agent (post_process.py:338-343).
- **For a brand-new sub-agent whose `get_or_load` returned None, the canary
  is NOT re-injected** — the system falls through silently.

---

## 8. Agent Registry (`src/proxy/agent/agent_registry.py` + `agent.py`)

### 8.1 `AgentRegistry` class (agent_registry.py, 29-603)
- LRU cache of `Agent` instances, backed by SQLite (chat_histories +
  `_load_from_sqlite`).
- `max_agents` default 100 (env: `AGENT_REGISTRY_MAX_AGENTS`).
- Lock hierarchy: `AgentRegistry._lock` (level 0) is the coarsest.
- `_in_flight_requests: dict[str, int]` (line 53) — protected by
  `request_scope()` (lines 64-82).

### 8.2 `get_or_load(session_id, caller_session_id=None)` (lines 92-197)
**Input**: session_id, optional caller_session_id.
**Output**: `Agent` or `None`.
**Fallback behavior** (in order):
1. **Authorization check** (lines 110-116): if `caller_session_id is not None`
   and != session_id → log warning, return `None`.
2. **Rate limit** (lines 119-136): Only for external callers (those passing
   `caller_session_id`). 100 unique session_ids per 60-second window.
3. **Fast path** (lines 138-147): Check `_agents` and `_agents_by_id` under
   lock; if hit, return.
4. **Slow path** (line 150): `_load_from_db(session_id)`.
5. **Unknown session** (line 154-155): Returns `None` (F3 SECURITY: prevents
   forged session IDs from creating phantom agents).
6. **Capacity check** (lines 167-176): If at max and all in-flight → return
   `None` (C-6 overflow block).
7. **Register** (line 178): `_register(agent)`.
8. **LRU eviction cleanup** (lines 188-195): `evicted_agent.clear_active_mode()`
   outside the lock (H-4 deadlock fix).

### 8.3 `_load_from_db(session_id)` (lines 199-266)
1. Check in-memory `chat_histories` cache.
2. If not cached, read through to SQLite via `_load_from_sqlite`.
3. If SQLite returns `None` → return `None` (forged session).
4. If SQLite returns `[]` (empty list) → session exists with no messages;
   valid for new sessions.
5. Construct `Agent(session_id, parent=None, config=AgentConfig())`.
6. `register_lifecycle_callbacks(agent)` (IoC deactivation).
7. Populate `_buffer` with loaded messages.
8. Return agent.

### 8.4 `register(agent)` (lines 285-335)
- Public entry point for registering a freshly constructed agent (e.g.
  the sub-agent auto-registration in `ContextOptimizeStage`).
- Acquires lock, double-checks for existing, evicts LRU if at capacity,
  calls `_register`.
- Does NOT wire up lifecycle callbacks — caller must do that.
- Returns `None` (after logging) on capacity overflow.

### 8.5 `request_scope(session_id)` (lines 64-82) — context manager
- Increments `_in_flight_requests[session_id]` on enter.
- Decrements on exit; pops when count reaches 0.
- Used by `RequestPipeline.run()` to prevent LRU eviction during tool loops.

### 8.6 `Agent` class (`agent.py`)
- Fields (lines 80-94): `agent_id`, `session_id`, `parent`, `config`,
  `is_subagent: bool = False` (line 84), `_metadata`, `_buffer`, `active_mode`,
  `_mode_lock`, `_ticket_mode_ts`, `telemetry`, `threshold_engine`.
- `__post_init__` (lines 96-105): derives `agent_id` from `session_id` if
  not provided; initializes `_mode_lock` with `make_lock(7, ...)`; creates
  `_deactivation_callbacks`.
- `set_active_mode` (lines 296-330): Atomic swap of `active_mode` under
  `_mode_lock`. Optionally CAS against `expected_mode`. Fires
  deactivation callbacks outside the lock.
- `clear_active_mode(expected_mode=None)` (lines 107-133): CAS clear.
  Returns True if CAS succeeded.
- `is_ticket_mode`, `is_skill_mode`, `is_workspace_mode` (lines 182-224):
  Read-only properties under `_mode_lock`.
- `snapshot_mode()`, `snapshot_mode_and_ticket_ts()` (lines 226-242):
  Atomic reads.
- `set_ticket_mode(bool)` (lines 332-351): Legacy wrapper.
- `messages` property (lines 155-179): Reads from `MessageBuffer`.
- `parent_agent_id` (lines 135-153): Returns `parent.agent_id` or legacy
  `_parent_agent_id`.

### 8.7 `Agent.is_subagent` — assignment
- Defined in `Agent` dataclass (agent.py:84) as `is_subagent: bool = False`.
- Set explicitly in **two** places in the codebase:
  1. `agent.py:84` — default.
  2. `stages.py:629` — `Agent(..., is_subagent=True)` in the
     `ContextOptimizeStage` auto-registration fallback.
- NOT set in the `spawn()` method (agent_registry.py:446-508) — that
  method does not pass `is_subagent` to the `Agent` constructor. This means
  agents created via `spawn()` (the sub-agent tree mechanism) have
  `is_subagent=False` unless they came from the auto-registration path.
- Read in many places (tool_guard.py, etc.) but never auto-derived from
  any other source.

### 8.8 `cleanup_stale_ticket_modes()` (agent_registry.py:409-440)
- Iterates all agents, skips those with in-flight requests, checks
  `agent.is_mode_expired()`, clears expired modes (with identity CAS).
- Called from `CleanupStage` (stages.py:414).

---

## 9. Tool Filtering (`src/proxy/tool_cache.py`)

### 9.1 `prepare_tooling_payload(body, model_cfg, ticket_mode, is_subagent, skill_name, session_id, agent_id)` (lines 67-304)
**Input**: body, model_cfg, ticket_mode bool, is_subagent bool, skill_name Optional[str], session_id, agent_id.
**Output**: `ToolingPayload(tools, tool_hash, cached, augmented_tools)`.

**Sub-agent handling**:
- Line 70-72: Diagnostic log of all parameters.
- Line 78-80: If `ticket_mode and not skill_name`: `skill_name = "ticket"`.
- Lines 84-177: **Skill mode path**: if `skill_name` is set:
  - Lines 91-99: `SkillContext(..., is_subagent=is_subagent)` passed to
    `skill.get_tool_filter(skill_ctx)`.
  - Lines 100-107: Filter `body["tools"]` by the returned tool set.
  - Lines 111-118: Append `search` tool from `MEMORY_TOOL_DEFINITIONS`.
  - Lines 120-132: Inject proxy-owned tools (e.g. `proxy_ticket` via
    `_get_injectable_memory_tools(skill_name)`).
  - Lines 138-157: Cache key includes `skill_name` to prevent cross-mode
    collisions.
- Lines 178+ (no skill): DEFAULT adapter path. Filters by
  `OPENCODE_TOOL_NAMES | PASSTHROUGH_TOOL_NAMES` (lines 239-244).
  `is_subagent` is **not used** in this branch.

### 9.2 `_generate_tool_hash_with_mode(tools, ticket_mode, skill_name)` (lines 307-328)
- Includes `skill_name` or `ticket` bit in the hash to prevent cross-mode
  cache pollution. SHA256, first 16 hex chars.

### 9.3 `cache_tool_schema`, `get_cached_tool_schema` (lines 330-359)
- Bounded LRU with `MAX_TOOL_CACHE_SIZE = 100`.

### 9.4 `_get_injectable_memory_tools(skill_name)` (lines 53-64)
- Returns `[PROXY_TICKET_TOOL_DEFINITION]` if `skill_name is not None`,
  else `[]`.
- Comment: "VFS is transparent (no LLM-visible tool). proxy_ticket is
  exposed only in SKILL mode where sub-agents need task management.
  DEFAULT mode gets no tool injection."

### 9.5 `PASSTHROUGH_TOOL_NAMES` (lines 19-50)
- Pulled from `ToolRegistry.get_passthrough_tools()` with hardcoded
  fallback: `{todowrite, ask, question, new_task, TaskCreateTool,
  TaskUpdateTool}`.

### 9.6 Sub-agent tool set summary
| Sub-agent state               | Tools received                                          |
|-------------------------------|---------------------------------------------------------|
| `Mode.SUBAGENT` (no skill)    | Filtered by `OPENCODE_TOOL_NAMES \| PASSTHROUGH` (no `proxy_ticket`) |
| `Mode.SKILL_SUBAGENT`         | Skill's `get_tool_filter()` ∩ `ToolRegistry` allowlist + `search` + `proxy_ticket` |
| Cascade fallback (agent=None) | Treated as DEFAULT → filtered adapter tools, no `proxy_ticket` |

---

## 10. Sub-Agent Flow Trace

**Scenario**: Parent agent in MODE.SKILL calls `task` tool → spawned sub-agent
request arrives at proxy with `prompt =
"[[__SUBAGENT__:SKILL:ticket|{json_state}]]\n<actual task>"`.

| # | Location | Action |
|---|----------|--------|
| 1 | `app.py:573-575` | Flask route `/v1/chat/completions` POST. |
| 2 | `app.py:576-583` | Rate limit check, return 429 if exceeded. |
| 3 | `app.py:587` | **`ctx = parse_request(request)`** (stages.py:73). |
| 4 | `stages.py:78-82` | Parse JSON body; raise ValueError on bad body. |
| 5 | `stages.py:101-107` | Extract and normalize model name. |
| 6 | `stages.py:115-116` | Resolve `session_id` from `body` + `client_hint`. |
| 7 | `stages.py:124-145` | Validate session token (if `PROXY_SESSION_SECRET` is set). |
| 8 | `stages.py:147` | `messages = body.get("messages", [])`. |
| 9 | `stages.py:149-155` | **`ingress = detect_subagent_ingress(body)`** → `(True, True, "ticket", "")`. |
| 10 | `canary.py:301-321` | `messages[0].content` starts with `[[__SUBAGENT__:SKILL:` → SKILL branch. |
| 11 | `canary.py:306` | `parse_canary(content)` → `CanaryParse(kind="skill", verified=True, skill_name="ticket", state=json_state)`. |
| 12 | `canary.py:202-205` | HMAC verify against `PROXY_SESSION_SECRET`. **In dev mode (no secret), skipped.** |
| 13 | `canary.py:209-215` | Verify `"ticket"` is registered. |
| 14 | `canary.py:311-313` | `result.is_subagent = True; result.inherited_skill = "ticket"; result.subagent_ticket_mode = True`. |
| 15 | `canary.py:308-310` | **Mutates** `messages[0].content` to strip the canary prefix. |
| 16 | `stages.py:166-168` | `unattended = body.get("unattended", False); if is_subagent: unattended = True`. |
| 17 | `stages.py:170-177` | Construct `RequestContext(body, model, ..., is_subagent=True, unattended=True, subagent_ticket_mode=True, inherited_skill="ticket", workspace_name="")`. |
| 18 | `stages.py:178` | `ctx.request_id = uuid.uuid4()[:12]`. |
| 19 | `app.py:588-614` | Set `ctx._features`, log request details (including `ctx.is_subagent`). |
| 20 | `app.py:616` | `pipeline = RequestPipeline()` (uses `DEFAULT_STAGES`). |
| 21 | `app.py:617-620` | Inject `_FEATURES` into `DispatchStage._features`. |
| 22 | `app.py:623` | `ctx.wants_stream` decides streaming path. |
| 23 | `app.py:649` (or `348` for streaming) | `final_response = pipeline.run(ctx)`. |
| 24 | `request_pipeline.py:83-87` | `with agent_registry.request_scope(session_id):` — prevent eviction. |
| 25 | `request_pipeline.py:92` | Loop over `self.stages` (= DEFAULT_STAGES). |
| 26 | **Stage 1: `CleanupStage.execute()`** (stages.py:381). Increments counter, maybe calls cleanup helpers. **is_subagent not read.** |
| 27 | **Stage 2: `DispatchStage.execute()`** (stages.py:438). Calls `run_dispatch(ctx, features)`. |
| 28 | `dispatch.py:271-273` | **`try_skill_activation` early-returns** because `ctx.is_subagent == True`. (Sub-agents can't activate new skills.) |
| 29 | `dispatch.py:81` | `try_enhancer` runs same as main agent (unless gated elsewhere). |
| 30 | Stage 3: `RouteResolutionStage.execute()` (stages.py:491). Sets `ctx.model_route` and `ctx.model_cfg`. |
| 31 | Stage 4: `SecurityStage.execute()` (stages.py:522). Scans `body["messages"]` — the canary-stripped content. **is_subagent not read.** |
| 32 | **Stage 5: `ContextOptimizeStage.execute()`** (stages.py:568). **The critical sub-agent stage.** |
| 33 | `stages.py:583` | `enforce_context_limits(ctx.body, model_cfg)`. |
| 34 | `stages.py:594` | `agent = agent_registry.get_or_load(ctx.session_id)`. |
| 35 | `agent_registry.py:138-147` | Fast-path check. If sub-agent was never registered, miss. |
| 36 | `agent_registry.py:150` | `_load_from_db(session_id)`. Reads from `chat_histories` → SQLite. |
| 37 | `agent_registry.py:243-244` | **If SQLite returns `None` → return `None`** (F3 security). |
| 38 | `stages.py:622` | **`if agent is None and getattr(ctx, "is_subagent", False):`** — auto-registration. |
| 39 | `stages.py:626-630` | Construct `Agent(session_id=ctx.session_id, agent_id=ctx.session_id[:12], is_subagent=True)`. |
| 40 | `stages.py:643` | `register_lifecycle_callbacks(new_agent)`. |
| 41 | `stages.py:653` | `agent_registry.register(new_agent)`. |
| 42 | `stages.py:665` | Re-resolve: `agent = agent_registry.get_or_load(ctx.session_id) or new_agent`. |
| 43 | `stages.py:682` | `body_original = copy.deepcopy(ctx.body)`. |
| 44 | `stages.py:691` | `if agent is not None:` (now True after auto-registration). |
| 45 | `stages.py:693` | **`mode = resolve_mode(ctx, agent)`** (mode.py:70). |
| 46 | `mode.py:85-86` | `_has_inherited_skill = True` (inherited_skill="ticket"). |
| 47 | `mode.py:92-104` | Snapshot agent mode (probably None for new sub-agent). |
| 48 | `mode.py:110-132` | Stale-clear check; for new sub-agent, no mode to clear. |
| 49 | `mode.py:138-160` | **`_has_inherited_skill` branch**: `skill = registry.get("ticket")` → call `_activate_skill(agent, skill, skill_ctx, ctx)`. Sets `is_ticket = True`. `ctx._skill_activated = True`. |
| 50 | `mode.py:178-185` | `mode = Mode.SKILL_SUBAGENT` (subagent + skill_active). |
| 51 | `mode.py:188` | `ctx._mode = Mode.SKILL_SUBAGENT`. |
| 52 | `stages.py:694` | `ctx.messages_original = copy.deepcopy(body_original.get("messages", []))`. |
| 53 | `stages.py:699` | **`ctx.body = apply_mode(ctx, agent, ctx.body, mode)`** (mode.py:194). |
| 54 | `mode.py:210-215` | `_is_skill_subagent = True`, `_is_subagent = True`, `_is_skill = True`. |
| 55 | `mode.py:256-260` | `_seed_task_context(ctx)` for SKILL/SKILL_SUBAGENT. |
| 56 | `mode.py:270` | `tooling = prepare_tooling_payload(body, model_cfg, ticket_mode=True, is_subagent=True, skill_name="ticket", session_id=ctx.session_id, agent_id=agent.agent_id)`. |
| 57 | `tool_cache.py:84-177` | Skill mode path. Filters tools by `skill.get_tool_filter(skill_ctx)`. Appends `search` + `proxy_ticket`. |
| 58 | `mode.py:285-306` | Apply skill tool filter (dual-auth: runtime + static). |
| 59 | `mode.py:320-386` | Build system prompt: stable prefix + detect_tool_mode + working_dir + mandatory block + skill_subagent prompt + task context + task anchor. |
| 60 | `mode.py:390-396` | `select_context(...)` for window pruning. |
| 61 | `stages.py:700` | `ctx.messages = copy.deepcopy(ctx.body.get("messages", []))`. |
| 62 | **Stage 6: `MotivatorStage.execute()`** (motivator_stage.py:40). |
| 63 | `motivator_stage.py:40-45` | `should_run()` returns True (subagent, has messages_original). |
| 64 | `motivator_stage.py:50` | `agent = registry.get_or_load(ctx.session_id)` — may return None for new sub-agent. |
| 65 | `motivator_stage.py:67-72` | Capture task, increment turn, reset thresholds on first turn. |
| 66 | `motivator_stage.py:73-100` | If turn% > 50: inject "[MOTIVATOR]" reminder into system message. |
| 67 | **Stage 7: `ProviderCompatStage.execute()`** (stages.py:818). `apply_provider_compat(llama_body, model_cfg)`. |
| 68 | **Stage 8: `UpstreamStage.execute()`** (stages.py:872). |
| 69 | `stages.py:884-886` | `upstream = prepare_upstream_request(...)`. `ctx.upstream = upstream`. |
| 70 | `stages.py:944` | `data = call_upstream(upstream.url, upstream.body, upstream.headers, False)`. |
| 71 | `stages.py:945` | `ctx.response_data = data`. |
| 72 | **Stage 9: `PostProcessStage.execute()`** (post_process.py:38). |
| 73 | `post_process.py:48-56` | Runs `_extract_minimax_reasoning`, `_scan_write_results`, `_sync_chat_history`, `_persist_messages`, `_populate_vfs_cache`, `_capture_cache_metrics`, `_handle_tool_interception`, `_assign_model_name`. |
| 74 | `post_process.py:84` | If `is_final_response` AND `not (ctx.is_subagent and ignore_subagents)`: `submit_sync(ctx.session_id, model_name=ctx.model_name)`. |
| 75 | `post_process.py:97` | `self._skill_post_process(ctx)`. |
| 76 | `post_process.py:332-349` | `agent = agent_registry.get_or_load(ctx.session_id)` — now finds the auto-registered agent. |
| 77 | `post_process.py:347` | **`ctx.response_data = apply_post_process(ctx, agent, ctx.response_data, mode)`** (mode.py:407). |
| 78 | `mode.py:415-434` | Defensive guard: if data malformed, optionally clear active mode. |
| 79 | `mode.py:448-467` | For SKILL_SUBAGENT: call `skill.post_process(skill_ctx, data)`. |
| 80 | `mode.py:469-509` | **`canary = skill.get_canary(skill_ctx)`** → e.g. `[[__SUBAGENT__:SKILL:ticket:sig:ts]]`. **`data = _inject_canary_into_task_calls(data, canary)`**. |
| 81 | `canary.py:467-508` | For each `task` tool call: strip existing canaries, prepend new canary. |
| 82 | `mode.py:512-530` | If `finish_reason == "stop"` and skill `should_deactivate`: clear active mode. |
| 83 | `app.py:649` | `final_response = pipeline.run(ctx)` returns `ctx.response_data`. |
| 84 | `app.py:680-685` | `return jsonify(final_response)`. |

---

## 11. Where Things Break — Failure Modes for Sub-Agent State

### 11.1 `is_subagent` could be `False` when it should be `True`

| Location | Failure mode |
|----------|--------------|
| `canary.py:202-205` | If HMAC verification fails AND `PROXY_SESSION_SECRET` is set: `parse_canary` returns `verified=False` → `detect_subagent_ingress` still sets `is_subagent = True` (lines 315-321) but `inherited_skill` and `subagent_ticket_mode` remain default. Sub-agent state is partially lost. |
| `canary.py:209-215` | If skill name is NOT registered: same partial-loss as above. |
| `stages.py:79` | `RequestContext.is_subagent: bool = False` default. If a future change bypasses `parse_request`, this default sticks. |
| `stages.py:313` | `is_subagent = is_subagent or body.get("is_subagent", False)` — clients could **forge** `body.is_subagent` to gain sub-agent privileges, but only via the response path (`handle_tool_interception`). This is a downstream concern. |
| `dispatch.py:271-273` | `try_skill_activation` early-returns for sub-agents. If `is_subagent` is incorrectly False here, the sub-agent could spuriously activate a skill. |
| `post_process.py:84` | `ignore_subagents` flag on model_cfg suppresses memory sync. If sub-agent state is lost, this gate doesn't apply. |
| `tool_pipeline.py:131-143` | `if name == "question" and (unattended or is_subagent)`: blocks the question tool. If `is_subagent` is False, sub-agents could call `question` and deadlock waiting for human input. |
| `tool_pipeline.py:146, 173, 493, 563` | VFS cache reads/writes use `is_subagent` for scoping. If lost, VFS pollution possible. |
| `tool_guard.py:376-451` | Multiple guard checks: `is_subagent` and `subagent_ticket_mode` together decide whether memory tools are allowed. If either is wrong, security violation. |
| `vfs.py:171-183` | `_vfs_active_for_context` decides VFS scope. Wrong value → cross-contamination. |
| `motivator_stage.py:42-45` | `should_run` requires `is_subagent`. If False, sub-agents miss turn-aware reminders. |
| `stages.py:622-674` | Auto-registration of new sub-agent agents depends on `is_subagent`. If False, no Agent is created and cascade fallback runs with `Mode.DEFAULT` instead of `Mode.SKILL_SUBAGENT`. |
| `mode.py:110, 161, 178` | Three mode-decision branches all read `is_subagent`. Wrong value → wrong mode. |
| `mode.py:212, 215` | `_is_subagent` and `_is_subagent_or_skill_subagent` flags derived from `mode.value`, not from `ctx.is_subagent` directly. They are recomputed in `apply_mode`, so a stale `ctx.is_subagent` between `resolve_mode` and `apply_mode` would only matter if `mode` itself were wrong. |

### 11.2 `inherited_skill` could be lost

| Location | Failure mode |
|----------|--------------|
| `canary.py:312` | Only set when `parse_canary` returns a `verified` skill carrying `skill_name`. Lost on HMAC failure or unregistered skill (`verified=False`). |
| `stages.py:154, 175` | `ctx_inherited_skill = ingress.inherited_skill` then passed to `RequestContext`. Single point of failure if ingress is malformed. |
| `stages.py:622-665` | Auto-registration of new sub-agent — does NOT propagate `inherited_skill` to the Agent's `config` or any storage. The skill is reactivated per-request by `resolve_mode`, not persisted on the Agent. |
| `mode.py:85-86, 138-160` | `resolve_mode` reads `inherited_skill` from `ctx` (not from agent). If `ctx` is reconstructed somewhere (e.g. across a redirect), the value would be lost. Currently `ctx` flows unchanged through the pipeline, so this is theoretical. |
| `mode.py:144-147` | `workspace_name` is read from `ctx.workspace_name` for the workspace skill_state. If `workspace_name` is lost, the skill activates without the workspace arg. |
| `apply_post_process` | Does not read `inherited_skill` from `ctx`. The canary to inject is decided by the agent's current `active_mode`, not by `ctx.inherited_skill`. **Mismatch possible**: sub-agent arrives with `inherited_skill="ticket"` but the Agent's `active_mode` was set differently. |

### 11.3 `subagent_ticket_mode` could be lost

| Location | Failure mode |
|----------|--------------|
| `canary.py:313` | Only `True` when `skill_name == "ticket"`. For `inherited_skill="workspace"`, this is `False` even though sub-agent mode should still propagate. |
| `stages.py:153, 174` | Single read from `ingress.subagent_ticket_mode`. |
| `mode.py:161-171` | Only the `elif` branch reads it. The `inherited_skill` branch (138-160) takes priority and also sets `is_ticket = True`. So `subagent_ticket_mode` is effectively **redundant** when `inherited_skill` is set. |
| `tool_guard.py:377-451` | Uses `subagent_ticket_mode` to allow memory tools. If False, sub-agents cannot use `proxy_ticket`. |

### 11.4 The auto-registration race (CRITICAL)

In `stages.py:622-674`, the auto-registration is wrapped in `try/except`, but
**does not raise** on failure. If `Agent(...)` constructor fails, or
`agent_registry.register(...)` fails, the code falls through with
`agent = None` and `_cascade_apply_default` runs. The sub-agent then
receives `Mode.DEFAULT` instead of `Mode.SKILL_SUBAGENT` (the BUG PATH
commented in `_cascade_apply_default`, lines 747-750).

This is the single most likely place for sub-agent handling to silently
degrade.

### 11.5 `apply_post_process` is the canary re-injection point

In `post_process.py:332-349`, if `agent_registry.get_or_load` returns `None`,
`apply_post_process` is **not called**. The canary is therefore not
re-injected into the response's `task` tool calls. The next-hop sub-agent
will not be marked as a sub-agent.

Even when `apply_post_process` is called, the canary comes from
`agent.snapshot_mode()` (mode.py:492). If the agent's mode is None (no
active skill), the fallback canary is the bare `[[__SUBAGENT__]]\n` (line
508), which sets `inherited_skill=None`. The sub-agent chain loses skill
inheritance at every hop.

---

## 12. Text Diagram of the Pipeline

```
HTTP POST /v1/chat/completions
        │
        ▼
┌──────────────────────────────────────────────────────────────┐
│ app.py:573 proxy_chat_completions()                          │
│  - rate_limit check                                          │
│  - parse_request(request)  ◄──── canary.py:detect_subagent_  │
│      │                                  ingress              │
│      │  ┌────────────────────────────────────┐               │
│      └─►│ RequestContext constructed:        │               │
│         │   is_subagent   = ingress.is_subagent            │
│         │   subagent_ticket_mode = ingress.subagent_...    │
│         │   inherited_skill = ingress.inherited_skill      │
│         │   workspace_name  = ingress.workspace_name       │
│         │   unattended = True (forced if is_subagent)      │
│         │   request_id = uuid4()[:12]                      │
│         └────────────────────────────────────┘              │
│  - RequestPipeline()  (with DEFAULT_STAGES)                 │
│  - if ctx.wants_stream: _keepalive_stream(ctx, pipeline)    │
│  - else: pipeline.run(ctx)                                  │
└──────────────┬───────────────────────────────────────────────┘
               │  pipeline.run() wraps in request_scope
               ▼
┌──────────────────────────────────────────────────────────────┐
│ RequestPipeline._run_stages (request_pipeline.py:91)        │
│   for stage in self.stages:                                 │
│       stage.execute(ctx)                                    │
│       (StageAbort short-circuits)                           │
└──────────────┬───────────────────────────────────────────────┘
               │
               ▼
  ┌─────────────────────────┐
  │ 1. CleanupStage         │  Increments counter. No sub-agent read.
  │    stages.py:381        │  Periodically cleans stale ticket modes.
  └────────────┬────────────┘
               ▼
  ┌─────────────────────────┐
  │ 2. DispatchStage        │  run_dispatch(ctx, features)
  │    stages.py:438        │  → try_skill_activation EARLY-RETURNS for sub-agents
  └────────────┬────────────┘   (dispatch.py:271-273)
               ▼
  ┌─────────────────────────┐
  │ 3. RouteResolutionStage │  Sets ctx.model_route, ctx.model_cfg
  │    stages.py:484        │  CRITICAL.
  └────────────┬────────────┘
               ▼
  ┌─────────────────────────┐
  │ 4. SecurityStage        │  Scans messages. No sub-agent read.
  │    stages.py:516        │  CRITICAL.
  └────────────┬────────────┘
               ▼
  ┌──────────────────────────────────────────────────────────┐
  │ 5. ContextOptimizeStage        stages.py:560             │
  │    - enforce_context_limits                              │
  │    - agent = get_or_load(session_id)                     │
  │      │ if None AND is_subagent:                          │
  │      │   auto-register Agent(is_subagent=True)  ◄─── KEY │
  │    - body_original = deepcopy(ctx.body)                  │
  │    - if agent:                                           │
  │        mode = resolve_mode(ctx, agent)                   │
  │            │ reads ctx.inherited_skill                   │
  │            │ activates skill via _activate_skill         │
  │            │ sets ctx._mode = SKILL_SUBAGENT             │
  │        ctx.body = apply_mode(ctx, agent, ctx.body, mode) │
  │            │ calls prepare_tooling_payload(is_subagent)  │
  │            │ builds system prompt (skill_subagent block) │
  │            │ appends task anchor                         │
  │            │ calls select_context(is_subagent)           │
  │    - else (cascade):                                     │
  │        _cascade_apply_default(ctx, body_original)        │
  │            │ BUG PATH: sub-agent gets Mode.DEFAULT       │
  │    - ctx.messages = deepcopy(body["messages"])           │
  │    - ctx.messages_original = deepcopy(...)               │
  │    CRITICAL.                                             │
  └────────────┬─────────────────────────────────────────────┘
               ▼
  ┌─────────────────────────┐
  │ 6. MotivatorStage       │  Only runs for sub-agents with
  │    motivator_stage.py   │  messages_original set.
  │    - injects turn       │  Skips silently if get_or_load
  │      reminders          │  returns None.
  └────────────┬────────────┘
               ▼
  ┌─────────────────────────┐
  │ 7. ProviderCompatStage  │  apply_provider_compat(llama_body)
  │    stages.py:818        │  Sets ctx.llama_body
  └────────────┬────────────┘
               ▼
  ┌─────────────────────────┐
  │ 8. UpstreamStage        │  prepare_upstream_request()
  │    stages.py:872        │  call_upstream() ──── HTTP to LLM
  │    CRITICAL.            │  Sets ctx.upstream, ctx.response_data
  └────────────┬────────────┘
               ▼
  ┌──────────────────────────────────────────────────────────┐
  │ 9. PostProcessStage             post_process.py:38       │
  │    - _extract_minimax_reasoning                         │
  │    - _scan_write_results (VFS, gates on is_subagent)     │
  │    - _sync_chat_history                                 │
  │    - _persist_messages (SQLite)                         │
  │    - _populate_vfs_cache                                │
  │    - _capture_cache_metrics                             │
  │    - _handle_tool_interception                           │
  │        │ passes is_subagent to ToolContext               │
  │    - _assign_model_name                                 │
  │    - if is_final AND not (is_subagent AND ignore_sub):   │
  │        submit_sync(session_id)                          │
  │    - _skill_post_process(ctx):                          │
  │        │ agent = get_or_load(session_id)                 │
  │        │ if agent:                                      │
  │        │   apply_post_process(ctx, agent, data, mode)    │
  │        │       │ CANARY INJECTION                        │
  │        │       │ canary = skill.get_canary(skill_ctx)    │
  │        │       │ _inject_canary_into_task_calls(...)     │
  │        │       │ clear_active_mode on finish_reason=stop│
  └────────────┬─────────────────────────────────────────────┘
               │
               ▼
        response returned to client

═══════════════════════════════════════════════════════════════════
        On the LLM response, if a `task` tool_call was generated:
        canary.py:_inject_canary_into_task_calls()
            - strips any existing canary from prompt
            - prepends new canary (from skill or bare [[__SUBAGENT__]])
        The next sub-agent request will hit the proxy with this canary
        prepended in its first user message.
═══════════════════════════════════════════════════════════════════
```

---

## 13. Cross-Reference Index

### Every assignment of `is_subagent` (full set, sorted by file:line)

| File:Line | Code |
|-----------|------|
| `types.py:31` | `is_subagent: bool = False` (SubagentIngress default) |
| `types.py:79` | `is_subagent: bool = False` (RequestContext default) |
| `agent/agent.py:84` | `is_subagent: bool = False` (Agent default) |
| `skills/skill_registry.py:49` | `is_subagent: bool = False` (SkillContext field) |
| `tool_pipeline/tool_pipeline.py:46` | `is_subagent: bool = False` (ToolContext field) |
| `canary.py:311` | `result.is_subagent = True` (SKILL canary parsed) |
| `canary.py:321` | `result.is_subagent = True` (SKILL canary unparseable) |
| `canary.py:329` | `result.is_subagent = True` (WORKSPACE canary parsed) |
| `canary.py:341` | `result.is_subagent = True` (WORKSPACE unparseable) |
| `canary.py:344` | `result.is_subagent = True` (bare canary) |
| `stages.py:152` | `is_subagent = ingress.is_subagent` (local var) |
| `stages.py:172` | `messages, flask_request.remote_addr or "", is_subagent,` (RequestContext ctor) |
| `stages.py:250` | `is_subagent: bool = False,` (handle_tool_interception param) |
| `stages.py:313` | `is_subagent = is_subagent or body.get("is_subagent", False)` |
| `stages.py:339` | `is_subagent=is_subagent,` (ToolContext ctor) |
| `stages.py:622` | `if agent is None and getattr(ctx, "is_subagent", False):` (READ) |
| `stages.py:629` | `is_subagent=True,` (Agent ctor in auto-register) |

Note: the only **writer** of `is_subagent` on `RequestContext` is
`parse_request` at `stages.py:172` (constructor). All other lines are READS
or local variables.

### Every assignment of `inherited_skill`

| File:Line | Code |
|-----------|------|
| `types.py:33` | `inherited_skill: Optional[str] = None` (SubagentIngress) |
| `types.py:99` | `inherited_skill: Optional[str] = None` (RequestContext) |
| `canary.py:312` | `result.inherited_skill = skill_name` (SKILL canary) |
| `canary.py:330` | `result.inherited_skill = "workspace"` (WORKSPACE canary) |
| `stages.py:154` | `ctx_inherited_skill = ingress.inherited_skill` (local var) |
| `stages.py:175` | `inherited_skill=ctx_inherited_skill,` (RequestContext ctor) |

### Every assignment of `subagent_ticket_mode`

| File:Line | Code |
|-----------|------|
| `types.py:32` | `subagent_ticket_mode: bool = False` (SubagentIngress) |
| `types.py:82` | `subagent_ticket_mode: bool = False` (RequestContext) |
| `canary.py:313` | `result.subagent_ticket_mode = (skill_name == "ticket")` |
| `stages.py:153` | `_ticket_mode_subagent = ingress.subagent_ticket_mode` (local var) |
| `stages.py:174` | `subagent_ticket_mode=_ticket_mode_subagent,` (RequestContext ctor) |

### Every assignment of `workspace_name`

| File:Line | Code |
|-----------|------|
| `types.py:34` | `workspace_name: str = ""` (SubagentIngress) |
| `types.py:100` | `workspace_name: str = ""` (RequestContext) |
| `canary.py:331` | `result.workspace_name = state.get("workspace_name", "")` |
| `stages.py:155` | `_workspace_name = ingress.workspace_name` (local var) |
| `stages.py:176` | `workspace_name=_workspace_name,` (RequestContext ctor) |
| `mode.py:145-147` | `workspace_name = getattr(ctx, 'workspace_name', ''); if workspace_name: skill_state = {"arg": workspace_name}` (uses, doesn't assign ctx) |

### Every assignment of `Agent.is_subagent`

| File:Line | Code |
|-----------|------|
| `agent/agent.py:84` | `is_subagent: bool = False` (default) |
| `stages.py:629` | `is_subagent=True,` (auto-registration only) |

The `Agent.spawn()` method (agent_registry.py:446-508) does NOT set
`is_subagent=True` when creating child agents. Only the
`ContextOptimizeStage` auto-registration path does.

---

## 14. Direct Answers to the Critical Questions

### Q: Where exactly does `is_subagent` get set?
- On `RequestContext`: only in `parse_request` at `stages.py:172` (constructor
  argument derived from `ingress.is_subagent` set in `canary.py:311, 321,
  329, 341, 344`).
- On `Agent`: only in the `ContextOptimizeStage` auto-registration at
  `stages.py:629` (NOT in `spawn()` at `agent_registry.py:489-495`).
- On `SubagentIngress`: in `detect_subagent_ingress` at `canary.py:311, 321,
  329, 341, 344`.
- On `ToolContext`: from the parameter passed by `handle_tool_interception`
  (`stages.py:313, 339`).
- On `SkillContext`: from `prepare_tooling_payload` (`tool_cache.py:98`).

### Q: Where exactly does `inherited_skill` get set?
- On `RequestContext`: only in `parse_request` at `stages.py:175` (from
  `ctx_inherited_skill = ingress.inherited_skill` at line 154).
- On `SubagentIngress`: in `detect_subagent_ingress` at `canary.py:312, 330`.

### Q: Where exactly does `subagent_ticket_mode` get set?
- On `RequestContext`: only in `parse_request` at `stages.py:174` (from
  `_ticket_mode_subagent = ingress.subagent_ticket_mode` at line 153).
- On `SubagentIngress`: in `detect_subagent_ingress` at `canary.py:313`.

### Q: How does `agent_registry.get_or_load()` work? What's the fallback?
- See section 8.2. Returns `Agent` from in-memory LRU or SQLite. Returns
  `None` if session is unknown (security), at capacity with all in-flight,
  rate-limited, or unauthorized.
- **Fallback for None**: in `ContextOptimizeStage`, if `is_subagent == True`,
  auto-registers a new `Agent(is_subagent=True)` (stages.py:622-674). If
  not sub-agent (or auto-register fails), the cascade fallback
  `_cascade_apply_default` runs with `Mode.DEFAULT`.

### Q: What is `_cascade_apply_default`? When does it run?
- See section 5.8. A fallback function in `stages.py:742-795`.
- Runs when `ContextOptimizeStage` has no `agent` (line 725) — typically
  because `get_or_load` returned `None` and the sub-agent auto-registration
  either didn't run (not a sub-agent) or failed.
- Always uses `Mode.DEFAULT`. For sub-agents, this is the documented "BUG
  PATH" — they get DEFAULT instead of SKILL_SUBAGENT.

### Q: What is `apply_post_process`? When does it run? What does it modify?
- See section 7.4. A function in `mode.py:407-535`.
- Runs from `PostProcessStage._skill_post_process` (post_process.py:347),
  which itself runs at the end of `PostProcessStage.execute` (line 97).
- Modifies:
  - The response `data` dict (canary injection, `skill.post_process`).
  - The agent's `active_mode` (via `clear_active_mode` calls).
- **Critical for sub-agents**: it injects the canary into `task` tool calls
  so the next sub-agent is properly tagged.

### Q: Where is the canary injected into the task prompt?
- `canary.py:467-508` (`_inject_canary_into_task_calls`). Called from
  `mode.py:509` (inside `apply_post_process`).

### Q: Where is the canary parsed from the sub-agent's request?
- `canary.py:271-367` (`detect_subagent_ingress`). Called from
  `stages.py:151` (inside `parse_request`).

### Q: What is the full chain from `ContextOptimizeStage.execute()` entry to the LLM call?
1. `stages.py:568` enter `ContextOptimizeStage.execute(ctx)`.
2. `stages.py:583` `enforce_context_limits`.
3. `stages.py:594` `agent = get_or_load(ctx.session_id)`.
4. `stages.py:622-674` if `agent is None and is_subagent`: auto-register.
5. `stages.py:682` `body_original = deepcopy(ctx.body)`.
6. `stages.py:693` `mode = resolve_mode(ctx, agent)` → sets `ctx._mode`,
   calls `_activate_skill` which sets `ctx._skill_activated = True`.
7. `stages.py:699` `ctx.body = apply_mode(ctx, agent, ctx.body, mode)` →
   mutates `body["tools"]`, `body["messages"]`, `ctx._tools_snapshot`.
8. `stages.py:700` `ctx.messages = deepcopy(ctx.body["messages"])`.
9. `motivator_stage.py:40-126` inject turn reminder (if is_subagent).
10. `stages.py:839` `llama_body = apply_provider_compat(...)`.
11. `stages.py:885` `upstream = prepare_upstream_request(llama_body, ...)`.
12. `stages.py:944` `data = call_upstream(upstream.url, upstream.body, ...)`.
13. `stages.py:945` `ctx.response_data = data`.

---

## 15. Key Observations

1. **Sub-agent state flows through `RequestContext` exclusively.** Once set
   in `parse_request` (stages.py:172, 174, 175, 176), `is_subagent`,
   `subagent_ticket_mode`, `inherited_skill`, and `workspace_name` are never
   re-derived — they are only READ by downstream stages.

2. **There is no HTTP header for sub-agent detection.** It is purely
   text-based in `messages[0].content`. The HMAC signature in the canary
   prevents forgery (when `PROXY_SESSION_SECRET` is set).

3. **The `is_subagent` on the `Agent` dataclass and the `is_subagent` on
   `RequestContext` are independent fields.** They are set in different
   places. The Agent field is set only via auto-registration in
   `ContextOptimizeStage` (stages.py:629), NOT in `spawn()`.

4. **The `apply_post_process` canary injection depends on `get_or_load`
   succeeding in `_skill_post_process`.** For brand-new sub-agent
   sessions, this is a chicken-and-egg: the agent was auto-registered in
   `ContextOptimizeStage`, so by the time `PostProcessStage` runs,
   `get_or_load` should find it. But if the auto-registration silently
   failed, the canary is not re-injected.

5. **The cascade fallback `_cascade_apply_default` is the most likely
   silent-failure point.** It runs in degraded mode and produces
   `Mode.DEFAULT` even for sub-agents. The fix is the auto-registration
   in `ContextOptimizeStage` (stages.py:622-674), which is wrapped in
   `try/except` and does NOT re-raise on failure.

6. **`subagent_ticket_mode` is effectively redundant** when
   `inherited_skill == "ticket"` — `resolve_mode` activates the skill via
   the `inherited_skill` branch (mode.py:138-160) and sets
   `is_ticket = True` directly. The `elif` branch for
   `subagent_ticket_mode` (mode.py:161-171) is only hit when
   `inherited_skill` is empty.

7. **Streaming path has a manual `request_scope`** (`_keepalive_stream` in
   app.py:346), but the non-streaming path relies on the automatic one
   inside `RequestPipeline.run` (request_pipeline.py:86).

8. **The `MotivatorStage` is the only stage that ONLY runs for sub-agents.**
   All other stages run for both main and sub-agent requests (some branch
   on `is_subagent` for behavior, but the stage itself runs).

---

**End of pipeline map. This document is descriptive only — no code was
modified during its production.**

---

## Appendix: Main-Agent VFS Cache-Safe Tail Behavior

Current main-agent VFS behavior governed by
[`docs/specs/main-agent-vfs-cache-safe-tail.md`](../specs/main-agent-vfs-cache-safe-tail.md):

### Canonical Main-Agent Order

```text
primary system
Tier-2 pin, if present
summaries
ancient history
completed retained conversation
main VFS context
entire live user-led exchange
```

The Tier-2 pin is emitted through the existing final pin path
(`_emit_tier2_pin`) and remains immediately after the primary system
message when present.

### Splitter Contract

`_split_completed_and_live_exchange()` in `src/proxy/context_v2.py`:

- Deep-copies the input conversation.
- Locates the latest user unit.
- Walks backward over every immediately preceding consecutive-user
  unit (Wave 4a unconditional walk-back).
- Keeps all later assistant and tool units live.
- Preserves assistant/tool atomicity.
- Returns no live suffix when no user unit exists.

### Main VFS Suppression

Main VFS is omitted when:

- `ctx is None`,
- no user-led live exchange exists,
- `VFS_ENABLED` is false,
- `PROXY_VFS_SCOPE` excludes main,
- the workspace cache is empty,
- all entries are expired (TTL filter at `vfs.py:548-551`),
- the smallest complete rendered block cannot fit the budget,
- or the actual augmented request cost exceeds the safe request budget.

### Path Coverage

Current VFS handling on:

- **Normal chain path** — `context_v2.py:1317+` (the `_select_context_v2` normal
  assembly after `_load_state_from_db`).
- **Fresh-session fit path** — `context_v2.py:1317+` (the early-return fast path
  for fresh sessions with no chain state, no `conversation_summaries`, no
  `message_summaries`).
- **Safe chain-computation fallback** — `context_v2.py:1199-1208` (the
  `except (TypeError, ValueError, KeyError, AttributeError)` handler around
  `compute_chain`).

Fresh-session VFS does not initialize chain state solely because VFS
exists; chain-state insertion is gated on the normal-path CAS.

### Workspace Resolution

The real `RequestContext`, including `client_hint`, reaches
`_build_vfs_context()`. Session-level workspace mapping remains
authoritative; client-level mapping is the fallback when no session
mapping exists.

### Provider Adaptation

- Non-MiniMax providers retain main VFS as `role="system"` and receive no
  synthetic VFS-only barrier.
- MiniMax converts system-origin VFS to `role="user"` and tracks the
  conversion in a per-message identity set so the consecutive-user
  merger skips VFS/current-user merging.
- Remaining user/user adjacency receives an exact
  `{"role": "assistant", "content": " "}` barrier in MiniMax.
- Assistant/multi-tool units remain contiguous across the barrier.

### Main/Inherited Isolation

- Main uses `_build_vfs_context()`; sub-agent uses
  `_build_vfs_inherited_context()`.
- Both builders have prompt-time scope enforcement via
  `_vfs_active_for_context(is_subagent)`.
- Both builders have agent-type defense-in-depth guards
  (`is_subagent` checks) at the top of each builder.
- Main and inherited markers remain distinct:
  - main → `[VFS CONTEXT — Cached file state]`
  - inherited → `[VFS INHERITED CONTEXT — Parent agent's recent file activity]`

### Lifecycle Prefix

- VFS-only mutation preserves the provider-bound prefix before VFS.
- `absent → present` and `present → expired` compare through the
  former insertion boundary.
- `present → changed` and workspace switching compare prefixes before
  each VFS.
- Growing conversation preserves the previous completed region; global
  indexes are not required to be equal.
- Stable rendering is deterministic across age within TTL and LRU
  access order because rows are sorted by path in the main renderer
  (`vfs.py:556`).

See the
[Main-Agent VFS Cache-Safe Tail Specification](../specs/main-agent-vfs-cache-safe-tail.md)
for the authoritative contract.
