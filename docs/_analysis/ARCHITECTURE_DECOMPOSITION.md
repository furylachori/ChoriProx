# ChoriProx — Architectural Decomposition (v2)

**Status:** CURRENT — verified against HEAD `03904a5` (2026-08-02)
**Scope:** Major-component map. Detailed per-topic diagrams are deferred (see §10).
**Companion:** `docs/_analysis/PIPELINE_MAP.md` (1,500-line request-flow walkthrough)

---

## The One-Liner

ChoriProx is an **LLM proxy** (port 6666) between AI coding clients and upstream
model backends. It intercepts every request/response to inject memory, manage
skills, execute tools locally, coordinate multi-agent swarms, and optimize
context — transparently, without client changes.

**Scale:** ~56K LOC in `src/`, ~170K LOC in `tests/`, 20 models across 10 backends.

---

## 1. Product Capability Map (what users see)

| Capability | README/US | Primary code |
|---|---|---|
| Multi-backend model routing | headline #1 | `clients/model_registry.py`, `proxy/model_registry.py`, `proxy_models.json` |
| Context optimization (V2 three-tier) | headline #2 | `proxy/context_v2.py` (2,432 LOC) |
| Memory recall & knowledge graph | headline #3 | `memory/` (SQLite + Neo4j + SOTA sync) |
| Tool interception + local execution + re-query | headline #4 | `proxy/tool_pipeline/`, `memory/tool_loop.py` (1,461 LOC) |
| Virtual Filesystem (cross-agent file sharing) | headline #5 | `proxy/vfs.py` (840 LOC) + state.py VFS block |
| Provider compatibility patches | headline #6 | `pipeline/stage_modules/provider_compat_stage.py` |
| Sub-agent canary filtering | headline #7 | `proxy/canary.py` (1,435 LOC) |
| Streaming SSE reconstruction | headline #8 | `proxy/upstream_sse.py` (766 LOC) |
| Circuit-breaker isolation | headline #9 | `memory/circuit_breaker.py` |
| Task management (ticket/work) | core product | `skills/ticket_skill.py`, `skills/work_skill.py`, 17 TaskActions |
| Workspace management | US-066..072 | `proxy/workspace_state.py`, `commands/workspace_commands.py`, `skills/workspace.py` |
| Custom JSON skills + research swarms | README feature | `proxy/skills/custom_skill_loader.py`, `SubagentPolicy` |
| `^^` command surface | README feature | `commands/` (389 LOC) |

---

## 2. The 8 Subsystems

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     CLIENT (opencode / kilocode)                        │
│                POST /v1/chat/completions (OpenAI-compatible)            │
└────────────────────────────────┬────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  1. REQUEST SHELL & ADMISSION                                          │
│     HTTP auth → parse_request → dedup admission → pipeline → response  │
│                                                                         │
│  ┌───────────┐  ┌────────────┐  ┌──────────────┐  ┌───────────────┐   │
│  │ Auth +     │→│ parse_     │→│ Dedup        │→│ SSE / JSON    │   │
│  │ Rate Limit │  │ request    │  │ (OWNER /     │  │ Response      │   │
│  │ (8 limiters│  │ → Request  │  │  FOLLOWER /  │  │ (streaming +  │   │
│  │  in state) │  │   Context  │  │  REPLAY /    │  │  non-stream)  │   │
│  └───────────┘  └────────────┘  │  CONFLICT)   │  └───────────────┘   │
│                                  └──────────────┘                       │
│  Key: dedup gates whether the pipeline runs AT ALL.                    │
│  REPLAY = zero upstream, zero pipeline (cached artifact).              │
│  FOLLOWER = attach to owner's completion event.                        │
└─────────────────────────────────────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  2. THE 9-STAGE PIPELINE (the spine)                                   │
│                                                                         │
│  [1 Cleanup] → [2 Dispatch] → [3 Route] → [4 Security]                │
│  → [5 ContextOptimize] → [6 Motivator] → [7 ProviderCompat]           │
│  → [8 Upstream] → [9 PostProcess]                                      │
│                                                                         │
│  CRITICAL (re-raise): 3, 4, 5, 8                                      │
│  Stages are shared singletons; only DispatchStage._features is mutable  │
└─────────────────────────────────────────────────────────────────────────┘
         │                    │                      │
         ▼                    ▼                      ▼
┌─────────────────┐ ┌──────────────────┐ ┌────────────────────────────┐
│ 3. CONTEXT V2   │ │ 4. SKILL / MODE  │ │ 5. TOOL EXECUTION          │
│    (pipeline-   │ │    / AGENT       │ │                            │
│     adjacent)   │ │                  │ │  ToolRegistry (30 tools)   │
│                 │ │  SkillRegistry   │ │  ToolPipeline              │
│  Chain-hash     │ │  TrustToken gate │ │    (parse → guard →        │
│  fork detection │ │  Custom JSON     │ │     execute → format)      │
│  Three-zone     │ │    loader        │ │  MemoryToolGuard           │
│  budget engine  │ │  Mode lifecycle  │ │    (8-check chain)         │
│  Summary binary │ │  Agent dataclass │ │  Tool re-query loop        │
│  search         │ │  AgentRegistry   │ │    (tool_loop.py 1,461 LOC)│
│  VFS placement  │ │    (lock #0,     │ │  17 TaskActions            │
│  Ancient        │ │     coarsest)    │ │  Tool arg normalization    │
│  compression    │ │  Sub-agent       │ │    (tool_parsing/ —        │
│  Tier-2 pin     │ │    canary +      │ │     shared leaf, 4+        │
│  emission       │ │    inheritance   │ │     consumers, THE         │
│                 │ │  Swarm policy    │ │     LANDMINE)              │
│  2,432 LOC      │ │    (SubagentPol.)│ │                            │
│  (context_v2.py)│ │  ^^command sys.  │ │                            │
│                 │ │  Workspace sys.  │ │                            │
└─────────────────┘ └──────────────────┘ └────────────────────────────┘
         │                    │                      │
         └────────────────────┼──────────────────────┘
                              ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  6. MEMORY & PERSISTENCE (background + storage)                        │
│                                                                         │
│  ┌─────────────────────┐  ┌─────────────────────┐  ┌────────────────┐ │
│  │ SQLite (memory.db)   │  │ Neo4j (graph)        │  │ SOTA Sync      │ │
│  │ 12 tables            │  │ :Entity, :Memory     │  │ (background    │ │
│  │ WAL, pool of 48      │  │ :Task, :Bug, :Review │  │  extraction)   │ │
│  │ Corruption auto-heal │  │ :WorkSession         │  │                │ │
│  │ db_pool.py 2,747 LOC │  │ :Workspace, :Session │  │ sync.py 1,404  │ │
│  │ (largest file)       │  │ :ReviewNote          │  │ sota_channel   │ │
│  │                      │  │ :_Migration          │  │ sota_client    │ │
│  │ event_log.db         │  │ [:BLOCKED_BY]        │  │ executors      │ │
│  │ (separate DB)        │  │ [:HAS_PARENT]        │  │ batch_manager  │ │
│  │                      │  │ [:HAS_NOTE]          │  │ extraction_val │ │
│  │                      │  │ [:MENTIONS]          │  │                │ │
│  │                      │  │ [:RELATES_TO]        │  │ 3,548 LOC      │ │
│  │                      │  │ [:LAST_USED_WS]      │  │ total          │ │
│  │                      │  │ 11 migrations        │  │                │ │
│  └─────────────────────┘  └─────────────────────┘  └────────────────┘ │
└─────────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  7. UPSTREAM TRANSPORT & PROVIDER COMPAT                               │
│                                                                         │
│  http_client.py (882) — unified HTTP call, malformed-response          │
│    validation, 1-retry-on-empty-usage                                   │
│  upstream_sse.py (766) — SSE accumulation, wall-clock budget,          │
│    3-class error hierarchy (WIRED + LOAD-BEARING)                       │
│  model_registry.py (336) — 6-layer fallback model resolution           │
│  tool_cache.py (504) — tool-schema caching/optimization                │
│  provider_compat — DeepSeek/Kimi/MiniMax/MiMo patches                  │
│  openai_compat.py — response shape builder + single-source validator   │
└─────────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  8. CROSS-CUTTING BACKBONE                                             │
│                                                                         │
│  State & Concurrency:                                                   │
│    state.py (705 LOC) — 15 ordered locks, ~14 counters, 9+ singletons, │
│    8 rate limiters. The shared-memory backbone every subsystem imports.  │
│    Lock violations caused 8+ adversarial-round deadlocks.               │
│                                                                         │
│  VFS (Virtual Filesystem):                                              │
│    vfs.py (840 LOC) + state.py VFS block + lock #9.                    │
│    Write-through, rollback snapshots, conflict detection,               │
│    cache-safe-tail placement. Own spec, own env (PROXY_VFS_SCOPE).     │
│                                                                         │
│  Client Adapters:                                                       │
│    src/clients/ (810 LOC) — ClientAdapter ABC, OpenCode/KiloCode,      │
│    ClientRegistry, ToolParserRegistry, sota_client.py (layering        │
│    firewall — ONLY legal non-memory path to SOTA).                     │
│                                                                         │
│  Security & Identity:                                                   │
│    auth.py, session_tokens.py (HMAC), SecurityStage (CRITICAL),        │
│    memory/security.py (Cypher injection guard).                        │
│                                                                         │
│  Error Model & Boundaries:                                              │
│    StageAbort (control flow), BackendError, InvalidRequestError,        │
│    InfrastructureError/ConstraintViolation/QueryTimeoutError,           │
│    SwarmCanaryError, UpstreamStreamError. 3-layer malformed-completion  │
│    boundary (http_client → PostProcess → SSE). docs/error-taxonomy.md. │
│                                                                         │
│  Observability:                                                         │
│    event_log.py (autocommit SQLite handler), metrics.py (Prometheus,   │
│    11 counters/histograms, /metrics endpoint).                         │
│                                                                         │
│  Resilience:                                                            │
│    Circuit breakers (Neo4j, SOTA), db_recovery.py (881 LOC auto-heal), │
│    dedup store + completion coordinator (admission control).           │
│                                                                         │
│  Operability:                                                           │
│    devtools/debug_session.py (1,520 LOC) — pytest-free harness.        │
│    16 runnable templates. 11 Neo4j migrations with auto-retry.         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 3. The 9 Pipeline Stages

| # | Stage | What it does | Key complexity |
|---|---|---|---|
| 1 | **Cleanup** | Request counter; every 100th: session/message/workspace GC + stale ticket-mode cleanup | Lightweight (4 cleanup responsibilities) |
| 2 | **Dispatch** | Priority: enhancer (dead, priority 1) → **skills** (priority 2) → **commands** (priority 4) | Two separate registries (SkillRegistry vs COMMAND_REGISTRY) |
| 3 | **RouteResolution** | Model name → backend URL + config | **6-layer** fallback matching (exact → wildcard → case-insensitive → normalization → backend_model → default) |
| 4 | **Security** | Malicious content detection on user messages | Fail-closed, CRITICAL |
| 5 | **ContextOptimize** | `enforce_context_limits` → `resolve_mode` → `apply_mode` (which builds the V2 context window internally) | Most computation per request; installs swarm policy |
| 6 | **Motivator** | Sub-agent context-usage reminders (50/75/90%) | Sub-agent only |
| 7 | **ProviderCompat** | Per-provider patches (MiniMax temp clamp, DeepSeek/Kimi quirks) + tool-schema optimization | Provider-specific |
| 8 | **Upstream** | `prepare_upstream_request` → `call_upstream` (in http_client.py) | Stream timeout, malformed-response validation, SSE reconstruction |
| 9 | **PostProcess** | **8 sub-methods** (reasoning extract, VFS write scan, chat history sync, SQLite persist, VFS cache populate, cache metrics, tool interception, model name assign) + 3 split skill ops (`apply_skill_response_transform` → `inject_task_canaries` → `finalize_skill_lifecycle` in `finally`) | **Heaviest by LOC** (843). Tool interception is unconditional (parsing is internal). SOTA sync gated on: final response + not ignored subagent + session_id + memory_enabled |

---

## 4. The Data Stores

```
┌──────────────────────────────────┐  ┌──────────────────────────────────┐
│     SQLite (memory.db)           │  │     Neo4j (graph)                │
│                                  │  │                                  │
│  12 tables:                      │  │  Labels:                         │
│  messages, conversation_summaries│  │  :Entity, :Memory, :Task, :Bug   │
│  extraction_batches, tasks       │  │  :Review, :ReviewNote            │
│  message_summaries, entities     │  │  :WorkSession, :Workspace        │
│  facts, supersessions            │  │  :Session, :_Migration           │
│  memory_log, sessions            │  │  :MigrationLock                  │
│  rate_limiter_state              │  │                                  │
│  tool_call_audit                 │  │  Relationships:                  │
│                                  │  │  [:BLOCKED_BY], [:HAS_PARENT]    │
│  WAL mode, pool of 48 conns     │  │  [:HAS_NOTE], [:MENTIONS]        │
│  db_pool.py: 2,747 LOC          │  │  [:RELATES_TO]                   │
│  (largest file in codebase)      │  │  [:LAST_USED_WORKSPACE]          │
│  Corruption auto-heal            │  │  [:HAS_WORK_SESSION]             │
│                                  │  │  [:FOUND_DURING], [:RESOLVED_BY] │
│  event_log.db (separate file)    │  │  [:HAS_REVIEW]                   │
│  — autocommit, daily purge       │  │                                  │
│                                  │  │  Enterprise, :7687 prod / :7688  │
│                                  │  │  test. 11 migrations (auto-retry │
│                                  │  │  on failure).                    │
└──────────────────────────────────┘  └──────────────────────────────────┘
```

---

## 5. Request Flow

```
Client POST /v1/chat/completions
  │
  ├─ Auth (PROXY_API_KEY) + rate limit (8 limiters)
  ├─ parse_request → RequestContext          ← app.py:896
  ├─ Dedup check (fingerprint)               ← app.py:933+ (AFTER parse)
  │   ├─ REPLAY → serve cached artifact, zero upstream, zero pipeline
  │   ├─ FOLLOWER → attach to owner's completion event, wait
  │   ├─ CONFLICT → 409 stream_conflict
  │   └─ OWNER ↓
  │
  ├─ Pipeline.run(ctx) [9 stages, wrapped in agent_registry.request_scope]
  │   ├─ Stage 2: Dispatch (skills priority 2, commands priority 4)
  │   ├─ Stage 5: enforce_context_limits → resolve_mode → apply_mode
  │   │   (apply_mode builds the V2 context window: system + summaries
  │   │    + VFS + live messages within budget; installs swarm policy)
  │   ├─ Stage 8: call upstream LLM (SSE reconstruction if stream=true)
  │   └─ Stage 9: PostProcess
  │       ├─ 8 sub-methods (unconditional, in order)
  │       │   #7 tool interception → ToolPipeline.run() internally
  │       │     parses for tool calls; if found: guard → execute → re-query
  │       ├─ SOTA sync (gated: final + not-ignored-subagent + session + memory)
  │       └─ 3 split skill ops (in finally):
  │           apply_skill_response_transform → inject_task_canaries
  │           → finalize_skill_lifecycle
  │
  └─ Response (SSE stream or JSON)
```

---

## 6. Dead / Vestigial (named explicitly)

| Component | Status | Evidence |
|---|---|---|
| `src/enhancer/` (`^^imp`) | **DEAD** — dispatch priority 1, always returns None | `ProxyFeatures` has no `enhancer_enabled` field |
| `src/qwen_tools.py` (273 LOC) | **DEAD** — zero importers | Not imported by any module |
| `recall` / `search` memory tools | **STUBS** — always return empty | Deliberately disabled |
| KiloCode `normalize_tool_args` | **VESTIGIAL** — wired but never called in production | Schema-driven path supersedes it |
| `OpenAIToolParser.parse` | **STUB** — returns `[]` | Production uses `parse_all_tools_from_content` directly |

---

## 7. Deferred Deep-Dive Topics (ranked by LOC)

Each of these deserves its own detailed diagram + explanation document.

| # | Topic | LOC | Why it deserves its own diagram |
|---|---|---|---|
| 1 | **SOTA sync pipeline** | 3,548 | Exchange identification → summarization → entity/fact extraction → Neo4j persistence → ancient compression. Own executor pools (4 pools, 14 workers). |
| 2 | **Context V2 internals** | 2,884 | Chain-hash fork detection, three-zone budget engine, summary binary search, VFS placement, finalizer safety net, Tier-2 pin emission, tool-pair integrity. |
| 3 | **Tool result handling** | 1,461 + 484 | Guard chain (8 checks) → local execution → re-query loop (up to 5 iterations) → closure finalization (COMPLETE/PARTIAL/FAILED). The core runtime behavior. |
| 4 | **SSE / upstream transport** | 1,785 | Stream accumulation, wall-clock budget, malformed-response 3-layer boundary (http_client → PostProcess → SSE), provider-compat patches, 6-layer model resolution. |
| 5 | **Canary / sub-agent ingress** | 1,435+ | HMAC signing, position-0 parsing, mode inheritance, swarm policy propagation, WORKSPACE canary (inherited_skill=None), session tokens. |
| 6 | **VFS lifecycle** | ~1,000+ | Write-through, edit-through, rollback snapshots, conflict detection, cache-safe-tail placement, workspace scoping, Tier-2 instruction capture. |
| 7 | **Dedup / completion coordination** | 889 + lock #14 | Fingerprint HMAC, 4-state admission (OWNER/FOLLOWER/REPLAY/CONFLICT), streaming follower lifecycle, replay safety gate, CompletionArtifact dual-purpose. |
| 8 | **Workspace system** | ~1,381 | state.py #6 → workspace_state → Neo4j persistence → canary inheritance → VFS scoping → commands. Cross-cuts 4 packages. |
| 9 | **Skill activation / mode lifecycle** | ~1,400 | TrustToken gate, custom JSON loading (atomic publication), swarm admission (two-constant split), per-turn deactivation contract, CAS compare-and-clear. |
| 10 | **Lock hierarchy / state management** | 705 + 149 | 15 positions (#0–#14), 8 rate limiters, ~14 counters, 9+ singletons. The concurrency backbone. Violations caused 8+ deadlocks. |

---

## 8. Key Architectural Invariants

1. **One Neo4j driver.** `get_shared_executor_from_env()` is the only constructor.
2. **15-position lock ordering.** Ascending only. #0 (AgentRegistry) coarsest, #14 (dedup) leaf.
3. **`proxy.state` is the shared-memory backbone.** Import as module, never named imports.
4. **Stages are shared singletons.** Only `DispatchStage._features` is mutable (reset per request).
5. **Dedup gates the pipeline.** FOLLOWER/REPLAY never call `pipeline.run()`.
6. **Tool interception is unconditional.** ToolPipeline.run() internally parses for tool calls.
7. **SOTA sync is fire-and-forget.** Gated on 4 conditions; runs in background pools.
8. **Skill lifecycle finalize runs in `finally`.** Always executes, even on abort.
9. **Custom skills are atomic.** One invalid skill → nothing published.
10. **`recall`/`search` are stubs.** Memory recall is deliberately disabled.

---

*Generated from full-repo audit at HEAD `03904a5`. See the document taxonomy in the internal documentation.*
