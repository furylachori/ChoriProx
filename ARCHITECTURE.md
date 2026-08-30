# ChoriProx Architecture

> Last verified: 2026-06-25 | Branch: main

## Request Flow

POST /v1/chat/completions
│
├─ parse_request() ── session_id, sub-agent canary detection
├─ Priority dispatch: enhancer (^^imp) → skills (^^ticket, ^^workspace) → commands (^^help, ^^status)
├─ resolve_model_route() ── multi-backend routing via proxy_models.json
├─ detect_malicious_content() ── blocks requests matching injection patterns (400)
├─ enforce_context_limits() ── token budget check
├─ ContextOptimizeStage (select_context, pipeline/stage_modules/context_optimize.py) ── V2 three-tier context assembly
├─ prepare_tooling_payload() ── tool filtering + memory tool schema append
├─ apply_proxy_recovery_system() ── recovery / circuit breaker
├─ ProviderCompatStage (apply_provider_compat, pipeline/provider_compat.py:155) ── DeepSeek / Kimi / MiniMax / MiMo patches
├─ prepare_upstream_request() ── per-model auth injection
├─ call_upstream() ── HTTP POST (3× retry, 429 handling, semaphores)
├─ PostProcessStage:
│   ├─ _extract_minimax_reasoning()
│   ├─ _scan_write_results() ── VFS write result scanning + stale rollback
│   ├─ _sync_chat_history()
│   ├─ _persist_messages()
│   ├─ _populate_vfs_cache() ── VFS read result caching
│   ├─ _handle_tool_interception() → memory_tool_loop() → re-query
│   └─ _assign_model_name()
│   then: submit_sync() ── background SOTA memory extraction
│   then: _skill_post_process() ── skill post-processing + apply_post_process()
└─ Stream SSE or return JSON

## Pipeline Stages

9 stages executed in order by RequestPipeline:

| # | Stage | Purpose |
|---|-------|---------|
| 1 | CleanupStage | Message normalization |
| 2 | DispatchStage | Command/enhancer/skill detection |
| 3 | RouteResolutionStage | Model → backend routing |
| 4 | SecurityStage | Malicious content detection (blocks with 400) |
| 5 | ContextOptimizeStage | Token budget, context assembly, memory injection |
| 6 | MotivatorStage | Sub-agent turn-aware reminder injection |
| 7 | ProviderCompatStage | Provider-specific parameter mutations |
| 8 | UpstreamStage | HTTP call to backend |
| 9 | PostProcessStage | Tool interception, persistence, VFS, sync |

## Directory Map

### Core Proxy (`src/proxy/`)
| File | Purpose |
|------|---------|
| `app.py` | Flask app, all HTTP route handlers |
| `admin.py` | Admin blueprint: /health, /v1/models, /v1/context/config, /v1/tools/cache, /v1/debug/agents |
| `state.py` | Single source of truth for ALL shared mutable state |
| `auth.py` | Authentication, error sanitization, malicious content detection |
| `context.py` | Token estimation and context optimization (V1) |
| `context_v2.py` | Three-tier context assembly (V2) |
| `canary.py` | Sub-agent canary parsing, HMAC signing, verification |
| `session_tokens.py` | HMAC-signed session tokens for client identity |
| `session.py` | Session management — histories, summaries, cleanup |
| `chat_history.py` | Chat history cache with LRU eviction |
| `http_client.py` | HTTP session management, call_upstream() |
| `streaming.py` | SSE formatting and streaming |
| `requery.py` | LLM requery after tool interception |
| `model_registry.py` | Model resolution, URL construction, adapter management |
| `rate_limiter.py` | Per-key sliding-window rate limiter |
| `mode.py` | Mode resolution and application (ticket, workspace) |
| `metrics.py` | Prometheus metrics for observability |
| `lock_ordering.py` | Runtime lock-ordering enforcement |
| `feature_flags.py` | Feature flags for incremental rollout |
| `tool_cache.py` | Tool schema caching |
| `tool_parsing/` | Multi-format tool call parsers (XML, JSON, YAML, DSML) |
| `types.py` | Dataclasses (RequestContext, TokenEstimate, etc.) |
| `utils.py` | Shared utility functions |
| `vfs.py` | Virtual Filesystem Layer — workspace-scoped LRU caching |
| `workspace_state.py` | Thread-safe per-session workspace state |
| `eviction_coordinator.py` | Cross-cutting eviction coordination |
| `json_formatter.py` | JSON log formatter |
| `skill_lifecycle.py` | IoC bridge between Agent and SkillRegistry |

### Pipeline (`src/proxy/pipeline/`)
| File | Purpose |
|------|---------|
| `stages.py` | Façade: re-exports 9 stages + DEFAULT_STAGES (impls in `stage_modules/`) |
| `stage_modules/` | 8 stage leaf modules (cleanup, dispatch_stage, route_resolution, security, context_optimize, provider_compat_stage, request_parse, upstream) |
| `request_pipeline.py` | Pipeline orchestrator |
| `dispatch.py` | Dispatch extraction — command/enhancer/skill handlers |
| `post_process.py` | Post-processing: tool interception, persistence, VFS, sync |
| `provider_compat.py` | Provider-specific parameter mutations (DeepSeek, Kimi, MiniMax, MiMo) |
| `motivator_stage.py` | Sub-agent turn-aware reminder injection |
| `agent_telemetry.py` | Per-agent state tracking (lock position #8) |
| `prompt_injector.py` | Static methods for message list augmentation |
| `conditional_stage.py` | Base class for non-crashing pipeline stages |
| `threshold_engine.py` | Percentage-based event trigger engine |

### Tool Pipeline (`src/proxy/tool_pipeline/`)
| File | Purpose |
|------|---------|
| `tool_pipeline.py` | Unified tool lifecycle: parse → classify → guard → execute → format |
| `tool_registry.py` | Single source of truth for all tool names/categories |
| `tool_guard.py` | Pre-execution guard pipeline (5 checks) |
| `tool_limits.py` | Tool call hard limits per request |
| `tool_call.py` | ToolCall dataclass |
| `tool_call_tracker.py` | Per-request call tracking with provenance |
| `action_registry.py` | Action dispatch for proxy_ticket |
| `actions/` | Action implementations (create_tasks, update_task, edit, etc.) |

### Agent (`src/proxy/agent/`)
| File | Purpose |
|------|---------|
| `agent.py` | Full Agent ownership model (mode, telemetry, message buffer) |
| `agent_registry.py` | Agent lifecycle, LRU caching, tree tracking |
| `agent_config.py` | Per-agent settings |
| `message_buffer.py` | Thread-safe message buffer with budget-aware trimming |
| `types.py` | ModeState, TaskProposal dataclasses |

### Memory (`src/memory/`)
| File | Purpose |
|------|---------|
| `db.py` | SQLite database layer with WAL mode |
| `graph.py` | Neo4j client for entity/fact graph |
| `query_executor.py` | Centralized Neo4j query execution with retry |
| `sync.py` | Background SOTA memory extraction sync |
| `sota_client.py` | SOTA LLM client for extraction/summarization |
| `sota_channel.py` | SOTA abstraction layer |
| `tool_loop.py` | Tool interception loop for memory tools |
| `recall.py` | Memory recall and context injection filtering |
| `security.py` | Memory poisoning defenses |
| `circuit_breaker.py` | Neo4j failure isolation |
| `task_graph.py` | Neo4j task graph operations |
| `task_graph_batch.py` | Batch task creation/updates |
| `task_graph_queries.py` | Task query operations |
| `task_graph_common.py` | Shared constants and types |
| `persistence.py` | Entity/fact storage to SQLite + Neo4j |
| `batch_manager.py` | Extraction batch management |
| `executors.py` | Bounded thread pool executors |
| `extraction_validator.py` | Entity/relationship type normalization |
| `tools/` | Memory tool implementations (recall, search, proxy_ticket, proxy_work) |
| `migrations/` | Neo4j schema versioning (11 migrations) |
| `prompts/` | LLM prompts for extraction + summary |

### Skills (`src/skills/`)
| File | Purpose |
|------|---------|
| `ticket_skill.py` | TicketSkill — task management via ^^ticket |
| `work_skill.py` | WorkSkill — work-mode task management |
| `workspace.py` | WorkspaceSkill — workspace-scoped management via ^^workspace |

### Clients (`src/clients/`)
| File | Purpose |
|------|---------|
| `interface.py` | ClientAdapter ABC, BackendAdapter, capability types |
| `kilocode.py` | KiloCode client adapter |
| `opencode.py` | OpenCode client adapter |
| `registry.py` | ClientRegistry singleton |
| `capabilities.py` | KILOCODE_CAPABILITIES, OPENCODE_CAPABILITIES |
| `orchestrator_config.py` | ProxyFeatures, HookCapabilities |
| `model_registry.py` | Client-side model configuration |
| `model_registry.py` | Client-side model configuration, `QwenXMLToolParser` (delegates to `proxy/tool_parsing/parsers.py`) |

### Commands & Enhancer
| File | Purpose |
|------|---------|
| `src/commands/__init__.py` | ^^help, ^^status, command detection/dispatch |
| `src/commands/task_commands.py` | ^^ticket trigger detection |
| `src/enhancer/__init__.py` | ^^imp trigger detection |
| `src/enhancer/core.py` | Enhancement context building + SOTA call |
| `src/enhancer/prompts.py` | Enhancement prompts |

## Tool Inventory

21 registered tools across 5 categories:

### EDITOR (7 tools)
| Tool | Available in Skills | Description |
|------|-------------------|-------------|
| `bash` | All | Shell command execution |
| `read` | ticket only | File reading |
| `glob` | ticket only | File pattern matching |
| `edit` | All | File editing (search/replace) |
| `write` | All | File writing |
| `grep` | ticket only | Content search (sandboxed) |
| `suggest` | All | Code review suggestions |

### WEB (2 tools)
| Tool | Available in Skills | Description |
|------|-------------------|-------------|
| `webfetch` | ticket only | URL content fetching |
| `MiniMax_web_search` | All | Web search via MiniMax |

### MEMORY (4 tools)
| Tool | Available in Skills | Description |
|------|-------------------|-------------|
| `recall` | All | Semantic + keyword memory search (STUB — returns empty) |
| `search` | ticket + work | Keyword search over chat history (STUB — returns empty) |
| `proxy_ticket` | ticket + work | Task CRUD operations (proxy_ticket action registry) |
| `proxy_work` | ticket + work | Work-mode task CRUD (proxy_work action registry) |

### TASK (1 tool)
| Tool | Available in Skills | Description |
|------|-------------------|-------------|
| `task` | All | Task planning |

### PASSTHROUGH (6 tools)
| Tool | Available in Skills | Description |
|------|-------------------|-------------|
| `todowrite` | ticket only | Todo list management |
| `ask` | All | User questions |
| `question` | ticket only | User questions (alias) |
| `new_task` | All | Create new task |
| `TaskCreateTool` | All | Task creation (alias) |
| `TaskUpdateTool` | All | Task update (alias) |

## Route Inventory

### Main Routes (app.py)
| URL | Method | Auth | Rate Limiter |
|-----|--------|------|-------------|
| `/v1/chat/completions` | POST | _require_api_key (localhost bypass) | chat_rate_limiter (60/min) |
| `/v2/chat/completions` | POST | Same | Same |
| `/metrics` | GET | None | admin_rate_limiter (30/min) |

### Admin Routes (admin.py)
| URL | Method | Auth | Purpose |
|-----|--------|------|---------|
| `/v1/models`, `/v2/models` | GET | None (exempt) | List registered models |
| `/health` | GET | None (exempt) | Health check (Neo4j + SQLite) |
| `/v1/context/config` | GET | check_config_auth | Get context config |
| `/v1/context/config/toggle` | POST | check_config_auth | Toggle token calculation |
| `/v1/context/sessions` | GET | check_config_auth | List active sessions |
| `/v1/context/session/<id>` | GET | check_config_auth | Session details |
| `/v1/tools/cache` | GET | check_config_auth | Tool cache status |
| `/v1/tools/cache/clear` | POST | check_config_auth | Clear tool cache |
| `/v1/tools/cache/<hash>` | GET | None | Get cached tools |
| `/v1/debug/agents` | GET | require_debug_auth | List agents |
| `/v1/debug/agents/<id>` | GET | require_debug_auth | Agent details |
| `/v1/debug/agents/<id>/history` | GET | require_debug_auth | Agent history |

## Storage

### SQLite Tables (12 tables)
| Table | Purpose |
|-------|---------|
| `messages` | Chat message storage with pre-computed token estimates |
| `conversation_summaries` | Per-session conversation summaries |
| `message_summaries` | Per-message summaries (V2 context) |
| `extraction_batches` | SOTA extraction batch tracking |
| `entities` | Entity storage (knowledge graph mirror) |
| `facts` | Fact/relationship storage |
| `supersessions` | Fact supersession tracking |
| `tasks` | Local task storage |
| `sessions` | Session existence tracking |
| `memory_log` | Recovery/audit log for Neo4j memories |
| `rate_limiter_state` | Persistent rate limiter state |
| `tool_call_audit` | Tool call audit log |

### Neo4j Schema
**Labels:** Entity, Task (dual-labeled :Entity:Task), Memory, _Migration, MigrationLock

**Relationships:**
| Relationship | Direction | Properties |
|-------------|-----------|-----------|
| `[:BLOCKED_BY]` | (blocked)→(blocker) | created_at |
| `[:HAS_PARENT]` | (child)→(parent) | created_at |
| `[:RELATES_TO]` | Entity-to-entity | — |
| `[:SUPERSEDES]` | Fact-to-fact | — |

**Task Node Properties:** uuid, short_hash, name, priority, status, summary, tag, planning_info, created_at, deleted_at, status_changed_at, workspace_id, session_id

**ReviewNote Nodes:**
`(:ReviewNote)` nodes are connected to Task nodes via `[:HAS_NOTE]` relationships.

| Label | Properties |
|-------|-----------|
| `ReviewNote` | uuid, content, author, created_at, workspace_id |

| Relationship | Direction | Properties |
|-------------|-----------|-----------|
| `[:HAS_NOTE]` | (task)→(review_note) | — |

## Lock Ordering

| Position | Lock | Type | Purpose |
|----------|------|------|---------|
| #0 | AgentRegistry._lock | Lock | Agent lifecycle |
| #1 | context_config_lock | RLock | Context configuration |
| #2 | chat_histories_lock | RLock | Chat history cache |
| #3 | chat_summaries_lock | Lock | Chat summaries |
| #4 | tool_cache_lock | Lock | Tool cache |
| #5 | request_counter_lock | Lock | Request counter |
| #6 | active_workspace_lock | RLock | Workspace state |
| #7 | Agent._mode_lock | Lock | Per-agent mode |
| #8 | AgentTelemetry._lock | Lock | Per-agent telemetry |
| #9 | vfs_lock | Lock | VFS cache + write log + rollback |
| #10 | _chain_state_lock | Lock | V2 chain-hash state per (session_id, model_name) |
| #11 | canary_verify_cache_lock | Lock | Canary verification cache |
| #12 | log_overflow_count_lock | Lock | Leaf — never acquires other locks |
| #13 | recovery_lock | RLock | Leaf — memory.db_recovery corruption-recovery serialization |
| #14 | dedup_lock | Lock | Leaf — DedupStore keyed completion map (CP4) |

Never acquire a lower-numbered lock while holding a higher-numbered one.

## Provider Compatibility

| Provider | Patches |
|----------|---------|
| DeepSeek | Inject empty reasoning_content on assistant messages missing it |
| Kimi | Inject reasoning_content on tool_calls messages; clamp temperature [0,1] |
| MiniMax | Convert system→user role; clamp temperature (0,1] |
| MiMo | Inject reasoning_content on assistant messages missing it |

## Known Limitations

- `recall` and `search` memory tools are STUBS (return empty results)
- Streaming is buffered: proxy forces stream:false upstream, parses full response, re-streams SSE
- Client adapter layer is partial — core routing does not go through clients/registry.py
- Phase-stable KV cache prefix: not yet designed
