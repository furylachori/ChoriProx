# ChoriProx — User Stories (Behavioral Contracts)

> **Purpose:** Non-negotiable behavioral contracts expressed as testable assertions.
> **Referenced by:** `AGENTS.md`
> **When to read:** Before modifying behavior in any covered subsystem.
> **Format:** `US-XXX: WHEN <condition> THEN <expected behavior> [AND MUST NOT <forbidden behavior>]`

---

## Circuit Breaker & Neo4j Isolation

### US-001: Neo4j Unavailable — Proxy Still Serves LLM Requests
**WHEN** Neo4j is unreachable or the circuit breaker is OPEN
**THEN** the proxy MUST still accept and serve `/v1/chat/completions` requests
**AND MUST NOT** return 5xx errors for chat completions due to Neo4j unavailability

### US-002: Circuit Breaker Half-Open Probe
**WHEN** the circuit breaker is HALF_OPEN
**THEN** exactly ONE probe request MUST be allowed through
**AND** if that probe succeeds, the circuit MUST transition to CLOSED
**AND** if that probe fails, the circuit MUST return to OPEN

### US-003: Circuit Breaker Health Check Timeout
**WHEN** `is_available()` is called on a Neo4j health check
**THEN** it MUST complete within 5 seconds
**AND MUST NOT** hang indefinitely on slow or unresponsive Neo4j

### US-004: QueryExecutor Error Hierarchy
**WHEN** a Neo4j operation fails
**THEN** it MUST raise `InfrastructureError` for transient failures, `ConstraintViolation` for uniqueness violations, or `QueryTimeoutError` for timeouts
**AND MUST NOT** leak raw `neo4j.exceptions` to callers

---

## Session Tokens & Canary System

### US-005: Unsigned Session Token Warning
**WHEN** `PROXY_SESSION_SECRET` is not set at startup
**THEN** a WARNING-level log MUST be emitted stating that session tokens are unsigned
**AND** `create_session_token()` MUST return an empty string (fail-safe)

### US-006: Canary Verification Fail-Safe
**WHEN** `PROXY_SESSION_SECRET` is not set
**THEN** `_verify_canary()` MUST accept any signature (return True)
**AND** a WARNING-level log MUST be emitted

### US-007: Canary Signature Validation
**WHEN** `PROXY_SESSION_SECRET` IS set and a canary has an invalid HMAC signature
**THEN** `_verify_canary()` MUST return False using `hmac.compare_digest()` (constant-time)
**AND MUST NOT** use `==` for signature comparison

### US-008: Session Token Expiry
**WHEN** a session token is older than the configured `max_age_seconds` (default 24h)
**THEN** `validate_session_token()` MUST reject it
**AND** the token MUST NOT be accepted for session identification

### US-009: Future Timestamp Guard
**WHEN** a session token has a `created_at` timestamp more than 5 minutes in the future
**THEN** `validate_session_token()` MUST reject it
**AND MUST NOT** accept tokens with clock-skewed timestamps

### US-010: Sub-Agent Canary Detection — First Message Only
**WHEN** `detect_subagent_ingress()` processes messages
**THEN** it MUST check only the FIRST user message for canary markers
**AND MUST NOT** scan tool call results or subsequent messages (preventing infinite recursion)

---

## Rate Limiting

### US-011: Rate Limiter Per-Bucket Independence
**WHEN** a command rate limit is exhausted for session X
**THEN** chat completions for session X MUST still be allowed
**AND** commands from session Y MUST still be allowed
**AND** each `RateLimiter` instance MUST operate independently

### US-012: Rate Limiter Localhost Exemption
**WHEN** a request originates from `127.0.0.1`
**THEN** the chat rate limiter MUST NOT block it
**AND** local health checks MUST NOT be rate-limited

### US-013: Rate Limiter Cleanup
**WHEN** `check()` is called on a RateLimiter
**THEN** expired window entries MUST be cleaned up at least every 10th call
**AND** stale entries MUST NOT accumulate unboundedly

---

## Skill System & Mode Management

### US-014: Skill Activation Snapshot
**WHEN** a skill is activated (`on_activate` called)
**THEN** the agent's mode MUST be snapshot-able via `snapshot_mode()`
**AND** the snapshot MUST include the mode name and activation timestamp
**AND MUST NOT** expose internal `_mode_lock`

### US-015: Skill Deactivation State Machine
**WHEN** `_deactivate_workspace` is called
**THEN** the multi-step deactivation protocol MUST be atomic
**AND** reading `agent.active_mode` during deactivation MUST return a consistent state (either pre- or post-deactivation, never mid-transition)

### US-016: Skill Mode Expiry
**WHEN** a skill has a positive `ttl_seconds` and the TTL has elapsed
**THEN** `agent.is_mode_expired()` MUST return True
**AND** the expired skill MUST be deactivated before the next request is processed

### US-017: Ticket MAIN Tool Surface
**WHEN** ticket mode is active and the MAIN (orchestrator) agent issues a tool call
**THEN** the allowed tool set MUST be exactly `{read, glob, grep, search, task, webfetch, proxy_ticket}`
**AND** `proxy_ticket` MUST be read-only for MAIN — `actions_allowed = frozenset({"list_tasks", "get_task"})`. All write actions (`create_tasks`, `update_task`, `edit`, `set_parent`, `clear_parent`, `clear_blocked_by`, `set_priority`, `append_review_note`) MUST be denied with the table's `deny_message`, instructing the MAIN to delegate via `task` sub-agents
**AND MUST NOT** `proxy_work` be reachable from MAIN in ticket mode — `proxy_work` is `visible=False` in `TICKET_CAPABILITY_TABLE` for `role="main"` (entry-point denies with `{"error": "mode_inactive", ...}` for non-work callers; tool guard reinforces)
**AND MUST NOT** allow `write`, `edit`, or `bash` tool calls
**AND** ticket write actions MUST be delegated to ticket sub-agents via the `task` tool (ticket sub-agents hold FULL `proxy_ticket` per US-017c)
**AND** work operations (`start_work`, `stop_work`, etc.) MUST be delegated to a work-mode child via `task` with `subagent_mode="work"` (Ticket MAIN has no `proxy_work` and no path to `stop_work` directly — see US-093)
**AND MUST NOT** any `proxy_*` tool be reachable from default mode (no active skill) — see US-098 for the entry-point mode-gate contract
**Code:** `TICKET_CAPABILITY_TABLE` (`src/proxy/skills/skill_definition.py:559`) — Ticket MAIN rows; `MemoryToolGuard._check_work_mode_actions` (`src/proxy/tool_pipeline/tool_guard.py:442-509`) and `_check_ticket_mode_orchestrator` (`src/proxy/tool_pipeline/tool_guard.py:510`) reinforce at the guard layer.

**Historical (SUPERSEDED):** An earlier draft listed `{read, glob, grep, search, proxy_ticket, proxy_work}` and added `proxy_work` under "SPEC Mode Design". That set is SUPERSEDED — `proxy_work` is no longer reachable from ticket MAIN after PROXY_MODES_FIX_PLAN C3 removed the work→ticket TTL fallback and the Bug #3 stop_work exemption. The current MAIN set is the row above; ticket writes go to sub-agents, work goes to a work-mode child.

### US-017a: Ticket Mode Is Non-Interactive (Fire-and-Forget)
**WHEN** ticket mode is activated via `^^ticket <description>` (plan action)
**THEN** the agent MUST produce exactly ONE response containing task operations (via `proxy_ticket` tool calls)
**AND** ticket mode MUST be deactivated after that response completes (TTL expiry)
**AND MUST NOT** enter a multi-turn interactive conversation in ticket mode
**AND MUST NOT** re-activate ticket mode on subsequent messages from the same context

### US-017b: ^^ticket Direct Commands MUST NOT Hit the LLM
**WHEN** `^^ticket` is invoked with a direct subcommand (`list`, `show <hash>`, `done <hashes>`, `close <hashes>`, `reopen <hashes>`, `invalidate <hashes>`)
**THEN** `handle_direct()` MUST return a response dict immediately without calling the LLM
**AND MUST NOT** activate ticket mode or forward the request to an upstream backend
**AND** ONLY the `plan` action (everything else) triggers mode activation + LLM call
**AND** `handle_direct()` MUST return an `error_response` (HTTP 409) when the active workspace is `default` — extends the read-only default-workspace policy of the non-default-workspace read-only `proxy_ticket` contract to the `^^ticket`/`^^work` commands.

### US-017c: Ticket SUB Tool Surface (No Nesting, Full proxy_ticket)
**WHEN** a sub-agent is spawned inside ticket mode
**THEN** the sub-agent's allowed tool set MUST be exactly `{read, glob, grep, search, webfetch, proxy_ticket}`
**AND** `proxy_ticket` MUST be FULL access for ticket sub-agents — `actions_allowed = None` in `TICKET_CAPABILITY_TABLE` for `role="subagent"` (all 10 ticket actions allowed: `create_tasks`, `update_task`, `list_tasks`, `get_task`, `edit`, `set_parent`, `clear_parent`, `clear_blocked_by`, `set_priority`, `append_review_note`). The MAIN's read-only restriction does NOT propagate to sub-agents.
**AND MUST NOT** have access to `write`, `edit`, or `bash` tools
**AND MUST NOT** `proxy_work` be reachable from ticket sub-agents — `proxy_work` is `visible=False, allowed=False` in `TICKET_CAPABILITY_TABLE` for `role="subagent"` with `deny_message="Ticket sub-agents cannot call proxy_work."`. Ticket sub-agents MUST NOT call `start_work` / `stop_work`; those are work-mode-only (US-093).
**AND MUST NOT** sub-agents be allowed to spawn sub-sub-agents — `task` and `question` are blocked via `TICKET_CAPABILITY_TABLE` (`visible=False, allowed=False` for `role="subagent"`) and via `blocked_tools_for_subagent = frozenset({"question", "task"})` on `TicketSkill`
**Code:** `TICKET_CAPABILITY_TABLE` (`src/proxy/skills/skill_definition.py:559`) — Ticket SUB rows; `TicketSkill.blocked_tools_for_subagent` (`src/skills/ticket_skill.py:59`).

**Historical (SUPERSEDED):** An earlier draft inherited the ticket MAIN set including `proxy_work` into ticket sub-agents. That inheritance is SUPERSEDED — `proxy_work` is denied to ticket sub-agents by the capability table and the entry-point mode gate (`execute_proxy_work` at `src/memory/tools/proxy_work.py:73-82` rejects non-work callers with `{"error": "mode_inactive", ...}`).

### US-017d: Agents MUST Execute in Task Mode, Not Meta-Plan
**WHEN** the agent is in ticket/task mode and the user says "add tasks for...", "close the tasks that...", or any imperative task instruction
**THEN** the agent MUST immediately issue `proxy_ticket` tool calls to create, update, or complete the tasks
**AND MUST NOT** ask "should I execute this?", "would you like me to proceed?", or produce a plan-without-execution
**AND MUST NOT** meta-plan (describe what it would do) instead of doing it

### US-017e: Context-Aware Task Decomposition from Conversation
**WHEN** the user says `^^ticket turn what we just discussed into tasks` or any context-referencing instruction (e.g., "turn this document into tasks", "break down what we reviewed")
**THEN** the agent MUST read the conversation history, decompose the discussed work into concrete tasks, and immediately call `proxy_ticket(action='create_tasks', ...)` to create them
**AND MUST NOT** return a proposal or list of suggested tasks without executing
**AND MUST NOT** ask "should I create these tasks?" or "here's what I'd suggest"

### US-017f: Deferred Execution After Planning Conversation
**WHEN** the user has had a multi-turn conversation about tasks and then says `^^ticket go now`, `^^ticket do it`, `^^ticket execute`, or any imperative trigger
**THEN** the agent MUST immediately execute all task operations discussed (create, update, close) via `proxy_ticket` tool calls in a single response
**AND MUST NOT** return another plan or ask for confirmation
**AND MUST NOT** re-describe what it's about to do before doing it

### US-017g: Discover-Then-Execute with Vague Criteria
**WHEN** the user says `^^ticket close everything related to X`, `^^ticket invalidate the old Y tasks`, or any instruction combining a mutation verb with discovery criteria
**THEN** the agent MUST first discover matching tasks (via `proxy_ticket(action='list_tasks')` or conversation context), then immediately execute the mutation (via `proxy_ticket(action='update_task', ...)`) — all in the same response
**AND MUST NOT** stop after listing/discovering — discover and execute MUST happen in one turn
**AND MUST NOT** ask "I found N tasks matching your criteria, should I close them?"

### US-017h: Audit and Flag — Analysis With Action
**WHEN** the user says `^^ticket look at what's been done this week and flag anything that slipped`, `^^ticket give me analysis on stale tasks`, or any analysis request
**THEN** the agent MUST analyze the task state, identify issues (stale, blocked, orphaned), and if actionable issues are found, immediately execute corrective operations (invalidate stale tasks, unblock where possible)
**AND MUST NOT** return only a report without taking action on clear issues
**AND** the analysis text MUST accompany the tool calls, not replace them

### US-017i: Imperative with Time-Based or Emotional Criteria
**WHEN** the user says `^^ticket just kill anything that's been pending for more than 2 weeks`, `^^ticket close everything older than a month`, or any imperative with time/vague criteria (including frustrated/emotional language)
**THEN** the agent MUST discover tasks matching the criteria and immediately invalidate/close them via `proxy_ticket` tool calls
**AND MUST NOT** ask for confirmation before executing
**AND MUST NOT** return a list of "tasks that would be affected" without executing
**AND** emotional/frustrated language MUST NOT change the execution behavior — the agent MUST execute the same way regardless of tone

### US-017j: ^^ticket/^^work Reject on the Default Workspace
**WHEN** `^^ticket` or `^^work` is invoked (any subcommand, including `plan`/`<hash>` activation) and the active workspace is `default`
**THEN** the proxy MUST reject with HTTP 409
**AND** the error message MUST guide the user to `^^ws create <name>`, `^^ws list`, or `^^ws help`
**AND** ticket/work mode MUST NOT activate on the default workspace
**NOTE:** extends the read-only default-workspace policy of the non-default-workspace read-only `proxy_ticket` contract (which governs the `proxy_ticket` tool) to the `^^ticket`/`^^work` commands.

### US-018: Workspace Tool Filter (SUPERSEDED by US-073)
**WHEN** workspace mode is active
**THEN** the allowed tool set MUST include `{read, glob, grep, search, webfetch, bash, edit, write, suggest, proxy_ticket, proxy_workspace, todowrite, ask, question, new_task}`
**AND** tools outside this set MUST be rejected
**SUPERSEDED:** Workspace mode is not a real mode (US-073). `^^workspace` subcommands are atomic one-shot local state mutations handled in `handle_direct()`; no LLM is invoked, no tool filter is applied. The `get_tool_filter` method and the `proxy_workspace` tool that this story was supposed to enable were dead code and have been removed.

---

## Workspace Skill Commands

### US-066: `^^workspace <name>` is a Local State Mutation, No LLM
**WHEN** the user invokes `^^workspace <name>` for a valid workspace name
**THEN** the active workspace MUST be set in `workspace_store` synchronously
**AND** a response MUST be returned immediately
**AND MUST NOT** invoke any upstream LLM backend
**AND MUST NOT** activate workspace mode
**AND MUST NOT** modify `agent.active_mode` or `agent.active_mode.skill_state`
**Rationale:** Workspace switching is a deterministic state mutation, not a planning task. The LLM round-trip adds latency (1-3s), costs tokens, and creates a useless failure mode.
**Code:** `src/skills/workspace.py:189-213` (`handle_direct`), `_handle_switch`.

### US-067: `^^workspace <name>` Validates Workspace Name Format
**WHEN** the user invokes `^^workspace <name>` where `<name>` is empty, longer than 64 chars, contains non-alphanumeric/non-hyphen characters, or is a reserved keyword (leave, default, recent, help, list, status, admin, system, all, create, forget)
**THEN** the command MUST return an error response with HTTP 400
**AND** the error message MUST identify the invalid input
**AND MUST NOT** mutate `workspace_store`
**AND MUST NOT** invoke the LLM
**Code:** `_is_valid_workspace_name` in `src/skills/workspace.py:40-46`.

### US-068: `^^workspace <name>` is Idempotent on Repeated Invocation
**WHEN** the user invokes `^^workspace <name>` for the currently active workspace
**THEN** the command MUST return a success response indicating the user is already in the named workspace
**AND MUST NOT** invoke the LLM
**Rationale:** Repeated invocations are common (user re-asserts intent). The no-op response distinguishes from a real switch.
**Code:** `_handle_switch` in `src/skills/workspace.py`.

### US-069: `^^workspace create <name>` Explicitly Creates a New Workspace
**WHEN** the user invokes `^^workspace create <name>` with a valid workspace name
**THEN** the workspace MUST be created in `workspace_store` (added to MRU, set as active)
**AND** the response MUST be returned immediately
**AND MUST NOT** invoke the LLM
**AND MUST NOT** apply the typo guard (explicit create bypasses it)
**Rationale:** Legitimate new workspaces must be creatable even if the name is close to an existing one. The user opts in to creation by using `create`.
**Code:** `_handle_create` in `src/skills/workspace.py`.

### US-070: `^^workspace forget <name>` Removes a Workspace from MRU
**WHEN** the user invokes `^^workspace forget <name>` with a valid workspace name
**THEN** `<name>` MUST be removed from the recent_workspaces list for that session
**AND** the active workspace MUST NOT be changed (use `^^workspace leave` to deactivate)
**AND** the response MUST be returned immediately
**AND MUST NOT** invoke the LLM
**WHEN** the user invokes `^^workspace forget <name>` where `<name>` is the currently active workspace
**THEN** the command MUST return an error response with HTTP 400
**Rationale:** Typos and stale workspaces accumulate in MRU. The user needs a way to clean them up without deactivating their current workspace. Forgetting the active workspace would create a "ghost" state (active but absent from MRU), so the command refuses.
**Code:** `workspace_store.forget` in `src/proxy/workspace_state.py`.

### US-071: `^^workspace <name>` Refuses Typos and Suggests Alternatives
**WHEN** the user invokes `^^workspace <name>` where `<name>` does NOT exactly match any existing workspace (in MRU or active) but is within `difflib.SequenceMatcher.ratio()` similarity (cutoff 0.7) of one or more existing workspaces
**THEN** the command MUST return a message identifying the close matches
**AND MUST NOT** mutate `workspace_store` or activate any workspace
**AND MUST NOT** invoke the LLM
**AND** the response MUST inform the user how to force creation (`^^workspace create <name>`)
**Rationale:** Dyslexic users and fast typists make typos. Auto-creating typos pollutes the MRU and cannot be undone without `forget`. The fuzzy match catches single-char typos, transpositions, double-char typos, and adjacent-key typos.
**Code:** `_handle_switch` in `src/skills/workspace.py`.

### US-072: `^^workspace recent`, `^^workspace list`, `^^workspace status`, `^^workspace help` Are Direct Commands
**WHEN** the user invokes `^^workspace <subcommand>` where `<subcommand>` is one of `recent`, `list`, `status`, `help`
**THEN** the command MUST be handled in `handle_direct` without activating workspace mode
**AND MUST NOT** invoke the LLM
**Code:** `src/skills/workspace.py:189-213` (`handle_direct`).

---

## Memory System & Injection

### US-019: Memory Type Gate
**WHEN** `filter_memories_for_context()` processes memories
**THEN** ONLY memories of type `fact` or `event` MUST pass through
**AND** memories of type `instruction` or `task` MUST be excluded from auto-injection

### US-020: Memory Confidence Gate
**WHEN** context budget exceeds 80% and a memory's confidence is below threshold
**THEN** `should_drop_memory()` MUST return True
**AND** the low-confidence memory MUST NOT be injected

### US-021: Memory Content Safety
**WHEN** memory content contains imperative verbs (`always`, `never`, `must`, `should`), tool names (`bash`), meta-instructions (`before responding`, `first call`), or credential patterns
**THEN** `filter_memory_content()` MUST flag or drop the memory
**AND MUST NOT** inject potentially dangerous content into LLM context

### US-022: Sub-Agent Sync Exclusion
**WHEN** a request is identified as a sub-agent (via canary detection)
**THEN** background SOTA sync MUST be skipped
**AND** the sub-agent's scratch work MUST NOT pollute the memory graph

---

## Tool Loop

### US-023: Tool Loop Message Cap
**WHEN** the tool interception loop runs
**THEN** the total message count MUST NOT exceed 69 messages
**AND** when the cap is reached, the loop MUST terminate and return the current state

### US-024: Tool Loop Orphan Cleanup
**WHEN** tool results exist without a matching assistant `tool_calls` message
**THEN** `_cleanup_orphaned_tool_results()` MUST remove those orphaned results
**AND** sending orphaned `tool_call_id` values to upstream MUST NOT occur (prevents 400 errors)

### US-025: Tool Loop Timeout Per Iteration
**WHEN** each iteration of the tool loop calls `call_upstream()`
**THEN** it MUST apply a `timeout_cap` for that single iteration
**AND** the overall tool loop MUST NOT hang indefinitely

---

## Context Optimization

### US-026: Context Pruning — System Prompt Preservation
**WHEN** `select_context()` prunes messages to fit token budget
**THEN** the system prompt at index 0 MUST be preserved
**AND** recent messages from the conversation tail MUST be appended
**AND MUST NOT** drop the system prompt to make room for history

### US-027: V2 Passthrough Threshold
**WHEN** total history is under 30% of the safe context budget
**THEN** the V2 pipeline MUST return full history unchanged
**AND MUST NOT** apply pruning or summarization

### US-028: Orphan Tool Result Sanitization
**WHEN** pruning removes an assistant message that contained `tool_calls`
**THEN** all `role: "tool"` messages referencing those tool_call_ids MUST also be removed
**AND MUST NOT** send orphaned tool results to upstream (prevents 400 errors)

---

## Grep Sandbox

### US-029: Sensitive Path Blocking
**WHEN** a `grep` tool call targets a path containing `.env`, `.pem`, `credential`, `secret`, `token`, `.password`, or `.git`
**THEN** the tool guard MUST reject the call
**AND** the rejection message MUST NOT reveal the full blocked path

### US-030: Workspace Path Resolution
**WHEN** a workspace-relative path is used in a tool call
**THEN** the path MUST be resolved against the workspace's base directory
**AND** path traversal (`../`) beyond the workspace root MUST be blocked

---

## Provider Compatibility

### US-031: DeepSeek reasoning_content Injection
**WHEN** the upstream backend is DeepSeek and an assistant message lacks `reasoning_content`
**THEN** an empty `reasoning_content` field MUST be injected
**AND** the request MUST NOT fail with a missing-field error
**Implementation:** `DeepSeekAdapter` in `proxy/pipeline/provider_compat.py`

### US-032: Kimi Temperature Clamping
**WHEN** the upstream backend is Kimi
**THEN** `temperature` MUST be clamped to [0, 1]
**AND** `reasoning_content` MUST be injected on `tool_calls` messages that lack it
**Implementation:** `KimiAdapter` in `proxy/pipeline/provider_compat.py`

### US-033: MiniMax Role Conversion
**WHEN** the upstream backend is MiniMax
**THEN** `role: "system"` MUST be converted to `role: "user"`
**AND** `temperature` MUST be clamped to (0, 1] (exclusive lower bound)
**Implementation:** `MiniMaxAdapter` in `proxy/pipeline/provider_compat.py`

---

## Tool-Call Normalization

### US-124: Tool-call names and argument names are reconciled to the client's schema
**GIVEN** the client sends a `tools[]` array defining the tool names (e.g. `read`) and each tool's argument schema (e.g. `filePath`)
**WHEN** the LLM emits a tool call with degraded, hallucinated, or differently-cased names (e.g. `read_file`, `file_path`, `File`, `old_string`)
**THEN** the proxy MUST return the tool call using only names present in the client's schema (`read`, `filePath`, `oldString`), via `canonicalize_tool_name` + `normalize_tool_arguments(valid_params=...)`
**AND** the resolver order MUST be: exact schema match → fuzzy (casing/separator) match → synonym alias (reconciled against the schema) → passthrough unchanged
**AND MUST NOT** hardcode a canonical casing as a target — opencode is camelCase (`filePath`), kilocode is snake_case (`file_path`); `param_mappings` targets are intermediate tokens fuzzy-resolved at runtime
**AND** a name that matches nothing in the schema MUST pass through unchanged (best-effort reconciliation, not a strict filter)
**Implementation:** `normalize_tool_arguments`, `canonicalize_tool_name`, `make_tool_calls(valid_params_by_tool=...)` in `proxy/tool_parsing/normalize.py`; `build_valid_params_by_tool` reads the client schema. See AGENTS.md "Tool-call argument normalization".

### US-125: Internally-executed memory tools use snake_case and never round-trip to the client
**GIVEN** a memory tool (`recall`, `search`, `save_memory`, …) that the proxy intercepts and executes locally
**WHEN** `normalize_tool_arguments` runs with `valid_params=None`
**THEN** it MUST use legacy snake_case canonicalization (these tools have no client schema and execute locally in snake_case)
**AND MUST NOT** emit these tool calls back to the client as `tool_calls` (they are executed and their results are fed back to the LLM)
**Implementation:** `normalize_tool_arguments(valid_params=None)` legacy path in `proxy/tool_parsing/normalize.py`; memory-tool interception in `proxy/tool_pipeline/tool_pipeline.py` + `memory/tool_loop.py`

---

## Post-Processing

### US-034: PostProcessStage Resilience
**WHEN** `PostProcessStage.execute()` processes multiple post-processing steps
**THEN** a failure in one step MUST NOT skip subsequent steps
**AND** each step MUST have independent error handling

### US-035: API Key Redaction in Error Responses
**WHEN** an error response is returned to the client
**THEN** `_sanitize_error_body()` MUST redact sensitive values by key name (`api_key`, `token`, `secret`, etc.), `Bearer` tokens, and `key=value` pairs in error bodies
**AND MUST NOT** leak upstream credentials

---

## Migrations

### US-036: Migration Idempotency
**WHEN** `MigrationRunner.run_pending()` runs a migration
**THEN** it MUST use MERGE-based concurrent worker protection
**AND** running the same migration twice MUST be safe (no duplicate data or constraint failures)

### US-037: Failed Migration Recovery
**WHEN** a migration fails
**THEN** its status MUST be set to `'failed'`
**AND** subsequent migration runs MUST skip failed migrations (not retry infinitely)
**AND** the proxy MUST still start and serve requests

---

## HTTP Client

### US-038: Upstream Retry — Transient Errors
**WHEN** `call_upstream()` receives a `ConnectionError`, `ChunkedEncodingError`, or `Timeout`
**THEN** it MUST retry up to 3 times with exponential backoff
**AND MUST NOT** retry on 4xx errors (client errors are not transient)

### US-039: Upstream Retry — 429 Handling
**WHEN** upstream returns HTTP 429 with a `Retry-After` header
**THEN** `call_upstream()` MUST honor the `Retry-After` value before retrying
**AND** if no `Retry-After` header, MUST use exponential backoff

### US-040: Endpoint Concurrency Semaphore
**WHEN** multiple concurrent requests target the same upstream endpoint
**THEN** a per-endpoint semaphore MUST limit concurrent connections
**AND MUST NOT** allow unbounded connection pool exhaustion

---

## State Module

### US-041: Lock Acquisition Order
**WHEN** multiple locks from `state.py` must be acquired
**THEN** they MUST be acquired in ascending lock number order
**AND** acquiring a lower-numbered lock while holding a higher-numbered one MUST NOT happen (prevents deadlocks)

### US-042: ChatHistoryCache LRU Tracking
**WHEN** `ChatHistoryCache.__getitem__` retrieves an entry
**THEN** it MUST call `_track_access` to update the LRU order
**AND** entries accessed via `__getitem__` MUST NOT be prematurely evicted

---

## Pipeline Error Handling

### US-043: UpstreamStage Preparation Failure — 502 Response
**WHEN** `prepare_upstream_request()` raises a non-`AuthConfigError` exception in UpstreamStage
**THEN** UpstreamStage MUST raise `StageAbort` with a 502 status and `upstream_error` type
**AND MUST NOT** return `None` (which would silently continue to PostProcessStage with empty data)

### US-044: Context Limit Exceeded — 413 Response
**WHEN** `enforce_context_limits()` raises `ValueError` in ContextOptimizeStage
**THEN** ContextOptimizeStage MUST raise `StageAbort` with a 413 status and `context_limit_exceeded` type
**AND** non-`ValueError` exceptions from `enforce_context_limits()` MUST be logged as warnings and treated as non-fatal

### US-044a: Malformed Chat Request — Structured 400 Envelope
**WHEN** `/v1/chat/completions` or `/v2/chat/completions` receives a structurally malformed body
**THEN** the proxy MUST return HTTP 400 with `{"error": {"message": "...", "type": "invalid_request_error", "param": "...", "code": "invalid_messages"}}`
**AND** the `code` MUST identify the validation class (`invalid_messages`, `invalid_stream`, `invalid_role`, `invalid_content`, `missing_tool_call_id`, etc.)
**AND MUST NOT** start SSE or call upstream, even when the malformed request includes `"stream": true`

### US-045: Neo4j Driver Singleton
**WHEN** multiple `QueryExecutor` instances are created with the same bolt URL
**THEN** exactly ONE `GraphDatabase.driver()` MUST be created (singleton pattern)
**AND MUST NOT** create a new driver per instantiation (prevents resource leaks)

---

## CLI Agent Requirements

### US-046: Clients are CLI Agents Running from Arbitrary Directories

**Context:** ChoriProx clients are CLI agents (Kilo, OpenCode, Cursor, etc.) that run from arbitrary directories on the user's machine. They are NOT web browsers or traditional HTTP clients.

**Requirement:** The proxy MUST NOT restrict clients based on:
- Working directory (clients can run from anywhere)
- File path access (clients access files outside the proxy's workspace)
- Origin IP (clients connect from localhost)

**Rationale:** CLI agents are trusted local tools invoked by the user. Restricting them by path or directory breaks legitimate workflows where the user runs an agent in a different project directory.

**Example:** User runs `kilo` in `/Users/alice/my-project/`. The agent connects to ChoriProx (running in `/Users/alice/ChoriProx/`) and needs to read/write files in `/Users/alice/my-project/`. This MUST work.

### US-047: /v1/models Endpoint Must Be Accessible Without Authentication

**Context:** CLI agents need to discover available models before making requests. The `/v1/models` endpoint provides this information.

**Requirement:** The `/v1/models` endpoint MUST be accessible without authentication. It MAY be rate-limited to prevent abuse.

**Rationale:** Model discovery is a prerequisite for making requests. Requiring authentication would create a chicken-and-egg problem where the client needs to know the models to authenticate, but needs to authenticate to know the models.

**Example:** Kilo starts up, calls `GET /v1/models` to get the list of available models, then uses one of those models in subsequent requests.

### US-048: Rate Limiting Must Not Block Legitimate CLI Agent Traffic

**Context:** CLI agents make bursts of requests during normal operation (e.g., reading multiple files, running multiple tools).

**Requirement:** Rate limits MUST be generous enough to accommodate legitimate CLI agent workflows:
- `/v1/chat/completions`: At least 100 requests/minute per client
- `/v1/models`, `/health`, `/metrics`: At least 10 requests/minute per client
- Tool calls: At least 50 calls/minute per session

**Rationale:** CLI agents are not abusive bots. They are tools performing legitimate work on behalf of the user. Overly restrictive rate limits break workflows.

**Example:** An agent reading 20 files in quick succession (to understand a codebase) should not be rate-limited.

---

## Task System

### US-049: (Reserved — Intentionally Skipped)
This story number was intentionally skipped and is reserved for future use.

### US-050: Fresh Task List for Agents
**WHEN** an agent enters ticket/task mode
**THEN** the agent MUST have access to a fresh list of tasks (via implicit list_tasks or cached data)
**AND MUST NOT** be required to perform an extra turn calling list_tasks before operating on tasks
**AND** the task list MUST be current as of the mode activation time
**Rationale:** Agents should be able to operate on tasks immediately upon entering task mode. When a user provides an explicit hash, the agent already has the fresh list (from this rule), so they can operate on it immediately without an extra turn.
**Implementation:** `_seed_task_context()` in `context.py:111-158` queries Neo4j and injects task list into system prompt during `apply_mode()`. Hash provenance tracker is pre-seeded via `register_seeded_hashes()`.

---

## Security & Authentication

### US-051: Authentication Fail-Closed (Except /v1/models)
**WHEN** PROXY_API_KEY is not set
**THEN** all endpoints EXCEPT /v1/models MUST return 500 (fail closed)
**AND** /v1/models MUST remain accessible without authentication (per US-047)
**AND** a CRITICAL-level log MUST be emitted warning about missing API key
**Rationale:** Security requires authentication by default. /v1/models is the only exception for model discovery.

### US-052: X-Forwarded-For Validation (Non-Localhost Only)
**WHEN** a request originates from a non-localhost IP and includes X-Forwarded-For header
**THEN** the proxy MUST validate the entire X-Forwarded-For chain against trusted proxies
**AND** localhost requests MUST remain exempt from X-Forwarded-For validation (per US-012)
**AND** rate limits MUST remain generous for CLI agents (per US-048)
**Rationale:** X-Forwarded-For spoofing is only relevant for non-localhost requests. Localhost CLI agents are trusted.

---

## VFS (Virtual Filesystem Layer)

### US-053: VFS Read Cache Hit — No Disk I/O
**WHEN** a `read` tool call targets a file in the VFS cache and the entry is not expired
**THEN** the proxy MUST return cached content without client round-trip
**AND MUST NOT** perform disk I/O for cache hits
**Code:** `proxy/tool_pipeline/tool_pipeline.py` (`_check_vfs_read_cache`)

### US-054: VFS Write-Through Cache Update
**WHEN** a `write` tool call is intercepted
**THEN** the VFS cache MUST be updated with the new content BEFORE the write is passed to the client
**AND** a rollback snapshot MUST be saved
**Code:** `proxy/vfs.py` (`_cache_file_write_through`)

### US-055: VFS Rollback on Write Failure
**WHEN** a write result indicates failure (detected by `_looks_like_failure` heuristic)
**THEN** the VFS cache MUST be rolled back to the pre-write state using the saved rollback snapshot
**Code:** `proxy/vfs.py` (`_rollback_snapshot`, `_looks_like_failure`), `proxy/pipeline/post_process.py` (`_scan_write_results`)

### US-056: VFS Blocklist Enforcement
**WHEN** a file path matches blocklist patterns (`.env`, `.pem`, `credential`, `secret`, `token`, `.git/`, `.password`)
**THEN** the file MUST NOT be cached by VFS
**AND** blocklist check MUST apply to both read caching and write-through
**Code:** `proxy/vfs.py`, `proxy/tool_pipeline/tool_pipeline.py`

### US-057: VFS Scope Guard
**WHEN** `PROXY_VFS_SCOPE` is `"subs"`
**THEN** VFS MUST be inactive for main agent requests
**WHEN** `PROXY_VFS_SCOPE` is `"main"`
**THEN** VFS MUST be inactive for sub-agent requests
**WHEN** `PROXY_VFS_SCOPE` is `"all"` (default)
**THEN** VFS MUST be active for both
**Code:** `proxy/vfs.py` (`_vfs_active_for_context`)

### US-058: VFS Inherited Context for Sub-Agents
**WHEN** a sub-agent request is processed and VFS is enabled
**THEN** the context MUST include an inherited context message listing the parent agent's recent file reads and writes
**AND** main agents MUST receive shared file state instead
**Code:** `proxy/vfs.py` (`_build_vfs_inherited_context`, `_build_vfs_context`), `proxy/context_v2.py`
**See also:** US-127 for main-agent placement and provider-bound prefix behavior.

### US-127: Main-Agent VFS Uses a Cache-Safe Pre-Live Tail

**WHEN** a main-agent request has renderable, in-scope VFS context and a
user-led live exchange
**THEN** completed retained conversation MUST precede the VFS block
**AND** the VFS block MUST appear immediately before the entire live
user-led exchange
**AND** the live exchange MUST include the latest consecutive-user run and
all following assistant tool calls and matching tool results
**AND** VFS MUST NOT split an `assistant(tool_calls) → tool result(s)` unit
**AND** VFS MUST remain distinct from current-user content
**AND** VFS appearance, mutation, expiration, or workspace switching MUST
preserve the complete provider-bound prefix before the VFS or former-VFS
boundary
**AND** stable VFS state MUST render byte-identically regardless of cache
age within TTL or LRU access order
**AND** expired VFS entries MUST disappear without changing completed
history or V2 boundary fields
**AND** main and inherited VFS builders MUST remain isolated and obey
`PROXY_VFS_SCOPE`
**AND** MiniMax MUST receive no system roles, no adjacent user roles, and
an exact assistant message with content `" "` where converted system-origin
VFS would otherwise adjoin the current user
**AND** non-MiniMax providers MUST retain main VFS as `role="system"` without
adding a VFS-only barrier
**AND** VFS MUST be omitted rather than trimming the active exchange,
splitting tool units, or exceeding the safe request budget.

**Spec:** See internal documentation for the main-agent VFS cache-safe tail contract.
**Code:** `src/proxy/context_v2.py`, `src/proxy/vfs.py`,
`src/proxy/pipeline/provider_compat.py`
**Tests:** `tests/unit/test_context_v2.py`,
`tests/unit/test_v2_barrier_case_insensitivity.py`,
`tests/unit/test_vfs_context.py`,
`tests/integration/test_vfs_cache_safe_tail.py`

---

## Streaming

### US-059: SSE Streaming Format
**WHEN** proxy re-streams upstream responses
**THEN** each chunk MUST be prefixed with `"data: "` and suffixed with `"\n\n"`
**AND** the final chunk MUST be `"data: [DONE]\n\n"`
**AND** the format MUST be OpenAI-compatible SSE
**Code:** `proxy/streaming.py` (`stream_response`, `make_sse_chunk`)

### US-060: Full-Lifecycle SSE Heartbeat
**WHEN** the proxy streams an SSE response (the client set `stream: true`)
**THEN** the proxy MUST emit a `: connected` SSE comment immediately on stream open, so the client knows the proxy accepted the request even if the upstream is still loading
**AND** while the completion thread is still running, the proxy MUST emit a `: keepalive` SSE comment every `streaming.heartbeat_interval` seconds (default 5.0s, configured in `config.py` under the `streaming` section) — covering preflight, upstream, tool loops, and post-processing; heartbeats MUST NOT stop just because upstream has returned data, because PostProcess may still be running
**AND** the final response MUST be emitted followed by exactly one `data: [DONE]\n\n` terminator
**AND** if the upstream or pipeline errors, the error MUST be surfaced as an SSE data event (not a JSON 5xx), per the canonical keepalive error path
**Implementation:** `_keepalive_stream` in `src/proxy/app.py` (called from the streaming path); `streaming.heartbeat_interval` in `src/config.py`

### US-060a: Malformed Upstream Completion

**WHEN** an upstream returns HTTP 200 but the body is not a valid OpenAI Chat Completion
**THEN** ChoriProx MUST NOT forward the malformed body to the client
**AND** a valid completion MUST contain a non-empty `choices` array whose first item and first item's `message` are objects
**AND** an explicit provider error MUST contain an `error` object and MUST use the existing error path
**AND** ChoriProx MAY retry a malformed response exactly once only when `usage.total_tokens` is absent or is integer zero
**AND** ChoriProx MUST NOT retry when `usage.total_tokens` is nonzero, negative, boolean, null, string, float, or otherwise present but untrustworthy
**AND** retry exhaustion or an unsafe retry MUST return `upstream_protocol_error` with code `invalid_chat_completion`
**AND** streaming failure MUST use HTTP 200 SSE with exactly one error data event and exactly one `[DONE]`
**AND** non-streaming failure MUST use HTTP 502
**AND** malformed responses MUST not run reasoning extraction, persistence, history synchronization, VFS mutation, cache metrics, tool interception, model assignment, or SOTA synchronization
**AND** skill lifecycle finalization MUST still classify the completed malformed turn as FAILED.

Implementation references:
- `proxy/openai_compat.py` — `chat_completion_shape_error`, `build_upstream_protocol_error` (validator + canonical error builder)
- `proxy/http_client.py` — unified `call_upstream()` semantic validation + at-most-one safe retry
- `proxy/pipeline/post_process.py` — `PostProcessStage` pre-method-list shape check + `StageAbort((body, 502))` + `finally` lifecycle finalization
- `proxy/app.py` — `_sse_from_response_data` final safety boundary (one error event + one `[DONE]`)

---

## Model Routing

### US-061: Model Route Resolution
**WHEN** a model alias is specified in the request
**THEN** proxy MUST resolve it via `proxy_models.json` to the correct backend URL, tool parser, and API key
**AND** unknown models MUST fall back to default backend
**Code:** `proxy/model_registry.py` (`resolve_model_route`)

---

## Admin Endpoints

### US-062: Health Endpoint
**WHEN** `/health` is called
**THEN** it MUST return 200 with Neo4j and SQLite availability status
**AND MUST** be accessible without authentication
**AND MUST** be rate-limited by `admin_rate_limiter`
**Code:** `proxy/admin.py` (`health`)

### US-063: Models Endpoint Without Auth
**WHEN** `/v1/models` or `/v2/models` is called
**THEN** it MUST return the list of registered models without requiring `PROXY_API_KEY` authentication
**AND MUST** remain accessible even when `PROXY_API_KEY` is not set (fail-closed exemption)
**Code:** `proxy/admin.py` (`proxy_models`)

---

## Agent Registry

### US-064: Agent Registry LRU Eviction
**WHEN** agent count exceeds `AGENT_REGISTRY_MAX_AGENTS` (default 100)
**THEN** the least recently used agent MUST be evicted
**AND** agents with active requests MUST NOT be evicted (request_scope protection)
**Code:** `proxy/agent/agent_registry.py` (`AgentRegistry`)

---

## Motivator

### US-065: Motivator Turn-Aware Injection
**WHEN** `PROXY_MOTIVATOR_ENABLED` is `"1"` and the request is from a sub-agent
**THEN** the MotivatorStage MUST inject a turn-aware reminder into the system prompt
**AND** the reminder intensity MUST scale with turn count relative to `PROXY_MOTIVATOR_MAX_TURNS`
**Code:** `proxy/pipeline/motivator_stage.py`

---

## Ticket-Driven Environment: Task Assignment

### US-073: start_work Claims Unassigned Task
**WHEN** an agent calls `start_work` on a task where `assigned_agent_id IS NULL`
**THEN** the task's `assigned_agent_id` MUST be set to the calling agent's `agent_id`
**AND** a `:WorkSession` MUST be created atomically with `status='active'`
**AND** the task's `status` MUST transition from `pending` to `in_progress` (if currently `pending`)
**AND** the operation MUST use `execute_transaction_no_retry`
**Code:** `src/proxy/tool_pipeline/actions/start_work.py`

### US-074: start_work Rejects When Different Agent Has Active Session
**WHEN** an agent calls `start_work` on a task with an active WorkSession from a different agent AND that session is less than 24h old
**THEN** the action MUST return `{"error": "task_locked", "active_agent": "<agent_id>"}`
**AND** no new WorkSession MUST be created
**Code:** `src/proxy/tool_pipeline/actions/start_work.py`

### US-075: Same Agent Can Start Multiple Sessions
**WHEN** an agent calls `start_work` on a task it already has an active session on
**THEN** a new WorkSession MUST be created (multiple work bursts allowed)
**AND** the existing session MUST NOT be closed
**Code:** `src/proxy/tool_pipeline/actions/start_work.py`

### US-076: Cypher WHERE Clause Enforces Assignment on MODIFY Actions
**WHEN** an agent attempts a MODIFY action (`edit`, `update_task`, `set_parent`, `clear_parent`, `clear_blocked_by`, `set_priority`, `start_work`, `stop_work`, `report_bug`, `handoff_task`) on a task where `assigned_agent_id` is set to a different agent
**THEN** the action's Cypher WHERE clause MUST reject the operation (returns empty result)
**AND** READ actions (`list_tasks`, `show`, `append_review_note`) MUST NOT be blocked
**AND** `validate_bug` MUST NOT be in this list — it enforces different-agent-than-reporter + not-reviewer + all-reviews-submitted (no Task node in MATCH)
**AND** `resolve_bug` MUST NOT be in this list — it is human-only (called via `^^ticket bug resolve`)
**Code:** Each action's Cypher (no Python gate — Cypher is authoritative)

---

## Ticket-Driven Environment: Work Sessions

### US-077: stop_work Closes Session + Updates Task Status
**WHEN** an agent calls `stop_work` with a `ws_uuid` on a task it is assigned to
**THEN** the specified WorkSession's `ended_at` MUST be set and `status` MUST be `completed`
**AND** if `new_status` is `complete` or `invalid`, the task's `status` and `completed_at` MUST be updated
**AND** if `git_commit_sha` is provided, it MUST be stored on the WorkSession
**Code:** `src/proxy/tool_pipeline/actions/stop_work.py`

### US-078: stop_work Disambiguates Multiple Sessions via ws_uuid
**WHEN** an agent has multiple active sessions on the same task and calls `stop_work`
**THEN** the `ws_uuid` parameter MUST identify which session to close
**AND** other active sessions MUST NOT be affected
**Code:** `src/proxy/tool_pipeline/actions/stop_work.py`

### US-079: stop_work Returns Error on No Active Session
**WHEN** an agent calls `stop_work` with no matching active session
**THEN** the action MUST return `{"error": "no_active_session"}`
**AND MUST NOT** silently succeed
**Code:** `src/proxy/tool_pipeline/actions/stop_work.py`

### US-080: Stale Session Lazy Cleanup in start_work
**WHEN** `start_work` is called on a task with an active session older than 24h
**THEN** the stale session MUST be set to `status='abandoned'` atomically
**AND** the new session MUST be created in the same transaction
**AND MUST NOT** require a scheduled watchdog job
**Code:** `src/proxy/tool_pipeline/actions/start_work.py` (stale session WHERE clause)

---

## Ticket-Driven Environment: Bug Reporting

### US-081: report_bug Creates Bug + FOUND_DURING Link
**WHEN** an agent calls `report_bug` during an active work session
**THEN** a `:Bug` node MUST be created with `status='open'`, `review_count=0`, `max_reviews=N`
**AND** the bug MUST be linked via `(Bug)-[:FOUND_DURING]->(WorkSession)`
**AND MUST NOT** block any task or auto-escalate priority
**AND** if no active session exists, MUST return `{"error": "no_active_session"}`
**Code:** `src/proxy/tool_pipeline/actions/report_bug.py`

### US-082: review_bug Merged Claim+Submit with Hardline N Limit
**WHEN** a reviewer calls `review_bug` on an open bug
**THEN** a `:Review` node MUST be created with `confidence_score`, `verdict`, `proposal`
**AND** `review_count` MUST be incremented atomically
**AND** if `review_count >= max_reviews`, the action MUST return an error (hardline N)
**AND** the same agent MUST NOT review twice (enforced via `NOT EXISTS`)
**AND** the reporter MUST NOT review their own bug (enforced via `reported_by_agent <> $caller_agent_id`)
**AND** concurrent reviews MUST be handled by Neo4j write-write conflict detection
**Code:** `src/proxy/tool_pipeline/actions/review_bug.py`

### US-083: validate_bug Requires Different Agent Than Reporter AND Reviewers
**WHEN** the verdict agent calls `validate_bug`
**THEN** the caller's `agent_id` MUST differ from `reported_by_agent` (enforced in Cypher)
**AND** the caller MUST NOT have submitted a review on this bug (enforced via `NOT EXISTS`)
**AND** `review_count` MUST equal `max_reviews` (all reviews submitted)
**Code:** `src/proxy/tool_pipeline/actions/validate_bug.py`

### US-084: resolve_bug Creates Task on Approval
**WHEN** a human calls `resolve_bug` with `resolution='approved'` on a validated bug
**THEN** a new `:Task` MUST be created with `name=bug.title`, `summary=bug.description`, `priority` mapped from severity
**AND** the bug MUST be linked via `(Bug)-[:RESOLVED_BY]->(Task)`
**AND** the bug's `status` MUST be set to `approved`
**Code:** `src/proxy/tool_pipeline/actions/resolve_bug.py`

### US-085: resolve_bug Dismisses Bug
**WHEN** a human calls `resolve_bug` with `resolution='dismissed'`
**THEN** the bug's `status` MUST be set to `dismissed`
**AND** `dismissed_at` and `dismissal_reason` MUST be set
**Code:** `src/proxy/tool_pipeline/actions/resolve_bug.py`

### US-086: Bug→Resolved When Task Completes
**WHEN** `update_task` sets a task to `status='complete'` and a Bug has `[:RESOLVED_BY]->(Task)`
**THEN** the Bug's `status` MUST be set to `resolved` atomically in the same transaction
**Code:** `src/memory/task_graph.py` (`update_task_status`, extended Cypher)

### US-087: Bug on Invalidated Task → Auto-Dismissed
**WHEN** `update_task` sets a task to `status='invalid'` and a Bug has `[:RESOLVED_BY]->(Task)`
**THEN** the Bug's `status` MUST be set to `dismissed` with `dismissal_reason='task_invalidated'`
**Code:** `src/memory/task_graph.py` (`update_task_status`, extended Cypher)

### US-088: report_bug Rate Limited
**WHEN** an agent calls `report_bug` more than 5 times in a single request
**THEN** subsequent calls MUST be rejected by `try_report_bug()` in `ToolCallTracker`
**Code:** `src/proxy/tool_pipeline/tool_call_tracker.py`

---

## Ticket-Driven Environment: Handoff

### US-089: handoff_task Closes Session + Reassigns
**WHEN** an agent calls `handoff_task` with `short_hash` and `to_agent_id`
**THEN** the source agent's active session MUST be closed with `status='transferred'` and `transfer_note`
**AND** the task's `assigned_agent_id` MUST be updated to `to_agent_id`
**AND** the target agent MUST be verified to exist in `agent_registry` (Python pre-check) before Cypher runs
**AND** if target not found, MUST return `{"error": "target_agent_not_found"}`
**Code:** `src/proxy/tool_pipeline/actions/handoff_task.py`

### US-090: Handoff History via Consecutive WorkSessions
**WHEN** a handoff occurs
**THEN** the old WorkSession MUST have `status='transferred'` and `transferred_from_agent` set on the new session
**AND** handoff history MUST be queryable via consecutive WorkSessions on the same Task with different `agent_id` values
**AND MUST NOT** create new node types (HandoffRecord, HandoffNote) or new relationships
**Code:** `src/proxy/tool_pipeline/actions/handoff_task.py`

---

## Ticket-Driven Environment: Action Protocol

### US-091: caller_agent_id Derived from session_id
**WHEN** any action is dispatched via `dispatch_action()`
**THEN** `caller_agent_id` MUST be resolved from `session_id` via `agent_registry.get_or_load(session_id).agent_id`
**AND** passed to the action as a keyword argument
**AND** if `get_or_load()` returns `None`, `dispatch_action()` MUST return `{"error": "agent_not_found"}`
**Code:** `src/proxy/tool_pipeline/action_registry.py`

---

## Ticket-Driven Environment: Work Mode

### US-092: ^^work Activation
**WHEN** a user or agent invokes `^^work <task_hash>`
**THEN** work mode MUST activate with TTL 300s
**AND** the task hash MUST be captured for system prompt anchoring
**AND** if ticket mode is currently active, it MUST be displaced
**AND** the agent's system prompt MUST include the task context (hash, name, priority, summary)
**Code:** `src/skills/work_skill.py`

### US-093: Work Lifecycle Is Owned by Work Mode
**WHEN** a work session is required for a task
**THEN** a work-mode agent MUST own the complete one-shot work lifecycle:
`start_work` → execution → `stop_work`. Only work mode can call `start_work` or `stop_work` — the `execute_proxy_work` entry-point mode gate at `src/memory/tools/proxy_work.py:33-82` allows ONLY `_mode_name == "work"` (and not expired). All other callers (Ticket MAIN, ticket sub-agents, default mode, workspace mode) receive `{"error": "mode_inactive", ...}`.
**AND** Ticket MAIN MUST delegate the work operation to a work-mode child using the `task` tool with `subagent_mode="work"`. Ticket MAIN has no `proxy_work` access and no path to call `stop_work` directly — see US-093/US-098 for the mode-gate contract.
**AND** Orphaned work sessions (where the original work-mode agent crashed or never reached `stop`) MUST be reclaimed by the 24h stale sweep in `work_sessions.py:cleanup_stale_sessions` (lazy, transaction-scoped — `start_work` ALSO runs the same sweep on the same task).
**AND MUST NOT** Ticket MAIN or ticket sub-agents ever call `start_work` / `stop_work` directly (those are work-mode-only entry points).
**AND MUST NOT** a work→ticket TTL fallback exist — the previous fallback machinery and Bug #3 stop_work exemption were DELETED in PROXY_MODES_FIX_PLAN C3.
**Code:** `src/memory/tools/proxy_work.py:33-82` (mode gate), `src/proxy/tool_pipeline/actions/start_work.py` + `stop_work.py` (the lifecycle actions themselves), `src/skills/work_skill.py` (work-mode activation).

### US-094: validate_bug on Non-Open Bug
**WHEN** `validate_bug` is called on a bug with `status` != `open` or `validated_at IS NOT NULL`
**THEN** the action MUST return `{"error": "bug_not_open"}` or `{"error": "already_validated"}`
**AND** the bug's status MUST NOT be changed
**Code:** `src/proxy/tool_pipeline/actions/validate_bug.py`

### US-095: Guard Result Shape — Denial String Wrapped to Tool Result
**WHEN** any guard check rejects a tool call
**THEN** `MemoryToolGuard.check()` MUST return a denial string (e.g. `"Work mode restricts proxy_ticket to read-only (list_tasks, get_task, append_review_note). For other operations, use proxy_work."`). It MUST NOT return a structured object such as `{"error": "work_mode_restricted", "action": "<action_name>", "message": "Use proxy_work or switch to ticket mode."}` — that envelope shape is obsolete.
**AND** `ToolPipeline.run()` MUST wrap the denial string into a tool-result envelope via `blocked_wrapped_execute_fn` (`src/proxy/tool_pipeline/tool_pipeline.py:516-524`). The wrapper returns `{"error": <denial_string>}` for guard-blocked calls — a `role=tool` soft-error payload that the LLM sees as a tool result and can recover from. Guard-blocked calls are filtered out before VFS write-through to prevent cache poisoning (M3-1).
**AND** The action-specific denial text originates from `CapabilityTable.check_action(tool, action, role)` (`src/proxy/skills/skill_definition.py:511-540`) — the table returns the entry's `deny_message` when the action is not in `actions_allowed`, or `None` if allowed. `MemoryToolGuard._check_work_mode_actions` (`src/proxy/tool_pipeline/tool_guard.py:442-509`) delegates to `check_action` when the active skill has a capability table; otherwise it falls back to a hardcoded deny string.
**AND MUST NOT** the original tool call be executed when the guard rejects it.

**Per-role capability table shape (the source of action denial):**

| Role | `proxy_ticket` allowed | `proxy_work` allowed |
|---|---|---|
| Ticket MAIN (`role="main"` in `TICKET_CAPABILITY_TABLE`) | `{list_tasks`, `get_task}` only | (not visible) |
| Ticket SUB (`role="subagent"` in `TICKET_CAPABILITY_TABLE`) | all 10 actions (`actions_allowed=None`) | (not visible) |
| Work MAIN (`role="main"` in `WORK_CAPABILITY_TABLE`) | `{list_tasks`, `get_task`, `append_review_note}` | all 7 actions (`actions_allowed=None`) |
| Work SUB (`role="subagent"` in `WORK_CAPABILITY_TABLE`) | `{list_tasks`, `get_task`, `append_review_note}` | `{report_bug`, `review_bug`, `validate_bug`, `resolve_bug`, `handoff_task}` (5 actions; `start_work`/`stop_work`/`complete_work` excluded) |

**For Ticket MAIN `proxy_ticket`:** allowed = `{list_tasks`, `get_task}`. Denied = all writes (`create_tasks`, `update_task`, `edit`, `set_parent`, `clear_parent`, `clear_blocked_by`, `set_priority`, `append_review_note`). Write actions MUST be delegated to ticket sub-agents via `task`.

**For Work mode `proxy_ticket`:** allowed = `{list_tasks`, `get_task`, `append_review_note}` (read-only + the one note-write). Denied = all 7 CRUD actions. The MAIN uses `proxy_work` for writes (work-mode writes do not go through `proxy_ticket`).

**Code:** `src/proxy/skills/skill_definition.py:559-624` (TICKET_CAPABILITY_TABLE + WORK_CAPABILITY_TABLE), `src/proxy/skills/skill_definition.py:511-540` (CapabilityTable.check_action), `src/proxy/tool_pipeline/tool_guard.py:442-509` (_check_work_mode_actions), `src/proxy/tool_pipeline/tool_pipeline.py:516-524` (blocked_wrapped_execute_fn wrapping the denial string into `{"error": ...}`).

### US-096: ^^ticket bug Subcommands
**WHEN** a user invokes `^^ticket bug list`, `^^ticket bug show <hash>`, or `^^ticket bug resolve <hash> approved|dismissed`
**THEN** the command MUST be handled in `handle_direct()` without invoking the LLM
**AND** `^^ticket bug list` MUST show bugs for the current workspace (default: `status=open`)
**AND** `^^ticket bug show <hash>` MUST show full bug details with reviews
**AND** `^^ticket bug resolve <hash> approved|dismissed` MUST call `resolve_bug` action
**Code:** `src/skills/ticket_skill.py`

### US-097: stop_work Does NOT Unassign
**WHEN** an agent calls `stop_work` on a task
**THEN** the task's `assigned_agent_id` MUST NOT be changed (stays assigned to the agent)
**AND** other agents MUST NOT be able to claim the task after stop_work
**AND** the agent MUST be able to call `start_work` again to resume
**Code:** `src/proxy/tool_pipeline/actions/stop_work.py`

### US-099: Workspace Mode is NOT a Real Mode
**WHEN** a user invokes any `^^workspace` subcommand (e.g. `^^workspace <name>`, `^^workspace list`, `^^workspace leave`)
**THEN** the command MUST be handled as an atomic one-shot local state mutation in `WorkspaceSkill.handle_direct()`
**AND** a response MUST be returned immediately
**AND** the LLM MUST NOT be invoked
**AND** no skill mode MUST be activated (workspace mode is not a real mode)
**AND** `agent.active_mode` MUST NOT be modified
**AND MUST NOT** `WorkspaceSkill.on_activate` be called (the method exists as a no-op stub for protocol compatibility only)
**AND MUST NOT** `WorkspaceSkill.get_tool_filter`, `get_system_prompt`, `get_canary`, `should_deactivate` be called in production
**AND MUST NOT** `proxy_workspace` be a registered tool (it is dead code; was scaffolded for an LLM-driven workspace mode that was never built)

**Rationale:** US-066 through US-072 cover `^^workspace` subcommands as deterministic one-shot state mutations. There is no LLM round-trip, no system prompt, no tool filter. The original code scaffolded a "workspace mode" (with `on_activate`, `get_tool_filter`, `proxy_workspace` tool) for a future feature that was never built. The scaffold was dead code reachable only via direct calls. This story locks in the contract: workspace mode is a one-shot local mutation, nothing more.

**Code:** `src/skills/workspace.py` (parse_subcommand, handle_direct, all `_handle_*` and `_show_*` methods). The removed dead code: `get_tool_filter`, `get_system_prompt`, `on_activate`, `on_deactivate`, `should_deactivate`, `get_canary`, `_deactivate_workspace`, and the entire `src/memory/tools/proxy_workspace.py` module.

### US-098: proxy_* Tools Are Mode-Gated at Entry Point
**WHEN** any agent (top-level or sub-agent) calls a `proxy_*` tool
**THEN** the tool's entry-point function MUST verify the agent's `active_mode` matches the tool's domain
**AND** the entry-point MUST return a denial envelope with the required-mode name in the message when the agent is not in the matching mode (the envelope code is `access_denied` for `proxy_ticket` non-default-workspace writes, and `mode_inactive` for `proxy_work` non-work callers)
**AND** the `^^*` command that activates the mode (e.g. `^^ticket`, `^^work`) MUST be the ONLY way to unlock the corresponding `proxy_*` tool
**AND** default mode (no active skill) MUST NOT reach any `proxy_*` action handler
**AND MUST NOT** any `proxy_*` tool's `execute_*()` function be reachable without the corresponding mode being active
**AND MUST NOT** the `tool_guard._check_mode_allowlist` be the only line of defense — the entry-point check is required as defense-in-depth

**Per-tool mode requirements:**
- `proxy_ticket` requires `active_mode.name ∈ {"ticket", "work"}` (work mode gets read-only enforcement via `MemoryToolGuard._check_work_mode_actions`). For non-default-workspace callers outside ticket/work mode, the entry point returns `{"error": "access_denied", ...}` with `list_tasks`/`get_task` still permitted (read-only fallback for non-default workspaces).
- `proxy_work` requires `active_mode.name == "work"` ONLY. Ticket MAIN and ticket sub-agents are denied with `{"error": "mode_inactive", ...}` from `execute_proxy_work` (`src/memory/tools/proxy_work.py:73-82`). Default mode and workspace mode are also denied. There is no legitimate path for ticket-mode MAIN or ticket sub-agents to call `proxy_work` (PROXY_MODES_FIX_PLAN C3 deleted the work→ticket TTL fallback and the Bug #3 stop_work exemption — see US-093).

**Note:** `proxy_workspace` was removed in the US-099 cleanup. There is no third proxy_* tool.

**Rationale:** The mode-gate at the tool entry point is the canonical authority for proxy_* access. `tool_guard._check_mode_allowlist` is the first line of defense; the entry-point check is the second. Both must agree. Without the entry-point check, a request that bypasses tool_guard (e.g. a future code path) would still expose proxy_* tools to default-mode callers.

**Code:** `src/memory/tools/proxy_ticket.py:110-200` (`execute_proxy_ticket` — ticket/work mode gate + non-default-workspace read-only fallback), `src/memory/tools/proxy_work.py:33-82` (`execute_proxy_work` — work-only mode gate).

---

## Workspace Model v5: Global Catalog (Replaces Per-Session US-066..US-072)

> **Supersedes:** US-066 through US-072 (per-session workspace model). The behavior of `^^workspace` subcommands at the UI level is preserved (still one-shot local mutations, no LLM), but the underlying data model changes from per-session (`session_id:workspace_name`) to global (`workspace_name` bare). Workspaces become a proxy-level object visible to all sessions, enabling multi-agent collaboration.

> **Spec:** See internal documentation for the workspace rework v5 contract.

### US-100: Workspace ID is a Bare Name (No Session Prefix)

**WHEN** any code in the proxy needs to represent a workspace identifier
**THEN** the `workspace_id` MUST be the bare workspace name (e.g. `"choriprox"`, `"luna"`, `"default"`)
**AND MUST NOT** contain a `:` separator
**AND MUST NOT** embed the session_id in the identifier
**AND** `make_workspace_id(workspace_name)` MUST return the bare name directly
**Rationale:** Workspaces are proxy-level objects shared by all sessions. Per-session scoping defeated the multi-agent collaboration use case (a task created by session A in "choriprox" was invisible to session B in the same workspace). The v4 compound-ID model is removed.
**Code:** `src/proxy/state.py:make_workspace_id`, `src/memory/task_graph.py:writes`, all 12 caller sites.

### US-101: Global Catalog of Workspaces

**WHEN** the user invokes `^^workspace list` from any session
**THEN** the response MUST list ALL `:Workspace` nodes that exist in Neo4j, sorted by `last_active_at DESC`
**AND MUST NOT** be limited to workspaces in the current session's MRU
**AND** the catalog MUST include the `default` workspace (always present)
**AND** if a workspace is in the global catalog but not in the current session's MRU, `^^workspace <name>` MUST still be able to switch to it
**Rationale:** Anyone on the proxy with proper access should be able to see all available workspaces. Per-session MRU is a UX convenience for "what have I been working on" but is not the source of truth for "what exists."
**Code:** `workspace_store.get_global_catalog()` returns the list, populated lazily by `recover_workspaces()` and invalidated on new `activate()` for a previously-unknown name.

### US-102: Task Visibility is Per-Workspace, Not Per-Session

**WHEN** a session activates workspace `X` (regardless of which session activated it originally)
**THEN** `^^ticket` MUST return all tasks with `:Task.workspace_id = "X"`
**AND MUST NOT** filter by `t.session_id` (any session's tasks in that workspace are visible)
**AND** any session that activates `X` sees the same task list
**Rationale:** Multi-agent collaboration requires all agents in a workspace to see the same work. If user A and user B are both in workspace "choriprox", they see the same task list — this is the feature's core value.
**Code:** All `t.workspace_id = $ws` queries in `task_graph_queries.py`, `task_graph.py`, `task_graph_batch.py`, `work_sessions.py`, `bug_tracker.py`.

### US-103: Migration 010 Strips Prefixes from Existing Data

**WHEN** migration 010 (`StripWorkspacePrefix`) runs at startup
**THEN** every `:Task.workspace_id` and `:Bug.workspace_id` containing a `:` MUST be replaced with the substring after the last `:`
**AND** the migration MUST be paginated (5K tasks per transaction) to bound memory
**AND** the migration MUST be idempotent (re-running on already-stripped data is a no-op)
**AND** the migration MUST NOT touch `:Workspace` nodes (the data migration is separate from the schema migration)
**Rationale:** The 291 existing tasks at `8d959d6d6f71ba93:choriprox` must become `choriprox` for the new model to be consistent. Without this migration, the new code's `t.workspace_id = $bare_ws` queries return zero rows for old compound-keyed tasks.
**Code:** `src/memory/migrations/010_strip_workspace_prefix.py` (NEW), registered in `src/memory/sync.py:init_db()`.

### US-104: Migration 010 Ordering — Code First, Then Restart

**WHEN** deploying the workspace model v5
**THEN** the new code MUST be deployed and the proxy restarted BEFORE the migration 010 runs
**AND** the migration runs automatically at startup as part of `init_db()`
**AND MUST NOT** be run manually (e.g. via `cypher-shell`) before the code deploy — running migration first with the old code still in place results in PRODUCTION DATA LOSS OF VISIBILITY (old code looks for compound IDs, finds none)
**Rationale:** The old code's `make_workspace_id` returns compound IDs. The new code returns bare names. If migration 010 runs first (data → bare), then the old code's queries with compound `workspace_id` values match zero tasks. This is a one-way data-visibility loss that requires manual re-prefixing to recover. The deployment runbook documents this order.
**Code:** See internal documentation §6 (Deployment Runbook) of the workspace rework v5 contract.

### US-105: Old Compound Workspace Nodes Kept as Soft Backup

**WHEN** the new code runs against existing data
**THEN** the old `:Workspace` nodes (keyed by compound IDs like `8d959d6d6f71ba93:choriprox`) MUST remain in Neo4j
**AND MUST NOT** be deleted by any code path automatically
**AND** they serve as provenance records (proving which session originally created each workspace)
**AND** they enable rollback: if the new code is reverted, the old `:Workspace` nodes provide the original session mapping for re-prefixing tasks
**AND** manual cleanup (`MATCH (w:Workspace) WHERE w.workspace_id CONTAINS ':' DELETE w`) is safe to run after 30+ days once the new model is confirmed stable
**Rationale:** The new `:Workspace` nodes (bare-named) coexist with the old compound-named ones. The old nodes are inert (new code never queries them) but preserve the data lineage. Deleting them is a separate decision that the user makes explicitly.
**Code:** No code touches old compound nodes. Documented in the internal workspace rework v5 contract §6.

### US-106: Per-Session Last-Used Workspace Tracked via `:LAST_USED_WORKSPACE`

**WHEN** a session activates a workspace (via `^^workspace <name>` or `^^workspace create <name>`)
**THEN** a `:Session` node MUST be created (if not exists) for that session_id
**AND** a `[:LAST_USED_WORKSPACE]` relationship MUST be created from the `:Session` to the `:Workspace {name: <name>}`
**AND** the relationship's `last_used_at` property MUST be set to `datetime()`
**AND** on subsequent activates, the relationship's `last_used_at` MUST be updated
**Rationale:** Per-session "what was I last working in?" is preserved across proxy restarts. Without this relationship, a cold session after restart would only see the global catalog and default to `default`. With it, recovery finds the session's last workspace and restores it.
**Code:** `src/memory/workspace_persistence.py:_persist_workspace` (new closure). New Cypher: `MERGE (s:Session {id: $sid}) MERGE (s)-[r:LAST_USED_WORKSPACE]->(w) SET r.last_used_at = datetime()`.

### US-107: Recovery on Cold Start Restores Per-Session Last-Used

**WHEN** a cold session (no in-memory `_active` entry) makes its first `get_active()` call after a proxy restart
**THEN** the recovery function MUST query Neo4j for the session's `[:LAST_USED_WORKSPACE]` relationship
**AND** if the relationship exists, `_active[session_id]` MUST be set to that workspace name
**AND** `_recovered_sessions` MUST be updated to mark the session as recovered
**AND** if the relationship does not exist, `_active[session_id]` MUST remain `default`
**AND** in either case, `_catalog_cache` MUST be populated with all `:Workspace` names (for `^^workspace list` performance)
**Rationale:** The catalog is global, but each session's "where am I right now?" is per-session. Recovery is "load catalog + restore per-session last-used". This combines the new global model with the existing per-session UX.
**Code:** `src/memory/workspace_persistence.py:recover_workspaces` (new). `src/proxy/workspace_state.py:get_active` (recovery gate).

### US-108: Typo Guard Checks Global Catalog

**WHEN** the user invokes `^^workspace <name>` where `<name>` does NOT exactly match any workspace
**THEN** the typo guard MUST fuzzy-match against ALL `:Workspace` nodes in the global catalog
**AND MUST NOT** fuzzy-match only against the current session's MRU
**AND** close matches within `difflib.SequenceMatcher.ratio()` cutoff (0.7) MUST be suggested to the user
**AND** the user can bypass the guard with `^^workspace create <name>`
**Rationale:** A user from a different team trying to type "choriprox" benefits from the typo guard even if they have never used the workspace. The catalog-based guard is more inclusive than the per-session MRU guard.
**Code:** `src/skills/workspace.py:_handle_switch` — uses `workspace_store.get_global_catalog()`.

### US-109: Workspace Creation is One-Shot Global, Not Per-Session

**WHEN** the user invokes `^^workspace create <name>` from any session
**THEN** a single `:Workspace {name: <name>}` node MUST be created in Neo4j (via MERGE — idempotent)
**AND** the caller's session MUST be activated into that workspace (sets `_active[session_id] = <name>`)
**AND** the workspace MUST be visible in `^^workspace list` for ALL sessions immediately (after the next recovery cycle, or immediately via the catalog invalidation)
**AND** the workspace MUST be persisted to Neo4j with `last_active_at = datetime()`
**AND MUST NOT** create a per-session compound-ID workspace
**Rationale:** Workspace creation is a global operation. Two sessions cannot both create a "private" copy of "choriprox" — they share. The user's mental model is "I'm creating a project space on the proxy," not "I'm creating a session-private variant."
**Code:** `src/proxy/workspace_state.py:activate()` (with the persist callback firing the new MERGE).

### US-110: Persist Callback is Retried on Every Call (Lazy Re-Wire)

**WHEN** the persist callback is invoked
**THEN** if `_persist` is currently `None` (Neo4j was down at module load), the dispatcher MUST re-attempt to acquire a callback before giving up
**AND** the dispatcher MUST catch all exceptions from the callback, log a WARNING, and continue
**AND MUST NOT** propagate exceptions to the caller (the in-memory state has already been set; persist is best-effort)
**Rationale:** Today, `_persist_cb` is `None` forever if Neo4j was down at module load. The v4 latent-bug audit flagged this. The new model retries on every activate call, so workspaces created while Neo4j is transiently down WILL eventually persist when Neo4j returns and a subsequent `^^workspace create` re-attempts the wire.
**Code:** `src/proxy/workspace_state.py:activate()` — wraps `_persist` with a lazy rewire attempt.

### US-111: Cleanup Evicts Per-Session State, NOT Global Workspaces

**WHEN** a session is evicted (TTL exceeded, chat-history LRU eviction, manual pop)
**THEN** the cleanup MUST remove the per-session in-memory state (`_active`, `_recent`, `_last_access`, `_recovered_sessions`)
**AND** the cleanup MUST remove the `:Session` node and `[:LAST_USED_WORKSPACE]` edge for that session
**AND MUST NOT** delete `:Workspace` nodes (they are global and may be used by other sessions)
**AND MUST NOT** delete tasks or bugs in any workspace
**Rationale:** Workspaces are global, so eviction of one session must not affect other sessions' access to the same workspace. The v4 model deleted `:Workspace` nodes on cleanup, which would orphan tasks mid-deploy. The v5 model only cleans per-session state.
**Code:** `cleanup_neo4j_sessions` (renamed from `cleanup_neo4j_workspaces`) deletes `:Session` + edge. `:Workspace` nodes untouched.

### US-112: SESSION_TTL Bumped to 8 Hours for Per-Session In-Memory State

**WHEN** a session is idle for more than 8 hours
**THEN** its in-memory `_active`, `_recent`, `_last_access` entries MUST be cleared
**AND** the `:Session` node and `[:LAST_USED_WORKSPACE]` edge MUST be removed
**AND** the `:Workspace` nodes MUST NOT be deleted (global, outlive any single session)
**AND** a subsequent `get_active()` for that session after idle period MUST trigger recovery, which finds the `:Session` via the relationship (if not yet TTL'd) or returns `default`
**Rationale:** 1-hour TTL (v4) was too aggressive. A user taking a 2-hour lunch break would lose all their workspace context. 8 hours covers a full workday. The `:Workspace` nodes are now global, so there's no storage pressure to evict them.
**Code:** `SESSION_TTL = 28800` in `src/proxy/state.py` (and propagated to `session.py`, `app.py`, `qwen_proxy.py`).

### US-113: Sub-Agent Inherits Parent's Workspace via Canary

**WHEN** a sub-agent request is processed and the canary contains `[[__SUBAGENT__:WORKSPACE:<name>:sig:ts]]`
**THEN** the sub-agent's `ctx.workspace_name` MUST be set to `<name>` (bare)
**AND** the sub-agent's first `get_active()` MUST seed `_active[subagent_session_id] = <name>` from the canary hint
**AND** the sub-agent's `^^ticket` MUST see tasks from the parent's workspace (`<name>`)
**AND MUST NOT** require the sub-agent to first invoke `^^workspace use <name>` (the canary IS the activation)
**Rationale:** The canary flow already extracts bare workspace names (the v4 design was always bare-name compatible; the per-session scoping was only in the persist path). Sub-agents get the parent's workspace without needing a second command.
**Code:** `src/proxy/canary.py:170-181` (already bare). New: `src/proxy/workspace_state.py:get_active()` accepts a `workspace_hint` from canary.

### US-114: Cross-Session Task Sharing is the Default

**WHEN** two sessions both activate the same workspace
**THEN** both sessions MUST see the same task list via `^^ticket`
**AND** if session A creates a task in workspace "X", session B (also in "X") MUST see it immediately
**AND** if session A sets a task to "in_progress", session B's `^^ticket` MUST show the new status
**AND** the parent-child relationship of tasks (HAS_PARENT, BLOCKED_BY) is preserved across sessions
**AND MUST NOT** task queries filter by `t.session_id` (session filtering is gone in v5)
**Rationale:** Multi-agent collaboration is the headline feature. Two humans on different machines, each running an agent in workspace "choriprox", see the same task list and can hand off work via the `proxy_ticket` handoff action. The v4 model made this impossible because tasks were session-scoped.
**Code:** All `t.workspace_id = $ws` queries in `task_graph_queries.py`, `task_graph.py`, `task_graph_batch.py`. Verified by `tests/integration/test_workspace_sharing.py`.

### US-115: Cleanup Does NOT Delete Last-Used Edge for Other Sessions

**WHEN** session S1 is evicted and the cleanup runs
**THEN** only S1's `:Session {id: S1}` and its `[:LAST_USED_WORKSPACE]` edge MUST be deleted
**AND MUST NOT** touch S2's `:Session {id: S2}` or its edge
**AND** `cleanup_neo4j_sessions([S1])` MUST scope the Cypher to `$session_ids = [S1]` only
**AND** the cleanup MUST run as a single batched UNWIND with parameter `session_ids`
**Rationale:** The cleanup is per-session. Other sessions' `[:LAST_USED_WORKSPACE]` edges are not collateral damage. Batching via UNWIND bounds transaction size.
**Code:** `cleanup_neo4j_sessions(session_ids)` in `src/memory/workspace_persistence.py`. Cypher: `UNWIND $session_ids AS sid MATCH (s:Session {id: sid}) OPTIONAL MATCH (s)-[r:LAST_USED_WORKSPACE]->() DELETE r, s`.

### US-116: VFS Cache Key is Bare Workspace Name

**WHEN** the VFS (Virtual Filesystem Layer) caches a file read or write
**THEN** the cache key MUST be the bare workspace name (e.g. `default`, `choriprox`)
**AND MUST NOT** be the compound `f"{session_id}:{workspace_name}"` form
**AND** all sessions in the same workspace share the VFS cache for that workspace
**AND** `vfs_caches[workspace_name]` is the lookup path
**Rationale:** VFS caching is per-workspace, not per-session. Two agents in the same workspace reading the same file should benefit from the same cache. The v4 compound key was per-session, which made VFS isolation the wrong layer.
**Code:** `src/proxy/vfs.py` — 6 f-string sites: `workspace_id = f"{session_id}:default"` → `workspace_id = "default"`.

### US-117: Migration 009 (Rewritten) Backfills Bare-Name Workspaces

**WHEN** migration 009 (`WorkspaceBackfill`, rewritten) runs at startup
**THEN** for every distinct `:Task.workspace_id` value, a `:Workspace {name: <that value>, ...}` node MUST be created (via MERGE — idempotent)
**AND** the new `:Workspace` node MUST NOT have a `session_id` property
**AND** the new `:Workspace` node MUST have `created_by_session` set to the session_id of the first task seen (audit only, never used for filtering)
**AND** the migration MUST be paginated (5K distinct workspace_ids per page)
**Rationale:** After migration 010 strips prefixes, migration 009 creates the corresponding `:Workspace` nodes. The 291 existing tasks produce one distinct `workspace_id = "choriprox"`, so 1 new `:Workspace` node is created (plus any from prior `^^workspace create` calls — currently "luna" too).
**Code:** `src/memory/migrations/009_workspace_backfill.py` (rewritten). New Cypher: `MERGE (w:Workspace {workspace_id: ws_id}) ON CREATE SET w.name = ws_id, w.created_at = earliest_created, w.created_by_session = $first_session`.

### US-118: Cleanup Race is Eliminated (v4 Latent Bug Fixed)

**WHEN** an `activate()` fires the persist callback (which MERGEs the `:Workspace` node)
**AND** the cleanup path runs concurrently (e.g. cleanup_stale_workspaces or cleanup_old_sessions)
**THEN** the cleanup MUST NOT delete the `:Workspace` node that the persist just wrote
**BECAUSE** the cleanup only matches `:Session` nodes and `[:LAST_USED_WORKSPACE]` edges (per US-111 / US-115)
**AND** it MUST NOT include any `MATCH (w:Workspace ...)` clause
**Rationale:** The v4 cleanup race (cleanup deletes a workspace the persist just wrote) is structurally impossible in v5 because the cleanup targets `:Session` not `:Workspace`. The two write paths are now independent.
**Code:** `cleanup_neo4j_sessions` Cypher: matches `:Session` only, not `:Workspace`. Verified by `tests/unit/test_workspace_persistence.py:test_cleanup_does_not_touch_workspaces`.

### US-119: All 12 `make_workspace_id` Caller Sites Use Bare Name

**WHEN** any of the 12 documented caller sites of `make_workspace_id` is invoked
**THEN** the result MUST be the bare workspace name passed as the first argument
**AND MUST NOT** include any session prefix
**AND** the signature MUST be `make_workspace_id(workspace_name) -> workspace_name` (single argument)
**Rationale:** The v4 signature `make_workspace_id(session_id, workspace) -> "session_id:workspace"` is gone. All callers updated to use the bare name. No compound IDs in queries or comparisons.
**Code:** `src/proxy/state.py:make_workspace_id` (rewritten). Caller sites: `src/proxy/workspace_state.py:77,231`, `src/proxy/context.py:134`, `src/skills/workspace.py:404`, `src/skills/work_skill.py:119`, `src/skills/ticket_skill.py:222,308,414,474,501,571,664`, `src/memory/tools/proxy_work.py:118`, `src/memory/tools/proxy_ticket.py:142`.

### US-120: Cleanup is Batched in 1000-Session Chunks

**WHEN** `cleanup_neo4j_sessions(session_ids)` is called with N session IDs
**THEN** the function MUST split the input into chunks of 1000 IDs
**AND** each chunk MUST be processed in a separate Cypher transaction (`UNWIND $session_ids AS sid MATCH ...`)
**AND** partial failure in one chunk MUST NOT stop subsequent chunks
**AND** the function MUST log per-chunk success/failure
**Rationale:** 10K stale sessions in one transaction would be too large. Per-chunk bounded transactions keep Neo4j memory usage in check. Failure isolation prevents one bad batch from blocking the whole cleanup.
**Code:** `src/memory/workspace_persistence.py:cleanup_neo4j_sessions` (renamed). BATCH_SIZE = 1000.

### US-121: `^^workspace list` Catalog is Cached with TTL

**WHEN** `get_global_catalog()` is called
**THEN** the first call MUST query Neo4j (populate `_catalog_cache`)
**AND** subsequent calls within 5 minutes MUST return the cached list
**AND** the cache MUST be invalidated on any `activate()` for a workspace name not previously in the cache
**AND** a cold process (no cache) MUST have an empty catalog until the first `get_active()` triggers recovery which populates the cache
**Rationale:** `^^workspace list` is called frequently. Hitting Neo4j on every call is wasteful. A 5-minute TTL keeps the catalog fresh enough for the user's experience while bounding the query rate.
**Code:** `src/proxy/workspace_state.py:get_global_catalog()` reads from `_catalog_cache` or queries Neo4j on miss.

### US-122: Recovery Does NOT Stampede Neo4j

**WHEN** 1000 cold sessions all make their first `get_active()` call after a proxy restart
**THEN** each call MAY trigger recovery (no special debouncing)
**AND** the `try_recover()` method's lock acquisition serializes concurrent calls for the same session (only one thread wins)
**AND** the recovery Cypher query is bounded to `LIMIT $limit` (default 10) results, so each query is O(1) cost
**AND MUST NOT** issue unbounded queries against Neo4j
**Rationale:** The 1000 cold sessions is a natural pattern (proxy restart, all clients reconnect). Each session's first request triggers a recovery. The bound on `LIMIT` keeps each query cheap. The lock on `try_recover` prevents the same session from re-recovering.
**Code:** `src/memory/workspace_persistence.py:recover_workspaces` — `LIMIT $limit` (default 10, matches `MAX_RECENT_WORKSPACES`).

### US-123: Failed Migration 010 Is Recoverable by Deleting `_Migration` Marker

**WHEN** migration 010 fails partway through
**THEN** the `_Migration {version: 10, status: "failed"}` node is set
**AND** the proxy continues startup (per US-037)
**AND** to retry, the operator MUST run:
```cypher
MATCH (m:_Migration {version: 10}) DELETE m
```
**AND** then restart the proxy
**AND** the migration re-runs idempotently (the `WHERE CONTAINS ':'` filter skips already-stripped data)
**Rationale:** Failed migration recovery is a documented pattern. The `:_Migration` deletion + restart is the standard remediation across all 10 migrations.
**Code:** `src/memory/migrations/runner.py:run_pending()` (no change). Documented in the internal workspace rework v5 contract §6.

---

## Tests (linked to the above stories)

| Story | Test file | What it covers |
|---|---|---|
| US-100 | `tests/unit/test_make_workspace_id.py` | Bare-name return, colon rejection, empty rejection |
| US-101 | `tests/integration/test_workspace_global_catalog.py` | `get_global_catalog` returns Neo4j state; cached |
| US-102 | `tests/integration/test_workspace_sharing.py` | Two sessions in same workspace see same tasks |
| US-103 | `tests/unit/test_migration_010.py` | Strip prefix, idempotency, soft-deleted, NULL, already-bare |
| US-104 | `tests/integration/test_workspace_global_catalog.py` | Migration runs at startup; data after migration is bare |
| US-105 | `tests/integration/test_workspace_sharing.py` | Old compound `:Workspace` nodes coexist |
| US-106 | `tests/unit/test_workspace_persistence.py` | Persist writes `:Session` and `[:LAST_USED_WORKSPACE]` |
| US-107 | `tests/unit/test_workspace_state.py` | Cold recovery restores per-session last-used |
| US-108 | `tests/skills/test_workspace_skill.py` | Typo guard against global catalog |
| US-109 | `tests/integration/test_workspace_sharing.py` | `^^workspace create` creates bare-name `:Workspace` |
| US-110 | `tests/unit/test_workspace_persistence.py` | Lazy rewire on `_persist=None` |
| US-111 | `tests/unit/test_workspace_persistence.py` | `cleanup_neo4j_sessions` does NOT touch `:Workspace` |
| US-112 | `tests/unit/test_workspace_state.py` | `SESSION_TTL=28800` evicts in-memory but not `:Workspace` |
| US-113 | `tests/proxy/test_canary.py` | Sub-agent inherits parent's workspace via canary |
| US-114 | `tests/integration/test_workspace_sharing.py` | Cross-session task visibility, status updates |
| US-115 | `tests/unit/test_workspace_persistence.py` | Cleanup scoped to specific session_ids |
| US-116 | `tests/unit/test_vfs_cache_keys.py` | VFS cache key is bare workspace name |
| US-117 | `tests/unit/test_migration_009.py` | Rewritten backfill creates bare-name nodes |
| US-118 | `tests/unit/test_workspace_persistence.py` | Cleanup race eliminated (no Workspace deletion) |
| US-119 | `tests/unit/test_make_workspace_id.py` | All 12 caller sites use bare name |
| US-120 | `tests/unit/test_workspace_persistence.py` | Cleanup batching, partial failure isolation |
| US-121 | `tests/unit/test_workspace_state.py` | Catalog cache TTL, invalidation on new activate |
| US-122 | `tests/unit/test_workspace_persistence.py` | Recovery query has `LIMIT $limit` |
| US-123 | `tests/integration/test_workspace_global_catalog.py` | Failed migration recovery by `_Migration` deletion |
| US-124 | `tests/unit/test_db_recovery.py` | (reserved — tool-call reconciliation story) |

---

## Deprecation Notice

**US-066 through US-072 are SUPERSEDED** by US-100 through US-123. The `^^workspace` subcommand behavior at the UI level is preserved (still one-shot local mutations, no LLM), but the underlying data model and lookup behavior change. The behavioral contract is documented in the new stories. The old stories remain in this file for historical reference but are marked as superseded.

**US-099 (workspace mode is not a real mode) is PRESERVED** — it covers the skill handler, not the data model. No change.

---

## US-126: SQLite Auto-Heal on Corruption

When `data/memory.db` hits `sqlite3.DatabaseError: database disk image is malformed`
(the 2026-07-06 prod incident pattern — abrupt SIGKILL during WAL write),
the proxy **self-heals without operator intervention**:

- **Startup:** `init_db()` runs `ensure_db_integrity(path)` BEFORE opening
  the migration connection. On False → `recover_corrupt_db(path)` rebuilds
  a clean db from the `.dump` (intact DDL + non-corrupt-table INSERTs) +
  `.recover` artifacts (corrupt-table rows → `lost_and_found` → mapped back
  to original tables by `rootpgno` → `sqlite_master.rootpage`). On success,
  atomic `os.replace(<db>.healing, <db>)` swaps the file in place. On failure,
  init_db raises RuntimeError so the proxy refuses to start with a corrupt db.
- **Pool mid-request:** `db_pool.ConnectionPool._create_connection()` catches
  `sqlite3.DatabaseError("malformed" | "not a database")`, runs the same
  recovery under `proxy.state.recovery_lock` (#13), then retries the connect.
  Concurrent threads serialize through `state.recovery_lock` so only one runs
  the recovery; subsequent threads see `state.recovery_in_progress = True` and
  re-check integrity instead of re-recovering.
- **Prevention:** `_graceful_db_shutdown` atexit handler runs BEFORE the existing
  thread-pool shutdown: drains in-flight sync jobs, then `PRAGMA wal_checkpoint(TRUNCATE)`,
  then pool close. Prevents the WAL + abrupt-exit corruption pattern at the source.
- **Operator-visible artifacts:** every recovery creates `data/.corrupt_backup_<ts>/`
  with the original db, `.dump`, `.recover`, and a backup.log; never silently overwrites.

**Rationale:** The 2026-07-06 incident required 20 min of manual ops. Automation
removes that from the on-call path. The recovery procedure is the same one
performed manually that day, codified.

