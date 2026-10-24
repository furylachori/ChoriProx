# `src/AGENTS.md` — ChoriProx Source Tree

> Loaded when working in any `src/` file. For module-specific rules see the
> nested `AGENTS.md` files (`src/proxy/AGENTS.md`, `src/memory/AGENTS.md`,
> `src/clients/AGENTS.md`, `src/skills/AGENTS.md`, etc.).

## Project Identity

ChoriProx is a Python/Flask LLM proxy (port 6666) sitting between AI coding
clients (opencode, kilocode) and upstream LLM backends (StepFun, DeepSeek, Kimi,
MiniMax, MiMo). Every `POST /v1/chat/completions` request flows through 9
ordered pipeline stages, gets memory-injected context, has tools normalized and
guarded, and is re-streamed to the client with history sync + SOTA background
extraction.

**Stack:** Python ≥3.9, Flask, Neo4j, SQLite, requests.
**Tests:** see `tests/AGENTS.md`. Do not cite a specific test count — run
`python3 -m pytest --collect-only` for the current count (the README badge
drifts).

## Directory Map

| Path | Purpose | Sub-AGENTS.md |
|---|---|---|
| `src/qwen_proxy.py` | 57 LOC backward-compat shim + Flask boot | (lives in `src/AGENTS.md`) |
| `src/proxy_recovery_system.py` | Tool-schema optimizer (called from ProviderCompatStage) | `src/proxy/AGENTS.md` |
| `src/tool_optimizer.py` | Pure data: pre-built OpenAI tool schemas | `src/proxy/AGENTS.md` |
| `src/proxy/` | Flask shell, 9-stage pipeline, state, skills infrastructure, agent, canary, tool pipeline, tool parsing, dispatch, modes | `src/proxy/AGENTS.md` |
| `src/memory/` | SQLite façade + pure leaves, Neo4j graph layer (tasks/bugs/workspaces), SOTA sync, tool loop, 4 memory tools, 11 migrations | `src/memory/AGENTS.md` |
| `src/clients/` | ClientAdapter ABC + OpenCode/KiloCode adapters + ClientRegistry + ModelRegistry + parsers | `src/clients/AGENTS.md` |
| `src/skills/` | TicketSkill, WorkSkill, WorkspaceSkill (BaseSkill subclasses) | `src/skills/AGENTS.md` |
| `src/commands/` | `^^help`/`^^status`/`^^workspace` command dispatcher | `src/commands/AGENTS.md` |
| `src/enhancer/` | `^^imp` placeholder (DEAD CODE — `try_enhancer` always returns None) | `src/enhancer/AGENTS.md` |
| `src/devtools/` | `debug_session` pytest-free harness + `_reset_all_shared_state` | `src/devtools/AGENTS.md` |

## Quick Commands

```bash
# Run unit tests
PYTHONPATH=src python3 -m pytest tests/unit -n 4 --dist=loadfile -v

# Run integration tests
PYTHONPATH=src python3 -m pytest tests/integration -n 4 --dist=loadfile -v

# Run proxy-layer tests
PYTHONPATH=src python3 -m pytest tests/proxy -n 4 --dist=loadfile -v

# Run all (unit + integration + proxy + root)
make test

# Start the proxy
PYTHONPATH=src NEO4J_AUTH="neo4j:password" PROXY_SOTA_API_KEY="sk-..." \
  PROXY_SOTA_ENDPOINT="https://api.stepfun.ai/step_plan/v1" \
  PROXY_SOTA_MODEL="step-3.5-flash-2603" \
  PROXY_TEST_DB="data/memory.db" \
  python3 src/qwen_proxy.py
```

## Cross-Cutting Invariants

These apply to ALL modules. See the linked sub-AGENTS.md for module-specific rules.

### STATE: Always import `proxy.state` as a module

```python
# CORRECT — module-level mutation via proxy.state reference
import proxy.state as state
state.tool_cache_size += 1

# WRONG — creates LOCAL variable, module-level unchanged
from proxy.state import tool_cache_size
tool_cache_size += 1
```

This applies to ALL integer counters and mutable collections in `state.py`.

### STATE: Never create new Lock() or dict() for shared state

All shared locks and dicts are defined ONCE in `state.py`. Every other module imports references.

### AGENT: Use public API for mode access

```python
# CORRECT
mode_name = agent.get_mode_name()
mode_state = agent.snapshot_mode()
expired = agent.is_mode_expired()

# WRONG — never access _mode_lock directly
with agent._mode_lock:  # NEVER
```

### NEO4J: Use QueryExecutor, never raw driver

```python
from memory.query_executor import InfrastructureError, ConstraintViolation

# CORRECT
rows = self._executor.execute_read(query, params)
# For non-idempotent writes: pre-generate UUIDs
full_uuid = f"entity-{uuid.uuid4().hex}"
rows = self._executor.execute_write_no_retry(query, {...})
```

Always handle `InfrastructureError` (re-raise for retry), `ConstraintViolation`
(duplicate), and `QueryTimeoutError` explicitly. Never use bare `except:` or
`except Exception: pass`.

### NEO4J: Never use SQL-on-GRAPH anti-patterns

When writing or modifying ANY Cypher query:

- **NEVER** use `WHERE` clauses to simulate JOINs — use pattern matching (`MATCH (a)-[:REL]->(b)`)
- **NEVER** use `UNION` to simulate `OR` across labels — use `WHERE` with label checks or separate `OPTIONAL MATCH`
- **NEVER** flatten graph structures into rows to aggregate — use `collect()`, `count()`, or pattern aggregations
- **NEVER** treat properties as foreign keys — use relationships
- **NEVER** use `MERGE` on properties that should be relationship-connected nodes
- **NEVER** create intermediate "join table" nodes when a direct relationship suffices

### DESIGN: No backwards compatibility unless explicitly stated

Backwards compatibility is a luxury we cannot afford. When modifying schemas,
APIs, interfaces, or data structures:

- **NEVER** add deprecation wrappers, shims, or compatibility layers unless the human explicitly requests it
- **NEVER** keep old code paths "just in case" — delete them
- **NEVER** add versioned endpoints or fields for migration purposes unless explicitly asked
- If a change breaks existing behavior, that is expected — clean breaks over accumulated tech debt

### TASK GRAPH: Use relationships, never JSON blobs

```python
# CORRECT — query via relationship traversal
MATCH (t:Task {short_hash: $hash})-[:BLOCKED_BY]->(blocker:Task)

# WRONG — never use attributes JSON blob
attrs = json.loads(task.get("attributes", "{}"))
blocked_by = attrs.get("blocked_by", [])

# WRONG — never use label-checking pattern
MATCH (e:Entity) WHERE 'Task' IN labels(e)

# CORRECT — use direct label match
MATCH (t:Task)
```

- NEVER use `attributes` JSON blob for task data — use explicit properties and relationships
- NEVER use `MATCH (e:Entity) WHERE 'Task' IN labels(e)` — use `MATCH (t:Task)` directly
- NEVER store foreign key UUIDs as properties — use relationships
- All task mutations that modify relationships MUST use `execute_transaction_no_retry()` for atomicity
- Migration 004 transformed all existing data. Any code expecting the old schema will fail.

### RATE LIMITING: Check before acting

```python
from proxy.state import command_rate_limiter
result = command_rate_limiter.check(session_id)
if not result.allowed:
    return jsonify({"error": "rate_limited", "retry_after": int(result.retry_after)}), 429
```

The 8 rate limiters live in `proxy/state.py:131-143` and are auto-discovered
by `state.reset_all_rate_limiters()`. Do NOT tighten limits back to the
pre-`872964c` values (60/min) without first implementing per-account rate
limiting.

### DEPENDENCY RESOLUTION: Batch, never N+1

All dependency resolution (task graph, entity relations, workspace inheritance)
must use batch queries (UNWIND, WHERE IN). Never loop with per-item queries.

### LOGGING: Include request_id

All operational log messages should include `request_id` when available in
context. Use `logger.info("message", extra={"request_id": ctx.request_id})`.

### TOOL NORMALIZATION: client `tools[]` is the single source of truth

Tool-call names AND argument names returned to the client MUST be reconciled
against the schema the client sent in the incoming request
(`tools[].function.parameters.properties`). LLMs emit degraded/hallucinated
names (`read_file`, `file_path`, `File`); the proxy heals them back to the
client's canonical names via `canonicalize_tool_name` +
`normalize_tool_arguments(valid_params=...)`.

See `src/proxy/tool_parsing/AGENTS.md` for the full invariant set (this has
been broken 5+ times and is the #1 landmine in the codebase).

## Bug Policy

**There are NO pre-existing issues or failing tests — only YOUR issues.**

- **While coding a feature/fix:** If you discover a bug related to what you're working on, fix it in place. Do not defer, do not call it a "pre-existing issue." It's your issue now.
- **While reviewing or exploring:** If you discover bugs unrelated to the review/exploration, do NOT just note them and move on. Provide a concrete fix plan (code changes, file paths, affected lines). No drive-by observations — every bug you see, you own.

The codebase has no "someone else's problem." You are the engineer. Act like it.

**Severity classification.** Tier findings into three buckets before labeling them:

- **Functional failure** — wrong output, lost data, auth bypass, broken flow. Highest urgency.
- **Degraded contract** — clean error becomes noisy error, fallback engages, observability degrades. Real but lower urgency.
- **Operational drift** — docs/code/tests out of sync, dead code, future risk. Cleanup-tier.

"Critical" / "ship-blocker" applies only to bucket 1.

## Fix Flow (non-trivial fixes)

For any non-trivial fix — multi-file, behavior change, refactor, or new
feature — follow these five steps. Trivial fixes (typo, one-line, <20 lines,
single file) skip this flow.

### 1. Plan

Write a brief plan: what changes, which files, edge cases, test strategy.
Keep it under one screen. If you can't write the plan, you don't understand
the work yet.

### 2. Adversarial review of plan (2 sub-agents, parallel)

Dispatch two sub-agents in parallel to find blind spots, scope creep, design
issues, and missing edge cases. Apply the feedback or note why you rejected
it. Do not skip this step — the plan is cheap to change, the implementation
is not.

### 3. Implement

Make the changes. Follow the conventions in this file and the relevant
sub-AGENTS.md. Add tests for new behavior AND edge cases the plan identified.

### 4. Adversarial review of implementation (2 sub-agents, parallel)

Dispatch two sub-agents in parallel to verify correctness, convention
adherence, test coverage, and no regressions. Each sub-agent runs the relevant
test layer and reads the diff end-to-end.

### 5. Commit

Single commit. Message explains the **why** (the bug or feature), the **what**
(the change), and the **verification** (tests added, regressions checked).
Reference related issues or prior commits.

## Sub-Agent Delegation Rule

Main agents (those with access to the `task` tool) must preserve their context
window for planning, coordination, and synthesis. **All implementation work is
delegated to sub-agents.**

- Any non-trivial edit (more than a single word or line change) MUST be performed by a sub-agent.
- Read-only investigation (grep, glob, reading files) may be done by the main agent when it needs fast feedback to decide next steps.
- Sub-agents handle: file edits, running tests, git operations, curl commands, and multi-step implementation tasks.
- Main agents handle: planning, dispatching sub-agents, reviewing results, synthesizing reports, and asking clarifying questions.

The goal is to keep the main agent's context clean and focused on the big picture while sub-agents do the detailed work.

## Cross-Cutting Landmines (top 10)

| # | Landmine | Where | What NOT to do |
|---|---|---|---|
| 1 | Tool-call argument normalization case folding | `src/proxy/tool_parsing/normalize.py` | Never add a `.lower()` fallback that bypasses `valid_params`. See `src/proxy/tool_parsing/AGENTS.md`. |
| 2 | Lock ordering 15 positions | `src/proxy/state.py:294-315` | Acquire locks in ascending order; never hold #0 while waiting for #7. Violations caused 8+ deadlocks. Position #13 = `recovery_lock` (RLock, `memory.db_recovery`); position #14 = `dedup_lock` (DedupStore, CP4). |
| 3 | Canary parser single-source | `src/proxy/canary.py:691` (`parse_canary`) | Never write inline position parsing. Always route through `_locate_canary` + `_ingress_from_parse`. |
| 4 | Canary position-0 requirement | `src/proxy/canary.py:721` | A canary mid-message strips prefix text (data loss). |
| 5 | WORKSPACE canary sets `inherited_skill=None` | `src/proxy/canary.py:864-865` | NEVER set `inherited_skill` to any non-None string for WORKSPACE canaries. |
| 6 | Neo4j test no-wipe policy | `tests/integration/neo4j_harness.py` | The harness NEVER wipes a shared DB (the `I_AM_AN_IDIOT_ASK_A_HUMAN` flag was REMOVED). Tests require the Docker test container on `:7688`; never remove the pytest.skip markers or point tests at prod `:7687`. Commit `2753fbe` caused 3 data losses. |
| 7 | `safe_db_name` invariants | `tests/integration/neo4j_harness_utils.py` | NOT idempotent; hash RAW input (not sanitized). Commit `5dda43a` caused 3+ data losses. |
| 8 | `apply_provider_compat` has no feature flag | `src/proxy/pipeline/stage_modules/provider_compat_stage.py` | Don't add a feature flag. Disabling is a call-site comment, not a flag. |
| 9 | Pipeline singleton-sharing + per-request reset | `src/proxy/pipeline/stages.py:18-28` | Only `DispatchStage._features` is mutable instance state; `app.py` resets it per request. |
| 10 | Tautological tests — green ≠ correct | any new test | Revert-test every fix; grep `src/` for callers; concurrency bugs need `threading.Barrier`. |

## Investigation Rigor (3 disciplines)

When investigating a bug, three disciplines prevent mis-attribution:

1. **Trace to the swallow site before naming the culprit line.** If flagging an exception handler as a fail-open path, follow the call chain upstream. A bare `except Exception: pass` may be a symptom — the real culprit can be 2–3 frames deeper.
2. **Verify the failure mode is reachable before calling it a bug.** When claiming an `AttributeError` / `TypeError` / `KeyError`, check the type definition. A dataclass-declared field is always set by its constructor; the missing-attribute mode cannot occur.
3. **"Zero callers today" ≠ ship-blocker.** Severity is about what could go wrong in production today, not what an antipattern could cause if wired in.

## Behavioral Contracts

`USER-STORIES.md` has the USER DEFINED expectations for key actions/flows/responses inside the application. Read it when modifying or implementing behavior in a covered subsystem. It defines what the user expects vs what the code currently does. It is NOT a standing ruleset to carry in context at all times — it's a reference document for disambiguation and implementation guidance.

## Quick links

- Top 10 sub-AGENTS.md files (mapping of every section of the old monolithic AGENTS.md to its new home)
- `src/proxy/AGENTS.md` — proxy shell + state + lock hierarchy
- `src/proxy/pipeline/AGENTS.md` — 9-stage pipeline lifecycle
- `src/proxy/tool_pipeline/AGENTS.md` — ToolRegistry + ToolGuard + 17 actions
- `src/proxy/tool_parsing/AGENTS.md` — schema-driven argument normalization (THE LANDMINE)
- `src/proxy/skills/AGENTS.md` — SkillRegistry + TrustToken + custom JSON loader
- `src/proxy/agent/AGENTS.md` — Agent dataclass + AgentRegistry (LRU + in-flight)
- `src/proxy/canary.py` (single file, see module docstring) — HMAC-signed sub-agent markers
- `src/memory/AGENTS.md` — SQLite + Neo4j + SOTA + tool loop
- `src/memory/tools/AGENTS.md` — 4 memory tools + action dispatch
- `src/memory/migrations/AGENTS.md` — 11 Neo4j schema migrations
- `src/clients/AGENTS.md` — ClientAdapter + adapters + registries
- `src/skills/AGENTS.md` — TicketSkill + WorkSkill + WorkspaceSkill
- `src/commands/AGENTS.md` — `^^help`/`^^status`/`^^workspace`
- `src/enhancer/AGENTS.md` — `^^imp` (DEAD CODE)
- `src/devtools/AGENTS.md` — debug_session + `_reset_all_shared_state`

## Repository artifact policy

Reviewers MUST flag and request rework when a PR commits a non-code artifact without an explicit justification in that originating PR identifying the file and its required purpose. Passing tests or calling a file "evidence" does not waive this rule.

Generated evidence, logs, traces, screenshots, recordings, dumps, and reports belong in the originating PR or issue, NEVER in the repository. Store bulky artifacts in Google Drive and link them from that PR or issue. Do not commit those artifacts as documentation or fixtures to evade this policy.

Source code and tests belong in the repository. Any minimal fixture, documentation, or configuration genuinely required by the implementation must have an explicit file-and-purpose justification in the originating PR; retain only what is required. If that justification is absent, request rework rather than accepting the artifact.

This AGENTS.md policy update is explicitly requested by the repository owner.
