# ChoriProx — Agent Operating Context

## Project Identity

ChoriProx is a Python/Flask LLM proxy (**port 6666**) that sits between AI coding
clients (opencode, kilocode, any OpenAI-compatible client) and upstream LLM
backends (StepFun, DeepSeek, Kimi, MiniMax, MiMo). Every `POST /v1/chat/completions`
flows through a **9-stage pipeline** (`Cleanup → Dispatch → RouteResolution →
Security → ContextOptimize → Motivator → ProviderCompat → Upstream → PostProcess`).

**Stack:** Python ≥3.9, Flask, Neo4j, SQLite, `requests`.

**Authoritative deep-dives** (when the root doc can't fit it):
- [docs/_analysis/PIPELINE_MAP.md](./docs/_analysis/PIPELINE_MAP.md) — 76 KB request-flow reference
- [USER-STORIES.md](./USER-STORIES.md) — US-XXX behavioral contracts
- [README.md](./README.md) — quick start + features
- [docs/TEST_PATTERNS.md](./docs/TEST_PATTERNS.md) — test architecture

## Navigation — `AGENTS.md` tree

You are here. **Load the root for universal invariants; load a sub-AGENTS.md when working in that subtree.**

```
/AGENTS.md                                     # YOU ARE HERE — slim root + universal landmines
├── /src/proxy/AGENTS.md                       # Flask shell, state.py (locks/dicts/counters), singletons, canary, session tokens
│   ├── /src/proxy/pipeline/AGENTS.md          # 9-stage pipeline, DEFAULT_STAGES, stage_modules leaf rule
│   ├── /src/proxy/tool_pipeline/AGENTS.md     # ToolRegistry, ToolPipeline.run(), MemoryToolGuard, 17 actions
│   ├── /src/proxy/agent/AGENTS.md             # Agent dataclass, AgentRegistry LRU, mode lifecycle
│   ├── /src/proxy/skills/AGENTS.md            # BaseSkill, SkillDefinition, TrustToken, registry, custom-JSON loader
│   ├── /src/proxy/tool_parsing/AGENTS.md      # schema-driven normalize.py (THE landmine), parsers.py, ticket/work prompts
├── /src/memory/AGENTS.md                      # SQLite façade, Neo4j, SOTA sync, task graph
│   ├── /src/memory/tools/AGENTS.md            # the 4 memory tools (recall / search / proxy_ticket / proxy_work)
│   ├── /src/memory/migrations/AGENTS.md       # 11 numbered migrations + runner
├── /src/clients/AGENTS.md                     # opencode + kilocode adapters + ClientRegistry
├── /src/skills/AGENTS.md                      # TicketSkill, WorkSkill, WorkspaceSkill
├── /src/commands/AGENTS.md                    # ^^help, ^^status, ^^workspace parsers
├── /src/enhancer/AGENTS.md                    # ^^imp (DEAD CODE — see file)
├── /src/devtools/AGENTS.md                    # debug_session harness + _reset_all_shared_state
└── /tests/AGENTS.md                           # test layers, builders, mocks, wipe gate, fixtures
```

For the per-section sources, see the internal documentation.

## Quick Commands

```bash
# Run unit tests
PYTHONPATH=src python3 -m pytest tests/unit -n 4 --dist=loadfile -v

# Run integration tests
PYTHONPATH=src python3 -m pytest tests/integration -n 4 --dist=loadfile -v

# Run proxy-layer tests
make test-proxy
# or: PYTHONPATH=src python3 -m pytest tests/proxy -n 4 --dist=loadfile -v

# Run all tests (unit + integration + proxy + root)
make test

# Run coverage (90% clients / 63% overall gates)
make coverage

# Override worker count for memory-constrained machines
make test WORKERS=2
# or: PYTHONPATH=src python3 -m pytest tests/unit -n 2 --dist=loadfile -v

# Start the proxy
PYTHONPATH=src NEO4J_AUTH="neo4j:password" PROXY_SOTA_API_KEY="sk-..." \
  PROXY_SOTA_ENDPOINT="https://api.stepfun.ai/step_plan/v1" \
  PROXY_SOTA_MODEL="step-3.5-flash-2603" \
  PROXY_TEST_DB="data/memory.db" \
  python3 src/qwen_proxy.py

# Collect all tests (don't run)
python3 -m pytest --collect-only
```

Test counts: do not cite a specific number. Run `--collect-only` for the current count. Different handover docs claim ~1571, ~1720, ~2507, ~3953 — all stale.

## Bug Policy

**There are NO pre-existing issues or failing tests — only YOUR issues.**

- **While coding a feature/fix:** if you discover a bug related to what you're working on, fix it in place. Do not defer, do not call it a "pre-existing issue." It's your issue now.
- **While reviewing or exploring:** if you discover bugs unrelated to the review/exploration, do NOT just note them and move on. Provide a concrete fix plan (code changes, file paths, affected lines). No drive-by observations — every bug you see, you own.

The codebase has no "someone else's problem." You are the engineer. Act like it.

### Severity classification

Tier findings into three buckets **before** labeling them:

| Bucket | Definition | Urgency |
|---|---|---|
| **Functional failure** | Wrong output, lost data, auth bypass, broken flow | Highest — ship-blocker |
| **Degraded contract** | Clean error → noisy error, fallback engages, observability degrades | Real, lower urgency |
| **Operational drift** | Docs/code/tests out of sync, dead code, future risk | Cleanup-tier |

`critical` / `ship-blocker` applies **only to bucket 1**. A dead-code antipattern labeled critical is misclassified even if the antipattern could cause damage if wired in — bucket 3 by current state.

## Fix Flow (non-trivial fixes)

For any non-trivial fix — multi-file, behavior change, refactor, or new feature — follow these five steps. Trivial fixes (typo, one-line, <20 lines, single file) skip this flow.

### 1. Plan
Write a brief plan: what changes, which files, edge cases, test strategy. Keep it under one screen. If you can't write the plan, you don't understand the work yet.

### 2. Adversarial review of plan (2 sub-agents, parallel)
Dispatch two sub-agents in parallel to find blind spots, scope creep, design issues, and missing edge cases. Apply the feedback or note why you rejected it. **Do not skip this step** — the plan is cheap to change, the implementation is not.

### 3. Implement
Make the changes. Follow the conventions in the relevant sub-AGENTS.md (use `proxy.state` as a module, use public Agent API, query Neo4j with patterns not SQL, etc.). Add tests for new behavior **and** edge cases the plan identified.

### 4. Adversarial review of implementation (2 sub-agents, parallel)
Dispatch two sub-agents in parallel to verify correctness, convention adherence, test coverage, and no regressions. Each sub-agent runs the relevant test layer and reads the diff end-to-end.

### 5. Commit
Single commit. Message explains the **why** (the bug or feature), the **what** (the change), and the **verification** (tests added, regressions checked). Reference related issues or prior commits.

## Sub-Agent Delegation Rule

Main agents (those with access to the `task` tool) must preserve their context window for planning, coordination, and synthesis. **All implementation work is delegated to sub-agents.**

- Any non-trivial edit (more than a single word or line change) MUST be performed by a sub-agent.
- Read-only investigation (grep, glob, reading files) may be done by the main agent when it needs fast feedback to decide next steps.
- Sub-agents handle: file edits, running tests, git operations, curl commands, and multi-step implementation tasks.
- Main agents handle: planning, dispatching sub-agents, reviewing results, synthesizing reports, and asking clarifying questions.

**Cross-references:**
- **Bug Policy** (above): every bug you see, you own. If a sub-agent spots a bug during review, fix it in the same commit or split it out — no drive-by observations.
- **No backwards compatibility** (DESIGN rule below): never add deprecation wrappers, shims, or "V2" flags. The current state is the only state. If a change breaks existing behavior, that is expected — clean breaks over accumulated tech debt.

## Universal Code Conventions

These apply **everywhere** in the codebase. Per-area conventions live in the relevant sub-AGENTS.md.

### STATE: Always import `proxy.state` as a module

```python
# CORRECT — module-level mutation via proxy.state reference
import proxy.state as state
state.tool_cache_size += 1

# WRONG — creates LOCAL variable, module-level unchanged
from proxy.state import tool_cache_size
tool_cache_size += 1
```

Applies to ALL integer counters and mutable collections in `state.py`. See [src/proxy/AGENTS.md](./src/proxy/AGENTS.md) for the full counter list and singleton rules.

### STATE: Never create new `Lock()` or `dict()` for shared state

All shared locks and dicts are defined ONCE in `state.py`. Every other module imports references. Adding a new lock without a state.py position is a deadlock waiting to happen.

### Lock ordering (positions #0–#14)

If you acquire multiple locks, follow the numbering in `state.py` comments. Violations cause deadlocks (8+ adversarial rounds).

| # | Lock | Where | Type |
|---|---|---|---|
| #0 | `AgentRegistry._lock` | `proxy/agent/agent_registry.py` | coarsest — global |
| #1 | `context_config_lock` | `proxy/state.py` | RLock |
| #2 | `chat_histories_lock` | `state.py` via `chat_history.py` | RLock |
| #3 | `chat_summaries_lock` | `state.py` | |
| #4 | `tool_cache_lock` | `state.py` | |
| #5 | `request_counter_lock` | `state.py` | |
| #6 | `active_workspace_lock` | `state.py` | RLock |
| #7 | `Agent._mode_lock` | `proxy/agent/agent.py` | **per-instance** |
| #8 | `AgentTelemetry._lock` | `pipeline/agent_telemetry.py` | **per-instance** |
| #9 | `vfs_lock` | `state.py` | |
| #10 | `_chain_state_lock` | `state.py` | |
| #11 | `canary_verify_cache_lock` | `state.py` | |
| #12 | `log_overflow_count_lock` | `state.py` | leaf — never acquires others |
| #13 | `recovery_lock` | `proxy/state.py` | RLock — leaf, corruption-recovery serialization (used by `memory/db_recovery.py`) |
| #14 | `dedup_lock` | `proxy/state.py` | leaf — DedupStore keyed completion map (CP4) |

**Rule:** acquire in ascending order; never hold a lower-numbered lock while acquiring a higher one. `#7` and `#8` are per-instance — not global. **Do NOT add a fake `_MODE_LOCK` global to `state.py`.**

### IMPORTS: Re-import pattern for mutable state

When a function needs to read mutable config that another module re-assigns, use the re-import pattern:
```python
def get_config():
    from proxy.module import _config  # re-import to get current value
    return _config
```

### LOGGING: Include `request_id`

All operational log messages should include `request_id` when available in context. Use `logger.info("message", extra={"request_id": ctx.request_id})`.

### DESIGN: No backwards compatibility unless explicitly stated

Backwards compatibility is a luxury we cannot afford. When modifying schemas, APIs, interfaces, or data structures:

- **NEVER** add deprecation wrappers, shims, or compatibility layers unless the human explicitly requests it and it is documented.
- **NEVER** keep old code paths "just in case" — delete them.
- **NEVER** add versioned endpoints or fields for migration purposes unless explicitly asked.
- If a change breaks existing behavior, that is expected — clean breaks over accumulated tech debt.

The only exception is when the human explicitly states backwards compatibility is required and clarifies it in the docs.

### Deep-copy message lists in pipeline stages

Always use `copy.deepcopy()`, never `list()`, when duplicating message lists in pipeline stages. `list()` shallow-copies dict items → aliasing mutation across turns.

## Universal Landmines

These landmines apply **everywhere** — no specific module. Module-specific landmines live in the relevant sub-AGENTS.md.

### 1. Neo4j test isolation — the harness NEVER wipes a database, and tests REQUIRE Docker

The test harness (`tests/integration/neo4j_harness.py`) **never** runs `MATCH (n) DETACH DELETE n` against any shared database. The old `I_AM_AN_IDIOT_ASK_A_HUMAN=yes` wipe gate has been **REMOVED entirely** — it was a foot-gun (agents set the flag without thinking and wiped production; hundreds of wipes are recorded in `backups/`). There is now **no flag and no override**: `_clear_all_data` refuses unconditionally, and the fixtures skip/mock when they cannot get an isolated database.

**Tests REQUIRE Docker.** Neo4j-backed tests run against the throwaway **Enterprise test container on `bolt://localhost:7688`** (`docker-compose.test.yml`), which supports per-test `CREATE DATABASE` isolation. Production lives on **`bolt://localhost:7687`** (`docker-compose.yml`, Enterprise + RBAC `DENY DELETE ON NODE Task`).

- **If Docker is not up, NO-ONE runs DB-touching tests.** Start the test container first: `docker compose -f docker-compose.test.yml up -d`. If `:7688` is unreachable, Neo4j tests **skip** — that is CORRECT behaviour, not a failure to "fix".
- **Never point tests at `:7687`.** The harness defaults to `:7688`; if `PROXY_NEO4J_BOLT_URL` targets `:7687`, fixtures skip/mock and `_clear_all_data` refuses. Production is un-wipeable by any code path, env var, or caller.
- **NEVER** re-introduce a wipe flag, remove the `pytest.skip(_SKIP_MSG)` calls, or remove the Neo4j-down / prod-guard skips to "fix" skipped tests or inflate the pass count. Commit `2753fbe` ("remove 9 skip markers") did exactly that and caused **3 production data losses in one day**. Skipped tests here mean "isolation unavailable" — fix the environment (start Docker), never the skip.

See [tests/AGENTS.md](./tests/AGENTS.md) for the full no-wipe policy and the regression tests that pin it.

### 2. Tautological tests — green ≠ correct

A test that PASSES does not prove the code is correct if the test is wired to the wrong code path, mocks the function under test, or exercises a simplified shortcut that bypasses the bug. This pattern cost **5 adversarial waves** to catch **4 real production bugs**. See [tests/AGENTS.md](./tests/AGENTS.md) for the full discipline; the 5 invariants in short:

1. **Exercise the REAL production call site.** Grep: does the test's call stack include the fixed line?
2. **Do NOT mock the function under test.** Use the real function with real test DB / real state.
3. **Revert-test every fix.** Revert the fix (one-line edit), run the test, confirm it FAILS. If it still passes → tautology.
4. **Dead-code check before testing.** Grep `src/` for production callers of the method under test. Zero callers = dead code = delete, don't test.
5. **Concurrency bugs need concurrency tests.** Use `threading.Thread` + `threading.Barrier` for race bugs. Sequential "two calls in a row" is a tautology.

### 3. Investigation rigor — three disciplines

1. **Trace to the swallow site before naming the culprit line.** A bare `except Exception: pass` may be a symptom — the real culprit can be 2–3 frames deeper where another handler swallows and returns `None`. Verify the exception actually escapes the caller before naming the handler line as the cause.
2. **Verify the failure mode is reachable before calling it a bug.** When claiming an `AttributeError` / `TypeError` / `KeyError`, check the type definition. A dataclass-declared field is always set by its constructor; the missing-attribute mode cannot occur. A 30-second type check prevents misclassifying a theoretical bug as an actual one.
3. **"Zero callers today" ≠ ship-blocker.** Severity is about what could go wrong in production today, not what an antipattern could cause if wired in. Reserve "ship-blocker" / "critical" for active security, data-loss, or functional-failure findings.

## Sub-Agent Ingress Canary (one-liner)

Sub-agent CLI forks from a parent request emit a position-0 canary marker that the proxy parses to determine inherited mode + workspace. Format:

```
[[__SUBAGENT__:SKILL:<name>:<64-hex-sig>:<ts>|<state-json>]]
```

- Single parser (`parse_canary` in `proxy/canary.py`) — never inline position parsing.
- HMAC binds `skill_name + state_str + timestamp`; 5-minute freshness window.
- WORKSPACE canary sets `inherited_skill = None` (NOT the string `"workspace"`).
- `subagent_mode="work"` escalation has a per-session rate limit (5/hour); prompt-injection mitigation, not a complete fix.

Full invariants in [src/proxy/AGENTS.md](./src/proxy/AGENTS.md) (canary section).

## Test Layers

| Layer | Directory | When to use | Builders |
|---|---|---|---|
| **unit** | `tests/unit/` | Pure logic, no I/O, fast feedback | `tests/builders.py` |
| **integration** | `tests/integration/` | DB-touching (SQLite/Neo4j); uses `neo4j_test_db` fixture | `tests/builders.py`, `tests/mocks.py` |
| **proxy** | `tests/proxy/` | End-to-end through Flask shell (`proxy_with_mock` fixture) | `tests/builders.py`, `tests/mocks.py` |
| **e2e** / **e2e_real** | `tests/e2e*/` | Real upstream LLM hits — use sparingly, costs tokens | — |
| **load** | `tests/load/` | Concurrency / throughput tests | `tests/concurrency_harness.py` |

`data_scripts` is **not** a current layer — removed. `--dist=loadfile` is the correct worker distribution (`--dist=loadgroup` was used in older handovers).

Coverage gates: 90% clients, 56% memory, 66% proxy, 63% overall (run via `make coverage`).

## Behavioral Contracts

[USER-STORIES.md](./USER-STORIES.md) has the USER DEFINED expectations for key actions/flows/responses.

**Usage:** Read USER-STORIES.md when modifying or implementing behavior in a covered subsystem. It defines what the user expects vs what the code currently does. It is NOT a standing ruleset to carry in context at all times — it's a reference document for disambiguation and implementation guidance.