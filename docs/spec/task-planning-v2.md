# Task Planning v2 — Specification

This document is the overview and entry point for the task-planning specification. The implementation details have been split into separate phase documents, each covering a focused deliverable. Refer to the table of contents below for navigation.

## Table of Contents

- **This Document** — Overview & Cross-Cutting Concerns (§1–3, §9–12)
- **Phase 1 — Context Assembly** — `task-planning/phase-1-context.md`
- **Phase 2 — Read-Only Tool Suite** — `task-planning/phase-2-tools.md`
- **Phase 3 — Planning Tool Loop** — `task-planning/phase-3-loop.md`
- **Phases 4–5 — Channel Extension & Integration** — `task-planning/phases-4-5-integration.md`

---

## 1. Overview / Goal

The current `^^task` command invokes a single-shot SOTA call with a forced `add_tasks` tool choice. The model receives only the user's raw description and has no visibility into the session's conversation history, codebase, or external resources. This produces generic, often misaligned task plans.

This specification redesigns `^^task` into a **multi-turn, read-only planning loop**:

1. **Rich context assembly** — injects the session summary and recent messages from the SQLite DB, plus raw message content up to a computed token budget.
2. **Read-only tool suite** — gives the planner access to `read_file`, `file_glob`, `code_search`, `web_fetch`, `recall`, and `search` so it can ground proposals in actual files, memory, and web resources.
3. **Multi-turn convergence** — the planner may iterate up to 5 times, calling read-only tools, receiving results, and refining its plan before emitting `add_tasks`.
4. **Safe execution** — all file reads are sandboxed to the working directory, capped, and screened for sensitive patterns.
5. **Reusable infrastructure** — extends `SOTAChannel` with an `ask_multi()` method and follows the same tool-loop pattern already proven in `src/memory/tool_loop.py`.

**Non-goals:**
- Allowing the planner to write files, execute shell commands, or edit code.
- Generating new session summaries during planning (reuse existing DB summaries only).
- Adding new Python dependencies (e.g., tiktoken) — token counting stays heuristic-based for now.

---

## 2. Current State & Problems

### 2.1 Current Implementation (`src/commands/task_commands.py`)

```python
planner = SOTAChannel(
    mode=SOTAMode.PLANNING,
    system_prompt=TASK_PLANNER_PROMPT,
    tools=[ADD_TASKS_TOOL],
    tool_choice={"type": "function", "function": {"name": "add_tasks"}},  # forced
    temperature=0.3,
    max_tokens=4096,
    parse_fn=_extract_task_proposals,
)
result = planner.ask(description, session_id=session_id)
```

The `ask()` method sends exactly two messages:

```python
messages = [
    {"role": "system", "content": self._system_prompt},
    {"role": "user", "content": prompt},
]
```

### 2.2 Problems

| # | Problem | Impact |
|---|---------|--------|
| 1 | **No session context** — `ask()` ignores `session_id` when building messages. | Plans lack awareness of prior conversation, decisions, or rejected ideas. |
| 2 | **Forced tool choice** — `tool_choice` hardcodes `add_tasks`. | Model cannot explore via intermediate tool calls; it must emit tasks immediately even if it needs more information. |
| 3 | **No grounding tools** — no file, web, or memory access. | Proposals are hallucinated; filenames, APIs, and dependencies may be fabricated. |
| 4 | **Single-turn only** — no re-query loop. | The model gets one chance. If the prompt is ambiguous, it guesses. |
| 5 | **No token budget management** — raw prompt is sent unchecked. | Risk of exceeding context window when long descriptions or pasted code are provided. |
| 6 | **No read-only sandbox** — not applicable today, but will be needed for safe tool use. | Without one, any future file-access expansion risks path traversal and secret leakage. |

### 2.3 What Already Works (Reuse)

- `SOTAChannel` with circuit breaker, rate limiting, retry logic, and result parsing.
- `get_summary(session_id)` and `get_last_messages(session_id, n=50)` in `src/memory/db.py`.
- The enhancer (`src/enhancer/core.py`) already fetches summaries + recent messages and assembles context — we mirror that pattern.
- `src/memory/tool_loop.py` already implements multi-turn tool interception with message evolution, trimming, and orphan cleanup — we follow its structure but keep it read-only.
- Memory tool schemas (`MEMORY_TOOL_DEFINITIONS`) already exist in `src/memory/tools.py`.

---

## 3. Architecture

### 3.1 High-Level Flow

```
User: ^^task <description>
         │
         ▼
┌─────────────────────┐
│   task_commands.py  │  plan_tasks()
│  (entry point)      │
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│    task_context.py  │  assemble_context()
│  (Phase 1)          │  - fetch summary from DB
│                     │  - fetch last N messages from DB
│                     │  - token budget → truncate
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│    task_loop.py     │  run_planning_loop()
│  (Phase 3)          │  - multi-turn loop (max 5 iter, 90s)
│                     │  - tool_choice="auto" each iteration
│                     │  - execute_readonly_tool() sandbox
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│   sota_channel.py   │  ask_multi()
│  (Phase 4)          │  - per-call tool_choice override
│                     │  - message array evolution
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│    task_commands.py  │  _extract_task_proposals()
│  (Phase 5)          │  - parse add_tasks result
│                     │  - deduplicate, store proposal
└─────────┬───────────┘
          │
          ▼
   Proposed Tasks → session state → user confirmation
```

### 3.2 Module Responsibilities

| Module | File | Responsibility |
|--------|------|--------------|
| Context assembler | `src/planning/task_context.py` | Build prompt from DB summary + messages within token budget |
| Tool schemas | `src/planning/task_tools.py` | OpenAI-format schemas for 6 read-only tools + `add_tasks` = 7 total |
| Tool executor | `src/planning/task_tools.py` | `execute_readonly_tool()` sandbox with security guards |
| Planning loop | `src/planning/task_loop.py` | Multi-turn convergence, iteration limits, timeout guards |
| SOTAChannel ext | `src/memory/sota_channel.py` | Add `ask_multi()`; `ask()` becomes thin wrapper |
| Integration | `src/commands/task_commands.py` | Wire everything together in `plan_tasks()` and `_get_planner()` |

---

---

## 9. Security & Safety

| Concern | Mitigation |
|---------|------------|
| **Path traversal** | `Path.resolve()` + `is_relative_to()` check + `..` rejection. Rejects paths escaping the project. |
| **Sensitive file reads** | Extended blocklist: `.env`, `.env.*`, `*.key`, `*secret*`, `.pem`, `.crt`, `.cer`, `.p12`, `.pfx`, `.jks`, `id_rsa`, `id_ed25519`, `id_ecdsa`, `credentials*.json`, `token*`, `private*`. Plus directory denylist: `.git/`, `.ssh/`, `.aws/`, `.npm/`, `.docker/`, `node_modules/`. |
| **Resource exhaustion (file read)** | Rejects files > 100KB before reading. Returns at most 500 lines. |
| **Resource exhaustion (glob)** | Generator stops after 50 matches — no full list materialization. |
| **Resource exhaustion (search)** | 50-match cap per `code_search`. Skips binary extensions (`jpg`, `png`, `zip`, `.so`, etc.). |
| **Resource exhaustion (loop)** | 5 iteration cap, 90s total timeout. |
| **Network egress abuse** | `web_fetch` allowlists only `http://` and `https://`. Rejects internal IPs / loopback / private addresses. Checks `Content-Length` header before body read. 10s default timeout, max 30s. |
| **Write / shell execution** | Not in `PLANNING_TOOLS`. Executor rejects unknown tools with `"Forbidden tool"`. |
| **Data leakage in tool results** | Tool results truncated to 8,000 chars before being sent back to SOTA. |
| **DoS via large regex** | `code_search` compiles regex; `re.error` caught and returned as error. No catastrophic backtracking prevention yet (see Future Work). |
| **Concurrent planning calls** | `threading.Lock()` in `_get_planner()` protects lazy init only. If concurrent `^^task` calls are possible, consider a per-session planner instance or a loop-level lock. |

---

## 10. Files to Create / Modify

| File | Action | Description |
|------|--------|-------------|
| `src/planning/__init__.py` | **CREATE** | Package init for planning subpackage. |
| `src/planning/task_context.py` | **CREATE** | Token budget model, `assemble_context()`, DB message fetching, truncation logic. |
| `src/planning/task_tools.py` | **CREATE** | `PLANNING_TOOLS` schemas, `execute_readonly_tool()`, and all tool handler implementations. |
| `src/planning/task_loop.py` | **CREATE** | `run_planning_loop()`, `PlanningLoopResult`, convergence guards. |
| `src/memory/sota_channel.py` | **MODIFY** | Add `ask_multi()` method; refactor `ask()` to delegate to `ask_multi()`. |
| `src/commands/task_commands.py` | **MODIFY** | Update `plan_tasks()` to call context assembly + planning loop. Update `_get_planner()` to remove forced tool_choice. |
| `src/proxy/app.py` (or caller) | **MODIFY** | Pass current system prompt to `plan_tasks()` as `system_prompt_hint` so `_extract_working_dir()` resolves paths correctly. |

---

## 11. Acceptance Criteria

- [ ] **AC-1** — Context assembly fetches `get_summary(session_id)` and `get_last_messages(session_id, n=50)` from the DB. No new summary is generated during planning.
- [ ] **AC-2** — Token budget computation uses the `len(text) / 3.5` heuristic and respects the 95,800-token available input limit. A TODO comment references future tiktoken migration.
- [ ] **AC-3** — `SOTAChannel.ask_multi()` exists and accepts a per-call `tool_choice` parameter ( `"auto"`, `"none"`, or force-dict).
- [ ] **AC-4** — The planning loop uses `tool_choice="auto"` on every iteration. After the final tool result, if `add_tasks` has not been called, the loop terminates gracefully unless `force_add_tasks_on_miss=True`.
- [ ] **AC-5** — `PLANNING_TOOLS` includes `read_file`, `file_glob`, `code_search`, `web_fetch`, `recall`, `search`, and `add_tasks`.
- [ ] **AC-6** — `execute_readonly_tool()` enforces working-directory containment, caps reads at 500 lines / 100KB, caps glob/search at 50 results, and blocks `.env` / `*.key` / `*secret*` patterns.
- [ ] **AC-7** — `plan_tasks()` deduplicates proposals by normalized name and stores them in session state exactly as it does today.
- [ ] **AC-8** — If the planning loop times out or hits max iterations without calling `add_tasks`, the user receives a clear message (not a raw error or empty task list).

---

## 12. Open Questions / Future Work

1. **tiktoken migration** — Replace `len(text) / 3.5` with `tiktoken.get_encoding("cl100k_base").encode()` for exact token counts. Deferred to avoid new dependencies.
2. **Catastrophic regex backtracking** — `code_search` uses `re.compile()` without a timeout. For untrusted planner-generated regexes, consider using `regex` package with `timeout` or a subprocess sandbox.
3. **Binary file filtering** — `code_search` reads all matched files as text. Should skip known binary extensions (`.jpg`, `.png`, `.zip`, `.so`, `.dll`).
4. **Memory tool activation** — `recall` and `search` are currently stubbed in `src/memory/tools.py`. When enabled, no code changes are needed in the planner — the existing dispatch to `execute_memory_tool()` will work.
5. **Planner temperature tuning** — `temperature=0.3` may be too low for creative exploration. Consider A/B testing `0.3` vs `0.5` for tasks involving novel architecture.
6. **Persistent tool result caching** — If the same planner reads the same file twice in one loop, we re-read from disk. A small per-loop LRU cache (e.g., 10 entries) would save tokens and latency.
7. **Structured output mode** — When the upstream model supports JSON Schema constrained output, `add_tasks` could be guaranteed valid, removing the need for `_parse_add_tasks()` retries.
8. **Planner metrics** — Emit `planning.iterations`, `planning.tool_calls`, `planning.finish_reason`, and `planning.latency_ms` as Prometheus-style metrics or structured logs for observability dashboards.

---
