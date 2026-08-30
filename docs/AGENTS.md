# `docs/` — Agent Operating Context

> Developer-facing guide to finding **CURRENT** vs **STALE** documentation in
> the ChoriProx repo. Companion to [`/AGENTS.md`](../../AGENTS.md) (project-
> level operating context) and [`docs/TEST_PATTERNS.md`](TEST_PATTERNS.md)
> (test-layer guide). This file covers the doc taxonomy, where the current
> authoritative source lives for each subsystem, and how to keep docs in
> sync with code.

## 1. The doc taxonomy

ChoriProx docs fall into 4 status categories. **The status of a doc is set by
this file**, not by the doc's own header — many stale docs still claim
"CURRENT" or "v4" or "Latest" in their title.

| Status | Meaning | Action |
|---|---|---|
| **CURRENT** | Verified against code; matches the implementation | Use as authoritative source |
| **STALE** | Partially drifted (line numbers, constants, counts) but still 80%+ accurate | Update line numbers + constants; verify code still matches before quoting |
| **SUPERSEDED** | Replaced by a newer doc but kept for historical reference | Do not use for current behavior; link to replacement from header |
| **DEAD** | No longer referenced; nothing cites it; safe to delete | Delete (open a PR) |

### Why docs go stale

Four forces drive docs out of sync:

1. **Refactors without doc updates.** Tier-2 / Tier-3 refactors (V2 context pipeline, `db.py` façade split, `tool_parsing/` package split) silently moved code without updating external references.
2. **Line drift.** A doc pins a behavior at `tool_loop.py:143`; a refactor moves the constant to `tool_loop.py:265`. The doc still describes correct behavior at the wrong line.
3. **Supersession.** A new spec (`docs/specs/ticket-manager-v2-spec.md`) replaces an older spec (`docs/specs/ticket-manager-v2-orchestrator-prompt.md`) — but the old one stays in tree and is grep-able, misleading search.
4. **Handover accumulation.** Every session left a `HANDOVER_*.md` file with the test count at the time of writing. Now there are 5 handovers citing different test counts.

### How to find the current doc for a subsystem

Use the **`docs/AGENTS_MAPPING.md` matrix** in this directory. It is the
authoritative status registry for documents it classifies; entries
are added as subsystems are reviewed. It is incrementally expanded and
is not yet a complete inventory of every file under `docs/`. Entries
carry: subsystem, status (CURRENT / STALE / SUPERSEDED / DEAD), and
the canonical source for that subsystem's current behavior.

## 2. The CURRENT docs (single source of truth per subsystem)

The table below lists each subsystem's authoritative doc. When you need
"what does this do today?", read the doc in this table — not the older
specs/handover/handbook docs.

| Subsystem | Current doc | Path | One-line summary |
|---|---|---|---|
| **Project identity, conventions, landmines** | `/AGENTS.md` | `/AGENTS.md` | 272 lines: project identity, quick commands, bug policy, fix flow (5 steps), code conventions, behavioral contracts, the 10 top landmines |
| **Architecture (synthesis)** | `_analysis/PIPELINE_MAP.md` | `docs/_analysis/PIPELINE_MAP.md` | 1500-line pipeline walkthrough; the most authoritative pipeline doc (generated from W3D wave) |
| **Test layer guide** | `TEST_PATTERNS.md` | `docs/TEST_PATTERNS.md` | US-XXX traceability, builder usage, layer choice, common pitfalls |
| **Test guide (agent view)** | `tests/AGENTS.md` | `tests/AGENTS.md` | Companion to TEST_PATTERNS.md: test infrastructure, autouse fixtures, builder pattern, pinned-behavior test pattern |
| **Env vars** | `environment-variables.md` | `docs/environment-variables.md` | Every env var, default, and override surface |
| **Feature flags** | `feature-flags.md` | `docs/feature-flags.md` | Every feature flag + kill switch (`PROXY_USE_ADAPTERS`, `PROXY_LOG_LEVEL`, etc.) |
| **Error taxonomy** | `error-taxonomy.md` | `docs/error-taxonomy.md` | All error types, status codes, and circuit-breaker semantics |
| **Architecture decisions (ADRs)** | `adr/001-011.md` | `docs/adr/001-queryexecutor-over-per-method-sessions.md` etc. | 11 ADRs explaining the "why" behind key design decisions |
| **Skills spec** | `specs/skill-system-spec.md` | `docs/specs/skill-system-spec.md` | Skill protocol API + SkillRegistry (single source for skill infrastructure) |
| **Ticket manager spec** | `specs/ticket-manager-v2-spec.md` | `docs/specs/ticket-manager-v2-spec.md` | Current ticket skill spec (replaces `apply-mode-unification.md`) |
| **Workspace model** | `specs/workspace-rework-v5.md` | `docs/specs/workspace-rework-v5.md` | v5 global-catalog workspace model (supersedes v4 isolation — `test_workspace_isolation.py` is deprecated) |
| **SOTA sync** | `SPEC_UNIFIED_SOTA_SYNC.md` | `docs/SPEC_UNIFIED_SOTA_SYNC.md` | Current unified sync_exchange pipeline (replaces sota-channel-ground-zero.md) |
| **Memory tools** | `MEMORY_TOOLS_SPEC.md` + `MEMORY_TOOLS_IMPLEMENTATION.md` | `docs/MEMORY_TOOLS_SPEC.md` | 6-tool model + implementation status (replaces `MEM_TOOLS_PROPOSAL_SPEC.md`) |
| **Debug session** | `devtools/debug-session-ops.md` | `docs/devtools/debug-session-ops.md` | Operator-facing guide for `devtools.debug_session.debug_session()` |
| **Deployment runbook (chain hash)** | `DEPLOYMENT_RUNBOOK_CHAIN_HASH.md` | `docs/DEPLOYMENT_RUNBOOK_CHAIN_HASH.md` | Chain-hash v2 deployment |
| **Deployment runbook (task mode)** | `DEPLOYMENT_RUNBOOK_TASK_MODE.md` | `docs/DEPLOYMENT_RUNBOOK_TASK_MODE.md` | Task mode deployment |
| **Neo4j migrations** | `migrations.md` (partial) | `docs/migrations.md` | Documents 001–011 (but see §3 — only 001/002 have detail) |
| **Adding models** | `ADDING_MODELS.md` | `docs/ADDING_MODELS.md` | Multi-model routing registration |
| **Custom JSON skills hardening** | `specs/custom-skills-remediation.md` | `docs/specs/custom-skills-remediation.md` | Wave 1 strict-schema loader, atomic startup, three-state tool policy, restart-only contract |
| **Main-agent VFS placement/cache safety** | `specs/main-agent-vfs-cache-safe-tail.md` | `docs/specs/main-agent-vfs-cache-safe-tail.md` | Implemented cache-safe pre-live placement, scope, budget, provider-role, and lifecycle-prefix contract |

## 3. The STALE / SUPERSEDED / DEAD docs

**DO NOT TRUST these docs without verifying against code.** The list below is
non-exhaustive; it captures the most commonly cited docs that contradict
the current implementation.

### STALE — partially accurate, drift in numbers or sections

| Doc | Path | What drifted | Action |
|---|---|---|---|
| `migrations.md` | `docs/migrations.md` | Documents 001 + 002 in detail; 003–011 only listed by name | Read 001/002 here; for 003–011 read source at `src/memory/migrations/00X_*.py` |
| `RCA-R3-ticket-mode-budget-exhaustion.md` | `docs/RCA-R3-ticket-mode-budget-exhaustion.md` | Claims `max_total_time=90.0`; actual is **900.0s** (`tool_loop.py:265`) | The RCA narrative is correct; the budget constant changed |
| `PROMPTS_CURRENT.md` | `docs/PROMPTS_CURRENT.md` | Claims MAIN ~503 tokens; `IMPLEMENTATION_HANDOVER.md` says 183 | Read source in `src/proxy/tool_parsing/prompts.py` for the actual count |
| `spec/08-kilocode-integration.md` | `docs/spec/08-kilocode-integration.md` | Kilocode-only; ChoriProx is multi-client (opencode + kilocode) | Replaced by `docs/skills/extension-guide.md` |
| `spec/09-system-prompt-architecture.md` | `docs/spec/09-system-prompt-architecture.md` | Pre-V2 prompt architecture; V2 redesign superseded it | See `_analysis/PIPELINE_MAP.md` §3-4 for current |
| `MEMORY_TOOLS_REMEDIATIONS.md` | `docs/MEMORY_TOOLS_REMEDIATIONS.md` | Pre-V2 tool definitions | See `MEMORY_TOOLS_SPEC.md` |
| `TASK_MODE_SPEC.md` | `docs/TASK_MODE_SPEC.md` | Pre-V2 task mode | See `specs/ticket-manager-v2-spec.md` |
| `specs/canary-ttl-expiry-fix.md` | `docs/specs/canary-ttl-expiry-fix.md` | Describes the fix; current HMAC binding is in `proxy/canary.py` docstring | Source docstring is current |
| `docs/v2-growing-prefix-plan.md` | `docs/v2-growing-prefix-plan.md` | Historical V2 sliding-window plan; operative VFS placement moved to the cache-safe-tail spec | See `docs/specs/main-agent-vfs-cache-safe-tail.md` |
| `docs/_analysis/MEMORY_TIER_FIX_PLAN.md` | `docs/_analysis/MEMORY_TIER_FIX_PLAN.md` | Broad memory-tier plan; cursor/summary/ancient/budget/compression/observability guidance still useful, but VFS placement and exact line references are stale | See `docs/specs/main-agent-vfs-cache-safe-tail.md` for VFS, `docs/_analysis/PIPELINE_MAP.md` for current pipeline |

### SUPERSEDED — replaced by a newer doc but kept for reference

| Doc | Path | Replaced by |
|---|---|---|
| `MEM_TOOLS_PROPOSAL_SPEC.md` | `docs/MEM_TOOLS_PROPOSAL_SPEC.md` | `MEMORY_TOOLS_SPEC.md` (6-tool model replaced by 4-tool `recall`/`search`/`proxy_ticket`/`proxy_work`) |
| `IMPLEMENTATION_HANDOVER.md` | `docs/IMPLEMENTATION_HANDOVER.md` | Token counts (183 MAIN / 57 SUB) are stale; read source |
| `specs/sota-channel-ground-zero.md` | `docs/specs/sota-channel-ground-zero.md` | `SPEC_UNIFIED_SOTA_SYNC.md` |
| `specs/apply-mode-unification.md` | `docs/specs/apply-mode-unification.md` | `specs/ticket-manager-v2-spec.md` |
| `specs/mode-prompt-restructure.md` | `docs/specs/mode-prompt-restructure.md` | V2 prompt pipeline (see `_analysis/PIPELINE_MAP.md`) |
| `specs/task-commands-fix-plan.md` | `docs/specs/task-commands-fix-plan.md` | `specs/task-commands-design.md` (current) + `specs/task-commands-implementation-plan.md` |
| `docs/HANDOVER_*.md` (all 5) | `docs/HANDOVER_*.md` | Session-bound handovers; superseded by `_analysis/PIPELINE_MAP.md` + `_analysis/ARCHITECTURE_BLUEPRINT.md` (in `.scratch_wave_reports/`) |
| `docs/spec/02-architecture.md` … `09-system-prompt-architecture.md` | `docs/spec/*.md` | Pre-V2 architecture; superseded by `_analysis/PIPELINE_MAP.md` |
| `docs/analysis/task-planning-v2-alternatives.md` | `docs/analysis/task-planning-v2-alternatives.md` | `docs/specs/workspace-rework-v5.md` |
| `docs/spec/task-planning/*.md` (4 files) | `docs/spec/task-planning/` | `docs/specs/workspace-rework-v5.md` |
| `docs/fix-plans/sub-agent-skill-inheritance.md` | `docs/fix-plans/sub-agent-skill-inheritance.md` | Implemented in `_resolve_task_canary` (`src/proxy/canary.py`) |
| `docs/plans/block-ticket-on-default-workspace.md` | `docs/plans/block-ticket-on-default-workspace.md` | Implemented; see `proxy/skill_resolver.py` |
| `docs/plans/connection-pool-fix.md` | `docs/plans/connection-pool-fix.md` | Implemented in `src/memory/db_pool.py` (Phase 3) |
| `docs/plans/task-graph-native-migration.md` | `docs/plans/task-graph-native-migration.md` | Migrations 003-005 implemented; see `migrations/` |
| `docs/rollback/rollback-5.4-mode-unification.md` | `docs/rollback/rollback-5.4-mode-unification.md` | The rollback was never invoked; current is mode unification |
| `docs/specs/architecture-refactor/*` | `docs/specs/architecture-refactor/` | The refactor was completed; see `_analysis/PIPELINE_MAP.md` |
| `docs/specs/consolidation-research/*` | `docs/specs/consolidation-research/` | Research output; current implementation lives in code |
| `docs/skills/custom-skills-plan.md` | `docs/skills/custom-skills-plan.md` | Replaced by `docs/specs/custom-skills-remediation.md` (Wave 1, 2026-07) — banner added at top of file |

### DEAD — already deleted from the repo, but cited in older docs

| Doc (cited name) | Status | Notes |
|---|---|---|
| `docs/ARCHITECTURE.md` | DELETED | Was 313 lines; partially stale (claimed 13-model counts that drifted, didn't reflect migration 004) |
| `docs/HANDOVER_2026_06_04.md` | DELETED | Rate limits cited (10/60s) are stale; current is 500/60s |
| `docs/MEMORY.md` | DELETED | Test count "100 tests for tests/skills/" outdated |
| `docs/TOOL_HANDLING_SYSTEM_REFERENCE.md` | DELETED | Mentioned deleted constants; tool count conflicts with current |
| `docs/HANDOVER.md` | DELETED | Test count 2507 stale |
| `docs/HANDOVER_CANARY_REFACTOR.md` | DELETED | LOW staleness risk |
| `docs/HANDOVER_SESSION_2026_06_03.md` | DELETED | Proxy was BROKEN at write time |
| `docs/HANDOVER_SESSION_2026_06_04.md` | DELETED | Connection pool fix spec; test count 1571 stale |
| `docs/IMPLEMENTATION_PLAN.md` | DELETED | Exchange-level context trimming plan; COMPLETED |
| `docs/PENDING-MINIMAX-FORMAT-RESEARCH.md` | DELETED | CLOSED, low staleness |
| `docs/SECURITY_AUDIT_REPORT.md` | DELETED | 21 SEC findings; paths may be stale |
| `docs/ADVERSARIAL_REVIEW_FINAL.md` | DELETED | 58 findings; VERY HIGH staleness |
| `docs/VALIDATED_FINDINGS_REPORT.md` | DELETED | 98 findings + 21 debunked; VERY HIGH staleness |
| `docs/ChoriProx_Codebase_Deep_Analysis_Report.md` | DELETED | Self-flagged as predating Tier 2 + 3; references deleted files |
| `docs/FOLLOW-UP-SOTA.md` | DELETED | SOTA timeout investigation |
| `docs/SPEC_TICKET_DRIVEN_ENVIRONMENT.md` | DELETED | 1128 lines; partially implemented |
| `src/qwen_tools.py` | EXISTS (273 LOC) | Dead code (not imported by any module); safe to delete |

> **Why are some DEAD docs cited as STALE?** Older handover docs (and AI
> training data) reference docs that have since been deleted. If you grep
> `docs/` for these names and find nothing, they were deleted — not stale.
> Don't try to "fix" a deleted doc; mark the citation as stale in your own
> doc.

## 4. How to write a new doc (docs-as-code)

### Step-by-step

1. **Pick the right subsystem folder.** Use the doc map in §2 as a guide:
   - `docs/specs/` — current specs (TicketSkill, SOTA, Workspace)
   - `docs/adr/` — architectural decisions
   - `docs/devtools/` — operator-facing docs
   - `docs/_analysis/` — synthesis / pipeline walks
   - `docs/skills/` — skill-extension guides
   - `docs/spec/` — pre-V2 architecture (DO NOT ADD HERE — that folder is legacy)
2. **Use the prompt-doc templates** if writing a prompt-related doc:
   - `docs/PROMPT_DOCUMENTATION_TEMPLATE.md` — template
   - `docs/PROMPT_DOCUMENTATION_EXAMPLE.md` — worked example
   - `docs/PROMPT_DOCUMENTATION_INDEX.md` — index of prompt docs
   - `docs/PROMPT_MIGRATION_GUIDE.md` — how to migrate old prompts to new format
3. **Reference `_analysis/PIPELINE_MAP.md`** if the doc touches pipeline code.
4. **Reference `/AGENTS.md`** if the doc states an invariant or landmine.
5. **Add an entry to `docs/AGENTS_MAPPING.md`** (this directory) with the new doc's path, subsystem, status (CURRENT), and one-line summary.
6. **If replacing an old doc**: open a PR that
   - Adds the new doc
   - Adds an entry to `docs/AGENTS_MAPPING.md` marking the old doc SUPERSEDED
   - Adds a `> SUPERSEDED: replaced by [new-doc](new-doc.md)` header at the top of the old doc
   - Updates any cross-references in other docs

### Naming conventions

- `NN-topic-area.md` for sequential specs (`01-overview.md`, `02-architecture.md`)
- `topic-name.md` for flat docs
- `topic-area-vN.md` for versioned specs (`workspace-rework-v5.md`)
- `topic-area-YYYY_MM_DD.md` for date-bound handovers (e.g., RCA reports)

### Don't

- Don't add new docs to `docs/spec/` (legacy pre-V2).
- Don't add a new `HANDOVER_*.md` — use `docs/devtools/` for runbooks instead.
- Don't duplicate content between `_analysis/PIPELINE_MAP.md` and a new doc — link instead.
- Don't claim "Current" or "Latest" in titles — the version is in the filename or in the `docs/AGENTS_MAPPING.md` status.

## 5. The doc review pattern

When you change code, the matching doc must be updated. The discipline:

1. **Find the doc** that covers the code you changed (use the `docs/AGENTS_MAPPING.md` table).
2. **Update line numbers.** If the doc cites `tool_loop.py:143` and you moved the constant to `tool_loop.py:265`, update the line.
3. **Update constants.** If the doc cites `max_total_time=90.0` and you changed it to 900.0, update the constant.
4. **If the doc is wrong in spirit (not just line drift)**, rewrite the section. Don't try to patch.
5. **Update `docs/AGENTS_MAPPING.md`** to reflect any status change (CURRENT → STALE, STALE → SUPERSEDED, etc.).
6. **If the doc is dead**, delete it (open a PR). Don't leave dead docs grep-able.

### When to update `_analysis/PIPELINE_MAP.md`

`_analysis/PIPELINE_MAP.md` is the synthesis doc generated from the W3D wave
report. Update it when:

- A stage is added or removed
- A stage's source file moves
- A behavior-critical section changes (e.g., the singleton-sharing pattern)

This doc is the most authoritative; keep it in sync.

## 6. The reference doc tree (one-paragraph summary each)

### `/AGENTS.md` (root)

272 lines. The project identity doc: stack (Python ≥3.9, Flask, Neo4j, SQLite, requests), quick commands, bug policy ("every bug you see is yours"), 5-step fix flow, sub-agent delegation rule, code conventions (state mutation, lock hierarchy, agent API, Neo4j query patterns, no backwards compat), known landmines (tool-call argument normalization, lock ordering, canary parsing, Neo4j wipe gate, safe_db_name, etc.). **Read first, always.**

### `docs/_analysis/PIPELINE_MAP.md`

1500 lines. The most authoritative pipeline walkthrough. Generated from the
W3D wave report. Covers: data structures (`RequestContext`, `SubagentIngress`,
`BackendError`), the 9 pipeline stages end-to-end, sub-agent ingress flow,
canary generation/injection, mode resolution. **When the code and this doc
disagree, this doc wins.** (Per the wave report contract.)

### `docs/TEST_PATTERNS.md`

260 lines. Test-layer guide: US-XXX traceability, builder pattern, layer
choice (unit/integration/e2e/e2e_real), disabling tests (`skip`/`xfail`/
`skipif`), common pitfalls (assertion-dead tests, heavy mocking, `time.sleep()`
band-aids, stateful scripts vs mocks, singleton leakage). Companion to
`tests/AGENTS.md`.

### `docs/environment-variables.md`

Every env var the proxy reads (`PROXY_API_KEY`, `PROXY_SESSION_SECRET`,
`PROXY_NEO4J_BOLT_URL`, `NEO4J_AUTH`, `PROXY_SOTA_API_KEY`, `PROXY_SOTA_ENDPOINT`,
`PROXY_SOTA_MODEL`, `PROXY_TEST_DB`, `PROXY_USE_ADAPTERS`, `PROXY_LOG_LEVEL`,
`PROXY_METRICS_ENABLED`, `PROXY_DUMP_REQUESTS`, `PROXY_SKILLS_DIR`, etc.)
with default value and override surface.

### `docs/feature-flags.md`

Feature flags and kill switches. Mirrors `feature-flags.md` (`PROXY_USE_ADAPTERS=0`,
`PROXY_LOG_LEVEL`, `PROXY_METRICS_ENABLED`, etc.). Includes the `enhancer`
flag (which doesn't exist — `^^imp` is dead code at runtime; see
`docs/AGENTS_MAPPING.md` → `src/enhancer/`).

### `docs/error-taxonomy.md`

All error types: `BackendError`, `InfrastructureError`, `ConstraintViolation`,
`QueryTimeoutError`, `AuthConfigError`, `StageAbort`, circuit-breaker states,
HTTP status code mapping (`401`, `403`, `413`, `502`).

### `docs/adr/001-011.md` (11 ADRs)

1. `001-queryexecutor-over-per-method-sessions.md` — single Neo4j driver
2. `002-ratelimiter-as-standalone-class.md` — RateLimiter class extraction
3. `003-workspacestatestore-compound-operations.md` — atomic compound ops
4. `004-eviction-coordinator-separate-module.md` — separate eviction module
5. `005-migration-merge-concurrency-protection.md` — MERGE concurrency safety
6. `006-union-branch-2-permanent-safety-net.md` — UNION fallback safety net
7. `007-postprocess-private-methods.md` — PostProcess private-method contract
8. `008-state-py-thin-reexport-layer.md` — `state.py` thin re-export
9. `009-eager-skill-registration.md` — eager skill registration at startup
10. `010-ioc-callbacks-for-deactivation.md` — IoC callbacks for deactivation
11. `011-custom-skill-restart-only.md` — custom-skill restart-only contract

### `docs/specs/skill-system-spec.md`

98KB. The current skill protocol API + SkillRegistry contract. Covers the
`SkillDefinition` dataclass, `TrustToken` trust gate, custom JSON skills,
trigger parsing, mode activation, prompt construction.

### `docs/specs/ticket-manager-v2-spec.md`

60KB. Current ticket skill spec. Covers the TicketSkill lifecycle (per-turn,
fire_and_forget=True), sub-agent inheritance, the work session → bug tracker
flow, planner contract.

### `docs/specs/workspace-rework-v5.md`

14KB. v5 global-catalog workspace model. **Supersedes v4 isolation**
(`test_workspace_isolation.py` is deprecated; see `test_workspace_sharing.py`).
Covers workspace canary, workspace_name inheritance, cross-session visibility.

### `docs/SPEC_UNIFIED_SOTA_SYNC.md`

870 lines. Current unified `sync_exchange` pipeline. Covers `submit_sync`,
the bounded thread pool, summary extraction, entity/fact persistence, batch
recovery. Supersedes `docs/specs/sota-channel-ground-zero.md`.

### `docs/MEMORY_TOOLS_SPEC.md`

Memory tool set: `recall` (read), `search` (read), `proxy_ticket`
(ticket-mode write), `proxy_work` (work-mode write). 4-tool model replaces
the 6-tool proposal (`docs/MEM_TOOLS_PROPOSAL_SPEC.md`).

### `docs/_analysis/ARCHITECTURE_BLUEPRINT.md` (in `.scratch_wave_reports/`)

665 lines. Synthesis doc from 22 sub-agent wave reports (W1A–W6B). The
project-state snapshot: module map, pipeline summary, memory architecture,
landmines (top 10), documentation state, the contradictions matrix.

**This is in `.scratch_wave_reports/`, not `docs/`** — it's a research artifact, not user-facing. When W6C/W6D land, the relevant findings migrate to `docs/AGENTS_MAPPING.md` and `docs/_analysis/PIPELINE_MAP.md`.

## See Also

- [`/AGENTS.md`](../../AGENTS.md) — project-level conventions and landmines
- [`docs/TEST_PATTERNS.md`](TEST_PATTERNS.md) — test-layer guide
- [`docs/_analysis/PIPELINE_MAP.md`](_analysis/PIPELINE_MAP.md) — pipeline walkthrough
- [`docs/AGENTS_MAPPING.md`](AGENTS_MAPPING.md) — the doc status matrix (this directory)
- [`tests/AGENTS.md`](../../tests/AGENTS.md) — test infrastructure guide