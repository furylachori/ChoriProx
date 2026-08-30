# ADR-011: No hot reload for custom JSON skills

## Status
Accepted (2026-07-11, Wave 7)

## Context

ChoriProx loads custom JSON skills via `populate_builtins()` at process startup
(`src/proxy/skills/skill_registry.py:399-440`). The strict-schema loader
(`src/proxy/skills/custom_skill_loader.py:_load_custom_skills`) parses each
file once, registers the resulting `SkillDefinition` via
`register_skill()`, and stores a `CustomSkillsStatus` snapshot on the
singleton.

The implementation currently has no runtime API for reloading or replacing
custom skills. `reset_default_registry()` is test-only; `register_skill()`
is documented test-only; `populate_builtins()` is called exactly once per
process lifetime. Operators who edit a JSON file expect changes to take
effect after a process restart — this is documented in the operator guide
and in the `AGENTS.md` file-table comment for `populate_builtins`.

The question is: should we add a hot-reload mechanism so JSON file edits
take effect without a restart?

## Considered Options

### Option 1 — No hot reload (CHOSEN)

Custom JSON files are loaded once at startup. Operators restart the
process to apply changes. Documented in the operator guide under
"Restart workflow" and in the `AGENTS.md` skill_registry.py file-table
comment.

**Pros:**
- Simple implementation. No file watcher, no inotify dep, no race
  conditions on partial reloads.
- Atomic publication: the entire registry is built once, never
  partially-mutated while serving requests.
- canary-verify cache (`proxy/canary.py:550`) is process-lifetime scoped;
  hot reload would invalidate the cache and require careful invalidation.
- Active agent modes reference `mode_state.skill_state` (per
  SkillDefinition, set at activation time). Hot reload would either
  (a) leave the active mode referencing a now-stale definition (silent
  inconsistency), or (b) force-clear the active mode (data loss).
- In-flight requests: a reload mid-request could leave a tool call
  resolved against a half-replaced skill definition.
- `DENIED_FOR_CUSTOM` policy would be re-evaluated, but already-installed
  ModeStates would still reference the OLD policy.

**Cons:**
- Operators must restart for any JSON change. For high-availability
  deployments this means a rolling restart per skill edit.

### Option 2 — File-watcher hot reload with atomic registry swap

Add a `watchdog` or `inotify`-based watcher on `PROXY_SKILLS_DIR`. On
file change: re-run the loader, construct a new `SkillRegistry`, swap
the singleton, fire deactivation callbacks for any active modes that
reference removed skills.

**Pros:** zero-downtime skill updates.

**Cons:**
- **Registry generation IDs.** Active modes would need to track which
  registry generation they were installed under. Compare-and-clear
  would need a per-generation snapshot, not just a per-mode snapshot.
- **Atomic snapshot swap.** The singleton currently uses a single
  `_default_registry` module global. Replacing it requires a generation
  pointer with compare-and-swap semantics (atomic ref swap), plus every
  reader going through a read-side indirection.
- **Active-mode policy.** What happens to a mode referencing a skill
  that's been REMOVED in the new generation? Force-clear? Mark stale?
  Reject the next message that tries to use it? Each option has UX cost.
- **Canary cache invalidation.** `canary_verify_cache` is keyed on
  (session_id, sig_hash, state_hash, secret_hash). A skill replacement
  changes the state hash for new canaries, but cached entries from the
  OLD generation would still verify. Either invalidate the cache
  per-generation (memory churn) or version-tag cache keys (lookup cost).
- **In-flight request draining.** Tool calls mid-flight reference the
  current definition. A reload mid-request could leave a tool resolved
  against a half-replaced definition.
- **Test infrastructure.** Every test that mutates JSON in
  `tmp_path/skills/` would now race against a background watcher.
  Tests that previously relied on `reset_default_registry()` between
  runs would need new isolation patterns.

### Option 3 — Operator-triggered reload (CLI command)

Add `^^reload` or a WSGI admin endpoint that operators invoke to
reload the skill registry on demand. Not automatic — operator-triggered.

**Pros:** simpler than file watcher; operator controls timing.

**Cons:** still requires:
- Generation tracking for active modes.
- Canary cache invalidation.
- In-flight request coordination.
- Test infrastructure changes.

Worst of both worlds: most of Option 2's complexity, none of its
zero-downtime benefit.

## Decision

**Option 1 — No hot reload.** Custom JSON files are loaded once at
process startup. Operators restart to apply changes.

The complexity of Options 2 and 3 is not justified by the use case
(custom skill JSON files are operator-authored, infrequent edits). The
restart workflow is already documented in the operator guide and the
`AGENTS.md` file-table comment.

## Prerequisites for Reconsidering

If a future requirement emerges (e.g., multi-tenant deployments with
frequent skill updates, or hot-deploy requirements during zero-downtime
rollouts), reconsider Option 2 ONLY if all of the following are
implemented and tested:

1. **Registry generation IDs.** Add `registry_generation: int` to the
   singleton, increment on reload. Active `ModeState` records its
   generation; compare-and-clear compares generations.
2. **Atomic snapshot swap.** Replace the module global with a
   generation-keyed pointer. Readers go through `get_active_registry()`
   which reads the current pointer atomically (no global lock held
   during request handling).
3. **Active-mode policy for removed skills.** Decide: force-clear?
   mark stale? reject next message? Document the choice and add tests.
4. **Canary cache invalidation.** Per-generation cache namespace; old
   generations evicted on swap.
5. **In-flight request draining.** Coordinate with the existing
   `RequestPipeline.request_scope` (`src/proxy/agent/agent_registry.py`).
   Wait for in-flight requests to complete before swapping.
6. **Test infrastructure.** Tests that mutate JSON in `tmp_path` need
   a `wait_for_reload()` helper or explicit reload API. `proxy_with_mock`
   needs an opt-in `enable_reload_watcher=True` flag.

Until all six are implemented, the no-hot-reload decision stands.

## Consequences

**Easier:**
- Simple implementation. The loader is called exactly once per process.
- Atomic publication: the entire registry is built once and never
  partially mutated while serving requests.
- No race conditions between reloads and active modes.
- No canary cache invalidation concerns.
- Test infrastructure remains simple (no watcher side-effects).

**More difficult:**
- Operators must restart for any JSON change.
- High-availability deployments need a rolling restart per skill edit.

**Neutral:**
- The CLI fail-fast behavior on bad config (Wave 2, `qwen_proxy.py:42-60`)
  becomes more important: a bad JSON file blocks the proxy from starting,
  forcing operators to fix it before continuing.

## Alternatives Considered

1. **File-watcher hot reload (Option 2 above)** — Rejected because of the
   generation / canary-cache / in-flight-drain complexity enumerated in
   the prerequisites.
2. **Operator-triggered reload (Option 3 above)** — Rejected as worst of
   both worlds: most of Option 2's complexity, none of its
   zero-downtime benefit.
