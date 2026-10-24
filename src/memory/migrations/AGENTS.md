# `src/memory/migrations/AGENTS.md` — Neo4j Schema Migrations

> 14 files: `__init__.py` + `base.py` + `runner.py` + 11 numbered migrations (001-011).
> Migration 004 is the BIG ONE — it transformed tasks to graph-native schema.
> Every migration is idempotent via `IF NOT EXISTS` + deterministic SHA-256 hashes.

## Module Purpose

Owns the versioned schema migration runner for Neo4j. Each migration is a
`Migration` subclass with `version: int` + `description: str` + `up(executor)`.
Idempotent on re-run. Backed by `(:_Migration)` nodes for completion tracking
and `(:MigrationLock)` nodes for cross-worker protection.

## Files in this module

| File | Purpose | Key lines |
|---|---|---|
| `__init__.py` | Imports all 11 migrations + exposes class names | line 1-26 |
| `base.py` | `Migration` base class — `version`, `description`, `up(executor)`, `down(executor)` | line 5 |
| `runner.py` | `MigrationRunner` — `register`, `ensure_schema`, `get_current_version`, `run_pending` with `MigrationLock` cross-worker protection + `_Migration` marker writes | line 9-107 |
| `001_backfill_parent_task_uuid.py` | `BackfillParentTaskUuid` — initial backfill (50 LOC) | |
| `002_verify_backfill.py` | `VerifyBackfill` — verification pass (22 LOC) | |
| `003_remove_project_relationships.py` | `RemoveProjectRelationships` — drop `:PROJECT` edges (21 LOC) | |
| `004_graph_native_tasks.py` | **`GraphNativeTasks` — the BIG transformation** (314 LOC, 6 steps) | line 11 |
| `005_ensure_graph_native.py` | `EnsureGraphNative` — safety net pass for migration 004 (305 LOC) | |
| `006_agent_task_assignment.py` | `AgentTaskAssignmentMigration` — `(short_hash, workspace_id)` uniqueness + `assigned_agent_id` index (29 LOC) | |
| `007_work_sessions.py` | `WorkSessionMigration` — `WorkSession` node + `HAS_WORK_SESSION` edge (30 LOC) | |
| `008_bug_tracking.py` | `BugTrackingMigration` — `Bug` + `Review` nodes + `HAS_REVIEW`/`FOUND_DURING`/`RESOLVED_BY` edges (44 LOC) | |
| `009_workspace_backfill.py` | `WorkspaceBackfill` — bare-name `:Workspace` nodes from existing Task data (98 LOC, 2 phases) | |
| `010_strip_workspace_prefix.py` | `StripWorkspacePrefix` — strip `session_id:` prefix from Task/Bug `workspace_id` (133 LOC, 5 phases) | |
| `011_entity_dedup_constraint.py` | `EntityDedupConstraint` — dedup `(name, entity_type)` Entity duplicates + create composite uniqueness constraint (231 LOC, 3 phases) | |

## Migration order

| # | Version | Class | Purpose | Critical? |
|---|---|---|---|---|
| 1 | 1 | `BackfillParentTaskUuid` | Initial backfill of `parent_task_uuid` from old schema | yes |
| 2 | 2 | `VerifyBackfill` | Verification pass for #1 | yes |
| 3 | 3 | `RemoveProjectRelationships` | Drop `:PROJECT` edges | yes |
| 4 | 4 | `GraphNativeTasks` | **THE big transformation** — `:BLOCKED_BY` / `:HAS_PARENT` / `:HAS_NOTE` + drop `attributes` blob + drop `entity_type` property + drop `review_notes` property + drop `parent_task_uuid` property | **critical** |
| 5 | 5 | `EnsureGraphNative` | Safety-net re-run of #4 for any data that slipped through | **critical** |
| 6 | 6 | `AgentTaskAssignmentMigration` | `(short_hash, workspace_id)` uniqueness + `assigned_agent_id` index | normal |
| 7 | 7 | `WorkSessionMigration` | `(:WorkSession)` uniqueness + status index | normal |
| 8 | 8 | `BugTrackingMigration` | `(:Bug)` + `(:Review)` + 4 constraints/indexes | normal |
| 9 | 9 | `WorkspaceBackfill` | Bare-name `:Workspace` nodes (2 phases: strip-prefix safety net + paginated backfill) | normal |
| 10 | 10 | `StripWorkspacePrefix` | Strip `session_id:` from Task/Bug `workspace_id` (5 phases) | normal |
| 11 | 11 | `EntityDedupConstraint` | Dedup `(name, entity_type)` Entity duplicates + `entity_name_type_unique` composite constraint (3 phases: paginated dedup, sentinel cleanup, constraint creation) | normal |

## MigrationRunner (runner.py:9)

```python
class MigrationRunner:
    def __init__(self, executor: QueryExecutor):
        self._executor = executor
        self._migrations: list = []

    def register(self, migration) -> None:
        self._migrations.append(migration)
        self._migrations.sort(key=lambda m: m.version)

    def ensure_schema(self) -> None:
        """Create unique constraint on (:_Migration {version})."""

    def get_current_version(self) -> int:
        """MAX(m.version) over (:_Migration)."""

    def run_pending(self) -> list[int]:
        """Run all migrations with version > current, with cross-worker lock."""
```

### `MigrationLock` cross-worker protection (runner.py:64-81)

For each migration, the runner:

```cypher
MERGE (lock:MigrationLock {name: $migration_name})
ON CREATE SET lock.status = 'running', lock.worker_id = $worker_id, lock.started_at = datetime()
ON MATCH SET lock.status = CASE
    WHEN lock.status = 'completed' THEN 'completed'
    WHEN lock.status = 'running' AND lock.worker_id = $worker_id THEN 'running'
    ELSE 'locked'
END
RETURN lock.status AS status
```

If `locked`, the worker SKIPS (another worker is running this migration).
If `completed`, the worker SKIPS (already done). If `running` with the
same `worker_id`, the worker proceeds (recovery from a prior crash).

After `up(executor)`, the runner writes `(:_Migration {version, status: 'completed', completed_at})`.

### B-12 safety guard (runner.py:37-59)

If `current == 0` AND `MATCH (t:Task) RETURN count(t) > 0`, the runner
**refuses** to start. This catches a restore scenario where `_Migration`
markers are missing but tasks exist (re-running on already-migrated data
might be safe today but is dangerous with future non-idempotent migrations).
The error message instructs the user to run the restore script to create
markers or manually MERGE `_Migration` nodes.

### Idempotency contract

All migrations MUST be safe to re-run. Conventions:

- Use `CREATE CONSTRAINT ... IF NOT EXISTS` and `CREATE INDEX ... IF NOT EXISTS`.
- Use `MERGE` for node creation; never `CREATE`.
- For Cypher updates, filter by sentinel property like `t._m004_s0 IS NULL` to skip already-processed rows.
- Use deterministic SHA-256 hashes for IDs where possible.

## Migration 004 — the BIG one (graph_native_tasks.py:11)

6 steps (`up()` at line 15):

| Step | Method | What |
|---|---|---|
| 0 | `_step0_extract_planning_review` | Pull `planning_info` and `review_notes` out of `attributes` JSON, promote to Task properties (sentinel `_m004_s0`) |
| 1 | `_step1_blocked_by_relationships` | Convert `attributes.blocked_by` JSON array → `[:BLOCKED_BY]` relationships (sentinel `_m004_s1`) |
| 2 | `_step2_has_parent_relationships` | Convert `parent_task_uuid` property → `[:HAS_PARENT]` relationships (sentinel `_m004_s2`) |
| 3 | `_step3_promote_tag_remove_entity_type` | Promote `tag` to top-level property, remove `entity_type` property (sentinel `_m004_s3`) |
| 4 | `_step4_remove_attributes` | Drop the `attributes` JSON blob entirely (sentinel `_m004_s4`) |
| 5 | `_step5_indexes_and_constraints` | Create the new graph-native indexes |

Each step is paginated (LIMIT 500) and uses sentinel properties to
guarantee idempotency.

After migration 004, the task graph schema is **graph-native, no JSON
blobs** — see `src/memory/AGENTS.md` §Task Graph Schema.

## Migration 005 — safety net (ensure_graph_native.py)

Re-runs the 004 logic with `count() = 0` exit conditions to catch any
data that slipped through. Less critical than 004 but important for
recovery scenarios.

## Migration 009 — WorkspaceBackfill (2 phases)

```python
class WorkspaceBackfill(Migration):
    version = 9

    def up(self, executor):
        BATCH_SIZE = 5000
        # Phase 1: safety-net strip compound IDs and backfill :Workspace nodes
        result = executor.execute_write(CYPHER_STRIP_AND_BACKFILL)
        # Phase 2: paginated backfill of bare-name :Workspace nodes
        offset = 0
        while True:
            result = executor.execute_write(CYPHER_BACKFILL, {"skip": offset, "limit": BATCH_SIZE})
            count = result[0].get("created", 0)
            if count < BATCH_SIZE:
                break
            offset += BATCH_SIZE

    def down(self, executor):
        pass  # no-op — re-prefixing requires original session_id mapping
```

## Migration 010 — StripWorkspacePrefix (5 phases)

```python
class StripWorkspacePrefix(Migration):
    version = 10

    def up(self, executor):
        BATCH_SIZE = 5000
        # Phase 1: strip Task prefixes (paginated)
        # Phase 2: strip Bug prefixes
        # Phase 3: backfill NULL Task workspace_id to 'default'
        # Phase 3b: backfill NULL Bug workspace_id to 'default'
        # Phase 4: re-key compound :Workspace nodes to bare names
        # Phase 5: delete orphaned compound :Workspace nodes
```

All phases are idempotent. The migration runner executes in version order.
010 runs AFTER 009 on a fresh deploy. The rewritten 009 internally strips
compound prefixes too, so the ordering is safe even on fresh data.

## Key Invariants

1. **All migrations are idempotent.** Use `IF NOT EXISTS` + sentinels.
2. **Migrations run in version order.** `MigrationRunner.register` sorts by `m.version`.
3. **Cross-worker protection via `MigrationLock`.** Two workers cannot run the same migration concurrently.
4. **`(:_Migration {version, status, started_at, completed_at, worker_id})` marker** is written on success.
5. **`down()` is best-effort.** Most migrations have `down()` as no-op because the original session_id mapping isn't preserved.
6. **Migration 004 is irreversible in practice** — `attributes` JSON blob is dropped, `entity_type` removed, `parent_task_uuid` replaced by `[:HAS_PARENT]`.
7. **`create_persist_callback()` is wired into `proxy.state.workspace_store`** so workspace writes also go through Neo4j via the callback path.
8. **Migrations NEVER touch `attributes` JSON post-004.** Code that reads/writes `attributes` is broken by design.

## Landmines

- **NEVER write code that parses `attributes` JSON.** Migration 004 transformed all existing data. Any code expecting the old schema will fail.
- **NEVER set `parent_task_uuid` as a Task property.** Use `[:HAS_PARENT]` relationships instead.
- **NEVER use `MATCH (e:Entity) WHERE 'Task' IN labels(e)`.** Use `MATCH (t:Task)` directly.
- **B-12 safety guard — DON'T disable it.** "Tasks exist but no `_Migration` markers" is a hard refusal. Restore via the documented runbook or manually MERGE markers.
- **Worker ID collision.** UUID4 is collision-resistant in practice; reusing worker_id only matters during a crash recovery within the same migration.
- **`MigrationLock` on a crashed worker.** A worker that crashed mid-migration leaves `lock.status = 'running'`. Another worker with a different `worker_id` sees `'locked'` and skips — the migration is effectively orphaned until manual intervention. **Future hardening**: add `lock.started_at` timeout.
- **`down()` for 004/005/009/010/011 is no-op.** Reverting requires the original session_id mapping (only preserved in backup snapshots).
- **`migration_004`'s `_step0_extract_planning_review` sentinel pattern.** Each step's sentinel (`_m004_s<N>`) marks processed rows. Re-running the same step is safe because the `WHERE ... IS NULL` filter skips already-processed rows.
- **Adding a new migration that needs a new sentinel** must use a unique sentinel (e.g. `_m<N>_<step>`) to avoid colliding with existing migrations.
- **Migration order matters.** `004` MUST run before `005`. `009` and `010` are order-safe due to internal stripping.

## Code Conventions

```python
# CORRECT — migration skeleton
from memory.migrations.base import Migration
from memory.query_executor import QueryExecutor

class MyNewMigration(Migration):
    version = 12
    description = "Short description"

    def up(self, executor: QueryExecutor) -> None:
        executor.execute_write(
            "CREATE CONSTRAINT my_constraint IF NOT EXISTS "
            "FOR (n:MyNode) REQUIRE n.uuid IS UNIQUE"
        )
        logger.info("Migration 012 applied: my_constraint")

# CORRECT — register at import time
from memory.migrations.runner import MigrationRunner
runner = MigrationRunner(executor)
runner.register(MyNewMigration())
runner.ensure_schema()
runner.run_pending()

# WRONG — non-idempotent statements
executor.execute_write("CREATE CONSTRAINT x")  # not IF NOT EXISTS — fails on re-run

# WRONG — writing to attributes JSON after 004
"MATCH (t:Task) SET t.attributes = $attrs"  # schema violation; t.attributes was removed
```

## External Dependencies

**Imports from:**
- `memory.query_executor` — `QueryExecutor` (write/read methods).
- `memory.migrations.base` — `Migration` base class.
- `proxy.utils` — `_now_iso` (timestamp formatting).

**Imported by:**
- `memory.graph.Neo4jClient.init_schema` (or equivalent) — runs migrations on startup.
- `src/proxy/AGENTS.md` (the boot path).
- Tests under `tests/integration/`, `tests/unit/` (for migration safety tests).

## Cross-References

- `src/memory/AGENTS.md` — task graph schema (post-004) + QueryExecutor + singleton driver.
- `src/proxy/AGENTS.md` — boot-time migration runner invocation.
- `tests/AGENTS.md` (planned) — `tests/integration/test_safe_db_name.py`, `tests/integration/test_migration_*.py`.

## Repository artifact policy

Reviewers MUST flag and request rework when a PR commits a non-code artifact without an explicit justification in that originating PR identifying the file and its required purpose. Passing tests or calling a file "evidence" does not waive this rule.

Generated evidence, logs, traces, screenshots, recordings, dumps, and reports belong in the originating PR or issue, NEVER in the repository. Store bulky artifacts in Google Drive and link them from that PR or issue. Do not commit those artifacts as documentation or fixtures to evade this policy.

Source code and tests belong in the repository. Any minimal fixture, documentation, or configuration genuinely required by the implementation must have an explicit file-and-purpose justification in the originating PR; retain only what is required. If that justification is absent, request rework rather than accepting the artifact.

This AGENTS.md policy update is explicitly requested by the repository owner.
