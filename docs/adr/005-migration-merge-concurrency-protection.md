# ADR-005: Migration MERGE for concurrency protection

## Status
Accepted (2026-06-02)

## Context

ChoriProx runs under Gunicorn with multiple worker processes. Each worker
calls `MigrationRunner.run_pending()` at startup to apply any pending Neo4j
schema migrations.

With N workers starting simultaneously, a race condition exists:

1. Worker A reads current version = 0, decides to run migration 1.
2. Worker B reads current version = 0, decides to run migration 1.
3. Both workers execute migration 1's `up()` method concurrently.
4. Migration 1 creates constraints and indexes. Running it twice causes
   `ConstraintAlreadyExists` errors or, worse, partial application where
   one worker's transaction conflicts with the other's.

The naive approach of checking version before running (read-then-write) is
a classic TOCTOU race. File-based locks don't work across processes on
all platforms, and external coordination (e.g., Redis lock) adds operational
dependencies.

## Decision

Use Neo4j's `MERGE ... ON CREATE ... ON MATCH` as an atomic compare-and-swap
for migration status tracking. The migration runner in
`src/memory/migrations/runner.py` uses this pattern:

```cypher
MERGE (m:_Migration {version: $v})
ON CREATE SET m.status = 'running', m.started_at = datetime()
ON MATCH SET m.status = CASE
    WHEN m.status = 'completed' THEN 'completed'
    ELSE 'running'
END
RETURN m.status AS status
```

The logic:

- **First worker** to reach `MERGE` creates the node with `status = 'running'`
  (ON CREATE branch). It proceeds to run the migration.
- **Subsequent workers** hit the ON MATCH branch. If status is already
  `'completed'`, they skip the migration. If status is `'running'` or
  `'failed'`, they set it to `'running'` but the first worker's transaction
  already committed, so the migration body runs only once.

A uniqueness constraint on `_Migration.version` ensures the MERGE operates
on exactly one node per version:

```cypher
CREATE CONSTRAINT migration_version_unique IF NOT EXISTS
FOR (m:_Migration) REQUIRE m.version IS UNIQUE
```

## Consequences

**Easier:**
- No external coordination mechanism needed. Neo4j's transaction isolation
  handles the race.
- Migration status is queryable at runtime (`MATCH (m:_Migration) RETURN
  m.version, m.status`), providing operational visibility.
- Failed migrations record their error message on the `_Migration` node,
  aiding debugging.

**More difficult:**
- Requires Neo4j 4.x+ for `MERGE` with `ON CREATE`/`ON MATCH` support.
  Older Neo4j versions cannot run ChoriProx's migration framework.
- The `MERGE` pattern is non-obvious to developers unfamiliar with Cypher.
  The runner includes comments explaining the concurrency semantics.
- Adding migration rollback requires careful handling of the `_Migration`
  node status (resetting from `'completed'` to allow re-run).

## Alternatives Considered

1. **Advisory file lock** — `fcntl.flock()` around migration execution.
   Rejected because it doesn't work on all platforms (Windows) and adds
   a file-system dependency.

2. **Single migration worker** — Only one designated worker runs migrations.
   Rejected because it requires worker identity management and doesn't work
   with auto-scaling or rolling deployments.

3. **Idempotent migrations only** — Design all migrations to be safely
   re-runnable. Rejected because some schema changes (e.g., backfill
   operations) are not naturally idempotent and would require complex
   "already done" checks within each migration.
