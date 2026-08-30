# Migration Framework

## Overview

ChoriProx uses a versioned migration system for Neo4j schema changes and data transformations. The framework provides:

- **Version tracking**: Migrations are numbered and tracked in Neo4j using `_Migration` nodes
- **Concurrent worker protection**: Multiple instances can start simultaneously without race conditions
- **Idempotent execution**: Already-completed migrations are skipped automatically
- **Failure handling**: Failed migrations are marked with error details for debugging
- **Rollback support**: Optional `down()` methods for reversible migrations

## Architecture

### Components

- **`Migration` base class** (`src/memory/migrations/base.py`): Defines the interface with `version`, `description`, `up()`, and optional `down()` methods
- **`MigrationRunner`** (`src/memory/migrations/runner.py`): Executes migrations in order, tracks state, handles failures
- **Migration files** (`src/memory/migrations/NNN_*.py`): Individual migration implementations

### State Tracking

Migrations are tracked in Neo4j with `_Migration` nodes:
```cypher
(m:_Migration {
  version: 1,
  status: 'completed',  // 'running', 'completed', 'failed'
  started_at: datetime(),
  completed_at: datetime(),
  error: '...'  // only if failed
})
```

## Creating a New Migration

### Step 1: Create the Migration File

Create `src/memory/migrations/NNN_description.py` where `NNN` is the next version number:

```python
"""Migration NNN: Brief description of what this migration does."""
from memory.migrations.base import Migration
from memory.query_executor import QueryExecutor


class MigrationName(Migration):
    version = 3  # Next version number
    description = "Add new index for performance"

    def up(self, executor: QueryExecutor) -> None:
        executor.execute_write(
            "CREATE INDEX entity_name_index IF NOT EXISTS "
            "FOR (e:Entity) ON (e.name)"
        )

    def down(self, executor: QueryExecutor) -> None:
        executor.execute_write("DROP INDEX entity_name_index IF EXISTS")
```

### Step 2: Register the Migration

Migrations must be registered with the `MigrationRunner` at startup. Check `src/memory/migrations/__init__.py` or the application startup code to see where migrations are registered:

```python
from memory.migrations.runner import MigrationRunner
from memory.migrations.001_backfill_parent_task_uuid import BackfillParentTaskUuid
from memory.migrations.002_verify_backfill import VerifyBackfill
from memory.migrations.003_description import MigrationName

runner = MigrationRunner(executor)
runner.register(BackfillParentTaskUuid())
runner.register(VerifyBackfill())
runner.register(MigrationName())
runner.run_pending()
```

## Naming Conventions

- **File format**: `NNN_description.py` (e.g., `001_backfill_parent_task_uuid.py`)
- **Class name**: PascalCase descriptive name (e.g., `BackfillParentTaskUuid`)
- **Version number**: Integer, must be unique and sequential
- **Description**: Brief, clear explanation of what the migration does

## How Migrations Run

### At Startup

1. Application creates a `MigrationRunner` with a `QueryExecutor`
2. Runner calls `ensure_schema()` to create the `_Migration` constraint
3. Runner calls `get_current_version()` to find the latest completed migration
4. Runner iterates through registered migrations in version order
5. For each migration with `version > current_version`:
   - Creates/updates `_Migration` node with `status = 'running'`
   - Checks if already completed (concurrent worker protection)
   - Calls `migration.up(executor)`
   - On success: sets `status = 'completed'`
   - On failure: sets `status = 'failed'` with error message, raises exception

### Concurrent Worker Protection

The runner uses a MERGE statement to atomically claim migrations:

```cypher
MERGE (m:_Migration {version: $v})
  ON CREATE SET m.status = 'running', m.started_at = datetime()
  ON MATCH SET m.status = CASE
    WHEN m.status = 'completed' THEN 'completed'
    ELSE 'running'
  END
  RETURN m.status AS status
```

If another worker already completed the migration, the status remains `'completed'` and the migration is skipped.

## Testing Migrations

### Unit Testing

Test migrations in isolation with a mock or test Neo4j instance:

```python
import pytest
from memory.migrations.003_description import MigrationName
from unittest.mock import Mock


def test_migration_003_creates_index():
    executor = Mock()
    migration = MigrationName()
    
    migration.up(executor)
    
    executor.execute_write.assert_called_once()
    call_args = executor.execute_write.call_args[0][0]
    assert "CREATE INDEX" in call_args


def test_migration_003_rollback():
    executor = Mock()
    migration = MigrationName()
    
    migration.down(executor)
    
    executor.execute_write.assert_called_once()
    call_args = executor.execute_write.call_args[0][0]
    assert "DROP INDEX" in call_args
```

### Integration Testing

Test against a real Neo4j test database:

```python
def test_migration_003_integration(test_executor):
    migration = MigrationName()
    
    migration.up(test_executor)
    
    result = test_executor.execute_read(
        "SHOW INDEXES WHERE name = 'entity_name_index'"
    )
    assert len(result) > 0
```

### Testing Migration Runner

```python
def test_runner_skips_completed_migrations(test_executor):
    test_executor.execute_write(
        "CREATE (m:_Migration {version: 1, status: 'completed'})"
    )
    
    runner = MigrationRunner(test_executor)
    runner.register(BackfillParentTaskUuid())
    runner.register(VerifyBackfill())
    
    applied = runner.run_pending()
    
    assert 1 not in applied
    assert 2 in applied
```

## Rollback Procedures

### Automatic Rollback

If a migration implements `down()`, it can be rolled back:

```python
def rollback_migration(version: int, executor: QueryExecutor):
    from memory.migrations import get_migration_by_version
    
    migration = get_migration_by_version(version)
    if not migration:
        raise ValueError(f"Migration {version} not found")
    
    migration.down(executor)
    
    executor.execute_write(
        "MATCH (m:_Migration {version: $v}) DELETE m",
        {"v": version}
    )
```

### Manual Rollback

For migrations without `down()` or when automatic rollback fails:

1. **Identify the migration version** that needs rollback
2. **Manually reverse the changes** using Cypher queries
3. **Delete the migration record**:
   ```cypher
   MATCH (m:_Migration {version: 3}) DELETE m
   ```
4. **Document the manual rollback** in the migration file comments

### Rollback Best Practices

- **Always implement `down()`** for reversible migrations
- **Test rollback procedures** before deploying to production
- **Backup data** before running irreversible migrations
- **Document manual rollback steps** in migration file comments
- **Consider data loss**: Some migrations (e.g., data deletions) cannot be safely rolled back

## Current Migrations

### Migration 001: Backfill Parent Task UUID

**File**: `src/memory/migrations/001_backfill_parent_task_uuid.py`  
**Class**: `BackfillParentTaskUuid`  
**Description**: Copies `parent_task_uuid` from JSON `attributes` field to top-level property

**What it does**:
- Finds all `Task` entities with `parent_task_uuid` in JSON attributes
- Extracts the value and sets it as a top-level property
- Processes in batches of 500 to avoid memory issues
- Sets empty string for tasks with null/missing parent_task_uuid

**Why**: Top-level properties are faster to query than JSON parsing

**Rollback**: Not implemented (data transformation is idempotent)

### Migration 002: Verify Backfill

**File**: `src/memory/migrations/002_verify_backfill.py`  
**Class**: `VerifyBackfill`  
**Description**: Verifies all tasks have `parent_task_uuid` as top-level property

**What it does**:
- Counts tasks that still need backfill
- Raises `RuntimeError` if any tasks remain unbackfilled
- Acts as a safety check after migration 001

**Why**: Ensures data integrity before proceeding with dependent migrations

**Rollback**: Not needed (verification only)

### Migration 011: Entity Dedup Constraint

**File**: `src/memory/migrations/011_entity_dedup_constraint.py`  
**Class**: `EntityDedupConstraint`  
**Description**: Dedups duplicate `(name, entity_type)` Entity nodes and creates the `entity_name_type_unique` composite uniqueness constraint

**What it does** (3 phases):
- Paginated dedup of duplicate `(name, entity_type)` groups — sentinel-guarded and atomic per-group
- Sentinel cleanup
- Constraint creation (`entity_name_type_unique`)

**Idempotent**: yes — sentinel `_m011_dedup` plus `IF NOT EXISTS` on the constraint

**Note**: The constraint is unfiltered (Neo4j 2026.05 does not support `WHERE` on `CREATE CONSTRAINT`); Task/Bug nodes have NULL `entity_type` and are auto-excluded.

**Rollback**: Not implemented

## Troubleshooting

### Migration Stuck in 'running' State

If a migration crashes without updating status:

```cypher
MATCH (m:_Migration {version: 3, status: 'running'})
SET m.status = 'failed', m.error = 'Manual reset after crash'
```

Then re-run the application to retry the migration.

### Migration Failed with Error

Check the error message:

```cypher
MATCH (m:_Migration {version: 3})
RETURN m.version, m.status, m.error
```

Fix the underlying issue, reset the migration:

```cypher
MATCH (m:_Migration {version: 3})
SET m.status = 'failed'
```

Then re-run the application.

### Concurrent Migration Conflicts

If multiple workers are running migrations:

1. Check which migrations are running:
   ```cypher
   MATCH (m:_Migration {status: 'running'})
   RETURN m.version, m.started_at
   ```

2. Wait for completion or manually reset stuck migrations:
   ```cypher
   MATCH (m:_Migration {version: 3, status: 'running'})
   WHERE m.started_at < datetime() - duration({minutes: 10})
   SET m.status = 'failed', m.error = 'Timeout - manual reset'
   ```

## Best Practices

1. **Keep migrations small and focused**: One logical change per migration
2. **Test thoroughly**: Unit test, integration test, and test rollback
3. **Make migrations idempotent**: Safe to run multiple times
4. **Document breaking changes**: Note if migration requires downtime
5. **Consider performance**: Large data migrations should batch operations
6. **Always implement `down()`**: Unless the migration is truly irreversible
7. **Use transactions**: Wrap related changes in a single transaction when possible
8. **Log progress**: For long-running migrations, log batch progress
9. **Review before deploying**: Have another developer review migration code
10. **Backup before production**: Always backup Neo4j before running migrations
