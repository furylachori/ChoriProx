"""Insurance tests for the ``MigrationRunner.run_pending`` safety guard (B-12).

The safety guard in ``src/memory/migrations/runner.py:34-60`` refuses to run
migrations when:

    current == 0 (no ``_Migration`` markers) AND
    there are existing ``:Task`` nodes in the graph

A buggy or absent guard would silently re-run migrations on data that may
already be partially migrated, which is currently safe but would be
DANGEROUS with a future non-idempotent migration. The guard also serves
as an early warning that the database is in an inconsistent state (e.g.
restored from a backup without ``_Migration`` markers).

These tests pin the contract:

1. ``current == 0`` AND tasks exist → raises ``RuntimeError``.
2. ``current == 0`` AND no tasks → proceeds normally.
3. ``current > 0`` → guard does not fire (even if tasks exist).
4. ``current == 0`` AND the task-count ``execute_read`` raises an
   InfrastructureError-style exception → guard does NOT block
   (we don't want a transient read error to halt the proxy).
5. ``RuntimeError`` message is actionable: includes the task count
   and points at the recovery procedure (restore script or manual MERGE).
"""
import unittest
from unittest.mock import MagicMock

from memory.migrations.runner import MigrationRunner


class _StubMigration:
    def __init__(self, version=1, description="stub"):
        self.version = version
        self.description = description
        self.up_calls = 0

    def up(self, executor):
        self.up_calls += 1


def _make_executor(version=None, task_count=None, read_side_effect=None):
    """Build a mock executor with controlled read responses.

    The first ``execute_read`` call is treated as the version query
    (``MATCH (m:_Migration) RETURN max(m.version) AS version``); the second
    is the task-count query (``MATCH (t:Task) RETURN count(t) AS c``).

    Pass ``read_side_effect`` to override the default routing (e.g. for
    transient-error tests).
    """
    executor = MagicMock()
    if read_side_effect is not None:
        executor.execute_read.side_effect = read_side_effect
        return executor

    def route(query, params=None, **kwargs):
        if "_Migration" in query and "max(m.version)" in query:
            return [{"version": version}]
        if "MATCH (t:Task)" in query:
            return [{"c": task_count}]
        return []

    executor.execute_read.side_effect = route
    return executor


class TestSafetyGuardTrigger(unittest.TestCase):
    """Cases where the safety guard fires."""

    def test_current_zero_with_tasks_raises(self):
        """current=0 and tasks>0 → RuntimeError (no migration runs)."""
        executor = _make_executor(version=0, task_count=5)
        runner = MigrationRunner(executor)
        migration = _StubMigration(version=1)
        runner.register(migration)

        with self.assertRaises(RuntimeError) as ctx:
            runner.run_pending()

        self.assertEqual(migration.up_calls, 0)
        self.assertIn("Tasks exist but no _Migration markers", str(ctx.exception))

    def test_current_zero_with_one_task_raises(self):
        """Edge case: exactly one task present → still raises."""
        executor = _make_executor(version=0, task_count=1)
        runner = MigrationRunner(executor)
        runner.register(_StubMigration(version=1))

        with self.assertRaises(RuntimeError):
            runner.run_pending()

    def test_error_message_includes_task_count(self):
        """The CRITICAL log line includes the count of tasks (the RuntimeError message is fixed text)."""
        executor = _make_executor(version=0, task_count=42)
        runner = MigrationRunner(executor)
        runner.register(_StubMigration(version=1))

        with self.assertLogs("memory.migrations.runner", level="CRITICAL") as cm:
            with self.assertRaises(RuntimeError):
                runner.run_pending()

        critical_msgs = [m for m in cm.output if "Migration version is 0" in m]
        self.assertEqual(len(critical_msgs), 1)
        self.assertIn("42", critical_msgs[0])

    def test_error_message_mentions_recovery(self):
        """The RuntimeError message must point at the recovery procedure."""
        executor = _make_executor(version=0, task_count=10)
        runner = MigrationRunner(executor)
        runner.register(_StubMigration(version=1))

        with self.assertRaises(RuntimeError) as ctx:
            runner.run_pending()

        message = str(ctx.exception).lower()
        self.assertTrue(
            "restore" in message or "merge" in message,
            f"error message should mention recovery; got: {ctx.exception!r}",
        )

    def test_guard_fires_logs_critical(self):
        """The CRITICAL log line includes the task count."""
        executor = _make_executor(version=0, task_count=7)
        runner = MigrationRunner(executor)
        runner.register(_StubMigration(version=1))

        with self.assertLogs("memory.migrations.runner", level="CRITICAL") as cm:
            with self.assertRaises(RuntimeError):
                runner.run_pending()

        critical_msgs = [m for m in cm.output if "Migration version is 0" in m]
        self.assertEqual(len(critical_msgs), 1)
        self.assertIn("7", critical_msgs[0])

    def test_guard_does_not_acquire_migration_lock(self):
        """When guard fires, no MigrationLock is acquired."""
        executor = _make_executor(version=0, task_count=3)
        runner = MigrationRunner(executor)
        runner.register(_StubMigration(version=1))

        with self.assertRaises(RuntimeError):
            runner.run_pending()

        lock_queries = [
            c for c in executor.execute_write.call_args_list
            if "MigrationLock" in str(c)
        ]
        self.assertEqual(
            lock_queries,
            [],
            f"guard should prevent lock acquisition; got: {lock_queries!r}",
        )


class TestSafetyGuardDoesNotFire(unittest.TestCase):
    """Cases where the safety guard does NOT fire."""

    def test_current_zero_with_no_tasks_proceeds(self):
        """current=0, no tasks → guard does not fire, migrations run normally."""
        executor = _make_executor(version=0, task_count=0)
        runner = MigrationRunner(executor)
        migration = _StubMigration(version=1)
        runner.register(migration)

        result = runner.run_pending()

        self.assertEqual(result, [1])
        self.assertEqual(migration.up_calls, 1)

    def test_current_positive_skips_task_count_check(self):
        """current>0 → guard does not even read the task count."""
        executor = MagicMock()
        executor.execute_read.return_value = [{"version": 5}]
        runner = MigrationRunner(executor)
        migration = _StubMigration(version=6)
        runner.register(migration)

        result = runner.run_pending()

        self.assertEqual(result, [6])
        self.assertEqual(migration.up_calls, 1)
        for c in executor.execute_read.call_args_list:
            query = c.args[0] if c.args else ""
            self.assertNotIn("MATCH (t:Task)", str(query))

    def test_current_positive_with_tasks_does_not_block(self):
        """current>0 AND tasks present → guard does not block (not a concern)."""
        executor = MagicMock()
        executor.execute_read.return_value = [{"version": 3}]
        runner = MigrationRunner(executor)
        migration = _StubMigration(version=4)
        runner.register(migration)

        result = runner.run_pending()

        self.assertEqual(result, [4])
        self.assertEqual(migration.up_calls, 1)

    def test_no_migrations_registered_proceeds(self):
        """No migrations → guard fires only if current==0 AND tasks exist."""
        executor = _make_executor(version=0, task_count=0)
        runner = MigrationRunner(executor)

        result = runner.run_pending()

        self.assertEqual(result, [])

    def test_transient_read_error_does_not_block(self):
        """A generic exception in the task-count read does not block the runner.

        Rationale: we don't want a transient Neo4j blip (timeouts, transient
        connection drops) to prevent migrations from running on first start.
        The guard is about catching an INCONSISTENT state, not about blocking
        on read errors. Note that ``RuntimeError`` is explicitly re-raised
        (see ``except RuntimeError: raise`` in the guard) — only non-RuntimeError
        exceptions are swallowed.
        """
        executor = MagicMock()
        executor.execute_read.side_effect = [
            [{"version": 0}],
            ConnectionError("transient: connection reset"),
        ]
        executor.execute_write.return_value = [{"status": "running"}]
        runner = MigrationRunner(executor)
        migration = _StubMigration(version=1)
        runner.register(migration)

        result = runner.run_pending()

        self.assertEqual(migration.up_calls, 1)
        self.assertEqual(result, [1])

    def test_infrastructure_error_does_not_block(self):
        """An InfrastructureError-shaped exception is treated as transient."""
        executor = MagicMock()
        executor.execute_read.side_effect = [
            [{"version": 0}],
            TimeoutError("Neo4j query timeout"),
        ]
        executor.execute_write.return_value = [{"status": "running"}]
        runner = MigrationRunner(executor)
        migration = _StubMigration(version=1)
        runner.register(migration)

        result = runner.run_pending()

        self.assertEqual(migration.up_calls, 1)
        self.assertEqual(result, [1])

    def test_runtime_error_from_guard_is_not_swallowed(self):
        """A RuntimeError raised by the guard propagates (not the catch-all except)."""
        executor = _make_executor(version=0, task_count=99)
        runner = MigrationRunner(executor)
        runner.register(_StubMigration(version=1))

        with self.assertRaises(RuntimeError) as ctx:
            runner.run_pending()

        self.assertIn("Tasks exist but no _Migration markers", str(ctx.exception))

    def test_empty_task_count_response_proceeds(self):
        """An empty task-count response (no rows) is treated as zero tasks."""
        executor = MagicMock()
        executor.execute_read.side_effect = [
            [{"version": 0}],
            [],
            [{"status": "running"}],
            [{"status": "running"}],
        ]
        runner = MigrationRunner(executor)
        migration = _StubMigration(version=1)
        runner.register(migration)

        result = runner.run_pending()

        self.assertEqual(migration.up_calls, 1)
        self.assertEqual(result, [1])

    def test_task_count_row_with_null_c_proceeds(self):
        """A row with c=None (unlikely, but defensive) is treated as zero."""
        executor = MagicMock()
        executor.execute_read.side_effect = [
            [{"version": 0}],
            [{"c": None}],
            [{"status": "running"}],
            [{"status": "running"}],
        ]
        runner = MigrationRunner(executor)
        migration = _StubMigration(version=1)
        runner.register(migration)

        result = runner.run_pending()

        self.assertEqual(migration.up_calls, 1)
        self.assertEqual(result, [1])


class TestSafetyGuardOrdering(unittest.TestCase):
    """The guard fires only ONCE per run_pending call."""

    def test_guard_runs_before_migration_loop(self):
        """Task count is read BEFORE any migration is attempted."""
        call_log = []

        executor = MagicMock()
        executor.execute_read.side_effect = [
            [{"version": 0}],
            [{"c": 5}],
        ]
        runner = MigrationRunner(executor)
        migration = _StubMigration(version=1)
        original_up = migration.up
        migration.up = MagicMock(side_effect=lambda exc: (call_log.append("up"), original_up(exc))[1])
        runner.register(migration)

        def record_read(query, *args, **kwargs):
            call_log.append(("read", str(query)[:30]))
            if "_Migration" in str(query) and "max(m.version)" in str(query):
                return [{"version": 0}]
            if "MATCH (t:Task)" in str(query):
                return [{"c": 5}]
            return []

        executor.execute_read.side_effect = record_read

        with self.assertRaises(RuntimeError):
            runner.run_pending()

        read_indices = [i for i, entry in enumerate(call_log) if entry[0] == "read"]
        up_indices = [i for i, entry in enumerate(call_log) if entry == "up"]
        self.assertEqual(len(read_indices), 2)
        self.assertEqual(len(up_indices), 0)
        self.assertLess(read_indices[1], 100)
