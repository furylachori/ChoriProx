import pytest
import importlib
from unittest.mock import MagicMock
from memory.migrations.runner import MigrationRunner
from memory.migrations.base import Migration


class TestMigrationRunner:
    def test_runner_runs_migrations_in_version_order(self):
        executor = MagicMock()

        def mock_execute_write(query, params=None, timeout=30):
            if 'MERGE' in query:
                return [{'status': 'running'}]
            return []

        executor.execute_write.side_effect = mock_execute_write
        executor.execute_read.return_value = [{'version': 0}]

        runner = MigrationRunner(executor)

        call_order = []

        m1 = MagicMock()
        m1.version = 1
        m1.description = "Migration 1"
        m1.up.side_effect = lambda e: call_order.append(1)

        m2 = MagicMock()
        m2.version = 2
        m2.description = "Migration 2"
        m2.up.side_effect = lambda e: call_order.append(2)

        m3 = MagicMock()
        m3.version = 3
        m3.description = "Migration 3"
        m3.up.side_effect = lambda e: call_order.append(3)

        runner.register(m2)
        runner.register(m1)
        runner.register(m3)

        applied = runner.run_pending()

        assert applied == [1, 2, 3]
        assert call_order == [1, 2, 3]

    def test_runner_skips_completed_migrations(self):
        executor = MagicMock()

        def mock_execute_write(query, params=None, timeout=30):
            if 'MERGE' in query:
                return [{'status': 'running'}]
            return []

        executor.execute_write.side_effect = mock_execute_write
        executor.execute_read.return_value = [{'version': 1}]

        runner = MigrationRunner(executor)

        m1 = MagicMock()
        m1.version = 1
        m1.description = "Migration 1"

        m2 = MagicMock()
        m2.version = 2
        m2.description = "Migration 2"

        runner.register(m1)
        runner.register(m2)

        applied = runner.run_pending()

        assert applied == [2]
        m1.up.assert_not_called()
        m2.up.assert_called_once()

    def test_runner_merge_prevents_concurrent_workers(self):
        executor = MagicMock()

        def mock_execute_write(query, params=None, timeout=30):
            if 'MERGE' in query:
                return [{'status': 'completed'}]
            return []

        executor.execute_write.side_effect = mock_execute_write
        executor.execute_read.return_value = [{'version': 0}]

        runner = MigrationRunner(executor)

        m1 = MagicMock()
        m1.version = 1
        m1.description = "Migration 1"

        runner.register(m1)

        applied = runner.run_pending()

        assert applied == []
        m1.up.assert_not_called()

    def test_runner_marks_failed_migration(self):
        """# US-037"""
        executor = MagicMock()

        def mock_execute_write(query, params=None, timeout=30):
            if 'MERGE' in query:
                return [{'status': 'running'}]
            return []

        executor.execute_write.side_effect = mock_execute_write
        executor.execute_read.return_value = [{'version': 0}]

        runner = MigrationRunner(executor)

        m1 = MagicMock()
        m1.version = 1
        m1.description = "Migration 1"
        m1.up.side_effect = Exception("Migration failed!")

        runner.register(m1)

        with pytest.raises(Exception, match="Migration failed!"):
            runner.run_pending()

        call_strings = [str(c) for c in executor.execute_write.call_args_list]
        assert any('failed' in s for s in call_strings)

    def test_ensure_schema_creates_constraint(self):
        executor = MagicMock()
        runner = MigrationRunner(executor)

        runner.ensure_schema()

        executor.execute_write.assert_called_once()
        call_args = executor.execute_write.call_args
        assert 'CREATE CONSTRAINT' in call_args[0][0]
        assert 'migration_version_unique' in call_args[0][0]

    def test_get_current_version_returns_zero_when_empty(self):
        executor = MagicMock()
        executor.execute_read.return_value = [{'version': None}]

        runner = MigrationRunner(executor)

        assert runner.get_current_version() == 0

    def test_get_current_version_returns_max_version(self):
        executor = MagicMock()
        executor.execute_read.return_value = [{'version': 5}]

        runner = MigrationRunner(executor)

        assert runner.get_current_version() == 5


class TestMigration001:
    def test_migration_001_backfills_parent_task_uuid(self):
        mod = importlib.import_module('memory.migrations.001_backfill_parent_task_uuid')
        BackfillParentTaskUuid = mod.BackfillParentTaskUuid

        executor = MagicMock()
        executor.execute_read.side_effect = [
            [
                {'nid': 1, 'attrs': '{"parent_task_uuid": "uuid-123"}'},
                {'nid': 2, 'attrs': '{"parent_task_uuid": "uuid-456"}'},
            ],
            [],
        ]

        migration = BackfillParentTaskUuid()
        migration.up(executor)

        assert executor.execute_write_no_retry.call_count == 1
        call_args = executor.execute_write_no_retry.call_args
        params = call_args[0][1]
        assert params['updates'] == [
            {'nid': 1, 'ptu': 'uuid-123'},
            {'nid': 2, 'ptu': 'uuid-456'},
        ]

    def test_migration_001_terminates_on_null_values(self):
        mod = importlib.import_module('memory.migrations.001_backfill_parent_task_uuid')
        BackfillParentTaskUuid = mod.BackfillParentTaskUuid

        executor = MagicMock()
        executor.execute_read.return_value = [
            {'nid': 1, 'attrs': '{"parent_task_uuid": null}'},
            {'nid': 2, 'attrs': '{"other_field": "value"}'},
        ]

        migration = BackfillParentTaskUuid()
        migration.up(executor)

        assert executor.execute_write_no_retry.call_count == 1
        call_args = executor.execute_write_no_retry.call_args
        query = call_args[0][0]
        params = call_args[0][1]
        assert "parent_task_uuid" in query
        assert "''" in query
        updates = params.get("updates", [])
        assert len(updates) == 2
        assert {u["nid"] for u in updates} == {1, 2}

    def test_migration_001_terminates_on_empty_batch(self):
        mod = importlib.import_module('memory.migrations.001_backfill_parent_task_uuid')
        BackfillParentTaskUuid = mod.BackfillParentTaskUuid

        executor = MagicMock()
        executor.execute_read.return_value = []

        migration = BackfillParentTaskUuid()
        migration.up(executor)

        executor.execute_write_no_retry.assert_not_called()


class TestMigration002:
    def test_migration_002_raises_on_unbackfilled(self):
        mod = importlib.import_module('memory.migrations.002_verify_backfill')
        VerifyBackfill = mod.VerifyBackfill

        executor = MagicMock()
        executor.execute_read.return_value = [{'unbackfilled': 5}]

        migration = VerifyBackfill()

        with pytest.raises(RuntimeError, match="5 tasks still need backfill"):
            migration.up(executor)

    def test_migration_002_passes_when_all_backfilled(self):
        mod = importlib.import_module('memory.migrations.002_verify_backfill')
        VerifyBackfill = mod.VerifyBackfill

        executor = MagicMock()
        executor.execute_read.return_value = [{'unbackfilled': 0}]

        migration = VerifyBackfill()
        migration.up(executor)

        executor.execute_read.assert_called_once()
        query = executor.execute_read.call_args[0][0]
        assert "parent_task_uuid" in query
        assert "count(" in query
        assert executor.execute_write.call_count == 0
        assert executor.execute_write_no_retry.call_count == 0
