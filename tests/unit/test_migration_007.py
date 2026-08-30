"""Tests for migration 007: Work session graph."""
import importlib
import inspect

import pytest

from memory.migrations.base import Migration
from memory.migrations.runner import MigrationRunner


class TestMigration007Import:
    def test_can_import_migration_module(self):
        mod = importlib.import_module("memory.migrations.007_work_sessions")
        assert mod is not None

    def test_migration_class_exported(self):
        mod = importlib.import_module("memory.migrations.007_work_sessions")
        assert hasattr(mod, "WorkSessionMigration")

    def test_migration_version_is_seven(self):
        mod = importlib.import_module("memory.migrations.007_work_sessions")
        cls = mod.WorkSessionMigration
        assert cls.version == 7

    def test_migration_inherits_from_migration_base(self):
        mod = importlib.import_module("memory.migrations.007_work_sessions")
        cls = mod.WorkSessionMigration
        assert issubclass(cls, Migration)

    def test_migration_description_mentions_work_session(self):
        mod = importlib.import_module("memory.migrations.007_work_sessions")
        cls = mod.WorkSessionMigration
        assert "work session" in cls.description.lower() or "worksession" in cls.description.lower()

    def test_migration_registered_in_package_init(self):
        from memory.migrations import WorkSessionMigration
        assert WorkSessionMigration is not None
        assert WorkSessionMigration.version == 7


class TestMigration007Up:
    def test_up_executes_two_statements(self):
        """up() must issue exactly two write calls: constraint + index."""
        mod = importlib.import_module("memory.migrations.007_work_sessions")
        calls = []

        class FakeExecutor:
            def execute_write(self, query, params=None, timeout=30):
                calls.append(query)
                return []

        migration = mod.WorkSessionMigration()
        migration.up(FakeExecutor())

        assert len(calls) == 2
        constraint_q, index_q = calls
        assert "CREATE CONSTRAINT work_session_uuid_unique IF NOT EXISTS" in constraint_q
        assert "FOR (ws:WorkSession) REQUIRE ws.uuid IS UNIQUE" in constraint_q
        assert "CREATE INDEX work_session_status IF NOT EXISTS" in index_q
        assert "(ws:WorkSession)" in index_q
        assert "(ws.status)" in index_q

    def test_up_is_idempotent_via_if_not_exists(self):
        """The statements include IF NOT EXISTS so re-running is safe."""
        mod = importlib.import_module("memory.migrations.007_work_sessions")
        migration = mod.WorkSessionMigration()
        src = inspect.getsource(migration.up)
        assert src.count("IF NOT EXISTS") == 2

    def test_up_propagates_infrastructure_error(self):
        """A failing write must not be silently swallowed by up()."""
        from memory.query_executor import InfrastructureError

        mod = importlib.import_module("memory.migrations.007_work_sessions")

        class FakeExecutor:
            def execute_write(self, query, params=None, timeout=30):
                raise InfrastructureError("simulated failure")

        migration = mod.WorkSessionMigration()
        with pytest.raises(InfrastructureError):
            migration.up(FakeExecutor())


class TestMigration007Registration:
    def test_migration_can_be_registered_with_runner(self):
        """Confirm the runner accepts the migration and sorts by version."""
        mod = importlib.import_module("memory.migrations.007_work_sessions")

        class FakeExecutor:
            pass

        runner = MigrationRunner(FakeExecutor())
        runner.register(mod.WorkSessionMigration())
        assert any(m.version == 7 for m in runner._migrations)
