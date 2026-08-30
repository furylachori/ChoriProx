"""Tests for migration 006: Agent-task assignment foundation."""
import importlib
import inspect

import pytest

from memory.migrations.base import Migration
from memory.migrations.runner import MigrationRunner
from memory.query_executor import QueryExecutor


class TestMigration006Import:
    def test_can_import_migration_module(self):
        mod = importlib.import_module("memory.migrations.006_agent_task_assignment")
        assert mod is not None

    def test_migration_class_exported(self):
        mod = importlib.import_module("memory.migrations.006_agent_task_assignment")
        assert hasattr(mod, "AgentTaskAssignmentMigration")

    def test_migration_version_is_six(self):
        mod = importlib.import_module("memory.migrations.006_agent_task_assignment")
        cls = mod.AgentTaskAssignmentMigration
        assert cls.version == 6

    def test_migration_inherits_from_migration_base(self):
        mod = importlib.import_module("memory.migrations.006_agent_task_assignment")
        cls = mod.AgentTaskAssignmentMigration
        assert issubclass(cls, Migration)

    def test_migration_description_mentions_assignment(self):
        mod = importlib.import_module("memory.migrations.006_agent_task_assignment")
        cls = mod.AgentTaskAssignmentMigration
        assert "assignment" in cls.description.lower()

    def test_migration_registered_in_package_init(self):
        from memory.migrations import AgentTaskAssignmentMigration
        assert AgentTaskAssignmentMigration is not None
        assert AgentTaskAssignmentMigration.version == 6


class TestMigration006Up:
    def test_up_executes_two_statements(self, monkeypatch):
        """up() must issue exactly two write calls: constraint + index."""
        mod = importlib.import_module("memory.migrations.006_agent_task_assignment")
        calls = []

        class FakeExecutor:
            def execute_write(self, query, params=None, timeout=30):
                calls.append(query)
                return []

        migration = mod.AgentTaskAssignmentMigration()
        migration.up(FakeExecutor())

        assert len(calls) == 2
        constraint_q, index_q = calls
        assert "CREATE CONSTRAINT task_short_hash_workspace_unique IF NOT EXISTS" in constraint_q
        assert "(t.short_hash, t.workspace_id) IS UNIQUE" in constraint_q
        assert "CREATE INDEX task_assigned_agent IF NOT EXISTS" in index_q
        assert "t.assigned_agent_id" in index_q

    def test_up_is_idempotent_via_if_not_exists(self):
        """The statements include IF NOT EXISTS so re-running is safe."""
        mod = importlib.import_module("memory.migrations.006_agent_task_assignment")
        migration = mod.AgentTaskAssignmentMigration()
        src = inspect.getsource(migration.up)
        assert src.count("IF NOT EXISTS") == 2

    def test_up_propagates_infrastructure_error(self):
        """A failing write must not be silently swallowed by up()."""
        from memory.query_executor import InfrastructureError

        mod = importlib.import_module("memory.migrations.006_agent_task_assignment")

        class FakeExecutor:
            def execute_write(self, query, params=None, timeout=30):
                raise InfrastructureError("simulated failure")

        migration = mod.AgentTaskAssignmentMigration()
        with pytest.raises(InfrastructureError):
            migration.up(FakeExecutor())


class TestMigration006Registration:
    def test_migration_can_be_registered_with_runner(self):
        """Confirm the runner accepts the migration and sorts by version."""
        mod = importlib.import_module("memory.migrations.006_agent_task_assignment")

        class FakeExecutor:
            pass

        runner = MigrationRunner(FakeExecutor())
        runner.register(mod.AgentTaskAssignmentMigration())
        assert any(m.version == 6 for m in runner._migrations)
