"""Tests for Migration 004: Graph-native task schema transformation."""
import json
import logging
import importlib
from unittest.mock import MagicMock, call
import pytest

migration_module = importlib.import_module("memory.migrations.004_graph_native_tasks")
GraphNativeTasks = migration_module.GraphNativeTasks


@pytest.fixture
def migration():
    return GraphNativeTasks()


@pytest.fixture
def executor():
    return MagicMock()


class TestStep1BlockedByRelationships:
    """Tests for _step1_blocked_by_relationships."""

    def test_successful_blocked_by_creation(self, migration, executor, caplog):
        """Test BLOCKED_BY relationships are created when blocker exists."""
        executor.execute_read.return_value = [
            {
                "child_uuid": "task-1",
                "attrs": json.dumps({"blocked_by": ["task-2", "task-3"]}),
            }
        ]
        executor.execute_write.return_value = [{"created": 2, "cleaned": 0}]

        with caplog.at_level(logging.INFO):
            migration._step1_blocked_by_relationships(executor)

        assert executor.execute_write.call_count == 3
        rels_call = executor.execute_write.call_args_list[0]
        assert "BLOCKED_BY" in rels_call[0][0]
        assert rels_call[0][1]["rels"] == [
            {"child_uuid": "task-1", "blocker_uuid": "task-2"},
            {"child_uuid": "task-1", "blocker_uuid": "task-3"},
        ]
        assert "created 2 BLOCKED_BY relationships, 0 orphaned" in caplog.text

    def test_orphaned_blocker_handling(self, migration, executor, caplog):
        """Test orphaned blocked_by references are counted and logged."""
        executor.execute_read.return_value = [
            {
                "child_uuid": "task-1",
                "attrs": json.dumps({"blocked_by": ["task-2", "nonexistent-task"]}),
            }
        ]
        executor.execute_write.return_value = [{"created": 1, "cleaned": 0}]

        with caplog.at_level(logging.INFO):
            migration._step1_blocked_by_relationships(executor)

        assert "1 orphaned blocked_by references" in caplog.text
        assert "created 1 BLOCKED_BY relationships, 1 orphaned" in caplog.text

    def test_multiple_orphaned_blockers(self, migration, executor, caplog):
        """Test multiple orphaned blocker references are counted correctly."""
        executor.execute_read.return_value = [
            {
                "child_uuid": "task-1",
                "attrs": json.dumps({"blocked_by": ["missing-1", "missing-2", "missing-3"]}),
            }
        ]
        executor.execute_write.return_value = [{"created": 0, "cleaned": 0}]

        with caplog.at_level(logging.WARNING):
            migration._step1_blocked_by_relationships(executor)

        assert "3 orphaned blocked_by references" in caplog.text

    def test_no_blocked_by_attributes(self, migration, executor, caplog):
        """Test migration handles tasks with no blocked_by in attributes."""
        executor.execute_read.return_value = [
            {"child_uuid": "task-1", "attrs": json.dumps({"other_field": "value"})}
        ]
        executor.execute_write.return_value = [{"cleaned": 0}]

        with caplog.at_level(logging.INFO):
            migration._step1_blocked_by_relationships(executor)

        assert executor.execute_write.call_count == 2
        assert "created 0 BLOCKED_BY relationships, 0 orphaned" in caplog.text

    def test_empty_attributes(self, migration, executor):
        """Test migration handles empty attributes gracefully."""
        executor.execute_read.return_value = [
            {"child_uuid": "task-1", "attrs": "{}"}
        ]
        executor.execute_write.return_value = [{"cleaned": 0}]

        migration._step1_blocked_by_relationships(executor)

        assert executor.execute_write.call_count == 2

    def test_malformed_json_attributes(self, migration, executor):
        """Test migration handles malformed JSON in attributes."""
        executor.execute_read.return_value = [
            {"child_uuid": "task-1", "attrs": "not valid json"}
        ]
        executor.execute_write.return_value = [{"cleaned": 0}]

        migration._step1_blocked_by_relationships(executor)

        assert executor.execute_write.call_count == 2


class TestStep2HasParentRelationships:
    """Tests for _step2_has_parent_relationships."""

    def test_successful_parent_creation(self, migration, executor, caplog):
        """Test HAS_PARENT relationships are created when parent exists."""
        executor.execute_write.side_effect = [
            [{"created": 3}],
            None,
        ]
        executor.execute_read.return_value = [{"orphans": 0}]

        with caplog.at_level(logging.INFO):
            migration._step2_has_parent_relationships(executor)

        assert executor.execute_write.call_count == 2
        first_call = executor.execute_write.call_args_list[0]
        assert "HAS_PARENT" in first_call[0][0]
        assert "created 3 HAS_PARENT relationships, 0 orphaned" in caplog.text

    def test_orphaned_parent_handling(self, migration, executor, caplog):
        """Test orphaned parent_task_uuid references are counted and logged."""
        executor.execute_write.side_effect = [
            [{"created": 2}],
            None,
        ]
        executor.execute_read.return_value = [{"orphans": 5}]

        with caplog.at_level(logging.INFO):
            migration._step2_has_parent_relationships(executor)

        assert "5 orphaned parent_task_uuid references" in caplog.text
        assert "created 2 HAS_PARENT relationships, 5 orphaned" in caplog.text

    def test_parent_task_uuid_removed_from_all_tasks(self, migration, executor):
        """Test parent_task_uuid is removed from all tasks after migration."""
        executor.execute_write.side_effect = [
            [{"created": 1}],
            None,
        ]
        executor.execute_read.return_value = [{"orphans": 2}]

        migration._step2_has_parent_relationships(executor)

        second_call = executor.execute_write.call_args_list[1]
        assert "REMOVE t.parent_task_uuid" in second_call[0][0]

    def test_all_parents_orphaned(self, migration, executor, caplog):
        """Test when all parent references are orphaned."""
        executor.execute_write.side_effect = [
            [{"created": 0}],
            None,
        ]
        executor.execute_read.return_value = [{"orphans": 10}]

        with caplog.at_level(logging.WARNING):
            migration._step2_has_parent_relationships(executor)

        assert "10 orphaned parent_task_uuid references" in caplog.text
        assert executor.execute_write.call_count == 2

    def test_no_parent_task_uuid(self, migration, executor, caplog):
        """Test migration handles tasks with no parent_task_uuid."""
        executor.execute_write.side_effect = [
            [{"created": 0}],
            None,
        ]
        executor.execute_read.return_value = [{"orphans": 0}]

        with caplog.at_level(logging.INFO):
            migration._step2_has_parent_relationships(executor)

        assert "created 0 HAS_PARENT relationships, 0 orphaned" in caplog.text

    def test_empty_write_result(self, migration, executor):
        """Test migration handles empty write result."""
        executor.execute_write.side_effect = [
            [],
            None,
        ]
        executor.execute_read.return_value = [{"orphans": 0}]

        migration._step2_has_parent_relationships(executor)

        assert executor.execute_write.call_count == 2


class TestStep0ExtractPlanningReview:
    """Tests for _step0_extract_planning_review."""

    def test_planning_info_extraction(self, migration, executor):
        """Test planning_info is extracted from attributes."""
        executor.execute_read.return_value = [
            {
                "nid": 1,
                "attrs": json.dumps({"planning_info": "Plan A", "other": "data"}),
            }
        ]
        executor.execute_write.return_value = [{"cleaned": 0}]

        migration._step0_extract_planning_review(executor)

        assert executor.execute_write.call_count == 3
        update_call = executor.execute_write.call_args_list[0]
        assert update_call[0][1]["updates"] == [
            {"nid": 1, "planning_info": "Plan A", "review_notes": None}
        ]

    def test_review_notes_extraction(self, migration, executor):
        """Test review_notes is extracted from attributes."""
        executor.execute_read.return_value = [
            {
                "nid": 2,
                "attrs": json.dumps({"review_notes": "Good work", "other": "data"}),
            }
        ]
        executor.execute_write.return_value = [{"cleaned": 0}]

        migration._step0_extract_planning_review(executor)

        update_call = executor.execute_write.call_args_list[0]
        assert update_call[0][1]["updates"] == [
            {"nid": 2, "planning_info": None, "review_notes": "Good work"}
        ]

    def test_both_planning_and_review(self, migration, executor):
        """Test both planning_info and review_notes are extracted."""
        executor.execute_read.return_value = [
            {
                "nid": 3,
                "attrs": json.dumps({
                    "planning_info": "Plan B",
                    "review_notes": "Needs revision",
                }),
            }
        ]
        executor.execute_write.return_value = [{"cleaned": 0}]

        migration._step0_extract_planning_review(executor)

        update_call = executor.execute_write.call_args_list[0]
        assert update_call[0][1]["updates"] == [
            {"nid": 3, "planning_info": "Plan B", "review_notes": "Needs revision"}
        ]

    def test_no_planning_or_review(self, migration, executor):
        """Test migration handles attributes without planning_info or review_notes."""
        executor.execute_read.return_value = [
            {"nid": 4, "attrs": json.dumps({"other_field": "value"})}
        ]
        executor.execute_write.return_value = [{"cleaned": 0}]

        migration._step0_extract_planning_review(executor)

        assert executor.execute_write.call_count == 2

    def test_multiple_tasks(self, migration, executor):
        """Test extraction from multiple tasks."""
        executor.execute_read.return_value = [
            {"nid": 1, "attrs": json.dumps({"planning_info": "Plan 1"})},
            {"nid": 2, "attrs": json.dumps({"review_notes": "Review 2"})},
            {"nid": 3, "attrs": json.dumps({"other": "data"})},
        ]
        executor.execute_write.return_value = [{"cleaned": 0}]

        migration._step0_extract_planning_review(executor)

        update_call = executor.execute_write.call_args_list[0]
        updates = update_call[0][1]["updates"]
        assert len(updates) == 2
        assert updates[0]["nid"] == 1
        assert updates[1]["nid"] == 2


class TestStep3PromoteTagRemoveEntityType:
    """Tests for _step3_promote_tag_remove_entity_type."""

    def test_tag_promotion(self, migration, executor):
        """Test tag is promoted from attributes to top-level property."""
        executor.execute_read.return_value = [
            {"nid": 1, "attrs": json.dumps({"tag": "bug", "other": "data"})}
        ]
        executor.execute_write.return_value = [{"cleaned": 0}]

        migration._step3_promote_tag_remove_entity_type(executor)

        assert executor.execute_write.call_count == 4
        tag_call = executor.execute_write.call_args_list[0]
        assert "SET t.tag = u.tag" in tag_call[0][0]
        assert tag_call[0][1]["updates"] == [{"nid": 1, "tag": "bug"}]

    def test_entity_type_removal(self, migration, executor):
        """Test entity_type is removed from all tasks."""
        executor.execute_read.return_value = []
        executor.execute_write.return_value = [{"cleaned": 0}]

        migration._step3_promote_tag_remove_entity_type(executor)

        assert executor.execute_write.call_count == 2
        call_args = executor.execute_write.call_args
        assert "REMOVE t.entity_type" in call_args[0][0]

    def test_no_tag_in_attributes(self, migration, executor):
        """Test migration handles attributes without tag."""
        executor.execute_read.return_value = [
            {"nid": 1, "attrs": json.dumps({"other_field": "value"})}
        ]
        executor.execute_write.return_value = [{"cleaned": 0}]

        migration._step3_promote_tag_remove_entity_type(executor)

        assert executor.execute_write.call_count == 3
        call_args = executor.execute_write.call_args
        assert "REMOVE t.entity_type" in call_args[0][0]


class TestStep4RemoveAttributes:
    """Tests for _step4_remove_attributes."""

    def test_attributes_removed(self, migration, executor):
        """Test attributes property is removed from all tasks."""
        executor.execute_write.return_value = [{"cleaned": 0}]

        migration._step4_remove_attributes(executor)

        executor.execute_write.assert_called_once()
        call_args = executor.execute_write.call_args
        assert "REMOVE t.attributes" in call_args[0][0]


class TestStep5IndexesAndConstraints:
    """Tests for _step5_indexes_and_constraints."""

    def test_indexes_created(self, migration, executor):
        """Test all required indexes are created."""
        migration._step5_indexes_and_constraints(executor)

        calls = [str(call) for call in executor.execute_write.call_args_list]
        assert any("CREATE CONSTRAINT task_uuid_unique" in c and "IS UNIQUE" in c for c in calls)
        assert any("task_label_short_hash_idx" in c for c in calls)
        assert any("task_label_workspace_idx" in c for c in calls)
        assert any("task_label_status_idx" in c for c in calls)
        assert any("task_label_created_at_idx" in c for c in calls)
        assert any("task_label_tag_idx" in c for c in calls)

    def test_unique_constraint_created(self, migration, executor):
        """Test unique constraint on uuid is created."""
        migration._step5_indexes_and_constraints(executor)

        calls = [str(call) for call in executor.execute_write.call_args_list]
        assert any("task_uuid_unique" in c and "UNIQUE" in c for c in calls)

    def test_name_workspace_constraint_recreated(self, migration, executor):
        """Test name+workspace constraint is dropped and recreated."""
        migration._step5_indexes_and_constraints(executor)

        calls = [str(call) for call in executor.execute_write.call_args_list]
        assert any("DROP CONSTRAINT task_name_workspace_unique" in c for c in calls)
        assert any("CREATE CONSTRAINT task_name_workspace_unique" in c for c in calls)

    def test_old_indexes_dropped(self, migration, executor):
        """Test old indexes are dropped."""
        migration._step5_indexes_and_constraints(executor)

        calls = [str(call) for call in executor.execute_write.call_args_list]
        assert any("DROP INDEX parent_task_uuid_idx" in c for c in calls)
        assert any("DROP INDEX short_hash_idx" in c for c in calls)
        assert any("DROP INDEX task_workspace_idx" in c for c in calls)
        assert any("DROP INDEX task_status_idx" in c for c in calls)


class TestMigrationIdempotency:
    """Tests for migration idempotency."""

    def test_step1_idempotent(self, migration, executor):
        """Test step1 can be run multiple times without creating duplicates."""
        executor.execute_read.return_value = [
            {"child_uuid": "task-1", "attrs": json.dumps({"blocked_by": ["task-2"]})}
        ]
        executor.execute_write.return_value = [{"created": 1, "cleaned": 0}]

        migration._step1_blocked_by_relationships(executor)
        migration._step1_blocked_by_relationships(executor)

        assert executor.execute_write.call_count == 6

    def test_step2_idempotent(self, migration, executor):
        """Test step2 can be run multiple times without creating duplicates."""
        executor.execute_write.side_effect = [
            [{"created": 1}],
            None,
            [{"created": 0}],
            None,
        ]
        executor.execute_read.return_value = [{"orphans": 0}]

        migration._step2_has_parent_relationships(executor)
        migration._step2_has_parent_relationships(executor)

        assert executor.execute_write.call_count == 4

    def test_full_migration_idempotent(self, migration, executor):
        """# US-036: Test full migration can be run twice safely."""
        executor.execute_read.return_value = []
        executor.execute_write.return_value = [{"created": 0, "orphans": 0, "cleaned": 0}]

        migration.up(executor)
        migration.up(executor)

        assert executor.execute_write.call_count > 0


class TestMigrationEdgeCases:
    """Tests for edge cases and error handling."""

    def test_empty_database(self, migration, executor):
        """Test migration handles empty database."""
        executor.execute_read.return_value = []
        executor.execute_write.return_value = [{"created": 0, "orphans": 0, "cleaned": 0}]

        migration.up(executor)

        assert executor.execute_write.call_count > 0

    def test_null_attributes(self, migration, executor):
        """Test migration handles NULL attributes."""
        executor.execute_read.return_value = [
            {"child_uuid": "task-1", "attrs": None}
        ]
        executor.execute_write.return_value = [{"cleaned": 0}]

        migration._step1_blocked_by_relationships(executor)

        assert executor.execute_write.call_count == 2

    def test_dict_attributes(self, migration, executor):
        """Test migration handles dict attributes (not JSON string)."""
        executor.execute_read.return_value = [
            {"child_uuid": "task-1", "attrs": {"blocked_by": ["task-2"]}}
        ]
        executor.execute_write.return_value = [{"cleaned": 0}]

        migration._step1_blocked_by_relationships(executor)

        assert executor.execute_write.call_count == 2

    def test_empty_blocked_by_list(self, migration, executor):
        """Test migration handles empty blocked_by list."""
        executor.execute_read.return_value = [
            {"child_uuid": "task-1", "attrs": json.dumps({"blocked_by": []})}
        ]
        executor.execute_write.return_value = [{"cleaned": 0}]

        migration._step1_blocked_by_relationships(executor)

        assert executor.execute_write.call_count == 2

    def test_migration_version_and_description(self, migration):
        """Test migration has correct version and description."""
        assert migration.version == 4
        assert "graph-native" in migration.description.lower()
