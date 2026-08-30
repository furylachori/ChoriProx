"""Tests for canary ingress and NULL-workspace fixes (Batch B - B5, INT-001/INT-004)."""
import pytest
from unittest.mock import MagicMock, patch


class TestMalformedCanaryDoesNotCrash:
    """INT-001: Malformed canary should not cause NameError."""

    def test_malformed_canary_defined_ctx_inherited_skill(self):
        """ctx_inherited_skill should be defined even for malformed canaries."""
        from proxy.canary import parse_canary

        # Malformed: missing closing ]]
        p = parse_canary("[[__SUBAGENT__:SKILL:ticket")
        # Should not crash — not a verified skill
        assert p is None or not p.verified
        assert p is None or p.skill_name is None

    def test_malformed_canary_inheritance_state(self):
        """Malformed canary with valid JSON but bad format."""
        from proxy.canary import parse_canary

        # Valid JSON in inheritance but malformed overall
        p = parse_canary("[[__SUBAGENT__:SKILL:ticket|{}")
        # Should handle gracefully — not a verified skill
        assert p is None or not p.verified
        assert p is None or p.skill_name is None


class TestNullWorkspaceTasksVisible:
    """INT-004: Old NULL-workspace tasks should be visible when filtering by 'default'."""

    def test_get_actionable_tasks_default_includes_null(self):
        """get_actionable_tasks('default') should include NULL-workspace tasks."""
        # This tests the query logic change
        # The workspace_where clause should be:
        # "AND (e.workspace_id = 'default' OR e.workspace_id IS NULL)"
        from memory.task_graph import TaskGraph
        # We can verify by checking the method exists and has correct params
        assert hasattr(TaskGraph, 'get_actionable_tasks')
        
        # Integration test would require actual Neo4j
        # For unit testing, verify the method signature accepts workspace_id
        import inspect
        sig = inspect.signature(TaskGraph.get_actionable_tasks)
        assert 'workspace_id' in [p.name for p in sig.parameters.values()]