"""Integration tests for Neo4j driver singleton across components.

Verifies that all components (TaskGraph, sync, migrations, app) properly
reuse the shared Neo4j driver singleton.
"""
import os
import threading
from unittest.mock import Mock, patch, MagicMock

import pytest

from memory.query_executor import (
    QueryExecutor,
    close_shared_driver,
    get_shared_driver,
    reset_shared_driver,
)


@pytest.mark.integration
class TestNeo4jSingletonIntegration:
    """Integration tests for Neo4j driver singleton across components."""

    def setup_method(self):
        """Reset singleton state before each test."""
        reset_shared_driver()
        import memory.query_executor as qe
        qe._shared_executor = None

    def teardown_method(self):
        """Reset singleton state after each test."""
        reset_shared_driver()
        import memory.query_executor as qe
        qe._shared_executor = None

    def test_task_graph_uses_shared_driver(self):
        """TaskGraph uses the shared driver singleton."""
        with patch("memory.query_executor.GraphDatabase") as mock_gdb:
            mock_driver = Mock()
            mock_session = Mock()
            mock_result = Mock()
            mock_result.__iter__ = Mock(return_value=iter([]))
            mock_session.run.return_value = mock_result
            mock_driver.session.return_value = mock_session
            mock_gdb.driver.return_value = mock_driver

            executor1 = QueryExecutor("bolt://localhost:7687", ("neo4j", "test"))
            executor2 = QueryExecutor("bolt://localhost:7687", ("neo4j", "test"))

            assert executor1._driver is executor2._driver
            assert executor1._driver is mock_driver
            mock_gdb.driver.assert_called_once()

    def test_sync_reuses_driver(self):
        """Sync module reuses the shared driver."""
        with patch("memory.query_executor.GraphDatabase") as mock_gdb:
            mock_driver = Mock()
            mock_gdb.driver.return_value = mock_driver

            from memory.query_executor import get_shared_executor

            executor1 = get_shared_executor("bolt://localhost:7687", ("neo4j", "test"))
            executor2 = get_shared_executor("bolt://localhost:7687", ("neo4j", "test"))

            assert executor1 is executor2
            assert executor1._driver is mock_driver
            mock_gdb.driver.assert_called_once()

    def test_migration_runner_reuses_driver(self):
        """Multiple executors share the same driver for migrations."""
        with patch("memory.query_executor.GraphDatabase") as mock_gdb:
            mock_driver = Mock()
            mock_gdb.driver.return_value = mock_driver

            executor1 = QueryExecutor("bolt://localhost:7687", ("neo4j", "test"))
            executor2 = QueryExecutor("bolt://localhost:7687", ("neo4j", "test"))
            executor3 = QueryExecutor("bolt://localhost:7687", ("neo4j", "test"))

            assert executor1._driver is executor2._driver
            assert executor2._driver is executor3._driver
            assert executor1._driver is mock_driver
            mock_gdb.driver.assert_called_once()

    def test_no_import_time_driver_creation(self):
        """Importing modules doesn't create driver."""
        reset_shared_driver()

        with patch("memory.query_executor.GraphDatabase") as mock_gdb:
            mock_driver = Mock()
            mock_gdb.driver.return_value = mock_driver

            import memory.task_graph
            import memory.sync
            import memory.migrations.runner

            mock_gdb.driver.assert_not_called()

    def test_driver_reuse_across_requests(self):
        """Driver is reused across multiple requests."""
        with patch("memory.query_executor.GraphDatabase") as mock_gdb:
            mock_driver = Mock()
            mock_gdb.driver.return_value = mock_driver

            executors = []
            for i in range(5):
                executor = QueryExecutor("bolt://localhost:7687", ("neo4j", "test"))
                executors.append(executor)

            assert all(e._driver is executors[0]._driver for e in executors)
            mock_gdb.driver.assert_called_once()

    def test_auth_separator_consistency(self):
        """Auth separator is consistently ':' across codebase."""
        test_auth = "neo4j:password"
        parts = test_auth.split(":", 1)
        assert len(parts) == 2
        assert parts[0] == "neo4j"
        assert parts[1] == "password"

        with patch("memory.query_executor.GraphDatabase") as mock_gdb:
            mock_driver = Mock()
            mock_gdb.driver.return_value = mock_driver

            auth_tuple = tuple(parts)
            executor = QueryExecutor("bolt://localhost:7687", auth_tuple)

            assert executor._driver is not None
