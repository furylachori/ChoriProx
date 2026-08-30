"""Tests for Neo4j driver singleton pattern.

Verifies that QueryExecutor uses a shared driver singleton to prevent
resource leaks from multiple driver instances.
"""
import threading
from unittest.mock import Mock, patch

import pytest

from memory.query_executor import (
    QueryExecutor,
    close_shared_driver,
    get_shared_driver,
    reset_shared_driver,
)


class TestNeo4jDriverSingleton:
    """Test suite for Neo4j driver singleton behavior."""

    def setup_method(self):
        """Reset singleton state before each test."""
        reset_shared_driver()

    def teardown_method(self):
        """Reset singleton state after each test."""
        reset_shared_driver()

    def test_singleton_basic(self):
        """Two QueryExecutor instances share the same driver."""
        with patch("memory.query_executor.GraphDatabase") as mock_gdb:
            mock_driver = Mock()
            mock_gdb.driver.return_value = mock_driver

            executor1 = QueryExecutor("bolt://localhost:7687", ("neo4j", "test"))
            executor2 = QueryExecutor("bolt://localhost:7687", ("neo4j", "test"))

            assert executor1._driver is executor2._driver
            assert executor1._driver is mock_driver
            mock_gdb.driver.assert_called_once()

    def test_thread_safety_100_threads(self):
        """100 threads with barrier ensure single driver creation."""
        with patch("memory.query_executor.GraphDatabase") as mock_gdb:
            mock_driver = Mock()
            mock_gdb.driver.return_value = mock_driver

            barrier = threading.Barrier(100)
            drivers = []

            def create_executor():
                barrier.wait()
                executor = QueryExecutor("bolt://localhost:7687", ("neo4j", "test"))
                drivers.append(executor._driver)

            threads = [threading.Thread(target=create_executor) for _ in range(100)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()

            assert len(set(id(d) for d in drivers)) == 1
            mock_gdb.driver.assert_called_once()

    def test_parameter_mismatch_warns(self):
        """Different params log warning but return existing driver."""
        with patch("memory.query_executor.GraphDatabase") as mock_gdb:
            mock_driver = Mock()
            mock_gdb.driver.return_value = mock_driver

            executor1 = QueryExecutor("bolt://localhost:7687", ("neo4j", "test1"))

            with patch("memory.query_executor.logger") as mock_logger:
                executor2 = QueryExecutor("bolt://localhost:7688", ("neo4j", "test2"))
                mock_logger.warning.assert_called_once()
                assert "different parameters" in mock_logger.warning.call_args[0][0]
                assert executor1._driver is executor2._driver

    def test_none_params_return_none(self):
        """None params don't create driver."""
        executor = QueryExecutor(None, None)
        assert executor._driver is None

    def test_none_then_valid_creates_driver(self):
        """None to valid transition creates driver."""
        with patch("memory.query_executor.GraphDatabase") as mock_gdb:
            mock_driver = Mock()
            mock_gdb.driver.return_value = mock_driver

            executor1 = QueryExecutor(None, None)
            assert executor1._driver is None

            executor2 = QueryExecutor("bolt://localhost:7687", ("neo4j", "test"))
            assert executor2._driver is mock_driver

    def test_cleanup_closes_driver(self):
        """close_shared_driver() closes the driver."""
        with patch("memory.query_executor.GraphDatabase") as mock_gdb:
            mock_driver = Mock()
            mock_gdb.driver.return_value = mock_driver

            executor = QueryExecutor("bolt://localhost:7687", ("neo4j", "test"))
            close_shared_driver()

            mock_driver.close.assert_called_once()

    def test_cleanup_idempotent(self):
        """Double close doesn't raise."""
        with patch("memory.query_executor.GraphDatabase") as mock_gdb:
            mock_driver = Mock()
            mock_gdb.driver.return_value = mock_driver

            executor = QueryExecutor("bolt://localhost:7687", ("neo4j", "test"))
            close_shared_driver()
            close_shared_driver()

    def test_reinit_after_close(self):
        """New driver created after close + reset."""
        with patch("memory.query_executor.GraphDatabase") as mock_gdb:
            mock_driver1 = Mock()
            mock_driver2 = Mock()
            mock_gdb.driver.side_effect = [mock_driver1, mock_driver2]

            executor1 = QueryExecutor("bolt://localhost:7687", ("neo4j", "test"))
            assert executor1._driver is mock_driver1

            close_shared_driver()
            reset_shared_driver()

            executor2 = QueryExecutor("bolt://localhost:7687", ("neo4j", "test"))
            assert executor2._driver is mock_driver2

    def test_shutdown_prevents_reinit(self):
        """After shutdown, RuntimeError raised."""
        with patch("memory.query_executor.GraphDatabase") as mock_gdb:
            mock_driver = Mock()
            mock_gdb.driver.return_value = mock_driver

            executor = QueryExecutor("bolt://localhost:7687", ("neo4j", "test"))
            close_shared_driver()

            with pytest.raises(RuntimeError, match="shut down"):
                QueryExecutor("bolt://localhost:7687", ("neo4j", "test"))

    def test_exception_during_creation_no_cache(self):
        """Failed creation leaves _driver_instance=None."""
        with patch("memory.query_executor.GraphDatabase") as mock_gdb:
            mock_gdb.driver.side_effect = Exception("Connection failed")

            with pytest.raises(Exception):
                QueryExecutor("bolt://localhost:7687", ("neo4j", "test"))

            mock_gdb.driver.side_effect = None
            mock_driver = Mock()
            mock_gdb.driver.return_value = mock_driver

            executor = QueryExecutor("bolt://localhost:7687", ("neo4j", "test"))
            assert executor._driver is mock_driver

    def test_lock_released_on_exception(self):
        """No deadlock after failed creation."""
        with patch("memory.query_executor.GraphDatabase") as mock_gdb:
            mock_gdb.driver.side_effect = Exception("Connection failed")

            with pytest.raises(Exception):
                QueryExecutor("bolt://localhost:7687", ("neo4j", "test"))

            mock_gdb.driver.side_effect = None
            mock_driver = Mock()
            mock_gdb.driver.return_value = mock_driver

            executor = QueryExecutor("bolt://localhost:7687", ("neo4j", "test"))
            assert executor._driver is mock_driver

    def test_executor_close_noop(self):
        """executor.close() doesn't close shared driver."""
        with patch("memory.query_executor.GraphDatabase") as mock_gdb:
            mock_driver = Mock()
            mock_gdb.driver.return_value = mock_driver

            executor = QueryExecutor("bolt://localhost:7687", ("neo4j", "test"))
            executor.close()

            mock_driver.close.assert_not_called()
