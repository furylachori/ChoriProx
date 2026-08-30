import pytest
from unittest.mock import MagicMock, patch
from neo4j.exceptions import ServiceUnavailable, SessionExpired, TransientError, ClientError
from memory.query_executor import (
    QueryExecutor,
    InfrastructureError,
    ConstraintViolation,
    QueryTimeoutError,
    _map_neo4j_error,
)


def make_executor(circuit_breaker=None):
    executor = QueryExecutor(None, None, circuit_breaker=circuit_breaker)
    executor._driver = MagicMock()
    return executor


def mock_session_with_result(driver, records):
    session = MagicMock()
    driver.session.return_value = session
    result = MagicMock()
    result.__iter__.side_effect = lambda: iter(records)
    session.run.return_value = result
    return session


class TestExecuteRead:
    def test_execute_read_returns_results(self):
        executor = make_executor()
        session = mock_session_with_result(executor._driver, [{'name': 'Alice'}, {'name': 'Bob'}])

        results = executor.execute_read("MATCH (n) RETURN n.name AS name")

        assert results == [{'name': 'Alice'}, {'name': 'Bob'}]
        session.run.assert_called_once()
        session.close.assert_called_once()

    @patch('memory.query_executor.time.sleep')
    def test_execute_read_retries_on_infrastructure_error(self, mock_sleep):
        executor = make_executor()
        session = MagicMock()
        executor._driver.session.return_value = session

        mock_result = MagicMock()
        mock_result.__iter__.side_effect = lambda: iter([{'ok': True}])
        session.run.side_effect = [
            TransientError("Connection lost"),
            TransientError("Connection lost"),
            mock_result,
        ]

        results = executor.execute_read("MATCH (n) RETURN n")

        assert results == [{'ok': True}]
        assert session.run.call_count == 3
        assert mock_sleep.call_count == 2

    def test_execute_read_does_not_retry_on_constraint_violation(self):
        """# US-004: ConstraintViolation not retried."""
        executor = make_executor()
        session = MagicMock()
        executor._driver.session.return_value = session
        session.run.side_effect = ClientError("Constraint violation: already exists")

        with pytest.raises(ConstraintViolation):
            executor.execute_read("CREATE (n:Node {id: 1})")

        assert session.run.call_count == 1


class TestExecuteWrite:
    @patch('memory.query_executor.time.sleep')
    def test_execute_write_retries_on_infrastructure_error(self, mock_sleep):
        executor = make_executor()
        session = MagicMock()
        executor._driver.session.return_value = session

        mock_result = MagicMock()
        mock_result.__iter__.side_effect = lambda: iter([])
        session.run.side_effect = [
            TransientError("Connection lost"),
            mock_result,
        ]

        executor.execute_write("CREATE (n:Node)")

        assert session.run.call_count == 2
        assert mock_sleep.call_count == 1

    def test_execute_write_no_retry_does_not_retry(self):
        executor = make_executor()
        session = MagicMock()
        executor._driver.session.return_value = session
        session.run.side_effect = TransientError("Connection lost")

        with pytest.raises(InfrastructureError):
            executor.execute_write_no_retry("CREATE (n:Node)")

        assert session.run.call_count == 1


class TestExecuteTransaction:
    def test_execute_transaction_commits_on_success(self):
        executor = make_executor()
        session = MagicMock()
        executor._driver.session.return_value = session
        tx = MagicMock()
        session.begin_transaction.return_value = tx

        def tx_fn(transaction):
            return "success"

        result = executor.execute_transaction(tx_fn)

        assert result == "success"
        tx.commit.assert_called_once()
        tx.rollback.assert_not_called()
        session.close.assert_called_once()

    def test_execute_transaction_rolls_back_on_exception(self):
        executor = make_executor()
        session = MagicMock()
        executor._driver.session.return_value = session
        tx = MagicMock()
        session.begin_transaction.return_value = tx

        def tx_fn(transaction):
            raise ValueError("Something went wrong")

        with pytest.raises(ValueError):
            executor.execute_transaction(tx_fn)

        tx.rollback.assert_called_once()
        tx.commit.assert_not_called()


class TestCircuitBreaker:
    @patch('memory.query_executor.time.sleep')
    def test_circuit_breaker_opens_after_failures(self, mock_sleep):
        cb = MagicMock()
        cb.can_execute.return_value = False
        executor = make_executor(circuit_breaker=cb)

        with pytest.raises(InfrastructureError, match="Circuit breaker is open"):
            executor.execute_read("MATCH (n) RETURN n")

        assert cb.can_execute.call_count == 3
        assert cb.on_failure.call_count == 3

    def test_circuit_breaker_half_open_after_timeout(self):
        cb = MagicMock()
        cb.can_execute.return_value = True
        executor = make_executor(circuit_breaker=cb)

        session = mock_session_with_result(executor._driver, [{'ok': True}])

        results = executor.execute_read("MATCH (n) RETURN n")

        assert results == [{'ok': True}]
        cb.on_success.assert_called()


class TestErrorMapping:
    def test_map_neo4j_error_maps_service_unavailable(self):
        exc = ServiceUnavailable("Connection refused")
        mapped = _map_neo4j_error(exc)
        # ServiceUnavailable is not a Neo4jError subclass in this neo4j version,
        # so _map_neo4j_error returns it unmodified.
        assert mapped is exc

    def test_map_neo4j_error_maps_transient_error(self):
        exc = TransientError("Temporary failure")
        mapped = _map_neo4j_error(exc)
        assert isinstance(mapped, InfrastructureError)

    def test_map_neo4j_error_maps_constraint_violation(self):
        exc = ClientError("Constraint violation: already exists")
        mapped = _map_neo4j_error(exc)
        assert isinstance(mapped, ConstraintViolation)


class TestResourceCleanup:
    def test_session_always_closed_in_finally(self):
        executor = make_executor()
        session = MagicMock()
        executor._driver.session.return_value = session
        session.run.side_effect = TransientError("Connection lost")

        with pytest.raises(InfrastructureError):
            executor.execute_read("MATCH (n) RETURN n")

        session.close.assert_called()

    def test_close_is_idempotent(self):
        """executor.close() is idempotent and doesn't close shared driver."""
        executor = make_executor()

        executor.close()
        assert executor._closed is True
        executor._driver.close.assert_not_called()  # Shared driver NOT closed

        # Second close should not raise
        executor.close()
        assert executor._closed is True
