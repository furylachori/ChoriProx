"""
Centralized Neo4j query execution with retry, circuit breaker integration,
and error mapping. Replaces ad-hoc session management scattered across modules.
"""
import logging
import threading
import time
from typing import Any, Callable, Optional

try:
    from neo4j import GraphDatabase
    from neo4j.exceptions import (
        ClientError,
        DriverError,
        ServiceUnavailable,
        SessionExpired,
        TransientError,
    )
except ImportError:
    GraphDatabase = None
    ClientError = None
    DriverError = None
    ServiceUnavailable = None
    SessionExpired = None
    TransientError = None

logger = logging.getLogger(__name__)

_driver_instance = None
_driver_lock = threading.Lock()
_shutdown_complete = False
_closing = False

__all__ = [
    "InfrastructureError",
    "ConstraintViolation",
    "QueryTimeoutError",
    "QueryExecutor",
    "get_shared_driver",
    "get_shared_executor",
    "close_shared_driver",
    "reset_shared_driver",
]


def get_shared_driver(bolt_url: Optional[str], auth: Optional[tuple]):
    """Get or create the shared Neo4j driver singleton.
    
    Thread-safe. Returns None if bolt_url or auth is None.
    Raises RuntimeError if shutdown has completed.
    """
    global _driver_instance, _shutdown_complete
    
    if bolt_url is None or auth is None:
        return None
    
    with _driver_lock:
        if _shutdown_complete:
            raise RuntimeError("Cannot create driver after shut down")
        
        if _driver_instance is not None:
            if _driver_instance._bolt_url != bolt_url or _driver_instance._auth != auth:
                logger.warning(
                    "get_shared_driver called with different parameters than existing driver"
                )
            return _driver_instance
        
        try:
            driver = GraphDatabase.driver(
                bolt_url,
                auth=auth,
                max_connection_pool_size=100,
                connection_acquisition_timeout=30,
                max_transaction_retry_time=5
            )
            driver._bolt_url = bolt_url
            driver._auth = auth
            _driver_instance = driver
            logger.info("Neo4j shared driver created")
            return _driver_instance
        except Exception:
            logger.exception("Failed to create Neo4j driver")
            raise


def close_shared_driver():
    """Close the shared Neo4j driver and executor.
    
    Thread-safe. Sets _closing flag before closing, then sets _shutdown_complete.
    """
    global _driver_instance, _shutdown_complete, _closing, _shared_executor
    
    with _driver_lock:
        _closing = True
        if _shared_executor is not None:
            _shared_executor = None
        if _driver_instance is not None:
            try:
                _driver_instance.close()
                logger.info("Neo4j shared driver closed")
            except Exception:
                logger.exception("Error closing Neo4j driver")
            finally:
                _driver_instance = None
        _shutdown_complete = True


def reset_shared_driver():
    """Reset the shared driver state (for testing only).
    
    Thread-safe. Does NOT close the driver, just resets state.
    """
    global _driver_instance, _shutdown_complete, _closing
    
    with _driver_lock:
        _driver_instance = None
        _shutdown_complete = False
        _closing = False


class InfrastructureError(Exception):
    """Transient Neo4j errors (ServiceUnavailable, SessionExpired) — retryable."""
    pass


class ConstraintViolation(Exception):
    """Uniqueness/relationship constraint violations — not retryable."""
    pass


class QueryTimeoutError(Exception):
    """Query exceeded timeout — may be retryable."""
    pass


_RETRYABLE = (InfrastructureError,)
_MAX_RETRIES = 3
_BACKOFF_BASE = 0.1

_shared_executor = None
_shared_lock = threading.Lock()


def get_shared_executor(bolt_url: Optional[str], auth: Optional[tuple]) -> "QueryExecutor":
    """Get or create the singleton shared QueryExecutor.
    
    All modules should use this instead of creating new QueryExecutor instances
    to ensure a single Neo4j driver is shared across the application.
    """
    global _shared_executor
    with _shared_lock:
        if _shared_executor is None:
            _shared_executor = QueryExecutor(bolt_url, auth)
        return _shared_executor


def get_shared_executor_from_env() -> Optional["QueryExecutor"]:
    """Get or create the shared QueryExecutor using env vars.

    Reads PROXY_NEO4J_BOLT_URL and NEO4J_AUTH. Returns None if Neo4j
    is unavailable (no driver created) or on failure.
    Thread-safe — uses existing _shared_lock.

    Matches init_db() auth parsing: if NEO4J_AUTH has no colon, return None
    (no driver created) rather than empty-string credentials.
    """
    import os

    bolt_url = os.environ.get("PROXY_NEO4J_BOLT_URL", "bolt://localhost:7687")
    neo4j_auth = os.environ.get("NEO4J_AUTH", "neo4j:password")
    if ":" not in neo4j_auth:
        return None
    parts = neo4j_auth.split(":", 1)
    auth = tuple(parts)

    try:
        executor = get_shared_executor(bolt_url, auth)
    except Exception:
        return None
    if executor._driver is None:
        return None
    return executor





def _map_neo4j_error(exc: Exception) -> Exception:
    from neo4j.exceptions import Neo4jError
    # neo4j 6.x split driver/connection failures (ServiceUnavailable,
    # SessionExpired) into a DriverError subtree separate from server-side
    # Neo4jError — both must map, or infra failures leak unmapped.
    root_types = (Neo4jError, DriverError) if DriverError is not None else (Neo4jError,)
    if not isinstance(exc, root_types):
        return exc
    if ServiceUnavailable and isinstance(exc, ServiceUnavailable):
        return InfrastructureError(str(exc))
    if SessionExpired and isinstance(exc, SessionExpired):
        return InfrastructureError(str(exc))
    if TransientError and isinstance(exc, TransientError):
        msg = str(exc).lower()
        if "timeout" in msg or "deadline" in msg:
            return QueryTimeoutError(str(exc))
        return InfrastructureError(str(exc))
    if ClientError and isinstance(exc, ClientError):
        msg = str(exc).lower()
        if "timeout" in msg or "deadline" in msg:
            return QueryTimeoutError(str(exc))
        if "constraint" in msg or "already exists" in msg or "duplicate" in msg:
            return ConstraintViolation(str(exc))
        return InfrastructureError(str(exc))
    return InfrastructureError(str(exc))


class QueryExecutor:
    def __init__(
        self,
        bolt_url: Optional[str],
        auth: Optional[tuple],
        circuit_breaker=None,
    ):
        self._driver = get_shared_driver(bolt_url, auth)
        self._circuit_breaker = circuit_breaker
        self._closed = False

    def execute_read(self, query: str, params: Optional[dict] = None, timeout: int = 30) -> list[dict]:
        """Execute read query with automatic retry on transient errors.

        Retries up to 3 times with exponential backoff (100ms, 200ms, 400ms)
        for InfrastructureError only.
        """
        return self._execute_with_retry(self._run_read, query, params, timeout)

    def execute_write(self, query: str, params: Optional[dict] = None, timeout: int = 30) -> list[dict]:
        """Execute write query with automatic retry on transient errors."""
        return self._execute_with_retry(self._run_write, query, params, timeout)

    def execute_write_no_retry(self, query: str, params: Optional[dict] = None, timeout: int = 30) -> list[dict]:
        """Execute write WITHOUT retry. For non-idempotent operations."""
        self._check_circuit_breaker()
        if not self._driver:
            raise InfrastructureError("No Neo4j driver configured")
        session = self._driver.session()
        try:
            from neo4j import Query
            result = session.run(Query(query, timeout=timeout), params or {})
            records = [dict(r) for r in result]
            result.consume()
            self._record_success()
            return records
        except Exception as e:
            mapped = _map_neo4j_error(e)
            self._record_failure(mapped)
            raise mapped from e
        finally:
            session.close()

    def execute_transaction(self, fn: Callable) -> Any:
        """Execute function within a managed write transaction.

        The function receives a Neo4j Transaction object and can run
        multiple queries atomically.
        """
        return self._execute_with_retry(self._run_transaction, fn)

    def execute_transaction_no_retry(self, fn: Callable) -> Any:
        """Execute function within a managed write transaction WITHOUT retry."""
        self._check_circuit_breaker()
        if not self._driver:
            raise InfrastructureError("No Neo4j driver configured")
        session = self._driver.session()
        try:
            tx = session.begin_transaction()
            try:
                result = fn(tx)
                tx.commit()
                self._record_success()
                return result
            except Exception:
                tx.rollback()
                raise
        except Exception as e:
            mapped = _map_neo4j_error(e)
            self._record_failure(mapped)
            raise mapped from e
        finally:
            session.close()

    def is_available(self) -> bool:
        """Check circuit breaker + driver connectivity."""
        if self._driver is None:
            return False
        if self._circuit_breaker and not self._circuit_breaker.can_execute():
            return False
        try:
            session = self._driver.session()
            try:
                from neo4j import Query
                result = session.run(Query("RETURN 1", timeout=2.0))
                result.consume()
                return True
            finally:
                session.close()
        except Exception:
            return False

    def session(self):
        """Return a new session from the shared driver.
        
        Raises RuntimeError if the driver is closing or closed.
        """
        if _closing or self._closed:
            raise RuntimeError("QueryExecutor is closed")
        if self._driver is None:
            raise InfrastructureError("No Neo4j driver configured")
        return self._driver.session()

    def close(self):
        """No-op. Shared driver is closed by close_shared_driver() at shutdown."""
        self._closed = True

    def _execute_with_retry(self, fn: Callable, *args) -> Any:
        last_error: Optional[Exception] = None
        for attempt in range(_MAX_RETRIES):
            try:
                self._check_circuit_breaker()
                result = fn(*args)
                self._record_success()
                return result
            except _RETRYABLE as e:
                last_error = e
                self._record_failure(e)
                if attempt < _MAX_RETRIES - 1:
                    delay = _BACKOFF_BASE * (2 ** attempt)
                    logger.debug(
                        "Retryable error (attempt %d/%d), backing off %.1fs: %s",
                        attempt + 1, _MAX_RETRIES, delay, e,
                    )
                    time.sleep(delay)
            except (ConstraintViolation, QueryTimeoutError):
                raise
            except Exception as e:
                mapped = _map_neo4j_error(e)
                if isinstance(mapped, _RETRYABLE):
                    last_error = mapped
                    self._record_failure(mapped)
                    if attempt < _MAX_RETRIES - 1:
                        delay = _BACKOFF_BASE * (2 ** attempt)
                        logger.debug(
                            "Retryable error (attempt %d/%d), backing off %.1fs: %s",
                            attempt + 1, _MAX_RETRIES, delay, mapped,
                        )
                        time.sleep(delay)
                elif isinstance(mapped, (ConstraintViolation, QueryTimeoutError)):
                    raise mapped
                else:
                    raise mapped
        raise last_error  # type: ignore[misc]

    def _check_circuit_breaker(self) -> None:
        if self._circuit_breaker and not self._circuit_breaker.can_execute():
            raise InfrastructureError("Circuit breaker is open")

    def _record_success(self) -> None:
        if self._circuit_breaker:
            self._circuit_breaker.on_success()

    def _record_failure(self, exc: Exception) -> None:
        if self._circuit_breaker:
            self._circuit_breaker.on_failure(exc)

    def _run_read(self, query: str, params: Optional[dict], timeout: int) -> list[dict]:
        if self._driver is None:
            raise InfrastructureError("No Neo4j driver configured")
        session = self._driver.session()
        t0 = time.time()
        try:
            from neo4j import Query
            result = session.run(Query(query, timeout=timeout), params or {})
            records = [dict(r) for r in result]
            result.consume()
            from proxy.metrics import record_neo4j_query
            record_neo4j_query("read", time.time() - t0)
            return records
        except Exception as e:
            mapped = _map_neo4j_error(e)
            from proxy.metrics import record_neo4j_error
            record_neo4j_error(type(mapped).__name__)
            raise mapped from e
        finally:
            session.close()

    def _run_write(self, query: str, params: Optional[dict], timeout: int) -> list[dict]:
        if self._driver is None:
            raise InfrastructureError("No Neo4j driver configured")
        session = self._driver.session()
        t0 = time.time()
        try:
            from neo4j import Query
            result = session.run(Query(query, timeout=timeout), params or {})
            records = [dict(r) for r in result]
            result.consume()
            from proxy.metrics import record_neo4j_query
            record_neo4j_query("write", time.time() - t0)
            return records
        except Exception as e:
            mapped = _map_neo4j_error(e)
            from proxy.metrics import record_neo4j_error
            record_neo4j_error(type(mapped).__name__)
            raise mapped from e
        finally:
            session.close()

    def _run_transaction(self, fn: Callable) -> Any:
        if self._driver is None:
            raise InfrastructureError("No Neo4j driver configured")
        session = self._driver.session()
        try:
            tx = session.begin_transaction()
            try:
                result = fn(tx)
                tx.commit()
                return result
            except Exception:
                tx.rollback()
                raise
        except Exception as e:
            raise _map_neo4j_error(e) from e
        finally:
            session.close()
