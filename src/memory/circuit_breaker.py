import logging
import threading
import time
from enum import Enum
from typing import Any, Callable, Optional, TypeVar

T = TypeVar("T")

logger = logging.getLogger("memory.circuit_breaker")


class CircuitState(Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitBreaker:
    """
    Thread-safe circuit breaker pattern implementation for Neo4j operations.

    States:
    - CLOSED: Normal operation, all requests pass through
    - OPEN: Circuit is tripped, requests fail fast
    - HALF_OPEN: Testing recovery, allowing a single probe request

    TOCTOU race condition is eliminated by:
    1. Per-operation state check (not startup-only)
    2. Atomic state transitions under lock
    3. Request tracking for HALF_OPEN probe
    """

    def __init__(
        self,
        failure_threshold: int = 3,
        recovery_timeout_seconds: float = 30.0,
        name: str = "neo4j",
    ) -> None:
        self._lock = threading.Lock()
        self._state: CircuitState = CircuitState.CLOSED
        self._failure_count: int = 0
        self._failure_threshold = failure_threshold
        self._recovery_timeout_seconds = recovery_timeout_seconds
        self._last_failure_time: Optional[float] = None
        self._name = name
        self._half_open_pending: bool = False

    @property
    def state(self) -> CircuitState:
        with self._lock:
            return self._state

    @property
    def failure_count(self) -> int:
        with self._lock:
            return self._failure_count

    def can_execute(self) -> bool:
        with self._lock:
            return self._can_execute_unsafe()

    def _can_execute_unsafe(self) -> bool:
        if self._state == CircuitState.CLOSED:
            return True

        if self._state == CircuitState.OPEN:
            if self._last_failure_time is None:
                return False

            elapsed = time.time() - self._last_failure_time
            if elapsed >= self._recovery_timeout_seconds:
                self._state = CircuitState.HALF_OPEN
                self._half_open_pending = True
                logger.info(
                    "[%s] Circuit transitioning to HALF_OPEN after %.1fs cooldown",
                    self._name,
                    elapsed,
                )
                return True
            return False

        if self._state == CircuitState.HALF_OPEN:
            if self._half_open_pending:
                return False
            self._half_open_pending = True
            return True

        return False

    def on_success(self) -> None:
        with self._lock:
            self._on_success_unsafe()

    def _on_success_unsafe(self) -> None:
        if self._state == CircuitState.HALF_OPEN:
            self._state = CircuitState.CLOSED
            self._failure_count = 0
            self._half_open_pending = False
            logger.info(
                "[%s] Probe request succeeded. Circuit CLOSED.",
                self._name,
            )
        elif self._state == CircuitState.CLOSED:
            if self._failure_count > 0:
                self._failure_count = 0
                logger.debug("[%s] Reset failure count to 0", self._name)

    def on_failure(self, exception: Optional[Exception] = None) -> None:
        with self._lock:
            self._on_failure_unsafe(exception)

    def _on_failure_unsafe(self, exception: Optional[Exception] = None) -> None:
        self._failure_count += 1
        self._last_failure_time = time.time()

        exc_str = f": {exception}" if exception else ""

        if self._state == CircuitState.HALF_OPEN:
            self._state = CircuitState.OPEN
            self._half_open_pending = False
            logger.warning(
                "[%s] Probe request failed. Circuit re-opened. Failures: %d%s",
                self._name,
                self._failure_count,
                exc_str,
            )
        elif (
            self._state == CircuitState.CLOSED
            and self._failure_count >= self._failure_threshold
        ):
            self._state = CircuitState.OPEN
            logger.warning(
                "[%s] Failure threshold (%d) reached. Circuit OPENED%s",
                self._name,
                self._failure_threshold,
                exc_str,
            )
        else:
            logger.debug(
                "[%s] Recorded failure %d/%d%s",
                self._name,
                self._failure_count,
                self._failure_threshold,
                exc_str,
            )

    def force_open(self) -> None:
        with self._lock:
            self._state = CircuitState.OPEN
            self._last_failure_time = time.time()
            logger.warning("[%s] Circuit FORCED OPEN by external request", self._name)

    def force_closed(self) -> None:
        with self._lock:
            self._state = CircuitState.CLOSED
            self._failure_count = 0
            self._last_failure_time = None
            self._half_open_pending = False
            logger.info("[%s] Circuit FORCED CLOSED by external request", self._name)

    def __call__(self, func: Callable[..., T]) -> Callable[..., T]:
        def wrapper(*args: Any, **kwargs: Any) -> T:
            if not self.can_execute():
                raise CircuitBreakerOpen(
                    f"[{self._name}] Circuit is {self._state.value}. "
                    f"Try again after {self._recovery_timeout_seconds}s."
                )

            try:
                result = func(*args, **kwargs)
                self.on_success()
                return result
            except CircuitBreakerOpen:
                raise
            except Exception as e:
                self.on_failure(e)
                raise

        return wrapper


class CircuitBreakerOpen(Exception):
    pass
