# ADR-001: QueryExecutor over per-method session management

## Status
Accepted (2026-06-02)

## Context

Early in ChoriProx's development, every Neo4j-accessing method in `graph.py`,
`task_graph.py`, and `sync.py` created its own `session.run()` calls. This
produced 25+ ad-hoc patterns for retry logic, error handling, session
lifecycle, and circuit breaker integration.

The problems were:

1. **Inconsistent retry behavior** — some methods retried on
   `ServiceUnavailable`, others did not. A transient Neo4j outage could crash
   some code paths while others recovered silently.
2. **No centralized error mapping** — Neo4j exceptions (`TransientError`,
   `ClientError`, `SessionExpired`) were caught and re-raised inconsistently,
   making it impossible for callers to handle infrastructure failures uniformly.
3. **Circuit breaker bypass** — methods that skipped the circuit breaker could
   hammer a failing Neo4j instance, worsening outages.
4. **Session leaks** — ad-hoc `session.run()` patterns occasionally failed to
   close sessions in error paths, leaking connections under load.
5. **Testing difficulty** — every method needed its own Neo4j mock setup,
   duplicating boilerplate across 80+ tests.

## Decision

All Neo4j access goes through `QueryExecutor` (`src/memory/query_executor.py`).
No direct `session.run()` calls are permitted anywhere in the codebase.

`QueryExecutor` provides three execution methods:

- `execute_read(query, params)` — for idempotent reads with automatic retry.
- `execute_write(query, params)` — for writes with automatic retry.
- `execute_write_no_retry(query, params)` — for non-idempotent writes where
  retry could cause duplicates (e.g., UUID-bearing entity creation).

All three methods route through a single `_execute()` core that handles:

- Session lifecycle (open/close in a `finally` block).
- Retry with exponential backoff (3 retries, 0.1s base).
- Error mapping to `InfrastructureError`, `ConstraintViolation`, or
  `QueryTimeoutError`.
- Circuit breaker integration (open circuit → immediate `InfrastructureError`).

## Consequences

**Easier:**
- Adding a new Neo4j query requires only a query string and params — no
  session management boilerplate.
- Retry policy, backoff timing, and error taxonomy are changed in one place.
- Tests mock `QueryExecutor` methods instead of Neo4j driver internals.
- Circuit breaker is enforced uniformly; no code path can bypass it.

**More difficult:**
- Write methods that must not retry (e.g., non-idempotent entity creation)
  must explicitly use `execute_write_no_retry`. Forgetting this can cause
  silent duplicates.
- The abstraction layer adds one level of indirection when debugging Neo4j
  query failures — stack traces include `QueryExecutor._execute()`.

## Alternatives Considered

1. **Per-module executor instances** — Each module (`graph.py`, `task_graph.py`)
   owns its own executor with module-specific retry config. Rejected because
   it fragments error handling and makes global circuit breaker integration
   impossible.

2. **Decorator-based retry** — A `@with_retry` decorator on each method.
   Rejected because it still requires per-method session management and makes
   circuit breaker integration implicit.

3. **Neo4j driver's built-in retry** — The driver has `max_transaction_retry_time`.
   Rejected because it does not integrate with our circuit breaker, does not
   map errors to our exception hierarchy, and does not distinguish retryable
   from non-retryable writes.
