# Error Taxonomy

## Overview

ChoriProx uses a structured error hierarchy to handle different failure modes across Neo4j operations, authentication, request validation, skill runtime state, and pipeline processing. Each error type has specific handling requirements and retry semantics.

## Invalid Request (400)

| Type | Code | Cause |
|---|---|---|
| `invalid_request_error` | `invalid_messages`, `invalid_stream`, `invalid_role`, `invalid_content`, `missing_tool_call_id`, etc. | Request body failed structural validation (spec §7.1). Envelope: `{"error": {"message", "type", "param", "code"}}`. |

## Skill State Conflict (409)

| Type | Code | Cause |
|---|---|---|
| `skill_state_conflict` | (int 409) | Active mode references a skill not in the registered snapshot (spec §7.2). Mode is cleared via compare-and-clear. |

## Skill Runtime Error (500)

| Type | Code | Cause |
|---|---|---|
| `skill_runtime_error` | (int 500) | Skill callback (prompt, tool policy, activation rewrite) raised an exception (spec §7.2 / §7.3). Mode is cleared. Underlying exception text is NOT echoed. |

## Error Classes

### InfrastructureError

- **Base class:** `Exception`
- **Module:** `src/memory/query_executor.py:34`
- **Raised when:**
  - Neo4j connection fails (`ServiceUnavailable`)
  - Session expires (`SessionExpired`)
  - Transient errors without timeout indicators
  - Circuit breaker is open
  - No Neo4j driver configured
- **Handling:** Caller should retry or degrade gracefully. The `QueryExecutor` automatically retries up to 3 times with exponential backoff (100ms, 200ms, 400ms).
- **Example:**
  ```python
  try:
      rows = executor.execute_read(query, params)
  except InfrastructureError:
      raise  # Let caller handle retry
  ```

### ConstraintViolation

- **Base class:** `Exception`
- **Module:** `src/memory/query_executor.py:39`
- **Raised when:**
  - Uniqueness constraint violated (duplicate entity)
  - Relationship constraint violated
  - Neo4j `ClientError` with "constraint", "already exists", or "duplicate" in message
- **Handling:** Not retryable. Caller should handle the conflict (e.g., return error to user, skip duplicate).
- **Example:**
  ```python
  try:
      rows = executor.execute_write_no_retry(query, params)
  except ConstraintViolation as e:
      return {"error": "duplicate", "message": str(e)}
  ```

### QueryTimeoutError

- **Base class:** `Exception`
- **Module:** `src/memory/query_executor.py:44`
- **Raised when:**
  - Query exceeds timeout limit
  - Neo4j `TransientError` or `ClientError` with "timeout" or "deadline" in message
- **Handling:** May be retryable. Caller can retry with longer timeout or degrade gracefully.
- **Example:**
  ```python
  try:
      rows = executor.execute_read(query, params, timeout=60)
  except QueryTimeoutError:
      logger.warning("Query timed out, retrying with longer timeout")
      rows = executor.execute_read(query, params, timeout=120)
  ```

### CircuitBreakerOpen

- **Base class:** `Exception`
- **Module:** `src/memory/circuit_breaker.py:182`
- **Raised when:**
  - Circuit breaker is in OPEN state (failure threshold exceeded)
  - Using `CircuitBreaker` as decorator and circuit is open
- **Handling:** Fail fast. Wait for recovery timeout (default 30s) before retrying.
- **Example:**
  ```python
  try:
      result = protected_function()
  except CircuitBreakerOpen as e:
      logger.warning(f"Circuit open: {e}")
      return fallback_response()
  ```

### AuthConfigError

- **Base class:** `Exception`
- **Module:** `src/proxy/auth.py:10`
- **Raised when:**
  - Model's `api_key_env` is configured but the environment variable is missing or empty
- **Handling:** Configuration error. Return 500 to client with helpful message (without exposing secrets).
- **Example:**
  ```python
  try:
      upstream = prepare_upstream_request(body, model_cfg, route)
  except AuthConfigError as e:
      return jsonify({
          "error": f"API key not configured for model '{e.model_name}'"
      }), 500
  ```

### StageAbort

- **Base class:** `Exception`
- **Module:** `src/proxy/pipeline/stages.py:33`
- **Raised when:**
  - Pipeline stage produces a complete response and processing should stop
  - Dispatch handlers fully handle the request (e.g., `^^help`, `^^status`)
  - Security stage blocks malicious content
- **Handling:** Not an error—control flow mechanism. Extract `response_data` and return to client.
- **Example:**
  ```python
  try:
      for stage in stages:
          result = stage.execute(ctx)
          if result:
              return result
  except StageAbort as e:
      return jsonify(e.response_data)
  ```

## Error Mapping

The `QueryExecutor._map_neo4j_error()` function translates Neo4j driver exceptions to ChoriProx error types:

| Neo4j Exception | Condition | ChoriProx Error |
|----------------|-----------|-----------------|
| `ServiceUnavailable` | Always | `InfrastructureError` |
| `SessionExpired` | Always | `InfrastructureError` |
| `TransientError` | Contains "timeout"/"deadline" | `QueryTimeoutError` |
| `TransientError` | Other | `InfrastructureError` |
| `ClientError` | Contains "timeout"/"deadline" | `QueryTimeoutError` |
| `ClientError` | Contains "constraint"/"already exists"/"duplicate" | `ConstraintViolation` |
| `ClientError` | Other | `InfrastructureError` |

## Circuit Breaker States

The `CircuitBreaker` class (`src/memory/circuit_breaker.py:18`) manages Neo4j failure isolation:

- **CLOSED:** Normal operation, all requests pass through
- **OPEN:** Circuit tripped after 3 failures (default), requests fail fast with `InfrastructureError("Circuit breaker is open")`
- **HALF_OPEN:** Testing recovery after 30s cooldown (default), allows single probe request

### State Transitions

```
CLOSED → OPEN: After failure_threshold (3) consecutive failures
OPEN → HALF_OPEN: After recovery_timeout_seconds (30s)
HALF_OPEN → CLOSED: Probe request succeeds
HALF_OPEN → OPEN: Probe request fails
```

## Best Practices

1. **Always handle InfrastructureError explicitly** in Neo4j operations
2. **Never retry ConstraintViolation**—it indicates a data conflict
3. **Use execute_write_no_retry()** for non-idempotent operations (pre-generate UUIDs)
4. **Check circuit breaker state** before expensive operations
5. **Log all errors** with request_id for debugging
6. **Sanitize error messages** before returning to clients (never expose API keys, internal paths)

## Complete Error Handling Pattern

```python
from memory.query_executor import (
    InfrastructureError,
    ConstraintViolation,
    QueryTimeoutError,
)
from memory.circuit_breaker import CircuitBreakerOpen

try:
    rows = executor.execute_write(query, params)
except InfrastructureError as e:
    logger.error("Neo4j infrastructure failure", extra={"request_id": ctx.request_id})
    raise  # Let caller retry or degrade
except ConstraintViolation as e:
    logger.warning("Duplicate entity", extra={"request_id": ctx.request_id})
    return {"error": "duplicate", "message": str(e)}
except QueryTimeoutError as e:
    logger.warning("Query timeout", extra={"request_id": ctx.request_id})
    return {"error": "timeout", "retry": True}
except CircuitBreakerOpen as e:
    logger.error("Circuit breaker open", extra={"request_id": ctx.request_id})
    return {"error": "service_unavailable", "retry_after": 30}
```
