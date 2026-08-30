# ADR-002: RateLimiter as standalone class vs inline

## Status
Accepted (2026-06-02)

## Context

ChoriProx enforces rate limits across four independent domains:

1. **Skill dispatch** — limits how often skills can be triggered per session.
2. **Skill dispatch (IP-scoped)** — limits skill triggers per IP address.
3. **Command dispatch** — limits `^^` command invocations per session.
4. **Task creation** — limits task graph mutations per session.

Initially, each domain stored its rate-limit state (dicts of timestamps,
locks, cleanup timers) directly in `proxy.state`. This led to:

- **Error-prone dict manipulation** — callers had to acquire the correct lock,
  check the deque length, append timestamps, and clean up stale entries.
  Missing any step caused either false positives (blocking legitimate
  requests) or false negatives (allowing unlimited requests).
- **No encapsulation** — rate-limit logic was copy-pasted across 4 call sites
  with subtle differences (window size, max count, cleanup interval).
- **Testing difficulty** — tests had to manipulate internal dicts and locks
  to simulate rate-limited states.
- **No persistence** — rate-limit state was lost on restart, allowing burst
  abuse after every proxy restart.

## Decision

Extract rate limiting into a standalone `RateLimiter` class in
`src/proxy/rate_limiter.py`. Each domain gets its own `RateLimiter` instance
with independent configuration:

```python
skill_rate_limiter = RateLimiter(max_requests=10, window_seconds=60.0, name="skill")
skill_ip_rate_limiter = RateLimiter(max_requests=30, window_seconds=60.0, name="skill_ip")
command_rate_limiter = RateLimiter(max_requests=20, window_seconds=60.0, name="command")
task_rate_limiter = RateLimiter(max_requests=50, window_seconds=60.0, name="task")
```

The class provides:

- `check(key) → RateLimitResult` — atomic check-and-record under lock.
- `check_only(key) → RateLimitResult` — check without recording (read-only).
- `cleanup()` — evict stale entries (auto-called every 10 invocations).
- Optional `persist=True` flag for SQLite-backed state survival across restarts.

`RateLimitResult` is a frozen dataclass with `allowed: bool` and
`retry_after: float`, providing a clean contract for callers.

## Consequences

**Easier:**
- Callers use a single method call instead of 5-step dict manipulation.
- Rate-limit configuration (window, max) is co-located with the instance,
  not scattered across call sites.
- Tests can instantiate `RateLimiter` directly without touching `proxy.state`.
- The `persist` flag enables optional state survival across restarts without
  changing call-site code.

**More difficult:**
- Existing callers had to migrate from direct dict access to `RateLimiter.check()`.
  This required updating 42 tests that manipulated internal rate-limit state.
- The `RateLimitResult` return type is slightly more verbose than a bare `bool`
  at call sites, though the `retry_after` field is needed for 429 responses.

## Alternatives Considered

1. **Inline helper functions** — `check_rate_limit(key, domain)` function in
   `state.py`. Rejected because it still centralizes all domain configs in one
   module and doesn't solve the testing problem.

2. **Middleware-based rate limiting** — Flask `before_request` hook. Rejected
   because rate limiting is domain-specific (skill vs command vs task), not
   request-global. Middleware can't distinguish domains without parsing the
   request body.

3. **Redis-backed rate limiting** — Use Redis for distributed rate state.
   Rejected because ChoriProx runs as a single-process proxy; Redis adds
   operational complexity without benefit.
