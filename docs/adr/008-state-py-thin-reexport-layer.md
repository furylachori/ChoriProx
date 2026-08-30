# ADR-008: state.py as thin re-export layer

## Status
Accepted (2026-06-02)

## Context

`proxy/state.py` was originally a 596-line monolith containing all shared
mutable state for the proxy: locks, dicts, counters, configuration, and
rate-limiting data. As ChoriProx grew, this file became a maintenance
bottleneck:

- Every module imported from `proxy.state`, creating a star-shaped dependency
  graph with `state.py` at the center.
- Changes to any state variable required understanding the entire file.
- The file was a frequent source of merge conflicts in parallel development.

A refactoring extracted logical groups into dedicated modules:

- `proxy.rate_limiter` — `RateLimiter` class and instances.
- `proxy.lock_ordering` — `make_lock()` with deadlock-prevention ordering.
- `proxy.workspace_state` — `WorkspaceStateStore` and workspace constants.
- `proxy.eviction_coordinator` — eviction cleanup coordination.

However, 67+ test files and ~20 production modules import names directly
from `proxy.state`:

```python
from proxy.state import tool_cache_lock, tool_cache, request_counter_lock
```

Changing all these imports simultaneously is high-risk: it requires
coordinated changes across 87+ files with no intermediate broken state.

## Decision

Keep `state.py` as a thin re-export layer that imports from the new modules
and re-exports under the original names. The file shrank from 596 to ~274
lines, containing:

- **Direct definitions** — locks, counters, and dicts that don't belong to
  any extracted module (e.g., `request_counter_lock`, `tool_cache`,
  `context_config`).
- **Re-exports** — names imported from extracted modules and re-exported:
  ```python
  from proxy.rate_limiter import RateLimiter, RateLimitResult
  from proxy.rate_limiter import (
      skill_rate_limiter, skill_ip_rate_limiter,
      command_rate_limiter, task_rate_limiter,
  )
  from proxy.workspace_state import (
      init_workspace_store, DEFAULT_WORKSPACE, WorkspaceStateStore,
  )
  from proxy.eviction_coordinator import build_and_process_deferred_clears
  ```

Existing imports from `proxy.state` continue to work unchanged. New code
can import directly from the source modules.

## Consequences

**Easier:**
- Zero breaking changes to existing code. All 67+ test files and 20+
  production modules continue to work without modification.
- The extracted modules can evolve independently. `rate_limiter.py` can add
  methods without touching `state.py`.
- `state.py` is now small enough (~274 lines) to understand at a glance.
- Gradual migration: callers can be moved to direct imports one at a time.

**More difficult:**
- `state.py` still exists as a module, which may confuse new developers who
  wonder where the "real" definitions live.
- Re-exports create two valid import paths for the same name, which can
  cause confusion in code reviews ("should I import from state or from
  rate_limiter?").
- IDE "go to definition" may navigate to the re-export in `state.py` instead
  of the source module, depending on the IDE's resolution strategy.

## Alternatives Considered

1. **Big-bang import migration** — Change all 87+ files in one PR. Rejected
   because it creates a large, risky PR that blocks all other development
   during merge conflict resolution.

2. **Deprecation warnings** — Add `__getattr__` to `state.py` that emits
   deprecation warnings for re-exported names. Rejected because it would
   flood test output and production logs before callers are migrated.

3. **Remove state.py entirely** — Force all imports from source modules.
   Rejected because it breaks backward compatibility with no incremental
   migration path.
