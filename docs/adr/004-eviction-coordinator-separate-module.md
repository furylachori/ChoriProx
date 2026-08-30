# ADR-004: Eviction coordinator as separate module

## Status
Accepted (2026-06-02)

## Context

When ChoriProx's LRU chat history cache evicts sessions, the eviction
triggers a cascade of cleanup across three subsystems:

1. **Chat history** — Remove the evicted session's message cache.
2. **Agent registry** — Clear the agent's active mode (skill mode) for
   evicted sessions, since stale modes waste memory and can cause incorrect
   dispatch decisions.
3. **Workspace state** — Deactivate workspaces associated with evicted
   sessions.

The cleanup logic was initially embedded in `chat_history.py`, which created
a circular dependency problem:

- `chat_history.py` needed to import from `agent_registry.py` (to clear modes)
  and `workspace_state.py` (to deactivate workspaces).
- `agent_registry.py` already imported from `chat_history.py` (to access
  chat histories during agent loading).
- `workspace_state.py` imported from `agent_registry.py` (for agent lookups
  during workspace activation).

This three-way circular dependency caused `ImportError` at module load time.
Attempts to break the cycle with lazy imports (`import` inside function
bodies) worked but scattered the coordination logic across modules, making
the eviction flow hard to understand and test.

## Decision

Extract the cross-cutting cleanup logic into a standalone module:
`src/proxy/eviction_coordinator.py`.

The module provides:

- `build_and_process_deferred_clears(evicted_session_ids)` — the public
  entry point called by `chat_history.py` after eviction.
- `_build_deferred_mode_clears(evicted_ids)` — reads agent registry and
  workspace state to determine which agents need mode clearing.
- `_process_deferred_clears(deferred)` — executes the actual mode clears
  and workspace deactivations.

The coordinator uses lazy imports for `agent_registry` and `workspace_state`
to avoid circular dependencies at module load time. Since it is only called
at runtime (during eviction), the lazy import cost is negligible.

`chat_history.py` calls `build_and_process_deferred_clears` **after**
releasing `chat_histories_lock` to prevent lock ordering violations with
the agent registry lock.

## Consequences

**Easier:**
- The eviction cleanup flow is visible in one file instead of scattered
  across three modules.
- Testing eviction behavior requires mocking only the coordinator's
  dependencies, not the full chat history cache.
- Adding new cleanup steps (e.g., clearing tool caches for evicted sessions)
  requires modifying only the coordinator.

**More difficult:**
- The coordination logic is separated from the data it operates on. A
  developer reading `chat_history.py` must follow the call to
  `eviction_coordinator.py` to understand the full eviction behavior.
- Lazy imports inside the coordinator are a code smell that signals
  architectural coupling, but the alternative (restructuring the three
  modules) would be a larger, riskier refactor.

## Alternatives Considered

1. **Event bus** — Publish eviction events; subscribers handle cleanup.
   Rejected because it adds infrastructure (event bus, subscriber
   registration) for a single use case. The coordinator is simpler.

2. **Restructure module boundaries** — Merge `chat_history.py` and
   `agent_registry.py` into one module. Rejected because they have
   distinct responsibilities and different change frequencies.

3. **Dependency injection** — Pass cleanup callbacks to `ChatHistoryCache`
   at construction time. Rejected because the callbacks need access to
   modules that aren't available at cache construction time (early in
   the import cycle).
