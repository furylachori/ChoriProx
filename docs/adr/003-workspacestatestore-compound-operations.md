# ADR-003: WorkspaceStateStore with compound operations

## Status
Accepted (2026-06-02)

## Context

Workspace state in ChoriProx involves multiple interrelated data structures:

- Active workspace per session (which workspace is currently selected).
- Workspace metadata (name, creation time, configuration).
- Workspace-scoped agent assignments.
- Workspace inheritance chains (child workspaces inheriting from parents).

Before `WorkspaceStateStore`, callers performed multi-dict read-modify-write
sequences under a shared lock. For example, activating a workspace required:

1. Read current active workspace from `active_workspaces` dict.
2. Deactivate old workspace (update `workspace_states` dict).
3. Activate new workspace (update `active_workspaces` and `workspace_states`).
4. Update agent registry references.

This sequence, if interrupted or reordered, could leave the system in an
inconsistent state (e.g., two active workspaces for one session, or an
active workspace with no corresponding entry in `workspace_states`).

The bugs were subtle and hard to reproduce because they depended on thread
scheduling. Two incidents in production involved workspace state corruption
under concurrent skill dispatch.

## Decision

Create `WorkspaceStateStore` (`src/proxy/workspace_state.py`) that provides
atomic compound operations:

- `activate(session_id, workspace_name)` — deactivates current workspace and
  activates the new one in a single atomic operation under the store's lock.
- `deactivate(session_id)` — deactivates the current workspace atomically.
- `snapshot(session_id)` — returns an immutable snapshot of the current
  workspace state (name, metadata, config) without holding the lock after
  return.

The store owns all workspace-related dicts and the shared `RLock`. External
code never accesses the internal dicts directly.

The `_deactivate_workspace` protocol orchestration (which involves skill
cleanup, agent mode changes, and memory updates) stays in `workspace.py`
because it cross-cuts multiple subsystems beyond state management.

## Consequences

**Easier:**
- Workspace activation/deactivation is a single method call with guaranteed
  atomicity.
- The `snapshot()` method provides safe read access without lock management.
- State invariants (e.g., "at most one active workspace per session") are
  enforced by the store, not by convention.
- Testing workspace state transitions requires only the store, not the full
  workspace orchestration module.

**More difficult:**
- The protocol orchestration for full workspace deactivation (skill cleanup,
  agent mode changes) is split between `WorkspaceStateStore` (state changes)
  and `workspace.py` (side effects). Developers must understand both.
- Adding new workspace state fields requires modifying the store's internal
  dicts and the `snapshot()` return type.

## Alternatives Considered

1. **Database transactions** — Use SQLite transactions for atomicity. Rejected
   because workspace state is hot-path, in-memory data accessed on every
   request. Database round-trips would add unacceptable latency.

2. **Optimistic locking with version counters** — Each workspace state has a
   version; CAS on update. Rejected because it pushes retry logic to every
   caller and doesn't prevent the inconsistent intermediate states during
   multi-step activation.

3. **Actor model** — Dedicated workspace state actor thread. Rejected as
   over-engineering for the access patterns; a single `RLock` with compound
   operations is simpler and sufficient.
