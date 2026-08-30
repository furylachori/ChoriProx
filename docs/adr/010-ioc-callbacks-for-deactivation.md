# ADR-010: IoC callbacks for deactivation

## Status
Accepted (2026-06-02)

## Context

When an agent's skill mode expires or changes, ChoriProx needs to notify
the skill system so it can clean up skill-specific state (workspace
associations, tool registrations, task context). The natural implementation
is for `Agent` to call `SkillRegistry.deactivate_skill(agent, skill_name)`
directly.

This creates a circular import:

- `Agent` (in `proxy/agent/agent.py`) needs to import `SkillRegistry`
  (in `proxy/skills/skill_registry.py`) to fire deactivation events.
- `SkillRegistry` needs to import `Agent` (or `AgentRegistry`) to look up
  agents during skill activation and dispatch.

Python's import system resolves circular imports only when the imported
names are available at the time of import. In this case, `Agent` is not
fully defined when `SkillRegistry` tries to import it, causing
`ImportError: cannot import name 'Agent'`.

Lazy imports (importing inside method bodies) worked but were fragile:
they had to be placed in exactly the right methods, and any refactoring
that moved code between methods could re-introduce the circular import.

## Decision

Use Inversion of Control (IoC) via callbacks. `Agent` fires a list of
registered callbacks when its mode changes, instead of importing
`SkillRegistry` directly.

In `Agent.__init__()`:
```python
self._deactivation_callbacks: List[Callable] = []
```

When mode changes (expiration, explicit deactivation, or replacement):
```python
def _fire_deactivation_callbacks(self, old_mode):
    for cb in self._deactivation_callbacks:
        try:
            cb(self, old_mode)
        except Exception as e:
            logger.error("Deactivation callback failed: %s", e)
```

Callbacks are registered at agent creation time by `skill_lifecycle.py`,
which acts as a bridge between the agent and skill systems:

```python
def register_skill_callbacks(agent):
    from proxy.skills.skill_registry import SkillRegistry
    registry = SkillRegistry()
    agent.register_deactivation_callback(
        lambda agent, mode: registry.on_agent_mode_change(agent, mode)
    )
```

`skill_lifecycle.py` imports both `Agent` and `SkillRegistry`, but it is
only imported at runtime (during agent creation), not at module load time.
This breaks the static circular import chain.

## Consequences

**Easier:**
- No circular import. `Agent` has zero knowledge of `SkillRegistry`.
- Multiple subsystems can register deactivation callbacks independently
  (e.g., workspace cleanup, tool cache invalidation) without modifying
  `Agent`.
- Callbacks are registered explicitly, making the wiring visible and
  testable. Tests can register mock callbacks to verify deactivation
  events fire correctly.
- Adding new deactivation behavior requires only a new callback
  registration, not modifying `Agent` internals.

**More difficult:**
- The deactivation flow is harder to trace: a developer reading `Agent`
  sees `_fire_deactivation_callbacks()` but must search the codebase to
  find what actually handles deactivation. Code comments document the
  bridge in `skill_lifecycle.py`.
- Callback ordering is implicit (registration order). If two callbacks
  have dependencies (e.g., workspace cleanup must happen before tool
  cache invalidation), the ordering must be managed in
  `skill_lifecycle.py`.
- Debugging deactivation issues requires understanding the IoC pattern.
  A developer unfamiliar with the pattern may not think to look for
  callback registrations.

## Alternatives Considered

1. **Direct import with lazy resolution** — `Agent` imports `SkillRegistry`
   inside the deactivation method. Rejected because it's fragile: any
   import reordering or module restructuring can re-introduce the circular
   import.

2. **Event bus** — A global event bus where `Agent` publishes
   `"mode_changed"` events and `SkillRegistry` subscribes. Rejected as
   over-engineering for a single publisher-subscriber pair. The callback
   pattern is simpler and more explicit.

3. **Mediator module** — A `mode_change_mediator.py` that both `Agent` and
   `SkillRegistry` import (no cycle). Rejected because it's functionally
   identical to the callback pattern but with an extra module. The
   callback list on `Agent` is more direct.
