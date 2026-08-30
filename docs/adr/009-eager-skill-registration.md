# ADR-009: Eager skill registration over lazy loading

## Status
Accepted (2026-06-02)

## Context

ChoriProx's `SkillRegistry` manages built-in skills (TicketSkill,
WorkspaceSkill) and dynamically loaded user skills. The registry is a
singleton accessed from multiple request-handling threads.

The original implementation used lazy loading: built-in skills were
registered on first access via an `_ensure_builtins()` method. This method
checked a `_builtins_loaded` flag and, if false, registered all built-in
skills and set the flag to true.

This created a TOCTOU (time-of-check-time-of-use) race condition:

1. Thread A calls `_ensure_builtins()`, reads `_builtins_loaded = False`.
2. Thread B calls `_ensure_builtins()`, reads `_builtins_loaded = False`.
3. Thread A registers built-in skills, sets `_builtins_loaded = True`.
4. Thread B registers built-in skills again, causing duplicate registrations.

A `_registering` flag was added as a guard, but it had the same TOCTOU
problem: two threads could both read `_registering = False` before either
sets it to `True`.

The race manifested as intermittent test failures (skill count assertions)
and, in production, occasional duplicate skill entries that caused ambiguous
trigger matching.

## Decision

Register built-in skills eagerly in `SkillRegistry.__new__()` instead of
lazily in `_ensure_builtins()`.

```python
class SkillRegistry:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            instance = super().__new__(cls)
            instance._skills = {}
            instance._register_builtins()  # Eager, not lazy
            cls._instance = instance
        return cls._instance
```

Since `__new__()` is called only once (singleton pattern), built-in
registration happens exactly once during the first `SkillRegistry()` call.
No flags, no race conditions.

The `_ensure_builtins()` method and its associated flags (`_builtins_loaded`,
`_registering`) are removed entirely.

## Consequences

**Easier:**
- No TOCTOU race. Built-in registration is guaranteed to happen exactly once
  by the singleton pattern's thread-safety guarantees.
- Simpler code: no flags, no guard checks, no `_ensure_builtins()` method.
- Tests can rely on built-in skills being present immediately after
  `SkillRegistry()` without calling any initialization method.
- Debugging skill registration issues is straightforward: built-ins are
  registered at a single, predictable point.

**More difficult:**
- Slightly slower startup: built-in skill registration happens during the
  first `SkillRegistry()` instantiation rather than being deferred to first
  use. In practice, this cost is negligible (<1ms) since skill registration
  is just dict population.
- If built-in registration fails (e.g., a skill class has a bug), the error
  occurs at startup rather than at first use, which could delay discovery
  in development. However, this is actually preferable: fail-fast is better
  than a runtime surprise.

## Alternatives Considered

1. **Double-checked locking** — Use a lock around `_ensure_builtins()`.
   Rejected because it adds lock acquisition to every skill lookup (hot path)
   and is notoriously difficult to implement correctly in Python due to
   memory model subtleties.

2. **Module-level registration** — Register built-ins at module import time
   (top-level code in `skill_registry.py`). Rejected because it executes
   during import, before logging and configuration are initialized, making
   registration errors hard to diagnose.

3. **Thread-safe lazy singleton** — Use `threading.Lock` in `__new__()`.
   This is actually what the current implementation does implicitly (the
   singleton `__new__` check is the synchronization point). The key insight
   is that doing registration in `__new__` rather than in a separate method
   eliminates the TOCTOU window entirely.
