# `src/proxy/agent/AGENTS.md` — Agent & AgentRegistry

> The Agent dataclass owns per-session message state, mode, parent link,
> and lifecycle. AgentRegistry is the LRU cache with in-flight protection.
> Lock ordering: #0 (AgentRegistry._lock, coarsest) → #7 (Agent._mode_lock,
> per-instance).

## Module Purpose

The agent subsystem manages the lifetime of an `Agent` instance per session:
- LRU eviction (`AgentRegistry`)
- In-flight request counting (`request_scope`)
- Mode lifecycle (`set_active_mode` / `clear_active_mode` / `snapshot_mode` / `is_mode_expired`)
- Message buffer + parent link + telemetry

## Files in this module

| File | Purpose | Key lines |
|---|---|---|
| `agent.py` | `Agent` dataclass (444 LOC) + `_activate_skill` / `_inherit_skill` helpers | line 154 dataclass; line 177 `_buffer_replace_lock` (per-instance); line 197 `Agent._mode_lock` (lock #7); line 214 `clear_active_mode`; line 326 `snapshot_mode`; line 354 `is_mode_expired`; line 396+ `set_active_mode` |
| `agent_registry.py` | `AgentRegistry` (546 LOC) — singleton + LRU + `_in_flight_requests` + `request_scope` + `_gol_query_log` | line 26 `AGENT_REGISTRY_MAX_AGENTS`; line 46 `AgentRegistry._lock` (lock #0); line 64 `request_scope`; line 92 `get_or_load` |
| `agent_config.py` | `AgentConfig` dataclass (provider, tools, budget, …) | |
| `message_buffer.py` | `MessageBuffer` — per-agent message storage with thread-safe append/extend | |
| `types.py` | `ModeState` (and related types) | |

## Agent dataclass (agent.py:154)

```python
@dataclass
class Agent:
    _buffer_replace_lock: threading.Lock = field(default_factory=threading.Lock)
    agent_id: str = ""
    session_id: str = ""
    parent: Optional["Agent"] = None
    config: AgentConfig = field(default_factory=AgentConfig)
    is_subagent: bool = False
    _metadata: Optional[dict] = None
    context_limit_override: Optional[int] = None
    _buffer: MessageBuffer = field(default_factory=MessageBuffer)
    active_mode: Optional[Any] = None  # ModeState or None
    _mode_lock: Optional[threading.Lock] = None
    _ticket_mode_ts: Optional[float] = None
    telemetry: Any = field(default_factory=_make_telemetry)
    threshold_engine: Any = field(default_factory=_make_threshold_engine)
```

`agent_id` is derived from `session_id` via `sha256[:12]` in `__post_init__`
(line 208-209) when not explicitly provided. This makes the ID deterministic
and stable across LRU eviction/reload cycles (assignment model).

### `_buffer_replace_lock` (line 177)

Per-instance `threading.Lock` (M4-1 — was ClassVar). Each agent has its own
buffer-replace lock. **Intentionally outside `state.py` lock-ordering
hierarchy** — this lock never interacts with any other lock.

### `_mode_lock` (line 197) — lock #7

**Per-instance `threading.Lock`** allocated in `__post_init__` via
`make_lock(7, "Agent._mode_lock", threading.Lock)`. NOT a global in
`state.py`. **Do NOT add a `_MODE_LOCK` global to `state.py`** — that was
a recurring bug proposal.

## Public API — Mode Access (always use this, never raw reads)

```python
# CORRECT
mode_name = agent.get_mode_name()
mode_state = agent.snapshot_mode()
expired = agent.is_mode_expired()
is_ticket = agent.is_ticket_mode

# WRONG — never access _mode_lock directly
with agent._mode_lock:  # NEVER
    mode = agent.active_mode  # racy raw read

# WRONG — reading active_mode without holding the lock returns stale data
mode = agent.active_mode  # TOCTOU — use snapshot_mode()
```

### `clear_active_mode(expected_mode=None) -> bool` (line 214)

Pattern A: snapshot state under lock, clear under lock, call
`on_deactivate()` outside the lock. Safe for concurrent calls.

`expected_mode` is an optional CAS check: only clear if `agent.active_mode`
is the same object (identity check). Prevents a race where a new mode is
set between a staleness check and the clear.

Returns `True` if cleared (CAS succeeded), `False` if another thread set a
new mode (CAS failed).

### `set_active_mode(mode_state, expected_mode=None) -> bool` (line 396)

Atomic CAS swap. Deactivation callbacks fired outside the lock.

### `snapshot_mode() -> Optional[ModeState]` (line 326)

The canonical atomic read. Use this instead of direct `active_mode`
access. Eliminates TOCTOU.

### `snapshot_mode_and_ticket_ts() -> tuple[ModeState | None, float | None]` (line 335)

Atomic multi-field read — eliminates TOCTOU between reading `active_mode`
and `_ticket_mode_ts`. Used by `execute_proxy_work` (memory/tools/proxy_work.py:56).

### `get_mode_name()` (line 344)

Convenience: returns `mode.name` or `None` under the lock. Replaces the
common pattern of locking + reading + getattr chains.

### `is_mode_expired()` (line 354)

Checks `mode.is_expired()` under the lock. Returns `False` if no mode is
active.

### `is_ticket_mode` (property, line 282)

Returns `True` when `active_mode` is set, `name == 'ticket'`, and TTL has
not expired. Uses `_mode_lock` (SEC-001). Falls back to `_ticket_mode_ts`
legacy timestamp for edge cases.

### `is_skill_mode` (line 303)

Generic version of `is_ticket_mode` — works for all skills.

### `is_workspace_mode` (line 316)

Returns `True` when `active_mode.name == 'workspace'`. Workspace mode is
persistent (`ttl_seconds=0`), so no TTL check.

## AgentRegistry (agent_registry.py:29)

```python
class AgentRegistry:
    def __init__(self, max_agents: int = AGENT_REGISTRY_MAX_AGENTS):
        self.max_agents = max_agents  # default 100, env AGENT_REGISTRY_MAX_AGENTS
        self._agents: Dict[str, Agent] = {}
        self._agents_by_id: Dict[str, Agent] = {}
        self._lru: OrderedDict = OrderedDict()
        self._lock = make_lock(0, "AgentRegistry._lock", threading.Lock)
        self._children: Dict[str, List[Agent]] = {}
        self._in_flight_requests: dict[str, int] = {}
        self._gol_query_log: deque[tuple[float, str]] = deque()
        self._gol_call_count: int = 0
```

### Singleton access

```python
from proxy.agent.agent_registry import registry as agent_registry
agent = agent_registry.get_or_load(session_id)
```

### `request_scope(session_id)` context manager (line 64)

Marks a request as in-flight for a session. Prevents LRU eviction and
TTL cleanup during the request's lifetime. **Used by `RequestPipeline.run()`**
to wrap the entire stage loop (R3-H8 / R3-H9).

```python
with agent_registry.request_scope(session_id):
    response = pipeline.run(ctx)
```

### `get_or_load(session_id, caller_session_id=None)` (line 92)

Load or retrieve an agent. **Has TWO defenses against session_id enumeration:**

1. **`caller_session_id` validation** — callers must prove ownership.
2. **Rate limit** — `_GET_OR_LOAD_RATE_LIMIT = 100 unique session_ids per 60s` (line 89-90), tracked via `_gol_query_log` deque (R2-HIGH-1, O(1) popleft, pruned periodically).

If rate limit exceeded, returns `None`.

## Lock Ordering

Per `src/proxy/AGENTS.md`:
- `#0 AgentRegistry._lock` ← coarsest; must be acquired before all others
- `#7 Agent._mode_lock` ← per-instance; mode resolution after workspace resolution

When acquiring multiple locks: ascending order only. Never hold `#0` while
waiting for `#7`. Violations caused 8+ adversarial-review-round deadlocks.

## Key Invariants

1. **`Agent._mode_lock` is per-instance (#7).** NOT a global in `state.py`. Use `proxy.lock_ordering.make_lock(7, ...)` to create it.
2. **Always use `snapshot_mode()` for reads.** Never read `agent.active_mode` raw — TOCTOU.
3. **Public API for mode access.** `get_mode_name` / `snapshot_mode` / `is_mode_expired` / `set_active_mode` / `clear_active_mode` / `is_ticket_mode` / `is_skill_mode` / `is_workspace_mode`. Never touch `_mode_lock`.
4. **`request_scope` is the canonical in-flight protection.** `RequestPipeline.run()` wraps the whole loop in it.
5. **`agent_id` is deterministic** via `sha256(session_id)[:12]`. Stable across eviction.
6. **`_buffer_replace_lock` is outside the lock hierarchy.** Per-instance buffer mutex, never interacts with other locks.
7. **ModeState is `frozen=True`.** Use `object.__setattr__` to bypass for internal writes (see `_activate_skill` line 93, `_inherit_skill` line 137).
8. **MappingProxyType wrap on `skill_state`.** `MappingProxyType(dict(skill_ctx.skill_state))` preserves the read-only invariant after activation (CRIT-S.1).

## Landmines

- **TOCTOU on `active_mode`** — read without `_mode_lock` returns stale data. Always `snapshot_mode()`.
- **Mode lock global** — don't add `_MODE_LOCK` to `state.py`. The lock is per-instance.
- **`active_mode` raw mutation** — `frozen=True` ModeState raises `FrozenInstanceError` on plain attribute assignment. Use `object.__setattr__`.
- **`_buffer_replace_lock` hierarchy** — never acquire it inside `_mode_lock` or `_lock`. Same-class mutex, not in hierarchy.
- **`request_scope` exit cleanup** — on exception inside the `with` block, `__exit__` decrements `_in_flight_requests`. Don't catch + swallow exceptions that prevent the cleanup.
- **Enumeration of `session_id`** — without `caller_session_id` validation, the rate limit (100/60s) blocks enumeration. `_gol_query_log` is pruned periodically (not every call) — burst traffic during the prune window can briefly exceed 100 unique IDs.
- **`AGENT_REGISTRY_MAX_AGENTS < 100`** — combined with content-derived session IDs, allows eviction attacks by varying the first message. Keep this value high.
- **`set_active_mode` deactivation callbacks** — fired OUTSIDE the lock. If the callback acquires another lock, do so after `_mode_lock` release.
- **CAS on `clear_active_mode`** — `expected_mode` is identity, not equality. `is` check.

## Code Conventions

```python
# CORRECT — public API
mode = agent.snapshot_mode()
if mode is None or agent.is_mode_expired():
    ...
cleared = agent.clear_active_mode(expected_mode=mode)

# CORRECT — request scope
from proxy.agent.agent_registry import registry as agent_registry
with agent_registry.request_scope(session_id):
    # ... process request
    ...

# WRONG — raw read
mode = agent.active_mode  # racy TOCTOU
with agent._mode_lock:
    mode = agent.active_mode

# CORRECT — read from agent registry
agent = agent_registry.get_or_load(session_id)
if agent is None:
    return ...  # rate-limited or session doesn't exist
```

## External Dependencies

**Imports from:**
- `proxy.lock_ordering.make_lock` — `Agent._mode_lock` (position #7) + `AgentRegistry._lock` (position #0).
- `proxy.state` (none directly — `_in_flight_requests` is on the registry).
- `proxy.skills.skill_definition.ModeState` — type annotations.
- `proxy.pipeline.agent_telemetry.AgentTelemetry` — `_make_telemetry()` (lazy).
- `proxy.pipeline.threshold_engine.ThresholdEngine` — `_make_threshold_engine()` (lazy).
- `proxy.skill_lifecycle.register_lifecycle_callbacks` — AgentRegistry init.
- `memory.task_graph.TaskGraph` — for `_get_task_graph_singleton` (used elsewhere).

**Imported by:**
- `proxy.app` — primary request path.
- `proxy.skill_resolver` — `resolve_mode`.
- `proxy.pipeline.context_optimize` — sub-agent auto-register.
- `proxy.tool_pipeline.tool_guard` — `_check_mode_allowlist`.
- `proxy.tool_pipeline.tool_pipeline` — `_check_ticket_mode_orchestrator`.
- `proxy.mode` — `_activate_skill` / `_inherit_skill` (defined in agent.py but used here).
- `proxy.workspace_state` — `resolve_inherited_workspace`.
- `memory.tools.proxy_ticket` / `memory.tools.proxy_work` — mode gates.
- `skills.ticket_skill` / `skills.work_skill` — `_activate_skill` / `_inherit_skill`.
- Tests under `tests/proxy/`, `tests/unit/`, `tests/integration/`.

## Cross-References

- `src/proxy/AGENTS.md` — lock hierarchy + 15 positions (#0–#14).
- `src/proxy/skills/AGENTS.md` — SkillRegistry + TrustToken + the mode lifecycle from `BaseSkill.to_skill_definition`.
- `src/proxy/pipeline/AGENTS.md` — `RequestPipeline.run()` wraps in `request_scope`.
- `src/proxy/canary.py` (module docstring) — `parse_canary` sets `inherited_skill` / `workspace_name` on the RequestContext, then ContextOptimizeStage calls `resolve_mode`.
- `src/skills/AGENTS.md` — TicketSkill.on_activate / WorkSkill.on_activate hook into `set_active_mode`.
- `src/memory/tools/AGENTS.md` — `execute_ticket_tool` / `execute_proxy_work` use `snapshot_mode_and_ticket_ts()` for atomic mode read.