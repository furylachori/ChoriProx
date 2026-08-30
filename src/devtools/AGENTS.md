# `src/devtools/AGENTS.md` — Debug Harness

> `debug_session.py` is a pytest-free Flask-app reproducer harness
> (~1520 LOC; measured post-edit via `wc -l src/devtools/debug_session.py`)
> that mirrors the `tests/integration/conftest.py:proxy_with_mock` fixture.
> It is the canonical entry point for running the proxy outside pytest —
> IPython, scratch scripts, debugging sessions.

## Module Purpose

Provides:
- `debug_session(...)` context manager that wires the proxy to a mock backend
- `DebugSession` class with chat / canary / subagent / activation helpers
- `_reset_all_shared_state()` — the canonical mirror of the test conftest
  reset (the spec additions: `escalation_rate_limiter`, `ToolCallTracker._instances`,
  `canary_verify_cache`, `_chain_state`, `vfs_caches`, `vfs_write_log`)
- `MockBackendHandler` + `_captured_bodies` for canned-response testing
  (caller-owned canned dicts are **deep-copied** before control keys are
  popped — they survive retries without mutation)
- `ProxyResponse` wrapper around Flask test client responses

## Files in this module

| File | Purpose |
|---|---|
| `__init__.py` | Package marker |
| `debug_session.py` | ~1520 LOC — `debug_session` context manager + `DebugSession` + `ProxyResponse` + `_reset_all_shared_state` + mock backend (caller-owned canned dicts are deep-copied before control-key pop) |
| `shell.py` | 33 LOC — IPython-friendly shell helpers |

## `debug_session(...)` context manager (line ~1273)

```python
@contextmanager
def debug_session(
    canned_responses: Optional[list[dict]] = None,
    db_path: Optional[str] = None,
    secret: str = "",
    api_key: str = "",
    neo4j_disabled: bool = True,
    vfs_enabled: bool = False,
    log_level: str = "DEBUG",
    log_format: str = "json",
    models: Optional[list[str]] = None,
    *,
    skills_dir: Optional[str] = None,
    enable_custom_skill_subagents: bool = False,
):
```

### What it does

1. **Env var setup (BEFORE import)** — sets `PROXY_ENV=dev`, `PROXY_SESSION_SECRET`, `PROXY_API_KEY`, `PROXY_VFS_ENABLED`, `PROXY_LOG_FORMAT`, `PROXY_DEBUG`, `TEST_API_KEY`. Saved to `saved_env` for restoration.
2. **When `skills_dir` is explicitly supplied (not None):** sets `PROXY_SKILLS_DIR` to that path (`""` = disabled, non-empty = override) and `PROXY_ENABLE_CUSTOM_SKILL_SUBAGENTS` to `"1"` (when `enable_custom_skill_subagents is True`) or `"0"` (deterministic default when only `skills_dir` is supplied). Both keys are added to `saved_env` BEFORE mutation and restored in `finally`.
3. **Patches `memory.sync.submit_sync` AND `memory.sync.submit_ancient_compress`** to no-ops BEFORE proxy import — UNCONDITIONALLY (regardless of `neo4j_disabled`). Mirrors the autouse `mock_sync_submit` fixture in `tests/conftest.py` for `submit_sync`, and additionally suppresses the bounded `_summary_pool` dispatch that `_apply_summary_budget` invokes after a trim.
4. **Pre-initializes the tmp SQLite schema** via `memory.db.init_db()` using the test DB path, so the proxy's rate-limiters / chat-history / post-process paths don't deadlock on never-created tables. The `:memory:` keeper block inside `init_db()` keeps the shared-cache schema alive across the migration-connection close.
5. **Spawns mock backend on dynamic port** — `HTTPServer(("127.0.0.1", 0), handler_cls)`. Port assigned by OS. Emits JSON only — no genuine upstream SSE is supported.
6. **Writes tmp `proxy_models.json`** pointing at the mock backend.
7. **When `skills_dir` is explicit:** calls `reset_default_registry()` + `get_default_registry()` to publish the custom JSON skills to the singleton registry.
8. **Imports `qwen_proxy`, wires registries/monkeypatches.**
9. **Yields a `DebugSession` bound to `app.test_client()`.**
10. **On exit:** shutdown server; reset default registry (when explicit); `_reset_all_shared_state()`; restore env.

### Custom-skill / swarm-capable configuration

```python
# Enable a swarm-capable custom skill:
with debug_session(
    canned_responses=canned,
    secret="...",                         # non-empty required for HMAC signing
    skills_dir="/path/to/skills",         # or '' to explicitly disable
    enable_custom_skill_subagents=True,   # must be True for task admission
) as c:
    ...
```

**Landmine — all three required.** A swarm-capable debug session requires EACH of: explicit `skills_dir`, `enable_custom_skill_subagents=True`, AND non-empty `secret`. Omitting any one means the registry build either rejects the config with a `swarm_*` configuration error, disables `task` admission, or falls back to the repo-default `skills_dir`.

**Landmine — first-request activation race.** The first `c.chat()` to a custom-skill trigger does NOT auto-register an `Agent` for the session — the agent load goes through `AgentRegistry.get_or_load()` which can return `None` in a fresh debug harness, and activation fails with `Skill activation failed`. Use `c.ensure_agent(session_id)` before the first call. Custom skills activate via the real `^^<name>` trigger (NOT via `activate_ticket`/`activate_work`).

### Critical: deterministic exit via `_finish()`

`debug_session` patches `memory.sync.submit_sync` AND
`memory.sync.submit_ancient_compress` to no-ops (UNCONDITIONALLY), so the
harness does not enqueue SOTA sync work during scenarios. Templates MUST
still call `_finish()` (in `tests/debug/templates/_helpers.py`) to convert
accumulated check results into a deterministic process exit code (0 all
passed, 1 any failed).

External reproducer code that bypasses `debug_session`'s `submit_sync`
patch may still encounter non-daemon executor shutdown behavior — in that
case end the script with `sys.stdout.flush(); os._exit(0)` to force exit.

## `DebugSession` class (line 388)

Public API (selected methods):

| Method | Purpose |
|---|---|
| `chat(text, ...)` | Send a chat message via the Flask test client |
| `send_as_subagent(...)` | Send a request with a canary marker. Optional `tools=` forwarded so the proxy has a client-set to filter. |
| `ensure_agent(session_id)` | Register a fresh `Agent` for `session_id` via the canonical public API (`AgentRegistry.register` + `register_lifecycle_callbacks`). Required before the **first** `c.chat()` to a custom-skill trigger — otherwise the dispatcher's agent load returns `None` and activation fails. Idempotent: uses `AgentRegistry.get_or_load` for the duplicate check and does NOT mutate private registry dictionaries. |
| `activate_ticket(...)` | Activate ticket mode for a session |
| `inspect_agent(session_id)` | Return the agent's current `ModeState` snapshot, or `None` (no active mode) |
| `get_chat_history(session_id)` | Read chat history from cache |
| `get_tool_counts(...)` | Return `ToolCallTracker` status |
| `set_tool_limit(tool, n)` | Override `ToolLimits` per-instance |
| `extract_task_prompts(response)` | Parse response JSON → list of task-call `prompt` strings ([]) for non-task |
| `extract_swarm_canary(response)` | Return the canary MARKER prefix from the first swarm-signed task call (None if none) |
| `dump_captured()` | Dump captured request bodies from the mock backend |
| `reset()` | Reset all shared state |
| `time_travel(seconds)` | Shift clock for TTL tests |
| `sign_canary(skill_name, state_json)` | HMAC-sign a canary string |
| `capture_canary(text)` | Parse and return the canary from a response (legacy) |
| `rate_limit_status(name)` | Inspect a named rate limiter |
| `last_tool_call(...)` | Return the last tool call seen |
| `canned` | Builder namespace for canned OpenAI-shape responses. New: `task_call`, `task_calls` |
| `state` | The shared `proxy.state` module reference |

## `_reset_all_shared_state()` (line 232)

The canonical state-reset helper. Mirrors
`tests/integration/conftest.py:_reset_all_shared_state` with the spec
additions. Called by `DebugSession.reset()` and on `debug_session` exit.

```python
def _reset_all_shared_state() -> None:
    # Rate limiters — delegate to single-source-of-truth helper
    import proxy.state as state
    state.reset_all_rate_limiters()

    # Chat histories + LRU
    from proxy.chat_history import chat_histories, chat_histories_lock, _chat_histories_lru
    with chat_histories_lock:
        chat_histories.clear()
        _chat_histories_lru.clear()

    # Chat summaries
    with state.chat_summaries_lock:
        state.chat_summaries.clear()

    # Workspace state
    with state.workspace_store._lock:
        state.workspace_store._active.clear()
        state.workspace_store._recent.clear()
        state.workspace_store._last_access.clear()

    # Tool cache
    with state.tool_cache_lock:
        state.tool_cache.clear()
    # ... (vfs_caches, canary_verify_cache, _chain_state, ToolCallTracker._instances, etc.)
```

Three call sites for this pattern (verified):

- `src/devtools/debug_session.py:_reset_all_shared_state` (line 232)
- `tests/integration/conftest.py:_reset_all_shared_state`
- `tests/conftest.py:reset_rate_limiters` (autouse fixture)

All three call `state.reset_all_rate_limiters()`, so adding a new limiter
auto-propagates.

## `ProxyResponse` (line 56)

```python
@dataclass
class ProxyResponse:
    status_code: int
    data: bytes
    _json: Optional[dict] = None

    @property
    def content(self) -> str:
        return self.data.decode("utf-8", errors="replace")

    def json(self) -> Optional[dict]:
        if self._json is not None:
            return self._json
        try:
            return json.loads(self.data)
        except (json.JSONDecodeError, TypeError):
            return None
```

Thin wrapper around Flask test client response. Lazy JSON parsing.

## Mock backend handler

`_make_handler_class(canned_responses, captured_bodies)` factory returns an
HTTP request handler class. Default response: walks through
`canned_responses` list, returns each as JSON, captures request bodies
to `_captured_bodies`. Used for deterministic responses in tests.

**Canned-response immutability.** When the handler dequeues a canned entry,
it `copy.deepcopy()` the entry BEFORE popping control keys (`_status`,
`_body`, `_retry_after`). The caller-owned fixtures remain unchanged
across retries — only the detached copy loses the control keys. Hidden
control keys (`_status`, `_body`, `_retry_after`) are NEVER serialized
into the JSON body that the proxy receives.

The mock backend emits JSON only — no genuine upstream SSE is supported
by these templates. Production callers still pass `wants_stream=False`
from `src/proxy/pipeline/stage_modules/upstream.py` and
`src/proxy/app.py`'s keepalive generator. Streaming-disconnect /
`GeneratorExit` remains the responsibility of unit-level tests
(`tests/unit/test_app_generator_exit.py`).

## Key Invariants

1. **`debug_session` is the ONLY pytest-free harness** for running the proxy end-to-end.
2. **`_reset_all_shared_state` mirrors the conftest pattern.** All three sites call `state.reset_all_rate_limiters()` for consistency.
3. **Env vars are set BEFORE import** of `qwen_proxy` (line ~1290). Reordering breaks module-load config.
4. **Mock backend on dynamic port** — `HTTPServer(("127.0.0.1", 0), handler_cls)`. Port assigned by OS to avoid collisions.
5. **`_SESSION_SECRET` is module-level cache** — test-only monkey-patch surface.
6. **`USE_ADAPTERS` env override** — `PROXY_USE_ADAPTERS=0` disables client adapters.
7. **`memory.sync.submit_sync` AND `memory.sync.submit_ancient_compress`
   are patched UNCONDITIONALLY** (regardless of `neo4j_disabled`). The
   Neo4j-specific patches (`memory.graph.Neo4jClient`,
   `memory.task_graph.Neo4jClient`, `memory.sync.init_db`,
   `memory.query_executor.get_shared_executor_from_env`,
   `memory.workspace_persistence._get_executor`) remain conditional.
8. **Mock backend deep-copies each canned entry before popping control
   keys.** Caller-owned canned dicts survive retries without mutation.
9. **`_finish()` remains required** for every template — ordinary
   templates retain the existing deterministic `os._exit()` contract
   after their context managers have closed. No executor template uses
   normal interpreter shutdown in this wave.

## Landmines

- **External reproducers bypassing `submit_sync` patch** — `debug_session` no-ops SOTA sync work, but external scripts that bypass the patch may need `os._exit(0)` to force shutdown of non-daemon `_sync_pool` workers.
- **`memory.sync.submit_ancient_compress` no-op** — the bounded
  `_summary_pool` dispatch that `_apply_summary_budget` invokes after a
  trim is patched UNCONDITIONALLY (not just under `neo4j_disabled`). The
  budget DECISION is real and exercised by templates; the post-trim
  SOTA side effect is intentionally absent. Do not patch it again in a
  template — a second patch re-enables real SOTA network traffic.
- **Env var save/restore race** — `saved_env` is captured BEFORE import but restored AFTER teardown. If anything mutates `os.environ` between, the restore may overwrite a real env value.
- **`proxy.state` import in `_reset_all_shared_state`** is INTENTIONAL — uses the module reference (per AGENTS.md "Proxy state import pattern") to avoid local shadows.
- **`MockBackendHandler` captures bodies** — `_captured_bodies` grows unbounded. For long sessions, manually clear it.
- **Canned-dict immutability** — the handler deep-copies the canned entry
  before popping control keys. Do NOT change this back to a shallow pop:
  the source-dict mutation would surface in caller-owned fixtures after a
  retry round-trip.
- **Mock backend is JSON-only** — `_make_handler_class` does NOT support
  `text/event-stream`. Production code that activates genuine upstream
  SSE is a separate behavior change; do NOT claim SSE support via these
  templates. Production callers still pass `wants_stream=False` from
  `src/proxy/pipeline/stage_modules/upstream.py` and the keepalive
  generator in `src/proxy/app.py`.
- **No executor template** — terminal `ThreadPoolExecutor` shutdown can
  hang in CPython before ordinary atexit handlers fire, so a wedged-executor
  template could mask the real production exit-code signal. Executor
  lifecycle remains unit/subprocess-owned; an `l_executor_shutdown.py`
  was intentionally NOT added in this wave.
- **`time_travel` only shifts clock via monkey-patching** — does NOT affect thread-spawned timers. Tests relying on thread-based timeouts may need additional patches.
- **`rate_limit_status(name)` uses `proxy.state` attribute scan** — same as `reset_all_rate_limiters()`. Future-drift guard in `reset_all_rate_limiters()` asserts non-empty `_name`.
- **`inspect_agent(session_id)` returns the live agent** — mutations propagate to subsequent requests.
- **`sign_canary` uses the proxy's HMAC signing** — equivalent to what `BaseSkill.build_skill_canary` produces.
- **`capture_canary()` is legacy and WRONG for swarm flows** — it reads `messages[-1].content` but swarm canaries live in `task_calls[].function.arguments.prompt`. Use `extract_swarm_canary(response)` for swarm parents instead.

## Runnable Template Inventory (14 worked-example templates)

| Template | Purpose |
|---|---|
| `a_single_turn.py` | One-shot `c.chat()` returns 200 with expected content |
| `b_mode_lifecycle.py` | Mode pre-seeding + per-turn stop-deactivation |
| `c_subagent_canary.py` | `c.sign_canary()` + `c.send_as_subagent()` + inheritance check |
| `d1_memory_tool_loop.py` | Two-canned memory-tool re-query |
| `d2_normalization_healing.py` | Tool-name / arg normalization (AGENTS.md #1 landmine) |
| `e_ttl_expiry.py` | TTL expiry two-session design |
| `f_provider_compat.py` | Kimi temperature clamp |
| `g_model_context_switch.py` | Multi-message trim and single-message trim across glm-5.2 / mini |
| `g_swarm_parent_child.py` | Custom research-swarm parent/child happy path |
| `h_swarm_guards.py` | Swarm runtime guards (budget, no-nesting, no-task pass-through, operator visibility) |
| `i_retry_lifecycle.py` | Negative and zero Retry-After end-to-end through Flask → pipeline → mock backend; caller-owned canned fixtures remain unmutated across retries; JSON wire body has no hidden control keys |
| `j_sqlite_lifecycle.py` | `:memory:` keeper + canonical message_summaries unique constraint; repeated `/health` SQLite ownership; pre-init raw `sqlite3` legacy seed + closed connection → `debug_session` re-runs the production migration |
| `k_summary_budget.py` | Real SQLite messages + summaries with misleading `rendered_*_tokens = 1`; production budget path uses actual rendered row cost (`[msg#{i}] {role}: {summary}`), drops oldest-first, retains a contiguous suffix within the 15% safe-budget cap; captured upstream body checked for current request marker + trimmed summary block |
| `l_vfs_cache_safe_tail.py` | Main-agent VFS cache-safe tail placement (US-127). Four scenarios through debug_session + real VFS builder + real V2 pipeline + mock backend: (A) canonical ordering — system → completed history → VFS → live user, VFS contains cached path, no live-user bleed; (B) §13.5 trim safety net — oversized live user trimmed with `... [truncated]`, VFS dropped per §13.1 #7/AC-15 priority, status 200 not 413; (C) byte-identical VFS content across multiple requests; (D) workspace switch replaces only the VFS block, prefix before VFS unchanged. Workspace activation via `state.workspace_store.set_active(session_id, workspace, client_hint=...)` (client-level mapping — the proxy derives session_id from request body and looks up workspace by client_hint fallback). |
| `_helpers.py` | Shared `_emit()` + `_finish()` — every template imports these |

No `l_executor_shutdown.py` was added in this wave — see Landmine above.

## Ownership Distinction

| Layer | Owns | Examples |
|---|---|---|
| **Unit tests** | exact retry sleep duration, exact cleanup ordering, exact edge-case decode behavior | `tests/unit/test_http_client_streaming.py`, `tests/unit/test_summary_budget.py`, `tests/unit/test_sync_pool_routing.py`, `tests/unit/test_db_recovery.py` |
| **Integration tests** | precise helper / database / contract assertions | `tests/integration/test_admin_endpoints.py`, `tests/integration/test_debug_session.py` |
| **Debug templates** | full Flask → 9-stage pipeline → mock-backend observable behavior (caller-owned canned fixtures, captured wire shapes, end-to-end exit codes) | `tests/debug/templates/*.py` |

A new test that mocks `_summary_token_cost`, `_apply_summary_budget`,
`_select_context_v2`, `store_message_summary`, `get_rich_summaries`, or
the captured upstream body should NOT be added — these debug templates
exercise the production call-stack with REAL production calls. The
`_make_handler_class` deep-copy and the unconditional no-op async patches
live exclusively in the harness; they do not propagate to production
code paths.

## Code Conventions

```python
# CORRECT — basic debug session
from devtools.debug_session import debug_session

with debug_session(
    canned_responses=[{"choices": [{"message": {"role": "assistant", "content": "Hello!"}}]}],
    secret="my-secret",
    api_key="sk-test-key",
) as session:
    response = session.chat("Hello, world!")
    print(response.json())

# CORRECT — subagent canary
with debug_session(secret="my-secret") as session:
    canary = session.sign_canary("ticket", '{"task_hash": "abc12345"}')
    response = session.send_as_subagent(canary_marker=canary, prompt="...")

# CORRECT — reset state mid-session
with debug_session() as session:
    session.reset()  # full _reset_all_shared_state
    response = session.chat("Fresh start")

# CORRECT — IPython-friendly shell
from devtools.shell import start_shell
start_shell(session)  # drops into IPython with session bound
```

## External Dependencies

**Imports from:**
- `proxy.state` — module reference for state mutations.
- `proxy.chat_history` — `chat_histories`, `chat_histories_lock`, `_chat_histories_lru`.
- `proxy.openai_compat` (indirect via `build_openai_response`).
- `memory.tools.*` (transitively via the proxy app).
- `requests` — outbound HTTP from reproducer scripts.
- `unittest.mock.MagicMock`, `patch` — for monkey-patching.
- `http.server.BaseHTTPRequestHandler`, `HTTPServer` — mock backend.

**Imported by:**
- `tests/integration/conftest.py:proxy_with_mock` — shares pattern (NOT direct import — duplicated for fixture isolation).
- `tests/debug/templates/_helpers.py` — `_finish()` exit helper.
- External reproducer scripts (IPython, scratch scripts).

## Cross-References

- `src/proxy/AGENTS.md` — `proxy.state` mutation pattern + rate limiter pool.
- `src/proxy/tool_pipeline/AGENTS.md` — `ToolCallTracker._instances` for reset.
- `src/memory/AGENTS.md` — `_sync_pool` shutdown via `atexit` / `os._exit(0)`.
- `tests/AGENTS.md` (planned) — the 3-site `_reset_all_shared_state` pattern.
- `src/proxy/canary.py` (module docstring) — `sign_canary` mirrors `canary.py:sign_canary`.