# `tests/` — Agent Operating Context

> Developer-facing guide to writing tests in ChoriProx. Companion to
> [`/AGENTS.md`](../../AGENTS.md) (project-level conventions and landmines) and
> [`docs/TEST_PATTERNS.md`](../../docs/TEST_PATTERNS.md) (US-XXX traceability
> and per-fixture recipes). This file covers the layer conventions, shared
> infrastructure, and the pinned-behavior test pattern.

## 1. The 4 test layers

ChoriProx tests are organized in 4 layers (NOT 5 — the `data_scripts/` reference
in `AGENTS.md` is a stale claim; the directory does not exist).

| Layer | Directory | Mocking scope | When to use | Speed |
|---|---|---|---|---|
| **Unit** | `tests/unit/` | All external deps mocked (LLM, Neo4j, filesystem, HTTP) | One function, one class, one module. Behavioral invariants. | ~ms |
| **Integration** | `tests/integration/` | LLM mocked; real SQLite; mock or real Neo4j | Multi-module interaction, real DB queries, mode transitions, request lifecycle | ~100ms |
| **E2E** | `tests/e2e/` | Mocked LLM (or `mock_sota_client`), real services | Full Flask client → pipeline → upstream; SSE format; auth headers | ~500ms |
| **E2E Real** | `tests/e2e_real/` | Real LLM at `localhost:7777`, real Neo4j | Pre-merge smoke for the SOTA / sync pipeline. Marked `@pytest.mark.e2e_real` | ~30s |

> **Sub-layers that exist as siblings, NOT separate layers:**
> `tests/proxy/`, `tests/skills/`, `tests/memory/`, `tests/integration/clients/`,
> `tests/unit/clients/`, `tests/unit/skills/`, `tests/proxy/pipeline/`. These
> inherit the layer conventions of their parent (`tests/proxy/` uses integration-
> layer fixtures; `tests/unit/clients/` uses unit-layer fixtures).
>
> **Load tests** live in `tests/load/` and are NOT part of the default test
> gate. They exercise `_HTTP_SESSION` under `-n WORKERS > 1` and require
> special handling (see §8 pitfalls).

**Rule of thumb:** Start at unit. Promote to integration when you need real
SQLite or mode state across modules. Promote to e2e when you need the full Flask
request → response cycle. Promote to `e2e_real` only for pre-merge smoke checks
with real LLM and Neo4j.

## 2. Shared infrastructure

The root `tests/conftest.py` (742 lines) hosts the shared fixtures that every
layer inherits. Sibling files provide specialized helpers.

| File | LOC | Purpose |
|---|---|---|
| `tests/conftest.py` | 742 | Root fixture host: 18 autouse fixtures, `client`, `app`, `mock_llama_response`, `AuthenticatedTestClient`, `MockAgent` (inline), empty-test linter, Neo4j bolt default, Neo4j-down skip |
| `tests/builders.py` | 264 | Plain-dict factories: `make_task`, `make_bug`, `make_review`, `make_work_session`, `make_message`, `make_tool_call`, `make_memory`. Each has smoke tests at the bottom |
| `tests/mocks.py` | 218 | `MockAgent` + `MockRequestContext` test doubles. **Test-only guard** at lines 48-54 — raises `ImportError` outside pytest/`PYTEST_CURRENT_TEST`/`CHORI_TEST_MODE=1` |
| `tests/mode_harness.py` | 158 | `create_test_agent`, `activate_ticket_mode`, `activate_work_mode`, `create_test_mode_state` (controllable clock for deterministic expiry) |
| `tests/concurrency_harness.py` | 171 | `run_concurrently` (uses `threading.Barrier` for synchronized start — deterministic, not flaky), `assert_no_deadlock`, `assert_consistent_state` |
| `tests/flask_harness.py` | 231 | `create_test_proxy_app`, `make_chat_request`, `make_tool_request`, `ProxyResponse` dataclass with `.content` / `.error` / `.stream_chunks` |
| `tests/integration/conftest.py` | 348 | `_reset_all_shared_state()` + `MockBackendHandler` + `mock_backend_server` + `proxy_with_mock` + `backup_live_db` (session fixture) |
| `tests/integration/neo4j_harness.py` | 418 | `neo4j_test_db` + `real_or_mock_neo4j` fixtures + the **no-wipe policy** (see §3) |
| `tests/integration/neo4j_harness_utils.py` | 30 | `safe_db_name(test_name)` — must remain non-idempotent; hash RAW input, not sanitized form (see AGENTS.md landmine) |

### The 18 autouse fixtures in `tests/conftest.py`

Every test in the suite runs under these fixtures (in declaration order — see
AGENTS.md "Race conditions in test setup" for why order matters).

| # | Fixture | Scope | Lines | Purpose |
|---|---|---|---|---|
| 1 | `block_all_http` | function | 158-213 | Patches `_HTTP_SESSION.request` + `requests.post` + `requests.get`. Skipped for `@pytest.mark.e2e_real` and `@pytest.mark.integration` |
| 2 | `block_openai_client` | session | 216-248 | Mocks `openai.OpenAI` + `openai.resources.chat.Completions` |
| 3 | `mock_sota_client` | function | 251-285 | Patches `memory.sota_client.SOTAClient`. **Skipped for `test_sota_tool_call_enforcement`** — that test needs the real class to verify `tool_calls` enforcement |
| 4 | `mock_sync_submit` | session | 288-301 | Patches `memory.sync.submit_sync` to prevent background threads |
| 5 | `_ensure_proxy_skills_dir` | session | 304-325 | Pins `PROXY_SKILLS_DIR` to a valid empty tmp path when unset (Wave 2 eager-init guard) |
| 6 | `cleanup_agent_registry` | function | 328-341 | Clears `registry._agents`, `_agents_by_id`, `_lru`, `_children` |
| 7 | `isolate_test_db` | function | 344-357 | Sets `PROXY_TEST_DB` to per-test `tmp_path / "memory.db"` |
| 8 | `reset_task_graph_singleton` | function | 493-505 | Resets `proxy.context._task_graph_singleton` to None |
| 9 | `_set_proxy_env_dev` | session | 508-516 | Sets `PROXY_ENV=dev` so `assert_session_secret_configured` does not raise at import time |
| 10 | `mock_startup_migrations` | session | 519-538 | Patches `memory.sync.init_db` — preserves `MigrationRunner` class for the Neo4j test harness's per-test isolated DBs |
| 11 | `reset_neo4j_singleton` | function | 541-549 | Calls `memory.query_executor.reset_shared_driver()` |
| 12 | `reset_neo4j_circuit_breaker` | function | 552-571 | `state.neo4j_circuit_breaker.force_closed()` + `TaskGraph._constraint_ensured = False` |
| 13 | `_isolate_workspace_neo4j` | function | 574-604 | Neutralizes `workspace_store._persist` + patches `create_persist_callback` → None — prevents test workspace writes from reaching prod Neo4j |
| 14 | `reset_sota_circuit_breaker` | function | 607-623 | `memory.sota_channel._sota_breaker.force_closed()` — prevents cascade SOTA failures across tests |
| 15 | `reset_rate_limiters` | function | 626-642 | Delegates to `proxy.state.reset_all_rate_limiters()` |
| 16 | `reset_tier2_instruction_state` | function | 645-662 | Clears `state.instruction_files_by_workspace` + `state._last_scanned_by_tool_call` (Tier-2 instruction file map reset) |
| 17 | `mock_agent_registry_get_or_load` | function | 665-725 | Patches `proxy.agent.agent_registry.registry.get_or_load` with an in-process MockAgent factory |
| 18 | `reset_dedup_store` | function | 728-742 | Clears the CP4 dedup store between tests (replay isolation) |

> **The `_reset_all_shared_state` pattern (3 sites).** The canonical reset
> lives at `tests/integration/conftest.py:18-166`. It is mirrored in
> `src/devtools/debug_session.py:225+` (for `proxy_with_mock`) and partially
> replicated in `tests/conftest.py:reset_rate_limiters` (autouse). All three
> delegate the rate-limiter reset to `proxy.state.reset_all_rate_limiters()`
> — the single source of truth that auto-discovers every `RateLimiter` and
> calls its public `reset()`.

## 3. The Neo4j no-wipe policy (CRITICAL — the harness NEVER wipes a database)

`tests/integration/neo4j_harness.py` **never** runs `MATCH (n) DETACH DELETE n`
against any shared database. The old `I_AM_AN_IDIOT_ASK_A_HUMAN=yes` wipe gate
has been **REMOVED entirely** — it was a foot-gun (agents set the flag without
thinking and wiped production; hundreds of wipes are recorded in `backups/`).
There is now **no flag and no override**. Protection is enforced in 3 layers:

| # | Layer | Where | Behavior |
|---|---|---|---|
| 1 | NO-WIPE | `_clear_all_data` | Refuses UNCONDITIONALLY — logs and returns. No session, no Cypher, no env var changes this. The destructive path is deleted, not gated. |
| 2 | PRODUCTION GUARD | `neo4j_test_db` / `real_or_mock_neo4j` fixtures + `_is_prod_bolt_url` | If `PROXY_NEO4J_BOLT_URL` targets `:7687`, fixtures skip (or fall back to a mock) — no `CREATE/DROP DATABASE`, no migrations, no wipe on prod. |
| 3 | ISOLATION DEFAULT | `_get_bolt_url` + root `conftest.py` | Tests default to the throwaway container `bolt://localhost:7688`. If it is down, `_neo4j_available()` is False and Neo4j tests skip — they never fall back to prod. |

**Tests REQUIRE Docker.** Neo4j-backed tests isolate into per-test Enterprise
databases on the test container (`docker-compose.test.yml`, `:7688`). Start it
before running DB-touching tests:

```bash
docker compose -f docker-compose.test.yml up -d        # test DB on :7688
# or use the Makefile target, which does this for you:
make test-integration
```

Production lives on `:7687` (`docker-compose.yml`, Enterprise + RBAC
`DENY DELETE ON NODE Task` via `scripts/neo4j-apply-rbac.sh`). **If Docker is
not up, NO-ONE runs DB-touching tests** — the Neo4j tests skip, and that is
CORRECT behaviour, not a failure to "fix".

**Why this MUST NOT be weakened.** Commit `2753fbe` ("remove 9 skip markers")
removed the isolation skips and **caused 3 production data losses in one day**
— the then-Community fallback ran `MATCH (n) DETACH DELETE n` against the
shared `neo4j` database, destroying all workspaces, tasks, sessions, and bugs.

**Pinned regression tests** (`tests/integration/test_neo4j_wipe_gate.py`):
- `test_clear_all_data_refuses_unconditionally` — `_clear_all_data` never opens a session, regardless of target.
- `test_legacy_idiot_flag_is_a_noop` — setting `I_AM_AN_IDIOT_ASK_A_HUMAN=yes` changes nothing; asserts the flag and `_backup_before_wipe` no longer exist and no `DETACH DELETE` runs.
- `test_clear_all_data_refuses_prod` — production URL is refused.
- `test_is_prod_bolt_url` — parametrized contract for the prod-URL discriminator (`:76870` must not match; no-port defaults to prod).

**Skip message:**
> "Neo4j test isolation is unavailable (need the Enterprise test container on
> bolt://localhost:7688 — start it with `docker compose -f
> docker-compose.test.yml up -d`). The test harness REFUSES to wipe any shared
> database and SKIPS instead. There is no override flag."

When you see this skip, the test is **correctly behaving** — fix the
environment (start Docker), never the skip.

### `safe_db_name` invariants (companion to isolation)

`tests/integration/neo4j_harness_utils.py:13-31` defines `safe_db_name(test_name)`
that has caused 3+ production data losses when bugs were introduced (commit
`5dda43a`). The 5 invariants (all must hold):

1. **Determinism** — `safe_db_name(x)` returns the same value every call.
2. **Uniqueness across sibling tests** in the same class.
3. **Uniqueness across inputs that sanitize to the same canonical form** — hash the RAW input (NOT the sanitized form) so `test_aaa` and `test-aaa` stay distinct.
4. **Length ≤ 63 chars** (Neo4j database name limit).
5. **First char letter; remaining chars `[a-zA-Z0-9.-]`** — 8-char MD5 digest of RAW `test_name` + 53-char sanitized prefix = max 62 chars; truncates to 63 if over.

Pinned at `tests/unit/test_safe_db_name.py`. **NOT idempotent** by design:
`safe_db_name(safe_db_name(x)) != safe_db_name(x)` because the digest is from
the raw input.

## 4. The pinned-behavior test pattern

**Every behavior invariant has at least one test that fails if you break it.**
This is the single most important pattern in the test suite. Without it,
"tests pass" does not prove "code works" — see AGENTS.md "Tautological tests"
for the recurring failure class.

### The "revert test" workflow

1. Identify the invariant you want to pin (e.g., "opencode's `filePath` must
   survive normalization").
2. Write a test that asserts the invariant.
3. **Revert the fix** (one-line edit in the production code).
4. **Run the test.** If it still passes, the test is tautological — rewrite
   it to exercise the real path (see AGENTS.md "Tautological tests" §5).
5. Restore the fix. Confirm the test passes.
6. Only commit after step 4 confirms the test fails on revert.

### The 6 canonical pinned-behavior tests

These are the tests you read first when investigating a regression in the
named subsystem.

| Test | Subsystem | Invariant | File |
|---|---|---|---|
| `test_argument_normalization_schema.py` | Tool argument normalization | Schema-driven resolver preserves opencode `filePath`; no case folding without schema | `tests/unit/` |
| `test_argument_normalization.py::TestCamelCasePreserved` | Tool argument normalization | `filePath` survives round-trip | `tests/unit/` |
| `test_lock_ordering.py` | Lock hierarchy | 15-position lock order (#0–#14); violations deadlocked 8+ times | `tests/unit/` |
| `test_pipeline_stages.py` | Pipeline | `DEFAULT_STAGES` order is fixed; singleton sharing pattern | `tests/unit/` |
| `test_tool_registry_single_source.py` | Tool registry | No `DEFAULT_TOOL_NAMES` / `MEMORY_TOOL_NAMES` literals; ToolRegistry is the single source | `tests/unit/` |
| `test_safe_db_name.py` | Neo4j DB naming | The 5 invariants in §3 above | `tests/unit/` |
| `test_neo4j_wipe_gate.py` | Neo4j no-wipe policy | `_clear_all_data` refuses unconditionally; the legacy wipe flag is a no-op; prod (:7687) is never touched (see §3) | `tests/integration/` |
| `test_canary.py::TestCanaryRefactorPinning` | Canary parser | `parse_canary` requires `loc.start == 0`; mid-message canaries strip prefix text | `tests/proxy/` |

### Tautological test guardrails (from AGENTS.md §"Tautological tests")

A test that PASSES does not prove the code is correct if any of these are true:

1. **Tests dead code.** Verify the method has production callers via `grep src/`. Zero callers = delete the method, do not test it.
2. **Tests a simplified path that bypasses the bug.** Use the real call site, not the underlying function in isolation.
3. **Masks the claim via manual reset.** Don't call `_update_batch_status(bid, "processing")` directly to bypass a broken `_set_batch_processing`.
4. **Mocks the function under test.** If you `monkeypatch` the function to return a canned value, you test the mock, not the code.
5. **Concurrency bug tested sequentially.** Use `threading.Barrier` (see `tests/concurrency_harness.py`) — never `time.sleep()`.

**The 5 invariants any new test for a bug fix MUST satisfy:**

1. Exercise the real production call site (grep the test's call stack for the fixed line).
2. Do not mock the function under test.
3. Revert-test: confirm the test FAILS when the fix is reverted.
4. Grep `src/` for callers of the method before testing (zero = dead code).
5. Concurrency bugs need `threading.Barrier` tests.

## 5. Builder pattern + per-file factories

### `tests/builders.py` — canonical dict factories

```python
from tests.builders import make_task, make_bug, make_message, make_tool_call

t = make_task(status="closed", priority="high", workspace_id="ws-alpha")
b = make_bug(workspace_id="ws-alpha", severity="high")
m = make_message(role="assistant", tool_calls=[make_tool_call()])
```

UUID prefixes are canonical: tasks → `entity-`, bugs → `bug-`, reviews →
`review-`, work_sessions → `ws-`, messages have integer `id`, memories →
`mem-`. Each builder returns a `dict` (not a mock) and accepts `**kwargs` for
fields not in the schema.

### Per-file `_make_*` factories

Most test files define a local factory near the top:

```python
def _make_ctx(session_id="sess-1", **overrides):
    return MockRequestContext(session_id=session_id, **overrides)
```

**When to add one:**
- The construction logic is repeated 3+ times in the file.
- The factory hides noise that distracts from the assertion.
- The factory sets up a real fixture (e.g., agent registry) that the test cannot do inline.

**When NOT to add one:**
- The construction is one line and adds noise to read.
- You need different defaults per test.

### `_FakeTx` script-driven Cypher recording

For tests that need to exercise real Cypher paths without a real Neo4j
(community edition), use `StatefulScript` (in `tests/integration/conftest.py`)
or `build_recording_tg` to capture and replay queries:

```python
class _FakeTx:
    """Records (query, params) pairs and returns scripted results."""
    def __init__(self, script: list):
        self._script = script
        self._idx = 0
    def run(self, query, params=None):
        q, p = self._script[self._idx]
        self._idx += 1
        return _RecordIter([{**q, **p}])
```

This exercises the real `TaskGraph` / action handler code path without needing
a live Neo4j. The `script` is a list of `(expected_query_substring, params)`
tuples that the test asserts were called in order.

### `MockBackendHandler` class-level attrs

`tests/integration/conftest.py:284-308` defines `MockBackendHandler`
(`BaseHTTPRequestHandler` subclass) with class-level configuration:

```python
class MockBackendHandler(BaseHTTPRequestHandler):
    _response_body: bytes = b'{"choices": [...]}'
    _status_code: int = 200
    _captured_request_body: bytes | None = None
    _captured_headers: dict = {}
    _request_count: int = 0
```

The `mock_backend_server` fixture resets all class attrs at setup and starts
a daemon-thread `HTTPServer` on `127.0.0.1:0` (kernel-assigned port).
Per-test customizations set these attrs directly before calling the client.

## 6. Mock backend setup (integration / e2e)

### The 3 mock backend handlers

| Class | Location | Use case |
|---|---|---|
| `MockBackendHandler` | `tests/integration/conftest.py:284+` | Default — captures a single request, returns canned response |
| `StreamingBackendHandler` | `tests/integration/test_streaming_pipeline.py:46+` | Sequential SSE responses via `_responses` list; optional `_fail_after: Optional[int]` for failure injection |
| `ToolLoopBackendHandler` | `tests/integration/test_tool_interception.py:24+` | Tool-loop scenarios — emits `tool_calls` finish_reasons, then a final stop |

### The 3 mock-backend fixtures

| Fixture | Returns | Use case |
|---|---|---|
| `mock_backend_server` (function) | `(server, handler_class, port)` | Start a daemon-thread HTTP server, capture requests |
| `proxy_with_mock` (function) | `(client, handler_class, mock_port)` | Wrap `devtools.debug_session.debug_session(...)`. **Single source of truth** for proxy + mock wiring (per Wave 1C refactor) |
| `client` (function) | `AuthenticatedTestClient` | Flask test client with auto-auth header; from root `tests/conftest.py:463-481` |

### Choosing the right fixture

- **`client`** for unit/e2e tests that don't need a real upstream (root conftest).
- **`mock_backend_server`** when you need raw HTTP capture (e.g., to assert headers sent).
- **`proxy_with_mock`** when you need the full Flask app + a configured mock upstream (most integration tests).

## 7. How to add a new test

### Step-by-step

1. **Pick the layer.** Unit first; promote if needed (see §1).
2. **Pick the right `tests/` subdirectory.** Mirror the source layout (e.g., `src/proxy/pipeline/foo.py` → `tests/unit/test_foo.py` or `tests/proxy/pipeline/test_foo.py`).
3. **Pick existing fixtures.** Don't duplicate. The 18 autouse fixtures in `tests/conftest.py` already cover most needs. Add a per-file autouse only if you need a domain-specific reset.
4. **Use `tests/builders.py`** for data shapes (Task, Bug, Message, etc.). Add a new builder to `builders.py` if the schema is not covered.
5. **Use `tests/mocks.py`** for `Agent` / `RequestContext` doubles. Outside pytest, this module raises `ImportError`.
6. **If testing Neo4j**, use `tests/integration/neo4j_harness.py` (`neo4j_test_db` for an isolated per-test DB, `real_or_mock_neo4j` for real-or-mock). These require the Docker test container on `:7688` (`docker compose -f docker-compose.test.yml up -d`) and **never** wipe a shared database — there is no override flag. If isolation is unavailable the test skips; never point it at production `:7687`.
7. **Add at least one assertion.** The empty-test linter in `tests/conftest.py:92-155` flags tests with only `pass`/`...`/docstring bodies. Add `assert` statements that capture the behavior.
8. **Add a `US-XXX` reference** in the docstring or a `# US-NNN` comment (see `docs/TEST_PATTERNS.md` §1).
9. **Run the layer's test command:**
   - `make test-unit` → `PYTHONPATH=src python3 -m pytest tests/unit -n 4 --dist=loadfile -v`
   - `make test-integration` → `tests/integration -n 4 --dist=loadfile -v`
   - `make test-proxy` → `tests/proxy -n 4 --dist=loadfile -v`
   - `make test-e2e` → `tests/e2e -n 4 --dist=loadfile -v`
   - `make test` → test-unit + test-integration + test-proxy + test-root (NOT e2e)
10. **Verify the test is NOT tautological** — revert the production fix and confirm the test fails (see §4).

### Minimal skeleton

```python
"""US-XXX: short description of the contract being tested."""
import pytest

from src.module.under_test import function_under_test


class TestFeatureX:
    def test_happy_path(self):
        result = function_under_test(input_val="x")
        assert result == expected_output

    def test_error_path(self):
        with pytest.raises(ValueError, match="expected message"):
            function_under_test(input_val=None)
```

## 8. Common pitfalls

### Don't mock `memory.sota_client.SOTAClient` directly

`SOTAClient` is bound at import time via `from memory.sota_client import
SOTAClient` in `memory/sync.py`. Patching `memory.sota_client.SOTAClient` does
NOT intercept the already-imported reference in `sync`. Use the `mock_sota_client`
autouse fixture (it patches at `memory.sync.SOTAClient`, the import location),
OR patch at the call site (`memory.sync._get_sota`) if you need a more
specific control.

### Don't patch `_HTTP_SESSION` per-test in load tests

`_HTTP_SESSION` patching is **not safe under `-n WORKERS > 1`**. Multiple
workers will race to apply patches to the shared module-level object, and one
worker's teardown will undo another's patch mid-test. Load tests must use
their own session fixture or run with `-n 1`.

### Don't import `tests.mocks` outside pytest

`tests/mocks.py` raises `ImportError` when imported outside pytest
(`PYTEST_CURRENT_TEST` unset AND `CHORI_TEST_MODE != "1"` AND `pytest` not in
`sys.modules`). This is the test-only guard at lines 48-54. If you see this
error in production, something is wrong — the guard is correct.

### Don't add `data_scripts/` as a test layer

The 5-layer claim (unit / integration / e2e / e2e_real / data_scripts) in some
older docs is **stale**. `tests/data_scripts/` does not exist. There are 4
layers (see §1). If you need data-driven fixtures, use `tests/fixtures/`
(`assert_byte_identical.py`, `golden_responses/`, `integration_fixtures.py`,
`schemas/`, `ticket_mode_responses/`, `unit_fixtures.py`).

### Don't use `time.sleep()` to wait for events

Sleep durations are platform-dependent and flake on slow CI. Use the
concurrency harness:

```python
from tests.concurrency_harness import run_concurrently

results = run_concurrently([
    lambda: write_a(),
    lambda: write_b(),
    lambda: write_c(),
])
assert all(r[1] is None for r in results)  # no exceptions
```

`run_concurrently` uses `threading.Barrier` for deterministic synchronized start.

### Don't use bare `except Exception: pass`

Always handle `InfrastructureError`, `ConstraintViolation`, and
`QueryTimeoutError` explicitly (from `memory.query_executor`). Bare `except`
silently swallows the bug class that the error hierarchy was designed to
surface.

### Don't use `--dist=loadgroup`

The Makefile uses `--dist=loadfile`. `loadgroup` distributes tests by class
boundaries and breaks fixtures that mutate module-level state across classes
(most fixtures do — see the 18 autouse fixtures in §2).

### `tests/enhanced_test_infrastructure.py` is dead code

This 454-line module (`ProbabilisticValidator`, `ContractValidator`,
`TestDataGenerator`, `EnhancedTestValidator`) is **not imported by any test**.
Per AGENTS.md "no dead code", it should be deleted in a follow-up. Don't add
new tests that depend on it.

### Mock the right way

`unittest.mock.MagicMock` is contagious. Prefer:

1. Real function with patched I/O (`patch("requests.post")`).
2. Real function with patched dependency (`patch("memory.graph.Neo4jClient")` — see the `mock_neo4j_client` fixture).
3. Real function with recorded protocol (`StatefulScript` or `build_recording_tg` for Cypher).
4. `MagicMock` only as a last resort for the unit under test.

## See Also

- [`/AGENTS.md`](../../AGENTS.md) — project-level conventions, known landmines, behavioral contracts
- [`docs/TEST_PATTERNS.md`](../../docs/TEST_PATTERNS.md) — US-XXX traceability, builder usage, layer choice, common pitfalls
- [`docs/_analysis/PIPELINE_MAP.md`](../../docs/_analysis/PIPELINE_MAP.md) — 1500-line pipeline walkthrough (use for stage-by-stage test design)
- [`USER-STORIES.md`](../../USER-STORIES.md) — `US-XXX` behavioral contracts