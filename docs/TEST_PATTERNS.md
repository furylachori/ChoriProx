# Test Patterns

> Companion to [`tests/AGENTS.md`](../../tests/AGENTS.md) (test-layer guide). This document covers the patterns for writing robust tests: how to thread user stories into tests, when to use the builder factory, how to choose a test layer, how to disable flaky tests, and the common pitfalls the linter can't catch.

## 1. US-XXX Traceability

Every behavioral contract in `USER-STORIES.md` is a `US-XXX` identifier. A test exercises a story if the test (or its enclosing class/module) carries a `US-XXX` reference somewhere in its docstring or in a comment that starts with `# US-`.

### How to add a reference

Pick the smallest scope that still makes sense:

```python
# Module-level: covers every test in the file
"""US-005: Unsigned session token when PROXY_SESSION_SECRET is not set."""
import pytest


def test_create_session_token_returns_empty_string():
    """US-005: must return ""."""
    ...


class TestUS005UnsignedSessionToken:
    """US-005: full contract — token empty AND warning logged."""

    def test_token_is_empty(self): ...
    def test_warning_is_logged(self, caplog): ...
```

### Detection rules

The inventory tooling looks for `US-NNN` in:

1. The module docstring (covers all tests in the file)
2. The class docstring (covers all tests in the class)
3. The test function docstring (covers that test only)
4. A comment line that begins with `# US-` (covers the next test)

If you write `US-5` instead of `US-005`, pad to three digits. Lowercase or extra characters are tolerated.

### When to skip the reference

- Pure infrastructure tests (harnesses, builders, conftest plumbing) — not a behavioral contract.
- Parametrized matrix tests where the parameter name is the contract (e.g., `test_handles_each_block_pattern[imperative_override]`) — link the parametrizing module/class instead.

## 2. The Builder Pattern

`tests/builders.py` exposes plain-dict factories that match the production schema (Neo4j node properties or SQLite row columns). Each builder returns a `dict`, never a mock object.

### When to use a builder

Use a builder when:

- You need a `Task`, `Bug`, `Review`, `WorkSession`, `Message`, `ToolCall`, or `Memory` shaped like a real production record.
- You want stable defaults so the test focuses on the field under test.
- You want to override only the fields the test cares about (`make_task(status="closed", priority="high")`).
- You want extra fields the production schema does not have — pass `**kwargs` and they are merged into the dict.

```python
from tests.builders import make_task, make_bug

def test_bug_creation_attaches_to_workspace():
    t = make_task(workspace_id="ws-alpha")
    b = make_bug(workspace_id="ws-alpha")
    assert b["workspace_id"] == t["workspace_id"]
```

### When to use a raw dict

Use a raw dict when:

- The test is asserting on the *shape* of the dict (a parsing or validation test). Builders would hide the shape.
- The data is not a known entity — ad-hoc fixtures, mock HTTP responses, expected error payloads.
- You need to deliberately break the schema (omit required fields, supply wrong types) to test validation.

```python
def test_rejects_task_without_uuid():
    bad = {"name": "no uuid", "status": "open"}  # missing 'uuid'
    with pytest.raises(ValidationError):
        TaskNode.from_dict(bad)
```

### Don't mock the builder

If a test wants a `MagicMock` task, do not call `make_task()` and then patch the result. Build the mock directly. Mixing mock and real data confuses failure traces and produces misleading errors.

## 3. Adding a New Test File

### Directory choice

| Layer | Directory | Mocking | When to use |
|-------|-----------|---------|-------------|
| Unit | `tests/unit/` | All external deps mocked (LLM, Neo4j, filesystem) | Single function, single class, single module. Fast (~ms). |
| Integration | `tests/integration/` | LLM mocked, real SQLite, real or mock Neo4j | Multi-module interaction, real DB queries, mode transitions, request lifecycle. |
| E2E | `tests/e2e/` | Mocked LLM, real services | Full Flask client → pipeline → upstream. Use `flask_harness.create_test_proxy_app()`. |
| E2E Real | `tests/e2e_real/` | Real LLM, real Neo4j | Pre-merge smoke for sync pipeline. Marked `@pytest.mark.e2e_real`. |

If unsure, start with unit. Promoting a test to integration is easier than demoting one.

### E2E Real skips

The 8 tests in `tests/e2e_real/` **skip automatically** when no `llama-server` is reachable at `http://localhost:7777`. The skip fixture is at `tests/e2e_real/conftest.py:98-102` (`skip_if_no_llama`).

This is **environmental**, not a code defect. The default test suite (~3953 tests) is fully self-contained — it does not require `llama-server` or any other external service.

To enable the 8 real-LLM tests locally:
1. Start `llama-server` on port 7777
2. Start ChoriProx on port 6666 (`./start.sh 6666`)
3. Run `pytest tests/e2e_real/ -v`

These tests are **not** in the default CI gate. They exist as a pre-merge smoke check for environments with the LLM available. If you add new e2e_real tests, use `@pytest.mark.e2e_real` and rely on the `skip_if_no_llama` autouse fixture.

### File naming

- `test_<module>.py` — one file per source module. E.g., `src/proxy/rate_limiter.py` → `tests/unit/test_rate_limiter.py`.
- `test_<feature>_unit.py` / `test_<feature>_integration.py` — when a feature spans modules. E.g., `tests/integration/test_mode_switching_persistence.py`.
- `test_<adversarial_issue>.py` — for regression tests against a specific bug. E.g., `tests/unit/test_c9_fail_open_intent.py`.

### Conftest fixtures

Most fixtures you need are autouse at the top level — `block_all_http`, `mock_openai_client`, `mock_sota_client`, `mock_sync_submit`, `reset_neo4j_singleton`, `reset_neo4j_circuit_breaker`, `reset_task_graph_singleton`. You rarely need a per-file conftest.

Add a per-directory conftest only when:

- The fixtures are expensive and you want a smaller scope.
- The fixtures depend on module state that a test must re-load (`reload_module`).
- The fixtures are domain-specific (e.g., `_ensure_registry_populated` for action-registry tests).

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

## 4. Disabling a Flaky Test

Three markers, three different intents.

### `@pytest.mark.skip` — "I know this test should not run."

```python
@pytest.mark.skip(reason="T2b: blocked by session-scoped mock fixture; covered in integration suite")
def test_legacy_path(): ...
```

Use when:

- The test exercises code that no longer exists.
- The test depends on infrastructure that cannot be mocked in this layer.
- A redirect is in place: "covered in `tests/integration/test_*.py::test_xxx`".

### `@pytest.mark.xfail` — "This is a known failure I expect to fix."

```python
@pytest.mark.xfail(reason="Ticket #NNN: bug in upstream serializer", strict=True)
def test_known_broken(): ...
```

Use when:

- The test reveals a real bug that has a tracking ticket.
- You want the test to *fail loudly* if the bug is fixed (with `strict=True`).
- You want the test to pass silently if the bug is fixed (without `strict=True`).

`xfail` shows up in the summary as `xpassed` or `xfailed` — both are acceptable signals.

### `@pytest.mark.skipif` — "Run only when condition is met."

```python
import os

@pytest.mark.skipif(not os.environ.get("NEO4J_AUTH"), reason="Neo4j not available")
def test_neo4j_live_query(): ...
```

Use when:

- The test requires a specific environment variable, service, or Python version.
- The test is optional and should not block CI.

For Neo4j-dependent tests, prefer the `neo4j_test_db` fixture from `tests/integration/neo4j_harness.py` — it handles the env check and fixture teardown for you.

### Document the why

Every disabled test must have a `reason=`. A bare `@pytest.mark.skip()` is a maintenance trap — nobody will know when it is safe to re-enable.

## 5. Common Pitfalls

### Assertion-dead tests

A test with zero `assert` statements passes trivially. The linter in `tests/conftest.py` flags tests that are *literally empty* (only docstring/pass/ellipsis). It cannot detect tests that execute code but never assert. Always end a test with at least one `assert` that captures the behavior under test.

### Heavy mocking

`unittest.mock.MagicMock` is contagious. If you mock the module that owns the function you want to test, the test becomes a test of your mock configuration, not of the code. Prefer:

1. Real function with patched I/O (`patch("requests.post")` not `patch("module.under_test")`).
2. Real function with patched dependency (`patch("memory.graph.Neo4jClient")` — see `mock_neo4j_client` fixture).
3. Real function with recorded protocol (`StatefulScript` or `build_recording_tg` for Cypher).
4. MagicMock only as a last resort for the unit under test.

### `time.sleep()` band-aids

Never use `time.sleep()` to wait for an event in a test. Sleep durations are platform-dependent and the test will flake on slow CI. Use the concurrency harness:

```python
from tests.concurrency_harness import run_concurrently, assert_no_deadlock


def test_concurrent_writes():
    results = run_concurrently([
        lambda: write_a(),
        lambda: write_b(),
        lambda: write_c(),
    ])
    assert all(r[1] is None for r in results)  # no exceptions
```

`run_concurrently` uses `threading.Barrier` so all threads start at exactly the same instant — deterministic, no sleep tuning.

### Stateful scripts vs. mock objects

For Cypher/query integration tests, use `StatefulScript` (in `tests/integration/conftest.py`) or `build_recording_tg` to capture and replay queries. These run the real action handlers against a recorded query log, so you exercise the real code path without needing a live Neo4j.

### Singleton leakage

The `reset_task_graph_singleton`, `reset_neo4j_singleton`, and `reset_neo4j_circuit_breaker` autouse fixtures exist because several modules cache singletons at import time. If your test imports a module that touches Neo4j before a patch is applied, the singleton survives. The autouse fixtures reset between tests, but if you import a module inside the test, reload it explicitly (`importlib.reload`).

### Race conditions in test setup

`@pytest.fixture(autouse=True)` runs in declaration order. Fixtures that read `proxy.state` must be declared *after* fixtures that mutate it. See the lock ordering comments in `src/proxy/state.py`.

### Streaming tests

SSE streams emit a leading comment chunk (`: Post-processing...`) before any data. Tests that parse the first chunk as data will fail. See `tests/integration/test_sse_streaming.py` for the canonical pattern.

## See Also

- `tests/conftest.py` — autouse fixtures, HTTP/Neo4j/SOTA mocking, empty-test linter.
- `tests/builders.py` — test data factories.
- `tests/mode_harness.py` — agent and mode test helpers.
- `tests/concurrency_harness.py` — `run_concurrently`, `assert_no_deadlock`, `assert_consistent_state`.
- `tests/flask_harness.py` — `create_test_proxy_app()`, `make_chat_request()`, `make_tool_request()`.
- `tests/integration/neo4j_harness.py` — opt-in Neo4j test DB.
- `USER-STORIES.md` — `US-XXX` behavioral contracts.
