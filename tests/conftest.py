"""Top-level conftest for the entire test suite.

Bootstrap required environment variables BEFORE any imports below this line
that might transitively trigger module-level strict reads.

Specifically, ``src/proxy/app.py`` calls ``assert_session_secret_configured()``
at module load (line 314), which raises ``RuntimeError`` when neither
``PROXY_SESSION_SECRET`` nor ``PROXY_ENV=dev`` is set. Several unit test
files import ``qwen_proxy`` (which imports ``proxy.app``) at collection time
— BEFORE any fixture runs — so a session-scoped autouse fixture is too late.

``PROXY_ENV=dev`` is the safest default because ``assert_session_secret_configured``
already has a documented dev-mode bypass and the existing
``_set_proxy_env_dev`` autouse fixture (further down) sets the same value at
runtime. ``setdefault`` preserves any pre-set production value.
"""
import os

os.environ.setdefault("PROXY_ENV", "dev")

import ast
import logging
import sys
import threading
from unittest.mock import MagicMock, patch

import pytest

logger = logging.getLogger(__name__)

pytest_plugins = ["tests.integration.neo4j_harness"]


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests (skip block_all_http)"
    )
    config.addinivalue_line(
        "markers", "e2e_real: end-to-end tests with real API calls"
    )
    config.addinivalue_line(
        "markers", "empty_test: flagged by the empty-test linter as having only docstring/pass body"
    )
    config.addinivalue_line(
        "markers",
        "real_agent_registry: marker indicating a test should use the real "
        "agent registry (skip the mock_agent_registry_get_or_load fixture)",
    )

    # Configure Neo4j credentials for the ISOLATED test container (matches
    # docker-compose.test.yml: neo4j:testpassword on :7688). The DEFAULT target
    # is the throwaway test container, NEVER the production graph on :7687.
    # The shared driver singleton is reused across module imports — only the
    # FIRST import creates a real connection. The reset_neo4j_singleton and
    # reset_neo4j_circuit_breaker fixtures below reset state between tests
    # without re-opening the connection pool excessively.
    import os
    # Default to the isolated test container, not prod. The Makefile's
    # test-integration target also sets :7688 explicitly; defaulting here
    # means a bare `pytest` still targets the throwaway test DB. Targeting
    # :7687 here would send tests to the production graph — the exact
    # data-loss class this default exists to prevent.
    os.environ.setdefault("PROXY_NEO4J_BOLT_URL", "bolt://localhost:7688")
    os.environ.setdefault("NEO4J_AUTH", "neo4j:testpassword")
    # Stub SOTA key so unit tests constructing the SOTA client are hermetic
    # on fresh clones. Real credentials (if exported) take precedence.
    os.environ.setdefault("PROXY_SOTA_API_KEY", "sk-stub-test-key")


_INFRA_FILES = {
    "tests/conftest.py",
    "tests/builders.py",
    "tests/mode_harness.py",
    "tests/concurrency_harness.py",
    "tests/flask_harness.py",
    "tests/integration/neo4j_harness.py",
}


def _is_nonExecuting(stmt):
    return (
        isinstance(stmt, ast.Pass)
        or (
            isinstance(stmt, ast.Expr)
            and isinstance(stmt.value, ast.Constant)
            and isinstance(stmt.value.value, (str, type(...)))
        )
    )


def _is_empty_body(fn_node):
    return all(_is_nonExecuting(s) for s in fn_node.body)


def _check_empty_tests(collected_items):
    from pathlib import Path

    flagged = []
    seen_paths = set()
    for item in collected_items:
        path = str(Path(str(item.fspath)).resolve())
        if path in seen_paths:
            continue
        seen_paths.add(path)

        try:
            rel = str(Path(str(item.fspath)).resolve().relative_to(Path.cwd()))
        except ValueError:
            continue

        if rel in _INFRA_FILES:
            continue
        if not rel.startswith("tests/unit/"):
            continue
        if not rel.split("/")[-1].startswith("test_"):
            continue

        try:
            with open(path, "r", encoding="utf-8") as f:
                source = f.read()
        except OSError:
            continue

        try:
            tree = ast.parse(source, filename=path)
        except SyntaxError:
            continue

        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            if not node.name.startswith("test"):
                continue
            if not _is_empty_body(node):
                continue
            flagged.append((rel, node.name, node.lineno))

    return flagged


def pytest_collection_modifyitems(config, items):
    flagged = _check_empty_tests(items)
    if not flagged:
        return
    print(
        "\n[empty-test linter] flagged {} empty test function(s) "
        "(body is only docstring/pass/ellipsis):".format(len(flagged)),
        file=sys.stderr,
    )
    for rel, name, lineno in flagged:
        print(
            "  {rel}:{lineno} {name}".format(rel=rel, lineno=lineno, name=name),
            file=sys.stderr,
        )
    print(
        "[empty-test linter] these are quality hints, not test failures.\n",
        file=sys.stderr,
    )


@pytest.fixture(autouse=True)
def block_all_http(request):
    """Block all real HTTP calls by patching _HTTP_SESSION.request.

    Skips for tests marked with 'e2e_real' which need real HTTP connections.
    """
    marker = request.node.get_closest_marker("e2e_real")
    if marker is not None:
        yield
        return
    if request.node.get_closest_marker("integration"):
        yield
        return

    import requests
    from qwen_proxy import _HTTP_SESSION

    default_response = MagicMock()
    default_response.status_code = 200
    default_response.json.return_value = {
        "choices": [
            {
                "message": {"role": "assistant", "content": "mock response"},
                "finish_reason": "stop",
            }
        ]
    }
    default_response.raise_for_status = MagicMock()

    default_get_response = MagicMock()
    default_get_response.status_code = 200
    default_get_response.json.return_value = {}
    default_get_response.raise_for_status = MagicMock()

    # Patch _HTTP_SESSION.request (instance-level) to allow test overrides
    def mock_request_side_effect(method, url, *args, **kwargs):
        if method.upper() == "GET":
            return default_get_response
        return default_response

    session_patcher = patch.object(
        _HTTP_SESSION, "request", side_effect=mock_request_side_effect
    )
    mock_session_request = session_patcher.start()

    # Also patch module-level requests.post/get for code that doesn't use _HTTP_SESSION
    post_patcher = patch("requests.post", return_value=default_response)
    get_patcher = patch("requests.get", return_value=default_get_response)
    post_patcher.start()
    get_patcher.start()

    yield

    session_patcher.stop()
    post_patcher.stop()
    get_patcher.stop()


@pytest.fixture(scope="session", autouse=True)
def block_openai_client():
    """Block all real OpenAI API calls by mocking the OpenAI client.

    This fixture is autouse=True to ensure NO test can ever make a real OpenAI API call.
    Mocks:
    - openai.OpenAI (the main client class)
    - openai.resources.chat.Completions (the chat completions resource)

    The mock returns a safe default response matching OpenAI's format.
    """
    mock_chat_response = MagicMock()
    mock_chat_response.choices = [
        MagicMock(
            message=MagicMock(role="assistant", content="mock response"),
            finish_reason="stop",
        )
    ]
    mock_chat_response.model = "gpt-4"
    mock_chat_response.usage = MagicMock(
        prompt_tokens=10, completion_tokens=5, total_tokens=15
    )

    mock_completions = MagicMock()
    mock_completions.create.return_value = mock_chat_response

    mock_openai_client = MagicMock()
    mock_openai_client.chat = MagicMock()
    mock_openai_client.chat.completions = mock_completions

    with patch("openai.OpenAI", return_value=mock_openai_client):
        with patch("openai.resources.chat.Completions", return_value=mock_completions):
            yield


@pytest.fixture(autouse=True)
def mock_sota_client(request):
    """Mock SOTAClient to prevent real API calls from background sync threads.

    The sync pipeline runs in background threads and imports SOTAClient via:
        from memory.sota_client import SOTAClient
    Since `from ... import ...` creates a direct reference in sync.py,
    we must patch at memory.sync.SOTAClient (the import location).

    NOTE: test_sota_tool_call_enforcement.py needs the REAL SOTAClient class
    to verify that SOTA methods use tool calls instead of response_format.
    This fixture skips mocking for those tests.
    """
    if "test_sota_tool_call_enforcement" in request.node.nodeid:
        yield
        return

    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.choices = [
        MagicMock(
            message=MagicMock(role="assistant", content="mock sota response"),
            finish_reason="stop",
        )
    ]
    mock_response.model = "sota-model"
    mock_response.usage = MagicMock(
        prompt_tokens=10, completion_tokens=5, total_tokens=15
    )
    mock_client.chat = MagicMock()
    mock_client.chat.completions = MagicMock()
    mock_client.chat.completions.create.return_value = mock_response

    with patch("memory.sota_client.SOTAClient", return_value=mock_client):
        yield mock_client


@pytest.fixture(scope="session", autouse=True)
def mock_sync_submit():
    """Mock submit_sync to prevent background threads from spawning.

    The sync pipeline runs in background threads and calls SOTA API.
    During tests, we want to prevent these threads from being created.
    """
    try:
        import memory.sync  # noqa: F401
    except (ImportError, SyntaxError):
        yield
        return
    with patch("memory.sync.submit_sync"):
        yield


@pytest.fixture(scope="session", autouse=True)
def _ensure_proxy_skills_dir():
    """Pin PROXY_SKILLS_DIR to a valid empty tmp path for the test session.

    Without this, qwen_proxy.py:__main__'s eager registry init (Wave 2) sees
    the test runner's PROXY_SKILLS_DIR (often unset → default <repo>/skills
    → missing → healthy disabled). Tests that want a different value use
    monkeypatch.setenv which is function-scoped and takes precedence for the
    duration of the test.

    The fixture sets the env var ONLY when unset, so a CI / developer who
    explicitly exports PROXY_SKILLS_DIR=… keeps that value. We do NOT
    pre-create the default <repo>/skills directory here — `resolve_skills_dir`
    treats a missing default as a healthy empty configuration per spec §5.1.
    """
    import os
    import tempfile
    if "PROXY_SKILLS_DIR" not in os.environ:
        tmp = tempfile.mkdtemp(prefix="test_skills_")
        os.environ["PROXY_SKILLS_DIR"] = tmp
    yield
    # Don't clean up — tests run in subprocesses anyway and the tmp dir is tiny.


@pytest.fixture(autouse=True)
def cleanup_agent_registry():
    """Clear agent registry state between tests."""
    from proxy.agent.agent_registry import registry

    registry._agents.clear()
    registry._agents_by_id.clear()
    registry._lru.clear()
    registry._children.clear()
    yield
    registry._agents.clear()
    registry._agents_by_id.clear()
    registry._lru.clear()
    registry._children.clear()


@pytest.fixture(autouse=True)
def isolate_test_db(tmp_path, monkeypatch):
    """Redirect PROXY_TEST_DB to a per-test tmpdir.

    Prevents future root tests from polluting shared data/memory.db
    between test runs. Mirrors the per-file autouse setup_test_db
    fixtures in tests/integration/test_sync_windowing.py and
    tests/integration/test_sync_pipeline_integration.py. The DB is
    only initialized if a downstream import actually calls
    memory.db.init_db — this fixture just sets the env var.
    """
    monkeypatch.setenv("PROXY_TEST_DB", str(tmp_path / "memory.db"))
    yield


@pytest.fixture
def mock_neo4j_client(monkeypatch):
    """Mock Neo4jClient for unit/integration tests."""
    from unittest.mock import MagicMock

    mock = MagicMock()
    mock.store_memory.return_value = True
    monkeypatch.setattr("memory.graph.Neo4jClient", lambda: mock)
    monkeypatch.setattr("memory.task_graph.Neo4jClient", lambda: mock)
    return mock


_llama_mock_patcher = None


def mock_llama_response_configure(data=None):
    """Configure mock LLM responses. Called by mock_llama_response fixture."""
    global _llama_mock_patcher
    if _llama_mock_patcher is not None:
        _llama_mock_patcher.stop()

    if data is None:
        data = {
            "choices": [
                {
                    "message": {"role": "assistant", "content": "mock response"},
                    "finish_reason": "stop",
                }
            ]
        }

    # Patch the module-level _HTTP_SESSION's request method
    from qwen_proxy import _HTTP_SESSION

    _llama_mock_patcher = patch.object(_HTTP_SESSION, "request")
    mock_request = _llama_mock_patcher.start()
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = data
    mock_response.raise_for_status = MagicMock()
    # A real headers dict (not a MagicMock attr) so call_upstream's streaming
    # branch can sniff the content type. ticket 199ece93 Part A: a client
    # stream=true request now reaches the SSE branch; a MagicMock .headers would
    # sniff as "" and raise UpstreamStreamError. application/json models an
    # upstream that ignored stream=true and returned JSON (the JSON-fallback).
    mock_response.headers = {"Content-Type": "application/json"}
    mock_request.return_value = mock_response


def mock_llama_response_stop():
    """Stop the mock LLM response patcher."""
    global _llama_mock_patcher
    if _llama_mock_patcher is not None:
        _llama_mock_patcher.stop()
        _llama_mock_patcher = None


@pytest.fixture
def mock_llama_response():
    """Fixture to configure mock LLM responses per test.

    Patches the _HTTP_SESSION.request method to return the configured mock data.
    The data should be OpenAI-format with finish_reason set correctly:
    - For tool calls: {"choices": [{"message": {"role": "assistant", "content": "..."}, "finish_reason": "tool_calls"}]}
    - For content only: {"choices": [{"message": {"role": "assistant", "content": "..."}, "finish_reason": "stop"}]}

    Note: block_all_http already patches Session.request globally,
    but this fixture provides more specific control over the response data.
    """
    yield mock_llama_response_configure

    mock_llama_response_stop()


class AuthenticatedTestClient:
    """Wrapper around Flask test client that automatically adds auth headers."""
    
    def __init__(self, client, api_key="test-proxy-key"):
        self._client = client
        self._api_key = api_key
    
    def _add_auth_header(self, kwargs):
        headers = kwargs.get("headers", {})
        if "Authorization" not in headers:
            headers["Authorization"] = f"Bearer {self._api_key}"
            kwargs["headers"] = headers
        return kwargs
    
    def post(self, *args, **kwargs):
        return self._client.post(*args, **self._add_auth_header(kwargs))
    
    def get(self, *args, **kwargs):
        return self._client.get(*args, **self._add_auth_header(kwargs))
    
    def put(self, *args, **kwargs):
        return self._client.put(*args, **self._add_auth_header(kwargs))
    
    def delete(self, *args, **kwargs):
        return self._client.delete(*args, **self._add_auth_header(kwargs))
    
    def patch(self, *args, **kwargs):
        return self._client.patch(*args, **self._add_auth_header(kwargs))


@pytest.fixture
def client(mock_llama_response, monkeypatch):
    """Create Flask test client with mocked LLM responses and automatic auth."""
    monkeypatch.setenv("PROXY_API_KEY", "test-proxy-key")
    mock_llama_response(
        data={
            "choices": [
                {
                    "message": {"role": "assistant", "content": "Mock response"},
                    "finish_reason": "stop",
                }
            ]
        }
    )

    from qwen_proxy import app

    app.config["TESTING"] = True
    return AuthenticatedTestClient(app.test_client())


@pytest.fixture
def app():
    """Create Flask app for testing."""
    from qwen_proxy import app

    app.config["TESTING"] = True
    return app


@pytest.fixture(autouse=True)
def reset_task_graph_singleton():
    """Reset the TaskGraph singleton before each test to prevent test pollution.
    
    The _get_task_graph_singleton() function caches a TaskGraph instance globally.
    When tests run together, the singleton may be created with a real TaskGraph
    before patches are applied, causing subsequent tests to fail.
    """
    import proxy.context as ctx_module
    original_singleton = ctx_module._task_graph_singleton
    ctx_module._task_graph_singleton = None
    yield
    ctx_module._task_graph_singleton = original_singleton


@pytest.fixture(autouse=True, scope="session")
def _set_proxy_env_dev():
    """Set PROXY_ENV=dev so the boot-time assert_session_secret_configured
    in proxy.app allows tests to import without setting PROXY_SESSION_SECRET.
    Phase 2A.
    """
    import os
    os.environ.setdefault("PROXY_ENV", "dev")
    yield


@pytest.fixture(autouse=True, scope="session")
def mock_startup_migrations():
    """Mock startup migrations to prevent Neo4j connection attempts during tests.

    The ``memory.sync.init_db()`` function runs pending migrations against the
    SHARED Neo4j database (``bolt://localhost:7687``).  When integration tests
    are run against a real Neo4j, accidentally calling ``init_db()`` would
    pollute the shared database with test migration state.  Mocking
    ``init_db`` keeps the real ``MigrationRunner`` class available for the
    Neo4j test harness (``tests/integration/neo4j_harness.py``), which creates
    isolated per-test databases and runs migrations against them.

    Previously this fixture patched ``MigrationRunner`` directly, but that
    broke the harness's ability to run migrations on its isolated test DBs.
    """
    from unittest.mock import patch, MagicMock

    mock_init_db = MagicMock(return_value=None)
    with patch('memory.sync.init_db', mock_init_db):
        yield mock_init_db


@pytest.fixture(autouse=True)
def reset_neo4j_singleton():
    """Reset Neo4j driver singleton between tests. Fast no-op if not initialized."""
    yield
    try:
        from memory.query_executor import reset_shared_driver
        reset_shared_driver()
    except ImportError:
        pass


@pytest.fixture(autouse=True)
def reset_neo4j_circuit_breaker():
    """Reset Neo4j circuit breaker state between tests.
    
    Prevents cascade failures when tests run together. If an earlier test
    triggers Neo4j connection failures, the circuit breaker opens and
    subsequent tests fail with "Circuit breaker is open" errors.
    """
    import proxy.state as state
    from memory.task_graph import TaskGraph
    
    # Reset before test
    state.neo4j_circuit_breaker.force_closed()
    TaskGraph._constraint_ensured = False
    
    yield
    
    # Reset after test
    state.neo4j_circuit_breaker.force_closed()
    TaskGraph._constraint_ensured = False


@pytest.fixture(autouse=True)
def _isolate_workspace_neo4j():
    """Structural guard against the 2753fbe-class workspace-Neo4j leak.

    At import time state.py:444 binds workspace_store._persist to the REAL
    _persist_workspace callback whenever the Neo4j driver is available (as it
    is in dev). Any test calling state.workspace_store.activate(...) then
    MERGEs a test :Workspace node into the shared production database — the
    exact vector that polluted prod with 230 test workspace nodes.

    This makes workspace-state writes purely in-memory for EVERY test:
      1. workspace_store._persist = None  → activate()/set_active() skip the
         persist block entirely (persist_payload stays None at workspace_state.py:81).
      2. patch create_persist_callback → None → belt-and-suspenders for the
         lazy-rewire path (workspace_state.py:94-96) and any store re-init.

    test_workspace_persistence.py exercises _persist_workspace directly with a
    mocked executor (not via the global store) — unaffected. neo4j_test_db
    tests write via TaskGraph / the harness on isolated DBs; the global
    _persist callback always targets the SHARED get_shared_executor_from_env()
    driver, so neutralizing it is correct there too.
    Guard test: tests/unit/test_workspace_neo4j_isolation.py.
    """
    import proxy.state as state
    original_persist = state.workspace_store._persist
    state.workspace_store._persist = None
    with patch(
        "memory.workspace_persistence.create_persist_callback", return_value=None
    ):
        yield
    state.workspace_store._persist = original_persist


@pytest.fixture(autouse=True)
def reset_sota_circuit_breaker():
    """Reset SOTA circuit breaker state between tests.

    Mirrors reset_neo4j_circuit_breaker. _sota_breaker (sota_channel.py:183,
    failure_threshold=5, recovery_timeout=120s) is a module-level singleton; a
    prior test that trips it OPEN leaves it open for the rest of the run,
    polluting every later test on the SOTA/summary path (summarization,
    ancient-compress, sync-pool).
    """
    from memory.sota_channel import _sota_breaker

    _sota_breaker.force_closed()

    yield

    _sota_breaker.force_closed()


@pytest.fixture(autouse=True)
def reset_rate_limiters():
    """Reset all rate limiter state between tests to prevent cross-test pollution.

    Delegates to proxy.state.reset_all_rate_limiters() — the single source
    of truth that auto-discovers every RateLimiter instance, asserts
    non-empty _name, and calls the public reset() (clears _entries,
    _call_count, _last_cleanup, _last_save, holds self._lock for thread
    safety). Mirrors tests/integration/conftest.py:_reset_all_shared_state
    and src/devtools/debug_session.py:_reset_all_shared_state.
    """
    try:
        import proxy.state as state
        state.reset_all_rate_limiters()
    except (ImportError, AttributeError) as e:
        logger.debug("reset partial (rate limiters): %s", e)
    yield


@pytest.fixture(autouse=True)
def reset_tier2_instruction_state():
    """Reset Tier-2 instruction file map between tests (spec §6.1 reset mandate).

    Without this, unit tests leak pinned sub-instructions across cases via the
    module-level instruction_files_by_workspace / _last_scanned_by_tool_call.
    """
    import proxy.state as state

    with state.vfs_lock:
        state.instruction_files_by_workspace.clear()
        state._last_scanned_by_tool_call.clear()

    yield

    with state.vfs_lock:
        state.instruction_files_by_workspace.clear()
        state._last_scanned_by_tool_call.clear()


@pytest.fixture(autouse=True)
def mock_agent_registry_get_or_load(request):
    """Mock agent_registry.get_or_load unless real_agent_registry is marked.

    The production Agent class gained snapshot_mode(), snapshot_mode_and_ticket_ts(),
    get_mode_name(), is_mode_expired(), and _deactivation_callbacks in Week 1.
    Tests that call agent_registry.get_or_load() need these methods available.
    """
    if "real_agent_registry" in request.keywords:
        yield None
        return

    class MockAgent:
        def __init__(self, session_id="test_session"):
            self.active_mode = None
            self._ticket_mode_ts = None
            self._deactivation_callbacks = []
            self._mode_lock = threading.RLock()
            self.session_id = session_id
            self.agent_id = f"agent_{session_id}"
            self.context_limit_override = None

        def set_context_limit_override(self, tokens):
            self.context_limit_override = tokens

        def get_context_limit_override(self):
            return self.context_limit_override

        def set_active_mode(self, mode_state, expected_mode=None):
            self.active_mode = mode_state
            return True
        
        def clear_active_mode(self, expected_mode=None):
            self.active_mode = None
            self._ticket_mode_ts = None
            return True
        
        def register_deactivation_callback(self, cb):
            self._deactivation_callbacks.append(cb)
        
        def snapshot_mode(self):
            return self.active_mode
        
        def snapshot_mode_and_ticket_ts(self):
            return self.active_mode, self._ticket_mode_ts
        
        def get_mode_name(self):
            return self.active_mode.name if self.active_mode else None
        
        def is_mode_expired(self):
            return False
    
    agents = {}

    def get_or_load_side_effect(session_id):
        if session_id not in agents:
            agents[session_id] = MockAgent(session_id)
        return agents[session_id]

    with patch('proxy.agent.agent_registry.registry.get_or_load', side_effect=get_or_load_side_effect):
        yield agents


@pytest.fixture(autouse=True)
def reset_dedup_store():
    """Clear the dedup store between tests (CP4-3 replay isolation).

    Without this, a COMPLETED entry from one test can be replayed in a
    subsequent test that expects to go through the full pipeline. Mirrors
    the per-file fixture in tests/unit/test_dedup.py and the reset in
    tests/integration/conftest.py:_reset_all_shared_state.
    """
    yield
    try:
        import proxy.state as state
        state.dedup_store.clear()
    except (ImportError, AttributeError):
        pass
