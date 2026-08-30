"""Integration test fixtures for ChoriProx model registration."""

import json
import logging
import os
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler
from unittest.mock import patch, MagicMock

import pytest
import requests
import shutil
import tempfile

logger = logging.getLogger(__name__)


def _reset_all_shared_state():
    """Reset all shared in-memory state to prevent cross-test contamination."""
    # Rate limiters — delegate to proxy.state.reset_all_rate_limiters()
    # (the single-source-of-truth helper). Auto-discovers every RateLimiter
    # instance in proxy.state, asserts non-empty _name, and calls the public
    # reset() (clears _entries, _call_count, _last_cleanup, _last_save,
    # holds self._lock for thread safety). Mirrors
    # src/devtools/debug_session.py:_reset_all_shared_state.
    try:
        import proxy.state as state
        state.reset_all_rate_limiters()
    except (ImportError, AttributeError) as e:
        logger.debug("reset partial (rate limiters): %s", e)
    
    # Chat histories
    try:
        from proxy.chat_history import chat_histories, chat_histories_lock, _chat_histories_lru
        with chat_histories_lock:
            chat_histories.clear()
            _chat_histories_lru.clear()
    except (ImportError, AttributeError) as e:
        logger.debug("reset partial (chat histories): %s", e)

    # Chat summaries
    try:
        from proxy.state import chat_summaries, chat_summaries_lock
        with chat_summaries_lock:
            chat_summaries.clear()
    except (ImportError, AttributeError) as e:
        logger.debug("reset partial (chat summaries): %s", e)

    # Workspace state
    try:
        from proxy.state import workspace_store
        with workspace_store._lock:
            workspace_store._active.clear()
            workspace_store._recent.clear()
            workspace_store._last_access.clear()
    except (ImportError, AttributeError) as e:
        logger.debug("reset partial (workspace): %s", e)

    # Tool cache
    try:
        from proxy.state import tool_cache, tool_cache_lock
        with tool_cache_lock:
            tool_cache.clear()
    except (ImportError, AttributeError) as e:
        logger.debug("reset partial (tool cache): %s", e)

    # Model config cache
    try:
        from proxy.state import model_config_cache
        model_config_cache.clear()
    except (ImportError, AttributeError) as e:
        logger.debug("reset partial (model config cache): %s", e)

    # Dedup store (CP4-2)
    try:
        import proxy.state as state
        state.dedup_store.clear()
    except (ImportError, AttributeError) as e:
        logger.debug("reset partial (dedup store): %s", e)

    # Agent registry
    try:
        from proxy.agent.agent_registry import registry
        with registry._lock:
            registry._agents.clear()
            registry._agents_by_id.clear()
            registry._lru.clear()
            registry._children.clear()
            registry._in_flight_requests.clear()
            registry._gol_query_log.clear()
            registry._gol_call_count = 0
    except (ImportError, AttributeError) as e:
        logger.debug("reset partial (agent registry): %s", e)

    # Counters (must use module reference for integers)
    try:
        import proxy.state as state
        state.request_counter = 0
        state.tool_cache_size = 0
        state.tool_cache_hits = 0
        state.tool_cache_misses = 0
        state.param_modifications = 0
        state.param_drops = 0
        state.param_clamps = 0
        state.param_injects = 0
        state.param_mod_details.clear()
        state.param_mod_log_timestamps.clear()
        state._last_cleanup_time = 0.0
    except (ImportError, AttributeError) as e:
        logger.debug("reset partial (counters): %s", e)

    # Neo4j circuit breaker
    try:
        from proxy.state import neo4j_circuit_breaker
        neo4j_circuit_breaker._failure_count = 0
        neo4j_circuit_breaker._state = "closed"
        neo4j_circuit_breaker._last_failure_time = None
    except (ImportError, AttributeError) as e:
        logger.debug("reset partial (neo4j circuit breaker): %s", e)

    # ToolCallTracker instances
    try:
        from proxy.tool_pipeline.tool_call_tracker import ToolCallTracker
        with ToolCallTracker._instances_lock:
            ToolCallTracker._instances.clear()
    except (ImportError, AttributeError):
        pass

    # Canary verification cache
    try:
        import proxy.state as state
        with state.canary_verify_cache_lock:
            state.canary_verify_cache.clear()
    except (ImportError, AttributeError):
        pass

    # Chain hash state
    try:
        import proxy.state as state
        with state._chain_state_lock:
            state._chain_state.clear()
    except (ImportError, AttributeError):
        pass

    # VFS caches
    try:
        import proxy.state as state
        with state.vfs_lock:
            state.vfs_caches.clear()
            state.vfs_write_log.clear()
            state.vfs_rollback_snapshots.clear()
            state.instruction_files_by_workspace.clear()
            state._last_scanned_by_tool_call.clear()
    except (ImportError, AttributeError):
        pass

    # Skill registry singleton — Wave 2 owns the singleton construction
    # eagerly at CLI boot. Integration tests that touch the registry must
    # see a fresh singleton between cases so the next get_default_registry()
    # call repopulates from the (test-set) PROXY_SKILLS_DIR.
    try:
        from proxy.skills.skill_registry import reset_default_registry
        reset_default_registry()
    except (ImportError, AttributeError) as e:
        logger.debug("reset partial (skill registry): %s", e)


@pytest.fixture(autouse=True)
def reset_shared_state():
    """Reset all shared in-memory state before and after each integration test."""
    _reset_all_shared_state()
    yield
    _reset_all_shared_state()


@pytest.fixture(autouse=True)
def reset_neo4j_singleton():
    """Reset Neo4j driver singleton between tests. Fast no-op if not initialized."""
    yield
    try:
        from memory.query_executor import reset_shared_driver
        reset_shared_driver()
    except ImportError:
        pass


def _check_neo4j_available():
    """Check if Neo4j is available without closing shared driver."""
    try:
        from memory.query_executor import QueryExecutor
        bolt_url = os.environ.get("PROXY_NEO4J_BOLT_URL", "bolt://localhost:7687")
        auth_str = os.environ.get("NEO4J_AUTH", "neo4j:password")
        auth = tuple(auth_str.split(":", 1)) if ":" in auth_str else None
        executor = QueryExecutor(bolt_url, auth)
        return executor.is_available()
    except Exception:
        return False


@pytest.fixture(autouse=True)
def mock_neo4j_for_integration(monkeypatch):
    """Mock Neo4jClient for integration tests to prevent real database connections."""
    mock = MagicMock()
    mock.store_memory.return_value = True
    monkeypatch.setattr("memory.graph.Neo4jClient", lambda: mock)
    monkeypatch.setattr("memory.task_graph.Neo4jClient", lambda: mock)
    return mock


@pytest.fixture(scope="session", autouse=True)
def setup_fixtures_dir():
    """Create and clean a session-scoped temp directory for test fixtures.

    Directory: /tmp/qwen_adapter_fixtures/
    """
    fixtures_dir = tempfile.mkdtemp(prefix="qwen_adapter_fixtures_")
    yield fixtures_dir
    shutil.rmtree(fixtures_dir, ignore_errors=True)


@pytest.fixture(scope="session", autouse=True)
def backup_live_db(request):
    """Back up data/memory.db at session start, restore at session end.

    Tests that touch the live DB (init_db, store_messages_batch, workspace_store,
    sync, etc.) modify the user's real data. This fixture preserves it.

    Uses file copy of all WAL-mode files (main + -shm + -wal) after a WAL
    checkpoint. The SQLite online backup API does not work reliably in this
    fixture's flow because the streaming protocol's transaction semantics leave
    the backup file incomplete.
    """
    test_db = os.environ.get("PROXY_TEST_DB")
    if test_db:
        yield
        return

    db_path = "data/memory.db"
    if not os.path.exists(db_path):
        yield
        return

    import sqlite3
    backup_dir = "data/.test_backup"
    os.makedirs(backup_dir, exist_ok=True)

    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    conn.close()

    backup_paths = {
        "": os.path.join(backup_dir, "memory.db.snapshot"),
        "-shm": os.path.join(backup_dir, "memory.db.snapshot-shm"),
        "-wal": os.path.join(backup_dir, "memory.db.snapshot-wal"),
    }
    for suffix, backup_path in backup_paths.items():
        try:
            shutil.copy(db_path + suffix, backup_path)
        except FileNotFoundError:
            pass

    def restore():
        for suffix, backup_path in backup_paths.items():
            try:
                shutil.copy(backup_path, db_path + suffix)
            except (FileNotFoundError, shutil.SameFileError):
                pass

    try:
        yield
    finally:
        try:
            restore()
        except Exception as e:
            logger.error("backup_live_db: FAILED to restore: %s. Live DB may be inconsistent.", e)

        for suffix, backup_path in backup_paths.items():
            try:
                os.remove(backup_path)
            except OSError:
                pass


class MockBackendHandler(BaseHTTPRequestHandler):
    """Configurable mock backend for integration tests."""

    _response_body = (
        b'{"choices":[{"message":{"role":"assistant","content":"default"}}]}'
    )
    _status_code = 200
    _captured_request_body = None
    _captured_headers = None
    _request_count = 0

    def do_POST(self):
        content_length = int(self.headers.get("Content-Length", 0))
        MockBackendHandler._captured_request_body = self.rfile.read(content_length)
        MockBackendHandler._captured_headers = dict(self.headers)
        MockBackendHandler._request_count += 1

        self.send_response(MockBackendHandler._status_code)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(MockBackendHandler._response_body)

    def log_message(self, format, *args):
        pass


@pytest.fixture
def mock_backend_server():
    """Start a mock HTTP server in a daemon thread.

    Returns (server, handler_class, port).
    Configure via handler_class._response_body, _status_code, etc.
    Reset state in each test.
    """
    server = HTTPServer(("127.0.0.1", 0), MockBackendHandler)
    port = server.server_address[1]

    MockBackendHandler._response_body = (
        b'{"choices":[{"message":{"role":"assistant","content":"default"}}]}'
    )
    MockBackendHandler._status_code = 200
    MockBackendHandler._captured_request_body = None
    MockBackendHandler._captured_headers = None
    MockBackendHandler._request_count = 0

    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    yield server, MockBackendHandler, port

    server.shutdown()


@pytest.fixture
def proxy_with_mock(tmp_path):
    """Wire proxy to use mock backend server with test model config.

    Returns (client, handler, port). Thin wrapper around debug_session()
    so there is exactly ONE implementation, not two — the canonical
    harness lives at `src/devtools/debug_session.py`.
    """
    from devtools.debug_session import debug_session

    with debug_session(db_path=str(tmp_path / "memory.db")) as c:
        yield c.client, c._handler_class, c.mock_port
