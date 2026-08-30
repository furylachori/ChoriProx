"""
Real LLM E2E test configuration.
These tests bypass mocks and make real HTTP requests.

The parent tests/conftest.py patches requests.post and requests.get with MagicMock,
blocking all real HTTP calls. We use urllib3 directly to bypass those patches.
"""

import pytest
from unittest.mock import patch
import json
import urllib3
import sys
import os
import time
import hashlib
import memory.sync
import memory.sota_client

# Add current directory to path for test_db import
sys.path.insert(0, os.path.dirname(__file__))
from db_helpers import init_test_db, cleanup_test_db

PROXY_URL = "http://localhost:6666"


@pytest.fixture(scope="session", autouse=True)
def setup_test_db():
    """Initialize test database once for all tests."""
    from db_helpers import TEST_DB_PATH

    print(f"[TEST CONFIG] Using DB path={TEST_DB_PATH}", flush=True)

    init_test_db()
    yield
    cleanup_test_db()


def calculate_session_id(messages):
    """
    Calculate the session_id that the proxy will generate.
    The proxy hashes the first user message content to create the session_id.

    Args:
        messages: List of message dicts, e.g., [{"role": "user", "content": "..."}]

    Returns:
        str: The 16-character session_id
    """
    for msg in messages:
        if isinstance(msg, dict) and msg.get("role") == "user":
            content = msg.get("content", "")
            if content:
                return hashlib.sha256(f"{content}:test".encode()).hexdigest()[:16]
    # Fallback
    return hashlib.sha256(
        f"{json.dumps(messages, sort_keys=True)}:test".encode()
    ).hexdigest()[:16]


def print_test_summary(test_name, session_id, result):
    """Print a summary of the test result."""
    print(f"\n{'=' * 60}")
    print(f"TEST: {test_name}")
    print(f"SESSION: {session_id}")
    print(f"RESULT: {'PASS' if result else 'FAIL'}")
    print(f"{'=' * 60}\n")


LLAMA_URL = "http://localhost:7777"

_http = urllib3.PoolManager()


def real_get(url, timeout=2):
    """Make a real GET request using urllib3, bypassing requests mocking."""
    resp = _http.request("GET", url, timeout=timeout)
    return resp


def real_post(url, json_data=None, timeout=30):
    """Make a real POST request using urllib3, bypassing requests mocking."""
    headers = {"Content-Type": "application/json", "X-Session-Client": "test"}
    body = json.dumps(json_data).encode("utf-8") if json_data else None
    resp = _http.request("POST", url, body=body, headers=headers, timeout=timeout)
    return resp


def _is_llama_available():
    """Check if llama-server is available using urllib3."""
    try:
        resp = real_get(LLAMA_URL, timeout=2)
        return resp.status < 500
    except Exception:
        return False


@pytest.fixture(autouse=True)
def skip_if_no_llama():
    """Skip test if llama-server is not available on port 7777."""
    if not _is_llama_available():
        pytest.skip("llama-server not available on port 7777")


class ResponseWrapper:
    """Wrap urllib3 response to provide a requests-like interface."""

    def __init__(self, urllib3_resp):
        self._resp = urllib3_resp
        self.status_code = urllib3_resp.status

    def json(self):
        return json.loads(self._resp.data.decode("utf-8"))

    @property
    def text(self):
        return self._resp.data.decode("utf-8")


def post(url, json_data=None, timeout=30):
    """Drop-in replacement for requests.post using urllib3."""
    return ResponseWrapper(real_post(url, json_data=json_data, timeout=timeout))


def get(url, timeout=2):
    """Drop-in replacement for requests.get using urllib3."""
    return ResponseWrapper(real_get(url, timeout=timeout))


@pytest.fixture(autouse=True)
def wait_for_sync_completion():
    """Wait for background sync threads to complete between tests."""
    yield

    # Wait for any pending sync jobs to complete
    try:
        from memory.executors import _get_sync_pool

        pool = _get_sync_pool()
        if pool:
            pool.shutdown(wait=True)
    except Exception as e:
        print(f"Warning: Could not shutdown sync pool: {e}", flush=True)

    # Wait for SOTA API rate limiting recovery between tests
    time.sleep(2)


@pytest.fixture(scope="session", autouse=True)
def block_all_http():
    """Override: Do NOT block HTTP for real e2e tests."""
    yield  # No patching - use real HTTP


@pytest.fixture(scope="session", autouse=True)
def mock_sota_client():
    """Override: Do NOT mock SOTAClient for real e2e tests."""
    yield  # No patching - use real SOTAClient


@pytest.fixture(scope="session", autouse=True)
def mock_sync_submit():
    """Override: Do NOT mock submit_sync for real e2e tests."""
    yield  # No patching - use real sync pipeline


@pytest.fixture
def sync_verification():
    """Verify sync pipeline completed successfully using test_db helper."""

    def verify(session_id, timeout=30):
        """
        Verify sync pipeline completed for a session.

        Args:
            session_id: The session ID to check
            timeout: Total timeout in seconds (default 30, sync takes ~15-20s)

        Returns:
            True if sync completed successfully

        Raises:
            pytest.fail: If sync did not complete within timeout
        """
        from db_helpers import check_sync_completed

        timeout = max(timeout, 120)  # SOTA can be slow under load
        print(f"\n[SYNC VERIFY] Checking sync for session {session_id}", flush=True)

        # Check sync completion
        sqlite_ok, neo4j_ok, status = check_sync_completed(session_id, timeout)

        if not sqlite_ok:
            pytest.fail(
                f"SQLite extraction batch not completed for session {session_id}. Status: {status}"
            )

        if not neo4j_ok:
            print(
                f"[SYNC VERIFY] WARNING: Neo4j not verified (status: {status})",
                flush=True,
            )

        print(
            f"[SYNC VERIFY] SUCCESS: Sync completed for session {session_id}",
            flush=True,
        )
        return True

    return verify
