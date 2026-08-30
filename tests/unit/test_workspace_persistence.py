"""Tests for memory.workspace_persistence (v5 global catalog persistence layer).

This module owns the v5 persist/recover/cleanup primitives:
  - _persist_workspace(workspace_name, session_id)  — write :Workspace,
    :Session, and :LAST_USED_WORKSPACE in a single Cypher.
  - recover_workspaces(store, session_id)          — load the global catalog
    and the per-session last-used pointer, then atomic try_recover.
  - cleanup_neo4j_sessions(session_ids)            — batch-delete :Session +
    :LAST_USED_WORKSPACE edges, NEVER :Workspace nodes.
  - get_global_catalog(executor, limit)            — bare-name catalog query.
  - create_persist_callback() / create_recovery_fn(store) — wiring helpers.

The tests use a mock executor that records every Cypher call and returns
scripted result rows. The store is a real WorkspaceStateStore (per-session
state is not Neo4j-backed, so no point in mocking it).
"""
import logging
import threading
from unittest.mock import MagicMock, patch

import pytest

from memory.workspace_persistence import (
    CATALOG_QUERY,
    CLEANUP_BATCH_SIZE,
    _persist_workspace,
    cleanup_neo4j_sessions,
    create_persist_callback,
    create_recovery_fn,
    get_global_catalog,
    recover_workspaces,
)
from proxy.workspace_state import WorkspaceStateStore


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _RecordingExecutor:
    """Mock executor that returns scripted results and records every call.

    `read_results` and `write_results` are queues: each call pulls the next
    item. If a queue is exhausted, returns `[]`.
    """

    def __init__(self, read_results=None, write_results=None):
        self.calls: list[dict] = []
        self._read_results = list(read_results or [])
        self._write_results = list(write_results or [])

    def execute_read(self, query, params=None, timeout=30):
        self.calls.append({"query": query, "params": params or {}, "op": "read"})
        if self._read_results:
            return self._read_results.pop(0)
        return []

    def execute_write(self, query, params=None, timeout=30):
        self.calls.append({"query": query, "params": params or {}, "op": "write"})
        if self._write_results:
            return self._write_results.pop(0)
        return []


def _first_query_containing(executor, needle: str) -> str:
    for c in executor.calls:
        if needle.lower() in c["query"].lower():
            return c["query"]
    raise AssertionError(f"No call with query containing {needle!r}")


def _all_queries_containing(executor, needle: str) -> list[str]:
    return [c["query"] for c in executor.calls if needle.lower() in c["query"].lower()]


@pytest.fixture
def lock():
    return threading.RLock()


@pytest.fixture
def store(lock):
    return WorkspaceStateStore(lock)


# ---------------------------------------------------------------------------
# _persist_workspace
# ---------------------------------------------------------------------------


class TestPersistCallback:
    def test_persist_callback_writes_bare_name(self):
        """_persist_workspace must MERGE on the bare name — never a compound.

        Cypher must use `w.workspace_id = $workspace_id` and pass the
        workspace_name as that parameter, with no embedded `session:name`
        literal in the persisted id.
        """
        executor = _RecordingExecutor()
        with patch(
            "memory.workspace_persistence._get_executor", return_value=executor
        ):
            _persist_workspace("choriprox", "sess1")

        assert len(executor.calls) == 1
        call = executor.calls[0]
        assert call["params"]["workspace_id"] == "choriprox"
        assert call["params"]["name"] == "choriprox"
        assert call["params"]["session_id"] == "sess1"
        # The Cypher must MERGE on the bare workspace_id property.
        assert "MERGE (w:Workspace {workspace_id: $workspace_id})" in call["query"]
        # The parameter $workspace_id must be passed the BARE name (no
        # session_id:workspace_name compound string in the params).
        passed_value = call["params"]["workspace_id"]
        assert ":" not in passed_value, (
            f"workspace_id param must be bare (no ':'), got {passed_value!r}"
        )

    def test_persist_callback_writes_session_and_edge(self):
        """The persist Cypher must also MERGE :Session and the edge.

        Required: `MERGE (s:Session {id: sid})` (with `sid` bound from
        $session_id via WITH) AND `MERGE (s)-[r:LAST_USED_WORKSPACE]->(w)`.
        """
        executor = _RecordingExecutor()
        with patch(
            "memory.workspace_persistence._get_executor", return_value=executor
        ):
            _persist_workspace("choriprox", "sess1")

        query = executor.calls[0]["query"]
        # The actual implementation binds $session_id to `sid` via WITH.
        assert "MERGE (s:Session {id: sid})" in query, (
            f"Persist Cypher must MERGE :Session:\n{query}"
        )
        assert "MERGE (s)-[r:LAST_USED_WORKSPACE]->(w)" in query, (
            f"Persist Cypher must MERGE :LAST_USED_WORKSPACE edge:\n{query}"
        )
        # The session_id is passed via $session_id param.
        assert executor.calls[0]["params"]["session_id"] == "sess1"

    def test_persist_callback_swallows_exceptions(self, caplog):
        """A failing executor must NOT propagate the exception. WARNING logged."""
        class _BoomExecutor:
            def execute_write(self, query, params=None, timeout=30):
                raise RuntimeError("simulated neo4j failure")

        with patch(
            "memory.workspace_persistence._get_executor", return_value=_BoomExecutor()
        ):
            with caplog.at_level(logging.WARNING, logger="memory.workspace_persistence"):
                # Must not raise.
                _persist_workspace("choriprox", "sess1")

        assert any("persist failed" in r.message.lower() or
                   "persist failed" in r.getMessage().lower()
                   for r in caplog.records), (
            "Expected a 'persist failed' WARNING in the log"
        )

    def test_persist_callback_noop_without_executor(self):
        """If _get_executor() returns None, _persist_workspace is a silent no-op."""
        with patch("memory.workspace_persistence._get_executor", return_value=None):
            # Must not raise.
            _persist_workspace("choriprox", "sess1")


# ---------------------------------------------------------------------------
# recover_workspaces
# ---------------------------------------------------------------------------


class TestRecovery:
    def test_recovery_populates_catalog(self, store):
        """recover_workspaces must load the catalog and set it on the store."""
        executor = _RecordingExecutor(read_results=[
            # Step 1: catalog read
            [{"name": "choriprox", "last_active_at": "2024-01-02"},
             {"name": "luna", "last_active_at": "2024-01-01"}],
            # Step 2: last-used read
            [{"name": "choriprox"}],
        ])
        with patch("memory.workspace_persistence._get_executor", return_value=executor):
            recover_workspaces(store, "sess1")

        # The store's catalog cache must contain both names + 'default' (always added).
        catalog = store.get_global_catalog()
        assert "choriprox" in catalog
        assert "luna" in catalog
        assert "default" in catalog

    def test_recovery_restores_per_session_last_used(self, store):
        """recover_workspaces must call try_recover with the :LAST_USED_WORKSPACE
        target — that lands in store._active[session_id].
        """
        executor = _RecordingExecutor(read_results=[
            [{"name": "choriprox", "last_active_at": "2024-01-01"}],
            [{"name": "choriprox"}],
        ])
        with patch("memory.workspace_persistence._get_executor", return_value=executor):
            recover_workspaces(store, "sess1")

        assert store._active.get("sess1") == "choriprox", (
            f"Expected sess1 → choriprox in _active, got {store._active}"
        )

    def test_recovery_filters_null_names(self, store):
        """Records with name=NULL (legacy data) must be filtered out of the catalog
        AND must not be passed to try_recover.
        """
        executor = _RecordingExecutor(read_results=[
            [{"name": None, "last_active_at": "2024-01-01"},
             {"name": "choriprox", "last_active_at": "2024-01-02"}],
            # Last-used query: returns choriprox (None would be filtered).
            [{"name": "choriprox"}],
        ])
        with patch("memory.workspace_persistence._get_executor", return_value=executor):
            recover_workspaces(store, "sess1")

        # try_recover must have been called with "choriprox", not None.
        assert store._active.get("sess1") == "choriprox", (
            f"Expected choriprox (None filtered), got {store._active.get('sess1')!r}"
        )
        # Catalog must not contain None.
        catalog = store.get_global_catalog()
        assert None not in catalog
        assert all(isinstance(n, str) for n in catalog)

    def test_recovery_no_last_used_stays_default(self, store):
        """If the last-used query returns empty, the session stays at 'default'."""
        executor = _RecordingExecutor(read_results=[
            # catalog (could be empty)
            [],
            # last-used: no rows → session has no prior record
            [],
        ])
        with patch("memory.workspace_persistence._get_executor", return_value=executor):
            recover_workspaces(store, "sess1")

        # No last-used → try_recover not called → _active stays empty.
        assert store._active.get("sess1", "default") == "default", (
            f"Expected sess1 → default (no last-used), got {store._active}"
        )


# ---------------------------------------------------------------------------
# get_global_catalog
# ---------------------------------------------------------------------------


class TestGetGlobalCatalog:
    def test_get_global_catalog_returns_sorted_by_last_active(self):
        """The catalog query must include ORDER BY last_active_at DESC.

        The mocked executor just returns rows in the order the Cypher
        produces them; we return data already in DESC order and assert
        the query's ORDER BY clause is present (which is what enforces
        the order in production).
        """
        # Already-sorted DESC by last_active_at — to mimic what Neo4j
        # would return when the query is `ORDER BY w.last_active_at DESC`.
        executor = _RecordingExecutor(read_results=[
            [{"name": "newest", "last_active_at": "2024-01-03T00:00:00Z"},
             {"name": "middle", "last_active_at": "2024-01-02T00:00:00Z"},
             {"name": "oldest", "last_active_at": "2024-01-01T00:00:00Z"}],
        ])

        with patch("memory.workspace_persistence._get_executor", return_value=executor):
            result = get_global_catalog(executor=executor, limit=10)

        # Names appear in the order returned by the (ORDER BY DESC) query.
        assert result == ["newest", "middle", "oldest", "default"]
        # And the actual query used by get_global_catalog has the ORDER BY.
        assert "ORDER BY w.last_active_at DESC" in CATALOG_QUERY

    def test_get_global_catalog_includes_default(self):
        """Even with an empty result, the catalog must return ['default']."""
        executor = _RecordingExecutor(read_results=[[]])

        with patch("memory.workspace_persistence._get_executor", return_value=executor):
            result = get_global_catalog(executor=executor)

        assert result == ["default"], (
            f"Empty catalog must be ['default'], got {result}"
        )

    def test_get_global_catalog_falls_back_to_default_on_failure(self):
        """A failing executor must return ['default'], not raise."""
        class _BoomExecutor:
            def execute_read(self, query, params=None, timeout=30):
                raise RuntimeError("simulated neo4j failure")

        with patch("memory.workspace_persistence._get_executor",
                   return_value=_BoomExecutor()):
            result = get_global_catalog()

        assert result == ["default"]

    def test_recovery_query_has_limit(self, store):
        """The catalog query must use LIMIT to bound results."""
        executor = _RecordingExecutor(read_results=[
            [{"name": "choriprox", "last_active_at": "2024-01-01"}],
            # No last-used result (test will not depend on it).
            [],
        ])
        with patch("memory.workspace_persistence._get_executor", return_value=executor):
            recover_workspaces(store, "sess1")

        # Find the catalog read call.
        catalog_call = next(
            (c for c in executor.calls if c["op"] == "read" and "LIMIT" in c["query"]),
            None,
        )
        assert catalog_call is not None, (
            "Catalog query must include LIMIT clause. Saw:\n"
            + "\n--\n".join(c["query"] for c in executor.calls)
        )
        assert catalog_call["params"]["limit"] == 100, (
            f"Default catalog limit must be 100, got {catalog_call['params'].get('limit')}"
        )


# ---------------------------------------------------------------------------
# cleanup_neo4j_sessions
# ---------------------------------------------------------------------------


class TestCleanupNeo4jSessions:
    def test_cleanup_neo4j_sessions_does_not_touch_workspaces(self):
        """The cleanup Cypher must touch :Session only — NEVER :Workspace."""
        executor = _RecordingExecutor()
        with patch("memory.workspace_persistence._get_executor", return_value=executor):
            cleanup_neo4j_sessions(["sess1"])

        assert len(executor.calls) == 1
        query = executor.calls[0]["query"]
        assert ":Session" in query, f"Cleanup must touch :Session:\n{query}"
        # The MATCH is on :Session — there must be no standalone MATCH on :Workspace
        # (i.e., it must not be deleting :Workspace nodes).
        # We assert by checking the DELETE clause.
        assert "DELETE r, s" in query or "DETACH DELETE s" in query, (
            f"Cleanup must delete :Session (and possibly edge), not :Workspace:\n{query}"
        )
        # Sanity: must not write `MATCH (w:Workspace)` (workspace deletion).
        assert "MATCH (w:Workspace)" not in query, (
            f"Cleanup must NOT touch :Workspace nodes:\n{query}"
        )

    def test_cleanup_neo4j_sessions_batches_1000(self):
        """2500 session IDs must be split into 3 batches of 1000/1000/500."""
        session_ids = [f"sess{i}" for i in range(2500)]
        executor = _RecordingExecutor()
        with patch("memory.workspace_persistence._get_executor", return_value=executor):
            cleanup_neo4j_sessions(session_ids)

        # 3 batches → 3 execute_write calls.
        assert len(executor.calls) == 3, (
            f"Expected 3 batched write calls for 2500 ids, got {len(executor.calls)}"
        )
        # First two batches are full (1000), last is partial (500).
        sizes = [len(c["params"]["session_ids"]) for c in executor.calls]
        assert sizes == [1000, 1000, 500], (
            f"Expected batch sizes [1000, 1000, 500], got {sizes}"
        )
        # And the BATCH_SIZE constant is 1000.
        assert CLEANUP_BATCH_SIZE == 1000

    def test_cleanup_neo4j_sessions_empty_list_is_noop(self):
        """An empty session_ids list must not execute any Cypher."""
        executor = _RecordingExecutor()
        with patch("memory.workspace_persistence._get_executor", return_value=executor):
            cleanup_neo4j_sessions([])

        assert len(executor.calls) == 0

    def test_cleanup_neo4j_sessions_swallows_exceptions(self, caplog):
        """A failing batch must not stop subsequent batches."""
        class _SometimesBoomExecutor:
            def __init__(self):
                self._calls = 0

            def execute_write(self, query, params=None, timeout=30):
                self._calls += 1
                if self._calls == 1:
                    raise RuntimeError("simulated batch failure")
                return []

        executor = _SometimesBoomExecutor()
        with patch("memory.workspace_persistence._get_executor", return_value=executor):
            # Must not raise.
            cleanup_neo4j_sessions([f"sess{i}" for i in range(2500)])

        # Both batches attempted (the second one succeeded).
        assert executor._calls == 3


# ---------------------------------------------------------------------------
# create_persist_callback / create_recovery_fn
# ---------------------------------------------------------------------------


class TestCallbackFactory:
    def test_create_persist_callback_returns_callable_when_executor_present(self):
        with patch(
            "memory.workspace_persistence._get_executor",
            return_value=MagicMock(),
        ):
            cb = create_persist_callback()
        assert cb is not None
        assert callable(cb)

    def test_create_persist_callback_returns_none_when_no_executor(self):
        with patch("memory.workspace_persistence._get_executor", return_value=None):
            cb = create_persist_callback()
        assert cb is None

    def test_create_recovery_fn_returns_callable_bound_to_store(self, store):
        fn = create_recovery_fn(store)
        assert callable(fn)
        # The returned partial should accept a single session_id arg.
        # Smoke-test: pass a session_id and confirm it doesn't raise
        # (the inner call will fail because we didn't patch the executor,
        # but the wrapping partial itself is callable).
        with patch("memory.workspace_persistence._get_executor", return_value=None):
            # With no executor, recover_workspaces is a no-op.
            fn("sess1")  # Must not raise.


# ---------------------------------------------------------------------------
# CATALOG_QUERY module-level constant
# ---------------------------------------------------------------------------


class TestCatalogQuery:
    def test_catalog_query_filters_compound_ids(self):
        """The catalog query must explicitly exclude compound workspace_ids."""
        assert "NOT w.workspace_id CONTAINS ':'" in CATALOG_QUERY, (
            "Catalog query must filter out compound IDs to avoid showing "
            "post-v4 / pre-migration data"
        )

    def test_catalog_query_returns_workspace_id_alias(self):
        """The catalog query must return the bare name under the 'name' alias."""
        assert "w.workspace_id AS name" in CATALOG_QUERY, (
            "Catalog query must alias workspace_id as 'name' for the persistence layer"
        )

    def test_catalog_query_has_limit(self):
        """The catalog query must use $limit for bounded results."""
        assert "LIMIT $limit" in CATALOG_QUERY
