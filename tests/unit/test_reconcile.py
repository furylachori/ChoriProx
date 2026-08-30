"""Tests for reconcile_sessions: SQLite↔Neo4j divergence detection.

Verifies that reconcile_sessions() returns the correct per-session
entity count structure and flags divergence without auto-repairing.
"""
import logging
from unittest.mock import MagicMock, patch

import pytest


def _make_db_conn_mock(rows):
    """Build a mock for memory.db.connection that returns `rows` from execute().fetchall().

    Backward-compatible: returns the same `rows` for every execute() call.
    Use _make_db_conn_mock_with_calls() when entity and fact queries need
    different fixtures.
    """
    mock_conn = MagicMock()
    mock_conn.__enter__ = MagicMock(return_value=mock_conn)
    mock_conn.__exit__ = MagicMock(return_value=False)
    mock_conn.execute.return_value.fetchall.return_value = rows
    return mock_conn


def _make_db_conn_mock_with_calls(*row_sets):
    """Build a mock that returns a different row set for each execute() call.

    Use this when entity and fact queries need different fixtures. The order
    of execute() calls in reconcile_sessions() is: entity query first, then
    fact query.
    """
    mock_conn = MagicMock()
    mock_conn.__enter__ = MagicMock(return_value=mock_conn)
    mock_conn.__exit__ = MagicMock(return_value=False)
    fetchall_results = [MagicMock(fetchall=MagicMock(return_value=rows)) for rows in row_sets]
    mock_conn.execute.side_effect = fetchall_results
    return mock_conn


def _make_graph_mock(available=True, execute_read_return=None, execute_read_side_effect=None):
    """Build a mock for memory.persistence._get_graph()."""
    g = MagicMock()
    g.is_available.return_value = available
    if execute_read_side_effect is not None:
        g._executor.execute_read.side_effect = execute_read_side_effect
    else:
        g._executor.execute_read.return_value = execute_read_return if execute_read_return is not None else []
    return g


def _make_graph_mock_with_queries(query_responses):
    """Build a mock that returns different responses based on query content.

    query_responses: dict mapping a substring to a response value or list of
    values (one per call). If the substring matches the query, that response
    is returned. First match wins.
    """
    g = MagicMock()
    g.is_available.return_value = True

    iterators = {k: (iter(v) if isinstance(v, list) else iter([v])) for k, v in query_responses.items()}

    def route(query, params=None):
        for key, it in iterators.items():
            if key in query:
                return next(it)
        return []

    g._executor.execute_read.side_effect = route
    return g


class TestReconcileSessions:
    """Tests for sync.reconcile_sessions()."""

    def setup_method(self):
        from memory import sync
        self._sync_module = sync

    def _call(self, session_id=None, db_rows=None, graph=None):
        """Invoke reconcile_sessions with mocked DB and graph."""
        mock_conn = _make_db_conn_mock(db_rows or [])
        with patch("memory.db_pool.connection", return_value=mock_conn), \
             patch.object(self._sync_module, "_get_graph", return_value=graph):
            return self._sync_module.reconcile_sessions(session_id=session_id)

    def test_returns_structure_with_matching_counts(self):
        """When SQLite and Neo4j agree, sessions[sid].divergent is False."""
        db_rows = [("sess-aaa", 5)]
        graph = _make_graph_mock(execute_read_return=[{"cnt": 5}])

        result = self._call(session_id=None, db_rows=db_rows, graph=graph)

        assert "sessions" in result
        assert "sess-aaa" in result["sessions"]
        assert result["sessions"]["sess-aaa"]["sqlite"] == 5
        assert result["sessions"]["sess-aaa"]["neo4j"] == 5
        assert result["sessions"]["sess-aaa"]["divergent"] is False

    def test_flags_divergence_when_counts_differ(self):
        """When SQLite and Neo4j disagree, sessions[sid].divergent is True."""
        db_rows = [("sess-bbb", 7)]
        graph = _make_graph_mock(execute_read_return=[{"cnt": 4}])

        result = self._call(db_rows=db_rows, graph=graph)

        assert result["sessions"]["sess-bbb"]["sqlite"] == 7
        assert result["sessions"]["sess-bbb"]["neo4j"] == 4
        assert result["sessions"]["sess-bbb"]["divergent"] is True

    def test_logs_warning_for_divergent_session(self, caplog):
        """Divergent sessions produce a [RECONCILE] warning log entry."""
        db_rows = [("sess-cccccccc", 10)]
        graph = _make_graph_mock(execute_read_return=[{"cnt": 3}])

        with caplog.at_level(logging.WARNING, logger="memory.sync"):
            self._call(db_rows=db_rows, graph=graph)

        assert any(
            "[RECONCILE]" in r.message and "divergent" in r.message
            for r in caplog.records
        )

    def test_logs_info_summary_after_check(self, caplog):
        """After all sessions checked, a summary info log is emitted."""
        db_rows = [("sess-a", 1), ("sess-b", 2)]
        graph = _make_graph_mock(execute_read_return=[{"cnt": 1}])

        with caplog.at_level(logging.INFO, logger="memory.sync"):
            self._call(db_rows=db_rows, graph=graph)

        assert any(
            "[RECONCILE]" in r.message and "sessions checked" in r.message
            for r in caplog.records
        )

    def test_neo4j_unavailable_returns_early(self):
        """If Neo4j is unavailable, the result still has a 'sessions' dict and
        a 'neo4j_available' flag set to False, and no Neo4j queries are run."""
        graph = _make_graph_mock(available=False)

        result = self._call(db_rows=[("sess-a", 3)], graph=graph)

        assert result["neo4j_available"] is False
        assert result["sessions"] == {}
        graph._executor.execute_read.assert_not_called()

    def test_neo4j_query_error_marked_with_negative_count(self):
        """If a Neo4j query raises, the session's neo4j count is -1 and divergent is False."""
        db_rows = [("sess-err", 5)]

        def _raise(_q, _p):
            raise RuntimeError("neo4j exploded")

        graph = _make_graph_mock(execute_read_side_effect=_raise)

        result = self._call(db_rows=db_rows, graph=graph)

        assert result["sessions"]["sess-err"]["sqlite"] == 5
        assert result["sessions"]["sess-err"]["neo4j"] == -1
        assert result["sessions"]["sess-err"]["divergent"] is False

    def test_session_id_filter_passed_to_sqlite(self):
        """When session_id is provided, both SQLite queries filter by it."""
        mock_conn = _make_db_conn_mock_with_calls([], [])
        graph = _make_graph_mock(execute_read_return=[])

        with patch("memory.db_pool.connection", return_value=mock_conn), \
             patch.object(self._sync_module, "_get_graph", return_value=graph):
            self._sync_module.reconcile_sessions(session_id="only-this")

        entity_sql = mock_conn.execute.call_args_list[0][0][0]
        entity_params = mock_conn.execute.call_args_list[0][0][1]
        fact_sql = mock_conn.execute.call_args_list[1][0][0]
        fact_params = mock_conn.execute.call_args_list[1][0][1]

        assert "FROM entities" in entity_sql
        assert "WHERE session_id = ?" in entity_sql
        assert entity_params == ("only-this",)
        assert "FROM facts" in fact_sql
        assert "WHERE session_id = ?" in fact_sql
        assert fact_params == ("only-this",)

    def test_no_session_id_filter_uses_groupby_query(self):
        """When session_id is None, both SQLite queries are the global GROUP BY."""
        mock_conn = _make_db_conn_mock_with_calls([], [])
        graph = _make_graph_mock(execute_read_return=[])

        with patch("memory.db_pool.connection", return_value=mock_conn), \
             patch.object(self._sync_module, "_get_graph", return_value=graph):
            self._sync_module.reconcile_sessions()

        entity_call = mock_conn.execute.call_args_list[0]
        fact_call = mock_conn.execute.call_args_list[1]

        entity_sql = entity_call[0][0]
        entity_params = entity_call[0][1] if len(entity_call[0]) > 1 else ()
        fact_sql = fact_call[0][0]
        fact_params = fact_call[0][1] if len(fact_call[0]) > 1 else ()

        assert "WHERE session_id = ?" not in entity_sql
        assert "GROUP BY session_id" in entity_sql
        assert entity_params == ()
        assert "WHERE session_id = ?" not in fact_sql
        assert "GROUP BY session_id" in fact_sql
        assert fact_params == ()

    def test_empty_sqlite_returns_empty_sessions(self):
        """If SQLite has no entity rows and Neo4j has no orphans, sessions dict is empty.

        Orphan detection still runs a Neo4j query, but if it returns nothing
        no sessions are added to the result.
        """
        graph = _make_graph_mock_with_queries({
            "DISTINCT e.session_id": [[]],
        })

        result = self._call(db_rows=[], graph=graph)

        assert result["sessions"] == {}

    def test_multiple_sessions_mixed_divergence(self):
        """Multiple sessions, some divergent some not, counted correctly."""
        db_rows = [("sess-1", 5), ("sess-2", 3), ("sess-3", 7)]
        counts = {"sess-1": [{"cnt": 5}], "sess-2": [{"cnt": 3}], "sess-3": [{"cnt": 0}]}
        graph = _make_graph_mock(
            execute_read_side_effect=lambda _q, p: counts[p["sid"]]
        )

        result = self._call(db_rows=db_rows, graph=graph)

        assert result["sessions"]["sess-1"]["divergent"] is False
        assert result["sessions"]["sess-2"]["divergent"] is False
        assert result["sessions"]["sess-3"]["divergent"] is True

    def test_does_not_auto_repair(self):
        """Function never writes to Neo4j — only execute_read is called."""
        db_rows = [("sess-a", 2), ("sess-b", 4)]
        graph = _make_graph_mock(execute_read_return=[{"cnt": 0}])

        self._call(db_rows=db_rows, graph=graph)

        graph._executor.execute_write.assert_not_called()
        graph._executor.execute_write_no_retry.assert_not_called()

    def test_fact_count_comparison(self):
        """Fact counts from SQLite are compared against RELATES_TO counts in Neo4j."""
        entity_rows = [("sess-match", 5), ("sess-diverg", 3)]
        fact_rows = [("sess-match", 4), ("sess-diverg", 6)]
        graph = _make_graph_mock_with_queries({
            "MATCH (e:Entity {session_id": [
                [{"cnt": 5}], [{"cnt": 3}],
                [{"cnt": 4}],
            ],
            "RELATES_TO": [
                [{"cnt": 4}],
                [{"cnt": 2}],
            ],
            "DISTINCT e.session_id": [[]],
        })
        mock_conn = _make_db_conn_mock_with_calls(entity_rows, fact_rows)

        with patch("memory.db_pool.connection", return_value=mock_conn), \
             patch.object(self._sync_module, "_get_graph", return_value=graph):
            result = self._sync_module.reconcile_sessions()

        assert result["sessions"]["sess-match"]["sqlite_facts"] == 4
        assert result["sessions"]["sess-match"]["neo4j_facts"] == 4
        assert result["sessions"]["sess-match"]["divergent"] is False

        assert result["sessions"]["sess-diverg"]["sqlite_facts"] == 6
        assert result["sessions"]["sess-diverg"]["neo4j_facts"] == 2
        assert result["sessions"]["sess-diverg"]["divergent"] is True

    def test_fact_divergence_logs_warning(self, caplog):
        """Fact-count divergence produces a [RECONCILE] facts divergent log line."""
        entity_rows = [("sess-wwwwwwww", 3)]
        fact_rows = [("sess-wwwwwwww", 9)]
        graph = _make_graph_mock_with_queries({
            "MATCH (e:Entity {session_id": [[{"cnt": 3}], [{"cnt": 1}]],
            "RELATES_TO": [[{"cnt": 1}]],
            "DISTINCT e.session_id": [[]],
        })
        mock_conn = _make_db_conn_mock_with_calls(entity_rows, fact_rows)

        with caplog.at_level(logging.WARNING, logger="memory.sync"), \
             patch("memory.db_pool.connection", return_value=mock_conn), \
             patch.object(self._sync_module, "_get_graph", return_value=graph):
            self._sync_module.reconcile_sessions()

        assert any(
            "[RECONCILE]" in r.message and "facts divergent" in r.message
            for r in caplog.records
        )

    def test_orphan_neo4j_session_detected(self):
        """A session in Neo4j but missing from SQLite is flagged as an orphan."""
        entity_rows = [("sess-known", 2)]
        fact_rows = [("sess-known", 1)]
        graph = _make_graph_mock_with_queries({
            "MATCH (e:Entity {session_id": [
                [{"cnt": 2}],
                [{"cnt": 7}],
            ],
            "RELATES_TO": [
                [{"cnt": 1}],
            ],
            "DISTINCT e.session_id": [
                [{"sid": "sess-known"}, {"sid": "sess-orphan-xyz"}],
            ],
        })
        mock_conn = _make_db_conn_mock_with_calls(entity_rows, fact_rows)

        with patch("memory.db_pool.connection", return_value=mock_conn), \
             patch.object(self._sync_module, "_get_graph", return_value=graph):
            result = self._sync_module.reconcile_sessions()

        assert "sess-orphan-xyz" in result["sessions"]
        orphan = result["sessions"]["sess-orphan-xyz"]
        assert orphan["sqlite"] == 0
        assert orphan["neo4j"] == 7
        assert orphan["sqlite_facts"] == 0
        assert orphan["neo4j_facts"] == 0
        assert orphan["orphan"] is True
        assert orphan["divergent"] is True

    def test_orphan_neo4j_session_logs_warning(self, caplog):
        """Orphan detection emits a [RECONCILE] orphan log line."""
        entity_rows = []
        fact_rows = []
        graph = _make_graph_mock_with_queries({
            "MATCH (e:Entity {session_id": [[{"cnt": 4}]],
            "DISTINCT e.session_id": [[{"sid": "sess-only-neo"}]],
        })
        mock_conn = _make_db_conn_mock_with_calls(entity_rows, fact_rows)

        with caplog.at_level(logging.WARNING, logger="memory.sync"), \
             patch("memory.db_pool.connection", return_value=mock_conn), \
             patch.object(self._sync_module, "_get_graph", return_value=graph):
            self._sync_module.reconcile_sessions()

        assert any(
            "[RECONCILE]" in r.message and "orphan" in r.message
            for r in caplog.records
        )

    def test_mixed_entity_fact_divergence(self):
        """Entity counts match but fact counts diverge → divergent=True, driven by facts."""
        entity_rows = [("sess-mix", 5)]
        fact_rows = [("sess-mix", 7)]
        graph = _make_graph_mock_with_queries({
            "MATCH (e:Entity {session_id": [[{"cnt": 5}], [{"cnt": 3}]],
            "RELATES_TO": [[{"cnt": 3}]],
            "DISTINCT e.session_id": [[]],
        })
        mock_conn = _make_db_conn_mock_with_calls(entity_rows, fact_rows)

        with patch("memory.db_pool.connection", return_value=mock_conn), \
             patch.object(self._sync_module, "_get_graph", return_value=graph):
            result = self._sync_module.reconcile_sessions()

        sess = result["sessions"]["sess-mix"]
        assert sess["sqlite"] == 5
        assert sess["neo4j"] == 5
        assert sess["sqlite_facts"] == 7
        assert sess["neo4j_facts"] == 3
        assert sess["divergent"] is True
        assert sess["orphan"] is False

    def test_orphan_detection_query_failure_does_not_crash(self):
        """If orphan detection itself raises, the function still returns a result."""
        entity_rows = [("sess-aaa", 1)]
        fact_rows = [("sess-aaa", 1)]

        def _raise(_q, _p=None):
            raise RuntimeError("orphan query boom")

        graph = _make_graph_mock(execute_read_side_effect=_raise)
        mock_conn = _make_db_conn_mock_with_calls(entity_rows, fact_rows)

        with patch("memory.db_pool.connection", return_value=mock_conn), \
             patch.object(self._sync_module, "_get_graph", return_value=graph):
            result = self._sync_module.reconcile_sessions()

        assert "sess-aaa" in result["sessions"]
        assert result["sessions"]["sess-aaa"]["sqlite"] == 1
