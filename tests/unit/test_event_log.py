import json
import logging
import queue
import sqlite3
from unittest.mock import patch, MagicMock

import pytest

from proxy.event_log import (
    SQLiteHandler,
    BoundedQueueHandler,
    log_message,
    parse_log_types,
    set_request_context,
    clear_request_context,
    _extract_payload,
    _matches,
    _now_iso,
    close_conn,
    _init_db,
)

_LEVEL_MAP = {
    logging.DEBUG: "DEBUG",
    logging.INFO: "INFO",
    logging.WARNING: "WARNING",
    logging.ERROR: "ERROR",
    logging.CRITICAL: "CRITICAL",
}


@pytest.fixture(autouse=True)
def reset_event_log_module():
    import proxy.event_log as mod
    old_conn = mod._conn
    mod._conn = None
    yield
    mod._conn = old_conn


@pytest.fixture
def temp_db():
    conn = sqlite3.connect(":memory:", check_same_thread=False, isolation_level="")
    _init_db(conn)
    yield conn
    conn.close()


def _make_record(name="test.logger", levelno=logging.INFO, msg="test message", **extra):
    record = logging.makeLogRecord({
        "name": name,
        "levelno": levelno,
        "levelname": _LEVEL_MAP.get(levelno, "NOTSET"),
        "msg": msg,
        "args": (),
    })
    for k, v in extra.items():
        setattr(record, k, v)
    return record


class TestSQLiteHandlerEmit:
    def test_all_levels_written(self, temp_db):
        with patch("proxy.event_log._get_conn", return_value=temp_db):
            handler = SQLiteHandler(allowed=None)
            for lvl in (logging.DEBUG, logging.INFO, logging.WARNING, logging.ERROR, logging.CRITICAL):
                handler.emit(_make_record(name="proxy.test", levelno=lvl, msg=f"msg-{lvl}"))
        rows = temp_db.execute("SELECT level FROM event_log ORDER BY id").fetchall()
        assert len(rows) == 5
        assert [r[0] for r in rows] == ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]

    def test_phase1_defaults(self, temp_db):
        with patch("proxy.event_log._get_conn", return_value=temp_db):
            handler = SQLiteHandler(allowed=None)
            handler.emit(_make_record(name="proxy.http_client", levelno=logging.INFO, msg="request"))
        row = temp_db.execute("SELECT subsystem, event_type FROM event_log").fetchone()
        assert row[0] == "proxy.http_client"
        assert row[1] == "general"

    def test_explicit_subsystem_and_event_type(self, temp_db):
        with patch("proxy.event_log._get_conn", return_value=temp_db):
            handler = SQLiteHandler(allowed=None)
            handler.emit(_make_record(name="proxy.http_client", levelno=logging.INFO, msg="req",
                                       subsystem="http", event_type="upstream_call"))
        row = temp_db.execute("SELECT subsystem, event_type FROM event_log").fetchone()
        assert row[0] == "http"
        assert row[1] == "upstream_call"

    def test_payload_json_null_when_no_extra(self, temp_db):
        with patch("proxy.event_log._get_conn", return_value=temp_db):
            handler = SQLiteHandler(allowed=None)
            record = _make_record(name="proxy.test", levelno=logging.INFO, msg="bare")
            handler.emit(record)
        row = temp_db.execute("SELECT payload_json FROM event_log").fetchone()
        assert row[0] is None

    def test_payload_json_valid_when_extra(self, temp_db):
        with patch("proxy.event_log._get_conn", return_value=temp_db):
            handler = SQLiteHandler(allowed=None)
            handler.emit(_make_record(name="proxy.test", levelno=logging.INFO, msg="extra",
                                       custom_field="value"))
        row = temp_db.execute("SELECT payload_json FROM event_log").fetchone()
        payload = json.loads(row[0])
        assert payload["custom_field"] == "value"

    def test_insert_failure_calls_handleError_no_recursion(self, temp_db):
        bad_conn = MagicMock()
        bad_conn.execute.side_effect = sqlite3.Error("boom")
        handler = SQLiteHandler(allowed=None)
        handle_error_calls = []
        handler.handleError = lambda r: handle_error_calls.append(r)

        root_logger = logging.getLogger()
        root_handler = MagicMock()
        root_logger.addHandler(root_handler)
        try:
            with patch("proxy.event_log._get_conn", return_value=bad_conn):
                handler.emit(_make_record(name="proxy.test", levelno=logging.INFO, msg="fail"))
            assert len(handle_error_calls) == 1
            assert handle_error_calls[0].msg == "fail"
            root_handler.emit.assert_not_called()
        finally:
            root_logger.removeHandler(root_handler)


class TestAllowListFilter:
    def test_parse_log_types_star_returns_none(self):
        assert parse_log_types("*") is None

    def test_parse_log_types_star_star_returns_none(self):
        assert parse_log_types("*:*") is None

    def test_parse_log_types_http_star_and_star_error(self):
        result = parse_log_types("http:*,*:ERROR")
        assert result == {("http", "*"), ("*", "ERROR")}

    def test_parse_log_types_single_subsystem(self):
        result = parse_log_types("http")
        assert result == {("http", "*")}

    def test_parse_log_types_empty_returns_none(self):
        assert parse_log_types("") is None

    def test_parse_log_types_commas_returns_none(self):
        assert parse_log_types(",,,") is None

    def test_parse_log_types_lone_colon_fails_open(self):
        assert parse_log_types(":") is None

    def test_parse_log_types_triple_colon_fails_open(self):
        assert parse_log_types(":::") is None

    def test_matches_http_star(self):
        allowed = {("http", "*")}
        assert _matches("http", "anything", allowed) is True
        assert _matches("memory", "sync", allowed) is False

    def test_matches_star_error(self):
        allowed = {("*", "ERROR")}
        assert _matches("anything", "ERROR", allowed) is True
        assert _matches("anything", "INFO", allowed) is False

    def test_matches_none_allowed(self):
        assert _matches("anything", "anything", None) is True

    def test_handler_filters_non_matching(self, temp_db):
        with patch("proxy.event_log._get_conn", return_value=temp_db):
            handler = SQLiteHandler(allowed={("http", "*")})
            handler.emit(_make_record(name="proxy.http", levelno=logging.INFO, msg="match",
                                       subsystem="http", event_type="upstream"))
            handler.emit(_make_record(name="proxy.memory", levelno=logging.INFO, msg="no-match",
                                       subsystem="memory", event_type="sync"))
        rows = temp_db.execute("SELECT message FROM event_log").fetchall()
        assert len(rows) == 1
        assert rows[0][0] == "match"


class TestLogMessage:
    def test_sets_subsystem_event_type_request_session(self, temp_db):
        logger = logging.getLogger("http")
        logger.setLevel(logging.DEBUG)
        with patch("proxy.event_log._get_conn", return_value=temp_db):
            handler = SQLiteHandler(allowed=None)
            logger.addHandler(handler)
            try:
                log_message("http", "upstream_call", "INFO", "test msg",
                            request_id="rid1", session_id="sid1", latency_ms=42)
            finally:
                logger.removeHandler(handler)
        row = temp_db.execute(
            "SELECT subsystem, event_type, request_id, session_id, payload_json FROM event_log"
        ).fetchone()
        assert row[0] == "http"
        assert row[1] == "upstream_call"
        assert row[2] == "rid1"
        assert row[3] == "sid1"
        payload = json.loads(row[4])
        assert payload["latency_ms"] == 42

    def test_drops_reserved_keys_and_warns(self, caplog):
        with caplog.at_level(logging.WARNING):
            log_message("http", "upstream_call", "INFO", "test msg", name="bad", levelno=99)
        assert "dropped reserved-key field" in caplog.text
        assert "name" in caplog.text

    def test_invalid_severity_raises(self):
        with pytest.raises(AttributeError):
            log_message("http", "upstream_call", "INVALID", "test msg")

    def test_valid_severities(self, temp_db):
        logger = logging.getLogger("http")
        logger.setLevel(logging.DEBUG)
        with patch("proxy.event_log._get_conn", return_value=temp_db):
            handler = SQLiteHandler(allowed=None)
            logger.addHandler(handler)
            try:
                for sev in ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"):
                    log_message("http", "test", sev, f"msg-{sev}")
            finally:
                logger.removeHandler(handler)
        rows = temp_db.execute("SELECT level FROM event_log ORDER BY id").fetchall()
        assert [r[0] for r in rows] == ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]


class TestQueueOverflow:
    def test_bump_log_overflow_called_on_full_queue(self):
        q = queue.Queue(maxsize=1)
        handler = BoundedQueueHandler(q)
        q.put_nowait("sentinel")
        bump_calls = []
        with patch("proxy.state.bump_log_overflow", side_effect=lambda: bump_calls.append(1)):
            handler.enqueue(_make_record(name="proxy.test", levelno=logging.INFO, msg="overflow"))
        assert len(bump_calls) == 1

    def test_log_overflow_count_increments(self):
        import proxy.state as state
        old = state.log_overflow_count
        q = queue.Queue(maxsize=1)
        handler = BoundedQueueHandler(q)
        q.put_nowait("sentinel")
        handler.enqueue(_make_record(name="proxy.test", levelno=logging.INFO, msg="overflow"))
        assert state.log_overflow_count == old + 1

    def test_cmd_status_renders_log_overflow_count(self):
        with patch.multiple(
            "proxy.state",
            request_counter=0,
            tool_cache_size=0,
            tool_cache_hits=0,
            tool_cache_misses=0,
            chat_histories={},
            log_overflow_count=7,
            create=True,
        ):
            from commands import CommandContext, dispatch_command
            ctx = CommandContext(raw_text="^^status", session_id="s1")
            resp = dispatch_command(ctx)
            content = resp["choices"][0]["message"]["content"]
            assert "Log lines dropped (queue overflow): 7" in content


class TestRequestContext:
    def test_contextvar_lands_in_record(self, temp_db):
        set_request_context("rid-ctx", "sid-ctx")
        try:
            with patch("proxy.event_log._get_conn", return_value=temp_db):
                handler = SQLiteHandler(allowed=None)
                handler.emit(_make_record(name="proxy.test", levelno=logging.INFO, msg="ctx"))
        finally:
            clear_request_context()
        row = temp_db.execute("SELECT request_id, session_id FROM event_log").fetchone()
        assert row[0] == "rid-ctx"
        assert row[1] == "sid-ctx"

    def test_explicit_extra_takes_precedence_over_contextvar(self, temp_db):
        set_request_context("rid-ctx", "sid-ctx")
        try:
            with patch("proxy.event_log._get_conn", return_value=temp_db):
                handler = SQLiteHandler(allowed=None)
                handler.emit(_make_record(name="proxy.test", levelno=logging.INFO, msg="prec",
                                           request_id="explicit-rid"))
        finally:
            clear_request_context()
        row = temp_db.execute("SELECT request_id, session_id FROM event_log").fetchone()
        assert row[0] == "explicit-rid"
        assert row[1] == "sid-ctx"

    def test_clear_request_context_removes_contextvar(self, temp_db):
        set_request_context("rid-ctx", "sid-ctx")
        clear_request_context()
        with patch("proxy.event_log._get_conn", return_value=temp_db):
            handler = SQLiteHandler(allowed=None)
            handler.emit(_make_record(name="proxy.test", levelno=logging.INFO, msg="cleared"))
        row = temp_db.execute("SELECT request_id, session_id FROM event_log").fetchone()
        assert row[0] is None
        assert row[1] is None

    def test_contextvar_survives_queue_thread_boundary(self, temp_db):
        import logging.handlers

        q = queue.Queue(maxsize=10)
        sqlite_h = SQLiteHandler(allowed=None)
        sqlite_h.setFormatter(logging.Formatter("%(message)s"))
        with patch("proxy.event_log._get_conn", return_value=temp_db):
            listener = logging.handlers.QueueListener(q, sqlite_h)
            listener.start()
            try:
                set_request_context("REQ-CROSS-THREAD", "SID-CROSS-THREAD")
                try:
                    qh = BoundedQueueHandler(q)
                    test_logger = logging.getLogger("proxy.test.cross_thread")
                    test_logger.handlers = []
                    test_logger.addHandler(qh)
                    test_logger.setLevel(logging.DEBUG)
                    test_logger.info("log from producer thread")
                finally:
                    clear_request_context()
            finally:
                listener.stop()

        import time as _t
        _t.sleep(0.1)
        row = temp_db.execute("SELECT request_id, session_id, message FROM event_log").fetchone()
        assert row is not None
        assert row[0] == "REQ-CROSS-THREAD", (
            f"contextvar did not survive thread boundary: got request_id={row[0]!r}")
        assert row[1] == "SID-CROSS-THREAD"
        assert "log from producer thread" in row[2]


class TestExtractPayload:
    def test_empty_extra_returns_none(self):
        record = _make_record(name="test", levelno=logging.INFO, msg="bare")
        assert _extract_payload(record) is None

    def test_classification_keys_excluded(self):
        record = _make_record(name="test", levelno=logging.INFO, msg="cls",
                               subsystem="http", event_type="upstream",
                               request_id="rid", session_id="sid",
                               custom="val")
        payload = json.loads(_extract_payload(record))
        assert "subsystem" not in payload
        assert "event_type" not in payload
        assert "request_id" not in payload
        assert "session_id" not in payload
        assert payload["custom"] == "val"

    def test_logrecord_reserved_keys_excluded(self):
        record = _make_record(name="test", levelno=logging.INFO, msg="reserved",
                               custom="val")
        payload = json.loads(_extract_payload(record))
        assert "name" not in payload
        assert "levelno" not in payload
        assert "msg" not in payload
        assert payload["custom"] == "val"

    def test_non_serializable_uses_default_str(self):
        obj = object()
        record = _make_record(name="test", levelno=logging.INFO, msg="nonser",
                               bad_field=obj)
        payload_str = _extract_payload(record)
        assert payload_str is not None
        payload = json.loads(payload_str)
        assert "bad_field" in payload
        assert str(obj) in payload["bad_field"]


class TestNowIso:
    def test_returns_iso8601_with_timezone(self):
        result = _now_iso()
        assert "T" in result
        assert "+" in result or "Z" in result


class TestCloseConn:
    def test_closes_connection_and_sets_none(self):
        import proxy.event_log as mod
        old = mod._conn
        try:
            conn = sqlite3.connect(":memory:")
            mod._conn = conn
            close_conn()
            assert mod._conn is None
        finally:
            mod._conn = old


class TestRetentionPurge:
    def test_deletes_old_rows_keeps_new(self, temp_db):
        temp_db.execute(
            "INSERT INTO event_log (timestamp, level, message) VALUES "
            "(datetime('now', '-31 days'), 'INFO', 'old'),"
            "(datetime('now', '-1 days'), 'INFO', 'new')"
        )
        temp_db.execute(
            "DELETE FROM event_log WHERE timestamp < datetime('now', '-30 days')"
        )
        rows = temp_db.execute("SELECT message FROM event_log ORDER BY id").fetchall()
        assert len(rows) == 1
        assert rows[0][0] == "new"

    def test_enforces_cap(self, temp_db):
        cap = 10
        for i in range(cap + 1):
            temp_db.execute(
                "INSERT INTO event_log (timestamp, level, message) VALUES (datetime('now'), 'INFO', ?)",
                (f"msg-{i}",),
            )
        temp_db.execute(
            "DELETE FROM event_log WHERE id NOT IN (SELECT id FROM event_log ORDER BY id DESC LIMIT ?)",
            (cap,),
        )
        count = temp_db.execute("SELECT COUNT(*) FROM event_log").fetchone()[0]
        assert count == cap


class TestAutocommitDurability:
    """Pins the autocommit fix: the handler conn (isolation_level=None) must not
    hold the WAL write lock after emit, and purge DELETEs must be durable."""

    @pytest.fixture(autouse=True)
    def _tmp_cwd(self, tmp_path, monkeypatch):
        (tmp_path / "data").mkdir()
        monkeypatch.chdir(tmp_path)
        close_conn()
        yield
        close_conn()

    def test_emit_does_not_hold_write_lock(self):
        handler = SQLiteHandler(allowed=None)
        handler.emit(_make_record(name="proxy.test", levelno=logging.INFO, msg="durable"))

        second = sqlite3.connect("data/event_log.db", timeout=0.5)
        try:
            second.execute("PRAGMA busy_timeout=500")
            # Durability: the inserted row is visible from another connection.
            count = second.execute("SELECT COUNT(*) FROM event_log").fetchone()[0]
            assert count == 1
            # No held write lock: a second-connection DELETE must succeed.
            second.execute("DELETE FROM event_log")
            second.commit()
        finally:
            second.close()

    def test_purge_deletes_are_durable(self):
        from datetime import datetime, timedelta, timezone

        from proxy.event_log import _get_conn

        conn = _get_conn()
        old_ts = (datetime.now(timezone.utc) - timedelta(days=40)).isoformat()
        conn.execute(
            "INSERT INTO event_log (timestamp, level, message) VALUES (?, 'INFO', 'ancient')",
            (old_ts,),
        )
        close_conn()

        # Replicate the purge body exactly as the fixed purge_event_log opens it.
        purge_conn = sqlite3.connect("data/event_log.db", isolation_level=None, timeout=30)
        try:
            purge_conn.execute("PRAGMA busy_timeout=5000")
            purge_conn.execute(
                "DELETE FROM event_log WHERE timestamp < datetime('now', ?)",
                ("-30 days",),
            )
            purge_conn.execute(
                "DELETE FROM event_log WHERE id NOT IN "
                "(SELECT id FROM event_log ORDER BY id DESC LIMIT ?)",
                (2_000_000,),
            )
        finally:
            purge_conn.close()

        fresh = sqlite3.connect("data/event_log.db")
        try:
            count = fresh.execute("SELECT COUNT(*) FROM event_log").fetchone()[0]
            assert count == 0, "purge DELETE was rolled back on close (not durable)"
        finally:
            fresh.close()


class TestSetRequestContextInParseRequest:
    def test_parse_request_calls_set_request_context(self):
        mock_flask_request = MagicMock()
        mock_flask_request.get_json.return_value = {
            "model": "test-model",
            "messages": [{"role": "user", "content": "hello"}],
        }
        mock_flask_request.headers.get.side_effect = lambda k, d=None: {
            "X-Session-Client": "test-client",
        }.get(k, d)
        mock_flask_request.remote_addr = "127.0.0.1"
        with patch("proxy.event_log.set_request_context") as mock_set, \
             patch("proxy.model_registry.resolve_session_id", return_value="sid-test"), \
             patch("proxy.canary.detect_subagent_ingress") as mock_ingress:
            mock_ingress.return_value = MagicMock(
                is_subagent=False, subagent_ticket_mode=False,
                inherited_skill=None, task_hash=None, workspace_name="",
            )
            from proxy.pipeline.stages import parse_request
            parse_request(mock_flask_request)
            assert mock_set.call_count == 1
            call_args = mock_set.call_args[0]
            assert isinstance(call_args[0], str) and len(call_args[0]) == 12
            assert call_args[1] == "sid-test"
