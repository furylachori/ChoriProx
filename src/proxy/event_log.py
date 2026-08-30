from __future__ import annotations

import contextvars
import fnmatch
import json
import logging
import logging.handlers
import os
import sqlite3
import threading
import time as _time

_conn = None
_conn_lock = threading.Lock()


def _get_conn() -> sqlite3.Connection:
    global _conn
    if _conn is not None:
        return _conn
    with _conn_lock:
        if _conn is None:
            # Autocommit: an uncommitted implicit transaction would hold the WAL write lock forever and block the purge.
            c = sqlite3.connect("data/event_log.db", check_same_thread=False,
                                isolation_level=None)
            c.execute("PRAGMA journal_mode=WAL")
            c.execute("PRAGMA busy_timeout=5000")
            c.execute("PRAGMA synchronous=NORMAL")
            _init_db(c)
            _conn = c
    return _conn


def close_conn() -> None:
    global _conn
    if _conn is not None:
        try:
            _conn.close()
        except Exception:
            pass
        _conn = None


def _init_db(c: sqlite3.Connection) -> None:
    c.execute("""
        CREATE TABLE IF NOT EXISTS event_log (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp     TEXT    NOT NULL,
            level         TEXT    NOT NULL,
            subsystem     TEXT,
            event_type    TEXT,
            logger_name   TEXT,
            request_id    TEXT,
            session_id    TEXT,
            message       TEXT    NOT NULL,
            payload_json  TEXT
        );
    """)
    c.execute("CREATE INDEX IF NOT EXISTS idx_event_log_ts        ON event_log(timestamp);")
    c.execute("CREATE INDEX IF NOT EXISTS idx_event_log_level_ts  ON event_log(timestamp) WHERE level IN ('WARNING','ERROR','CRITICAL');")
    c.execute("CREATE INDEX IF NOT EXISTS idx_event_log_session   ON event_log(session_id, timestamp);")
    c.execute("CREATE INDEX IF NOT EXISTS idx_event_log_subsystem ON event_log(subsystem, event_type, timestamp);")


_LOGRECORD_RESERVED = frozenset(logging.makeLogRecord({}).__dict__.keys()) | {"message"}
_CLASSIFICATION_KEYS = frozenset({"subsystem", "event_type", "request_id", "session_id"})


def _extract_payload(record) -> str | None:
    extra = {k: v for k, v in record.__dict__.items()
             if k not in _LOGRECORD_RESERVED and k not in _CLASSIFICATION_KEYS
             and not k.startswith("_")}
    return json.dumps(extra, default=str) if extra else None


def _matches(subsystem: str, event_type: str, allowed: set) -> bool:
    if allowed is None:
        return True
    return any(fnmatch.fnmatch(subsystem, sg) and fnmatch.fnmatch(event_type, tg)
               for (sg, tg) in allowed)


def _now_iso() -> str:
    from proxy.utils import _now_iso
    return _now_iso()


_current_request: contextvars.ContextVar[dict | None] = contextvars.ContextVar("_current_request", default=None)


def set_request_context(request_id: str | None, session_id: str | None) -> None:
    _current_request.set({"request_id": request_id, "session_id": session_id})


def clear_request_context() -> None:
    _current_request.set(None)


def _current_request_id() -> str | None:
    ctx = _current_request.get()
    return ctx["request_id"] if ctx else None


def _current_session_id() -> str | None:
    ctx = _current_request.get()
    return ctx["session_id"] if ctx else None


class SQLiteHandler(logging.Handler):
    def __init__(self, allowed):
        super().__init__()
        self._allowed = allowed

    def emit(self, record: logging.LogRecord) -> None:
        try:
            subsystem = getattr(record, "subsystem", record.name)
            event_type = getattr(record, "event_type", "general")
            if self._allowed is not None and not _matches(subsystem, event_type, self._allowed):
                return
            rid = getattr(record, "request_id", None) or _current_request_id()
            sid = getattr(record, "session_id", None) or _current_session_id()
            _get_conn().execute(
                "INSERT INTO event_log "
                "(timestamp, level, subsystem, event_type, logger_name, request_id, session_id, message, payload_json) "
                "VALUES (?,?,?,?,?,?,?,?,?)",
                (_now_iso(), record.levelname, subsystem, event_type, record.name,
                 rid, sid, self.format(record), _extract_payload(record)),
            )
        except sqlite3.Error:
            self.handleError(record)
        except Exception:
            self.handleError(record)


class BoundedQueueHandler(logging.handlers.QueueHandler):
    def prepare(self, record: logging.LogRecord) -> logging.LogRecord:
        record = super().prepare(record)
        ctx = _current_request.get()
        if ctx:
            if getattr(record, "request_id", None) is None:
                record.request_id = ctx["request_id"]
            if getattr(record, "session_id", None) is None:
                record.session_id = ctx["session_id"]
        return record

    def enqueue(self, record: logging.LogRecord) -> None:
        try:
            self.queue.put_nowait(record)
        except Exception:
            import proxy.state as state
            state.bump_log_overflow()
            try:
                self.queue.put_nowait(
                    logging.makeLogRecord({
                        "name": "proxy.event_log",
                        "levelno": logging.WARNING,
                        "msg": "Log queue overflow: record dropped (queue size=%d)",
                        "args": (self.queue.maxsize,),
                    })
                )
            except Exception:
                pass


def log_message(subsystem: str, event_type: str, severity: str, message: str,
                *, request_id: str | None = None, session_id: str | None = None,
                **fields):
    safe = {k: v for k, v in fields.items() if k not in _LOGRECORD_RESERVED}
    dropped = set(fields) - set(safe)
    if dropped:
        logging.getLogger(__name__).warning(
            "log_message: dropped reserved-key field(s): %s", sorted(dropped))
    extra = {"subsystem": subsystem, "event_type": event_type}
    if request_id is not None:
        extra["request_id"] = request_id
    if session_id is not None:
        extra["session_id"] = session_id
    extra.update(safe)
    logging.getLogger(subsystem).log(getattr(logging, severity.upper()), message, extra=extra)


def parse_log_types(raw: str | None = None) -> set | None:
    if raw is None:
        raw = os.environ.get("PROXY_LOG_TYPES", "*")
    raw = raw.strip()
    if not raw or raw == "*" or raw == "*:*":
        return None
    result = set()
    for pair in raw.split(","):
        pair = pair.strip()
        if not pair:
            continue
        if ":" not in pair:
            result.add((pair, "*"))
        else:
            sub, typ = pair.split(":", 1)
            sub = sub.strip()
            typ = typ.strip()
            if not sub or not typ:
                logging.getLogger(__name__).warning(
                    "PROXY_LOG_TYPES: skipping malformed token %r (empty sub/type)", pair)
                continue
            result.add((sub, typ))
    if not result:
        logging.getLogger(__name__).warning(
            "PROXY_LOG_TYPES=%r parsed to empty set — failing open to all-on", raw)
        return None
    return result


def purge_event_log(retention_days: int = 30, max_rows: int = 2_000_000) -> None:
    logger = logging.getLogger(__name__)
    while True:
        try:
            _time.sleep(86400)
            conn = sqlite3.connect("data/event_log.db", isolation_level=None, timeout=30)
            try:
                conn.execute("PRAGMA busy_timeout=5000")
                conn.execute(
                    "DELETE FROM event_log WHERE timestamp < datetime('now', ?)",
                    (f"-{retention_days} days",),
                )
                conn.execute(
                    "DELETE FROM event_log WHERE id NOT IN "
                    "(SELECT id FROM event_log ORDER BY id DESC LIMIT ?)",
                    (max_rows,),
                )
            finally:
                conn.close()
        except Exception:
            logger.warning("event_log purge failed", exc_info=True)
