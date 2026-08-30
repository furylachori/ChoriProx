"""JSON log formatter for structured logging pipelines."""
import json
import logging
import time


class JSONFormatter(logging.Formatter):
    """Formats log records as single-line JSON objects.

    Includes timestamp, level, logger, message, module, function, line.
    Optionally includes request_id and session_id from record extras.
    """

    _STANDARD_FIELDS = {
        'name', 'msg', 'args', 'created', 'filename', 'funcName',
        'levelname', 'levelno', 'lineno', 'module', 'msecs',
        'message', 'pathname', 'process', 'processName', 'relativeCreated',
        'thread', 'threadName', 'exc_info', 'exc_text', 'stack_info',
    }

    def format(self, record):
        log_entry = {
            "timestamp": time.strftime(
                "%Y-%m-%dT%H:%M:%S", time.gmtime(record.created)
            ) + f".{int(record.msecs):03d}Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }

        request_id = getattr(record, 'request_id', None)
        if request_id:
            log_entry["request_id"] = request_id

        session_id = getattr(record, 'session_id', None)
        if session_id:
            log_entry["session_id"] = session_id

        extra = {}
        for key, value in record.__dict__.items():
            if key not in self._STANDARD_FIELDS and not key.startswith('_'):
                extra[key] = value
        if extra:
            log_entry["extra"] = extra

        if record.exc_info and not record.exc_text:
            record.exc_text = self.formatException(record.exc_info)
        if record.exc_text:
            log_entry["exception"] = record.exc_text

        return json.dumps(log_entry, default=str, ensure_ascii=False)
