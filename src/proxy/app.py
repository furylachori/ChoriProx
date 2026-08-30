"""Flask app — all HTTP route handlers for the Qwen Tool Proxy."""

import atexit
import ipaddress
import os
import json
import logging
import signal
import threading
import time
import hmac
from typing import Optional


from flask import Flask, request, jsonify, Response

from proxy.types import BackendError, InvalidRequestError
import proxy.state as state
from proxy.state import (
    AppState,
)
from proxy.model_registry import (
    MAX_RESPONSE_TOKENS,
    USE_ADAPTERS,
    _model_registry,
    _sanitize_url,
)
from proxy.auth import (
    _sanitize_error_body,
    AuthConfigError,
)
from proxy.streaming import stream_response, make_sse_chunk
from proxy.session import (
    CLEANUP_INTERVAL,
    SESSION_TTL,
)
from proxy.session_tokens import create_session_token
from proxy.admin import admin_bp
from proxy.feature_flags import USE_DISPATCH_EXTRACTION
from proxy.openai_compat import (
    build_upstream_protocol_error,
    chat_completion_shape_error,
)
from proxy.rate_limit_response import rate_limited_response
from proxy.upstream_sse import (
    UpstreamStreamError,
    UpstreamStreamProtocolError,
    UpstreamStreamLimitError,
)
from proxy.pipeline.exceptions import StageAbort
from proxy.pipeline.request_pipeline import RequestPipeline
from proxy.pipeline.stages import parse_request
from proxy.completion_coordinator import (
    AdmissionKind,
    AdmissionResult,
    CompletionArtifact,
    CompletionCapacityError,
    CompletionRecord,
    CompletionState,
    admit,
    shutdown as _shutdown_completion_coordinator,
)
from proxy.owner_setup import owner_setup
from memory.query_executor import close_shared_driver


_logging_config_lock = threading.Lock()


def _safe_configure_logging() -> None:
    """Configure the logging handler bus: QueueHandler → QueueListener → [Stream, RotatedFile, SQLite].

    Safe for:
    - Production Gunicorn/Uvicorn (preserves their configured handlers)
    - Multi-worker/forked environments
    - Multi-threaded imports (double-checked locking)
    """
    import logging.handlers
    import queue
    import os as _os

    root_logger = logging.getLogger()

    if root_logger.handlers:
        return

    with _logging_config_lock:
        if root_logger.handlers:
            return

        from proxy.event_log import (
            SQLiteHandler, BoundedQueueHandler, parse_log_types,
            purge_event_log,
        )

        queue_size = int(_os.environ.get("PROXY_LOG_QUEUE", "8192"))
        log_queue = queue.Queue(maxsize=queue_size)

        json_fmt = _os.environ.get("PROXY_LOG_FORMAT", "text").lower() == "json"
        if json_fmt:
            from proxy.json_formatter import JSONFormatter
            stream_fmt = JSONFormatter()
        else:
            stream_fmt = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

        from proxy.json_formatter import JSONFormatter
        file_fmt = JSONFormatter()

        sqlite_fmt = logging.Formatter("%(message)s")

        stream_h = logging.StreamHandler()
        stream_h.setFormatter(stream_fmt)

        _os.makedirs("data/logs", exist_ok=True)
        file_days = int(_os.environ.get("PROXY_LOG_FILE_DAYS", "14"))
        file_h = logging.handlers.TimedRotatingFileHandler(
            "data/logs/proxy.log", when="midnight",
            backupCount=file_days, utc=True,
        )
        file_h.setFormatter(file_fmt)

        allowed = parse_log_types()
        sqlite_h = SQLiteHandler(allowed)
        sqlite_h.setFormatter(sqlite_fmt)

        root_logger.handlers = []
        queue_handler = BoundedQueueHandler(log_queue)
        root_logger.addHandler(queue_handler)

        _listener = logging.handlers.QueueListener(log_queue, stream_h, file_h, sqlite_h)
        _listener.start()

        _safe_configure_logging._listener = _listener
        _safe_configure_logging._file_h = file_h
        _safe_configure_logging._sqlite_h = sqlite_h

        _log_level = logging.DEBUG if _os.environ.get("PROXY_DEBUG") else logging.INFO
        root_logger.setLevel(_log_level)

        logging.getLogger("openai._base_client").setLevel(logging.WARNING)
        logging.getLogger("httpcore.http11").setLevel(logging.WARNING)
        logging.getLogger("httpx").setLevel(logging.WARNING)

        retention_days = int(_os.environ.get("PROXY_EVENT_LOG_RETENTION_DAYS", "30"))
        purge_thread = threading.Thread(
            target=purge_event_log,
            args=(retention_days, 2_000_000),
            daemon=True,
            name="event_log_purge",
        )
        purge_thread.start()


_safe_configure_logging()

_FEATURES = None

logger = logging.getLogger(__name__)

try:
    from clients.orchestrator_config import ProxyFeatures

    _FEATURES = ProxyFeatures()

    if getattr(_FEATURES, "ticket_commands_enabled", True):
        import commands.task_commands
except ImportError:
    logger.warning("Command system disabled: import failed")
    _FEATURES = None

_PROXY_TRACE = os.environ.get("PROXY_TRACE", "0") != "0"


_DOG_MESSAGES: list[str] = []
# The 15-dog-message rotation table that used to live here has been removed:
# _keepalive_stream no longer yields rotating dog comments. The new contract
# (ticket 199ece93 §4) is fixed ASCII comments — `: connected` once on
# open, `: keepalive` per wait timeout while pending, `: [DONE]` once
# after final response — so the rotating content is dead code.


app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16 MB

app.config["APP_STATE"] = AppState(
    llama_url="http://localhost:7777/v1/chat/completions",
    max_response_tokens=MAX_RESPONSE_TOKENS,
    cleanup_interval=CLEANUP_INTERVAL,
    session_ttl=SESSION_TTL,
    test_mode=False,
)

def _parse_trusted_proxies():
    raw = os.environ.get("PROXY_TRUSTED_PROXIES", "")
    entries = [e.strip() for e in raw.split(",") if e.strip()]
    networks = []
    for entry in entries:
        try:
            networks.append(ipaddress.ip_network(entry, strict=False))
        except ValueError:
            logger.warning("PROXY_TRUSTED_PROXIES: invalid entry %r ignored", entry)
    return networks

TRUSTED_PROXY_NETWORKS = _parse_trusted_proxies()

_DISABLE_LOCALHOST_BYPASS = os.environ.get(
    "PROXY_DISABLE_LOCALHOST_BYPASS", ""
).lower() in ("1", "true", "yes")

_LOCALHOST_IPS = {"127.0.0.1", "::1"}


def _is_local_request():
    """Check if the current request originates from localhost.

    Uses get_client_ip() to respect trusted-proxy validation, and
    normalizes IPv4-mapped IPv6 addresses (::ffff:127.0.0.1 → 127.0.0.1).
    """
    if _DISABLE_LOCALHOST_BYPASS:
        return False
    addr = get_client_ip()
    if not addr or addr == "unknown":
        return False
    if addr.startswith("::ffff:"):
        addr = addr[7:]
    return addr in _LOCALHOST_IPS


def _is_trusted_proxy(ip_str):
    try:
        addr = ipaddress.ip_address(ip_str)
    except ValueError:
        return False
    return any(addr in net for net in TRUSTED_PROXY_NETWORKS)


def get_client_ip():
    remote_addr = request.remote_addr or "unknown"

    if remote_addr in _LOCALHOST_IPS:
        return remote_addr

    forwarded_for = request.headers.get("X-Forwarded-For")

    if not TRUSTED_PROXY_NETWORKS:
        if forwarded_for:
            logger.warning(
                "X-Forwarded-For header present from %s but PROXY_TRUSTED_PROXIES not configured — ignoring header",
                remote_addr,
            )
        return remote_addr

    if not _is_trusted_proxy(remote_addr):
        if forwarded_for:
            logger.warning(
                "X-Forwarded-For from untrusted source %s — ignoring header",
                remote_addr,
            )
        return remote_addr

    chain = [ip.strip() for ip in forwarded_for.split(",") if ip.strip()] if forwarded_for else []
    chain.append(remote_addr)

    for i in range(len(chain) - 2, -1, -1):
        if not _is_trusted_proxy(chain[i]):
            return chain[i]

    return chain[0]





def _cleanup_stale_sessions():
    """Clean up in-memory state from previous runs on startup.

    All in-memory state (agent registry, workspace state, chat histories,
    rate limiters) is empty on fresh start. This function ensures any
    persistent state (SQLite sessions table) is consistent.
    """
    try:
        from memory.db import connection
        with connection(pooled=True) as conn:
            session_count = conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0]
        logger.info("Startup: in-memory state is fresh, %d sessions in SQLite from previous runs", session_count)
    except Exception as e:
        logger.warning("Startup: session cleanup check failed (non-fatal): %s", e)


def _eager_init_registry() -> None:
    """Construct the singleton SkillRegistry eagerly.

    Called explicitly from CLI startup (qwen_proxy.py:__main__) and
    transitively from the WSGI boot path (gunicorn on_starting hook
    → ``initialize_proxy_at_startup`` → ``_eager_init_registry``).
    NOT called at module-import time because test fixtures set
    PROXY_SKILLS_DIR via monkeypatch AFTER importing app.py.

    On SkillRegistryConfigurationError: the singleton is left unconstructed;
    the caller decides whether to exit (CLI) or fail the WSGI worker.

    Idempotent: if the singleton is already constructed, returns without
    re-running populate_builtins.
    """
    from proxy.skills.skill_registry import get_default_registry
    get_default_registry()


def initialize_proxy_at_startup() -> None:
    """Canonical startup sequence for the proxy.

    Performs the ordered steps that every cold start must run, in the
    exact order required for correctness:

      1. ``_eager_init_registry()`` — construct the SkillRegistry
         singleton BEFORE any request path can reference it.
      2. ``memory.sync.init_db()`` — create the SQLite schema
         (rate_limiter_state, sessions, etc.) BEFORE any limiter tries
         to read from it.
      3. ``proxy.state.load_all_rate_limiter_state()`` — hydrate every
         persist=True RateLimiter from SQLite. Step ordering is the
         whole point of this function: previously these limiters were
         constructed at module-import time and tried to read the
         schema before init_db() ran, emitting 7 "no such table:
         rate_limiter_state" warnings on every cold start.
      4. ``_cleanup_stale_sessions()`` — non-fatal in-memory check that
         logs the previous-run session count.

    Idempotent: calling this twice (e.g. from a test that exercises both
    CLI boot and WSGI boot) is safe — each step is a no-op on the
    second pass.

    Failures during steps 2/3 propagate. Step 1 surfaces
    ``SkillRegistryConfigurationError`` to the caller (the CLI converts
    this to a sys.exit(1)). Step 4 swallows its own DB error (it is
    diagnostic only).

    This function is callable from both CLI (qwen_proxy.py) and the
    WSGI boot path. CLI startup in qwen_proxy.py deliberately calls
    ``_eager_init_registry()`` SEPARATELY before this function so it
    can catch ``SkillRegistryConfigurationError`` and exit with code
    1 — the spec requires a specific CLI exit code on skill-config
    failure, which is not the right shape for a generic
    ``initialize_proxy_at_startup()`` body. The WSGI path wires this
    function via the ``on_starting`` hook in ``gunicorn.conf.py`` at
    the repo root (auto-loaded by ``gunicorn qwen_proxy:app``), so a
    production WSGI deploy invokes it exactly once in the master
    process before workers fork.
    """
    _eager_init_registry()
    from memory.sync import init_db
    init_db()
    print("[MEMORY] Database schemas initialized (SQLite + Neo4j)")
    import proxy.state as state
    state.load_all_rate_limiter_state()
    _cleanup_stale_sessions()


from proxy.assertions import assert_session_secret_configured
assert_session_secret_configured()


if not os.environ.get("PROXY_SESSION_SECRET"):
    logger.warning("PROXY_SESSION_SECRET not set — session tokens and canary signing are DISABLED. Set this env var in production.")

if not os.environ.get("PROXY_API_KEY", "").strip():
    logger.critical("PROXY_API_KEY not set — all endpoints except /v1/models will reject requests (fail-closed). Set PROXY_API_KEY for production use.")

def _unwrap_stage_abort(response_data):
    """Normalize StageAbort response_data into a plain dict for SSE streaming.

    Handles three shapes produced by pipeline stages:
      - tuple[dict, int] → extract the dict (status code discarded for streaming)
      - flask.Response     → extract JSON body via get_json()
      - dict               → passthrough as-is
    """
    from flask import Response as FlaskResponse

    if isinstance(response_data, tuple):
        return response_data[0]
    if isinstance(response_data, FlaskResponse):
        if getattr(response_data, 'content_type', '') == 'text/event-stream':
            return response_data  # Streaming SSE — caller must yield from it
        try:
            body = response_data.get_json(silent=True)
            if body is None:
                return {"error": {"message": "Non-JSON dispatch response", "type": "proxy_error"}}
            return body
        except Exception:
            return {"error": {"message": "Non-JSON dispatch response", "type": "proxy_error"}}
    return response_data


def _response_from_stage_abort(response_data):
    """Convert StageAbort response_data into a (body, status_code) pair.

    Handles three shapes produced by pipeline stages:
      - tuple[dict, int] → jsonify(body), status_code
      - flask.Response     → return as-is (carries its own status_code)
      - dict               → extract status from {"error": {"code": N}} if present, else 200
    """
    from flask import Response as FlaskResponse

    if isinstance(response_data, tuple):
        body, status = response_data
        return jsonify(body), status

    if isinstance(response_data, FlaskResponse):
        return response_data

    if isinstance(response_data, dict):
        code = 200
        error_block = response_data.get("error")
        if isinstance(error_block, dict):
            maybe_code = error_block.get("code")
            if isinstance(maybe_code, int) and 100 <= maybe_code <= 599:
                code = maybe_code
        return jsonify(response_data), code

    return jsonify(response_data), 200


def _map_exception_to_envelope(exc):
    """Map an unexpected pipeline exception to a structured SSE error envelope.

    F8 fix (four-day-change-audit-2026-07-17.md, finding F8): the non-streaming
    route maps ``RuntimeError→502``, ``ValueError→400``, ``AuthConfigError→401``,
    ``BackendError→upstream envelope``. The streaming path used to collapse
    every unexpected exception to a single opaque ``"Internal stream error"``
    SSE event. This helper mirrors the non-streaming mapping so the client
    sees a typed, diagnostic envelope. Only unmapped exception types fall
    back to the generic ``proxy_error``.

    ``UpstreamStreamError`` (and its ``UpstreamStreamProtocolError`` /
    ``UpstreamStreamLimitError`` subclasses, raised by ``proxy.upstream_sse``)
    map to a sanitized ``upstream_error`` envelope with HTTP 502 — the
    sanitized ``.code`` is surfaced but the message (which may contain
    provider detail) is replaced with a generic string.
    """
    if isinstance(exc, AuthConfigError):
        return {
            "error": {
                "message": "Model configuration error",
                "type": "auth_error",
                "code": "auth_config_error",
            }
        }, 401
    if isinstance(exc, BackendError):
        status = 502 if exc.status_code >= 500 else exc.status_code
        sanitized = _sanitize_error_body(exc.error_body)
        if isinstance(sanitized, dict) and isinstance(sanitized.get("error"), dict):
            env = dict(sanitized["error"])
        else:
            env = {"message": "Upstream backend error"}
        env["type"] = "upstream_error"
        return {"error": env}, status
    if isinstance(exc, ValueError):
        return {
            "error": {
                "message": "Invalid request parameters",
                "type": "validation_error",
                "code": "invalid_request",
            }
        }, 400
    if isinstance(exc, RuntimeError):
        return {
            "error": {
                "message": "Backend processing error",
                "type": "runtime_error",
                "code": "backend_error",
            }
        }, 502
    if isinstance(exc, UpstreamStreamError):
        # UpstreamStreamError (+ Protocol/Limit subclasses) carry a sanitized
        # machine-readable `.code` but a message that may contain provider
        # detail. Emit a generic message; surface only the sanitized code.
        code = getattr(exc, "code", None)
        if not isinstance(code, str):
            code = "upstream_stream_error"
        return {
            "error": {
                "message": "Upstream streaming error",
                "type": "upstream_error",
                "code": code,
            }
        }, 502
    # Unmapped exception types — last resort.
    return {
        "error": {
            "message": "Internal stream error",
            "type": "proxy_error",
            "code": "internal_error",
        }
    }, 500


def _freeze_stream_identity(data):
    """Stamp stable id/created onto a success response dict BEFORE rendering,
    IN PLACE, so stream_response's int(time.time()) fallback (streaming.py:21,23)
    cannot fire at delivery/replay time. Mutates data so the stamped values are
    captured into the frozen terminal_chunks (rendered exactly once). Real
    upstream responses already carry id/created, so this is a no-op for them."""
    if "id" not in data:
        data["id"] = f"chatcmpl-{int(time.time())}"
    if "created" not in data:
        data["created"] = int(time.time())
    return data


def _materialize_artifact(outcome, ctx):
    """Render the terminal SSE chunks for a completed outcome ONCE, in the
    completion worker thread (CP3-3). Returns an immutable CompletionArtifact.

    Reuses the existing renderers in place. Captures NO Flask request-local
    (no flask.request / current_app / jsonify): uses _unwrap_stage_abort (not
    _response_from_stage_abort), _sse_from_response_data, _map_exception_to_envelope,
    _sanitize_error_body, make_sse_chunk — all request-context-free. Safe to run
    off the request thread.
    """
    request_id = str(getattr(ctx, "request_id", "?"))
    model = (
        getattr(ctx.upstream, "original_model", "unknown")
        if hasattr(ctx, "upstream") and ctx.upstream
        else "unknown"
    )

    if outcome["stage_abort"] is not None:
        data = _unwrap_stage_abort(outcome["stage_abort"])
        chunks = tuple(_sse_from_response_data(data, model))
        return CompletionArtifact(
            terminal_chunks=chunks,
            logical_status="stage_abort",
            original_request_id=request_id,
        )

    if outcome["backend_error"] is not None:
        err = outcome["backend_error"]
        sanitized = _sanitize_error_body(err.error_body)
        if not isinstance(sanitized, dict):
            sanitized = {}
        error_block = sanitized.get("error")
        if isinstance(error_block, dict):
            error_block = dict(error_block)
            error_block["type"] = "upstream_error"
            sanitized["error"] = error_block
        else:
            sanitized["error"] = {
                "message": str(error_block) if error_block is not None else "Upstream error",
                "type": "upstream_error",
            }
        chunks = (make_sse_chunk(sanitized), "data: [DONE]\n\n")
        return CompletionArtifact(
            terminal_chunks=chunks,
            logical_status="backend_error",
            original_request_id=request_id,
        )

    if outcome["error"] is not None:
        mapped, _status = _map_exception_to_envelope(outcome["error"])
        chunks = (make_sse_chunk(mapped), "data: [DONE]\n\n")
        return CompletionArtifact(
            terminal_chunks=chunks,
            logical_status="error",
            original_request_id=request_id,
        )

    data = outcome["result"] if isinstance(outcome["result"], dict) else None
    if isinstance(data, dict):
        _freeze_stream_identity(data)

    # CP4-3: detect tool calls so the replay safety gate can reject them.
    _has_tool_calls = False
    if isinstance(data, dict):
        _choices = data.get("choices") or []
        for _choice in _choices:
            _msg = _choice.get("message") if isinstance(_choice, dict) else None
            if isinstance(_msg, dict):
                if _msg.get("tool_calls") or _msg.get("function_call"):
                    _has_tool_calls = True
                    break
            _fr = _choice.get("finish_reason") if isinstance(_choice, dict) else None
            if _fr in ("tool_calls", "function_call"):
                _has_tool_calls = True
                break

    chunks = tuple(_sse_from_response_data(data, model))
    return CompletionArtifact(
        terminal_chunks=chunks,
        logical_status="ok",
        original_request_id=request_id,
        has_tool_calls=_has_tool_calls,
    )


def _keepalive_stream(ctx, record, *, on_follower_exit=None):
    """SSE generator: bounded completion, full-lifecycle heartbeat.

    Architecture (ticket 199ece93 §4 + F1 bounded admission + CP3-2 subscriber):

      1. A single completion worker runs the entire 9-stage pipeline
         (``pipeline.run(ctx)``) exactly once inside the bounded
         ``completion_coordinator`` pool. The worker is the canonical owner
         of the request lifecycle: it sets the event-log ContextVars (which
         do not propagate to new threads), runs the pipeline, lets
         PostProcessStage finalize the skill lifecycle (which sets
         ``ctx.skill_lifecycle_finalized`` and clears / preserves the active
         mode based on the upstream ``finish_reason``), and publishes a
         captured ``outcome`` dict onto the ``CompletionRecord``.

         Admission control is SYNCHRONOUS in ``proxy_chat_completions``:
         ``completion_coordinator.admit()`` acquires a bounded semaphore
         BEFORE this generator is returned to Flask. On capacity exhaustion
         the route returns 503 — which is only possible because no streaming
         status has been committed yet.

      2. This generator (the WSGI-returned object) yields the SSE artifact:

            - ``": connected\\n\\n"`` immediately,
            - ``": keepalive\\n\\n"`` every ``heartbeat_interval`` seconds while
              the completion worker is still running (covering preflight,
              upstream, tool loops, and post-processing),
            - then the worker-materialized ``record.artifact.terminal_chunks``
              emitted VERBATIM. Delivery renders NOTHING (CP3-3) — the artifact
              already contains exactly one final ``data:`` chunk followed by
              exactly one ``data: [DONE]\\n\\n``.

      3. CP3-2/CP3-3 subscriber contract: delivery is a PURE SUBSCRIBER to the
         per-request ``CompletionRecord``. It waits on ``record.completion_event``
         (never ``future.result()`` / ``cancel()`` / ``join()``) and, once the
         event fires, emits ``record.artifact.terminal_chunks`` verbatim — the
         terminal SSE was already rendered ONCE by the completion worker (with
         frozen id/created). Delivery performs no classification and no
         rendering. Disconnect semantics: ``GeneratorExit`` is logged,
         re-raised, and otherwise IGNORED. The completion worker continues in
         the pool, the record is left intact, and mode-clear is the
         responsibility of PostProcessStage (which has either already run or is
         still running). A downstream disconnect no longer touches mode.
    """
    # Re-import CONFIG so a test that monkeypatches the env var post-import
    # still picks up the configured value.
    from proxy.state import CONFIG

    request_id = getattr(ctx, "request_id", "?")
    session_id = getattr(ctx, "session_id", "?")
    log_extra = {"request_id": request_id, "session_id": session_id}

    from config import _resolve_heartbeat_interval

    streaming_section = (CONFIG or {}).get("streaming", {}) or {}
    heartbeat_interval = _resolve_heartbeat_interval(
        streaming_section.get("heartbeat_interval", 5.0)
    )

    try:
        logger.info("stream_opened", extra=log_extra)
        yield ": connected\n\n"

        # Pure subscriber: wait on the record's completion event. Never
        # touches the worker Future's result/cancel/join — delivery only
        # observes completion via the event (and a non-blocking liveness
        # check for the cancelled-worker backstop below).
        while True:
            if record.completion_event.wait(timeout=heartbeat_interval):
                break
            # Backstop: if the worker Future finished WITHOUT publishing
            # completion (only reachable when shutdown(cancel_futures=True)
            # cancelled a not-yet-started worker), the event will never fire.
            # In HEAD, future.result() raised CancelledError and the stream
            # terminated; preserve that "worker collapsed -> stream ends"
            # behavior with a canonical error instead of looping keepalives
            # forever. `.done()` is a non-blocking status check, not a result
            # consume/cancel/join.
            if record.future is not None and record.future.done() and not record.completion_event.is_set():
                logger.warning("stream worker ended without publishing completion", extra=log_extra)
                yield make_sse_chunk({
                    "error": {
                        "message": "Completion worker unavailable",
                        "type": "proxy_error",
                        "code": "worker_cancelled",
                    }
                })
                yield "data: [DONE]\n\n"
                logger.info("delivery_finished", extra=log_extra)
                return
            yield ": keepalive\n\n"

        # Completion published. Emit the worker-materialized terminal artifact
        # verbatim — delivery renders NOTHING (CP3-3). The artifact already
        # contains exactly one `data: [DONE]`.
        artifact = getattr(record, "artifact", None)
        if artifact is not None:
            yield from artifact.terminal_chunks
            logger.info("delivery_finished", extra=log_extra)
            return

        # Materialization failed in the worker (artifact stayed None). Emit a
        # canonical error + one [DONE] rather than nothing.
        logger.error("no completion artifact; emitting canonical error", extra=log_extra)
        yield make_sse_chunk({
            "error": {
                "message": "Completion rendering failed",
                "type": "proxy_error",
                "code": "materialize_failed",
            }
        })
        yield "data: [DONE]\n\n"
        logger.info("delivery_finished", extra=log_extra)
    except GeneratorExit:
        # Disconnect. The completion thread keeps running. Do NOT clear
        # mode, do NOT join the thread, do NOT cancel upstream — the
        # pipeline's PostProcessStage is still the sole authority on mode
        # lifecycle and the completion thread will reach it.
        logger.info("delivery_detached", extra=log_extra)
        raise
    finally:
        if on_follower_exit is not None:
            on_follower_exit()


def _replay_stream(ctx, artifact):
    """SSE generator for replayed completions (CP4-3).

    Emits the pre-materialized terminal artifact verbatim. No heartbeat
    (artifact is ready), no backstop (no worker), no rendering.
    """
    request_id = getattr(ctx, "request_id", "?")
    session_id = getattr(ctx, "session_id", "?")
    log_extra = {"request_id": request_id, "session_id": session_id}
    try:
        logger.info("completion_replayed", extra=log_extra)
        yield ": connected\n\n"
        if artifact is not None:
            yield from artifact.terminal_chunks
        else:
            yield make_sse_chunk({
                "error": {
                    "message": "Replay artifact unavailable",
                    "type": "proxy_error",
                    "code": "replay_artifact_missing",
                }
            })
            yield "data: [DONE]\n\n"
        logger.info("delivery_finished", extra=log_extra)
    except GeneratorExit:
        logger.info("delivery_detached", extra=log_extra)
        raise


def _sse_from_response_data(data, model_name):
    """Convert a response dict to SSE chunks, handling both success and error shapes.

    Unlike the old _sse_from_dict, this tolerates error-shaped responses
    (StageAbort payloads, malformed completions, etc.) without KeyError.

    US-060a: malformed chat completions (None, {}, [], choices:None,
    choices:[], choices:[None], choices:[{"message": None}], etc.) are
    refused up-front and replaced with the canonical upstream protocol
    error. Explicit upstream error envelopes (``{"error": {...}}`` with
    no ``choices``) are preserved verbatim so the client sees the real
    failure shape.
    """
    from proxy.streaming import make_sse_chunk, stream_response

    # Belt-and-suspenders: if a flask.Response somehow reaches us
    # without going through _unwrap_stage_abort, extract its JSON body
    from flask import Response as FlaskResponse
    if isinstance(data, FlaskResponse):
        try:
            data = data.get_json(silent=True)
            if data is None:
                yield make_sse_chunk({"error": {"message": "Non-JSON response in SSE path", "type": "proxy_error"}})
                yield "data: [DONE]\n\n"
                return
        except Exception:
            yield make_sse_chunk({"error": {"message": "Non-JSON response in SSE path", "type": "proxy_error"}})
            yield "data: [DONE]\n\n"
            return

    # US-060a: explicit upstream error envelopes are preserved verbatim.
    if (
        isinstance(data, dict)
        and isinstance(data.get("error"), dict)
        and not data.get("choices")
    ):
        yield make_sse_chunk(data)
        yield "data: [DONE]\n\n"
        return

    # US-060a: refuse to emit any malformed chat completion body. None,
    # {}, [], and structurally broken choices all land here. The
    # canonical protocol error is the only thing allowed on the wire
    # for a malformed upstream response.
    shape_error = chat_completion_shape_error(data)
    if shape_error is not None:
        logger.error(
            "Refusing to emit malformed chat completion over SSE: %s",
            shape_error,
        )
        yield make_sse_chunk(build_upstream_protocol_error())
        yield "data: [DONE]\n\n"
        return

    # US-060a: direct indexing is safe — chat_completion_shape_error has
    # already confirmed choices is a non-empty list with a dict message.
    choices = data["choices"]
    message = choices[0]["message"]
    yield from stream_response(message, data, model_name)


def _require_api_key(f):
    """Decorator: reject requests without a valid PROXY_API_KEY.

    Fail-closed: when PROXY_API_KEY is not set, returns 500 instead of
    allowing unauthenticated access.

    Localhost bypass: connections from 127.0.0.1 or ::1 are allowed without
    authentication, since this is a local development proxy.
    """
    from functools import wraps

    @wraps(f)
    def decorated(*args, **kwargs):
        if _is_local_request():
            return f(*args, **kwargs)

        expected = os.environ.get("PROXY_API_KEY", "").strip()
        if not expected:
            logger.critical("PROXY_API_KEY not set — rejecting request to %s from %s (fail-closed)",
                          request.path, request.remote_addr)
            return jsonify({"error": "Server authentication not configured", "type": "auth_config_error"}), 500

        provided = None
        auth_header = request.headers.get("Authorization", "")
        if auth_header.startswith("Bearer "):
            provided = auth_header[7:].strip()
        if not provided:
            provided = request.headers.get("X-API-Key", "").strip()

        if not provided:
            logger.warning("Auth failed: no credentials provided for %s from %s",
                         request.path, request.remote_addr)
            return jsonify({"error": "Unauthorized", "type": "auth_error"}), 401

        if not hmac.compare_digest(provided, expected):
            logger.warning("Auth failed: invalid key for %s from %s (provided=%s... expected=%s...)",
                         request.path, request.remote_addr,
                         provided[:6] if provided else "none",
                         expected[:6] if expected else "none")
            return jsonify({"error": "Unauthorized", "type": "auth_error"}), 401

        return f(*args, **kwargs)

    return decorated


_FAIL_CLOSED_LOGGED = False


@app.before_request
def _enforce_fail_closed():
    """Fail-closed: when PROXY_API_KEY is not set, reject all endpoints except /v1/models, /v2/models, and /health.

    Per US-051, /v1/models must remain accessible without authentication (US-047).
    Localhost connections are always allowed (local development proxy).
    All other endpoints return 500 to prevent unauthenticated access.
    """
    global _FAIL_CLOSED_LOGGED
    expected = os.environ.get("PROXY_API_KEY", "").strip()
    if expected:
        return None
    if _is_local_request():
        return None
    if request.path in ("/v1/models", "/v2/models", "/health"):
        return None
    if not _FAIL_CLOSED_LOGGED:
        logger.critical("PROXY_API_KEY not set — all endpoints except /v1/models and /health will reject non-localhost requests")
        _FAIL_CLOSED_LOGGED = True
    return jsonify({"error": "Server authentication not configured", "type": "auth_config_error"}), 500


@app.route("/v1/chat/completions", methods=["POST"])
@app.route("/v2/chat/completions", methods=["POST"])
@_require_api_key
def proxy_chat_completions():
    try:
        from proxy.state import chat_rate_limiter
        
        rl_result = chat_rate_limiter.check(get_client_ip())
        if not rl_result.allowed:
            logger.warning("[?] rate limited: addr=%s retry_after=%s", request.remote_addr, int(rl_result.retry_after))
            body, status = rate_limited_response(int(rl_result.retry_after))
            return jsonify(body), status

        from proxy.agent.agent_registry import registry as agent_registry

        ctx = parse_request(request)
        _incoming_msg_count = len(ctx.body.get("messages", []))
        _incoming_chars = sum(
            len(m.get("content", "")) if isinstance(m.get("content", ""), str) else 0
            for m in ctx.body.get("messages", [])
        )
        logger.info(
            "[%s] request received: %d messages, %d chars (%.1f KB) for session %s model=%s",
            getattr(ctx, 'request_id', '?'),
            _incoming_msg_count,
            _incoming_chars,
            _incoming_chars / 1024,
            ctx.session_id,
            ctx.model_name,
            extra={"request_id": getattr(ctx, 'request_id', '?'), "session_id": ctx.session_id},
        )

        try:
            from proxy.http_client import dump_inbound_request_body
            dump_inbound_request_body(
                session_id=ctx.session_id,
                request_id=getattr(ctx, "request_id", None),
                body=ctx.body,
            )
        except Exception as _dump_exc:
            logger.warning("Inbound request dump failed: %s", _dump_exc)

        _incoming_tools = ctx.body.get("tools", [])
        _incoming_tool_names = [
            t.get("function", {}).get("name", t.get("name", "?"))
            for t in (_incoming_tools or [])
        ]
        logger.debug(
            "Incoming tools: is_subagent=%s incoming=%d tools: %s",
            ctx.is_subagent, len(_incoming_tools), _incoming_tool_names,
        )

        # ── CP4-2/CP4-3/CP4-4: transparent dedup (streaming + non-streaming) ──
        # Compute the request fingerprint and atomically look up or insert
        # into the dedup store. OWNER: admit + run pipeline. FOLLOWER: attach
        # to the owner's record, wait on the same event, get the same artifact.
        # REPLAY (CP4-3): COMPLETED + safe → serve cached artifact directly.
        # CONFLICT (CP4-4): PENDING entry with mismatched stream mode → 409.
        _dedup_fingerprint: Optional[str] = None
        _dedup_admission: Optional[AdmissionResult] = None  # set for FOLLOWER/REPLAY path
        _dedup_record: Optional[CompletionRecord] = None  # set for OWNER path
        _dedup_on_completed = None  # set for OWNER streaming path
        _dedup_kind = None
        _dedup_entry = None
        try:
            from proxy.dedup import compute_fingerprint, FollowerCapExceeded
            import proxy.state as _state

            # Include the active mode name in the fingerprint to prevent
            # mode collisions (same body, different mode → different response).
            _mode_name = ""
            try:
                from proxy.agent.agent_registry import registry as _agent_registry
                _agent = _agent_registry.get_or_load(ctx.session_id)
                if _agent is not None:
                    _mode_name = _agent.get_mode_name() or ""
            except Exception:
                pass

            _dedup_fingerprint = compute_fingerprint(ctx, mode_name=_mode_name)

            def _make_record():
                return CompletionRecord(owner_request_id=str(getattr(ctx, "request_id", "?")))

            try:
                _dedup_entry, _dedup_kind = _state.dedup_store.lookup_or_insert(
                    _dedup_fingerprint, _make_record, is_stream=ctx.wants_stream,
                )
            except FollowerCapExceeded:
                logger.warning(
                    "dedup follower cap exceeded: fp=%s",
                    _dedup_fingerprint[:8] if _dedup_fingerprint else "?",
                    extra={"request_id": getattr(ctx, "request_id", "?")},
                )
                return jsonify({
                    "error": {
                        "message": "Too many in-flight requests for this fingerprint",
                        "type": "rate_limit",
                        "code": "dedup_follower_cap",
                    }
                }), 503, {"Retry-After": "1"}

            if _dedup_kind == AdmissionKind.OWNER:
                # OWNER path: use the record from lookup_or_insert.
                _dedup_record = _dedup_entry.record
                _dedup_record.fingerprint = _dedup_fingerprint
                _fp_for_hook = _dedup_fingerprint

                if ctx.wants_stream:
                    def _dedup_on_completed():
                        _art = _dedup_record.artifact
                        if _art is not None and getattr(_art, "logical_status", None) == "ok":
                            _state.dedup_store.mark_completed(_fp_for_hook, _art)
                        else:
                            _state.dedup_store.mark_failed(_fp_for_hook)
                # Non-stream on_completed is handled inline (synchronous path).

                logger.debug(
                    "dedup owner: fp=%s stream=%s",
                    _dedup_fingerprint[:8] if _dedup_fingerprint else "?",
                    ctx.wants_stream,
                    extra={"request_id": getattr(ctx, "request_id", "?")},
                )
            elif _dedup_kind == AdmissionKind.FOLLOWER:
                # FOLLOWER path: attach to the owner's record.
                _dedup_admission = AdmissionResult(
                    kind=AdmissionKind.FOLLOWER,
                    future=_dedup_entry.record.future if _dedup_entry.record else None,
                    record=_dedup_entry.record,
                    fingerprint=_dedup_fingerprint,
                    original_request_id=_dedup_entry.record.owner_request_id if _dedup_entry.record else None,
                )
                logger.info(
                    "completion_follower_attached: fp=%s owner=%s",
                    _dedup_fingerprint[:8] if _dedup_fingerprint else "?",
                    _dedup_admission.original_request_id,
                    extra={"request_id": getattr(ctx, "request_id", "?"), "session_id": getattr(ctx, "session_id", "?")},
                )
            elif _dedup_kind == AdmissionKind.REPLAY:
                # REPLAY path (CP4-3): serve cached artifact directly.
                _dedup_admission = AdmissionResult(
                    kind=AdmissionKind.REPLAY,
                    record=_dedup_entry.record,
                    fingerprint=_dedup_fingerprint,
                    original_request_id=_dedup_entry.record.owner_request_id if _dedup_entry.record else None,
                )
                logger.info(
                    "completion_replayed: fp=%s owner=%s",
                    _dedup_fingerprint[:8] if _dedup_fingerprint else "?",
                    _dedup_admission.original_request_id,
                    extra={"request_id": getattr(ctx, "request_id", "?"), "session_id": getattr(ctx, "session_id", "?")},
                )
            elif _dedup_kind == AdmissionKind.CONFLICT:
                # CONFLICT path (CP4-4): stream mode mismatch for idempotent request.
                logger.warning(
                    "completion_conflict: fp=%s entry_stream=%s request_stream=%s",
                    _dedup_fingerprint[:8] if _dedup_fingerprint else "?",
                    _dedup_entry.is_stream if _dedup_entry else "?",
                    ctx.wants_stream,
                    extra={"request_id": getattr(ctx, "request_id", "?"), "session_id": getattr(ctx, "session_id", "?")},
                )
                return jsonify({
                    "error": {
                        "message": "Stream mode conflict for idempotent request",
                        "type": "conflict",
                        "code": "stream_conflict",
                    }
                }), 409
        except Exception:  # noqa: BLE001 — dedup must never break the request
            logger.debug(
                "dedup fingerprint computation failed; proceeding without dedup",
                extra={"request_id": getattr(ctx, "request_id", "?")},
                exc_info=True,
            )

        pipeline = RequestPipeline()

        # ── Streaming path with bounded admission + keepalive ──────
        if ctx.wants_stream:
            # CP4-3: REPLAY path — emit cached artifact directly.
            if _dedup_admission is not None and _dedup_admission.kind == AdmissionKind.REPLAY:
                admission = _dedup_admission
                final_response = _replay_stream(
                    ctx,
                    admission.record.artifact if admission.record else None,
                )
            # CP4-2: FOLLOWER path skips admit() entirely — the owner's
            # record is already in-flight; we just wait on the same event.
            elif _dedup_admission is not None:
                admission = _dedup_admission
                if _dedup_admission.kind == AdmissionKind.FOLLOWER:
                    _clean_fp = _dedup_fingerprint
                    _released = False
                    def _release_follower():
                        nonlocal _released
                        if _released or _clean_fp is None:
                            return
                        _released = True
                        try:
                            _state.dedup_store.release_follower(_clean_fp)
                        except Exception:
                            logger.debug(
                                "dedup release_follower failed",
                                exc_info=True,
                                extra={"request_id": getattr(ctx, "request_id", "?")},
                            )
                    final_response = _keepalive_stream(
                        ctx, admission.record, on_follower_exit=_release_follower,
                    )
                else:
                    final_response = _keepalive_stream(ctx, admission.record)
            else:
                # OWNER path: admit + run pipeline in completion worker.
                try:
                    admission = admit(
                        pipeline, ctx,
                        on_owner_start=lambda: owner_setup(ctx, pipeline, _FEATURES),
                        materialize=_materialize_artifact,
                        on_completed=_dedup_on_completed,
                        record=_dedup_record,
                    )
                except CompletionCapacityError:
                    # F1: admission control is SYNCHRONOUS — we reject BEFORE
                    # any streaming status is committed, so 503 is valid.
                    # CP4-2: clean up the orphaned PENDING dedup entry so
                    # followers don't attach to a record that will never complete.
                    if _dedup_fingerprint is not None:
                        try:
                            _state.dedup_store.mark_failed(_dedup_fingerprint)
                        except Exception:  # noqa: BLE001
                            pass
                    logger.warning(
                        "completion_capacity_rejected",
                        extra={"request_id": getattr(ctx, 'request_id', '?'), "session_id": ctx.session_id},
                    )
                    return jsonify({
                        "error": {
                            "message": "Proxy at capacity — too many in-flight completions",
                            "type": "rate_limit",
                            "code": "over_capacity",
                        }
                    }), 503, {"Retry-After": "1"}
                final_response = _keepalive_stream(ctx, admission.record)
            # StageAbort is caught inside _keepalive_stream and streamed as SSE.
            # Generator construction is deferred to Flask iteration, so an outer
            # try/except here would be dead code.

            _stream_headers = {
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            }
            if admission.kind == AdmissionKind.REPLAY:
                _stream_headers["X-ChoriProx-Replay"] = "hit"
            if os.environ.get("PROXY_SESSION_SECRET"):
                _session_token_value = create_session_token(client_id=ctx.client_hint, session_id=ctx.session_id)
                if _session_token_value:
                    _stream_headers["X-Session-Token"] = _session_token_value
            # Lifecycle log: "stream delivery finished" is emitted by the
            # generator itself once the final chunks have been yielded. We
            # do NOT log "response complete" here — the previous log fired
            # before any delivery, which was misleading. (ticket 199ece93 §4)
            return Response(
                final_response,
                content_type="text/event-stream",
                headers=_stream_headers,
            )

        # ── Non-streaming path ─────────────────────────────────────
        # CP4-4: non-stream dedup (OWNER/FOLLOWER/REPLAY/CONFLICT).
        if _dedup_admission is not None and _dedup_admission.kind == AdmissionKind.REPLAY:
            # REPLAY: return cached body directly.
            _replay_body = _dedup_admission.record.artifact.body if (_dedup_admission.record and _dedup_admission.record.artifact) else None
            if _replay_body is not None:
                _resp = jsonify(_replay_body)
                _resp.headers["X-ChoriProx-Replay"] = "hit"
                if os.environ.get("PROXY_SESSION_SECRET"):
                    _token = create_session_token(client_id=ctx.client_hint, session_id=ctx.session_id)
                    if _token:
                        _resp.headers["X-Session-Token"] = _token
                return _resp
            # Artifact missing — fall through to OWNER path.

        if _dedup_admission is not None and _dedup_admission.kind == AdmissionKind.FOLLOWER:
            # FOLLOWER: block on the owner's completion_event, then read body.
            import proxy.state as _state
            _record = _dedup_admission.record
            if _record is not None:
                _follower_timeout = float(os.environ.get("PROXY_DEDUP_NONSTREAM_FOLLOWER_TIMEOUT", "60"))
                _event_fired = _record.completion_event.wait(timeout=_follower_timeout)
                if _event_fired and _record.artifact is not None and getattr(_record.artifact, "body", None) is not None:
                    _resp = jsonify(_record.artifact.body)
                    if os.environ.get("PROXY_SESSION_SECRET"):
                        _token = create_session_token(client_id=ctx.client_hint, session_id=ctx.session_id)
                        if _token:
                            _resp.headers["X-Session-Token"] = _token
                    return _resp
                # Timeout or no artifact — release follower slot to free the cap,
                # then return 504.
                if _dedup_fingerprint is not None:
                    try:
                        _state.dedup_store.release_follower(_dedup_fingerprint)
                    except Exception:
                        logger.debug(
                            "dedup release_follower failed",
                            exc_info=True,
                            extra={"request_id": getattr(ctx, "request_id", "?")},
                        )
                logger.warning("dedup follower timeout: fp=%s", _dedup_fingerprint[:8] if _dedup_fingerprint else "?",
                               extra={"request_id": getattr(ctx, "request_id", "?")})
                return jsonify({
                    "error": {
                        "message": "Completion owner did not finish in time",
                        "type": "timeout",
                        "code": "dedup_follower_timeout",
                    }
                }), 504, {"Retry-After": "5"}

        # OWNER path (or dedup disabled / fallback):
        # CP4-4: completion_event.set() MUST fire in all cases (success,
        # StageAbort, other exceptions) so followers never hang. The artifact
        # is stored on the record BEFORE the event fires; mark_completed/
        # mark_failed on the store happens after.
        _stage_abort_response = None
        try:
            owner_setup(ctx, pipeline, _FEATURES)
            final_response = pipeline.run(ctx)
            logger.info("[%s] dispatch complete", getattr(ctx, 'request_id', '?'),
                        extra={"request_id": getattr(ctx, 'request_id', '?'), "session_id": ctx.session_id})
            # Build and store the artifact on the record (before event fires).
            if _dedup_fingerprint is not None and _dedup_record is not None:
                _has_tc = False
                if isinstance(final_response, dict):
                    for _ch in (final_response.get("choices") or []):
                        _msg = _ch.get("message") if isinstance(_ch, dict) else None
                        if isinstance(_msg, dict) and (_msg.get("tool_calls") or _msg.get("function_call")):
                            _has_tc = True
                            break
                        if isinstance(_ch, dict) and _ch.get("finish_reason") in ("tool_calls", "function_call"):
                            _has_tc = True
                            break
                _ns_artifact = CompletionArtifact(
                    terminal_chunks=(),
                    logical_status="ok",
                    original_request_id=str(getattr(ctx, "request_id", "?")),
                    has_tool_calls=_has_tc,
                    body=final_response if isinstance(final_response, dict) else None,
                )
                _dedup_record.artifact = _ns_artifact
                _dedup_record.state = CompletionState.COMPLETED
                _dedup_record.completed_at = time.time()
        except StageAbort as e:
            # CP4-4: mark_failed so commands/errors are not cached.
            if _dedup_fingerprint is not None:
                try:
                    import proxy.state as _state
                    _state.dedup_store.mark_failed(_dedup_fingerprint)
                except Exception:
                    pass
            logger.info("[%s] dispatch aborted", getattr(ctx, 'request_id', '?'))
            _stage_abort_response = _response_from_stage_abort(e.response_data)
        except Exception:
            # CP4-4: mark_failed on any pipeline failure so the entry is not
            # left PENDING (which would absorb followers for 600s).
            if _dedup_fingerprint is not None:
                try:
                    import proxy.state as _state
                    _state.dedup_store.mark_failed(_dedup_fingerprint)
                except Exception:
                    pass
            raise
        finally:
            # ALWAYS signal followers, even on failure paths.
            if _dedup_record is not None:
                _dedup_record.completion_event.set()

        if _stage_abort_response is not None:
            return _stage_abort_response

        # CP4-4: update the dedup store entry (after event is set — the record
        # already carries the artifact for followers; this updates the store
        # entry for future REPLAY lookups).
        if _dedup_fingerprint is not None and _dedup_record is not None:
            try:
                import proxy.state as _state
                _art = _dedup_record.artifact
                if _art is not None and getattr(_art, "has_tool_calls", False):
                    _state.dedup_store.mark_failed(_dedup_fingerprint)
                elif _art is not None:
                    _state.dedup_store.mark_completed(_dedup_fingerprint, _art)
            except Exception:
                logger.debug("dedup non-stream store update failed", exc_info=True,
                             extra={"request_id": getattr(ctx, "request_id", "?")})

        if _PROXY_TRACE:
            upstream = ctx.upstream
            logger.debug(
                json.dumps(
                    {
                        "event": "proxy_trace_request",
                        "url": upstream.url,
                        "body": upstream.body,
                        "headers": {
                            k: v
                            for k, v in upstream.headers.items()
                            if k.lower()
                            not in (
                                "authorization",
                                "x-api-key",
                                "api-key",
                            )
                        },
                    }
                )
            )

        logger.info("[%s] response complete: stream=False", getattr(ctx, 'request_id', '?'),
                    extra={"request_id": getattr(ctx, 'request_id', '?'), "session_id": ctx.session_id})
        _response = jsonify(final_response)
        if os.environ.get("PROXY_SESSION_SECRET"):
            _session_token_value = create_session_token(client_id=ctx.client_hint, session_id=ctx.session_id)
            if _session_token_value:
                _response.headers["X-Session-Token"] = _session_token_value
        return _response

    except InvalidRequestError as exc:
        _rid = getattr(ctx, 'request_id', '?') if 'ctx' in locals() else '?'
        _sid = getattr(ctx, 'session_id', '?') if 'ctx' in locals() else '?'
        logger.warning("[%s] invalid request: %s", _rid, exc.message,
                       extra={"request_id": _rid, "session_id": _sid})
        return jsonify({
            "error": {
                "message": exc.message,
                "type": "invalid_request_error",
                "param": exc.param,
                "code": exc.code,
            }
        }), 400
    except ValueError as e:
        _rid = getattr(ctx, 'request_id', '?') if 'ctx' in locals() else '?'
        _sid = getattr(ctx, 'session_id', '?') if 'ctx' in locals() else '?'
        logger.warning("[%s] validation error: %s", _rid, e,
                       extra={"request_id": _rid, "session_id": _sid})
        return jsonify({"error": "Invalid request parameters", "type": "validation_error"}), 400
    except StageAbort as e:
        _rid = getattr(ctx, 'request_id', '?') if 'ctx' in locals() else '?'
        logger.warning("[%s] request aborted during parsing", _rid,
                       extra={"request_id": _rid})
        return e.response_data
    except AuthConfigError as e:
        _rid = getattr(ctx, 'request_id', '?') if 'ctx' in locals() else '?'
        _sid = getattr(ctx, 'session_id', '?') if 'ctx' in locals() else '?'
        logger.error("[%s] auth config error: %s", _rid, e,
                     extra={"request_id": _rid, "session_id": _sid})
        return jsonify({"error": "Model configuration error", "type": "auth_error"}), 401
    except BackendError as e:
        sanitized = _sanitize_error_body(e.error_body)
        status = 502 if e.status_code >= 500 else e.status_code
        return jsonify(sanitized), status
    except RuntimeError as e:
        _rid = getattr(ctx, 'request_id', '?') if 'ctx' in locals() else '?'
        _sid = getattr(ctx, 'session_id', '?') if 'ctx' in locals() else '?'
        logger.exception("[%s] runtime error in proxy pipeline", _rid,
                         extra={"request_id": _rid, "session_id": _sid})
        return jsonify({"error": "Backend processing error", "type": "runtime_error"}), 502
    except Exception as e:
        _rid = getattr(ctx, 'request_id', '?') if 'ctx' in locals() else '?'
        _sid = getattr(ctx, 'session_id', '?') if 'ctx' in locals() else '?'
        logger.exception("[%s] internal proxy error", _rid,
                         extra={"request_id": _rid, "session_id": _sid})
        return jsonify({"error": "Internal proxy error"}), 500


@app.after_request
def add_v1_deprecation_headers(response):
    if request.path.startswith("/v1/"):
        response.headers["Deprecation"] = "true"
        response.headers["Sunset"] = "Sat, 01 Nov 2026 00:00:00 GMT"
        response.headers["Link"] = f"</v2{request.path[3:]}>; rel=\"successor-version\""
    from proxy.metrics import record_http_request
    duration = time.time() - getattr(request, '_start_time', time.time())
    record_http_request(request.method, request.path, response.status_code, duration)
    return response


@app.route('/metrics')
def metrics():
    from proxy.state import admin_rate_limiter
    client_ip = get_client_ip()
    rl_result = admin_rate_limiter.check(client_ip)
    if not rl_result.allowed:
        body, status = rate_limited_response(int(rl_result.retry_after))
        return jsonify(body), status
    from proxy.metrics import generate_metrics
    result = generate_metrics()
    if isinstance(result, tuple):
        body, status, headers = result
        return body, status, headers
    return result


@app.before_request
def _record_request_start():
    request._start_time = time.time()


app.register_blueprint(admin_bp)


_shutdown_in_progress = False


def _graceful_shutdown():
    """Comprehensive shutdown: drain requests, close pools, stop threads.

    Registered via atexit to ensure clean shutdown on process exit.
    Idempotent - safe to call multiple times.
    """
    global _shutdown_in_progress
    if _shutdown_in_progress:
        return
    _shutdown_in_progress = True

    deadline = time.monotonic() + 30.0

    def remaining():
        return max(0.0, deadline - time.monotonic())

    logger.info("Graceful shutdown initiated")

    try:
        completions_drained = _shutdown_completion_coordinator(
            grace_period=remaining()
        )
    except Exception as exc:
        logger.warning("Completion coordinator shutdown failed: %s", exc)
        return

    if not completions_drained:
        logger.warning(
            "Graceful shutdown stopped: completion workers still active"
        )
        return

    try:
        from proxy.agent.agent_registry import registry

        while True:
            with registry._lock:
                in_flight = sum(registry._in_flight_requests.values())

            if in_flight == 0:
                break

            wait_for = remaining()
            if wait_for <= 0:
                logger.warning(
                    "Graceful shutdown stopped: %d requests still in-flight",
                    in_flight,
                )
                return

            logger.info("Waiting for %d in-flight requests...", in_flight)
            time.sleep(min(0.5, wait_for))
    except Exception as exc:
        logger.warning("In-flight request drain failed: %s", exc)
        return

    executor_budget = remaining()
    if executor_budget <= 0:
        logger.warning(
            "Graceful shutdown stopped: deadline expired before memory drain"
        )
        return

    try:
        from memory.executors import _shutdown_thread_pool
        _shutdown_thread_pool(timeout=executor_budget / 4.0)
    except Exception as exc:
        logger.warning("Thread pool shutdown failed: %s", exc)
        return

    if remaining() <= 0:
        logger.warning(
            "Graceful shutdown stopped: memory drain exhausted deadline"
        )
        return

    try:
        from memory.db import close_all_connections
        close_all_connections()
    except Exception as exc:
        logger.warning("SQLite connection close failed: %s", exc)
        return

    logger.info("Graceful shutdown complete")


atexit.register(close_shared_driver)


def _shutdown_logging():
    try:
        _safe_configure_logging._listener.stop()
    except Exception:
        pass
    try:
        _safe_configure_logging._file_h.close()
    except Exception:
        pass
    try:
        _safe_configure_logging._sqlite_h.close()
    except Exception:
        pass
    try:
        from proxy.event_log import close_conn
        close_conn()
    except Exception:
        pass


atexit.register(_shutdown_logging)
atexit.register(_graceful_shutdown)


@app.teardown_request
def _clear_request_context(_exception=None):
    from proxy.event_log import clear_request_context
    clear_request_context()


if __name__ == "__main__":
    print("=" * 60)
    print("Qwen Tool Proxy for Kilocode")
    print("=" * 60)
    print(f"  Listening on    : http://0.0.0.0:6666")
    if USE_ADAPTERS:
        reg = _model_registry
        if reg:
            if reg.list_models():
                print(f"  Default backend : {_sanitize_url(reg.get_default().backend)}")
                print(f"  Registered models:")
                for name, cfg in reg.list_models():
                    print(
                        f"    {name} → {_sanitize_url(cfg.backend)} (parser: {cfg.tool_parser})"
                    )
                if USE_ADAPTERS and reg:
                    _missing_keys = []
                    for _alias, _cfg in reg.list_models():
                        if (
                            _cfg.api_key_env
                            and not os.environ.get(_cfg.api_key_env, "").strip()
                        ):
                            _missing_keys.append(_alias)
                    if _missing_keys:
                        print(
                            f"[WARN] {len(_missing_keys)} models configured but missing API keys: {', '.join(_missing_keys)}"
                        )
            else:
                logger.warning(
                    "No models registered; proxy_models.json is empty or missing"
                )
    print(f"  Streaming       : kilocode streams -> proxy buffers -> streams back")
    print("=" * 60)
    if not _DISABLE_LOCALHOST_BYPASS and os.environ.get("PROXY_API_KEY"):
        logger.warning(
            "Localhost auth bypass is ACTIVE. Set PROXY_DISABLE_LOCALHOST_BYPASS=1 "
            "if deploying behind a reverse proxy to prevent unauthenticated access."
        )
    try:
        initialize_proxy_at_startup()
    except Exception as e:
        print(f"[MEMORY] Database initialization failed: {e}")
