"""HTTP session management and upstream calls."""

import hashlib
import json
import logging
import os
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

import requests

from proxy.openai_compat import (
    build_upstream_protocol_error,
    chat_completion_shape_error,
)
from proxy.state import _HTTP_SESSION, CONFIG
from proxy.utils import _now_iso
from proxy.types import BackendError
from proxy.upstream_sse import OpenAIStreamAccumulator, UpstreamStreamError

_REPO_ROOT = Path(__file__).resolve().parents[2]

logger = logging.getLogger(__name__)

_RETRYABLE = (
    requests.exceptions.ConnectionError,
    requests.exceptions.ChunkedEncodingError,
    requests.exceptions.Timeout,
)

_RETRYABLE_STATUS = {502, 503, 504, 520, 524}

_DEBUG_5XX = os.environ.get("PROXY_DEBUG_5XX", "0") == "1"
_DEBUG_5XX_DIR = os.environ.get("PROXY_DEBUG_5XX_DIR", "data/debug_5xx")

_endpoint_semaphores: dict = {}
_endpoint_sem_lock = threading.Lock()


def _estimate_body_kb(body: dict) -> float:
    """Estimate serialized body size in KB without full JSON serialization."""
    total = 64
    for msg in body.get("messages", []):
        total += 30
        total += len(msg.get("role", ""))
        content = msg.get("content", "")
        if isinstance(content, str):
            total += len(content)
        elif isinstance(content, list):
            for part in content:
                if isinstance(part, dict):
                    total += len(part.get("text", "")) + 40
        for tc in (msg.get("tool_calls") or []):
            func = tc.get("function", {})
            total += len(func.get("name", "")) + len(func.get("arguments", "")) + 60
        for field in ("reasoning_content", "reasoning"):
            val = msg.get(field)
            if isinstance(val, str):
                total += len(val)
    for key in ("tools", "functions"):
        val = body.get(key)
        if val:
            total += len(json.dumps(val, default=str))
    return total / 1024


def _resolve_upstream_timeout(body: dict) -> int:
    """Resolve the upstream read timeout from config, scaling for complex prompts.

    Config keys (in ``proxy_config.json`` under ``upstream``):
        timeout              - base timeout in seconds (default 1800 = 30 min)
        timeout_multiplier   - multiplier applied when prompt exceeds thresholds

    Scaling triggers when the estimated body > 40 KB or message count > 20.
    """
    cfg = CONFIG.get("upstream", {})
    raw_timeout = cfg.get("timeout")
    if raw_timeout is None or not isinstance(raw_timeout, (int, float)):
        base = 1800
    else:
        base = int(raw_timeout)
    mult = float(cfg.get("timeout_multiplier", 2.0))

    msg_count = len(body.get("messages", []))
    body_kb = _estimate_body_kb(body)

    if body_kb > 40 or msg_count > 20:
        scaled = min(int(base * mult), 1800)
        logger.debug(
            "Large upstream request (%d msgs, %.1f KB) — scaled timeout %d -> %ds",
            msg_count,
            body_kb,
            base,
            scaled,
        )
        return scaled

    return base


def _retry_config() -> dict:
    """Read retry settings from config, returning sensible defaults."""
    cfg = CONFIG.get("upstream", {})
    return {
        "max_attempts": int(cfg.get("retry_max_attempts", 3)),
        "base_delay": float(cfg.get("retry_base_delay", 2.0)),
        "max_delay": float(cfg.get("retry_max_delay", 30.0)),
    }


def _get_endpoint_semaphore(url: str) -> Optional[threading.Semaphore]:
    parsed = urlparse(url)
    endpoint_key = f"{parsed.scheme}://{parsed.netloc}"
    ec_map = CONFIG.get("upstream", {}).get("endpoint_concurrency", {})
    if endpoint_key not in ec_map:
        return None
    if ec_map[endpoint_key] == 0:
        return None
    with _endpoint_sem_lock:
        if endpoint_key not in _endpoint_semaphores:
            sem = threading.Semaphore(ec_map[endpoint_key])
            _endpoint_semaphores[endpoint_key] = sem
            logger.info(
                "Concurrency limiter: %s — max %d in-flight",
                endpoint_key,
                ec_map[endpoint_key],
            )
        return _endpoint_semaphores[endpoint_key]


def get_http_session():
    """Return the shared HTTP session. Per-request auth via headers= param."""
    return _HTTP_SESSION


def _dump_5xx_payload(
    url: str,
    body: dict,
    error_body: dict,
    status: int,
    attempt: int,
    retry_exhausted: bool,
) -> None:
    """Opt-in diagnostic dump of upstream 5xx request/response to disk.

    Activated by PROXY_DEBUG_5XX=1. PROXY_DEBUG_5XX_DIR overrides the directory.
    Failure to write is non-fatal; logged at warning. Does not alter retry flow.
    """
    if not _DEBUG_5XX:
        return
    try:
        import proxy.state as _state
        request_id = getattr(_state, "request_id", None) or uuid.uuid4().hex[:12]
    except Exception:
        request_id = uuid.uuid4().hex[:12]

    now = datetime.now(timezone.utc)
    timestamp_iso = _now_iso()
    timestamp_safe = now.strftime("%Y%m%dT%H%M%S%f")

    messages = body.get("messages", []) if isinstance(body, dict) else []
    per_tool_message_schema = []
    for idx, msg in enumerate(messages):
        if isinstance(msg, dict) and msg.get("role") == "tool":
            content = msg.get("content")
            if isinstance(content, str):
                content_len = len(content)
            elif isinstance(content, (list, dict)):
                content_len = len(content)
            else:
                content_len = 0
            per_tool_message_schema.append({
                "index": idx,
                "role": msg.get("role"),
                "tool_call_id": msg.get("tool_call_id"),
                "name": msg.get("name"),
                "content_len": content_len,
                "has_name": "name" in msg,
            })

    parsed = urlparse(url)
    sanitized_url = f"{parsed.scheme}://{parsed.netloc}{parsed.path}"

    dump = {
        "request_id": request_id,
        "url": sanitized_url,
        "method": "POST",
        "timestamp": timestamp_iso,
        "status": status,
        "request_body": body,
        "response_body": error_body,
        "is_subagent": body.get("is_subagent") if isinstance(body, dict) else None,
        "message_count": len(messages),
        "per_tool_message_schema": per_tool_message_schema,
        "attempt": attempt,
        "retry_exhausted": retry_exhausted,
    }

    try:
        dump_dir = Path(_DEBUG_5XX_DIR)
        dump_dir.mkdir(parents=True, exist_ok=True)
        filepath = dump_dir / f"{request_id}-{timestamp_safe}.json"
        filepath.write_text(json.dumps(dump, default=str, indent=2))
    except Exception as exc:
        logger.warning("Failed to write 5xx debug dump: %s", exc)


def _dump_request_payload(url: str, body: dict) -> None:
    """Opt-in diagnostic dump of outbound upstream request body to disk.

    Activated by PROXY_DUMP_REQUESTS=1. PROXY_DUMP_REQUESTS_DIR overrides the
    directory (resolved relative to the repo root). Failure to write is
    non-fatal; logged at warning. Does not alter the upstream call.
    """
    if os.environ.get("PROXY_DUMP_REQUESTS", "0") != "1":
        return
    dump_dir_rel = os.environ.get("PROXY_DUMP_REQUESTS_DIR", "data/debug_requests")

    try:
        import proxy.state as _state
        request_id = getattr(_state, "request_id", None) or uuid.uuid4().hex[:12]
    except Exception:
        request_id = uuid.uuid4().hex[:12]

    now = datetime.now(timezone.utc)
    timestamp_iso = _now_iso()
    timestamp_safe = now.strftime("%Y%m%dT%H%M%S%f")

    parsed = urlparse(url)
    sanitized_url = f"{parsed.scheme}://{parsed.netloc}{parsed.path}"

    if isinstance(body, dict):
        messages = body.get("messages") or []
        model = body.get("model")
        tools = body.get("tools")
        tool_count = len(tools) if isinstance(tools, list) else 0
    else:
        messages = []
        model = None
        tool_count = 0

    # Stable SHA-256 of the canonical (sorted-keys) body so the hash survives
    # dict-order changes between runs. Compute BEFORE the dump so the value
    # is available for logging even if the file write fails.
    try:
        body_sha256 = hashlib.sha256(
            json.dumps(body, sort_keys=True, default=str).encode("utf-8")
        ).hexdigest()
    except Exception:
        body_sha256 = None

    def _hash_msg_content(msg):
        content = msg.get("content", "") if isinstance(msg, dict) else msg
        return hashlib.sha256(
            json.dumps(content, default=str, sort_keys=True).encode("utf-8")
        ).hexdigest()

    first_msg_sha256 = None
    last_msg_sha256 = None
    if messages:
        try:
            first_msg_sha256 = _hash_msg_content(messages[0])
        except Exception:
            first_msg_sha256 = None
        try:
            last_msg_sha256 = _hash_msg_content(messages[-1])
        except Exception:
            last_msg_sha256 = None

    messages_by_role = {"user": 0, "assistant": 0, "tool": 0, "system": 0}
    for msg in messages:
        if isinstance(msg, dict):
            role = msg.get("role")
            if role in messages_by_role:
                messages_by_role[role] += 1

    dump = {
        "timestamp": timestamp_iso,
        "request_id": request_id,
        "url": sanitized_url,
        "model": model,
        "message_count": len(messages) if isinstance(messages, list) else 0,
        "tool_count": tool_count,
        "messages_by_role": messages_by_role,
        "first_msg_sha256": first_msg_sha256,
        "last_msg_sha256": last_msg_sha256,
        "body_sha256": body_sha256,
        "request_body": body,
    }

    try:
        dump_dir = Path(dump_dir_rel)
        if not dump_dir.is_absolute():
            dump_dir = _REPO_ROOT / dump_dir
        dump_dir.mkdir(parents=True, exist_ok=True)
        filepath = dump_dir / f"{request_id}-{timestamp_safe}.json"

        # body_bytes reflects the size of request_body as serialized to disk.
        body_bytes = len(json.dumps(body, default=str))
        dump["body_bytes"] = body_bytes

        filepath.write_text(json.dumps(dump, default=str))

        logger.info(
            "Dumped outbound request body (sha256=%s, %d bytes) to %s",
            body_sha256,
            body_bytes,
            filepath,
        )
    except Exception as exc:
        logger.warning("Failed to write request debug dump: %s", exc)


def _enforce_dump_retention(dump_dir: Path) -> None:
    """Delete the oldest dump files until the count is under the cap.

    Cap is read from PROXY_DUMP_MAX_FILES (default 250). Cleanup failure
    is swallowed and logged at warning — a retention error must never
    break a request dump.
    """
    try:
        max_files = int(os.environ.get("PROXY_DUMP_MAX_FILES", "250"))
    except ValueError:
        max_files = 250
    if max_files <= 0:
        return
    try:
        entries = [
            (dump_dir / name, os.path.getmtime(dump_dir / name))
            for name in os.listdir(dump_dir)
            if (dump_dir / name).is_file() and name.endswith(".json")
        ]
        # Leave room for the new dump about to be written: delete down to
        # max_files - 1 so the post-write count is <= max_files.
        overflow = len(entries) - (max_files - 1)
        if overflow <= 0:
            return
        entries.sort(key=lambda item: item[1])
        for path, _mtime in entries[:overflow]:
            try:
                os.remove(path)
            except OSError:
                logger.warning("Failed to remove old dump file %s", path)
    except Exception as exc:
        logger.warning("Dump retention cleanup failed: %s", exc)


def dump_inbound_request_body(
    session_id: Optional[str],
    request_id: Optional[str],
    body: dict,
) -> None:
    """Opt-in diagnostic dump of inbound client request body to disk.

    Activated by the same switch as outbound: PROXY_DUMP_REQUESTS=1.
    PROXY_DUMP_INBOUND_DIR overrides the directory (resolved relative to
    the repo root). Default: data/debug_inbound/.

    PROXY_DUMP_MAX_FILES caps the number of retained dump files (default
    250). Before writing a new dump, the oldest files (by mtime) are
    deleted until the count is under the cap. Cleanup failure is non-fatal
    (logged at warning) — a request is never broken by retention.

    The dump includes a `_meta` block with session_id, request_id,
    timestamp, and body_bytes — and the full `request_body` dict. Failure
    to write is non-fatal; logged at warning. Does not alter the inbound
    request handling.
    """
    if os.environ.get("PROXY_DUMP_REQUESTS", "0") != "1":
        return
    dump_dir_rel = os.environ.get("PROXY_DUMP_INBOUND_DIR", "data/debug_inbound")

    rid = request_id or uuid.uuid4().hex[:12]
    now = datetime.now(timezone.utc)
    timestamp_iso = _now_iso()
    timestamp_safe = now.strftime("%Y%m%dT%H%M%S%f")

    body_bytes = len(json.dumps(body, default=str))
    dump = {
        "_meta": {
            "session_id": session_id,
            "request_id": rid,
            "timestamp": timestamp_iso,
            "body_bytes": body_bytes,
        },
        "request_body": body,
    }

    try:
        dump_dir = Path(dump_dir_rel)
        if not dump_dir.is_absolute():
            dump_dir = _REPO_ROOT / dump_dir
        dump_dir.mkdir(parents=True, exist_ok=True)

        _enforce_dump_retention(dump_dir)

        filepath = dump_dir / f"{rid}-{timestamp_safe}.json"
        filepath.write_text(json.dumps(dump, default=str, indent=2))
        logger.info(
            "Dumped inbound request body (session_id=%s, %d bytes) to %s",
            session_id,
            body_bytes,
            filepath,
        )
    except Exception as exc:
        logger.warning("Failed to write inbound request debug dump: %s", exc)


def _select_streaming_media_type(content_type_header: str) -> str:
    """Return one of: 'sse', 'json', or '' (unsupported).

    Strict header normalization. Does NOT sniff the body. Does NOT
    inspect response.text or response.raw. Returns '' for missing,
    empty, comma-joined, malformed, or unrecognized media types.
    """
    raw = content_type_header or ""
    media_type = raw.split(";", 1)[0].strip().lower()
    if media_type == "text/event-stream":
        return "sse"
    if media_type == "application/json":
        return "json"
    if media_type.startswith("application/") and media_type.endswith("+json"):
        prefix_len = len("application/")
        suffix_len = len("+json")
        if len(media_type) > prefix_len + suffix_len:
            return "json"
    return ""


def _reported_total_tokens(data):
    """Return (present, value) for usage.total_tokens."""
    if not isinstance(data, dict):
        return False, None

    usage = data.get("usage")
    if not isinstance(usage, dict) or "total_tokens" not in usage:
        return False, None

    return True, usage["total_tokens"]


def _capture_upstream_sse_enabled() -> bool:
    """Diagnostic: force upstream stream=true and tee the raw SSE to disk.

    Activated by PROXY_CAPTURE_UPSTREAM_SSE=1. Purpose: capture a real
    provider's upstream SSE bytes (e.g. MiniMax) so the parser can be fixed
    against ground truth WITHOUT crashing the pipeline. When enabled, the SSE
    branch buffers the raw lines, writes them verbatim to disk BEFORE parsing,
    and any parser failure is downgraded to a canonical protocol error (clean
    502) instead of propagating — so a single call yields the raw bytes even
    when reconstruction fails. OFF by default; production is unaffected.
    """
    return os.environ.get("PROXY_CAPTURE_UPSTREAM_SSE", "0") == "1"


def _dump_upstream_sse(url: str, raw_lines: list) -> Optional[str]:
    """Write captured raw upstream SSE lines verbatim to disk. Best-effort;
    returns the file path on success, else None. Gated by the caller."""
    try:
        import proxy.state as _state
        request_id = getattr(_state, "request_id", None) or uuid.uuid4().hex[:12]
    except Exception:
        request_id = uuid.uuid4().hex[:12]
    try:
        now = datetime.now(timezone.utc)
        timestamp_safe = now.strftime("%Y%m%dT%H%M%S%f")
        dump_dir_rel = os.environ.get("PROXY_CAPTURE_UPSTREAM_SSE_DIR", "data/debug_upstream_sse")
        dump_dir = Path(dump_dir_rel)
        if not dump_dir.is_absolute():
            dump_dir = _REPO_ROOT / dump_dir
        dump_dir.mkdir(parents=True, exist_ok=True)
        parsed = urlparse(url)
        header = (
            f"# upstream SSE capture\n"
            f"# url: {parsed.scheme}://{parsed.netloc}{parsed.path}\n"
            f"# request_id: {request_id}\n"
            f"# captured_at: {_now_iso()}\n"
            f"# line_count: {len(raw_lines)}\n"
            f"# NOTE: lines are as produced by iter_lines(decode_unicode=False)"
            f" (terminators stripped); rejoined with \\n.\n\n"
        ).encode("utf-8")
        body = b"\n".join(
            (ln if isinstance(ln, (bytes, bytearray)) else str(ln).encode("utf-8"))
            for ln in raw_lines
        )
        filepath = dump_dir / f"{request_id}-{timestamp_safe}.sse"
        filepath.write_bytes(header + body)
        logger.warning(
            "Captured upstream SSE (%d lines) to %s", len(raw_lines), filepath,
        )
        return str(filepath)
    except Exception as exc:
        logger.warning("Failed to write upstream SSE capture: %s", exc)
        return None


def call_upstream(url: str, body: dict, headers: dict, wants_stream: bool,
                  timeout_cap: Optional[float] = None) -> dict:
    """Make HTTP POST to upstream backend with configurable timeout + retry.

    Retries on transient connection errors with exponential backoff.

    When wants_stream is True, the Requests transport is invoked with
    stream=True so the body can be consumed incrementally via
    response.iter_lines(decode_unicode=False). The mode selector
    (SSE vs JSON vs fail-closed) runs on the Content-Type header
    BEFORE any body is consumed. No Response, iterator, or generator
    escapes this function.

    Args:
        timeout_cap: If set, cap the computed timeout at this value (seconds).
                     Used by the tool loop to enforce its per-iteration budget.
                     Applies to the per-attempt read timeout, not total retry duration.
    """
    timeout = _resolve_upstream_timeout(body)
    if timeout_cap is not None:
        timeout = min(timeout, timeout_cap)
    rcfg = _retry_config()
    dumped = False
    malformed_retry_used = False

    # Diagnostic capture: force upstream SSE and tee raw bytes to disk even when
    # the caller requested non-streaming, so a real provider's stream can be
    # captured for parser fixes without crashing the pipeline.
    capture_sse = _capture_upstream_sse_enabled()
    effective_stream = wants_stream or capture_sse

    # OpenAI-compatible providers decide streaming from the request BODY's
    # `stream` field, not the Requests transport flag. To actually receive SSE
    # under capture we must also set body["stream"]=true (post-revert the body
    # carries stream=false). Copy so we never mutate the caller's dict.
    if capture_sse and isinstance(body, dict) and body.get("stream") is not True:
        body = {**body, "stream": True}
        logger.warning(
            "PROXY_CAPTURE_UPSTREAM_SSE=1: forcing upstream body stream=true to "
            "capture real SSE (diagnostic; a parse failure returns a clean 502)."
        )

    for attempt in range(1, rcfg["max_attempts"] + 1):
        sem = _get_endpoint_semaphore(url)
        acquired = False
        if sem:
            try:
                acquired = sem.acquire(timeout=timeout + 30)
            except Exception:
                logger.exception("Unexpected error acquiring concurrency slot on %s", url)
                raise
            if not acquired:
                logger.error("Timed out waiting for concurrency slot on %s", url)
                raise RuntimeError(f"Concurrency slot timeout: {url}")

        response = None
        accumulator = None
        retry_delay = None
        try:
            _start = time.monotonic()

            if not dumped:
                _dump_request_payload(url=url, body=body)
                dumped = True

            response = _HTTP_SESSION.post(
                url,
                json=body,
                timeout=(30, timeout),  # (connect, read)
                allow_redirects=False,
                headers=headers,
                stream=effective_stream,
            )
            _duration = time.monotonic() - _start
            logger.info(
                json.dumps(
                    {
                        "event": "remote_model_call",
                        "url": url,
                        "status": response.status_code,
                        "duration_ms": round(_duration * 1000),
                        "attempt": attempt,
                    }
                )
            )

            if response.status_code == 429:
                retry_after = response.headers.get("Retry-After")
                if retry_after:
                    try:
                        parsed_retry_after = int(retry_after)
                        if parsed_retry_after < 0:
                            raise ValueError("Retry-After must be non-negative")
                        delay = min(parsed_retry_after, 60)
                        logger.info(
                            json.dumps({"event": "rate_limited", "retry_after": retry_after, "sleeping": delay})
                        )
                    except ValueError:
                        logger.info(
                            json.dumps({"event": "rate_limited", "retry_after": retry_after})
                        )
                        delay = min(
                            rcfg["base_delay"] * (2 ** (attempt - 1)),
                            rcfg["max_delay"],
                        )
                else:
                    delay = min(
                        rcfg["base_delay"] * (2 ** (attempt - 1)),
                        rcfg["max_delay"],
                    )
                    logger.warning("Rate limited (429) with no Retry-After, backing off %.1fs", delay)

                # M1 #2: skip the sleep on the terminal attempt — sleeping
                # then raising wastes a full Retry-After with no recovery.
                if attempt >= rcfg["max_attempts"]:
                    try:
                        backend_body = response.json()
                    except (json.JSONDecodeError, ValueError):
                        backend_body = {"error": "Rate limit exceeded"}
                    if "error" not in backend_body:
                        backend_body["error"] = "Rate limit exceeded"
                    raise BackendError(429, backend_body)
                retry_delay = delay
                continue
            response.raise_for_status()

            if not effective_stream:
                try:
                    data = response.json()
                except (json.JSONDecodeError, ValueError):
                    logger.error(
                        f"Backend returned non-JSON response (length: {len(response.text)}, status: {response.status_code})"
                    )
                    raise RuntimeError("Backend returned invalid JSON response")
            else:
                raw_content_type = response.headers.get("Content-Type", "")
                mode = _select_streaming_media_type(raw_content_type)
                if mode == "":
                    raise UpstreamStreamError(
                        "unsupported upstream streaming content type",
                        code="unsupported_content_type",
                        semantic_started=False,
                    )
                if mode == "json":
                    try:
                        data = response.json()
                    except (json.JSONDecodeError, ValueError):
                        logger.error(
                            f"Backend returned non-JSON response (length: {len(response.text)}, status: {response.status_code})"
                        )
                        raise RuntimeError("Backend returned invalid JSON response")
                else:
                    # mode == "sse"
                    accumulator = OpenAIStreamAccumulator()
                    if capture_sse:
                        # Buffer + tee raw bytes BEFORE parsing so a parser
                        # rejection still yields the captured stream. A parse
                        # failure under capture is downgraded to a canonical
                        # protocol error (clean 502) instead of propagating.
                        raw_lines = list(response.iter_lines(decode_unicode=False))
                        _dump_upstream_sse(url, raw_lines)
                        try:
                            for line in raw_lines:
                                accumulator.feed_line(line)
                            data = accumulator.finish()
                        except UpstreamStreamError as exc:
                            logger.warning(
                                "Upstream SSE reconstruction failed under capture "
                                "(code=%s): %s — returning protocol error; raw bytes captured.",
                                getattr(exc, "code", None), exc,
                            )
                            return build_upstream_protocol_error()
                    else:
                        lines = response.iter_lines(decode_unicode=False)
                        # Wall-clock bound on total SSE consumption. The per-chunk
                        # read timeout does NOT bound a slow-but-alive stream; under
                        # the no-cancel model that would pin the endpoint semaphore
                        # and a completion-pool slot indefinitely. Budget is the
                        # effective per-attempt read timeout.
                        _stream_start = time.monotonic()
                        _stream_budget = timeout
                        for line in lines:
                            if time.monotonic() - _stream_start > _stream_budget:
                                raise UpstreamStreamError(
                                    "upstream stream exceeded time budget",
                                    code="stream_timeout",
                                    semantic_started=accumulator.semantic_started,
                                )
                            accumulator.feed_line(line)
                        data = accumulator.finish()

            shape_error = chat_completion_shape_error(data)
            if shape_error is None:
                return data

            total_tokens_present, total_tokens = _reported_total_tokens(data)
            safe_to_retry = (
                not total_tokens_present
                or (
                    type(total_tokens) is int
                    and total_tokens == 0
                )
            )

            response_id = data.get("id") if isinstance(data, dict) else None
            choices_type = (
                type(data.get("choices")).__name__
                if isinstance(data, dict)
                else type(data).__name__
            )
            if total_tokens_present and type(total_tokens) is not int:
                reported_token_diagnostic = type(total_tokens).__name__
            else:
                reported_token_diagnostic = total_tokens

            if (
                safe_to_retry
                and not malformed_retry_used
                and attempt < rcfg["max_attempts"]
            ):
                malformed_retry_used = True
                retry_delay = min(
                    rcfg["base_delay"] * 2 ** (attempt - 1),
                    rcfg["max_delay"],
                )
                logger.warning(
                    "Upstream returned malformed chat completion (attempt %d/%d): %s. "
                    "choices_type=%s response_id=%s reported_total_tokens=%s. "
                    "Retrying in %.1fs ...",
                    attempt,
                    rcfg["max_attempts"],
                    shape_error,
                    choices_type,
                    response_id,
                    reported_token_diagnostic,
                    retry_delay,
                )
                continue

            logger.error(
                "Upstream returned malformed chat completion (attempt %d/%d): %s. "
                "choices_type=%s response_id=%s reported_total_tokens=%s "
                "malformed_retry_used=%s",
                attempt,
                rcfg["max_attempts"],
                shape_error,
                choices_type,
                response_id,
                reported_token_diagnostic,
                malformed_retry_used,
            )
            return build_upstream_protocol_error()
        except UpstreamStreamError:
            raise
        except requests.exceptions.SSLError:
            raise
        except _RETRYABLE as e:
            if accumulator is not None and accumulator.semantic_started:
                logger.error(
                    "Upstream transport broken after semantic start — not retrying: %s",
                    e,
                )
                raise
            if attempt < rcfg["max_attempts"]:
                delay = min(
                    rcfg["base_delay"] * (2 ** (attempt - 1)),
                    rcfg["max_delay"],
                )
                retry_delay = delay
                logger.warning(
                    "Upstream connection broken (attempt %d/%d): %s. "
                    "Retrying in %.1fs ...",
                    attempt,
                    rcfg["max_attempts"],
                    e,
                    delay,
                )
            else:
                logger.error(
                    "Upstream transport broken — exhausted %d retries: %s",
                    rcfg["max_attempts"],
                    e,
                )
                raise RuntimeError(
                    f"Backend unreachable after {rcfg['max_attempts']} retries: {url}"
                ) from e
        except requests.exceptions.HTTPError as e:
            sanitized = url
            if hasattr(e, "response") and e.response is not None:
                status = e.response.status_code
                try:
                    error_body = e.response.json()
                except (json.JSONDecodeError, ValueError):
                    error_body = {"error": e.response.text or "Unknown error"}
                logger.info(
                    json.dumps(
                        {
                            "event": "proxy_trace_response",
                            "url": sanitized,
                            "status": status,
                            "body": error_body,
                        }
                    )
                )

                if status >= 500:
                    try:
                        _body_summary = {
                            "model": body.get("model") if isinstance(body, dict) else None,
                            "message_count": len(body.get("messages", [])) if isinstance(body, dict) else 0,
                            "tool_count": len(body.get("tools", [])) if isinstance(body, dict) else 0,
                            "stream": body.get("stream") if isinstance(body, dict) else None,
                            "request_body_preview": json.dumps(body)[:2048] if isinstance(body, dict) else str(body)[:2048],
                            "hint": "Set PROXY_DEBUG_5XX=1 to dump the full body to data/debug_5xx/",
                        }
                    except Exception as _e:
                        _body_summary = {"preview_error": str(_e)}
                    logger.error(
                        json.dumps(
                            {
                                "event": "upstream_5xx_body_summary",
                                "url": sanitized,
                                "status": status,
                                "summary": _body_summary,
                            }
                        )
                    )
                    _dump_5xx_payload(
                        url=url,
                        body=body,
                        error_body=error_body,
                        status=status,
                        attempt=attempt,
                        retry_exhausted=attempt == rcfg["max_attempts"],
                    )

                if status in _RETRYABLE_STATUS:
                    if attempt < rcfg["max_attempts"]:
                        delay = min(
                            rcfg["base_delay"] * (2 ** (attempt - 1)),
                            rcfg["max_delay"],
                        )
                        retry_delay = delay
                        logger.warning(
                            "Upstream returned %d (attempt %d/%d): %s. "
                            "Retrying in %.1fs ...",
                            status,
                            attempt,
                            rcfg["max_attempts"],
                            error_body,
                            delay,
                        )
                        continue
                    else:
                        logger.error(
                            "Upstream returned %d — exhausted %d retries: %s",
                            status,
                            rcfg["max_attempts"],
                            error_body,
                        )

                raise BackendError(status, error_body)
            raise RuntimeError(f"Backend error: {url}") from e
        finally:
            try:
                if response is not None:
                    response.close()
            except Exception as close_exc:
                logger.warning(
                    "Upstream response close failed (suppressed): %s",
                    close_exc,
                )
            if acquired:
                try:
                    sem.release()
                except Exception as release_exc:
                    logger.warning(
                        "Upstream semaphore release failed (suppressed): %s",
                        release_exc,
                    )
            if retry_delay is not None:
                time.sleep(retry_delay)