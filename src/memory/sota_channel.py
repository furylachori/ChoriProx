# sota_channel.py — SOTA Abstraction Layer

import os
import re
import json
import logging
import time
import random
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Optional

from openai import APITimeoutError as OpenAITimeout, APIError as OpenAIAPIError

from memory.circuit_breaker import CircuitBreaker, CircuitBreakerOpen

# SOTAMode is defined in _sota_http.py; imported lazily at module scope to avoid circular deps


def __getattr__(name: str):
    if name == "SOTAMode":
        from memory._sota_http import SOTAMode
        return SOTAMode
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

logger = logging.getLogger("memory.sota_channel")

# Retry-After cap (deployment-time knob, read once at module load).
# 60s is the sane max; longer is almost certainly malicious or buggy upstream.
# Env override: PROXY_SOTA_RETRY_AFTER_CAP, clamped to [1, 300].
try:
    _RETRY_AFTER_CAP_SECONDS = int(os.environ.get("PROXY_SOTA_RETRY_AFTER_CAP", "60"))
    if not (1 <= _RETRY_AFTER_CAP_SECONDS <= 300):
        logger.warning(
            "PROXY_SOTA_RETRY_AFTER_CAP=%s out of range [1, 300], clamping",
            _RETRY_AFTER_CAP_SECONDS,
        )
        _RETRY_AFTER_CAP_SECONDS = max(1, min(300, _RETRY_AFTER_CAP_SECONDS))
except ValueError:
    logger.warning(
        "PROXY_SOTA_RETRY_AFTER_CAP=%r is not a valid integer — defaulting to 60",
        os.environ.get("PROXY_SOTA_RETRY_AFTER_CAP"),
    )
    _RETRY_AFTER_CAP_SECONDS = 60


# ── Error hierarchy ──────────────────────────────────────────────────────────

class SOTATimeoutError(Exception):
    pass


class SOTAAPIError(Exception):
    def __init__(self, message, status_code=None):
        super().__init__(message)
        self.status_code = status_code


class SOTARateLimitError(SOTAAPIError):
    def __init__(self, message, retry_after=2.5):
        super().__init__(message, status_code=429)
        self.retry_after = retry_after


class SOTAParseError(Exception):
    pass


# Re-export CircuitBreakerOpen for convenience
# (already imported above)


# ── Regex constants ───────────────────────────────────────────────────────────

_CONTROL_CHAR_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_TOOL_RE = re.compile(r"<tool\b[^>]*>[\s\S]*?</tool>", re.DOTALL | re.IGNORECASE)
_TOOL_UNCLOSED_RE = re.compile(r"<tool\b[^>]*>[\s\S]*$", re.DOTALL | re.IGNORECASE)


# ── Helper functions ──────────────────────────────────────────────────────────

def _clean_sota_response(content: str) -> str:
    """Strip thinking tags, markdown code blocks, and non-JSON text from SOTA response."""
    if not content:
        return ""

    content = content.strip()

    # Strip <thinking>...</thinking> blocks
    content = re.sub(r"<thinking>[\s\S]*?</thinking>", "", content, flags=re.DOTALL)

    # Strip [KEY]...[/KEY] blocks (alternative format)
    content = re.sub(r"\[KEY\][\s\S]*?\[/KEY\]", "", content, flags=re.DOTALL)

    # Strip [ctokol]...[/ctokol] blocks
    content = re.sub(r"\[ctokol\][\s\S]*?\[/ctokol\]", "", content, flags=re.DOTALL)

    # Strip think tags (literal angle brackets, closed)
    content = re.sub(r"<think\b[\s\S]*?> response", "", content, flags=re.DOTALL | re.IGNORECASE)

    # Strip unclosed think tags (model started thinking but never closed)
    content = re.sub(r"<think\b[\s\S]*$", "", content, flags=re.DOTALL | re.IGNORECASE)

    # Strip think tags (unicode corner brackets ...</think> U+300C U+300D)
    content = re.sub(r"\u300c[\s\S]*?\u300d", "", content, flags=re.DOTALL)

    # Strip <TOOL> blocks and unclosed tags
    content = _TOOL_RE.sub("", content)
    content = _TOOL_UNCLOSED_RE.sub("", content)

    # Strip ```json code blocks
    content = re.sub(r"```json\s*", "", content)
    content = re.sub(r"```\s*$", "", content, flags=re.MULTILINE)

    # Strip any remaining ``` markers
    content = content.replace("```", "")

    # Strip text before JSON object (e.g., "Here's the JSON: {" or "Sure! {" etc.)
    json_start = content.find("{")
    if json_start > 0:
        content = content[json_start:]

    # Strip text after JSON object (in case model added trailing text)
    json_end = content.rfind("}")
    if json_end > 0 and json_end < len(content) - 1:
        content = content[: json_end + 1]

    return content.strip()


def _sanitize_text(text: str) -> str:
    """Sanitize free-form text before storage/injection."""
    if not isinstance(text, str):
        return ""
    text = _CONTROL_CHAR_RE.sub("", text)
    if len(text) > 5000:
        text = text[:5000] + "..."
    return text


def _validate_content(item):
    """Ensure content exists. Strip control chars, truncate."""
    content = item.get("content", "")
    if not isinstance(content, str) or not content.strip():
        item["content"] = "(empty)"
        return
    content = _CONTROL_CHAR_RE.sub("", content)
    if len(content) > 2000:
        content = content[:2000] + "..."
    item["content"] = content


def parse_retry_after(value, default=2.5) -> float:
    """Parse Retry-After header value into seconds (capped at _RETRY_AFTER_CAP_SECONDS).

    The cap is a deployment-time knob via PROXY_SOTA_RETRY_AFTER_CAP (default 60s,
    clamped to [1, 300]). A malicious or buggy upstream returning an oversized
    Retry-After can otherwise pin a _sync_pool worker for days.
    """
    if not value:
        return default
    result = default
    try:
        result = float(value)
    except ValueError:
        try:
            from email.utils import parsedate_to_datetime
            dt = parsedate_to_datetime(value)
            seconds = (dt - datetime.now(timezone.utc)).total_seconds()
            result = seconds if seconds > 0 else default
        except Exception as e:
            logger.debug("Failed to parse retry-after header: %s", e)
            return default
    if result > _RETRY_AFTER_CAP_SECONDS:
        logger.warning(
            "Retry-After capped: server value %.1fs exceeds cap %ds",
            result, _RETRY_AFTER_CAP_SECONDS,
        )
        return float(_RETRY_AFTER_CAP_SECONDS)
    return result


def _validate_input(prompt):
    """Validate prompt input. Raises ValueError if invalid."""
    # Lazy import to avoid circular dependency
    from memory._sota_rate_limiter import SOTA_MAX_INPUT_CHARS
    if prompt is None:
        raise ValueError("prompt cannot be None")
    if not isinstance(prompt, str):
        raise ValueError("prompt must be a string")
    if not prompt.strip():
        raise ValueError("prompt cannot be empty or whitespace")
    if len(prompt) > SOTA_MAX_INPUT_CHARS:
        logger.warning(
            "Prompt length %d exceeds SOTA_MAX_INPUT_CHARS %d (session_id context not available)",
            len(prompt), int(SOTA_MAX_INPUT_CHARS)
        )


# ── SOTAResult dataclass ──────────────────────────────────────────────────────

@dataclass
class SOTAResult:
    raw: Any
    parsed: Any
    elapsed_ms: float
    attempts: int
    model: str
    usage: Optional[object] = None


# ── Module-level circuit breaker instance ──────────────────────────────────────

_sota_breaker = CircuitBreaker(failure_threshold=5, recovery_timeout_seconds=120, name="sota")


# ── SOTAChannel class ──────────────────────────────────────────────────────────

class SOTAChannel:
    """Abstraction layer for SOTA API calls with circuit breaker, rate limiting, and retry."""

    def __init__(
        self,
        mode,
        system_prompt: str,
        tools: Optional[list] = None,
        tool_choice: Optional[dict] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        reasoning_effort: str = "low",
        parse_fn: Optional[callable] = None,
        model: Optional[str] = None,
    ) -> None:
        from memory._sota_http import SOTAMode as SOTAModeEnum, _MODE_MODEL, _MODE_DEFAULTS
        from memory._sota_rate_limiter import _SOTA_RETRY_DELAY as _sota_retry_delay
        self._mode = mode
        self._system_prompt = system_prompt
        self._tools = tools if tools is not None else None
        self._tool_choice = tool_choice if tool_choice is not None else None
        self._reasoning_effort = reasoning_effort
        self._parse_fn = parse_fn

        defaults = _MODE_DEFAULTS[mode]
        self._temperature = temperature if temperature is not None else defaults.get("temperature")
        self._max_tokens = max_tokens if max_tokens is not None else defaults.get("max_tokens")
        self._model = model if model is not None else _MODE_MODEL[mode]

    @property
    def model(self) -> str:
        return self._model

    def ask(self, prompt: str, session_id: str = "") -> SOTAResult:
        # Lazy imports to avoid circular dependency
        from memory._sota_http import get_sota_client
        from memory._sota_rate_limiter import claim_slot as _claim_slot, _SOTA_RETRY_DELAY as _sota_retry_delay

        # Step 1: Validate input
        _validate_input(prompt)

        # Step 2: Circuit breaker check
        if not _sota_breaker.can_execute():
            raise CircuitBreakerOpen(
                f"[sota] Circuit is {_sota_breaker.state.value}. "
                f"Try again after 120s."
            )

        # Step 3: Get client and timeout
        client = get_sota_client()
        _sota_http_timeout = client.timeout

        _elapsed = 0.0
        raw = ""
        parsed = None
        usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

        # Step 5: Claim slot once outside retry loop
        with _claim_slot():
            for attempt in range(5):
                try:
                    _call_start = time.monotonic()

                    messages = [
                        {"role": "system", "content": self._system_prompt},
                        {"role": "user", "content": prompt},
                    ]

                    extra_body = {"reasoning_effort": self._reasoning_effort}

                    response = client.chat.completions.create(
                        model=self._model,
                        messages=messages,
                        max_tokens=self._max_tokens,
                        temperature=self._temperature,
                        extra_body=extra_body,
                        tools=self._tools if self._tools else None,
                        tool_choice=self._tool_choice if self._tool_choice else None,
                    )

                    _elapsed = time.monotonic() - _call_start

                    # Runtime guard: verify response structure
                    if not response.choices or response.choices[0].message is None:
                        raise SOTAParseError(
                            "SOTA response missing choices or message "
                            f"(attempt {attempt + 1}/5 session_id={session_id})"
                        )

                    message = response.choices[0].message

                    # Mode branching: tool-call mode vs content-only mode
                    if self._tools:
                        # Tool-call mode
                        if message.tool_calls and len(message.tool_calls) > 0:
                            raw_args = message.tool_calls[0].function.arguments
                            try:
                                raw = json.loads(raw_args)
                            except json.JSONDecodeError as e:
                                raise SOTAParseError(
                                    f"Failed to parse tool-call arguments: {e} "
                                    f"(attempt {attempt + 1}/5 session_id={session_id})"
                                )
                        elif message.content:
                            cleaned = _clean_sota_response(message.content)
                            try:
                                raw = json.loads(cleaned)
                            except json.JSONDecodeError as e:
                                raise SOTAParseError(
                                    f"Tool-call mode: content parsed as JSON failed: {e} "
                                    f"(attempt {attempt + 1}/5 session_id={session_id})"
                                )
                            if not isinstance(raw, dict):
                                raise SOTAParseError(
                                    f"Tool-call mode: content parsed but not a dict (got {type(raw).__name__}) "
                                    f"(attempt {attempt + 1}/5 session_id={session_id})"
                                )
                        else:
                            raise SOTAParseError(
                                f"No tool_calls and no content in tool-call mode "
                                f"(attempt {attempt + 1}/5 session_id={session_id})"
                            )
                    else:
                        # Content-only mode
                        if message.content:
                            raw = _clean_sota_response(message.content)
                        else:
                            raise SOTAParseError(
                                f"No content in content-only mode "
                                f"(attempt {attempt + 1}/5 session_id={session_id})"
                            )

                    parsed = self._parse_fn(raw) if self._parse_fn else raw

                    _sota_breaker.on_success()

                    usage = getattr(response, "usage", None) if hasattr(response, "usage") else None

                    break  # exit retry loop on success

                except SOTAParseError:
                    if attempt == 4:
                        raise
                    logger.warning(
                        "SOTAParseError (attempt %d/5 session_id=%s), retrying in %.1fs",
                        attempt + 1, session_id,
                        _sota_retry_delay * (2 ** attempt) + random.uniform(0, 1)
                    )
                    time.sleep(_sota_retry_delay * (2 ** attempt) + random.uniform(0, 1))

                except (OpenAITimeout, OpenAIAPIError) as e:
                    status = getattr(e, "status_code", None)

                    if status == 429:
                        retry_after = parse_retry_after(
                            getattr(getattr(e, "response", None), "headers", {}).get("Retry-After"),
                            default=2.5
                        )
                        logger.warning(
                            "Rate limited (429) in SOTAChannel (attempt %d/5 session_id=%s), backing off %.1fs",
                            attempt + 1, session_id, retry_after
                        )
                        # M5-2: count every 429 toward the breaker. A steady
                        # 429 stream is exactly the failure mode the breaker
                        # is designed to detect; without this, a persistent
                        # rate-limit burns MAX_ATTEMPTS × Retry-After sleeps
                        # before the breaker ever sees a failure.
                        _sota_breaker.on_failure()
                        if attempt == 4:
                            raise SOTARateLimitError(f"Rate limited after {attempt + 1} attempts")
                        time.sleep(retry_after)
                        continue

                    if status in (502, 503, 504):
                        sleep = _sota_retry_delay * (2 ** attempt) + random.uniform(0, 1)
                        logger.warning(
                            "Server error %s in SOTAChannel (attempt %d/5 session_id=%s), backing off %.1fs",
                            status, attempt + 1, session_id, sleep
                        )
                        time.sleep(sleep)
                        if attempt == 4:
                            _sota_breaker.on_failure()
                            raise SOTAAPIError(f"Server error {status} after 5 attempts", status_code=status)
                        continue

                    if isinstance(e, OpenAITimeout):
                        if attempt == 4:
                            _sota_breaker.on_failure()
                            raise SOTATimeoutError(f"API call timed out after {_sota_http_timeout}s")
                        logger.warning(
                            "SOTA timeout in SOTAChannel (attempt %d/5 session_id=%s), retrying",
                            attempt + 1, session_id
                        )
                        time.sleep(_sota_retry_delay)
                        continue

                    # General API error
                    if attempt == 4:
                        _sota_breaker.on_failure()
                        raise SOTAAPIError(str(e), status_code=status)
                    logger.warning(
                        "SOTA API error in SOTAChannel (attempt %d/5 session_id=%s): %s, retrying",
                        attempt + 1, session_id, e
                    )
                    time.sleep(_sota_retry_delay)

        return SOTAResult(
            raw=raw,
            parsed=parsed,
            elapsed_ms=_elapsed * 1000,
            attempts=attempt + 1,
            model=self._model,
            usage=usage,
        )