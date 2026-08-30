"""Unit tests for src/proxy/http_client.py Wave 2B streaming transport.

Tests invoke real call_upstream and real accumulator behavior. Parser
methods and semantic state are not mocked. Strict concrete doubles
drive Session and Response lifecycles with shared operation logs.
"""

from __future__ import annotations

import json
from typing import Any, Optional

import pytest
import requests
from requests.structures import CaseInsensitiveDict

from proxy import http_client
from proxy.http_client import call_upstream
from proxy.upstream_sse import (
    DEFAULT_UPSTREAM_STREAM_LIMITS,
    UpstreamStreamError,
    UpstreamStreamLimitError,
    UpstreamStreamProtocolError,
)


# ---------------------------------------------------------------------------
# Strict response double
# ---------------------------------------------------------------------------


class StrictResponse:
    """Concrete requests.Response double with configurable lifecycle."""

    def __init__(
        self,
        status_code: int = 200,
        headers: Optional[dict] = None,
        lines: Optional[list] = None,
        json_value: Any = None,
        json_exception: Optional[BaseException] = None,
        raise_for_status_exception: Optional[BaseException] = None,
        iter_lines_exception: Optional[BaseException] = None,
        close_exception: Optional[BaseException] = None,
        text: str = "",
        operation_log: Optional[list] = None,
    ) -> None:
        self.status_code = status_code
        self.headers = CaseInsensitiveDict(headers if headers is not None else {})
        self._lines = lines if lines is not None else []
        self._json_value = json_value
        self._json_exception = json_exception
        self._raise_for_status_exception = raise_for_status_exception
        self._iter_lines_exception = iter_lines_exception
        self._close_exception = close_exception
        self.text = text
        self.operation_log = (
            operation_log if operation_log is not None else []
        )
        self.json_calls = 0
        self.iter_lines_calls = 0
        self.close_calls = 0
        self.requested_decode_unicode: object = None
        self._closed = False

    def raise_for_status(self) -> None:
        self.operation_log.append("raise_for_status")
        if self._raise_for_status_exception is not None:
            raise self._raise_for_status_exception
        if self.status_code >= 400:
            err = requests.exceptions.HTTPError(
                f"{self.status_code} http error", response=self
            )
            raise err

    def json(self) -> Any:
        if self._closed:
            raise AssertionError("json() called after close()")
        self.operation_log.append("json")
        self.json_calls += 1
        if self._json_exception is not None:
            raise self._json_exception
        return self._json_value

    def iter_lines(self, decode_unicode: bool = False):
        if self._closed:
            raise AssertionError("iter_lines() called after close()")
        if decode_unicode is not False:
            raise AssertionError(
                "iter_lines(decode_unicode=...) must be False, got "
                f"{decode_unicode!r}"
            )
        self.operation_log.append("iter_lines")
        self.iter_lines_calls += 1
        self.requested_decode_unicode = decode_unicode
        if self._iter_lines_exception is not None:
            def _gen():
                for line in self._lines:
                    yield line
                raise self._iter_lines_exception
            return _gen()
        return iter(self._lines)

    def close(self) -> None:
        self.operation_log.append("close")
        self.close_calls += 1
        if self._close_exception is not None:
            raise self._close_exception
        self._closed = True


# ---------------------------------------------------------------------------
# Strict session double
# ---------------------------------------------------------------------------


class ScriptedSession:
    """Concrete session double with explicit stream= parameter support."""

    def __init__(self, operation_log: Optional[list] = None) -> None:
        self.post_calls: list = []
        self.responses: list = []
        self.operation_log = (
            operation_log if operation_log is not None else []
        )

    def post(self, url, json=None, timeout=None, allow_redirects=False,
             headers=None, stream=False):
        if stream is not True and stream is not False:
            raise AssertionError(f"post() stream must be bool, got {stream!r}")
        self.post_calls.append({
            "url": url,
            "json": json,
            "timeout": timeout,
            "allow_redirects": allow_redirects,
            "headers": headers,
            "stream": stream,
        })
        self.operation_log.append("post")
        if not self.responses:
            raise AssertionError("No scripted response left")
        entry = self.responses.pop(0)
        if isinstance(entry, BaseException):
            raise entry
        return entry


# ---------------------------------------------------------------------------
# Recording semaphore
# ---------------------------------------------------------------------------


class RecordingSemaphore:
    """Concrete semaphore that records acquire/release ordering."""

    def __init__(
        self,
        operation_log: Optional[list] = None,
        raise_on_release: Optional[BaseException] = None,
    ) -> None:
        self.acquire_calls = 0
        self.release_calls = 0
        self._held = False
        self.operation_log = (
            operation_log if operation_log is not None else []
        )
        self._raise_on_release = raise_on_release

    def acquire(self, timeout=None) -> bool:
        self.operation_log.append("acquire")
        self.acquire_calls += 1
        self._held = True
        return True

    def release(self) -> None:
        self.operation_log.append("release")
        self.release_calls += 1
        self._held = False
        if self._raise_on_release is not None:
            raise self._raise_on_release


# ---------------------------------------------------------------------------
# Fixtures and helpers
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _fast_retry_config(monkeypatch):
    """Default fast retry config for every test in this module."""
    monkeypatch.setitem(http_client.CONFIG, "upstream", {
        "timeout": 30,
        "retry_max_attempts": 3,
        "retry_base_delay": 0.01,
        "retry_max_delay": 0.1,
    })


def _patch_session(monkeypatch, session: ScriptedSession) -> None:
    monkeypatch.setattr(http_client, "_HTTP_SESSION", session)


def _patch_endpoint_semaphore(monkeypatch, sem: RecordingSemaphore) -> None:
    monkeypatch.setattr(http_client, "_get_endpoint_semaphore", lambda url: sem)


def _minimal_sse_lines(content: str = "ok") -> list:
    return [
        b'data: {"id":"x","choices":[{"index":0,"delta":{"role":"assistant"}}]}',
        b"",
        (
            b'data: {"choices":[{"index":0,"delta":{"content":"'
            + content.encode("utf-8")
            + b'"},"finish_reason":"stop"}]}'
        ),
        b"",
        b"data: [DONE]",
        b"",
    ]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_stream_true_passes_requests_stream_true(monkeypatch):
    session = ScriptedSession()
    session.responses.append(StrictResponse(
        status_code=200,
        headers={"Content-Type": "text/event-stream"},
        lines=_minimal_sse_lines(),
    ))
    _patch_session(monkeypatch, session)

    result = call_upstream("http://test/v1", {"model": "m"}, {}, True)

    assert result["object"] == "chat.completion"
    assert result["choices"][0]["message"]["content"] == "ok"
    assert session.post_calls[0]["stream"] is True


def test_stream_false_passes_requests_stream_false_and_uses_json(monkeypatch):
    session = ScriptedSession()
    response = StrictResponse(
        status_code=200,
        json_value={"choices": [{"message": {"role": "assistant", "content": "hi"}}]},
    )
    session.responses.append(response)
    _patch_session(monkeypatch, session)

    result = call_upstream("http://test/v1", {"model": "m"}, {}, False)

    assert result == {"choices": [{"message": {"role": "assistant", "content": "hi"}}]}
    assert session.post_calls[0]["stream"] is False
    assert response.json_calls == 1


def test_sse_content_type_with_parameters_reconstructs_complete_response(monkeypatch):
    session = ScriptedSession()
    session.responses.append(StrictResponse(
        status_code=200,
        headers={"Content-Type": "text/event-stream; charset=utf-8"},
        lines=_minimal_sse_lines("hello"),
    ))
    _patch_session(monkeypatch, session)

    result = call_upstream("http://test/v1", {}, {}, True)

    assert result["choices"][0]["message"]["content"] == "hello"
    assert result["choices"][0]["finish_reason"] == "stop"


def test_sse_branch_calls_iter_lines_with_decode_unicode_false_not_json(monkeypatch):
    session = ScriptedSession()
    response = StrictResponse(
        status_code=200,
        headers={"Content-Type": "text/event-stream"},
        lines=_minimal_sse_lines(),
    )
    session.responses.append(response)
    _patch_session(monkeypatch, session)

    call_upstream("http://test/v1", {}, {}, True)

    assert response.iter_lines_calls == 1
    assert response.json_calls == 0
    assert response.requested_decode_unicode is False


def test_application_json_fallback_uses_json_not_iter_lines(monkeypatch):
    session = ScriptedSession()
    response = StrictResponse(
        status_code=200,
        headers={"Content-Type": "application/json"},
        json_value={"choices": [{"message": {"role": "assistant", "content": "x"}}]},
    )
    session.responses.append(response)
    _patch_session(monkeypatch, session)

    result = call_upstream("http://test/v1", {}, {}, True)

    assert result["choices"][0]["message"]["content"] == "x"
    assert response.json_calls == 1
    assert response.iter_lines_calls == 0


def test_vendor_application_json_fallback_is_accepted(monkeypatch):
    session = ScriptedSession()
    response = StrictResponse(
        status_code=200,
        headers={"Content-Type": "application/vnd.example+json"},
        json_value={"choices": [{"message": {"role": "assistant", "content": "v"}}]},
    )
    session.responses.append(response)
    _patch_session(monkeypatch, session)

    result = call_upstream("http://test/v1", {}, {}, True)

    assert result["choices"][0]["message"]["content"] == "v"


def test_missing_content_type_fails_without_consuming_body(monkeypatch):
    from proxy.upstream_sse import UpstreamStreamError

    session = ScriptedSession()
    response = StrictResponse(
        status_code=200,
        headers={},
        lines=_minimal_sse_lines(),
    )
    session.responses.append(response)
    _patch_session(monkeypatch, session)

    with pytest.raises(UpstreamStreamError) as excinfo:
        call_upstream("http://test/v1", {}, {}, True)

    assert excinfo.value.code == "unsupported_content_type"
    assert response.json_calls == 0
    assert response.iter_lines_calls == 0
    assert response.close_calls == 1


def test_unsupported_content_type_fails_without_consuming_body(monkeypatch):
    from proxy.upstream_sse import UpstreamStreamError

    session = ScriptedSession()
    response = StrictResponse(
        status_code=200,
        headers={"Content-Type": "text/json"},
        lines=_minimal_sse_lines(),
    )
    session.responses.append(response)
    _patch_session(monkeypatch, session)

    with pytest.raises(UpstreamStreamError):
        call_upstream("http://test/v1", {}, {}, True)

    assert response.json_calls == 0
    assert response.iter_lines_calls == 0


def test_malformed_vendor_json_content_type_is_rejected(monkeypatch):
    from proxy.upstream_sse import UpstreamStreamError

    session = ScriptedSession()
    response = StrictResponse(
        status_code=200,
        headers={"Content-Type": "application/+json"},
        lines=_minimal_sse_lines(),
    )
    session.responses.append(response)
    _patch_session(monkeypatch, session)

    with pytest.raises(UpstreamStreamError):
        call_upstream("http://test/v1", {}, {}, True)


def test_malformed_sse_does_not_fall_back_to_json(monkeypatch):
    session = ScriptedSession()
    response = StrictResponse(
        status_code=200,
        headers={"Content-Type": "text/event-stream"},
        lines=[
            b"data: {not-json",
            b"",
        ],
    )
    session.responses.append(response)
    _patch_session(monkeypatch, session)

    with pytest.raises(UpstreamStreamProtocolError):
        call_upstream("http://test/v1", {}, {}, True)

    assert session.post_calls[-1]["stream"] is True


def test_sse_parser_limit_error_is_not_retried(monkeypatch):
    session = ScriptedSession()
    too_long = b"x" * (DEFAULT_UPSTREAM_STREAM_LIMITS.max_line_bytes + 1)
    response = StrictResponse(
        status_code=200,
        headers={"Content-Type": "text/event-stream"},
        lines=[b"data: " + too_long, b""],
    )
    session.responses.append(response)
    _patch_session(monkeypatch, session)

    with pytest.raises(UpstreamStreamLimitError):
        call_upstream("http://test/v1", {}, {}, True)

    assert len(session.post_calls) == 1


def test_requests_connection_error_from_post_retries(monkeypatch):
    session = ScriptedSession()
    session.responses.append(requests.exceptions.ConnectionError("dropped"))
    session.responses.append(StrictResponse(
        status_code=200,
        json_value={"choices": [{"message": {"role": "assistant", "content": "ok"}}]},
    ))
    _patch_session(monkeypatch, session)

    result = call_upstream("http://test/v1", {}, {}, False)

    assert result["choices"][0]["message"]["content"] == "ok"
    assert len(session.post_calls) == 2
    assert session.post_calls[0] == session.post_calls[1]


def test_iter_lines_connection_error_before_semantic_start_retries(monkeypatch):
    session = ScriptedSession()
    pre_semantic_error_response = StrictResponse(
        status_code=200,
        headers={"Content-Type": "text/event-stream"},
        lines=[b": keepalive", b""],
        iter_lines_exception=requests.exceptions.ConnectionError("dropped mid-stream"),
    )
    second_response = StrictResponse(
        status_code=200,
        headers={"Content-Type": "text/event-stream"},
        lines=_minimal_sse_lines("retry-ok"),
    )
    session.responses.append(pre_semantic_error_response)
    session.responses.append(second_response)
    _patch_session(monkeypatch, session)

    result = call_upstream("http://test/v1", {}, {}, True)

    assert result["choices"][0]["message"]["content"] == "retry-ok"
    assert len(session.post_calls) == 2


def test_iter_lines_connection_error_after_semantic_start_does_not_retry(monkeypatch):
    session = ScriptedSession()
    post_semantic_error_response = StrictResponse(
        status_code=200,
        headers={"Content-Type": "text/event-stream"},
        lines=[
            b'data: {"choices":[{"index":0,"delta":{"role":"assistant"}}]}',
            b"",
        ],
        iter_lines_exception=requests.exceptions.ConnectionError("dropped after role"),
    )
    session.responses.append(post_semantic_error_response)
    _patch_session(monkeypatch, session)

    with pytest.raises(requests.exceptions.ConnectionError):
        call_upstream("http://test/v1", {}, {}, True)

    assert len(session.post_calls) == 1


def test_each_retry_uses_fresh_accumulator_state(monkeypatch):
    session = ScriptedSession()
    session.responses.append(StrictResponse(
        status_code=200,
        headers={"Content-Type": "text/event-stream"},
        lines=[b": comment", b""],
        iter_lines_exception=requests.exceptions.ConnectionError("first"),
    ))
    session.responses.append(StrictResponse(
        status_code=200,
        headers={"Content-Type": "text/event-stream"},
        lines=_minimal_sse_lines("final"),
    ))
    _patch_session(monkeypatch, session)

    captured_accumulators: list = []
    real_accumulator = http_client.OpenAIStreamAccumulator

    class RecordingAccumulator(real_accumulator):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            captured_accumulators.append(self)

    monkeypatch.setattr(
        http_client, "OpenAIStreamAccumulator", RecordingAccumulator
    )

    result = call_upstream("http://test/v1", {}, {}, True)

    assert result["choices"][0]["message"]["content"] == "final"
    assert len(captured_accumulators) == 2
    assert captured_accumulators[0] is not captured_accumulators[1]


def test_response_closes_on_json_success_before_semaphore_release(monkeypatch):
    sem = RecordingSemaphore()
    session = ScriptedSession()
    response = StrictResponse(
        status_code=200,
        json_value={"choices": [{"message": {"role": "assistant", "content": "ok"}}]},
    )
    session.responses.append(response)
    _patch_session(monkeypatch, session)
    _patch_endpoint_semaphore(monkeypatch, sem)

    events: list = []

    original_release = sem.release
    def tracking_release():
        events.append(("release", len(events)))
        original_release()
    sem.release = tracking_release

    original_close = response.close
    def tracking_close():
        events.append(("close", len(events)))
        original_close()
    response.close = tracking_close

    call_upstream("http://test/v1", {}, {}, False)

    close_idx = next(i for i, (op, _) in enumerate(events) if op == "close")
    release_idx = next(i for i, (op, _) in enumerate(events) if op == "release")
    assert close_idx < release_idx


def test_response_closes_on_sse_success_before_semaphore_release(monkeypatch):
    sem = RecordingSemaphore()
    session = ScriptedSession()
    response = StrictResponse(
        status_code=200,
        headers={"Content-Type": "text/event-stream"},
        lines=_minimal_sse_lines(),
    )
    session.responses.append(response)
    _patch_session(monkeypatch, session)
    _patch_endpoint_semaphore(monkeypatch, sem)

    events: list = []

    original_release = sem.release
    def tracking_release():
        events.append("release")
        original_release()
    sem.release = tracking_release

    original_close = response.close
    def tracking_close():
        events.append("close")
        original_close()
    response.close = tracking_close

    call_upstream("http://test/v1", {}, {}, True)

    assert events.index("close") < events.index("release")


def test_response_closes_on_parser_error_before_semaphore_release(monkeypatch):
    from proxy.upstream_sse import UpstreamStreamProtocolError

    log: list = []
    sem = RecordingSemaphore(operation_log=log)
    session = ScriptedSession(operation_log=log)
    response = StrictResponse(
        status_code=200,
        headers={"Content-Type": "text/event-stream"},
        lines=[b"data: {not-json", b""],
        operation_log=log,
    )
    session.responses.append(response)
    _patch_session(monkeypatch, session)
    _patch_endpoint_semaphore(monkeypatch, sem)

    with pytest.raises(UpstreamStreamProtocolError):
        call_upstream("http://test/v1", {}, {}, True)

    iter_lines_idx = log.index("iter_lines")
    close_idx = log.index("close")
    release_idx = log.index("release")
    assert iter_lines_idx < close_idx < release_idx


def test_retry_after_delay_is_preserved_after_close_and_release(monkeypatch):
    monkeypatch.setitem(http_client.CONFIG, "upstream", {
        "timeout": 30,
        "retry_max_attempts": 3,
        "retry_base_delay": 0.01,
        "retry_max_delay": 0.1,
    })

    events: list = []

    sem = RecordingSemaphore(operation_log=events)
    session = ScriptedSession(operation_log=events)

    first = StrictResponse(
        status_code=429,
        headers={"Retry-After": "5"},
        json_value={"error": "rate limited"},
        operation_log=events,
    )
    session.responses.append(first)

    second = StrictResponse(
        status_code=200,
        json_value={"choices": [{"message": {"role": "assistant", "content": "ok"}}]},
        operation_log=events,
    )
    session.responses.append(second)

    _patch_session(monkeypatch, session)
    _patch_endpoint_semaphore(monkeypatch, sem)

    acquire_count = [0]
    original_acquire = sem.acquire
    def tracking_acquire(timeout=None):
        acquire_count[0] += 1
        events.append(f"acquire:{acquire_count[0]}")
        return original_acquire(timeout)
    sem.acquire = tracking_acquire

    release_count = [0]
    original_release = sem.release
    def tracking_release():
        release_count[0] += 1
        events.append(f"release:{release_count[0]}")
        original_release()
    sem.release = tracking_release

    first_original_close = first.close
    def first_close_tracking():
        events.append("close:first")
        first_original_close()
    first.close = first_close_tracking

    second_original_close = second.close
    def second_close_tracking():
        events.append("close:second")
        second_original_close()
    second.close = second_close_tracking

    sleep_calls: list = []

    def fake_sleep(duration):
        events.append(("sleep", duration))
        sleep_calls.append(duration)

    monkeypatch.setattr(http_client.time, "sleep", fake_sleep)

    result = call_upstream("http://test/v1", {}, {}, False)

    assert result["choices"][0]["message"]["content"] == "ok"
    assert len([e for e in events if e == "post"]) == 2
    assert len(sleep_calls) == 1
    assert sleep_calls[0] == 5
    assert first.close_calls == 1
    assert second.close_calls == 1

    post_indices = [i for i, e in enumerate(events) if e == "post"]
    first_close_idx = events.index("close:first")
    first_release_idx = events.index("release:1")
    second_acquire_idx = events.index("acquire:2")
    second_post_idx = post_indices[1]
    sleep_event_idx = None
    for i, e in enumerate(events):
        if isinstance(e, tuple) and e[0] == "sleep":
            sleep_event_idx = i
            break

    assert (
        post_indices[0]
        < first_close_idx
        < first_release_idx
    ), f"first attempt order: {events}"
    assert (
        first_release_idx < sleep_event_idx
    ), f"sleep order: {events}"
    assert (
        sleep_event_idx < second_acquire_idx < second_post_idx
    ), f"second attempt order: {events}"


def test_negative_retry_after_uses_exponential_delay_after_cleanup(
    monkeypatch,
):
    monkeypatch.setitem(http_client.CONFIG, "upstream", {
        "timeout": 30,
        "retry_max_attempts": 3,
        "retry_base_delay": 0.25,
        "retry_max_delay": 1.0,
    })

    events: list = []
    sem = RecordingSemaphore(operation_log=events)
    session = ScriptedSession(operation_log=events)

    first = StrictResponse(
        status_code=429,
        headers={"Retry-After": "-5"},
        json_value={"error": "rate limited"},
        operation_log=events,
    )
    second = StrictResponse(
        status_code=200,
        json_value={
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": "ok",
                },
            }],
        },
        operation_log=events,
    )
    session.responses.extend([first, second])

    _patch_session(monkeypatch, session)
    _patch_endpoint_semaphore(monkeypatch, sem)

    sleep_calls: list = []

    def fake_sleep(delay):
        if delay < 0:
            raise ValueError("sleep length must be non-negative")
        events.append(("sleep", delay))
        sleep_calls.append(delay)

    monkeypatch.setattr(http_client.time, "sleep", fake_sleep)

    result = call_upstream(
        "http://test/v1",
        {},
        {},
        False,
    )

    assert result["choices"][0]["message"]["content"] == "ok"
    assert len(session.post_calls) == 2
    assert sleep_calls == [0.25]
    assert first.close_calls == 1
    assert second.close_calls == 1
    assert sem.acquire_calls == 2
    assert sem.release_calls == 2

    first_post = events.index("post")
    first_close = events.index("close")
    first_release = events.index("release")
    sleep = events.index(("sleep", 0.25))
    second_acquire = events.index("acquire", first_release + 1)
    second_post = events.index("post", first_post + 1)

    assert first_post < first_close < first_release < sleep
    assert sleep < second_acquire < second_post


def test_zero_retry_after_retries_immediately(monkeypatch):
    monkeypatch.setitem(http_client.CONFIG, "upstream", {
        "timeout": 30,
        "retry_max_attempts": 3,
        "retry_base_delay": 0.25,
        "retry_max_delay": 1.0,
    })

    session = ScriptedSession()
    first = StrictResponse(
        status_code=429,
        headers={"Retry-After": "0"},
        json_value={"error": "rate limited"},
    )
    second = StrictResponse(
        status_code=200,
        json_value={
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": "ok",
                },
            }],
        },
    )
    session.responses.extend([first, second])
    _patch_session(monkeypatch, session)

    sleep_calls: list = []
    monkeypatch.setattr(
        http_client.time,
        "sleep",
        lambda delay: sleep_calls.append(delay),
    )

    result = call_upstream(
        "http://test/v1",
        {},
        {},
        False,
    )

    assert result["choices"][0]["message"]["content"] == "ok"
    assert len(session.post_calls) == 2
    assert sleep_calls == [0]
    assert first.close_calls == 1
    assert second.close_calls == 1


def test_missing_retry_after_uses_exponential_delay(monkeypatch):
    monkeypatch.setitem(http_client.CONFIG, "upstream", {
        "timeout": 30,
        "retry_max_attempts": 3,
        "retry_base_delay": 0.25,
        "retry_max_delay": 1.0,
    })

    events: list = []
    sem = RecordingSemaphore(operation_log=events)
    session = ScriptedSession(operation_log=events)

    first = StrictResponse(
        status_code=429,
        headers={},
        json_value={"error": "rate limited"},
        operation_log=events,
    )
    session.responses.append(first)
    second = StrictResponse(
        status_code=200,
        json_value={"choices": [{"message": {"role": "assistant", "content": "ok"}}]},
        operation_log=events,
    )
    session.responses.append(second)

    _patch_session(monkeypatch, session)
    _patch_endpoint_semaphore(monkeypatch, sem)

    sleep_calls: list = []

    def fake_sleep(duration):
        sleep_calls.append(duration)

    monkeypatch.setattr(http_client.time, "sleep", fake_sleep)

    result = call_upstream("http://test/v1", {}, {}, False)

    assert result["choices"][0]["message"]["content"] == "ok"
    assert sleep_calls == [0.25]
    assert first.close_calls == 1
    close_idx = events.index("close")
    release_idx = events.index("release")
    assert close_idx < release_idx


def test_terminal_429_stream_response_uses_existing_json_error_path_and_closes(monkeypatch):
    from proxy.types import BackendError

    monkeypatch.setitem(http_client.CONFIG, "upstream", {
        "timeout": 30,
        "retry_max_attempts": 1,
        "retry_base_delay": 0.01,
        "retry_max_delay": 0.1,
    })

    session = ScriptedSession()
    response = StrictResponse(
        status_code=429,
        headers={"Content-Type": "application/json"},
        json_value={"error": "rate limit"},
    )
    session.responses.append(response)
    _patch_session(monkeypatch, session)

    sleep_calls: list = []

    def fake_sleep(duration):
        sleep_calls.append(duration)

    monkeypatch.setattr(http_client.time, "sleep", fake_sleep)

    with pytest.raises(BackendError) as excinfo:
        call_upstream("http://test/v1", {}, {}, True)

    assert excinfo.value.status_code == 429
    assert response.close_calls == 1
    assert response.json_calls == 1
    assert response.iter_lines_calls == 0
    assert sleep_calls == []


def test_terminal_429_with_negative_retry_after_does_not_sleep(
    monkeypatch,
):
    from proxy.types import BackendError

    monkeypatch.setitem(http_client.CONFIG, "upstream", {
        "timeout": 30,
        "retry_max_attempts": 1,
        "retry_base_delay": 0.25,
        "retry_max_delay": 1.0,
    })

    sem = RecordingSemaphore()
    session = ScriptedSession()
    response = StrictResponse(
        status_code=429,
        headers={
            "Content-Type": "application/json",
            "Retry-After": "-5",
        },
        json_value={"error": "rate limited"},
    )
    session.responses.append(response)
    _patch_session(monkeypatch, session)
    _patch_endpoint_semaphore(monkeypatch, sem)

    sleep_calls: list = []
    monkeypatch.setattr(
        http_client.time,
        "sleep",
        lambda delay: sleep_calls.append(delay),
    )

    with pytest.raises(BackendError) as excinfo:
        call_upstream(
            "http://test/v1",
            {},
            {},
            True,
        )

    assert excinfo.value.status_code == 429
    assert len(session.post_calls) == 1
    assert response.json_calls == 1
    assert response.iter_lines_calls == 0
    assert response.close_calls == 1
    assert sem.acquire_calls == 1
    assert sem.release_calls == 1
    assert sleep_calls == []


def test_retryable_transport_then_parser_error_does_not_sleep_twice(monkeypatch):
    monkeypatch.setitem(http_client.CONFIG, "upstream", {
        "timeout": 30,
        "retry_max_attempts": 3,
        "retry_base_delay": 0.01,
        "retry_max_delay": 0.1,
    })

    session = ScriptedSession()
    transport_error = StrictResponse(
        status_code=200,
        headers={"Content-Type": "text/event-stream"},
        lines=[b": keepalive", b""],
        iter_lines_exception=requests.exceptions.ConnectionError("first"),
    )
    session.responses.append(transport_error)
    parser_error = StrictResponse(
        status_code=200,
        headers={"Content-Type": "text/event-stream"},
        lines=[b"data: {not-json", b""],
    )
    session.responses.append(parser_error)
    _patch_session(monkeypatch, session)

    sleep_calls: list = []

    def fake_sleep(duration):
        sleep_calls.append(duration)

    monkeypatch.setattr(http_client.time, "sleep", fake_sleep)

    with pytest.raises(UpstreamStreamProtocolError):
        call_upstream("http://test/v1", {}, {}, True)

    assert len(sleep_calls) == 1
    assert len(session.post_calls) == 2
    assert transport_error.close_calls == 1
    assert parser_error.close_calls == 1


def test_retryable_status_then_unsupported_content_type_does_not_sleep_twice(monkeypatch):
    from proxy.upstream_sse import UpstreamStreamError

    monkeypatch.setitem(http_client.CONFIG, "upstream", {
        "timeout": 30,
        "retry_max_attempts": 3,
        "retry_base_delay": 0.01,
        "retry_max_delay": 0.1,
    })

    session = ScriptedSession()
    retryable = StrictResponse(
        status_code=503,
        headers={"Content-Type": "application/json"},
        json_value={"error": "first"},
    )
    session.responses.append(retryable)
    unsupported = StrictResponse(
        status_code=200,
        headers={"Content-Type": "text/plain"},
        lines=_minimal_sse_lines(),
    )
    session.responses.append(unsupported)
    _patch_session(monkeypatch, session)

    sleep_calls: list = []

    def fake_sleep(duration):
        sleep_calls.append(duration)

    monkeypatch.setattr(http_client.time, "sleep", fake_sleep)

    with pytest.raises(UpstreamStreamError) as excinfo:
        call_upstream("http://test/v1", {}, {}, True)

    assert excinfo.value.code == "unsupported_content_type"
    assert len(sleep_calls) == 1
    assert len(session.post_calls) == 2


def test_retryable_transport_then_json_decode_error_does_not_sleep_twice(monkeypatch):
    monkeypatch.setitem(http_client.CONFIG, "upstream", {
        "timeout": 30,
        "retry_max_attempts": 3,
        "retry_base_delay": 0.01,
        "retry_max_delay": 0.1,
    })

    session = ScriptedSession()
    session.responses.append(requests.exceptions.ConnectionError("first"))
    second = StrictResponse(
        status_code=200,
        headers={"Content-Type": "application/json"},
        json_exception=ValueError("bad json"),
    )
    session.responses.append(second)
    _patch_session(monkeypatch, session)

    sleep_calls: list = []

    def fake_sleep(duration):
        sleep_calls.append(duration)

    monkeypatch.setattr(http_client.time, "sleep", fake_sleep)

    with pytest.raises(RuntimeError, match="Backend returned invalid JSON response"):
        call_upstream("http://test/v1", {}, {}, True)

    assert len(sleep_calls) == 1
    assert len(session.post_calls) == 2
    assert second.close_calls == 1


def test_retryable_status_then_nonretryable_http_error_does_not_sleep_twice(monkeypatch):
    from proxy.types import BackendError

    monkeypatch.setitem(http_client.CONFIG, "upstream", {
        "timeout": 30,
        "retry_max_attempts": 3,
        "retry_base_delay": 0.01,
        "retry_max_delay": 0.1,
    })

    session = ScriptedSession()
    retryable = StrictResponse(
        status_code=503,
        headers={"Content-Type": "application/json"},
        json_value={"error": "first"},
    )
    session.responses.append(retryable)
    nonretryable = StrictResponse(
        status_code=400,
        headers={"Content-Type": "application/json"},
        json_value={"error": "bad request"},
    )
    session.responses.append(nonretryable)
    _patch_session(monkeypatch, session)

    sleep_calls: list = []

    def fake_sleep(duration):
        sleep_calls.append(duration)

    monkeypatch.setattr(http_client.time, "sleep", fake_sleep)

    with pytest.raises(BackendError) as excinfo:
        call_upstream("http://test/v1", {}, {}, False)

    assert excinfo.value.status_code == 400
    assert len(sleep_calls) == 1
    assert len(session.post_calls) == 2


def test_iter_lines_connection_error_after_content_does_not_retry(monkeypatch):
    session = ScriptedSession()
    response = StrictResponse(
        status_code=200,
        headers={"Content-Type": "text/event-stream"},
        lines=[
            b'data: {"choices":[{"index":0,"delta":{"content":"x"}}]}',
            b"",
        ],
        iter_lines_exception=requests.exceptions.ConnectionError("dropped after content"),
    )
    session.responses.append(response)
    _patch_session(monkeypatch, session)

    sleep_calls: list = []

    def fake_sleep(duration):
        sleep_calls.append(duration)

    monkeypatch.setattr(http_client.time, "sleep", fake_sleep)

    with pytest.raises(requests.exceptions.ConnectionError):
        call_upstream("http://test/v1", {}, {}, True)

    assert len(session.post_calls) == 1
    assert sleep_calls == []
    assert response.close_calls == 1


def test_iter_lines_connection_error_after_tool_fragment_does_not_retry(monkeypatch):
    session = ScriptedSession()
    response = StrictResponse(
        status_code=200,
        headers={"Content-Type": "text/event-stream"},
        lines=[
            b'data: {"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"call-1","type":"function","function":{"name":"read","arguments":""}}]}}]}',
            b"",
        ],
        iter_lines_exception=requests.exceptions.ConnectionError("dropped after tool"),
    )
    session.responses.append(response)
    _patch_session(monkeypatch, session)

    sleep_calls: list = []

    def fake_sleep(duration):
        sleep_calls.append(duration)

    monkeypatch.setattr(http_client.time, "sleep", fake_sleep)

    with pytest.raises(requests.exceptions.ConnectionError):
        call_upstream("http://test/v1", {}, {}, True)

    assert len(session.post_calls) == 1
    assert sleep_calls == []
    assert response.close_calls == 1


def test_iter_lines_connection_error_after_finish_reason_does_not_retry(monkeypatch):
    session = ScriptedSession()
    response = StrictResponse(
        status_code=200,
        headers={"Content-Type": "text/event-stream"},
        lines=[
            b'data: {"choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}',
            b"",
        ],
        iter_lines_exception=requests.exceptions.ConnectionError("dropped after finish"),
    )
    session.responses.append(response)
    _patch_session(monkeypatch, session)

    sleep_calls: list = []

    def fake_sleep(duration):
        sleep_calls.append(duration)

    monkeypatch.setattr(http_client.time, "sleep", fake_sleep)

    with pytest.raises(requests.exceptions.ConnectionError):
        call_upstream("http://test/v1", {}, {}, True)

    assert len(session.post_calls) == 1
    assert sleep_calls == []
    assert response.close_calls == 1


def test_semaphore_remains_held_through_sse_iteration_and_close(monkeypatch):
    sem = RecordingSemaphore()
    held_observations: list = []

    class ObservingSemaphore(RecordingSemaphore):
        def __init__(self):
            super().__init__()
            self._held_during_iteration = False
            self._held_during_close = False

        def release(self):
            held_observations.append(("release", self._held))
            super().release()

    obs_sem = ObservingSemaphore()

    session = ScriptedSession()

    def make_response():
        class ObservingResponse(StrictResponse):
            def __init__(self):
                super().__init__(
                    status_code=200,
                    headers={"Content-Type": "text/event-stream"},
                    lines=_minimal_sse_lines("ok"),
                )
                self._sem = obs_sem

            def iter_lines(self, decode_unicode=False):
                obs_sem._held_during_iteration = obs_sem._held
                return super().iter_lines(decode_unicode)

            def close(self):
                obs_sem._held_during_close = obs_sem._held
                return super().close()

        return ObservingResponse()

    response = make_response()
    session.responses.append(response)
    _patch_session(monkeypatch, session)
    _patch_endpoint_semaphore(monkeypatch, obs_sem)

    result = call_upstream("http://test/v1", {}, {}, True)

    assert result["choices"][0]["message"]["content"] == "ok"
    assert obs_sem._held_during_iteration is True
    assert obs_sem._held_during_close is True
    assert obs_sem._held is False
    assert obs_sem.release_calls == 1


def test_ssl_error_is_not_retried(monkeypatch):
    session = ScriptedSession()
    session.responses.append(requests.exceptions.SSLError("certificate failed"))
    _patch_session(monkeypatch, session)
    sem = RecordingSemaphore()
    _patch_endpoint_semaphore(monkeypatch, sem)

    sleep_calls: list = []

    def fake_sleep(duration):
        sleep_calls.append(duration)

    monkeypatch.setattr(http_client.time, "sleep", fake_sleep)

    with pytest.raises(requests.exceptions.SSLError):
        call_upstream("http://test/v1", {}, {}, False)

    assert len(session.post_calls) == 1
    assert sleep_calls == []
    assert sem.release_calls == 1


def test_semaphore_release_failure_is_logged_without_replacing_success(monkeypatch, caplog):
    import logging
    sem = RecordingSemaphore(
        raise_on_release=RuntimeError("release failed")
    )
    session = ScriptedSession()
    response = StrictResponse(
        status_code=200,
        json_value={"choices": [{"message": {"role": "assistant", "content": "ok"}}]},
    )
    session.responses.append(response)
    _patch_session(monkeypatch, session)
    _patch_endpoint_semaphore(monkeypatch, sem)

    with caplog.at_level(logging.WARNING, logger="proxy.http_client"):
        result = call_upstream("http://test/v1", {}, {}, False)

    assert result["choices"][0]["message"]["content"] == "ok"
    assert sem.release_calls == 1
    warning_records = [
        r for r in caplog.records
        if "semaphore release failed" in r.message.lower()
    ]
    assert len(warning_records) >= 1
    for record in warning_records:
        assert "body" not in record.message.lower()
        assert "credential" not in record.message.lower()


def test_close_failure_does_not_replace_json_success_and_releases_semaphore(monkeypatch):
    sem = RecordingSemaphore()
    session = ScriptedSession()
    response = StrictResponse(
        status_code=200,
        json_value={"choices": [{"message": {"role": "assistant", "content": "ok"}}]},
        close_exception=RuntimeError("close failed"),
    )
    session.responses.append(response)
    _patch_session(monkeypatch, session)
    _patch_endpoint_semaphore(monkeypatch, sem)

    result = call_upstream("http://test/v1", {}, {}, False)

    assert result["choices"][0]["message"]["content"] == "ok"
    assert sem.release_calls == 1


def test_close_failure_does_not_replace_sse_success_and_releases_semaphore(monkeypatch):
    sem = RecordingSemaphore()
    session = ScriptedSession()
    response = StrictResponse(
        status_code=200,
        headers={"Content-Type": "text/event-stream"},
        lines=_minimal_sse_lines("ok"),
        close_exception=RuntimeError("close failed"),
    )
    session.responses.append(response)
    _patch_session(monkeypatch, session)
    _patch_endpoint_semaphore(monkeypatch, sem)

    result = call_upstream("http://test/v1", {}, {}, True)

    assert result["choices"][0]["message"]["content"] == "ok"
    assert sem.release_calls == 1


def test_close_failure_does_not_replace_parser_error_and_releases_semaphore(monkeypatch):
    sem = RecordingSemaphore()
    session = ScriptedSession()
    response = StrictResponse(
        status_code=200,
        headers={"Content-Type": "text/event-stream"},
        lines=[b"data: {not-json", b""],
        close_exception=RuntimeError("close failed"),
    )
    session.responses.append(response)
    _patch_session(monkeypatch, session)
    _patch_endpoint_semaphore(monkeypatch, sem)

    with pytest.raises(UpstreamStreamProtocolError):
        call_upstream("http://test/v1", {}, {}, True)

    assert response.close_calls == 1
    assert sem.release_calls == 1


def test_lowercase_content_type_header_selects_sse(monkeypatch):
    session = ScriptedSession()
    session.responses.append(StrictResponse(
        status_code=200,
        headers={"content-type": "text/event-stream; charset=utf-8"},
        lines=_minimal_sse_lines("hello"),
    ))
    _patch_session(monkeypatch, session)

    result = call_upstream("http://test/v1", {}, {}, True)

    assert result["choices"][0]["message"]["content"] == "hello"
    assert result["choices"][0]["finish_reason"] == "stop"


# ---------------------------------------------------------------------------
# US-060a: malformed-completion semantic validation + safe single retry
# ---------------------------------------------------------------------------


def test_malformed_then_valid_json_retries_with_semantic_validation(monkeypatch):
    """US-060a: A. Null choices with zero total_tokens retries once; second post wins."""
    monkeypatch.setitem(http_client.CONFIG, "upstream", {
        "timeout": 30,
        "retry_max_attempts": 3,
        "retry_base_delay": 0.25,
        "retry_max_delay": 1.0,
    })

    events: list = []
    sem = RecordingSemaphore(operation_log=events)
    session = ScriptedSession(operation_log=events)

    malformed = StrictResponse(
        status_code=200,
        json_value={
            "id": "malformed-1",
            "choices": None,
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        },
        operation_log=events,
    )
    valid = StrictResponse(
        status_code=200,
        json_value={"choices": [{"message": {"role": "assistant", "content": "recovered"}}]},
        operation_log=events,
    )
    session.responses.append(malformed)
    session.responses.append(valid)
    _patch_session(monkeypatch, session)
    _patch_endpoint_semaphore(monkeypatch, sem)

    sleep_calls: list = []

    def fake_sleep(duration):
        sleep_calls.append(duration)
        events.append(("sleep", duration))

    monkeypatch.setattr(http_client.time, "sleep", fake_sleep)

    result = call_upstream("http://test/v1", {}, {}, False)

    assert result["choices"][0]["message"]["content"] == "recovered"
    assert len([e for e in events if e == "post"]) == 2
    assert malformed.close_calls == 1
    assert valid.close_calls == 1
    assert sem.acquire_calls == 2
    assert sem.release_calls == 2
    assert sleep_calls == [0.25]

    post_indices = [i for i, e in enumerate(events) if e == "post"]
    first_close = events.index("close")
    first_release = events.index("release")
    sleep_idx = events.index(("sleep", 0.25))
    second_acquire = events.index("acquire", first_release + 1)
    second_post = post_indices[1]
    last_close = len(events) - 1 - events[::-1].index("close")
    last_release = len(events) - 1 - events[::-1].index("release")

    assert post_indices[0] < first_close < first_release < sleep_idx
    assert sleep_idx < second_acquire < second_post
    assert second_post < last_close < last_release


def test_malformed_twice_uses_two_posts_and_leaves_one_unused(monkeypatch):
    """US-060a: B. Two malformed zero-token posts then canonical error; third unused."""
    monkeypatch.setitem(http_client.CONFIG, "upstream", {
        "timeout": 30,
        "retry_max_attempts": 3,
        "retry_base_delay": 0.01,
        "retry_max_delay": 0.1,
    })

    session = ScriptedSession()
    malformed_a = StrictResponse(
        status_code=200,
        json_value={
            "id": "m-a",
            "choices": None,
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        },
    )
    malformed_b = StrictResponse(
        status_code=200,
        json_value={
            "id": "m-b",
            "choices": None,
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        },
    )
    unused_valid = StrictResponse(
        status_code=200,
        json_value={"choices": [{"message": {"role": "assistant", "content": "x"}}]},
    )
    session.responses.extend([malformed_a, malformed_b, unused_valid])
    _patch_session(monkeypatch, session)

    sleep_calls: list = []

    def fake_sleep(duration):
        sleep_calls.append(duration)

    monkeypatch.setattr(http_client.time, "sleep", fake_sleep)

    from proxy.openai_compat import build_upstream_protocol_error
    result = call_upstream("http://test/v1", {}, {}, False)

    assert result == build_upstream_protocol_error()
    assert len(session.post_calls) == 2
    assert len(session.responses) == 1
    assert malformed_a.close_calls == 1
    assert malformed_b.close_calls == 1
    assert sleep_calls == [0.01]


def test_non_dict_response_then_valid_retries(monkeypatch):
    """US-060a: C. Non-dict first response is malformed; retry succeeds with valid JSON."""
    monkeypatch.setitem(http_client.CONFIG, "upstream", {
        "timeout": 30,
        "retry_max_attempts": 3,
        "retry_base_delay": 0.01,
        "retry_max_delay": 0.1,
    })

    session = ScriptedSession()
    session.responses.append(StrictResponse(
        status_code=200,
        json_value=[],
    ))
    session.responses.append(StrictResponse(
        status_code=200,
        json_value={"choices": [{"message": {"role": "assistant", "content": "ok"}}]},
    ))
    _patch_session(monkeypatch, session)

    sleep_calls: list = []

    def fake_sleep(duration):
        sleep_calls.append(duration)

    monkeypatch.setattr(http_client.time, "sleep", fake_sleep)

    result = call_upstream("http://test/v1", {}, {}, False)

    assert result["choices"][0]["message"]["content"] == "ok"
    assert len(session.post_calls) == 2
    assert sleep_calls == [0.01]


def test_nonzero_total_tokens_does_not_retry(monkeypatch):
    """US-060a: D. Malformed with total_tokens=123 does NOT retry (unsafe)."""
    monkeypatch.setitem(http_client.CONFIG, "upstream", {
        "timeout": 30,
        "retry_max_attempts": 3,
        "retry_base_delay": 0.25,
        "retry_max_delay": 1.0,
    })

    session = ScriptedSession()
    malformed = StrictResponse(
        status_code=200,
        json_value={
            "id": "m-nonzero",
            "choices": None,
            "usage": {"prompt_tokens": 100, "completion_tokens": 23, "total_tokens": 123},
        },
    )
    unused_valid = StrictResponse(
        status_code=200,
        json_value={"choices": [{"message": {"role": "assistant", "content": "unused"}}]},
    )
    session.responses.append(malformed)
    session.responses.append(unused_valid)
    _patch_session(monkeypatch, session)

    sleep_calls: list = []

    def fake_sleep(duration):
        sleep_calls.append(duration)

    monkeypatch.setattr(http_client.time, "sleep", fake_sleep)

    from proxy.openai_compat import build_upstream_protocol_error
    result = call_upstream("http://test/v1", {}, {}, False)

    assert result == build_upstream_protocol_error()
    assert len(session.post_calls) == 1
    assert sleep_calls == []
    assert malformed.close_calls == 1
    assert len(session.responses) == 1


@pytest.mark.parametrize(
    "bad_total_tokens",
    [-1, False, "0", "123", 0.0, None],
    ids=["neg-int", "bool-false", "str-zero", "str-123", "float-zero", "none"],
)
def test_present_untrustworthy_total_tokens_does_not_retry(monkeypatch, bad_total_tokens):
    """US-060a: E. Present-but-non-int-zero total_tokens is unsafe; no retry."""
    monkeypatch.setitem(http_client.CONFIG, "upstream", {
        "timeout": 30,
        "retry_max_attempts": 3,
        "retry_base_delay": 0.25,
        "retry_max_delay": 1.0,
    })

    session = ScriptedSession()
    malformed = StrictResponse(
        status_code=200,
        json_value={
            "id": "m-bad",
            "choices": None,
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": bad_total_tokens},
        },
    )
    unused_valid = StrictResponse(
        status_code=200,
        json_value={"choices": [{"message": {"role": "assistant", "content": "x"}}]},
    )
    session.responses.append(malformed)
    session.responses.append(unused_valid)
    _patch_session(monkeypatch, session)

    sleep_calls: list = []

    def fake_sleep(duration):
        sleep_calls.append(duration)

    monkeypatch.setattr(http_client.time, "sleep", fake_sleep)

    from proxy.openai_compat import build_upstream_protocol_error
    result = call_upstream("http://test/v1", {}, {}, False)

    assert result == build_upstream_protocol_error()
    assert len(session.post_calls) == 1
    assert sleep_calls == []
    assert len(session.responses) == 1


def test_absent_total_tokens_retries_to_valid(monkeypatch):
    """US-060a: F. Malformed body with no usage field retries once; second post wins."""
    monkeypatch.setitem(http_client.CONFIG, "upstream", {
        "timeout": 30,
        "retry_max_attempts": 3,
        "retry_base_delay": 0.01,
        "retry_max_delay": 0.1,
    })

    session = ScriptedSession()
    session.responses.append(StrictResponse(
        status_code=200,
        json_value={"id": "m-no-usage", "choices": None},
    ))
    session.responses.append(StrictResponse(
        status_code=200,
        json_value={"choices": [{"message": {"role": "assistant", "content": "ok"}}]},
    ))
    _patch_session(monkeypatch, session)

    sleep_calls: list = []

    def fake_sleep(duration):
        sleep_calls.append(duration)

    monkeypatch.setattr(http_client.time, "sleep", fake_sleep)

    result = call_upstream("http://test/v1", {}, {}, False)

    assert result["choices"][0]["message"]["content"] == "ok"
    assert len(session.post_calls) == 2
    assert sleep_calls == [0.01]


def test_explicit_error_envelope_passes_through(monkeypatch):
    """US-060a: G. Dict-valued error envelope is treated as explicit provider error."""
    monkeypatch.setitem(http_client.CONFIG, "upstream", {
        "timeout": 30,
        "retry_max_attempts": 3,
        "retry_base_delay": 0.25,
        "retry_max_delay": 1.0,
    })

    session = ScriptedSession()
    envelope = {
        "error": {"message": "provider rejected request", "type": "provider_error"},
    }
    session.responses.append(StrictResponse(
        status_code=200,
        json_value=envelope,
    ))
    unused_valid = StrictResponse(
        status_code=200,
        json_value={"choices": [{"message": {"role": "assistant", "content": "x"}}]},
    )
    session.responses.append(unused_valid)
    _patch_session(monkeypatch, session)

    sleep_calls: list = []

    def fake_sleep(duration):
        sleep_calls.append(duration)

    monkeypatch.setattr(http_client.time, "sleep", fake_sleep)

    result = call_upstream("http://test/v1", {}, {}, False)

    assert result == envelope
    assert len(session.post_calls) == 1
    assert sleep_calls == []
    assert len(session.responses) == 1


def test_string_valued_error_is_malformed_and_retries(monkeypatch):
    """US-060a: H. String-valued error is malformed; retry once; second malformed -> canonical error."""
    monkeypatch.setitem(http_client.CONFIG, "upstream", {
        "timeout": 30,
        "retry_max_attempts": 3,
        "retry_base_delay": 0.01,
        "retry_max_delay": 0.1,
    })

    session = ScriptedSession()
    session.responses.append(StrictResponse(
        status_code=200,
        json_value={"error": "provider rejected request"},
    ))
    session.responses.append(StrictResponse(
        status_code=200,
        json_value={
            "id": "m-string-error",
            "choices": None,
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        },
    ))
    _patch_session(monkeypatch, session)

    sleep_calls: list = []

    def fake_sleep(duration):
        sleep_calls.append(duration)

    monkeypatch.setattr(http_client.time, "sleep", fake_sleep)

    from proxy.openai_compat import build_upstream_protocol_error
    result = call_upstream("http://test/v1", {}, {}, False)

    assert result == build_upstream_protocol_error()
    assert len(session.post_calls) == 2
    assert sleep_calls == [0.01]


def test_streaming_request_with_json_content_type_retries(monkeypatch):
    """US-060a: I. wants_stream=True with application/json content goes through shared validation."""
    monkeypatch.setitem(http_client.CONFIG, "upstream", {
        "timeout": 30,
        "retry_max_attempts": 3,
        "retry_base_delay": 0.01,
        "retry_max_delay": 0.1,
    })

    session = ScriptedSession()
    session.responses.append(StrictResponse(
        status_code=200,
        headers={"Content-Type": "application/json"},
        json_value={
            "id": "stream-json-malformed",
            "choices": None,
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        },
    ))
    session.responses.append(StrictResponse(
        status_code=200,
        headers={"Content-Type": "application/json"},
        json_value={"choices": [{"message": {"role": "assistant", "content": "recovered"}}]},
    ))
    _patch_session(monkeypatch, session)

    sleep_calls: list = []

    def fake_sleep(duration):
        sleep_calls.append(duration)

    monkeypatch.setattr(http_client.time, "sleep", fake_sleep)

    result = call_upstream("http://test/v1", {}, {}, True)

    assert result["choices"][0]["message"]["content"] == "recovered"
    assert len(session.post_calls) == 2
    assert sleep_calls == [0.01]


def test_sse_branch_runs_shared_semantic_validation(monkeypatch):
    """US-060a: J. Reconstructed SSE dict is validated by the shared shape check."""
    from unittest.mock import patch

    real_validator = http_client.chat_completion_shape_error

    with patch.object(
        http_client, "chat_completion_shape_error", wraps=real_validator
    ) as mock:
        session = ScriptedSession()
        session.responses.append(StrictResponse(
            status_code=200,
            headers={"Content-Type": "text/event-stream"},
            lines=_minimal_sse_lines("sse-valid"),
        ))
        _patch_session(monkeypatch, session)

        result = call_upstream("http://test/v1", {}, {}, True)

    assert result["choices"][0]["message"]["content"] == "sse-valid"
    assert mock.call_count == 1
    called_with = mock.call_args.args[0]
    assert called_with["id"] == "x"
    assert called_with["choices"][0]["message"]["content"] == "sse-valid"


def test_json_null_then_valid_retries_once(monkeypatch):
    """US-060a: actual upstream JSON null triggers shared shape validation and retry once."""
    monkeypatch.setitem(http_client.CONFIG, "upstream", {
        "timeout": 30,
        "retry_max_attempts": 3,
        "retry_base_delay": 0.01,
        "retry_max_delay": 0.1,
    })

    session = ScriptedSession()
    null_response = StrictResponse(status_code=200, json_value=None)
    valid_response = StrictResponse(
        status_code=200,
        json_value={"choices": [{"message": {"role": "assistant", "content": "recovered-null"}}]},
    )
    session.responses.append(null_response)
    session.responses.append(valid_response)
    _patch_session(monkeypatch, session)

    sleep_calls: list = []

    def fake_sleep(duration):
        sleep_calls.append(duration)

    monkeypatch.setattr(http_client.time, "sleep", fake_sleep)

    result = call_upstream("http://test/v1", {}, {}, False)

    assert result["choices"][0]["message"]["content"] == "recovered-null"
    assert len(session.post_calls) == 2
    assert sleep_calls == [0.01]
    assert null_response.close_calls == 1
    assert valid_response.close_calls == 1
    assert len(session.responses) == 0


def test_json_null_twice_returns_canonical_error_without_third_request(monkeypatch):
    """US-060a: two JSON null responses exhaust the single malformed-retry budget."""
    monkeypatch.setitem(http_client.CONFIG, "upstream", {
        "timeout": 30,
        "retry_max_attempts": 3,
        "retry_base_delay": 0.01,
        "retry_max_delay": 0.1,
    })

    session = ScriptedSession()
    null_a = StrictResponse(status_code=200, json_value=None)
    null_b = StrictResponse(status_code=200, json_value=None)
    unused_valid = StrictResponse(
        status_code=200,
        json_value={"choices": [{"message": {"role": "assistant", "content": "never-used"}}]},
    )
    session.responses.extend([null_a, null_b, unused_valid])
    _patch_session(monkeypatch, session)

    sleep_calls: list = []

    def fake_sleep(duration):
        sleep_calls.append(duration)

    monkeypatch.setattr(http_client.time, "sleep", fake_sleep)

    from proxy.openai_compat import build_upstream_protocol_error
    result = call_upstream("http://test/v1", {}, {}, False)

    assert result == build_upstream_protocol_error()
    assert len(session.post_calls) == 2
    assert sleep_calls == [0.01]
    assert null_a.close_calls == 1
    assert null_b.close_calls == 1
    assert len(session.responses) == 1


@pytest.mark.parametrize(
    "transport_exception",
    [
        requests.exceptions.ConnectionError("connection reset"),
        requests.exceptions.Timeout("timed out"),
        requests.exceptions.ChunkedEncodingError("truncated"),
    ],
    ids=["connection-error", "timeout", "chunked-encoding-error"],
)
def test_shared_attempt_budget_for_connection_failure_then_malformed(
    monkeypatch, transport_exception
):
    """US-060a: retry_max_attempts=2: real transport exception then malformed; no third request."""
    monkeypatch.setitem(http_client.CONFIG, "upstream", {
        "timeout": 30,
        "retry_max_attempts": 2,
        "retry_base_delay": 0.01,
        "retry_max_delay": 0.1,
    })

    session = ScriptedSession()
    session.responses.append(transport_exception)
    malformed = StrictResponse(
        status_code=200,
        json_value={
            "id": "terminal-malformed-after-transport",
            "choices": None,
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        },
    )
    unused_valid = StrictResponse(
        status_code=200,
        json_value={"choices": [{"message": {"role": "assistant", "content": "never-used"}}]},
    )
    session.responses.append(malformed)
    session.responses.append(unused_valid)
    _patch_session(monkeypatch, session)

    sleep_calls: list = []

    def fake_sleep(duration):
        sleep_calls.append(duration)

    monkeypatch.setattr(http_client.time, "sleep", fake_sleep)

    from proxy.openai_compat import build_upstream_protocol_error
    result = call_upstream("http://test/v1", {}, {}, False)

    assert result == build_upstream_protocol_error()
    assert len(session.post_calls) == 2
    assert malformed.close_calls == 1
    assert len(session.responses) == 1
    assert sleep_calls == [0.01]


def test_shared_attempt_budget_for_http_503_then_malformed(monkeypatch):
    """US-060a: K. retry_max_attempts=2: 503 then malformed; no third request."""
    monkeypatch.setitem(http_client.CONFIG, "upstream", {
        "timeout": 30,
        "retry_max_attempts": 2,
        "retry_base_delay": 0.01,
        "retry_max_delay": 0.1,
    })

    session = ScriptedSession()
    transport_retryable = StrictResponse(
        status_code=503,
        headers={"Content-Type": "application/json"},
        json_value={"error": "first"},
    )
    malformed = StrictResponse(
        status_code=200,
        json_value={
            "id": "m-budget",
            "choices": None,
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        },
    )
    session.responses.append(transport_retryable)
    session.responses.append(malformed)
    _patch_session(monkeypatch, session)

    sleep_calls: list = []

    def fake_sleep(duration):
        sleep_calls.append(duration)

    monkeypatch.setattr(http_client.time, "sleep", fake_sleep)

    from proxy.openai_compat import build_upstream_protocol_error
    result = call_upstream("http://test/v1", {}, {}, False)

    assert result == build_upstream_protocol_error()
    assert len(session.post_calls) == 2
    assert transport_retryable.close_calls == 1
    assert malformed.close_calls == 1
    assert sleep_calls == [0.01]


def test_malformed_retry_log_does_not_contain_body_or_sentinel(monkeypatch, caplog):
    """US-060a: L. Sentinel value and full body never appear in log output."""
    import logging

    monkeypatch.setitem(http_client.CONFIG, "upstream", {
        "timeout": 30,
        "retry_max_attempts": 3,
        "retry_base_delay": 0.25,
        "retry_max_delay": 1.0,
    })

    sentinel = "DO_NOT_LOG_THIS_VALUE"
    session = ScriptedSession()
    session.responses.append(StrictResponse(
        status_code=200,
        json_value={
            "id": "leak-candidate",
            "choices": None,
            "secret_malformed_body_sentinel": sentinel,
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        },
    ))
    session.responses.append(StrictResponse(
        status_code=200,
        json_value={"choices": [{"message": {"role": "assistant", "content": "ok"}}]},
    ))
    _patch_session(monkeypatch, session)

    def fake_sleep(duration):
        pass

    monkeypatch.setattr(http_client.time, "sleep", fake_sleep)

    with caplog.at_level(logging.WARNING, logger="proxy.http_client"):
        result = call_upstream("http://test/v1", {}, {}, False)

    assert result["choices"][0]["message"]["content"] == "ok"

    all_messages = "\n".join(r.getMessage() for r in caplog.records)
    assert sentinel not in all_messages
    assert "secret_malformed_body_sentinel" not in all_messages
    assert "choices is not an array" in all_messages
    assert "leak-candidate" in all_messages


# ---------------------------------------------------------------------------
# A5 — wall-clock budget on SSE consumption (ticket 199ece93 Part A)
# ---------------------------------------------------------------------------


def test_sse_exceeds_wall_clock_budget_raises_stream_timeout(monkeypatch):
    """A slow-but-alive SSE stream must be bounded by a total wall-clock budget
    (= the per-attempt read timeout), not just the per-chunk read timeout.
    Exceeding it raises a NON-retryable UpstreamStreamError(code="stream_timeout").
    Revert-test: deleting the budget guard makes this hang/loop forever.
    """
    class _JumpClock:
        def __init__(self):
            self.t = 0.0

        def __call__(self):
            self.t += 10_000.0
            return self.t

    monkeypatch.setattr(http_client.time, "monotonic", _JumpClock())

    session = ScriptedSession()
    session.responses.append(StrictResponse(
        status_code=200,
        headers={"Content-Type": "text/event-stream"},
        lines=_minimal_sse_lines(),
    ))
    _patch_session(monkeypatch, session)
    sem = RecordingSemaphore()
    _patch_endpoint_semaphore(monkeypatch, sem)

    with pytest.raises(UpstreamStreamError) as ei:
        call_upstream("http://test/v1", {"model": "m"}, {}, True)

    assert ei.value.code == "stream_timeout"
    assert len(session.post_calls) == 1
    assert sem.release_calls >= 1


# ---------------------------------------------------------------------------
# PROXY_CAPTURE_UPSTREAM_SSE — safe raw-SSE capture diagnostic
# ---------------------------------------------------------------------------


def test_capture_sse_dumps_raw_bytes_and_does_not_crash_on_bad_stream(monkeypatch, tmp_path):
    """With PROXY_CAPTURE_UPSTREAM_SSE=1, a malformed provider stream must NOT
    propagate an exception: the raw bytes are written to disk and a canonical
    protocol error is returned (clean 502), so one call captures ground truth
    without breaking production. Even a NON-streaming caller (wants_stream=False)
    is forced to stream under the flag.
    """
    monkeypatch.setenv("PROXY_CAPTURE_UPSTREAM_SSE", "1")
    monkeypatch.setenv("PROXY_CAPTURE_UPSTREAM_SSE_DIR", str(tmp_path))

    # A genuinely malformed stream: [DONE] without a finish_reason.
    bad_lines = [
        b'data: {"choices":[{"index":0,"delta":{"role":"assistant"}}]}',
        b"",
        b'data: {"choices":[{"index":0,"delta":{"content":"hi"}}]}',
        b"",
        b"data: [DONE]",
        b"",
    ]
    session = ScriptedSession()
    session.responses.append(StrictResponse(
        status_code=200,
        headers={"Content-Type": "text/event-stream"},
        lines=bad_lines,
    ))
    _patch_session(monkeypatch, session)

    result = call_upstream("http://test/v1", {"model": "m"}, {}, False)

    assert session.post_calls[0]["stream"] is True
    assert session.post_calls[0]["json"]["stream"] is True
    assert isinstance(result, dict)
    assert result.get("error", {}).get("code") == "invalid_chat_completion"

    captures = list(tmp_path.glob("*.sse"))
    assert captures, "no .sse capture file written"
    blob = captures[0].read_bytes()
    assert b'"role":"assistant"' in blob
    assert b'"content":"hi"' in blob
