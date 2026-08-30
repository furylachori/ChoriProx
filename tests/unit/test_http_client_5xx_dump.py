"""Tests for opt-in 5xx diagnostic dump in http_client.call_upstream.

US-XXX: PROXY_DEBUG_5XX opt-in dump for post-mortem analysis of upstream
5xx errors. The dump captures the full request body, response body,
per-tool-message schema, attempt number, and retry exhaustion flag.
"""

import json
import time
from unittest.mock import MagicMock, patch

import pytest
from requests.exceptions import HTTPError

import proxy.state as state
from proxy.http_client import _dump_5xx_payload, call_upstream


def _make_5xx_response(status_code=520, body=None, text=None):
    resp = MagicMock()
    resp.status_code = status_code
    resp.json.return_value = body if body is not None else {"error": "upstream down"}
    resp.text = text if text is not None else json.dumps(resp.json.return_value)
    resp.headers = {}
    resp.raise_for_status.side_effect = HTTPError(response=resp)
    return resp


@pytest.fixture
def debug_env(tmp_path, monkeypatch):
    monkeypatch.setattr("proxy.http_client._DEBUG_5XX", True)
    monkeypatch.setattr("proxy.http_client._DEBUG_5XX_DIR", str(tmp_path))
    monkeypatch.setattr(state, "request_id", "test-rid-12345")
    return tmp_path


class TestDump5xxPayloadFires:
    def test_dump_fires_on_5xx_response(self, debug_env, tmp_path):
        mock_resp = _make_5xx_response(status_code=520)

        with patch("proxy.http_client._HTTP_SESSION") as mock_session:
            mock_session.post.return_value = mock_resp
            with patch("proxy.http_client.time.sleep"):
                with patch("proxy.http_client._retry_config") as mock_retry:
                    mock_retry.return_value = {
                        "max_attempts": 1,
                        "base_delay": 0.1,
                        "max_delay": 1.0,
                    }
                    with pytest.raises(Exception):
                        call_upstream(
                            "https://api.x.com/v1/chat/completions?key=SECRET",
                            {
                                "model": "m",
                                "messages": [
                                    {"role": "user", "content": "hi"},
                                ],
                            },
                            {"Authorization": "Bearer SECRET"},
                            False,
                        )

        dumps = list(tmp_path.glob("*.json"))
        assert len(dumps) == 1, f"expected exactly one dump file, got {dumps}"
        payload = json.loads(dumps[0].read_text())
        assert payload["status"] == 520
        assert payload["request_body"]["messages"][0]["content"] == "hi"

    def test_dump_does_not_fire_when_debug_disabled(self, tmp_path, monkeypatch):
        monkeypatch.setattr("proxy.http_client._DEBUG_5XX", False)
        monkeypatch.setattr("proxy.http_client._DEBUG_5XX_DIR", str(tmp_path))
        monkeypatch.setattr(state, "request_id", "test-rid-disabled")

        mock_resp = _make_5xx_response(status_code=502)

        with patch("proxy.http_client._HTTP_SESSION") as mock_session:
            mock_session.post.return_value = mock_resp
            with patch("proxy.http_client.time.sleep"):
                with patch("proxy.http_client._retry_config") as mock_retry:
                    mock_retry.return_value = {
                        "max_attempts": 1,
                        "base_delay": 0.1,
                        "max_delay": 1.0,
                    }
                    with pytest.raises(Exception):
                        call_upstream(
                            "https://api.x.com/v1/chat",
                            {"model": "m", "messages": []},
                            {},
                            False,
                        )

        assert list(tmp_path.glob("*.json")) == [], "no dump should be written"

    def test_dump_does_not_fire_on_4xx_response(self, debug_env, tmp_path):
        mock_resp = _make_5xx_response(status_code=400, body={"error": "bad"})

        with patch("proxy.http_client._HTTP_SESSION") as mock_session:
            mock_session.post.return_value = mock_resp
            with patch("proxy.http_client._retry_config") as mock_retry:
                mock_retry.return_value = {
                    "max_attempts": 1,
                    "base_delay": 0.1,
                    "max_delay": 1.0,
                }
                with pytest.raises(Exception):
                    call_upstream(
                        "https://api.x.com/v1/chat",
                        {"model": "m", "messages": []},
                        {},
                        False,
                    )

        assert list(tmp_path.glob("*.json")) == [], "4xx must not trigger dump"


class TestDumpPath:
    def test_filename_uses_state_request_id(self, debug_env, tmp_path):
        _dump_5xx_payload(
            url="https://api.x.com/v1/chat",
            body={"messages": []},
            error_body={"error": "x"},
            status=520,
            attempt=1,
            retry_exhausted=True,
        )

        files = list(tmp_path.glob("*.json"))
        assert len(files) == 1
        name = files[0].name
        assert name.startswith("test-rid-12345-"), f"unexpected filename: {name}"
        assert name.endswith(".json")

    def test_filename_is_unique_across_calls(self, debug_env, tmp_path):
        for _ in range(3):
            _dump_5xx_payload(
                url="https://api.x.com/v1/chat",
                body={"messages": []},
                error_body={"error": "x"},
                status=520,
                attempt=1,
                retry_exhausted=True,
            )
            time.sleep(0.002)

        names = [f.name for f in tmp_path.glob("*.json")]
        assert len(names) == 3
        assert len(set(names)) == 3, "filenames must be unique"

    def test_request_id_fallback_when_state_empty(self, monkeypatch, tmp_path):
        monkeypatch.setattr("proxy.http_client._DEBUG_5XX", True)
        monkeypatch.setattr("proxy.http_client._DEBUG_5XX_DIR", str(tmp_path))
        monkeypatch.setattr(state, "request_id", "")

        _dump_5xx_payload(
            url="https://api.x.com/v1/chat",
            body={"messages": []},
            error_body={"error": "x"},
            status=520,
            attempt=1,
            retry_exhausted=True,
        )

        files = list(tmp_path.glob("*.json"))
        assert len(files) == 1
        name = files[0].name
        prefix = name.split("-")[0]
        assert len(prefix) == 12, f"expected 12-char uuid prefix, got {prefix!r}"


class TestDumpPayload:
    def test_all_documented_fields_present(self, debug_env, tmp_path):
        body = {
            "model": "minimax",
            "messages": [
                {"role": "system", "content": "sys"},
                {"role": "user", "content": "hi"},
                {
                    "role": "tool",
                    "tool_call_id": "call_abc",
                    "name": "search",
                    "content": "result text",
                },
            ],
            "is_subagent": True,
        }
        _dump_5xx_payload(
            url="https://api.x.com/v1/chat/completions?key=SECRET",
            body=body,
            error_body={"error": "upstream boom", "code": 520},
            status=520,
            attempt=2,
            retry_exhausted=True,
        )

        files = list(tmp_path.glob("*.json"))
        assert len(files) == 1
        payload = json.loads(files[0].read_text())

        assert payload["request_id"] == "test-rid-12345"
        assert payload["url"] == "https://api.x.com/v1/chat/completions"
        assert "key=SECRET" not in payload["url"], "query string must be stripped"
        assert payload["method"] == "POST"
        assert "T" in payload["timestamp"] and "+" in payload["timestamp"]
        assert payload["status"] == 520
        assert payload["request_body"] == body
        assert payload["response_body"] == {"error": "upstream boom", "code": 520}
        assert payload["is_subagent"] is True
        assert payload["message_count"] == 3
        assert payload["attempt"] == 2
        assert payload["retry_exhausted"] is True

    def test_url_sanitized_strips_query(self, debug_env, tmp_path):
        _dump_5xx_payload(
            url="https://api.x.com/v1/chat?api_key=SECRET&model=foo",
            body={"messages": []},
            error_body={"error": "x"},
            status=520,
            attempt=1,
            retry_exhausted=True,
        )
        payload = json.loads(list(tmp_path.glob("*.json"))[0].read_text())
        assert payload["url"] == "https://api.x.com/v1/chat"

    def test_is_subagent_none_when_absent(self, debug_env, tmp_path):
        _dump_5xx_payload(
            url="https://api.x.com/v1/chat",
            body={"messages": []},
            error_body={"error": "x"},
            status=520,
            attempt=1,
            retry_exhausted=True,
        )
        payload = json.loads(list(tmp_path.glob("*.json"))[0].read_text())
        assert payload["is_subagent"] is None


class TestNowIsoMigration:
    """M2-2 (Batch 1 follow-up): http_client.py must use the canonical
    ``_now_iso()`` helper from ``proxy.utils`` for ISO timestamps.

    Pre-fix, 3 sites in http_client.py built ISO timestamps inline as
    ``datetime.now(timezone.utc).isoformat()``. Migrating to the helper
    keeps the timestamp format centralized and grep-detectable.

    These tests pin the migration by directly importing the module
    attribute and asserting it points at the canonical helper.
    """

    def test_http_client_imports_canonical_now_iso(self):
        """http_client must re-export _now_iso from proxy.utils."""
        from proxy import http_client
        from proxy.utils import _now_iso as canonical_now_iso

        assert hasattr(http_client, "_now_iso"), (
            "http_client.py must import _now_iso from proxy.utils (M2-2 "
            "migration). Without this import, the 3 timestamp sites fall "
            "back to inline datetime.now(timezone.utc).isoformat() calls "
            "and diverge from the canonical helper."
        )
        assert http_client._now_iso is canonical_now_iso, (
            "http_client._now_iso must be the same function object as "
            "proxy.utils._now_iso (no local redefinition or aliasing)."
        )

    def test_no_inline_isoformat_in_http_client(self):
        """Belt-and-suspenders: the inline `now.isoformat()` pattern must
        not have crept back into http_client.py after the migration.

        The only allowed call is via ``_now_iso()`` from proxy.utils.
        """
        from proxy import http_client

        source = open(http_client.__file__).read()
        # `.isoformat()` is what `_now_iso` returns internally — but the
        # file should NOT build it inline at the call sites.
        assert "now.isoformat()" not in source, (
            "http_client.py still has inline `now.isoformat()` calls. "
            "Migrate to `_now_iso()` from proxy.utils."
        )


class TestPerToolMessageSchema:
    def test_extracts_tool_messages_with_schema(self, debug_env, tmp_path):
        body = {
            "messages": [
                {"role": "user", "content": "do thing"},
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {"name": "search", "arguments": "{}"},
                        }
                    ],
                },
                {
                    "role": "tool",
                    "tool_call_id": "call_1",
                    "name": "search",
                    "content": "search result 12345",
                },
                {
                    "role": "tool",
                    "tool_call_id": "call_2",
                    "content": "no-name tool",
                },
                {
                    "role": "assistant",
                    "content": "done",
                },
            ]
        }
        _dump_5xx_payload(
            url="https://api.x.com/v1/chat",
            body=body,
            error_body={"error": "x"},
            status=520,
            attempt=1,
            retry_exhausted=True,
        )
        payload = json.loads(list(tmp_path.glob("*.json"))[0].read_text())
        schema = payload["per_tool_message_schema"]

        assert len(schema) == 2
        assert schema[0] == {
            "index": 2,
            "role": "tool",
            "tool_call_id": "call_1",
            "name": "search",
            "content_len": 19,
            "has_name": True,
        }
        assert schema[1] == {
            "index": 3,
            "role": "tool",
            "tool_call_id": "call_2",
            "name": None,
            "content_len": 12,
            "has_name": False,
        }

    def test_no_tool_messages_yields_empty_schema(self, debug_env, tmp_path):
        body = {
            "messages": [
                {"role": "system", "content": "sys"},
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": "hello"},
            ]
        }
        _dump_5xx_payload(
            url="https://api.x.com/v1/chat",
            body=body,
            error_body={"error": "x"},
            status=520,
            attempt=1,
            retry_exhausted=True,
        )
        payload = json.loads(list(tmp_path.glob("*.json"))[0].read_text())
        assert payload["per_tool_message_schema"] == []
        assert payload["message_count"] == 3

    def test_tool_message_with_list_content(self, debug_env, tmp_path):
        body = {
            "messages": [
                {
                    "role": "tool",
                    "tool_call_id": "call_x",
                    "name": "fetch",
                    "content": [{"text": "a"}, {"text": "b"}],
                }
            ]
        }
        _dump_5xx_payload(
            url="https://api.x.com/v1/chat",
            body=body,
            error_body={"error": "x"},
            status=520,
            attempt=1,
            retry_exhausted=True,
        )
        payload = json.loads(list(tmp_path.glob("*.json"))[0].read_text())
        assert payload["per_tool_message_schema"][0]["content_len"] == 2


class TestDumpDoesNotAlterRetryPath:
    def test_5xx_still_raises_backend_error(self, debug_env, tmp_path):
        from proxy.types import BackendError

        mock_resp = _make_5xx_response(status_code=502)

        with patch("proxy.http_client._HTTP_SESSION") as mock_session:
            mock_session.post.return_value = mock_resp
            with patch("proxy.http_client._retry_config") as mock_retry:
                mock_retry.return_value = {
                    "max_attempts": 1,
                    "base_delay": 0.1,
                    "max_delay": 1.0,
                }
                with pytest.raises(BackendError) as exc_info:
                    call_upstream(
                        "https://api.x.com/v1/chat",
                        {"messages": []},
                        {},
                        False,
                    )

        assert exc_info.value.status_code == 502
        assert list(tmp_path.glob("*.json")), "dump should still have fired"

    def test_retry_exhausted_flag_reflects_last_attempt(self, tmp_path, monkeypatch):
        monkeypatch.setattr("proxy.http_client._DEBUG_5XX", True)
        monkeypatch.setattr("proxy.http_client._DEBUG_5XX_DIR", str(tmp_path))
        monkeypatch.setattr(state, "request_id", "rid-retry")

        _dump_5xx_payload(
            url="https://api.x.com/v1/chat",
            body={"messages": []},
            error_body={"error": "x"},
            status=520,
            attempt=1,
            retry_exhausted=False,
        )
        _dump_5xx_payload(
            url="https://api.x.com/v1/chat",
            body={"messages": []},
            error_body={"error": "x"},
            status=520,
            attempt=2,
            retry_exhausted=False,
        )
        _dump_5xx_payload(
            url="https://api.x.com/v1/chat",
            body={"messages": []},
            error_body={"error": "x"},
            status=520,
            attempt=3,
            retry_exhausted=True,
        )

        payloads = [
            json.loads(f.read_text()) for f in sorted(tmp_path.glob("*.json"))
        ]
        assert [p["attempt"] for p in payloads] == [1, 2, 3]
        assert [p["retry_exhausted"] for p in payloads] == [False, False, True]


class TestBodySummaryLog:
    def test_body_summary_logged_on_5xx(self, caplog, monkeypatch):
        monkeypatch.setattr("proxy.http_client._DEBUG_5XX", False)
        body = {
            "model": "mini-m3",
            "messages": [
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": "hello"},
            ],
            "tools": [{"type": "function", "function": {"name": "bash"}}],
            "stream": True,
        }
        with caplog.at_level("ERROR", logger="proxy.http_client"):
            with patch("proxy.http_client._HTTP_SESSION") as session_mock:
                session_mock.post.return_value = _make_5xx_response(
                    status_code=500,
                    body={"type": "error", "error": {"message": "unknown error, 512 (1000)"}},
                )
                with patch("proxy.http_client._get_endpoint_semaphore", return_value=None):
                    from proxy.types import BackendError
                    with pytest.raises(BackendError):
                        call_upstream(
                            "https://api.minimax.io/v1/chat/completions",
                            body,
                            headers={},
                            wants_stream=True,
                        )

        summary_records = [
            r for r in caplog.records
            if r.name == "proxy.http_client" and "upstream_5xx_body_summary" in r.getMessage()
        ]
        assert len(summary_records) == 1
        logged = json.loads(summary_records[0].getMessage())
        assert logged["event"] == "upstream_5xx_body_summary"
        assert logged["status"] == 500
        assert logged["summary"]["model"] == "mini-m3"
        assert logged["summary"]["message_count"] == 2
        assert logged["summary"]["tool_count"] == 1
        assert logged["summary"]["stream"] is True
        assert "hi" in logged["summary"]["request_body_preview"]
        assert "PROXY_DEBUG_5XX=1" in logged["summary"]["hint"]

    def test_body_summary_not_logged_on_4xx(self, caplog, monkeypatch):
        monkeypatch.setattr("proxy.http_client._DEBUG_5XX", False)
        body = {"model": "m", "messages": [{"role": "user", "content": "x"}]}
        with caplog.at_level("ERROR", logger="proxy.http_client"):
            with patch("proxy.http_client._HTTP_SESSION") as session_mock:
                session_mock.post.return_value = _make_5xx_response(
                    status_code=400, body={"error": "bad request"}
                )
                with patch("proxy.http_client._get_endpoint_semaphore", return_value=None):
                    from proxy.types import BackendError
                    with pytest.raises(BackendError):
                        call_upstream(
                            "https://api.x.com/v1/chat", body, headers={}, wants_stream=False
                        )

        summary_records = [
            r for r in caplog.records
            if r.name == "proxy.http_client" and "upstream_5xx_body_summary" in r.getMessage()
        ]
        assert summary_records == []
