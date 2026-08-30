"""Tests for opt-in request body dump in http_client.call_upstream.

US-XXX: PROXY_DUMP_REQUESTS opt-in dump of the outbound upstream request
body BEFORE the HTTP POST fires. Captures the actual body the proxy is
about to send, so post-mortem analysis can correlate client intent with
what the backend saw on the wire.

Off by default; enabled via PROXY_DUMP_REQUESTS=1. Override the directory
via PROXY_DUMP_REQUESTS_DIR (default data/debug_requests/, resolved
relative to the repo root).
"""

import json
from unittest.mock import MagicMock, patch

import pytest
import requests

from proxy.http_client import call_upstream


def _make_2xx_response(body=None):
    resp = MagicMock()
    resp.status_code = 200
    resp.json.return_value = body if body is not None else {
        "choices": [{"message": {"role": "assistant", "content": "ok"}}]
    }
    resp.headers = {}
    resp.raise_for_status.return_value = None
    resp.text = json.dumps(resp.json.return_value)
    return resp


class TestDumpEnabled:
    def test_dump_writes_file_when_env_on(self, tmp_path, monkeypatch):
        monkeypatch.setenv("PROXY_DUMP_REQUESTS", "1")
        monkeypatch.setenv("PROXY_DUMP_REQUESTS_DIR", str(tmp_path))

        mock_resp = _make_2xx_response()
        captured = {}

        real_write_text = type(tmp_path).write_text

        def spy_write_text(self, data, *args, **kwargs):
            captured["path"] = self
            captured["content"] = data
            return real_write_text(self, data, *args, **kwargs)

        with patch("proxy.http_client._HTTP_SESSION") as mock_session:
            mock_session.post.return_value = mock_resp
            with patch.object(type(tmp_path), "write_text", spy_write_text):
                result = call_upstream(
                    "https://api.x.com/v1/chat/completions?key=SECRET",
                    {
                        "model": "minimax",
                        "messages": [
                            {"role": "system", "content": "be brief"},
                            {"role": "user", "content": "hello"},
                        ],
                        "tools": [{"type": "function", "function": {"name": "bash"}}],
                    },
                    {"Authorization": "Bearer SECRET"},
                    False,
                )

        assert result == mock_resp.json.return_value

        assert "path" in captured, "expected write_text to have been called"
        assert captured["path"].parent == tmp_path
        assert captured["path"].suffix == ".json"

        payload = json.loads(captured["content"])
        assert set(payload.keys()) >= {
            "timestamp",
            "request_id",
            "url",
            "model",
            "message_count",
            "tool_count",
            "body_bytes",
            "body_sha256",
            "messages_by_role",
            "first_msg_sha256",
            "last_msg_sha256",
            "request_body",
        }

        assert payload["url"] == "https://api.x.com/v1/chat/completions"
        assert "key=SECRET" not in payload["url"]
        assert payload["model"] == "minimax"
        assert payload["message_count"] == 2
        assert payload["tool_count"] == 1
        assert payload["messages_by_role"] == {
            "user": 1,
            "assistant": 0,
            "tool": 0,
            "system": 1,
        }
        assert isinstance(payload["body_bytes"], int)
        assert payload["body_bytes"] > 0
        assert isinstance(payload["body_sha256"], str)
        assert len(payload["body_sha256"]) == 64
        assert isinstance(payload["first_msg_sha256"], str)
        assert isinstance(payload["last_msg_sha256"], str)
        assert payload["request_body"]["model"] == "minimax"

    def test_filename_starts_with_request_id_or_uuid(
        self, tmp_path, monkeypatch
    ):
        monkeypatch.setenv("PROXY_DUMP_REQUESTS", "1")
        monkeypatch.setenv("PROXY_DUMP_REQUESTS_DIR", str(tmp_path))

        mock_resp = _make_2xx_response()
        with patch("proxy.http_client._HTTP_SESSION") as mock_session:
            mock_session.post.return_value = mock_resp
            call_upstream(
                "https://api.x.com/v1/chat",
                {"model": "m", "messages": [{"role": "user", "content": "hi"}]},
                {},
                False,
            )

        files = list(tmp_path.glob("*.json"))
        assert len(files) == 1
        name = files[0].name
        assert name.endswith(".json")
        prefix = name.split("-")[0]
        assert len(prefix) == 12, f"expected 12-char id prefix, got {prefix!r}"

    def test_body_sha256_is_stable_across_dict_order(self, tmp_path, monkeypatch):
        monkeypatch.setenv("PROXY_DUMP_REQUESTS", "1")
        monkeypatch.setenv("PROXY_DUMP_REQUESTS_DIR", str(tmp_path))

        body_a = {"model": "m", "messages": [{"role": "user", "content": "hi"}]}
        body_b = {"messages": [{"role": "user", "content": "hi"}], "model": "m"}

        def _dump_once(body):
            mock_resp = _make_2xx_response()
            with patch("proxy.http_client._HTTP_SESSION") as mock_session:
                mock_session.post.return_value = mock_resp
                call_upstream(
                    "https://api.x.com/v1/chat",
                    body,
                    {},
                    False,
                )
            files = sorted(tmp_path.glob("*.json"))
            assert len(files) == 1
            payload = json.loads(files[0].read_text())
            tmp_path.glob("*.json")  # noqa: B018
            for f in files:
                f.unlink()
            return payload["body_sha256"]

        h1 = _dump_once(body_a)
        h2 = _dump_once(body_b)
        assert h1 == h2, "body_sha256 must use sort_keys for stability"

    def test_first_and_last_msg_hashes_distinct(
        self, tmp_path, monkeypatch
    ):
        monkeypatch.setenv("PROXY_DUMP_REQUESTS", "1")
        monkeypatch.setenv("PROXY_DUMP_REQUESTS_DIR", str(tmp_path))

        body = {
            "model": "m",
            "messages": [
                {"role": "user", "content": "first"},
                {"role": "assistant", "content": "middle"},
                {"role": "user", "content": "last"},
            ],
        }
        mock_resp = _make_2xx_response()
        with patch("proxy.http_client._HTTP_SESSION") as mock_session:
            mock_session.post.return_value = mock_resp
            call_upstream("https://api.x.com/v1/chat", body, {}, False)

        payload = json.loads(list(tmp_path.glob("*.json"))[0].read_text())
        assert payload["first_msg_sha256"] != payload["last_msg_sha256"]


class TestDumpDisabled:
    def test_no_file_when_env_unset(self, tmp_path, monkeypatch):
        monkeypatch.delenv("PROXY_DUMP_REQUESTS", raising=False)
        monkeypatch.setenv("PROXY_DUMP_REQUESTS_DIR", str(tmp_path))

        mock_resp = _make_2xx_response()
        with patch("proxy.http_client._HTTP_SESSION") as mock_session:
            mock_session.post.return_value = mock_resp
            result = call_upstream(
                "https://api.x.com/v1/chat",
                {"model": "m", "messages": [{"role": "user", "content": "hi"}]},
                {},
                False,
            )

        assert result == mock_resp.json.return_value
        assert list(tmp_path.glob("*.json")) == []

    def test_no_file_when_env_zero(self, tmp_path, monkeypatch):
        monkeypatch.setenv("PROXY_DUMP_REQUESTS", "0")
        monkeypatch.setenv("PROXY_DUMP_REQUESTS_DIR", str(tmp_path))

        mock_resp = _make_2xx_response()
        with patch("proxy.http_client._HTTP_SESSION") as mock_session:
            mock_session.post.return_value = mock_resp
            call_upstream(
                "https://api.x.com/v1/chat",
                {"model": "m", "messages": []},
                {},
                False,
            )

        assert list(tmp_path.glob("*.json")) == []


class TestDumpFailureMode:
    def test_write_failure_does_not_break_call(self, tmp_path, monkeypatch):
        monkeypatch.setenv("PROXY_DUMP_REQUESTS", "1")
        monkeypatch.setenv("PROXY_DUMP_REQUESTS_DIR", str(tmp_path))

        mock_resp = _make_2xx_response()

        def boom(self, *args, **kwargs):
            raise OSError("disk full")

        with patch("proxy.http_client._HTTP_SESSION") as mock_session:
            mock_session.post.return_value = mock_resp
            with patch.object(type(tmp_path), "write_text", boom):
                result = call_upstream(
                    "https://api.x.com/v1/chat",
                    {"model": "m", "messages": [{"role": "user", "content": "hi"}]},
                    {},
                    False,
                )

        assert result == mock_resp.json.return_value
        assert mock_session.post.called, "POST must still have been called"

    def test_write_failure_logs_warning(
        self, tmp_path, monkeypatch, caplog
    ):
        monkeypatch.setenv("PROXY_DUMP_REQUESTS", "1")
        monkeypatch.setenv("PROXY_DUMP_REQUESTS_DIR", str(tmp_path))

        mock_resp = _make_2xx_response()

        def boom(self, *args, **kwargs):
            raise OSError("disk full")

        with patch("proxy.http_client._HTTP_SESSION") as mock_session:
            mock_session.post.return_value = mock_resp
            with patch.object(type(tmp_path), "write_text", boom):
                with caplog.at_level("WARNING", logger="proxy.http_client"):
                    call_upstream(
                        "https://api.x.com/v1/chat",
                        {"model": "m", "messages": []},
                        {},
                        False,
                    )

        warning_records = [
            r for r in caplog.records
            if r.name == "proxy.http_client" and r.levelname == "WARNING"
        ]
        assert warning_records, "expected a WARNING log on dump failure"
        assert "Failed to write request debug dump" in warning_records[0].getMessage()


class TestDumpOneAttemptOnly:
    def test_dump_fires_only_on_first_attempt(
        self, tmp_path, monkeypatch
    ):
        """On retry, the same body is resent — the dump must not duplicate it."""
        monkeypatch.setenv("PROXY_DUMP_REQUESTS", "1")
        monkeypatch.setenv("PROXY_DUMP_REQUESTS_DIR", str(tmp_path))

        call_count = {"n": 0}

        def flaky_post(*args, **kwargs):
            call_count["n"] += 1
            if call_count["n"] == 1:
                raise requests.exceptions.ConnectionError("dropped")
            return _make_2xx_response()

        with patch("proxy.http_client._HTTP_SESSION") as mock_session:
            mock_session.post.side_effect = flaky_post
            with patch("proxy.http_client.time.sleep"):
                with patch("proxy.http_client._retry_config") as mock_retry:
                    mock_retry.return_value = {
                        "max_attempts": 3,
                        "base_delay": 0.01,
                        "max_delay": 0.05,
                    }
                    result = call_upstream(
                        "https://api.x.com/v1/chat",
                        {"model": "m", "messages": [{"role": "user", "content": "hi"}]},
                        {},
                        False,
                    )

        assert result is not None
        assert call_count["n"] == 2, "should have retried once"
        files = list(tmp_path.glob("*.json"))
        assert len(files) == 1, (
            f"expected exactly one dump across attempts, got {len(files)}"
        )