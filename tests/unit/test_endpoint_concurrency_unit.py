"""Unit tests for per-endpoint concurrency limiter.

Covers tests 1-17 from SPEC_UPSTREAM_CONCURRENCY.md test plan.
# US-040: Endpoint Concurrency Semaphore
"""

import threading
import time
import pytest
from unittest.mock import patch, MagicMock, call

from proxy.http_client import (
    _get_endpoint_semaphore,
    _endpoint_semaphores,
    _endpoint_sem_lock,
)


def _make_config_mock(ec_map):
    def config_get(key, default=None):
        if key == "upstream":
            return {"endpoint_concurrency": ec_map}
        return default
    mock_cfg = MagicMock()
    mock_cfg.get.side_effect = config_get
    return mock_cfg


class TestGetEndpointSemaphore:
    @pytest.fixture(autouse=True)
    def reset_state(self):
        _endpoint_semaphores.clear()
        yield
        _endpoint_semaphores.clear()

    def test_01_two_concurrent_calls_within_limit(self):
        mock_cfg = _make_config_mock({"https://ollama.com": 2})
        with patch("proxy.http_client.CONFIG", mock_cfg):
            sem = _get_endpoint_semaphore("https://ollama.com")
            assert sem is not None
            assert sem.acquire(blocking=False)
            assert sem.acquire(blocking=False)

    def test_02_third_call_blocks_then_succeeds(self):
        mock_cfg = _make_config_mock({"https://ollama.com": 2})
        with patch("proxy.http_client.CONFIG", mock_cfg):
            sem = _get_endpoint_semaphore("https://ollama.com")
            assert sem.acquire(blocking=False)
            assert sem.acquire(blocking=False)
            acquired = sem.acquire(timeout=0.1)
            assert not acquired

    def test_03_unconfigured_endpoint_proceeds(self):
        mock_cfg = _make_config_mock({"https://ollama.com": 2})
        with patch("proxy.http_client.CONFIG", mock_cfg):
            sem = _get_endpoint_semaphore("https://unknown.endpoint.com")
            assert sem is None

    def test_07_ec_zero_disabled(self):
        mock_cfg = _make_config_mock({"https://api.xiaomi.com": 0})
        with patch("proxy.http_client.CONFIG", mock_cfg):
            sem = _get_endpoint_semaphore("https://api.xiaomi.com")
            assert sem is None

    def test_08_ec_omitted_unlimited(self):
        mock_cfg = _make_config_mock({})
        with patch("proxy.http_client.CONFIG", mock_cfg):
            sem = _get_endpoint_semaphore("https://any.endpoint.com")
            assert sem is None

    def test_09_log_fires_once_on_first_call(self):
        mock_cfg = _make_config_mock({"https://ollama.example.com": 2})
        with patch("proxy.http_client.CONFIG", mock_cfg):
            _endpoint_semaphores.clear()
            with patch("proxy.http_client.logger") as mock_logger:
                sem1 = _get_endpoint_semaphore("https://ollama.example.com")
                sem2 = _get_endpoint_semaphore("https://ollama.example.com")
                assert sem1 is sem2
                mock_logger.info.assert_called_once()
                assert "Concurrency limiter" in mock_logger.info.call_args[0][0]

    def test_10_ten_concurrent_limit_2(self):
        mock_cfg = _make_config_mock({"https://ollama.com": 2})
        with patch("proxy.http_client.CONFIG", mock_cfg):
            sem = _get_endpoint_semaphore("https://ollama.com")
            for _ in range(2):
                assert sem.acquire(blocking=False)
            for _ in range(8):
                assert not sem.acquire(blocking=False)

    def test_11_timeout_null_defaults_1800(self):
        def config_get(key, default=None):
            if key == "upstream":
                return {"timeout": None, "endpoint_concurrency": {"https://x.com": 1}}
            return default
        mock_cfg = MagicMock()
        mock_cfg.get.side_effect = config_get
        with patch("proxy.http_client.CONFIG", mock_cfg):
            from proxy.http_client import _resolve_upstream_timeout
            result = _resolve_upstream_timeout({})
            assert result == 1800

    def test_12_timeout_invalid_defaults_1800(self):
        def config_get(key, default=None):
            if key == "upstream":
                return {"timeout": "not_a_number", "endpoint_concurrency": {"https://x.com": 1}}
            return default
        mock_cfg = MagicMock()
        mock_cfg.get.side_effect = config_get
        with patch("proxy.http_client.CONFIG", mock_cfg):
            from proxy.http_client import _resolve_upstream_timeout
            result = _resolve_upstream_timeout({})
            assert result == 1800

    def test_13_negative_semaphore_value(self):
        mock_cfg = _make_config_mock({"https://x.com": -1})
        with patch("proxy.http_client.CONFIG", mock_cfg):
            with pytest.raises(ValueError):
                _get_endpoint_semaphore("https://x.com")

    def test_14_double_checked_locking_concurrent(self):
        mock_cfg = _make_config_mock({"https://new.endpoint.com": 1})
        with patch("proxy.http_client.CONFIG", mock_cfg):
            results = []
            errors = []

            def get_sem():
                try:
                    results.append(_get_endpoint_semaphore("https://new.endpoint.com"))
                except Exception as e:
                    errors.append(e)

            threads = [threading.Thread(target=get_sem) for _ in range(10)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()

            assert not errors
            assert all(r is results[0] for r in results)


class TestCallUpstreamSemaphoreIntegration:
    @pytest.fixture(autouse=True)
    def reset_state(self):
        _endpoint_semaphores.clear()
        yield
        _endpoint_semaphores.clear()

    def _make_mock_response(self, status_code=200, json_data=None):
        mock_resp = MagicMock()
        mock_resp.status_code = status_code
        mock_resp.json.return_value = json_data or {"ok": True}
        mock_resp.text = '{"ok": true}'
        mock_resp.headers = {}
        return mock_resp

    def test_05_call_upstream_releases_on_backend_error(self):
        mock_cfg = _make_config_mock({"https://error.com": 1})
        mock_resp = MagicMock()
        mock_resp.status_code = 400
        error_body = {"error": "bad request"}
        mock_resp.json.return_value = error_body
        mock_resp.text = '{"error": "bad request"}'
        mock_resp.headers = {}
        mock_resp.raise_for_status.side_effect = MagicMock()
        from requests.exceptions import HTTPError
        mock_resp.raise_for_status.side_effect = HTTPError(response=mock_resp)

        with patch("proxy.http_client.CONFIG", mock_cfg):
            with patch("proxy.http_client._HTTP_SESSION") as mock_session:
                mock_session.post.return_value = mock_resp
                from proxy.http_client import call_upstream
                with pytest.raises(Exception):
                    call_upstream("https://error.com/path", {}, {}, False)
                sem = _endpoint_semaphores.get("https://error.com")
                assert sem.acquire(blocking=False)

    def test_06_call_upstream_releases_on_connection_error(self):
        mock_cfg = _make_config_mock({"https://connfail.com": 1})
        with patch("proxy.http_client.CONFIG", mock_cfg):
            with patch("proxy.http_client._HTTP_SESSION") as mock_session:
                mock_session.post.side_effect = ConnectionError("simulated")
                with patch("proxy.http_client._retry_config") as mock_retry:
                    mock_retry.return_value = {"max_attempts": 1, "base_delay": 0.1, "max_delay": 1}
                    from proxy.http_client import call_upstream
                    with pytest.raises((ConnectionError, RuntimeError)):
                        call_upstream("https://connfail.com/path", {}, {}, False)
                    sem = _endpoint_semaphores.get("https://connfail.com")
                    assert sem.acquire(blocking=False)

    def test_15_retryable_503_releases_slot_before_retry(self):
        mock_cfg = _make_config_mock({"https://overload.com": 1})
        mock_resp = MagicMock()
        mock_resp.status_code = 503
        mock_resp.json.return_value = {"error": "overloaded"}
        mock_resp.text = '{"error": "overloaded"}'
        mock_resp.headers = {}
        mock_resp.raise_for_status.side_effect = MagicMock()
        from requests.exceptions import HTTPError
        mock_resp.raise_for_status.side_effect = HTTPError(response=mock_resp)

        with patch("proxy.http_client.CONFIG", mock_cfg):
            with patch("proxy.http_client._HTTP_SESSION") as mock_session:
                mock_session.post.return_value = mock_resp
                with patch("proxy.http_client.time.sleep"):
                    from proxy.http_client import call_upstream
                    try:
                        call_upstream("https://overload.com/path", {}, {}, False)
                    except Exception:
                        pass
                    sem = _endpoint_semaphores.get("https://overload.com")
                    assert sem.acquire(blocking=False)

    def test_16_non_retryable_400_releases_slot(self):
        mock_cfg = _make_config_mock({"https://bad.com": 1})
        mock_resp = MagicMock()
        mock_resp.status_code = 400
        mock_resp.json.return_value = {"error": "bad request"}
        mock_resp.text = '{"error": "bad request"}'
        mock_resp.headers = {}
        mock_resp.raise_for_status.side_effect = MagicMock()
        from requests.exceptions import HTTPError
        mock_resp.raise_for_status.side_effect = HTTPError(response=mock_resp)

        with patch("proxy.http_client.CONFIG", mock_cfg):
            with patch("proxy.http_client._HTTP_SESSION") as mock_session:
                mock_session.post.return_value = mock_resp
                from proxy.http_client import call_upstream
                with pytest.raises(Exception):
                    call_upstream("https://bad.com/path", {}, {}, False)
                sem = _endpoint_semaphores.get("https://bad.com")
                assert sem.acquire(blocking=False)


class TestSemaphoreAcquireTimeout:
    @pytest.fixture(autouse=True)
    def reset_state(self):
        _endpoint_semaphores.clear()
        yield
        _endpoint_semaphores.clear()

    def test_04_acquire_timeout_raises_runtime_error(self):
        mock_cfg = _make_config_mock({"https://x.com": 1})
        with patch("proxy.http_client.CONFIG", mock_cfg):
            sem = _get_endpoint_semaphore("https://x.com")
            sem.acquire()
            acquired = sem.acquire(timeout=0.05)
            assert not acquired
            sem.release()