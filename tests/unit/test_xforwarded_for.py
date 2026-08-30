"""US-052: X-Forwarded-For validation for non-localhost requests.

The proxy MUST validate X-Forwarded-For headers and MUST NOT trust
spoofed headers from untrusted sources. The rate limiter uses
get_client_ip() for per-key limiting.
"""

from unittest.mock import patch

import pytest
from flask import Flask

from proxy.app import get_client_ip


def _app():
    a = Flask(__name__)
    a.config["TESTING"] = True
    return a


class TestUS052XForwardedForValidation:
    """US-052: get_client_ip() must not blindly trust X-Forwarded-For."""

    def test_xff_ignored_when_no_trusted_proxies_configured(self):
        """X-Forwarded-For from any source is ignored when
        PROXY_TRUSTED_PROXIES is not set — returns remote_addr.
        """
        app = _app()
        with app.test_request_context(
            "/",
            environ_overrides={"REMOTE_ADDR": "203.0.113.1"},
            headers={"X-Forwarded-For": "1.2.3.4"},
        ):
            with patch("proxy.app.TRUSTED_PROXY_NETWORKS", []):
                result = get_client_ip()

        assert result == "203.0.113.1", (
            "Without trusted proxies, X-Forwarded-For must be ignored. "
            f"Expected remote_addr 203.0.113.1, got {result!r}"
        )

    def test_xff_ignored_from_untrusted_remote_addr(self):
        """X-Forwarded-For is ignored when remote_addr is not in
        PROXY_TRUSTED_PROXIES — even if the header is present.
        """
        import ipaddress

        trusted = [ipaddress.ip_network("10.0.0.0/8")]
        app = _app()
        with app.test_request_context(
            "/",
            environ_overrides={"REMOTE_ADDR": "203.0.113.1"},
            headers={"X-Forwarded-For": "1.2.3.4, 10.0.0.1"},
        ):
            with patch("proxy.app.TRUSTED_PROXY_NETWORKS", trusted):
                result = get_client_ip()

        assert result == "203.0.113.1", (
            "Untrusted remote_addr must not propagate X-Forwarded-For. "
            f"Expected 203.0.113.1, got {result!r}"
        )

    def test_xff_trusted_from_localhost_remote_addr(self):
        """When remote_addr is localhost (127.0.0.1), X-Forwarded-For
        is returned as-is — localhost is implicitly trusted as a proxy.
        """
        app = _app()
        with app.test_request_context(
            "/",
            environ_overrides={"REMOTE_ADDR": "127.0.0.1"},
            headers={"X-Forwarded-For": "203.0.113.5"},
        ):
            result = get_client_ip()

        assert result == "127.0.0.1", (
            "Localhost remote_addr returns localhost directly (no chain walk). "
            f"Got {result!r}"
        )

    def test_xff_walks_chain_from_trusted_proxy(self):
        """When remote_addr IS a trusted proxy, the chain is walked
        to find the first non-trusted IP (the real client).
        """
        import ipaddress

        trusted = [ipaddress.ip_network("10.0.0.0/8")]
        app = _app()
        with app.test_request_context(
            "/",
            environ_overrides={"REMOTE_ADDR": "10.0.0.1"},
            headers={"X-Forwarded-For": "203.0.113.99, 10.0.0.2"},
        ):
            with patch("proxy.app.TRUSTED_PROXY_NETWORKS", trusted):
                result = get_client_ip()

        assert result == "203.0.113.99", (
            "Chain walk from trusted proxy must find the first non-trusted IP. "
            f"Expected 203.0.113.99, got {result!r}"
        )

    def test_spoofed_xff_from_untrusted_source_is_rejected(self):
        """An attacker spoofing X-Forwarded-For from an untrusted IP
        cannot inject a fake client IP for rate limiting.
        """
        import ipaddress

        trusted = [ipaddress.ip_network("10.0.0.0/8")]
        app = _app()
        with app.test_request_context(
            "/",
            environ_overrides={"REMOTE_ADDR": "198.51.100.1"},
            headers={"X-Forwarded-For": "127.0.0.1"},
        ):
            with patch("proxy.app.TRUSTED_PROXY_NETWORKS", trusted):
                result = get_client_ip()

        assert result == "198.51.100.1", (
            "Spoofed X-Forwarded-For from untrusted source must be rejected. "
            f"Expected remote_addr 198.51.100.1, got {result!r}"
        )
