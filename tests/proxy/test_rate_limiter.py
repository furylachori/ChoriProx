import threading
from unittest.mock import patch

from proxy.rate_limiter import RateLimiter, RateLimitResult


def test_check_allows_under_limit():
    limiter = RateLimiter(max_requests=3, window_seconds=10.0)
    with patch("proxy.rate_limiter.time.time", return_value=100.0):
        r1 = limiter.check("k")
        r2 = limiter.check("k")
        r3 = limiter.check("k")
    assert r1.allowed is True
    assert r2.allowed is True
    assert r3.allowed is True


def test_check_denies_over_limit():
    limiter = RateLimiter(max_requests=2, window_seconds=10.0)
    with patch("proxy.rate_limiter.time.time", return_value=100.0):
        limiter.check("k")
        limiter.check("k")
        result = limiter.check("k")
    assert result.allowed is False
    assert result.retry_after == 10.0


def test_check_only_does_not_record():
    limiter = RateLimiter(max_requests=2, window_seconds=10.0)
    with patch("proxy.rate_limiter.time.time", return_value=100.0):
        r1 = limiter.check_only("k")
        r2 = limiter.check_only("k")
        r3 = limiter.check_only("k")
        r4 = limiter.check("k")
    assert r1.allowed is True
    assert r2.allowed is True
    assert r3.allowed is True
    assert r4.allowed is True


def test_sliding_window_expires_old_entries():
    limiter = RateLimiter(max_requests=2, window_seconds=10.0)
    with patch("proxy.rate_limiter.time.time", return_value=100.0):
        limiter.check("k")
        limiter.check("k")
        denied = limiter.check("k")
        assert denied.allowed is False

    with patch("proxy.rate_limiter.time.time", return_value=111.0):
        result = limiter.check("k")
    assert result.allowed is True


def test_per_key_isolation():
    limiter = RateLimiter(max_requests=1, window_seconds=10.0)
    with patch("proxy.rate_limiter.time.time", return_value=100.0):
        a1 = limiter.check("A")
        a2 = limiter.check("A")
        b1 = limiter.check("B")
    assert a1.allowed is True
    assert a2.allowed is False
    assert b1.allowed is True


def test_cleanup_removes_expired_entries():
    limiter = RateLimiter(max_requests=5, window_seconds=10.0)
    with patch("proxy.rate_limiter.time.time", return_value=100.0):
        limiter.check("k1")
        limiter.check("k2")

    with patch("proxy.rate_limiter.time.time", return_value=111.0):
        limiter.cleanup()

    assert "k1" not in limiter._entries
    assert "k2" not in limiter._entries


def test_cleanup_preserves_active_entries():
    limiter = RateLimiter(max_requests=5, window_seconds=10.0)
    with patch("proxy.rate_limiter.time.time", return_value=100.0):
        limiter.check("k1")

    with patch("proxy.rate_limiter.time.time", return_value=105.0):
        limiter.cleanup()

    assert "k1" in limiter._entries
    assert len(limiter._entries["k1"]) == 1


def test_thread_safety_concurrent_checks():
    limiter = RateLimiter(max_requests=50, window_seconds=10.0)
    results = []
    errors = []
    barrier = threading.Barrier(5)

    def worker():
        try:
            barrier.wait()
            for _ in range(20):
                r = limiter.check("shared")
                results.append(r)
        except Exception as e:
            errors.append(e)

    with patch("proxy.rate_limiter.time.time", return_value=100.0):
        threads = [threading.Thread(target=worker) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

    assert len(errors) == 0
    assert len(results) == 100
    allowed_count = sum(1 for r in results if r.allowed)
    denied_count = sum(1 for r in results if not r.allowed)
    assert allowed_count == 50
    assert denied_count == 50


def test_retry_after_calculation():
    limiter = RateLimiter(max_requests=2, window_seconds=10.0)
    with patch("proxy.rate_limiter.time.time", return_value=100.0):
        limiter.check("k")

    with patch("proxy.rate_limiter.time.time", return_value=103.0):
        limiter.check("k")
        result = limiter.check("k")

    assert result.allowed is False
    assert result.retry_after == 7.0


def test_rate_limit_result_dataclass():
    allowed_result = RateLimitResult(allowed=True)
    assert allowed_result.allowed is True
    assert allowed_result.retry_after == 0.0

    denied_result = RateLimitResult(allowed=False, retry_after=5.5)
    assert denied_result.allowed is False
    assert denied_result.retry_after == 5.5
