"""Load tests for ChoriProx proxy endpoints.

Run with: PYTHONPATH=src python -m pytest tests/load/ -v --tb=short

These tests measure throughput and latency under concurrent load
using Flask test client with mocked upstream backends.
"""
import time
import concurrent.futures
import statistics
import pytest


def _make_chat_request(client, session_id=None):
    """Make a single chat completion request and return (status_code, latency)."""
    start = time.time()
    response = client.post('/v1/chat/completions', json={
        "model": "test-model",
        "messages": [{"role": "user", "content": "Hello, load test!"}],
        "stream": False,
    })
    elapsed = time.time() - start
    return response.status_code, elapsed


def _make_health_request(client):
    """Make a single health check request and return (status_code, latency)."""
    start = time.time()
    response = client.get('/health')
    elapsed = time.time() - start
    return response.status_code, elapsed


def _make_models_request(client):
    """Make a single models list request and return (status_code, latency)."""
    start = time.time()
    response = client.get('/v1/models')
    elapsed = time.time() - start
    return response.status_code, elapsed


def _run_concurrent(client, request_fn, num_requests, max_workers):
    """Run concurrent requests and collect results."""
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(request_fn, client) for _ in range(num_requests)]
        results = [f.result() for f in concurrent.futures.as_completed(futures)]
    return results


def _compute_stats(results):
    """Compute statistics from results list of (status_code, latency)."""
    status_codes = [r[0] for r in results]
    latencies = sorted([r[1] for r in results])
    n = len(latencies)
    
    server_errors = sum(1 for c in status_codes if c >= 500 and c != 503)
    rate_limited = sum(1 for c in status_codes if c == 429)
    degraded = sum(1 for c in status_codes if c == 503)
    success = sum(1 for c in status_codes if c == 200)
    
    return {
        "total": n,
        "success": success,
        "server_errors": server_errors,
        "rate_limited": rate_limited,
        "degraded": degraded,
        "status_codes": dict(
            (code, status_codes.count(code)) for code in set(status_codes)
        ),
        "mean": statistics.mean(latencies),
        "median": statistics.median(latencies),
        "stdev": statistics.stdev(latencies) if n > 1 else 0.0,
        "min": latencies[0],
        "max": latencies[-1],
        "p50": latencies[int(n * 0.50)],
        "p95": latencies[int(n * 0.95)] if n > 1 else latencies[-1],
        "p99": latencies[int(n * 0.99)] if n > 1 else latencies[-1],
        "total_time": sum(latencies),
    }


class TestLoadChatCompletions:
    """Load tests for POST /v1/chat/completions."""

    def test_burst_10_concurrent(self, client):
        """10 concurrent requests should all succeed within 5 seconds."""
        results = _run_concurrent(client, _make_chat_request, 10, 10)
        stats = _compute_stats(results)

        assert stats["server_errors"] == 0, (
            f"Got {stats['server_errors']} server errors: {stats['status_codes']}"
        )
        assert stats["mean"] < 5.0, f"Mean latency {stats['mean']:.2f}s exceeds 5s"
        assert stats["max"] < 10.0, f"Max latency {stats['max']:.2f}s exceeds 10s"

    def test_burst_50_concurrent(self, client):
        """50 concurrent requests — measure throughput and error rate."""
        results = _run_concurrent(client, _make_chat_request, 50, 50)
        stats = _compute_stats(results)

        error_rate = stats["server_errors"] / stats["total"]
        assert error_rate < 0.1, (
            f"Error rate {error_rate:.1%} exceeds 10%: {stats['status_codes']}"
        )
        assert stats["p95"] < 10.0, f"p95 latency {stats['p95']:.2f}s exceeds 10s"

    def test_sustained_100_sequential(self, client):
        """100 sequential requests — measure p50/p95/p99 latency distribution."""
        results = []
        for _ in range(100):
            results.append(_make_chat_request(client))
        stats = _compute_stats(results)

        assert stats["server_errors"] == 0, (
            f"Got {stats['server_errors']} server errors: {stats['status_codes']}"
        )
        assert stats["p50"] < 3.0, f"p50 latency {stats['p50']:.2f}s exceeds 3s"
        assert stats["p95"] < 5.0, f"p95 latency {stats['p95']:.2f}s exceeds 5s"
        assert stats["p99"] < 10.0, f"p99 latency {stats['p99']:.2f}s exceeds 10s"

    def test_concurrent_25_workers_100_requests(self, client):
        """100 requests with 25 concurrent workers — mixed load pattern."""
        results = _run_concurrent(client, _make_chat_request, 100, 25)
        stats = _compute_stats(results)

        error_rate = stats["server_errors"] / stats["total"]
        assert error_rate < 0.15, (
            f"Error rate {error_rate:.1%} exceeds 15%: {stats['status_codes']}"
        )
        assert stats["mean"] < 5.0, f"Mean latency {stats['mean']:.2f}s exceeds 5s"

    def test_streaming_requests_concurrent(self, client):
        """10 concurrent streaming requests should all succeed."""
        def make_streaming_request(c):
            start = time.time()
            response = c.post('/v1/chat/completions', json={
                "model": "test-model",
                "messages": [{"role": "user", "content": "Stream test"}],
                "stream": True,
            })
            elapsed = time.time() - start
            return response.status_code, elapsed

        results = _run_concurrent(client, make_streaming_request, 10, 10)
        stats = _compute_stats(results)

        assert stats["server_errors"] == 0, (
            f"Got {stats['server_errors']} server errors: {stats['status_codes']}"
        )

    def test_large_payload_requests(self, client):
        """5 concurrent requests with large message payloads."""
        large_content = "x" * 10000

        def make_large_request(c):
            start = time.time()
            response = c.post('/v1/chat/completions', json={
                "model": "test-model",
                "messages": [
                    {"role": "system", "content": large_content},
                    {"role": "user", "content": large_content},
                ],
                "stream": False,
            })
            elapsed = time.time() - start
            return response.status_code, elapsed

        results = _run_concurrent(client, make_large_request, 5, 5)
        stats = _compute_stats(results)

        assert stats["server_errors"] == 0, (
            f"Got {stats['server_errors']} server errors: {stats['status_codes']}"
        )


class TestLoadHealth:
    """Load tests for GET /health."""

    def test_concurrent_health_checks_100(self, client):
        """100 concurrent health checks should return 200 or 503 (degraded)."""
        results = _run_concurrent(client, _make_health_request, 100, 100)
        stats = _compute_stats(results)

        assert stats["server_errors"] == 0, (
            f"Got {stats['server_errors']} server errors: {stats['status_codes']}"
        )
        assert stats["mean"] < 2.0, f"Mean latency {stats['mean']:.2f}s exceeds 2s"
        assert stats["max"] < 5.0, f"Max latency {stats['max']:.2f}s exceeds 5s"

    def test_sustained_health_500(self, client):
        """500 sequential health checks — measure stability."""
        results = []
        for _ in range(500):
            results.append(_make_health_request(client))
        stats = _compute_stats(results)

        assert stats["server_errors"] == 0, (
            f"Got {stats['server_errors']} server errors: {stats['status_codes']}"
        )
        assert stats["p99"] < 2.0, f"p99 latency {stats['p99']:.2f}s exceeds 2s"


class TestLoadModels:
    """Load tests for GET /v1/models."""

    def test_concurrent_models_list_50(self, client):
        """50 concurrent model list requests should all succeed."""
        results = _run_concurrent(client, _make_models_request, 50, 50)
        stats = _compute_stats(results)

        assert stats["server_errors"] == 0, (
            f"Got {stats['server_errors']} server errors: {stats['status_codes']}"
        )
        assert stats["mean"] < 3.0, f"Mean latency {stats['mean']:.2f}s exceeds 3s"


class TestLoadMixed:
    """Mixed endpoint load tests."""

    def test_chat_during_health_checks(self, client):
        """Concurrent chat and health requests should not interfere."""
        def run_both():
            chat_result = _make_chat_request(client)
            health_result = _make_health_request(client)
            return chat_result, health_result

        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(run_both) for _ in range(10)]
            all_results = [f.result() for f in concurrent.futures.as_completed(futures)]

        chat_results = [r[0] for r in all_results]
        health_results = [r[1] for r in all_results]

        chat_stats = _compute_stats(chat_results)
        health_stats = _compute_stats(health_results)

        assert chat_stats["server_errors"] == 0, (
            f"Chat server errors: {chat_stats['status_codes']}"
        )
        assert health_stats["server_errors"] == 0, (
            f"Health server errors: {health_stats['status_codes']}"
        )
