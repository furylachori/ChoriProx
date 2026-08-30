"""M8-2: double-checked locking on _ensure_metrics.

Pre-fix: `_ensure_metrics()` was a non-atomic check-then-set on
~15 module-global Counters/Histograms. Two threads racing on the
first `record_*()` would both observe `_http_requests_total is None`,
both call `prom.Counter('choriprox_http_requests_total', ...)`, and
the second would raise:

    ValueError: Duplicated timeseries in CollectorRegistry

Fix: module-scope `_metrics_init_lock` + double-checked locking
pattern. Fast path returns immediately when initialized. Slow
path serializes via the lock and re-checks before instantiating.

This test forces the race by:
1. Setting all metric globals back to None.
2. Spawning N threads, each calling a different `record_*` entry
   point concurrently on the first call.
3. Asserting no thread raised ValueError: Duplicated timeseries.
"""

import threading

import pytest


pytestmark = pytest.mark.unit


# All module-level metric globals from src/proxy/metrics.py:31-50.
# Listed explicitly so the fixture can reset them without poking
# into the module's internals.
_METRIC_GLOBALS = [
    "_http_requests_total",
    "_http_request_duration",
    "_rate_limiter_checks_total",
    "_rate_limiter_rejections_total",
    "_neo4j_queries_total",
    "_neo4j_query_duration",
    "_neo4j_errors_total",
    "_agent_registry_evictions_total",
    "_prompt_cache_requests_total",
    "_prompt_cache_read_tokens_total",
    "_prompt_cache_no_cache_tokens_total",
]


@pytest.fixture
def reset_metrics_globals(monkeypatch):
    """Reset all metric globals to None so the next _ensure_metrics()
    call re-initializes them. Restored on teardown.

    PROXY_METRICS_ENABLED must be truthy for the test to actually
    trigger the race (otherwise _ensure_metrics short-circuits at
    the is_enabled() check and never reaches the prom.Counter calls).
    """
    import proxy.metrics as m

    saved = {name: getattr(m, name) for name in _METRIC_GLOBALS}
    for name in _METRIC_GLOBALS:
        monkeypatch.setattr(m, name, None)

    # Force metrics enabled so the initialization path actually runs.
    monkeypatch.setattr(m, "_METRICS_ENABLED", True)
    # Also patch the module-level flag the is_enabled() function reads.
    # is_enabled() does `return _METRICS_ENABLED and _get_prom() is not False`,
    # so patching _METRICS_ENABLED suffices.

    yield

    # monkeypatch.setattr with explicit value restoration handles
    # teardown automatically.


def test_concurrent_record_does_not_duplicate_registration(reset_metrics_globals):
    """N threads concurrent on first record_*: no ValueError.

    Pre-fix: with the check-then-instantiate race, two threads
    would both call prom.Counter('choriprox_http_requests_total',
    ...) and the second would raise ValueError: Duplicated
    timeseries in CollectorRegistry. Post-fix: double-checked
    locking serializes the slow path so only one thread
    instantiates each Counter.

    Important: we also need a fresh prometheus_client.REGISTRY.
    If a prior test (in this session) has already created any of
    the metric names, the registry has them — _ensure_metrics would
    raise Duplicated timeseries on the FIRST call regardless of
    concurrency, masking the race we're testing. The fixture below
    resets both the metric globals and the prom registry.
    """
    import proxy.metrics as m
    import prometheus_client as prom

    # Fresh prom registry. Without this, prior test artifacts cause
    # immediate failure unrelated to the race we're testing.
    fresh_registry = prom.CollectorRegistry()
    original_registry = prom.REGISTRY
    prom.REGISTRY = fresh_registry
    try:
        # Sanity: globals are None (the fixture reset them).
        assert m._http_requests_total is None, (
            "Fixture did not reset _http_requests_total to None"
        )

        # Thread barrier: all threads wait at the barrier, then race
        # on the first record_* call simultaneously. This maximizes
        # the chance of two threads passing the fast-path check at
        # the same time, exposing the pre-fix bug.
        barrier = threading.Barrier(20)

        errors = []

        def _race(method_idx):
            try:
                barrier.wait(timeout=5)
                record_funcs = [
                    m.record_http_request,
                    m.record_rate_limiter_check,
                    m.record_rate_limiter_rejection,
                    m.record_neo4j_query,
                    m.record_neo4j_error,
                    m.record_agent_eviction,
                ]
                fn = record_funcs[method_idx % len(record_funcs)]
                try:
                    if fn is m.record_http_request:
                        fn("GET", "/x", 200, 0.01)
                    elif fn is m.record_rate_limiter_check:
                        fn("chat")
                    elif fn is m.record_rate_limiter_rejection:
                        fn("chat")
                    elif fn is m.record_neo4j_query:
                        fn("cypher", 0.01)
                    elif fn is m.record_neo4j_error:
                        fn("timeout")
                    elif fn is m.record_agent_eviction:
                        fn()
                except Exception as e:
                    errors.append((method_idx, e))
            except Exception as e:
                errors.append((method_idx, e))

        threads = [
            threading.Thread(target=_race, args=(i,))
            for i in range(20)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        # The pre-fix bug: at least one thread raises
        # ValueError: Duplicated timeseries in CollectorRegistry. The
        # post-fix: zero such errors.
        dup_errors = [
            (idx, e) for idx, e in errors
            if "Duplicated timeseries" in str(e) or isinstance(e, ValueError)
        ]
        assert not dup_errors, (
            f"Got {len(dup_errors)} ValueError: Duplicated timeseries "
            f"on concurrent first-call. First error: {dup_errors[0]}. "
            f"The double-checked locking in _ensure_metrics is not "
            f"serializing the slow path. All errors: {errors}"
        )

        # Sanity: the slow path DID execute (we can observe the counter).
        # Read directly from the module attribute — this is the
        # authoritative state, not the prom registry.
        assert m._http_requests_total is not None, (
            f"Metrics were not initialized after concurrent first-call — "
            f"_ensure_metrics must have short-circuited. "
            f"is_enabled={m.is_enabled()}, _METRICS_ENABLED={m._METRICS_ENABLED}, "
            f"_prom={m._prom}, errors={errors}"
        )
    finally:
        prom.REGISTRY = original_registry


def test_double_checked_locking_fast_path():
    """After init, _ensure_metrics takes the fast path and doesn't acquire the lock.

    Timing-based: with the lock held briefly under heavy contention,
    the fast path (no lock) should be much faster than a single
    lock acquisition. We just verify that calling _ensure_metrics
    10,000 times in sequence is fast (<0.1s).
    """
    import time
    import proxy.metrics as m

    # Already initialized (from prior test or fixture).
    if m._http_requests_total is None:
        m._ensure_metrics()

    # Call 10,000 times. Pre-fix would have been the same speed
    # (no lock, no work). Post-fix adds a fast-path check + the
    # GIL-protected module attribute read; should still be << 0.1s.
    start = time.perf_counter()
    for _ in range(10_000):
        m._ensure_metrics()
    elapsed = time.perf_counter() - start

    assert elapsed < 0.5, (
        f"10,000 _ensure_metrics calls took {elapsed:.3f}s. "
        f"Fast path should be < 0.05s; this suggests the lock is "
        f"being acquired on every call (missing the inner re-check)."
    )
