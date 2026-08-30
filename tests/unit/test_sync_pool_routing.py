"""M5-1: sync pool starvation when ancient-compress jobs are queued.

Pre-fix: submit_ancient_compress routed through `_get_sync_pool()` —
the same 4-worker pool that submit_sync uses. Ancient-compress blocks
a worker for the full SOTA round-trip (≤180s + 5 internal retries).
Eviction burst → all 4 workers stuck on ancient-compress →
sync_exchange can't run → batches pile up.

Fix: route submit_ancient_compress through the summary pool
(`_get_summary_pool()`, 4 workers, separate queue). The two work
classes now have independent worker queues — sync_exchange workers
are not blocked by ancient-compress SOTA calls.

This test is the load test: prove the two pools are actually
independent by submitting ancient-compress jobs that BLOCK the
summary pool, then verifying sync_exchange still runs promptly.
"""

import threading
from concurrent.futures import ThreadPoolExecutor

import pytest

pytestmark = pytest.mark.unit


@pytest.fixture
def isolated_memory_executors(monkeypatch):
    """Install test-owned executor singletons for one routing test."""
    import memory.executors as executors

    sync_pool = ThreadPoolExecutor(
        max_workers=4,
        thread_name_prefix="test-mem-sync",
    )
    query_pool = ThreadPoolExecutor(
        max_workers=2,
        thread_name_prefix="test-mem-query",
    )
    summary_pool = ThreadPoolExecutor(
        max_workers=4,
        thread_name_prefix="test-mem-summary",
    )
    timeout_pool = ThreadPoolExecutor(
        max_workers=4,
        thread_name_prefix="test-timeout-guard",
    )

    monkeypatch.setattr(executors, "_sync_pool", sync_pool)
    monkeypatch.setattr(executors, "_query_pool", query_pool)
    monkeypatch.setattr(executors, "_summary_pool", summary_pool)
    monkeypatch.setattr(executors, "_timeout_pool", timeout_pool)
    monkeypatch.setattr(executors, "_shutdown_complete", False)
    monkeypatch.setattr(executors, "_shutdown_requested", False)
    monkeypatch.setattr(executors, "_session_locks", {})

    try:
        yield {
            "sync": sync_pool,
            "query": query_pool,
            "summary": summary_pool,
            "timeout": timeout_pool,
        }
    finally:
        timeout_pool.shutdown(wait=True, cancel_futures=True)
        summary_pool.shutdown(wait=True, cancel_futures=True)
        query_pool.shutdown(wait=True, cancel_futures=True)
        sync_pool.shutdown(wait=True, cancel_futures=True)


def test_ancient_compress_does_not_starve_sync_pool(
    monkeypatch,
    isolated_memory_executors,
):
    """Four real ancient-compress dispatches must not starve sync work."""
    import memory.sync as sync

    # The session-scoped autouse ``mock_sync_submit`` fixture
    # (tests/conftest.py) replaces ``memory.sync.submit_sync`` with a
    # MagicMock to prevent background threads from spawning in the broader
    # test suite. This test specifically exercises pool routing, so we
    # rebind ``submit_sync`` to the real underlying implementation
    # (``memory.executors.submit_sync``, aliased as
    # ``memory.sync._exec_submit_sync``) for the duration of this test.
    # monkeypatch restores the autouse MagicMock on teardown.
    def _real_submit_sync(session_id):
        return sync._exec_submit_sync(sync.sync_exchange, session_id)

    monkeypatch.setattr(sync, "submit_sync", _real_submit_sync)

    workers_started = threading.Barrier(5)
    release_workers = threading.Event()
    ancient_futures = []

    def _blocking_ancient_worker(session_id, summary_payload):
        workers_started.wait(timeout=5.0)
        assert release_workers.wait(timeout=5.0)

    def _quick_sync(session_id):
        return f"sync:{session_id}"

    monkeypatch.setattr(
        sync,
        "_ancient_compress_worker",
        _blocking_ancient_worker,
    )
    monkeypatch.setattr(sync, "sync_exchange", _quick_sync)
    monkeypatch.setattr(sync, "_ancient_compress_pending", 0)
    monkeypatch.setattr(
        "memory.sota_channel._sota_breaker.can_execute",
        lambda: True,
    )

    try:
        for index in range(4):
            future = sync.submit_ancient_compress(
                f"ancient-routing-{index}",
                [{"exchange_index": index, "summary": "payload"}],
            )
            assert future is not None
            ancient_futures.append(future)

        workers_started.wait(timeout=5.0)

        sync_future = sync.submit_sync("sync-routing")
        assert sync_future is not None
        assert sync_future.result(timeout=1.0) == "sync:sync-routing"

        assert release_workers.is_set() is False
    finally:
        release_workers.set()
        for future in ancient_futures:
            future.result(timeout=5.0)

    assert sync._ancient_compress_pending == 0


def test_ancient_compress_uses_summary_pool_not_sync_pool(
    monkeypatch,
    isolated_memory_executors,
):
    """Ancient compression must dispatch only to the summary pool."""
    import memory.sync as sync

    sync_pool = isolated_memory_executors["sync"]
    summary_pool = isolated_memory_executors["summary"]

    sync_pool_calls = []
    summary_pool_calls = []

    real_sync_submit = sync_pool.submit
    real_summary_submit = summary_pool.submit

    def spy_sync_submit(*args, **kwargs):
        sync_pool_calls.append((args, kwargs))
        return real_sync_submit(*args, **kwargs)

    def spy_summary_submit(*args, **kwargs):
        summary_pool_calls.append((args, kwargs))
        return real_summary_submit(*args, **kwargs)

    monkeypatch.setattr(sync_pool, "submit", spy_sync_submit)
    monkeypatch.setattr(summary_pool, "submit", spy_summary_submit)
    monkeypatch.setattr(
        sync,
        "_ancient_compress_worker",
        lambda session_id, summary_payload: None,
    )
    monkeypatch.setattr(sync, "_ancient_compress_pending", 0)
    monkeypatch.setattr(
        "memory.sota_channel._sota_breaker.can_execute",
        lambda: True,
    )

    future = sync.submit_ancient_compress(
        "s-ancient-pool-test",
        [{"exchange_index": 0, "summary": "x"}],
    )

    assert future is not None
    future.result(timeout=5.0)

    assert len(summary_pool_calls) == 1
    assert len(sync_pool_calls) == 0
    assert sync._ancient_compress_pending == 0