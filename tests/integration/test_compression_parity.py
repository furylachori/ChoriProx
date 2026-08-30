"""Integration tests for C6 — compression parity + RMW serialization (D1, D2).

D1 — Worker deletes what it compressed (parity with sync path).
D2 — Per-session ancient lock serializes concurrent compress paths.
D2.5 — Lock cleanup removes stale entries.

Each test uses a unique file-backed SQLite database supplied by tmp_path.
"""
import hashlib
import threading
import time
from unittest.mock import patch

import pytest


pytestmark = pytest.mark.integration


SESSION_BASE = "test-c6-compression-parity"


@pytest.fixture(autouse=True)
def _isolate_db(tmp_path, monkeypatch):
    db_path = tmp_path / "compression-parity.db"
    monkeypatch.setenv("PROXY_TEST_DB", str(db_path))
    from memory.db import init_db
    init_db(str(db_path))
    yield str(db_path)


def _session_id(test_name):
    return f"{SESSION_BASE}-{test_name}"


def _content_hash(content):
    return hashlib.sha256(content.encode()).hexdigest()[:16]


def _seed_summary(session_id, exchange_index, summary, message_index=None,
                   end_message_index=None):
    from memory.db_summaries import store_message_summary
    if message_index is None:
        message_index = exchange_index * 2
    if end_message_index is None:
        end_message_index = message_index + 1
    store_message_summary(
        session_id=session_id,
        exchange_index=exchange_index,
        role="user",
        summary=summary,
        tool_summary="",
        content_hash=_content_hash(f"seed-{exchange_index}-{summary}"),
        model_used="seed",
        token_estimate=50,
        end_message_index=end_message_index,
        message_index=message_index,
    )


def _count_summaries(session_id):
    from memory.db import connection
    with connection() as conn:
        return conn.execute(
            "SELECT COUNT(*) as c FROM message_summaries WHERE session_id = ?",
            (session_id,),
        ).fetchone()["c"]


# ============================================================================
# Test 1: Worker deletes what it compressed (D1)
# ============================================================================


class TestWorkerDeleteParity:
    """D1: worker must delete compressed summaries after writing ancient."""

    def test_worker_deletes_compressed_summaries(self):
        from memory import sync as sync_mod
        from memory.db_summaries import get_ancient_history, get_rich_summaries

        SESSION = _session_id("worker-deletes")
        _seed_summary(SESSION, 0, "ZERO_SENTINEL_A")
        _seed_summary(SESSION, 1, "ONE_SENTINEL_B")
        _seed_summary(SESSION, 2, "TWO_SENTINEL_C")
        _seed_summary(SESSION, 3, "THREE_SENTINEL_KEEP")

        payload = get_rich_summaries(SESSION)[:3]
        assert len(payload) == 3, f"Test setup: expected 3 summaries, got {len(payload)}"

        with patch.object(sync_mod, "_get_sota") as mock_sota:
            mock_sota.return_value.compress_ancient_history.return_value = (
                "NEW_ANCIENT_FROM_WORKER_ABC_99999"
            )
            sync_mod._ancient_compress_worker(SESSION, payload)

        assert _count_summaries(SESSION) == 1, (
            f"Worker should have deleted 3 summaries, but "
            f"{_count_summaries(SESSION)} remain"
        )
        kept = get_rich_summaries(SESSION)
        assert len(kept) == 1
        assert kept[0]["exchange_index"] == 3
        assert kept[0]["summary"] == "THREE_SENTINEL_KEEP"

        ancient = get_ancient_history(SESSION)
        assert "NEW_ANCIENT_FROM_WORKER_ABC_99999" in ancient, (
            f"Worker must have written new ancient. Got: {ancient!r}"
        )


# ============================================================================
# Test 2: Barrier race (D2) — concurrent compress paths serialize
# ============================================================================


class TestBarrierRaceSerialization:
    """D2: per-session ancient lock serializes concurrent compress paths."""

    def test_concurrent_compress_paths_serialize(self):
        from memory import sync as sync_mod
        from memory.db_summaries import get_ancient_history, get_rich_summaries

        SESSION = _session_id("barrier-race")
        for i in range(6):
            _seed_summary(SESSION, i, f"summary-{i}")

        all_summaries = get_rich_summaries(SESSION)
        path_a_payload = all_summaries[:3]
        path_b_payload = all_summaries[3:]

        barrier = threading.Barrier(2)
        sota_latency = 0.3
        call_log = []
        call_log_lock = threading.Lock()

        def slow_compress(session_id, existing_ancient, payload):
            with call_log_lock:
                call_log.append({
                    "existing_len": len(existing_ancient or ""),
                    "payload_size": len(payload),
                })
            time.sleep(sota_latency)
            return f"COMPRESSED_V{len(call_log):03d}_{len(payload)}SUMS"

        def path_a_worker():
            barrier.wait()
            sync_mod._ancient_compress_worker(SESSION, path_a_payload)

        def path_b_worker():
            barrier.wait()
            sync_mod._ancient_compress_worker(SESSION, path_b_payload)

        with patch.object(sync_mod, "_get_sota") as mock_sota:
            mock_sota.return_value.compress_ancient_history.side_effect = slow_compress
            t_a = threading.Thread(target=path_a_worker)
            t_b = threading.Thread(target=path_b_worker)
            t_a.start()
            t_b.start()
            t_a.join(timeout=10)
            t_b.join(timeout=10)

        assert not t_a.is_alive(), "Path A thread did not finish in 10s"
        assert not t_b.is_alive(), "Path B thread did not finish in 10s"

        assert len(call_log) == 2, (
            f"Both SOTA calls must execute (serialized via lock). "
            f"Got {len(call_log)} calls: {call_log}"
        )

        first_existing_len = call_log[0]["existing_len"]
        second_existing_len = call_log[1]["existing_len"]
        assert second_existing_len > first_existing_len, (
            f"Second compress call must read a longer existing_ancient "
            f"(path B observes path A's write). "
            f"call_log={call_log}"
        )

        assert _count_summaries(SESSION) == 0, (
            f"After concurrent compress, all 6 summaries should be deleted. "
            f"Got {_count_summaries(SESSION)} remaining."
        )

        final_ancient = get_ancient_history(SESSION)
        assert final_ancient.startswith("COMPRESSED_V") and "SUMS" in final_ancient, (
            f"Final ancient must be a valid SOTA-compressed string. "
            f"Got: {final_ancient!r}"
        )


# ============================================================================
# Test 3: Per-session dedup (no double-write)
# ============================================================================


class TestNoDoubleQueuePerSession:
    """After first worker deletes summaries, second run with empty payload
    must not overwrite ancient."""

    def test_second_dispatch_with_empty_payload_is_noop(self):
        from memory import sync as sync_mod
        from memory.db_summaries import get_ancient_history, get_rich_summaries

        SESSION = _session_id("dedup")
        _seed_summary(SESSION, 0, "A")
        _seed_summary(SESSION, 1, "B")
        _seed_summary(SESSION, 2, "C")
        payload = get_rich_summaries(SESSION)

        with patch.object(sync_mod, "_get_sota") as mock_sota:
            mock_sota.return_value.compress_ancient_history.return_value = (
                "FIRST_WRITE_ABC"
            )
            sync_mod._ancient_compress_worker(SESSION, payload)

        assert _count_summaries(SESSION) == 0
        first_ancient = get_ancient_history(SESSION)
        assert first_ancient == "FIRST_WRITE_ABC"

        with patch.object(sync_mod, "_get_sota") as mock_sota:
            mock_sota.return_value.compress_ancient_history.return_value = (
                "SECOND_SHOULD_NOT_FIRE"
            )
            sync_mod._ancient_compress_worker(SESSION, [])

        assert get_ancient_history(SESSION) == "FIRST_WRITE_ABC", (
            "Second dispatch with empty payload must not overwrite ancient."
        )


# ============================================================================
# Test 4: Lock cleanup
# ============================================================================


class TestAncientLockCleanup:
    """_ancient_locks must have cleanup, mirror _cleanup_sync_locks."""

    def test_cleanup_removes_stale_session_entries(self):
        from memory import sync as sync_mod

        fake_session = _session_id("cleanup-stale")
        sync_mod._get_ancient_lock(fake_session)
        assert fake_session in sync_mod._ancient_locks

        cleaned = sync_mod._cleanup_ancient_locks({fake_session})
        assert cleaned == 1
        assert fake_session not in sync_mod._ancient_locks

    def test_cleanup_returns_zero_when_no_stale(self):
        from memory import sync as sync_mod

        cleaned = sync_mod._cleanup_ancient_locks({_session_id("cleanup-missing")})
        assert cleaned == 0

    def test_cleanup_handles_mixed_real_and_missing(self):
        from memory import sync as sync_mod

        real_session = _session_id("cleanup-mixed-real")
        sync_mod._get_ancient_lock(real_session)
        assert real_session in sync_mod._ancient_locks

        cleaned = sync_mod._cleanup_ancient_locks(
            {real_session, _session_id("cleanup-missing-1"), _session_id("cleanup-missing-2")}
        )
        assert cleaned == 1
        assert real_session not in sync_mod._ancient_locks

    def test_cleanup_skips_held_locks(self):
        from memory import sync as sync_mod
        import threading

        held_session = _session_id("cleanup-held")
        lock = sync_mod._get_ancient_lock(held_session)

        ready = threading.Event()
        done = threading.Event()

        def hold():
            lock.acquire()
            ready.set()
            done.wait()
            lock.release()

        t = threading.Thread(target=hold)
        t.start()
        ready.wait()
        try:
            cleaned = sync_mod._cleanup_ancient_locks({held_session})
            assert cleaned == 0
            assert held_session in sync_mod._ancient_locks
        finally:
            done.set()
            t.join()

        cleaned = sync_mod._cleanup_ancient_locks({held_session})
        assert cleaned == 1
        assert held_session not in sync_mod._ancient_locks
