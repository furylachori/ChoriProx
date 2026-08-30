"""CP4-1/CP4-2/CP4-3: canonical JSON fingerprint + DedupStore with in-flight dedup + replay.

CP4-1 pinned fingerprint determinism, store CRUD, eviction, and caps.
CP4-2 adds lookup_or_insert (atomic owner/follower admission) and
FollowerCapExceeded. insert_owner is retired.
CP4-3 adds is_safe_for_replay safety gate and REPLAY admission kind.
"""
import json
import threading
import time
import types

import pytest

from proxy.completion_coordinator import AdmissionKind
from proxy.dedup import (
    DedupEntry,
    DedupState,
    DedupStore,
    FollowerCapExceeded,
    canonical_json,
    compute_fingerprint,
    is_safe_for_replay,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _clean_dedup_store():
    """Ensure test isolation: clear the module-level store before/after each test."""
    import proxy.state as state
    state.dedup_store.clear()
    yield
    state.dedup_store.clear()


def _ctx(**overrides):
    defaults = dict(
        session_id="sess-abc",
        client_hint="127.0.0.1",
        model_name="test-model",
        workspace_name="",
        is_subagent=False,
        body={"model": "test-model", "messages": [{"role": "user", "content": "hi"}]},
    )
    defaults.update(overrides)
    return types.SimpleNamespace(**defaults)


# ---------------------------------------------------------------------------
# canonical_json
# ---------------------------------------------------------------------------


class TestCanonicalJson:
    def test_sorted_keys(self):
        result = canonical_json({"b": 1, "a": 2})
        assert result == b'{"a":2,"b":1}'

    def test_compact_separators(self):
        result = canonical_json({"key": "value"})
        assert b" " not in result  # no whitespace

    def test_nested_sorted(self):
        result = canonical_json({"z": {"b": 1, "a": 2}, "a": 0})
        parsed = json.loads(result)
        assert list(parsed.keys()) == ["a", "z"]
        assert list(parsed["z"].keys()) == ["a", "b"]

    def test_unicode_preserved(self):
        result = canonical_json({"msg": "你好"})
        assert "你好".encode("utf-8") in result

    def test_nan_rejected(self):
        with pytest.raises(ValueError):
            canonical_json({"temperature": float("nan")})

    def test_infinity_rejected(self):
        with pytest.raises(ValueError):
            canonical_json({"value": float("inf")})

    def test_empty_dict(self):
        assert canonical_json({}) == b"{}"

    def test_list_preserved(self):
        result = canonical_json({"items": [3, 1, 2]})
        assert json.loads(result)["items"] == [3, 1, 2]

    def test_deterministic(self):
        body = {"model": "m", "messages": [{"role": "user", "content": "hi"}], "stream": True}
        assert canonical_json(body) == canonical_json(body)


# ---------------------------------------------------------------------------
# compute_fingerprint
# ---------------------------------------------------------------------------


class TestComputeFingerprint:
    def test_deterministic(self):
        ctx = _ctx()
        assert compute_fingerprint(ctx) == compute_fingerprint(ctx)

    def test_truncated_to_32(self):
        assert len(compute_fingerprint(_ctx())) == 32

    def test_hex_chars(self):
        fp = compute_fingerprint(_ctx())
        assert all(c in "0123456789abcdef" for c in fp)

    def test_different_body_different_fp(self):
        fp1 = compute_fingerprint(_ctx(body={"messages": [{"role": "user", "content": "a"}]}))
        fp2 = compute_fingerprint(_ctx(body={"messages": [{"role": "user", "content": "b"}]}))
        assert fp1 != fp2

    def test_different_session_different_fp(self):
        fp1 = compute_fingerprint(_ctx(session_id="s1"))
        fp2 = compute_fingerprint(_ctx(session_id="s2"))
        assert fp1 != fp2

    def test_different_client_different_fp(self):
        fp1 = compute_fingerprint(_ctx(client_hint="1.2.3.4"))
        fp2 = compute_fingerprint(_ctx(client_hint="5.6.7.8"))
        assert fp1 != fp2

    def test_different_model_different_fp(self):
        fp1 = compute_fingerprint(_ctx(model_name="model-a"))
        fp2 = compute_fingerprint(_ctx(model_name="model-b"))
        assert fp1 != fp2

    def test_different_workspace_different_fp(self):
        fp1 = compute_fingerprint(_ctx(workspace_name="ws-a"))
        fp2 = compute_fingerprint(_ctx(workspace_name="ws-b"))
        assert fp1 != fp2

    def test_subagent_flag_affects_fp(self):
        fp1 = compute_fingerprint(_ctx(is_subagent=False))
        fp2 = compute_fingerprint(_ctx(is_subagent=True))
        assert fp1 != fp2

    def test_nan_body_does_not_crash(self):
        """compute_fingerprint raises ValueError on NaN body — the app.py
        wire-in wraps this in try/except. Here we pin the raise."""
        ctx = _ctx(body={"temperature": float("nan")})
        with pytest.raises(ValueError):
            compute_fingerprint(ctx)

    def test_stream_field_stripped_from_fingerprint(self):
        """CP4-4: the stream field is stripped from body before fingerprinting.
        Same body with/without stream produces the same fingerprint."""
        body_with_stream = {"model": "m", "messages": [{"role": "user", "content": "hi"}], "stream": True}
        body_without_stream = {"model": "m", "messages": [{"role": "user", "content": "hi"}]}
        body_stream_false = {"model": "m", "messages": [{"role": "user", "content": "hi"}], "stream": False}
        fp1 = compute_fingerprint(_ctx(body=body_with_stream))
        fp2 = compute_fingerprint(_ctx(body=body_without_stream))
        fp3 = compute_fingerprint(_ctx(body=body_stream_false))
        assert fp1 == fp2 == fp3


# ---------------------------------------------------------------------------
# DedupStore
# ---------------------------------------------------------------------------


class TestDedupStore:
    def test_lookup_miss(self):
        store = DedupStore()
        assert store.lookup("nonexistent") is None

    def test_insert_and_lookup(self):
        store = DedupStore()
        record = types.SimpleNamespace(owner_request_id="r1")
        entry, kind = store.lookup_or_insert("fp1", lambda: record, is_stream=True)
        assert kind == AdmissionKind.OWNER
        assert entry.state == DedupState.PENDING
        assert entry.record is record
        assert entry.is_stream is True
        # Verify lookup finds it
        found = store.lookup("fp1")
        assert found is entry

    def test_mark_completed(self):
        store = DedupStore()
        record = types.SimpleNamespace(owner_request_id="r1")
        store.lookup_or_insert("fp1", lambda: record, is_stream=True)
        artifact = types.SimpleNamespace(
            terminal_chunks=("data: {}\n\n", "data: [DONE]\n\n"),
            logical_status="ok",
        )
        store.mark_completed("fp1", artifact)
        entry = store.lookup("fp1")
        assert entry.state == DedupState.COMPLETED
        assert entry.artifact is artifact
        assert entry.completed_at is not None
        assert entry.stored_bytes > 0

    def test_mark_failed_evicted_on_next_insert(self):
        store = DedupStore()
        record1 = types.SimpleNamespace(owner_request_id="r1")
        store.lookup_or_insert("fp1", lambda: record1, is_stream=True)
        store.mark_failed("fp1")
        # FAILED entry is visible via lookup (test inspection)
        entry = store.lookup("fp1")
        assert entry is not None
        assert entry.state == DedupState.FAILED
        # Next lookup_or_insert evicts FAILED and creates new OWNER
        record2 = types.SimpleNamespace(owner_request_id="r2")
        entry2, kind = store.lookup_or_insert("fp1", lambda: record2, is_stream=True)
        assert kind == AdmissionKind.OWNER
        assert entry2.state == DedupState.PENDING
        assert entry2.record is record2

    def test_completed_ttl_expiry(self):
        store = DedupStore()
        store.completed_ttl_seconds = 0.1
        record = types.SimpleNamespace(owner_request_id="r1")
        store.lookup_or_insert("fp1", lambda: record, is_stream=True)
        artifact = types.SimpleNamespace(terminal_chunks=("data: [DONE]\n\n",), logical_status="ok")
        store.mark_completed("fp1", artifact)
        assert store.lookup("fp1") is not None
        time.sleep(0.15)
        assert store.lookup("fp1") is None

    def test_pending_max_age_expiry(self):
        store = DedupStore()
        store.pending_max_age_seconds = 0.1
        record = types.SimpleNamespace(owner_request_id="r1")
        store.lookup_or_insert("fp1", lambda: record, is_stream=True)
        assert store.lookup("fp1") is not None
        time.sleep(0.15)
        assert store.lookup("fp1") is None

    def test_lru_eviction_max_completed(self):
        store = DedupStore()
        store.max_completed = 2
        for i in range(3):
            fp = f"fp{i}"
            record = types.SimpleNamespace(owner_request_id=f"r{i}")
            store.lookup_or_insert(fp, lambda r=record: r, is_stream=True)
            artifact = types.SimpleNamespace(terminal_chunks=(f"data: {i}\n\n",), logical_status="ok")
            store.mark_completed(fp, artifact)
        # Oldest (fp0) should be evicted
        assert store.lookup("fp0") is None
        assert store.lookup("fp1") is not None
        assert store.lookup("fp2") is not None

    def test_pending_never_evicted_by_completed_cap(self):
        store = DedupStore()
        store.max_completed = 1
        # Insert a PENDING entry
        record_p = types.SimpleNamespace(owner_request_id="rp")
        store.lookup_or_insert("fp_pending", lambda: record_p, is_stream=True)
        # Insert 2 COMPLETED entries (exceeds max_completed=1)
        for i in range(2):
            fp = f"fp_c{i}"
            record = types.SimpleNamespace(owner_request_id=f"rc{i}")
            store.lookup_or_insert(fp, lambda r=record: r, is_stream=True)
            artifact = types.SimpleNamespace(terminal_chunks=(f"data: {i}\n\n",), logical_status="ok")
            store.mark_completed(fp, artifact)
        # PENDING entry must survive
        assert store.lookup("fp_pending") is not None
        assert store.lookup("fp_pending").state == DedupState.PENDING

    def test_clear(self):
        store = DedupStore()
        record = types.SimpleNamespace(owner_request_id="r1")
        store.lookup_or_insert("fp1", lambda: record, is_stream=True)
        store.clear()
        assert store.lookup("fp1") is None
        assert store.stats()["pending_entries"] == 0

    def test_stats(self):
        store = DedupStore()
        record = types.SimpleNamespace(owner_request_id="r1")
        store.lookup_or_insert("fp1", lambda: record, is_stream=True)
        stats = store.stats()
        assert stats["pending_entries"] == 1
        assert stats["completed_entries"] == 0
        artifact = types.SimpleNamespace(terminal_chunks=("data: x\n\n",), logical_status="ok")
        store.mark_completed("fp1", artifact)
        stats = store.stats()
        assert stats["pending_entries"] == 0
        assert stats["completed_entries"] == 1
        assert stats["total_stored_bytes"] > 0

    def test_stored_bytes_uses_utf8(self):
        store = DedupStore()
        record = types.SimpleNamespace(owner_request_id="r1")
        store.lookup_or_insert("fp1", lambda: record, is_stream=True)
        # CJK content: 3 bytes per char in UTF-8
        artifact = types.SimpleNamespace(terminal_chunks=("data: 你好\n\n",), logical_status="ok")
        store.mark_completed("fp1", artifact)
        entry = store.lookup("fp1")
        expected = len("data: 你好\n\n".encode("utf-8"))
        assert entry.stored_bytes == expected

    def test_lookup_or_insert_follower_attaches(self):
        """Second call with same fingerprint attaches as follower."""
        store = DedupStore()
        record = types.SimpleNamespace(owner_request_id="r1")
        entry1, kind1 = store.lookup_or_insert("fp1", lambda: record, is_stream=True)
        assert kind1 == AdmissionKind.OWNER
        assert entry1.follower_count == 0

        # Second call: follower attaches
        entry2, kind2 = store.lookup_or_insert("fp1", lambda: record, is_stream=True)
        assert kind2 == AdmissionKind.FOLLOWER
        assert entry2 is entry1  # same entry
        assert entry2.follower_count == 1

    def test_lookup_or_insert_follower_cap_exceeded(self):
        """FollowerCapExceeded raised when cap is reached."""
        store = DedupStore()
        store.max_followers_per_key = 2
        record = types.SimpleNamespace(owner_request_id="r1")
        store.lookup_or_insert("fp1", lambda: record, is_stream=True)

        # Attach 2 followers (cap=2)
        store.lookup_or_insert("fp1", lambda: record, is_stream=True)
        store.lookup_or_insert("fp1", lambda: record, is_stream=True)

        # Third follower exceeds cap
        with pytest.raises(FollowerCapExceeded):
            store.lookup_or_insert("fp1", lambda: record, is_stream=True)

    def test_lookup_or_insert_completed_safe_returns_replay(self):
        """CP4-3: COMPLETED + safe artifact → REPLAY, entry NOT evicted."""
        store = DedupStore()
        record = types.SimpleNamespace(owner_request_id="r1")
        store.lookup_or_insert("fp1", lambda: record, is_stream=True)
        artifact = types.SimpleNamespace(
            terminal_chunks=("data: [DONE]\n\n",),
            logical_status="ok",
            has_tool_calls=False,
        )
        store.mark_completed("fp1", artifact)

        new_record = types.SimpleNamespace(owner_request_id="r2")
        entry, kind = store.lookup_or_insert("fp1", lambda: new_record, is_stream=True)
        assert kind == AdmissionKind.REPLAY
        # Entry NOT evicted — still COMPLETED with original artifact
        assert entry.state == DedupState.COMPLETED
        assert entry.artifact is artifact

    def test_lookup_or_insert_completed_unsafe_evicts(self):
        """CP4-3: COMPLETED + unsafe artifact (has_tool_calls) → evict + new OWNER."""
        store = DedupStore()
        record = types.SimpleNamespace(owner_request_id="r1")
        store.lookup_or_insert("fp1", lambda: record, is_stream=True)
        artifact = types.SimpleNamespace(
            terminal_chunks=("data: [DONE]\n\n",),
            logical_status="ok",
            has_tool_calls=True,
        )
        store.mark_completed("fp1", artifact)

        new_record = types.SimpleNamespace(owner_request_id="r2")
        entry, kind = store.lookup_or_insert("fp1", lambda: new_record, is_stream=True)
        assert kind == AdmissionKind.OWNER
        assert entry.state == DedupState.PENDING
        assert entry.record is new_record

    def test_lock_injection(self):
        """DedupStore uses the lock passed at construction."""
        shared_lock = threading.Lock()
        store = DedupStore(lock=shared_lock)
        assert store._lock is shared_lock

    def test_lock_injection_default(self):
        """DedupStore creates its own lock when none is passed."""
        store = DedupStore()
        assert isinstance(store._lock, type(threading.Lock()))

    def test_lookup_or_insert_conflict_stream_mismatch(self):
        """CP4-4: PENDING entry with is_stream=True, lookup with is_stream=False → CONFLICT."""
        store = DedupStore()
        record = types.SimpleNamespace(owner_request_id="r1")
        entry1, kind1 = store.lookup_or_insert("fp1", lambda: record, is_stream=True)
        assert kind1 == AdmissionKind.OWNER
        assert entry1.is_stream is True

        # Same fingerprint, different stream mode → CONFLICT
        entry2, kind2 = store.lookup_or_insert("fp1", lambda: record, is_stream=False)
        assert kind2 == AdmissionKind.CONFLICT
        assert entry2 is entry1  # same entry returned
        assert store.conflicts == 1

    def test_nonstream_stored_bytes_nonzero(self):
        """CP4-4: mark_completed with body → stored_bytes > 0."""
        store = DedupStore()
        record = types.SimpleNamespace(owner_request_id="r1")
        store.lookup_or_insert("fp1", lambda: record, is_stream=False)
        body = {"choices": [{"message": {"content": "hello"}}]}
        artifact = types.SimpleNamespace(
            terminal_chunks=(),
            logical_status="ok",
            body=body,
        )
        store.mark_completed("fp1", artifact)
        entry = store.lookup("fp1")
        assert entry.state == DedupState.COMPLETED
        assert entry.stored_bytes > 0
        # Verify it matches the compact JSON encoding of the body
        expected = len(json.dumps(body, separators=(",", ":")).encode("utf-8"))
        assert entry.stored_bytes == expected


# ---------------------------------------------------------------------------
# is_safe_for_replay (CP4-3)
# ---------------------------------------------------------------------------


class TestIsSafeForReplay:
    def test_is_safe_for_replay_ok(self):
        """COMPLETED + ok + no tool_calls → True."""
        entry = DedupEntry(
            fingerprint="fp1",
            state=DedupState.COMPLETED,
            artifact=types.SimpleNamespace(
                logical_status="ok",
                has_tool_calls=False,
                terminal_chunks=("data: [DONE]\n\n",),
            ),
        )
        assert is_safe_for_replay(entry) is True

    def test_is_safe_for_replay_tool_calls(self):
        """COMPLETED + ok + has_tool_calls=True → False."""
        entry = DedupEntry(
            fingerprint="fp1",
            state=DedupState.COMPLETED,
            artifact=types.SimpleNamespace(
                logical_status="ok",
                has_tool_calls=True,
                terminal_chunks=("data: [DONE]\n\n",),
            ),
        )
        assert is_safe_for_replay(entry) is False

    def test_is_safe_for_replay_error(self):
        """COMPLETED + logical_status='error' → False."""
        entry = DedupEntry(
            fingerprint="fp1",
            state=DedupState.COMPLETED,
            artifact=types.SimpleNamespace(
                logical_status="error",
                has_tool_calls=False,
                terminal_chunks=("data: [DONE]\n\n",),
            ),
        )
        assert is_safe_for_replay(entry) is False

    def test_is_safe_for_replay_none_artifact(self):
        """COMPLETED + artifact=None → False."""
        entry = DedupEntry(
            fingerprint="fp1",
            state=DedupState.COMPLETED,
            artifact=None,
        )
        assert is_safe_for_replay(entry) is False

    def test_is_safe_for_replay_not_completed(self):
        """PENDING state → False."""
        entry = DedupEntry(
            fingerprint="fp1",
            state=DedupState.PENDING,
            artifact=types.SimpleNamespace(
                logical_status="ok",
                has_tool_calls=False,
                terminal_chunks=("data: [DONE]\n\n",),
            ),
        )
        assert is_safe_for_replay(entry) is False


# ---------------------------------------------------------------------------
# release_follower: idempotent follower_count decrement (Issue 4)
# ---------------------------------------------------------------------------


class TestReleaseFollower:
    """release_follower: idempotent decrement of follower_count on PENDING entries."""

    def _make_entry(self, fingerprint="fp-1", state=DedupState.PENDING, count=0):
        from proxy.completion_coordinator import CompletionRecord
        return DedupEntry(
            fingerprint=fingerprint,
            record=CompletionRecord(owner_request_id="req-1"),
            state=state,
        )

    def test_release_follower_decrements_pending(self):
        store = DedupStore()
        entry = self._make_entry(count=3)
        store._store["fp-1"] = entry
        entry.follower_count = 3
        assert store.release_follower("fp-1") is True
        assert entry.follower_count == 2

    def test_release_follower_no_op_on_completed(self):
        store = DedupStore()
        entry = self._make_entry(state=DedupState.COMPLETED)
        store._store["fp-1"] = entry
        entry.follower_count = 3
        assert store.release_follower("fp-1") is False
        assert entry.follower_count == 3

    def test_release_follower_no_op_on_failed(self):
        store = DedupStore()
        entry = self._make_entry(state=DedupState.FAILED)
        store._store["fp-1"] = entry
        entry.follower_count = 3
        assert store.release_follower("fp-1") is False
        assert entry.follower_count == 3

    def test_release_follower_no_op_on_missing(self):
        store = DedupStore()
        assert store.release_follower("nonexistent") is False

    def test_release_follower_floor_zero(self):
        store = DedupStore()
        entry = self._make_entry()
        store._store["fp-1"] = entry
        entry.follower_count = 0
        assert store.release_follower("fp-1") is False
        assert entry.follower_count == 0

    def test_release_follower_increments_stats_counter(self):
        store = DedupStore()
        entry = self._make_entry()
        store._store["fp-1"] = entry
        entry.follower_count = 3
        store.release_follower("fp-1")
        assert store.stats()["releases"] == 1
        # No-op cases don't increment
        store.release_follower("nonexistent")
        assert store.stats()["releases"] == 1
        store.release_follower("fp-1")  # count is now 1
        store.release_follower("fp-1")  # count is now 0
        store.release_follower("fp-1")  # floor-0 no-op
        assert store.stats()["releases"] == 3

    def test_release_follower_after_real_lookup_or_insert(self):
        """Real follow-up: attach via lookup_or_insert, then release."""
        from proxy.completion_coordinator import CompletionRecord
        store = DedupStore()
        record_factory = lambda: CompletionRecord(owner_request_id="req-1")
        # OWNER
        entry, kind1 = store.lookup_or_insert("fp-1", record_factory, is_stream=True)
        assert kind1 == AdmissionKind.OWNER
        # FOLLOWER
        entry, kind2 = store.lookup_or_insert("fp-1", record_factory, is_stream=True)
        assert kind2 == AdmissionKind.FOLLOWER
        assert entry.follower_count == 1
        # Release
        assert store.release_follower("fp-1") is True
        assert entry.follower_count == 0
        assert store.stats()["releases"] == 1

    def test_release_follower_concurrency(self):
        """16 followers attach via lookup_or_insert, then release concurrently.

        Exercises the real _lock-serialized decrement path. Uses threading.Barrier
        for synchronized start to make the test deterministic, not flaky.
        """
        from proxy.completion_coordinator import CompletionRecord
        from tests.concurrency_harness import run_concurrently
        store = DedupStore()
        record_factory = lambda: CompletionRecord(owner_request_id="req-1")
        # OWNER
        entry, _ = store.lookup_or_insert("fp-1", record_factory, is_stream=True)
        # 16 followers
        for _ in range(16):
            store.lookup_or_insert("fp-1", record_factory, is_stream=True)
        assert entry.follower_count == 16

        def worker():
            store.release_follower("fp-1")

        run_concurrently([worker] * 16, timeout=10)
        assert entry.follower_count == 0
        assert store.stats()["releases"] == 16


# ---------------------------------------------------------------------------
# Lock ordering revert-test
# ---------------------------------------------------------------------------


def test_dedup_lock_uses_position_14():
    """Revert-test: state.py must use make_lock(14, ...) for dedup_lock."""
    import inspect
    import proxy.state as state_mod
    source = inspect.getsource(state_mod)
    assert "make_lock(14," in source, (
        "dedup_lock must be at position 14 in the lock hierarchy"
    )


# ---------------------------------------------------------------------------
# /health observability (CP4-5)
# ---------------------------------------------------------------------------


def test_health_includes_dedup_stats(monkeypatch):
    """CP4-5: the /health endpoint includes a dedup block with stats().

    Neo4j + SQLite are mocked so /health returns a deterministic 200, and the
    admin rate limiter is patched to a permissive non-persistent limiter so the
    rate-limit gate never interferes. The autouse ``_clean_dedup_store`` fixture
    clears the store, so the block reports zeroed counters — we assert every
    documented stats() key is present.
    """
    from unittest.mock import MagicMock, patch

    from proxy.rate_limiter import RateLimiter
    monkeypatch.setattr(
        "proxy.admin.admin_rate_limiter",
        RateLimiter(max_requests=1000, window_seconds=60, name="admin_dedup_test", persist=False),
    )

    mock_connection = MagicMock()
    conn = mock_connection.return_value.__enter__.return_value
    conn.execute.return_value.fetchone.return_value = (1,)

    from proxy.app import app as flask_app
    flask_app.config["TESTING"] = True

    with patch("memory.db.connection", mock_connection), \
         patch("proxy.context._get_task_graph_singleton") as mock_tg:
        mock_tg.return_value._executor.is_available.return_value = True
        client = flask_app.test_client()
        resp = client.get("/health")

    assert resp.status_code == 200
    data = resp.get_json()
    assert "dedup" in data
    dedup = data["dedup"]
    assert "pending_entries" in dedup
    assert "completed_entries" in dedup
    assert "total_stored_bytes" in dedup
    assert "hits" in dedup
    assert "misses" in dedup
    assert "evictions" in dedup
    assert "conflicts" in dedup
