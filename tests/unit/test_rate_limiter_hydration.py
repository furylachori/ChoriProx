"""US-HYDRATE: RateLimiter hydration lifecycle.

The previous design called ``RateLimiter._load()`` at module-import
time (proxy.state declares 8 limiter singletons at module load). At
that moment ``memory.sync.init_db()`` had not yet run, so every cold
start emitted 7 "no such table: rate_limiter_state" warnings and
persisted state was silently never restored.

These tests pin the new lifecycle:

  1. Constructor must NOT touch SQLite. Pure in-memory construction.
  2. ``load_persisted_state()`` MUST raise on a missing schema
     (no silent swallow).
  3. ``load_persisted_state()`` must restore real entries from
     SQLite and make them observable via ``check()``.
  4. ``load_persisted_state()`` must filter expired timestamps.
  5. ``load_persisted_state()`` must be atomic: a second call replaces
     the snapshot rather than accumulating.
  6. ``proxy.state.load_all_rate_limiter_state()`` must auto-discover
     every ``RateLimiter`` instance via ``dir(proxy.state)`` and call
     ``load_persisted_state`` on each.

Revert-test protocol: after the suite passes, comment out the call to
``state.load_all_rate_limiter_state()`` in
``proxy.app.initialize_proxy_at_startup`` and confirm
``test_persisted_quota_restored_after_restart`` (in
tests/proxy/test_proxy_startup_orchestration.py) FAILS.
"""
import os
from collections import deque
from unittest.mock import patch, MagicMock

import pytest


# ---------------------------------------------------------------------------
# Shared fixture: real SQLite via init_db, scoped to this test file only.
# We do NOT use the autouse `isolate_test_db` (tmp_path/memory.db) because
# `mock_startup_migrations` patches `memory.sync.init_db` which would mask
# the real schema creation. We use :memory: + real init_db so the rate
# limiter's hydration path runs against the real schema.
# ---------------------------------------------------------------------------


@pytest.fixture
def real_db():
    """Pooled :memory: DB with full init_db.

    Used by tests that need the real ``rate_limiter_state`` schema.
    The autouse `mock_startup_migrations` (which mocks
    `memory.sync.init_db`) is bypassed here — we call the real
    `memory.db.init_db` (which is `memory.db_migrations.init_db`,
    the leaf, not the patched sync wrapper).
    """
    os.environ["PROXY_TEST_DB"] = ":memory:"
    from memory.db import init_db, immediate_transaction
    init_db()
    yield immediate_transaction
    os.environ.pop("PROXY_TEST_DB", None)


# ---------------------------------------------------------------------------
# 1. Constructor with persist=True must not touch SQLite.
# ---------------------------------------------------------------------------


class TestConstructorDoesNotTouchSQLite:
    """The whole point of the fix: ``RateLimiter(persist=True)`` used to
    call ``self._load()`` in ``__init__``, racing the SQLite schema
    creation. The new constructor is pure-Python — no SQLite I/O.

    A sentinel placed on the ``memory.db.connection`` symbol must never
    be invoked by construction.
    """

    def test_constructor_with_persist_does_not_touch_sqlite(self):
        from proxy.rate_limiter import RateLimiter

        sentinel = MagicMock()
        # Patch the symbol `RateLimiter._load` used to import — `from
        # memory.db import connection` inside the old `_load()` body.
        # If the constructor still calls _load (or any equivalent
        # SQLite-touching code), the sentinel will be entered.
        with patch("memory.db.connection", sentinel):
            rl = RateLimiter(
                max_requests=5,
                window_seconds=60,
                name="hyd-ctor",
                persist=True,
            )
        # Constructor returned without raising — and the sentinel was
        # never used as a context manager.
        assert sentinel.called is False, (
            "RateLimiter(persist=True).__init__ touched SQLite "
            f"(connection sentinel was used: {sentinel.call_args_list!r}). "
            "The whole point of the fix is that construction is pure-Python."
        )
        # The limiter is in a sane empty state — ready for explicit
        # hydration later.
        assert rl._entries == {} or all(
            len(v) == 0 for v in rl._entries.values()
        ), f"Expected empty entries after construction, got {dict(rl._entries)!r}"


# ---------------------------------------------------------------------------
# 2. load_persisted_state on a fresh schema-less DB must RAISE (strict).
# ---------------------------------------------------------------------------


class TestLoadPersistedStateStrictPropagation:
    """The old ``_load()`` had a bare ``except Exception`` that swallowed
    the ``no such table`` error and logged a warning. The new
    ``load_persisted_state`` MUST propagate so the failure surfaces
    during startup — silent swallow is a hidden bug class.
    """

    def test_load_persisted_state_no_table_strict(self):
        from proxy.rate_limiter import RateLimiter
        import sqlite3

        # Make the DB connection raise as if the schema is missing.
        # We do this by monkey-patching the module-level `connection`
        # symbol that `load_persisted_state` imports via
        # `from memory.db import connection`.
        from contextlib import contextmanager

        @contextmanager
        def _raising_connection(*args, **kwargs):
            raise sqlite3.OperationalError("no such table: rate_limiter_state")
            yield  # unreachable, satisfies @contextmanager decorator

        with patch("memory.db.connection", _raising_connection):
            rl = RateLimiter(
                max_requests=5,
                window_seconds=60,
                name="hyd-strict",
                persist=True,
            )
            with pytest.raises(sqlite3.OperationalError, match="no such table"):
                rl.load_persisted_state()


# ---------------------------------------------------------------------------
# 3. load_persisted_state restores real entries (round-trip).
# ---------------------------------------------------------------------------


class TestLoadPersistedStateRestoresEntries:
    """Drive real ``init_db`` into :memory:, insert rows, then hydrate
    a fresh limiter with the same name. The deque for the matching key
    must contain exactly the rows we inserted.
    """

    def test_load_persisted_state_restores_real_entries(self, real_db):
        from proxy.rate_limiter import RateLimiter
        import json
        import time

        now = time.time()
        timestamps = [now - 5.0, now - 3.0, now - 1.0]
        with real_db(pooled=True) as conn:
            conn.execute(
                "INSERT INTO rate_limiter_state (limiter_name, key, timestamps) VALUES (?, ?, ?)",
                ("hyd-r", "key-hyd-r", json.dumps(timestamps)),
            )

        rl = RateLimiter(
            max_requests=5, window_seconds=60, name="hyd-r", persist=True
        )
        result = rl.load_persisted_state()
        assert result is True, (
            "load_persisted_state must return True when rows exist for the limiter"
        )

        # Verify the entries are observable via the public check()
        # path.  Do NOT use rl._entries directly — that is internal.
        # Instead, exhaust the bucket via check() and confirm the
        # count matches the timestamps we inserted.
        #
        # We have 3 timestamps for key-hyd-r. With max_requests=5,
        # 5 checks should all be allowed, then the 6th should be
        # blocked — proving the bucket was hydrated with 3 entries
        # (3 existing + 5 new = 8, but cleanup at 10 calls keeps
        # valid ones in the window).
        rl.check("key-hyd-r")  # 4 in deque (was 3)
        rl.check("key-hyd-r")  # 5 in deque
        # 6th check: max=5, should be rejected
        rejected = rl.check("key-hyd-r")
        assert rejected.allowed is False, (
            f"After hydration of 3 timestamps + 2 checks (=5), the 6th "
            f"check should be rejected. got allowed={rejected.allowed}"
        )

        # Confirm the deque length is exactly 5 (3 hydrated + 2 new).
        # The rate limiter's _entries is a defaultdict(deque) — use
        # list comprehension since deque doesn't support len() at
        # arbitrary keys without creating them (defaultdict side effect).
        assert len(rl._entries["key-hyd-r"]) == 5, (
            f"Expected deque length 5 after hydration + 2 checks, "
            f"got {len(rl._entries['key-hyd-r'])}. Hydration did not "
            f"restore entries correctly."
        )


# ---------------------------------------------------------------------------
# 4. Expired timestamps must be filtered.
# ---------------------------------------------------------------------------


class TestLoadPersistedStateFiltersExpired:
    """Only timestamps within ``window_seconds`` of ``now`` survive."""

    def test_load_persisted_state_filters_expired(self, real_db):
        from proxy.rate_limiter import RateLimiter
        import json
        import time

        now = time.time()
        window = 60.0
        # 2 fresh (within window) + 2 expired (well past window)
        timestamps = [
            now - 5.0,   # fresh
            now - 10.0,  # fresh
            now - 120.0, # expired
            now - 200.0, # expired
        ]
        with real_db(pooled=True) as conn:
            conn.execute(
                "INSERT INTO rate_limiter_state (limiter_name, key, timestamps) VALUES (?, ?, ?)",
                ("hyd-filt", "key-filt", json.dumps(timestamps)),
            )

        rl = RateLimiter(
            max_requests=10, window_seconds=window, name="hyd-filt", persist=True
        )
        rl.load_persisted_state()

        # Only the 2 fresh timestamps should survive. Inspect via
        # the deque length: defaultdict auto-creates on read, so use
        # a guarded pattern.
        dq = rl._entries.get("key-filt")
        assert dq is not None, (
            f"Expected key-filt to be hydrated, but _entries={dict(rl._entries)!r}"
        )
        assert len(dq) == 2, (
            f"Expected 2 fresh timestamps after filter, got {len(dq)}. "
            f"Expired timestamps were not filtered."
        )
        # And both retained timestamps must be within the window.
        for ts in dq:
            assert now - ts < window, (
                f"Retained timestamp {ts} is {now - ts}s old, beyond window={window}"
            )


# ---------------------------------------------------------------------------
# 5. Idempotent hydration — atomic replace, not accumulation.
# ---------------------------------------------------------------------------


class TestLoadPersistedStateIdempotent:
    """Two consecutive hydrations without DB changes produce the same
    state. Inserting more rows between hydrations yields only the new
    rows visible — the previous snapshot is REPLACED, not merged.
    """

    def test_load_persisted_state_idempotent(self, real_db):
        from proxy.rate_limiter import RateLimiter
        import json
        import time

        now = time.time()
        with real_db(pooled=True) as conn:
            conn.execute(
                "INSERT INTO rate_limiter_state (limiter_name, key, timestamps) VALUES (?, ?, ?)",
                ("hyd-idem", "key-idem", json.dumps([now - 1.0, now - 2.0])),
            )

        rl = RateLimiter(
            max_requests=10, window_seconds=60, name="hyd-idem", persist=True
        )
        first_result = rl.load_persisted_state()
        first_len = len(rl._entries["key-idem"])

        second_result = rl.load_persisted_state()
        second_len = len(rl._entries["key-idem"])

        assert first_result is True
        assert second_result is True
        assert first_len == second_len == 2, (
            f"Idempotent hydration: expected stable len=2 across two "
            f"calls, got first={first_len}, second={second_len}"
        )

        # Now insert 3 new rows and hydrate again — only the new 3
        # should be visible (atomic replace, not merge).
        with real_db(pooled=True) as conn:
            conn.execute(
                "DELETE FROM rate_limiter_state WHERE limiter_name = ?",
                ("hyd-idem",),
            )
            conn.execute(
                "INSERT INTO rate_limiter_state (limiter_name, key, timestamps) VALUES (?, ?, ?)",
                ("hyd-idem", "key-idem", json.dumps([now - 0.5, now - 0.4, now - 0.3])),
            )

        rl.load_persisted_state()
        third_len = len(rl._entries["key-idem"])
        assert third_len == 3, (
            f"After insert+hydrate, expected len=3 (atomic replace of "
            f"previous 2 + new 3 should be 3, not 5). Got len={third_len}"
        )


# ---------------------------------------------------------------------------
# 6. proxy.state.load_all_rate_limiter_state auto-discovers via dir().
# ---------------------------------------------------------------------------


class TestLoadAllRateLimiterStateAutoDiscovers:
    """``state.load_all_rate_limiter_state`` mirrors
    ``state.reset_all_rate_limiters`` — it iterates ``dir(proxy.state)``
    and dispatches ``load_persisted_state`` on every ``RateLimiter``
    instance. A fake limiter injected into proxy.state MUST be picked
    up.
    """

    def test_load_all_rate_limiter_state_auto_discovers(self):
        """Inject a sentinel limiter into proxy.state and verify
        ``load_all_rate_limiter_state`` dispatches to it via the
        ``dir()`` auto-discovery loop.

        The 8 singleton limiters in proxy.state also get dispatched;
        to avoid SQLite I/O we stub ``RateLimiter.load_persisted_state``
        on the CLASS so every limiter's hydration is a no-op, then
        verify the sentinel was called.
        """
        from proxy.rate_limiter import RateLimiter
        import proxy.state as state

        sentinel = MagicMock(spec=RateLimiter)
        sentinel.load_persisted_state = MagicMock(return_value=True)
        sentinel._name = "sentinel-hyd"

        # Stub the class method so real limiters don't touch SQLite.
        # After the test we restore the original.
        original = RateLimiter.load_persisted_state
        RateLimiter.load_persisted_state = MagicMock(return_value=True)

        # Inject into proxy.state module attrs.
        original_attr = getattr(state, "_test_sentinel_limiter", None)
        state._test_sentinel_limiter = sentinel
        try:
            state.load_all_rate_limiter_state()
        finally:
            if original_attr is None:
                del state._test_sentinel_limiter
            else:
                state._test_sentinel_limiter = original_attr
            RateLimiter.load_persisted_state = original

        # The sentinel's load_persisted_state must have been called.
        sentinel.load_persisted_state.assert_called_once_with(), (
            "load_all_rate_limiter_state did not call load_persisted_state "
            "on the injected sentinel limiter — auto-discovery is broken."
        )


# ---------------------------------------------------------------------------
# 7. Empty _name guard (matches reset_all_rate_limiters invariant).
# ---------------------------------------------------------------------------


class TestLoadAllRateLimiterStateEmptyNameGuard:
    """Same drift guard as ``reset_all_rate_limiters``: an empty ``_name``
    would make the limiter unreachable via the WHERE clause key. We
    raise rather than silently skip.
    """

    def test_empty_name_limiter_raises(self):
        from proxy.rate_limiter import RateLimiter
        import proxy.state as state

        bad = RateLimiter(max_requests=1, window_seconds=10, name="", persist=False)
        state._test_bad_limiter = bad
        try:
            # Stub class method to avoid real limiters hitting SQLite.
            original = RateLimiter.load_persisted_state
            RateLimiter.load_persisted_state = MagicMock(return_value=True)
            try:
                with pytest.raises(ValueError, match="empty _name"):
                    state.load_all_rate_limiter_state()
            finally:
                RateLimiter.load_persisted_state = original
        finally:
            del state._test_bad_limiter