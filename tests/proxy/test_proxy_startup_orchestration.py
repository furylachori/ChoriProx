"""US-HYDRATE: Proxy startup orchestration.

These tests pin the canonical startup sequence:

  1. ``_eager_init_registry()`` (existing)
  2. ``memory.sync.init_db()`` — create schema BEFORE any rate limiter
     reads from it.
  3. ``proxy.state.load_all_rate_limiter_state()`` — hydrate every
     persist=True limiter from SQLite.
  4. ``_cleanup_stale_sessions()`` (existing)

Tests:
  - test_startup_no_warning_on_fresh_db — fresh tmp DB → no
    "RateLimiter[…] no such table" warning at startup.
  - test_startup_hydration_runs_after_init_db — log ordering shows
    init_db BEFORE load_persisted_state.
  - test_persisted_quota_restored_after_restart — write a timestamp
    via one limiter, instantiate a fresh limiter with the same name,
    call the full startup sequence, verify the slot is restored.

These are UNIT-level tests that exercise the canonical
``initialize_proxy_at_startup`` function WITHOUT spinning up the full
Flask app. We monkeypatch ``PROXY_TEST_DB`` to a tmp file per-test,
bypass the ``mock_startup_migrations`` autouse (which would replace
real init_db with a MagicMock — we need real schema), and call the
function directly.

Revert-test protocol: after the suite passes, comment out
``state.load_all_rate_limiter_state()`` in
``proxy.app.initialize_proxy_at_startup`` and confirm
``test_persisted_quota_restored_after_restart`` FAILS.
"""
import logging
import os
import re
from unittest.mock import patch

import pytest


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def real_init_db_path(tmp_path, monkeypatch):
    """Real init_db against a tmp_path SQLite file.

    ``isolate_test_db`` (autouse) already sets PROXY_TEST_DB to a
    tmp_path/memory.db — we just need init_db to actually run, but
    ``mock_startup_migrations`` (autouse, session scope) replaces
    ``memory.sync.init_db`` with a MagicMock.

    The ``memory.db.init_db`` symbol (re-exported from
    ``memory.db_migrations``) is NOT patched by that fixture — only
    ``memory.sync.init_db`` is. ``proxy.app.initialize_proxy_at_startup``
    does ``from memory.sync import init_db`` which WOULD hit the mock.

    For these tests we override the startup function to use
    ``memory.db.init_db`` (the leaf, real implementation) by patching
    ``memory.sync.init_db`` with the real ``memory.db.init_db`` so we
    exercise the full real schema-creation path while still running
    through the proxy.app orchestration code.
    """
    from memory.db import init_db as real_init_db

    # Replace memory.sync.init_db with the real implementation.
    monkeypatch.setattr("memory.sync.init_db", real_init_db)
    yield


# ---------------------------------------------------------------------------
# 1. Fresh-DB startup must NOT emit "no such table" warnings.
# ---------------------------------------------------------------------------


class TestStartupNoWarningOnFreshDB:
    """Cold start against an empty tmp DB: zero "no such table" warnings.

    Pre-fix: 7 "RateLimiter[…] no such table: rate_limiter_state"
    warnings at every cold start, because the limiters tried to read
    SQLite BEFORE init_db() created the schema.

    Post-fix: init_db() runs first, schema is created, hydration runs
    second. No warnings.
    """

    def test_startup_no_warning_on_fresh_db(self, real_init_db_path, tmp_path, caplog):
        from proxy.app import initialize_proxy_at_startup

        # Set a fresh tmp DB path BEFORE running startup.
        os.environ["PROXY_TEST_DB"] = str(tmp_path / "fresh.db")

        caplog.set_level(logging.WARNING)
        with caplog.at_level(logging.WARNING):
            initialize_proxy_at_startup()

        # Find any "no such table: rate_limiter_state" warnings.
        offending = [
            r for r in caplog.records
            if "no such table" in r.getMessage().lower()
            and "rate_limiter_state" in r.getMessage().lower()
        ]
        assert not offending, (
            f"Fresh-DB startup emitted 'no such table: rate_limiter_state' "
            f"warnings: {[r.getMessage() for r in offending]}. "
            f"The fix should ensure init_db() runs BEFORE the rate limiters "
            f"try to read the schema."
        )

        os.environ.pop("PROXY_TEST_DB", None)


# ---------------------------------------------------------------------------
# 2. init_db must run BEFORE load_persisted_state (log ordering).
# ---------------------------------------------------------------------------


class TestStartupOrderingInitDbFirst:
    """The startup sequence is ordered: init_db → load_all_rate_limiter_state.

    This test uses a tracer that records the order in which init_db and
    load_all_rate_limiter_state are called. init_db must come first —
    otherwise we re-introduce the race the fix was designed to prevent.
    """

    def test_startup_hydration_runs_after_init_db(self, real_init_db_path, tmp_path):
        from memory.db import init_db as real_init_db
        from proxy.app import initialize_proxy_at_startup

        os.environ["PROXY_TEST_DB"] = str(tmp_path / "ord.db")

        # Wrap init_db to record the call timestamp.
        call_order = []

        def tracing_init_db(*args, **kwargs):
            call_order.append("init_db")
            return real_init_db(*args, **kwargs)

        # Wrap state.load_all_rate_limiter_state to record its call.
        import proxy.state as state

        original_load = state.load_all_rate_limiter_state

        def tracing_load():
            call_order.append("load_all_rate_limiter_state")
            return original_load()

        with patch("memory.sync.init_db", tracing_init_db):
            with patch.object(state, "load_all_rate_limiter_state", tracing_load):
                initialize_proxy_at_startup()

        assert "init_db" in call_order, (
            f"init_db was not called during startup. order={call_order}"
        )
        assert "load_all_rate_limiter_state" in call_order, (
            f"load_all_rate_limiter_state was not called during startup. order={call_order}"
        )
        assert call_order.index("init_db") < call_order.index(
            "load_all_rate_limiter_state"
        ), (
            f"init_db must run BEFORE load_all_rate_limiter_state to avoid "
            f"the 'no such table' race. Actual order: {call_order}"
        )

        os.environ.pop("PROXY_TEST_DB", None)


# ---------------------------------------------------------------------------
# 3. Persisted quota restored after restart.
# ---------------------------------------------------------------------------


class TestPersistedQuotaRestoredAfterRestart:
    """Insert a timestamp via one limiter, then create a NEW instance
    with the same name, run the full startup, verify the slot is
    restored.
    """

    def test_persisted_quota_restored_after_restart(self, real_init_db_path, tmp_path):
        from memory.db import immediate_transaction, init_db as real_init_db
        from proxy.rate_limiter import RateLimiter
        from proxy.app import initialize_proxy_at_startup
        import proxy.state as state

        db_path = str(tmp_path / "restart.db")
        os.environ["PROXY_TEST_DB"] = db_path

        # First "process" — initialize the schema, then write a row
        # via a real limiter with persist=True. We must init_db BEFORE
        # the limiter tries to write — that's the whole point of the
        # fix. Use a unique name to avoid colliding with the
        # singleton limiters in proxy.state.
        limiter_name = "restart-test"

        # Initialize the schema (the "first process" startup).
        real_init_db()

        first_limiter = RateLimiter(
            max_requests=5,
            window_seconds=60,
            name=limiter_name,
            persist=True,
        )
        # Drive 3 checks so the limiter's bucket has 3 timestamps.
        first_limiter.check("restart-key")
        first_limiter.check("restart-key")
        first_limiter.check("restart-key")
        # _save is a private method, but we use it here to deterministically
        # write the current state without waiting for the time gate.
        first_limiter._save()

        # Verify the row exists in SQLite.
        with immediate_transaction(pooled=True) as conn:
            rows = conn.execute(
                "SELECT key, timestamps FROM rate_limiter_state WHERE limiter_name = ?",
                (limiter_name,),
            ).fetchall()
        assert len(rows) == 1, (
            f"Expected 1 row written by first limiter, got {len(rows)}. "
            f"Rows: {rows!r}"
        )

        # Second "process" — create a NEW limiter with the same name
        # (in-memory state is fresh), inject it into proxy.state so the
        # auto-discovery loop in state.load_all_rate_limiter_state finds
        # it, and run the full startup sequence.
        second_limiter = RateLimiter(
            max_requests=5,
            window_seconds=60,
            name=limiter_name,
            persist=True,
        )
        # Pre-condition: second_limiter's _entries is empty (fresh process).
        assert len(second_limiter._entries) == 0, (
            "Pre-hydration: second limiter should start with empty _entries"
        )

        # Inject into proxy.state so auto-discovery picks it up.
        import proxy.state as state
        state._test_restart_limiter = second_limiter
        try:
            # Drive startup — must hydrate second_limiter via auto-discovery.
            with patch("memory.sync.init_db", real_init_db):
                initialize_proxy_at_startup()
        finally:
            del state._test_restart_limiter

        # Post-condition: second_limiter's bucket for 'restart-key'
        # should be hydrated with the 3 entries we wrote.
        dq = second_limiter._entries.get("restart-key")
        assert dq is not None, (
            f"After startup, expected 'restart-key' bucket to be "
            f"hydrated, but _entries={dict(second_limiter._entries)!r}"
        )
        assert len(dq) == 3, (
            f"Expected 3 entries hydrated after restart, got len={len(dq)}. "
            f"startup did not restore persisted state."
        )

        os.environ.pop("PROXY_TEST_DB", None)