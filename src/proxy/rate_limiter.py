"""Per-key sliding-window rate limiter.

Replaces the ad-hoc rate limiting state (locks, dicts, cleanup timers)
that was previously scattered across state.py.
"""
import logging
import sqlite3
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# M8-4: time-based gates. The % 10 call-count gates only fire on the
# 10th, 20th, ... call. Under low traffic, the persistent limiters
# never write to SQLite and the per-key dict grows unbounded. These
# intervals bound the staleness independent of call volume.
_CLEANUP_INTERVAL_S: float = 60.0
_SAVE_INTERVAL_S: float = 30.0


@dataclass
class RateLimitResult:
    """Result of a rate limit check."""
    allowed: bool
    retry_after: float = 0.0


class RateLimiter:
    """Per-key sliding-window rate limiter. One instance per domain."""

    def __init__(self, max_requests: int, window_seconds: float, name: str = "", persist: bool = False):
        self._lock = threading.Lock()
        self._entries: dict[str, deque] = defaultdict(deque)
        self._max = max_requests
        self._window = window_seconds
        self._name = name
        self._call_count = 0
        self._persist = persist
        # Initialize timestamps so the time gates measure "since last
        # event" not "since epoch". Re-anchored on load_persisted_state()
        # so the gate covers the rehydration path too — but the SQLite
        # call is deferred until init_db() has run (see proxy.app
        # initialize_proxy_at_startup). Constructing a persist=True
        # limiter at module-import time used to call _load() here,
        # which raced init_db() and emitted 7 "no such table" warnings
        # on every cold start.
        self._last_cleanup: float = time.time()
        self._last_save: float = time.time()

    def check(self, key: str) -> RateLimitResult:
        """Check if allowed. Records the request if allowed."""
        with self._lock:
            now = time.time()
            self._call_count += 1
            dq = self._entries[key]
            # M8-4: cleanup fires on call-count modulo OR time elapsed.
            if self._call_count % 10 == 0 or (now - self._last_cleanup > _CLEANUP_INTERVAL_S):
                self._cleanup(now)
                self._last_cleanup = now
            cutoff = now - self._window
            while dq and dq[0] < cutoff:
                dq.popleft()
            if len(dq) >= self._max:
                retry_after = dq[0] + self._window - now
                result = RateLimitResult(allowed=False, retry_after=max(0.0, retry_after))
                from proxy.metrics import record_rate_limiter_check, record_rate_limiter_rejection
                record_rate_limiter_check(self._name)
                record_rate_limiter_rejection(self._name)
            else:
                dq.append(now)
                result = RateLimitResult(allowed=True)
                from proxy.metrics import record_rate_limiter_check
                record_rate_limiter_check(self._name)

            # M8-4: same pattern for save — modulo OR time elapsed.
            # (save only happens for persistent limiters; the gate
            # also covers the call to _save() at the end so the
            # timestamp is updated atomically with the save. The
            # _save() method has its own internal try/except, but
            # this outer try/except also guards against the rare case
            # of an _save override (e.g. test mocks) that doesn't
            # itself swallow exceptions.)
            if self._persist and (self._call_count % 10 == 0 or (now - self._last_save > _SAVE_INTERVAL_S)):
                try:
                    self._save()
                except Exception as _save_err:
                    logger.warning("RateLimiter[%s] _save raised: %s", self._name, _save_err)
                self._last_save = now
            elif not self._persist and (now - self._last_save > _SAVE_INTERVAL_S):
                # M8-4: for non-persistent limiters, the gate still
                # tracks "since-last-event" so the next persist switch
                # (if the limiter is ever reconfigured) would not
                # immediately fire. No _save() call here.
                self._last_save = now

            return result

    def check_only(self, key: str) -> RateLimitResult:
        """Check without recording."""
        with self._lock:
            now = time.time()
            # M8-4: check_only also gets time-based cleanup so the
            # per-key dict can't grow unbounded under gate-before-
            # commit traffic. No save here — check_only doesn't
            # consume a slot so it shouldn't be persisted.
            if now - self._last_cleanup > _CLEANUP_INTERVAL_S:
                self._cleanup(now)
                self._last_cleanup = now
            dq = self._entries[key]
            cutoff = now - self._window
            while dq and dq[0] < cutoff:
                dq.popleft()
            if len(dq) >= self._max:
                retry_after = dq[0] + self._window - now
                return RateLimitResult(allowed=False, retry_after=max(0.0, retry_after))
            return RateLimitResult(allowed=True)

    def _cleanup(self, now: float) -> None:
        cutoff = now - self._window
        empty_keys = []
        for key, dq in self._entries.items():
            while dq and dq[0] < cutoff:
                dq.popleft()
            if not dq:
                empty_keys.append(key)
        for key in empty_keys:
            del self._entries[key]

    def cleanup(self) -> None:
        """Public cleanup method for external periodic cleanup."""
        with self._lock:
            self._cleanup(time.time())

    def reset(self) -> None:
        """Clear all rate limit entries. Used for test cleanup."""
        with self._lock:
            self._entries.clear()
            self._call_count = 0
            # Reset timestamps so the time gates re-fire immediately.
            self._last_cleanup = time.time()
            self._last_save = time.time()

    def load_persisted_state(self) -> bool:
        """Restore entries from SQLite. Returns True if data was loaded, False otherwise.

        Public method — called explicitly by the startup sequence
        (``proxy.app.initialize_proxy_at_startup`` → ``proxy.state.load_all_rate_limiter_state``)
        AFTER ``memory.sync.init_db()`` has created the schema. The
        constructor no longer calls this; the SQLite round-trip is
        deferred so cold starts don't emit "no such table" warnings.

        - No-op when ``self._persist`` is False (in-memory limiters).
        - Reads rows OUTSIDE the limiter lock (SQLite I/O would block
          unrelated check() calls otherwise).
        - Filters expired timestamps against ``self._window``.
        - Acquires ``self._lock`` ONCE for the atomic replace of
          ``self._entries`` and re-anchor of ``_last_cleanup`` /
          ``_last_save``.
        - Propagates exceptions. Startup is responsible for handling
          failures; we do NOT silently swallow (per AGENTS.md "no
          silent failure" — startup catches at the boundary).
        """
        if not self._persist:
            return False
        from memory.db import connection
        import json

        # Phase 1: read rows OUTSIDE self._lock so concurrent check()
        # calls on this limiter are not blocked by SQLite I/O.
        with connection(pooled=True) as conn:
            rows = conn.execute(
                "SELECT key, timestamps FROM rate_limiter_state WHERE limiter_name = ?",
                (self._name,),
            ).fetchall()

        # Phase 2: filter expired timestamps (also outside the lock —
        # pure CPU work, no need to hold the limiter's critical
        # section against unrelated check() calls).
        now = time.time()
        new_entries: dict[str, deque] = {}
        for row in rows:
            timestamps = json.loads(row["timestamps"])
            valid = [t for t in timestamps if now - t < self._window]
            if valid:
                new_entries[row["key"]] = deque(valid)

        # Phase 3: acquire the limiter lock ONCE and atomically swap
        # _entries + re-anchor the time gates. check() may have been
        # racing with this read; under the lock, we now replace the
        # snapshot in one critical section.
        loaded = bool(new_entries) or bool(rows)
        with self._lock:
            self._entries = defaultdict(deque, new_entries)
            # M8-4: re-anchor the time gates to "since rehydration" so
            # the first post-restart check doesn't immediately fire
            # (preserves the modulo-10 contract for the first 10 calls
            # after a restart on a previously-idle limiter).
            self._last_cleanup = now
            self._last_save = now
        return loaded

    def _save(self):
        """Persist entries to SQLite."""
        try:
            from memory.db import immediate_transaction
            import json
            with immediate_transaction(pooled=True) as conn:
                for key, timestamps in self._entries.items():
                    conn.execute(
                        "INSERT OR REPLACE INTO rate_limiter_state (limiter_name, key, timestamps) VALUES (?, ?, ?)",
                        (self._name, key, json.dumps(list(timestamps)))
                    )
        except (OSError, sqlite3.Error) as _save_err:
            logger.warning("RateLimiter[%s] _save to SQLite failed: %s", self._name, _save_err)
