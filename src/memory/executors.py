"""
Bounded thread pool executors for memory system operations.

Two separate pools prevent cross-contamination: a 90-second tool re-query
loop must not starve sync operations.
"""

import atexit
import os
import signal
import threading
import time
import logging
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger("memory.executors")

_sync_pool = None
_query_pool = None
_summary_pool = None
_timeout_pool = None
_pool_lock = threading.Lock()
_shutdown_complete = False
_shutdown_requested = False

# Per-session lock to prevent concurrent submit_sync for the same session_id.
_session_locks: dict[str, threading.Lock] = {}
_session_locks_lock = threading.Lock()

# Track hung workers for observability. The _timeout_guard fires after 600s
# and assumes the worker is stuck. The future's _on_done callback removes
# the entry when the worker eventually completes.
#
# Keyed by id(future) because Python thread objects don't have a stable
# identity after the underlying thread terminates. The dict is bounded by
# the 4-worker pool — at most 4 entries can exist concurrently.
_hung_futures: dict[int, tuple[str, float]] = {}
_hung_futures_lock = threading.Lock()


def _check_shutdown_locked() -> None:
    """Raise if executor shutdown has begun.

    Called from every pool getter and every submit path under
    ``_pool_lock`` so concurrent creation/submission cannot race
    past the shutdown snapshot. After shutdown has been requested,
    no new executors are created and no new work is submitted.
    """
    if _shutdown_requested or _shutdown_complete:
        raise RuntimeError("Memory executors are shutting down")


def _get_sync_pool():
    global _sync_pool
    if _sync_pool is None:
        with _pool_lock:
            _check_shutdown_locked()
            if _sync_pool is None:
                _sync_pool = ThreadPoolExecutor(max_workers=4, thread_name_prefix="mem-sync")
    else:
        # Fast path: even if the pool is set, we must reject if
        # shutdown has been requested. Hold the lock briefly to
        # read the flag.
        with _pool_lock:
            _check_shutdown_locked()
    return _sync_pool


def _get_query_pool():
    global _query_pool
    if _query_pool is None:
        with _pool_lock:
            _check_shutdown_locked()
            if _query_pool is None:
                _query_pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix="mem-query")
    else:
        with _pool_lock:
            _check_shutdown_locked()
    return _query_pool


def _get_summary_pool():
    global _summary_pool
    if _summary_pool is None:
        with _pool_lock:
            _check_shutdown_locked()
            if _summary_pool is None:
                _summary_pool = ThreadPoolExecutor(max_workers=4, thread_name_prefix="mem-summary")
    else:
        with _pool_lock:
            _check_shutdown_locked()
    return _summary_pool


def _get_timeout_pool():
    global _timeout_pool
    if _timeout_pool is None:
        with _pool_lock:
            _check_shutdown_locked()
            if _timeout_pool is None:
                _timeout_pool = ThreadPoolExecutor(max_workers=4, thread_name_prefix="timeout-guard")
    else:
        with _pool_lock:
            _check_shutdown_locked()
    return _timeout_pool


def _check_future_error(future):
    """Log any exception that occurred in the submitted task."""
    try:
        exc = future.exception()
        if exc is not None:
            logger.error("Uncaught exception in sync task: %s", exc, exc_info=exc)
    except Exception:
        pass


def _shutdown_thread_pool(timeout=30):
    """Gracefully shutdown thread pools, waiting up to ``timeout`` seconds.

    FIX c7-arch-iter4: the previous implementation called
    ``pool.shutdown(wait=True, cancel_futures=False)`` which blocks
    indefinitely until ALL pending tasks finish — the ``timeout``
    parameter was logged but ignored, so a wedged worker could hang the
    atexit handler forever.

    Bounded-wait pattern: each pool's ``shutdown`` runs in a daemon
    thread; we ``Event.wait(timeout)`` for completion. If the grace
    window expires, we log a WARNING and return control to the caller
    (the interpreter); the daemon shutdown will complete asynchronously
    on process exit. Subsequent ``ThreadPoolExecutor.shutdown()`` calls
    on an already-shutting-down pool are no-ops per the
    ``concurrent.futures`` spec, so the existing ``atexit`` registration
    of ``_shutdown_thread_pool`` (the live atexit path via ``_graceful_db_shutdown``) is unaffected.

    Idempotent — safe to call multiple times.

    The locked snapshot prevents a leaked executor generation. Submission
    to an existing executor after snapshot is not prevented — callers
    must check shutdown in submit paths.

    Args:
        timeout: Maximum seconds to wait for pool drain. ``<= 0`` skips
            the wait entirely and forces ``cancel_futures=True``.
    """
    global _shutdown_complete, _shutdown_requested
    with _pool_lock:
        if _shutdown_complete:
            return
        _shutdown_requested = True
        _shutdown_complete = True
        pools = [
            ("sync", _sync_pool),
            ("query", _query_pool),
            ("summary", _summary_pool),
            ("timeout", _timeout_pool),
        ]
        # Reset tracking dict under the pool lock so a concurrent
        # _timeout_guard cannot re-populate during shutdown. The
        # watchdog submits to _timeout_pool, which is also being
        # shut down, so no new entries can arrive after this clear.
        with _hung_futures_lock:
            _hung_futures.clear()

    logger.info("Gracefully shutting down memory thread pools (timeout=%ds)...", timeout)

    if timeout <= 0:
        # Hard shutdown, no waiting
        for name, pool in pools:
            if pool is not None:
                try:
                    pool.shutdown(wait=False, cancel_futures=True)
                except Exception as e:
                    logger.warning("Failed to shut down %s pool: %s", name, e)
        return

    # Best-effort bounded wait: run shutdown in a daemon thread, join via
    # Event.wait(timeout). The daemon thread completes asynchronously even
    # if the wait times out — no leaked thread on process exit.
    for name, pool in pools:
        if pool is None:
            continue
        shutdown_done = threading.Event()

        # Bind every variable to the default. In particular, the
        # done event must be bound to THIS pool's event — otherwise a
        # late finish from pool A could erroneously set pool B's
        # event, and the shutdown caller would believe B drained
        # while B is still using SQLite.
        def _do_shutdown(
            p=pool,
            n=name,
            done=shutdown_done,
        ):
            try:
                p.shutdown(wait=True, cancel_futures=False)
                logger.debug("Shut down %s pool", n)
            except Exception as e:
                logger.warning("Failed to shut down %s pool: %s", n, e)
            finally:
                done.set()

        t = threading.Thread(
            target=_do_shutdown,
            daemon=True,
            name=f"shutdown-{name}",
        )
        t.start()
        if not shutdown_done.wait(timeout=timeout):
            logger.warning(
                "_shutdown_thread_pool: drain timed out after %ds for %s pool; "
                "shutdown continuing asynchronously",
                timeout, name,
            )

    logger.info("Memory thread pools gracefully shut down")


def _signal_handler(signum, frame):
    """Signal handler: set flag only, re-raise with default.

    Actual cleanup happens via atexit (_graceful_db_shutdown -> _shutdown_thread_pool) on process exit.
    Calling pool.shutdown() from a signal handler is unsafe due to
    re-entrant lock acquisition if the signal arrives mid-submit.
    """
    global _shutdown_requested
    _shutdown_requested = True
    signal.signal(signum, signal.SIG_DFL)
    os.kill(os.getpid(), signum)


# Importing memory.db_pool registers the terminal-shutdown atexit fallback
# FIRST. The graceful db shutdown is registered AFTER it so atexit's LIFO
# order is:
#   1. _graceful_db_shutdown  (drain workers, then DB terminal shutdown)
#   2. db_pool._shutdown_global_pool  (idempotent fallback)
from memory import db_pool  # noqa: E402, F401  # registers terminal atexit


# ---------------------------------------------------------------------------
# C7+ graceful shutdown: drain thread pools → DB terminal shutdown.
#
# The db_pool fallback is registered earlier; this is the authoritative
# graceful callback. LIFO order is:
#   1. this function
#   2. db_pool._shutdown_global_pool  (idempotent, no-op when terminal)
# ---------------------------------------------------------------------------
def _graceful_db_shutdown(timeout=10):
    """atexit handler: drain thread pools → terminally shutdown the pool.

    The thread-pool drain consumes most of the deadline. The remaining
    time is passed to db_pool._shutdown_global_pool so the entire
    graceful path has a single shared, bounded budget.
    """
    # Split the deadline: 70% to thread-pool drain, 30% to db pool.
    db_budget = max(0.5, timeout * 0.3)
    pool_budget = max(0.5, timeout - db_budget)

    try:
        _shutdown_thread_pool(timeout=pool_budget)
    except Exception as exc:
        logger.warning(
            "graceful_db_shutdown: thread-pool drain failed: %s",
            exc,
        )

    try:
        db_pool._shutdown_global_pool(timeout=db_budget)
    except Exception as exc:
        logger.warning(
            "graceful_db_shutdown: database shutdown failed: %s",
            exc,
        )


atexit.register(_graceful_db_shutdown)


# SIGTERM/SIGINT handlers (only register if we're in main thread)
try:
    signal.signal(signal.SIGTERM, _signal_handler)
    signal.signal(signal.SIGINT, _signal_handler)
except ValueError:
    # Not in main thread, signal handlers can't be registered
    pass


def _release_session_token(sid, token):
    """Identity-checked release of a session ownership token.

    Used by every code path that may remove the token: completion
    callback, timeout guard, watchdog failure path, and submission
    failure. Each path passes the exact token it acquired; a
    mismatched token means the entry has been replaced by a newer
    submit for the same session.
    """
    with _session_locks_lock:
        if _session_locks.get(sid) is token:
            del _session_locks[sid]


def _on_done(f, sid, token):
    """Normal completion callback. Releases the session token AND clears
    the hung-worker tracking entry (if the timeout guard previously
    marked this future as hung). The dict is bounded by the 4-worker
    pool — at most 4 concurrent entries can exist.
    """
    _release_session_token(sid, token)
    with _hung_futures_lock:
        _hung_futures.pop(id(f), None)


def _timeout_guard(f, sid, token, timeout_sec=600):
    """600s watchdog for a sync_pool future. If the future hasn't finished
    by ``timeout_sec``, the worker is presumed hung. The per-session
    debounce token is released either way (normal completion or timeout)
    so the session can accept new submits.

    The guard is strictly observable: it logs a WARNING with the
    session_id and current hung count, and an ERROR when all 4 workers
    are presumed hung. Python threads cannot be killed, so the fix is
    visibility + bounding, not thread murder.
    """
    try:
        f.result(timeout=timeout_sec)
    except Exception:
        with _hung_futures_lock:
            _hung_futures[id(f)] = (sid, time.time())
            count = len(_hung_futures)
        logger.warning(
            "Sync worker hung after %ds: session=%s; "
            "current hung-worker count: %d / 4",
            timeout_sec, sid, count,
        )
        if count >= 4:
            logger.error(
                "All 4 sync workers presumed hung. "
                "Session_ids: %s",
                [s for s, _ in _hung_futures.values()],
            )
    finally:
        _release_session_token(sid, token)


def submit_sync(fn, *args, **kwargs):
    """Submit a sync operation to the sync pool.

    Per-session lock prevents concurrent sync jobs for the same
    session_id. If a sync is already queued or running for that
    session, returns None.

    The session ownership token is removed only via the
    identity-checked ``_release_session_token`` helper. The normal
    completion callback AND the timeout guard AND the watchdog
    submission-failure path AND the cancellation path all funnel
    through that helper, so a stale completion can never delete a
    newer submit's ownership token.
    """
    if args:
        session_id = args[0]
    else:
        session_id = kwargs.get("session_id")

    if session_id:
        _token = object()
        with _session_locks_lock:
            if session_id in _session_locks:
                logger.info(
                    "Sync debounced for session=%s "
                    "(already queued/running)", session_id
                )
                return None
            _session_locks[session_id] = _token

        try:
            future = _get_sync_pool().submit(fn, *args, **kwargs)
        except Exception:
            _release_session_token(session_id, _token)
            raise

        # Identity-checked done callback. Only the exact token can be
        # removed by this callback; a newer submit for the same
        # session will see its own token and the older completion
        # is a no-op. Closure captures the per-submit session_id + token;
        # the actual work is in the module-level _on_done (so it can be
        # imported + tested in isolation).
        def _on_done_cb(_f, sid=session_id, token=_token):
            _on_done(_f, sid, token)

        future.add_done_callback(_check_future_error)
        future.add_done_callback(_on_done_cb)

        def _timeout_guard_cb(f, sid=session_id, token=_token, timeout_sec=600):
            _timeout_guard(f, sid, token, timeout_sec)

        try:
            _get_timeout_pool().submit(_timeout_guard_cb)
        except Exception as exc:
            # Watchdog submission failed. The sync future is still
            # running. Keep the ownership token: the identity-checked
            # done callback will release it when the future finishes.
            # If the future has not yet started, attempt to cancel.
            try:
                cancelled = future.cancel()
            except Exception:
                cancelled = False
            if cancelled:
                _release_session_token(session_id, _token)
            else:
                logger.warning(
                    "submit_sync: watchdog submission failed for "
                    "session=%s; ownership will release on completion: %s",
                    session_id, exc,
                )
        return future
    else:
        future = _get_sync_pool().submit(fn, *args, **kwargs)
        future.add_done_callback(_check_future_error)
        return future


def submit_summary(fn, *args, **kwargs):
    """Submit a summarization operation to the summary pool."""
    future = _get_summary_pool().submit(fn, *args, **kwargs)
    future.add_done_callback(_check_future_error)
    return future


# ---------------------------------------------------------------------------
# Public API
#  submit_sync, submit_summary
# ---------------------------------------------------------------------------
