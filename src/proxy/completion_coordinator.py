"""Bounded completion admission + execution for streaming requests.

F1 fix (four-day-change-audit-2026-07-17.md, finding F1): replaces the
raw ``threading.Thread``-per-request in ``app.py`` with a semaphore-gated,
bounded ``ThreadPoolExecutor`` so a disconnect storm cannot accumulate
unbounded threads that eventually saturate the SQLite pool (48 slots,
see db_pool.py).

Admission is SYNCHRONOUS in the Flask route: the semaphore is acquired
BEFORE the streaming response is handed to Flask. On capacity exhaustion
``CompletionCapacityError`` is raised and the route returns ``503 +
Retry-After`` — while the HTTP status can still be set, because no
streaming response has been committed yet.

The completion worker calls ``pipeline.run(ctx)`` exactly once —
preserving the W1-A/F10 contract (commit 39ac2d7) that
``ContextOptimizeStage`` sets ``ctx.mode_snapshot_pre_upstream`` BEFORE
upstream and ``PostProcessStage`` consumes it.

``admit(pipeline, ctx, on_owner_start=...)`` accepts an opaque owner hook
(CP2B) that the worker runs once, immediately before ``pipeline.run(ctx)``,
so owner-only mutations never fire for rejected (503) or coalesced follower
requests. The coordinator stays generic and never imports the hook's module.

State lives at module level; the module-local ``_lock`` is a documented
leaf lock that never acquires any of the 14 numbered locks.
"""

import atexit
import enum
import logging
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Optional

from proxy.pipeline.exceptions import StageAbort
from proxy.types import BackendError

logger = logging.getLogger(__name__)


class CompletionCapacityError(Exception):
    """Raised when the completion coordinator is at capacity."""


class AdmissionKind(enum.Enum):
    """Result kind for admission decisions (CP4-2)."""
    OWNER = "owner"
    FOLLOWER = "follower"
    REPLAY = "replay"        # CP4-3
    CONFLICT = "conflict"    # CP4-4+


@dataclass(frozen=True)
class AdmissionResult:
    """Typed result of admit() / dedup admission.

    CP4-2: kind is now a typed AdmissionKind enum. future defaults to None
    (followers don't have their own future — they share the owner's record).
    """

    kind: AdmissionKind
    future: Any = None
    record: Optional[Any] = None
    fingerprint: Optional[str] = None
    original_request_id: Optional[str] = None


class CompletionState(enum.Enum):
    """Lifecycle state of a CompletionRecord."""

    PENDING = "pending"
    COMPLETED = "completed"


@dataclass
class CompletionRecord:
    """Per-request completion record — the subscriber handoff object (CP3-2).

    NOT stored in any shared map (keyless CP3). Single-writer (the completion
    worker) / single-reader (the Flask delivery generator). ``completion_event``
    is the happens-before barrier: the worker assigns ``outcome`` / ``state`` /
    ``completed_at`` and then calls ``completion_event.set()`` LAST, so a
    delivery thread that observes ``wait() == True`` sees a fully-published
    record WITHOUT a lock. Two invariants keep this lock-free:

      1. Every field the subscriber reads is assigned BEFORE ``set()``.
      2. No third reader touches the record without gating on
         ``completion_event`` (the record is never placed in a shared map, so
         this holds by construction in CP3).

    CP4-2: ``fingerprint`` is set by the route for dedup wiring (the dedup
    store key that this record is associated with).
    """

    owner_request_id: str
    future: Any = None
    state: CompletionState = CompletionState.PENDING
    completion_event: threading.Event = field(default_factory=threading.Event)
    outcome: Optional[dict] = None
    created_at: float = field(default_factory=time.time)
    completed_at: Optional[float] = None
    artifact: Optional[Any] = None
    fingerprint: Optional[str] = None


@dataclass(frozen=True)
class CompletionArtifact:
    """Immutable terminal artifact — SSE chunks for streaming
    (``terminal_chunks``), JSON body for non-streaming (``body``).

    Materialized ONCE by the completion worker (CP3-3). ``terminal_chunks``
    are the fully-rendered SSE strings including exactly one
    ``data: [DONE]\\n\\n`` terminator. Delivery emits these verbatim — it
    never renders. Materializing in the worker (with frozen id/created) is
    the byte-identity foundation CP4 replay depends on.

    CP4-3: ``has_tool_calls`` is set by ``_materialize_artifact`` (app.py) when
    the success response contains tool_calls or function_call in any choice
    message, or a tool_calls/function_call finish_reason. It is checked by the
    replay safety gate (``is_safe_for_replay`` in dedup.py) — artifacts with
    tool calls are never replayed (they require live pipeline processing).

    CP4-4: ``body`` carries the JSON response dict for non-streaming requests.
    When set, ``terminal_chunks`` is an empty tuple. The replay path for
    non-stream requests reads ``body`` directly.
    """

    terminal_chunks: tuple
    logical_status: str
    original_request_id: str
    content_type: str = "text/event-stream"
    wire_status: int = 200
    has_tool_calls: bool = False
    body: Optional[dict] = None


# ---------------------------------------------------------------------------
# Configuration (read once at import)
# ---------------------------------------------------------------------------

_MAX_COMPLETION_RAW = os.environ.get("PROXY_MAX_CONCURRENT_COMPLETIONS", "16")
try:
    MAX_CONCURRENT_COMPLETIONS = int(_MAX_COMPLETION_RAW)
    if MAX_CONCURRENT_COMPLETIONS < 1:
        MAX_CONCURRENT_COMPLETIONS = 16
except ValueError:
    MAX_CONCURRENT_COMPLETIONS = 16

_POOL_WORKERS_RAW = os.environ.get("PROXY_COMPLETION_POOL_WORKERS", "0")
try:
    _configured_workers = int(_POOL_WORKERS_RAW)
except ValueError:
    _configured_workers = 0
if _configured_workers < 1:
    import multiprocessing as _mp

    _configured_workers = min(32, (_mp.cpu_count() or 1) * 4)

# ---------------------------------------------------------------------------
# Singleton executor + semaphore + leaf lock
# ---------------------------------------------------------------------------

_executor = ThreadPoolExecutor(
    max_workers=_configured_workers,
    thread_name_prefix="completion-worker-",
)

_semaphore = threading.Semaphore(MAX_CONCURRENT_COMPLETIONS)

_lock = threading.Lock()  # leaf lock — never acquires any numbered lock
_pending = 0
_shutdown = False


def _run_completion(pipeline, ctx, record, on_owner_start=None, materialize=None, on_completed=None):
    """Worker body: runs the full 9-stage pipeline inside a pool thread.

    Catches and classifies every outcome into a dict so the caller's
    ``Future.result()`` never re-raises. Event-log ContextVars do not
    propagate to new threads; the worker installs + clears its own.

    ``on_owner_start`` is an optional owner-only hook (CP2B). When provided,
    it runs inside the try block AFTER the event-log context is installed and
    BEFORE ``pipeline.run(ctx)``, so owner-only mutations execute exactly once
    on the owner execution path. A hook exception is captured into
    ``outcome["error"]`` (never crashes the worker) and skips ``pipeline.run``.

    CP3-2: the worker publishes completion onto ``record`` in the ``finally``
    (``state=COMPLETED``, ``outcome``, ``completed_at``, then
    ``completion_event.set()`` LAST). ``set()`` is the last statement so a
    woken subscriber observes a fully-populated record without a lock. The
    published ``record.outcome`` is the SAME dict object that is returned.

    CP3-3: when ``materialize`` is provided, the worker renders the terminal
    SSE artifact ONCE (in this pool thread) and publishes it onto
    ``record.artifact`` in the ``finally``, immediately after ``record.outcome``
    and BEFORE ``completion_event.set()``. The materialize call is wrapped in a
    try/except so a render failure leaves ``artifact=None`` but NEVER skips
    ``event.set()`` (the wait loop would otherwise hang forever). ``materialize``
    is an opaque owner-only hook — the coordinator neither imports nor
    interprets it.

    CP4-2: ``on_completed`` is an optional hook invoked in the ``finally``
    AFTER ``record.completion_event.set()``. Used by the dedup wire-in to
    mark the dedup entry COMPLETED. A hook exception is logged but never
    propagates (the completion is already published).
    """
    from proxy.event_log import clear_request_context, set_request_context

    request_id = getattr(ctx, "request_id", "?")
    session_id = getattr(ctx, "session_id", "?")
    log_extra = {"request_id": request_id, "session_id": session_id}

    outcome: dict = {
        "result": None,
        "stage_abort": None,
        "backend_error": None,
        "error": None,
    }
    set_request_context(request_id, session_id)
    logger.info("completion_owner_started", extra=log_extra)
    try:
        if on_owner_start is not None:
            on_owner_start()
        outcome["result"] = pipeline.run(ctx)
        logger.info("completion_finalized", extra=log_extra)
    except StageAbort as exc:
        outcome["stage_abort"] = exc.response_data
        logger.info("completion_finalized (stage_abort)", extra=log_extra)
    except BackendError as exc:
        outcome["backend_error"] = exc
        logger.warning("completion_finalized (backend_error)", extra=log_extra)
    except Exception as exc:  # noqa: BLE001 — last-resort worker safety
        outcome["error"] = exc
        logger.exception("completion_finalized (error)", extra=log_extra)
    finally:
        clear_request_context()
        record.outcome = outcome
        if materialize is not None:
            try:
                record.artifact = materialize(outcome, ctx)
            except Exception:  # noqa: BLE001 — render failure must not hang delivery
                record.artifact = None
                logger.exception(
                    "completion artifact materialization failed", extra=log_extra
                )
        record.state = CompletionState.COMPLETED
        record.completed_at = time.time()
        record.completion_event.set()
        if on_completed is not None:
            try:
                on_completed()
            except Exception:  # noqa: BLE001
                logger.exception("on_completed hook failed", extra=log_extra)
    return outcome


def admit(pipeline, ctx, *, on_owner_start=None, materialize=None, on_completed=None, record=None):
    """Admit a completion request and return its ``AdmissionResult``.

    ``on_owner_start`` (CP2B) is an opaque Optional[Callable] owner hook run
    inside the worker BEFORE ``pipeline.run(ctx)`` (see ``_run_completion``).
    The coordinator stays generic — it neither imports nor interprets the hook.

    ``materialize`` (CP3-3) is an opaque owner-only render hook (like
    ``on_owner_start``) that the worker calls in its ``finally`` to build the
    terminal SSE artifact published onto ``record.artifact``. The coordinator
    never imports or interprets it; a render failure inside it leaves
    ``artifact=None`` without hanging delivery.

    ``on_completed`` (CP4-2) is an opaque hook invoked in the worker's
    ``finally`` AFTER ``record.completion_event.set()``. Used by the dedup
    wire-in to mark the dedup entry COMPLETED.

    ``record`` (CP4-2) is an optional pre-created CompletionRecord. When
    provided (by the dedup owner path), the coordinator uses it instead of
    creating a new one. This allows the dedup store to hold the record
    reference before admit() is called.

    R2-A (#3 admission mutates shared state pre-503): the decision
    (would_accept) is computed under one lock BEFORE any state mutation.
    A 503 must not mutate any counter — the prior implementation
    acquired the semaphore as the capacity check, so the semaphore
    counter was decremented before ``_shutdown`` (or any future 503
    gate) was consulted. This rewrite separates the two phases:

      Phase 1 — DECIDE (under ``_lock``, no mutation):
        * Read ``_shutdown`` and ``_pending``.
        * Compute ``would_accept = (not _shutdown) and (_pending < MAX)``.
        * If ``not would_accept``, raise ``CompletionCapacityError`` with
          a specific message — the lock is released cleanly, no state
          has been mutated, and the next request sees the true count.

      Phase 2 — MUTATE (only if decision is accept):
        * Acquire the semaphore (still non-blocking; the count may
          have changed between Phase 1 and now, so this is a
          defensive double-check; if the semaphore now reports full,
          release and re-raise — the executor's max-workers is
          ``>=`` ``MAX_CONCURRENT_COMPLETIONS`` so under normal
          configuration this never fires).
        * Increment ``_pending`` under ``_lock``.
        * Submit to the executor; if submit raises ``RuntimeError``
          (executor shut down between check and submit), roll back
          both the semaphore and the counter.

    The semaphore remains the canonical atomic check-and-mutate
    primitive for the capacity counter; the rewrite just sequences it
    AFTER the shutdown check so the shutdown decision does not leave
    a briefly-decremented semaphore behind.

    Returns:
        AdmissionResult(kind=AdmissionKind.OWNER, future=<Future>, record=<CompletionRecord>)
        — the Future resolves to the outcome dict, and ``record`` is the
        per-request ``CompletionRecord`` the worker publishes completion onto
        (via ``record.completion_event``). The 503 capacity/shutdown paths
        raise CompletionCapacityError before any AdmissionResult is constructed.
    """
    global _pending

    request_id = getattr(ctx, "request_id", "?")

    # ── Phase 1: DECIDE (no mutation) ──────────────────────────────
    with _lock:
        if _shutdown:
            raise CompletionCapacityError("completion coordinator shutting down")
        if _pending >= MAX_CONCURRENT_COMPLETIONS:
            raise CompletionCapacityError(
                f"completion capacity {MAX_CONCURRENT_COMPLETIONS} exceeded"
            )

    # ── Phase 2: MUTATE (only because the decision is accept) ──────
    acquired = _semaphore.acquire(blocking=False)
    if not acquired:
        # Defensive: a concurrent admit already took the last slot
        # between Phase 1 and now. No mutation has happened yet —
        # the counter reads as the truth we sampled in Phase 1, plus
        # one in-flight we cannot see (the winner of the race). Raise
        # without mutating; the next admit() will see the updated count.
        raise CompletionCapacityError(
            f"completion capacity {MAX_CONCURRENT_COMPLETIONS} exceeded"
        )

    with _lock:
        # Defensive re-check under the same lock used for the counter.
        # If shutdown was requested between Phase 1 and the semaphore
        # acquire, release the semaphore (which we own now) and
        # raise. No leak.
        if _shutdown:
            _semaphore.release()
            raise CompletionCapacityError("completion coordinator shutting down")
        _pending += 1

    logger.info(
        "completion admitted: pending=%d/%d",
        _pending,
        MAX_CONCURRENT_COMPLETIONS,
        extra={"request_id": request_id},
    )

    if record is None:
        record = CompletionRecord(owner_request_id=str(request_id))

    try:
        future = _executor.submit(_run_completion, pipeline, ctx, record, on_owner_start, materialize, on_completed)
    except RuntimeError:
        # executor rejected the submission (e.g. shut down between
        # mutation and submit). Roll back both the semaphore and the
        # counter so no slot is leaked.
        _semaphore.release()
        with _lock:
            _pending -= 1
        raise CompletionCapacityError("completion executor unavailable")

    def _on_done(_fut):
        global _pending
        _semaphore.release()
        with _lock:
            _pending -= 1

    record.future = future
    future.add_done_callback(_on_done)
    return AdmissionResult(kind=AdmissionKind.OWNER, future=future, record=record)


def pending_count():
    """Return the number of currently admitted (in-flight) completions."""
    with _lock:
        return _pending


def max_capacity():
    """Return the configured max concurrent completions."""
    return MAX_CONCURRENT_COMPLETIONS


def shutdown(grace_period: float = 30.0) -> bool:
    """Stop accepting new completions and drain in-flight ones.

    Outcomes:
      - Drained before deadline: executor shutdown uses ``wait=True``,
        returns ``True``.
      - Deadline expired: executor shutdown uses ``wait=False,
        cancel_futures=True``, returns ``False``.
    """
    global _shutdown
    with _lock:
        _shutdown = True
    deadline = time.monotonic() + max(0.0, grace_period)
    while True:
        with _lock:
            pending = _pending

        if pending == 0:
            _executor.shutdown(wait=True)
            logger.info("completion coordinator shut down")
            return True

        remaining = deadline - time.monotonic()
        if remaining <= 0:
            logger.warning(
                "completion coordinator shutdown timed out: "
                "_pending=%d after grace_period=%.1fs",
                pending,
                grace_period,
            )
            _executor.shutdown(wait=False, cancel_futures=True)
            return False

        time.sleep(min(0.05, remaining))


atexit.register(shutdown)
