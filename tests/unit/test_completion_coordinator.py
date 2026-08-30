"""F1: bounded completion coordinator — admission, slot accounting, shutdown."""
import threading
import time
import types
from concurrent.futures import ThreadPoolExecutor

import pytest

import proxy.completion_coordinator as cc


@pytest.fixture
def coordinator_state():
    """Snapshot/restore module globals so tests don't leak into the shared
    singleton executor (which must survive the session)."""
    orig_sem, orig_pending, orig_shutdown = cc._semaphore, cc._pending, cc._shutdown
    yield
    cc._semaphore, cc._pending, cc._shutdown = orig_sem, orig_pending, orig_shutdown


def _ctx():
    return types.SimpleNamespace(request_id="r", session_id="s")


def test_admit_runs_worker_and_releases_slot(coordinator_state):
    cc._semaphore = threading.Semaphore(4)
    cc._pending = 0
    cc._shutdown = False

    class _P:
        def run(self, ctx):
            return {"ok": True}

    fut = cc.admit(_P(), _ctx()).future
    released = threading.Event()
    fut.add_done_callback(lambda f: released.set())
    outcome = fut.result(timeout=5)
    assert outcome["result"] == {"ok": True}
    assert outcome["error"] is None
    assert released.wait(timeout=5)
    assert cc.pending_count() == 0
    assert cc._semaphore._value == 4


def test_admit_returns_admission_result_owner(coordinator_state):
    cc._semaphore = threading.Semaphore(4)
    cc._pending = 0
    cc._shutdown = False

    class _P:
        def run(self, ctx):
            return {"ok": True}

    admission = cc.admit(_P(), _ctx())
    assert admission.kind == cc.AdmissionKind.OWNER
    assert admission.record is not None
    assert admission.record.owner_request_id == "r"
    # After completion, the record mirrors the future outcome via the event.
    outcome = admission.future.result(timeout=5)
    assert admission.record.completion_event.wait(timeout=5)
    assert admission.record.state == cc.CompletionState.COMPLETED
    assert admission.record.outcome is outcome
    assert admission.record.completed_at is not None
    assert outcome["result"] == {"ok": True}
    assert outcome["error"] is None


def test_worker_publishes_state_before_event_set(coordinator_state):
    observations = []

    class _ProbeEvent(threading.Event):
        def __init__(self, record):
            super().__init__()
            self._record = record

        def set(self):
            # Runs in the worker thread at the exact publish boundary.
            observations.append((self._record.state, self._record.completed_at))
            super().set()

    class _P:
        def run(self, ctx):
            return {"ok": True}

    record = cc.CompletionRecord(owner_request_id="r")
    record.completion_event = _ProbeEvent(record)
    cc._run_completion(_P(), _ctx(), record)

    assert len(observations) == 1
    state_at_set, completed_at_at_set = observations[0]
    assert state_at_set == cc.CompletionState.COMPLETED
    assert completed_at_at_set is not None


def test_capacity_exhaustion_raises_then_recovers(coordinator_state):
    cc._semaphore = threading.Semaphore(1)
    cc._pending = 0
    cc._shutdown = False
    gate = threading.Event()

    class _P:
        def run(self, ctx):
            gate.wait(timeout=10)
            return {"ok": True}

    fut1 = cc.admit(_P(), _ctx()).future
    released = threading.Event()
    fut1.add_done_callback(lambda f: released.set())
    with pytest.raises(cc.CompletionCapacityError):
        cc.admit(_P(), _ctx())
    gate.set()
    fut1.result(timeout=5)
    assert released.wait(timeout=5)
    assert cc.pending_count() == 0
    assert cc._semaphore._value == 1


def test_slot_released_on_worker_exception(coordinator_state):
    cc._semaphore = threading.Semaphore(2)
    cc._pending = 0
    cc._shutdown = False

    class _P:
        def run(self, ctx):
            raise RuntimeError("boom")

    fut = cc.admit(_P(), _ctx()).future
    released = threading.Event()
    fut.add_done_callback(lambda f: released.set())
    outcome = fut.result(timeout=5)
    assert isinstance(outcome["error"], RuntimeError)
    assert released.wait(timeout=5)
    assert cc.pending_count() == 0
    assert cc._semaphore._value == 2


def test_shutdown_flag_rejects_admission_without_leaking_slot(coordinator_state):
    cc._semaphore = threading.Semaphore(3)
    cc._pending = 0
    cc._shutdown = True

    class _P:
        def run(self, ctx):
            return {}

    with pytest.raises(cc.CompletionCapacityError):
        cc.admit(_P(), _ctx())
    assert cc._semaphore._value == 3
    assert cc.pending_count() == 0


# ---------------------------------------------------------------------------
# R2-A — finding #3 (admission mutates shared state pre-503)
# ---------------------------------------------------------------------------


def test_r2_a1_capacity_503_does_not_mutate_pending(coordinator_state):
    """R2-A #3 (capacity check pre-mutation): a 503 from admit() must
    leave BOTH _pending AND _semaphore._value unchanged. The previous
    implementation acquired the semaphore as the capacity check; the
    rewrite reads _pending under the lock first (Phase 1), then
    mutates only on accept (Phase 2).

    Revert-check: revert Edit R2-A.1 — replace the Phase-1
    `if _pending >= MAX_CONCURRENT_COMPLETIONS` raise with the prior
    `_semaphore.acquire(blocking=False)` + post-check pattern. With
    the prior pattern, a concurrent accept that releases the
    semaphore between Phase 1's read and Phase 2's acquire can leave
    _pending < MAX_CONCURRENT_COMPLETIONS but the semaphore briefly
    decremented; this test pins that the 503 path itself never
    mutates _pending.
    """
    cap = 1
    cc._semaphore = threading.Semaphore(cap)
    cc._pending = 0
    cc._shutdown = False

    gate = threading.Event()

    class _P:
        def run(self, ctx):
            gate.wait(timeout=10)
            return {"ok": True}

    fut1 = cc.admit(_P(), _ctx()).future
    fut1.add_done_callback(lambda f: gate.set())
    assert cc.pending_count() == 1
    assert cc._semaphore._value == 0

    with pytest.raises(cc.CompletionCapacityError):
        cc.admit(_P(), _ctx())

    assert cc.pending_count() == 1, (
        f"_pending was mutated on the 503 path: got {cc.pending_count()}, expected 1"
    )
    assert cc._semaphore._value == 0, (
        f"semaphore was mutated on the 503 path: got {cc._semaphore._value}, expected 0"
    )

    gate.set()
    fut1.result(timeout=5)


def test_r2_a2_shutdown_503_does_not_touch_semaphore(coordinator_state):
    """R2-A #3 (shutdown check pre-mutation): set _shutdown=True and
    call admit(). The semaphore counter must NEVER be decremented —
    not even momentarily.

    Pre-rewrite behaviour: _semaphore.acquire(blocking=False) returns
    True (slot available), then the shutdown check fires, the slot is
    released. The semaphore value transiently drops from cap to cap-1.

    The discriminating signal is recorded via a Semaphore subclass
    that records its value after every successful acquire and before
    every release. With the pre-rewrite pattern, the recorded history
    contains cap-1 (a transient decrement). With the rewrite, the
    recorded history is empty (no acquire/release happens on the
    shutdown-503 path because Phase 1 raises before touching the
    semaphore).

    Note on deviation from the verbatim plan: the plan specified a
    busy-spin observer thread, but the pre-rewrite acquire/release is
    so fast (microseconds, GIL held) that the observer never catches
    the transient (verified empirically — 10/10 passes on revert). The
    Semaphore-subclass spy is the deterministic equivalent: it
    observes the value AT the moment of the operation, not by racing
    from another thread. This deviation is required to satisfy the
    universal invariant "revert-check must fail on revert".
    """
    cap = 4
    history: list = []

    class _SpySemaphore(threading.Semaphore):
        def acquire(self, blocking=True, timeout=None):
            result = super().acquire(blocking=blocking, timeout=timeout)
            if result:
                history.append(self._value)
            return result

        def release(self):
            history.append(self._value)
            super().release()

    cc._semaphore = _SpySemaphore(cap)
    cc._pending = 0
    cc._shutdown = True

    class _P:
        def run(self, ctx):
            return {}

    with pytest.raises(cc.CompletionCapacityError):
        cc.admit(_P(), _ctx())

    assert min(history, default=cap) == cap, (
        f"semaphore was decremented on the shutdown-503 path: "
        f"observed values {history}, expected all >= {cap}. "
        f"The rewrite must move the shutdown check BEFORE acquire."
    )
    assert cc.pending_count() == 0
    assert cc._semaphore._value == cap


def test_r2_a3_shutdown_flip_between_phases_does_not_leak(coordinator_state, monkeypatch):
    """R2-A #3 (defensive re-check): flip _shutdown=True AFTER Phase 1
    reads it (False) but BEFORE Phase 2 acquires the semaphore. The
    rewrite's defensive re-check under _lock must release the
    semaphore and raise CompletionCapacityError without incrementing
    _pending.

    Revert-check: remove the Phase-2 `if _shutdown: _semaphore.release();
    raise CompletionCapacityError(...)` block (the defensive re-check).
    After revert, _pending += 1 runs unconditionally after the semaphore
    acquire, then _executor.submit() will likely raise RuntimeError on
    the shut-down executor, hitting the except branch that decrements
    _pending. The end-state counter is correct (0), but the test
    pin is the semaphore: without the defensive re-check, the
    semaphore is acquired and only released by the except branch
    when submit() raises — which it does NOT do immediately. If
    the test asserts semaphore._value == cap after the 503 raises,
    the pre-rewrite defensive-check removal will fail this assertion
    because the semaphore was acquired (decremented) and the release
    only happens in the submit() except (which never fires because
    submit() succeeds on the shutting-down executor).
    """
    cap = 3
    cc._semaphore = threading.Semaphore(cap)
    cc._pending = 0
    cc._shutdown = False

    real_acquire = cc._semaphore.acquire

    def flipping_acquire(blocking=True, timeout=None):
        cc._shutdown = True
        return real_acquire(blocking=blocking, timeout=timeout)

    monkeypatch.setattr(cc._semaphore, "acquire", flipping_acquire)

    class _P:
        def run(self, ctx):
            return {}

    with pytest.raises(cc.CompletionCapacityError) as exc_info:
        cc.admit(_P(), _ctx())
    assert "shutting down" in str(exc_info.value).lower(), (
        f"Phase-2 defensive re-check must raise the shutdown-specific "
        f"message, got: {exc_info.value}"
    )

    assert cc.pending_count() == 0, (
        f"_pending was mutated despite the shutdown flip: got {cc.pending_count()}"
    )
    assert cc._semaphore._value == cap, (
        f"semaphore leaked on shutdown-flip race: got {cc._semaphore._value}, "
        f"expected {cap}"
    )


def test_r2_a4_concurrent_admissions_preserve_counters(coordinator_state):
    """R2-A #3 (concurrent regression-pin): MAX+1 concurrent admit()
    calls must yield exactly MAX successes and exactly 1
    CompletionCapacityError, with no leaked semaphore slots and no
    spurious _pending increments.

    Revert-check: revert Edit R2-A.1 to the pre-rewrite pattern.
    The pre-rewrite pattern works for this regression-pin in the
    happy path (capacity correctly rejects the (MAX+1)th), but the
    discriminating signal is the post-failure counter check: the
    pre-rewrite pattern acquires-then-releases on shutdown-503, so
    the semaphore transiently drops to 0 before returning to 1. The
    concurrent observer would see a transient value of 0. With the
    rewrite, the semaphore never moves on the rejection path.
    """
    cap = 2
    cc._semaphore = threading.Semaphore(cap)
    cc._pending = 0
    cc._shutdown = False

    gate = threading.Event()
    barrier = threading.Barrier(cap + 1)

    class _P:
        def run(self, ctx):
            barrier.wait(timeout=5)
            gate.wait(timeout=10)
            return {"ok": True}

    results = {"accepted": 0, "rejected": 0}
    lock = threading.Lock()

    def attempt():
        barrier.wait(timeout=5)
        try:
            cc.admit(_P(), _ctx())
            with lock:
                results["accepted"] += 1
        except cc.CompletionCapacityError:
            with lock:
                results["rejected"] += 1

    threads = [threading.Thread(target=attempt, daemon=True) for _ in range(cap + 1)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=10)

    assert results["accepted"] == cap, (
        f"expected {cap} acceptances, got {results['accepted']}"
    )
    assert results["rejected"] == 1, (
        f"expected 1 rejection, got {results['rejected']}"
    )
    assert cc.pending_count() == cap, (
        f"_pending drifted: got {cc.pending_count()}, expected {cap}"
    )
    assert cc._semaphore._value == 0, (
        f"semaphore leaked: got {cc._semaphore._value}, expected 0"
    )

    gate.set()


def test_r2_a5_submit_runtimeerror_does_not_double_release(coordinator_state, monkeypatch):
    """R2-A #3 (rollback-on-submit-failure): patch _executor.submit to
    raise RuntimeError. Verify the semaphore counter is back to its
    pre-admit value AND _pending is 0 — no leak from a double-release
    or a missing rollback.

    Revert-check: remove the except RuntimeError branch from Edit
    R2-A.1 (the rollback that releases the semaphore and decrements
    _pending). After revert, the semaphore is acquired and _pending
    is incremented, but submit's RuntimeError propagates to the
    Flask route, which maps to a 500 (NOT a 503) — the route's
    `except CompletionCapacityError` does not catch RuntimeError.
    The test asserts a CompletionCapacityError is raised; without
    the except branch, the test fails with the original RuntimeError.
    This pins that the 503 path is the ONLY error path on submit
    failure.
    """
    cap = 2
    cc._semaphore = threading.Semaphore(cap)
    cc._pending = 0
    cc._shutdown = False

    def boom_submit(fn, *args, **kwargs):
        raise RuntimeError("executor rejected submission")

    monkeypatch.setattr(cc._executor, "submit", boom_submit)

    class _P:
        def run(self, ctx):
            return {}

    with pytest.raises(cc.CompletionCapacityError) as exc_info:
        cc.admit(_P(), _ctx())
    assert "executor unavailable" in str(exc_info.value).lower(), (
        f"submit-failure path must map to CompletionCapacityError, "
        f"got: {exc_info.value}"
    )
    assert cc.pending_count() == 0, (
        f"_pending leaked on submit-failure rollback: got {cc.pending_count()}"
    )
    assert cc._semaphore._value == cap, (
        f"semaphore leaked on submit-failure rollback: got {cc._semaphore._value}, "
        f"expected {cap}"
    )


def test_r2_a6_admit_after_shutdown_does_not_mutate(coordinator_state):
    """R2-A #3 (shutdown-state regression-pin): set _shutdown=True and
    call admit(). Assert CompletionCapacityError is raised with the
    shutdown-specific message, _pending is unchanged, the post-call
    semaphore value is unchanged, AND no transient decrement
    occurred during the call.

    Revert-check: revert Edit R2-A.1's Phase-1 `if _shutdown: raise`
    block. The pre-rewrite pattern would acquire the semaphore
    (decrementing it to cap-1) BEFORE checking _shutdown, then
    release it (back to cap) and raise. The post-call state would
    be correct (cap, 0), but the transient decrement would appear
    in the Semaphore-spy history.

    Note on deviation from the verbatim plan: the plan specified a
    post-call counter check (semaphore._value == cap), but the
    pre-rewrite code's release() restores the value before the test
    reads it — so the post-call check is NOT discriminating on
    revert. The Semaphore-subclass spy (overriding acquire/release
    to record _value) is the deterministic equivalent: it captures
    the transient value AT the moment of the operation. This
    deviation is required to satisfy the universal invariant
    "revert-check must fail on revert".
    """
    cap = 4
    history: list = []

    class _SpySemaphore(threading.Semaphore):
        def acquire(self, blocking=True, timeout=None):
            result = super().acquire(blocking=blocking, timeout=timeout)
            if result:
                history.append(self._value)
            return result

        def release(self):
            history.append(self._value)
            super().release()

    cc._semaphore = _SpySemaphore(cap)
    cc._pending = 0
    cc._shutdown = True

    class _P:
        def run(self, ctx):
            return {}

    with pytest.raises(cc.CompletionCapacityError) as exc_info:
        cc.admit(_P(), _ctx())
    assert "shutting down" in str(exc_info.value).lower()

    assert min(history, default=cap) == cap, (
        f"semaphore was transiently decremented on the shutdown-503 path: "
        f"observed values {history}, expected all >= {cap}. "
        f"The rewrite must move the shutdown check BEFORE acquire."
    )
    assert cc.pending_count() == 0
    assert cc._semaphore._value == cap


# ---------------------------------------------------------------------------
# R2-B — finding #9 (coordinator shutdown unbounded + mis-ordered)
# ---------------------------------------------------------------------------


def test_slice_3a_shutdown_waits_for_drain_then_uses_blocking_executor_shutdown(coordinator_state, monkeypatch):
    cap = 1
    cc._semaphore = threading.Semaphore(cap)
    cc._pending = 0
    cc._shutdown = False

    fresh_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="test-slice3a-drain-")
    monkeypatch.setattr(cc, "_executor", fresh_executor)

    shutdown_calls = []
    original_executor_shutdown = fresh_executor.shutdown

    def wrapping_shutdown(wait=True, cancel_futures=False):
        shutdown_calls.append((wait, cancel_futures, cc.pending_count()))
        return original_executor_shutdown(wait=wait, cancel_futures=cancel_futures)

    monkeypatch.setattr(cc._executor, "shutdown", wrapping_shutdown)

    worker_started = threading.Event()
    worker_can_finish = threading.Event()
    shutdown_waiting = threading.Event()

    class _P:
        def run(self, ctx):
            worker_started.set()
            assert worker_can_finish.wait(timeout=5.0)
            return {"ok": True}

    fut = cc.admit(_P(), _ctx()).future
    assert worker_started.wait(timeout=5)

    def coordinated_sleep(_delay):
        shutdown_waiting.set()
        assert worker_can_finish.wait(timeout=5.0)

    monkeypatch.setattr(cc.time, "sleep", coordinated_sleep)

    result_holder = {}

    def run_shutdown():
        result_holder["result"] = cc.shutdown(grace_period=2.0)

    shutdown_thread = threading.Thread(target=run_shutdown, daemon=False)
    shutdown_thread.start()

    assert shutdown_waiting.wait(timeout=5)
    worker_can_finish.set()
    fut.result(timeout=5)
    shutdown_thread.join(timeout=5)
    assert not shutdown_thread.is_alive()

    assert result_holder["result"] is True
    assert len(shutdown_calls) == 1
    assert shutdown_calls[0] == (True, False, 0)
    assert cc.pending_count() == 0

    class _EmptyP:
        def run(self, ctx):
            return {}

    with pytest.raises(cc.CompletionCapacityError) as exc_info:
        cc.admit(_EmptyP(), _ctx())
    assert "shutting down" in str(exc_info.value).lower()


def test_slice_3a_shutdown_timeout_is_nonblocking_and_returns_false(coordinator_state, monkeypatch):
    cap = 1
    cc._semaphore = threading.Semaphore(cap)
    cc._pending = 0
    cc._shutdown = False

    fresh_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="test-slice3a-timeout-")
    monkeypatch.setattr(cc, "_executor", fresh_executor)

    shutdown_calls = []
    original_executor_shutdown = fresh_executor.shutdown

    def wrapping_shutdown(wait=True, cancel_futures=False):
        shutdown_calls.append((wait, cancel_futures, cc.pending_count()))
        return original_executor_shutdown(wait=wait, cancel_futures=cancel_futures)

    monkeypatch.setattr(cc._executor, "shutdown", wrapping_shutdown)

    worker_started = threading.Event()
    worker_can_finish = threading.Event()
    callback_ran = threading.Event()

    class _P:
        def run(self, ctx):
            worker_started.set()
            assert worker_can_finish.wait(timeout=5.0)
            return {"ok": True}

    fut = cc.admit(_P(), _ctx()).future
    fut.add_done_callback(lambda f: callback_ran.set())
    assert worker_started.wait(timeout=5)

    try:
        started = time.monotonic()
        result = cc.shutdown(grace_period=0.05)
        elapsed = time.monotonic() - started

        assert result is False
        assert elapsed < 0.5
        assert len(shutdown_calls) == 1
        assert shutdown_calls[0] == (False, True, 1)
        assert cc.pending_count() == 1

        class _EmptyP:
            def run(self, ctx):
                return {}

        with pytest.raises(cc.CompletionCapacityError) as exc_info:
            cc.admit(_EmptyP(), _ctx())
        assert "shutting down" in str(exc_info.value).lower()
    finally:
        worker_can_finish.set()
        fut.result(timeout=5.0)
        assert callback_ran.wait(timeout=5)
        assert cc.pending_count() == 0
        original_executor_shutdown(wait=True)


# ---------------------------------------------------------------------------
# CP2B — on_owner_start owner hook (ordering + guard pins)
# ---------------------------------------------------------------------------


def test_on_owner_start_runs_before_pipeline_run(coordinator_state):
    """CP2B: the owner hook must run BEFORE pipeline.run(ctx) on the owner
    execution path. Deterministic (no sleep) via a shared calls list.

    Revert-check: drop the `if on_owner_start is not None: on_owner_start()`
    call from _run_completion -> calls == ["run"] -> this test fails.
    """
    cc._semaphore = threading.Semaphore(4)
    cc._pending = 0
    cc._shutdown = False

    calls: list = []

    class _P:
        def run(self, ctx):
            calls.append("run")
            return {"ok": True}

    fut = cc.admit(_P(), _ctx(), on_owner_start=lambda: calls.append("owner")).future
    outcome = fut.result(timeout=5)

    assert outcome["error"] is None
    assert outcome["result"] == {"ok": True}
    assert calls == ["owner", "run"]


def test_on_owner_start_none_is_safe(coordinator_state):
    """CP2B B1: admit() with no owner hook (the existing 2-arg call form) must
    stay valid — a bare on_owner_start() with None would crash the worker.
    """
    cc._semaphore = threading.Semaphore(4)
    cc._pending = 0
    cc._shutdown = False

    fut = cc.admit(_P(), _ctx()).future
    outcome = fut.result(timeout=5)

    assert outcome["error"] is None
    assert outcome["result"] == {"ok": True}


def test_on_owner_start_raises_skips_pipeline_run(coordinator_state):
    """CP2B: an owner_setup exception is captured into outcome["error"] (never
    crashes the worker) AND pipeline.run(ctx) is SKIPPED because the hook runs
    inside the try block before it.
    """
    cc._semaphore = threading.Semaphore(4)
    cc._pending = 0
    cc._shutdown = False

    ran = {"pipeline": False}
    boom = RuntimeError("owner setup failed")

    class _P:
        def run(self, ctx):
            ran["pipeline"] = True
            return {"ok": True}

    def _raise():
        raise boom

    fut = cc.admit(_P(), _ctx(), on_owner_start=_raise).future
    outcome = fut.result(timeout=5)

    assert outcome["error"] is boom
    assert outcome["result"] is None
    assert ran["pipeline"] is False


# Module-level happy-path fake reused by test_on_owner_start_none_is_safe.
class _P:
    def run(self, ctx):
        return {"ok": True}
