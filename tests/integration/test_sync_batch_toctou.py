"""M5-3a: TOCTOU race on _persist_extraction_results.

Pre-fix: _persist_extraction_results did read-then-write on batch
status. Two concurrent syncs both passed the check, both called
the unconditional UPDATE, both wrote entities/facts -> duplicate
Neo4j writes + duplicate memory_log rows.

Fix: terminal "completed" status is now a conditional UPDATE
(_update_batch_status_if with WHERE status IN ('extracted',
'processing')). rowcount=0 means another worker won the race.
The conditional UPDATE is the last step, so we never do the work
and then fail to claim.
"""

import os
import threading

import pytest


pytestmark = pytest.mark.integration


@pytest.fixture(autouse=True)
def setup_test_db(tmp_path, monkeypatch):
    db_path = str(tmp_path / "memory.db")
    monkeypatch.setenv("PROXY_TEST_DB", db_path)

    from memory.db import init_db

    init_db(db_path)
    yield db_path


def _make_batch(status="extracted"):
    from memory.db_summaries import record_extraction_batch

    return record_extraction_batch(
        "sess-toctou",
        0,
        1,
        start_message_index=0,
        end_message_index=1,
        status=status,
    )


# ---------------------------------------------------------------------------
# SQL-helper unit tests — the core invariant of the fix
# ---------------------------------------------------------------------------


def test_conditional_update_returns_false_on_claim_loss(setup_test_db):
    """The new helper returns False when another worker has already moved
    the row out of the expected status. This is the atomicity gate that
    replaces the read-then-write at the top of _persist_extraction_results.
    """
    from memory.batch_manager import _update_batch_status, _update_batch_status_if
    from memory.db_summaries import get_batch_status

    bid = _make_batch(status="pending")
    assert get_batch_status(bid) == "pending"

    assert _update_batch_status_if(bid, "pending", "processing") is True
    assert get_batch_status(bid) == "processing"

    assert _update_batch_status_if(bid, "pending", "processing") is False
    assert get_batch_status(bid) == "processing"

    assert _update_batch_status_if(bid, ("extracted", "processing"), "completed") is True
    assert get_batch_status(bid) == "completed"

    assert _update_batch_status_if(bid, ("extracted", "processing"), "completed") is False
    assert get_batch_status(bid) == "completed"

    _update_batch_status(bid, "pending")


def test_conditional_update_unknown_id_returns_false(setup_test_db):
    """A missing row -> WHERE matches nothing -> rowcount=0 -> False."""
    from memory.batch_manager import _update_batch_status_if

    assert _update_batch_status_if(424242, "pending", "completed") is False


def test_conditional_update_atomic_under_concurrency(setup_test_db):
    """Two threads race on the same row via the SQL helper. Exactly one
    claims the transition; the other returns False. This is the actual
    atomicity guarantee the conditional UPDATE provides.
    """
    from memory.batch_manager import _update_batch_status_if
    from memory.db_summaries import get_batch_status

    bid = _make_batch(status="extracted")

    barrier = threading.Barrier(8)
    results = []
    lock = threading.Lock()

    def worker():
        barrier.wait()
        result = _update_batch_status_if(
            bid, ("extracted", "processing"), "completed"
        )
        with lock:
            results.append(result)

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert results.count(True) == 1, (
        f"exactly one of the 8 racers must claim 'completed', got "
        f"{results.count(True)} True(s) in {results}"
    )
    assert results.count(False) == 7
    assert get_batch_status(bid) == "completed"


# ---------------------------------------------------------------------------
# _persist_extraction_results happy-path tests
# ---------------------------------------------------------------------------


def test_persist_fresh_batch_transitions_to_completed(setup_test_db, monkeypatch):
    """A fresh batch in 'processing' state is correctly transitioned to
    'completed' by the conditional UPDATE at the end. Returns True.
    """
    from memory import sync as sync_mod
    from memory.db_summaries import get_batch_status

    _stub_persistence(monkeypatch)

    bid = _make_batch(status="processing")

    assert sync_mod._persist_extraction_results(
        "sess-toctou", _extraction_with_entities(), bid, 1, end_exchange=1
    ) is True
    assert get_batch_status(bid) == "completed"


def test_persist_extracted_batch_transitions_to_completed(setup_test_db, monkeypatch):
    """A batch in 'extracted' state (recovered from a crashed previous run)
    is correctly transitioned to 'completed' by the conditional UPDATE.
    Returns True.

    The orchestrator's recovery path (memory/sync.py:1134) handles the
    'extracted' state by first calling `_set_batch_processing` to reset
    to 'processing' before re-entering `_persist_extraction_results`.

    BL5 (arch-review-2): this test used to bypass the claim by calling
    `_update_batch_status(bid, "processing")` directly. That masked the
    bug in 5dc8801 where _set_batch_processing's WHERE clause lacked
    'extracted'. Now the test calls the actual claim function
    (_set_batch_processing) the same way restart_recovery does — which
    proves the SQL invariant holds end-to-end.
    """
    from memory import sync as sync_mod
    from memory.batch_manager import _set_batch_processing
    from memory.db_summaries import get_batch_status

    _stub_persistence(monkeypatch)

    bid = _make_batch(status="extracted")

    # Recovery reset via the REAL claim function (not a manual UPDATE).
    # _set_batch_processing must accept 'extracted' as a valid from-status
    # to support the post-crash recovery cycle. Pre-fix this returns
    # (False, False) and the row stays 'extracted'.
    claimed, was_failed = _set_batch_processing(bid)
    assert claimed is True, (
        f"_set_batch_processing must accept an 'extracted' batch (the "
        f"crash-mid-completion state). Pre-fix this returned "
        f"(False, False); post-fix returns (True, was_previously_failed)."
    )
    assert was_failed is False, (
        "'extracted' is not 'failed', so was_previously_failed should "
        f"be False; got {was_failed}"
    )
    assert get_batch_status(bid) == "processing", (
        f"After _set_batch_processing on an 'extracted' batch, status "
        f"must be 'processing' (the transition target). Got "
        f"{get_batch_status(bid)!r}."
    )

    # Now the inline _persist_extraction_results completes the row.
    assert sync_mod._persist_extraction_results(
        "sess-toctou", _extraction_with_entities(), bid, 1, end_exchange=1
    ) is True
    assert get_batch_status(bid) == "completed"


def test_persist_empty_extraction_still_claims_completed(setup_test_db, monkeypatch):
    """An empty extraction (no entities/facts) still transitions to
    'completed' via the conditional UPDATE — returns True on a fresh row.
    """
    from memory import sync as sync_mod
    from memory.db_summaries import get_batch_status

    _stub_persistence(monkeypatch)

    bid = _make_batch(status="processing")

    assert sync_mod._persist_extraction_results(
        "sess-toctou", {"entities": [], "facts": []}, bid, 1, end_exchange=1
    ) is True
    assert get_batch_status(bid) == "completed"


def test_persist_failed_status_marks_failed(setup_test_db, monkeypatch):
    """If the persistence helpers report errors, the batch is marked
    'failed' (unconditional, per the spec — failed paths are not contended).
    """
    from memory import sync as sync_mod
    from memory.db_summaries import get_batch_status
    from memory import persistence as persistence_mod

    def fake_store_entities_and_facts(session_id, extraction):
        return {"errors": ["boom"]}

    def fake_store_memories(session_id, events, instructions, tasks):
        return True

    monkeypatch.setattr(sync_mod, "_store_entities_and_facts", fake_store_entities_and_facts)
    monkeypatch.setattr(sync_mod, "_store_memories", fake_store_memories)
    monkeypatch.setattr(sync_mod._db(), "set_last_summarized_exchange_index", lambda *a, **kw: None)

    bid = _make_batch(status="processing")

    assert sync_mod._persist_extraction_results(
        "sess-toctou", _extraction_with_entities(), bid, 1, end_exchange=1
    ) is False
    assert get_batch_status(bid) == "failed"


# ---------------------------------------------------------------------------
# M5-3a Window 2 — claim at top of _persist_extraction_results
# ---------------------------------------------------------------------------


def test_concurrent_persist_one_wins(setup_test_db, monkeypatch):
    """Two threads racing on _persist_extraction_results → only one returns True.

    Window 1 (concurrent race): both threads try to claim the same batch
    at the same time. The first to acquire the claim UPDATE sets
    status='extracted'. The second sees status='extracted' (not 'pending')
    and the claim returns False. So one returns True, one returns False.
    """
    from memory import sync as sync_mod
    from memory.db_summaries import get_batch_status

    _stub_persistence(monkeypatch)

    bid = _make_batch(status="pending")

    barrier = threading.Barrier(2)
    results = []
    lock = threading.Lock()
    payload = _extraction_with_entities()

    def worker():
        barrier.wait()
        result = sync_mod._persist_extraction_results(
        "sess-toctou", payload, bid, 1, end_exchange=1
        )
        with lock:
            results.append(result)

    threads = [threading.Thread(target=worker) for _ in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert results.count(True) == 1, (
        f"exactly one of the 2 racers must claim 'extracted', got "
        f"{results.count(True)} True(s) in {results}"
    )
    assert results.count(False) == 1
    assert get_batch_status(bid) == "completed"


def test_sequential_persist_after_completion_does_not_duplicate(
    setup_test_db, monkeypatch
):
    """Thread A completes. Thread B starts AFTER A finishes.

    Window 2 (sequential after completion): Thread B's _claim_batch_for_extraction
    finds status='completed' (not 'pending') and the claim returns False.
    Thread B returns False without re-doing the persist work.
    """
    from memory import sync as sync_mod
    from memory.db_summaries import get_batch_status

    _stub_persistence(monkeypatch)

    bid = _make_batch(status="pending")
    payload = _extraction_with_entities()

    # Thread A: full flow → status="completed"
    result_a = sync_mod._persist_extraction_results(
        "sess-toctou", payload, bid, 1, end_exchange=1
    )
    assert result_a is True
    assert get_batch_status(bid) == "completed"

    # Thread B: starts after A is fully done
    result_b = sync_mod._persist_extraction_results(
        "sess-toctou", payload, bid, 1, end_exchange=1
    )
    assert result_b is False  # claim failed — status was "completed"
    assert get_batch_status(bid) == "completed"  # status unchanged


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _stub_persistence(monkeypatch):
    """Stub Neo4j-touching helpers so the test runs without a real graph."""
    from memory import sync as sync_mod

    def fake_store_entities_and_facts(session_id, extraction):
        return {"errors": []}

    def fake_store_memories(session_id, events, instructions, tasks):
        return True

    monkeypatch.setattr(sync_mod, "_store_entities_and_facts", fake_store_entities_and_facts)
    monkeypatch.setattr(sync_mod, "_store_memories", fake_store_memories)
    monkeypatch.setattr(sync_mod._db(), "set_last_summarized_exchange_index", lambda *a, **kw: None)


def _extraction_with_entities():
    return {
        "entities": [{"name": "alpha", "entity_type": "Concept"}],
        "facts": [{"content": "alpha is a concept"}],
        "events": [],
        "instructions": [],
        "tasks": [],
    }
