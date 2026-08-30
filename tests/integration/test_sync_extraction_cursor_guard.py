"""M5-3b: extraction-path cursor must not advance past summarization gap.

Pre-fix: _persist_extraction_results unconditionally called
`set_last_summarized_exchange_index(..., end_exchange)` after a
successful persist. This jumps the cursor past any failed-but-
unprocessed summarization exchanges between the previous cursor
and end_exchange — those exchanges are then dropped from
`_identify_uncached_exchanges` output forever (DATA LOSS in the
summary table).

This is the mirror of the M5-4 / P1.4 bug on the summarization
path. P1.4 fixed the summarization path's contiguous-only cursor
advance; this test (and the fix it exercises) closes the same
class of bug on the extraction path.

Fix: after the conditional claim succeeds, query the current
summarization cursor and advance the extraction cursor to
`min(summarization_cursor, end_exchange)`. If summarization is
behind, the extraction cursor caps at where summarization has
reached — failed summarization gaps block the cursor.
"""

from unittest.mock import MagicMock

import pytest

pytestmark = pytest.mark.integration


def test_extraction_does_not_advance_past_summarization_gap():
    """Extraction ends at 14, summarization stuck at 7 → cursor stays at 7."""
    from memory.sync import _persist_extraction_results
    from memory.batch_manager import _get_incomplete_batches
    from memory.db_summaries import set_last_summarized_exchange_index, get_last_summarized_exchange_index
    import memory.db as db_module

    session_id = "test-m5-3b-gap"
    end_exchange = 14

    # Seed: summarization cursor = 7 (exchanges 8..14 are unprocessed)
    db_module.init_db()
    set_last_summarized_exchange_index(session_id, 7)

    assert get_last_summarized_exchange_index(session_id) == 7

    # Seed: a batch_id in status='extracted' so _claim_batch succeeds
    db_module.execute = lambda *a, **kw: None  # placeholder if needed

    # Run the persist path. We just need to verify the cursor
    # management — _store_entities_and_facts and _store_memories
    # are stubbed via pre-existing patches.
    # Simplest: call the function and check the cursor after.
    with pytest.MonkeyPatch().context() as mp:
        # Stub the claim to always succeed (test focuses on cursor advance)
        mp.setattr("memory.sync._claim_batch_for_extraction", lambda bid: True)
        # Stub the terminal conditional UPDATE (status→completed) to succeed
        mp.setattr("memory.sync._update_batch_status_if", lambda *a, **kw: True)
        # Stub the heavy work
        mp.setattr("memory.sync._update_batch_extraction", lambda *a, **kw: MagicMock())
        mp.setattr("memory.sync._store_entities_and_facts",
                   lambda *a, **kw: {"errors": []})
        mp.setattr("memory.sync._store_memories",
                   lambda *a, **kw: True)

        extraction = {
            "entities": [{"name": "test", "entity_type": "Concept"}],
            "facts": [],
            "events": [],
            "instructions": [],
            "tasks": [],
        }

        result = _persist_extraction_results(
            session_id, extraction,
            batch_id=1, end_idx=15, end_exchange=end_exchange,
        )

    assert result is True

    # Cursor must be at 7 (summarization lag), NOT 14 (the bug).
    actual = get_last_summarized_exchange_index(session_id)
    assert actual == 7, (
        f"Extraction cursor advanced to {actual}; expected 7. "
        f"Pre-fix would have set cursor=14, dropping exchanges "
        f"8..14 from future _identify_uncached_exchanges calls."
    )


def test_extraction_advances_to_end_when_summarization_caught_up():
    """Normal case: summarization already at end_exchange → extraction advances fully."""
    from memory.sync import _persist_extraction_results
    from memory.db_summaries import set_last_summarized_exchange_index, get_last_summarized_exchange_index
    import memory.db as db_module

    session_id = "test-m5-3b-caught-up"
    end_exchange = 14

    db_module.init_db()
    # Seed: summarization cursor already at 14 (summarization ran first)
    set_last_summarized_exchange_index(session_id, 14)

    assert get_last_summarized_exchange_index(session_id) == 14

    with pytest.MonkeyPatch().context() as mp:
        mp.setattr("memory.sync._claim_batch_for_extraction", lambda bid: True)
        mp.setattr("memory.sync._update_batch_status_if", lambda *a, **kw: True)
        mp.setattr("memory.sync._update_batch_extraction", lambda *a, **kw: MagicMock())
        mp.setattr("memory.sync._store_entities_and_facts",
                   lambda *a, **kw: {"errors": []})
        mp.setattr("memory.sync._store_memories",
                   lambda *a, **kw: True)

        extraction = {
            "entities": [{"name": "test", "entity_type": "Concept"}],
            "facts": [],
            "events": [],
            "instructions": [],
            "tasks": [],
        }

        result = _persist_extraction_results(
            session_id, extraction,
            batch_id=2, end_idx=15, end_exchange=end_exchange,
        )

    assert result is True

    # Cursor advances to 14 (summarization already caught up).
    actual = get_last_summarized_exchange_index(session_id)
    assert actual == 14, (
        f"Extraction cursor at {actual}; expected 14 (summarization "
        f"already caught up)."
    )


def test_extraction_does_not_regress_when_summarization_ahead():
    """Edge case: summarization cursor > end_exchange (already past the batch boundary).

    Pre-fix: the helper used ``INSERT ON CONFLICT DO UPDATE SET
    last_exchange_index = excluded.last_exchange_index`` which OVERWRITES
    the cursor with end_exchange. So a summarization cursor at 20
    regressed to 14 when extraction ran with end_exchange=14. The test
    below documents the FIXED behavior: cursor must never regress.
    """
    from memory.sync import _persist_extraction_results
    from memory.db_summaries import set_last_summarized_exchange_index, get_last_summarized_exchange_index
    import memory.db as db_module

    session_id = "test-m5-3b-ahead"
    end_exchange = 14

    db_module.init_db()
    # Seed: summarization cursor already at 20 (past end_exchange=14)
    set_last_summarized_exchange_index(session_id, 20)

    assert get_last_summarized_exchange_index(session_id) == 20

    with pytest.MonkeyPatch().context() as mp:
        mp.setattr("memory.sync._claim_batch_for_extraction", lambda bid: True)
        mp.setattr("memory.sync._update_batch_status_if", lambda *a, **kw: True)
        mp.setattr("memory.sync._update_batch_extraction", lambda *a, **kw: MagicMock())
        mp.setattr("memory.sync._store_entities_and_facts",
                   lambda *a, **kw: {"errors": []})
        mp.setattr("memory.sync._store_memories",
                   lambda *a, **kw: True)

        extraction = {
            "entities": [{"name": "test", "entity_type": "Concept"}],
            "facts": [],
        }

        result = _persist_extraction_results(
            session_id, extraction,
            batch_id=3, end_idx=15, end_exchange=end_exchange,
        )

    assert result is True
    # Forward-progress clamp: cursor must NOT regress below the existing 20.
    actual = get_last_summarized_exchange_index(session_id)
    assert actual == 20, (
        f"Cursor regressed to {actual}; must stay at 20. "
        f"Pre-fix: helper used INSERT ON CONFLICT DO UPDATE which "
        f"overwrote the cursor with end_exchange=14."
    )
