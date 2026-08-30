import os
import pytest
from contextlib import contextmanager
from unittest.mock import MagicMock, patch
from memory.sync import sync_exchange, _update_batch_status, _update_batch_extraction
from memory.sota_client import SOTATimeoutError, SOTAAPIError, SOTAParseError


# Reusable summaries list for get_rich_summaries mock (3+ meaningful → extraction gate passes)
_STANDARD_SUMMARIES = [
    {"message_index": 1, "exchange_index": 1, "summary": "User asked about Python async patterns.", "rendered_narrative_tokens": 100},
    {"message_index": 2, "exchange_index": 2, "summary": "Assistant explained asyncio event loop.", "rendered_narrative_tokens": 120},
    {"message_index": 3, "exchange_index": 3, "summary": "User requested a code example.", "rendered_narrative_tokens": 90},
]

_STANDARD_EXTRACTION = {"entities": [{"name": "test", "entity_type": "Concept", "summary": "test"}], "facts": []}


class TestUnifiedSyncIntegration:
    # ------------------------------------------------------------------
    # 1. Single SOTA call
    # ------------------------------------------------------------------
    def test_sync_exchange_single_sota_call(self):
        with patch("memory.sync._get_sota") as mock_get_sota:
            mock_sota = MagicMock()
            mock_get_sota.return_value = mock_sota
            mock_sota.extract_and_summarize_from_summaries.return_value = _STANDARD_EXTRACTION

            with patch("memory.sync._summarize_uncached_exchanges", return_value={"stored": 0, "found_uncached": False}):
                with patch("memory.db.get_max_end_exchange_index", return_value=-1):
                    with patch("memory.db.get_max_exchange_index", return_value=5):
                        mock_conn = MagicMock()
                        mock_conn.execute.return_value.fetchone.return_value = {"start_idx": 0, "end_idx": 5}

                        @contextmanager
                        def mock_connection_ctx(*args, **kwargs):
                            yield mock_conn

                        with patch("memory.db_pool.connection", side_effect=mock_connection_ctx):
                            with patch("memory.db.get_messages_in_range", return_value=[{"role": "user", "content": "Hello"}]):
                                with patch("memory.db.get_extraction_batch", return_value=None):
                                    with patch("memory.db.record_extraction_batch", return_value="batch_123"):
                                        with patch("memory.batch_manager._set_batch_processing", return_value=(True, False)):
                                            with patch("memory.db.get_summary_age", return_value=None):
                                                with patch("memory.db.get_last_summarized_exchange_index", return_value=-1):
                                                    with patch("memory.db.get_latest_message_index", return_value=5):
                                                        with patch("memory.db.get_rich_summaries", return_value=_STANDARD_SUMMARIES):
                                                            with patch("memory.db.get_batch_status", return_value="processing"):
                                                                with patch("memory.sync._store_entities_and_facts", return_value={"errors": []}):
                                                                    with patch("memory.db.set_last_summarized_exchange_index"):
                                                                        with patch("memory.sync._update_batch_extraction"):
                                                                            with patch("memory.sync._update_batch_status"):
                                                                                sync_exchange("session_single")

        mock_sota.extract_and_summarize_from_summaries.assert_called_once()

    # ------------------------------------------------------------------
    # 2. Stores extraction and summary
    # ------------------------------------------------------------------
    def test_sync_exchange_stores_extraction_and_summary(self):
        with patch("memory.sync._get_sota") as mock_get_sota:
            mock_sota = MagicMock()
            mock_get_sota.return_value = mock_sota
            mock_sota.extract_and_summarize_from_summaries.return_value = {
                "entities": [{"name": "extracted_fact", "entity_type": "Concept"}],
                "facts": [],
            }

            with patch("memory.sync._summarize_uncached_exchanges", return_value={"stored": 0, "found_uncached": False}):
                with patch("memory.db.get_max_end_exchange_index", return_value=-1):
                    with patch("memory.db.get_max_exchange_index", return_value=5):
                        mock_conn = MagicMock()
                        mock_conn.execute.return_value.fetchone.return_value = {"start_idx": 0, "end_idx": 5}

                        @contextmanager
                        def mock_connection_ctx(*args, **kwargs):
                            yield mock_conn

                        with patch("memory.db_pool.connection", side_effect=mock_connection_ctx):
                            with patch("memory.db.get_messages_in_range", return_value=[{"role": "user", "content": "test"}]):
                                with patch("memory.db.get_extraction_batch", return_value=None):
                                    with patch("memory.db.record_extraction_batch", return_value="batch_456"):
                                        with patch("memory.batch_manager._set_batch_processing", return_value=(True, False)):
                                            with patch("memory.db.get_summary_age", return_value=None):
                                                with patch("memory.db.get_last_summarized_exchange_index", return_value=-1):
                                                    with patch("memory.db.get_latest_message_index", return_value=5):
                                                        with patch("memory.db.get_rich_summaries", return_value=_STANDARD_SUMMARIES):
                                                            with patch("memory.db.get_batch_status", return_value="processing"):
                                                                with patch("memory.sync._store_entities_and_facts", return_value={"errors": []}):
                                                                    with patch("memory.db.set_last_summarized_exchange_index"):
                                                                        with patch("memory.sync._claim_batch_for_extraction", return_value=True):
                                                                            with patch("memory.sync._update_batch_extraction") as mock_extraction:
                                                                                with patch("memory.sync._update_batch_status"):
                                                                                    sync_exchange("session_store")

        mock_extraction.assert_called()

    # ------------------------------------------------------------------
    # 3. Summary gate skips when not due
    # ------------------------------------------------------------------
    def test_sync_exchange_skips_when_gating_not_due(self):
        with patch("memory.sync._get_sota") as mock_get_sota:
            mock_sota = MagicMock()
            mock_get_sota.return_value = mock_sota

            with patch("memory.sync._summarize_uncached_exchanges", return_value={"stored": 0, "found_uncached": False}):
                with patch("memory.db.get_max_end_exchange_index", return_value=-1):
                    with patch("memory.db.get_max_exchange_index", return_value=5):
                        mock_conn = MagicMock()
                        mock_conn.execute.return_value.fetchone.return_value = {"start_idx": 0, "end_idx": 5}

                        @contextmanager
                        def mock_connection_ctx(*args, **kwargs):
                            yield mock_conn

                        with patch("memory.db_pool.connection", side_effect=mock_connection_ctx):
                            with patch("memory.db.get_messages_in_range", return_value=[{"role": "user", "content": "test"}]):
                                with patch("memory.db.get_extraction_batch", return_value=None):
                                    with patch("memory.db.record_extraction_batch", return_value="batch_gated"):
                                        with patch("memory.batch_manager._set_batch_processing", return_value=(True, False)):
                                            with patch("memory.db.get_summary_age", return_value=30):
                                                with patch("memory.db.get_last_summarized_exchange_index", return_value=3):
                                                    with patch("memory.db.get_latest_message_index", return_value=5):
                                                        with patch("memory.sync._update_batch_status") as mock_status:
                                                            with patch.dict(os.environ, {"PROXY_SOTA_SUMMARY_GATE": "true"}):
                                                                sync_exchange("session_gated")

        mock_sota.extract_and_summarize_from_summaries.assert_not_called()
        mock_status.assert_called_with("batch_gated", "skipped")

    # ------------------------------------------------------------------
    # 4. Gate disabled → extraction proceeds
    # ------------------------------------------------------------------
    def test_sync_exchange_runs_when_gating_disabled(self):
        with patch("memory.sync._get_sota") as mock_get_sota:
            mock_sota = MagicMock()
            mock_get_sota.return_value = mock_sota
            mock_sota.extract_and_summarize_from_summaries.return_value = {"entities": [], "facts": []}

            with patch("memory.sync._summarize_uncached_exchanges", return_value={"stored": 0, "found_uncached": False}):
                with patch("memory.db.get_max_end_exchange_index", return_value=-1):
                    with patch("memory.db.get_max_exchange_index", return_value=5):
                        mock_conn = MagicMock()
                        mock_conn.execute.return_value.fetchone.return_value = {"start_idx": 0, "end_idx": 5}

                        @contextmanager
                        def mock_connection_ctx(*args, **kwargs):
                            yield mock_conn

                        with patch("memory.db_pool.connection", side_effect=mock_connection_ctx):
                            with patch("memory.db.get_messages_in_range", return_value=[{"role": "user", "content": "test"}]):
                                with patch("memory.db.get_extraction_batch", return_value=None):
                                    with patch("memory.db.record_extraction_batch", return_value="batch_no_gate"):
                                        with patch("memory.batch_manager._set_batch_processing", return_value=(True, False)):
                                            with patch("memory.db.get_summary_age", return_value=None):
                                                with patch("memory.db.get_last_summarized_exchange_index", return_value=-1):
                                                    with patch("memory.db.get_latest_message_index", return_value=5):
                                                        with patch("memory.db.get_rich_summaries", return_value=_STANDARD_SUMMARIES):
                                                            with patch("memory.db.get_batch_status", return_value="processing"):
                                                                with patch("memory.sync._update_batch_status"):
                                                                    with patch.dict(os.environ, {"PROXY_SOTA_SUMMARY_GATE": "false"}):
                                                                        sync_exchange("session_no_gate")

        mock_sota.extract_and_summarize_from_summaries.assert_called_once()

    # ------------------------------------------------------------------
    # 5. Retry on transient error (extraction failure path)
    # ------------------------------------------------------------------
    def test_retry_on_transient_error(self):
        with patch("memory.sync._get_sota") as mock_get_sota:
            mock_sota = MagicMock()
            mock_get_sota.return_value = mock_sota
            mock_sota.extract_and_summarize_from_summaries.side_effect = SOTAAPIError("Transient error")

            with patch("memory.sync._summarize_uncached_exchanges", return_value={"stored": 0, "found_uncached": False}):
                with patch("memory.db.get_max_end_exchange_index", return_value=-1):
                    with patch("memory.db.get_max_exchange_index", return_value=5):
                        mock_conn = MagicMock()
                        mock_conn.execute.return_value.fetchone.return_value = {"start_idx": 0, "end_idx": 5}

                        @contextmanager
                        def mock_connection_ctx(*args, **kwargs):
                            yield mock_conn

                        with patch("memory.db_pool.connection", side_effect=mock_connection_ctx):
                            with patch("memory.db.get_messages_in_range", return_value=[{"role": "user", "content": "test"}]):
                                with patch("memory.db.get_extraction_batch", return_value=None):
                                    with patch("memory.db.record_extraction_batch", return_value="batch_retry"):
                                        with patch("memory.batch_manager._set_batch_processing", return_value=(True, False)):
                                            with patch("memory.db.get_summary_age", return_value=None):
                                                with patch("memory.db.get_last_summarized_exchange_index", return_value=-1):
                                                    with patch("memory.db.get_latest_message_index", return_value=5):
                                                        with patch("memory.db.get_rich_summaries", return_value=_STANDARD_SUMMARIES):
                                                            with patch("memory.sync._update_batch_status") as mock_status:
                                                                sync_exchange("session_retry")

        mock_status.assert_called_with("batch_retry", "failed", error="extraction_error: Transient error")

    # ------------------------------------------------------------------
    # 6. Retry exhausted (extraction failure path)
    # ------------------------------------------------------------------
    def test_retry_exhausted_raises(self):
        with patch("memory.sync._get_sota") as mock_get_sota:
            mock_sota = MagicMock()
            mock_get_sota.return_value = mock_sota
            mock_sota.extract_and_summarize_from_summaries.side_effect = SOTAAPIError("Permanent error")

            with patch("memory.sync._summarize_uncached_exchanges", return_value={"stored": 0, "found_uncached": False}):
                with patch("memory.db.get_max_end_exchange_index", return_value=-1):
                    with patch("memory.db.get_max_exchange_index", return_value=5):
                        mock_conn = MagicMock()
                        mock_conn.execute.return_value.fetchone.return_value = {"start_idx": 0, "end_idx": 5}

                        @contextmanager
                        def mock_connection_ctx(*args, **kwargs):
                            yield mock_conn

                        with patch("memory.db_pool.connection", side_effect=mock_connection_ctx):
                            with patch("memory.db.get_messages_in_range", return_value=[{"role": "user", "content": "test"}]):
                                with patch("memory.db.get_extraction_batch", return_value=None):
                                    with patch("memory.db.record_extraction_batch", return_value="batch_exhausted"):
                                        with patch("memory.batch_manager._set_batch_processing", return_value=(True, False)):
                                            with patch("memory.db.get_summary_age", return_value=None):
                                                with patch("memory.db.get_last_summarized_exchange_index", return_value=-1):
                                                    with patch("memory.db.get_latest_message_index", return_value=5):
                                                        with patch("memory.db.get_rich_summaries", return_value=_STANDARD_SUMMARIES):
                                                            with patch("memory.sync._update_batch_status") as mock_status:
                                                                sync_exchange("session_exhausted")

        mock_status.assert_called_with("batch_exhausted", "failed", error="extraction_error: Permanent error")

    # ------------------------------------------------------------------
    # 7. Concurrent sync claims batch → extraction not called
    # ------------------------------------------------------------------
    def test_concurrent_sync_same_session(self):
        with patch("memory.sync._get_sota") as mock_get_sota:
            mock_sota = MagicMock()
            mock_get_sota.return_value = mock_sota

            with patch("memory.sync._summarize_uncached_exchanges", return_value={"stored": 0, "found_uncached": False}):
                with patch("memory.db.get_max_end_exchange_index", return_value=-1):
                    with patch("memory.db.get_max_exchange_index", return_value=5):
                        mock_conn = MagicMock()
                        mock_conn.execute.return_value.fetchone.return_value = {"start_idx": 0, "end_idx": 5}

                        @contextmanager
                        def mock_connection_ctx(*args, **kwargs):
                            yield mock_conn

                        with patch("memory.db_pool.connection", side_effect=mock_connection_ctx):
                            with patch("memory.db.get_messages_in_range", return_value=[{"role": "user", "content": "test"}]):
                                with patch("memory.db.get_extraction_batch", return_value=None):
                                    with patch("memory.db.record_extraction_batch", return_value="batch_concurrent"):
                                        with patch("memory.batch_manager._set_batch_processing", return_value=(False, False)):
                                            with patch("memory.sync._update_batch_status"):
                                                sync_exchange("session_concurrent")

        mock_sota.extract_and_summarize_from_summaries.assert_not_called()

    # ------------------------------------------------------------------
    # 8. Batch status updated after successful extraction + persistence
    # ------------------------------------------------------------------
    def test_neo4j_failure_skips_summary_upsert(self):
        with patch("memory.sync._get_sota") as mock_get_sota:
            mock_sota = MagicMock()
            mock_get_sota.return_value = mock_sota
            mock_sota.extract_and_summarize_from_summaries.return_value = _STANDARD_EXTRACTION

            with patch("memory.sync._summarize_uncached_exchanges", return_value={"stored": 0, "found_uncached": False}):
                with patch("memory.db.get_max_end_exchange_index", return_value=-1):
                    with patch("memory.db.get_max_exchange_index", return_value=5):
                        mock_conn = MagicMock()
                        mock_conn.execute.return_value.fetchone.return_value = {"start_idx": 0, "end_idx": 5}

                        @contextmanager
                        def mock_connection_ctx(*args, **kwargs):
                            yield mock_conn

                        with patch("memory.db_pool.connection", side_effect=mock_connection_ctx):
                            with patch("memory.db.get_messages_in_range", return_value=[{"role": "user", "content": "test"}]):
                                with patch("memory.db.get_extraction_batch", return_value=None):
                                    with patch("memory.db.record_extraction_batch", return_value="batch_neo4j_fail"):
                                        with patch("memory.batch_manager._set_batch_processing", return_value=(True, False)):
                                            with patch("memory.db.get_summary_age", return_value=None):
                                                with patch("memory.db.get_last_summarized_exchange_index", return_value=-1):
                                                    with patch("memory.db.get_latest_message_index", return_value=5):
                                                        with patch("memory.db.get_rich_summaries", return_value=_STANDARD_SUMMARIES):
                                                            with patch("memory.db.get_batch_status", return_value="processing"):
                                                                with patch("memory.sync._store_entities_and_facts", return_value={"errors": []}):
                                                                    with patch("memory.db.set_last_summarized_exchange_index"):
                                                                        with patch("memory.sync._claim_batch_for_extraction", return_value=True):
                                                                            with patch("memory.sync._update_batch_extraction"):
                                                                                with patch("memory.sync._update_batch_status_if", return_value=True) as mock_status_if:
                                                                                    with patch("memory.sync._update_batch_status") as mock_status:
                                                                                        sync_exchange("session_neo4j_fail")

        assert mock_status_if.call_count >= 1

    # ------------------------------------------------------------------
    # 9. Empty extraction → batch marked completed
    # ------------------------------------------------------------------
    def test_empty_extraction_marks_completed(self):
        with patch("memory.sync._get_sota") as mock_get_sota:
            mock_sota = MagicMock()
            mock_get_sota.return_value = mock_sota
            mock_sota.extract_and_summarize_from_summaries.return_value = {"entities": [], "facts": []}

            with patch("memory.sync._summarize_uncached_exchanges", return_value={"stored": 0, "found_uncached": False}):
                with patch("memory.db.get_max_end_exchange_index", return_value=-1):
                    with patch("memory.db.get_max_exchange_index", return_value=5):
                        mock_conn = MagicMock()
                        mock_conn.execute.return_value.fetchone.return_value = {"start_idx": 0, "end_idx": 5}

                        @contextmanager
                        def mock_connection_ctx(*args, **kwargs):
                            yield mock_conn

                        with patch("memory.db_pool.connection", side_effect=mock_connection_ctx):
                            with patch("memory.db.get_messages_in_range", return_value=[{"role": "user", "content": "test"}]):
                                with patch("memory.db.get_extraction_batch", return_value=None):
                                    with patch("memory.db.record_extraction_batch", return_value="batch_empty"):
                                        with patch("memory.batch_manager._set_batch_processing", return_value=(True, False)):
                                            with patch("memory.db.get_summary_age", return_value=None):
                                                with patch("memory.db.get_last_summarized_exchange_index", return_value=-1):
                                                    with patch("memory.db.get_latest_message_index", return_value=5):
                                                        with patch("memory.db.get_rich_summaries", return_value=_STANDARD_SUMMARIES):
                                                            with patch("memory.db.get_batch_status", return_value="processing"):
                                                                with patch("memory.sync._claim_batch_for_extraction", return_value=True):
                                                                    with patch("memory.sync._update_batch_status_if", return_value=True) as mock_status_if:
                                                                        with patch("memory.sync._update_batch_status") as mock_status:
                                                                            sync_exchange("session_empty")

        completed_calls = [c for c in mock_status_if.call_args_list
                           if c[0][0] == "batch_empty" and c.kwargs.get("to_status") == "completed"]
        assert len(completed_calls) > 0

    # ------------------------------------------------------------------
    # 10. SOTA timeout → extraction error (inner handler)
    # ------------------------------------------------------------------
    def test_batch_status_on_sota_timeout(self):
        with patch("memory.sync._get_sota") as mock_get_sota:
            mock_sota = MagicMock()
            mock_get_sota.return_value = mock_sota
            mock_sota.extract_and_summarize_from_summaries.side_effect = SOTATimeoutError("Timeout")

            with patch("memory.sync._summarize_uncached_exchanges", return_value={"stored": 0, "found_uncached": False}):
                with patch("memory.db.get_max_end_exchange_index", return_value=-1):
                    with patch("memory.db.get_max_exchange_index", return_value=5):
                        mock_conn = MagicMock()
                        mock_conn.execute.return_value.fetchone.return_value = {"start_idx": 0, "end_idx": 5}

                        @contextmanager
                        def mock_connection_ctx(*args, **kwargs):
                            yield mock_conn

                        with patch("memory.db_pool.connection", side_effect=mock_connection_ctx):
                            with patch("memory.db.get_messages_in_range", return_value=[{"role": "user", "content": "test"}]):
                                with patch("memory.db.get_extraction_batch", return_value=None):
                                    with patch("memory.db.record_extraction_batch", return_value="batch_timeout"):
                                        with patch("memory.batch_manager._set_batch_processing", return_value=(True, False)):
                                            with patch("memory.db.get_summary_age", return_value=None):
                                                with patch("memory.db.get_last_summarized_exchange_index", return_value=-1):
                                                    with patch("memory.db.get_latest_message_index", return_value=5):
                                                        with patch("memory.db.get_rich_summaries", return_value=_STANDARD_SUMMARIES):
                                                            with patch("memory.sync._update_batch_status") as mock_status:
                                                                sync_exchange("session_timeout")

        mock_status.assert_called_with("batch_timeout", "failed", error="extraction_error: Timeout")

    # ------------------------------------------------------------------
    # 11. SOTA parse error → extraction error (inner handler)
    # ------------------------------------------------------------------
    def test_batch_status_on_parse_error(self):
        with patch("memory.sync._get_sota") as mock_get_sota:
            mock_sota = MagicMock()
            mock_get_sota.return_value = mock_sota
            mock_sota.extract_and_summarize_from_summaries.side_effect = SOTAParseError("Parse error")

            with patch("memory.sync._summarize_uncached_exchanges", return_value={"stored": 0, "found_uncached": False}):
                with patch("memory.db.get_max_end_exchange_index", return_value=-1):
                    with patch("memory.db.get_max_exchange_index", return_value=5):
                        mock_conn = MagicMock()
                        mock_conn.execute.return_value.fetchone.return_value = {"start_idx": 0, "end_idx": 5}

                        @contextmanager
                        def mock_connection_ctx(*args, **kwargs):
                            yield mock_conn

                        with patch("memory.db_pool.connection", side_effect=mock_connection_ctx):
                            with patch("memory.db.get_messages_in_range", return_value=[{"role": "user", "content": "test"}]):
                                with patch("memory.db.get_extraction_batch", return_value=None):
                                    with patch("memory.db.record_extraction_batch", return_value="batch_parse"):
                                        with patch("memory.batch_manager._set_batch_processing", return_value=(True, False)):
                                            with patch("memory.db.get_summary_age", return_value=None):
                                                with patch("memory.db.get_last_summarized_exchange_index", return_value=-1):
                                                    with patch("memory.db.get_latest_message_index", return_value=5):
                                                        with patch("memory.db.get_rich_summaries", return_value=_STANDARD_SUMMARIES):
                                                            with patch("memory.sync._update_batch_status") as mock_status:
                                                                sync_exchange("session_parse")

        mock_status.assert_called_with("batch_parse", "failed", error="extraction_error: Parse error")

    # ------------------------------------------------------------------
    # 12. _store_entities_and_facts raises → outer catch-all → "unexpected"
    # ------------------------------------------------------------------
    def test_sqlite_failure_after_neo4j_success(self):
        with patch("memory.sync._get_sota") as mock_get_sota:
            mock_sota = MagicMock()
            mock_get_sota.return_value = mock_sota
            mock_sota.extract_and_summarize_from_summaries.return_value = _STANDARD_EXTRACTION

            with patch("memory.sync._summarize_uncached_exchanges", return_value={"stored": 0, "found_uncached": False}):
                with patch("memory.db.get_max_end_exchange_index", return_value=-1):
                    with patch("memory.db.get_max_exchange_index", return_value=5):
                        mock_conn = MagicMock()
                        mock_conn.execute.return_value.fetchone.return_value = {"start_idx": 0, "end_idx": 5}

                        @contextmanager
                        def mock_connection_ctx(*args, **kwargs):
                            yield mock_conn

                        with patch("memory.db_pool.connection", side_effect=mock_connection_ctx):
                            with patch("memory.db.get_messages_in_range", return_value=[{"role": "user", "content": "test"}]):
                                with patch("memory.db.get_extraction_batch", return_value=None):
                                    with patch("memory.db.record_extraction_batch", return_value="batch_sqlite_fail"):
                                        with patch("memory.batch_manager._set_batch_processing", return_value=(True, False)):
                                            with patch("memory.db.get_summary_age", return_value=None):
                                                with patch("memory.db.get_last_summarized_exchange_index", return_value=-1):
                                                    with patch("memory.db.get_latest_message_index", return_value=5):
                                                        with patch("memory.db.get_rich_summaries", return_value=_STANDARD_SUMMARIES):
                                                            with patch("memory.db.get_batch_status", return_value="processing"):
                                                                with patch("memory.sync._claim_batch_for_extraction", return_value=True):
                                                                    with patch("memory.sync._update_batch_extraction"):
                                                                        with patch("memory.sync._update_batch_status") as mock_status:
                                                                            with patch("memory.sync._store_entities_and_facts",
                                                                                       side_effect=Exception("SQLite error")):
                                                                                sync_exchange("session_sqlite_fail")

        mock_status.assert_called_with("batch_sqlite_fail", "failed", error="unexpected")

    # ------------------------------------------------------------------
    # 13. Rate limiter serializes → single extraction call
    # ------------------------------------------------------------------
    def test_rate_limiter_serializes(self):
        with patch("memory.sync._get_sota") as mock_get_sota:
            mock_sota = MagicMock()
            mock_get_sota.return_value = mock_sota
            mock_sota.extract_and_summarize_from_summaries.return_value = {"entities": [], "facts": []}

            with patch("memory.sync._summarize_uncached_exchanges", return_value={"stored": 0, "found_uncached": False}):
                with patch("memory.db.get_max_end_exchange_index", return_value=-1):
                    with patch("memory.db.get_max_exchange_index", return_value=50):
                        mock_conn = MagicMock()
                        mock_conn.execute.return_value.fetchone.return_value = {"start_idx": 0, "end_idx": 50}

                        @contextmanager
                        def mock_connection_ctx(*args, **kwargs):
                            yield mock_conn

                        with patch("memory.db_pool.connection", side_effect=mock_connection_ctx):
                            messages = [{"role": "user", "content": f"msg{i}"} for i in range(50)]
                            with patch("memory.db.get_messages_in_range", return_value=messages):
                                with patch("memory.db.get_extraction_batch", return_value=None):
                                    with patch("memory.db.record_extraction_batch", return_value="batch_rate"):
                                        with patch("memory.batch_manager._set_batch_processing", return_value=(True, False)):
                                            with patch("memory.db.get_summary_age", return_value=None):
                                                with patch("memory.db.get_last_summarized_exchange_index", return_value=-1):
                                                    with patch("memory.db.get_latest_message_index", return_value=50):
                                                        with patch("memory.db.get_rich_summaries", return_value=_STANDARD_SUMMARIES):
                                                            with patch("memory.db.get_batch_status", return_value="processing"):
                                                                with patch("memory.sync._update_batch_status"):
                                                                    sync_exchange("session_rate")

        assert mock_sota.extract_and_summarize_from_summaries.call_count == 1

    # ------------------------------------------------------------------
    # 14. Reasoning token logging → extraction called
    # ------------------------------------------------------------------
    def test_reasoning_token_logging(self):
        with patch("memory.sync._get_sota") as mock_get_sota:
            mock_sota = MagicMock()
            mock_get_sota.return_value = mock_sota
            mock_sota.extract_and_summarize_from_summaries.return_value = {"entities": [], "facts": []}

            with patch("memory.sync._summarize_uncached_exchanges", return_value={"stored": 0, "found_uncached": False}):
                with patch("memory.db.get_max_end_exchange_index", return_value=-1):
                    with patch("memory.db.get_max_exchange_index", return_value=5):
                        mock_conn = MagicMock()
                        mock_conn.execute.return_value.fetchone.return_value = {"start_idx": 0, "end_idx": 5}

                        @contextmanager
                        def mock_connection_ctx(*args, **kwargs):
                            yield mock_conn

                        with patch("memory.db_pool.connection", side_effect=mock_connection_ctx):
                            with patch("memory.db.get_messages_in_range", return_value=[{"role": "user", "content": "test"}]):
                                with patch("memory.db.get_extraction_batch", return_value=None):
                                    with patch("memory.db.record_extraction_batch", return_value="batch_reasoning"):
                                        with patch("memory.batch_manager._set_batch_processing", return_value=(True, False)):
                                            with patch("memory.db.get_summary_age", return_value=None):
                                                with patch("memory.db.get_last_summarized_exchange_index", return_value=-1):
                                                    with patch("memory.db.get_latest_message_index", return_value=5):
                                                        with patch("memory.db.get_rich_summaries", return_value=_STANDARD_SUMMARIES):
                                                            with patch("memory.db.get_batch_status", return_value="processing"):
                                                                with patch("memory.sync._update_batch_status"):
                                                                    sync_exchange("session_reasoning")

        mock_sota.extract_and_summarize_from_summaries.assert_called_once()

    # ------------------------------------------------------------------
    # 15. Prompt truncation logic → extraction called with long messages
    # ------------------------------------------------------------------
    def test_prompt_truncation_logic(self):
        with patch("memory.sync._get_sota") as mock_get_sota:
            mock_sota = MagicMock()
            mock_get_sota.return_value = mock_sota
            mock_sota.extract_and_summarize_from_summaries.return_value = {"entities": [], "facts": []}

            long_content = "x" * 50000

            with patch("memory.sync._summarize_uncached_exchanges", return_value={"stored": 0, "found_uncached": False}):
                with patch("memory.db.get_max_end_exchange_index", return_value=-1):
                    with patch("memory.db.get_max_exchange_index", return_value=5):
                        mock_conn = MagicMock()
                        mock_conn.execute.return_value.fetchone.return_value = {"start_idx": 0, "end_idx": 5}

                        @contextmanager
                        def mock_connection_ctx(*args, **kwargs):
                            yield mock_conn

                        with patch("memory.db_pool.connection", side_effect=mock_connection_ctx):
                            with patch("memory.db.get_messages_in_range", return_value=[{"role": "user", "content": long_content}]):
                                with patch("memory.db.get_extraction_batch", return_value=None):
                                    with patch("memory.db.record_extraction_batch", return_value="batch_truncate"):
                                        with patch("memory.batch_manager._set_batch_processing", return_value=(True, False)):
                                            with patch("memory.db.get_summary_age", return_value=None):
                                                with patch("memory.db.get_last_summarized_exchange_index", return_value=-1):
                                                    with patch("memory.db.get_latest_message_index", return_value=5):
                                                        with patch("memory.db.get_rich_summaries", return_value=_STANDARD_SUMMARIES):
                                                            with patch("memory.db.get_batch_status", return_value="processing"):
                                                                with patch("memory.sync._update_batch_status"):
                                                                    sync_exchange("session_truncate")

        mock_sota.extract_and_summarize_from_summaries.assert_called_once()

    # ------------------------------------------------------------------
    # 16. Batch status recheck → existing batch claimed by concurrent sync
    # ------------------------------------------------------------------
    def test_batch_status_recheck(self):
        with patch("memory.sync._get_sota") as mock_get_sota:
            mock_sota = MagicMock()
            mock_get_sota.return_value = mock_sota

            with patch("memory.sync._summarize_uncached_exchanges", return_value={"stored": 0, "found_uncached": False}):
                with patch("memory.db.get_max_end_exchange_index", return_value=-1):
                    with patch("memory.db.get_max_exchange_index", return_value=5):
                        mock_conn = MagicMock()
                        mock_conn.execute.return_value.fetchone.return_value = {"start_idx": 0, "end_idx": 5}

                        @contextmanager
                        def mock_connection_ctx(*args, **kwargs):
                            yield mock_conn

                        with patch("memory.db_pool.connection", side_effect=mock_connection_ctx):
                            with patch("memory.db.get_messages_in_range", return_value=[{"role": "user", "content": "test"}]):
                                with patch("memory.db.get_extraction_batch") as mock_get_batch:
                                    mock_get_batch.return_value = {
                                        "id": "batch_recheck",
                                        "status": "processing",
                                        "start_message_index": 0,
                                        "end_message_index": 5,
                                        "start_exchange_index": 0,
                                        "end_exchange_index": 5,
                                        "session_id": "session_recheck",
                                    }
                                    with patch("memory.db.get_batch_age", return_value=0):
                                        with patch("memory.batch_manager._set_batch_processing", return_value=(False, False)):
                                            sync_exchange("session_recheck")

        mock_sota.extract_and_summarize_from_summaries.assert_not_called()

    # ------------------------------------------------------------------
    # 17. NEW: Extraction skipped when < 3 meaningful summaries
    # ------------------------------------------------------------------
    def test_extraction_skipped_when_few_summaries(self):
        with patch("memory.sync._get_sota") as mock_get_sota:
            mock_sota = MagicMock()
            mock_get_sota.return_value = mock_sota

            with patch("memory.sync._summarize_uncached_exchanges", return_value={"stored": 0, "found_uncached": False}):
                with patch("memory.db.get_max_end_exchange_index", return_value=-1):
                    with patch("memory.db.get_max_exchange_index", return_value=5):
                        mock_conn = MagicMock()
                        mock_conn.execute.return_value.fetchone.return_value = {"start_idx": 0, "end_idx": 5}

                        @contextmanager
                        def mock_connection_ctx(*args, **kwargs):
                            yield mock_conn

                        with patch("memory.db_pool.connection", side_effect=mock_connection_ctx):
                            with patch("memory.db.get_messages_in_range", return_value=[{"role": "user", "content": "test"}]):
                                with patch("memory.db.get_extraction_batch", return_value=None):
                                    with patch("memory.db.record_extraction_batch", return_value="batch_few"):
                                        with patch("memory.batch_manager._set_batch_processing", return_value=(True, False)):
                                            with patch("memory.db.get_summary_age", return_value=None):
                                                with patch("memory.db.get_last_summarized_exchange_index", return_value=-1):
                                                    with patch("memory.db.get_latest_message_index", return_value=5):
                                                        few_summaries = [
                                                            {"message_index": 1, "exchange_index": 1, "summary": "First.", "rendered_narrative_tokens": 50},
                                                            {"message_index": 2, "exchange_index": 2, "summary": "Second.", "rendered_narrative_tokens": 50},
                                                        ]
                                                        with patch("memory.db.get_rich_summaries", return_value=few_summaries):
                                                            with patch("memory.sync._update_batch_status") as mock_status:
                                                                sync_exchange("session_few")

        mock_status.assert_called_with("batch_few", "skipped")
        mock_sota.extract_and_summarize_from_summaries.assert_not_called()

    # ------------------------------------------------------------------
    # 18. NEW: Extraction skipped when all summaries are trivial
    # ------------------------------------------------------------------
    def test_extraction_skipped_when_trivial_summaries(self):
        with patch("memory.sync._get_sota") as mock_get_sota:
            mock_sota = MagicMock()
            mock_get_sota.return_value = mock_sota

            with patch("memory.sync._summarize_uncached_exchanges", return_value={"stored": 0, "found_uncached": False}):
                with patch("memory.db.get_max_end_exchange_index", return_value=-1):
                    with patch("memory.db.get_max_exchange_index", return_value=5):
                        mock_conn = MagicMock()
                        mock_conn.execute.return_value.fetchone.return_value = {"start_idx": 0, "end_idx": 5}

                        @contextmanager
                        def mock_connection_ctx(*args, **kwargs):
                            yield mock_conn

                        with patch("memory.db_pool.connection", side_effect=mock_connection_ctx):
                            with patch("memory.db.get_messages_in_range", return_value=[{"role": "user", "content": "test"}]):
                                with patch("memory.db.get_extraction_batch", return_value=None):
                                    with patch("memory.db.record_extraction_batch", return_value="batch_trivial"):
                                        with patch("memory.batch_manager._set_batch_processing", return_value=(True, False)):
                                            with patch("memory.db.get_summary_age", return_value=None):
                                                with patch("memory.db.get_last_summarized_exchange_index", return_value=-1):
                                                    with patch("memory.db.get_latest_message_index", return_value=5):
                                                        trivial_summaries = [
                                                            {"message_index": 1, "exchange_index": 1, "summary": "(no response)", "rendered_narrative_tokens": 10},
                                                            {"message_index": 2, "exchange_index": 2, "summary": "(empty message)", "rendered_narrative_tokens": 10},
                                                            {"message_index": 3, "exchange_index": 3, "summary": "(parse error)", "rendered_narrative_tokens": 10},
                                                            {"message_index": 4, "exchange_index": 4, "summary": "   ", "rendered_narrative_tokens": 10},
                                                        ]
                                                        with patch("memory.db.get_rich_summaries", return_value=trivial_summaries):
                                                            with patch("memory.sync._update_batch_status") as mock_status:
                                                                sync_exchange("session_trivial")

        mock_status.assert_called_with("batch_trivial", "skipped")
        mock_sota.extract_and_summarize_from_summaries.assert_not_called()
