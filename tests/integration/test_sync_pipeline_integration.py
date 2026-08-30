import pytest
from contextlib import contextmanager
from unittest.mock import MagicMock, patch
from memory.sync import sync_exchange, _update_batch_status
from memory.sota_client import SOTATimeoutError, SOTAAPIError, SOTAParseError, SyncResult


def _mock_connection_factory(conn):
    @contextmanager
    def _ctx(*args, **kwargs):
        yield conn
    return _ctx


class TestSyncPipelineIntegration:
    @pytest.fixture(autouse=True)
    def setup_test_db(self, tmp_path, monkeypatch):
        db_path = str(tmp_path / "memory.db")
        monkeypatch.setenv("PROXY_TEST_DB", db_path)
        from memory.db import init_db
        init_db(db_path)
        return db_path

    @patch("memory.db_pool.connection")
    @patch("memory.db.get_max_end_exchange_index", return_value=-1)
    @patch("memory.db.get_max_exchange_index", return_value=2)
    @patch("memory.db.get_messages_in_range")
    @patch("memory.db.get_extraction_batch")
    @patch("memory.sync._update_batch_extraction")
    @patch("memory.sync._update_batch_status")
    @patch("memory.sync._get_sota")
    @patch("memory.sync._get_graph")
    @patch("memory.db.record_extraction_batch")
    def test_sync_exchange_records_batch(
        self,
        mock_record,
        mock_graph,
        mock_sota,
        mock_status,
        mock_extraction,
        mock_get_batch,
        mock_get_messages_in_range,
        mock_get_max_exchange_index,
        mock_get_max_end_exchange_index,
        mock_connection,
    ):
        mock_conn = MagicMock()
        mock_conn.execute.return_value.fetchone.return_value = {"start_idx": 0, "end_idx": 1}
        mock_connection.side_effect = _mock_connection_factory(mock_conn)
        mock_get_messages_in_range.return_value = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there"},
        ]
        mock_get_batch.return_value = None
        mock_record.return_value = "batch_123"
        mock_graph_instance = MagicMock()
        mock_graph.return_value = mock_graph_instance
        mock_graph_instance.store_memory.return_value = True

        mock_sota_instance = MagicMock()
        mock_sota.return_value = mock_sota_instance

        sync_exchange("session123")

        mock_record.assert_called_once()

    @patch("memory.db.get_rich_summaries")
    @patch("memory.db.get_latest_message_index", return_value=5)
    @patch("memory.db.get_last_summarized_exchange_index", return_value=-1)
    @patch("memory.db.get_summary_age", return_value=None)
    @patch("memory.db.get_summary", return_value="")
    @patch("memory.batch_manager._set_batch_processing", return_value=(True, False))
    @patch("memory.db_pool.connection")
    @patch("memory.db.get_max_end_exchange_index", return_value=-1)
    @patch("memory.db.get_max_exchange_index", return_value=1)
    @patch("memory.db.get_messages_in_range", return_value=[{"role": "user", "content": "test message"}])
    @patch("memory.db.get_extraction_batch")
    @patch("memory.sync._update_batch_extraction")
    @patch("memory.sync._update_batch_status")
    @patch("memory.sync._get_sota")
    @patch("memory.sync._get_graph")
    @patch("memory.db.record_extraction_batch")
    def test_sync_exchange_calls_sota_extraction(
        self, mock_record, mock_graph, mock_sota, mock_extraction, mock_status,
        mock_get_batch, mock_get_messages_in_range, mock_get_max_exchange_index,
        mock_get_max_end_exchange_index, mock_connection, mock_set_processing,
        mock_get_summary, mock_get_summary_age, mock_get_last_summarized,
        mock_get_latest, mock_get_rich_summaries,
    ):
        mock_conn = MagicMock()
        mock_conn.execute.return_value.fetchone.return_value = {"start_idx": 0, "end_idx": 1}
        mock_connection.side_effect = _mock_connection_factory(mock_conn)
        mock_get_batch.return_value = None
        mock_record.return_value = "batch_456"
        mock_graph_instance = MagicMock()
        mock_graph.return_value = mock_graph_instance
        mock_graph_instance.store_memory.return_value = True

        mock_sota_instance = MagicMock()
        mock_sota.return_value = mock_sota_instance
        mock_sota_instance.extract_and_summarize_from_summaries.return_value = {
            "extraction_timestamp": "2025-01-01T00:00:00Z",
            "entities": [],
            "facts": [],
        }

        mock_get_rich_summaries.return_value = [
            {"summary": "User said hello", "role": "user", "message_index": 1, "exchange_index": 1},
            {"summary": "Assistant responded", "role": "assistant", "message_index": 2, "exchange_index": 2},
            {"summary": "User asked about Python", "role": "user", "message_index": 3, "exchange_index": 3},
        ]

        sync_exchange("session456")

        mock_sota_instance.extract_and_summarize_from_summaries.assert_called_once()

    @patch("memory.db.get_rich_summaries")
    @patch("memory.db.get_latest_message_index", return_value=5)
    @patch("memory.db.get_last_summarized_exchange_index", return_value=-1)
    @patch("memory.db.get_summary_age", return_value=None)
    @patch("memory.db.get_summary", return_value="")
    @patch("memory.batch_manager._set_batch_processing", return_value=(True, False))
    @patch("memory.db_pool.connection")
    @patch("memory.db.get_max_end_exchange_index", return_value=-1)
    @patch("memory.db.get_max_exchange_index", return_value=1)
    @patch("memory.db.get_messages_in_range", return_value=[{"role": "user", "content": "info"}])
    @patch("memory.db.get_extraction_batch")
    @patch("memory.sync._update_batch_extraction")
    @patch("memory.sync._claim_batch_for_extraction", return_value=True)
    @patch("memory.sync._update_batch_status_if", return_value=True)
    @patch("memory.sync._update_batch_status")
    @patch("memory.sync._get_graph")
    @patch("memory.sync._get_sota")
    @patch("memory.db.record_extraction_batch")
    def test_sync_exchange_stores_to_neo4j(
        self,
        mock_record,
        mock_sota,
        mock_graph,
        mock_status,
        mock_status_if,
        mock_claim,
        mock_extraction,
        mock_get_batch,
        mock_get_messages_in_range,
        mock_get_max_exchange_index,
        mock_get_max_end_exchange_index,
        mock_connection,
        mock_set_processing,
        mock_get_summary,
        mock_get_summary_age,
        mock_get_last_summarized,
        mock_get_latest,
        mock_get_rich_summaries,
    ):
        mock_conn = MagicMock()
        mock_conn.execute.return_value.fetchone.return_value = {"start_idx": 0, "end_idx": 1}
        mock_connection.side_effect = _mock_connection_factory(mock_conn)
        mock_get_batch.return_value = None
        mock_record.return_value = "batch_789"
        mock_graph_instance = MagicMock()
        mock_graph.return_value = mock_graph_instance
        mock_graph_instance.store_memory.return_value = True

        mock_sota_instance = MagicMock()
        mock_sota.return_value = mock_sota_instance
        mock_sota_instance.extract_and_summarize_from_summaries.return_value = {
            "extraction_timestamp": "2025-01-01T00:00:00Z",
            "entities": [{"name": "Python", "type": "Technology"}],
            "facts": [{"content": "Project uses Python", "topic_key": "project/stack"}],
        }

        mock_get_rich_summaries.return_value = [
            {"summary": "User info provided", "role": "user", "message_index": 1, "exchange_index": 1},
            {"summary": "Assistant acknowledged", "role": "assistant", "message_index": 2, "exchange_index": 2},
            {"summary": "User shared context", "role": "user", "message_index": 3, "exchange_index": 3},
        ]

        sync_exchange("session789")

        mock_status_if.assert_called()
        mock_extraction.assert_called()

    @patch("memory.batch_manager._set_batch_processing", return_value=(True, False))
    @patch("memory.db_pool.connection")
    @patch("memory.db.get_max_end_exchange_index", return_value=-1)
    @patch("memory.db.get_max_exchange_index", return_value=1)
    @patch("memory.db.get_messages_in_range", return_value=[{"role": "user", "content": "test"}])
    @patch("memory.db.get_extraction_batch")
    @patch("memory.sync._update_batch_status")
    @patch("memory.sync._get_sota")
    @patch("memory.db.record_extraction_batch")
    @patch("memory.sync._run_extraction_phase")
    def test_sync_exchange_handles_sota_timeout(
        self, mock_run_extraction, mock_record, mock_sota, mock_status, mock_get_batch,
        mock_get_messages_in_range, mock_get_max_exchange_index, mock_get_max_end_exchange_index,
        mock_connection, mock_set_processing, caplog,
    ):
        import logging
        caplog.set_level(logging.WARNING, logger="memory.sync")
        mock_conn = MagicMock()
        mock_conn.execute.return_value.fetchone.return_value = {"start_idx": 0, "end_idx": 1}
        mock_connection.side_effect = _mock_connection_factory(mock_conn)
        mock_get_batch.return_value = None
        mock_record.return_value = "batch_timeout"
        mock_sota_instance = MagicMock()
        mock_sota.return_value = mock_sota_instance
        mock_run_extraction.side_effect = SOTATimeoutError("Timeout")

        result = sync_exchange("session_timeout")

        assert result is None
        assert mock_status.called
        assert mock_status.call_args[0][0] == "batch_timeout"
        assert mock_status.call_args[0][1] == "failed"
        assert mock_status.call_args.kwargs.get("error") == "timeout"
        assert any("timeout" in r.message.lower() for r in caplog.records)

    @patch("memory.batch_manager._set_batch_processing", return_value=(True, False))
    @patch("memory.db_pool.connection")
    @patch("memory.db.get_max_end_exchange_index", return_value=-1)
    @patch("memory.db.get_max_exchange_index", return_value=1)
    @patch("memory.db.get_messages_in_range", return_value=[{"role": "user", "content": "test"}])
    @patch("memory.db.get_extraction_batch")
    @patch("memory.sync._update_batch_status")
    @patch("memory.sync._get_sota")
    @patch("memory.db.record_extraction_batch")
    @patch("memory.sync._run_extraction_phase")
    def test_sync_exchange_handles_sota_api_error(
        self, mock_run_extraction, mock_record, mock_sota, mock_status, mock_get_batch,
        mock_get_messages_in_range, mock_get_max_exchange_index, mock_get_max_end_exchange_index,
        mock_connection, mock_set_processing, caplog,
    ):
        import logging
        caplog.set_level(logging.WARNING, logger="memory.sync")
        mock_conn = MagicMock()
        mock_conn.execute.return_value.fetchone.return_value = {"start_idx": 0, "end_idx": 1}
        mock_connection.side_effect = _mock_connection_factory(mock_conn)
        mock_get_batch.return_value = None
        mock_record.return_value = "batch_error"
        mock_sota_instance = MagicMock()
        mock_sota.return_value = mock_sota_instance
        mock_run_extraction.side_effect = SOTAAPIError("API Error")

        result = sync_exchange("session_error")

        assert result is None
        assert mock_status.called
        assert mock_status.call_args[0][0] == "batch_error"
        assert mock_status.call_args[0][1] == "failed"
        assert mock_status.call_args.kwargs.get("error") == "API Error"
        assert any("api error" in r.message.lower() for r in caplog.records)

    @patch("memory.batch_manager._set_batch_processing", return_value=(True, False))
    @patch("memory.db_pool.connection")
    @patch("memory.db.get_max_end_exchange_index", return_value=-1)
    @patch("memory.db.get_max_exchange_index", return_value=1)
    @patch("memory.db.get_messages_in_range", return_value=[{"role": "user", "content": "test"}])
    @patch("memory.db.get_extraction_batch")
    @patch("memory.sync._update_batch_status")
    @patch("memory.sync._get_sota")
    @patch("memory.db.record_extraction_batch")
    @patch("memory.sync._run_extraction_phase")
    def test_sync_exchange_handles_parse_error(
        self, mock_run_extraction, mock_record, mock_sota, mock_status, mock_get_batch,
        mock_get_messages_in_range, mock_get_max_exchange_index, mock_get_max_end_exchange_index,
        mock_connection, mock_set_processing, caplog,
    ):
        import logging
        caplog.set_level(logging.WARNING, logger="memory.sync")
        mock_conn = MagicMock()
        mock_conn.execute.return_value.fetchone.return_value = {"start_idx": 0, "end_idx": 1}
        mock_connection.side_effect = _mock_connection_factory(mock_conn)
        mock_get_batch.return_value = None
        mock_record.return_value = "batch_parse"
        mock_sota_instance = MagicMock()
        mock_sota.return_value = mock_sota_instance
        mock_run_extraction.side_effect = SOTAParseError("Parse error")

        result = sync_exchange("session_parse")

        assert result is None
        assert mock_status.called
        assert mock_status.call_args[0][0] == "batch_parse"
        assert mock_status.call_args[0][1] == "failed"
        assert mock_status.call_args.kwargs.get("error") == "parse_error"
        assert any("parse_error" in r.message.lower() or "parse error" in r.message.lower() for r in caplog.records)

    @patch("memory.sync._get_sota")
    @patch("memory.db_pool.connection")
    @patch("memory.db.get_max_end_exchange_index", return_value=-1)
    @patch("memory.db.get_max_exchange_index", return_value=1)
    @patch("memory.db.get_messages_in_range", return_value=[{"role": "user", "content": "test"}])
    @patch("memory.db.get_extraction_batch")
    def test_sync_exchange_skips_completed_batch(self, mock_get_batch, mock_get_messages_in_range, mock_get_max_exchange_index, mock_get_max_end_exchange_index, mock_connection, mock_sota):
        mock_conn = MagicMock()
        mock_conn.execute.return_value.fetchone.return_value = {"start_idx": 0, "end_idx": 1}
        mock_connection.side_effect = _mock_connection_factory(mock_conn)
        mock_get_batch.return_value = {"id": "batch_done", "status": "completed"}

        sync_exchange("session_done")

        mock_sota.return_value.sync.assert_not_called()

    @patch("memory.db.get_rich_summaries")
    @patch("memory.db.get_latest_message_index", return_value=5)
    @patch("memory.db.get_last_summarized_exchange_index", return_value=-1)
    @patch("memory.db.get_summary_age", return_value=None)
    @patch("memory.db.get_summary", return_value="")
    @patch("memory.batch_manager._set_batch_processing", return_value=(True, False))
    @patch("memory.db_pool.connection")
    @patch("memory.db.get_max_end_exchange_index", return_value=-1)
    @patch("memory.db.get_max_exchange_index", return_value=1)
    @patch("memory.db.get_messages_in_range", return_value=[{"role": "user", "content": "test"}])
    @patch("memory.db.get_extraction_batch")
    @patch("memory.sync._update_batch_extraction")
    @patch("memory.sync._update_batch_status")
    @patch("memory.sync._get_sota")
    @patch("memory.sync._get_graph")
    @patch("memory.db.record_extraction_batch")
    def test_sync_exchange_generates_summary(
        self,
        mock_record,
        mock_graph,
        mock_sota,
        mock_extraction,
        mock_status,
        mock_get_batch,
        mock_get_messages_in_range,
        mock_get_max_exchange_index,
        mock_get_max_end_exchange_index,
        mock_connection,
        mock_set_processing,
        mock_get_summary,
        mock_get_summary_age,
        mock_get_last_summarized,
        mock_get_latest,
        mock_get_rich_summaries,
    ):
        mock_conn = MagicMock()
        mock_conn.execute.return_value.fetchone.return_value = {"start_idx": 0, "end_idx": 1}
        mock_connection.side_effect = _mock_connection_factory(mock_conn)
        mock_get_batch.return_value = None
        mock_record.return_value = "batch_summary"
        mock_graph_instance = MagicMock()
        mock_graph.return_value = mock_graph_instance
        mock_graph_instance.store_memory.return_value = True

        mock_sota_instance = MagicMock()
        mock_sota.return_value = mock_sota_instance
        mock_sota_instance.extract_and_summarize_from_summaries.return_value = {
            "extraction_timestamp": "2025-01-01T00:00:00Z",
            "entities": [],
            "facts": [],
        }

        mock_get_rich_summaries.return_value = [
            {"summary": "Session test summary", "role": "user", "message_index": 1, "exchange_index": 1},
            {"summary": "Session test response", "role": "assistant", "message_index": 2, "exchange_index": 2},
            {"summary": "Session test followup", "role": "user", "message_index": 3, "exchange_index": 3},
        ]

        sync_exchange("session_summary")

        mock_sota_instance.extract_and_summarize_from_summaries.assert_called_once()

    @patch("memory.db.get_rich_summaries")
    @patch("memory.db.get_latest_message_index", return_value=5)
    @patch("memory.db.get_last_summarized_exchange_index", return_value=3)
    @patch("memory.db.get_summary_age", return_value=None)
    @patch("memory.db.get_summary", return_value="")
    @patch("memory.batch_manager._set_batch_processing", return_value=(True, True))
    @patch("memory.db_pool.connection")
    @patch("memory.db.get_max_end_exchange_index", return_value=-1)
    @patch("memory.db.get_max_exchange_index", return_value=4)
    @patch("memory.db.get_messages_in_range", return_value=[{"role": "user", "content": "test"}])
    @patch("memory.db.get_extraction_batch")
    @patch("memory.sync._update_batch_extraction")
    @patch("memory.sync._update_batch_status")
    @patch("memory.sync._get_sota")
    @patch("memory.sync._get_graph")
    @patch("memory.db.record_extraction_batch")
    def test_sync_exchange_tombstones_second_failure(
        self,
        mock_record,
        mock_graph,
        mock_sota,
        mock_status,
        mock_extraction,
        mock_get_batch,
        mock_get_messages_in_range,
        mock_get_max_exchange_index,
        mock_get_max_end_exchange_index,
        mock_connection,
        mock_set_processing,
        mock_get_summary,
        mock_get_summary_age,
        mock_get_last_summarized,
        mock_get_latest,
        mock_get_rich_summaries,
    ):
        """When a previously-failed batch fails extraction again, it should be tombstoned as 'skipped'."""
        mock_conn = MagicMock()
        mock_conn.execute.return_value.fetchone.return_value = {"start_idx": 0, "end_idx": 4}
        mock_connection.side_effect = _mock_connection_factory(mock_conn)
        # Pre-existing batch in 'failed' status — was_previously_failed will be True
        mock_get_batch.return_value = {"id": "batch_xyz", "status": "failed"}
        mock_graph_instance = MagicMock()
        mock_graph.return_value = mock_graph_instance

        mock_sota_instance = MagicMock()
        mock_sota.return_value = mock_sota_instance
        # Extraction fails (second failure)
        mock_sota_instance.extract_and_summarize_from_summaries.side_effect = SOTAParseError("Parse error")

        mock_get_rich_summaries.return_value = [
            {"summary": "User said hello", "role": "user", "message_index": 1, "exchange_index": 1},
            {"summary": "Assistant responded", "role": "assistant", "message_index": 2, "exchange_index": 2},
            {"summary": "User asked about Python", "role": "user", "message_index": 3, "exchange_index": 3},
        ]

        sync_exchange("session_tombstone")

        # Verify tombstone: status should be 'skipped', NOT 'failed'
        mock_status.assert_called_once()
        call_args = mock_status.call_args[0]
        assert call_args[0] == "batch_xyz"
        assert call_args[1] == "skipped", f"Expected 'skipped' tombstone, got '{call_args[1]}'"
