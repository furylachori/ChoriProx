


def test_log_and_emit_infra_error_returns_correct_shape():
    """Helper must return the canonical 503 envelope and log at WARNING."""
    from unittest.mock import MagicMock
    from proxy.tool_pipeline.actions._errors import log_and_emit_infra_error

    logger = MagicMock()
    exc = RuntimeError("boom")
    result = log_and_emit_infra_error(logger, "proxy_ticket", "test_op", exc, "sess-1")

    assert result["ok"] is False
    assert result["tool"] == "proxy_ticket"
    assert result["operation"] == "test_op"
    assert result["status"] == "error"
    assert result["data"] is None
    assert result["session_id"] == "sess-1"
    assert result["error"]["code"] == "infrastructure_error"
    assert result["error"]["message"] == "Database temporarily unavailable"
    assert result["error"]["retryable"] is True
    assert result["error"]["category"] == "infrastructure"
    logger.warning.assert_called_once()
    args = logger.warning.call_args[0]
    assert "proxy_ticket" in args
    assert "test_op" in args
    assert "boom" in str(args)


def test_log_and_emit_infra_error_handles_empty_session_id():
    from unittest.mock import MagicMock
    from proxy.tool_pipeline.actions._errors import log_and_emit_infra_error

    logger = MagicMock()
    exc = RuntimeError("boom")
    result = log_and_emit_infra_error(logger, "proxy_ticket", "test_op", exc, "")

    assert result["session_id"] == ""


def test_infra_error_envelope_helper_returns_canonical_shape():
    """The non-logging helper also returns a canonical envelope."""
    from proxy.tool_pipeline.actions._errors import infra_error_envelope

    result = infra_error_envelope("proxy_ticket", "test_op", "sid")

    assert result["ok"] is False
    assert result["tool"] == "proxy_ticket"
    assert result["operation"] == "test_op"
    assert result["status"] == "error"
    assert result["data"] is None
    assert result["session_id"] == "sid"
    assert result["error"]["code"] == "infrastructure_error"
    assert result["error"]["message"] == "Database temporarily unavailable"
    assert result["error"]["retryable"] is True
    assert result["error"]["category"] == "infrastructure"


def test_append_review_note_catches_infrastructure_errors():
    """End-to-end: the migrated action must use the helper on
    InfrastructureError. Asserts the response shape, the log call,
    AND that reverting the migration back to inline breaks this test."""
    from unittest.mock import MagicMock, patch
    from proxy.tool_pipeline.actions import AppendReviewNoteAction
    from memory.query_executor import InfrastructureError

    tg = MagicMock()
    tg.append_review_note.side_effect = InfrastructureError("boom")

    with patch("proxy.tool_pipeline.actions.append_review_note.logger") as mock_logger:
        result = AppendReviewNoteAction.execute(
            arguments={"short_hash": "abc", "note": "test", "author": "me"},
            session_id="sess-1",
            tg=tg,
            workspace_id="ws-1",
            caller_agent_id="agent-1",
        )

    assert result["error"]["code"] == "infrastructure_error"
    assert result["status"] == "error"
    assert result["ok"] is False
    assert result["session_id"] == "sess-1"
    assert mock_logger.warning.called
