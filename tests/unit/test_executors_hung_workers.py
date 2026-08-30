"""CG-3: hung sync workers are visible (logged + tracked) after the 600s
_timeout_guard fires. The dict is bounded by the 4-worker pool; the
on_done callback removes entries when the worker eventually completes.
"""
import logging
import threading
import time
from unittest.mock import MagicMock

import pytest

from memory.executors import (
    _hung_futures,
    _hung_futures_lock,
    _on_done,
    _timeout_guard,
)


@pytest.fixture(autouse=True)
def _clear_hung_futures():
    """Clear the module-level tracking dict before/after each test."""
    _hung_futures.clear()
    yield
    _hung_futures.clear()


def _make_mock_future(timeout_result=False):
    """Build a mock future that either times out or completes."""
    mock_future = MagicMock()
    if timeout_result:
        mock_future.result.side_effect = TimeoutError("simulated timeout")
    else:
        mock_future.result.return_value = None
    return mock_future


class TestTimeoutGuard:
    def test_normal_completion_no_warning(self, caplog):
        """A worker that completes before the timeout emits no warning."""
        mock_future = _make_mock_future(timeout_result=False)
        with caplog.at_level(logging.WARNING, logger="memory.executors"):
            _timeout_guard(mock_future, sid="sess-normal", token=object(), timeout_sec=10)
        assert "Sync worker hung" not in caplog.text
        assert len(_hung_futures) == 0

    def test_hung_worker_logs_warning_and_tracked(self, caplog):
        """A worker that times out emits WARNING and is added to the tracking dict."""
        mock_future = _make_mock_future(timeout_result=True)
        with caplog.at_level(logging.WARNING, logger="memory.executors"):
            _timeout_guard(mock_future, sid="sess-hung", token=object(), timeout_sec=600)
        assert any(
            "Sync worker hung" in r.message and r.name == "memory.executors"
            for r in caplog.records
        ), f"Expected warning log, got: {[r.message for r in caplog.records]}"
        assert len(_hung_futures) == 1
        assert _hung_futures[id(mock_future)][0] == "sess-hung"

    def test_hung_worker_count_4_logs_error(self, caplog):
        """When all 4 workers are presumed hung, log ERROR."""
        mocks = []
        for i in range(3):
            mock = MagicMock()
            mock.result.side_effect = TimeoutError()
            mocks.append(mock)
            _timeout_guard(mock, sid=f"sess-pre-{i}", token=object(), timeout_sec=600)
        mock = MagicMock()
        mock.result.side_effect = TimeoutError()
        mocks.append(mock)
        with caplog.at_level(logging.WARNING, logger="memory.executors"):
            _timeout_guard(mock, sid="sess-4th", token=object(), timeout_sec=600)
        assert any(
            "All 4 sync workers presumed hung" in r.message and r.name == "memory.executors"
            for r in caplog.records
        ), f"Expected ERROR log, got: {[r.message for r in caplog.records]}"
        assert len(_hung_futures) == 4


class TestOnDoneClears:
    def test_on_done_removes_from_tracking(self):
        """When the worker eventually completes, _on_done removes its entry."""
        mock_future = _make_mock_future(timeout_result=True)
        _timeout_guard(mock_future, sid="sess-done", token=object(), timeout_sec=600)
        assert len(_hung_futures) == 1
        _on_done(mock_future, sid="sess-done", token=object())
        assert len(_hung_futures) == 0

    def test_on_done_no_op_for_untracked_future(self):
        """_on_done for a future that was never tracked is a no-op."""
        mock_future = _make_mock_future(timeout_result=False)
        _on_done(mock_future, sid="sess-clean", token=object())
        assert len(_hung_futures) == 0
