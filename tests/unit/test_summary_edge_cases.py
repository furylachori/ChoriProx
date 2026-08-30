"""Edge-case tests for summary and context assembly paths.

Covers:
  1.12 — get_rich_summaries() returns [] on DB error (silently)
  1.13 — _select_context_v2() with no summaries, no ancient history ("broken" state)
  1.14 — submit_sync() per-session locking
  1.15 — _cleanup_tool_pair_integrity() post-budget overflow (orphan tool results)
"""

import sqlite3
import threading
import time
from contextlib import ExitStack
from unittest.mock import MagicMock, patch

import pytest

from proxy.context_v2 import _select_context_v2, _cleanup_tool_pair_integrity


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _system(content: str) -> dict:
    return {"role": "system", "content": content}


def _user(content: str) -> dict:
    return {"role": "user", "content": content}


def _assistant(content: str) -> dict:
    return {"role": "assistant", "content": content}


def _assistant_with_tool_call(tc_id: str, name: str = "read_file") -> dict:
    return {
        "role": "assistant",
        "content": "",
        "tool_calls": [{"id": tc_id, "type": "function", "function": {"name": name, "arguments": "{}"}}],
    }


def _tool_result(tc_id: str, content: str = "result") -> dict:
    return {"role": "tool", "tool_call_id": tc_id, "content": content}


def _no_db_mocks(cfg):
    """Return a list of patches that suppress DB and VFS side-effects."""
    return [
        patch("proxy.context.get_context_config", return_value=cfg),
        patch("memory.db.get_ancient_history", return_value=""),
        patch("memory.db.get_rich_summaries", return_value=[]),
        patch("proxy.vfs._build_vfs_inherited_context", return_value=None),
        patch("proxy.vfs._build_vfs_context", return_value=None),
    ]


# ---------------------------------------------------------------------------
# 1.12 — get_rich_summaries() silent empty on DB error
# ---------------------------------------------------------------------------

class TestGetRichSummariesDBError:
    """get_rich_summaries must return [] and log the error when SQLite fails."""

    def test_returns_empty_on_operational_error(self, caplog):
        """Mock connection() to raise sqlite3.OperationalError; verify [] returned."""
        import logging
        from memory.db_summaries import get_rich_summaries

        mock_conn_ctx = MagicMock()
        mock_conn_ctx.__enter__ = MagicMock(side_effect=sqlite3.OperationalError("database is locked"))
        mock_conn_ctx.__exit__ = MagicMock(return_value=False)

        with patch("memory.db_pool.connection", return_value=mock_conn_ctx):
            with caplog.at_level(logging.ERROR, logger="memory.db"):
                result = get_rich_summaries("session-1.12")

        assert result == [], f"Expected [] on DB error, got {result}"
        assert any("Failed to get rich summaries" in r.message for r in caplog.records), (
            "Expected error log message about failed rich summaries"
        )


# ---------------------------------------------------------------------------
# 1.13 — No summaries available ("broken" state)
# ---------------------------------------------------------------------------

class TestNoSummariesBrokenState:
    """When summaries and ancient history are both empty, context_v2 must
    still produce a valid output with recent raw messages and system prompt.
    """

    @pytest.fixture
    def tight_budget_config(self):
        return {
            "max_context_tokens": 40_000,
            "tokens_per_char": 0.25,
            "safe_context_tokens": 10_000,
            "raw_exchange_count": 8,
        }

    def test_no_crash_and_raw_window_preserved(self, tight_budget_config):
        """150 user messages, no summaries, no ancient history.

        With safe=10000 and 150 msgs × 100 tok = 15000 > 6500 low-watermark,
        eviction fires and trims to ~65 messages. Must not crash and must
        preserve the system prompt + a trimmed raw block.
        """
        pad = "X" * 400  # 100 tokens each
        messages = [_system("You are a helpful assistant.")] + [
            _user(f"user {i} {pad}") for i in range(150)
        ]

        with ExitStack() as stack:
            for p in _no_db_mocks(tight_budget_config):
                stack.enter_context(p)
            result = _select_context_v2("sess-broken", messages, model_name="test")

        assert isinstance(result, list)
        assert len(result) > 0, "Must return at least some messages"

        # System prompt must be preserved
        system_msgs = [m for m in result if m.get("role") == "system"]
        assert len(system_msgs) >= 1, "System prompt must be present"

        # Eviction must have trimmed the raw block below the input size.
        non_system = [m for m in result if m.get("role") != "system"]
        assert len(non_system) < 150, (
            f"Eviction should trim below 150 raw messages, got {len(non_system)}"
        )
        assert len(non_system) > 0, "Must keep some raw messages"

    def test_no_summary_content_in_output(self, tight_budget_config):
        """Verify no summary/past-context blocks are injected when summaries are empty.

        150 messages force eviction; summaries are mocked [] so the summary
        block must not appear even though the eviction path runs.
        """
        pad = "X" * 400
        messages = [_system("sys")] + [_user(f"u {i} {pad}") for i in range(150)]

        with ExitStack() as stack:
            for p in _no_db_mocks(tight_budget_config):
                stack.enter_context(p)
            result = _select_context_v2("sess-no-summary", messages, model_name="test")

        summary_msgs = [
            m for m in result
            if m.get("role") == "system" and "PAST CONTEXT" in m.get("content", "")
        ]
        assert len(summary_msgs) == 0, "Should not have PAST CONTEXT when summaries are empty"


# ---------------------------------------------------------------------------
# 1.14 — submit_sync() per-session locking
# ---------------------------------------------------------------------------

class TestSubmitSyncPerSessionLocking:
    """submit_sync() must prevent concurrent sync for the same session_id."""

    def test_second_call_returns_none_while_first_running(self):
        """A second submit_sync() for the same session must return None."""
        from memory.executors import submit_sync, _session_locks, _session_locks_lock

        barrier = threading.Event()
        call_log = []

        def slow_fn(session_id):
            call_log.append("started")
            barrier.wait(timeout=5)
            call_log.append("finished")

        try:
            future1 = submit_sync(slow_fn, "session-lock-test")
            assert future1 is not None, "First submit should return a future"
            time.sleep(0.05)  # Let it start

            future2 = submit_sync(slow_fn, "session-lock-test")
            assert future2 is None, "Second submit for same session should return None"

            barrier.set()
            future1.result(timeout=5)
        finally:
            # Cleanup lock
            with _session_locks_lock:
                _session_locks.pop("session-lock-test", None)

    def test_subsequent_call_succeeds_after_first_completes(self):
        """After the first sync completes, a new submit_sync for the same session must succeed."""
        from memory.executors import submit_sync, _session_locks, _session_locks_lock

        call_log = []

        def fast_fn(session_id):
            call_log.append("executed")
            return "done"

        try:
            f1 = submit_sync(fast_fn, "session-after-test")
            assert f1 is not None
            f1.result(timeout=5)
            time.sleep(0.05)  # Let callback clean up lock

            f2 = submit_sync(fast_fn, "session-after-test")
            assert f2 is not None, "Second call after first completed should return a future"
            result = f2.result(timeout=5)
            assert result == "done"
            assert len(call_log) == 2
        finally:
            with _session_locks_lock:
                _session_locks.pop("session-after-test", None)

    def test_different_sessions_can_run_concurrently(self):
        """Two different sessions must not block each other."""
        from memory.executors import submit_sync, _session_locks, _session_locks_lock

        barrier = threading.Event()
        results = []

        def fn(session_id):
            barrier.wait(timeout=5)
            results.append(session_id)

        try:
            f1 = submit_sync(fn, "session-A")
            f2 = submit_sync(fn, "session-B")
            assert f1 is not None, "session-A should get a future"
            assert f2 is not None, "session-B should get a future"

            barrier.set()
            f1.result(timeout=5)
            f2.result(timeout=5)
            assert set(results) == {"session-A", "session-B"}
        finally:
            with _session_locks_lock:
                _session_locks.pop("session-A", None)
                _session_locks.pop("session-B", None)


# ---------------------------------------------------------------------------
# 1.15 — _cleanup_tool_pair_integrity post-budget overflow
# ---------------------------------------------------------------------------

class TestCleanupToolPairIntegrityPostBudget:
    """When tool_call messages exist but their tool results are missing (pruned
    by budget), _cleanup_tool_pair_integrity must inject synthetic placeholders.
    _select_context_v2 must not crash and must stay within budget.
    """

    @pytest.fixture
    def cfg(self):
        return {
            "max_context_tokens": 40_000,
            "tokens_per_char": 0.25,
            "safe_context_tokens": 10_000,
            "raw_exchange_count": 8,
        }

    def test_no_context_limit_error_with_orphan_tool_calls(self, cfg):
        """10 assistant tool_call messages whose results are NOT in raw window.
        The cleanup function must inject synthetic results. _select_context_v2
        must not raise ContextLimitError.
        """
        pad = "X" * 400  # 100 tokens each
        # Build messages: 10 tool-call exchanges, then a normal tail
        messages = [_system("sys")]
        for i in range(10):
            tc_id = f"call_{i}"
            messages.append(_assistant_with_tool_call(tc_id, f"tool_{i}"))
            messages.append(_user(f"follow-up {i} {pad}"))
        # Final user message
        messages.append(_user(f"last message {pad}"))

        patches = _no_db_mocks(cfg)
        with ExitStack() as stack:
            for p in patches:
                stack.enter_context(p)
            result = _select_context_v2("sess-tools", messages, model_name="test")

        assert isinstance(result, list)
        # Must not raise ContextLimitError

    def test_budget_stays_within_safe_tokens(self, cfg):
        """Verify the final context does not exceed safe_context_tokens."""
        pad = "X" * 400
        messages = [_system("sys")]
        for i in range(10):
            tc_id = f"call_{i}"
            messages.append(_assistant_with_tool_call(tc_id, f"tool_{i}"))
            messages.append(_user(f"msg {i} {pad}"))
        messages.append(_user(f"final {pad}"))

        patches = _no_db_mocks(cfg)
        with ExitStack() as stack:
            for p in patches:
                stack.enter_context(p)
            result = _select_context_v2("sess-budget", messages, model_name="test")

        total_chars = sum(len(m.get("content", "")) for m in result if isinstance(m.get("content"), str))
        # tool_calls data also contributes, so count those
        for m in result:
            for tc in (m.get("tool_calls") or []):
                total_chars += len(str(tc))
        estimated_tokens = int(total_chars * cfg["tokens_per_char"])
        tool_schema_tokens = 2000
        assert estimated_tokens + tool_schema_tokens <= cfg["safe_context_tokens"], (
            f"Estimated {estimated_tokens} + {tool_schema_tokens} tool_schema exceeds "
            f"safe budget {cfg['safe_context_tokens']}"
        )

    def test_synthetic_results_injected_for_missing_tool_results(self):
        """Standalone test: assistant with tool_calls but no matching tool result
        must get a synthetic placeholder injected.
        """
        messages = [
            _user("do something"),
            _assistant_with_tool_call("tc_missing"),
            _user("what happened?"),
        ]
        cleaned = _cleanup_tool_pair_integrity(messages)
        tool_msgs = [m for m in cleaned if m.get("role") == "tool"]
        assert len(tool_msgs) == 1, "Should have injected one synthetic tool result"
        assert tool_msgs[0]["tool_call_id"] == "tc_missing"
        assert "pruned" in tool_msgs[0]["content"].lower()
