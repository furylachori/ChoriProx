"""US-F2: Previous-context index space — _build_previous_context receives message_index, not exchange_index."""
import time

import pytest


pytestmark = pytest.mark.integration


@pytest.fixture(autouse=True)
def _isolate_db(tmp_path, monkeypatch):
    """Per-test fresh DB."""
    db_path = str(tmp_path / "memory.db")
    monkeypatch.setenv("PROXY_TEST_DB", db_path)
    from memory.db import init_db
    init_db(db_path)
    yield db_path


def _now():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _seed_user_assistant(session_id, exchange_idx, msg_idx, user_content, assistant_content):
    """Insert one user + one assistant row at consecutive message indices."""
    now = _now()
    from memory.db_pool import connection
    with connection() as conn:
        conn.execute(
            "INSERT INTO messages (session_id, message_index, role, content, "
            "exchange_index, content_hash, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (session_id, msg_idx, "user", user_content, exchange_idx,
             f"hash-ex{exchange_idx}-u", now),
        )
        conn.execute(
            "INSERT INTO messages (session_id, message_index, role, content, "
            "exchange_index, content_hash, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (session_id, msg_idx + 1, "assistant", assistant_content, exchange_idx,
             f"hash-ex{exchange_idx}-a", now),
        )
        conn.commit()


class _RecordingSOTA:
    """Mock SOTA client that captures every summarize_message call."""

    def __init__(self):
        self.calls = []

    def summarize_message(self, *args, **kwargs):
        self.calls.append({"args": args, "kwargs": dict(kwargs)})
        return {
            "summary": f"summary for exchange {args[1]}",
            "tool_summary": "",
            "token_estimate": 10,
        }

    model = "test-sota-model"


def _call_summarize(session_id, sota, start_exchange=0, end_exchange=2):
    from memory.sync import _summarize_uncached_exchanges
    return _summarize_uncached_exchanges(session_id, sota, start_exchange, end_exchange)


class TestPreviousContextIndexSpace:
    """The _build_previous_context argument must be a message_index, not an exchange_index."""

    def test_previous_context_includes_preceding_exchange(self):
        """Seed 3 exchanges (6 messages); assert the [Previous Context] block for the
        LAST exchange (exchange_index=2, message_index=4) contains the assistant content
        from exchange 1 (message_index=3). With the bug (exchange_idx=2 passed), only
        messages 0,1 (exchange 0) would be included and exchange 1 would be missing.
        """
        from memory.sync import _summarize_uncached_exchanges

        session_id = "test-prev-context-fix"

        _seed_user_assistant(
            session_id, exchange_idx=0, msg_idx=0,
            user_content="user-zero-content sentinel ZERO_USER_0",
            assistant_content="assistant-zero sentinel ZERO_ASST_0",
        )
        _seed_user_assistant(
            session_id, exchange_idx=1, msg_idx=2,
            user_content="user-one-content sentinel ONE_USER_1",
            assistant_content="assistant-one sentinel ONE_ASST_1",
        )
        _seed_user_assistant(
            session_id, exchange_idx=2, msg_idx=4,
            user_content="user-two-content sentinel TWO_USER_2",
            assistant_content="assistant-two sentinel TWO_ASST_2",
        )

        sota = _RecordingSOTA()
        _call_summarize(session_id, sota, start_exchange=0, end_exchange=2)

        assert len(sota.calls) == 3, (
            f"expected 3 summarize_message calls (one per uncached exchange); "
            f"got {len(sota.calls)}"
        )

        last_call = sota.calls[-1]
        last_exchange_idx = last_call["args"][1]
        assert last_exchange_idx == 2, (
            f"last call must target exchange_index=2; got {last_exchange_idx}"
        )

        previous_context = last_call["kwargs"].get("previous_messages_text", "")

        assert "ONE_ASST_1" in previous_context, (
            f"previous_messages_text for exchange 2 must include the assistant content "
            f"from exchange 1 (message_index=3). Bug signature: with exchange_idx=2 the "
            f"filter r['message_index'] < 2 only matches messages 0,1 (exchange 0), "
            f"omitting message 3.\nGot: {previous_context!r}"
        )

        assert "ZERO_ASST_0" in previous_context, (
            f"previous_messages_text for exchange 2 must include content from "
            f"exchange 0 too.\nGot: {previous_context!r}"
        )

        assert "TWO_ASST_2" not in previous_context, (
            f"previous_messages_text must NOT include the current exchange's own "
            f"assistant message (filter is strictly < target).\nGot: {previous_context!r}"
        )

    def test_exchange_index_unchanged_for_other_purposes(self):
        """The SOTA call's exchange_index positional argument must stay in EXCHANGE_INDEX
        space, not message_index space. Guards against accidentally swapping the wrong
        argument at line 453.
        """
        from memory.sync import _summarize_uncached_exchanges

        session_id = "test-prev-context-exchange-idx"

        _seed_user_assistant(
            session_id, exchange_idx=0, msg_idx=0,
            user_content="user-zero-content sentinel ZZ_USER_0",
            assistant_content="assistant-zero sentinel ZZ_ASST_0",
        )
        _seed_user_assistant(
            session_id, exchange_idx=1, msg_idx=2,
            user_content="user-one-content sentinel ZZ_USER_1",
            assistant_content="assistant-one sentinel ZZ_ASST_1",
        )
        _seed_user_assistant(
            session_id, exchange_idx=2, msg_idx=4,
            user_content="user-two-content sentinel ZZ_USER_2",
            assistant_content="assistant-two sentinel ZZ_ASST_2",
        )

        sota = _RecordingSOTA()
        _call_summarize(session_id, sota, start_exchange=0, end_exchange=2)

        assert len(sota.calls) == 3

        last_call = sota.calls[-1]
        actual_ex_idx = last_call["args"][1]

        assert actual_ex_idx == 2, (
            f"summarize_message positional arg 1 for the LAST exchange must be 2 "
            f"(exchange_index space), got {actual_ex_idx}. If accidentally swapped "
            f"with the previous-context argument, this would be 4 (message_index)."
        )

        assert actual_ex_idx != 4, (
            f"summarize_message positional arg 1 must NOT be 4 (message_index=4 for "
            f"exchange 2). The fix only changes the previous-context argument, not "
            f"the SOTA call's exchange_index."
        )
