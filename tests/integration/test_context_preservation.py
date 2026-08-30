"""Integration tests for context preservation, summarization, and continuation.

Exercises the real Flask shell + 9-stage pipeline via ``proxy_with_mock``:
- ``test_large_conversation_handling`` — proxy handles 20+ msg conversation
  end-to-end AND ``sync_exchange`` writes a row to ``conversation_summaries``
- ``test_summary_generation`` — ``sync_exchange`` after a long conversation
  writes rows to ``message_summaries`` (SOTA path triggered, returns canned
  summaries via ``mock_sota_client``)
- ``test_continuation_after_summary`` — after summarization, the next request
  still completes 200 AND the cursor in ``conversation_summaries`` advanced

The unit tests drive ``_select_context_v2`` directly:
- ``test_context_window_optimization`` — V2 chain-hash trims a 60-message
  conversation down when its tokens-per-char budget is exceeded
- ``test_oversized_single_message`` — V2 does not crash on a 2000-word single
  message; returns a valid list containing the system prompt
"""

import json

import pytest


pytestmark = pytest.mark.integration


def _patch_sota_with_string_summaries(monkeypatch):
    """Replace SOTAClient.summarize_message with a callable that returns
    real-string dicts so SQLite can bind them. The autouse ``mock_sota_client``
    fixture returns MagicMock objects that SQLite rejects with
    "Error binding parameter 4 - probably unsupported type".
    """
    from unittest.mock import MagicMock

    fake_client = MagicMock()
    fake_client.model = "fake-sota-model"

    def fake_summarize(session_id, exchange_index, raw_content, role,
                       tool_calls_json=None, previous_messages_text=""):
        return {
            "summary": f"summary for exchange {exchange_index}",
            "tool_summary": "",
            "token_estimate": 25,
        }

    fake_client.summarize_message = fake_summarize
    monkeypatch.setattr("memory.sync._get_sota", lambda: fake_client)


def _init_db_once():
    from memory.db import init_db

    init_db()


def _drive_turn(client, handler, messages, model="kimi"):
    handler._response_body = json.dumps(
        {
            "id": "chatcmpl-test",
            "object": "chat.completion",
            "created": 0,
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "ok"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12},
        }
    ).encode()
    return client.post(
        "/v1/chat/completions",
        json={"model": model, "messages": messages, "stream": False},
    )


def _session_id_for(messages, client_hint="127.0.0.1"):
    from proxy.model_registry import resolve_session_id

    return resolve_session_id({"messages": messages}, client_hint)


def _conversation_summaries_rows(session_id):
    from memory.db_pool import connection

    with connection(pooled=True) as conn:
        rows = conn.execute(
            "SELECT * FROM conversation_summaries WHERE session_id = ?",
            (session_id,),
        ).fetchall()
    return rows


def _message_summaries_count(session_id):
    from memory.db_pool import connection

    with connection(pooled=True) as conn:
        row = conn.execute(
            "SELECT COUNT(*) AS n FROM message_summaries WHERE session_id = ?",
            (session_id,),
        ).fetchone()
    return int(row["n"] or 0)


def _max_exchange_index(session_id):
    from memory.db_pool import connection

    with connection(pooled=True) as conn:
        row = conn.execute(
            "SELECT COALESCE(MAX(exchange_index), -1) AS max_ei "
            "FROM messages WHERE session_id = ?",
            (session_id,),
        ).fetchone()
    return int(row["max_ei"])


def test_large_conversation_handling(proxy_with_mock, monkeypatch):
    _init_db_once()
    _patch_sota_with_string_summaries(monkeypatch)
    client, handler, port = proxy_with_mock

    messages = [{"role": "system", "content": "You are a helpful assistant."}]
    for i in range(20):
        if i % 2 == 0:
            messages.append(
                {"role": "user", "content": f"Question {i} about topic {i % 5}."}
            )
        else:
            messages.append(
                {"role": "assistant", "content": f"Answer {i} for topic {i % 5}."}
            )
    messages.append({"role": "user", "content": "What were just discussing?"})

    resp = _drive_turn(client, handler, messages)
    assert resp.status_code == 200
    data = resp.get_json()
    assert data.get("choices"), f"No choices in response: {data}"
    content = data["choices"][0].get("message", {}).get("content", "")
    assert len(content) > 0

    session_id = _session_id_for(messages)
    from memory.sync import sync_exchange
    sync_exchange(session_id)

    max_ei = _max_exchange_index(session_id)
    assert max_ei >= 0, (
        f"Expected proxy to persist messages with exchange_index, "
        f"max exchange_index={max_ei} for session_id={session_id}"
    )

    rows = _conversation_summaries_rows(session_id)
    assert any(r["last_exchange_index"] >= 1 for r in rows), (
        f"Expected at least one row in conversation_summaries with "
        f"last_exchange_index >= 1, got {rows!r}"
    )


def test_summary_generation(proxy_with_mock, monkeypatch):
    _init_db_once()
    _patch_sota_with_string_summaries(monkeypatch)
    client, handler, port = proxy_with_mock

    messages = [{"role": "system", "content": "You are a helpful assistant."}]
    for i in range(10):
        messages.append(
            {"role": "user", "content": f"User message number {i}: " + "x" * 50}
        )
        messages.append({"role": "assistant", "content": f"Assistant reply {i}."})

    resp = _drive_turn(client, handler, messages)
    assert resp.status_code == 200

    session_id = _session_id_for(messages)
    from memory.sync import sync_exchange
    sync_exchange(session_id)

    rows = _conversation_summaries_rows(session_id)
    advanced = [r for r in rows if r["last_exchange_index"] >= 0]
    assert advanced, (
        f"Expected at least one conversation_summaries row with "
        f"last_exchange_index >= 0 after sync_exchange, got {rows!r}"
    )

    msg_summary_count = _message_summaries_count(session_id)
    assert msg_summary_count > 0, (
        f"Expected at least one row in message_summaries after sync_exchange, "
        f"got {msg_summary_count}"
    )


def test_continuation_after_summary(proxy_with_mock, monkeypatch):
    _init_db_once()
    _patch_sota_with_string_summaries(monkeypatch)
    client, handler, port = proxy_with_mock

    messages = [{"role": "system", "content": "You are a helpful assistant."}]
    for i in range(10):
        messages.append(
            {"role": "user", "content": f"Setup exchange {i}: " + "x" * 80}
        )
        messages.append({"role": "assistant", "content": f"Setup reply {i}."})

    resp = _drive_turn(client, handler, messages)
    assert resp.status_code == 200

    session_id = _session_id_for(messages)
    from memory.sync import sync_exchange
    sync_exchange(session_id)

    rows_after = _conversation_summaries_rows(session_id)
    advanced = [r for r in rows_after if r["last_exchange_index"] >= 0]
    assert advanced, (
        f"Pre-condition: sync_exchange should advance the cursor, got "
        f"{rows_after!r}"
    )
    cursor_after_first_sync = max(r["last_exchange_index"] for r in advanced)
    assert cursor_after_first_sync > 0, (
        f"Pre-condition: first sync_exchange must advance cursor past 0 "
        f"to make the continuation >= comparison meaningful, got "
        f"cursor_after_first_sync={cursor_after_first_sync}"
    )

    messages.append(
        {"role": "user", "content": "Continue working on the last topic we discussed"}
    )
    resp2 = _drive_turn(client, handler, messages)
    assert resp2.status_code == 200
    data2 = resp2.get_json()
    assert data2.get("choices"), f"No choices in continuation response: {data2}"
    content2 = data2["choices"][0].get("message", {}).get("content", "")
    tool_calls2 = data2["choices"][0].get("message", {}).get("tool_calls", [])
    assert len(content2) > 0 or len(tool_calls2) > 0

    sync_exchange(session_id)
    rows_final = _conversation_summaries_rows(session_id)
    advanced_final = [r for r in rows_final if r["last_exchange_index"] >= 0]
    assert advanced_final, (
        f"Expected conversation_summaries to retain an advanced cursor after "
        f"continuation, got {rows_final!r}"
    )
    cursor_final = max(r["last_exchange_index"] for r in advanced_final)
    assert cursor_final >= cursor_after_first_sync, (
        f"Cursor must not regress: was {cursor_after_first_sync} after first "
        f"sync, now {cursor_final}"
    )


def test_context_window_optimization(monkeypatch):
    from unittest.mock import patch as _patch

    from proxy.context_v2 import _select_context_v2

    small_cfg = {
        "max_context_tokens": 2_000,
        "safe_context_tokens": 2_000,
        "optimal_context_tokens": 940,
        "tokens_per_char": 0.25,
        "chars_per_token": 4,
        "context_reduction_ratio": 0.75,
        "raw_exchange_count": 100,
    }

    big_pad = "Y" * 400
    messages = [{"role": "system", "content": "You are a helpful assistant."}]
    for i in range(60):
        messages.append({"role": "user", "content": f"User msg {i}: {big_pad}"})
        messages.append({"role": "assistant", "content": f"Assistant msg {i}: {big_pad}"})

    from unittest.mock import MagicMock
    ctx = MagicMock()
    ctx.v2_boundary_idx = None
    ctx.v2_boundary_chain = None
    ctx.is_subagent = False
    ctx.workspace_name = None

    from proxy.state import _chain_state as state_chain
    from proxy.state import chat_summaries as state_summaries
    from proxy.state import instruction_files_by_workspace as state_instructions

    state_chain.clear()
    state_summaries.clear()
    state_instructions.clear()

    with _patch("proxy.context.get_context_config", return_value=small_cfg), \
         _patch("proxy.model_registry.get_context_config", return_value=small_cfg), \
         _patch("proxy.vfs._build_vfs_context", return_value=None), \
         _patch("proxy.vfs._build_vfs_inherited_context", return_value=None):
        result = _select_context_v2(
            "sess-ctx-window-test",
            messages,
            model_name="test-small",
            model_cfg=None,
            is_subagent=False,
            ctx=ctx,
            tools=None,
        )

    assert isinstance(result, list)
    assert len(result) > 0
    assert result[0].get("role") == "system"
    assert result[0].get("content") == "You are a helpful assistant."
    assert len(result) < len(messages), (
        f"Expected V2 to trim 60-message conversation, got "
        f"len(result)={len(result)} len(input)={len(messages)}"
    )


def test_oversized_single_message(monkeypatch):
    from unittest.mock import patch as _patch

    from proxy.context_v2 import _select_context_v2

    small_cfg = {
        "max_context_tokens": 2_000,
        "safe_context_tokens": 2_000,
        "optimal_context_tokens": 940,
        "tokens_per_char": 0.25,
        "chars_per_token": 4,
        "context_reduction_ratio": 0.75,
        "raw_exchange_count": 100,
    }

    huge = " ".join(["word"] * 2000)
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": f"Analyze this large text: {huge}. What are the main themes?"},
    ]

    from unittest.mock import MagicMock
    ctx = MagicMock()
    ctx.v2_boundary_idx = None
    ctx.v2_boundary_chain = None
    ctx.is_subagent = False
    ctx.workspace_name = None

    from proxy.state import _chain_state as state_chain
    from proxy.state import chat_summaries as state_summaries
    from proxy.state import instruction_files_by_workspace as state_instructions

    state_chain.clear()
    state_summaries.clear()
    state_instructions.clear()

    with _patch("proxy.context.get_context_config", return_value=small_cfg), \
         _patch("proxy.model_registry.get_context_config", return_value=small_cfg), \
         _patch("proxy.vfs._build_vfs_context", return_value=None), \
         _patch("proxy.vfs._build_vfs_inherited_context", return_value=None):
        result = _select_context_v2(
            "sess-oversized-test",
            messages,
            model_name="oversized-test",
            model_cfg=None,
            is_subagent=False,
            ctx=ctx,
            tools=None,
        )

    assert result is not None
    assert isinstance(result, list)
    assert len(result) > 0
    assert result[0].get("role") == "system"
    assert result[0].get("content") == "You are a helpful assistant."
    assert any(m.get("role") == "user" for m in result), (
        f"Expected the oversized user message to survive in result, "
        f"got roles={[m.get('role') for m in result]}"
    )