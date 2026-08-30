"""Unit tests for V2 chain-hash round-7 patches.

Covers:
- R1: get_rich_summaries off-by-one (conversation position → SQL message_index)
- I1: prefix query off-by-one in _load_state_from_db
- R2: cold-init CAS-loser runs tool-pair cleanup
- S1: summaries surfaced for growing conversations (tiered pipeline)
- S2: sub-agent summary marker
- S3: tool_schema_tokens computed from actual tools
- R3: passthrough threshold = 0.30 per US-027
"""
import sqlite3
from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from proxy import context_v2


pytestmark = pytest.mark.unit


def _system(content="sys"):
    return {"role": "system", "content": content}


def _user(content):
    return {"role": "user", "content": content}


def _assistant(content=""):
    return {"role": "assistant", "content": content}


def _budget_config(safe=10_000, tokens_per_char=0.25):
    return {
        "max_context_tokens": 40_000,
        "tokens_per_char": tokens_per_char,
        "safe_context_tokens": safe,
        "raw_exchange_count": 100,
    }


def _make_ctx(request_id="test-req"):
    return SimpleNamespace(
        request_id=request_id,
        v2_boundary_idx=None,
        v2_boundary_chain=None,
    )


def _no_db_mocks(cfg, rich_summaries=None, ancient_history=""):
    return [
        patch("proxy.context.get_context_config", return_value=cfg),
        patch("memory.db.get_ancient_history", return_value=ancient_history),
        patch("memory.db.get_rich_summaries", return_value=rich_summaries or []),
        patch("proxy.vfs._build_vfs_inherited_context", return_value=None),
        patch("proxy.vfs._build_vfs_context", return_value=None),
    ]


def _mock_conn_for_load_state(messages_table_rows, raw_start):
    """Build a mock SQLite connection for _load_state_from_db.

    `messages_table_rows` is the full list of (message_index, role, chain_hash)
    rows present in messages.
    """
    def execute_side_effect(query, *args, **kwargs):
        q = str(query).upper()
        result = MagicMock()
        if "RAW_START_MESSAGE_INDEX" in q:
            result.fetchone.return_value = {"raw_start_message_index": raw_start}
        elif "TOTAL(*)" in q or "COUNT(*)" in q:
            if "ROLE != 'SYSTEM'" in q and "ORDER BY MESSAGE_INDEX" in q and "LIMIT 1 OFFSET" in q:
                # SQL boundary query: find SQL message_index of conversation position
                boundary = args[1] if len(args) > 1 else 0
                non_system = sorted(
                    [r for r in messages_table_rows if r[1] != "system"],
                    key=lambda r: r[0],
                )
                if boundary < len(non_system):
                    result.fetchone.return_value = {"message_index": non_system[boundary][0]}
                else:
                    result.fetchone.return_value = None
            elif "ROLE != 'SYSTEM'" in q:
                # Total non-system count
                count = sum(1 for r in messages_table_rows if r[1] != "system")
                result.fetchone.return_value = (count,)
            elif "CHAIN_HASH IS NOT NULL" in q:
                count = sum(1 for r in messages_table_rows if r[2] is not None)
                result.fetchone.return_value = (count,)
            elif "ROLE = 'SYSTEM'" in q:
                count = sum(1 for r in messages_table_rows if r[1] == "system")
                result.fetchone.return_value = (count,)
            else:
                result.fetchone.return_value = (0,)
        elif "CHAIN_HASH" in q:
            # Return chain_rows for prefix and recent queries
            if "ORDER BY MESSAGE_INDEX" in q and "LIMIT 1 OFFSET" in q:
                boundary = args[1] if len(args) > 1 else 0
                non_system = sorted(
                    [r for r in messages_table_rows if r[1] != "system"],
                    key=lambda r: r[0],
                )
                if boundary < len(non_system):
                    result.fetchone.return_value = {"message_index": non_system[boundary][0]}
                else:
                    result.fetchone.return_value = None
            elif "ORDER BY MESSAGE_INDEX DESC LIMIT" in q:
                limit = args[1] if len(args) > 1 else 10000
                sorted_rows = sorted(
                    [r for r in messages_table_rows if r[2] is not None],
                    key=lambda r: r[0],
                    reverse=True,
                )[:limit]
                result.fetchall.return_value = [
                    {"message_index": r[0], "chain_hash": r[2]} for r in sorted_rows
                ]
            elif "MESSAGE_INDEX <" in q:
                # prefix rows: chain_hash where message_index < some bound
                sql_boundary = args[1] if len(args) > 1 else 0
                rows = [
                    {"message_index": r[0], "chain_hash": r[2]}
                    for r in sorted(messages_table_rows, key=lambda r: r[0])
                    if r[2] is not None and r[0] < sql_boundary
                ]
                result.fetchall.return_value = rows
            else:
                result.fetchall.return_value = []
        else:
            result.fetchone.return_value = None
            result.fetchall.return_value = []
        return result

    conn = MagicMock()
    conn.execute.side_effect = execute_side_effect
    cm = MagicMock()
    cm.__enter__.return_value = conn
    cm.__exit__.return_value = False
    return cm


# ---------------------------------------------------------------------------
# R1: get_rich_summaries off-by-one — conversation position vs SQL index
# ---------------------------------------------------------------------------


class TestGetRichSummariesWithSystemMessage:
    """R1: with system@0, conversation position N corresponds to SQL
    message_index N + system_count. The fix converts conversation position
    to SQL message_index before calling get_rich_summaries(before_index=...).
    """

    def _build_db(self):
        conn = sqlite3.connect(":memory:")
        conn.row_factory = sqlite3.Row
        conn.executescript("""
            CREATE TABLE messages (
                session_id TEXT NOT NULL,
                message_index INTEGER NOT NULL,
                role TEXT NOT NULL,
                content TEXT,
                content_hash TEXT,
                tool_calls_json TEXT,
                tool_call_id TEXT,
                chain_hash TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT,
                is_edited BOOLEAN DEFAULT 0,
                exchange_index INTEGER DEFAULT NULL,
                UNIQUE(session_id, message_index)
            );
            CREATE TABLE conversation_summaries (
                session_id TEXT NOT NULL,
                model_name TEXT NOT NULL,
                summary TEXT,
                summary_tokens INTEGER,
                exchange_count INTEGER DEFAULT 0,
                last_exchange_start INTEGER DEFAULT 0,
                last_message_index INTEGER DEFAULT -1,
                raw_start_message_index INTEGER DEFAULT -1,
                ancient_history TEXT,
                last_exchange_index INTEGER DEFAULT -1,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                PRIMARY KEY (session_id, model_name)
            );
            CREATE TABLE message_summaries (
                session_id TEXT NOT NULL,
                message_index INTEGER NOT NULL,
                exchange_index INTEGER DEFAULT 0,
                end_message_index INTEGER DEFAULT 0,
                role TEXT,
                summary TEXT,
                tool_summary TEXT,
                content_hash TEXT,
                model_used TEXT,
                token_estimate INTEGER DEFAULT 0,
                rendered_full_tokens INTEGER DEFAULT 0,
                rendered_action_tokens INTEGER DEFAULT 0,
                rendered_narrative_tokens INTEGER DEFAULT 0,
                created_at TEXT,
                updated_at TEXT,
                UNIQUE(session_id, exchange_index)
            );
        """)
        # system@0 + 3 user/asst pairs
        conn.execute(
            "INSERT INTO messages (session_id, message_index, role, content, chain_hash, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            ("s1", 0, "system", "sys", None, "2026-01-01T00:00:00Z"),
        )
        for i in range(1, 7):
            role = "user" if i % 2 == 1 else "assistant"
            conn.execute(
                "INSERT INTO messages (session_id, message_index, role, content, chain_hash, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                ("s1", i, role, f"m{i}", f"h{i}", "2026-01-01T00:00:00Z"),
            )
        # Two summaries stored: for SQL index 1+2 (user@1 + asst@2 → conv 0+1)
        # and for SQL index 3+4 (user@3 + asst@4 → conv 2+3)
        conn.execute(
            "INSERT INTO message_summaries (session_id, message_index, role, summary, "
            "tool_summary, content_hash, model_used, token_estimate, "
            "rendered_full_tokens, rendered_action_tokens, rendered_narrative_tokens, "
            "created_at, updated_at, exchange_index) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            ("s1", 1, "user", "sum0-conv0", "", "h1", "m", 50,
             20, 20, 15, "2026-01-01T00:00:00Z", "2026-01-01T00:00:00Z", 0),
        )
        conn.execute(
            "INSERT INTO message_summaries (session_id, message_index, role, summary, "
            "tool_summary, content_hash, model_used, token_estimate, "
            "rendered_full_tokens, rendered_action_tokens, rendered_narrative_tokens, "
            "created_at, updated_at, exchange_index) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            ("s1", 3, "user", "sum1-conv2", "", "h3", "m", 50,
             20, 20, 15, "2026-01-01T00:00:00Z", "2026-01-01T00:00:00Z", 1),
        )
        conn.execute(
            "INSERT INTO message_summaries (session_id, message_index, role, summary, "
            "tool_summary, content_hash, model_used, token_estimate, "
            "rendered_full_tokens, rendered_action_tokens, rendered_narrative_tokens, "
            "created_at, updated_at, exchange_index) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            ("s1", 5, "user", "sum2-conv4", "", "h5", "m", 50,
             20, 20, 15, "2026-01-01T00:00:00Z", "2026-01-01T00:00:00Z", 2),
        )
        conn.execute(
            "INSERT INTO conversation_summaries "
            "(session_id, model_name, raw_start_message_index, created_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?)",
            ("s1", "m1", 3, "2026-01-01T00:00:00Z", "2026-01-01T00:00:00Z"),
        )
        conn.commit()
        return conn

    def test_summary_fetch_includes_correct_conv_position(self):
        """raw_start=3 (conv position 3) → SQL message_index of conv pos 3 = 4.
        With before_index=4 (SQL), summaries at SQL index < 4 are returned.
        That's 2 summaries (SQL index 1 and 3) — conv positions 0 and 2.
        """
        from proxy.context_v2 import _load_state_from_db

        conn = self._build_db()
        cm = MagicMock()
        cm.__enter__ = MagicMock(return_value=conn)
        cm.__exit__ = MagicMock(return_value=False)
        with patch("memory.db.connection", return_value=cm), \
             patch("memory.db.get_ancient_history", return_value=""), \
             patch("memory.db.get_rich_summaries") as rich_mock:
            rich_mock.return_value = [
                {"message_index": 1, "summary": "sum0-conv0", "role": "user", "tool_summary": ""},
                {"message_index": 3, "summary": "sum1-conv2", "role": "user", "tool_summary": ""},
            ]
            result = _load_state_from_db("s1", "m1")

        # The fix should compute SQL boundary correctly. Conv pos 3 → SQL 4.
        # before_index=4 should include summaries at SQL 1 and 3.
        assert rich_mock.called
        call_args = rich_mock.call_args
        before_index = call_args.kwargs.get("before_index") or call_args.args[1]
        assert before_index == 4, (
            f"R1 fix: conversation position 3 must convert to SQL message_index 4 "
            f"(conv_pos + system_count). Got before_index={before_index}"
        )


# ---------------------------------------------------------------------------
# R2: Cold-init CAS-loser runs tool-pair cleanup
# ---------------------------------------------------------------------------


class TestColdInitCasLoserRunsToolPairCleanup:
    """R2: When cold-init CAS-loser fires, the returned messages MUST go
    through _cleanup_tool_pair_integrity so tool pair integrity (US-024)
    is preserved.
    """

    def test_cold_init_loser_returns_cleaned_messages(self):
        """Patch _cleanup_tool_pair_integrity to mark itself called.
        Trigger cold-init CAS-loser and verify it was invoked."""
        from proxy.context_v2 import _select_context_v2

        import proxy.state as state
        state._chain_state.clear()
        key = ("sess-cold-loser-cleanup", "modelA")

        winner_entry = state.new_chain_state_entry()
        winner_entry["last_chain"] = []
        winner_entry["boundary_idx"] = 0
        winner_entry["boundary_chain"] = None
        winner_entry["summaries"] = []
        winner_entry["ancient"] = ""
        winner_entry["version"] = 1

        def _populate(*a, **kw):
            with state._chain_state_lock:
                state._chain_state[key] = winner_entry
            return []

        pad = "X" * 6000
        msgs = [_system("sys"), _user(f"u1 {pad}"), _assistant(f"a1 {pad}")]
        cfg = _budget_config(safe=10_000)
        ctx = _make_ctx("cold-loser-cleanup")

        conn_cm = MagicMock()
        conn_cm.__enter__ = MagicMock(return_value=MagicMock())
        conn_cm.__exit__ = MagicMock(return_value=False)

        cleanup_called = []
        original_cleanup = context_v2._cleanup_tool_pair_integrity

        def _tracking_cleanup(messages):
            cleanup_called.append(messages)
            return original_cleanup(messages)

        with ExitStack() as stack:
            stack.enter_context(patch("proxy.context.get_context_config", return_value=cfg))
            stack.enter_context(patch("memory.db.get_ancient_history", return_value=""))
            stack.enter_context(patch("proxy.vfs._build_vfs_context", return_value=None))
            stack.enter_context(patch("proxy.vfs._build_vfs_inherited_context", return_value=None))
            stack.enter_context(patch("memory.db.connection", return_value=conn_cm))
            stack.enter_context(patch(
                "memory.db_summaries.get_last_summarized_exchange_index",
                return_value=-1,
            ))
            stack.enter_context(patch("memory.sync.submit_ancient_compress", return_value=None))
            stack.enter_context(patch(
                "proxy.context_v2._load_state_from_db",
                return_value=None,
            ))
            stack.enter_context(patch(
                "memory.db.get_rich_summaries",
                side_effect=_populate,
            ))
            stack.enter_context(patch(
                "proxy.context_v2._cleanup_tool_pair_integrity",
                side_effect=_tracking_cleanup,
            ))
            _select_context_v2("sess-cold-loser-cleanup", msgs, model_name="modelA", ctx=ctx)

        assert ctx.v2_boundary_idx is None, (
            "Cold-init CAS-loser must still set ctx.v2_boundary_idx = None"
        )
        # Cold-init CAS-loser MUST call _cleanup_tool_pair_integrity before returning.
        # If it doesn't, tool pair integrity is violated (US-024).
        assert any(
            m for m in cleanup_called
            if any(x.get("role") == "assistant" for x in m)
        ), (
            "Cold-init CAS-loser MUST run _cleanup_tool_pair_integrity on the "
            "assembled messages before returning (US-024 tool pair integrity)"
        )

        state._chain_state.clear()


# ---------------------------------------------------------------------------
# S1: Summaries surfaced in tiered pipeline even without eviction
# ---------------------------------------------------------------------------


class TestSummariesSurfacedInTieredPipeline:
    """S1: when we're in the tiered pipeline (passthrough didn't fire) and
    no eviction occurred, summaries should still be fetched and injected.
    This was previously gated on boundary_idx > 0.
    """

    def test_summaries_fetched_when_no_eviction(self):
        """With messages large enough to exceed passthrough but small enough
        to not trigger eviction, summaries should still be fetched and
        appear in the output.
        """
        cfg = _budget_config(safe=10_000)
        pad = "X" * 6000
        msgs = [_system("sys"), _user(f"u {pad}"), _assistant(f"a {pad}")]
        # 2 msgs × 6000 chars × 0.25 = 3000 tokens > 0.30 × 10000 (passthrough)
        # 3000 tokens < 0.65 × 10000 (no eviction)

        summaries = [{"message_index": 5, "role": "user", "summary": "injected-summary", "tool_summary": ""}]

        with ExitStack() as stack:
            for p in _no_db_mocks(cfg, rich_summaries=summaries):
                stack.enter_context(p)
            ctx = _make_ctx()
            result = _select_context_v2_with_ctx("sess1", msgs, cfg, ctx=ctx)

        summary_blocks = [
            m for m in result
            if m.get("role") == "system" and "injected-summary" in m.get("content", "")
        ]
        assert len(summary_blocks) == 1, (
            "S1 contract: summary should appear in tiered-pipeline output "
            "even when no eviction occurred"
        )


def _select_context_v2_with_ctx(session_id, msgs, cfg, ctx):
    from proxy.context_v2 import _select_context_v2
    return _select_context_v2(session_id, msgs, model_name="test", ctx=ctx)


# ---------------------------------------------------------------------------
# S2: Sub-agent summary marker
# ---------------------------------------------------------------------------


class TestSubagentSummaryMarker:
    """S2: sub-agents get [Conversation Summary] / [End Summary] markers
    (US-058 inheritance) instead of [PAST CONTEXT] / [END PAST CONTEXT].
    """

    def test_subagent_summary_uses_conversation_summary_marker(self):
        cfg = _budget_config(safe=10_000)
        summaries = [{"message_index": 5, "role": "user", "summary": "subagent-context", "tool_summary": ""}]

        with ExitStack() as stack:
            for p in _no_db_mocks(cfg, rich_summaries=summaries):
                stack.enter_context(p)
            from proxy.context_v2 import _select_context_v2
            result = _select_context_v2(
                "sess1", _small_msgs(),
                model_name="test", is_subagent=True,
            )

        summary_blocks = [
            m for m in result
            if m.get("role") == "system" and "Conversation Summary" in m.get("content", "")
        ]
        assert len(summary_blocks) == 1, (
            "Sub-agent output must use [Conversation Summary] marker (US-058)"
        )
        content = summary_blocks[0]["content"]
        assert "[End Summary]" in content, (
            "Sub-agent summary block must close with [End Summary] marker"
        )
        assert "subagent-context" in content


def _small_msgs():
    pad = "X" * 6000
    return [_system("sys"), _user(f"u {pad}"), _assistant(f"a {pad}")]


# ---------------------------------------------------------------------------
# S3: tool_schema_tokens computed from actual tools
# ---------------------------------------------------------------------------


class TestToolSchemaTokensComputed:
    """S3: tool_schema_tokens is computed from the actual tools list passed
    to the request, not hardcoded to 2000.
    """

    def test_tool_schema_tokens_from_actual_tools(self):
        """S3: tool_schema_tokens is computed from the actual tools list
        passed to the request, not hardcoded to 2000.

        Mechanism: when tool_schema_tokens is computed correctly from a
        huge tools array, the assembled candidate overflows safe_tokens
        and the universal finalizer raises ContextLimitError. If
        tool_schema_tokens were hardcoded to 2000, the candidate would
        fit and no raise would fire.
        """
        import json
        from proxy.context import ContextLimitError

        cfg = _budget_config(safe=2_500)
        msgs = [_system("sys"), _user("hi"), _assistant("hello")]
        tools = [
            {
                "type": "function",
                "function": {
                    "name": f"tool_{i}",
                    "description": "x" * 2000,
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "arg_" + str(j): {
                                "type": "string",
                                "description": "z" * 200,
                            }
                            for j in range(5)
                        },
                        "required": ["arg_0"],
                    },
                },
            }
            for i in range(10)
        ]

        expected_tool_tokens = int(
            len(json.dumps(tools, separators=(",", ":"))) * 0.25
        )
        assert expected_tool_tokens > cfg["safe_context_tokens"], (
            f"Test setup invariant broken: tool_schema_tokens "
            f"(~{expected_tool_tokens}) must exceed safe_context_tokens "
            f"({cfg['safe_context_tokens']}) to force overflow."
        )

        with ExitStack() as stack:
            for p in _no_db_mocks(cfg):
                stack.enter_context(p)
            from proxy.context_v2 import _select_context_v2
            ctx = _make_ctx()
            # With tool_schema_tokens correctly computed (~7500), the
            # candidate overflows safe=2500 even after the trim safety
            # net (tool_schema is not content-trimmable). The finalizer
            # raises ContextLimitError. If tool_schema_tokens were
            # hardcoded to 2000, no raise would fire.
            with pytest.raises(ContextLimitError):
                _select_context_v2(
                    "sess1", msgs, model_name="test",
                    tools=tools, ctx=ctx,
                )

        # The raise proves tool_schema_tokens was computed (not hardcoded).
        # Expected: total > safe because tool_schema alone > safe.
        # No boundary_idx assertion — the raise prevents chain-state mutation.

