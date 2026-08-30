"""Unit tests for proxy.context_v2 sub-agent token budget guard.

C1: Regression tests for the sub-agent context assembly in
_select_context_v2. The sub-agent path must guarantee the task (first
role=="user") and the most recent unit survive, and must never return a
set that overflows the safe budget (system + raw + tool_schema).
"""
import copy
from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import patch, MagicMock

import pytest

import proxy.state as state
from proxy.types import RequestContext
from proxy.context_v2 import _select_context_v2


def _system(content: str) -> dict:
    return {"role": "system", "content": content}


def _user(content: str) -> dict:
    return {"role": "user", "content": content}


def _assistant(content: str = "", tool_calls=None) -> dict:
    msg = {"role": "assistant", "content": content}
    if tool_calls is not None:
        msg["tool_calls"] = tool_calls
    return msg


def _tool_call(tc_id: str, name: str = "read_file", args: str = '{"path":"/f.py"}') -> dict:
    return {"id": tc_id, "type": "function", "function": {"name": name, "arguments": args}}


def _tool(tc_id: str, content: str = "result") -> dict:
    return {"role": "tool", "tool_call_id": tc_id, "content": content}


def _assert_pair_integrity(messages: list) -> None:
    """Every tool result has its parent assistant; every assistant tool_call
    has a (real or synthetic) result."""
    assistant_tc_ids = set()
    for m in messages:
        if m.get("role") == "assistant" and isinstance(m.get("tool_calls"), list):
            for tc in m["tool_calls"]:
                if tc.get("id"):
                    assistant_tc_ids.add(tc["id"])
    present_tool_ids = {m.get("tool_call_id") for m in messages if m.get("role") == "tool"}
    for m in messages:
        if m.get("role") == "tool":
            assert m.get("tool_call_id") in assistant_tc_ids, "Orphan tool result"
    assert assistant_tc_ids <= present_tool_ids, "Assistant tool_call missing its result"


@pytest.fixture
def tight_budget_config():
    """Config where total_available leaves a small recent_raw budget.

    tokens_per_char=0.25 → 400 chars = 100 tokens.
    safe_context_tokens=10000, tools=2000, system=0 →
    total_available = 8000, recent_raw_budget = int(10000 * 0.65) = 6500.
    """
    return {
        "max_context_tokens": 40_000,
        "tokens_per_char": 0.25,
        "safe_context_tokens": 10_000,
        "raw_exchange_count": 100,
    }


def _no_db_mocks(cfg):
    """Return a list of patches that suppress DB and VFS side-effects."""
    return [
        patch("proxy.context.get_context_config", return_value=cfg),
        patch("memory.db.get_ancient_history", return_value=""),
        patch("memory.db.get_rich_summaries", return_value=[]),
        patch("proxy.vfs._build_vfs_inherited_context", return_value=None),
        patch("proxy.vfs._build_vfs_context", return_value=None),
    ]


class TestSubagentTokenBudgetGuard:
    """C1: _select_context_v2 sub-agent path must respect the safe-token
    budget accounting for system + raw + tool_schema. When the exchange
    overflows safe, the task and last unit are preserved via trimming.
    """

    def test_subagent_falls_through_when_tokens_exceed_budget(self, tight_budget_config):
        """Sub-agent with many large messages (system+raw+tools > safe_tokens)
        must NOT short-circuit. It must run the sub-agent overflow path,
        which preserves [system, first-user, trimmed-last-unit].

        Budget math (tokens_per_char=0.25):
          safe_context_tokens=10000, system=0, tools=2000
          110 user messages = 11000 tokens; 0 + 11000 + 2000 = 13000 > 10000
          → overflow → preserves [system, first-user, last-user].
        """
        pad = "X" * 400  # 100 tokens per message
        messages = [_system("")] + [_user(f"user {i} {pad}") for i in range(110)]

        with ExitStack() as stack:
            for p in _no_db_mocks(tight_budget_config):
                stack.enter_context(p)
            result = _select_context_v2(
                "sess1", messages, model_name="test", is_subagent=True,
            )

        assert isinstance(result, list)
        # The overflow path returns [system, first-user(task), last-unit].
        # The full tier-injection path would have included summaries;
        # the presence of trimming proves the overflow path ran.
        non_system = [m for m in result if m.get("role") != "system"]
        assert len(non_system) < len(messages) - 1, (
            f"Sub-agent should preserve task + last-unit only when "
            f"system+raw+tools overflows safe, but {len(non_system)} messages "
            f"of {len(messages) - 1} non-system survived"
        )
        # The task (first user) must survive.
        assert any("user 0" in m.get("content", "") for m in result), \
            "Sub-agent task (first user message) must survive overflow"

    def test_subagent_short_circuits_when_exchange_fits_safe(self, tight_budget_config):
        """Sub-agent whose system + raw + tool_schema fits within safe must
        return all messages.

        96 messages × 100 tok = 9600 raw tokens. With safe=12000, system=0,
        tools=2000: 0 + 9600 + 2000 = 11600 <= 12000 → case 1 returns all.
        (safe=12000 also makes recent_raw_budget=7800 < 9600, so the
        main-agent budget pipeline would trim — proving the sub-agent
        path returned all messages rather than the main trim path.)
        """
        fits_config = {
            "max_context_tokens": 40_000,
            "tokens_per_char": 0.25,
            "safe_context_tokens": 12_000,
            "raw_exchange_count": 100,
        }
        pad = "X" * 400  # 100 tokens per message
        messages = [_system("")] + [_user(f"user {i} {pad}") for i in range(96)]

        with ExitStack() as stack:
            for p in _no_db_mocks(fits_config):
                stack.enter_context(p)
            result = _select_context_v2(
                "sess1", messages, model_name="test", is_subagent=True,
            )

        assert isinstance(result, list)
        assert len(result) == len(messages), (
            f"Sub-agent whose exchange fits safe should return all messages, "
            f"got {len(result)} of {len(messages)}"
        )

    def test_non_subagent_unaffected_by_token_guard(self, tight_budget_config):
        """Non-sub-agent with estimated_tokens > safe_tokens must NOT enter
        the sub-agent path. The normal budget pipeline trims as expected.
        """
        pad = "X" * 400  # 100 tokens per message
        # 110 messages × 100 tok = 11000 > safe=10000
        messages = [_system("")] + [_user(f"user {i} {pad}") for i in range(110)]

        with ExitStack() as stack:
            for p in _no_db_mocks(tight_budget_config):
                stack.enter_context(p)
            result = _select_context_v2(
                "sess1", messages, model_name="test", is_subagent=False,
            )

        assert isinstance(result, list)
        # Non-sub-agent: the sub-agent path does NOT apply, so the normal
        # budget pipeline trims to recent_raw_budget=6500 tokens.
        non_system = [m for m in result if m.get("role") != "system"]
        assert len(non_system) < len(messages) - 1, (
            f"Non-sub-agent should run normal budget pipeline regardless of "
            f"token count, but {len(non_system)} of {len(messages) - 1} survived"
        )

    def test_subagent_guard_boundary_equal_tokens(self, tight_budget_config):
        """When estimated_tokens == model_safe_tokens, system + raw + tools
        still overflows safe (by the tool_schema reservation), so the
        sub-agent overflow path runs and preserves task + last unit.

        Budget math (tokens_per_char=0.25):
          safe_context_tokens=10000. 100 user messages = 10000 raw tokens.
          0 + 10000 + 2000 = 12000 > 10000 → overflow path.
        """
        pad = "X" * 400  # 100 tokens per message
        messages = [_system("")] + [_user(f"user {i} {pad}") for i in range(100)]

        with ExitStack() as stack:
            for p in _no_db_mocks(tight_budget_config):
                stack.enter_context(p)
            result = _select_context_v2(
                "sess1", messages, model_name="test", is_subagent=True,
            )

        assert isinstance(result, list)
        non_system = [m for m in result if m.get("role") != "system"]
        assert len(non_system) < 100, (
            f"Sub-agent at boundary should run overflow path, "
            f"but {len(non_system)} of 100 survived"
        )

    def test_slice_4e_subagent_caps_complete_rendered_summary(self):
        """Sub-agent inheritance uses its full wrapper in the 15% cap."""
        session_id = "slice-4e-subagent-summary"
        summaries = [
            {
                "message_index": index,
                "role": "user",
                "summary": "S" * 400,
                "tool_summary": "",
            }
            for index in range(200, 220)
        ]
        cfg = {
            "max_context_tokens": 40_000,
            "tokens_per_char": 0.25,
            "safe_context_tokens": 10_000,
            "raw_exchange_count": 100,
        }
        ctx = SimpleNamespace(
            is_subagent=True,
            session_id=session_id,
            client_hint="",
            request_id="slice-4e-subagent-summary",
            parent_agent_id="parent",
        )
        messages = [
            _system("system"),
            _user("complete this task"),
        ]

        with ExitStack() as stack:
            stack.enter_context(
                patch(
                    "proxy.context.get_context_config",
                    return_value=cfg,
                )
            )
            stack.enter_context(
                patch(
                    "memory.db.get_rich_summaries",
                    return_value=summaries,
                )
            )
            stack.enter_context(
                patch(
                    "proxy.vfs._build_vfs_inherited_context",
                    return_value=None,
                )
            )
            dispatch = stack.enter_context(
                patch(
                    "memory.sync.submit_ancient_compress",
                    return_value=None,
                )
            )
            result = _select_context_v2(
                session_id,
                messages,
                model_name="test",
                is_subagent=True,
                ctx=ctx,
            )

        summary_messages = [
            message
            for message in result
            if (
                message.get("role") == "system"
                and "[Conversation Summary]"
                in message.get("content", "")
            )
        ]
        assert len(summary_messages) == 1

        content = summary_messages[0]["content"]
        assert content.count("[msg#") == 13
        assert "[msg#207]" in content
        assert "[msg#219]" in content
        assert "[msg#206]" not in content
        assert "[End Summary]" in content
        assert int(len(content) * 0.25) == 1_399
        assert int(len(content) * 0.25) <= 1_500

        dispatch.assert_not_called()


class TestSubagentTaskPreservation:
    """Bug #1: a sub-agent whose single exchange exceeds safe must keep its
    task (first role=="user"). The exchange is sized to genuinely exceed
    safe (system + raw + tool_schema) so the test is non-vacuous.
    """

    def test_task_survives_when_exchange_exceeds_safe(self, tight_budget_config):
        """One exchange of ~14 large file-read tool results pushes raw past
        safe. The task (first user) must survive; tool results may be
        trimmed/dropped with pair integrity preserved.

        safe=10000, system=0, tools=2000. Each tool result ≈ 850 tok
        (3400-char pad); 14 results ≈ 11900 + assistant ≈ 90 + task ≈ 10
        → system+raw+tools ≈ 14000 > 10000 → overflow path.
        """
        pad = "X" * 3400  # ~850 tokens per tool result
        n_tools = 14
        tc_ids = [f"tc_{i}" for i in range(n_tools)]
        tool_calls = [_tool_call(tc_id) for tc_id in tc_ids]
        messages = [
            _system(""),
            _user("TASK: read all these files and summarize"),
            _assistant(content="reading files", tool_calls=tool_calls),
        ]
        for tc_id in tc_ids:
            messages.append(_tool(tc_id, content=f"file {pad}"))

        with ExitStack() as stack:
            for p in _no_db_mocks(tight_budget_config):
                stack.enter_context(p)
            result = _select_context_v2(
                "sess1", messages, model_name="test", is_subagent=True,
            )

        # The task MUST survive.
        task_present = any(
            m.get("role") == "user" and "TASK" in m.get("content", "")
            for m in result
        )
        assert task_present, (
            f"Bug #1: sub-agent task must survive when exchange exceeds safe; "
            f"result roles: {[m.get('role') for m in result]}"
        )
        # Pair integrity must hold (dropped tool results get synthetic replacements).
        _assert_pair_integrity(result)
        # The result must fit within safe (system + raw + tools). With tools=None
        # post-Slice-4A, tool_schema_tokens=0 (was 2000 before Slice 4A).
        from proxy.context import _count_msg_chars
        non_system = [m for m in result if m.get("role") != "system"]
        raw_tokens = sum(int(_count_msg_chars(m) * 0.25) for m in non_system)
        assert raw_tokens <= 10000, (
            f"Sub-agent result must not overflow safe: {raw_tokens} > 10000"
        )


class TestSubagentSystemPromptOverflow:
    """Bug #8: a sub-agent with a large skill system prompt must not return
    a set that overflows safe. The old guard compared estimated (which
    excluded system) against safe, so a large system prompt slipped through.
    """

    def test_large_system_prompt_does_not_overflow_safe(self):
        """system=3000 tok + exchange ≈ 5700 tok + tools=2000 = 10700 > 10000.
        The old guard saw estimated(5700) < safe(10000) and returned all,
        overflowing upstream. The fix trims to fit safe while preserving
        the task.
        """
        cfg = {
            "max_context_tokens": 40_000,
            "tokens_per_char": 0.25,
            "safe_context_tokens": 10_000,
            "raw_exchange_count": 100,
        }
        system = _system("S" * 12000)  # 3000 tokens
        n_tools = 10
        tc_ids = [f"tc_{i}" for i in range(n_tools)]
        tool_calls = [_tool_call(tc_id) for tc_id in tc_ids]
        tool_pad = "X" * 2240  # ~560 tokens per tool result
        messages = [system, _user("TASK: do the work")]
        messages.append(_assistant(content="working", tool_calls=tool_calls))
        for tc_id in tc_ids:
            messages.append(_tool(tc_id, content=f"r {tool_pad}"))

        with ExitStack() as stack:
            for p in _no_db_mocks(cfg):
                stack.enter_context(p)
            result = _select_context_v2(
                "sess1", messages, model_name="test", is_subagent=True,
            )

        from proxy.context import _count_msg_chars
        # The system prompt must be present (index 0).
        assert result[0].get("role") == "system"
        # The task must survive.
        assert any(
            m.get("role") == "user" and "TASK" in m.get("content", "")
            for m in result
        ), "Sub-agent task must survive"
        # system + raw + tools must NOT overflow safe. With tools=None
        # post-Slice-4A, tool_schema_tokens=0 (was 2000 before Slice 4A).
        sys_tokens = int(len(result[0].get("content", "")) * 0.25)
        non_system = [m for m in result if m.get("role") != "system"]
        raw_tokens = sum(int(_count_msg_chars(m) * 0.25) for m in non_system)
        assert sys_tokens + raw_tokens <= 10000, (
            f"Bug #8: sub-agent result overflows safe: "
            f"{sys_tokens}+{raw_tokens} > 10000"
        )
        _assert_pair_integrity(result)


# ---------------------------------------------------------------------------
# Assembly order (barrier + VFS + last_user-after-barrier)
# ---------------------------------------------------------------------------

def _assembly_budget_config(safe=10_000):
    return {
        "max_context_tokens": 40_000,
        "tokens_per_char": 0.25,
        "safe_context_tokens": safe,
        "raw_exchange_count": 100,
    }


def _assembly_msgs(count, chars=400):
    pad = "X" * chars
    return [{"role": "user", "content": f"msg {i} {pad}"} for i in range(count)]


def _assembly_mock_conn():
    """Mock for the C5 fast-exit SELECT 1 queries + downstream reads.

    Returns rows for conversation_summaries / message_summaries so the
    fast-exit does NOT fire (preserving the original test intent: the
    normal assembly path runs with no DB rows for chain/restore). Other
    queries return shape-correct empty results.
    """
    def execute_side_effect(query, *args, **kwargs):
        q = str(query).upper()
        result = MagicMock()
        if "SELECT 1 FROM CONVERSATION_SUMMARIES" in q or "SELECT 1 FROM MESSAGE_SUMMARIES" in q:
            result.fetchone.return_value = (1,)
        elif "RAW_START_MESSAGE_INDEX" in q:
            result.fetchone.return_value = None
        elif "MIN(MESSAGE_INDEX)" in q:
            result.fetchone.return_value = None
        elif "CHAIN_HASH" in q:
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


class TestAssemblyOrder:
    """Assembly order per Wave 4 canonical placement:
    [sys, summaries, ancient, completed_history, vfs, live_exchange].

    Wave 4 removed the context-layer barrier emission from V2 — the
    MiniMax user-user adjacency break is delegated to the adapter
    (MiniMaxAdapter.apply) downstream. The completed/live split is
    produced by `_split_completed_and_live_exchange`; VFS is appended only
    when live_exchange is non-empty.
    """

    def test_no_summaries_assembly_is_sys_completed_live(self):
        """Without summaries/ancient: assembly is [sys, completed_history,
        live_exchange] — no injected system blocks, no barrier."""
        cfg = _assembly_budget_config()
        msgs = [{"role": "system", "content": "sys"}] + _assembly_msgs(40)

        ctx = SimpleNamespace(request_id="t", v2_boundary_idx=None, v2_boundary_chain=None)
        conn_cm = _assembly_mock_conn()
        with ExitStack() as stack:
            stack.enter_context(patch("proxy.context.get_context_config", return_value=cfg))
            stack.enter_context(patch("memory.db.get_rich_summaries", return_value=[]))
            stack.enter_context(patch("memory.db.get_ancient_history", return_value=""))
            stack.enter_context(patch("proxy.vfs._build_vfs_context", return_value=None))
            stack.enter_context(patch("proxy.vfs._build_vfs_inherited_context", return_value=None))
            stack.enter_context(patch("memory.db.connection", return_value=conn_cm))
            stack.enter_context(
                patch("memory.db_summaries.get_last_summarized_exchange_index", return_value=-1)
            )
            result = _select_context_v2("sess1", msgs, model_name="test", ctx=ctx)

        assert result[0].get("role") == "system"
        injected_systems = [
            m for m in result[1:]
            if m.get("role") == "system"
            and ("[PAST CONTEXT" in m.get("content", "")
                 or "[TIMELINE" in m.get("content", "")
                 or "[VFS]" in m.get("content", ""))
        ]
        assert len(injected_systems) == 0, "No summaries/ancient/VFS expected"
        barriers = [m for m in result if m.get("role") == "assistant" and m.get("content") == " "]
        assert len(barriers) == 0, "Wave 4 must NOT emit context-layer barrier"

    def test_slice_4b_live_multi_tool_exchange_is_reserved_before_history_trim(self):
        session_id = "slice-4b-session"
        model_name = "slice-4b-model"
        cfg = {
            "safe_context_tokens": 10_000,
            "max_context_tokens": 40_000,
            "tokens_per_char": 0.25,
            "raw_exchange_count": 100,
        }

        old_user = {
            "role": "user",
            "content": "OLD-USER-" + ("o" * 6000),
        }
        old_assistant = {
            "role": "assistant",
            "content": "OLD-ASSISTANT-" + ("a" * 6000),
        }
        current_user = {
            "role": "user",
            "content": "CURRENT-USER-" + ("u" * 10000),
        }
        current_assistant = {
            "role": "assistant",
            "content": "CURRENT-ASSISTANT-" + ("c" * 10000),
            "tool_calls": [
                {
                    "id": "slice4b-a",
                    "type": "function",
                    "function": {
                        "name": "read",
                        "arguments": "{\"path\":\"/a\"}",
                    },
                },
                {
                    "id": "slice4b-b",
                    "type": "function",
                    "function": {
                        "name": "read",
                        "arguments": "{\"path\":\"/b\"}",
                    },
                },
            ],
        }
        tool_a = {
            "role": "tool",
            "tool_call_id": "slice4b-a",
            "content": "RESULT-A",
        }
        tool_b = {
            "role": "tool",
            "tool_call_id": "slice4b-b",
            "content": "RESULT-B",
        }

        live_exchange = [
            current_user,
            current_assistant,
            tool_a,
            tool_b,
        ]
        messages = [
            {"role": "system", "content": "system"},
            old_user,
            old_assistant,
            *live_exchange,
        ]
        before = copy.deepcopy(messages)

        tools = [
            {
                "type": "function",
                "function": {
                    "name": "large_schema",
                    "description": "d" * 16000,
                    "parameters": {
                        "type": "object",
                        "properties": {},
                    },
                },
            },
        ]

        ctx = SimpleNamespace(
            request_id="slice-4b",
            v2_boundary_idx=None,
            v2_boundary_chain=None,
        )
        conn_cm = _assembly_mock_conn()

        warm_entry = state.new_chain_state_entry()
        warm_entry["version"] = 1
        warm_entry["boundary_idx"] = 0
        warm_entry["boundary_chain"] = None

        with state._chain_state_lock:
            state._chain_state[(session_id, model_name)] = warm_entry

        with ExitStack() as stack:
            stack.enter_context(
                patch(
                    "proxy.context.get_context_config",
                    return_value=cfg,
                )
            )
            stack.enter_context(
                patch(
                    "memory.db.get_rich_summaries",
                    return_value=[],
                )
            )
            stack.enter_context(
                patch(
                    "memory.db.get_ancient_history",
                    return_value="",
                )
            )
            stack.enter_context(
                patch(
                    "proxy.vfs._build_vfs_context",
                    return_value=None,
                )
            )
            stack.enter_context(
                patch(
                    "proxy.vfs._build_vfs_inherited_context",
                    return_value=None,
                )
            )
            stack.enter_context(
                patch(
                    "memory.db.connection",
                    return_value=conn_cm,
                )
            )
            stack.enter_context(
                patch(
                    "memory.db_summaries.get_last_summarized_exchange_index",
                    return_value=-1,
                )
            )
            result = _select_context_v2(
                session_id,
                messages,
                model_name=model_name,
                ctx=ctx,
                tools=tools,
            )

        assert messages == before

        result_by_marker = {
            message.get("content", "")[:32]: message
            for message in result
            if isinstance(message.get("content"), str)
        }

        assert not any(
            message.get("content", "").startswith("OLD-USER-")
            for message in result
        )
        assert not any(
            message.get("content", "").startswith("OLD-ASSISTANT-")
            for message in result
        )

        current_index = next(
            index
            for index, message in enumerate(result)
            if message.get("content", "").startswith("CURRENT-USER-")
        )
        assert result[current_index:] == live_exchange
        assert result[current_index] == current_user
        assert result[current_index + 1] == current_assistant
        assert result[current_index + 2] == tool_a
        assert result[current_index + 3] == tool_b

        assert result_by_marker[current_user["content"][:32]] == current_user
        assert result_by_marker[current_assistant["content"][:32]] == current_assistant

    def test_slice_4c_sub_agent_return_is_mandatory_under_history_pressure(self):
        session_id = "slice-4c-session"
        model_name = "slice-4c-model"
        cfg = {
            "safe_context_tokens": 10_000,
            "max_context_tokens": 40_000,
            "tokens_per_char": 0.25,
            "raw_exchange_count": 100,
        }

        old_user = {
            "role": "user",
            "content": "OLD-USER-" + ("o" * 6000),
        }
        old_assistant = {
            "role": "assistant",
            "content": "OLD-ASSISTANT-" + ("a" * 6000),
        }
        sentinel = {
            "role": "tool",
            "content": "SUB-AGENT-RESULT-" + ("s" * 12000),
            "tool_call_id": "child-agent-slice-4c",
            "name": "sub_agent_return",
            "_sub_agent_return": True,
        }
        current_user = {
            "role": "user",
            "content": "CURRENT-USER-" + ("u" * 8000),
        }

        messages = [
            {"role": "system", "content": "system"},
            old_user,
            old_assistant,
            sentinel,
            current_user,
        ]
        before = copy.deepcopy(messages)

        tools = [
            {
                "type": "function",
                "function": {
                    "name": "large_schema",
                    "description": "d" * 16000,
                    "parameters": {
                        "type": "object",
                        "properties": {},
                    },
                },
            },
        ]

        ctx = SimpleNamespace(
            request_id="slice-4c",
            v2_boundary_idx=None,
            v2_boundary_chain=None,
        )
        conn_cm = _assembly_mock_conn()

        warm_entry = state.new_chain_state_entry()
        warm_entry["version"] = 1
        warm_entry["boundary_idx"] = 0
        warm_entry["boundary_chain"] = None

        with state._chain_state_lock:
            state._chain_state[(session_id, model_name)] = warm_entry

        with ExitStack() as stack:
            stack.enter_context(
                patch(
                    "proxy.context.get_context_config",
                    return_value=cfg,
                )
            )
            stack.enter_context(
                patch(
                    "memory.db.get_rich_summaries",
                    return_value=[],
                )
            )
            stack.enter_context(
                patch(
                    "memory.db.get_ancient_history",
                    return_value="",
                )
            )
            stack.enter_context(
                patch(
                    "proxy.vfs._build_vfs_context",
                    return_value=None,
                )
            )
            stack.enter_context(
                patch(
                    "proxy.vfs._build_vfs_inherited_context",
                    return_value=None,
                )
            )
            stack.enter_context(
                patch(
                    "memory.db.connection",
                    return_value=conn_cm,
                )
            )
            stack.enter_context(
                patch(
                    "memory.db_summaries.get_last_summarized_exchange_index",
                    return_value=-1,
                )
            )
            result = _select_context_v2(
                session_id,
                messages,
                model_name=model_name,
                ctx=ctx,
                tools=tools,
            )

        assert messages == before

        sentinels = [
            message
            for message in result
            if message.get("_sub_agent_return") is True
        ]
        assert sentinels == [sentinel]
        assert sentinels[0] == sentinel

        sentinel_index = result.index(sentinels[0])
        current_index = next(
            index
            for index, message in enumerate(result)
            if message.get("content", "").startswith("CURRENT-USER-")
        )
        assert sentinel_index < current_index
        assert result[current_index] == current_user

        assert not any(
            message.get("content", "").startswith("OLD-USER-")
            for message in result
        )
        assert not any(
            message.get("content", "").startswith("OLD-ASSISTANT-")
            for message in result
        )

    def test_slice_4d_mandatory_overflow_raises_when_discretionary_history_already_empty(self):
        """Slice 4D: when discretionary history is empty and the mandatory
        sentinel + live user + Tier-2 + tools still overflow safe, the
        terminal helper raises ContextLimitError instead of silently
        returning an overflowing candidate.
        """
        from proxy.context import ContextLimitError

        session_id = "slice-4d-overflow-session"
        model_name = "slice-4d-overflow-model"
        cfg = {
            "safe_context_tokens": 10_000,
            "max_context_tokens": 40_000,
            "tokens_per_char": 0.25,
            "raw_exchange_count": 100,
        }

        sentinel = {
            "role": "tool",
            "content": "MANDATORY-SENTINEL-" + ("s" * 30000),
            "tool_call_id": "slice-4d-mandatory",
            "name": "sub_agent_return",
            "_sub_agent_return": True,
        }
        current_user = {
            "role": "user",
            "content": "CURRENT-USER-" + ("u" * 8000),
        }

        messages = [
            {"role": "system", "content": "system"},
            sentinel,
            current_user,
        ]

        tools = [{
            "type": "function",
            "function": {
                "name": "large_schema",
                "description": "d" * 16000,
                "parameters": {"type": "object", "properties": {}},
            },
        }]

        ctx = SimpleNamespace(
            request_id="slice-4d-overflow",
            v2_boundary_idx=None,
            v2_boundary_chain=None,
        )
        conn_cm = _assembly_mock_conn()

        warm_entry = state.new_chain_state_entry()
        warm_entry["version"] = 1
        warm_entry["boundary_idx"] = 0
        warm_entry["boundary_chain"] = None

        with state._chain_state_lock:
            state._chain_state[(session_id, model_name)] = warm_entry

        with ExitStack() as stack:
            stack.enter_context(patch("proxy.context.get_context_config", return_value=cfg))
            stack.enter_context(patch("memory.db.get_rich_summaries", return_value=[]))
            stack.enter_context(patch("memory.db.get_ancient_history", return_value=""))
            stack.enter_context(patch("proxy.vfs._build_vfs_context", return_value=None))
            stack.enter_context(patch("proxy.vfs._build_vfs_inherited_context", return_value=None))
            stack.enter_context(patch("memory.db.connection", return_value=conn_cm))
            stack.enter_context(patch(
                "memory.db_summaries.get_last_summarized_exchange_index",
                return_value=-1,
            ))
            with pytest.raises(ContextLimitError) as excinfo:
                _select_context_v2(
                    session_id,
                    messages,
                    model_name=model_name,
                    ctx=ctx,
                    tools=tools,
                )

        assert "Final context exceeds safe token budget" in str(excinfo.value)

    def test_slice_4d_optional_vfs_dropped_when_mandatory_history_fits_without_it(self):
        """Slice 4D: an optional main-path VFS that pushes the candidate over
        ``safe_tokens`` is dropped by ``_finalize_context_result``; the
        remainder must fit and survive.

        Correction: activate the ACTUAL request ``session_id`` under the
        workspace so ``_build_vfs_context`` resolves the workspace via
        ``workspace_store`` and returns a non-None VFS message.
        """
        from proxy.vfs import _cache_file, _build_vfs_context
        from proxy.context import _count_msg_chars

        session_id = "slice-4d-vfs-drop-session"
        model_name = "slice-4d-vfs-drop-model"
        workspace_name = "slice-4d-vfs-drop-ws"
        client_hint = "slice-4d-client"

        cfg = {
            "safe_context_tokens": 4_000,
            "max_context_tokens": 40_000,
            "tokens_per_char": 1.0,
            "raw_exchange_count": 100,
        }

        long_content = "X" * 1980
        messages = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": long_content},
            {"role": "assistant", "content": long_content},
        ]

        # Activate the ACTUAL request session under the workspace so the
        # VFS builder resolves it via workspace_store.
        state.workspace_store.activate(
            session_id, workspace_name, client_hint=client_hint,
        )
        _cache_file(
            "/slice-4d/file.py", "slice-4d cache content",
            workspace_name, session_id,
        )

        ctx = SimpleNamespace(
            request_id="slice-4d-vfs-drop",
            v2_boundary_idx=None,
            v2_boundary_chain=None,
            session_id=session_id,
            workspace_name=workspace_name,
            client_hint=client_hint,
        )
        conn_cm = _assembly_mock_conn()

        warm_entry = state.new_chain_state_entry()
        warm_entry["version"] = 1
        warm_entry["boundary_idx"] = 0
        warm_entry["boundary_chain"] = None

        with state._chain_state_lock:
            state._chain_state[(session_id, model_name)] = warm_entry

        # Wrap the real VFS builder to record the returned object.
        real_vfs_builder = _build_vfs_context
        built_vfs_objects = []

        def _tracking_vfs_builder(sess, ctx_arg, budget, tpc):
            result = real_vfs_builder(sess, ctx_arg, budget, tpc)
            if result is not None:
                built_vfs_objects.append(result)
            return result

        with ExitStack() as stack:
            stack.enter_context(patch("proxy.context.get_context_config", return_value=cfg))
            stack.enter_context(patch("memory.db.get_rich_summaries", return_value=[]))
            stack.enter_context(patch("memory.db.get_ancient_history", return_value=""))
            stack.enter_context(patch("proxy.vfs._build_vfs_inherited_context", return_value=None))
            stack.enter_context(
                patch("proxy.vfs._build_vfs_context", side_effect=_tracking_vfs_builder)
            )
            stack.enter_context(patch("memory.db.connection", return_value=conn_cm))
            stack.enter_context(patch(
                "memory.db_summaries.get_last_summarized_exchange_index",
                return_value=-1,
            ))
            result = _select_context_v2(
                session_id,
                messages,
                model_name=model_name,
                ctx=ctx,
            )

        # The VFS builder was called and returned a non-None VFS.
        assert built_vfs_objects, (
            "VFS builder must have been called and returned non-None"
        )
        built_vfs = built_vfs_objects[0]
        assert "/slice-4d/file.py" in built_vfs.get("content", ""), (
            "Built VFS must contain the cached file path"
        )

        # Arithmetic preconditions: candidate without VFS fits, with VFS overflows.
        candidate_without_vfs_chars = sum(
            _count_msg_chars(m) for m in messages
        )
        vfs_chars = _count_msg_chars(built_vfs)
        without_vfs_tokens = int(candidate_without_vfs_chars * 1.0)
        with_vfs_tokens = int((candidate_without_vfs_chars + vfs_chars) * 1.0)
        assert without_vfs_tokens <= 4_000, (
            f"Precondition: without VFS ({without_vfs_tokens}) must fit safe (4000)"
        )
        assert with_vfs_tokens > 4_000, (
            f"Precondition: with VFS ({with_vfs_tokens}) must overflow safe (4000)"
        )

        # VFS was dropped — the built VFS object is absent from the result.
        assert not any(m is built_vfs for m in result), (
            "The exact built VFS object must be absent from the result"
        )
        assert not any("[VFS" in m.get("content", "") for m in result), (
            f"Optional VFS should have been dropped; got "
            f"{[m.get('content', '')[:40] for m in result]}"
        )

        # Final result fits safe.
        result_tokens = int(sum(_count_msg_chars(m) for m in result) * 1.0)
        assert result_tokens <= 4_000, (
            f"Final result ({result_tokens}) must fit safe (4000)"
        )

        # Current live user content survives.
        assert any(
            m.get("content") == long_content and m.get("role") == "user"
            for m in result
        ), "Live user content must survive VFS drop"

    def test_slice_4d_count_failure_cannot_bypass_final_budget(self):
        """Slice 4D correction: when ``_count_msg_chars`` raises on a
        malformed nested ``tool_calls`` entry, ``_finalize_context_result``
        must raise ``ContextLimitError`` — never silently count the message
        as zero tokens.
        """
        from proxy.context import ContextLimitError
        from proxy.context_v2 import _finalize_context_result

        candidate = [{
            "role": "assistant",
            "content": "X" * 100_000,
            "tool_calls": [{"function": "not-an-object"}],
        }]

        with pytest.raises(ContextLimitError) as excinfo:
            _finalize_context_result(
                candidate,
                "slice-4d-count-fail",
                None,
                tool_schema_tokens=0,
                model_safe_tokens=10_000,
                tokens_per_char=0.25,
            )

        assert "Cannot validate final context budget" in str(excinfo.value)
        assert isinstance(excinfo.value.__cause__, AttributeError)

    def test_slice_4d_overflow_does_not_advance_warm_chain_state(self):
        """Slice 4D correction: when the slow path's final validation raises
        ``ContextLimitError``, the warm chain-state entry must remain
        untouched — no version bump, no boundary change, no ctx mutation.
        """
        import copy
        from proxy.context import ContextLimitError

        session_id = "slice-4d-atomic-session"
        model_name = "slice-4d-atomic-model"
        cfg = {
            "safe_context_tokens": 10_000,
            "max_context_tokens": 40_000,
            "tokens_per_char": 0.25,
            "raw_exchange_count": 100,
        }

        # Mandatory sentinel + current user: combined mandatory cost
        # exceeds safe even after discretionary trim.
        sentinel = {
            "role": "tool",
            "content": "MANDATORY-SENTINEL-" + ("s" * 30000),
            "tool_call_id": "slice-4d-atomic",
            "name": "sub_agent_return",
            "_sub_agent_return": True,
        }
        current_user = {
            "role": "user",
            "content": "CURRENT-USER-" + ("u" * 8000),
        }
        messages = [
            {"role": "system", "content": "system"},
            sentinel,
            current_user,
        ]

        tools = [{
            "type": "function",
            "function": {
                "name": "large_schema",
                "description": "d" * 16000,
                "parameters": {"type": "object", "properties": {}},
            },
        }]

        ctx = SimpleNamespace(
            request_id="slice-4d-atomic",
            v2_boundary_idx=None,
            v2_boundary_chain=None,
        )
        conn_cm = _assembly_mock_conn()

        warm_entry = state.new_chain_state_entry()
        warm_entry["version"] = 1
        warm_entry["boundary_idx"] = 0
        warm_entry["boundary_chain"] = None
        warm_key = (session_id, model_name)

        with state._chain_state_lock:
            state._chain_state[warm_key] = warm_entry
        entry_before = copy.deepcopy(warm_entry)
        ctx_boundary_idx_before = ctx.v2_boundary_idx
        ctx_boundary_chain_before = ctx.v2_boundary_chain

        with ExitStack() as stack:
            stack.enter_context(patch("proxy.context.get_context_config", return_value=cfg))
            stack.enter_context(patch("memory.db.get_rich_summaries", return_value=[]))
            stack.enter_context(patch("memory.db.get_ancient_history", return_value=""))
            stack.enter_context(patch("proxy.vfs._build_vfs_context", return_value=None))
            stack.enter_context(patch("proxy.vfs._build_vfs_inherited_context", return_value=None))
            stack.enter_context(patch("memory.db.connection", return_value=conn_cm))
            stack.enter_context(patch(
                "memory.db_summaries.get_last_summarized_exchange_index",
                return_value=-1,
            ))
            with pytest.raises(ContextLimitError):
                _select_context_v2(
                    session_id,
                    messages,
                    model_name=model_name,
                    ctx=ctx,
                    tools=tools,
                )

        with state._chain_state_lock:
            entry_after = state._chain_state.get(warm_key)

        assert entry_after is not None, "Chain-state entry must still exist"
        assert entry_after == entry_before, (
            f"Chain-state entry must be unchanged after rejected turn; "
            f"before={entry_before!r} after={entry_after!r}"
        )
        assert ctx.v2_boundary_idx == ctx_boundary_idx_before, (
            "ctx.v2_boundary_idx must not advance on rejected turn"
        )
        assert ctx.v2_boundary_chain == ctx_boundary_chain_before, (
            "ctx.v2_boundary_chain must not advance on rejected turn"
        )

    def test_with_summaries_canonical_order_with_vfs_and_live(self):
        """With summaries + VFS: assembly is
        [sys, summaries, ancient, completed_history, vfs, live_exchange].
        Under the Wave 4a unconditional walk-back, completed_history is
        empty for an all-consecutive-users input (the splitter widens the
        live suffix back over the entire user run), but the section
        ordering must still hold when content is present.
        """
        cfg = _assembly_budget_config()
        msgs = [{"role": "system", "content": "sys"}] + _assembly_msgs(100)
        summaries = [{"message_index": 5, "role": "user", "summary": "sum", "tool_summary": ""}]
        vfs = {"role": "system", "content": "[VFS] workspace"}
        ctx = SimpleNamespace(request_id="t", v2_boundary_idx=None, v2_boundary_chain=None)
        conn_cm = _assembly_mock_conn()
        with ExitStack() as stack:
            stack.enter_context(patch("proxy.context.get_context_config", return_value=cfg))
            stack.enter_context(patch("memory.db.get_rich_summaries", return_value=summaries))
            stack.enter_context(patch("memory.db.get_ancient_history", return_value="ancient text"))
            stack.enter_context(patch("proxy.vfs._build_vfs_context", return_value=vfs))
            stack.enter_context(patch("proxy.vfs._build_vfs_inherited_context", return_value=None))
            stack.enter_context(patch("memory.db.connection", return_value=conn_cm))
            stack.enter_context(
                patch("memory.db_summaries.get_last_summarized_exchange_index", return_value=-1)
            )
            result = _select_context_v2(
                "sess1", msgs, model_name="test", ctx=ctx,
            )

        summaries_idx = next(
            i for i, m in enumerate(result) if "[PAST CONTEXT" in m.get("content", "")
        )
        ancient_idx = next(
            i for i, m in enumerate(result) if "[TIMELINE" in m.get("content", "")
        )
        vfs_idx = next(
            i for i, m in enumerate(result) if "[VFS]" in m.get("content", "")
        )
        first_user_idx = next(
            (i for i, m in enumerate(result) if m.get("role") == "user"),
            len(result),
        )

        assert summaries_idx < ancient_idx, "summaries < ancient"
        assert ancient_idx < vfs_idx, "ancient < vfs"
        assert vfs_idx < first_user_idx, (
            "vfs must precede live_exchange (canonical placement); got "
            f"vfs@{vfs_idx} first_user@{first_user_idx}"
        )

        barriers = [
            m for m in result
            if m.get("role") == "assistant" and m.get("content") == " "
        ]
        assert len(barriers) == 0, "Wave 4 must NOT emit context-layer barrier"

    def test_minimax_adapter_handles_vfs_user_adjacency(self):
        """Wave 4: V2 emits no barrier. Multi-user live_exchange is the
        caller's responsibility to handle downstream. With MiniMaxAdapter,
        the adapter must break [user][user] adjacency in the V2 output
        so the upstream provider doesn't reject with 400.
        """
        from proxy.pipeline.provider_compat import MiniMaxAdapter

        cfg = _assembly_budget_config()
        msgs = [{"role": "system", "content": "sys"}] + _assembly_msgs(100)
        summaries = [{"message_index": 5, "role": "user", "summary": "sum", "tool_summary": ""}]
        vfs = {"role": "system", "content": "[VFS] workspace"}
        model_cfg = SimpleNamespace(backend_model="MiniMax-M3")

        ctx = SimpleNamespace(request_id="t", v2_boundary_idx=None, v2_boundary_chain=None)
        conn_cm = _assembly_mock_conn()
        with ExitStack() as stack:
            stack.enter_context(patch("proxy.context.get_context_config", return_value=cfg))
            stack.enter_context(patch("memory.db.get_rich_summaries", return_value=summaries))
            stack.enter_context(patch("memory.db.get_ancient_history", return_value="ancient text"))
            stack.enter_context(patch("proxy.vfs._build_vfs_context", return_value=vfs))
            stack.enter_context(patch("proxy.vfs._build_vfs_inherited_context", return_value=None))
            stack.enter_context(patch("memory.db.connection", return_value=conn_cm))
            stack.enter_context(
                patch("memory.db_summaries.get_last_summarized_exchange_index", return_value=-1)
            )
            v2_result = _select_context_v2(
                "sess1", msgs, model_name="test", model_cfg=model_cfg, ctx=ctx,
            )

        assert not any(
            m.get("role") == "assistant" and m.get("content") == " " for m in v2_result
        ), "Wave 4 V2 output must NOT contain a context-layer barrier"

        adapter_result = MiniMaxAdapter.apply(
            {"messages": [dict(m) for m in v2_result], "temperature": 0.5},
            model_cfg,
        )["messages"]

        roles = [m.get("role") for m in adapter_result]
        for i in range(len(roles) - 1):
            assert not (roles[i] == "user" and roles[i + 1] == "user"), (
                f"Adjacent role=user at indices {i},{i+1} survived MiniMax adapter. "
                f"Roles: {roles[i-1:i+3] if i > 0 else roles[i:i+3]}"
            )

    def test_with_summaries_non_minimax_no_barrier_in_v2(self):
        """Wave 4: V2 emits no context-layer barrier regardless of adapter.
        Non-MiniMax backends likewise never see a V2-emitted barrier;
        canonical order is sys < summaries < ancient < completed < VFS < live.
        """
        cfg = _assembly_budget_config()
        msgs = [{"role": "system", "content": "sys"}] + _assembly_msgs(100)
        summaries = [{"message_index": 5, "role": "user", "summary": "sum", "tool_summary": ""}]
        vfs = {"role": "system", "content": "[VFS] workspace"}
        model_cfg = SimpleNamespace(backend_model="gpt-4o")

        ctx = SimpleNamespace(request_id="t", v2_boundary_idx=None, v2_boundary_chain=None)
        conn_cm = _assembly_mock_conn()
        with ExitStack() as stack:
            stack.enter_context(patch("proxy.context.get_context_config", return_value=cfg))
            stack.enter_context(patch("memory.db.get_rich_summaries", return_value=summaries))
            stack.enter_context(patch("memory.db.get_ancient_history", return_value="ancient text"))
            stack.enter_context(patch("proxy.vfs._build_vfs_context", return_value=vfs))
            stack.enter_context(patch("proxy.vfs._build_vfs_inherited_context", return_value=None))
            stack.enter_context(patch("memory.db.connection", return_value=conn_cm))
            stack.enter_context(
                patch("memory.db_summaries.get_last_summarized_exchange_index", return_value=-1)
            )
            result = _select_context_v2(
                "sess1", msgs, model_name="test", model_cfg=model_cfg, ctx=ctx,
            )

        barriers = [
            m for m in result
            if m.get("role") == "assistant" and m.get("content") == " "
        ]
        assert len(barriers) == 0, (
            "Wave 4 must NOT emit context-layer barrier for any adapter"
        )

        summaries_idx = next(
            i for i, m in enumerate(result) if "[PAST CONTEXT" in m.get("content", "")
        )
        ancient_idx = next(
            i for i, m in enumerate(result) if "[TIMELINE" in m.get("content", "")
        )
        vfs_idx = next(
            i for i, m in enumerate(result) if "[VFS]" in m.get("content", "")
        )
        first_user_idx = next(
            (i for i, m in enumerate(result) if m.get("role") == "user"),
            len(result),
        )
        assert summaries_idx < ancient_idx < vfs_idx < first_user_idx, (
            "Canonical order: summaries < ancient < vfs < live_exchange"
        )

    def test_slice_4e_empty_vfs_reserves_zero_history_budget(self):
        """Absent VFS must not consume the slow path's five-percent carve."""
        from proxy.context import _count_msg_chars
        from proxy.context_v2 import (
            _ancient_max_chars,
            _format_ancient,
            _format_summaries,
            _truncate_ancient,
        )
        from proxy.vfs import _build_vfs_context

        session_id = "slice-4e-empty-vfs"
        model_name = "slice-4e-model"
        workspace = "slice-4e-empty-workspace"
        cfg = {
            "safe_context_tokens": 10_000,
            "max_context_tokens": 40_000,
            "tokens_per_char": 0.25,
            "raw_exchange_count": 100,
        }

        old_user_marker = "SLICE-4E-OLD-USER-"
        old_assistant_marker = "SLICE-4E-OLD-ASSISTANT-"
        live_marker = "SLICE-4E-LIVE-USER-"

        old_user = {
            "role": "user",
            "content": (
                old_user_marker
                + "u" * (1_200 - len(old_user_marker))
            ),
        }
        old_assistant = {
            "role": "assistant",
            "content": (
                old_assistant_marker
                + "a" * (1_200 - len(old_assistant_marker))
            ),
        }
        live_user = {
            "role": "user",
            "content": (
                live_marker
                + "l" * (20_000 - len(live_marker))
            ),
        }
        messages = [
            old_user,
            old_assistant,
            live_user,
        ]
        before = copy.deepcopy(messages)

        summary = {
            "message_index": 200,
            "role": "user",
            "summary": "S" * 5_761,
            "tool_summary": "",
        }
        summary_content = _format_summaries([summary])
        assert len(summary_content) == 6_000
        assert int(len(summary_content) * 0.25) == 1_500

        ancient = _truncate_ancient(
            "A" * 10_000,
            _ancient_max_chars(10_000, 0.25),
            session_id,
        )
        ancient_content = _format_ancient(ancient)
        assert len(ancient) == 3_943
        assert len(ancient_content) == 4_000
        assert int(len(ancient_content) * 0.25) == 1_000

        tools = [{
            "type": "function",
            "function": {
                "name": "tool",
                "description": "d" * 5_888,
                "parameters": {
                    "type": "object",
                    "properties": {},
                },
            },
        }]
        import json
        encoded_tools = json.dumps(
            tools,
            separators=(",", ":"),
        )
        assert len(encoded_tools) == 6_000
        assert int(len(encoded_tools) * 0.25) == 1_500

        ctx = SimpleNamespace(
            request_id="slice-4e-empty-vfs",
            session_id=session_id,
            client_hint="",
            workspace_name=workspace,
            is_subagent=False,
            v2_boundary_idx=None,
            v2_boundary_chain=None,
        )

        warm_entry = state.new_chain_state_entry()
        warm_entry["version"] = 1
        warm_entry["boundary_idx"] = 0
        warm_entry["boundary_chain"] = None
        warm_entry["summaries"] = [summary]
        warm_entry["ancient"] = ancient

        key = (session_id, model_name)
        with state._chain_state_lock:
            state._chain_state[key] = warm_entry

        try:
            with ExitStack() as stack:
                stack.enter_context(
                    patch(
                        "proxy.context.get_context_config",
                        return_value=cfg,
                    )
                )
                stack.enter_context(
                    patch(
                        "proxy.workspace_state.resolve_inherited_workspace",
                        return_value=workspace,
                    )
                )
                vfs_builder = stack.enter_context(
                    patch(
                        "proxy.vfs._build_vfs_context",
                        wraps=_build_vfs_context,
                    )
                )
                result = _select_context_v2(
                    session_id,
                    messages,
                    model_name=model_name,
                    ctx=ctx,
                    tools=tools,
                )
        finally:
            with state._chain_state_lock:
                state._chain_state.pop(key, None)

        assert messages == before
        vfs_builder.assert_called_once()
        assert not any(
            "[VFS CONTEXT" in message.get("content", "")
            for message in result
        )

        assert any(
            message.get("content", "").startswith(old_user_marker)
            for message in result
        )
        assert any(
            message.get("content", "").startswith(
                old_assistant_marker
            )
            for message in result
        )
        assert any(
            message.get("content", "").startswith(live_marker)
            for message in result
        )

        message_tokens = int(
            sum(
                _count_msg_chars(message)
                for message in result
            )
            * 0.25
        )
        assert message_tokens == 8_100
        assert message_tokens + 1_500 == 9_600


class TestSubagentVfsInheritedWiring:
    """Wire `_build_vfs_inherited_context` into the sub-agent path.

    The function previously had zero production callers — the only guarded
    call site (`if not is_subagent:`) sat after the sub-agent early return.
    These tests exercise the production path
    `_select_context_v2(..., is_subagent=True, ctx=ctx)` and assert the
    returned message list (NOT a direct call to the function), so a wiring
    regression fails the test.
    """

    @pytest.fixture(autouse=True)
    def _vfs_sandbox(self, monkeypatch):
        monkeypatch.setattr(state, "VFS_ENABLED", True)
        monkeypatch.setattr(state, "VFS_SCOPE", "all")
        monkeypatch.setattr(state, "VFS_TTL_SECONDS", 60)
        with state.vfs_lock:
            state.vfs_caches.clear()
            state.vfs_write_log.clear()
        yield
        with state.vfs_lock:
            state.vfs_caches.clear()
            state.vfs_write_log.clear()

    @staticmethod
    def _cfg():
        return {
            "max_context_tokens": 40_000,
            "tokens_per_char": 0.25,
            "safe_context_tokens": 10_000,
            "raw_exchange_count": 100,
        }

    def test_inherited_block_injected_when_cache_populated(self):
        """Wiring proof: with a populated VFS cache, the sub-agent context
        must contain a `[VFS INHERITED CONTEXT` message."""
        from proxy.vfs import _cache_file

        _cache_file("src/app.py", "content", "default", "parent_session")

        ctx = SimpleNamespace(
            is_subagent=True,
            session_id="child_session",
            client_hint="opencode",
            request_id="req-wiring-1",
            parent_agent_id="parent_session",
        )
        messages = [_system("sys"), _user("do the task")]

        with ExitStack() as stack:
            stack.enter_context(
                patch("proxy.context.get_context_config", return_value=self._cfg())
            )
            stack.enter_context(patch("memory.db.get_ancient_history", return_value=""))
            stack.enter_context(patch("memory.db.get_rich_summaries", return_value=[]))
            mock_ws = stack.enter_context(patch("proxy.workspace_state.workspace_store"))
            mock_ws.get_active.return_value = "default"
            result = _select_context_v2(
                "child_session", messages, model_name="test",
                is_subagent=True, ctx=ctx,
            )

        inherited = [
            m for m in result
            if "[VFS INHERITED CONTEXT" in m.get("content", "")
        ]
        assert inherited, (
            f"Expected a [VFS INHERITED CONTEXT] message in the sub-agent "
            f"context; roles returned: {[m.get('role') for m in result]}"
        )

    def test_no_inherited_block_when_cache_empty(self):
        """Empty-cache no-op: with no cached entries, no inherited message
        must appear — proves the wiring consults the function rather than
        hardcoding output."""
        ctx = SimpleNamespace(
            is_subagent=True,
            session_id="child_session",
            client_hint="opencode",
            request_id="req-wiring-2",
            parent_agent_id="parent_session",
        )
        messages = [_system("sys"), _user("do the task")]

        with ExitStack() as stack:
            stack.enter_context(
                patch("proxy.context.get_context_config", return_value=self._cfg())
            )
            stack.enter_context(patch("memory.db.get_ancient_history", return_value=""))
            stack.enter_context(patch("memory.db.get_rich_summaries", return_value=[]))
            mock_ws = stack.enter_context(patch("proxy.workspace_state.workspace_store"))
            mock_ws.get_active.return_value = "default"
            result = _select_context_v2(
                "child_session", messages, model_name="test",
                is_subagent=True, ctx=ctx,
            )

        inherited = [
            m for m in result
            if "[VFS INHERITED CONTEXT" in m.get("content", "")
        ]
        assert not inherited, (
            f"No [VFS INHERITED CONTEXT] expected with empty cache; "
            f"got: {[m.get('content', '')[:40] for m in inherited]}"
        )

    def test_inherited_build_exception_is_swallowed(self):
        """Exception path: when `_build_vfs_inherited_context` raises, the
        try/except in `_select_context_v2` must swallow it (mirroring the
        main-agent VFS block) so the request still succeeds with no
        inherited message."""
        ctx = SimpleNamespace(
            is_subagent=True,
            session_id="child_session",
            client_hint="opencode",
            request_id="req-wiring-3",
            parent_agent_id="parent_session",
        )
        messages = [_system("sys"), _user("do the task")]

        def _boom(*args, **kwargs):
            raise RuntimeError("simulated inherited-context failure")

        with ExitStack() as stack:
            stack.enter_context(
                patch("proxy.context.get_context_config", return_value=self._cfg())
            )
            stack.enter_context(patch("memory.db.get_ancient_history", return_value=""))
            stack.enter_context(patch("memory.db.get_rich_summaries", return_value=[]))
            stack.enter_context(
                patch("proxy.vfs._build_vfs_inherited_context", side_effect=_boom)
            )
            result = _select_context_v2(
                "child_session", messages, model_name="test",
                is_subagent=True, ctx=ctx,
            )

        assert isinstance(result, list), (
            "Request must still succeed when inherited-context build raises"
        )
        inherited = [
            m for m in result
            if "[VFS INHERITED CONTEXT" in m.get("content", "")
        ]
        assert not inherited, (
            "No inherited message expected when the builder raised"
        )


class TestMainVfsRealRequestContext:
    """US-058 + AC-12: Wave 2 — main VFS must receive the real RequestContext,
    including client_hint, so client-level workspace routing reaches the same
    named workspace used for cache population. ctx=None skips VFS while
    preserving the rest of normal assembly.
    """

    CLIENT_HINT = "wave2-client-hint"
    NAMED_WORKSPACE = "wave2-alpha"
    UNRELATED_WORKSPACE = "wave2-beta"
    SEED_SESSION = "wave2-seed-session"
    TARGET_SESSION = "wave2-target-session"
    NONE_SESSION = "wave2-none-session"
    PARITY_SESSION = "wave2-parity-session"
    MODEL_NAME = "wave2-test-model"

    @pytest.fixture(autouse=True)
    def _wave2_state(self, monkeypatch):
        monkeypatch.setattr(state, "VFS_ENABLED", True)
        monkeypatch.setattr(state, "VFS_SCOPE", "all")
        monkeypatch.setattr(state, "VFS_TTL_SECONDS", 60)
        monkeypatch.setattr(state, "VFS_MAX_ENTRIES", 500)
        monkeypatch.setattr(state, "VFS_MAX_ENTRY_BYTES", 1_048_576)
        monkeypatch.setattr(state, "VFS_MAX_WORKSPACES", 20)

        session_ids = (
            self.SEED_SESSION,
            self.TARGET_SESSION,
            self.NONE_SESSION,
            self.PARITY_SESSION,
        )

        for session_id in session_ids:
            state.workspace_store.deactivate(session_id)
            state.workspace_store.pop_session(session_id)

        state.workspace_store.deactivate(
            self.SEED_SESSION,
            client_hint=self.CLIENT_HINT,
        )

        with state.vfs_lock:
            state.vfs_caches.clear()
            state.vfs_write_log.clear()
            state.vfs_rollback_snapshots.clear()

        yield

        for session_id in session_ids:
            state.workspace_store.deactivate(session_id)
            state.workspace_store.pop_session(session_id)

        state.workspace_store.deactivate(
            self.SEED_SESSION,
            client_hint=self.CLIENT_HINT,
        )

        with state.vfs_lock:
            state.vfs_caches.clear()
            state.vfs_write_log.clear()
            state.vfs_rollback_snapshots.clear()

    @staticmethod
    def _messages():
        return [
            {"role": "system", "content": "wave2 system"},
            {"role": "user", "content": "completed user"},
            {"role": "assistant", "content": "completed assistant"},
            {"role": "user", "content": "current user"},
        ]

    @staticmethod
    def _cfg():
        return {
            "max_context_tokens": 40_000,
            "tokens_per_char": 0.25,
            "safe_context_tokens": 10_000,
            "raw_exchange_count": 100,
        }

    @staticmethod
    def _ctx(session_id, client_hint, messages, workspace_name):
        from proxy.types import RequestContext
        return RequestContext(
            body={"messages": messages},
            model_name=TestMainVfsRealRequestContext.MODEL_NAME,
            wants_stream=False,
            session_id=session_id,
            client_hint=client_hint,
            messages=messages,
            remote_addr="127.0.0.1",
            workspace_name=workspace_name,
        )

    @classmethod
    def _install_warm_state(cls, session_id, messages, model_name):
        from proxy.chain_hash import compute_chain as _compute_chain
        conversation = messages[1:]
        chain, _ = _compute_chain(conversation)

        entry = state.new_chain_state_entry()
        entry.update(
            {
                "last_chain": list(chain),
                "boundary_idx": 1,
                "boundary_chain": chain[0],
                "summaries": [],
                "ancient": "",
                "version": 1,
            }
        )

        with state._chain_state_lock:
            state._chain_state[(session_id, model_name)] = entry

    def test_real_request_context_routes_main_vfs_by_client_hint(
        self,
        monkeypatch,
    ):
        """Wave 2 AC-12: real ctx propagates client_hint so the resolver
        selects the same named workspace used for cache population."""
        from proxy.vfs import _cache_file as _cache

        messages = self._messages()

        state.workspace_store.activate(
            self.SEED_SESSION,
            self.NAMED_WORKSPACE,
            client_hint=self.CLIENT_HINT,
        )
        state.workspace_store.pop_session(self.SEED_SESSION)

        active, _ = state.workspace_store.snapshot(self.TARGET_SESSION)
        assert active == "default"

        _cache(
            "/wave2/default-only.py",
            "default cache",
            "default",
            self.SEED_SESSION,
        )
        _cache(
            "/wave2/alpha-only.py",
            "alpha cache",
            self.NAMED_WORKSPACE,
            self.SEED_SESSION,
        )
        _cache(
            "/wave2/beta-only.py",
            "beta cache",
            self.UNRELATED_WORKSPACE,
            self.SEED_SESSION,
        )

        self._install_warm_state(
            self.TARGET_SESSION, messages, self.MODEL_NAME,
        )

        ctx = self._ctx(
            self.TARGET_SESSION,
            self.CLIENT_HINT,
            messages,
            workspace_name=self.NAMED_WORKSPACE,
        )

        monkeypatch.setattr(
            "proxy.context.get_context_config",
            lambda *_args, **_kwargs: self._cfg(),
        )

        result = _select_context_v2(
            self.TARGET_SESSION,
            messages,
            model_name=self.MODEL_NAME,
            ctx=ctx,
        )

        vfs_messages = [
            message
            for message in result
            if "[VFS CONTEXT" in message.get("content", "")
        ]

        assert len(vfs_messages) == 1
        content = vfs_messages[0]["content"]

        assert f"Workspace: {self.NAMED_WORKSPACE}" in content
        assert "alpha-only.py" in content
        assert "default-only.py" not in content
        assert "beta-only.py" not in content

        active, _ = state.workspace_store.snapshot(self.TARGET_SESSION)
        assert active == self.NAMED_WORKSPACE

        assert ctx.v2_boundary_idx == 1
        assert ctx.v2_boundary_chain is not None

    def test_ctx_none_skips_vfs_and_preserves_normal_assembly(
        self,
        monkeypatch,
    ):
        """Wave 2: ctx=None skips VFS construction; the rest of normal
        assembly is byte-identical to the real-context baseline routed by
        client_hint to an EMPTY named workspace."""
        from proxy.vfs import _cache_file as _cache

        messages = self._messages()

        state.workspace_store.activate(
            self.SEED_SESSION,
            self.NAMED_WORKSPACE,
            client_hint=self.CLIENT_HINT,
        )
        state.workspace_store.pop_session(self.SEED_SESSION)

        state.workspace_store.set_active(self.NONE_SESSION, "default")

        _cache(
            "/wave2/default-only.py",
            "default cache",
            "default",
            self.SEED_SESSION,
        )

        self._install_warm_state(
            self.NONE_SESSION, messages, self.MODEL_NAME,
        )
        self._install_warm_state(
            self.PARITY_SESSION, messages, self.MODEL_NAME,
        )

        real_ctx = self._ctx(
            self.PARITY_SESSION,
            self.CLIENT_HINT,
            messages,
            workspace_name=self.NAMED_WORKSPACE,
        )

        monkeypatch.setattr(
            "proxy.context.get_context_config",
            lambda *_args, **_kwargs: self._cfg(),
        )

        without_ctx = _select_context_v2(
            self.NONE_SESSION,
            messages,
            model_name=self.MODEL_NAME,
            ctx=None,
        )

        with_ctx = _select_context_v2(
            self.PARITY_SESSION,
            messages,
            model_name=self.MODEL_NAME,
            ctx=real_ctx,
        )

        assert without_ctx == with_ctx
        assert without_ctx
        assert without_ctx[0] == {
            "role": "system",
            "content": "wave2 system",
        }
        assert any(
            message.get("content") == "current user"
            for message in without_ctx
        )
        assert not any(
            "[VFS CONTEXT" in message.get("content", "")
            for message in without_ctx
        )

        assert real_ctx.v2_boundary_idx == 1
        assert real_ctx.v2_boundary_chain is not None


class TestMainVfsCacheSafeTailPlacement:
    """Wave 4: canonical normal-path placement under cache-safe tail spec (US-127).

    Verifies main VFS appears AFTER completed retained history and BEFORE
    the entire live user-led exchange, with no synthetic context-layer
    barrier emitted solely because VFS exists.
    """

    CLIENT_HINT = "wave4-client-hint"
    NAMED_WORKSPACE = "wave4-alpha"
    SEED_SESSION = "wave4-seed-session"
    TARGET_SESSION = "wave4-target-session"
    MODEL_NAME = "wave4-test-model"

    @pytest.fixture(autouse=True)
    def _wave4_state(self, monkeypatch):
        monkeypatch.setattr(state, "VFS_ENABLED", True)
        monkeypatch.setattr(state, "VFS_SCOPE", "all")
        monkeypatch.setattr(state, "VFS_TTL_SECONDS", 60)
        monkeypatch.setattr(state, "VFS_MAX_ENTRIES", 500)
        monkeypatch.setattr(state, "VFS_MAX_ENTRY_BYTES", 1_048_576)
        monkeypatch.setattr(state, "VFS_MAX_WORKSPACES", 20)

        session_ids = (self.SEED_SESSION, self.TARGET_SESSION)
        for session_id in session_ids:
            state.workspace_store.deactivate(session_id)
            state.workspace_store.pop_session(session_id)
        state.workspace_store.deactivate(
            self.SEED_SESSION, client_hint=self.CLIENT_HINT,
        )

        with state.vfs_lock:
            state.vfs_caches.clear()
            state.vfs_write_log.clear()
            state.vfs_rollback_snapshots.clear()

        yield

        for session_id in session_ids:
            state.workspace_store.deactivate(session_id)
            state.workspace_store.pop_session(session_id)
        state.workspace_store.deactivate(
            self.SEED_SESSION, client_hint=self.CLIENT_HINT,
        )
        with state.vfs_lock:
            state.vfs_caches.clear()
            state.vfs_write_log.clear()
            state.vfs_rollback_snapshots.clear()

    @staticmethod
    def _cfg():
        return {
            "max_context_tokens": 40_000,
            "tokens_per_char": 0.25,
            "safe_context_tokens": 10_000,
            "raw_exchange_count": 100,
        }

    @staticmethod
    def _ctx(session_id, client_hint, messages, workspace_name):
        return RequestContext(
            body={"messages": messages},
            model_name=TestMainVfsCacheSafeTailPlacement.MODEL_NAME,
            wants_stream=False,
            session_id=session_id,
            client_hint=client_hint,
            messages=messages,
            remote_addr="127.0.0.1",
            workspace_name=workspace_name,
        )

    @classmethod
    def _install_warm_state(cls, session_id, messages, model_name,
                             boundary_idx=0, boundary_chain=None):
        from proxy.chain_hash import compute_chain
        conversation = messages[1:] if messages[0].get("role") == "system" else messages
        chain, _ = compute_chain(conversation)
        if boundary_idx == 0:
            entry_chain = None
        else:
            entry_chain = boundary_chain if boundary_chain is not None else (
                chain[boundary_idx - 1] if boundary_idx - 1 < len(chain) else None
            )
        entry = state.new_chain_state_entry()
        entry.update({
            "last_chain": list(chain),
            "boundary_idx": boundary_idx,
            "boundary_chain": entry_chain,
            "summaries": [],
            "ancient": "",
            "version": 1,
        })
        with state._chain_state_lock:
            state._chain_state[(session_id, model_name)] = entry

    def _setup_workspace_with_vfs(self):
        from proxy.vfs import _cache_file
        state.workspace_store.activate(
            self.SEED_SESSION,
            self.NAMED_WORKSPACE,
            client_hint=self.CLIENT_HINT,
        )
        state.workspace_store.pop_session(self.SEED_SESSION)
        _cache_file(
            "/wave4/alpha-only.py",
            "alpha cache",
            self.NAMED_WORKSPACE,
            self.SEED_SESSION,
        )
        _cache_file(
            "/wave4/default-only.py",
            "default cache",
            "default",
            self.SEED_SESSION,
        )

    def test_ordinary_turn_places_vfs_after_completed_before_live(self, monkeypatch):
        self._setup_workspace_with_vfs()
        messages = [
            {"role": "system", "content": "wave4 system"},
            {"role": "user", "content": "old user"},
            {"role": "assistant", "content": "old assistant"},
            {"role": "user", "content": "current user"},
        ]
        self._install_warm_state(
            self.TARGET_SESSION, messages, self.MODEL_NAME,
        )
        ctx = self._ctx(
            self.TARGET_SESSION, self.CLIENT_HINT, messages,
            workspace_name=self.NAMED_WORKSPACE,
        )
        monkeypatch.setattr(
            "proxy.context.get_context_config",
            lambda *_a, **_kw: self._cfg(),
        )

        result = _select_context_v2(
            self.TARGET_SESSION, messages,
            model_name=self.MODEL_NAME, ctx=ctx,
        )

        def _idx(predicate):
            for i, m in enumerate(result):
                if predicate(m):
                    return i
            return -1

        sys_idx = _idx(lambda m: m.get("role") == "system"
                       and "wave4 system" in m.get("content", ""))
        old_user_idx = _idx(lambda m: m.get("content") == "old user")
        old_asst_idx = _idx(lambda m: m.get("content") == "old assistant")
        vfs_idx = _idx(lambda m: m.get("role") == "system"
                       and "[VFS" in m.get("content", ""))
        curr_idx = _idx(lambda m: m.get("content") == "current user")

        assert sys_idx >= 0 and old_user_idx >= 0 and old_asst_idx >= 0
        assert vfs_idx >= 0 and curr_idx >= 0
        assert sys_idx < old_user_idx < old_asst_idx < vfs_idx < curr_idx

        for m in result:
            if m.get("role") == "assistant" and m.get("content") == " ":
                pytest.fail(f"Wave 4 must not emit context-layer barrier; found: {m}")

    def test_multi_tool_continuation_preserves_assistant_tool_atomicity(self, monkeypatch):
        self._setup_workspace_with_vfs()
        messages = [
            {"role": "system", "content": "wave4 system"},
            {"role": "user", "content": "old user"},
            {"role": "assistant", "content": "old assistant"},
            {"role": "user", "content": "current user"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {"id": "A", "type": "function",
                     "function": {"name": "read", "arguments": "{}"}},
                    {"id": "B", "type": "function",
                     "function": {"name": "read", "arguments": "{}"}},
                ],
            },
            {"role": "tool", "tool_call_id": "A", "content": "result A"},
            {"role": "tool", "tool_call_id": "B", "content": "result B"},
        ]
        self._install_warm_state(
            self.TARGET_SESSION, messages, self.MODEL_NAME,
        )
        ctx = self._ctx(
            self.TARGET_SESSION, self.CLIENT_HINT, messages,
            workspace_name=self.NAMED_WORKSPACE,
        )
        monkeypatch.setattr(
            "proxy.context.get_context_config",
            lambda *_a, **_kw: self._cfg(),
        )

        result = _select_context_v2(
            self.TARGET_SESSION, messages,
            model_name=self.MODEL_NAME, ctx=ctx,
        )

        asst_tc_idx = next(
            i for i, m in enumerate(result)
            if m.get("role") == "assistant"
            and isinstance(m.get("tool_calls"), list)
            and len(m["tool_calls"]) == 2
        )
        tool_a_idx = next(
            i for i, m in enumerate(result[asst_tc_idx + 1:], asst_tc_idx + 1)
            if m.get("role") == "tool" and m.get("tool_call_id") == "A"
        )
        tool_b_idx = next(
            i for i, m in enumerate(result[asst_tc_idx + 1:], asst_tc_idx + 1)
            if m.get("role") == "tool" and m.get("tool_call_id") == "B"
        )
        curr_idx = next(
            i for i, m in enumerate(result)
            if m.get("content") == "current user"
        )

        assert tool_a_idx == asst_tc_idx + 1
        assert tool_b_idx == asst_tc_idx + 2
        assert curr_idx < asst_tc_idx

    def test_consecutive_current_users_stay_in_live_suffix(self, monkeypatch):
        self._setup_workspace_with_vfs()
        messages = [
            {"role": "system", "content": "wave4 system"},
            {"role": "assistant", "content": "completed"},
            {"role": "user", "content": "current one"},
            {"role": "user", "content": "current two"},
        ]
        self._install_warm_state(
            self.TARGET_SESSION, messages, self.MODEL_NAME,
        )
        ctx = self._ctx(
            self.TARGET_SESSION, self.CLIENT_HINT, messages,
            workspace_name=self.NAMED_WORKSPACE,
        )
        monkeypatch.setattr(
            "proxy.context.get_context_config",
            lambda *_a, **_kw: self._cfg(),
        )

        result = _select_context_v2(
            self.TARGET_SESSION, messages,
            model_name=self.MODEL_NAME, ctx=ctx,
        )

        completed_idx = next(
            i for i, m in enumerate(result) if m.get("content") == "completed"
        )
        curr1_idx = next(
            i for i, m in enumerate(result) if m.get("content") == "current one"
        )
        curr2_idx = next(
            i for i, m in enumerate(result) if m.get("content") == "current two"
        )
        vfs_idx = next(
            i for i, m in enumerate(result)
            if m.get("role") == "system" and "[VFS" in m.get("content", "")
        )

        assert completed_idx < vfs_idx < curr1_idx < curr2_idx

    def test_no_user_normal_path_skips_vfs(self, monkeypatch):
        from proxy.vfs import _cache_file

        state.workspace_store.activate(
            self.SEED_SESSION, self.NAMED_WORKSPACE,
            client_hint=self.CLIENT_HINT,
        )
        state.workspace_store.pop_session(self.SEED_SESSION)
        _cache_file(
            "/wave4/alpha.py", "alpha cache",
            self.NAMED_WORKSPACE, self.SEED_SESSION,
        )

        messages = [
            {"role": "system", "content": "wave4 system"},
            {"role": "assistant", "content": "alone assistant"},
        ]
        self._install_warm_state(
            self.TARGET_SESSION, messages, self.MODEL_NAME, boundary_idx=0,
        )
        ctx = self._ctx(
            self.TARGET_SESSION, self.CLIENT_HINT, messages,
            workspace_name=self.NAMED_WORKSPACE,
        )
        monkeypatch.setattr(
            "proxy.context.get_context_config",
            lambda *_a, **_kw: self._cfg(),
        )

        result = _select_context_v2(
            self.TARGET_SESSION, messages,
            model_name=self.MODEL_NAME, ctx=ctx,
        )

        assert not any(
            "[VFS" in m.get("content", "")
            for m in result
        ), f"No-user path must not emit VFS; got: {[m.get('content','')[:40] for m in result]}"


class _FastPathNoSummaryConn:
    """Context manager mock for memory.db.connection on the fast path.

    Records both summary-existence queries and returns no rows for either,
    routing the fresh-session fast return. Fails on unexpected SQL so a
    hidden fallthrough cannot pass silently.
    """

    def __init__(self):
        self.queries = []

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        return False

    def execute(self, sql, params=None):
        self.queries.append((sql, params))
        if "FROM conversation_summaries" in sql:
            return self
        if "FROM message_summaries" in sql:
            return self
        raise AssertionError(f"Unexpected SQL on fast path: {sql}")

    def fetchone(self):
        return None


class TestFreshSessionVfsFastPath:
    """Wave 5: fresh-session VFS fast path on the cleaned-fit branch.

    Pins the invariants documented in the spec: when the cleaned input
    fits the safe-token budget AND the session has no prior chain state
    AND no conversation/message summaries, the V2 selector may invoke
    the real VFS builder and emit VFS in canonical tail position before
    falling back to the cleaned-fast return on overflow.
    """

    @pytest.fixture(autouse=True)
    def _wave5_fast_path_state(self, monkeypatch):
        """Isolate VFS caches, workspace mappings, and Tier-2 instructions per test."""
        monkeypatch.setattr(state, "VFS_ENABLED", True)
        monkeypatch.setattr(state, "VFS_SCOPE", "all")
        monkeypatch.setattr(state, "VFS_TTL_SECONDS", 3600)
        monkeypatch.setattr(state, "VFS_MAX_ENTRIES", 500)
        monkeypatch.setattr(state, "VFS_MAX_ENTRY_BYTES", 1_048_576)
        monkeypatch.setattr(state, "VFS_MAX_WORKSPACES", 20)

        yield

        session_ids = (
            *getattr(self, "_wave5_session_ids", ()),
            getattr(self, "_wave5_target_session", ""),
            getattr(self, "_wave5_seed_session", ""),
        )
        unique = tuple(s for s in set(session_ids) if s)
        for sid in unique:
            try:
                state.workspace_store.deactivate(sid)
                state.workspace_store.pop_session(sid)
            except Exception:
                pass
        client_hint = getattr(self, "_wave5_client_hint", "")
        if client_hint:
            try:
                state.workspace_store.deactivate(
                    getattr(self, "_wave5_seed_session", ""),
                    client_hint=client_hint,
                )
            except Exception:
                pass
        with state.vfs_lock:
            state.vfs_caches.clear()
            state.vfs_write_log.clear()
            state.vfs_rollback_snapshots.clear()
        with state.vfs_lock:
            state.instruction_files_by_workspace.clear()
            state._last_scanned_by_tool_call.clear()
        chain_pairs = (
            (getattr(self, "_wave5_target_session", ""),
             getattr(self, "_wave5_model_name", "")),
            (getattr(self, "_wave5_seed_session", ""),
             getattr(self, "_wave5_model_name", "")),
        )
        chain_pairs = tuple(p for p in chain_pairs if p[0] and p[1])
        with state._chain_state_lock:
            for pair in chain_pairs:
                state._chain_state.pop(pair, None)

    @staticmethod
    def _cfg(safe=10_000, tpc=0.25):
        return {
            "max_context_tokens": 40_000,
            "tokens_per_char": tpc,
            "safe_context_tokens": safe,
            "raw_exchange_count": 100,
        }

    @staticmethod
    def _ctx(session_id, model_name, client_hint, workspace_name, messages):
        return RequestContext(
            body={"messages": messages},
            model_name=model_name,
            wants_stream=False,
            session_id=session_id,
            client_hint=client_hint,
            messages=messages,
            remote_addr="127.0.0.1",
            workspace_name=workspace_name,
        )

    @staticmethod
    def _assert_no_chain_state(session_id, model_name):
        with state._chain_state_lock:
            assert (session_id, model_name) not in state._chain_state, (
                f"Fresh fast path must not insert chain-state for "
                f"({session_id!r}, {model_name!r}); found "
                f"{state._chain_state.get((session_id, model_name))!r}"
            )

    @staticmethod
    def _assert_tier2_pin_present(result, sentinel_content):
        pin_indices = [
            i for i, m in enumerate(result)
            if m.get("role") == "system" and "AGENTS.md" in m.get("content", "")
        ]
        assert len(pin_indices) == 1, (
            f"Exactly one Tier-2 pin expected; got positions={pin_indices}"
        )
        pin_idx = pin_indices[0]
        assert sentinel_content in result[pin_idx]["content"], (
            f"Stable sentinel missing from Tier-2 pin: "
            f"{result[pin_idx].get('content', '')[:200]!r}"
        )

    def test_empty_vfs_preserves_fresh_fast_return(self, monkeypatch):
        """Empty resolved VFS + base fits + no chain-state → preserves cheap
        cleaned fast return. No VFS, no summaries/ancient, no chain-state key."""
        from proxy.context_v2 import _cleanup_tool_pair_integrity
        session_id = "wave5-empty-session"
        model_name = "wave5-empty-model"
        client_hint = "wave5-empty-hint"
        workspace_name = "wave5-empty-alpha"
        self._wave5_session_ids = (session_id,)
        self._wave5_target_session = session_id
        self._wave5_seed_session = "wave5-empty-seed"
        self._wave5_model_name = model_name
        self._wave5_client_hint = client_hint

        messages = [
            {"role": "system", "content": "wave5 system"},
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi back"},
        ]
        cleaned_expected = _cleanup_tool_pair_integrity(messages)

        conn_mock = _FastPathNoSummaryConn()
        monkeypatch.setattr(
            "proxy.context.get_context_config",
            lambda *_a, **_kw: self._cfg(),
        )

        def _no_summary_conn(*_a, **_kw):
            return conn_mock

        monkeypatch.setattr("memory.db.connection", _no_summary_conn)

        ctx = self._ctx(
            session_id, model_name, client_hint, workspace_name, messages,
        )

        from proxy.context_v2 import _select_context_v2
        result = _select_context_v2(
            session_id, messages, model_name=model_name, ctx=ctx,
        )

        assert not any(
            "[VFS" in m.get("content", "") for m in result
        ), (
            f"Empty VFS path must not emit VFS; got roles="
            f"{[m.get('role') for m in result]}"
        )
        assert not any(
            m.get("role") == "assistant" and m.get("content") == " "
            for m in result
        ), "Wave 4 invariant: no context-layer barrier"
        non_tier2 = [m for m in result if m.get("role") != "system"
                     or "AGENTS.md" not in m.get("content", "")]
        assert non_tier2 == cleaned_expected, (
            f"Non-Tier-2 output must equal cleaned; got diff: "
            f"{[(i, e, r) for i, (e, r) in enumerate(zip(cleaned_expected, non_tier2)) if e != r]}"
        )
        self._assert_no_chain_state(session_id, model_name)
        assert ctx.v2_boundary_idx is None
        assert ctx.v2_boundary_chain is None
        assert len(conn_mock.queries) == 2
        assert any("FROM conversation_summaries" in q[0] for q in conn_mock.queries)
        assert any("FROM message_summaries" in q[0] for q in conn_mock.queries)

    def test_warm_vfs_uses_canonical_tail_on_fresh_fast_path(self, monkeypatch):
        """Warm populated named-workspace VFS + base fits → real builder
        runs, canonical-tail assembly order, exact ordering, no chain-state
        initialization. Revert A's primary discriminator."""
        from proxy.vfs import _cache_file
        session_id = "wave5-warm-session"
        model_name = "wave5-warm-model"
        client_hint = "wave5-warm-hint"
        workspace_name = "wave5-warm-alpha"
        seed_session = "wave5-warm-seed"
        self._wave5_session_ids = (session_id, seed_session)
        self._wave5_target_session = session_id
        self._wave5_seed_session = seed_session
        self._wave5_model_name = model_name
        self._wave5_client_hint = client_hint

        state.workspace_store.activate(
            seed_session, workspace_name, client_hint=client_hint,
        )
        state.workspace_store.pop_session(seed_session)
        _cache_file("/wave5/alpha-only.py", "alpha cache",
                    workspace_name, seed_session)
        _cache_file("/wave5/default-only.py", "default cache",
                    "default", seed_session)
        _cache_file("/wave5/beta-only.py", "beta cache",
                    "wave5-warm-beta", seed_session)

        messages = [
            {"role": "system", "content": "wave5 system"},
            {"role": "user", "content": "old user"},
            {"role": "assistant", "content": "old assistant"},
            {"role": "user", "content": "current user"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {"id": "A", "type": "function",
                     "function": {"name": "read", "arguments": "{}"}},
                    {"id": "B", "type": "function",
                     "function": {"name": "read", "arguments": "{}"}},
                ],
            },
            {"role": "tool", "tool_call_id": "A", "content": "result A"},
            {"role": "tool", "tool_call_id": "B", "content": "result B"},
        ]
        messages_before = copy.deepcopy(messages)

        conn_mock = _FastPathNoSummaryConn()
        monkeypatch.setattr(
            "proxy.context.get_context_config",
            lambda *_a, **_kw: self._cfg(),
        )
        monkeypatch.setattr("memory.db.connection", lambda *_a, **_kw: conn_mock)

        ctx = self._ctx(
            session_id, model_name, client_hint, workspace_name, messages,
        )

        from proxy.context_v2 import _select_context_v2
        result = _select_context_v2(
            session_id, messages, model_name=model_name, ctx=ctx,
        )

        def _idx(predicate):
            for i, m in enumerate(result):
                if predicate(m):
                    return i
            return -1

        sys_idx = _idx(lambda m: m.get("role") == "system"
                       and "wave5 system" in m.get("content", ""))
        old_user_idx = _idx(lambda m: m.get("content") == "old user")
        old_asst_idx = _idx(lambda m: m.get("content") == "old assistant")
        vfs_idx = _idx(lambda m: m.get("role") == "system"
                       and "[VFS" in m.get("content", ""))
        curr_idx = _idx(lambda m: m.get("content") == "current user")
        asst_tc_idx = _idx(lambda m: m.get("role") == "assistant"
                           and isinstance(m.get("tool_calls"), list)
                           and len(m["tool_calls"]) == 2)
        tool_a_idx = _idx(lambda m: m.get("role") == "tool"
                          and m.get("tool_call_id") == "A")
        tool_b_idx = _idx(lambda m: m.get("role") == "tool"
                          and m.get("tool_call_id") == "B")

        assert sys_idx >= 0
        assert old_user_idx >= 0
        assert old_asst_idx >= 0
        assert vfs_idx >= 0
        assert curr_idx >= 0
        assert asst_tc_idx >= 0
        assert tool_a_idx >= 0
        assert tool_b_idx >= 0

        assert sys_idx < old_user_idx < old_asst_idx < vfs_idx < curr_idx < asst_tc_idx
        assert tool_a_idx == asst_tc_idx + 1
        assert tool_b_idx == asst_tc_idx + 2

        vfs_msg = result[vfs_idx]
        assert "alpha-only.py" in vfs_msg["content"]
        assert "default-only.py" not in vfs_msg["content"]
        assert "beta-only.py" not in vfs_msg["content"]

        assert not any(
            m.get("role") == "assistant" and m.get("content") == " "
            for m in result
        ), "Wave 4 invariant: no context-layer barrier"

        self._assert_no_chain_state(session_id, model_name)
        assert ctx.v2_boundary_idx is None
        assert ctx.v2_boundary_chain is None

        assert messages == messages_before, (
            "Input messages must not be mutated by the fast path"
        )

    def test_ctx_none_skips_populated_vfs_on_fresh_fast_path(self, monkeypatch):
        """ctx=None + populated VFS → cleaned fast return, no VFS emitted,
        no fabricated context."""
        from proxy.vfs import _cache_file
        from proxy.context_v2 import _cleanup_tool_pair_integrity
        session_id = "wave5-ctxnone-session"
        model_name = "wave5-ctxnone-model"
        client_hint = "wave5-ctxnone-hint"
        workspace_name = "wave5-ctxnone-alpha"
        seed_session = "wave5-ctxnone-seed"
        self._wave5_session_ids = (session_id, seed_session)
        self._wave5_target_session = session_id
        self._wave5_seed_session = seed_session
        self._wave5_model_name = model_name
        self._wave5_client_hint = client_hint

        state.workspace_store.activate(
            seed_session, workspace_name, client_hint=client_hint,
        )
        state.workspace_store.pop_session(seed_session)
        _cache_file("/wave5/ctxnone.py", "alpha cache",
                    workspace_name, seed_session)
        _cache_file("/wave5/default-only.py", "default cache",
                    "default", seed_session)

        messages = [
            {"role": "system", "content": "wave5 system"},
            {"role": "user", "content": "hello"},
        ]
        cleaned_expected = _cleanup_tool_pair_integrity(messages)

        conn_mock = _FastPathNoSummaryConn()
        monkeypatch.setattr(
            "proxy.context.get_context_config",
            lambda *_a, **_kw: self._cfg(),
        )
        monkeypatch.setattr("memory.db.connection", lambda *_a, **_kw: conn_mock)

        from proxy.context_v2 import _select_context_v2
        result = _select_context_v2(
            session_id, messages, model_name=model_name, ctx=None,
        )

        assert not any("[VFS" in m.get("content", "") for m in result)
        non_tier2 = [m for m in result if m.get("role") != "system"
                     or "AGENTS.md" not in m.get("content", "")]
        assert non_tier2 == cleaned_expected, (
            f"Output must equal cleaned; got diff: "
            f"{[(i, e, r) for i, (e, r) in enumerate(zip(cleaned_expected, non_tier2)) if e != r]}"
        )
        self._assert_no_chain_state(session_id, model_name)
        assert not any(
            m.get("role") == "assistant" and m.get("content") == " "
            for m in result
        )

    def test_actual_vfs_cost_overflow_omits_without_trimming(self, monkeypatch):
        """Base fits but augmented (with real VFS) does not → omit VFS without
        trimming. Revert B's discriminator.

        Slice 4A removed the implicit 2000-token tool_schema fallback for
        ``tools=None``. To preserve the original test intent (VFS overflow
        discriminator), we now pass an explicit ~2000-token tool schema so
        the budget math matches the pre-Slice-4A invariant.
        """
        import json as _json
        from proxy.vfs import _cache_file, _build_vfs_context
        from proxy.context_v2 import _cleanup_tool_pair_integrity
        from proxy.context import _count_msg_chars
        session_id = "wave5-overflow-session"
        model_name = "wave5-overflow-model"
        client_hint = "wave5-overflow-hint"
        workspace_name = "wave5-overflow-alpha"
        seed_session = "wave5-overflow-seed"
        self._wave5_session_ids = (session_id, seed_session)
        self._wave5_target_session = session_id
        self._wave5_seed_session = seed_session
        self._wave5_model_name = model_name
        self._wave5_client_hint = client_hint

        state.workspace_store.activate(
            seed_session, workspace_name, client_hint=client_hint,
        )
        state.workspace_store.pop_session(seed_session)
        _cache_file("/wave5/overflow.py", "overflow cache",
                    workspace_name, seed_session)

        safe = 4_900
        tpc = 1.0

        long_user_content = "X" * 1400
        long_asst_content = "Y" * 1400
        messages = [
            {"role": "system", "content": "wave5 system"},
            {"role": "user", "content": long_user_content},
            {"role": "assistant", "content": long_asst_content},
        ]

        # Explicit tool schema sized to consume ~2000 tokens at tpc=1.0
        # (2000 chars JSON-serialized). Restores the pre-Slice-4A budget
        # invariant without relying on the removed fallback.
        tool_description = "T" * 1950
        tools = [{
            "type": "function",
            "function": {
                "name": "schema_pad",
                "description": tool_description,
                "parameters": {"type": "object", "properties": {}},
            },
        }]
        tool_tokens = int(len(_json.dumps(tools, separators=(",", ":"))) * tpc)

        test_ctx = self._ctx(
            session_id, model_name, client_hint, workspace_name, messages,
        )
        actual_vfs = _build_vfs_context(
            session_id, test_ctx, int(safe * 0.05), tpc,
        )
        assert actual_vfs is not None, (
            "Test setup invariant: VFS builder must return non-None for "
            "populated cache under generous budget"
        )
        actual_vfs_tokens = int(_count_msg_chars(actual_vfs) * tpc)
        base_tokens = sum(
            int(_count_msg_chars(m) * tpc) for m in messages
            if m.get("role") != "system"
        )
        sys_tokens = int(_count_msg_chars(messages[0]) * tpc)

        assert base_tokens + tool_tokens + sys_tokens <= safe, (
            f"base ({base_tokens}) + tools ({tool_tokens}) + sys ({sys_tokens}) "
            f"= {base_tokens + tool_tokens + sys_tokens} must fit safe ({safe})"
        )
        assert (
            base_tokens + actual_vfs_tokens + tool_tokens + sys_tokens
            > safe
        ), (
            f"base ({base_tokens}) + vfs ({actual_vfs_tokens}) + tools "
            f"({tool_tokens}) + sys ({sys_tokens}) = "
            f"{base_tokens + actual_vfs_tokens + tool_tokens + sys_tokens} "
            f"must overflow safe ({safe})"
        )

        conn_mock = _FastPathNoSummaryConn()
        monkeypatch.setattr(
            "proxy.context.get_context_config",
            lambda *_a, **_kw: self._cfg(safe=safe, tpc=tpc),
        )
        monkeypatch.setattr("memory.db.connection", lambda *_a, **_kw: conn_mock)

        from proxy.context_v2 import _select_context_v2
        result = _select_context_v2(
            session_id, messages, model_name=model_name, ctx=test_ctx,
            tools=tools,
        )

        assert not any("[VFS" in m.get("content", "") for m in result), (
            f"Overflow path must omit VFS; got roles="
            f"{[m.get('role') for m in result]}"
        )
        cleaned_expected = _cleanup_tool_pair_integrity(messages)
        non_tier2 = [m for m in result if m.get("role") != "system"
                     or "AGENTS.md" not in m.get("content", "")]
        assert non_tier2 == cleaned_expected, (
            f"Overflow must return cleaned exactly; got diff: "
            f"{[(i, e, r) for i, (e, r) in enumerate(zip(cleaned_expected, non_tier2)) if e != r]}"
        )
        self._assert_no_chain_state(session_id, model_name)
        assert test_ctx.v2_boundary_idx is None
        assert test_ctx.v2_boundary_chain is None

    def test_actual_vfs_cost_exact_fit_includes_vfs(self, monkeypatch):
        """Exact equality fit includes VFS — pins `<=`, not `<`."""
        from proxy.vfs import _cache_file, _build_vfs_context
        from proxy.context import _count_msg_chars
        session_id = "wave5-exactfit-session"
        model_name = "wave5-exactfit-model"
        client_hint = "wave5-exactfit-hint"
        workspace_name = "wave5-exactfit-alpha"
        seed_session = "wave5-exactfit-seed"
        self._wave5_session_ids = (session_id, seed_session)
        self._wave5_target_session = session_id
        self._wave5_seed_session = seed_session
        self._wave5_model_name = model_name
        self._wave5_client_hint = client_hint

        state.workspace_store.activate(
            seed_session, workspace_name, client_hint=client_hint,
        )
        state.workspace_store.pop_session(seed_session)
        _cache_file("/wave5/exactfit.py", "exactfit cache",
                    workspace_name, seed_session)

        tpc = 1.0
        safe = 10_000

        long_user_content = "X" * 1000
        long_asst_content = "Y" * 1000
        messages = [
            {"role": "system", "content": "wave5 system"},
            {"role": "user", "content": long_user_content},
            {"role": "assistant", "content": long_asst_content},
        ]

        test_ctx = self._ctx(
            session_id, model_name, client_hint, workspace_name, messages,
        )

        actual_vfs = _build_vfs_context(
            session_id, test_ctx, int(safe * 0.05), tpc,
        )
        assert actual_vfs is not None
        actual_vfs_tokens = int(_count_msg_chars(actual_vfs) * tpc)
        sys_tokens = int(_count_msg_chars(messages[0]) * tpc)
        tool_tokens = 2000

        target_total = safe
        target_base = target_total - tool_tokens - sys_tokens - actual_vfs_tokens
        if target_base < 0:
            pytest.fail(
                f"VFS alone ({actual_vfs_tokens}) exceeds available budget; "
                f"cannot construct exact-fit fixture"
            )
        long_asst_chars = 1000
        long_user_chars = target_base - long_asst_chars
        if long_user_chars <= 0:
            long_asst_chars = target_base // 2
            long_user_chars = target_base - long_asst_chars
        assert long_user_chars > 0

        messages[1]["content"] = "X" * long_user_chars
        messages[2]["content"] = "Y" * long_asst_chars

        recomputed_base = sum(
            int(_count_msg_chars(m) * tpc) for m in messages
            if m.get("role") != "system"
        )
        assert (
            recomputed_base + actual_vfs_tokens + tool_tokens + sys_tokens
            == safe
        ), "Fixture must satisfy exact-equality invariant"

        conn_mock = _FastPathNoSummaryConn()
        monkeypatch.setattr(
            "proxy.context.get_context_config",
            lambda *_a, **_kw: self._cfg(safe=safe, tpc=tpc),
        )
        monkeypatch.setattr("memory.db.connection", lambda *_a, **_kw: conn_mock)

        from proxy.context_v2 import _select_context_v2
        result = _select_context_v2(
            session_id, messages, model_name=model_name, ctx=test_ctx,
        )

        assert any("[VFS" in m.get("content", "") for m in result), (
            f"Exact fit must include VFS; got roles="
            f"{[m.get('role') for m in result]}"
        )
        self._assert_no_chain_state(session_id, model_name)

    def test_no_user_fresh_fast_path_does_not_consult_vfs(self, monkeypatch):
        """Assistant-only / no-user request → VFS builder not invoked;
        cleaned return preserved."""
        import proxy.vfs as _vfs_module
        from proxy.vfs import _cache_file
        from proxy.context_v2 import _cleanup_tool_pair_integrity
        session_id = "wave5-nouser-session"
        model_name = "wave5-nouser-model"
        client_hint = "wave5-nouser-hint"
        workspace_name = "wave5-nouser-alpha"
        seed_session = "wave5-nouser-seed"
        self._wave5_session_ids = (session_id, seed_session)
        self._wave5_target_session = session_id
        self._wave5_seed_session = seed_session
        self._wave5_model_name = model_name
        self._wave5_client_hint = client_hint

        state.workspace_store.activate(
            seed_session, workspace_name, client_hint=client_hint,
        )
        state.workspace_store.pop_session(seed_session)
        _cache_file("/wave5/nouser.py", "nouser cache",
                    workspace_name, seed_session)

        messages = [
            {"role": "system", "content": "wave5 system"},
            {"role": "assistant", "content": "alone"},
        ]
        cleaned_expected = _cleanup_tool_pair_integrity(messages)

        conn_mock = _FastPathNoSummaryConn()
        monkeypatch.setattr(
            "proxy.context.get_context_config",
            lambda *_a, **_kw: self._cfg(),
        )
        monkeypatch.setattr("memory.db.connection", lambda *_a, **_kw: conn_mock)

        ctx = self._ctx(
            session_id, model_name, client_hint, workspace_name, messages,
        )

        real_builder = _vfs_module._build_vfs_context
        with patch.object(
            _vfs_module, "_build_vfs_context", wraps=real_builder,
        ) as spy:
            from proxy.context_v2 import _select_context_v2
            result = _select_context_v2(
                session_id, messages, model_name=model_name, ctx=ctx,
            )

        assert spy.call_count == 0, (
            f"No-user fast path must not invoke _build_vfs_context; "
            f"called {spy.call_count} times"
        )
        assert not any("[VFS" in m.get("content", "") for m in result)
        non_tier2 = [m for m in result if m.get("role") != "system"
                     or "AGENTS.md" not in m.get("content", "")]
        assert non_tier2 == cleaned_expected
        self._assert_no_chain_state(session_id, model_name)
        assert ctx.v2_boundary_idx is None
        assert ctx.v2_boundary_chain is None

    def test_vfs_builder_failure_preserves_fresh_fast_return(self, monkeypatch):
        """Builder exception → cleaned fast return preserved; warning logged."""
        from proxy.vfs import _cache_file
        from proxy.context_v2 import _cleanup_tool_pair_integrity
        session_id = "wave5-fail-session"
        model_name = "wave5-fail-model"
        client_hint = "wave5-fail-hint"
        workspace_name = "wave5-fail-alpha"
        seed_session = "wave5-fail-seed"
        self._wave5_session_ids = (session_id, seed_session)
        self._wave5_target_session = session_id
        self._wave5_seed_session = seed_session
        self._wave5_model_name = model_name
        self._wave5_client_hint = client_hint

        state.workspace_store.activate(
            seed_session, workspace_name, client_hint=client_hint,
        )
        state.workspace_store.pop_session(seed_session)
        _cache_file("/wave5/fail.py", "fail cache",
                    workspace_name, seed_session)

        messages = [
            {"role": "system", "content": "wave5 system"},
            {"role": "user", "content": "hello"},
        ]
        cleaned_expected = _cleanup_tool_pair_integrity(messages)

        conn_mock = _FastPathNoSummaryConn()
        monkeypatch.setattr(
            "proxy.context.get_context_config",
            lambda *_a, **_kw: self._cfg(),
        )
        monkeypatch.setattr("memory.db.connection", lambda *_a, **_kw: conn_mock)

        def _boom(*_a, **_kw):
            raise RuntimeError("simulated builder failure")

        monkeypatch.setattr(
            "proxy.workspace_state.resolve_inherited_workspace",
            _boom,
        )

        ctx = self._ctx(
            session_id, model_name, client_hint, workspace_name, messages,
        )

        from proxy.context_v2 import _select_context_v2
        result = _select_context_v2(
            session_id, messages, model_name=model_name, ctx=ctx,
        )

        assert not any("[VFS" in m.get("content", "") for m in result)
        non_tier2 = [m for m in result if m.get("role") != "system"
                     or "AGENTS.md" not in m.get("content", "")]
        assert non_tier2 == cleaned_expected
        self._assert_no_chain_state(session_id, model_name)
        assert ctx.v2_boundary_idx is None
        assert ctx.v2_boundary_chain is None

    @pytest.mark.parametrize(
        "branch",
        [
            pytest.param("empty_vfs", id="empty-vfs"),
            pytest.param("warm_affordable", id="warm-affordable"),
            pytest.param("warm_overflow", id="warm-overflow"),
            pytest.param("ctx_none", id="ctx-none"),
        ],
    )
    def test_stable_tier2_survives_fresh_fast_path_branches(self, monkeypatch, branch):
        """Tier-2 pin stable across every fast-path branch."""
        from proxy.vfs import _cache_file
        session_id = f"wave5-tier2-{branch}-session"
        model_name = f"wave5-tier2-{branch}-model"
        client_hint = "wave5-tier2-hint"
        workspace_name = "wave5-tier2-alpha"
        seed_session = f"wave5-tier2-{branch}-seed"
        self._wave5_session_ids = (session_id, seed_session)
        self._wave5_target_session = session_id
        self._wave5_seed_session = seed_session
        self._wave5_model_name = model_name
        self._wave5_client_hint = client_hint

        sentinel_path = "WAVE5_TIER2_AGENTS_SENTINEL.md"
        sentinel_content = "WAVE5_STABLE_TIER2_SENTINEL"
        with state.vfs_lock:
            state.record_instruction_file(
                workspace_name, sentinel_path, sentinel_content,
                session_id=seed_session,
            )

        if branch != "empty_vfs":
            state.workspace_store.activate(
                seed_session, workspace_name, client_hint=client_hint,
            )
            state.workspace_store.pop_session(seed_session)
            _cache_file(f"/wave5/tier2-{branch}.py", "tier2 cache",
                        workspace_name, seed_session)

        if branch == "ctx_none":
            state.workspace_store.activate(
                session_id, workspace_name,
            )
            self._wave5_session_ids = (session_id, seed_session, session_id)

        if branch == "empty_vfs":
            messages = [
                {"role": "system", "content": "wave5 system"},
                {"role": "user", "content": "hello"},
            ]
        elif branch == "warm_affordable":
            messages = [
                {"role": "system", "content": "wave5 system"},
                {"role": "user", "content": "hello"},
            ]
        elif branch == "warm_overflow":
            safe = 3_600
            tpc = 1.0
            long_content = "X" * 1400
            messages = [
                {"role": "system", "content": "wave5 system"},
                {"role": "user", "content": long_content},
                {"role": "assistant", "content": long_content},
            ]
            ctx = self._ctx(
                session_id, model_name, client_hint, workspace_name, messages,
            )
            conn_mock = _FastPathNoSummaryConn()
            monkeypatch.setattr(
                "proxy.context.get_context_config",
                lambda *_a, **_kw: self._cfg(safe=safe, tpc=tpc),
            )
            monkeypatch.setattr(
                "memory.db.connection", lambda *_a, **_kw: conn_mock,
            )
            from proxy.context_v2 import _select_context_v2
            result = _select_context_v2(
                session_id, messages, model_name=model_name, ctx=ctx,
            )
            self._assert_tier2_pin_present(result, sentinel_content)
            assert not any("[VFS" in m.get("content", "") for m in result)
            self._assert_no_chain_state(session_id, model_name)
            return
        else:
            messages = [
                {"role": "system", "content": "wave5 system"},
                {"role": "user", "content": "hello"},
            ]

        if branch != "warm_overflow":
            conn_mock = _FastPathNoSummaryConn()
            monkeypatch.setattr(
                "proxy.context.get_context_config",
                lambda *_a, **_kw: self._cfg(),
            )
            monkeypatch.setattr(
                "memory.db.connection", lambda *_a, **_kw: conn_mock,
            )
            ctx = None if branch == "ctx_none" else self._ctx(
                session_id, model_name, client_hint, workspace_name, messages,
            )

        from proxy.context_v2 import _select_context_v2
        result = _select_context_v2(
            session_id, messages, model_name=model_name, ctx=ctx,
        )

        self._assert_tier2_pin_present(result, sentinel_content)
        self._assert_no_chain_state(session_id, model_name)


class TestChainFailureVfsFallback:
    """US: Wave 6 chain-failure handler — optional VFS on compute_chain failure."""

    @pytest.fixture(autouse=True)
    def _wave6_state(self, monkeypatch):
        import uuid
        import proxy.state as state
        self._wave6_run_id = uuid.uuid4().hex[:8]
        self._wave6_session_ids = ()
        self._wave6_seed_session = ""
        self._wave6_client_hint = ""
        self._wave6_workspace = ""
        self._wave6_vfs_tokens = ()
        self._wave6_chain_keys = ()

        monkeypatch.setattr(state, "VFS_ENABLED", True)
        monkeypatch.setattr(state, "VFS_SCOPE", "all")
        monkeypatch.setattr(state, "VFS_TTL_SECONDS", 3600)
        monkeypatch.setattr(state, "VFS_MAX_ENTRIES", 500)
        monkeypatch.setattr(state, "VFS_MAX_ENTRY_BYTES", 1_048_576)
        monkeypatch.setattr(state, "VFS_MAX_WORKSPACES", 20)

        yield

        for sid in self._wave6_session_ids:
            try:
                state.workspace_store.deactivate(sid)
                state.workspace_store.pop_session(sid)
            except Exception:
                pass
        if self._wave6_seed_session and self._wave6_client_hint:
            try:
                state.workspace_store.deactivate(
                    self._wave6_seed_session, client_hint=self._wave6_client_hint,
                )
            except Exception:
                pass
        with state.vfs_lock:
            state.vfs_caches.clear()
            state.vfs_write_log.clear()
            state.vfs_rollback_snapshots.clear()
            state.instruction_files_by_workspace.clear()
            state._last_scanned_by_tool_call.clear()
        for key in self._wave6_chain_keys:
            with state._chain_state_lock:
                state._chain_state.pop(key, None)

    @staticmethod
    def _cfg(safe=10_000, tpc=0.25):
        return {
            "max_context_tokens": 40_000,
            "tokens_per_char": tpc,
            "safe_context_tokens": safe,
            "raw_exchange_count": 100,
        }

    def _build_ctx(self, session_id, model_name, client_hint, workspace_name, messages,
                   run_id=None):
        run_id = run_id or self._wave6_run_id
        ctx = RequestContext(
            body={"messages": messages},
            model_name=model_name,
            wants_stream=False,
            session_id=session_id,
            client_hint=client_hint,
            messages=messages,
            remote_addr="127.0.0.1",
            workspace_name=workspace_name,
        )
        # Real-context boundary sentinels (must reset to None on every return).
        ctx.v2_boundary_idx = 37
        ctx.v2_boundary_chain = "prior-boundary"
        return ctx

    def _seed_workspace_and_cache(self, seed_session, target_session, workspace, client_hint,
                                    cache_path, cache_content, run_id=None):
        """Activate workspace under client_hint, populate named-workspace cache."""
        import proxy.state as state
        from proxy.vfs import _cache_file
        run_id = run_id or self._wave6_run_id
        self._wave6_session_ids = (seed_session, target_session)
        self._wave6_seed_session = seed_session
        self._wave6_client_hint = client_hint
        self._wave6_workspace = workspace
        state.workspace_store.activate(seed_session, workspace, client_hint=client_hint)
        state.workspace_store.pop_session(seed_session)
        _cache_file(cache_path, cache_content, workspace, seed_session)

    def _force_chain_failure(self, monkeypatch, exc_cls=ValueError, msg="forced Wave 6 chain failure"):
        from proxy import chain_hash
        monkeypatch.setattr(
            chain_hash, "compute_chain",
            lambda *a, **kw: (_ for _ in ()).throw(exc_cls(msg)),
        )

    def _assert_no_chain_state(self, key):
        import proxy.state as state
        with state._chain_state_lock:
            assert key not in state._chain_state

    def test_affordable_real_vfs_precedes_complete_multi_tool_live_exchange(self, monkeypatch):
        """Affordable VFS on chain failure with multi-tool continuation.

        Primary revert discriminator.
        """
        import uuid
        run_id = uuid.uuid4().hex[:8]
        SESSION = f"wave6-afford-{run_id}"
        MODEL = f"wave6-afford-model-{run_id}"
        CLIENT = f"wave6-afford-client-{run_id}"
        WORKSPACE = f"wave6-afford-alpha-{run_id}"
        SEED = f"wave6-afford-seed-{run_id}"
        self._wave6_chain_keys = [(SESSION, MODEL)]

        CACHE_PATH = f"/wave6/afford-{run_id}.py"
        CACHE_CONTENT = f"alpha cache content {run_id}"
        self._seed_workspace_and_cache(SEED, SESSION, WORKSPACE, CLIENT, CACHE_PATH, CACHE_CONTENT)

        messages = [
            {"role": "system", "content": "wave6 system"},
            {"role": "user", "content": "completed user"},
            {"role": "assistant", "content": "completed assistant"},
            {"role": "user", "content": "current user"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {"id": "A", "type": "function",
                     "function": {"name": "read", "arguments": "{}"}},
                    {"id": "B", "type": "function",
                     "function": {"name": "read", "arguments": "{}"}},
                ],
            },
            {"role": "tool", "tool_call_id": "A", "content": "result A"},
            {"role": "tool", "tool_call_id": "B", "content": "result B"},
        ]
        messages_before = [dict(m) for m in messages]
        ctx = self._build_ctx(SESSION, MODEL, CLIENT, WORKSPACE, messages, run_id=run_id)

        monkeypatch.setattr(
            "proxy.context.get_context_config",
            lambda *_a, **_kw: self._cfg(),
        )
        self._force_chain_failure(monkeypatch)

        from proxy.context_v2 import _select_context_v2
        result = _select_context_v2(
            SESSION, messages, model_name=MODEL, ctx=ctx,
        )

        def _idx(predicate):
            for i, m in enumerate(result):
                if predicate(m):
                    return i
            return -1

        vfs_indices = [
            i for i, m in enumerate(result)
            if "[VFS CONTEXT" in (m.get("content") or "")
        ]
        assert len(vfs_indices) == 1, (
            f"Exactly one [VFS CONTEXT expected; got positions={vfs_indices}"
        )
        vfs_idx = vfs_indices[0]
        vfs_content = result[vfs_idx].get("content", "")
        assert CACHE_PATH in vfs_content, (
            f"Real cached filename {CACHE_PATH!r} must appear in VFS block"
        )

        sys_idx = _idx(lambda m: m.get("role") == "system"
                       and "wave6 system" in (m.get("content") or ""))
        old_user_idx = _idx(lambda m: m.get("content") == "completed user")
        old_asst_idx = _idx(lambda m: m.get("content") == "completed assistant")
        curr_user_idx = _idx(lambda m: m.get("content") == "current user")
        asst_tc_idx = _idx(lambda m: m.get("role") == "assistant"
                            and isinstance(m.get("tool_calls"), list)
                            and len(m["tool_calls"]) == 2)
        tool_a_idx = _idx(lambda m: m.get("role") == "tool"
                          and m.get("tool_call_id") == "A")
        tool_b_idx = _idx(lambda m: m.get("role") == "tool"
                          and m.get("tool_call_id") == "B")

        assert sys_idx >= 0 and old_user_idx >= 0 and old_asst_idx >= 0
        assert curr_user_idx >= 0 and asst_tc_idx >= 0
        assert tool_a_idx >= 0 and tool_b_idx >= 0

        assert sys_idx < old_user_idx < old_asst_idx < vfs_idx < curr_user_idx
        assert tool_a_idx == asst_tc_idx + 1
        assert tool_b_idx == asst_tc_idx + 2

        assert not any(
            m.get("role") == "assistant" and m.get("content") == " "
            for m in result
        )

        assert messages == messages_before

        assert ctx.v2_boundary_idx is None
        assert ctx.v2_boundary_chain is None

        self._assert_no_chain_state((SESSION, MODEL))

    def test_no_user_led_exchange_returns_cleaned_fallback_without_vfs(self, monkeypatch):
        """No user-led live exchange → no VFS; builder not consulted."""
        from unittest.mock import patch as _patch
        from proxy import vfs as _vfs_module
        import uuid
        run_id = uuid.uuid4().hex[:8]
        SESSION = f"wave6-nouser-{run_id}"
        MODEL = f"wave6-nouser-model-{run_id}"
        CLIENT = f"wave6-nouser-client-{run_id}"
        WORKSPACE = f"wave6-nouser-alpha-{run_id}"
        SEED = f"wave6-nouser-seed-{run_id}"
        self._wave6_chain_keys = [(SESSION, MODEL)]

        self._seed_workspace_and_cache(
            SEED, SESSION, WORKSPACE, CLIENT,
            f"/wave6/nouser-{run_id}.py", f"nouser cache {run_id}",
        )

        messages = [
            {"role": "system", "content": "wave6 system"},
            {"role": "assistant", "content": "alone assistant"},
        ]
        ctx = self._build_ctx(SESSION, MODEL, CLIENT, WORKSPACE, messages, run_id=run_id)

        monkeypatch.setattr(
            "proxy.context.get_context_config",
            lambda *_a, **_kw: self._cfg(),
        )
        self._force_chain_failure(monkeypatch)

        with _patch.object(
            _vfs_module, "_build_vfs_context",
            side_effect=lambda *a, **kw: _vfs_module._build_vfs_context(*a, **kw),
        ) as spy:
            from proxy.context_v2 import _select_context_v2
            result = _select_context_v2(
                SESSION, messages, model_name=MODEL, ctx=ctx,
            )

        assert spy.call_count == 0, (
            f"No-user chain fallback must not invoke _build_vfs_context; "
            f"called {spy.call_count} times"
        )
        assert not any("[VFS" in m.get("content", "") for m in result)
        assert ctx.v2_boundary_idx is None
        assert ctx.v2_boundary_chain is None
        self._assert_no_chain_state((SESSION, MODEL))

    def test_assistant_tool_only_unit_returns_cleaned_fallback_without_vfs(self, monkeypatch):
        """Assistant/tool-only input → no user manufactured; no VFS."""
        import uuid
        run_id = uuid.uuid4().hex[:8]
        SESSION = f"wave6-assttool-{run_id}"
        MODEL = f"wave6-assttool-model-{run_id}"
        CLIENT = f"wave6-assttool-client-{run_id}"
        WORKSPACE = f"wave6-assttool-alpha-{run_id}"
        SEED = f"wave6-assttool-seed-{run_id}"
        self._wave6_chain_keys = [(SESSION, MODEL)]

        self._seed_workspace_and_cache(
            SEED, SESSION, WORKSPACE, CLIENT,
            f"/wave6/assttool-{run_id}.py", f"assttool cache {run_id}",
        )

        messages = [
            {"role": "system", "content": "wave6 system"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {"id": "A", "type": "function",
                     "function": {"name": "read", "arguments": "{}"}},
                ],
            },
            {"role": "tool", "tool_call_id": "A", "content": "result A"},
        ]
        ctx = self._build_ctx(SESSION, MODEL, CLIENT, WORKSPACE, messages, run_id=run_id)

        monkeypatch.setattr(
            "proxy.context.get_context_config",
            lambda *_a, **_kw: self._cfg(),
        )
        self._force_chain_failure(monkeypatch)

        from proxy.context_v2 import _select_context_v2
        result = _select_context_v2(
            SESSION, messages, model_name=MODEL, ctx=ctx,
        )

        tool_indices = [
            i for i, m in enumerate(result) if m.get("role") == "tool"
        ]
        assert tool_indices, "Tool message must survive cleaned fallback"
        assert not any("[VFS" in m.get("content", "") for m in result)
        assert not any(
            m.get("role") == "user" for m in result
        ), "No user must be manufactured"
        assert not any(
            m.get("role") == "assistant" and m.get("content") == " "
            for m in result
        )
        assert ctx.v2_boundary_idx is None
        assert ctx.v2_boundary_chain is None
        self._assert_no_chain_state((SESSION, MODEL))

    def test_orphan_tool_tail_uses_cleanup_and_suppresses_vfs(self, monkeypatch):
        """Orphan tool tail → real cleanup authoritative; no VFS."""
        import uuid
        run_id = uuid.uuid4().hex[:8]
        SESSION = f"wave6-orphan-{run_id}"
        MODEL = f"wave6-orphan-model-{run_id}"
        CLIENT = f"wave6-orphan-client-{run_id}"
        WORKSPACE = f"wave6-orphan-alpha-{run_id}"
        SEED = f"wave6-orphan-seed-{run_id}"
        self._wave6_chain_keys = [(SESSION, MODEL)]

        self._seed_workspace_and_cache(
            SEED, SESSION, WORKSPACE, CLIENT,
            f"/wave6/orphan-{run_id}.py", f"orphan cache {run_id}",
        )

        messages = [
            {"role": "system", "content": "wave6 system"},
            {"role": "user", "content": "current user"},
            {"role": "tool", "tool_call_id": "orphan", "content": "stray"},
        ]
        messages_before = [dict(m) for m in messages]
        ctx = self._build_ctx(SESSION, MODEL, CLIENT, WORKSPACE, messages, run_id=run_id)

        monkeypatch.setattr(
            "proxy.context.get_context_config",
            lambda *_a, **_kw: self._cfg(),
        )
        self._force_chain_failure(monkeypatch)

        from proxy.context_v2 import _cleanup_tool_pair_integrity
        expected_cleaned = _cleanup_tool_pair_integrity(messages)

        from proxy.context_v2 import _select_context_v2
        result = _select_context_v2(
            SESSION, messages, model_name=MODEL, ctx=ctx,
        )

        # Real cleanup removed the orphan before the chain-failure handler ran.
        assert not any(m.get("tool_call_id") == "orphan" for m in result)
        # No VFS in the chain-failure fallback.
        assert not any("[VFS" in m.get("content", "") for m in result)
        # Result equals authoritative cleaned fallback (Tier-2 pin may precede).
        non_tier2 = [m for m in result if "AGENTS.md" not in m.get("content", "")]
        assert non_tier2 == expected_cleaned
        # Input unchanged.
        assert messages == messages_before
        # Boundaries None.
        assert ctx.v2_boundary_idx is None
        assert ctx.v2_boundary_chain is None
        self._assert_no_chain_state((SESSION, MODEL))

    def test_missing_tool_result_uses_cleanup_and_suppresses_vfs(self, monkeypatch):
        """Assistant(tool_calls) without matching tool → cleanup supplies synthetic result; no VFS."""
        import uuid
        run_id = uuid.uuid4().hex[:8]
        SESSION = f"wave6-missing-{run_id}"
        MODEL = f"wave6-missing-model-{run_id}"
        CLIENT = f"wave6-missing-client-{run_id}"
        WORKSPACE = f"wave6-missing-alpha-{run_id}"
        SEED = f"wave6-missing-seed-{run_id}"
        self._wave6_chain_keys = [(SESSION, MODEL)]

        self._seed_workspace_and_cache(
            SEED, SESSION, WORKSPACE, CLIENT,
            f"/wave6/missing-{run_id}.py", f"missing cache {run_id}",
        )

        messages = [
            {"role": "system", "content": "wave6 system"},
            {"role": "user", "content": "current user"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {"id": "Z", "type": "function",
                     "function": {"name": "read", "arguments": "{}"}},
                ],
            },
        ]
        ctx = self._build_ctx(SESSION, MODEL, CLIENT, WORKSPACE, messages, run_id=run_id)

        monkeypatch.setattr(
            "proxy.context.get_context_config",
            lambda *_a, **_kw: self._cfg(),
        )
        self._force_chain_failure(monkeypatch)

        from proxy.context_v2 import _select_context_v2
        result = _select_context_v2(
            SESSION, messages, model_name=MODEL, ctx=ctx,
        )

        # No VFS.
        assert not any("[VFS" in m.get("content", "") for m in result)
        # Cleanup synthesized a tool result for the missing pair.
        tool_indices = [i for i, m in enumerate(result) if m.get("role") == "tool"]
        assert tool_indices, (
            "Cleanup should supply a synthetic tool result for missing pair"
        )
        # Boundaries None.
        assert ctx.v2_boundary_idx is None
        assert ctx.v2_boundary_chain is None
        self._assert_no_chain_state((SESSION, MODEL))

    def test_actual_vfs_cost_overflow_returns_untrimmed_cleaned_fallback(self, monkeypatch):
        """VFS budget overflow → cleaned fallback without trimming.

        Revert B discriminator.
        """
        import uuid
        run_id = uuid.uuid4().hex[:8]
        SESSION = f"wave6-overflow-{run_id}"
        MODEL = f"wave6-overflow-model-{run_id}"
        CLIENT = f"wave6-overflow-client-{run_id}"
        WORKSPACE = f"wave6-overflow-alpha-{run_id}"
        SEED = f"wave6-overflow-seed-{run_id}"
        self._wave6_chain_keys = [(SESSION, MODEL)]

        self._seed_workspace_and_cache(
            SEED, SESSION, WORKSPACE, CLIENT,
            f"/wave6/overflow-{run_id}.py",
            (f"big cache content-{run_id}-" + "z" * 800),
        )

        # Tight safe budget at tpc=1.0 for exact integer arithmetic.
        tpc = 1.0
        safe = 4_400
        long_user = "X" * 600   # 600 tokens
        long_asst = "Y" * 600   # 600 tokens
        messages = [
            {"role": "system", "content": "wave6 system"},
            {"role": "user", "content": long_user},
            {"role": "assistant", "content": long_asst},
        ]
        # base = 600 + 600 = 1200 tokens (excluding system).
        # sys tokens ~ 12. Custom tools list inflated so tool_schema_tokens
        # is large enough that adding VFS overflows the safe budget while the
        # unaugmented base still fits.
        oversize_tools = [{
            "type": "function",
            "function": {
                "name": f"bigfunc-{run_id}",
                "description": "z" * 2_950,
                "parameters": {"type": "object", "properties": {}},
            },
        }]
        import json as _json
        tool_tokens = int(len(_json.dumps(oversize_tools, separators=(",", ":"))) * tpc)
        # Reset safe to land just under base+tools+sys.
        safe = tool_tokens + 1200 + 12 + 50  # 50 = buffer
        vfs_budget = int(safe * 0.05)

        from proxy.context_v2 import _count_msg_chars
        from proxy.vfs import _build_vfs_context
        from proxy.types import RequestContext

        test_ctx = RequestContext(
            body={"messages": messages},
            model_name=MODEL,
            wants_stream=False,
            session_id=SESSION,
            client_hint=CLIENT,
            messages=messages,
            remote_addr="127.0.0.1",
            workspace_name=WORKSPACE,
        )
        test_ctx.v2_boundary_idx = 37
        test_ctx.v2_boundary_chain = "prior-boundary"
        actual_vfs = _build_vfs_context(
            SESSION, test_ctx, int(safe * 0.05), tpc,
        )
        assert actual_vfs is not None, (
            f"VFS budget ({int(safe * 0.05)}) must accommodate the rendered block"
        )
        actual_vfs_tokens = int(_count_msg_chars(actual_vfs) * tpc)
        base_tokens = sum(
            int(_count_msg_chars(m) * tpc) for m in messages
            if m.get("role") != "system"
        )
        sys_tokens = int(_count_msg_chars(messages[0]) * tpc)

        assert base_tokens + tool_tokens + sys_tokens <= safe, (
            f"base ({base_tokens}) + tools ({tool_tokens}) + sys ({sys_tokens}) "
            f"= {base_tokens + tool_tokens + sys_tokens} must fit safe ({safe})"
        )
        assert (
            base_tokens + actual_vfs_tokens + tool_tokens + sys_tokens
            > safe
        ), (
            f"base + vfs ({actual_vfs_tokens}) + tools + sys must exceed safe "
            f"({base_tokens + actual_vfs_tokens + tool_tokens + sys_tokens} <= {safe})"
        )

        # Now call through the chain-failure handler.
        ctx = self._build_ctx(SESSION, MODEL, CLIENT, WORKSPACE, messages, run_id=run_id)
        messages_before = [dict(m) for m in messages]
        monkeypatch.setattr(
            "proxy.context.get_context_config",
            lambda *_a, **_kw: self._cfg(safe=safe, tpc=tpc),
        )
        self._force_chain_failure(monkeypatch)

        from proxy.context_v2 import _select_context_v2
        result = _select_context_v2(
            SESSION, messages, model_name=MODEL, ctx=ctx,
            tools=oversize_tools,
        )

        assert not any("[VFS" in m.get("content", "") for m in result)
        non_tier2 = [m for m in result if "AGENTS.md" not in m.get("content", "")]
        from proxy.context_v2 import _cleanup_tool_pair_integrity
        expected_cleaned = _cleanup_tool_pair_integrity(messages)
        assert non_tier2 == expected_cleaned
        assert messages == messages_before
        assert ctx.v2_boundary_idx is None
        assert ctx.v2_boundary_chain is None
        self._assert_no_chain_state((SESSION, MODEL))

    def test_actual_vfs_cost_exact_fit_includes_vfs(self, monkeypatch):
        """Exact equality fit includes VFS — pins `<=`, not `<`."""
        import uuid
        run_id = uuid.uuid4().hex[:8]
        SESSION = f"wave6-exact-{run_id}"
        MODEL = f"wave6-exact-model-{run_id}"
        CLIENT = f"wave6-exact-client-{run_id}"
        WORKSPACE = f"wave6-exact-alpha-{run_id}"
        SEED = f"wave6-exact-seed-{run_id}"
        self._wave6_chain_keys = [(SESSION, MODEL)]

        self._seed_workspace_and_cache(
            SEED, SESSION, WORKSPACE, CLIENT,
            f"/wave6/exact-{run_id}.py", f"exact cache {run_id}",
        )

        tpc = 1.0
        safe = 10_000
        long_user = "X" * 1000
        long_asst = "Y" * 1000
        messages = [
            {"role": "system", "content": "wave6 system"},
            {"role": "user", "content": long_user},
            {"role": "assistant", "content": long_asst},
        ]
        from proxy.context_v2 import _count_msg_chars
        from proxy.vfs import _build_vfs_context
        from proxy.types import RequestContext

        test_ctx = RequestContext(
            body={"messages": messages},
            model_name=MODEL,
            wants_stream=False,
            session_id=SESSION,
            client_hint=CLIENT,
            messages=messages,
            remote_addr="127.0.0.1",
            workspace_name=WORKSPACE,
        )
        test_ctx.v2_boundary_idx = 37
        test_ctx.v2_boundary_chain = "prior-boundary"
        actual_vfs = _build_vfs_context(
            SESSION, test_ctx, int(10_000 * 0.05), tpc,
        )
        assert actual_vfs is not None
        actual_vfs_tokens = int(_count_msg_chars(actual_vfs) * tpc)
        base_tokens = sum(
            int(_count_msg_chars(m) * tpc) for m in messages
            if m.get("role") != "system"
        )
        sys_tokens = int(_count_msg_chars(messages[0]) * tpc)
        tool_tokens = 2000

        # Resize to hit exact equality.
        target_base = safe - tool_tokens - sys_tokens - actual_vfs_tokens
        if target_base < 0:
            pytest.fail("VFS alone exceeds available budget")
        asst_chars = 1000
        user_chars = target_base - asst_chars
        if user_chars <= 0:
            asst_chars = target_base // 2
            user_chars = target_base - asst_chars
        assert user_chars > 0
        messages[1]["content"] = "X" * user_chars
        messages[2]["content"] = "Y" * asst_chars

        ctx = self._build_ctx(SESSION, MODEL, CLIENT, WORKSPACE, messages, run_id=run_id)
        monkeypatch.setattr(
            "proxy.context.get_context_config",
            lambda *_a, **_kw: self._cfg(safe=safe, tpc=tpc),
        )
        self._force_chain_failure(monkeypatch)

        from proxy.context_v2 import _select_context_v2
        result = _select_context_v2(
            SESSION, messages, model_name=MODEL, ctx=ctx,
        )

        assert any("[VFS" in m.get("content", "") for m in result), (
            f"Exact fit must include VFS; roles={[m.get('role') for m in result]}"
        )
        self._assert_no_chain_state((SESSION, MODEL))

    def test_ctx_none_returns_cleaned_fallback_without_vfs(self, monkeypatch):
        """ctx=None → cleaned fallback, no VFS, no fabricated context."""
        import uuid
        run_id = uuid.uuid4().hex[:8]
        SESSION = f"wave6-ctxnone-{run_id}"
        MODEL = f"wave6-ctxnone-model-{run_id}"
        CLIENT = f"wave6-ctxnone-client-{run_id}"
        WORKSPACE = f"wave6-ctxnone-alpha-{run_id}"
        SEED = f"wave6-ctxnone-seed-{run_id}"
        self._wave6_chain_keys = [(SESSION, MODEL)]

        self._seed_workspace_and_cache(
            SEED, SESSION, WORKSPACE, CLIENT,
            f"/wave6/ctxnone-{run_id}.py", f"ctxnone cache {run_id}",
        )

        messages = [
            {"role": "system", "content": "wave6 system"},
            {"role": "user", "content": "hello"},
        ]
        monkeypatch.setattr(
            "proxy.context.get_context_config",
            lambda *_a, **_kw: self._cfg(),
        )
        self._force_chain_failure(monkeypatch)

        from proxy.context_v2 import _select_context_v2
        result = _select_context_v2(
            SESSION, messages, model_name=MODEL, ctx=None,
        )

        assert not any("[VFS" in m.get("content", "") for m in result)
        # Result equals cleaned input (modulo Tier-2 pin).
        from proxy.context_v2 import _cleanup_tool_pair_integrity
        expected_cleaned = _cleanup_tool_pair_integrity(messages)
        non_tier2 = [m for m in result if "AGENTS.md" not in m.get("content", "")]
        assert non_tier2 == expected_cleaned
        self._assert_no_chain_state((SESSION, MODEL))

    def test_vfs_builder_failure_returns_cleaned_fallback(self, monkeypatch):
        """Lower builder dependency raises → real builder entered; cleaned fallback."""
        import uuid
        run_id = uuid.uuid4().hex[:8]
        SESSION = f"wave6-builderfail-{run_id}"
        MODEL = f"wave6-builderfail-model-{run_id}"
        CLIENT = f"wave6-builderfail-client-{run_id}"
        WORKSPACE = f"wave6-builderfail-alpha-{run_id}"
        SEED = f"wave6-builderfail-seed-{run_id}"
        self._wave6_chain_keys = [(SESSION, MODEL)]

        self._seed_workspace_and_cache(
            SEED, SESSION, WORKSPACE, CLIENT,
            f"/wave6/builderfail-{run_id}.py", f"builderfail cache {run_id}",
        )

        messages = [
            {"role": "system", "content": "wave6 system"},
            {"role": "user", "content": "current user"},
        ]
        ctx = self._build_ctx(SESSION, MODEL, CLIENT, WORKSPACE, messages, run_id=run_id)

        monkeypatch.setattr(
            "proxy.context.get_context_config",
            lambda *_a, **_kw: self._cfg(),
        )
        # Force resolve_inherited_workspace to raise — propagates into _build_vfs_context.
        def _boom(*_a, **_kw):
            raise RuntimeError("forced builder dependency failure")
        monkeypatch.setattr(
            "proxy.workspace_state.resolve_inherited_workspace", _boom,
        )
        self._force_chain_failure(monkeypatch)

        from proxy.context_v2 import _select_context_v2
        result = _select_context_v2(
            SESSION, messages, model_name=MODEL, ctx=ctx,
        )

        assert not any("[VFS" in m.get("content", "") for m in result)
        from proxy.context_v2 import _cleanup_tool_pair_integrity
        expected_cleaned = _cleanup_tool_pair_integrity(messages)
        non_tier2 = [m for m in result if "AGENTS.md" not in m.get("content", "")]
        assert non_tier2 == expected_cleaned
        assert ctx.v2_boundary_idx is None
        assert ctx.v2_boundary_chain is None
        self._assert_no_chain_state((SESSION, MODEL))

    def test_cleanup_failure_after_chain_failure_preserves_full_message_fallback(self, monkeypatch):
        """Slice 4D correction: malformed (non-dict) messages that break
        both chain and cleanup now raise ``ContextLimitError`` at final
        validation rather than silently returning the malformed candidate.
        The chain-failure handler still catches the cleanup exception and
        routes the original messages to the terminal boundary, which then
        rejects the non-dict message.
        """
        from proxy.context import ContextLimitError

        import uuid
        run_id = uuid.uuid4().hex[:8]
        SESSION = f"wave6-cleanupfail-{run_id}"
        MODEL = f"wave6-cleanupfail-model-{run_id}"
        CLIENT = f"wave6-cleanupfail-client-{run_id}"
        WORKSPACE = f"wave6-cleanupfail-alpha-{run_id}"
        SEED = f"wave6-cleanupfail-seed-{run_id}"
        self._wave6_chain_keys = [(SESSION, MODEL)]

        messages = [
            {"role": "system", "content": "wave6 system"},
            None,  # triggers AttributeError in m.get(...)
        ]
        ctx = self._build_ctx(SESSION, MODEL, CLIENT, WORKSPACE, messages, run_id=run_id)
        monkeypatch.setattr(
            "proxy.context.get_context_config",
            lambda *_a, **_kw: self._cfg(),
        )
        self._force_chain_failure(monkeypatch)

        from proxy.context_v2 import _select_context_v2
        with pytest.raises(ContextLimitError) as excinfo:
            _select_context_v2(
                SESSION, messages, model_name=MODEL, ctx=ctx,
            )

        assert "Cannot validate final context budget" in str(excinfo.value)
        assert ctx.v2_boundary_idx is None
        assert ctx.v2_boundary_chain is None
        self._assert_no_chain_state((SESSION, MODEL))

    @pytest.mark.parametrize(
        "branch",
        [
            pytest.param("affordable", id="affordable"),
            pytest.param("overflow", id="overflow"),
            pytest.param("no_user", id="no-user"),
            pytest.param("ctx_none", id="ctx-none"),
            pytest.param("builder_fail", id="builder-fail"),
        ],
    )
    def test_stable_tier2_survives_chain_failure_fallback_branches(self, monkeypatch, branch):
        """Tier-2 pin stable across every chain-failure branch.

        No _capture_tier2_instructions() call occurs.
        """
        import uuid
        from unittest.mock import patch as _patch
        import proxy.state as state
        from proxy.context_v2 import _capture_tier2_instructions as _real_capture

        run_id = uuid.uuid4().hex[:8]
        SESSION = f"wave6-tier2-{branch}-{run_id}"
        MODEL = f"wave6-tier2-{branch}-model-{run_id}"
        CLIENT = f"wave6-tier2-{branch}-client-{run_id}"
        WORKSPACE = f"wave6-tier2-{branch}-alpha-{run_id}"
        SEED = f"wave6-tier2-{branch}-seed-{run_id}"
        self._wave6_chain_keys = [(SESSION, MODEL)]

        sentinel_path = f"WAVE6_TIER2_SENTINEL_{branch}_{run_id}.md"
        sentinel_content = f"WAVE6_STABLE_TIER2_{branch}_{run_id}"
        # Record Tier-2 sentinel under the workspace that the branch will resolve.
        # For ctx_none, _emit_tier2_pin sees ctx=None and falls back to
        # resolve_inherited_workspace(session_id) → "default" workspace.
        tier2_workspace = "default" if branch == "ctx_none" else WORKSPACE
        with state.vfs_lock:
            state.record_instruction_file(
                tier2_workspace, sentinel_path, sentinel_content,
                session_id=SESSION,
            )

        # Workspace + cache for branches that need them.
        if branch in ("affordable", "overflow", "no_user", "builder_fail"):
            self._seed_workspace_and_cache(
                SEED, SESSION, WORKSPACE, CLIENT,
                f"/wave6/tier2-{branch}-{run_id}.py",
                f"tier2 cache {branch} {run_id}",
            )

        # Build messages per branch.
        if branch == "affordable":
            messages = [
                {"role": "system", "content": "wave6 system"},
                {"role": "user", "content": "hello"},
            ]
        elif branch == "overflow":
            safe = 2_000
            tpc = 1.0
            long_user = "X" * 600
            long_asst = "Y" * 600
            messages = [
                {"role": "system", "content": "wave6 system"},
                {"role": "user", "content": long_user},
                {"role": "assistant", "content": long_asst},
            ]
            cfg = self._cfg(safe=safe, tpc=tpc)
            monkeypatch.setattr(
                "proxy.context.get_context_config",
                lambda *_a, **_kw: cfg,
            )
            ctx = self._build_ctx(SESSION, MODEL, CLIENT, WORKSPACE, messages, run_id=run_id)
            self._force_chain_failure(monkeypatch)
            with _patch.object(
                __import__("proxy.context_v2", fromlist=["_capture_tier2_instructions"]),
                "_capture_tier2_instructions",
                side_effect=lambda *a, **kw: _real_capture(*a, **kw),
            ) as cap_spy:
                from proxy.context_v2 import _select_context_v2
                result = _select_context_v2(
                    SESSION, messages, model_name=MODEL, ctx=ctx,
                )
            assert cap_spy.call_count == 0, (
                f"Chain-failure path must not call _capture_tier2_instructions; "
                f"called {cap_spy.call_count} times"
            )
            assert not any("[VFS" in m.get("content", "") for m in result)
            self._assert_no_chain_state((SESSION, MODEL))
            return
        elif branch == "no_user":
            messages = [
                {"role": "system", "content": "wave6 system"},
                {"role": "assistant", "content": "alone"},
            ]
        elif branch == "ctx_none":
            messages = [
                {"role": "system", "content": "wave6 system"},
                {"role": "user", "content": "hello"},
            ]
        elif branch == "builder_fail":
            messages = [
                {"role": "system", "content": "wave6 system"},
                {"role": "user", "content": "hello"},
            ]
            monkeypatch.setattr(
                "proxy.workspace_state.resolve_inherited_workspace",
                lambda *a, **kw: (_ for _ in ()).throw(
                    RuntimeError("forced builder dep failure")
                ),
            )

        if branch != "overflow":
            monkeypatch.setattr(
                "proxy.context.get_context_config",
                lambda *_a, **_kw: self._cfg(),
            )
        if branch != "ctx_none":
            ctx = self._build_ctx(SESSION, MODEL, CLIENT, WORKSPACE, messages, run_id=run_id)
        else:
            ctx = None
        self._force_chain_failure(monkeypatch)

        with _patch.object(
            __import__("proxy.context_v2", fromlist=["_capture_tier2_instructions"]),
            "_capture_tier2_instructions",
            side_effect=lambda *a, **kw: _real_capture(*a, **kw),
        ) as cap_spy:
            from proxy.context_v2 import _select_context_v2
            result = _select_context_v2(
                SESSION, messages, model_name=MODEL, ctx=ctx,
            )

        assert cap_spy.call_count == 0, (
            f"Chain-failure path must not call _capture_tier2_instructions; "
            f"called {cap_spy.call_count} times in branch={branch}"
        )
        # Tier-2 pin present with sentinel.
        pin_indices = [
            i for i, m in enumerate(result)
            if m.get("role") == "system" and "AGENTS.md" in m.get("content", "")
        ]
        assert len(pin_indices) == 1, (
            f"Exactly one Tier-2 pin expected in branch={branch}; "
            f"got positions={pin_indices}"
        )
        assert sentinel_content in result[pin_indices[0]]["content"], (
            f"Stable sentinel missing from Tier-2 pin in branch={branch}"
        )
        # VFS presence matches branch.
        vfs_present = any("[VFS" in m.get("content", "") for m in result)
        if branch == "affordable":
            assert vfs_present, f"VFS expected in {branch}"
        else:
            assert not vfs_present, f"VFS NOT expected in {branch}"
        self._assert_no_chain_state((SESSION, MODEL))

    def test_nonpositive_budget_preserves_minimal_fallback_without_vfs(self, monkeypatch):
        """Slice 4D: a nonempty candidate cannot fit a zero-token budget.
        ``compute_chain`` must still not be called (the budget check fires
        first); the call now raises ``ContextLimitError`` instead of
        returning a silent fallback.
        """
        from unittest.mock import patch as _patch
        from proxy import chain_hash as _chain
        from proxy.context import ContextLimitError
        import uuid
        run_id = uuid.uuid4().hex[:8]
        SESSION = f"wave6-nonpos-{run_id}"
        MODEL = f"wave6-nonpos-model-{run_id}"

        # Use a SafeConfig with safe=0 to trigger the nonpositive-budget early return.
        cfg = {
            "max_context_tokens": 40_000,
            "tokens_per_char": 0.25,
            "safe_context_tokens": 0,
            "raw_exchange_count": 100,
        }
        monkeypatch.setattr(
            "proxy.context.get_context_config",
            lambda *_a, **_kw: cfg,
        )

        messages = [{"role": "user", "content": "this message is long enough to overflow a zero-token budget"}]
        # Use a wrapping spy to assert compute_chain is NOT called.
        with _patch.object(_chain, "compute_chain") as chain_spy:
            from proxy.context_v2 import _select_context_v2
            with pytest.raises(ContextLimitError) as excinfo:
                _select_context_v2(
                    SESSION, messages, model_name=MODEL, ctx=None,
                )

        assert chain_spy.call_count == 0, (
            f"Nonpositive budget must not invoke compute_chain; "
            f"called {chain_spy.call_count} times"
        )
        assert "Final context exceeds safe token budget" in str(excinfo.value)

    def test_empty_messages_preserve_empty_fallback_without_vfs(self, monkeypatch):
        """Empty messages return before compute_chain; no VFS."""
        from unittest.mock import patch as _patch
        from proxy import chain_hash as _chain
        import uuid
        run_id = uuid.uuid4().hex[:8]
        SESSION = f"wave6-empty-{run_id}"
        MODEL = f"wave6-empty-model-{run_id}"

        monkeypatch.setattr(
            "proxy.context.get_context_config",
            lambda *_a, **_kw: self._cfg(),
        )

        messages = []
        with _patch.object(_chain, "compute_chain") as chain_spy:
            from proxy.context_v2 import _select_context_v2
            result = _select_context_v2(
                SESSION, messages, model_name=MODEL, ctx=None,
            )

        assert chain_spy.call_count == 0, (
            f"Empty messages must not invoke compute_chain; "
            f"called {chain_spy.call_count} times"
        )
        assert not any("[VFS" in m.get("content", "") for m in result)
        # Result equals the existing empty fallback ([]).
        assert result == []



