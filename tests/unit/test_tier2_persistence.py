"""Tier-2 persistence across V2 eviction — spec §9.4.

The core value proposition: Tier-2 sub-AGENTS.md content is captured into
the durable map BEFORE eviction, then re-emitted as a stable system pin
AFTER eviction has wiped the original tool result from the outbound.

Every test asserts its PRECONDITION (eviction actually fired) then the
OUTCOME (pin still present). If eviction doesn't fire, the test is vacuous
and will FAIL loudly.
"""
import copy
import uuid
from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

import proxy.state as state
from proxy.context_v2 import _select_context_v2

from tests.fixtures.tier2_payload import (
    RAW_TOOL_CONTENT,
    INSTRUCTION_BLOCKS,
)
from tests.fixtures.v2_eviction_harness import run_v2_turns


pytestmark = pytest.mark.unit

_SUB_INSTR_TAG = "<CLIENT_SUB_INSTRUCTIONS"
_CLIENT_SUB_CLOSE = "</CLIENT_SUB_INSTRUCTIONS>"


def _system(content="sys"):
    return {"role": "system", "content": content}


def _user(content):
    return {"role": "user", "content": content}


def _assistant(content=""):
    return {"role": "assistant", "content": content}


def _tool(tc_id, content):
    return {"role": "tool", "tool_call_id": tc_id, "content": content}


def _assistant_with_tool_call(tc_id, content=""):
    return {
        "role": "assistant",
        "content": content,
        "tool_calls": [{
            "id": tc_id,
            "type": "function",
            "function": {"name": "read", "arguments": "{}"},
        }],
    }


def _budget_config(safe=8000, tokens_per_char=0.25):
    return {
        "max_context_tokens": 40_000,
        "tokens_per_char": tokens_per_char,
        "safe_context_tokens": safe,
        "raw_exchange_count": 100,
    }


def _make_ctx(turn, workspace_name=""):
    return SimpleNamespace(
        request_id=f"req-persist-{turn}-{uuid.uuid4().hex[:6]}",
        v2_boundary_idx=None,
        v2_boundary_chain=None,
        workspace_name=workspace_name,
    )


@pytest.fixture(autouse=True)
def _clean_tier2_state():
    with state.vfs_lock:
        state.instruction_files_by_workspace.clear()
        state._last_scanned_by_tool_call.clear()
    yield
    with state.vfs_lock:
        state.instruction_files_by_workspace.clear()
        state._last_scanned_by_tool_call.clear()


def _build_turns_with_growing_conversation(workspace, tool_tc_id="tc-tier2-read",
                                           num_turns=5, filler_size=2000):
    """Build a list of full-history message lists for each turn.

    Turn 0: system + Tier-2 tool result + small user question.
    Turn N: opencode re-sends full history + new user/assistant exchange.

    Returns (turn_msg_lists, tool_tc_id, workspace).
    """
    tool_msg = _tool(tool_tc_id, RAW_TOOL_CONTENT)
    base = [_system("sys"), tool_msg]

    turn_lists = []
    cumulative = list(base)
    for turn in range(num_turns):
        if turn == 0:
            cumulative.append(_user("initial question"))
        else:
            cumulative.append(_user(f"follow-up question {turn} " + "Q" * filler_size))
            cumulative.append(_assistant(f"response {turn} " + "R" * filler_size))
        turn_lists.append(copy.deepcopy(cumulative))

    return turn_lists, tool_tc_id, workspace


def _assert_eviction_fired(results, tool_tc_id):
    """Assert that eviction actually fired: boundary_idx advanced AND the
    tool_call_id disappeared from some turn's outbound.

    Returns the first turn index where the tool disappeared.
    """
    boundary_values = [r["boundary_idx"] for r in results]
    max_boundary = max(boundary_values) if boundary_values else 0

    tool_visible_turns = []
    for r in results:
        if tool_tc_id in r["outbound_tool_call_ids"]:
            tool_visible_turns.append(r["turn"])

    assert max_boundary > 0, (
        f"PRECONDITION FAILED: boundary_idx never advanced. "
        f"Per-turn values: {boundary_values}. "
        f"Eviction did not fire — test is vacuous."
    )

    assert len(tool_visible_turns) < len(results), (
        f"PRECONDITION FAILED: tool {tool_tc_id!r} present in ALL turns "
        f"(visible in turns {tool_visible_turns}). "
        f"Eviction did not remove it — test is vacuous."
    )

    evicted_turn = None
    for r in results:
        if tool_tc_id not in r["outbound_tool_call_ids"]:
            evicted_turn = r["turn"]
            break

    return evicted_turn, boundary_values


class TestTier2PersistenceMultiTurn:
    """§9.4: Multi-turn persistence — Tier-2 survives eviction."""

    def test_tier2_pin_survives_eviction(self):
        """Two-phase: Phase 1 runs with a LARGE budget so the Tier-2 tool
        result IS in raw_block at capture time → map populates. Phase 2
        runs with a SMALL budget so the cold-path boundary advances PAST
        the tool result → tool evicts. Pin survives from the map.

        PRECONDITION: the tool message is NOT in the phase-2 outbound AND
        boundary_idx advanced beyond the tool's position.
        OUTCOME: the pin IS present in the phase-2 outbound despite the
        tool result being gone.
        """
        ws = f"ws-persist-{uuid.uuid4().hex[:8]}"
        tool_tc_id = "tc-tier2-read"

        tool_msg = _tool(tool_tc_id, RAW_TOOL_CONTENT)
        conversation = [
            _system("sys"),
            _user("read file " + "Q" * 6000),
            _assistant_with_tool_call(tool_tc_id),
            tool_msg,
            _user("follow-up " + "Z" * 5000),
        ]

        large_cfg = _budget_config(safe=40_000)
        phase1_ctx = _make_ctx(0, workspace_name=ws)
        state._chain_state.clear()
        mock_conn = MagicMock()
        mock_conn_cm = MagicMock()
        mock_conn_cm.__enter__ = MagicMock(return_value=mock_conn)
        mock_conn_cm.__exit__ = MagicMock(return_value=False)

        with ExitStack() as stack:
            stack.enter_context(patch("proxy.context.get_context_config", return_value=large_cfg))
            stack.enter_context(patch("memory.db.get_ancient_history", return_value=""))
            stack.enter_context(patch("memory.db.get_rich_summaries", return_value=[]))
            stack.enter_context(patch("proxy.vfs._build_vfs_inherited_context", return_value=None))
            stack.enter_context(patch("proxy.vfs._build_vfs_context", return_value=None))
            stack.enter_context(patch(
                "memory.db_summaries.get_last_summarized_exchange_index",
                return_value=-1,
            ))
            stack.enter_context(patch("memory.sync.submit_ancient_compress", return_value=None))
            stack.enter_context(patch(
                "proxy.context_v2._load_state_from_db",
                return_value=None,
            ))
            stack.enter_context(patch("memory.db.connection", return_value=mock_conn_cm))
            phase1_outbound = _select_context_v2(
                "sess-persist", copy.deepcopy(conversation),
                model_name="test-model", ctx=phase1_ctx,
            )

        phase1_map = state.get_instruction_files(ws)
        assert phase1_map, (
            "PRECONDITION FAILED: Phase 1 (large budget) must populate the map "
            "from the Tier-2 tool result. Capture runs on raw_block BEFORE "
            "_cleanup_tool_pair_integrity drops orphan tools."
        )

        small_cfg = _budget_config(safe=20_000)
        phase2_ctx = _make_ctx(1, workspace_name=ws)

        with ExitStack() as stack:
            stack.enter_context(patch("proxy.context.get_context_config", return_value=small_cfg))
            stack.enter_context(patch("memory.db.get_ancient_history", return_value=""))
            stack.enter_context(patch("memory.db.get_rich_summaries", return_value=[]))
            stack.enter_context(patch("proxy.vfs._build_vfs_inherited_context", return_value=None))
            stack.enter_context(patch("proxy.vfs._build_vfs_context", return_value=None))
            stack.enter_context(patch(
                "memory.db_summaries.get_last_summarized_exchange_index",
                return_value=-1,
            ))
            stack.enter_context(patch("memory.sync.submit_ancient_compress", return_value=None))
            stack.enter_context(patch(
                "proxy.context_v2._load_state_from_db",
                return_value=None,
            ))
            stack.enter_context(patch("memory.db.connection", return_value=mock_conn_cm))
            phase2_outbound = _select_context_v2(
                "sess-persist", copy.deepcopy(conversation),
                model_name="test-model", ctx=phase2_ctx,
            )

        # The instruction map MUST survive regardless of phase 2 outcome.
        phase2_map = state.get_instruction_files(ws)
        assert phase2_map, (
            "Tier-2 instruction map MUST survive phase 2 (capture persisted "
            "from phase 1). Map state is the cross-phase invariant."
        )

        phase2_tool_present = tool_tc_id in {
            m.get("tool_call_id") for m in phase2_outbound if m.get("role") == "tool"
        }
        assert not phase2_tool_present, (
            f"PRECONDITION FAILED: tool {tool_tc_id!r} must be EVICTED in "
            f"Phase 2 (small budget). boundary_idx={phase2_ctx.v2_boundary_idx}. "
            "Eviction did not fire."
        )

        pin_msgs = [
            m for m in phase2_outbound
            if m.get("role") == "system" and _SUB_INSTR_TAG in m.get("content", "")
        ]
        assert len(pin_msgs) == 1, (
            f"OUTCOME FAILED: expected exactly 1 Tier-2 pin in Phase 2 "
            f"outbound, got {len(pin_msgs)}. "
            f"Tool was evicted (boundary_idx={phase2_ctx.v2_boundary_idx}), "
            f"but pin did not survive."
        )

        pin_content = pin_msgs[0]["content"]
        assert _CLIENT_SUB_CLOSE in pin_content, "pin must be properly closed"

        for instr_path, _ in INSTRUCTION_BLOCKS:
            assert instr_path in pin_content, (
                f"pin missing captured path {instr_path}"
            )

    def test_persistence_reported_diagnostics(self):
        """Diagnostic test: two-phase approach showing capture-then-evict."""
        ws = f"ws-diag-{uuid.uuid4().hex[:8]}"
        tool_tc_id = "tc-tier2-diag"

        tool_msg = _tool(tool_tc_id, RAW_TOOL_CONTENT)
        conversation = [
            _system("sys"),
            _user("read file " + "Q" * 6000),
            _assistant_with_tool_call(tool_tc_id),
            tool_msg,
            _user("follow-up " + "Z" * 5000),
        ]

        lines = []

        for phase_name, safe in [("CAPTURE", 40_000), ("EVICT", 20_000)]:
            ctx = _make_ctx(0, workspace_name=ws)
            cfg = _budget_config(safe=safe)
            state._chain_state.clear()
            mock_conn = MagicMock()
            mock_conn_cm = MagicMock()
            mock_conn_cm.__enter__ = MagicMock(return_value=mock_conn)
            mock_conn_cm.__exit__ = MagicMock(return_value=False)

            with ExitStack() as stack:
                stack.enter_context(patch("proxy.context.get_context_config", return_value=cfg))
                stack.enter_context(patch("memory.db.get_ancient_history", return_value=""))
                stack.enter_context(patch("memory.db.get_rich_summaries", return_value=[]))
                stack.enter_context(patch("proxy.vfs._build_vfs_inherited_context", return_value=None))
                stack.enter_context(patch("proxy.vfs._build_vfs_context", return_value=None))
                stack.enter_context(patch(
                    "memory.db_summaries.get_last_summarized_exchange_index",
                    return_value=-1,
                ))
                stack.enter_context(patch("memory.sync.submit_ancient_compress", return_value=None))
                stack.enter_context(patch(
                    "proxy.context_v2._load_state_from_db",
                    return_value=None,
                ))
                stack.enter_context(patch("memory.db.connection", return_value=mock_conn_cm))
                outbound = _select_context_v2(
                    "sess-diag", copy.deepcopy(conversation),
                    model_name="test-model", ctx=ctx,
                )

            tool_present = tool_tc_id in {
                m.get("tool_call_id") for m in outbound if m.get("role") == "tool"
            }
            pin_count = sum(
                1 for m in outbound
                if m.get("role") == "system" and _SUB_INSTR_TAG in m.get("content", "")
            )
            map_size = len(state.get_instruction_files(ws))
            lines.append(
                f"  phase={phase_name} safe={safe} boundary={ctx.v2_boundary_idx} "
                f"tool_present={tool_present} pins={pin_count} map_size={map_size}"
            )

        diag = "\n".join(lines)
        print(f"\n[persistence-diagnostics]\n{diag}")

        # Persistence contract: the EVICT phase (small budget) MUST emit
        # the Tier-2 pin in its outbound (capture persisted from CAPTURE
        # phase, pin re-emitted post-eviction). This makes the diagnostic
        # test non-tautological per tests/AGENTS.md §4.
        evict_line = next(
            line for line in lines if line.strip().startswith("phase=EVICT")
        )
        assert "pins=1" in evict_line, (
            f"Persistence contract failed: EVICT phase must emit exactly 1 "
            f"Tier-2 pin (capture persisted across phases). diag=\n{diag}"
        )

    def test_persistence_revert_test(self):
        """Revert-test: temporarily disable _emit_tier2_pin (make it return
        messages unchanged); confirm NO pin appears after eviction.

        Two-phase: Phase 1 populates the map (emitter disabled AFTER capture
        but we need capture to still work — so we only disable emitter for
        Phase 2). This proves Phase 2's pin comes from _emit_tier2_pin, not
        from anywhere else.
        """
        ws = f"ws-revert-{uuid.uuid4().hex[:8]}"
        tool_tc_id = "tc-tier2-revert"

        tool_msg = _tool(tool_tc_id, RAW_TOOL_CONTENT)
        conversation = [
            _system("sys"),
            _user("read file " + "Q" * 6000),
            _assistant_with_tool_call(tool_tc_id),
            tool_msg,
            _user("follow-up " + "Z" * 5000),
        ]

        large_cfg = _budget_config(safe=40_000)
        state._chain_state.clear()
        mock_conn = MagicMock()
        mock_conn_cm = MagicMock()
        mock_conn_cm.__enter__ = MagicMock(return_value=mock_conn)
        mock_conn_cm.__exit__ = MagicMock(return_value=False)

        with ExitStack() as stack:
            stack.enter_context(patch("proxy.context.get_context_config", return_value=large_cfg))
            stack.enter_context(patch("memory.db.get_ancient_history", return_value=""))
            stack.enter_context(patch("memory.db.get_rich_summaries", return_value=[]))
            stack.enter_context(patch("proxy.vfs._build_vfs_inherited_context", return_value=None))
            stack.enter_context(patch("proxy.vfs._build_vfs_context", return_value=None))
            stack.enter_context(patch(
                "memory.db_summaries.get_last_summarized_exchange_index",
                return_value=-1,
            ))
            stack.enter_context(patch("memory.sync.submit_ancient_compress", return_value=None))
            stack.enter_context(patch(
                "proxy.context_v2._load_state_from_db",
                return_value=None,
            ))
            stack.enter_context(patch("memory.db.connection", return_value=mock_conn_cm))
            _select_context_v2(
                "sess-revert", copy.deepcopy(conversation),
                model_name="test-model",
                ctx=_make_ctx(0, workspace_name=ws),
            )

        assert state.get_instruction_files(ws), (
            "Phase 1 must populate the map (emitter was live)"
        )

        import proxy.context_v2 as cv2_module
        original_emit = cv2_module._emit_tier2_pin

        def _noop_emit(messages, session_id, ctx=None):
            return messages

        cv2_module._emit_tier2_pin = _noop_emit
        try:
            small_cfg = _budget_config(safe=8000)
            phase2_ctx = _make_ctx(1, workspace_name=ws)

            with ExitStack() as stack:
                stack.enter_context(patch("proxy.context.get_context_config", return_value=small_cfg))
                stack.enter_context(patch("memory.db.get_ancient_history", return_value=""))
                stack.enter_context(patch("memory.db.get_rich_summaries", return_value=[]))
                stack.enter_context(patch("proxy.vfs._build_vfs_inherited_context", return_value=None))
                stack.enter_context(patch("proxy.vfs._build_vfs_context", return_value=None))
                stack.enter_context(patch(
                    "memory.db_summaries.get_last_summarized_exchange_index",
                    return_value=-1,
                ))
                stack.enter_context(patch("memory.sync.submit_ancient_compress", return_value=None))
                stack.enter_context(patch(
                    "proxy.context_v2._load_state_from_db",
                    return_value=None,
                ))
                stack.enter_context(patch("memory.db.connection", return_value=mock_conn_cm))
                phase2_outbound = _select_context_v2(
                    "sess-revert", copy.deepcopy(conversation),
                    model_name="test-model", ctx=phase2_ctx,
                )

            phase2_tool_present = tool_tc_id in {
                m.get("tool_call_id") for m in phase2_outbound if m.get("role") == "tool"
            }
            assert not phase2_tool_present, (
                "PRECONDITION: tool must be evicted in Phase 2"
            )

            pin_msgs = [
                m for m in phase2_outbound
                if m.get("role") == "system" and _SUB_INSTR_TAG in m.get("content", "")
            ]

            assert len(pin_msgs) == 0, (
                "REVERT-TEST FAILED: pin should NOT be present when "
                "_emit_tier2_pin is disabled, but found "
                f"{len(pin_msgs)} pin(s). The persistence test is "
                "tautological — it passes regardless of whether the "
                "pin emitter works."
            )

            print(
                f"\n[revert-test] Tool evicted (boundary={phase2_ctx.v2_boundary_idx}), "
                f"NO pin present (as expected with emitter disabled)."
            )
        finally:
            cv2_module._emit_tier2_pin = original_emit


class TestTier2SubAgentLag:
    """§8/§9.4: Sub-agent lag — map populates on a main turn, pin emits
    on the next sub-agent turn."""

    def test_sub_agent_lag_two_phase(self):
        """Two-phase: turn N sub-agent request, map EMPTY → no pin.
        Turn N+1 main request reads subfolder → map populates.
        Turn N+2 sub-agent → pin NOW emits."""
        ws = f"ws-lag-{uuid.uuid4().hex[:8]}"
        sub_sid = f"sub-lag-{uuid.uuid4().hex[:8]}"

        tool_msg = _tool("tc-lag-read", RAW_TOOL_CONTENT)

        sub_only_msgs = [
            _system("sys"),
            _user("sub-agent task"),
            _assistant("working..."),
        ]

        main_with_read_msgs = [
            _system("sys"),
            tool_msg,
            _user("main question"),
            _assistant("main answer"),
        ]

        sub_after_populate_msgs = [
            _system("sys"),
            _user("sub-agent follow-up"),
            _assistant("still working..."),
        ]

        turn_lists = [sub_only_msgs, main_with_read_msgs, sub_after_populate_msgs]

        results = run_v2_turns(
            session_id=sub_sid,
            turn_msg_lists=turn_lists,
            cfg=_budget_config(safe=30_000),
            ctx_factory=lambda turn: _make_ctx(turn, workspace_name=ws),
        )

        turn0_pins = [
            m for m in results[0]["outbound"]
            if m.get("role") == "system" and _SUB_INSTR_TAG in m.get("content", "")
        ]
        assert len(turn0_pins) == 0, (
            "Phase 1: sub-agent with empty map must NOT produce a pin"
        )

        assert results[1]["captured_map"], (
            "Phase 2: main turn with Tier-2 read must populate the map"
        )

        turn2_pins = [
            m for m in results[2]["outbound"]
            if m.get("role") == "system" and _SUB_INSTR_TAG in m.get("content", "")
        ]
        assert len(turn2_pins) == 1, (
            f"Phase 3: sub-agent after map populated must emit pin, "
            f"got {len(turn2_pins)} pins"
        )


class TestTier2RootSkip:
    """§9.4: Root-skip — a Tier-2 suffix that includes the Tier-1 root
    path → root NOT in the captured map."""

    def test_root_path_not_in_captured_map(self):
        """A Tier-2 suffix containing the Tier-1 root path alongside a
        valid sub-AGENTS.md → root path is excluded from the map."""
        ws = f"ws-rootskip-{uuid.uuid4().hex[:8]}"
        root_path = "/repo/project/AGENTS.md"
        sub_path = "/repo/project/src/proxy/sub/AGENTS.md"

        sub_content = "# sub-agents content\n# landmine: tool_parsing/AGENTS.md"
        root_content = "# ROOT AGENTS.md — Tier-1, must NOT be in Tier-2 map"

        suffix = (
            "<system-reminder>\n"
            f"Instructions from: {sub_path}\n"
            f"{sub_content}\n"
            f"Instructions from: {root_path}\n"
            f"{root_content}\n"
            "</system-reminder>"
        )
        read_path = "/repo/project/src/proxy/sub/route.py"
        raw_content = (
            f"<path>{read_path}</path><type>file</type><content>\n"
            "def foo(): pass\n"
            "</content>" + suffix
        )

        tool_msg = _tool("tc-rootskip", raw_content)
        msgs = [
            _system(f"Instructions from: {root_path}\n{root_content}"),
            _user("read the file " + "X" * 2000),
            _assistant_with_tool_call("tc-rootskip"),
            tool_msg,
            _user("current question " + "Y" * 2000),
        ]

        ctx = _make_ctx(0, workspace_name=ws)
        cfg = _budget_config(safe=3_000)

        state._chain_state.clear()
        mock_conn = MagicMock()
        mock_conn_cm = MagicMock()
        mock_conn_cm.__enter__ = MagicMock(return_value=mock_conn)
        mock_conn_cm.__exit__ = MagicMock(return_value=False)

        with ExitStack() as stack:
            stack.enter_context(patch("proxy.context.get_context_config", return_value=cfg))
            stack.enter_context(patch("memory.db.get_ancient_history", return_value=""))
            stack.enter_context(patch("memory.db.get_rich_summaries", return_value=[]))
            stack.enter_context(patch("proxy.vfs._build_vfs_inherited_context", return_value=None))
            stack.enter_context(patch("proxy.vfs._build_vfs_context", return_value=None))
            stack.enter_context(patch(
                "memory.db_summaries.get_last_summarized_exchange_index",
                return_value=-1,
            ))
            stack.enter_context(patch("memory.sync.submit_ancient_compress", return_value=None))
            stack.enter_context(patch(
                "proxy.context_v2._load_state_from_db",
                return_value=None,
            ))
            stack.enter_context(patch("memory.db.connection", return_value=mock_conn_cm))
            _select_context_v2(
                "sess-rootskip", copy.deepcopy(msgs),
                model_name="test-model", ctx=ctx,
            )

        files = state.get_instruction_files(ws)
        assert sub_path in files, "sub-AGENTS.md must be in the map"
        assert root_path not in files, (
            f"root path {root_path} must NOT be in the captured map "
            f"(root-skip defensive anchor)"
        )
