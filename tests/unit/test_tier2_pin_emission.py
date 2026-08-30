"""Tier-2 sub-AGENTS.md pin emission — parametrized across EVERY return path.

Spec §9.3: for EACH return path in _select_context_v2 and _build_subagent_context,
assert the pin is present in the output of THAT path with the correct wrapper,
escape behavior, and position.

Return paths covered:
  1. Main (line ~1338 — the tail return, common case)
  2. Fast-exit (chain-state miss + no DB rows — C5 replacement for the
     removed passthrough return)
  3. compute_chain-fail (line ~1040 — chain hash computation raises)
  4. warm-CAS-fail (line ~1296 — version mismatch on warm path, falls to main return)
  5. cold-CAS-fail (line ~1322 — cold-init CAS lost, falls to main return)
  6. Sub-agent fits (:314)
  7. Sub-agent no-first-user (:335)
  8. Sub-agent trimmed-fits (:361)
  9. Sub-agent fallback (:369)
"""
import copy
import uuid
from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

import proxy.state as state
from proxy.context_v2 import _select_context_v2

from tests.fixtures.subagent_request_builder import (
    _detect_return_path,
    make_subagent_request,
)
from tests.fixtures.tier2_payload import RAW_TOOL_CONTENT, INSTRUCTION_BLOCKS


pytestmark = pytest.mark.unit

_SUB_INSTR_TAG = "<CLIENT_SUB_INSTRUCTIONS"
_PROJ_INSTR_TAG = "<CLIENT_PROJECT_INSTRUCTIONS"
_CLIENT_SUB_CLOSE = "</CLIENT_SUB_INSTRUCTIONS>"
# _escape_prompt_field escapes structural prompt tags (TASK_PAYLOAD, TASK_CONTEXT,
# MANDATORY_SYSTEM_INSTRUCTIONS, CLIENT_PROJECT_INSTRUCTIONS) — not generic
# <script>. We inject TASK_PAYLOAD as the dangerous-tag canary.
_DANGER_INJECT = "<TASK_PAYLOAD>"
_ESCAPED_DANGER = "&lt;TASK_PAYLOAD&gt;"


def _system(content="sys"):
    return {"role": "system", "content": content}


def _user(content):
    return {"role": "user", "content": content}


def _assistant(content=""):
    return {"role": "assistant", "content": content}


def _tool(tc_id, content):
    return {"role": "tool", "tool_call_id": tc_id, "content": content}


def _budget_config(safe=10_000, tokens_per_char=0.25):
    return {
        "max_context_tokens": 40_000,
        "tokens_per_char": tokens_per_char,
        "safe_context_tokens": safe,
        "raw_exchange_count": 100,
    }


def _make_ctx(request_id="test-req", workspace_name=""):
    return SimpleNamespace(
        request_id=request_id,
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
    from proxy.agent.agent_registry import registry as _reg
    with _reg._lock:
        _reg._agents.clear()
        _reg._agents_by_id.clear()
        _reg._lru.clear()


def _populate_map_with_script(workspace, session_id=None):
    """Populate the instruction map with content containing a dangerous
    structural tag that _escape_prompt_field will HTML-entity-encode.

    Uses a unique path that won't collide with the capture pipeline's
    recorded blocks (which use the fixture's real AGENTS.md paths).
    """
    with state.vfs_lock:
        state.record_instruction_file(
            workspace,
            "/repo/project/src/proxy/sub/INJECTED_DANGER.md",
            "Landmine content with <TASK_PAYLOAD> inside.",
            session_id=session_id,
        )


def _build_main_path_conversation(tool_content=None, safe=30_000, padding_size=6000):
    """Build a conversation that reaches the main V2 path (not fast-exit).

    The conversation size is irrelevant to the main path now that the
    passthrough threshold is removed (C5). All conversations flow through
    the tiered pipeline; the size just controls whether eviction fires.
    """
    if tool_content is None:
        tool_content = RAW_TOOL_CONTENT
    msgs = [_system("sys")]
    msgs.append(_tool("tc-read-main", tool_content))
    for i in range(3):
        msgs.append(_user(f"later user {i} " + "X" * padding_size))
        msgs.append(_assistant(f"later asst {i} " + "Y" * padding_size))
    msgs.append(_user("current question"))
    return msgs


def _assert_pin_present(result, workspace):
    """Assert the Tier-2 pin is present with correct wrapper and escape."""
    pin_msg = None
    pin_idx = -1
    for i, m in enumerate(result):
        if m.get("role") != "system":
            continue
        content = m.get("content", "")
        if _SUB_INSTR_TAG in content:
            pin_msg = m
            pin_idx = i
            break
    assert pin_msg is not None, (
        f"pin not found in result for workspace={workspace!r}. "
        f"System msgs: {[m.get('content','')[:80] for m in result if m.get('role')=='system']}"
    )
    content = pin_msg["content"]
    # Distinct wrapper
    assert _SUB_INSTR_TAG in content, "pin must use <CLIENT_SUB_INSTRUCTIONS>"
    assert _PROJ_INSTR_TAG not in content, "pin must NOT use <CLIENT_PROJECT_INSTRUCTIONS>"
    assert _CLIENT_SUB_CLOSE in content, "pin must be properly closed"
    return pin_idx


def _assert_escape(pin_content):
    """Assert dangerous structural tags are escaped correctly by
    _escape_prompt_field: &lt;TASK_PAYLOAD&gt; present, raw <TASK_PAYLOAD>
    absent (outside the escaped form), no double-escape (&amp;lt;).
    """
    assert _ESCAPED_DANGER in pin_content, (
        f"escaped dangerous tag must appear as {_ESCAPED_DANGER!r} in pin content"
    )
    # Strip all escaped occurrences, then check no raw form remains
    stripped = pin_content.replace(_ESCAPED_DANGER, "")
    assert _DANGER_INJECT not in stripped, (
        f"raw {_DANGER_INJECT!r} must NOT appear in pin content (outside escaped form)"
    )
    assert "&amp;lt;" not in pin_content, (
        "double-escaped &amp;lt; must NOT appear (no double-escape)"
    )


def _assert_position(result, pin_idx):
    """Assert pin_idx < summaries_idx (if summaries exist) and pin_idx < ancient_idx.

    The pin must come BEFORE summaries and ancient context blocks. When no
    system message precedes the pin (sub-agent degenerate paths), pin_idx=0
    is correct — the pin IS the first system message.
    """
    summaries_idx = None
    ancient_idx = None
    for i, m in enumerate(result):
        if m.get("role") != "system":
            continue
        content = m.get("content", "")
        if ("[PAST CONTEXT" in content or "[Conversation Summary" in content
                and summaries_idx is None):
            summaries_idx = i
        if "[TIMELINE" in content and ancient_idx is None:
            ancient_idx = i
    # pin must come before summaries
    if summaries_idx is not None:
        assert pin_idx < summaries_idx, (
            f"pin_idx={pin_idx} must be < summaries_idx={summaries_idx}"
        )
    # pin must come before ancient
    if ancient_idx is not None:
        assert pin_idx < ancient_idx, (
            f"pin_idx={pin_idx} must be < ancient_idx={ancient_idx}"
        )


def _run_v2(
    session_id,
    messages,
    ctx,
    cfg,
    is_subagent=False,
    summaries=None,
    **kwargs,
):
    """Run _select_context_v2 with standard mocks."""
    state._chain_state.clear()
    mock_conn = MagicMock()
    mock_conn_cm = MagicMock()
    mock_conn_cm.__enter__ = MagicMock(return_value=mock_conn)
    mock_conn_cm.__exit__ = MagicMock(return_value=False)
    with ExitStack() as stack:
        stack.enter_context(patch("proxy.context.get_context_config", return_value=cfg))
        stack.enter_context(patch("memory.db.get_ancient_history", return_value=""))
        stack.enter_context(
            patch(
                "memory.db.get_rich_summaries",
                return_value=[] if summaries is None else summaries,
            )
        )
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
        result = _select_context_v2(
            session_id, copy.deepcopy(messages),
            model_name="test-model", ctx=ctx,
            is_subagent=is_subagent,
            **kwargs,
        )
    return result


class TestPinEmissionMainPath:
    """Pin emission on the main return path (line ~1338)."""

    def test_main_path_pin_emitted(self):
        ws = f"ws-main-{uuid.uuid4().hex[:8]}"
        _populate_map_with_script(ws)
        msgs = _build_main_path_conversation()
        ctx = _make_ctx(workspace_name=ws)
        cfg = _budget_config(safe=30_000)
        result = _run_v2("sess-main", msgs, ctx, cfg)
        pin_idx = _assert_pin_present(result, ws)
        _assert_position(result, pin_idx)
        _assert_escape(next(m["content"] for m in result if _SUB_INSTR_TAG in m.get("content", "")))

    def test_main_path_no_tier1_collision(self):
        """The Tier-2 pin uses <CLIENT_SUB_INSTRUCTIONS>, NOT
        <CLIENT_PROJECT_INSTRUCTIONS>. Tier-1 is emitted by MotivatorStage
        (outside _select_context_v2 scope) — here we just verify the Tier-2
        pin doesn't accidentally use the Tier-1 wrapper."""
        ws = f"ws-main-one-{uuid.uuid4().hex[:8]}"
        _populate_map_with_script(ws)
        msgs = _build_main_path_conversation()
        ctx = _make_ctx(workspace_name=ws)
        cfg = _budget_config(safe=30_000)
        result = _run_v2("sess-tier1", msgs, ctx, cfg)
        sub_count = sum(
            1 for m in result
            if _SUB_INSTR_TAG in m.get("content", "")
        )
        assert sub_count == 1, f"expected exactly 1 Tier-2 pin, got {sub_count}"
        # The Tier-2 pin content must NOT contain the Tier-1 wrapper
        pin_content = next(
            m["content"] for m in result
            if _SUB_INSTR_TAG in m.get("content", "")
        )
        assert _PROJ_INSTR_TAG not in pin_content, (
            "Tier-2 pin must NOT use <CLIENT_PROJECT_INSTRUCTIONS> wrapper"
        )


class TestPinEmissionFastExit:
    """Pin emission on the fast-exit return path (C5 replacement for the
    removed passthrough return).

    The fast-exit fires when chain-state is a miss AND no
    conversation_summaries / message_summaries rows exist. It still runs
    `_emit_tier2_pin` on the way out.
    """

    def test_fast_exit_pin_emitted(self):
        ws = f"ws-pass-{uuid.uuid4().hex[:8]}"
        _populate_map_with_script(ws)
        # Small conversation — fits under fast-exit (chain-state miss +
        # no DB rows)
        msgs = [
            _system("sys"),
            _user("hello"),
            _assistant("hi there"),
        ]
        ctx = _make_ctx(workspace_name=ws)
        cfg = _budget_config(safe=100_000)

        state._chain_state.clear()
        mock_conn = MagicMock()
        mock_conn_cm = MagicMock()
        mock_conn_cm.__enter__ = MagicMock(return_value=mock_conn)
        mock_conn_cm.__exit__ = MagicMock(return_value=False)
        mock_conn.execute.return_value = MagicMock(fetchone=MagicMock(return_value=None))

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
            result = _select_context_v2(
                "sess-pass", copy.deepcopy(msgs),
                model_name="test-model", ctx=ctx,
            )
        # Verify we took fast-exit (boundary_idx is None)
        assert ctx.v2_boundary_idx is None, (
            "precondition: must have taken fast-exit path"
        )
        pin_idx = _assert_pin_present(result, ws)
        _assert_position(result, pin_idx)
        _assert_escape(next(m["content"] for m in result if _SUB_INSTR_TAG in m.get("content", "")))


class TestPinEmissionComputeChainFail:
    """Pin emission on the compute_chain-fail return path (line ~1040).

    When compute_chain raises, the pipeline falls back to emitting the full
    messages with the pin.
    """

    def test_compute_chain_fail_pin_emitted(self):
        ws = f"ws-chainfail-{uuid.uuid4().hex[:8]}"
        _populate_map_with_script(ws)
        msgs = _build_main_path_conversation()
        ctx = _make_ctx(workspace_name=ws)
        cfg = _budget_config(safe=30_000)
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
            stack.enter_context(patch(
                "proxy.chain_hash.compute_chain",
                side_effect=TypeError("mock chain failure"),
            ))
            result = _select_context_v2(
                "sess-chainfail", copy.deepcopy(msgs),
                model_name="test-model", ctx=ctx,
            )
        # Verify we took the compute_chain-fail path (boundary_idx is None)
        assert ctx.v2_boundary_idx is None, (
            "precondition: compute_chain-fail path must set boundary_idx to None"
        )
        pin_idx = _assert_pin_present(result, ws)
        _assert_position(result, pin_idx)
        _assert_escape(next(m["content"] for m in result if _SUB_INSTR_TAG in m.get("content", "")))


class TestPinEmissionCASFail:
    """Pin emission on the warm-CAS-fail and cold-CAS-fail paths.

    CAS failures don't return early — they set cas_failed=True and fall through
    to the main return at line ~1338. We test both by forcing the CAS condition.
    """

    def test_warm_cas_fail_pin_emitted(self):
        """Warm-path CAS failure: snap_version is not None, but version mismatch
        occurs. The pin still emits from the shared main return."""
        ws = f"ws-warmcas-{uuid.uuid4().hex[:8]}"
        _populate_map_with_script(ws)
        msgs = _build_main_path_conversation()
        ctx = _make_ctx(workspace_name=ws)
        cfg = _budget_config(safe=30_000)
        state._chain_state.clear()
        mock_conn = MagicMock()
        mock_conn_cm = MagicMock()
        mock_conn_cm.__enter__ = MagicMock(return_value=mock_conn)
        mock_conn_cm.__exit__ = MagicMock(return_value=False)
        # Pre-populate chain_state with an entry that has version > 0
        # so snap_version is not None on the warm path.
        key = ("sess-warmcas", "test-model")
        entry = state.new_chain_state_entry()
        entry["last_chain"] = []
        entry["boundary_idx"] = 0
        entry["boundary_chain"] = None
        entry["summaries"] = []
        entry["ancient"] = ""
        entry["version"] = 5
        state._chain_state[key] = entry
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
            # Simulate a concurrent modification: between the snap read
            # (line ~1046) and the CAS write (line ~1286), another thread
            # bumps the version. We do this by directly mutating the entry
            # in chain_state via a hook on find_fork_point (which runs
            # between snap and CAS).
            from proxy.chain_hash import find_fork_point as _real_find_fork
            def _concurrent_bump(*args, **kwargs):
                # Another thread bumps the version
                if key in state._chain_state:
                    state._chain_state[key]["version"] = 999
                return _real_find_fork(*args, **kwargs)
            stack.enter_context(patch(
                "proxy.chain_hash.find_fork_point",
                side_effect=_concurrent_bump,
            ))
            result = _select_context_v2(
                "sess-warmcas", copy.deepcopy(msgs),
                model_name="test-model", ctx=ctx,
            )
        # CAS fail sets boundary_idx to None
        assert ctx.v2_boundary_idx is None, (
            f"precondition: warm-CAS-fail must set boundary_idx to None, "
            f"got {ctx.v2_boundary_idx}"
        )
        pin_idx = _assert_pin_present(result, ws)
        _assert_position(result, pin_idx)
        _assert_escape(next(m["content"] for m in result if _SUB_INSTR_TAG in m.get("content", "")))

    def test_cold_cas_fail_pin_emitted(self):
        """Cold-init CAS failure: state_entry is None from DB, then between
        check and write, a concurrent thread populates _chain_state. The pin
        still emits from the shared main return."""
        ws = f"ws-coldcas-{uuid.uuid4().hex[:8]}"
        _populate_map_with_script(ws)
        msgs = _build_main_path_conversation()
        ctx = _make_ctx(workspace_name=ws)
        cfg = _budget_config(safe=30_000)
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
            # _load_state_from_db returns None → cold path
            stack.enter_context(patch(
                "proxy.context_v2._load_state_from_db",
                return_value=None,
            ))
            stack.enter_context(patch("memory.db.connection", return_value=mock_conn_cm))
            # After cold-init, the CAS write checks if another thread populated
            # the key. We simulate that by injecting into _chain_state during
            # the cold-path compute.
            original_cold = state.new_chain_state_entry
            key = ("sess-coldcas", "test-model")
            def _inject_concurrent_entry():
                """Simulate a concurrent thread populating chain_state."""
                concurrent = original_cold()
                concurrent["version"] = 1
                state._chain_state[key] = concurrent
            # Patch _cold_path_compute to inject a concurrent entry before
            # the CAS write checks
            from proxy.context_v2 import _cold_path_compute as _real_cold
            def _patched_cold(*args, **kwargs):
                result = _real_cold(*args, **kwargs)
                _inject_concurrent_entry()
                return result
            stack.enter_context(patch(
                "proxy.context_v2._cold_path_compute",
                side_effect=_patched_cold,
            ))
            result = _select_context_v2(
                "sess-coldcas", copy.deepcopy(msgs),
                model_name="test-model", ctx=ctx,
            )
        # cold-init CAS fail sets boundary_idx to None
        assert ctx.v2_boundary_idx is None, (
            "precondition: cold-init CAS fail must set boundary_idx to None"
        )
        pin_idx = _assert_pin_present(result, ws)
        _assert_position(result, pin_idx)
        _assert_escape(next(m["content"] for m in result if _SUB_INSTR_TAG in m.get("content", "")))


class TestPinEmissionSubAgent:
    """Pin emission on all 4 sub-agent return paths.

    Each path is reached via _build_subagent_context through make_subagent_request.

    Slice 4D correction: ``no-first-user`` and ``trimmed-fits`` paths now
    reserve Tier-2 cost before trimming, so the post-trim candidate plus
    Tier-2 pin fits ``safe_tokens``.

    Slice 4D restoration (2026-07-18): the ``fallback`` path previously
    raised ``ContextLimitError`` because the universal finalizer had no
    content-trim safety net. With the restored safety net (§13.5), the
    finalizer content-trims the oversized first-user message with a
    ``... [truncated]`` marker rather than raising. The test now asserts
    the trimmed-fits outcome instead of the raise.
    """

    @pytest.mark.parametrize("target_return", [
        "fits",
        "no-first-user",
        "trimmed-fits",
        "fallback",
    ])
    def test_sub_agent_pin_emitted(self, target_return):
        ws = f"ws-sub-{target_return}-{uuid.uuid4().hex[:8]}"
        sub_sid = f"sub-sess-{uuid.uuid4().hex[:8]}"
        _populate_map_with_script(ws, session_id=sub_sid)
        msgs, ctx = make_subagent_request(
            parent_workspace=ws,
            target_return=target_return,
            session_id=sub_sid,
        )
        cfg = _budget_config()

        result = _run_v2(sub_sid, msgs, ctx, cfg, is_subagent=True)
        assert _detect_return_path(msgs, result) == target_return
        pin_idx = _assert_pin_present(result, ws)
        _assert_position(result, pin_idx)
        pin_content = next(m["content"] for m in result if _SUB_INSTR_TAG in m.get("content", ""))
        _assert_escape(pin_content)

    def test_summary_bearing_trimmed_fit_uses_exact_rendered_candidate(self):
        from proxy.context import _count_msg_chars

        workspace = f"ws-sub-summary-{uuid.uuid4().hex[:8]}"
        session_id = f"sub-summary-{uuid.uuid4().hex[:8]}"
        _populate_map_with_script(workspace, session_id=session_id)

        messages = [
            {
                "role": "system",
                "content": "SYSTEM-4D-" + ("P" * 1990),
            },
            {
                "role": "user",
                "content": "please complete this assignment",
            },
            {
                "role": "assistant",
                "content": "ASSISTANT-4D-" + ("A" * 50000),
            },
        ]
        messages, ctx = make_subagent_request(
            parent_workspace=workspace,
            messages=messages,
            session_id=session_id,
        )
        summaries = [
            {
                "message_index": 5,
                "role": "user",
                "summary": "SUMMARY-4D-" + ("S" * 400),
                "tool_summary": "",
            },
        ]
        cfg = _budget_config(
            safe=10_000,
            tokens_per_char=0.25,
        )

        result = _run_v2(
            session_id,
            messages,
            ctx,
            cfg,
            is_subagent=True,
            summaries=summaries,
        )

        assert _detect_return_path(messages, result) == "trimmed-fits"

        summary_messages = [
            message
            for message in result
            if (
                message.get("role") == "system"
                and "[Conversation Summary]" in message.get("content", "")
            )
        ]
        assert len(summary_messages) == 1
        assert "SUMMARY-4D-" in summary_messages[0]["content"]

        tier2_messages = [
            message
            for message in result
            if _SUB_INSTR_TAG in message.get("content", "")
        ]
        assert len(tier2_messages) == 1

        final_tokens = int(
            sum(_count_msg_chars(message) for message in result)
            * cfg["tokens_per_char"]
        )
        assert final_tokens <= cfg["safe_context_tokens"]


class TestPinEmissionNegative:
    """Negative tests: empty map → no pin; no Tier-2 in inbound → no pin."""

    def test_empty_map_no_pin(self):
        """When the instruction map is empty, NO <CLIENT_SUB pin emits."""
        ws = f"ws-empty-{uuid.uuid4().hex[:8]}"
        # Do NOT populate the map, and use a conversation with no Tier-2
        # tool result so the capture pipeline doesn't populate it either.
        msgs = _build_main_path_conversation(
            tool_content="plain tool result without any tier2 suffix",
        )
        ctx = _make_ctx(workspace_name=ws)
        cfg = _budget_config(safe=30_000)
        result = _run_v2("sess-empty", msgs, ctx, cfg)
        for m in result:
            assert _SUB_INSTR_TAG not in m.get("content", ""), (
                "empty map must NOT produce a Tier-2 pin"
            )

    def test_empty_map_no_sub_in_output(self):
        """Even when the conversation would engage the V2 pipeline, if the
        map is empty, NO <CLIENT_SUB substring appears anywhere in output."""
        ws = f"ws-truly-empty-{uuid.uuid4().hex[:8]}"
        # Deliberately do NOT populate the map
        msgs = _build_main_path_conversation(
            tool_content="plain tool result without tier2",
        )
        ctx = _make_ctx(workspace_name=ws)
        cfg = _budget_config(safe=30_000)
        result = _run_v2("sess-truly-empty", msgs, ctx, cfg)
        for m in result:
            assert "CLIENT_SUB" not in m.get("content", ""), (
                "empty map must NOT produce any CLIENT_SUB pin"
            )


class TestPinEmissionRevertTests:
    """Revert-tests: prove that the pin emission is wired at ALL returns.

    Strategy: monkeypatch _emit_tier2_pin to only fire on the main return
    (detectable by checking call stack depth or a flag). Then assert that
    the sub-agent cases FAIL (proving they depend on their own _emit_tier2_pin
    call, not the main one).

    We implement this by patching _emit_tier2_pin with a wrapper that
    tracks whether it was called from a sub-agent path vs main path.
    """

    @pytest.mark.parametrize("target_return", ["fits", "no-first-user", "trimmed-fits", "fallback"])
    def test_sub_agent_pin_independent_of_main_return(self, target_return):
        """Revert-test: prove sub-agent pin emission does NOT come from the
        main return path.

        Strategy: patch _build_subagent_context to track calls, then verify
        the pin (or the mandatory-overflow raise for fallback) comes from
        within the sub-agent builder, not the main return.

        Slice 4D correction: ``no-first-user``/``trimmed-fits`` paths now
        succeed with Tier-2 reservation.

        Slice 4D restoration (2026-07-18): ``fallback`` previously raised
        but now succeeds via the finalizer content-trim safety net (§13.5).
        """
        ws = f"ws-revert-{target_return}-{uuid.uuid4().hex[:8]}"
        sub_sid = f"sub-revert-{uuid.uuid4().hex[:8]}"
        _populate_map_with_script(ws, session_id=sub_sid)
        msgs, ctx = make_subagent_request(
            parent_workspace=ws,
            target_return=target_return,
            session_id=sub_sid,
        )
        cfg = _budget_config()
        import proxy.context_v2 as cv2_module
        original_build = cv2_module._build_subagent_context
        build_calls = []

        def tracking_build(*args, **kwargs):
            build_calls.append(True)
            return original_build(*args, **kwargs)

        with patch.object(
            cv2_module, "_build_subagent_context", side_effect=tracking_build,
        ):
            result = _run_v2(sub_sid, msgs, ctx, cfg, is_subagent=True)

        # Proof: _build_subagent_context WAS entered for every branch.
        assert build_calls, (
            "_build_subagent_context was NOT entered — sub-agent path not taken"
        )

        # Proof for successful branches: the requested branch produced the
        # result and emitted its own pin.
        assert _detect_return_path(msgs, result) == target_return
        pin_idx = _assert_pin_present(result, ws)
        _assert_position(result, pin_idx)
        pin_content = next(m["content"] for m in result if _SUB_INSTR_TAG in m.get("content", ""))
        _assert_escape(pin_content)


class TestSlice4DFinalBudgetValidation:
    """Slice 4D: ``_finalize_context_result`` is the universal terminal
    boundary. Tier-2 cost is included in the final measurement; overflow
    that would have leaked silently now raises ``ContextLimitError``.

    Each test calculates the arithmetic preconditions so the overflow source
    is behaviorally explicit.
    """

    def test_slice_4d_fast_exit_tier2_pushes_total_over_raises(self):
        """Fast-exit branch: base candidate plus tools fits, but the emitted
        Tier-2 pin pushes the rendered total past ``safe_tokens``. The
        terminal helper raises ``ContextLimitError`` instead of returning
        the overflowing candidate.
        """
        from proxy.context import ContextLimitError

        ws = f"ws-4d-fast-{uuid.uuid4().hex[:8]}"
        _populate_map_with_script(ws)
        msgs = [
            _system("sys"),
            _user("hello"),
            _assistant("hi there"),
        ]
        ctx = _make_ctx(workspace_name=ws)

        # Base + Tier-2 must exceed safe; tools absent so tool_schema_tokens=0.
        # sys(3) + user(5) + asst(8) = 16 chars; pin content is large.
        # With safe=20, tpc=1.0, base=16 tokens, pin content (~700 chars
        # from the populated script) easily exceeds 4-char remainder.
        cfg = _budget_config(safe=20, tokens_per_char=1.0)

        state._chain_state.clear()
        mock_conn = MagicMock()
        mock_conn_cm = MagicMock()
        mock_conn_cm.__enter__ = MagicMock(return_value=mock_conn)
        mock_conn_cm.__exit__ = MagicMock(return_value=False)
        mock_conn.execute.return_value = MagicMock(fetchone=MagicMock(return_value=None))

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
            stack.enter_context(patch("proxy.context_v2._load_state_from_db", return_value=None))
            stack.enter_context(patch("memory.db.connection", return_value=mock_conn_cm))
            with pytest.raises(ContextLimitError) as excinfo:
                _select_context_v2(
                    "sess-4d-fast", copy.deepcopy(msgs),
                    model_name="test-model", ctx=ctx,
                )

        assert ctx.v2_boundary_idx is None, (
            "precondition: fast-exit path must set boundary_idx to None"
        )
        assert "Final context exceeds safe token budget" in str(excinfo.value)

    def test_slice_4d_compute_chain_fail_tier2_pushes_total_over_raises(self):
        """Compute-chain failure branch: forcing ``compute_chain`` to raise
        ``TypeError`` routes through the chain-failure handler. Tier-2 pin
        emission on the fallback candidate pushes the total past
        ``safe_tokens`` and the terminal helper raises.
        """
        from proxy.context import ContextLimitError

        ws = f"ws-4d-chain-{uuid.uuid4().hex[:8]}"
        _populate_map_with_script(ws)
        # Large enough that the chain-failure fallback path is well-defined;
        # small enough that Tier-2 pin (~700 tokens at tpc=1.0) overflows
        # safe=20.
        msgs = [
            _system("sys"),
            _user("hello"),
            _assistant("hi there"),
        ]
        ctx = _make_ctx(workspace_name=ws)
        cfg = _budget_config(safe=20, tokens_per_char=1.0)

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
            stack.enter_context(patch("proxy.context_v2._load_state_from_db", return_value=None))
            stack.enter_context(patch("memory.db.connection", return_value=mock_conn_cm))
            stack.enter_context(patch(
                "proxy.chain_hash.compute_chain",
                side_effect=TypeError("forced for slice 4d"),
            ))
            with pytest.raises(ContextLimitError) as excinfo:
                _select_context_v2(
                    "sess-4d-chain", copy.deepcopy(msgs),
                    model_name="test-model", ctx=ctx,
                )

        assert ctx.v2_boundary_idx is None, (
            "precondition: compute_chain-fail path must set boundary_idx to None"
        )
        assert "Final context exceeds safe token budget" in str(excinfo.value)

    def test_slice_4d_subagent_fit_drops_inherited_vfs(self):
        """Sub-agent fit branch: the cleaned exchange fits safe on its own,
        but the inherited VFS block would overflow. The terminal helper
        drops the inherited VFS and returns the VFS-free candidate.

        Asserts:
        - all original non-system messages survive
        - Tier-2 pin present exactly once
        - inherited VFS absent
        - final rendered total fits ``safe_tokens``
        """
        ws = f"ws-4d-sub-fit-{uuid.uuid4().hex[:8]}"
        sub_sid = f"sub-4d-fit-{uuid.uuid4().hex[:8]}"
        _populate_map_with_script(ws, session_id=sub_sid)

        # Small cleaned exchange that fits comfortably.
        msgs = [
            {"role": "user", "content": "do the sub-agent task"},
            {"role": "assistant", "content": "working on it"},
        ]
        ctx = SimpleNamespace(
            request_id="slice-4d-sub-fit",
            v2_boundary_idx=None,
            v2_boundary_chain=None,
            is_subagent=True,
            session_id=sub_sid,
            workspace_name=ws,
            parent_agent_id=None,
            inherited_skill=None,
        )

        # Register parent + sub-agent so workspace inheritance resolves.
        from proxy.agent.agent import Agent
        from proxy.agent.agent_config import AgentConfig
        from proxy.agent.agent_registry import registry
        parent_sid = f"parent-4d-fit-{uuid.uuid4().hex[:8]}"
        parent_agent = Agent(
            session_id=parent_sid, parent=None, config=AgentConfig(),
        )
        with registry._lock:
            registry._register(parent_agent)
            state.workspace_store.activate(parent_sid, ws)
            sub_agent = Agent(
                session_id=sub_sid, parent=parent_agent,
                config=AgentConfig(), is_subagent=True,
            )
            registry._register(sub_agent)
            state.workspace_store.activate(sub_sid, ws)

        # Inherited VFS block sized so candidate+VFS overflows safe but
        # candidate-only fits. safe=200, tpc=1.0.
        # Cleaned = 23 + 13 = 36 chars; pin ~700 chars; total ~736 chars.
        # VFS block 1000 chars → candidate+VFS ~1736 > 200? Wait that's huge.
        # Use safe=1500, tpc=1.0: cleaned+pin ~736 ≤ 1500; +VFS(1000)=1736 > 1500.
        vfs_inherited_block = {
            "role": "system",
            "content": "[VFS INHERITED CONTEXT]\n" + ("Z" * 1000) + "\n[END VFS]",
        }
        cfg = _budget_config(safe=1500, tokens_per_char=1.0)

        state._chain_state.clear()
        mock_conn = MagicMock()
        mock_conn_cm = MagicMock()
        mock_conn_cm.__enter__ = MagicMock(return_value=mock_conn)
        mock_conn_cm.__exit__ = MagicMock(return_value=False)

        with ExitStack() as stack:
            stack.enter_context(patch("proxy.context.get_context_config", return_value=cfg))
            stack.enter_context(patch("memory.db.get_ancient_history", return_value=""))
            stack.enter_context(patch("memory.db.get_rich_summaries", return_value=[]))
            stack.enter_context(patch(
                "proxy.vfs._build_vfs_inherited_context",
                return_value=vfs_inherited_block,
            ))
            stack.enter_context(patch("proxy.vfs._build_vfs_context", return_value=None))
            stack.enter_context(patch(
                "memory.db_summaries.get_last_summarized_exchange_index",
                return_value=-1,
            ))
            stack.enter_context(patch("memory.sync.submit_ancient_compress", return_value=None))
            stack.enter_context(patch("memory.db.connection", return_value=mock_conn_cm))
            result = _select_context_v2(
                sub_sid, copy.deepcopy(msgs),
                model_name="test-model", ctx=ctx, is_subagent=True,
            )

        # All original non-system messages survive.
        result_non_system = [m for m in result if m.get("role") != "system"]
        original_non_system = [m for m in msgs if m.get("role") != "system"]
        assert len(result_non_system) == len(original_non_system), (
            f"All non-system messages must survive; got "
            f"{[m.get('content')[:30] for m in result_non_system]}"
        )
        # Tier-2 pin present exactly once.
        pin_count = sum(
            1 for m in result if _SUB_INSTR_TAG in m.get("content", "")
        )
        assert pin_count == 1, f"expected exactly 1 Tier-2 pin, got {pin_count}"
        # Inherited VFS absent.
        assert not any("[VFS INHERITED" in m.get("content", "") for m in result), (
            "Inherited VFS must be dropped to fit safe_tokens"
        )
        # Final rendered total fits safe.
        from proxy.context import _count_msg_chars
        rendered_tokens = sum(
            int(_count_msg_chars(m) * 1.0) for m in result
        )
        assert rendered_tokens <= 1500, (
            f"Final rendered total {rendered_tokens} must fit safe=1500"
        )

    def test_slice_4d_subagent_fallback_trims_when_task_overflows(self):
        """Sub-agent fallback branch: when even the first user message plus
        required system/tools/Tier-2 cannot fit ``safe_tokens``, the
        terminal helper previously raised ``ContextLimitError`` (Slice 4D
        contract).

        Slice 4D restoration (2026-07-18): with the finalizer content-trim
        safety net (§13.5), the finalizer content-trims the oversized
        first-user message with a ``... [truncated]`` marker and succeeds
        rather than raising. The test asserts the trimmed-fits outcome.
        """
        ws = f"ws-4d-sub-fb-{uuid.uuid4().hex[:8]}"
        sub_sid = f"sub-4d-fb-{uuid.uuid4().hex[:8]}"
        _populate_map_with_script(ws, session_id=sub_sid)

        # Huge first-user content alone overflows safe.
        huge = "C" * 100_000
        msgs = [
            {"role": "user", "content": huge},
            {"role": "assistant", "content": "tail"},
        ]
        ctx = SimpleNamespace(
            request_id="slice-4d-sub-fb",
            v2_boundary_idx=None,
            v2_boundary_chain=None,
            is_subagent=True,
            session_id=sub_sid,
            workspace_name=ws,
            parent_agent_id=None,
            inherited_skill=None,
        )

        from proxy.agent.agent import Agent
        from proxy.agent.agent_config import AgentConfig
        from proxy.agent.agent_registry import registry
        parent_sid = f"parent-4d-fb-{uuid.uuid4().hex[:8]}"
        parent_agent = Agent(
            session_id=parent_sid, parent=None, config=AgentConfig(),
        )
        with registry._lock:
            registry._register(parent_agent)
            state.workspace_store.activate(parent_sid, ws)
            sub_agent = Agent(
                session_id=sub_sid, parent=parent_agent,
                config=AgentConfig(), is_subagent=True,
            )
            registry._register(sub_agent)
            state.workspace_store.activate(sub_sid, ws)

        cfg = _budget_config(safe=10_000, tokens_per_char=0.25)

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
            stack.enter_context(patch("memory.db.connection", return_value=mock_conn_cm))
            result = _select_context_v2(
                sub_sid, copy.deepcopy(msgs),
                model_name="test-model", ctx=ctx, is_subagent=True,
            )

        # Slice 4D restoration: the finalizer content-trimmed the user
        # message; result is a list with at least one trimmed message.
        assert isinstance(result, list)
        truncated = [
            m for m in result
            if "... [truncated]" in str(m.get("content", ""))
        ]
        assert truncated, (
            "Slice 4D restoration: expected at least one message with "
            "'... [truncated]' marker after finalizer content-trim"
        )
        # The trimmed user content must be shorter than the original huge.
        user_msgs = [m for m in result if m.get("role") == "user"]
        assert user_msgs, "expected at least one user message in result"
        trimmed_user = user_msgs[-1]
        assert len(trimmed_user.get("content", "")) < len(huge), (
            f"trimmed user content ({len(trimmed_user.get('content', ''))}) "
            f"must be shorter than original ({len(huge)})"
        )
