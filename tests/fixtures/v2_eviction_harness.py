"""Multi-turn V2-eviction harness (spec §9.10 #3).

Drives _select_context_v2 once per turn with REAL CAS (do NOT mock
state._chain_state). Each turn_msg_lists[i] is the FULL inbound messages
for turn i (opencode re-sends full history each turn). Returns a list of
per-turn dicts with diagnostics so tests can prove eviction actually fired.

Uses a CAPPED cfg (small safe_context_tokens) so eviction engages within
a few turns. Mock only the I/O deps (db summaries, vfs, _load_state_from_db)
like test_tier2_capture does; keep the chain-state + eviction REAL.
"""
import copy
import uuid
from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import proxy.state as state
from proxy.context_v2 import _select_context_v2


def _default_budget_config(safe=8000, tokens_per_char=0.25):
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


def run_v2_turns(session_id, turn_msg_lists, cfg=None, ctx_factory=None,
                 model_name="test-model"):
    """Drive _select_context_v2 once per turn with REAL CAS.

    Parameters
    ----------
    session_id : str
        Stable session ID across turns (opencode re-sends full history).
    turn_msg_lists : list[list[dict]]
        turn_msg_lists[i] is the FULL inbound messages for turn i.
    cfg : dict | None
        Model config dict. Defaults to a capped cfg (safe=8000) so eviction
        engages within a few turns.
    ctx_factory : callable | None
        Called as ctx_factory(turn_index) -> SimpleNamespace. Defaults to
        _make_ctx with a unique request_id per turn.
    model_name : str
        Model name for chain-state keying.

    Returns
    -------
    list[dict]
        Per-turn result dicts with keys:
        - "outbound": the final message list from _select_context_v2
        - "boundary_idx": ctx.v2_boundary_idx after the call
        - "captured_map": snapshot of instruction_files_by_workspace for
          the workspace (empty dict if no workspace)
        - "turn": turn index (0-based)
        - "outbound_tool_call_ids": set of tool_call_id values in outbound
        - "outbound_size_chars": total char count of outbound messages
    """
    if cfg is None:
        cfg = _default_budget_config()
    if ctx_factory is None:
        ctx_factory = lambda turn: _make_ctx(
            request_id=f"req-{turn}-{uuid.uuid4().hex[:6]}",
            workspace_name="",
        )

    return _run_v2_turns_impl(session_id, turn_msg_lists, cfg, ctx_factory, model_name)


def _run_v2_turns_impl(session_id, turn_msg_lists, cfg, ctx_factory, model_name):
    if ctx_factory is None:
        ctx_factory = lambda turn: _make_ctx(
            request_id=f"req-{turn}-{uuid.uuid4().hex[:6]}",
            workspace_name="",
        )

    state._chain_state.clear()

    results = []
    workspace_key = None

    for turn_idx, messages in enumerate(turn_msg_lists):
        ctx = ctx_factory(turn_idx)

        turn_cfg = cfg[turn_idx] if isinstance(cfg, list) else cfg

        mock_conn = MagicMock()
        mock_conn_cm = MagicMock()
        mock_conn_cm.__enter__ = MagicMock(return_value=mock_conn)
        mock_conn_cm.__exit__ = MagicMock(return_value=False)

        with ExitStack() as stack:
            stack.enter_context(patch("proxy.context.get_context_config", return_value=turn_cfg))
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
                session_id, copy.deepcopy(messages),
                model_name=model_name, ctx=ctx,
            )

        workspace_name = getattr(ctx, "workspace_name", None)
        if workspace_name and workspace_key is None:
            workspace_key = workspace_name

        captured = {}
        if workspace_name:
            captured = state.get_instruction_files(workspace_name)

        tool_ids = set()
        total_chars = 0
        for m in outbound:
            tc_id = m.get("tool_call_id")
            if tc_id:
                tool_ids.add(tc_id)
            content = m.get("content", "")
            if isinstance(content, str):
                total_chars += len(content)

        results.append({
            "outbound": outbound,
            "boundary_idx": ctx.v2_boundary_idx,
            "captured_map": dict(captured),
            "turn": turn_idx,
            "outbound_tool_call_ids": tool_ids,
            "outbound_size_chars": total_chars,
        })

    return results
