"""Verify the V2 barrier insertion handles mixed-case backend_model names.

Bug: production has backend_model='MiniMax-M3' (mixed case from
src/proxy_models.json:48). _get_adapter's can_handle uses 'minimax' in backend
which does not match 'MiniMax-M3'. _get_adapter returned None → no barrier
inserted. MiniMaxAdapter's apply() then concatenated summaries + ancient +
last_user into one 92K-char user message → entire [PAST CONTEXT] block landed
in no_cache per turn.

Fix: lowercase at the call site (Change 1) AND inside _get_adapter
(Change 2, defense-in-depth).
"""
from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from proxy.context_v2 import _select_context_v2
from proxy.pipeline import provider_compat
from proxy.pipeline.provider_compat import (
    DeepSeekAdapter,
    MiniMaxAdapter,
    _get_adapter,
)


pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _user(content: str) -> dict:
    return {"role": "user", "content": content}


def _assistant(content: str) -> dict:
    return {"role": "assistant", "content": content}


def _assistant_tool_calls(call_ids, name="delegate_task", content=""):
    return {
        "role": "assistant",
        "content": content,
        "tool_calls": [
            {
                "id": cid,
                "type": "function",
                "function": {"name": name, "arguments": "{}"},
            }
            for cid in call_ids
        ],
    }


def _tool_result(call_id, content=""):
    return {"role": "tool", "tool_call_id": call_id, "content": content}


def _no_db_mocks(cfg):
    return [
        patch("proxy.context.get_context_config", return_value=cfg),
        patch("memory.db.get_ancient_history", return_value=""),
        patch("memory.db.get_rich_summaries", return_value=[]),
        patch("proxy.vfs._build_vfs_inherited_context", return_value=None),
        patch("proxy.vfs._build_vfs_context", return_value=None),
    ]


def _budget_config():
    return {
        "max_context_tokens": 40_000,
        "tokens_per_char": 0.25,
        "safe_context_tokens": 10_000,
        "raw_exchange_count": 100,
    }


def _large_raw(count=150, chars=400):
    pad = "X" * chars
    return [_user(f"msg {i} {pad}") for i in range(count)]


def _has_barrier(messages: list) -> bool:
    return any(
        m.get("role") == "assistant" and m.get("content") == " "
        for m in messages
    )


def _vfs_ctx(session_id):
    """Explicit request-context double for VFS-specific _select_context_v2 calls.

    Wave 2 removed the production SimpleNamespace fallback that fabricated a
    fake context when none was supplied. These tests need VFS to fire, so they
    must pass a real-looking context double.
    """
    return SimpleNamespace(
        request_id=f"vfs-{session_id}",
        session_id=session_id,
        client_hint="",
        workspace_name="default",
        is_subagent=False,
        v2_boundary_idx=None,
        v2_boundary_chain=None,
    )


# ---------------------------------------------------------------------------
# Defense-in-depth: _get_adapter lowercases input
# ---------------------------------------------------------------------------

class TestGetAdapterLowercasesInput:
    """Verify _get_adapter normalizes input to lowercase before matching.

    This is defense-in-depth: even if a future caller forgets to lowercase
    the backend_model, _get_adapter still returns the correct adapter.
    """

    def test_lowercase_input_returns_adapter(self):
        assert _get_adapter("minimax-m3") is MiniMaxAdapter

    def test_mixed_case_input_returns_adapter(self):
        assert _get_adapter("MiniMax-M3") is MiniMaxAdapter

    def test_uppercase_input_returns_adapter(self):
        assert _get_adapter("MINIMAX-M3") is MiniMaxAdapter

    def test_chaotic_case_input_returns_adapter(self):
        assert _get_adapter("mInImAx-M3") is MiniMaxAdapter

    def test_empty_string_returns_none(self):
        assert _get_adapter("") is None

    def test_none_input_returns_none(self):
        assert _get_adapter(None) is None

    def test_unknown_backend_returns_none(self):
        assert _get_adapter("openai-gpt-4") is None
        assert _get_adapter("anthropic-claude") is None

    def test_other_adapters_lowercased_too(self):
        assert _get_adapter("DeepSeek-V3") is DeepSeekAdapter
        assert _get_adapter("DEEPSEEK-V3") is DeepSeekAdapter

    def test_returns_registry_adapter_not_unregistered(self):
        from proxy.pipeline.provider_compat import _REGISTRY
        for adapter_cls in _REGISTRY:
            if adapter_cls is MiniMaxAdapter:
                continue
            backend_marker = adapter_cls.__name__.replace("Adapter", "").lower()
            assert _get_adapter(backend_marker.upper()) is adapter_cls
            assert _get_adapter(backend_marker.capitalize()) is adapter_cls


# ---------------------------------------------------------------------------
# Provider-bound adjacency handling moved out of V2 (Wave 4)
# ---------------------------------------------------------------------------
#
# Wave 4 removes the context-layer barrier emission from `_select_context_v2`.
# The MiniMax user-merge problem is now handled by MiniMaxAdapter.apply
# downstream of V2 — see `tests/unit/test_context_v2.py::TestAssemblyOrder::
# test_minimax_adapter_handles_vfs_user_adjacency` for the provider-bound test.
#
# The classes `TestV2BarrierCaseInsensitivity`, `TestV2BarrierPreservesToolPair`,
# and `TestV2BarrierOnlyWhenMergeNeeded` previously pinned V2-internal barrier
# emission; they are intentionally absent here because Wave 4 deletes that
# code path. Tests that asserted `_has_barrier(result)` or inspected
# `last_user` directly were tied to a structural pattern V2 no longer uses.


# ---------------------------------------------------------------------------
# Defense-in-depth: _cleanup_tool_pair_integrity strips wedged barriers.
# ---------------------------------------------------------------------------

class TestCleanupStripsWedgedBarrier:
    """Belt-and-suspenders: _cleanup_tool_pair_integrity must drop any
    barrier assistant message wedged between an assistant tool_call
    and its matching role=tool result. Defends against future barrier sites.
    """

    def test_drops_wedged_barrier_between_tool_call_and_result(self):
        from proxy.context_v2 import _cleanup_tool_pair_integrity

        messages = [
            _user("hi"),
            _assistant_tool_calls(["call_real_1"]),
            _assistant(" "),
            _tool_result("call_real_1", '{"ok":true}'),
        ]

        cleaned = _cleanup_tool_pair_integrity(messages)

        roles = [m.get("role") for m in cleaned]
        assert " " not in [
            m.get("content") for m in cleaned if m.get("role") == "assistant"
        ], (
            f"Wedged barrier must be dropped. Got cleaned messages: {cleaned}"
        )

        tool_call_idx = next(
            i for i, m in enumerate(cleaned)
            if m.get("role") == "assistant" and isinstance(m.get("tool_calls"), list)
        )
        assert cleaned[tool_call_idx + 1].get("role") == "tool", (
            f"Tool result must follow tool_call immediately. Got: {cleaned[tool_call_idx + 1]}"
        )
        assert cleaned[tool_call_idx + 1].get("tool_call_id") == "call_real_1"

    def test_preserves_barrier_not_wedged_between_tool_pair(self):
        from proxy.context_v2 import _cleanup_tool_pair_integrity

        messages = [
            _user("hi"),
            _assistant_tool_calls(["call_x"]),
            _tool_result("call_x", "ok"),
            _assistant(" "),
            _user("next question"),
        ]

        cleaned = _cleanup_tool_pair_integrity(messages)

        assert _has_barrier(cleaned), (
            "Barrier NOT wedged between tool_pair must be preserved. "
            f"Got roles: {[m.get('role') for m in cleaned]}"
        )

    def test_drops_wedged_barrier_only_when_tool_call_id_matches(self):
        from proxy.context_v2 import _cleanup_tool_pair_integrity

        messages = [
            _user("hi"),
            _assistant_tool_calls(["call_real_1"]),
            _assistant(" "),
            _tool_result("call_other_999", "unrelated"),
        ]

        cleaned = _cleanup_tool_pair_integrity(messages)

        barrier_msgs = [
            m for m in cleaned
            if m.get("role") == "assistant" and m.get("content") == " "
        ]
        assert len(barrier_msgs) == 1, (
            f"Barrier must be preserved when tool_call_id does not match the "
            f"preceding assistant's tool_calls. Got cleaned: {cleaned}"
        )


# ---------------------------------------------------------------------------
# Wave 4 canonical placement: VFS inserted AFTER completed_history and
# BEFORE live_exchange. This guarantees the assistant(tool_calls) ->
# role=tool transition is never wedged by a system message and that the
# conversation chain stays contiguous inside each section.
# ---------------------------------------------------------------------------

class TestV2VfsDoesNotWedgeToolPair:
    """VFS must be inserted BETWEEN completed_history and live_exchange
    in the canonical placement. This keeps the assistant(tool_calls) →
    role=tool transition contiguous inside the completed_history section.
    """

    def test_vfs_does_not_wedge_tool_pair(self):
        """Strengthened: two assistant tool calls + two contiguous tool
        results must survive with their atomic assistant-tool adjacency,
        with VFS appearing BEFORE that pair (it's part of completed_history)
        and never between the assistant and its tool results.
        """
        cfg = _budget_config()
        messages = (
            _large_raw(count=150)
            + [
                _assistant_tool_calls(["call_A", "call_B"]),
                _tool_result("call_A", "result A"),
                _tool_result("call_B", "result B"),
                _user("now do something else"),
            ]
        )
        vfs_context = {"role": "system", "content": "[VFS] workspace: ..."}

        model_cfg = SimpleNamespace(backend_model="MiniMax-M3")

        with ExitStack() as stack:
            for p in _no_db_mocks(cfg):
                stack.enter_context(p)
            stack.enter_context(patch(
                "proxy.vfs._build_vfs_context",
                return_value=vfs_context,
            ))
            stack.enter_context(patch(
                "proxy.vfs._build_vfs_inherited_context",
                return_value=None,
            ))
            result = _select_context_v2(
                "sess-vfs-wedge", messages, model_name="test", model_cfg=model_cfg,
                ctx=_vfs_ctx("sess-vfs-wedge"),
            )

        vfs_count = sum(
            1 for m in result
            if m.get("role") == "system" and "[VFS]" in (m.get("content") or "")
        )
        assert vfs_count == 1, (
            f"Exactly one mocked VFS block expected, got {vfs_count}. "
            f"VFS did not fire — this test would otherwise pass trivially "
            f"on a no-VFS assembly."
        )

        asst_idx = next(
            i for i, m in enumerate(result)
            if m.get("role") == "assistant"
            and isinstance(m.get("tool_calls"), list)
            and len(m["tool_calls"]) == 2
        )
        tool_a_idx = next(
            i for i, m in enumerate(result[asst_idx + 1:], asst_idx + 1)
            if m.get("role") == "tool" and m.get("tool_call_id") == "call_A"
        )
        tool_b_idx = next(
            i for i, m in enumerate(result[asst_idx + 1:], asst_idx + 1)
            if m.get("role") == "tool" and m.get("tool_call_id") == "call_B"
        )

        assert tool_a_idx == asst_idx + 1
        assert tool_b_idx == asst_idx + 2

        current_user_idx = next(
            i for i, m in enumerate(result)
            if m.get("role") == "user" and m.get("content") == "now do something else"
        )
        vfs_idx = next(
            i for i, m in enumerate(result)
            if m.get("role") == "system" and "[VFS]" in (m.get("content") or "")
        )
        assert asst_idx < vfs_idx < current_user_idx, (
            f"VFS must appear AFTER completed_history (whose tail is the "
            f"assistant+tool_calls pair) and BEFORE live_exchange "
            f"(which starts at the live user); "
            f"asst@{asst_idx} vfs@{vfs_idx} current@{current_user_idx}"
        )

    def test_vfs_follows_completed_history_and_precedes_live_exchange(self):
        """Wave 4 canonical order: VFS appears AFTER completed_history and
        BEFORE live_exchange (the user-led live suffix).
        """
        cfg = _budget_config()
        # small enough that the early "msg 0" user survives the trim
        messages = (
            _large_raw(count=50)
            + [
                _assistant_tool_calls(["call_xyz"]),
                _tool_result("call_xyz", "<result>ok</result>"),
                _user("next"),
            ]
        )
        vfs_context = {"role": "system", "content": "[VFS] workspace: ..."}

        model_cfg = SimpleNamespace(backend_model="MiniMax-M3")

        with ExitStack() as stack:
            for p in _no_db_mocks(cfg):
                stack.enter_context(p)
            stack.enter_context(patch(
                "proxy.vfs._build_vfs_context",
                return_value=vfs_context,
            ))
            stack.enter_context(patch(
                "proxy.vfs._build_vfs_inherited_context",
                return_value=None,
            ))
            result = _select_context_v2(
                "sess-vfs-front", messages, model_name="test", model_cfg=model_cfg,
                ctx=_vfs_ctx("sess-vfs-front"),
            )

        vfs_positions = [
            i for i, m in enumerate(result)
            if m.get("role") == "system" and "[VFS]" in (m.get("content") or "")
        ]
        assert len(vfs_positions) == 1, (
            f"Exactly one VFS system message expected, got positions={vfs_positions}"
        )

        old_user_idx = next(
            (i for i, m in enumerate(result)
             if m.get("role") == "user" and "msg 0 " in m.get("content", "")),
            None,
        )
        old_asst_idx = next(
            (i for i, m in enumerate(result)
             if m.get("role") == "assistant"
             and isinstance(m.get("tool_calls"), list)),
            None,
        )
        current_user_idx = next(
            (i for i, m in enumerate(result)
             if m.get("role") == "user" and m.get("content") == "next"),
            None,
        )

        assert old_user_idx is not None
        assert old_asst_idx is not None
        assert current_user_idx is not None
        vfs_idx = vfs_positions[0]

        assert old_user_idx < old_asst_idx < vfs_idx < current_user_idx, (
            f"Canonical order required: old_user@{old_user_idx} < "
            f"old_asst@{old_asst_idx} < vfs@{vfs_idx} < current_user@{current_user_idx}"
        )

