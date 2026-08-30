"""Spec §9.5 — Tier-2 strip: byte-integrity, non-mutation, close-tag preserved."""
import copy
from types import SimpleNamespace

import pytest

import proxy.state as state
from proxy.state import record_instruction_file
from tests.fixtures.tier2_payload import (
    FILE_BODY,
    RAW_TOOL_CONTENT,
    RAW_TOOL_BODY_WITH_LITERAL_CLOSE,
    TIER2_SUFFIX,
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


def _ctx():
    return SimpleNamespace(workspace_name="ws-strip")


def _populate_map():
    with state.vfs_lock:
        record_instruction_file(
            "ws-strip",
            "/repo/src/proxy/sub/AGENTS.md",
            "sub-agents body",
        )


def test_byte_integrity_stripped_content_equals_file_body():
    _populate_map()
    from proxy.context_v2 import _emit_tier2_pin

    tool_msg = {"role": "tool", "content": RAW_TOOL_CONTENT, "tool_call_id": "tc-1"}
    result = _emit_tier2_pin([tool_msg], "sess-strip", ctx=_ctx())
    tool_out = [m for m in result if m.get("role") == "tool"]
    assert len(tool_out) == 1
    assert tool_out[0]["content"] == FILE_BODY


def test_stripped_content_endswith_close_tag():
    _populate_map()
    from proxy.context_v2 import _emit_tier2_pin

    tool_msg = {"role": "tool", "content": RAW_TOOL_CONTENT, "tool_call_id": "tc-1"}
    result = _emit_tier2_pin([tool_msg], "sess-strip", ctx=_ctx())
    tool_out = [m for m in result if m.get("role") == "tool"]
    assert tool_out[0]["content"].endswith("</content>")


def test_suffix_removed_from_output():
    _populate_map()
    from proxy.context_v2 import _emit_tier2_pin

    tool_msg = {"role": "tool", "content": RAW_TOOL_CONTENT, "tool_call_id": "tc-1"}
    result = _emit_tier2_pin([tool_msg], "sess-strip", ctx=_ctx())
    tool_out = [m for m in result if m.get("role") == "tool"]
    assert "<system-reminder>" not in tool_out[0]["content"]
    assert TIER2_SUFFIX not in tool_out[0]["content"]


def test_non_mutation_input_list_unchanged():
    _populate_map()
    from proxy.context_v2 import _emit_tier2_pin

    tool_msg = {"role": "tool", "content": RAW_TOOL_CONTENT, "tool_call_id": "tc-1"}
    input_list = [tool_msg]
    input_before = copy.deepcopy(input_list)
    _emit_tier2_pin(input_list, "sess-strip", ctx=_ctx())
    assert input_list == input_before
    assert input_list[0] is tool_msg


def test_non_mutation_input_dict_identity():
    _populate_map()
    from proxy.context_v2 import _emit_tier2_pin

    tool_msg = {"role": "tool", "content": RAW_TOOL_CONTENT, "tool_call_id": "tc-1"}
    original_id = id(tool_msg)
    result = _emit_tier2_pin([tool_msg], "sess-strip", ctx=_ctx())
    tool_out = [m for m in result if m.get("role") == "tool"]
    assert id(tool_out[0]) != original_id
    assert tool_msg["content"] == RAW_TOOL_CONTENT


def test_last_user_is_tool_role_gets_stripped():
    _populate_map()
    from proxy.context_v2 import _emit_tier2_pin

    msgs = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "query"},
        {"role": "tool", "content": RAW_TOOL_CONTENT, "tool_call_id": "tc-2"},
    ]
    result = _emit_tier2_pin(msgs, "sess-strip", ctx=_ctx())
    assert result[-1]["content"] == FILE_BODY


def test_no_strip_when_map_empty():
    from proxy.context_v2 import _emit_tier2_pin

    tool_msg = {"role": "tool", "content": RAW_TOOL_CONTENT, "tool_call_id": "tc-3"}
    result = _emit_tier2_pin([tool_msg], "sess-strip")
    assert len(result) == 1
    assert result[0]["content"] == RAW_TOOL_CONTENT


def test_non_tool_messages_untouched():
    _populate_map()
    from proxy.context_v2 import _emit_tier2_pin

    sys_msg = {"role": "system", "content": "system content"}
    user_msg = {"role": "user", "content": "user query"}
    asst_msg = {"role": "assistant", "content": "assistant reply"}
    result = _emit_tier2_pin([sys_msg, user_msg, asst_msg], "sess-strip", ctx=_ctx())
    non_pin = [m for m in result if m.get("role") != "system" or "CLIENT_SUB" not in m.get("content", "")]
    assert non_pin[0] is sys_msg
    assert non_pin[1] is user_msg
    assert non_pin[2] is asst_msg


def test_literal_close_not_followed_by_system_reminder_not_stripped():
    _populate_map()
    from proxy.context_v2 import _emit_tier2_pin

    tool_msg = {
        "role": "tool",
        "content": RAW_TOOL_BODY_WITH_LITERAL_CLOSE,
        "tool_call_id": "tc-4",
    }
    result = _emit_tier2_pin([tool_msg], "sess-strip", ctx=_ctx())
    tool_out = [m for m in result if m.get("role") == "tool"]
    assert tool_out[0]["content"] == RAW_TOOL_BODY_WITH_LITERAL_CLOSE


def test_pin_present_after_strip():
    _populate_map()
    from proxy.context_v2 import _emit_tier2_pin

    tool_msg = {"role": "tool", "content": RAW_TOOL_CONTENT, "tool_call_id": "tc-5"}
    result = _emit_tier2_pin([tool_msg], "sess-strip", ctx=_ctx())
    sys_msgs = [m for m in result if m.get("role") == "system"]
    sub_pins = [m for m in sys_msgs if "CLIENT_SUB_INSTRUCTIONS" in m.get("content", "")]
    assert len(sub_pins) == 1
