"""# US-028: Orphan Tool Result Sanitization
US-024: Tool Loop Orphan Cleanup — pure function unit tests."""
import copy

from memory.tool_loop import _cleanup_orphaned_tool_results


def _assistant_with_calls(*call_ids):
    return {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": cid,
                "type": "function",
                "function": {"name": "recall", "arguments": "{}"},
            }
            for cid in call_ids
        ],
    }


def _tool_result(call_id, content='{"results":[]}'):
    return {"role": "tool", "tool_call_id": call_id, "content": content}


class TestCleanupOrphanedToolResults:
    def test_drop_orphan_tool_result(self):
        messages = [
            {"role": "user", "content": "hi"},
            _assistant_with_calls("call_real_1"),
            _tool_result("call_real_1"),
            _tool_result("call_orphan_999"),
        ]
        result = _cleanup_orphaned_tool_results(messages)

        tool_ids = [m["tool_call_id"] for m in result if m.get("role") == "tool"]
        assert "call_real_1" in tool_ids
        assert "call_orphan_999" not in tool_ids
        assert len([m for m in result if m.get("role") == "tool"]) == 1

    def test_preserve_valid_tool_result(self):
        messages = [
            {"role": "user", "content": "hi"},
            _assistant_with_calls("call_keep_1"),
            _tool_result("call_keep_1", '{"results":[{"x":1}]}'),
        ]
        result = _cleanup_orphaned_tool_results(messages)

        tool_msgs = [m for m in result if m.get("role") == "tool"]
        assert len(tool_msgs) == 1
        assert tool_msgs[0]["tool_call_id"] == "call_keep_1"
        assert tool_msgs[0]["content"] == '{"results":[{"x":1}]}'

    def test_prune_partial_tool_calls(self):
        messages = [
            _assistant_with_calls("call_A", "call_B"),
            _tool_result("call_A"),
        ]
        result = _cleanup_orphaned_tool_results(messages)

        assistant_msg = next(m for m in result if m.get("role") == "assistant")
        surviving_ids = [tc["id"] for tc in assistant_msg["tool_calls"]]
        assert surviving_ids == ["call_A"]
        assert "call_B" not in surviving_ids
        assert "tool_calls" in assistant_msg

    def test_remove_tool_calls_entirely(self):
        messages = [
            _assistant_with_calls("call_solo"),
        ]
        result = _cleanup_orphaned_tool_results(messages)

        assistant_msg = next(m for m in result if m.get("role") == "assistant")
        assert "tool_calls" not in assistant_msg
        assert assistant_msg["role"] == "assistant"
        assert assistant_msg["content"] is None

    def test_pass_through_no_changes(self):
        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Question?"},
            {"role": "assistant", "content": "Answer."},
        ]
        original_copy = copy.deepcopy(messages)
        result = _cleanup_orphaned_tool_results(messages)

        assert result == original_copy
        assert result is not messages
        result.append({"role": "user", "content": "mutated"})
        assert len(messages) == 3

    def test_mixed_orphans_and_valids(self):
        messages = [
            {"role": "system", "content": "sys"},
            _assistant_with_calls("call_keep_1"),
            _tool_result("call_keep_1"),
            _tool_result("call_orphan_x"),
            {"role": "user", "content": "follow up"},
            _assistant_with_calls("call_keep_2", "call_drop_2"),
            _tool_result("call_keep_2"),
        ]
        result = _cleanup_orphaned_tool_results(messages)

        tool_ids = {m["tool_call_id"] for m in result if m.get("role") == "tool"}
        assert tool_ids == {"call_keep_1", "call_keep_2"}
        assert "call_orphan_x" not in tool_ids

        assistants = [m for m in result if m.get("role") == "assistant"]
        first_ids = [tc["id"] for tc in assistants[0]["tool_calls"]]
        assert first_ids == ["call_keep_1"]
        second_ids = [tc["id"] for tc in assistants[1]["tool_calls"]]
        assert second_ids == ["call_keep_2"]
        assert "call_drop_2" not in second_ids

    def test_empty_input(self):
        assert _cleanup_orphaned_tool_results([]) == []

    def test_no_tool_messages(self):
        messages = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
        ]
        original = copy.deepcopy(messages)
        result = _cleanup_orphaned_tool_results(messages)

        assert result == original
        assert len(result) == 3

    def test_no_tool_calls_in_assistants(self):
        messages = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "plain reply, no tool_calls"},
            _tool_result("call_unmatched_1"),
            _tool_result("call_unmatched_2"),
        ]
        result = _cleanup_orphaned_tool_results(messages)

        tool_msgs = [m for m in result if m.get("role") == "tool"]
        assert tool_msgs == []
        assert len(result) == 2
