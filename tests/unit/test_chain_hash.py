"""Unit tests for src/proxy/chain_hash.py — V2 conversation identity hash chain.

Covers §11 "Chain hash unit tests" of docs/v2-chain-hash-redesign.md:
canonicalization, message hashing, cumulative chain computation, fork
detection, multimodal consistency with memory.db._normalize_content, and
performance.
"""
import hashlib
import json
import time

import pytest

pytestmark = pytest.mark.unit

from memory.db import _normalize_content
from proxy.chain_hash import (
    _canonical,
    _message_hash,
    compute_chain,
    find_fork_point,
)


def _msg(role="user", content="hi", tool_calls=None, tool_call_id=None):
    m = {"role": role, "content": content}
    if tool_calls is not None:
        m["tool_calls"] = tool_calls
    if tool_call_id is not None:
        m["tool_call_id"] = tool_call_id
    return m


class TestCanonical:
    def test_string_content_deterministic(self):
        a = _canonical({"role": "user", "content": "hi"})
        b = _canonical({"role": "user", "content": "hi"})
        assert a == b
        assert a.startswith("user\x00hi\x00")

    def test_list_content_text_part_joined(self):
        a = _canonical({"role": "user", "content": [{"type": "text", "text": "hi"}]})
        b = _canonical({"role": "user", "content": "hi"})
        assert a == b

    def test_list_content_multiple_parts_joined_with_newline(self):
        a = _canonical({
            "role": "user",
            "content": [
                {"type": "text", "text": "line1"},
                {"type": "text", "text": "line2"},
            ],
        })
        b = _canonical({"role": "user", "content": "line1\nline2"})
        assert a == b

    def test_list_content_non_text_part_dropped(self):
        # Non-text dict parts (e.g. image parts) are dropped to match
        # memory.db._normalize_content (db.py:869-888), which is applied at
        # store time. This keeps DB-stored and runtime chain_hash byte-identical.
        with_image = _canonical({
            "role": "user",
            "content": [
                {"type": "text", "text": "a"},
                {"type": "image_url", "image_url": {"url": "x"}},
            ],
        })
        text_only = _canonical({"role": "user", "content": "a"})
        assert with_image == text_only

    def test_empty_string_content(self):
        a = _canonical({"role": "user", "content": ""})
        assert a.startswith("user\x00\x00")

    def test_none_content_normalizes_to_empty(self):
        assert _canonical({"role": "user", "content": None}) == _canonical({"role": "user", "content": ""})

    def test_missing_content_key_normalizes_to_empty(self):
        assert _canonical({"role": "user"}) == _canonical({"role": "user", "content": None})

    def test_tool_calls_string_args_parsed_to_dict(self):
        str_args = [{"id": "c1", "type": "function", "function": {"name": "f", "arguments": '{"x": 1}'}}]
        dict_args = [{"id": "c1", "type": "function", "function": {"name": "f", "arguments": {"x": 1}}}]
        assert _canonical({"role": "assistant", "content": "", "tool_calls": str_args}) == \
            _canonical({"role": "assistant", "content": "", "tool_calls": dict_args})

    def test_tool_calls_dict_args_used_directly(self):
        dict_args = [{"id": "c1", "type": "function", "function": {"name": "f", "arguments": {"x": 1}}}]
        canonical = _canonical({"role": "assistant", "content": "", "tool_calls": dict_args})
        tool_calls_part = canonical.split("\x00")[2]
        expected = json.dumps([{
            "id": "c1",
            "type": "function",
            "function": {"name": "f", "arguments": {"x": 1}},
        }], sort_keys=True, separators=(",", ":"))
        assert tool_calls_part == expected

    def test_tool_calls_invalid_json_args_kept_as_string(self):
        bad_args = [{"id": "c1", "type": "function", "function": {"name": "f", "arguments": "not-json"}}]
        canonical = _canonical({"role": "assistant", "content": "", "tool_calls": bad_args})
        tool_calls_part = canonical.split("\x00")[2]
        expected = json.dumps([{
            "id": "c1",
            "type": "function",
            "function": {"name": "f", "arguments": "not-json"},
        }], sort_keys=True, separators=(",", ":"))
        assert tool_calls_part == expected

    def test_tool_calls_extra_fields_dropped(self):
        with_extras = [{
            "id": "c1",
            "type": "function",
            "function": {"name": "f", "arguments": "{}"},
            "timestamp": 12345,
            "internal_id": "abc",
        }]
        canonical_only = [{"id": "c1", "type": "function", "function": {"name": "f", "arguments": "{}"}}]
        assert _canonical({"role": "assistant", "content": "", "tool_calls": with_extras}) == \
            _canonical({"role": "assistant", "content": "", "tool_calls": canonical_only})

    def test_tool_call_id_dropped(self):
        with_id = {"role": "tool", "content": "result", "tool_call_id": "abc-123"}
        without_id = {"role": "tool", "content": "result"}
        assert _canonical(with_id) == _canonical(without_id)

    def test_role_casing_lowercased(self):
        assert _canonical({"role": "User", "content": "hi"}) == _canonical({"role": "user", "content": "hi"})
        assert _canonical({"role": "USER", "content": "hi"}) == _canonical({"role": "user", "content": "hi"})


class TestMessageHash:
    def test_same_message_same_hash(self):
        msg = {"role": "user", "content": "hello world"}
        assert _message_hash(msg) == _message_hash(dict(msg))

    def test_different_role_different_hash(self):
        a = _message_hash({"role": "user", "content": "x"})
        b = _message_hash({"role": "assistant", "content": "x"})
        assert a != b

    def test_different_content_different_hash(self):
        a = _message_hash({"role": "user", "content": "x"})
        b = _message_hash({"role": "user", "content": "y"})
        assert a != b

    def test_hash_is_16_hex_chars(self):
        h = _message_hash({"role": "user", "content": "x"})
        assert len(h) == 16
        assert all(c in "0123456789abcdef" for c in h)


class TestComputeChain:
    def test_empty_list_returns_empty(self):
        assert compute_chain([]) == ([], {})

    def test_all_system_messages_skipped(self):
        msgs = [
            {"role": "system", "content": "sys1"},
            {"role": "system", "content": "sys2"},
        ]
        assert compute_chain(msgs) == ([], {})

    def test_single_non_system_message(self):
        msg = _msg(content="hi")
        chain_list, chain_to_position = compute_chain([msg])
        expected = _message_hash(msg)
        assert chain_list == [expected]
        assert chain_to_position == {expected: 0}

    def test_system_then_user_only_user_in_chain(self):
        msgs = [
            {"role": "system", "content": "sys"},
            _msg(content="hi"),
        ]
        chain_list, chain_to_position = compute_chain(msgs)
        assert len(chain_list) == 1
        assert chain_list[0] == _message_hash(_msg(content="hi"))
        assert len(chain_to_position) == 1

    def test_first_chain_hash_seeds_with_msg_hash(self):
        msgs = [_msg(content="first"), _msg(content="second")]
        chain_list, _ = compute_chain(msgs)
        assert chain_list[0] == _message_hash(_msg(content="first"))
        expected_second = hashlib.sha256(
            f"{chain_list[0]}:{_message_hash(_msg(content='second'))}".encode("utf-8")
        ).hexdigest()[:16]
        assert chain_list[1] == expected_second

    def test_edit_at_k_invalidates_chain_from_k(self):
        base = [_msg(content=f"m{i}") for i in range(5)]
        edited = [_msg(content=f"m{i}") for i in range(5)]
        edited[2] = _msg(content="EDITED")
        base_chain, _ = compute_chain(base)
        edited_chain, _ = compute_chain(edited)
        assert base_chain[0] == edited_chain[0]
        assert base_chain[1] == edited_chain[1]
        for i in range(2, 5):
            assert base_chain[i] != edited_chain[i]

    def test_chain_stable_when_appending(self):
        base = [_msg(content=f"m{i}") for i in range(5)]
        longer = [_msg(content=f"m{i}") for i in range(7)]
        base_chain, _ = compute_chain(base)
        longer_chain, _ = compute_chain(longer)
        for i in range(5):
            assert base_chain[i] == longer_chain[i]
        assert len(longer_chain) == 7

    def test_system_message_change_does_not_shift_chain(self):
        msgs_a = [{"role": "system", "content": "sysA"}, _msg(content="u1"), _msg(content="u2")]
        msgs_b = [{"role": "system", "content": "sysB"}, _msg(content="u1"), _msg(content="u2")]
        chain_a, _ = compute_chain(msgs_a)
        chain_b, _ = compute_chain(msgs_b)
        assert chain_a == chain_b

    def test_chain_to_position_maps_correctly(self):
        msgs = [{"role": "system", "content": "sys"}, _msg(content="a"), _msg(content="b")]
        chain_list, chain_to_position = compute_chain(msgs)
        assert chain_to_position[chain_list[0]] == 0
        assert chain_to_position[chain_list[1]] == 1

    def test_perf_4350_messages_under_50ms(self):
        msgs = [_msg(content=f"message {i}") for i in range(4350)]
        start = time.perf_counter()
        compute_chain(msgs)
        elapsed_ms = (time.perf_counter() - start) * 1000
        assert elapsed_ms < 50, f"compute_chain took {elapsed_ms:.1f}ms"


class TestFindForkPoint:
    def test_identical_chains_fork_at_end(self):
        chain = ["a", "b", "c"]
        assert find_fork_point(chain, chain) == 3

    def test_edit_at_k_forks_at_k(self):
        new_chain = ["a", "b", "c", "d"]
        old_chain = ["a", "b", "X", "d"]
        assert find_fork_point(new_chain, old_chain) == 2

    def test_append_new_longer_forks_at_old_length(self):
        new_chain = ["a", "b", "c"]
        old_chain = ["a", "b"]
        assert find_fork_point(new_chain, old_chain) == 2

    def test_empty_old_chain_forks_at_zero(self):
        assert find_fork_point(["a", "b"], []) == 0

    def test_empty_new_chain_does_not_crash(self):
        assert find_fork_point([], ["a", "b"]) == 0

    def test_both_empty_forks_at_zero(self):
        assert find_fork_point([], []) == 0


class TestMultimodalConsistency:
    def test_list_text_content_matches_string_content(self):
        # HIGH 2: list content [{"type":"text","text":"hi"}] must produce the
        # same canonical as string "hi", matching _normalize_content.
        raw_list = _msg(content=[{"type": "text", "text": "hi"}])
        raw_str = _msg(content="hi")
        assert _canonical(raw_list) == _canonical(raw_str)
        assert _normalize_content(raw_list["content"]) == "hi"

    def test_db_stored_chain_hash_matches_runtime(self):
        # The DB stores content after applying _normalize_content. The DB
        # backfill/recovery reconstructs the message from the stored (normalized)
        # content column and computes chain_hash via _canonical. The runtime
        # computes chain_hash from the raw (possibly list) content. Both must
        # agree for restart recovery to load the correct chain.
        raw_msg = _msg(content=[{"type": "text", "text": "hello"}, {"type": "text", "text": "world"}])
        stored_content = _normalize_content(raw_msg["content"])
        db_row_msg = {"role": raw_msg["role"], "content": stored_content}

        runtime_chain, _ = compute_chain([raw_msg])
        db_chain, _ = compute_chain([db_row_msg])
        assert runtime_chain == db_chain
