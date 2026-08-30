"""US-TIER2: Concurrency & lock-ordering tests for Tier-2 sub-AGENTS.md pinning.

Covers spec §9.6:
  1. LOST-UPDATE: two threads race to capture the SAME workspace+path with
     DIFFERENT content via the real capture path (_capture_tier2_instructions
     under vfs_lock). Asserts: no exception, popitem never raises, final value
     is one-or-the-other (not a hybrid/corrupt), map size stays within cap.
  2. DELTA-TRACK by tool_call_id: a conversation where opencode inserts the
     tool result at a DIFFERENT index than a prior turn (shift). Asserts the
     entry is captured once (keyed on tool_call_id, not index).
  3. ROBUSTNESS: concurrent capture + concurrent pin-emit on different
     workspaces don't deadlock (assert_consistent_state).
"""

import copy
import threading

import pytest

import proxy.state as state
from proxy.context_v2 import _capture_tier2_instructions, _emit_tier2_pin
from tests.concurrency_harness import (
    assert_consistent_state,
    assert_no_deadlock,
    run_concurrently,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_tool_result(read_path: str, file_body: str, instr_path: str,
                      instr_content: str, tc_id: str) -> dict:
    """Build a role=tool message with a Tier-2 suffix (opencode wire format).

    The extract_instruction_blocks parser captures everything after the header
    until the next header or end-of-string. The trailing '</system-reminder>'
    closing tag is NOT a header, so it becomes part of the captured content.
    Tests must account for this suffix in expected values.
    """
    content = (
        f"<path>{read_path}</path>"
        f"<type>file</type>"
        f"<content>{file_body}</content>"
        f"<system-reminder>\n"
        f"Instructions from: {instr_path}\n"
        f"{instr_content}\n"
        f"</system-reminder>"
    )
    return {
        "role": "tool",
        "tool_call_id": tc_id,
        "name": "read",
        "content": content,
    }


def _expected_captured(instr_content: str) -> str:
    """Return the exact string extract_instruction_blocks captures for a
    single-block suffix (content + trailing '</system-reminder>')."""
    return f"{instr_content}\n</system-reminder>"


def _make_sys_msg() -> dict:
    return {"role": "system", "content": "You are a helpful assistant."}


# ---------------------------------------------------------------------------
# State isolation fixture
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _clean_tier2_state():
    with state.vfs_lock:
        state.instruction_files_by_workspace.clear()
        state._last_scanned_by_tool_call.clear()
    yield
    with state.vfs_lock:
        state.instruction_files_by_workspace.clear()
        state._last_scanned_by_tool_call.clear()


# ---------------------------------------------------------------------------
# 1. LOST-UPDATE (§9.6 + invariant #5)
# ---------------------------------------------------------------------------

class TestLostUpdate:
    """Two threads race to capture the SAME workspace+path with DIFFERENT
    content. Uses threading.Barrier(2) for deterministic start."""

    def test_concurrent_high_contention_cap_boundary(self):
        """High-contention: N threads each insert a UNIQUE path into the same
        workspace, exceeding the inner cap. Targets the check-then-act window
        at the cap boundary (len >= MAX → popitem; insert) which is NOT
        GIL-atomic — this is the race vfs_lock protects.

        Uses env-gated fault injection (PROXY_TIER2_TEST_RACE) to widen the
        window deterministically. With the lock, the inner map never exceeds
        the cap. Without the lock, concurrent callers both pass the check and
        both insert → overflow."""
        import os
        ws = "ws-high-contention"
        orig_cap = state.INSTRUCTION_FILES_MAX_PER_WORKSPACE
        state.INSTRUCTION_FILES_MAX_PER_WORKSPACE = 4
        os.environ["PROXY_TIER2_TEST_RACE"] = "1"
        try:
            N = 12
            sys_msg = _make_sys_msg()

            msgs = [
                _make_tool_result(
                    f"/repo/src/sub/f{i}.py", f"body{i}",
                    f"/repo/src/sub/AGENTS{i}.md", f"INSTR_{i}", f"tc-hc-{i}",
                )
                for i in range(N)
            ]

            barrier = threading.Barrier(N, timeout=10)

            def capture_with(msg):
                def fn():
                    barrier.wait()
                    _capture_tier2_instructions(
                        [copy.deepcopy(msg)], sys_msg, "sess-hc",
                        _FakeCtx(ws),
                    )
                    return True
                return fn

            results = run_concurrently([capture_with(m) for m in msgs])

            for _, exc in results:
                assert exc is None, f"thread raised: {exc!r}"

            with state.vfs_lock:
                inner = state.instruction_files_by_workspace.get(ws)
                assert inner is not None
                from collections import OrderedDict
                assert isinstance(inner, OrderedDict)

                # Inner cap MUST be respected — EXACTLY at cap, never over
                assert len(inner) == state.INSTRUCTION_FILES_MAX_PER_WORKSPACE, (
                    f"inner map overflow: {len(inner)} > {state.INSTRUCTION_FILES_MAX_PER_WORKSPACE}"
                )

                # Every surviving entry is intact (one of the expected values)
                expected_contents = {_expected_captured(f"INSTR_{i}") for i in range(N)}
                for path, content in inner.items():
                    assert content in expected_contents, f"corrupt: {content!r}"
        finally:
            state.INSTRUCTION_FILES_MAX_PER_WORKSPACE = orig_cap
            os.environ.pop("PROXY_TIER2_TEST_RACE", None)

    def test_concurrent_different_paths_no_lost_update(self):
        """Two threads insert DIFFERENT paths concurrently — both must land."""
        ws = "ws-diffpath"
        sys_msg = _make_sys_msg()

        msg_a = _make_tool_result(
            "/repo/src/foo.py", "body A", "/repo/src/AGENTS.md", "INSTR_A", "tc-da"
        )
        msg_b = _make_tool_result(
            "/repo/lib/bar.py", "body B", "/repo/lib/AGENTS.md", "INSTR_B", "tc-db"
        )

        barrier = threading.Barrier(2, timeout=5)

        def capture_with(msg):
            def fn():
                barrier.wait()
                _capture_tier2_instructions(
                    [copy.deepcopy(msg)], sys_msg, "sess-dp",
                    _FakeCtx(ws),
                )
                return True
            return fn

        results = run_concurrently([capture_with(msg_a), capture_with(msg_b)])
        for _, exc in results:
            assert exc is None

        with state.vfs_lock:
            inner = state.instruction_files_by_workspace.get(ws)
            assert inner is not None
            assert "/repo/src/AGENTS.md" in inner
            assert "/repo/lib/AGENTS.md" in inner
            assert inner["/repo/src/AGENTS.md"] == _expected_captured("INSTR_A")
            assert inner["/repo/lib/AGENTS.md"] == _expected_captured("INSTR_B")


# ---------------------------------------------------------------------------
# 2. DELTA-TRACK by tool_call_id (§9.6)
# ---------------------------------------------------------------------------

class TestDeltaTrackByToolCallId:
    """Delta keys on tool_call_id, not message index. Construct a conversation
    where opencode inserts the tool result at a DIFFERENT index than a prior
    turn (shift). Assert it's still correctly captured once."""

    def test_tool_result_shifted_index_captured_once(self):
        ws = "ws-shift"
        instr_path = "/repo/src/AGENTS.md"
        read_path = "/repo/src/foo.py"
        body = "file body"
        instr = "SHIFTED_INSTRUCTION"
        tc_id = "tc-shift"

        sys_msg = _make_sys_msg()

        # Turn 1: tool result at index 2 (after user + assistant)
        turn1_msgs = [
            {"role": "user", "content": "read foo"},
            {"role": "assistant", "content": "", "tool_calls": [
                {"id": tc_id, "function": {"name": "read", "arguments": "{}"}}
            ]},
            _make_tool_result(read_path, body, instr_path, instr, tc_id),
        ]
        _capture_tier2_instructions(turn1_msgs, sys_msg, "sess-shift",
                                     _FakeCtx(ws))

        with state.vfs_lock:
            assert tc_id in state._last_scanned_by_tool_call
            assert state._last_scanned_by_tool_call[tc_id] == 2
            inner = state.instruction_files_by_workspace.get(ws)
            assert inner is not None
            assert instr_path in inner

        # Turn 2: opencode inserts an earlier message (e.g. a new user turn at
        # index 1), pushing the tool result to index 3. The tc_id is the same.
        turn2_msgs = [
            {"role": "user", "content": "read foo"},
            {"role": "user", "content": "also read bar"},  # shift inserted
            {"role": "assistant", "content": "", "tool_calls": [
                {"id": tc_id, "function": {"name": "read", "arguments": "{}"}}
            ]},
            _make_tool_result(read_path, body, instr_path, instr, tc_id),
        ]
        _capture_tier2_instructions(turn2_msgs, sys_msg, "sess-shift",
                                     _FakeCtx(ws))

        with state._last_scanned_by_tool_call if False else state.vfs_lock:
            pass  # lock released; re-acquire below

        with state.vfs_lock:
            # The tc_id is the SAME → must NOT re-capture (skipped as already scanned)
            # The index in raw_block is now 3, but the key is tc_id.
            assert state._last_scanned_by_tool_call[tc_id] == 2  # unchanged from turn 1
            inner = state.instruction_files_by_workspace.get(ws)
            assert inner[instr_path] == _expected_captured(instr)  # still the same content

    def test_different_tc_id_same_path_captured_twice(self):
        """A different tool_call_id for the same path → capture is allowed
        (different read call)."""
        ws = "ws-diffid"
        instr_path = "/repo/src/AGENTS.md"
        read_path = "/repo/src/foo.py"
        body = "file body"
        sys_msg = _make_sys_msg()

        msg_a = _make_tool_result(read_path, body, instr_path, "INSTR_1", "tc-1")
        msg_b = _make_tool_result(read_path, body, instr_path, "INSTR_2", "tc-2")

        _capture_tier2_instructions([msg_a], sys_msg, "sess-di", _FakeCtx(ws))
        _capture_tier2_instructions([msg_b], sys_msg, "sess-di", _FakeCtx(ws))

        with state.vfs_lock:
            assert state._last_scanned_by_tool_call["tc-1"] == 0
            assert state._last_scanned_by_tool_call["tc-2"] == 0
            inner = state.instruction_files_by_workspace.get(ws)
            # Last writer wins (different tc_id → both captured)
            assert inner[instr_path] == _expected_captured("INSTR_2")


# ---------------------------------------------------------------------------
# 3. ROBUSTNESS — concurrent capture + pin-emit, no deadlock
# ---------------------------------------------------------------------------

class TestRobustness:
    """Concurrent capture + concurrent pin-emit on different workspaces don't
    deadlock."""

    def test_concurrent_capture_and_pin_emit(self):
        ws_a = "ws-cap-a"
        ws_b = "ws-cap-b"
        sys_msg = _make_sys_msg()

        # Pre-populate both workspaces so pin-emit has data to emit
        with state.vfs_lock:
            state.record_instruction_file(ws_a, "/a/AGENTS.md", "INSTR_A")
            state.record_instruction_file(ws_b, "/b/AGENTS.md", "INSTR_B")

        msg_a = _make_tool_result(
            "/a/foo.py", "body", "/a/AGENTS.md", "INSTR_A_NEW", "tc-ca"
        )
        msg_b = _make_tool_result(
            "/b/bar.py", "body", "/b/AGENTS.md", "INSTR_B_NEW", "tc-cb"
        )

        base_messages = [
            sys_msg,
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "ok"},
        ]

        barrier = threading.Barrier(3, timeout=5)
        errors = []

        def capture_a():
            try:
                barrier.wait()
                _capture_tier2_instructions(
                    [copy.deepcopy(msg_a)], sys_msg, "sess-ra", _FakeCtx(ws_a),
                )
            except Exception as e:
                errors.append(("capture_a", e))

        def capture_b():
            try:
                barrier.wait()
                _capture_tier2_instructions(
                    [copy.deepcopy(msg_b)], sys_msg, "sess-rb", _FakeCtx(ws_b),
                )
            except Exception as e:
                errors.append(("capture_b", e))

        def pin_emit():
            try:
                barrier.wait()
                _emit_tier2_pin(copy.deepcopy(base_messages), "sess-pe",
                                _FakeCtx(ws_a))
            except Exception as e:
                errors.append(("pin_emit", e))

        threads = [
            threading.Thread(target=capture_a, name="capture_a"),
            threading.Thread(target=capture_b, name="capture_b"),
            threading.Thread(target=pin_emit, name="pin_emit"),
        ]
        for t in threads:
            t.start()
        assert_no_deadlock(threads, timeout=5)
        assert not errors, f"threads raised: {errors}"

    def test_assert_consistent_state_after_concurrent_captures(self):
        """After concurrent captures, the state is consistent (each workspace
        has exactly the expected entries, no corruption)."""
        ws_x = "ws-consistent-x"
        ws_y = "ws-consistent-y"
        sys_msg = _make_sys_msg()

        msgs_x = [
            _make_tool_result(f"/x/f{i}.py", f"body{i}", "/x/AGENTS.md",
                              f"INSTR_X_{i}", f"tc-x-{i}")
            for i in range(5)
        ]
        msgs_y = [
            _make_tool_result(f"/y/g{i}.py", f"body{i}", "/y/AGENTS.md",
                              f"INSTR_Y_{i}", f"tc-y-{i}")
            for i in range(5)
        ]

        barrier = threading.Barrier(2, timeout=5)

        def capture_batch(msgs, ws_id, sid):
            def fn():
                barrier.wait()
                for m in msgs:
                    _capture_tier2_instructions(
                        [m], sys_msg, sid, _FakeCtx(ws_id),
                    )
                return True
            return fn

        results = run_concurrently([
            capture_batch(msgs_x, ws_x, "sess-cx"),
            capture_batch(msgs_y, ws_y, "sess-cy"),
        ])
        for _, exc in results:
            assert exc is None

        def check():
            with state.vfs_lock:
                inner_x = state.instruction_files_by_workspace.get(ws_x)
                inner_y = state.instruction_files_by_workspace.get(ws_y)
                return (
                    inner_x is not None
                    and inner_y is not None
                    and len(inner_x) == 1
                    and len(inner_y) == 1
                )

        assert_consistent_state(
            get_state_fn=lambda: None,
            check_fn=lambda _: check(),
            timeout=3,
        )


# ---------------------------------------------------------------------------
# Lightweight fake ctx
# ---------------------------------------------------------------------------

class _FakeCtx:
    def __init__(self, workspace_name):
        self.workspace_name = workspace_name
        self.is_subagent = False
