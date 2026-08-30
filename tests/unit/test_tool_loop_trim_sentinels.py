"""H1/H2 arch-review-2 regression: tool_loop trim helpers preserve
``_sub_agent_return`` sentinel messages.

Pre-fix, the inline trim blocks in ``run_memory_tool_loop`` dropped oldest
non-system messages indiscriminately under token pressure or when the
count cap was hit. If a ``_sub_agent_return`` sentinel was among the
older messages, the production trim silently dropped it — which is
sub-agent round-trip state. The parent's record of the child's output
was lost.

Commit ``f79e802`` fixed the inline trim blocks (the actual production
code path). The f79e802 tests, however, were pinned to
``MessageBuffer.trim_to_budget`` and ``MessageBuffer.enforce_hard_cap``,
which are LIBRARY CODE with zero production callers in ``src/``. The
tests pinned dead code while the real production path went untested.

This file ports the H1/H2 sentinel-preservation assertions to the actual
production helpers (``_trim_working_messages_by_tokens`` and
``_enforce_working_messages_count_cap``), which are the extracted module-
level functions that ``run_memory_tool_loop`` calls. If the production
trim regresses, these tests catch it.
"""

import pytest


pytestmark = pytest.mark.unit


def _mk(role, content="x", sentinel=False):
    """Build a minimal message dict, optionally tagged as a sentinel.

    The content length controls token cost (tokens = len(content) *
    tokens_per_char). Long content => expensive message.
    """
    m = {"role": role, "content": content}
    if sentinel:
        m["_sub_agent_return"] = True
        m["tool_call_id"] = "call_test_subagent"
    return m


# Mirror the H2 fixture pattern. tokens_per_char must be a fixed value
# so token math is deterministic.
_TOKENS_PER_CHAR = 0.25


class TestTrimWorkingMessagesByTokensSentinelPreservation:
    """H2 arch-review-2: the production token-budget trim must preserve
    ``_sub_agent_return`` sentinel messages."""

    def test_preserves_sentinel_under_token_pressure(self):
        """A sentinel among the older messages survives a trim that drops
        many non-sentinel messages.

        Setup: budget small enough that only ~2 cheap messages fit. The
        sentinel is at index 0 (oldest non-system message) and would be
        the natural drop target.
        """
        from memory.tool_loop import _trim_working_messages_by_tokens

        msgs = [
            {"role": "system", "content": "you are helpful"},
            _mk("tool", "sentinel", sentinel=True),
        ]
        for i in range(20):
            role = "user" if i % 2 == 0 else "assistant"
            msgs.append(_mk(role, "x" * 200))

        assert len(msgs) == 22, f"test setup invariant: got {len(msgs)}, expected 22"

        result = _trim_working_messages_by_tokens(
            msgs, tokens_per_char=_TOKENS_PER_CHAR, max_tokens=5
        )

        survived_sentinels = [m for m in result if m.get("_sub_agent_return")]
        assert survived_sentinels, (
            "Sentinel was dropped by _trim_working_messages_by_tokens. "
            "H2 invariant: _sub_agent_return messages must survive under "
            "aggressive budget cuts in the production trim path."
        )

    def test_preserves_multiple_sentinels_across_buffer(self):
        """Multiple sentinels scattered through the buffer all survive."""
        from memory.tool_loop import _trim_working_messages_by_tokens

        msgs = [{"role": "system", "content": "sys"}]
        for i in range(12):
            if i in (1, 5, 9):
                msgs.append(_mk("tool", f"sentinel-{i}", sentinel=True))
            else:
                role = "user" if i % 2 == 0 else "assistant"
                msgs.append(_mk(role, "y" * 200))

        assert len(msgs) == 13

        result = _trim_working_messages_by_tokens(
            msgs, tokens_per_char=_TOKENS_PER_CHAR, max_tokens=10
        )

        surviving_sentinels = [m for m in result if m.get("_sub_agent_return")]
        assert len(surviving_sentinels) == 3, (
            f"Expected all 3 sentinels preserved; got "
            f"{len(surviving_sentinels)} surviving: "
            f"{[m.get('content', '')[:20] for m in surviving_sentinels]}"
        )

    def test_preserves_sentinel_when_only_sentinels_would_be_dropped(self):
        """If budget pressure forces drops and the ONLY candidates are
        sentinels, we accept overrun rather than lose them.

        Mirror of the count cap's overrun-acceptance invariant.
        """
        from memory.tool_loop import _trim_working_messages_by_tokens

        msgs = [
            {"role": "system", "content": "sys"},
            _mk("tool", "sentinel-old", sentinel=True),
            _mk("tool", "sentinel-new", sentinel=True),
            _mk("user", "newest user"),
        ]

        result = _trim_working_messages_by_tokens(
            msgs, tokens_per_char=_TOKENS_PER_CHAR, max_tokens=0
        )

        surviving_sentinels = [m for m in result if m.get("_sub_agent_return")]
        assert len(surviving_sentinels) == 2, (
            f"_trim_working_messages_by_tokens(max_tokens=0) must preserve "
            f"both sentinels (overrun accepted); got "
            f"{len(surviving_sentinels)} surviving."
        )
        assert surviving_sentinels[0]["content"] == "sentinel-old"
        assert surviving_sentinels[1]["content"] == "sentinel-new"

    def test_no_sentinels_preserves_normal_drop_behavior(self):
        """Without sentinels, the production trim behaves as before —
        drops oldest until budget fits. Regression guard: the fix must
        NOT change behavior for non-sentinel buffers.
        """
        from memory.tool_loop import _trim_working_messages_by_tokens

        msgs = [{"role": "system", "content": "sys"}]
        for i in range(6):
            msgs.append(_mk("user", f"message-{i}" + "x" * 30))

        assert len(msgs) == 7

        result = _trim_working_messages_by_tokens(
            msgs, tokens_per_char=_TOKENS_PER_CHAR, max_tokens=25
        )

        # System always survives.
        assert result[0].get("role") == "system", (
            "System message must always be preserved."
        )

        # Last user always survives.
        last_user = result[-1]
        assert last_user.get("role") == "user", (
            f"Last user message must survive; got {last_user}"
        )
        assert last_user["content"].startswith("message-5"), (
            f"Last user must be the chronologically last user; "
            f"got {last_user['content'][:30]!r}"
        )

        # No sentinels in buffer means nothing extra gets preserved.
        sentinels = [m for m in result if m.get("_sub_agent_return")]
        assert sentinels == [], (
            "Sentinel-preservation logic must NOT activate when there "
            "are no sentinels in the buffer. Found: "
            f"{[m.get('content') for m in sentinels]}"
        )

    def test_does_not_mutate_input_list(self):
        """The helper must return a NEW list; the input list is not mutated."""
        from memory.tool_loop import _trim_working_messages_by_tokens

        msgs = [
            {"role": "system", "content": "sys"},
            _mk("tool", "sentinel", sentinel=True),
            _mk("user", "x" * 200),
        ]
        original = list(msgs)
        result = _trim_working_messages_by_tokens(
            msgs, tokens_per_char=_TOKENS_PER_CHAR, max_tokens=5
        )
        assert msgs == original, "Input list was mutated by trim helper"
        assert result is not msgs, "Helper returned the same list object"


class TestEnforceWorkingMessagesCountCapSentinelPreservation:
    """H1 arch-review-2: the production count-based hard cap must preserve
    ``_sub_agent_return`` sentinel messages."""

    def test_preserves_sentinel_when_over_cap(self):
        """A sentinel among the older messages survives a count-cap trim."""
        from memory.tool_loop import _enforce_working_messages_count_cap

        msgs = [_mk("user", "u0"), _mk("tool", "sentinel", sentinel=True)]
        for i in range(1, 20):
            role = "assistant" if i % 2 else "user"
            msgs.append(_mk(role, f"m{i}"))

        assert len(msgs) == 21, f"test setup invariant: got {len(msgs)}, expected 21"

        result = _enforce_working_messages_count_cap(msgs, max_messages=10)
        survived = [m for m in result if m.get("_sub_agent_return")]
        assert len(survived) == 1, (
            f"Sentinel was dropped by _enforce_working_messages_count_cap. "
            f"Surviving sentinels: {len(survived)}. "
            f"This is H1 from arch-review-2 — the production hard cap "
            f"silently lost sub-agent round-trip state."
        )
        assert survived[0]["content"] == "sentinel"

    def test_preserves_multiple_sentinels_when_over_cap(self):
        """Multiple sentinels scattered through the buffer all survive."""
        from memory.tool_loop import _enforce_working_messages_count_cap

        msgs = []
        msgs.append(_mk("tool", "sentinel-A", sentinel=True))
        for i in range(99):
            msgs.append(_mk("user", f"u{i}"))
        msgs.append(_mk("tool", "sentinel-B", sentinel=True))
        for i in range(399):
            msgs.append(_mk("user", f"u{i}"))
        msgs.append(_mk("tool", "sentinel-C", sentinel=True))
        for i in range(600):
            msgs.append(_mk("assistant", f"a{i}"))

        result = _enforce_working_messages_count_cap(msgs, max_messages=100)
        survived = [m for m in result if m.get("_sub_agent_return")]
        assert len(survived) == 3, (
            f"Expected 3 sentinels preserved, got {len(survived)}. "
            f"Contents: {[m['content'] for m in survived]}. "
            f"H1 arch-review-2 bug: production hard cap dropped sentinels."
        )

    def test_no_drops_when_under_cap(self):
        """No messages dropped when count is at or below cap."""
        from memory.tool_loop import _enforce_working_messages_count_cap

        msgs = [_mk("user", f"u{i}") for i in range(50)]
        msgs.append(_mk("tool", "sentinel", sentinel=True))
        msgs += [_mk("assistant", f"a{i}") for i in range(49)]

        result = _enforce_working_messages_count_cap(msgs, max_messages=1000)
        assert len(result) == 100

    def test_sentinels_in_middle_preserved_under_cap_with_older_droppables(self):
        """Sentinel mid-buffer survives; older non-sentinels are dropped."""
        from memory.tool_loop import _enforce_working_messages_count_cap

        msgs = [_mk("user", f"u{i}") for i in range(10)]
        msgs.append(_mk("tool", "SENTINEL", sentinel=True))
        msgs += [_mk("user", f"u{i}") for i in range(1990)]

        result = _enforce_working_messages_count_cap(msgs, max_messages=1000)
        sentinel = [m for m in result if m.get("_sub_agent_return")]
        assert len(sentinel) == 1, (
            f"Sentinel at index 10 was dropped by the production hard cap. "
            f"H1 arch-review-2: _enforce_working_messages_count_cap lost "
            f"sub-agent round-trip state."
        )

    def test_system_messages_always_preserved(self):
        """System messages are preserved alongside sentinels."""
        from memory.tool_loop import _enforce_working_messages_count_cap

        msgs = [
            _mk("system", "sys1"),
            _mk("system", "sys2"),
        ]
        for i in range(2005):
            msgs.append(_mk("user", f"u{i}"))
        msgs.append(_mk("tool", "sub-return", sentinel=True))

        result = _enforce_working_messages_count_cap(msgs, max_messages=1000)
        sys_count = sum(1 for m in result if m.get("role") == "system")
        sentinel_count = sum(1 for m in result if m.get("_sub_agent_return"))
        assert sys_count == 2, f"Systems preserved = {sys_count}, expected 2"
        assert sentinel_count == 1, f"Sentinels preserved = {sentinel_count}, expected 1"

    def test_does_not_mutate_input_list(self):
        """The helper must return a NEW list; the input list is not mutated."""
        from memory.tool_loop import _enforce_working_messages_count_cap

        msgs = [_mk("user", f"u{i}") for i in range(50)]
        msgs.append(_mk("tool", "sentinel", sentinel=True))
        msgs += [_mk("assistant", f"a{i}") for i in range(49)]
        original = list(msgs)
        result = _enforce_working_messages_count_cap(msgs, max_messages=10)
        assert msgs == original, "Input list was mutated by count-cap helper"
        assert result is not msgs, "Helper returned the same list object"
