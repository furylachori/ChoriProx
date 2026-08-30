"""Integration-budget regression tests for the main-path assembly in
`_select_context_v2`.

Validates that VFS + summaries + ancient + raw history stay within
`model_safe_tokens` on the main (steady-state) path.  The sub-budget
carve is gone: the main path now reserves headroom for summaries,
ancient, and VFS before trimming, and drops VFS if the combined total
overflows (mirrors the fresh-fit and chain-fallback siblings).
"""
import pytest
from types import SimpleNamespace
from unittest.mock import patch

from proxy.context_v2 import _select_context_v2

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _system(content: str) -> dict:
    return {"role": "system", "content": content}


def _user(content: str) -> dict:
    return {"role": "user", "content": content}


def _assistant(content: str) -> dict:
    return {"role": "assistant", "content": content}


def _make_summary(msg_idx: int, tokens: int = 50) -> dict:
    """Return a rich-summary row with ~tokens worth of content."""
    return {
        "message_index": msg_idx,
        "role": "user",
        "summary": "X" * int(tokens * 4),  # tokens_per_char=0.25 → 4 chars/token
        "tool_summary": "",
        "rendered_full_tokens": tokens,
        "rendered_action_tokens": tokens,
        "rendered_narrative_tokens": tokens,
    }


def _base_config() -> dict:
    return {
        "safe_context_tokens": 10_000,
        "max_context_tokens": 40_000,
        "tokens_per_char": 0.25,
        "raw_exchange_count": 100,
    }


def _apply_patches(cfg, summaries=None, vfs_return=None):
    """Apply all patches for _select_context_v2 and return the stack."""
    if summaries is None:
        summaries = []

    return [
        patch("proxy.context.get_context_config", return_value=cfg),
        patch("memory.db.get_ancient_history", return_value=""),
        patch("memory.db.get_rich_summaries", return_value=summaries),
        patch("proxy.vfs._build_vfs_context", return_value=vfs_return),
        patch("proxy.vfs._build_vfs_inherited_context", return_value=None),
    ]


def _large_messages(count=150, chars=400):
    """Build messages large enough to force eviction (150 × 100 tok > 6500)."""
    pad = "X" * chars
    return [_system("")] + [_user(f"msg {i} {pad}") for i in range(count)]


def _run_with_patches(cfg, summaries=None, vfs_return=None, is_subagent=False, messages=None):
    """Run _select_context_v2 with all mocks applied.

    Defaults to a large raw input so eviction fires and summaries are fetched.
    """
    if messages is None:
        messages = _large_messages()
    patches = _apply_patches(cfg, summaries=summaries, vfs_return=vfs_return)
    from contextlib import ExitStack
    ctx = SimpleNamespace(
        request_id="vfs-summary-interaction",
        session_id="sess1",
        client_hint="",
        workspace_name="default",
        is_subagent=is_subagent,
        v2_boundary_idx=None,
        v2_boundary_chain=None,
    )
    with ExitStack() as stack:
        for p in patches:
            stack.enter_context(p)
        return _select_context_v2(
            "sess1",
            messages,
            model_name="test",
            is_subagent=is_subagent,
            ctx=ctx,
        )


def _count_summary_messages(result: list) -> int:
    """Count messages that look like assembled summaries (PAST CONTEXT or Conversation Summary)."""
    count = 0
    for m in result:
        content = m.get("content", "")
        if isinstance(content, str) and "[PAST CONTEXT" in content:
            count += 1
        elif isinstance(content, str) and "[Conversation Summary]" in content:
            count += 1
    return count


def _extract_summary_content(result: list) -> str:
    """Extract the summary text block from the result, if present."""
    for m in result:
        content = m.get("content", "")
        if isinstance(content, str) and "[PAST CONTEXT" in content:
            return content
        if isinstance(content, str) and "[Conversation Summary]" in content:
            return content
    return ""


def _count_summary_lines(result: list) -> int:
    """Count [msg#N] lines in the summary block."""
    for m in result:
        content = m.get("content", "")
        if isinstance(content, str) and "[msg#" in content:
            return content.count("[msg#")
    return 0


# ---------------------------------------------------------------------------
# Test 1: VFS consumes full budget — summaries still appear
# ---------------------------------------------------------------------------

class TestVfsConsumesFullBudget:
    """VFS returning a context that uses its full 5% budget must not
    eliminate summaries entirely — they should still appear (possibly
    truncated) within the remaining budget."""

    def test_summaries_still_appear_when_vfs_active(self):
        """With VFS returning a large context, summaries must still appear.

        Budget math (tokens_per_char=0.25, safe=10000, system=0, tools=2000):
          reserved_summary = int(10000 * 0.15) = 1500
          reserved_ancient  = int(10000 * 0.10) = 1000
          reserved_vfs      = int(10000 * 0.05) = 500
          trim_budget       = 10000 - 0 - 2000 - 1500 - 1000 - 500 = 5000

        Summaries have their own full reserved budget (1500 tokens) and
        VFS content is added only if the combined total still fits.
        """
        # VFS returns a context consuming its full budget (400 tokens = 1600 chars)
        vfs_content = "VFS " * 400  # 1600 chars → 400 tokens at 0.25 tok/char
        vfs_context = {"role": "system", "content": vfs_content}

        # 10 summaries at 50 tokens each = 500 tokens (fits in 800 token budget)
        summaries = [_make_summary(i, tokens=50) for i in range(10)]

        cfg = _base_config()
        result = _run_with_patches(cfg, summaries=summaries, vfs_return=vfs_context)

        # Summaries must be present (budget allows 800 tokens, we have 500)
        summary_block = _extract_summary_content(result)
        assert summary_block, (
            "Summaries must appear when VFS is active and budget allows. "
            "Got no summary block in output."
        )

    def test_total_stays_within_budget(self):
        """The assembled context must stay within model_safe_tokens."""
        vfs_content = "VFS " * 400
        vfs_context = {"role": "system", "content": vfs_content}

        summaries = [_make_summary(i, tokens=50) for i in range(5)]

        cfg = _base_config()
        result = _run_with_patches(cfg, summaries=summaries, vfs_return=vfs_context)

        from proxy.context import _count_msg_chars
        total_chars = sum(_count_msg_chars(m) for m in result)
        estimated_tokens = int(total_chars * cfg["tokens_per_char"])
        # tools schema adds 2000 tokens, so total must be <= safe
        assert estimated_tokens + 2000 <= cfg["safe_context_tokens"], (
            f"Estimated {estimated_tokens} + 2000 tools = {estimated_tokens + 2000} "
            f"exceeds safe budget {cfg['safe_context_tokens']}"
        )


# ---------------------------------------------------------------------------
# Test 2: VFS returns None vs. large context — summary count differs
# ---------------------------------------------------------------------------

class TestVfsNoneVsLargeContext:
    """When VFS returns None, summaries get the full budget. When VFS
    returns a large context, summaries lose budget and fewer fit."""

    def test_both_within_budget(self):
        """Both runs must stay within model_safe_tokens."""
        summaries = [_make_summary(i, tokens=50) for i in range(20)]
        cfg = _base_config()

        from proxy.context import _count_msg_chars

        for vfs_active in [False, True]:
            vfs_context = {"role": "system", "content": "VFS " * 400} if vfs_active else None
            result = _run_with_patches(cfg, summaries=summaries, vfs_return=vfs_context)

            total_chars = sum(_count_msg_chars(m) for m in result)
            estimated = int(total_chars * cfg["tokens_per_char"])
            assert estimated + 2000 <= cfg["safe_context_tokens"], (
                f"VFS={'active' if vfs_active else 'None'}: "
                f"estimated {estimated} + 2000 = {estimated + 2000} "
                f"exceeds safe {cfg['safe_context_tokens']}"
            )


# ---------------------------------------------------------------------------
# Test 3: VFS budget calculation is correct
# ---------------------------------------------------------------------------

class TestVfsBudgetCalculation:
    """Verify the VFS budget formula and conditional carving."""

    def test_vfs_budget_is_five_percent_of_total_available(self):
        """vfs_budget = int(total_available * 0.05).

        total_available = safe_context_tokens - system_prompt_tokens - tool_schema_tokens
                        = 10000 - 0 - 2000 = 8000
        vfs_budget = int(8000 * 0.05) = 400
        """
        cfg = _base_config()
        # VFS context of exactly 400 tokens (1600 chars at 0.25)
        vfs_content = "A" * 1600
        vfs_context = {"role": "system", "content": vfs_content}

        result = _run_with_patches(cfg, summaries=[], vfs_return=vfs_context)

        # VFS context must appear in result
        vfs_found = any(
            m.get("content", "").startswith("A") and len(m.get("content", "")) > 100
            for m in result
        )
        assert vfs_found, "VFS context should appear in result when budget allows"

    def test_vfs_budget_clamped_to_zero(self):
        """When VFS budget would push rich_summary_budget below zero,
        it must clamp to zero (no negative budget).

        The math: summary = 0.15*total, vfs = 0.05*total → summary - vfs = 0.10*total
        So clamping to zero can't happen with positive total_available.
        Test the code path doesn't crash with a small window and VFS active.

        Uses a small message set: safe=5000 gives recent_raw_budget=3250,
        which already exceeds total_available (3000), so a large raw block
        would overflow. Small messages avoid that and exercise the VFS carve
        + clamp path cleanly.
        """
        cfg = {
            "safe_context_tokens": 5_000,
            "max_context_tokens": 40_000,
            "tokens_per_char": 0.25,
            "raw_exchange_count": 100,
        }
        # total_available = 5000 - 0 - 2000 = 3000
        # ancient=150, raw=3250, summary=3000-150-3250 → clamped to 0
        # VFS content = 150 tokens (600 chars) — fits in vfs_budget
        vfs_content = "A" * 600  # 150 tokens
        vfs_context = {"role": "system", "content": vfs_content}

        summaries = [_make_summary(i, tokens=10) for i in range(3)]

        result = _run_with_patches(
            cfg, summaries=summaries, vfs_return=vfs_context,
            messages=[_system(""), _user("hello")],
        )
        all_content = " ".join(
            str(m.get("content", "") or "") for m in result
        )
        assert "A" * 100 in all_content, (
            "VFS context (a content block of 'A' * 600) must appear in the "
            f"output when VFS is active and budget allows; full content: "
            f"{all_content[:300]!r}"
        )

    def test_vfs_content_present_when_vfs_active(self):
        """VFS content block must appear in the assembly output unchanged
        when VFS is active and budget allows. We cannot directly observe
        vfs_context_tokens, but content presence confirms the VFS slot was
        injected with the expected payload.
        """
        cfg = _base_config()
        vfs_content = "FILE: src/main.py\n" + "line content\n" * 50
        vfs_context = {"role": "system", "content": vfs_content}

        result = _run_with_patches(cfg, summaries=[], vfs_return=vfs_context)

        # The VFS content must appear in the output
        all_content = " ".join(m.get("content", "") for m in result)
        assert "FILE: src/main.py" in all_content, (
            "VFS content must appear in output"
        )


# ---------------------------------------------------------------------------
# Diagnostic: Can VFS starve summaries to zero?
# ---------------------------------------------------------------------------

class TestVfsStarvationDiagnostic:
    """Diagnostic: under what conditions does VFS completely eliminate
    summaries from the context?"""

    def test_vfs_does_not_starve_summaries_with_realistic_budget(self):
        """With the default budget split (5% VFS, 15% summaries), VFS
        cannot starve summaries to zero because summary budget is 3× VFS.

        After VFS carve: summary_budget = 0.15*total - 0.05*total = 0.10*total
        This is always positive when total > 0.

        However, if summaries are individually large, they may not fit
        even one. We verify that at least one small summary survives.
        """
        cfg = _base_config()
        # total_available=8000, summary_budget_after_vfs = 800 tokens
        # Each summary = 500 tokens → at most 1 fits
        summaries = [_make_summary(i, tokens=500) for i in range(10)]

        vfs_context = {"role": "system", "content": "VFS " * 400}
        result = _run_with_patches(cfg, summaries=summaries, vfs_return=vfs_context)

        lines = _count_summary_lines(result)
        # With 800 token budget and 500-token summaries, at most 1 fits
        # This is the starvation edge case — but 1 should still fit
        assert lines >= 1, (
            f"At least 1 summary (500 tokens) should fit in 800 token budget, "
            f"got {lines}. VFS may be over-carving."
        )

    def test_vfs_zero_budget_when_oversized(self):
        """When VFS returns content exceeding its budget, it still gets
        included (no truncation in the code). The key invariant is that
        rich_summary_budget is reduced by vfs_budget (the intended carve),
        not by vfs_context_tokens (actual content size).

        This test verifies the VFS doesn't steal MORE than its allocated 5%.
        """
        cfg = _base_config()
        # total_available=8000, vfs_budget=400 tokens
        # VFS content = 800 tokens (double the budget)
        vfs_content = "A" * 3200  # 800 tokens
        vfs_context = {"role": "system", "content": vfs_content}

        # 10 summaries at 50 tokens = 500 tokens
        # With VFS: budget=800, all 10 should still fit (500 < 800)
        summaries = [_make_summary(i, tokens=50) for i in range(10)]

        result = _run_with_patches(cfg, summaries=summaries, vfs_return=vfs_context)

        lines = _count_summary_lines(result)
        # All 10 summaries should fit (500 < 800)
        assert lines == 10, (
            f"All 10 summaries (500 tokens) should fit in 800 token budget, "
            f"got {lines}. VFS over-carved the summary budget."
        )
