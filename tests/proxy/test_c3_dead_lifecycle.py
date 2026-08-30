"""C3 (Commit 3): dead-lifecycle deletion behavioral tests.

After C3:
- resolve_mode_with_fallback is DELETED (zero production callers).
- Bug #3 stop_work exemption is DELETED (one-shot semantics: only work-
  mode agents run work sessions, so MAIN in ticket mode cannot call
  proxy_work at all).
- WorkSkill.fallback_skill = None (no builtin has a fallback).
- Expired work mode resolves to Mode.DEFAULT (no ticket ModeState
  installed).
- All guard/tool denial strings tell the MODEL what it can do, not
  what ^^ command to run.

These tests pin the cross-cutting invariants at the BEHAVIORAL boundary
— through real hooks and real guards, not by inspecting constants.
Each test has a corresponding revert-test in C3 STEP 4.
"""
import re
import unittest
from unittest.mock import MagicMock, patch


class TestStopWorkBlockedForMainInTicketMode(unittest.TestCase):
    """C3 (a): ticket-MAIN calling proxy_work(stop_work) is BLOCKED.

    Revert-test 4.1: restoring the Bug #3 exemption lets the call through.
    """

    @patch("proxy.agent.agent_registry.registry")
    def test_stop_work_blocked_for_main_in_ticket_mode(self, mock_reg):
        from proxy.tool_pipeline.tool_call import ToolCall
        from proxy.tool_pipeline.tool_guard import MemoryToolGuard

        tool_call = ToolCall(
            id="t1",
            name="proxy_work",
            arguments={"action": "stop_work", "short_hash": "abc123", "ws_uuid": "ws-1"},
            source="native",
            from_cache=False,
        )

        # Build a ctx with is_subagent=False (MAIN) and a ticket-mode agent.
        ctx = MagicMock()
        ctx.session_id = "sess-c3-stopwork"
        ctx.is_subagent = False
        ctx.client_hint = ""

        # Patch the agent registry to return a ticket-mode agent.
        agent = MagicMock()
        ms = MagicMock()
        ms.name = "ticket"
        agent.snapshot_mode.return_value = ms
        mock_reg.get_or_load.return_value = agent

        guard = MemoryToolGuard()
        result = guard._check_ticket_mode_orchestrator(tool_call, ctx)

        # After C3: ticket MAIN calling proxy_work(stop_work) MUST be blocked.
        self.assertIsNotNone(
            result,
            f"C3 REGRESSION: ticket MAIN proxy_work(stop_work) was NOT blocked; "
            f"result={result!r}",
        )
        # Error message must use the new "delegate" framing.
        result_lower = result.lower()
        self.assertTrue(
            "blocked" in result_lower or "delegate" in result_lower,
            f"C3 error message must contain 'blocked' or 'delegate'; got: {result!r}",
        )


class TestNoUseCaretCaretInDenialStrings(unittest.TestCase):
    """C3 (b): no denial string tells the model to 'use ^^' or 'activate a skill'.

    The actor test: is the model the actor ("use ^^X" = BAD) or relaying
    ("tell the user to run ^^X" = OK)?
    Revert-test 4.2: reverting any denial string to old text makes this fail.

    NOTE: the previous AST-based extractor only caught bare `return "string"`
    patterns; denial strings live in dict literals and f-strings, which are
    invisible to the AST. This version reads raw source text and matches
    with regex — it inspects every string in the file, including dict
    literals, f-strings, and concatenated strings.
    """

    FILES = [
        "src/proxy/skill_guard.py",
        "src/proxy/tool_pipeline/tool_guard.py",
        "src/memory/tools/proxy_ticket.py",
        "src/memory/tools/proxy_work.py",
    ]

    def _read_source(self, path):
        with open(path) as f:
            return f.read()

    def test_no_use_caret_caret_in_denial_strings(self):
        """No source contains 'use ^^' (model as actor, BAD)."""
        pattern = re.compile(r"use\s+\^\^", re.IGNORECASE)
        for path in self.FILES:
            source = self._read_source(path)
            matches = pattern.findall(source)
            self.assertFalse(
                matches,
                f"C3 REGRESSION: {path} tells the model to 'use ^^': {matches}"
            )

    def test_no_activate_a_skill_in_denial_strings(self):
        """No source contains 'activate a skill' (model as actor, BAD)."""
        pattern = re.compile(r"activate\s+a\s+skill", re.IGNORECASE)
        for path in self.FILES:
            source = self._read_source(path)
            matches = pattern.findall(source)
            self.assertFalse(
                matches,
                f"C3 REGRESSION: {path} tells the model to 'activate a skill': {matches}"
            )

    def test_no_switch_to_mode_in_denial_strings(self):
        """No source contains 'switch to ... mode' (model can't switch)."""
        pattern = re.compile(r"switch\s+to\s+\w+\s+mode", re.IGNORECASE)
        for path in self.FILES:
            source = self._read_source(path)
            matches = pattern.findall(source)
            self.assertFalse(
                matches,
                f"C3 REGRESSION: {path} tells the model to 'switch to mode': {matches}"
            )

    def test_caret_caret_only_in_user_relay_context(self):
        """`^^` mentions must NOT be in 'use ^^X' or 'activate ^^X' (model as actor)."""
        bad_patterns = [
            re.compile(r"use\s+\^\^", re.IGNORECASE),
            re.compile(r"activate\s+\^\^", re.IGNORECASE),
        ]
        for path in self.FILES:
            source = self._read_source(path)
            for pattern in bad_patterns:
                matches = pattern.findall(source)
                self.assertFalse(
                    matches,
                    f"C3 REGRESSION: {path} contains bad actor pattern: {matches}"
                )


class TestExpiredWorkModeResolvesToDefault(unittest.TestCase):
    """C3 (c): expired work mode resolves to Mode.DEFAULT, no ticket ModeState installed.

    This pins the deletion of resolve_mode_with_fallback: without it,
    expired work mode can't fall back to ticket mode (no orphan cleanup
    path — orphaned sessions are handled by the 24h stale sweep).
    """

    def test_expired_work_mode_resolves_to_default(self):
        """An agent with an expired work ModeState resolves to Mode.DEFAULT."""
        from proxy.skill_resolver import resolve_mode
        from proxy.skills.skill_registry import SkillRegistry
        from proxy.skills.skill_definition import ModeState
        from proxy.mode import Mode

        # Build an agent with an expired work ModeState.
        agent = MagicMock()
        expired_mode = MagicMock(spec=ModeState)
        expired_mode.name = "work"
        expired_mode.is_expired.return_value = True
        agent.snapshot_mode.return_value = expired_mode
        agent.snapshot_mode_and_ticket_ts.return_value = (expired_mode, None)

        ctx = MagicMock()
        ctx.is_subagent = False
        ctx.subagent_ticket_mode = False
        ctx.inherited_skill = None
        ctx.workspace_name = ""
        ctx._skill_activated = False

        registry = MagicMock(spec=SkillRegistry)

        result = resolve_mode(ctx, agent, registry)

        # After C3: expired work mode resolves to Mode.DEFAULT. The resolver
        # returns str(Mode.DEFAULT.value) — i.e., the stringified int value —
        # to survive `importlib.reload(proxy.mode)` (see skill_resolver.py:_mode_value).
        self.assertEqual(result, str(Mode.DEFAULT.value), (
            f"C3 REGRESSION: expired work mode did not resolve to Mode.DEFAULT; got {result!r}"
        ))


class TestWorkSkillFallbackSkillIsNone(unittest.TestCase):
    """C3 (d): WorkSkill.fallback_skill is None (the work→ticket fallback is deleted).

    Revert-test 4.3: reverting to "ticket" makes this fail.
    """

    def test_workskill_fallback_skill_is_none(self):
        from skills.work_skill import WorkSkill
        from proxy.skills.skill_definition import TrustToken, _held_trust_token

        # This test module is not in TRUSTED_TRUST_CALLERS, so
        # `TrustToken.for_trusted_caller()` would raise PermissionError in
        # production. Patch the registry allowlist to include our module
        # so the token mint succeeds (the call-frame check on this test
        # is exempt because `tests.*` callers are trusted per the docs).
        with patch(
            "proxy.skills.skill_registry.TRUSTED_TRUST_CALLERS",
            frozenset({"tests.proxy.test_c3_dead_lifecycle"}),
        ):
            with _held_trust_token(TrustToken.for_trusted_caller()):
                sd = WorkSkill().to_skill_definition()

        self.assertIsNone(
            sd.fallback_skill,
            f"C3 REGRESSION: WorkSkill.fallback_skill must be None; got {sd.fallback_skill!r}",
        )


if __name__ == "__main__":
    unittest.main()