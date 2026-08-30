"""C2 (Commit 2): ticket MAIN/SUB prompts are role-split.

Pre-C2: TICKET_MAIN_PREAMBLE + TICKET_MANAGER_PROMPT was sent to MAIN;
TICKET_MANAGER_PROMPT alone was sent to SUB. The body opened with
"You are a focused sub-agent..." which contradicted MAIN's orchestrator
identity, requiring a NOTE band-aid to explain.

Post-C2: TICKET_MAIN_PROMPT contains MAIN's orchestrator identity plus
the shared reference body. TICKET_SUB_PROMPT contains the sub-agent
identity plus the same shared reference body. The 10-action table,
hash rules, and guardrails live in ONE shared constant
TICKET_SHARED_REFERENCE — included by both MAIN and SUB.

This file pins:
1. MAIN prompt = TICKET_MAIN_PROMPT, does NOT contain "focused sub-agent".
2. SUB prompt = TICKET_SUB_PROMPT, DOES contain "focused sub-agent".
3. Both prompts include the shared reference (one source of truth).
4. Exports updated to TICKET_MAIN_PROMPT / TICKET_SUB_PROMPT.
"""
from __future__ import annotations

import pytest

from proxy.skills.skill_definition import SkillRequestContext
from skills.ticket_skill import TicketSkill
from proxy.tool_parsing.ticket_prompts import (
    TICKET_MAIN_PROMPT,
    TICKET_SUB_PROMPT,
    TICKET_SHARED_REFERENCE,
)


def _ctx(*, is_subagent: bool) -> SkillRequestContext:
    return SkillRequestContext(
        session_id="sess-ticket-main",
        is_subagent=is_subagent,
        inherited_skill=None,
        subagent_ticket_mode=False,
        workspace_name="",
        client_hint=None,
        body={},
        seeded_tasks=[],
        mode_state=None,
    )


class TestTicketMainPromptIdentity:
    """MAIN prompt has the orchestrator identity, NOT the sub-agent identity."""

    def test_main_get_system_prompt_returns_ticket_main_prompt(self):
        ctx = _ctx(is_subagent=False)
        prompt = TicketSkill().get_system_prompt(ctx)
        assert prompt == TICKET_MAIN_PROMPT, (
            "MAIN get_system_prompt must return TICKET_MAIN_PROMPT directly "
            "(no preamble+manager concatenation; C2 split)."
        )

    def test_main_does_not_say_focused_sub_agent(self):
        ctx = _ctx(is_subagent=False)
        prompt = TicketSkill().get_system_prompt(ctx)
        assert "focused sub-agent" not in prompt, (
            "MAIN prompt must NOT contain 'focused sub-agent' — that's the "
            "sub-agent identity. C2 splits the prompts so MAIN gets the "
            "orchestrator identity instead."
        )

    def test_main_says_planning_orchestrator(self):
        ctx = _ctx(is_subagent=False)
        prompt = TicketSkill().get_system_prompt(ctx)
        assert "PLANNING ORCHESTRATOR" in prompt, (
            "MAIN prompt must open with the PLANNING ORCHESTRATOR identity."
        )

    def test_main_mentions_task_tool_and_delegate(self):
        ctx = _ctx(is_subagent=False)
        prompt = TicketSkill().get_system_prompt(ctx)
        assert "task" in prompt.lower(), "MAIN prompt must mention the task tool"
        assert "delegate" in prompt.lower(), "MAIN prompt must say 'delegate'"

    def test_main_prompt_no_stop_work_exception(self):
        """C6.5: TICKET_MAIN_PROMPT must not reference the deleted
        proxy_work(stop_work) Bug #3 exemption (C4.5 cleaned prompt text)."""
        from proxy.tool_parsing.ticket_prompts import TICKET_MAIN_PROMPT
        lower = TICKET_MAIN_PROMPT.lower()
        # The stale "except proxy_work(stop_work)" phrasing must be gone
        assert "except `proxy_work(stop_work)`" not in lower, (
            f"C4.5 fix 2: TICKET_MAIN_PROMPT still contains stale stop_work exemption"
        )
        # Also check for the broader stale phrasing
        assert "expired work session" not in lower, (
            f"C4.5 fix 2: TICKET_MAIN_PROMPT still references 'expired work session'"
        )


class TestTicketSubPromptIdentity:
    """SUB prompt has the sub-agent identity."""

    def test_sub_get_subagent_prompt_returns_ticket_sub_prompt(self):
        ctx = _ctx(is_subagent=True)
        prompt = TicketSkill().get_subagent_prompt(ctx)
        assert prompt == TICKET_SUB_PROMPT, (
            "SUB get_subagent_prompt must return TICKET_SUB_PROMPT directly "
            "(no longer the bare TICKET_MANAGER_PROMPT)."
        )

    def test_sub_says_focused_sub_agent(self):
        ctx = _ctx(is_subagent=True)
        prompt = TicketSkill().get_subagent_prompt(ctx)
        assert "focused sub-agent" in prompt, (
            "SUB prompt must retain 'focused sub-agent' identity (the workers)."
        )

    def test_sub_does_not_say_planning_orchestrator(self):
        ctx = _ctx(is_subagent=True)
        prompt = TicketSkill().get_subagent_prompt(ctx)
        assert "PLANNING ORCHESTRATOR" not in prompt, (
            "SUB prompt must NOT contain 'PLANNING ORCHESTRATOR' — that's the "
            "MAIN identity."
        )


class TestTicketSharedReference:
    """The 10-action table, hash rules, and guardrails live in ONE constant."""

    def test_main_and_sub_share_reference_body(self):
        """Both MAIN and SUB include TICKET_SHARED_REFERENCE verbatim."""
        assert TICKET_SHARED_REFERENCE in TICKET_MAIN_PROMPT, (
            "TICKET_MAIN_PROMPT must include TICKET_SHARED_REFERENCE"
        )
        assert TICKET_SHARED_REFERENCE in TICKET_SUB_PROMPT, (
            "TICKET_SUB_PROMPT must include TICKET_SHARED_REFERENCE"
        )

    def test_shared_reference_lists_all_ten_actions(self):
        """The 10-action inventory lives ONLY in the shared reference."""
        for action in (
            "create_tasks", "update_task", "list_tasks", "get_task", "edit",
            "set_parent", "clear_parent", "clear_blocked_by",
            "set_priority", "append_review_note",
        ):
            assert action in TICKET_SHARED_REFERENCE, (
                f"TICKET_SHARED_REFERENCE must list action {action!r}"
            )

    def test_shared_reference_appears_once_in_each_prompt(self):
        """TICKET_SHARED_REFERENCE must be included EXACTLY ONCE in each prompt.

        The structural invariant is: each prompt contains the shared
        reference as a single substring (no fork). Counting action words
        like 'create_tasks' is unreliable because the shared reference
        itself mentions each action multiple times.
        """
        assert TICKET_MAIN_PROMPT.count(TICKET_SHARED_REFERENCE) == 1, (
            f"TICKET_SHARED_REFERENCE must appear exactly once in "
            f"TICKET_MAIN_PROMPT (got {TICKET_MAIN_PROMPT.count(TICKET_SHARED_REFERENCE)})"
        )
        assert TICKET_SUB_PROMPT.count(TICKET_SHARED_REFERENCE) == 1, (
            f"TICKET_SHARED_REFERENCE must appear exactly once in "
            f"TICKET_SUB_PROMPT (got {TICKET_SUB_PROMPT.count(TICKET_SHARED_REFERENCE)})"
        )


class TestTicketMainPromptExports:
    """The new constants are exported from proxy.tool_parsing."""

    def test_main_prompt_exported(self):
        from proxy.tool_parsing import TICKET_MAIN_PROMPT as _exp  # noqa: F401

        import proxy.tool_parsing as pkg

        assert "TICKET_MAIN_PROMPT" in pkg.__all__

    def test_sub_prompt_exported(self):
        from proxy.tool_parsing import TICKET_SUB_PROMPT as _exp  # noqa: F401

        import proxy.tool_parsing as pkg

        assert "TICKET_SUB_PROMPT" in pkg.__all__

    def test_shared_reference_exported(self):
        from proxy.tool_parsing import TICKET_SHARED_REFERENCE as _exp  # noqa: F401

        import proxy.tool_parsing as pkg

        assert "TICKET_SHARED_REFERENCE" in pkg.__all__

    def test_old_constants_removed(self):
        """The old TICKET_MANAGER_PROMPT and TICKET_MAIN_PREAMBLE are DELETED."""
        import proxy.tool_parsing.ticket_prompts as mod
        import proxy.tool_parsing as pkg

        assert not hasattr(mod, "TICKET_MANAGER_PROMPT"), (
            "TICKET_MANAGER_PROMPT must be deleted (no backcompat alias)"
        )
        assert not hasattr(mod, "TICKET_MAIN_PREAMBLE"), (
            "TICKET_MAIN_PREAMBLE must be deleted (no backcompat alias)"
        )
        assert "TICKET_MANAGER_PROMPT" not in pkg.__all__
        assert "TICKET_MAIN_PREAMBLE" not in pkg.__all__