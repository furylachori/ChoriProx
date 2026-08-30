"""Slice 6 — prompt retry-policy contract tests.

Asserts the ticket prompt contains a consistent retry/error-handling policy
and does NOT contain the old contradictory phrases.
"""

import pytest
from proxy.tool_parsing.ticket_prompts import TICKET_SHARED_REFERENCE, TICKET_SUB_PROMPT, TICKET_MAIN_PROMPT


class TestRetryPolicyPresent:
    """The replacement policy block must be present in TICKET_SHARED_REFERENCE."""

    def test_bounded_total_attempts(self):
        assert "Maximum 3 total attempts" in TICKET_SHARED_REFERENCE

    def test_retry_only_after_change(self):
        assert "Retry only after changing the invalid argument" in TICKET_SHARED_REFERENCE

    def test_no_retry_for_access_denial(self):
        assert "Do not repeat the same call" in TICKET_SHARED_REFERENCE

    def test_mandatory_closure_prefix(self):
        assert "Always end with COMPLETE, PARTIAL, or FAILED" in TICKET_SHARED_REFERENCE

    def test_raw_dump_prohibition(self):
        assert "never a valid final response" in TICKET_SHARED_REFERENCE

    def test_infrastructure_retry_once(self):
        assert "Retry once if the result says retryable=true" in TICKET_SHARED_REFERENCE

    def test_validation_budget_matches_prompt_spec(self):
        """Cross-reference: SemanticRetryTracker.BUDGETS['validation'] must agree
        with the prompt's stated maximum attempts. Drift between these is silent
        until a user sees earlier-than-expected retry exhaustion."""
        from proxy.tool_pipeline.tool_limits import SemanticRetryTracker
        from proxy.tool_parsing.ticket_prompts import TICKET_SHARED_REFERENCE

        budget = SemanticRetryTracker.BUDGETS["validation"]
        assert f"Maximum {budget} total attempts" in TICKET_SHARED_REFERENCE, (
            f"SemanticRetryTracker.BUDGETS['validation']={budget} does not match "
            f"the prompt's stated maximum. Update either the code budget or the prompt."
        )

    def test_same_error_repeat_stop(self):
        assert "If the same error repeats twice, stop retrying" in TICKET_SHARED_REFERENCE


class TestOldContradictoryPhrasesAbsent:
    """The old contradictory phrases must be ABSENT."""

    def test_no_retry_and_proceed_phrase(self):
        assert "After 3, report error and proceed" not in TICKET_SHARED_REFERENCE

    def test_no_stop_on_non_success(self):
        assert "Non-success from proxy_ticket: print it and STOP" not in TICKET_SHARED_REFERENCE

    def test_no_duplicate_max_3_retries_in_hard_limits(self):
        lines = TICKET_SHARED_REFERENCE.splitlines()
        in_hard_limits = False
        for line in lines:
            if "Hard Limits" in line:
                in_hard_limits = True
            if in_hard_limits and "Max 3 retries" in line:
                assert False, "Max 3 retries should not be in Hard Limits section"

    def test_no_hardcoded_max_3_retries_line(self):
        for line in TICKET_SHARED_REFERENCE.splitlines():
            if line.strip() == "- Max 3 retries per individual call.":
                assert False, f"Standalone 'Max 3 retries' line still present: {line!r}"


class TestPolicyInheritedBySubAndMain:
    """Both MAIN and SUB prompts include TICKET_SHARED_REFERENCE."""

    def test_sub_prompt_has_policy(self):
        assert "Maximum 3 total attempts" in TICKET_SUB_PROMPT

    def test_main_prompt_has_policy(self):
        assert "Maximum 3 total attempts" in TICKET_MAIN_PROMPT
