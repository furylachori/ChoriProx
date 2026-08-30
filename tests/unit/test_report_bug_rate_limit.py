"""US-088: report_bug rate limited to 5 calls per request.

Verifies the ``try_count_report_bug()`` method on ``ToolLimits`` (the
underlying primitive used by ``ToolCallTracker``) enforces the
per-request hard cap of 5 ``report_bug`` calls. The 6th call MUST be
rejected with ``False`` (the guard then converts this to a user-facing
error message).
"""
import pytest

from proxy.tool_pipeline.tool_call_tracker import ToolCallTracker
from proxy.tool_pipeline.tool_limits import MAX_REPORT_BUG_CALLS, ToolLimits


SESS = "sess-rate-bug-1"


class TestReportBugRateLimit:
    """US-088: report_bug is rate limited to 5 calls per request."""

    def setup_method(self):
        """Reset the global tracker and create a fresh limits instance."""
        ToolCallTracker.reset_instance(SESS)
        self.limits = ToolLimits()

    def teardown_method(self):
        ToolCallTracker.reset_instance(SESS)

    def test_first_five_calls_allowed(self):
        """The first 5 report_bug calls MUST be allowed (return True)."""
        for i in range(MAX_REPORT_BUG_CALLS):
            assert self.limits.try_count_report_bug(f"call-{i}") is True, (
                f"Call {i + 1} of {MAX_REPORT_BUG_CALLS} should be allowed"
            )

    def test_sixth_call_rejected(self):
        """The 6th report_bug call MUST be rejected (return False)."""
        for i in range(MAX_REPORT_BUG_CALLS):
            self.limits.try_count_report_bug(f"call-{i}")
        assert self.limits.try_count_report_bug("call-6") is False, (
            "6th call must be rejected — MAX_REPORT_BUG_CALLS is 5"
        )

    def test_count_is_at_max_after_five_calls(self):
        """After 5 allowed calls, ``report_bug_count`` is 5."""
        for i in range(MAX_REPORT_BUG_CALLS):
            self.limits.try_count_report_bug(f"call-{i}")
        assert self.limits.report_bug_count == MAX_REPORT_BUG_CALLS

    def test_duplicate_call_id_is_idempotent(self):
        """Calling with the same call_id twice counts as one (idempotent)."""
        assert self.limits.try_count_report_bug("dup-call") is True
        assert self.limits.try_count_report_bug("dup-call") is True
        assert self.limits.report_bug_count == 1

    def test_max_constant_is_five(self):
        """Sanity guard: MAX_REPORT_BUG_CALLS is exactly 5 per US-088."""
        assert MAX_REPORT_BUG_CALLS == 5

    def test_separate_call_ids_each_counted_separately(self):
        """Five distinct call_ids exhaust the budget; a 6th unique id
        is rejected.
        """
        distinct_ids = [f"unique-{i}" for i in range(5)]
        for cid in distinct_ids:
            assert self.limits.try_count_report_bug(cid) is True
        assert self.limits.try_count_report_bug("unique-6") is False

    def test_reset_clears_count(self):
        """reset() MUST clear the report_bug counter (new request)."""
        for i in range(MAX_REPORT_BUG_CALLS):
            self.limits.try_count_report_bug(f"call-{i}")
        assert self.limits.try_count_report_bug("after-limit") is False

        self.limits.reset()

        assert self.limits.report_bug_count == 0
        assert self.limits.try_count_report_bug("fresh-call") is True


class TestReportBugRateLimitViaToolCallTracker:
    """US-088: verify the same limit flows through ToolCallTracker
    (the production entry point used by tool_guard.py).
    """

    def setup_method(self):
        ToolCallTracker.reset_instance(SESS)
        self.tracker = ToolCallTracker.get_instance(SESS)

    def teardown_method(self):
        ToolCallTracker.reset_instance(SESS)

    def test_tracker_rejects_sixth_call(self):
        """ToolCallTracker.try_count_report_bug rejects the 6th call."""
        for i in range(MAX_REPORT_BUG_CALLS):
            assert self.tracker.try_count_report_bug(f"call-{i}") is True
        assert self.tracker.try_count_report_bug("call-6") is False

    def test_tracker_count_property_reflects_state(self):
        """tracker.report_bug_count reads the underlying ToolLimits count."""
        for i in range(3):
            self.tracker.try_count_report_bug(f"call-{i}")
        assert self.tracker.report_bug_count == 3
