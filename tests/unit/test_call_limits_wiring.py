"""Tests for SkillDefinition.call_limits wiring into MemoryToolGuard.

Verifies that per-skill `call_limits` dicts (declared on SkillDefinition)
are honored by `_check_call_limits` for `proxy_ticket`, `proxy_work`, and
`task` tools, and that fall-back to global defaults happens when no skill
is active or the skill omits a key.

Covers commit 04514a9 (proxy_ticket + task) and the proxy_work wiring
follow-up (previously dead `{"proxy_work": 50}` key on WorkSkill).
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from proxy.tool_pipeline.tool_call import ToolCall
from proxy.tool_pipeline.tool_call_tracker import ToolCallTracker
from proxy.tool_pipeline.tool_limits import (
    MAX_PROXY_TICKET_CALLS,
    MAX_PROXY_WORK_CALLS,
    MAX_SUBAGENTS,
)
from proxy.tool_pipeline.tool_guard import MemoryToolGuard


SESS = "sess-call-limits-1"


def _make_mode_state(name: str):
    ms = MagicMock()
    ms.name = name
    return ms


def _make_agent(mode_name: str | None):
    a = MagicMock()
    if mode_name is None:
        a.snapshot_mode.return_value = None
    else:
        a.snapshot_mode.return_value = _make_mode_state(mode_name)
    return a


def _make_skill_def(call_limits: dict):
    sd = MagicMock()
    sd.call_limits = call_limits
    return sd


def _make_ctx(session_id: str = SESS, is_subagent: bool = False):
    ctx = MagicMock()
    ctx.session_id = session_id
    ctx.is_subagent = is_subagent
    ctx.subagent_ticket_mode = False
    return ctx


def _patch_skill(call_limits: dict, mode_name: str = "work"):
    """Patch agent_registry + skill registry to return a skill with call_limits."""
    return (
        patch(
            "proxy.agent.agent_registry.registry.get_or_load",
            return_value=_make_agent(mode_name),
        ),
        patch(
            "proxy.skills.skill_registry.get_default_registry",
            return_value=MagicMock(get_skill=MagicMock(return_value=_make_skill_def(call_limits))),
        ),
    )


class TestCallLimitsProxyTicketFromSkill:
    """proxy_ticket call limit is enforced from the active skill's call_limits."""

    def setup_method(self):
        ToolCallTracker.reset_instance(SESS)
        self.guard = MemoryToolGuard()

    def teardown_method(self):
        ToolCallTracker.reset_instance(SESS)

    def test_call_limits_proxy_ticket_enforced_from_skill(self):
        """Skill with call_limits={'proxy_ticket': 3} -> 4th call blocked at 3/3."""
        p1, p2 = _patch_skill({"proxy_ticket": 3}, mode_name="ticket")
        with p1, p2:
            for i in range(3):
                tc = ToolCall(id=f"pt-{i}", name="proxy_ticket", arguments={"action": "list_tasks"})
                err = self.guard._check_call_limits(tc, _make_ctx())
                assert err is None, f"call {i+1} should pass, got: {err}"
            tc = ToolCall(id="pt-3", name="proxy_ticket", arguments={"action": "list_tasks"})
            err = self.guard._check_call_limits(tc, _make_ctx())
            assert err is not None
            assert "proxy_ticket call limit reached (3/3)" in err


class TestCallLimitsTaskFromSkill:
    """task spawn limit is enforced from the active skill's call_limits."""

    def setup_method(self):
        ToolCallTracker.reset_instance(SESS)
        self.guard = MemoryToolGuard()

    def teardown_method(self):
        ToolCallTracker.reset_instance(SESS)

    def test_call_limits_task_enforced_from_skill(self):
        """WorkSkill call_limits={'task': 3} -> 4th spawn blocked. Limit=3, not MAX_SUBAGENTS."""
        p1, p2 = _patch_skill({"task": 3}, mode_name="work")
        with p1, p2:
            for i in range(3):
                tc = ToolCall(id=f"t-{i}", name="task", arguments={})
                err = self.guard._check_call_limits(tc, _make_ctx())
                assert err is None, f"spawn {i+1} should pass, got: {err}"
            tc = ToolCall(id="t-3", name="task", arguments={})
            err = self.guard._check_call_limits(tc, _make_ctx())
            assert err is not None
            assert "Sub-agent limit reached (3/3)" in err
            # Confirm the limit came from the skill (3), not the global MAX_SUBAGENTS.
            assert str(MAX_SUBAGENTS) != "3" or "3/3" in err


class TestCallLimitsProxyWorkFromSkill:
    """proxy_work call limit is enforced from the active skill's call_limits."""

    def setup_method(self):
        ToolCallTracker.reset_instance(SESS)
        self.guard = MemoryToolGuard()

    def teardown_method(self):
        ToolCallTracker.reset_instance(SESS)

    def test_call_limits_proxy_work_enforced_from_skill(self):
        """WorkSkill call_limits={'proxy_work': 5} -> 6th call blocked at 5/5."""
        p1, p2 = _patch_skill({"proxy_work": 5}, mode_name="work")
        with p1, p2:
            for i in range(5):
                tc = ToolCall(id=f"pw-{i}", name="proxy_work", arguments={"action": "start_work", "short_hash": "abc12345"})
                err = self.guard._check_call_limits(tc, _make_ctx())
                assert err is None, f"call {i+1} should pass, got: {err}"
            tc = ToolCall(id="pw-5", name="proxy_work", arguments={"action": "start_work", "short_hash": "abc12345"})
            err = self.guard._check_call_limits(tc, _make_ctx())
            assert err is not None
            assert "proxy_work call limit reached (5/5)" in err

    def test_call_limits_proxy_work_total_runs_before_report_bug(self):
        """Total proxy_work counter runs before the per-action report_bug check."""
        # Set proxy_work limit to 2; report_bug is a separate sub-limit (MAX_REPORT_BUG_CALLS=5).
        # After 2 proxy_work calls, the 3rd (even report_bug) is blocked by the TOTAL counter.
        p1, p2 = _patch_skill({"proxy_work": 2}, mode_name="work")
        with p1, p2:
            for i in range(2):
                tc = ToolCall(id=f"pwrb-{i}", name="proxy_work", arguments={"action": "report_bug", "task_short_hash": "abc12345"})
                err = self.guard._check_call_limits(tc, _make_ctx())
                assert err is None, f"call {i+1} should pass, got: {err}"
            tc = ToolCall(id="pwrb-2", name="proxy_work", arguments={"action": "report_bug", "task_short_hash": "abc12345"})
            err = self.guard._check_call_limits(tc, _make_ctx())
            assert err is not None
            # Total counter fires first
            assert "proxy_work call limit reached (2/2)" in err


class TestCallLimitsFallback:
    """Fall back to global defaults when skill omits a key or no skill active."""

    def setup_method(self):
        ToolCallTracker.reset_instance(SESS)
        self.guard = MemoryToolGuard()

    def teardown_method(self):
        ToolCallTracker.reset_instance(SESS)

    def test_call_limits_falls_back_to_global_when_skill_has_no_key(self):
        """Skill with call_limits={} -> proxy_ticket uses MAX_PROXY_TICKET_CALLS."""
        p1, p2 = _patch_skill({}, mode_name="ticket")
        with p1, p2:
            # Exhaust the global limit
            for i in range(MAX_PROXY_TICKET_CALLS):
                tc = ToolCall(id=f"ptg-{i}", name="proxy_ticket", arguments={"action": "list_tasks"})
                err = self.guard._check_call_limits(tc, _make_ctx())
                assert err is None, f"call {i+1} should pass, got: {err}"
            # Next call blocked with the GLOBAL max in the message
            tc = ToolCall(id="ptg-block", name="proxy_ticket", arguments={"action": "list_tasks"})
            err = self.guard._check_call_limits(tc, _make_ctx())
            assert err is not None
            assert f"proxy_ticket call limit reached ({MAX_PROXY_TICKET_CALLS}/{MAX_PROXY_TICKET_CALLS})" in err

    def test_call_limits_falls_back_to_global_when_no_skill_active(self):
        """No active mode (DEFAULT) -> proxy_work uses MAX_PROXY_WORK_CALLS."""
        with patch(
            "proxy.agent.agent_registry.registry.get_or_load",
            return_value=_make_agent(None),
        ), patch(
            "proxy.skills.skill_registry.get_default_registry",
            return_value=MagicMock(get_skill=MagicMock(return_value=None)),
        ):
            for i in range(MAX_PROXY_WORK_CALLS):
                tc = ToolCall(id=f"pwg-{i}", name="proxy_work", arguments={"action": "start_work", "short_hash": "abc12345"})
                err = self.guard._check_call_limits(tc, _make_ctx())
                assert err is None, f"call {i+1} should pass, got: {err}"
            tc = ToolCall(id="pwg-block", name="proxy_work", arguments={"action": "start_work", "short_hash": "abc12345"})
            err = self.guard._check_call_limits(tc, _make_ctx())
            assert err is not None
            assert f"proxy_work call limit reached ({MAX_PROXY_WORK_CALLS}/{MAX_PROXY_WORK_CALLS})" in err


class TestCallLimitsMaxZero:
    """A call_limits value of 0 blocks ALL calls (zero allowed)."""

    def setup_method(self):
        ToolCallTracker.reset_instance(SESS)
        self.guard = MemoryToolGuard()

    def teardown_method(self):
        ToolCallTracker.reset_instance(SESS)

    def test_call_limits_max_limit_zero_blocks_all(self):
        """call_limits={'proxy_ticket': 0} -> first call blocked (0 allowed)."""
        p1, p2 = _patch_skill({"proxy_ticket": 0}, mode_name="ticket")
        with p1, p2:
            tc = ToolCall(id="pt-zero", name="proxy_ticket", arguments={"action": "list_tasks"})
            err = self.guard._check_call_limits(tc, _make_ctx())
            assert err is not None
            assert "proxy_ticket call limit reached (0/0)" in err


class TestTryCountProxyWorkIdempotent:
    """try_count_proxy_work is idempotent by call_id (same as proxy_ticket)."""

    def setup_method(self):
        ToolCallTracker.reset_instance(SESS)
        self.tracker = ToolCallTracker.get_instance(SESS)

    def teardown_method(self):
        ToolCallTracker.reset_instance(SESS)

    def test_try_count_proxy_work_idempotent_by_call_id(self):
        """Calling try_count_proxy_work('call-1') twice counts once."""
        assert self.tracker.try_count_proxy_work("call-1") is True
        assert self.tracker.try_count_proxy_work("call-1") is True  # idempotent
        assert self.tracker.proxy_work_count == 1
        # A different call_id counts again
        assert self.tracker.try_count_proxy_work("call-2") is True
        assert self.tracker.proxy_work_count == 2

    def test_try_count_proxy_work_respects_max_limit(self):
        """max_limit=2 -> 3rd distinct call_id blocked."""
        assert self.tracker.try_count_proxy_work("a", max_limit=2) is True
        assert self.tracker.try_count_proxy_work("b", max_limit=2) is True
        assert self.tracker.try_count_proxy_work("c", max_limit=2) is False

    def test_status_summary_includes_proxy_work(self):
        """status_summary() includes proxy_work_calls and proxy_work_max."""
        self.tracker.try_count_proxy_work("x")
        s = self.tracker.status_summary()
        assert s["proxy_work_calls"] == 1
        assert s["proxy_work_max"] == MAX_PROXY_WORK_CALLS