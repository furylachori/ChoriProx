"""M4 #3 and M4 #4: per-response cap on the canary escalation rate limiter.

Both fixes touch the same primitive (state.escalation_rate_limiter) and
live in ``src/proxy/canary.py``. They are SECURELY RELEVANT: the 5/hr
budget caps the blast radius of prompt-injection that forces repeated
ticket->work escalations. A regression in either fix would let an
attacker burn the entire budget in a single response (M4 #4) or per
failed escalation (M4 #3).

M4 #3: ``_resolve_task_canary`` used to call
``state.escalation_rate_limiter.check()`` UNCONDITIONALLY at the gate,
which CONSUMED the slot before ``_build_work_canary_for_request`` had
a chance to run. If the build returned None (e.g., no parent state to
inherit), the slot was consumed but the default canary was used -- a
free 5/hr budget leak per failed escalation. Fix: use ``check_only()``
at the gate, commit (``check()``) only after the build returns a
non-None canary.

M4 #4: the loop over ``native_tc_list`` in ``_inject_canary_into_task_calls``
called ``_resolve_task_canary`` per tool call, which hit the rate
limiter once per call. A response with 6 task tool calls burned the
entire 5/hr budget in a single response. Fix: hoist the slot commit
out of the loop. ``_resolve_task_canary`` still runs ``check_only()``
per call (so it can fall back gracefully), but the slot commit happens
once at the first matched ``tc`` via the ``escalation_committed`` flag.
"""
import json
import unittest
from collections import deque
from unittest.mock import MagicMock, patch

import proxy.state as state


class TestEscalationSlotNotBurnedWhenBuildFails(unittest.TestCase):
    """M4 #3: the rate-limit slot is committed only after the work canary
    is actually built. If ``_build_work_canary_for_request`` returns
    None, the escalation is denied and the slot stays free."""

    def setUp(self):
        from proxy.state import escalation_rate_limiter
        self._limiter = escalation_rate_limiter
        self._limiter.reset()
        self._orig_max = self._limiter._max
        self._orig_window = self._limiter._window

    def tearDown(self):
        self._limiter.reset()
        self._limiter._max = self._orig_max
        self._limiter._window = self._orig_window

    def _ctx(self, session_id):
        ctx = MagicMock()
        ctx.session_id = session_id
        ctx.agent_id = "agent_m4_3_" + session_id[-8:]
        ctx.request_id = "req_m4_3_" + session_id[-8:]
        ctx.is_subagent = False
        ctx.client_hint = "test"
        return ctx

    def test_build_failure_does_not_consume_slot(self):
        from proxy.canary import _resolve_task_canary

        session_id = "sess_m4_3_build_fails_42"
        ctx = self._ctx(session_id)
        args = {"subagent_mode": "work"}

        with patch(
            "proxy.canary._build_work_canary_for_request",
            return_value=None,
        ):
            canary, kind = _resolve_task_canary(
                default_canary="DEFAULT_CANARY",
                args=args,
                default_mode_name="ticket",
                request_ctx=ctx,
            )

        self.assertEqual(
            canary, "DEFAULT_CANARY",
            "build failure must fall back to default canary",
        )
        self.assertIsNone(
            kind,
            f"build failure must NOT escalate (kind={kind!r}); "
            "pre-fix code would have set kind='work' AND consumed a slot",
        )
        self.assertEqual(
            state.escalation_rate_limiter._call_count, 0,
            f"slot must NOT be committed when build fails. "
            f"_call_count={state.escalation_rate_limiter._call_count} (expected 0). "
            f"Pre-fix code committed the slot via check() BEFORE calling _build_work_canary_for_request.",
        )
        self.assertEqual(
            len(state.escalation_rate_limiter._entries.get(session_id, [])), 0,
            f"deque for session must be empty. Pre-fix would have 1 entry.",
        )

    def test_build_failure_then_successful_escalation_still_has_budget(self):
        """The post-fix invariant: a failed escalation does NOT consume the
        5/hr budget, so a subsequent successful escalation can still
        proceed. Pre-fix would have burned one slot on the failed
        attempt, leaving only 4 for the rest of the hour."""
        from proxy.canary import _resolve_task_canary

        from proxy.rate_limiter import RateLimiter
        test_limiter = RateLimiter(
            max_requests=5, window_seconds=3600, persist=False,
        )
        with patch("proxy.state.escalation_rate_limiter", test_limiter):
            ctx = self._ctx("sess_m4_3_budget_retention")
            args = {"subagent_mode": "work"}

            with patch(
                "proxy.canary._build_work_canary_for_request",
                return_value=None,
            ):
                for _ in range(3):
                    canary, kind = _resolve_task_canary(
                        default_canary="DEFAULT_CANARY",
                        args=args,
                        default_mode_name="ticket",
                        request_ctx=ctx,
                    )
                    self.assertIsNone(
                        kind,
                        "3 failed escalations must all return kind=None "
                        "(no escalation, no slot burned)",
                    )

            self.assertEqual(
                test_limiter._call_count, 0,
                f"3 failed escalations must commit 0 slots. "
                f"Got _call_count={test_limiter._call_count}. Pre-fix would be 3.",
            )

            with patch(
                "proxy.canary._build_work_canary_for_request",
                return_value="WORK_CANARY",
            ):
                for i in range(5):
                    canary, kind = _resolve_task_canary(
                        default_canary="DEFAULT_CANARY",
                        args=args,
                        default_mode_name="ticket",
                        request_ctx=ctx,
                    )
                    self.assertEqual(
                        canary, "WORK_CANARY",
                        f"escalation {i+1} after 3 failures must still succeed "
                        f"(budget preserved)",
                    )
                    self.assertEqual(
                        kind, "work",
                        f"escalation {i+1} must succeed; pre-fix would have "
                        f"rate-limited this since 3 slots were already burned",
                    )


class TestEscalationCommittedOncePerResponse(unittest.TestCase):
    """M4 #4: a single response with N task tool calls all requesting
    ``subagent_mode='work'`` commits ONE slot total, not N. The
    ``escalation_committed`` flag in ``_inject_canary_into_task_calls``
    hoists the commit out of the per-tool-call loop."""

    def setUp(self):
        from proxy.state import escalation_rate_limiter
        self._limiter = escalation_rate_limiter
        self._limiter.reset()
        self._orig_max = self._limiter._max
        self._orig_window = self._limiter._window

    def tearDown(self):
        self._limiter.reset()
        self._limiter._max = self._orig_max
        self._limiter._window = self._orig_window

    def _ctx(self, session_id):
        ctx = MagicMock()
        ctx.session_id = session_id
        ctx.agent_id = "agent_m4_4_" + session_id[-8:]
        ctx.request_id = "req_m4_4_" + session_id[-8:]
        ctx.is_subagent = False
        ctx.client_hint = "test"
        return ctx

    def _response_with_n_task_calls(self, n: int, subagent_mode: str) -> dict:
        return {
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": f"call_{i}",
                            "type": "function",
                            "function": {
                                "name": "task",
                                "arguments": json.dumps({
                                    "prompt": f"do thing {i}",
                                    "subagent_mode": subagent_mode,
                                }),
                            },
                        }
                        for i in range(n)
                    ],
                },
            }],
        }

    def test_six_task_tool_calls_commit_one_slot(self):
        from proxy.canary import _inject_canary_into_task_calls

        from proxy.rate_limiter import RateLimiter
        test_limiter = RateLimiter(
            max_requests=5, window_seconds=3600, persist=False,
        )
        with patch("proxy.state.escalation_rate_limiter", test_limiter):
            session_id = "sess_m4_4_per_response_42"
            ctx = self._ctx(session_id)
            data = self._response_with_n_task_calls(6, subagent_mode="work")

            with patch(
                "proxy.canary._build_work_canary_for_request",
                return_value="WORK_CANARY",
            ):
                result = _inject_canary_into_task_calls(
                    data, "DEFAULT_CANARY",
                    request_ctx=ctx,
                    default_mode_name="ticket",
                )

        committed = len(test_limiter._entries.get(session_id, []))
        self.assertEqual(
            committed, 1,
            f"6 task tool calls in one response must commit EXACTLY 1 slot "
            f"(per-response cap). Got committed={committed}. "
            f"Pre-fix would have committed 6, burning the entire 5/hr "
            f"budget in a single response. "
            f"_call_count={test_limiter._call_count} (informational).",
        )

    def test_response_with_mixed_modes_commits_at_most_one_slot(self):
        """A response with 6 task calls (4 with subagent_mode='work', 2
        with subagent_mode='ticket') still commits only 1 slot -- the
        'work' escalations collapse into one response-level event. The
        'ticket' calls are no-ops for the rate limiter."""
        from proxy.canary import _inject_canary_into_task_calls

        from proxy.rate_limiter import RateLimiter
        test_limiter = RateLimiter(
            max_requests=5, window_seconds=3600, persist=False,
        )
        with patch("proxy.state.escalation_rate_limiter", test_limiter):
            session_id = "sess_m4_4_mixed_42"
            ctx = self._ctx(session_id)
            data = {
                "choices": [{
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": f"call_{i}",
                                "type": "function",
                                "function": {
                                    "name": "task",
                                    "arguments": json.dumps({
                                        "prompt": f"do {i}",
                                        "subagent_mode": (
                                            "work" if i < 4 else "ticket"
                                        ),
                                    }),
                                },
                            }
                            for i in range(6)
                        ],
                    },
                }],
            }

            with patch(
                "proxy.canary._build_work_canary_for_request",
                return_value="WORK_CANARY",
            ):
                result = _inject_canary_into_task_calls(
                    data, "DEFAULT_CANARY",
                    request_ctx=ctx,
                    default_mode_name="ticket",
                )

        committed = len(test_limiter._entries.get(session_id, []))
        self.assertEqual(
            committed, 1,
            f"4 work + 2 ticket task calls in one response must commit 1 "
            f"slot (work collapses to one response-level escalation). "
            f"Got committed={committed}.",
        )


        committed = len(test_limiter._entries.get(session_id, []))
        self.assertEqual(
            committed, 1,
            f"4 work + 2 ticket task calls in one response must commit 1 "
            f"slot (work collapses to one response-level escalation). "
            f"Got committed={committed}.",
        )


class TestEscalationDeniedWhenLimitExceeded(unittest.TestCase):
    """BATCH 2.2 (M4-3 followup): when the per-session escalation
    budget is exhausted, ``_resolve_task_canary`` must fall back to
    the DEFAULT canary (not silently grant the WORK canary). The
    rate-limit gate lives IN the function via a real ``check()``
    call, so direct callers AND the production
    ``_inject_canary_into_task_calls`` path both honor the 5/hr cap
    on the 6th concurrent (or back-to-back) request hitting the
    cap. Pre-fix code had the gate in the outer loop and the
    function used a ``check_only()`` peek — bypassable for direct
    callers."""

    def setUp(self):
        from proxy.state import escalation_rate_limiter
        self._limiter = escalation_rate_limiter
        self._limiter.reset()
        self._orig_max = self._limiter._max
        self._orig_window = self._limiter._window

    def tearDown(self):
        self._limiter.reset()
        self._limiter._max = self._orig_max
        self._limiter._window = self._orig_window

    def _ctx(self, session_id):
        ctx = MagicMock()
        ctx.session_id = session_id
        ctx.agent_id = "agent_b22_" + session_id[-8:]
        ctx.request_id = "req_b22_" + session_id[-8:]
        ctx.is_subagent = False
        ctx.client_hint = "test"
        return ctx

    def _response_with_task_call(self, subagent_mode: str) -> dict:
        return {
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [{
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "task",
                            "arguments": json.dumps({
                                "prompt": "do thing",
                                "subagent_mode": subagent_mode,
                            }),
                        },
                    }],
                },
            }],
        }

    def test_work_canary_denied_when_budget_exhausted(self):
        """When the limiter is at cap, the response uses the DEFAULT
        canary, not the WORK canary.

        The gate now lives in ``_resolve_task_canary`` itself: it does
        a real ``check()`` AFTER a successful canary build, and if the
        slot is denied, returns ``("DEFAULT_CANARY", "rate_limited")``.
        No need to mock the function — the real one will deny at cap.
        Pre-fix code (the M4-3 followup that called ``check()`` in
        the outer loop) would have used WORK_CANARY if the function
        were mocked to bypass the gate; the new design is not
        bypassable for direct callers.
        """
        from proxy.canary import _inject_canary_into_task_calls
        from proxy.rate_limiter import RateLimiter
        import time as _time

        test_limiter = RateLimiter(
            max_requests=5, window_seconds=3600, persist=False,
        )
        session_id = "sess_b22_cap_42"
        # Pre-fill the limiter to its 5/hr cap so the next check() is denied.
        with test_limiter._lock:
            now = _time.time()
            test_limiter._entries[session_id] = deque([now, now, now, now, now])
        with patch("proxy.state.escalation_rate_limiter", test_limiter):
            ctx = self._ctx(session_id)
            data = self._response_with_task_call(subagent_mode="work")

            result = _inject_canary_into_task_calls(
                data, "DEFAULT_CANARY",
                request_ctx=ctx,
                default_mode_name="ticket",
            )

        args = json.loads(result["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"])
        self.assertFalse(
            args["prompt"].startswith("WORK_CANARY"),
            f"When budget is exhausted, the prompt must NOT use the "
            f"WORK canary. Got prompt[:80]={args['prompt'][:80]!r}. "
            f"Pre-fix code would have used WORK_CANARY, bypassing "
            f"the 5/hr cap on the 6th request.",
        )
        self.assertTrue(
            args["prompt"].startswith("DEFAULT_CANARY"),
            f"When budget is exhausted, the prompt must use the "
            f"DEFAULT canary. Got prompt[:80]={args['prompt'][:80]!r}.",
        )

    def test_no_commit_when_budget_exhausted(self):
        """When the limiter is at cap, no slot is committed. The
        limiter rejects the request (no new entry).

        The gate is ``_resolve_task_canary``'s own ``check()``: when
        the limiter is at cap, ``check()`` returns ``not allowed``
        WITHOUT appending. So a 6th call hits the cap and the deque
        stays at 5 entries. Pre-fix code (with the gate in the outer
        loop and the function peeking) would have either appended a
        6th entry (the original bypass bug) or required a separate
        ``check_only``-then-``check`` sequence that was bypassable
        for direct callers."""
        from proxy.canary import _inject_canary_into_task_calls
        from proxy.rate_limiter import RateLimiter
        import time as _time

        test_limiter = RateLimiter(
            max_requests=5, window_seconds=3600, persist=False,
        )
        session_id = "sess_b22_no_commit"
        # Pre-fill to cap.
        with test_limiter._lock:
            now = _time.time()
            test_limiter._entries[session_id] = deque([now, now, now, now, now])
        with patch("proxy.state.escalation_rate_limiter", test_limiter):
            ctx = self._ctx(session_id)
            data = self._response_with_task_call(subagent_mode="work")

            _inject_canary_into_task_calls(
                data, "DEFAULT_CANARY",
                request_ctx=ctx,
                default_mode_name="ticket",
            )

        self.assertEqual(
            len(test_limiter._entries.get(session_id, [])), 5,
            f"Denial must NOT consume a slot. Got "
            f"len={len(test_limiter._entries.get(session_id, []))}. "
            f"Pre-fix code would have appended a 6th entry (bypassing the cap).",
        )


if __name__ == "__main__":
    unittest.main()
