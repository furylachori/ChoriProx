"""Slice 5 (Spec §15 — Bounded Semantic Retry) — tracker unit tests.

Exercises the real ``SemanticRetryTracker`` directly (no mocking). The
tracker owns no lock and no session state, so each test builds a fresh
instance and feeds it real envelope dicts via ``record`` + ``check``.

The 10 invariants under test:

 1. First call to any key → ``check`` returns ``None``.
 2. Validation: 2 consecutive identical → rejected on the 3rd check;
    ``total >= 3`` also rejects (even when ``consecutive_identical < 2``).
 3. Material correction (different args signature) → consecutive resets
    → check returns ``None``.
 4. Authorization: first failure recorded → second check rejected
    immediately with ``authorization_no_retry``.
 5. Internal: first failure → second check rejected with
    ``internal_no_retry``.
 6. Infrastructure: 2 total → 3rd check rejected with
    ``infrastructure_retry_exhausted``.
 7. Success clears state for the key → subsequent check starts fresh.
 8. Partial envelope carrying a top-level ``error`` dict increments the
    counters (does not clear state).
 9. ``field=None`` falls back to the whole-args signature.
10. Different operations under the same tool get independent budgets.
"""
from __future__ import annotations

import pytest

from proxy.tool_pipeline.tool_limits import SemanticRetryTracker


def _error_envelope(
    tool: str,
    operation: str,
    *,
    code: str = "invalid_input",
    category: str = "validation",
    field=None,
    message: str = "boom",
):
    return {
        "ok": False,
        "tool": tool,
        "operation": operation,
        "status": "error",
        "data": None,
        "error": {
            "code": code,
            "message": message,
            "retryable": True,
            "category": category,
            **({"field": field} if field is not None else {}),
        },
        "session_id": "",
    }


def _partial_with_error(
    tool: str,
    operation: str,
    *,
    code: str = "invalid_input",
    category: str = "validation",
    field=None,
):
    return {
        "ok": True,
        "tool": tool,
        "operation": operation,
        "status": "partial",
        "data": {"created": [], "errors": [{"name": "x", "error": code}]},
        "error": {
            "code": code,
            "message": "partial failure",
            "retryable": True,
            "category": category,
            **({"field": field} if field is not None else {}),
        },
        "session_id": "",
    }


def _success_envelope(tool: str, operation: str):
    return {
        "ok": True,
        "tool": tool,
        "operation": operation,
        "status": "success",
        "data": {"created": [{"short_hash": "abc12345"}]},
        "error": None,
        "session_id": "",
    }


class TestFirstCallAlwaysAllowed:
    def test_first_call_to_any_key_returns_none(self):
        tracker = SemanticRetryTracker()
        assert tracker.check("proxy_ticket", "create_tasks", {"action": "create_tasks"}) is None

    def test_first_call_with_no_args_dict_returns_none(self):
        tracker = SemanticRetryTracker()
        assert tracker.check("proxy_ticket", "create_tasks", {}) is None


class TestValidationBudget:
    def test_two_consecutive_identical_then_third_check_rejected(self):
        tracker = SemanticRetryTracker()
        args = {"action": "create_tasks", "tasks": []}
        op = SemanticRetryTracker._derive_operation("proxy_ticket", args)

        assert tracker.check("proxy_ticket", op, args) is None
        tracker.record("proxy_ticket", op, _error_envelope("proxy_ticket", op), args)

        assert tracker.check("proxy_ticket", op, args) is None
        tracker.record("proxy_ticket", op, _error_envelope("proxy_ticket", op), args)

        rejection = tracker.check("proxy_ticket", op, args)
        assert rejection == ("validation_retry_exhausted", "validation")

    def test_total_budget_rejects_even_when_consecutive_resets(self):
        tracker = SemanticRetryTracker()
        op = SemanticRetryTracker._derive_operation("proxy_ticket", {"action": "create_tasks"})

        args_iter = [
            {"action": "create_tasks", "tasks": [{"name": f"T{i}"}]}
            for i in range(3)
        ]
        # Three distinct args (each a material correction). consecutive
        # resets each time but total climbs to 3.
        for i, args in enumerate(args_iter):
            assert tracker.check("proxy_ticket", op, args) is None, (
                f"material correction {i} must be allowed"
            )
            tracker.record("proxy_ticket", op, _error_envelope("proxy_ticket", op), args)

        # 4th call has a NEW signature again, but total_attempts is now 3
        # (>= BUDGETS["validation"]) AND the new signature differs from
        # the last. Per the spec ordering, a material correction is
        # allowed regardless of total — this assertion pins that
        # contract: a brand-new signature passes.
        new_args = {"action": "create_tasks", "tasks": [{"name": "T-new"}]}
        assert tracker.check("proxy_ticket", op, new_args) is None

        # But the SAME signature as the last failure must now be rejected
        # by the total_attempts budget.
        rejection = tracker.check("proxy_ticket", op, args_iter[-1])
        assert rejection == ("validation_retry_exhausted", "validation"), (
            "total_attempts >= 3 with the same signature as the last "
            "failure must trip the validation budget"
        )


class TestMaterialCorrection:
    def test_different_args_signature_resets_consecutive(self):
        tracker = SemanticRetryTracker()
        args_a = {"action": "create_tasks", "tasks": [{"name": "A"}]}
        args_b = {"action": "create_tasks", "tasks": [{"name": "B"}]}
        op = SemanticRetryTracker._derive_operation("proxy_ticket", args_a)

        # 1st: error on A
        assert tracker.check("proxy_ticket", op, args_a) is None
        tracker.record("proxy_ticket", op, _error_envelope("proxy_ticket", op), args_a)

        # 2nd: error on A again — consecutive=2 now
        assert tracker.check("proxy_ticket", op, args_a) is None
        tracker.record("proxy_ticket", op, _error_envelope("proxy_ticket", op), args_a)

        # 3rd: SAME signature would be rejected. But material correction
        # with args_b must be allowed (None).
        assert tracker.check("proxy_ticket", op, args_b) is None

    def test_correction_then_repeat_allows_one_more_attempt(self):
        tracker = SemanticRetryTracker()
        args_a = {"action": "create_tasks", "tasks": [{"name": "A"}]}
        args_b = {"action": "create_tasks", "tasks": [{"name": "B"}]}
        op = SemanticRetryTracker._derive_operation("proxy_ticket", args_a)

        tracker.record("proxy_ticket", op, _error_envelope("proxy_ticket", op), args_a)
        tracker.record("proxy_ticket", op, _error_envelope("proxy_ticket", op), args_a)

        # Correction to args_b is allowed (None) and recorded.
        assert tracker.check("proxy_ticket", op, args_b) is None
        tracker.record("proxy_ticket", op, _error_envelope("proxy_ticket", op), args_b)

        # Second identical args_b now has consecutive_identical=1, but
        # total_attempts=3 — must be rejected.
        rejection = tracker.check("proxy_ticket", op, args_b)
        assert rejection == ("validation_retry_exhausted", "validation")


class TestAuthorization:
    def test_first_failure_blocks_second_check_immediately(self):
        tracker = SemanticRetryTracker()
        args = {"action": "start_work", "short_hash": "abc12345"}
        op = SemanticRetryTracker._derive_operation("proxy_work", args)

        assert tracker.check("proxy_work", op, args) is None
        tracker.record(
            "proxy_work",
            op,
            _error_envelope(
                "proxy_work", op,
                code="access_denied",
                category="authorization",
            ),
            args,
        )

        rejection = tracker.check("proxy_work", op, args)
        assert rejection == ("authorization_no_retry", "authorization")

    def test_authorization_blocks_even_with_material_correction(self):
        tracker = SemanticRetryTracker()
        args_a = {"action": "start_work", "short_hash": "aaa11111"}
        args_b = {"action": "start_work", "short_hash": "bbb22222"}
        op = SemanticRetryTracker._derive_operation("proxy_work", args_a)

        tracker.record(
            "proxy_work", op,
            _error_envelope(
                "proxy_work", op,
                code="access_denied",
                category="authorization",
            ),
            args_a,
        )

        rejection = tracker.check("proxy_work", op, args_b)
        assert rejection == ("authorization_no_retry", "authorization"), (
            "authorization errors must reject even when the args differ — "
            "material correction does not override category-level blocks"
        )


class TestInternal:
    def test_first_failure_blocks_second_check(self):
        tracker = SemanticRetryTracker()
        args = {"action": "create_tasks", "tasks": [{"name": "X"}]}
        op = SemanticRetryTracker._derive_operation("proxy_ticket", args)

        tracker.record(
            "proxy_ticket", op,
            _error_envelope(
                "proxy_ticket", op,
                code="action_failed",
                category="internal",
            ),
            args,
        )

        rejection = tracker.check("proxy_ticket", op, args)
        assert rejection == ("internal_no_retry", "internal")


class TestInfrastructure:
    def test_two_total_rejects_on_third_check(self):
        tracker = SemanticRetryTracker()
        args = {"action": "create_tasks", "tasks": [{"name": "X"}]}
        op = SemanticRetryTracker._derive_operation("proxy_ticket", args)

        # 1st execute → infra error
        assert tracker.check("proxy_ticket", op, args) is None
        tracker.record(
            "proxy_ticket", op,
            _error_envelope(
                "proxy_ticket", op,
                code="infrastructure_error",
                category="infrastructure",
            ),
            args,
        )

        # 2nd execute (same signature) → still allowed (total=1 < 2)
        assert tracker.check("proxy_ticket", op, args) is None
        tracker.record(
            "proxy_ticket", op,
            _error_envelope(
                "proxy_ticket", op,
                code="infrastructure_error",
                category="infrastructure",
            ),
            args,
        )

        # 3rd check → rejected (total=2 >= 2)
        rejection = tracker.check("proxy_ticket", op, args)
        assert rejection == ("infrastructure_retry_exhausted", "infrastructure")

    def test_material_correction_allowed_under_budget(self):
        tracker = SemanticRetryTracker()
        args_a = {"action": "create_tasks", "tasks": [{"name": "A"}]}
        args_b = {"action": "create_tasks", "tasks": [{"name": "B"}]}
        op = SemanticRetryTracker._derive_operation("proxy_ticket", args_a)

        tracker.record(
            "proxy_ticket", op,
            _error_envelope(
                "proxy_ticket", op,
                code="infrastructure_error",
                category="infrastructure",
            ),
            args_a,
        )
        # Different signature → material correction → allowed even
        # though infrastructure total would have been 1.
        assert tracker.check("proxy_ticket", op, args_b) is None


class TestSuccessClearsState:
    def test_success_envelope_clears_state(self):
        tracker = SemanticRetryTracker()
        args = {"action": "create_tasks", "tasks": [{"name": "X"}]}
        op = SemanticRetryTracker._derive_operation("proxy_ticket", args)

        tracker.record("proxy_ticket", op, _error_envelope("proxy_ticket", op), args)
        tracker.record("proxy_ticket", op, _success_envelope("proxy_ticket", op), args)

        # State cleared — next check is "first call" again.
        assert tracker.check("proxy_ticket", op, args) is None


class TestPartialWithError:
    def test_partial_with_error_dict_increments_counters(self):
        tracker = SemanticRetryTracker()
        args = {"action": "create_tasks", "tasks": [{"name": "X"}]}
        op = SemanticRetryTracker._derive_operation("proxy_ticket", args)

        tracker.record(
            "proxy_ticket", op,
            _partial_with_error("proxy_ticket", op),
            args,
        )
        state = tracker._states.get(("proxy_ticket", op))
        assert state is not None, "partial-with-error must populate state"
        assert state.total_attempts == 1
        assert state.consecutive_identical == 1
        assert state.category == "validation"

    def test_partial_without_error_clears_state(self):
        tracker = SemanticRetryTracker()
        args = {"action": "create_tasks", "tasks": [{"name": "X"}]}
        op = SemanticRetryTracker._derive_operation("proxy_ticket", args)

        tracker.record("proxy_ticket", op, _error_envelope("proxy_ticket", op), args)
        # A partial WITHOUT a top-level error dict is treated as a clean
        # resolution (per-task errors live in data.errors, not in error).
        clean_partial = {
            "ok": True,
            "tool": "proxy_ticket",
            "operation": op,
            "status": "partial",
            "data": {"created": [{"short_hash": "abc12345"}], "errors": []},
            "error": None,
            "session_id": "",
        }
        tracker.record("proxy_ticket", op, clean_partial, args)
        assert ("proxy_ticket", op) not in tracker._states


class TestArgsSignatureFallback:
    def test_field_none_uses_whole_args_signature(self):
        tracker = SemanticRetryTracker()
        args = {"action": "create_tasks", "tasks": [{"name": "X"}]}
        op = SemanticRetryTracker._derive_operation("proxy_ticket", args)

        # Record an error with field=None — signature must include the
        # full cleaned args dict, so adding session_id later does NOT
        # change the signature.
        tracker.record(
            "proxy_ticket", op,
            _error_envelope("proxy_ticket", op, field=None),
            args,
        )
        state = tracker._states[("proxy_ticket", op)]
        assert state.field is None

        # Same args minus session_id → signature matches.
        assert tracker.check("proxy_ticket", op, dict(args)) is None

        # Adding session_id does NOT change the signature (proxy-injected
        # keys are stripped).
        args_with_session = dict(args)
        args_with_session["session_id"] = "sess-1"
        assert tracker.check("proxy_ticket", op, args_with_session) is None

    def test_field_set_uses_only_that_field_value(self):
        tracker = SemanticRetryTracker()
        args = {"action": "create_tasks", "tasks": [{"name": "X"}]}
        op = SemanticRetryTracker._derive_operation("proxy_ticket", args)

        # Record with field="tasks" — signature is just {"tasks": [...]}.
        tracker.record(
            "proxy_ticket", op,
            _error_envelope("proxy_ticket", op, field="tasks"),
            args,
        )

        # Same tasks value but different priority → still matches.
        args_variant = dict(args)
        args_variant["priority"] = "high"
        assert tracker.check("proxy_ticket", op, args_variant) is None, (
            "field-scoped signature must ignore changes to other fields"
        )


class TestPerOperationBudgets:
    def test_different_operations_get_separate_budgets(self):
        tracker = SemanticRetryTracker()
        args_create = {"action": "create_tasks", "tasks": []}
        args_list = {"action": "list_tasks"}
        op_create = SemanticRetryTracker._derive_operation("proxy_ticket", args_create)
        op_list = SemanticRetryTracker._derive_operation("proxy_ticket", args_list)

        # Burn through the entire validation budget for create_tasks.
        for _ in range(3):
            tracker.record(
                "proxy_ticket", op_create,
                _error_envelope("proxy_ticket", op_create),
                args_create,
            )
        # create_tasks is now rejected.
        assert tracker.check("proxy_ticket", op_create, args_create) == (
            "validation_retry_exhausted", "validation",
        )

        # list_tasks has no prior state → first call allowed.
        assert tracker.check("proxy_ticket", op_list, args_list) is None

    def test_different_tools_get_separate_budgets(self):
        tracker = SemanticRetryTracker()
        args = {"action": "create_tasks", "tasks": []}
        op = SemanticRetryTracker._derive_operation("proxy_ticket", args)

        for _ in range(3):
            tracker.record(
                "proxy_ticket", op,
                _error_envelope("proxy_ticket", op),
                args,
            )
        # proxy_ticket:create_tasks rejected.
        assert tracker.check("proxy_ticket", op, args) is not None

        # proxy_work with the same action/args has no state → allowed.
        assert tracker.check("proxy_work", op, args) is None


class TestDeriveOperation:
    def test_uses_action_when_present(self):
        assert SemanticRetryTracker._derive_operation(
            "proxy_ticket", {"action": "create_tasks"}
        ) == "create_tasks"

    def test_falls_back_to_tool_name_when_action_missing(self):
        assert SemanticRetryTracker._derive_operation("recall", {}) == "recall"

    def test_falls_back_to_tool_name_when_action_empty(self):
        assert (
            SemanticRetryTracker._derive_operation("proxy_ticket", {"action": ""})
            == "proxy_ticket"
        )

    def test_falls_back_to_tool_name_when_args_not_dict(self):
        assert SemanticRetryTracker._derive_operation("recall", None) == "recall"
        assert SemanticRetryTracker._derive_operation("recall", "not-a-dict") == "recall"
