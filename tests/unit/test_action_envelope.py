"""Pure-shape unit tests for the canonical action envelope + error taxonomy.

These tests do NOT exercise any action handler or the dispatcher. They pin
the shape of the three envelope builders and the taxonomy lookup so that
Cluster 1 (handler migration) and Cluster 0 (dispatcher cutover) can rely on
the contract.
"""
from __future__ import annotations

from proxy.tool_pipeline.action_envelope import (
    error_envelope,
    partial_envelope,
    success_envelope,
)
from proxy.tool_pipeline.error_taxonomy import (
    CATEGORY_AUTHORIZATION,
    CATEGORY_INFRASTRUCTURE,
    CATEGORY_INTERNAL,
    CATEGORY_VALIDATION,
    classify_error,
)


# ---------------------------------------------------------------------------
# success_envelope
# ---------------------------------------------------------------------------

def test_success_envelope_has_exactly_seven_keys():
    env = success_envelope("proxy_ticket", "create_tasks", {"created": []}, "sess-1")
    assert set(env.keys()) == {
        "ok",
        "tool",
        "operation",
        "status",
        "data",
        "error",
        "session_id",
    }


def test_success_envelope_shape():
    env = success_envelope("proxy_ticket", "create_tasks", {"created": []}, "sess-1")
    assert env["ok"] is True
    assert env["tool"] == "proxy_ticket"
    assert env["operation"] == "create_tasks"
    assert env["status"] == "success"
    assert env["data"] == {"created": []}
    assert env["error"] is None
    assert env["session_id"] == "sess-1"
    # No top-level valid_actions leak.
    assert "valid_actions" not in env


# ---------------------------------------------------------------------------
# partial_envelope
# ---------------------------------------------------------------------------

def test_partial_envelope_shape():
    payload = {
        "created": [{"short_hash": "abc123"}],
        "errors": [{"name": "bad", "reason": "duplicate"}],
        "wiring_errors": [],
        "total_created": 1,
        "total_errors": 1,
    }
    env = partial_envelope("proxy_ticket", "create_tasks", payload, "sess-1")
    assert env["ok"] is True
    assert env["status"] == "partial"
    assert env["data"] is payload
    assert env["error"] is None
    assert env["session_id"] == "sess-1"


def test_partial_envelope_has_exactly_seven_keys():
    env = partial_envelope("proxy_ticket", "create_tasks", {}, "sess-1")
    assert set(env.keys()) == {
        "ok",
        "tool",
        "operation",
        "status",
        "data",
        "error",
        "session_id",
    }


# ---------------------------------------------------------------------------
# error_envelope
# ---------------------------------------------------------------------------

def test_error_envelope_basic_shape():
    env = error_envelope(
        "proxy_ticket",
        "create_tasks",
        "invalid_input",
        "create_tasks requires a non-empty tasks array",
        "sess-1",
        field="tasks",
    )
    assert env["ok"] is False
    assert env["tool"] == "proxy_ticket"
    assert env["operation"] == "create_tasks"
    assert env["status"] == "error"
    assert env["data"] is None
    assert env["session_id"] == "sess-1"
    assert env["error"] == {
        "code": "invalid_input",
        "message": "create_tasks requires a non-empty tasks array",
        "field": "tasks",
        "retryable": True,
        "category": "validation",
    }


def test_error_envelope_operation_can_be_none():
    env = error_envelope(
        "proxy_ticket",
        None,
        "invalid_action",
        "Unknown action: 'foo'",
        "sess-1",
    )
    assert env["operation"] is None
    assert env["error"]["code"] == "invalid_action"


def test_error_envelope_field_included_only_when_passed():
    with_field = error_envelope(
        "proxy_ticket", "create_tasks", "invalid_input", "msg", "sess-1", field="tasks",
    )
    assert with_field["error"]["field"] == "tasks"

    without_field = error_envelope(
        "proxy_ticket", "create_tasks", "invalid_input", "msg", "sess-1",
    )
    assert "field" not in without_field["error"]


def test_error_envelope_valid_actions_included_only_when_passed():
    with_va = error_envelope(
        "proxy_ticket",
        None,
        "invalid_action",
        "Unknown action: 'foo'",
        "sess-1",
        valid_actions=["create_tasks", "list_tasks"],
    )
    assert with_va["valid_actions"] == ["create_tasks", "list_tasks"]

    without_va = error_envelope(
        "proxy_ticket",
        None,
        "invalid_action",
        "Unknown action: 'foo'",
        "sess-1",
    )
    assert "valid_actions" not in without_va


def test_error_envelope_seven_keys_without_valid_actions():
    env = error_envelope("proxy_ticket", "create_tasks", "not_found", "msg", "sess-1")
    assert set(env.keys()) == {
        "ok",
        "tool",
        "operation",
        "status",
        "data",
        "error",
        "session_id",
    }


def test_error_envelope_eight_keys_with_valid_actions():
    env = error_envelope(
        "proxy_ticket", None, "invalid_action", "msg", "sess-1",
        valid_actions=["create_tasks"],
    )
    assert set(env.keys()) == {
        "ok",
        "tool",
        "operation",
        "status",
        "data",
        "error",
        "session_id",
        "valid_actions",
    }


def test_error_envelope_error_dict_has_required_keys():
    env = error_envelope("proxy_ticket", "create_tasks", "not_found", "msg", "sess-1")
    error = env["error"]
    assert set(error.keys()) == {"code", "message", "retryable", "category"}


# ---------------------------------------------------------------------------
# session_id coercion
# ---------------------------------------------------------------------------

def test_success_envelope_coerces_none_session_id():
    env = success_envelope("proxy_ticket", "list_tasks", {}, None)
    assert env["session_id"] == ""


def test_partial_envelope_coerces_none_session_id():
    env = partial_envelope("proxy_ticket", "create_tasks", {}, None)
    assert env["session_id"] == ""


def test_error_envelope_coerces_none_session_id():
    env = error_envelope("proxy_ticket", "create_tasks", "not_found", "msg", None)
    assert env["session_id"] == ""


# ---------------------------------------------------------------------------
# classify_error / taxonomy
# ---------------------------------------------------------------------------

def test_classify_error_unknown_code_returns_internal_not_retryable():
    assert classify_error("nonexistent_code") == (CATEGORY_INTERNAL, False)


def test_classify_error_each_category_has_a_resolving_code():
    # Validation
    assert classify_error("invalid_input") == (CATEGORY_VALIDATION, True)
    # Authorization
    assert classify_error("access_denied") == (CATEGORY_AUTHORIZATION, False)
    # Infrastructure
    assert classify_error("infrastructure_error") == (CATEGORY_INFRASTRUCTURE, True)
    # Internal
    assert classify_error("action_failed") == (CATEGORY_INTERNAL, False)


def test_classify_error_infrastructure_codes_are_retryable():
    for code in ("infrastructure_error", "query_timeout", "constraint_violation"):
        category, retryable = classify_error(code)
        assert category == CATEGORY_INFRASTRUCTURE
        assert retryable is True


def test_classify_error_authorization_codes_are_not_retryable():
    for code in ("access_denied", "mode_inactive", "not_authorized"):
        category, retryable = classify_error(code)
        assert category == CATEGORY_AUTHORIZATION
        assert retryable is False


def test_classify_error_validation_codes_are_retryable():
    for code in ("invalid_input", "ambiguous_hash", "not_found", "duplicate_name"):
        category, retryable = classify_error(code)
        assert category == CATEGORY_VALIDATION
        assert retryable is True


def test_classify_error_internal_codes_are_not_retryable():
    for code in ("action_failed", "tool_execution_failed", "unknown_memory_tool"):
        category, retryable = classify_error(code)
        assert category == CATEGORY_INTERNAL
        assert retryable is False
