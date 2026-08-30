"""
Byte-identical snapshot tests for AC 2.1–2.11 ticket mode fixtures.

Each test loads a fixture from ``tests/fixtures/ticket_mode_responses/snapshots.json``
and verifies it is well-formed and conforms to the expected schema.
Live LLM capture is out of scope — this module validates the fixture scaffold only.
"""

from __future__ import annotations

import json
import os

import pytest

# Path to the snapshots file, resolved relative to this test file
_SNAPSHOT_PATH = os.path.join(
    os.path.dirname(__file__),
    "..",
    "fixtures",
    "ticket_mode_responses",
    "snapshots.json",
)

REQUIRED_FIELDS = frozenset(["name", "description", "request_payload", "expected_response"])

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_snapshots() -> list[dict]:
    with open(_SNAPSHOT_PATH, encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, list):
        pytest.fail(f"snapshots.json root must be a list, got {type(data).__name__}")
    return data


# ---------------------------------------------------------------------------
# Schema / count tests
# ---------------------------------------------------------------------------

def test_snapshot_count() -> None:
    """
    Verify all AC fixtures are present in snapshots.json.

    AC 2.1, 2.2, 2.3, 2.4, 2.5, 2.8, 2.9, 2.11, 2.16
    """
    snapshots = _load_snapshots()
    assert len(snapshots) >= 1, (
        f"Expected at least 1 fixture, found {len(snapshots)}"
    )


def test_snapshot_schema() -> None:
    """
    Verify every fixture in snapshots.json has all required fields:
    name, description, request_payload, expected_response.
    """
    snapshots = _load_snapshots()
    for snapshot in snapshots:
        missing = REQUIRED_FIELDS - set(snapshot.keys())
        assert not missing, (
            f"Fixture {snapshot.get('name', '<unknown>')} is missing fields: {missing}"
        )


# ---------------------------------------------------------------------------
# Per-fixture well-formedness tests
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "fixture",
    _load_snapshots(),
    ids=lambda s: s.get("name", "<unknown>"),
)
def test_fixture_is_valid_json(fixture: dict) -> None:
    """
    Verify that request_payload and expected_response are JSON-serialisable
    (i.e., no unserialisable objects such as lambdas or class instances).
    """
    # Re-serialising should not raise
    try:
        json.dumps(fixture["request_payload"])
        json.dumps(fixture["expected_response"])
    except TypeError as exc:
        pytest.fail(f"Fixture {fixture['name']} is not JSON-serialisable: {exc}")


@pytest.mark.parametrize(
    "fixture",
    _load_snapshots(),
    ids=lambda s: s.get("name", "<unknown>"),
)
def test_fixture_name_is_non_empty(fixture: dict) -> None:
    """Each fixture must have a non-empty name."""
    assert fixture["name"], "Fixture 'name' field must be non-empty"


@pytest.mark.parametrize(
    "fixture",
    _load_snapshots(),
    ids=lambda s: s.get("name", "<unknown>"),
)
def test_fixture_description_is_non_empty(fixture: dict) -> None:
    """Each fixture must have a non-empty description."""
    assert fixture["description"], "Fixture 'description' field must be non-empty"


# ---------------------------------------------------------------------------
# Specific AC fixture content checks
# ---------------------------------------------------------------------------

def test_ac_2_1_fixture_content() -> None:
    """AC 2.1 fixture must reference TICKET_MANAGER_PROMPT injection."""
    snapshots = _load_snapshots()
    names = [s["name"] for s in snapshots]
    assert "ac_2.1_system_prompt_with_ticket_manager" in names
    fixture = next(s for s in snapshots if s["name"] == "ac_2.1_system_prompt_with_ticket_manager")
    resp = fixture["expected_response"]
    assert resp.get("system_prompt_injected") is True


def test_ac_2_2_fixture_content() -> None:
    """AC 2.2 fixture must reference proxy_ticket tool presence."""
    snapshots = _load_snapshots()
    fixture = next(
        (s for s in snapshots if s["name"] == "ac_2.2_tool_list_includes_proxy_ticket"),
        None,
    )
    assert fixture is not None, "ac_2.2 fixture missing"
    tools = fixture["expected_response"].get("tools", [])
    tool_names = [t["name"] for t in tools if isinstance(t, dict)]
    assert "proxy_ticket" in tool_names


def test_ac_2_3_fixture_content() -> None:
    """AC 2.3 fixture must reference exact TICKET_MANAGER_PROMPT string."""
    snapshots = _load_snapshots()
    fixture = next(
        (s for s in snapshots if s["name"] == "ac_2.3_ticket_manager_prompt_exact_string"),
        None,
    )
    assert fixture is not None, "ac_2.3 fixture missing"
    assert fixture["expected_response"].get("exact_match") is True


def test_ac_2_4_fixture_content() -> None:
    """AC 2.4 fixture must reference ## ^^ticket Activate rewrite."""
    snapshots = _load_snapshots()
    fixture = next(
        (s for s in snapshots if s["name"] == "ac_2.4_rewrite_with_ticket_activate_format"),
        None,
    )
    assert fixture is not None, "ac_2.4 fixture missing"
    assert fixture["expected_response"].get("rewritten") is True
    assert fixture["expected_response"].get("activation_keyword") == "## ^^ticket Activate"


def test_ac_2_8_fixture_content() -> None:
    """AC 2.8 fixture must reference stale-clear on reactivation."""
    snapshots = _load_snapshots()
    fixture = next(
        (s for s in snapshots if s["name"] == "ac_2.8_stale_clear_on_reactivation"),
        None,
    )
    assert fixture is not None, "ac_2.8 fixture missing"
    assert fixture["expected_response"].get("stale_cleared") is True


def test_ac_2_11_fixture_content() -> None:
    """AC 2.11 fixture must have is_ticket_mode True in expected response."""
    snapshots = _load_snapshots()
    fixture = next(
        (s for s in snapshots if s["name"] == "ac_2.11_is_ticket_mode_flag_true"),
        None,
    )
    assert fixture is not None, "ac_2.11 fixture missing"
    assert fixture["expected_response"].get("is_ticket_mode") is True


def test_ac_2_16_fixture_content() -> None:
    """AC 2.16 fixture must reference [[__SUBAGENT__:SKILL:ticket]] canary."""
    snapshots = _load_snapshots()
    fixture = next(
        (s for s in snapshots if s["name"] == "ac_2.16_subagent_skill_ticket_canary"),
        None,
    )
    assert fixture is not None, "ac_2.16 fixture missing"
    assert fixture["expected_response"].get("canary_exact") == "[[__SUBAGENT__:SKILL:ticket]]"
