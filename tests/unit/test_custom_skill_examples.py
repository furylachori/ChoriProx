"""Tests for the shipped example custom-skill files in `examples/skills/`.

Per spec §9: the shipped example skills are moved to `examples/skills/`
as starting points for operators. They are NOT silently enabled in
default source deployments.

Tests in this file:
- Verify each shipped example parses cleanly through the strict-schema
  loader.
- Verify the loader can register all examples atomically.
- Skip cleanly if `examples/skills/` doesn't exist yet (Agent 4
  relocation pending).
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from proxy.skills.custom_skill_loader import (
    CustomSkillsStatus,
    SkillRegistryConfigurationError,
    load_custom_skills,
)
from proxy.tool_pipeline.tool_registry import registry as tool_catalog


EXAMPLES_DIR = (
    Path(__file__).resolve().parents[2]
    / "examples"
    / "skills"
)


def _examples_dir() -> Path | None:
    """Return the examples/skills/ path if it exists and has JSON files."""
    if not EXAMPLES_DIR.exists() or not EXAMPLES_DIR.is_dir():
        return None
    if not any(EXAMPLES_DIR.glob("*.json")):
        return None
    return EXAMPLES_DIR


pytestmark = pytest.mark.skipif(
    _examples_dir() is None,
    reason="examples/skills/ not yet populated (Agent 4 relocation pending)",
)


def test_examples_dir_exists():
    """examples/skills/ exists and is a directory."""
    assert EXAMPLES_DIR.exists()
    assert EXAMPLES_DIR.is_dir()


def test_examples_have_at_least_one_json_file():
    """At least one JSON file in examples/skills/."""
    files = list(EXAMPLES_DIR.glob("*.json"))
    assert len(files) >= 1, f"no JSON files in {EXAMPLES_DIR}"


def test_each_example_parses_cleanly():
    """Each shipped example parses through the strict-schema loader."""
    examples = _examples_dir()
    assert examples is not None
    files = list(examples.glob("*.json"))
    for path in files:
        data = json.loads(path.read_text())
        # Strict schema invariants for shipped examples:
        assert "name" in data, f"{path}: missing name"
        assert "trigger" in data, f"{path}: missing trigger"
        assert "description" in data, f"{path}: missing description"
        assert data["description"], f"{path}: empty description"
        assert len(data["description"]) <= 160, (
            f"{path}: description > 160 chars"
        )
        assert data["trigger"] == f"^^{data['name']}", (
            f"{path}: trigger != ^^<name>"
        )
        assert "tool_whitelist" in data, f"{path}: missing tool_whitelist"
        assert isinstance(data["tool_whitelist"], list), (
            f"{path}: tool_whitelist not a list"
        )
        assert len(data["tool_whitelist"]) >= 1, (
            f"{path}: empty tool_whitelist"
        )
        # Must have prompt OR prompt_file (exactly one)
        has_p = "prompt" in data
        has_pf = "prompt_file" in data
        assert has_p ^ has_pf, (
            f"{path}: must have exactly one of prompt/prompt_file"
        )


def test_loader_accepts_all_examples(tmp_path):
    """The loader parses every shipped example without issues."""
    examples = _examples_dir()
    assert examples is not None
    parsed, issues, status = load_custom_skills(examples, tool_catalog)
    assert issues == (), (
        f"loader found {len(issues)} issues in examples/skills/: "
        f"{[i.code for i in issues]}"
    )
    assert status.state == "ok"
    expected = sorted(p.stem for p in examples.glob("*.json"))
    actual = sorted(status.loaded_names)
    assert actual == expected, (
        f"loaded_names mismatch: {actual} vs expected {expected}"
    )


def test_examples_cannot_override_builtins():
    """No shipped example uses a reserved builtin name (ticket, work)."""
    examples = _examples_dir()
    assert examples is not None
    for path in examples.glob("*.json"):
        data = json.loads(path.read_text())
        assert data["name"] not in ("ticket", "work"), (
            f"{path}: example uses reserved builtin name"
        )


def test_examples_use_known_tools():
    """Every tool in shipped examples must be a known catalog entry."""
    examples = _examples_dir()
    assert examples is not None
    known = set(tool_catalog.get_tool_names())
    for path in examples.glob("*.json"):
        data = json.loads(path.read_text())
        for tool in data["tool_whitelist"]:
            assert tool in known, (
                f"{path}: unknown tool {tool!r} in tool_whitelist"
            )


def test_examples_no_denied_tools():
    """Shipped examples must not include proxy_ticket / proxy_work / task."""
    examples = _examples_dir()
    assert examples is not None
    DENIED = {"proxy_ticket", "proxy_work", "task"}
    for path in examples.glob("*.json"):
        data = json.loads(path.read_text())
        for tool in data["tool_whitelist"]:
            assert tool not in DENIED, (
                f"{path}: example uses denied tool {tool!r}"
            )


def test_examples_pure_schema_keys():
    """No example includes a rejected top-level key."""
    examples = _examples_dir()
    assert examples is not None
    from proxy.skills.custom_skill_loader import ALLOWED_TOP_LEVEL_KEYS
    for path in examples.glob("*.json"):
        data = json.loads(path.read_text())
        extras = set(data.keys()) - ALLOWED_TOP_LEVEL_KEYS
        assert not extras, (
            f"{path}: unknown top-level keys {sorted(extras)!r}"
        )