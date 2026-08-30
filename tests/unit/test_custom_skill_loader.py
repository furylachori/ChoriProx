"""Tests for the strict-schema JSON custom-skill loader.

Verifies the loader at `proxy.skills.custom_skill_loader` correctly parses,
validates, and aggregates issues from JSON files in the configured skills
directory. Coverage:

- Required fields, name/trigger/description rules
- Aliases, ttl_seconds
- Unknown keys rejected
- tool_whitelist: list, non-empty, no unknown names,
  proxy_ticket/proxy_work/task denied
- Filename/name/trigger identity, reserved names, cross-file
- Size and bound limits
- Atomic fail-closed loading

These are unit-level tests — direct calls to `load_custom_skills`,
`parse_custom_skill_file`, `discover_custom_skill_files`, and
`build_custom_skill_definition` with the tool-catalog singleton from
`proxy.tool_pipeline.tool_registry`. They do NOT go through the
singleton startup path — that's exercised in
`tests/unit/test_custom_skill_startup_contract.py`.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from proxy.skills.custom_skill_loader import (
    ALLOWED_TOP_LEVEL_KEYS,
    CONDITIONALLY_ADMITTED_FOR_CUSTOM,
    MAX_ALIASES_PER_SKILL,
    MAX_CUSTOM_SKILL_FILES,
    MAX_CUSTOM_SKILL_TTL_SECONDS,
    MAX_DESCRIPTION_LENGTH,
    MAX_SKILL_DEFINITION_BYTES,
    MAX_SKILL_PROMPT_BYTES,
    MAX_TOOLS_PER_WHITELIST,
    NAME_PATTERN,
    PERMANENTLY_DENIED_FOR_CUSTOM,
    ParsedCustomSkill,
    RESERVED_BUILTIN_NAMES,
    RESERVED_COMMAND_TRIGGERS,
    TRIGGER_PATTERN,
    CustomSkillsStatus,
    SkillLoadIssue,
    SkillRegistryConfigurationError,
    build_custom_skill_definition,
    discover_custom_skill_files,
    load_custom_skills,
    parse_custom_skill_file,
    resolve_skills_dir,
)
from proxy.tool_pipeline.tool_registry import registry as tool_catalog


def _write(root: Path, name: str, payload) -> Path:
    """Write a JSON file in `root`. Returns the path."""
    p = root / f"{name}.json"
    if isinstance(payload, str):
        p.write_text(payload)
    else:
        p.write_text(json.dumps(payload))
    return p


def _valid_payload(name: str = "ok", **overrides) -> dict:
    """Build a payload that passes every strict check by default."""
    base = {
        "name": name,
        "trigger": f"^^{name}",
        "description": f"Default description for {name}.",
        "prompt": f"Do {name} work.",
        "tool_whitelist": ["read"],
    }
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


def test_loads_valid_json_skill(tmp_path):
    """A well-formed JSON file is parsed and a ParsedCustomSkill is returned."""
    root = tmp_path / "skills"
    root.mkdir()
    _write(root, "test", _valid_payload("test"))
    parsed, issues, status = load_custom_skills(root, tool_catalog)
    assert len(parsed) == 1
    assert issues == ()
    assert status.state == "ok"
    assert status.loaded_names == ("test",)
    assert parsed[0].name == "test"
    assert parsed[0].trigger == "^^test"
    assert parsed[0].allowed_tool_names == frozenset({"read"})
    assert parsed[0].ttl_seconds == 300  # default


# ---------------------------------------------------------------------------
# Required-field validation
# ---------------------------------------------------------------------------


def test_missing_name_field_rejects_with_issue(tmp_path):
    """JSON without 'name' fails with 'name_not_string' or 'name_invalid_format'."""
    root = tmp_path / "skills"
    root.mkdir()
    _write(root, "noname", {
        "trigger": "^^noname", "description": "d",
        "prompt": "p", "tool_whitelist": ["read"],
    })
    parsed, issues, status = load_custom_skills(root, tool_catalog)
    assert parsed == ()
    assert status.state == "degraded"
    codes = [i.code for i in issues]
    assert "name_not_string" in codes or "name_invalid_format" in codes


def test_missing_trigger_field_rejects_with_issue(tmp_path):
    """JSON without 'trigger' fails with 'trigger_not_string' or 'trigger_invalid_format'."""
    root = tmp_path / "skills"
    root.mkdir()
    _write(root, "notrigger", {
        "name": "notrigger", "description": "d",
        "prompt": "p", "tool_whitelist": ["read"],
    })
    parsed, issues, _ = load_custom_skills(root, tool_catalog)
    assert parsed == ()
    codes = [i.code for i in issues]
    assert "trigger_not_string" in codes or "trigger_invalid_format" in codes


def test_invalid_name_rejects_with_issue(tmp_path):
    """An uppercase name does not match NAME_PATTERN."""
    root = tmp_path / "skills"
    root.mkdir()
    _write(root, "testupper", {
        "name": "Test", "trigger": "^^Test",
        "description": "d", "prompt": "p", "tool_whitelist": ["read"],
    })
    parsed, issues, _ = load_custom_skills(root, tool_catalog)
    assert parsed == ()
    codes = [i.code for i in issues]
    assert "name_invalid_format" in codes


def test_missing_prompt_rejects_with_issue(tmp_path):
    """JSON without prompt and prompt_file fails with code 'missing_prompt'."""
    root = tmp_path / "skills"
    root.mkdir()
    _write(root, "noprompt", {
        "name": "noprompt", "trigger": "^^noprompt",
        "description": "d", "tool_whitelist": ["read"],
    })
    parsed, issues, _ = load_custom_skills(root, tool_catalog)
    assert parsed == ()
    codes = [i.code for i in issues]
    assert "missing_prompt" in codes


# ---------------------------------------------------------------------------
# prompt_file handling
# ---------------------------------------------------------------------------


def test_prompt_file_loads_content(tmp_path):
    """A 'prompt_file' reference is read into ParsedCustomSkill.prompt_text."""
    root = tmp_path / "skills"
    root.mkdir()
    prompt_text = "Run the build pipeline."
    (root / "build_prompt.md").write_text(prompt_text)
    _write(root, "build", {
        "name": "build", "trigger": "^^build",
        "description": "Run build.",
        "prompt_file": "build_prompt.md", "tool_whitelist": ["bash", "read"],
    })
    parsed, _, _ = load_custom_skills(root, tool_catalog)
    assert parsed[0].prompt_text == prompt_text


def test_prompt_file_path_traversal_rejected(tmp_path):
    """prompt_file that escapes the skills dir is rejected."""
    root = tmp_path / "skills"
    root.mkdir()
    _write(root, "evil", {
        "name": "evil", "trigger": "^^evil",
        "description": "d", "prompt_file": "../../etc/passwd",
        "tool_whitelist": ["read"],
    })
    parsed, issues, _ = load_custom_skills(root, tool_catalog)
    assert parsed == ()
    codes = [i.code for i in issues]
    assert "prompt_unreadable" in codes or "prompt_file_unreadable" in codes


# ---------------------------------------------------------------------------
# JSON validity
# ---------------------------------------------------------------------------


def test_invalid_json_rejects_with_issue(tmp_path):
    """A file with malformed JSON fails with code 'invalid_json'."""
    root = tmp_path / "skills"
    root.mkdir()
    _write(root, "broken", "{ this is not valid json")
    parsed, issues, _ = load_custom_skills(root, tool_catalog)
    assert parsed == ()
    codes = [i.code for i in issues]
    assert "invalid_json" in codes


def test_non_json_files_ignored(tmp_path):
    """Non-JSON files in the dir are silently ignored."""
    root = tmp_path / "skills"
    root.mkdir()
    (root / "README.md").write_text("# Skills")
    _write(root, "ok", _valid_payload("ok"))
    parsed, issues, _ = load_custom_skills(root, tool_catalog)
    assert len(parsed) == 1
    assert parsed[0].name == "ok"


# ---------------------------------------------------------------------------
# Optional fields
# ---------------------------------------------------------------------------


def test_aliases_optional(tmp_path):
    """A skill without 'aliases' gets an empty aliases tuple."""
    root = tmp_path / "skills"
    root.mkdir()
    _write(root, "ali", _valid_payload("ali"))
    parsed, _, _ = load_custom_skills(root, tool_catalog)
    assert parsed[0].aliases == ()


def test_multiple_skills_loaded(tmp_path):
    """Three valid JSON files produce three ParsedCustomSkills."""
    root = tmp_path / "skills"
    root.mkdir()
    for n in ("alpha", "beta", "gamma"):
        _write(root, n, _valid_payload(n))
    parsed, _, status = load_custom_skills(root, tool_catalog)
    assert len(parsed) == 3
    assert set(status.loaded_names) == {"alpha", "beta", "gamma"}


# ---------------------------------------------------------------------------
# Atomic duplicate-name failure
# ---------------------------------------------------------------------------


def test_duplicate_skill_name_fails_atomically(tmp_path):
    """Two files with the same name → whole load fails (no partial)."""
    root = tmp_path / "skills"
    root.mkdir()
    _write(root, "first", _valid_payload("first"))
    _write(root, "second", {
        "name": "first", "trigger": "^^first",
        "description": "d", "prompt": "p", "tool_whitelist": ["read"],
    })
    parsed, issues, status = load_custom_skills(root, tool_catalog)
    assert parsed == ()
    assert status.state == "degraded"
    assert status.loaded_names == ()
    codes = [i.code for i in issues]
    assert "name_filename_mismatch" in codes or "duplicate_name" in codes


# ---------------------------------------------------------------------------
# Skills dir missing
# ---------------------------------------------------------------------------


def test_skills_dir_missing_is_noop():
    """When root is None, the loader returns a disabled status."""
    parsed, issues, status = load_custom_skills(None, tool_catalog)
    assert parsed == ()
    assert issues == ()
    assert status.state == "disabled"


# ---------------------------------------------------------------------------
# SkillDefinition behavior callbacks
# ---------------------------------------------------------------------------


def test_get_system_prompt_returns_prompt_text(tmp_path):
    """build_custom_skill_definition installs get_system_prompt returning the prompt."""
    root = tmp_path / "skills"
    root.mkdir()
    _write(root, "tester", _valid_payload(
        "tester", prompt="Use bash to run tests.",
        tool_whitelist=["bash"],
    ))
    parsed, _, _ = load_custom_skills(root, tool_catalog)
    sd = build_custom_skill_definition(parsed[0])
    assert sd.get_system_prompt(None) == "Use bash to run tests."


def test_parse_subcommand_signature_matches_dispatcher(tmp_path):
    """parse_subcommand accepts a single text arg and returns (action, text)."""
    root = tmp_path / "skills"
    root.mkdir()
    _write(root, "sig", _valid_payload("sig"))
    parsed, _, _ = load_custom_skills(root, tool_catalog)
    sd = build_custom_skill_definition(parsed[0])
    action, arg = sd.parse_subcommand("^^sig hello")
    assert action == "__direct__"
    assert arg == "^^sig hello"


def test_parse_subcommand_returns_action_not_none(tmp_path):
    """parse_subcommand returns a non-None action so dispatch routes to activation."""
    root = tmp_path / "skills"
    root.mkdir()
    _write(root, "act", _valid_payload("act"))
    parsed, _, _ = load_custom_skills(root, tool_catalog)
    sd = build_custom_skill_definition(parsed[0])
    action, _ = sd.parse_subcommand("^^act hello")
    assert action is not None


def test_parse_subcommand_activates_on_alias(tmp_path):
    """parse_subcommand must accept text starting with any of the skill's aliases."""
    root = tmp_path / "skills"
    root.mkdir()
    _write(root, "alias", {
        "name": "alias", "trigger": "^^alias",
        "description": "d", "prompt": "p",
        "tool_whitelist": ["read"],
        "aliases": ["^^bar", "^^baz"],
    })
    parsed, _, _ = load_custom_skills(root, tool_catalog)
    sd = build_custom_skill_definition(parsed[0])
    for prefix in ("^^bar", "^^baz", "^^alias"):
        action, arg = sd.parse_subcommand(f"{prefix} hello")
        assert action == "__direct__"
        assert arg == f"{prefix} hello"
    action_none, _ = sd.parse_subcommand("nothing")
    assert action_none is None


def test_parse_subcommand_works_with_no_aliases(tmp_path):
    """Aliases=() does not break the trigger path."""
    root = tmp_path / "skills"
    root.mkdir()
    _write(root, "noal", _valid_payload("noal"))
    parsed, _, _ = load_custom_skills(root, tool_catalog)
    sd = build_custom_skill_definition(parsed[0])
    action, _ = sd.parse_subcommand("^^noal hi")
    assert action == "__direct__"


def test_should_deactivate_main_agent_clears_on_stop(tmp_path):
    """Default should_deactivate returns True on stop."""
    root = tmp_path / "skills"
    root.mkdir()
    _write(root, "sd", _valid_payload("sd"))
    parsed, _, _ = load_custom_skills(root, tool_catalog)
    sd = build_custom_skill_definition(parsed[0])
    from unittest.mock import MagicMock
    assert sd.should_deactivate(MagicMock(is_subagent=False), "stop") is True
    assert sd.should_deactivate(MagicMock(is_subagent=True), "stop") is True


def test_get_activation_rewrite_includes_user_request(tmp_path):
    """Returns 'User request: <arg>' — prompt itself is NOT in the rewrite."""
    root = tmp_path / "skills"
    root.mkdir()
    _write(root, "rew", _valid_payload("rew"))
    parsed, _, _ = load_custom_skills(root, tool_catalog)
    sd = build_custom_skill_definition(parsed[0])
    from unittest.mock import MagicMock
    ctx = MagicMock(raw_user_text="^^rew some args")
    result = sd.get_activation_rewrite(ctx)
    assert "User request: some args" in result
    assert "Do rew work." not in result


def test_get_activation_rewrite_no_args_returns_empty(tmp_path):
    """If raw_user_text equals the trigger exactly, return empty."""
    root = tmp_path / "skills"
    root.mkdir()
    _write(root, "noarg", _valid_payload("noarg"))
    parsed, _, _ = load_custom_skills(root, tool_catalog)
    sd = build_custom_skill_definition(parsed[0])
    from unittest.mock import MagicMock
    assert sd.get_activation_rewrite(MagicMock(raw_user_text="^^noarg")) == ""


def test_get_activation_rewrite_empty_raw_user_text(tmp_path):
    """Empty raw_user_text returns empty."""
    root = tmp_path / "skills"
    root.mkdir()
    _write(root, "emp", _valid_payload("emp"))
    parsed, _, _ = load_custom_skills(root, tool_catalog)
    sd = build_custom_skill_definition(parsed[0])
    from unittest.mock import MagicMock
    assert sd.get_activation_rewrite(MagicMock(raw_user_text="")) == ""


def test_get_subagent_prompt_returns_prompt_text(tmp_path):
    """get_subagent_prompt returns the JSON prompt text."""
    root = tmp_path / "skills"
    root.mkdir()
    _write(root, "sapt", _valid_payload("sapt"))
    parsed, _, _ = load_custom_skills(root, tool_catalog)
    sd = build_custom_skill_definition(parsed[0])
    assert sd.get_subagent_prompt(None) == "Do sapt work."


def test_get_canary_returns_none(tmp_path):
    """Custom skills don't spawn sub-agents by default."""
    root = tmp_path / "skills"
    root.mkdir()
    _write(root, "gc", _valid_payload("gc"))
    parsed, _, _ = load_custom_skills(root, tool_catalog)
    sd = build_custom_skill_definition(parsed[0])
    from unittest.mock import MagicMock
    assert sd.get_canary(MagicMock()) is None


def test_handle_direct_returns_none(tmp_path):
    """handle_direct passthrough returns None."""
    root = tmp_path / "skills"
    root.mkdir()
    _write(root, "hd", _valid_payload("hd"))
    parsed, _, _ = load_custom_skills(root, tool_catalog)
    sd = build_custom_skill_definition(parsed[0])
    from unittest.mock import MagicMock
    assert sd.handle_direct(MagicMock(), "any", "args") is None


def test_post_process_passthrough(tmp_path):
    """post_process returns the response unchanged."""
    root = tmp_path / "skills"
    root.mkdir()
    _write(root, "pp", _valid_payload("pp"))
    parsed, _, _ = load_custom_skills(root, tool_catalog)
    sd = build_custom_skill_definition(parsed[0])
    from unittest.mock import MagicMock
    response = {"choices": [{"message": {"content": "x"}}]}
    assert sd.post_process(MagicMock(), response) is response


# ---------------------------------------------------------------------------
# Alias / trigger validation
# ---------------------------------------------------------------------------


def test_alias_with_double_caret_accepted(tmp_path):
    """^^-prefixed alias is accepted."""
    root = tmp_path / "skills"
    root.mkdir()
    _write(root, "gooda", {
        "name": "gooda", "trigger": "^^gooda",
        "description": "d", "prompt": "p",
        "tool_whitelist": ["read"],
        "aliases": ["^^ga"],
    })
    parsed, _, _ = load_custom_skills(root, tool_catalog)
    assert "^^ga" in parsed[0].aliases


def test_duplicate_trigger_rejected(tmp_path):
    """Two files with the same trigger (via alias collision) — second rejected."""
    root = tmp_path / "skills"
    root.mkdir()
    _write(root, "aaa", _valid_payload("aaa"))
    _write(root, "bbb", {
        "name": "bbb", "trigger": "^^bbb",
        "description": "d", "prompt": "p",
        "tool_whitelist": ["read"],
        "aliases": ["^^aaa"],
    })
    parsed, issues, _ = load_custom_skills(root, tool_catalog)
    assert parsed == ()
    codes = [i.code for i in issues]
    assert "duplicate_alias_cross_file" in codes


def test_duplicate_alias_rejected(tmp_path):
    """Two files where the second's alias collides with the first's trigger."""
    root = tmp_path / "skills"
    root.mkdir()
    _write(root, "aaa", _valid_payload("aaa"))
    _write(root, "bbb", {
        "name": "bbb", "trigger": "^^bbb",
        "description": "d", "prompt": "p",
        "tool_whitelist": ["read"],
        "aliases": ["^^aaa"],
    })
    parsed, issues, _ = load_custom_skills(root, tool_catalog)
    assert parsed == ()
    codes = [i.code for i in issues]
    assert "duplicate_alias_cross_file" in codes


def test_custom_skill_cannot_override_reserved_builtin(tmp_path):
    """A JSON file with name='ticket' or name='work' is rejected."""
    root = tmp_path / "skills"
    root.mkdir()
    _write(root, "ticket", {
        "name": "ticket", "trigger": "^^ticket",
        "description": "fake", "prompt": "fake",
        "tool_whitelist": ["read"],
    })
    _write(root, "work", {
        "name": "work", "trigger": "^^work",
        "description": "fake", "prompt": "fake",
        "tool_whitelist": ["read"],
    })
    parsed, issues, _ = load_custom_skills(root, tool_catalog)
    assert parsed == ()
    codes = [i.code for i in issues]
    assert codes.count("reserved_name") == 2


def test_non_dict_json_rejects_with_issue(tmp_path):
    """Top-level non-object JSON fails with code 'not_object'."""
    root = tmp_path / "skills"
    root.mkdir()
    (root / "arr.json").write_text("[1, 2, 3]")
    (root / "str.json").write_text('"hello"')
    (root / "num.json").write_text("42")
    (root / "null.json").write_text("null")
    parsed, issues, _ = load_custom_skills(root, tool_catalog)
    assert parsed == ()
    codes = [i.code for i in issues]
    assert codes.count("not_object") == 4


def test_invalid_filename_rejects(tmp_path):
    """Files not matching FILENAME_PATTERN are silently skipped."""
    root = tmp_path / "skills"
    root.mkdir()
    (root / "Uppercase.json").write_text(json.dumps(_valid_payload("uppercase")))
    (root / "1abc.json").write_text(json.dumps(_valid_payload("oneabc")))
    (root / ".hidden.json").write_text(json.dumps(_valid_payload("hidden")))
    files = discover_custom_skill_files(root)
    assert files == ()


def test_unknown_trigger_does_not_crash_loader(tmp_path):
    """A non-validating trigger produces an issue, not a crash."""
    root = tmp_path / "skills"
    root.mkdir()
    _write(root, "badtrig", {
        "name": "badtrig", "trigger": "no_caret_prefix",
        "description": "d", "prompt": "p", "tool_whitelist": ["read"],
    })
    parsed, issues, _ = load_custom_skills(root, tool_catalog)
    assert parsed == ()
    codes = [i.code for i in issues]
    assert "trigger_invalid_format" in codes


def test_duplicate_name_fails_atomically(tmp_path):
    """Same name appearing as collision → whole-load failure (no partial)."""
    root = tmp_path / "skills"
    root.mkdir()
    _write(root, "first", _valid_payload("first"))
    _write(root, "second", {
        "name": "first", "trigger": "^^first",
        "description": "d", "prompt": "p", "tool_whitelist": ["read"],
    })
    parsed, _, status = load_custom_skills(root, tool_catalog)
    assert parsed == ()
    assert status.state == "degraded"
    assert status.loaded_names == ()


def test_get_activation_rewrite_strips_alias_prefix(tmp_path):
    """User typed an alias → rewrite strips the alias prefix."""
    root = tmp_path / "skills"
    root.mkdir()
    _write(root, "als", {
        "name": "als", "trigger": "^^als",
        "description": "d", "prompt": "p",
        "tool_whitelist": ["read"],
        "aliases": ["^^a"],
    })
    parsed, _, _ = load_custom_skills(root, tool_catalog)
    sd = build_custom_skill_definition(parsed[0])
    from unittest.mock import MagicMock
    ctx = MagicMock(raw_user_text="^^a run tests")
    result = sd.get_activation_rewrite(ctx)
    assert result == "User request: run tests"
    assert "^^a" not in result


# ---------------------------------------------------------------------------
# Description validation
# ---------------------------------------------------------------------------


def test_missing_description_rejects_with_issue(tmp_path):
    """Missing description key fails."""
    root = tmp_path / "skills"
    root.mkdir()
    _write(root, "nodec", {
        "name": "nodec", "trigger": "^^nodec",
        "prompt": "p", "tool_whitelist": ["read"],
    })
    parsed, issues, _ = load_custom_skills(root, tool_catalog)
    assert parsed == ()
    codes = [i.code for i in issues]
    assert "description_not_string" in codes


def test_empty_description_rejects_with_issue(tmp_path):
    """Empty description string fails with code 'description_empty'."""
    root = tmp_path / "skills"
    root.mkdir()
    _write(root, "emptydec", {
        "name": "emptydec", "trigger": "^^emptydec",
        "description": "", "prompt": "p", "tool_whitelist": ["read"],
    })
    parsed, issues, _ = load_custom_skills(root, tool_catalog)
    assert parsed == ()
    codes = [i.code for i in issues]
    assert "description_empty" in codes


def test_too_long_description_rejects_with_issue(tmp_path):
    """Description > MAX_DESCRIPTION_LENGTH fails."""
    root = tmp_path / "skills"
    root.mkdir()
    _write(root, "longdec", {
        "name": "longdec", "trigger": "^^longdec",
        "description": "x" * (MAX_DESCRIPTION_LENGTH + 1),
        "prompt": "p", "tool_whitelist": ["read"],
    })
    parsed, issues, _ = load_custom_skills(root, tool_catalog)
    assert parsed == ()
    codes = [i.code for i in issues]
    assert "description_too_long" in codes


def test_description_with_control_chars_rejects_with_issue(tmp_path):
    """Description with a control char (e.g. \\x01) fails."""
    root = tmp_path / "skills"
    root.mkdir()
    _write(root, "ctrl", {
        "name": "ctrl", "trigger": "^^ctrl",
        "description": "hello\x01world",
        "prompt": "p", "tool_whitelist": ["read"],
    })
    parsed, issues, _ = load_custom_skills(root, tool_catalog)
    assert parsed == ()
    codes = [i.code for i in issues]
    assert "description_control_chars" in codes


# ---------------------------------------------------------------------------
# prompt / prompt_file mutual exclusion
# ---------------------------------------------------------------------------


def test_both_prompt_and_prompt_file_rejects_with_issue(tmp_path):
    """Both 'prompt' and 'prompt_file' defined → code 'both_prompt_sources'."""
    root = tmp_path / "skills"
    root.mkdir()
    (root / "extra_prompt.md").write_text("extra")
    _write(root, "both", {
        "name": "both", "trigger": "^^both",
        "description": "d", "prompt": "inline",
        "prompt_file": "extra_prompt.md", "tool_whitelist": ["read"],
    })
    parsed, issues, _ = load_custom_skills(root, tool_catalog)
    assert parsed == ()
    codes = [i.code for i in issues]
    assert "both_prompt_sources" in codes


# ---------------------------------------------------------------------------
# tool_whitelist validation
# ---------------------------------------------------------------------------


def test_required_tool_whitelist_missing_rejects_with_issue(tmp_path):
    """Missing tool_whitelist key fails with 'tool_whitelist_missing'."""
    root = tmp_path / "skills"
    root.mkdir()
    _write(root, "notw", {
        "name": "notw", "trigger": "^^notw",
        "description": "d", "prompt": "p",
    })
    parsed, issues, _ = load_custom_skills(root, tool_catalog)
    assert parsed == ()
    codes = [i.code for i in issues]
    assert "tool_whitelist_missing" in codes


def test_empty_tool_whitelist_accepted_as_deny_all(tmp_path):
    """Empty tool_whitelist list is accepted (spec §4.4 — empty list = no tools, fail-closed)."""
    root = tmp_path / "skills"
    root.mkdir()
    _write(root, "empty", {
        "name": "empty", "trigger": "^^empty",
        "description": "d", "prompt": "p",
        "tool_whitelist": [],
    })
    parsed, issues, _ = load_custom_skills(root, tool_catalog)
    assert parsed == (ParsedCustomSkill(
        name="empty",
        trigger="^^empty",
        description="d",
        aliases=(),
        prompt_text="p",
        allowed_tool_names=frozenset(),
        ttl_seconds=300,
        source_file="empty.json",
    ),)
    assert parsed[0].allowed_tool_names == frozenset()
    assert issues == ()


def test_unknown_tool_rejects_with_issue(tmp_path):
    """A tool not in the canonical catalog fails with 'unknown_tool'."""
    root = tmp_path / "skills"
    root.mkdir()
    _write(root, "unknown", {
        "name": "unknown", "trigger": "^^unknown",
        "description": "d", "prompt": "p",
        "tool_whitelist": ["not_a_real_tool"],
    })
    parsed, issues, _ = load_custom_skills(root, tool_catalog)
    assert parsed == ()
    codes = [i.code for i in issues]
    assert "unknown_tool" in codes


def test_proxy_ticket_in_whitelist_rejects_with_issue(tmp_path):
    """proxy_ticket is forbidden for custom skills."""
    root = tmp_path / "skills"
    root.mkdir()
    _write(root, "pt", {
        "name": "pt", "trigger": "^^pt",
        "description": "d", "prompt": "p",
        "tool_whitelist": ["proxy_ticket"],
    })
    parsed, issues, _ = load_custom_skills(root, tool_catalog)
    assert parsed == ()
    codes = [i.code for i in issues]
    assert "tool_whitelist_denied" in codes


def test_proxy_work_in_whitelist_rejects_with_issue(tmp_path):
    """proxy_work is forbidden for custom skills."""
    root = tmp_path / "skills"
    root.mkdir()
    _write(root, "pw", {
        "name": "pw", "trigger": "^^pw",
        "description": "d", "prompt": "p",
        "tool_whitelist": ["proxy_work"],
    })
    parsed, issues, _ = load_custom_skills(root, tool_catalog)
    assert parsed == ()
    codes = [i.code for i in issues]
    assert "tool_whitelist_denied" in codes


def test_task_without_subagents_in_whitelist_rejects_with_swarm_subagents_missing(tmp_path):
    """task in whitelist but no `subagents` policy → swarm_subagents_missing.

    Per spec §0.1, `task` is conditionally admitted (not permanently
    denied). The parent loader rejects it ONLY when the same JSON object
    does not declare a structurally-valid `subagents` policy.
    """
    root = tmp_path / "skills"
    root.mkdir()
    _write(root, "tk", {
        "name": "tk", "trigger": "^^tk",
        "description": "d", "prompt": "p",
        "tool_whitelist": ["task"],
    })
    parsed, issues, _ = load_custom_skills(root, tool_catalog)
    assert parsed == ()
    codes = [i.code for i in issues]
    assert "swarm_subagents_missing" in codes
    # REGRESSION-PIN: must NOT use the old permanent-denial code.
    assert "tool_whitelist_denied" not in codes


def test_duplicate_tool_in_whitelist_rejects_with_issue(tmp_path):
    """Duplicate tool entries in tool_whitelist fail with 'duplicate_tool'."""
    root = tmp_path / "skills"
    root.mkdir()
    _write(root, "dup", {
        "name": "dup", "trigger": "^^dup",
        "description": "d", "prompt": "p",
        "tool_whitelist": ["read", "read"],
    })
    parsed, issues, _ = load_custom_skills(root, tool_catalog)
    assert parsed == ()
    codes = [i.code for i in issues]
    assert "duplicate_tool" in codes


# ---------------------------------------------------------------------------
# Unknown top-level keys
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "field",
    [
        "subagent_tool_filter",
        "subagent_proxy_ticket_access",
        "subagent_proxy_work_access",
        "on_activate",
        "on_deactivate",
        "handle_direct",
        "parse_subcommand",
        "post_process",
        "should_deactivate",
        "get_canary",
        "get_subagent_prompt",
        "get_activation_rewrite",
        "fallback_skill",
        "fire_and_forget",
        "blocked_tools_for_subagent",
        "call_limits",
        "check_action",
        "enabled",
        "tools",
        "vars",
        "metadata",
    ],
)
def test_unknown_top_level_key_rejects_with_issue(tmp_path, field):
    """Every non-schema top-level key produces code 'unknown_field'."""
    root = tmp_path / "skills"
    root.mkdir(exist_ok=True)
    payload = _valid_payload("bad")
    payload[field] = "anything"
    _write(root, "bad", payload)
    parsed, issues, _ = load_custom_skills(root, tool_catalog)
    assert parsed == ()
    codes = [i.code for i in issues]
    assert "unknown_field" in codes, f"field {field!r} did not produce 'unknown_field'"


# ---------------------------------------------------------------------------
# Identity rules
# ---------------------------------------------------------------------------


def test_filename_name_mismatch_rejects_with_issue(tmp_path):
    """name field must equal filename stem."""
    root = tmp_path / "skills"
    root.mkdir()
    _write(root, "mismatch", {
        "name": "different", "trigger": "^^different",
        "description": "d", "prompt": "p", "tool_whitelist": ["read"],
    })
    parsed, issues, _ = load_custom_skills(root, tool_catalog)
    assert parsed == ()
    codes = [i.code for i in issues]
    assert "name_filename_mismatch" in codes


def test_trigger_name_mismatch_rejects_with_issue(tmp_path):
    """trigger must equal '^^' + name."""
    root = tmp_path / "skills"
    root.mkdir()
    _write(root, "trigmm", {
        "name": "trigmm", "trigger": "^^other",
        "description": "d", "prompt": "p", "tool_whitelist": ["read"],
    })
    parsed, issues, _ = load_custom_skills(root, tool_catalog)
    assert parsed == ()
    codes = [i.code for i in issues]
    assert "trigger_name_mismatch" in codes


def test_alias_equals_trigger_rejects_with_issue(tmp_path):
    """An alias equal to the primary trigger fails with 'alias_equals_trigger'."""
    root = tmp_path / "skills"
    root.mkdir()
    _write(root, "self", {
        "name": "self", "trigger": "^^self",
        "description": "d", "prompt": "p",
        "tool_whitelist": ["read"],
        "aliases": ["^^self"],
    })
    parsed, issues, _ = load_custom_skills(root, tool_catalog)
    assert parsed == ()
    codes = [i.code for i in issues]
    assert "alias_equals_trigger" in codes


def test_alias_without_double_caret_rejects_with_issue(tmp_path):
    """An alias without '^^' prefix fails with 'alias_invalid_format'."""
    root = tmp_path / "skills"
    root.mkdir()
    _write(root, "bada", {
        "name": "bada", "trigger": "^^bada",
        "description": "d", "prompt": "p",
        "tool_whitelist": ["read"],
        "aliases": ["t"],
    })
    parsed, issues, _ = load_custom_skills(root, tool_catalog)
    assert parsed == ()
    codes = [i.code for i in issues]
    assert "alias_invalid_format" in codes


# ---------------------------------------------------------------------------
# Reserved name / trigger
# ---------------------------------------------------------------------------


def test_reserved_name_rejects_with_issue(tmp_path):
    """Reserved builtin names (ticket, work) are rejected."""
    root = tmp_path / "skills"
    root.mkdir()
    _write(root, "ticket", {
        "name": "ticket", "trigger": "^^ticket",
        "description": "d", "prompt": "p", "tool_whitelist": ["read"],
    })
    parsed, issues, _ = load_custom_skills(root, tool_catalog)
    codes = [i.code for i in issues]
    assert "reserved_name" in codes


def test_reserved_trigger_rejects_with_issue(tmp_path):
    """Reserved command triggers (^^help, ^^status, ^^workspace, ^^ws) are rejected."""
    root = tmp_path / "skills"
    root.mkdir()
    _write(root, "help", {
        "name": "help", "trigger": "^^help",
        "description": "d", "prompt": "p", "tool_whitelist": ["read"],
    })
    parsed, issues, _ = load_custom_skills(root, tool_catalog)
    assert parsed == ()
    codes = [i.code for i in issues]
    assert "reserved_trigger" in codes


# ---------------------------------------------------------------------------
# Atomic failure
# ---------------------------------------------------------------------------


def test_atomic_failure_no_partial_load(tmp_path):
    """3 files: 2 valid + 1 invalid → loader returns 0 parsed (atomic)."""
    root = tmp_path / "skills"
    root.mkdir()
    _write(root, "valid1", _valid_payload("valid1"))
    _write(root, "valid2", _valid_payload("valid2"))
    _write(root, "invalid", {
        "name": "invalid", "trigger": "^^invalid",
        "description": "", "prompt": "p", "tool_whitelist": ["read"],
    })
    parsed, issues, status = load_custom_skills(root, tool_catalog)
    assert parsed == ()
    assert status.state == "degraded"
    assert status.loaded_names == ()
    assert any(i.code == "description_empty" for i in issues)


# ---------------------------------------------------------------------------
# Size limits
# ---------------------------------------------------------------------------


def test_size_limit_json(tmp_path):
    """A JSON file > MAX_SKILL_DEFINITION_BYTES fails with code 'definition_too_large'."""
    root = tmp_path / "skills"
    root.mkdir()
    padding = "x" * (MAX_SKILL_DEFINITION_BYTES + 1)
    payload = {
        "name": "huge", "trigger": "^^huge",
        "description": "huge", "prompt": padding, "tool_whitelist": ["read"],
    }
    _write(root, "huge", payload)
    parsed, issues, _ = load_custom_skills(root, tool_catalog)
    assert parsed == ()
    codes = [i.code for i in issues]
    assert "definition_too_large" in codes


def test_size_limit_prompt(tmp_path):
    """prompt_file > MAX_SKILL_PROMPT_BYTES fails."""
    root = tmp_path / "skills"
    root.mkdir()
    padding = "y" * (MAX_SKILL_PROMPT_BYTES + 1)
    (root / "huge_prompt.md").write_text(padding)
    _write(root, "huge", {
        "name": "huge", "trigger": "^^huge",
        "description": "d", "prompt_file": "huge_prompt.md",
        "tool_whitelist": ["read"],
    })
    parsed, issues, _ = load_custom_skills(root, tool_catalog)
    assert parsed == ()
    codes = [i.code for i in issues]
    assert "prompt_too_large" in codes or "prompt_file_unreadable" in codes


def test_max_files_exceeded(tmp_path):
    """More than MAX_CUSTOM_SKILL_FILES matching files raises discovery error."""
    root = tmp_path / "skills"
    root.mkdir()
    for i in range(MAX_CUSTOM_SKILL_FILES + 1):
        _write(root, f"f{i:04d}", _valid_payload(f"f{i:04d}"))
    parsed, issues, _ = load_custom_skills(root, tool_catalog)
    assert parsed == ()
    codes = [i.code for i in issues]
    assert "too_many_files" in codes


# ---------------------------------------------------------------------------
# ttl_seconds validation
# ---------------------------------------------------------------------------


def test_ttl_zero_rejects(tmp_path):
    """ttl_seconds=0 is below the minimum (1)."""
    root = tmp_path / "skills"
    root.mkdir()
    _write(root, "tt0", {
        "name": "tt0", "trigger": "^^tt0",
        "description": "d", "prompt": "p", "tool_whitelist": ["read"],
        "ttl_seconds": 0,
    })
    parsed, issues, _ = load_custom_skills(root, tool_catalog)
    assert parsed == ()
    codes = [i.code for i in issues]
    assert "ttl_too_small" in codes


def test_ttl_negative_rejects(tmp_path):
    """Negative ttl_seconds is rejected."""
    root = tmp_path / "skills"
    root.mkdir()
    _write(root, "ttneg", {
        "name": "ttneg", "trigger": "^^ttneg",
        "description": "d", "prompt": "p", "tool_whitelist": ["read"],
        "ttl_seconds": -1,
    })
    parsed, issues, _ = load_custom_skills(root, tool_catalog)
    assert parsed == ()
    codes = [i.code for i in issues]
    assert "ttl_too_small" in codes


def test_ttl_bool_rejects(tmp_path):
    """bool is rejected even though isinstance(True, int) is True."""
    root = tmp_path / "skills"
    root.mkdir()
    _write(root, "ttb", {
        "name": "ttb", "trigger": "^^ttb",
        "description": "d", "prompt": "p", "tool_whitelist": ["read"],
        "ttl_seconds": True,
    })
    parsed, issues, _ = load_custom_skills(root, tool_catalog)
    assert parsed == ()
    codes = [i.code for i in issues]
    assert "ttl_bool_rejected" in codes


def test_ttl_float_rejects(tmp_path):
    """Float ttl_seconds is rejected."""
    root = tmp_path / "skills"
    root.mkdir()
    _write(root, "ttf", {
        "name": "ttf", "trigger": "^^ttf",
        "description": "d", "prompt": "p", "tool_whitelist": ["read"],
        "ttl_seconds": 30.5,
    })
    parsed, issues, _ = load_custom_skills(root, tool_catalog)
    assert parsed == ()
    codes = [i.code for i in issues]
    assert "ttl_not_int" in codes


def test_ttl_string_rejects(tmp_path):
    """String ttl_seconds is rejected."""
    root = tmp_path / "skills"
    root.mkdir()
    _write(root, "tts", {
        "name": "tts", "trigger": "^^tts",
        "description": "d", "prompt": "p", "tool_whitelist": ["read"],
        "ttl_seconds": "300",
    })
    parsed, issues, _ = load_custom_skills(root, tool_catalog)
    assert parsed == ()
    codes = [i.code for i in issues]
    assert "ttl_not_int" in codes


def test_ttl_max_accepted(tmp_path):
    """ttl_seconds = MAX_CUSTOM_SKILL_TTL_SECONDS is accepted."""
    root = tmp_path / "skills"
    root.mkdir()
    _write(root, "ttmax", {
        "name": "ttmax", "trigger": "^^ttmax",
        "description": "d", "prompt": "p", "tool_whitelist": ["read"],
        "ttl_seconds": MAX_CUSTOM_SKILL_TTL_SECONDS,
    })
    parsed, issues, _ = load_custom_skills(root, tool_catalog)
    assert len(parsed) == 1
    assert issues == ()
    assert parsed[0].ttl_seconds == MAX_CUSTOM_SKILL_TTL_SECONDS


def test_ttl_max_plus_one_rejects(tmp_path):
    """ttl_seconds > MAX_CUSTOM_SKILL_TTL_SECONDS is rejected."""
    root = tmp_path / "skills"
    root.mkdir()
    _write(root, "ttp1", {
        "name": "ttp1", "trigger": "^^ttp1",
        "description": "d", "prompt": "p", "tool_whitelist": ["read"],
        "ttl_seconds": MAX_CUSTOM_SKILL_TTL_SECONDS + 1,
    })
    parsed, issues, _ = load_custom_skills(root, tool_catalog)
    assert parsed == ()
    codes = [i.code for i in issues]
    assert "ttl_too_large" in codes


# ---------------------------------------------------------------------------
# Array bounds
# ---------------------------------------------------------------------------


def test_aliases_too_many_rejects(tmp_path):
    """aliases list with > MAX_ALIASES_PER_SKILL entries is rejected."""
    root = tmp_path / "skills"
    root.mkdir()
    aliases = [f"^^a{i:03d}" for i in range(MAX_ALIASES_PER_SKILL + 1)]
    _write(root, "many", {
        "name": "many", "trigger": "^^many",
        "description": "d", "prompt": "p", "tool_whitelist": ["read"],
        "aliases": aliases,
    })
    parsed, issues, _ = load_custom_skills(root, tool_catalog)
    assert parsed == ()
    codes = [i.code for i in issues]
    assert "aliases_too_many" in codes


def test_tools_too_many_rejects(tmp_path):
    """tool_whitelist with > MAX_TOOLS_PER_WHITELIST entries is rejected."""
    root = tmp_path / "skills"
    root.mkdir()
    tools = [f"fake_tool_{i}" for i in range(MAX_TOOLS_PER_WHITELIST + 1)]
    _write(root, "manyt", {
        "name": "manyt", "trigger": "^^manyt",
        "description": "d", "prompt": "p", "tool_whitelist": tools,
    })
    parsed, issues, _ = load_custom_skills(root, tool_catalog)
    assert parsed == ()
    codes = [i.code for i in issues]
    assert "tool_whitelist_too_many" in codes


# ---------------------------------------------------------------------------
# UTF-8 validation
# ---------------------------------------------------------------------------


def test_utf8_bom_in_json_rejects(tmp_path):
    """A JSON file starting with UTF-8 BOM fails."""
    root = tmp_path / "skills"
    root.mkdir()
    bom_payload = b"\xef\xbb\xbf" + json.dumps(_valid_payload("bom")).encode()
    (root / "bom.json").write_bytes(bom_payload)
    parsed, issues, _ = load_custom_skills(root, tool_catalog)
    assert parsed == ()
    codes = [i.code for i in issues]
    assert "definition_bom" in codes


def test_utf8_bom_in_prompt_rejects(tmp_path):
    """A prompt_file starting with UTF-8 BOM fails."""
    root = tmp_path / "skills"
    root.mkdir()
    bom_prompt = b"\xef\xbb\xbf" + b"prompt content"
    (root / "bom_prompt.md").write_bytes(bom_prompt)
    _write(root, "bomp", {
        "name": "bomp", "trigger": "^^bomp",
        "description": "d", "prompt_file": "bom_prompt.md",
        "tool_whitelist": ["read"],
    })
    parsed, issues, _ = load_custom_skills(root, tool_catalog)
    assert parsed == ()
    codes = [i.code for i in issues]
    assert "prompt_bom" in codes


def test_invalid_utf8_json_rejects(tmp_path):
    """A JSON file with invalid UTF-8 bytes fails with 'definition_invalid_utf8'."""
    root = tmp_path / "skills"
    root.mkdir()
    bad_bytes = b'{"name":"ok","trigger":"^^ok","description":"\xff\xfe","prompt":"p","tool_whitelist":["read"]}'
    (root / "badutf8.json").write_bytes(bad_bytes)
    parsed, issues, _ = load_custom_skills(root, tool_catalog)
    assert parsed == ()
    codes = [i.code for i in issues]
    assert "definition_invalid_utf8" in codes


def test_invalid_utf8_prompt_rejects(tmp_path):
    """A prompt_file with invalid UTF-8 bytes fails."""
    root = tmp_path / "skills"
    root.mkdir()
    bad = b"prompt \xff\xfe content"
    (root / "bad.md").write_bytes(bad)
    _write(root, "badp", {
        "name": "badp", "trigger": "^^badp",
        "description": "d", "prompt_file": "bad.md", "tool_whitelist": ["read"],
    })
    parsed, issues, _ = load_custom_skills(root, tool_catalog)
    assert parsed == ()
    codes = [i.code for i in issues]
    assert "prompt_invalid_utf8" in codes


def test_parse_custom_skill_file_prompt_bom_returns_tuple_not_raise(tmp_path):
    """prompt_file with UTF-8 BOM → tuple (None, (issue,)) — NEVER raises.

    The 'never raises' contract covers the prompt_file branch too.
    """
    root = tmp_path / "skills"
    root.mkdir()
    bom_prompt = b"\xef\xbb\xbf" + b"prompt content"
    (root / "bom.md").write_bytes(bom_prompt)
    _write(root, "bomp", {
        "name": "bomp", "trigger": "^^bomp",
        "description": "d", "prompt_file": "bom.md",
        "tool_whitelist": ["read"],
    })
    parsed, issues = parse_custom_skill_file(
        root / "bomp.json", root, tool_catalog
    )
    assert parsed is None
    codes = [i.code for i in issues]
    assert "prompt_bom" in codes


def test_parse_custom_skill_file_prompt_invalid_utf8_returns_tuple_not_raise(tmp_path):
    """prompt_file with invalid UTF-8 → tuple (None, (issue,)) — NEVER raises."""
    root = tmp_path / "skills"
    root.mkdir()
    bad = b"prompt \xff\xfe content"
    (root / "bad.md").write_bytes(bad)
    _write(root, "badp", {
        "name": "badp", "trigger": "^^badp",
        "description": "d", "prompt_file": "bad.md",
        "tool_whitelist": ["read"],
    })
    parsed, issues = parse_custom_skill_file(
        root / "badp.json", root, tool_catalog
    )
    assert parsed is None
    codes = [i.code for i in issues]
    assert "prompt_invalid_utf8" in codes


def test_load_custom_skills_prompt_bom_aggregates_into_status(tmp_path):
    """Integration: prompt_file BOM joins the aggregated issue tuple via load_custom_skills."""
    root = tmp_path / "skills"
    root.mkdir()
    bom_prompt = b"\xef\xbb\xbf" + b"content"
    (root / "bom.md").write_bytes(bom_prompt)
    _write(root, "bomp", {
        "name": "bomp", "trigger": "^^bomp",
        "description": "d", "prompt_file": "bom.md",
        "tool_whitelist": ["read"],
    })
    parsed, issues, status = load_custom_skills(root, tool_catalog)
    assert parsed == ()
    assert status.state == "degraded"
    codes = [i.code for i in issues]
    assert "prompt_bom" in codes


# ---------------------------------------------------------------------------
# resolve_skills_dir semantics
# ---------------------------------------------------------------------------


def test_resolve_skills_dir_env_unset_default_missing(monkeypatch):
    """When env unset and default missing, returns a healthy status."""
    monkeypatch.delenv("PROXY_SKILLS_DIR", raising=False)
    resolved = resolve_skills_dir()
    assert resolved.source in ("default", "disabled")
    assert resolved.enabled in (True, False)


def test_resolve_skills_dir_env_empty_disables(monkeypatch):
    """PROXY_SKILLS_DIR='' explicitly disables."""
    monkeypatch.setenv("PROXY_SKILLS_DIR", "")
    resolved = resolve_skills_dir()
    assert resolved.source == "disabled"
    assert resolved.enabled is False
    assert resolved.root is None


def test_resolve_skills_dir_env_set_missing_raises(monkeypatch):
    """PROXY_SKILLS_DIR=/nonexistent → fatal SkillRegistryConfigurationError."""
    monkeypatch.setenv("PROXY_SKILLS_DIR", "/nonexistent/path/at/all")
    with pytest.raises(SkillRegistryConfigurationError):
        resolve_skills_dir()


# ---------------------------------------------------------------------------
# build_custom_skill_definition — basic shape
# ---------------------------------------------------------------------------


def test_build_custom_skill_definition_installs_callbacks(tmp_path):
    """build_custom_skill_definition produces a SkillDefinition with all callbacks."""
    root = tmp_path / "skills"
    root.mkdir()
    _write(root, "ok", {
        "name": "ok", "trigger": "^^ok",
        "description": "OK.", "prompt": "Do ok.",
        "tool_whitelist": ["read", "bash"],
        "aliases": ["^^o"],
    })
    parsed, _, _ = load_custom_skills(root, tool_catalog)
    sd = build_custom_skill_definition(parsed[0])
    assert sd.name == "ok"
    assert sd.trigger == "^^ok"
    assert sd.aliases == ("^^o",)
    assert sd.allowed_tool_names == frozenset({"read", "bash"})
    assert sd.get_system_prompt(None) == "Do ok."
    assert sd.get_subagent_prompt(None) == "Do ok."
    assert sd.parse_subcommand("^^o hi") == ("__direct__", "^^o hi")
    assert sd.parse_subcommand("^^ok hi") == ("__direct__", "^^ok hi")
    assert sd.parse_subcommand("nothing") == (None, "nothing")
    assert sd.handle_direct(None, "x", "y") is None
    assert sd.post_process(None, {"k": "v"}) == {"k": "v"}
    assert sd.should_deactivate(None, "stop") is True
    assert sd.get_canary(None) is None


# ---------------------------------------------------------------------------
# Pattern constants sanity
# ---------------------------------------------------------------------------


def test_name_pattern_matches_simple_names():
    """NAME_PATTERN accepts lowercase starts-with-letter names."""
    assert NAME_PATTERN.match("foo")
    assert NAME_PATTERN.match("foo-bar")
    assert NAME_PATTERN.match("foo_bar")
    assert NAME_PATTERN.match("a1")
    assert not NAME_PATTERN.match("Foo")
    assert not NAME_PATTERN.match("1foo")
    assert not NAME_PATTERN.match("-foo")
    assert not NAME_PATTERN.match("")


def test_trigger_pattern_requires_double_caret():
    """TRIGGER_PATTERN requires the '^^' prefix."""
    assert TRIGGER_PATTERN.match("^^foo")
    assert TRIGGER_PATTERN.match("^^a")
    assert not TRIGGER_PATTERN.match("foo")
    assert not TRIGGER_PATTERN.match("^foo")
    assert not TRIGGER_PATTERN.match("^^Foo")


def test_reserved_constants_have_expected_values():
    """Constants match the spec."""
    assert RESERVED_BUILTIN_NAMES == frozenset({"ticket", "work"})
    assert RESERVED_COMMAND_TRIGGERS == frozenset(
        {"^^help", "^^status", "^^workspace", "^^ws"}
    )
    assert PERMANENTLY_DENIED_FOR_CUSTOM == frozenset(
        {"proxy_ticket", "proxy_work"}
    )
    assert CONDITIONALLY_ADMITTED_FOR_CUSTOM == frozenset({"task"})
    assert "name" in ALLOWED_TOP_LEVEL_KEYS
    assert "trigger" in ALLOWED_TOP_LEVEL_KEYS
    assert "description" in ALLOWED_TOP_LEVEL_KEYS
    assert "tool_whitelist" in ALLOWED_TOP_LEVEL_KEYS
    # Reject known inert fields
    assert "fallback_skill" not in ALLOWED_TOP_LEVEL_KEYS
    assert "fire_and_forget" not in ALLOWED_TOP_LEVEL_KEYS
    assert "on_activate" not in ALLOWED_TOP_LEVEL_KEYS
    assert "subagent_tool_filter" not in ALLOWED_TOP_LEVEL_KEYS
    # Swarm opt-in key is admitted (spec §0.1 override)
    assert "subagents" in ALLOWED_TOP_LEVEL_KEYS
    # DENIED_FOR_CUSTOM must be GONE (the spec removes this alias)
    import proxy.skills.custom_skill_loader as m
    assert not hasattr(m, "DENIED_FOR_CUSTOM")


# ---------------------------------------------------------------------------
# parse_custom_skill_file direct API
# ---------------------------------------------------------------------------


def test_parse_custom_skill_file_direct(tmp_path):
    """parse_custom_skill_file returns (parsed, ()) on success.

    Signature changed from raising SkillLoadIssue to returning
    (Optional[ParsedCustomSkill], tuple[SkillLoadIssue, ...]).
    """
    root = tmp_path / "skills"
    root.mkdir()
    _write(root, "direct", _valid_payload("direct"))
    parsed, issues = parse_custom_skill_file(
        root / "direct.json", root, tool_catalog
    )
    assert parsed is not None
    assert parsed.name == "direct"
    assert parsed.source_file == "direct.json"
    assert issues == ()


def test_parse_custom_skill_file_returns_issue_tuple_on_error(tmp_path):
    """parse_custom_skill_file returns (None, (issue,)) on validation failure.

    It must NOT raise SkillLoadIssue — the function is a pure validator
    that returns a tuple per spec.
    """
    root = tmp_path / "skills"
    root.mkdir()
    (root / "bad.json").write_text("{not json")
    parsed, issues = parse_custom_skill_file(
        root / "bad.json", root, tool_catalog
    )
    assert parsed is None
    codes = [i.code for i in issues]
    assert "invalid_json" in codes


# ---------------------------------------------------------------------------
# discover_custom_skill_files API
# ---------------------------------------------------------------------------


def test_discover_custom_skill_files_sorted(tmp_path):
    """Discover returns sorted matching files."""
    root = tmp_path / "skills"
    root.mkdir()
    for name in ("zeta", "alpha", "beta"):
        _write(root, name, _valid_payload(name))
    files = discover_custom_skill_files(root)
    assert [f.name for f in files] == ["alpha.json", "beta.json", "zeta.json"]


def test_discover_custom_skill_files_excludes_non_matching(tmp_path):
    """Files not matching FILENAME_PATTERN are excluded from discovery."""
    root = tmp_path / "skills"
    root.mkdir()
    (root / "Upper.json").write_text("{}")
    (root / "README.md").write_text("")
    (root / ".hidden.json").write_text("{}")
    _write(root, "ok", _valid_payload("ok"))
    files = discover_custom_skill_files(root)
    assert [f.name for f in files] == ["ok.json"]


# ---------------------------------------------------------------------------
# CustomSkillsStatus shape
# ---------------------------------------------------------------------------


def test_custom_skills_status_disabled_shape():
    """A disabled CustomSkillsStatus has expected fields."""
    s = CustomSkillsStatus(
        enabled=False, directory=None, source="disabled",
        state="disabled", discovered=0, loaded_names=(),
        issues=(), restart_required=False,
    )
    assert s.enabled is False
    assert s.directory is None
    assert s.source == "disabled"
    assert s.state == "disabled"


def test_custom_skills_status_ok_shape(tmp_path):
    """A successful load produces an 'ok' status."""
    root = tmp_path / "skills"
    root.mkdir()
    _write(root, "ok", _valid_payload("ok"))
    _, _, status = load_custom_skills(root, tool_catalog)
    assert status.state == "ok"
    assert status.enabled is True
    assert status.directory == str(root)
    assert status.loaded_names == ("ok",)
    assert status.issues == ()


# ---------------------------------------------------------------------------
# SkillRegistryConfigurationError carries issues
# ---------------------------------------------------------------------------


def test_skill_registry_configuration_error_carries_issues():
    """SkillRegistryConfigurationError exposes .issues."""
    issue = SkillLoadIssue(file="bad.json", code="invalid_json", message="bad")
    err = SkillRegistryConfigurationError(issues=(issue,))
    assert err.issues == (issue,)
    assert "1 issue" in str(err)


# ---------------------------------------------------------------------------
# CRIT-1: Alias hijack — reserved builtin + reserved command triggers
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "reserved_alias",
    [
        "^^ticket", "^^work",
        "^^help", "^^status", "^^workspace", "^^ws",
    ],
)
def test_alias_reserved_trigger_rejected(tmp_path, reserved_alias):
    """CRIT-1: an alias matching any reserved trigger is rejected with 'alias_reserved'.

    Covers both reserved builtin triggers (``^^ticket``, ``^^work``) and
    reserved command triggers (``^^help``, ``^^status``, ``^^workspace``,
    ``^^ws``). Each reserved alias must produce a single ``alias_reserved``
    ``SkillLoadIssue`` and the whole file must fail (atomic).
    """
    root = tmp_path / "skills"
    root.mkdir(exist_ok=True)
    # File-name uses the alias body so each parametrize case is unique.
    name = "sneak_" + reserved_alias[2:]
    _write(root, name, {
        "name": name, "trigger": f"^^{name}",
        "description": "d", "prompt": "p",
        "tool_whitelist": ["read"],
        "aliases": [reserved_alias],
    })
    parsed, issues, status = load_custom_skills(root, tool_catalog)
    assert parsed == ()
    assert status.state == "degraded"
    codes = [i.code for i in issues]
    assert codes == ["alias_reserved"], (
        f"reserved alias {reserved_alias!r} should yield exactly one "
        f"alias_reserved issue, got codes={codes}"
    )
    msg = issues[0].message
    assert "reserved" in msg
    assert reserved_alias in msg


def test_alias_reserved_full_payload_degraded(tmp_path):
    """CRIT-1 (spec payload): the full hijack payload produces a clear error.

    The spec payload aliases the skill to every reserved trigger at once.
    The loader short-circuits on the first reserved alias (matching the
    short-circuit pattern of every other check in this file) — at least
    one ``alias_reserved`` issue is raised, the whole file fails, and
    the status reports ``degraded`` with zero loaded skills.
    """
    root = tmp_path / "skills"
    root.mkdir()
    _write(
        root, "sneaky",
        {
            "name": "sneaky",
            "trigger": "^^sneaky",
            "description": "d",
            "prompt": "p",
            "tool_whitelist": ["read"],
            "aliases": [
                "^^ticket", "^^work",
                "^^help", "^^status", "^^workspace", "^^ws",
            ],
        },
    )
    parsed, issues, status = load_custom_skills(root, tool_catalog)
    assert parsed == ()
    assert status.state == "degraded"
    assert status.loaded_names == ()
    codes = [i.code for i in issues]
    assert "alias_reserved" in codes
    # The error envelope is constructable from the issues tuple.
    err = SkillRegistryConfigurationError(issues=issues)
    assert err.issues == issues


def test_alias_cross_file_collision_one_trigger_one_alias(tmp_path):
    """CRIT-1 (cross-file): one file uses ^^foo as trigger, another as alias."""
    root = tmp_path / "skills"
    root.mkdir()
    _write(root, "alpha", _valid_payload("alpha"))
    _write(root, "beta", {
        "name": "beta", "trigger": "^^beta",
        "description": "d", "prompt": "p",
        "tool_whitelist": ["read"],
        "aliases": ["^^alpha"],
    })
    parsed, issues, _ = load_custom_skills(root, tool_catalog)
    assert parsed == ()
    codes = [i.code for i in issues]
    assert "duplicate_alias_cross_file" in codes


def test_alias_cross_file_collision_two_aliases_sharing(tmp_path):
    """CRIT-1 (cross-file): two files each list the same alias — second rejected."""
    root = tmp_path / "skills"
    root.mkdir()
    _write(root, "first", {
        "name": "first", "trigger": "^^first",
        "description": "d", "prompt": "p",
        "tool_whitelist": ["read"],
        "aliases": ["^^shared"],
    })
    _write(root, "second", {
        "name": "second", "trigger": "^^second",
        "description": "d", "prompt": "p",
        "tool_whitelist": ["read"],
        "aliases": ["^^shared"],
    })
    parsed, issues, _ = load_custom_skills(root, tool_catalog)
    assert parsed == ()
    codes = [i.code for i in issues]
    assert "duplicate_alias_cross_file" in codes


def test_alias_nonreserved_still_accepted(tmp_path):
    """Regression: a non-reserved alias still passes after CRIT-1 added checks."""
    root = tmp_path / "skills"
    root.mkdir()
    _write(root, "ok", {
        "name": "ok", "trigger": "^^ok",
        "description": "d", "prompt": "p",
        "tool_whitelist": ["read"],
        "aliases": ["^^alpha", "^^beta"],
    })
    parsed, issues, _ = load_custom_skills(root, tool_catalog)
    assert len(parsed) == 1
    assert issues == ()
    assert set(parsed[0].aliases) == {"^^alpha", "^^beta"}


# ---------------------------------------------------------------------------
# Swarm-capable custom-skill tests
#
# These tests exercise conditional task admission, the startup rejection
# matrix, the nested schema, and the per-request budget.
#
# Tests that construct a SubagentPolicy or pass `subagents=` to
# SkillDefinition require the data-class + SkillDefinition field.
# ---------------------------------------------------------------------------

from dataclasses import FrozenInstanceError  # noqa: E402  (used by Section J)


@pytest.fixture
def swarm_env(monkeypatch):
    """Set PROXY_ENABLE_CUSTOM_SKILL_SUBAGENTS=1 and PROXY_SESSION_SECRET."""
    monkeypatch.setenv("PROXY_ENABLE_CUSTOM_SKILL_SUBAGENTS", "1")
    monkeypatch.setenv("PROXY_SESSION_SECRET", "x" * 32)
    yield


# ---------------------------------------------------------------------------
# Section B — Ordinary skills (no subagents)
# ---------------------------------------------------------------------------


def test_ordinary_skill_no_subagents_no_task_accepted(tmp_path):
    """No subagents, no parent task → accepted (ordinary custom skill)."""
    root = tmp_path / "skills"
    root.mkdir()
    _write(root, "ord", _valid_payload("ord", tool_whitelist=["read"]))
    parsed, issues, _ = load_custom_skills(root, tool_catalog)
    assert len(parsed) == 1
    assert parsed[0].subagents is None
    assert issues == ()


def test_ordinary_skill_empty_parent_whitelist_accepted(tmp_path):
    """Empty tool_whitelist remains accepted as deny-all without subagents."""
    root = tmp_path / "skills"
    root.mkdir()
    _write(root, "emp", {
        "name": "emp", "trigger": "^^emp",
        "description": "d", "prompt": "p",
        "tool_whitelist": [],
    })
    parsed, _, _ = load_custom_skills(root, tool_catalog)
    assert parsed[0].allowed_tool_names == frozenset()
    assert parsed[0].subagents is None


def test_proxy_ticket_in_whitelist_still_always_rejects(tmp_path):
    """proxy_ticket unconditionally fires tool_whitelist_denied.

    REGRESSION-PIN: must NOT become swarm_subagents_missing.
    """
    root = tmp_path / "skills"
    root.mkdir()
    _write(root, "pt", {
        "name": "pt", "trigger": "^^pt",
        "description": "d", "prompt": "p",
        "tool_whitelist": ["proxy_ticket"],
    })
    parsed, issues, _ = load_custom_skills(root, tool_catalog)
    assert parsed == ()
    codes = [i.code for i in issues]
    assert "tool_whitelist_denied" in codes
    assert "swarm_subagents_missing" not in codes


def test_proxy_work_in_whitelist_still_always_rejects(tmp_path):
    """proxy_work unconditionally fires tool_whitelist_denied.

    REGRESSION-PIN: must NOT become swarm_subagents_missing.
    """
    root = tmp_path / "skills"
    root.mkdir()
    _write(root, "pw", {
        "name": "pw", "trigger": "^^pw",
        "description": "d", "prompt": "p",
        "tool_whitelist": ["proxy_work"],
    })
    parsed, issues, _ = load_custom_skills(root, tool_catalog)
    assert parsed == ()
    codes = [i.code for i in issues]
    assert "tool_whitelist_denied" in codes
    assert "swarm_subagents_missing" not in codes


# ---------------------------------------------------------------------------
# Section C — Valid swarm (depends on Agent 2's SubagentPolicy)
# ---------------------------------------------------------------------------


def test_valid_swarm_loads_and_parses_policy(tmp_path, swarm_env):
    """A valid swarm file passes every check and exposes a SubagentPolicy.

    Requires Agent 2's SubagentPolicy dataclass. Marked expected-to-fail
    until Agent 2 lands.
    """
    root = tmp_path / "skills"
    root.mkdir()
    _write(root, "research", {
        "name": "research", "trigger": "^^research",
        "description": "Research skill",
        "prompt": "Research.",
        "tool_whitelist": ["read", "glob", "task"],
        "subagents": {
            "tool_whitelist": ["read", "glob"],
            "max_task_calls_per_request": 4,
        },
    })
    parsed, issues, _ = load_custom_skills(root, tool_catalog)
    assert issues == ()
    assert len(parsed) == 1
    p = parsed[0]
    assert p.allowed_tool_names == frozenset({"read", "glob", "task"})
    # SkillDefinition carry-through — depends on Agent 2.
    sd = build_custom_skill_definition(p)
    assert sd.subagents is p.subagents  # identity-stable


# ---------------------------------------------------------------------------
# Section D — Prompt-only child (depends on Agent 2)
# ---------------------------------------------------------------------------


def test_swarm_prompt_only_child_loads_empty_tool_whitelist(tmp_path, swarm_env):
    """Empty child tool_whitelist=[] produces SubagentPolicy(frozenset())."""
    root = tmp_path / "skills"
    root.mkdir()
    _write(root, "promptonly", {
        "name": "promptonly", "trigger": "^^promptonly",
        "description": "d", "prompt": "p",
        "tool_whitelist": ["task"],
        "subagents": {
            "tool_whitelist": [],
            "max_task_calls_per_request": 1,
        },
    })
    parsed, issues, _ = load_custom_skills(root, tool_catalog)
    assert issues == ()
    assert parsed[0].subagents.tool_whitelist == frozenset()


# ---------------------------------------------------------------------------
# Section E — Closed nested schema (parametrize)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "extra_key",
    [
        "enabled", "children", "max_children", "prompt", "prompt_file",
        "child_prompt", "ttl_seconds", "call_limits",
    ],
)
def test_swarm_subagents_unknown_key_rejects(
    tmp_path, swarm_env, extra_key
):
    """Any unknown top-level key inside `subagents` rejects the file."""
    root = tmp_path / "skills"
    root.mkdir(exist_ok=True)
    subagents = {
        "tool_whitelist": ["read"],
        "max_task_calls_per_request": 4,
        extra_key: "anything",
    }
    _write(root, "bad", {
        "name": "bad", "trigger": "^^bad",
        "description": "d", "prompt": "p",
        "tool_whitelist": ["read", "task"],
        "subagents": subagents,
    })
    parsed, issues, _ = load_custom_skills(root, tool_catalog)
    assert parsed == ()
    codes = [i.code for i in issues]
    assert "swarm_subagents_unknown_key" in codes


# ---------------------------------------------------------------------------
# Section F — Child whitelist failures (parametrize)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "raw_tw,expected_code",
    [
        (None, "swarm_tool_whitelist_missing"),
        ("not-a-list", "swarm_tool_whitelist_not_list"),
        ([123], "swarm_tool_whitelist_entry_invalid"),   # non-string
        ([""], "swarm_tool_whitelist_entry_invalid"),    # empty string
        (["unknown_tool_xyz"], "swarm_unknown_tool"),
        (["read", "read"], "swarm_duplicate_tool"),
        (["task"], "swarm_child_forbidden_tool"),
        (["question"], "swarm_child_forbidden_tool"),
        (["proxy_ticket"], "swarm_child_forbidden_tool"),
        (["proxy_work"], "swarm_child_forbidden_tool"),
    ],
)
def test_swarm_child_tool_whitelist_failures(
    tmp_path, swarm_env, raw_tw, expected_code
):
    """Every malformed child tool_whitelist shape emits its specific code."""
    root = tmp_path / "skills"
    root.mkdir(exist_ok=True)
    _write(root, "bad", {
        "name": "bad", "trigger": "^^bad",
        "description": "d", "prompt": "p",
        "tool_whitelist": ["read", "task"],
        "subagents": {
            "tool_whitelist": raw_tw,
            "max_task_calls_per_request": 4,
        },
    })
    parsed, issues, _ = load_custom_skills(root, tool_catalog)
    assert parsed == ()
    codes = [i.code for i in issues]
    assert expected_code in codes


# ---------------------------------------------------------------------------
# Section G — Budget failures (parametrize)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "raw_max,expected_code,omit_key",
    [
        (None, "swarm_max_calls_missing", True),     # missing key
        (None, "swarm_max_calls_missing", False),    # explicit null
        (True, "swarm_max_calls_not_integer", False),
        (False, "swarm_max_calls_not_integer", False),
        ("4", "swarm_max_calls_not_integer", False),
        (4.5, "swarm_max_calls_not_integer", False),
        (0, "swarm_max_calls_out_of_range", False),
        (-1, "swarm_max_calls_out_of_range", False),
        (17, "swarm_max_calls_out_of_range", False),
        (1, None, False),    # valid
        (16, None, False),   # valid
    ],
)
def test_swarm_max_calls_budget_failures(
    tmp_path, swarm_env, raw_max, expected_code, omit_key
):
    """Every malformed max_task_calls_per_request value emits its code."""
    root = tmp_path / "skills"
    root.mkdir(exist_ok=True)
    subagents: dict = {"tool_whitelist": ["read"]}
    if omit_key:
        # raw_max is None — leave the key out entirely.
        pass
    else:
        subagents["max_task_calls_per_request"] = raw_max
    _write(root, "bad", {
        "name": "bad", "trigger": "^^bad",
        "description": "d", "prompt": "p",
        "tool_whitelist": ["read", "task"],
        "subagents": subagents,
    })
    parsed, issues, _ = load_custom_skills(root, tool_catalog)
    if expected_code is None:
        # Valid case — but the file still needs task + subagents to be a
        # valid swarm. With valid budget + read parent + swarm_env, the
        # load should succeed.
        assert issues == ()
        assert len(parsed) == 1
    else:
        assert parsed == ()
        codes = [i.code for i in issues]
        assert expected_code in codes


# ---------------------------------------------------------------------------
# Section H — Contradiction and deployment matrix
# ---------------------------------------------------------------------------


def test_swarm_parent_present_no_subagents_rejects(tmp_path, swarm_env):
    """Parent has 'task' but no `subagents` key → swarm_subagents_missing."""
    root = tmp_path / "skills"
    root.mkdir()
    _write(root, "bad", {
        "name": "bad", "trigger": "^^bad",
        "description": "d", "prompt": "p",
        "tool_whitelist": ["read", "task"],
    })
    parsed, issues, _ = load_custom_skills(root, tool_catalog)
    assert parsed == ()
    codes = [i.code for i in issues]
    assert "swarm_subagents_missing" in codes


def test_swarm_subagents_present_no_parent_task_rejects(tmp_path, swarm_env):
    """subagents present but parent's tool_whitelist lacks 'task' → swarm_parent_task_missing."""
    root = tmp_path / "skills"
    root.mkdir()
    _write(root, "bad", {
        "name": "bad", "trigger": "^^bad",
        "description": "d", "prompt": "p",
        "tool_whitelist": ["read"],
        "subagents": {
            "tool_whitelist": ["read"],
            "max_task_calls_per_request": 4,
        },
    })
    parsed, issues, _ = load_custom_skills(root, tool_catalog)
    assert parsed == ()
    codes = [i.code for i in issues]
    assert "swarm_parent_task_missing" in codes


def test_swarm_flag_not_set_rejects(tmp_path, monkeypatch):
    """Valid swarm but PROXY_ENABLE_CUSTOM_SKILL_SUBAGENTS unset → swarm_flag_disabled."""
    monkeypatch.delenv("PROXY_ENABLE_CUSTOM_SKILL_SUBAGENTS", raising=False)
    monkeypatch.setenv("PROXY_SESSION_SECRET", "x" * 32)
    root = tmp_path / "skills"
    root.mkdir()
    _write(root, "research", {
        "name": "research", "trigger": "^^research",
        "description": "d", "prompt": "p",
        "tool_whitelist": ["read", "task"],
        "subagents": {
            "tool_whitelist": ["read"],
            "max_task_calls_per_request": 4,
        },
    })
    parsed, issues, _ = load_custom_skills(root, tool_catalog)
    assert parsed == ()
    codes = [i.code for i in issues]
    assert "swarm_flag_disabled" in codes


def test_swarm_flag_empty_string_rejects(tmp_path, monkeypatch):
    """PROXY_ENABLE_CUSTOM_SKILL_SUBAGENTS='' → swarm_flag_disabled."""
    monkeypatch.setenv("PROXY_ENABLE_CUSTOM_SKILL_SUBAGENTS", "")
    monkeypatch.setenv("PROXY_SESSION_SECRET", "x" * 32)
    root = tmp_path / "skills"
    root.mkdir()
    _write(root, "research", {
        "name": "research", "trigger": "^^research",
        "description": "d", "prompt": "p",
        "tool_whitelist": ["read", "task"],
        "subagents": {
            "tool_whitelist": ["read"],
            "max_task_calls_per_request": 4,
        },
    })
    parsed, issues, _ = load_custom_skills(root, tool_catalog)
    assert parsed == ()
    codes = [i.code for i in issues]
    assert "swarm_flag_disabled" in codes


def test_swarm_flag_non_one_value_rejects(tmp_path, monkeypatch):
    """PROXY_ENABLE_CUSTOM_SKILL_SUBAGENTS='true' / '0' / 'yes' → swarm_flag_disabled.

    Only literal "1" is accepted (spec §3).
    """
    for bad_value in ("true", "0", "yes", "on", "enabled"):
        monkeypatch.setenv("PROXY_ENABLE_CUSTOM_SKILL_SUBAGENTS", bad_value)
        monkeypatch.setenv("PROXY_SESSION_SECRET", "x" * 32)
        root = tmp_path / "skills"
        root.mkdir(exist_ok=True)
        _write(root, "research", {
            "name": "research", "trigger": "^^research",
            "description": "d", "prompt": "p",
            "tool_whitelist": ["read", "task"],
            "subagents": {
                "tool_whitelist": ["read"],
                "max_task_calls_per_request": 4,
            },
        })
        parsed, issues, _ = load_custom_skills(root, tool_catalog)
        assert parsed == (), f"flag={bad_value!r} should reject"
        codes = [i.code for i in issues]
        assert "swarm_flag_disabled" in codes, f"flag={bad_value!r}"


def test_swarm_secret_missing_rejects(tmp_path, monkeypatch):
    """Flag set but PROXY_SESSION_SECRET missing → swarm_missing_secret."""
    monkeypatch.setenv("PROXY_ENABLE_CUSTOM_SKILL_SUBAGENTS", "1")
    monkeypatch.delenv("PROXY_SESSION_SECRET", raising=False)
    root = tmp_path / "skills"
    root.mkdir()
    _write(root, "research", {
        "name": "research", "trigger": "^^research",
        "description": "d", "prompt": "p",
        "tool_whitelist": ["read", "task"],
        "subagents": {
            "tool_whitelist": ["read"],
            "max_task_calls_per_request": 4,
        },
    })
    parsed, issues, _ = load_custom_skills(root, tool_catalog)
    assert parsed == ()
    codes = [i.code for i in issues]
    assert "swarm_missing_secret" in codes


def test_swarm_secret_empty_string_rejects(tmp_path, monkeypatch):
    """Flag set but PROXY_SESSION_SECRET='' → swarm_missing_secret."""
    monkeypatch.setenv("PROXY_ENABLE_CUSTOM_SKILL_SUBAGENTS", "1")
    monkeypatch.setenv("PROXY_SESSION_SECRET", "")
    root = tmp_path / "skills"
    root.mkdir()
    _write(root, "research", {
        "name": "research", "trigger": "^^research",
        "description": "d", "prompt": "p",
        "tool_whitelist": ["read", "task"],
        "subagents": {
            "tool_whitelist": ["read"],
            "max_task_calls_per_request": 4,
        },
    })
    parsed, issues, _ = load_custom_skills(root, tool_catalog)
    assert parsed == ()
    codes = [i.code for i in issues]
    assert "swarm_missing_secret" in codes


# ---------------------------------------------------------------------------
# Section I — Aggregation across multiple files
# ---------------------------------------------------------------------------


def test_swarm_multi_file_failure_aggregates_all_issues(tmp_path, swarm_env):
    """3 files with different swarm validation failures → all 3 issues appear.

    All three failure paths exercise code that does NOT require
    SubagentPolicy instantiation (so the test runs without Agent 2):
      bad1 — task in parent without `subagents` → swarm_subagents_missing
      bad2 — `subagents: {}` → swarm_tool_whitelist_missing + swarm_max_calls_missing
      bad3 — child whitelist contains forbidden tool → swarm_child_forbidden_tool
    """
    root = tmp_path / "skills"
    root.mkdir()
    _write(root, "bad1", {
        "name": "bad1", "trigger": "^^bad1",
        "description": "d", "prompt": "p",
        "tool_whitelist": ["read", "task"],
    })
    _write(root, "bad2", {
        "name": "bad2", "trigger": "^^bad2",
        "description": "d", "prompt": "p",
        "tool_whitelist": ["read", "task"],
        "subagents": {},
    })
    _write(root, "bad3", {
        "name": "bad3", "trigger": "^^bad3",
        "description": "d", "prompt": "p",
        "tool_whitelist": ["read", "task"],
        "subagents": {
            "tool_whitelist": ["task"],
            "max_task_calls_per_request": 4,
        },
    })
    parsed, issues, _ = load_custom_skills(root, tool_catalog)
    assert parsed == ()
    codes = sorted([i.code for i in issues])
    assert "swarm_subagents_missing" in codes
    assert "swarm_tool_whitelist_missing" in codes
    assert "swarm_max_calls_missing" in codes
    assert "swarm_child_forbidden_tool" in codes


def test_swarm_single_file_flag_disabled_and_secret_missing_reports_both(
    tmp_path, monkeypatch
):
    """A structurally-valid swarm file with BOTH flag disabled AND secret missing
    → BOTH `swarm_flag_disabled` and `swarm_missing_secret` reported."""
    monkeypatch.delenv("PROXY_ENABLE_CUSTOM_SKILL_SUBAGENTS", raising=False)
    monkeypatch.delenv("PROXY_SESSION_SECRET", raising=False)
    root = tmp_path / "skills"
    root.mkdir()
    _write(root, "research", {
        "name": "research", "trigger": "^^research",
        "description": "d", "prompt": "p",
        "tool_whitelist": ["read", "task"],
        "subagents": {
            "tool_whitelist": ["read"],
            "max_task_calls_per_request": 4,
        },
    })
    parsed, issues, _ = load_custom_skills(root, tool_catalog)
    assert parsed == ()
    codes = [i.code for i in issues]
    assert "swarm_flag_disabled" in codes
    assert "swarm_missing_secret" in codes


# ---------------------------------------------------------------------------
# Section J — Immutable status (swarm fields)
# ---------------------------------------------------------------------------


def test_custom_skills_status_no_swarm(tmp_path, monkeypatch):
    """No swarm skills → swarm_enabled=False, swarm_capable_skills=()."""
    monkeypatch.delenv("PROXY_ENABLE_CUSTOM_SKILL_SUBAGENTS", raising=False)
    root = tmp_path / "skills"
    root.mkdir()
    _write(root, "ok", _valid_payload("ok"))
    _, _, status = load_custom_skills(root, tool_catalog)
    assert status.swarm_enabled is False
    assert status.swarm_capable_skills == ()


def test_custom_skills_status_one_swarm(tmp_path, swarm_env):
    """One valid swarm file → swarm_enabled=True, single canonical name."""
    root = tmp_path / "skills"
    root.mkdir()
    _write(root, "research", {
        "name": "research", "trigger": "^^research",
        "description": "d", "prompt": "p",
        "tool_whitelist": ["read", "task"],
        "subagents": {
            "tool_whitelist": ["read"],
            "max_task_calls_per_request": 4,
        },
    })
    _, _, status = load_custom_skills(root, tool_catalog)
    assert status.swarm_enabled is True
    assert status.swarm_capable_skills == ("research",)


def test_custom_skills_status_multiple_swarms_sorted(tmp_path, swarm_env):
    """3 valid swarm files → sorted canonical names in swarm_capable_skills."""
    root = tmp_path / "skills"
    root.mkdir()
    for name in ("zeta", "alpha", "mu"):
        _write(root, name, {
            "name": name, "trigger": f"^^{name}",
            "description": "d", "prompt": "p",
            "tool_whitelist": ["read", "task"],
            "subagents": {
                "tool_whitelist": ["read"],
                "max_task_calls_per_request": 4,
            },
        })
    _, _, status = load_custom_skills(root, tool_catalog)
    assert status.swarm_enabled is True
    assert status.swarm_capable_skills == ("alpha", "mu", "zeta")


def test_custom_skills_status_is_frozen():
    """CustomSkillsStatus fields cannot be reassigned (frozen dataclass)."""
    s = CustomSkillsStatus(
        enabled=False, directory=None, source="disabled",
        state="disabled", discovered=0, loaded_names=(),
        issues=(), restart_required=False,
    )
    with pytest.raises(FrozenInstanceError):
        s.swarm_enabled = True  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Section K — Additional invariants
# ---------------------------------------------------------------------------


def test_swarm_subagents_null_value_rejects_swarm_subagents_not_object(
    tmp_path, swarm_env
):
    """subagents: null → swarm_subagents_not_object (not swarm_subagents_missing).

    Per spec §10, `subagents: null` is a malformed declared policy —
    the key is present but the value is not an object. The loader
    distinguishes "key absent" from "key present but malformed".
    """
    root = tmp_path / "skills"
    root.mkdir()
    _write(root, "bad", {
        "name": "bad", "trigger": "^^bad",
        "description": "d", "prompt": "p",
        "tool_whitelist": ["read", "task"],
        "subagents": None,
    })
    parsed, issues, _ = load_custom_skills(root, tool_catalog)
    assert parsed == ()
    codes = [i.code for i in issues]
    assert "swarm_subagents_not_object" in codes
    assert "swarm_subagents_missing" not in codes


def test_swarm_subagents_string_value_rejects_swarm_subagents_not_object(
    tmp_path, swarm_env
):
    """subagents: "string" → swarm_subagents_not_object (not swarm_subagents_missing)."""
    root = tmp_path / "skills"
    root.mkdir()
    _write(root, "bad", {
        "name": "bad", "trigger": "^^bad",
        "description": "d", "prompt": "p",
        "tool_whitelist": ["read", "task"],
        "subagents": "not-an-object",
    })
    parsed, issues, _ = load_custom_skills(root, tool_catalog)
    assert parsed == ()
    codes = [i.code for i in issues]
    assert "swarm_subagents_not_object" in codes
    assert "swarm_subagents_missing" not in codes


def test_swarm_subagents_omitted_with_task_in_whitelist_rejects_swarm_subagents_missing(
    tmp_path, swarm_env
):
    """Parent has 'task' but no `subagents` key (omitted) → swarm_subagents_missing.

    Distinct from `subagents: null` (key present, value malformed).
    """
    root = tmp_path / "skills"
    root.mkdir()
    _write(root, "bad", {
        "name": "bad", "trigger": "^^bad",
        "description": "d", "prompt": "p",
        "tool_whitelist": ["read", "task"],
        # NO "subagents" key
    })
    parsed, issues, _ = load_custom_skills(root, tool_catalog)
    assert parsed == ()
    codes = [i.code for i in issues]
    assert "swarm_subagents_missing" in codes
    assert "swarm_subagents_not_object" not in codes


def test_swarm_subagents_empty_object_reports_two_nested_missing(
    tmp_path, swarm_env
):
    """`subagents: {}` reports both swarm_tool_whitelist_missing AND swarm_max_calls_missing."""
    root = tmp_path / "skills"
    root.mkdir()
    _write(root, "bad", {
        "name": "bad", "trigger": "^^bad",
        "description": "d", "prompt": "p",
        "tool_whitelist": ["read", "task"],
        "subagents": {},
    })
    parsed, issues, _ = load_custom_skills(root, tool_catalog)
    assert parsed == ()
    codes = [i.code for i in issues]
    assert "swarm_tool_whitelist_missing" in codes
    assert "swarm_max_calls_missing" in codes
    assert "swarm_subagents_unknown_key" not in codes  # empty object → no extras