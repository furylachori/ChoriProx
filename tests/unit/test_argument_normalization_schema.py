"""
Unit tests for schema-driven tool argument normalization.

Covers the fix where normalize_tool_arguments reconciles arg names against
the client's actual tool schema (camelCase preservation for opencode).

US-124: Tool-call names and argument names are reconciled to the client's schema.
"""

import json

import pytest

from proxy.tool_parsing import (
    build_valid_params_by_tool,
    make_tool_calls,
    normalize_tool_arguments,
    provide_defaults_for_optional_params,
)


def _opencode_read_schema():
    return {
        "type": "function",
        "function": {
            "name": "read",
            "parameters": {
                "type": "object",
                "properties": {
                    "filePath": {"type": "string"},
                    "offset": {"type": "integer"},
                    "limit": {"type": "integer"},
                },
            },
        },
    }


class TestCamelCasePreserved:
    """opencode emits camelCase; the schema must win over snake_case aliases."""

    def test_filepath_exact_preserved(self):
        """filePath is an exact schema match and must be preserved."""
        vp = {"filePath", "offset", "limit"}
        result = normalize_tool_arguments("read", {"filePath": "/tmp/a.txt"}, valid_params=vp)
        assert result == {"filePath": "/tmp/a.txt"}

    def test_degraded_file_path_to_filepath(self):
        """file_path (snake) resolves to filePath (camel) via fuzzy match."""
        vp = {"filePath", "offset", "limit"}
        result = normalize_tool_arguments("read", {"file_path": "/tmp/a.txt"}, valid_params=vp)
        assert result == {"filePath": "/tmp/a.txt"}

    def test_casing_drift_to_filepath(self):
        """File / FilePath / FILEPATH all resolve to filePath."""
        vp = {"filePath", "offset", "limit"}
        for degraded in ("File", "FilePath", "FILEPATH"):
            result = normalize_tool_arguments("read", {degraded: "/tmp/a.txt"}, valid_params=vp)
            assert result == {"filePath": "/tmp/a.txt"}, f"failed for {degraded}"


class TestEditCamelCase:
    """edit old_string/new_string reconcile to oldString/newString."""

    def test_edit_snake_to_camel(self):
        vp = {"filePath", "oldString", "newString"}
        result = normalize_tool_arguments(
            "edit",
            {"file_path": "/tmp/a.txt", "old_string": "a", "new_string": "b"},
            valid_params=vp,
        )
        assert result == {"filePath": "/tmp/a.txt", "oldString": "a", "newString": "b"}


class TestBashSynonymReconciled:
    """cmd alias reconciles to the real schema name in both snake and camel."""

    def test_cmd_to_command_snake(self):
        vp = {"command"}
        result = normalize_tool_arguments("bash", {"cmd": "ls"}, valid_params=vp)
        assert result == {"command": "ls"}

    def test_cmd_to_command_camel(self):
        vp = {"Command"}
        result = normalize_tool_arguments("bash", {"cmd": "ls"}, valid_params=vp)
        assert result == {"Command": "ls"}


class TestLegacyBehaviorPreserved:
    """valid_params=None keeps the original snake_case synonym behavior."""

    def test_snake_case_alias_when_no_schema(self):
        result = normalize_tool_arguments("read_file", {"path": "/tmp/a.txt"})
        assert result == {"file_path": "/tmp/a.txt"}

    def test_file_path_stays_when_no_schema(self):
        result = normalize_tool_arguments("read", {"file_path": "/tmp/a.txt"})
        assert result == {"file_path": "/tmp/a.txt"}

    def test_unknown_key_passthrough(self):
        result = normalize_tool_arguments("read", {"offset": 5})
        assert result == {"offset": 5}


class TestIdempotency:
    """normalize(normalize(x, vp), vp) == normalize(x, vp)."""

    @pytest.mark.parametrize(
        "tool,args,vp",
        [
            ("read", {"filePath": "/x"}, {"filePath", "offset", "limit"}),
            ("read", {"file_path": "/x"}, {"filePath", "offset", "limit"}),
            ("read", {"path": "/x"}, {"filePath", "offset", "limit"}),
            ("edit", {"old_string": "a", "new_string": "b"}, {"oldString", "newString"}),
            ("bash", {"cmd": "ls"}, {"command"}),
        ],
    )
    def test_idempotent(self, tool, args, vp):
        once = normalize_tool_arguments(tool, dict(args), valid_params=vp)
        twice = normalize_tool_arguments(tool, dict(once), valid_params=vp)
        assert once == twice


class TestMakeToolCallsSchemaAware:
    """make_tool_calls threads valid_params_by_tool end to end."""

    def test_emits_camelcase_filepath(self):
        tc = [{"name": "read", "arguments": {"filePath": "/tmp/a.txt"}}]
        vpbt = {"read": {"filePath", "offset", "limit"}}
        result = make_tool_calls(tc, valid_params_by_tool=vpbt)
        assert result is not None
        args = json.loads(result[0]["function"]["arguments"])
        assert "filePath" in args
        assert "file_path" not in args

    def test_emits_camelcase_from_snake_input(self):
        tc = [{"name": "read", "arguments": {"file_path": "/tmp/a.txt"}}]
        vpbt = {"read": {"filePath", "offset", "limit"}}
        result = make_tool_calls(tc, valid_params_by_tool=vpbt)
        args = json.loads(result[0]["function"]["arguments"])
        assert args == {"filePath": "/tmp/a.txt"}


class TestBuildValidParamsByTool:
    """build_valid_params_by_tool extracts real param names from tools[]."""

    def test_extracts_names_from_openai_schema(self):
        tools = [
            _opencode_read_schema(),
            {
                "type": "function",
                "function": {
                    "name": "bash",
                    "parameters": {
                        "type": "object",
                        "properties": {"command": {"type": "string"}, "workdir": {"type": "string"}},
                    },
                },
            },
        ]
        result = build_valid_params_by_tool(tools)
        assert result["read"] == {"filePath", "offset", "limit"}
        assert result["bash"] == {"command", "workdir"}

    def test_returns_empty_for_none(self):
        assert build_valid_params_by_tool(None) == {}

    def test_returns_empty_for_empty_list(self):
        assert build_valid_params_by_tool([]) == {}

    def test_returns_empty_for_non_list(self):
        assert build_valid_params_by_tool({"not": "a list"}) == {}

    def test_tolerates_missing_parameters(self):
        tools = [{"type": "function", "function": {"name": "bare"}}]
        assert build_valid_params_by_tool(tools) == {}

    def test_tolerates_malformed_entries(self):
        tools = ["not a dict", None, 42, {"function": None}, {"function": {"name": "ok"}}]
        assert build_valid_params_by_tool(tools) == {}


class TestProvideDefaultsSchemaAware:
    """provide_defaults_for_optional_params injects reconciled real names."""

    def test_injects_real_camel_name(self):
        args = {}
        result = provide_defaults_for_optional_params("task", args, valid_params={"subagentType"})
        assert result == {"subagentType": "general"}

    def test_does_not_duplicate_when_real_present(self):
        args = {"subagentType": "coder"}
        result = provide_defaults_for_optional_params("task", args, valid_params={"subagentType"})
        assert result == {"subagentType": "coder"}

    def test_injects_snake_when_no_schema(self):
        args = {}
        result = provide_defaults_for_optional_params("task", args)
        assert result == {"subagent_type": "general"}


class TestRegressionOpencodeRead:
    """Regression: a read tool_call through make_tool_calls with an opencode-shaped
    schema yields filePath (not file_path)."""

    def test_read_through_make_tool_calls(self):
        tools = [_opencode_read_schema()]
        vpbt = build_valid_params_by_tool(tools)
        tc = [{"name": "read", "arguments": {"filePath": "/tmp/a.txt", "offset": 10}}]
        result = make_tool_calls(tc, valid_params_by_tool=vpbt)
        assert result is not None
        args = json.loads(result[0]["function"]["arguments"])
        assert args == {"filePath": "/tmp/a.txt", "offset": 10}
        assert "file_path" not in args


class TestTaskModeAlias:
    """Regression: the `mode` alias on the `task` tool must map to subagent_mode,
    not subagent_type. canary.py reads args.get("subagent_mode")."""

    def test_mode_alias_without_schema(self):
        """valid_params=None keeps the legacy alias mapping mode -> subagent_mode."""
        result = normalize_tool_arguments("task", {"mode": "work"})
        assert result == {"subagent_mode": "work"}

    def test_mode_alias_with_schema(self):
        """With a schema advertising subagent_mode, mode still reconciles to it."""
        vp = frozenset({"prompt", "subagent_mode"})
        result = normalize_tool_arguments("task", {"mode": "work"}, valid_params=vp)
        assert result == {"subagent_mode": "work"}


class TestTaskProxyInternalControlKeys:
    """H1: subagent_mode / subagent_type are proxy-internal control keys that
    are NEVER injected into the client tool schema. Their paraphrase aliases
    MUST apply unconditionally regardless of valid_params so canary.py
    (_resolve_task_canary, which reads args.get("subagent_mode")) sees the
    normalized value in production.

    The REAL production `task` tool schema is {prompt, subagent_type,
    description} — it does NOT include subagent_mode (which is a proxy-internal
    control key). Without the special-case in _resolve, the aliases would pass
    through unchanged and args.get("subagent_mode") would miss them.
    """

    # The real production task schema — no subagent_mode.
    PROD_TASK_VP = frozenset({"prompt", "subagent_type", "description"})

    def test_mode_alias_applies_without_subagent_mode_in_schema(self):
        """`mode` paraphrase alias applies even when subagent_mode is NOT in
        the client schema (production reality)."""
        result = normalize_tool_arguments(
            "task", {"mode": "work"}, valid_params=self.PROD_TASK_VP
        )
        assert result == {"subagent_mode": "work"}

    def test_subagentmode_alias_applies_without_subagent_mode_in_schema(self):
        result = normalize_tool_arguments(
            "task", {"subagentmode": "work"}, valid_params=self.PROD_TASK_VP
        )
        assert result == {"subagent_mode": "work"}

    def test_agent_mode_alias_applies_without_subagent_mode_in_schema(self):
        result = normalize_tool_arguments(
            "task", {"agent_mode": "work"}, valid_params=self.PROD_TASK_VP
        )
        assert result == {"subagent_mode": "work"}

    def test_subagent_mode_camelcase_alias_applies(self):
        result = normalize_tool_arguments(
            "task", {"subagentMode": "work"}, valid_params=self.PROD_TASK_VP
        )
        assert result == {"subagent_mode": "work"}

    def test_subagent_type_paraphrase_aliases_apply(self):
        """subagent_type paraphrase aliases also apply unconditionally."""
        for alias in ("agent_type", "agent", "type"):
            result = normalize_tool_arguments(
                "task", {alias: "coder"}, valid_params=self.PROD_TASK_VP
            )
            assert result == {"subagent_type": "coder"}, f"failed for {alias!r}"

    def test_exact_subagent_mode_preserved_when_in_schema(self):
        """When subagent_mode IS in the schema (non-production test), exact
        match is preserved."""
        vp = frozenset({"prompt", "subagent_mode"})
        result = normalize_tool_arguments(
            "task", {"subagent_mode": "work"}, valid_params=vp
        )
        assert result == {"subagent_mode": "work"}

    def test_other_keys_still_schema_driven(self):
        """Non-control keys keep the schema-driven order (exact match wins)."""
        result = normalize_tool_arguments(
            "task",
            {"prompt": "hi", "mode": "work"},
            valid_params=self.PROD_TASK_VP,
        )
        assert result == {"prompt": "hi", "subagent_mode": "work"}


class TestProxyControlKeySchemaCollision:
    """M-2: the control-key override is a synonym-alias-class operation and
    MUST run AFTER the exact-schema-match check. If a future client exposes a
    `task`-like tool whose schema legitimately contains a real `type` or `mode`
    param, that exact match MUST win — the key must NOT be clobbered to the
    proxy-internal control-key name (subagent_type / subagent_mode). The alias
    only applies when the key is NOT in the schema (a paraphrase needing heal).
    """

    def test_type_not_clobbered_when_real_schema_param(self):
        """`type` IS a real schema param -> schema wins, NOT rewritten to
        subagent_type. MUST fail if the exact-match guard is reverted."""
        vp = frozenset({"prompt", "type", "description"})
        result = normalize_tool_arguments("task", {"type": "bug"}, valid_params=vp)
        assert result == {"type": "bug"}

    def test_mode_not_clobbered_when_real_schema_param(self):
        """`mode` IS a real schema param -> schema wins, NOT rewritten to
        subagent_mode. MUST fail if the exact-match guard is reverted."""
        vp = frozenset({"prompt", "mode", "description"})
        result = normalize_tool_arguments("task", {"mode": "fast"}, valid_params=vp)
        assert result == {"mode": "fast"}

    def test_mode_alias_applies_when_not_in_schema(self):
        """`mode` is NOT in schema -> alias applies (H1 fix still works)."""
        vp = frozenset({"prompt", "subagent_type", "description"})
        result = normalize_tool_arguments("task", {"mode": "work"}, valid_params=vp)
        assert result == {"subagent_mode": "work"}

    def test_type_alias_applies_with_no_schema(self):
        """valid_params=None -> alias applies (byte-identical legacy behavior)."""
        result = normalize_tool_arguments("task", {"type": "bug"}, valid_params=None)
        assert result == {"subagent_type": "bug"}
