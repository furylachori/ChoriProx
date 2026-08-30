"""
Unit tests for tool_cache module.
Tests prepare_tooling_payload, cache_tool_schema, get_cached_tool_schema.
"""

import logging

import pytest
from unittest.mock import patch, MagicMock

from proxy.tool_cache import prepare_tooling_payload


def _attach_ticket_main_capability_table(mock_skill):
    """Attach a CapabilityTable that returns the ticket MAIN injectable set.

    Slice 8b decomposed proxy_ticket into 10 fine-grained tools. The ticket
    MAIN role only sees the two read-only tools.
    """
    from proxy.skills.skill_definition import CapabilityTable, ToolCapability
    mock_skill.capability_table = CapabilityTable(entries=(
        ToolCapability("proxy_ticket_list_tasks", "main", True, True),
        ToolCapability("proxy_ticket_get_task", "main", True, True),
    ))


def _attach_ticket_subagent_capability_table(mock_skill):
    """Attach a CapabilityTable returning the ticket SUBAGENT injectable set.

    Ticket sub-agents get the FULL decomposed ticket surface (10 tools) — the
    MAIN restriction (write actions blocked, must delegate) does NOT apply.
    """
    from proxy.skills.skill_definition import CapabilityTable, ToolCapability
    mock_skill.capability_table = CapabilityTable(entries=tuple(
        ToolCapability(name, "subagent", True, True)
        for name in (
            "proxy_ticket_list_tasks", "proxy_ticket_get_task",
            "proxy_ticket_create_tasks", "proxy_ticket_update_task",
            "proxy_ticket_edit", "proxy_ticket_set_parent",
            "proxy_ticket_clear_parent", "proxy_ticket_clear_blocked_by",
            "proxy_ticket_set_priority", "proxy_ticket_append_review_note",
        )
    ))


def _attach_work_main_capability_table(mock_skill):
    """Attach a CapabilityTable that returns the work MAIN injectable set.

    Work MAIN sees the three read/append ticket tools plus proxy_work.
    """
    from proxy.skills.skill_definition import CapabilityTable, ToolCapability
    mock_skill.capability_table = CapabilityTable(entries=(
        ToolCapability("proxy_ticket_list_tasks", "main", True, True),
        ToolCapability("proxy_ticket_get_task", "main", True, True),
        ToolCapability("proxy_ticket_append_review_note", "main", True, True),
        ToolCapability("proxy_work", "main", True, True),
    ))


def _attach_ticket_only_capability_table(mock_skill):
    """Attach a CapabilityTable returning only proxy_ticket_list_tasks.

    Used for the "unknown skill with proxy_ticket fallback" regression test —
    after Slice 8b, the fallback is gone; the mock skill now declares its own
    capability_table.
    """
    from proxy.skills.skill_definition import CapabilityTable, ToolCapability
    mock_skill.capability_table = CapabilityTable(entries=(
        ToolCapability("proxy_ticket_list_tasks", "main", True, True),
    ))


def _make_ticket_skill_mock():
    """Create a mock skill that mimics TicketSkill's get_tool_filter."""
    mock_skill = MagicMock()
    mock_skill.get_tool_filter.return_value = frozenset({
        "read", "glob", "grep", "search",
        "proxy_ticket_list_tasks", "proxy_ticket_get_task",
    })
    mock_skill.name = "ticket"
    _attach_ticket_main_capability_table(mock_skill)
    return mock_skill


class TestTicketModeTools:
    """Tests for ticket mode tool filtering in prepare_tooling_payload."""

    @patch("proxy.tool_cache.get_cached_tool_schema")
    @patch("proxy.tool_cache._generate_tool_hash_with_mode")
    @patch("proxy.skills.skill_registry.SkillRegistry")
    def test_grep_emitted_in_ticket_mode(self, mock_sr_cls, mock_gen_hash, mock_get_cached):
        """prepare_tooling_payload with ticket_mode=True should emit grep."""
        mock_get_cached.return_value = None  # cache miss
        mock_gen_hash.return_value = "test_hash_12345"

        mock_registry = MagicMock()
        mock_registry.get.return_value = _make_ticket_skill_mock()
        mock_sr_cls.return_value = mock_registry

        incoming_tools = [
            {
                "type": "function",
                "function": {
                    "name": "read",
                    "description": "Read a file",
                    "parameters": {"type": "object", "properties": {}, "required": []},
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "grep",
                    "description": "Search file contents",
                    "parameters": {"type": "object", "properties": {}, "required": []},
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "bash",
                    "description": "Run shell command",
                    "parameters": {"type": "object", "properties": {}, "required": []},
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "webfetch",
                    "description": "Fetch URL",
                    "parameters": {"type": "object", "properties": {}, "required": []},
                },
            },
        ]

        body = {"tools": incoming_tools}
        model_cfg = None

        result = prepare_tooling_payload(body, model_cfg, ticket_mode=True)

        tool_names = [
            t.get("function", {}).get("name", "") for t in result.tools
        ]

        assert "grep" in tool_names, f"grep not emitted; tools={tool_names}"
        assert "bash" not in tool_names, f"bash should be excluded in ticket mode; tools={tool_names}"
        assert "proxy_ticket_list_tasks" in tool_names, (
            f"proxy_ticket_list_tasks should be appended in ticket mode; tools={tool_names}"
        )
        assert "proxy_ticket_get_task" in tool_names, (
            f"proxy_ticket_get_task should be appended in ticket mode; tools={tool_names}"
        )


_SAMPLE_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "read",
            "description": "Read a file",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "grep",
            "description": "Search file contents",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
]


class TestIsSubagentToolFiltering:
    """Tests for is_subagent parameter propagation in prepare_tooling_payload."""

    @patch("proxy.tool_cache.get_cached_tool_schema")
    @patch("proxy.tool_cache._generate_tool_hash_with_mode")
    @patch("proxy.skills.skill_registry.SkillContext")
    @patch("proxy.skills.skill_registry.SkillRegistry")
    def test_is_subagent_true_passed_to_skill_context(
        self, mock_sr_cls, mock_ctx_cls, mock_gen_hash, mock_get_cached
    ):
        mock_get_cached.return_value = None
        mock_gen_hash.return_value = "hash_sub_true"
        mock_skill = _make_ticket_skill_mock()
        mock_registry = MagicMock()
        mock_registry.get.return_value = mock_skill
        mock_sr_cls.return_value = mock_registry

        prepare_tooling_payload(
            {"tools": _SAMPLE_TOOLS}, None, ticket_mode=True, is_subagent=True
        )

        mock_ctx_cls.assert_called_once()
        assert mock_ctx_cls.call_args.kwargs["is_subagent"] is True

    @patch("proxy.tool_cache.get_cached_tool_schema")
    @patch("proxy.tool_cache._generate_tool_hash_with_mode")
    @patch("proxy.skills.skill_registry.SkillContext")
    @patch("proxy.skills.skill_registry.SkillRegistry")
    def test_is_subagent_false_passed_to_skill_context(
        self, mock_sr_cls, mock_ctx_cls, mock_gen_hash, mock_get_cached
    ):
        mock_get_cached.return_value = None
        mock_gen_hash.return_value = "hash_sub_false"
        mock_skill = _make_ticket_skill_mock()
        mock_registry = MagicMock()
        mock_registry.get.return_value = mock_skill
        mock_sr_cls.return_value = mock_registry

        prepare_tooling_payload(
            {"tools": _SAMPLE_TOOLS}, None, ticket_mode=True, is_subagent=False
        )

        mock_ctx_cls.assert_called_once()
        assert mock_ctx_cls.call_args.kwargs["is_subagent"] is False

    @patch("proxy.tool_cache.get_cached_tool_schema")
    @patch("proxy.tool_cache._generate_tool_hash_with_mode")
    @patch("proxy.skills.skill_registry.SkillContext")
    @patch("proxy.skills.skill_registry.SkillRegistry")
    def test_is_subagent_tool_filter_receives_context(
        self, mock_sr_cls, mock_ctx_cls, mock_gen_hash, mock_get_cached
    ):
        mock_get_cached.return_value = None
        mock_gen_hash.return_value = "hash_ctx_check"
        mock_skill = _make_ticket_skill_mock()
        mock_registry = MagicMock()
        mock_registry.get.return_value = mock_skill
        mock_sr_cls.return_value = mock_registry

        sentinel_ctx = MagicMock()
        sentinel_ctx.is_subagent = True
        mock_ctx_cls.return_value = sentinel_ctx

        prepare_tooling_payload(
            {"tools": _SAMPLE_TOOLS}, None, ticket_mode=True, is_subagent=True
        )

        mock_skill.get_tool_filter.assert_called_once_with(sentinel_ctx)

    @patch("proxy.tool_cache.get_cached_tool_schema")
    @patch("proxy.tool_cache._generate_tool_hash_with_mode")
    @patch("proxy.skills.skill_registry.SkillContext")
    @patch("proxy.skills.skill_registry.SkillRegistry")
    def test_is_subagent_with_skill_name(
        self, mock_sr_cls, mock_ctx_cls, mock_gen_hash, mock_get_cached
    ):
        mock_get_cached.return_value = None
        mock_gen_hash.return_value = "hash_skill_name"
        mock_skill = _make_ticket_skill_mock()
        mock_registry = MagicMock()
        mock_registry.get.return_value = mock_skill
        mock_sr_cls.return_value = mock_registry

        prepare_tooling_payload(
            {"tools": _SAMPLE_TOOLS}, None, skill_name="workspace", is_subagent=True
        )

        mock_sr_cls.return_value.get.assert_called_once_with("workspace")
        mock_ctx_cls.assert_called_once()
        assert mock_ctx_cls.call_args.kwargs["is_subagent"] is True

    @patch("proxy.tool_pipeline.tool_registry.registry")
    def test_is_subagent_no_skill_mode(self, mock_tool_reg):
        mock_tool_reg.get_default_tools.return_value = {"read", "grep"}
        with patch("proxy.model_registry.ACTIVE_CLIENT", None), \
             patch("proxy.model_registry.USE_ADAPTERS", False):
            result = prepare_tooling_payload(
                {"tools": _SAMPLE_TOOLS}, None, is_subagent=True
            )

        tool_names = [
            t.get("function", {}).get("name", "") for t in result.tools
        ]
        assert "read" in tool_names
        assert "grep" in tool_names


class TestProxyOwnedToolInjection:
    """Tests for re-injecting proxy-owned tool schemas (VFS, MEMORY) for sub-agents."""

    @patch("proxy.tool_cache.get_cached_tool_schema")
    @patch("proxy.tool_cache._generate_tool_hash_with_mode")
    @patch("proxy.model_registry.ACTIVE_CLIENT", None)
    @patch("proxy.model_registry.USE_ADAPTERS", False)
    def test_no_injection_in_default_mode(self, mock_gen_hash, mock_get_cached):
        """DEFAULT mode must NOT inject any proxy-owned tools (VFS is transparent)."""
        mock_get_cached.return_value = None
        mock_gen_hash.return_value = "hash_default"

        incoming = [
            {"type": "function", "function": {"name": "bash", "description": "shell", "parameters": {"type": "object", "properties": {}}}},
            {"type": "function", "function": {"name": "MiniMax_web_search", "description": "search", "parameters": {"type": "object", "properties": {}}}},
        ]
        with patch("proxy.tool_pipeline.tool_registry.registry") as mock_reg:
            mock_reg.get_default_tools.return_value = {"bash", "MiniMax_web_search"}
            mock_reg.get_passthrough_tools.return_value = set()
            mock_reg._tools = {
                "proxy_ticket_list_tasks": MagicMock(available_in_skills=frozenset({"ticket"})),
            }
            result = prepare_tooling_payload({"tools": incoming}, None)

        names = [t.get("function", {}).get("name", "") for t in result.tools]
        assert "bash" in names
        assert "MiniMax_web_search" in names
        assert "proxy_ticket_list_tasks" not in names

    @patch("proxy.tool_cache.get_cached_tool_schema")
    @patch("proxy.tool_cache._generate_tool_hash_with_mode")
    @patch("proxy.skills.skill_registry.SkillRegistry")
    def test_inject_proxy_ticket_only_for_skill_mode(self, mock_sr_cls, mock_gen_hash, mock_get_cached):
        """SKILL mode (ticket) should inject the decomposed proxy_ticket_* tools."""
        mock_get_cached.return_value = None
        mock_gen_hash.return_value = "hash_skill"

        mock_skill = _make_ticket_skill_mock()
        mock_registry = MagicMock()
        mock_registry.get.return_value = mock_skill
        mock_sr_cls.return_value = mock_registry

        incoming = [
            {"type": "function", "function": {"name": "read", "description": "r", "parameters": {"type": "object", "properties": {}}}},
            {"type": "function", "function": {"name": "grep", "description": "g", "parameters": {"type": "object", "properties": {}}}},
            {"type": "function", "function": {"name": "search", "description": "s", "parameters": {"type": "object", "properties": {}}}},
        ]
        with patch("proxy.tool_pipeline.tool_registry.registry") as mock_reg:
            mock_reg._tools = {
                "proxy_ticket_list_tasks": MagicMock(available_in_skills=frozenset({"ticket"})),
                "proxy_ticket_get_task": MagicMock(available_in_skills=frozenset({"ticket"})),
            }
            result = prepare_tooling_payload({"tools": incoming}, None, skill_name="ticket")

        names = [t.get("function", {}).get("name", "") for t in result.tools]
        assert "proxy_ticket_list_tasks" in names
        assert "proxy_ticket_get_task" in names
        assert "proxy_ticket" not in names, (
            f"legacy monolithic proxy_ticket should not be injected; got {names}"
        )

    @patch("proxy.tool_cache.get_cached_tool_schema")
    @patch("proxy.tool_cache._generate_tool_hash_with_mode")
    @patch("proxy.skills.skill_registry.SkillRegistry")
    def test_subagent_ticket_mode_inherits_proxy_ticket(
        self, mock_sr_cls, mock_gen_hash, mock_get_cached
    ):
        """REGRESSION (subagent-ticket-tool): a sub-agent spawned from a parent
        in ticket mode (Phase 5 structured path: is_subagent=True, ticket_mode=True)
        must receive the decomposed proxy_ticket_* tool schemas in its
        augmented tool list.

        This is the exact parameter shape that ContextOptimizeStage passes to
        prepare_tooling_payload for a sub-agent in skill mode after resolve_mode
        activates the inherited ticket skill. If the proxy_ticket_* tools are
        missing here, the sub-agent cannot manage tasks.
        """
        mock_get_cached.return_value = None
        mock_gen_hash.return_value = "hash_subagent_ticket"

        mock_skill = MagicMock()
        mock_skill.get_tool_filter.return_value = frozenset({
            "read", "glob", "grep", "search",
            "proxy_ticket_list_tasks", "proxy_ticket_get_task",
            "proxy_work",
            "task", "webfetch",
        })
        mock_skill.name = "ticket"
        _attach_ticket_subagent_capability_table(mock_skill)
        mock_registry = MagicMock()
        mock_registry.get.return_value = mock_skill
        mock_sr_cls.return_value = mock_registry

        incoming = [
            {"type": "function", "function": {"name": "read", "description": "r", "parameters": {"type": "object", "properties": {}}}},
            {"type": "function", "function": {"name": "grep", "description": "g", "parameters": {"type": "object", "properties": {}}}},
            {"type": "function", "function": {"name": "search", "description": "s", "parameters": {"type": "object", "properties": {}}}},
            {"type": "function", "function": {"name": "bash", "description": "b", "parameters": {"type": "object", "properties": {}}}},
        ]
        with patch("proxy.tool_pipeline.tool_registry.registry") as mock_reg:
            mock_reg._tools = {
                "proxy_ticket_list_tasks": MagicMock(available_in_skills=frozenset({"ticket"})),
                "proxy_ticket_get_task": MagicMock(available_in_skills=frozenset({"ticket"})),
                "search": MagicMock(available_in_skills=frozenset({"ticket"})),
            }
            result = prepare_tooling_payload(
                {"tools": incoming}, None,
                ticket_mode=True, is_subagent=True, skill_name="ticket",
                session_id="sess_sub", agent_id="agent_sub",
            )

        names = [t.get("function", {}).get("name", "") for t in result.tools]
        assert "proxy_ticket_list_tasks" in names, (
            f"REGRESSION: sub-agent in ticket mode did not receive "
            f"proxy_ticket_list_tasks; got tools={names}"
        )
        assert "proxy_ticket_get_task" in names, (
            f"REGRESSION: sub-agent in ticket mode did not receive "
            f"proxy_ticket_get_task; got tools={names}"
        )
        assert "bash" not in names, (
            f"REGRESSION: bash should be filtered out in ticket mode; got tools={names}"
        )
        assert "read" in names, f"expected read in tools={names}"
        assert "grep" in names, f"expected grep in tools={names}"

    @patch("proxy.tool_cache.get_cached_tool_schema")
    @patch("proxy.tool_cache._generate_tool_hash_with_mode")
    @patch("proxy.skills.skill_registry.SkillRegistry")
    def test_no_duplicate_injection(self, mock_sr_cls, mock_gen_hash, mock_get_cached):
        """If a proxy_ticket_* tool is already in incoming tools, don't add it again in SKILL mode."""
        mock_get_cached.return_value = None
        mock_gen_hash.return_value = "hash_dup"

        mock_skill = _make_ticket_skill_mock()
        mock_registry = MagicMock()
        mock_registry.get.return_value = mock_skill
        mock_sr_cls.return_value = mock_registry

        # Mock the skill filter to KEEP the decomposed proxy_ticket tools in the
        # initial filtered set so the dedup branch is exercised.
        mock_skill.get_tool_filter.return_value = frozenset({
            "read", "proxy_ticket_list_tasks", "proxy_ticket_get_task",
        })

        incoming = [
            {"type": "function", "function": {"name": "read", "description": "r", "parameters": {"type": "object", "properties": {}}}},
            {"type": "function", "function": {"name": "proxy_ticket_list_tasks", "description": "t", "parameters": {"type": "object", "properties": {}}}},
            {"type": "function", "function": {"name": "proxy_ticket_get_task", "description": "t", "parameters": {"type": "object", "properties": {}}}},
        ]
        with patch("proxy.tool_pipeline.tool_registry.registry") as mock_reg:
            mock_reg._tools = {
                "proxy_ticket_list_tasks": MagicMock(available_in_skills=frozenset({"ticket"})),
                "proxy_ticket_get_task": MagicMock(available_in_skills=frozenset({"ticket"})),
            }
            result = prepare_tooling_payload({"tools": incoming}, None, skill_name="ticket")

        names = [t.get("function", {}).get("name", "") for t in result.tools]
        assert names.count("proxy_ticket_list_tasks") == 1
        assert names.count("proxy_ticket_get_task") == 1

    @patch("proxy.tool_cache.get_cached_tool_schema")
    @patch("proxy.tool_cache._generate_tool_hash_with_mode")
    @patch("proxy.model_registry.ACTIVE_CLIENT", None)
    @patch("proxy.model_registry.USE_ADAPTERS", False)
    def test_skill_restricted_tools_not_in_default(self, mock_gen_hash, mock_get_cached):
        """DEFAULT mode injects NO proxy-owned tools (VFS is transparent, proxy_ticket is skill-only)."""
        mock_get_cached.return_value = None
        mock_gen_hash.return_value = "hash_restrict"

        incoming = [
            {"type": "function", "function": {"name": "bash", "description": "shell", "parameters": {"type": "object", "properties": {}}}},
        ]
        with patch("proxy.tool_pipeline.tool_registry.registry") as mock_reg:
            mock_reg.get_default_tools.return_value = {"bash"}
            mock_reg.get_passthrough_tools.return_value = set()
            mock_reg._tools = {
                "proxy_ticket_list_tasks": MagicMock(available_in_skills=frozenset({"ticket"})),
                "search": MagicMock(available_in_skills=frozenset({"ticket"})),
            }
            result = prepare_tooling_payload({"tools": incoming}, None)

        names = [t.get("function", {}).get("name", "") for t in result.tools]
        assert "proxy_ticket_list_tasks" not in names
        assert "search" not in names


class TestInjectableMemoryToolsSkillAware:
    """Tests for skill-aware proxy_work injection (C1)."""

    @patch("proxy.tool_cache.get_cached_tool_schema")
    @patch("proxy.tool_cache._generate_tool_hash_with_mode")
    @patch("proxy.skills.skill_registry.SkillRegistry")
    def test_inject_proxy_ticket_for_default_skill_name(
        self, mock_sr_cls, mock_gen_hash, mock_get_cached
    ):
        """skill_name=None → no proxy_* injection (default mode)."""
        mock_get_cached.return_value = None
        mock_gen_hash.return_value = "hash_c1_default"
        result = prepare_tooling_payload(
            {"tools": _SAMPLE_TOOLS}, None, skill_name=None,
            session_id="sess_c1_default", agent_id="agent_c1_default",
        )
        names = [t.get("function", {}).get("name", "") for t in result.tools]
        assert "proxy_ticket" not in names, f"default mode must NOT inject proxy_ticket; got {names}"
        assert "proxy_work" not in names, f"default mode must NOT inject proxy_work; got {names}"

    @patch("proxy.tool_cache.get_cached_tool_schema")
    @patch("proxy.tool_cache._generate_tool_hash_with_mode")
    @patch("proxy.skills.skill_registry.SkillRegistry")
    def test_inject_proxy_ticket_for_ticket_skill_name(
        self, mock_sr_cls, mock_gen_hash, mock_get_cached
    ):
        """skill_name='ticket' → decomposed proxy_ticket_list_tasks/get_task ONLY, not proxy_work."""
        mock_get_cached.return_value = None
        mock_gen_hash.return_value = "hash_c1_ticket"
        mock_skill = _make_ticket_skill_mock()
        mock_registry = MagicMock()
        mock_registry.get.return_value = mock_skill
        mock_sr_cls.return_value = mock_registry
        with patch("proxy.tool_pipeline.tool_registry.registry") as mock_reg:
            mock_reg._tools = {
                "proxy_ticket_list_tasks": MagicMock(available_in_skills=frozenset({"ticket"})),
                "proxy_ticket_get_task": MagicMock(available_in_skills=frozenset({"ticket"})),
                "proxy_work": MagicMock(available_in_skills=frozenset({"work"})),
            }
            result = prepare_tooling_payload(
                {"tools": _SAMPLE_TOOLS}, None,
                skill_name="ticket",
                session_id="sess_c1_ticket", agent_id="agent_c1_ticket",
            )
        names = [t.get("function", {}).get("name", "") for t in result.tools]
        assert "proxy_ticket_list_tasks" in names, (
            f"ticket mode must inject proxy_ticket_list_tasks; got {names}"
        )
        assert "proxy_ticket_get_task" in names, (
            f"ticket mode must inject proxy_ticket_get_task; got {names}"
        )
        assert "proxy_work" not in names, f"ticket mode must NOT inject proxy_work; got {names}"

    @patch("proxy.tool_cache.get_cached_tool_schema")
    @patch("proxy.tool_cache._generate_tool_hash_with_mode")
    @patch("proxy.skills.skill_registry.SkillRegistry")
    def test_inject_proxy_work_only_in_work_skill_name(
        self, mock_sr_cls, mock_gen_hash, mock_get_cached
    ):
        """skill_name='work' → BOTH decomposed ticket tools AND proxy_work injected."""
        mock_get_cached.return_value = None
        mock_gen_hash.return_value = "hash_c1_work"
        mock_skill = MagicMock()
        mock_skill.get_tool_filter.return_value = frozenset({
            "read", "write", "edit", "bash", "glob", "grep", "search",
            "task", "proxy_work",
            "proxy_ticket_list_tasks", "proxy_ticket_get_task",
            "proxy_ticket_append_review_note",
        })
        mock_skill.name = "work"
        _attach_work_main_capability_table(mock_skill)
        mock_registry = MagicMock()
        mock_registry.get.return_value = mock_skill
        mock_sr_cls.return_value = mock_registry
        with patch("proxy.tool_pipeline.tool_registry.registry") as mock_reg:
            mock_reg._tools = {
                "proxy_ticket_list_tasks": MagicMock(available_in_skills=frozenset({"ticket", "work"})),
                "proxy_ticket_get_task": MagicMock(available_in_skills=frozenset({"ticket", "work"})),
                "proxy_ticket_append_review_note": MagicMock(available_in_skills=frozenset({"work"})),
                "proxy_work": MagicMock(available_in_skills=frozenset({"work"})),
            }
            result = prepare_tooling_payload(
                {"tools": _SAMPLE_TOOLS}, None,
                skill_name="work",
                session_id="sess_c1_work", agent_id="agent_c1_work",
            )
        names = [t.get("function", {}).get("name", "") for t in result.tools]
        assert "proxy_ticket_list_tasks" in names, (
            f"work mode must inject proxy_ticket_list_tasks; got {names}"
        )
        assert "proxy_ticket_get_task" in names, (
            f"work mode must inject proxy_ticket_get_task; got {names}"
        )
        assert "proxy_work" in names, f"work mode must inject proxy_work; got {names}"

    @patch("proxy.tool_cache.get_cached_tool_schema")
    @patch("proxy.tool_cache._generate_tool_hash_with_mode")
    @patch("proxy.skills.skill_registry.SkillRegistry")
    def test_inject_proxy_work_for_unknown_skill_falls_back_to_ticket_only(
        self, mock_sr_cls, mock_gen_hash, mock_get_cached
    ):
        """skill_name='build' (unknown) with capability_table → proxy_ticket_list_tasks ONLY.

        Slice 8b removed the auto-fallback that injected proxy_ticket for any
        unknown skill. The mock skill now declares its own capability_table
        with proxy_ticket_list_tasks visible; the contract is "the capability
        table is the source of truth, and proxy_work must NOT be injected for
        a non-work skill".
        """
        mock_get_cached.return_value = None
        mock_gen_hash.return_value = "hash_c1_unknown"
        mock_skill = MagicMock()
        mock_skill.get_tool_filter.return_value = frozenset({
            "read", "write", "edit", "bash", "glob", "grep", "search",
        })
        mock_skill.name = "build"
        _attach_ticket_only_capability_table(mock_skill)
        mock_registry = MagicMock()
        mock_registry.get.return_value = mock_skill
        mock_sr_cls.return_value = mock_registry
        with patch("proxy.tool_pipeline.tool_registry.registry") as mock_reg:
            mock_reg._tools = {
                "proxy_ticket_list_tasks": MagicMock(available_in_skills=frozenset({"ticket", "build"})),
                "proxy_work": MagicMock(available_in_skills=frozenset({"work"})),
            }
            result = prepare_tooling_payload(
                {"tools": _SAMPLE_TOOLS}, None,
                skill_name="build",
                session_id="sess_c1_unknown", agent_id="agent_c1_unknown",
            )
        names = [t.get("function", {}).get("name", "") for t in result.tools]
        assert "proxy_ticket_list_tasks" in names, (
            f"capability_table declares proxy_ticket_list_tasks; got {names}"
        )
        assert "proxy_work" not in names, (
            f"unknown skill must NOT inject proxy_work; got {names}"
        )

    @patch("proxy.tool_cache.get_cached_tool_schema")
    @patch("proxy.tool_cache._generate_tool_hash_with_mode")
    @patch("proxy.skills.skill_registry.SkillRegistry")
    def test_cache_bleed_between_ticket_and_work_skill_names(
        self, mock_sr_cls, mock_gen_hash, mock_get_cached
    ):
        """Cache key must isolate per-skill_name: priming with work must not bleed into ticket."""
        from proxy.tool_cache import clear_tool_cache
        clear_tool_cache()
        mock_get_cached.return_value = None
        mock_gen_hash.return_value = "hash_c1_bleed_default"

        mock_skill_ticket = _make_ticket_skill_mock()
        mock_skill_work = MagicMock()
        mock_skill_work.get_tool_filter.return_value = frozenset({
            "read", "write", "edit", "bash", "glob", "grep", "search",
            "task", "proxy_work",
            "proxy_ticket_list_tasks", "proxy_ticket_get_task",
            "proxy_ticket_append_review_note",
        })
        mock_skill_work.name = "work"
        _attach_work_main_capability_table(mock_skill_work)
        mock_registry = MagicMock()
        mock_registry.get.side_effect = lambda name: mock_skill_work if name == "work" else mock_skill_ticket
        mock_sr_cls.return_value = mock_registry

        with patch("proxy.tool_pipeline.tool_registry.registry") as mock_reg:
            mock_reg._tools = {
                "proxy_ticket_list_tasks": MagicMock(available_in_skills=frozenset({"ticket", "work"})),
                "proxy_ticket_get_task": MagicMock(available_in_skills=frozenset({"ticket", "work"})),
                "proxy_ticket_append_review_note": MagicMock(available_in_skills=frozenset({"work"})),
                "proxy_work": MagicMock(available_in_skills=frozenset({"work"})),
            }

            work_result = prepare_tooling_payload(
                {"tools": _SAMPLE_TOOLS}, None,
                skill_name="work",
                session_id="sess_c1_work_bleed", agent_id="agent_c1_work_bleed",
            )
            work_names = [t.get("function", {}).get("name", "") for t in work_result.tools]
            assert "proxy_work" in work_names, f"work prime must inject proxy_work; got {work_names}"

            ticket_result = prepare_tooling_payload(
                {"tools": _SAMPLE_TOOLS}, None,
                skill_name="ticket",
                session_id="sess_c1_ticket_bleed", agent_id="agent_c1_ticket_bleed",
            )
            ticket_names = [t.get("function", {}).get("name", "") for t in ticket_result.tools]
            assert "proxy_ticket_list_tasks" in ticket_names, (
                f"ticket must inject proxy_ticket_list_tasks; got {ticket_names}"
            )
            assert "proxy_work" not in ticket_names, (
                f"CACHE BLEED: ticket-mode call returned proxy_work; "
                f"cache key is not isolating per-skill_name. got {ticket_names}"
            )


# ---------------------------------------------------------------------------
# H4: tool_cache passthrough fallback must catch AttributeError too
# ---------------------------------------------------------------------------

class TestPassthroughFallbackAttributeError:
    """H4: when `_tool_reg.get_passthrough_tools()` raises AttributeError
    (e.g. typo, missing method, partial-init mock), tool_cache must fall
    back to the hardcoded set and log a WARNING.
    """

    def test_passthrough_fallback_on_attribute_error_logs_warning(self, caplog):
        """AttributeError from get_passthrough_tools triggers hardcoded fallback + WARNING."""
        import importlib
        import logging
        import proxy.tool_cache as tool_cache_mod

        # Force the `from proxy.tool_pipeline.tool_registry import registry`
        # to succeed (use a stub module that provides `registry`) so we can
        # make `registry.get_passthrough_tools` raise AttributeError.

        class _StubToolRegistry:
            @staticmethod
            def get_passthrough_tools():
                raise AttributeError(
                    "module 'proxy.tool_pipeline.tool_registry' has no attribute 'get_passthrough_tools'"
                )

        stub_module = MagicMock()
        stub_module.registry = _StubToolRegistry()

        with patch.dict("sys.modules", {
            "proxy.tool_pipeline.tool_registry": stub_module,
        }):
            with caplog.at_level(logging.WARNING, logger="proxy.tool_cache"):
                importlib.reload(tool_cache_mod)

        # Hardcoded fallback set should be in use
        expected_fallback = {
            "todowrite", "ask", "question", "new_task",
            "TaskCreateTool", "TaskUpdateTool",
        }
        assert tool_cache_mod.PASSTHROUGH_TOOL_NAMES == expected_fallback

        # A WARNING should have been logged with the exception type/name
        warning_records = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert any("get_passthrough_tools" in r.getMessage() for r in warning_records), (
            f"Expected a WARNING mentioning get_passthrough_tools, got: "
            f"{[r.getMessage() for r in warning_records]}"
        )
        assert any("AttributeError" in r.getMessage() for r in warning_records), (
            "Expected WARNING to include the AttributeError type"
        )

    def test_passthrough_fallback_on_import_error_logs_warning(self, caplog):
        """ImportError fallback path also logs a WARNING (regression)."""
        import importlib
        import logging
        import proxy.tool_cache as tool_cache_mod

        # Make the inner import raise ImportError by hiding the module.
        with patch.dict("sys.modules", {
            "proxy.tool_pipeline.tool_registry": None,  # forces ImportError
        }):
            with caplog.at_level(logging.WARNING, logger="proxy.tool_cache"):
                importlib.reload(tool_cache_mod)

        expected_fallback = {
            "todowrite", "ask", "question", "new_task",
            "TaskCreateTool", "TaskUpdateTool",
        }
        assert tool_cache_mod.PASSTHROUGH_TOOL_NAMES == expected_fallback

        warning_records = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert any("ModuleNotFoundError" in r.getMessage() for r in warning_records), (
            "Expected WARNING to mention ModuleNotFoundError (subclass of ImportError)"
        )

    def test_passthrough_no_fallback_on_success(self, caplog):
        """Sanity: when get_passthrough_tools returns normally, no WARNING fires
        and the returned set is used (not the hardcoded fallback)."""
        import importlib
        import logging
        import proxy.tool_cache as tool_cache_mod

        class _StubToolRegistry:
            @staticmethod
            def get_passthrough_tools():
                return {"custom_passthrough_a", "custom_passthrough_b"}

        stub_module = MagicMock()
        stub_module.registry = _StubToolRegistry()

        with patch.dict("sys.modules", {
            "proxy.tool_pipeline.tool_registry": stub_module,
        }):
            with caplog.at_level(logging.WARNING, logger="proxy.tool_cache"):
                importlib.reload(tool_cache_mod)

        assert tool_cache_mod.PASSTHROUGH_TOOL_NAMES == {
            "custom_passthrough_a", "custom_passthrough_b",
        }
        # No warning when the call succeeds
        warning_records = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert not any("get_passthrough_tools" in r.getMessage() for r in warning_records)

    def test_passthrough_fallback_on_wrong_return_type(self, caplog):
        """If get_passthrough_tools returns None (or any non-collection type),
        tool_cache must fall back to the hardcoded set rather than crashing
        downstream when the value is unioned with OPENCODE_TOOL_NAMES."""
        import importlib
        import logging
        import proxy.tool_cache as tool_cache_mod

        class _StubToolRegistry:
            @staticmethod
            def get_passthrough_tools():
                return None  # invalid return type

        stub_module = MagicMock()
        stub_module.registry = _StubToolRegistry()

        with patch.dict("sys.modules", {
            "proxy.tool_pipeline.tool_registry": stub_module,
        }):
            with caplog.at_level(logging.WARNING, logger="proxy.tool_cache"):
                importlib.reload(tool_cache_mod)

        expected_fallback = {
            "todowrite", "ask", "question", "new_task",
            "TaskCreateTool", "TaskUpdateTool",
        }
        assert tool_cache_mod.PASSTHROUGH_TOOL_NAMES == expected_fallback

        # WARNING should be logged mentioning the wrong return type
        warning_records = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert any("get_passthrough_tools" in r.getMessage() for r in warning_records), (
            f"Expected a WARNING mentioning get_passthrough_tools, got: "
            f"{[r.getMessage() for r in warning_records]}"
        )

    def test_passthrough_accepts_list_return_value(self, caplog):
        """A list return is accepted (defensive) and converted to a set."""
        import importlib
        import logging
        import proxy.tool_cache as tool_cache_mod

        class _StubToolRegistry:
            @staticmethod
            def get_passthrough_tools():
                return ["alpha", "beta", "gamma"]

        stub_module = MagicMock()
        stub_module.registry = _StubToolRegistry()

        with patch.dict("sys.modules", {
            "proxy.tool_pipeline.tool_registry": stub_module,
        }):
            with caplog.at_level(logging.WARNING, logger="proxy.tool_cache"):
                importlib.reload(tool_cache_mod)

        assert tool_cache_mod.PASSTHROUGH_TOOL_NAMES == {"alpha", "beta", "gamma"}
        warning_records = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert not any("get_passthrough_tools" in r.getMessage() for r in warning_records)


# ---------------------------------------------------------------------------
# Regression: default-mode cache-hit must not reject small-but-valid toolsets
# ---------------------------------------------------------------------------

def _make_valid_tool(name, description="d"):
    """Build a tool dict that passes validate_tool_schema."""
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {"type": "object", "properties": {}},
        },
    }


class TestSmallToolsetCacheHit:
    """Regression tests for the default-mode cache-hit path.

    The cache hash is content-derived (SHA-256[:16] of the canonicalized
    filtered tools), so a cache hit guarantees byte-identical content and any
    "suspiciously small" size heuristic can only produce false positives. These
    tests pin that small-but-valid toolsets are returned silently on hit.
    """

    def test_small_toolset_cache_hit_returns_cached_and_no_warning(self, caplog):
        """Pin the fix for the "suspiciously small" false-positive.

        Pre-fix: a 5-tool cached hit was invalidated with a WARNING because the
        old threshold ``max(3, len(get_default_tools())//2)`` computed to 7
        (15 defaults // 2), and 5 < 7 rejected the hit. Post-fix the hit is
        returned silently.

        Reproduction note: ``get_default_tools()`` is mocked to 15 names so the
        pre-fix threshold would be ``max(3, 15//2) == 7`` and the 5-tool hit
        (5 < 7) reproduces the rejection. The filter still admits only the 5
        fixture tools (the incoming body only contains those 5), so
        ``augmented_tools`` has 5 entries and is cached under its content hash.
        """
        from proxy.tool_cache import clear_tool_cache
        clear_tool_cache()

        fixture_names = ["bash", "glob", "grep", "read", "webfetch"]
        incoming = [_make_valid_tool(n) for n in fixture_names]
        # 15 default names: the 5 fixture names plus 10 dummies, so the old
        # threshold max(3, 15//2) == 7 rejects the 5-tool hit (5 < 7).
        default_names = set(fixture_names) | {f"t{i}" for i in range(10)}
        assert len(default_names) == 15

        with patch("proxy.tool_pipeline.tool_registry.registry") as mock_reg, \
             patch("proxy.model_registry.ACTIVE_CLIENT", None), \
             patch("proxy.model_registry.USE_ADAPTERS", False):
            mock_reg.get_default_tools.return_value = default_names
            mock_reg.get_passthrough_tools.return_value = set()
            mock_reg._tools = {}

            # Prime the cache.
            result1 = prepare_tooling_payload({"tools": incoming}, None)
            assert result1.cached is False

            caplog.clear()
            caplog.set_level(logging.WARNING, logger="proxy.tool_cache")

            result2 = prepare_tooling_payload({"tools": incoming}, None)

        assert result2.cached is True
        names2 = [t.get("function", {}).get("name", "") for t in result2.tools]
        assert set(names2) == set(fixture_names)

        warnings = [
            r for r in caplog.records
            if r.name == "proxy.tool_cache" and r.levelno >= logging.WARNING
        ]
        assert not warnings, (
            f"Expected no WARNING from proxy.tool_cache on cache hit, got: "
            f"{[(r.levelname, r.getMessage()) for r in warnings]}"
        )

    def test_two_tool_cache_hit_returns_cached(self, caplog):
        """Pin the floor: a 2-tool cached hit must be returned.

        The original da56b1a stale-entry was 2 tools; we must not re-introduce a
        threshold that rejects small-but-valid hits. With 15 defaults, any "half
        of default" threshold would reject 2.
        """
        from proxy.tool_cache import clear_tool_cache
        clear_tool_cache()

        fixture_names = ["bash", "MiniMax_web_search"]
        incoming = [_make_valid_tool(n) for n in fixture_names]
        default_names = set(fixture_names) | {f"d{i}" for i in range(13)}
        assert len(default_names) == 15

        with patch("proxy.tool_pipeline.tool_registry.registry") as mock_reg, \
             patch("proxy.model_registry.ACTIVE_CLIENT", None), \
             patch("proxy.model_registry.USE_ADAPTERS", False):
            mock_reg.get_default_tools.return_value = default_names
            mock_reg.get_passthrough_tools.return_value = set()
            mock_reg._tools = {}

            result1 = prepare_tooling_payload({"tools": incoming}, None)
            assert result1.cached is False

            caplog.clear()
            caplog.set_level(logging.WARNING, logger="proxy.tool_cache")

            result2 = prepare_tooling_payload({"tools": incoming}, None)

        assert result2.cached is True
        names2 = [t.get("function", {}).get("name", "") for t in result2.tools]
        assert set(names2) == set(fixture_names)

        warnings = [
            r for r in caplog.records
            if r.name == "proxy.tool_cache" and r.levelno >= logging.WARNING
        ]
        assert not warnings, (
            f"Expected no WARNING from proxy.tool_cache on cache hit, got: "
            f"{[(r.levelname, r.getMessage()) for r in warnings]}"
        )
