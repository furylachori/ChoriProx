"""Unified ToolRegistry — single source of truth for all tool names and categories.

Single source of truth for all tool names and categories.
All separate allowlists were deleted in Tier 3 Phase 3.

The public catalog API (`get_tool_names`, `get`, `is_proxy_admin_tool`) is
the single source of truth that the custom-skill loader (Agent 2) uses to
validate `tool_whitelist` entries. There is no second static list of valid
tool names — see `docs/specs/custom-skills-remediation.md` §6.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional


class ToolCategory(Enum):
    """Categories for tool classification."""
    EDITOR = auto()       # bash, read, glob, edit, write, grep, suggest
    WEB = auto()          # webfetch
    MEMORY = auto()       # recall, search, proxy_ticket
    TASK = auto()         # task
    PASSTHROUGH = auto()  # todowrite, ask, new_task, TaskCreateTool, TaskUpdateTool


@dataclass(frozen=True)
class ToolMeta:
    """Immutable metadata for a single tool."""
    name: str
    category: ToolCategory
    available_in_skills: Optional[frozenset[str]] = None  # None = available in all skills
    is_passthrough: bool = False


class ToolRegistry:
    """Registry of all known tools.

    Usage:
        registry = ToolRegistry()
        registry.register(ToolMeta("bash", ToolCategory.EDITOR))
        ...
        registry.get_default_tools()       # non-passthrough tool names
        registry.get_passthrough_tools()   # passthrough tool names
        registry.get_memory_tool_names()   # MEMORY category tool names
        registry.get_tools_for_skill(skill_name)  # tools for a skill
    """

    def __init__(self) -> None:
        self._tools: dict[str, ToolMeta] = {}

    def register(self, meta: ToolMeta) -> None:
        """Register a tool. Overwrites if already registered."""
        self._tools[meta.name] = meta

    def get_memory_tool_names(self) -> set[str]:
        """Return tool names in the MEMORY category."""
        return {n for n, m in self._tools.items() if m.category == ToolCategory.MEMORY}

    def get_tools_for_skill(self, skill_name: str) -> set[str]:
        """Return tool names available for a specific skill.
        
        Formula: tools where available_in_skills is None OR contains skill_name
        """
        return {
            n for n, m in self._tools.items()
            if m.available_in_skills is None or skill_name in m.available_in_skills
        }

    def get_default_tools(self) -> set[str]:
        """Return tool names for default (non-passthrough) LLM tool set.

        Excludes proxy_* admin tools (``proxy_ticket``, ``proxy_work``)
        which require a skill context to be safe — bare sub-agents (no
        canary skill) must not see them in their tool list or the tool
        guard will block them at execution time.

        Other tools with ``available_in_skills`` set (e.g. ``read``,
        ``glob``, ``grep``, ``webfetch``, ``search``) ARE still included
        here — being in a skill filter is orthogonal to being available
        in default mode. Skills can further restrict via
        ``SkillRegistry.get_tool_filter``.
        """
        return {
            n for n, m in self._tools.items()
            if not m.is_passthrough and n not in _PROXY_ADMIN_TOOLS
        }

    def get_passthrough_tools(self) -> set[str]:
        """Return passthrough tool names (client-specific, not sent to LLM)."""
        return {n for n, m in self._tools.items() if m.is_passthrough}

    def get_tool_names(self) -> frozenset[str]:
        """Return the canonical set of tool names known to this registry.

        Single-source-of-truth view of EVERY registered tool — default
        tools, passthrough tools, admin tools, and any future categories.
        The custom-skill loader (Agent 2) validates every
        ``tool_whitelist`` entry against this set; there is NO second
        static list of valid tool names.

        The returned frozenset is a stable snapshot of ``self._tools``
        keys at call time. Mutations to the registry after this call
        are NOT reflected in the returned value.
        """
        return frozenset(self._tools.keys())

    def get(self, name: str) -> Optional[ToolMeta]:
        """Look up a tool by name. Returns ``None`` if not registered.

        Public API — replaces the prior ``hasattr(tool_registry, 'get')``
        pattern in `proxy/skill_tools.py` (the registry now exposes
        ``get`` unconditionally). The loader (Agent 2) also calls this
        to validate ``tool_whitelist`` entries: an unknown name is a
        hard reject (see `docs/specs/custom-skills-remediation.md` §4.4).
        """
        return self._tools.get(name)

    def is_proxy_admin_tool(self, name: str) -> bool:
        """Check whether ``name`` is a proxy admin tool.

        Proxy admin tools are ``proxy_ticket`` / ``proxy_work`` (see
        ``_PROXY_ADMIN_TOOLS``). They require skill context to be safe:
        bare sub-agents (no canary skill) MUST NOT see them in their
        LLM-facing tool list — ``get_default_tools()`` excludes them
        and ``MemoryToolGuard._check_hierarchy`` blocks them at
        execution time. The loader (Agent 2) rejects them in custom
        JSON ``tool_whitelist`` entries regardless of author intent
        (spec §4.4).
        """
        return name in _PROXY_ADMIN_TOOLS


# Tools that require skill context to be safe. Bare sub-agents (no canary
# skill) must not see these in their LLM-facing tool list.
_PROXY_ADMIN_TOOLS = frozenset({
    "proxy_ticket", "proxy_work",
    "proxy_ticket_list_tasks",
    "proxy_ticket_get_task",
    "proxy_ticket_create_tasks",
    "proxy_ticket_update_task",
    "proxy_ticket_edit",
    "proxy_ticket_set_parent",
    "proxy_ticket_clear_parent",
    "proxy_ticket_clear_blocked_by",
    "proxy_ticket_set_priority",
    "proxy_ticket_append_review_note",
})


# ---------------------------------------------------------------------------
# Global singleton — initialised at import time
# ---------------------------------------------------------------------------
registry = ToolRegistry()

# EDITOR tools
registry.register(ToolMeta("bash",    ToolCategory.EDITOR))
registry.register(ToolMeta("read",     ToolCategory.EDITOR,       available_in_skills=frozenset({"ticket", "work"})))
registry.register(ToolMeta("glob",     ToolCategory.EDITOR,       available_in_skills=frozenset({"ticket", "work"})))
registry.register(ToolMeta("edit",     ToolCategory.EDITOR))
registry.register(ToolMeta("write",    ToolCategory.EDITOR))
registry.register(ToolMeta("grep",     ToolCategory.EDITOR,       available_in_skills=frozenset({"ticket", "work"})))
registry.register(ToolMeta("suggest",  ToolCategory.EDITOR))

# WEB tools
registry.register(ToolMeta("webfetch", ToolCategory.WEB,           available_in_skills=frozenset({"ticket"})))
registry.register(ToolMeta("MiniMax_web_search", ToolCategory.WEB))

# MEMORY tools
registry.register(ToolMeta("recall",   ToolCategory.MEMORY))
registry.register(ToolMeta("search",   ToolCategory.MEMORY,       available_in_skills=frozenset({"ticket"})))
registry.register(ToolMeta("proxy_ticket", ToolCategory.MEMORY))
registry.register(ToolMeta("proxy_work", ToolCategory.MEMORY, available_in_skills=frozenset({"ticket", "work"})))

# Decomposed ticket tools (Slice 8a — dormant until 8b)
for _tname in (
    "proxy_ticket_list_tasks",
    "proxy_ticket_get_task",
    "proxy_ticket_create_tasks",
    "proxy_ticket_update_task",
    "proxy_ticket_edit",
    "proxy_ticket_set_parent",
    "proxy_ticket_clear_parent",
    "proxy_ticket_clear_blocked_by",
    "proxy_ticket_set_priority",
    "proxy_ticket_append_review_note",
):
    registry.register(ToolMeta(_tname, ToolCategory.MEMORY, available_in_skills=frozenset({"ticket", "work"})))

# TASK tools
registry.register(ToolMeta("task",    ToolCategory.TASK))

# PASSTHROUGH tools
registry.register(ToolMeta("todowrite",   ToolCategory.PASSTHROUGH, is_passthrough=True, available_in_skills=frozenset({"ticket"})))
registry.register(ToolMeta("ask",         ToolCategory.PASSTHROUGH, is_passthrough=True))
registry.register(ToolMeta("question",    ToolCategory.PASSTHROUGH, is_passthrough=True, available_in_skills=frozenset()))
registry.register(ToolMeta("new_task",    ToolCategory.PASSTHROUGH, is_passthrough=True))
registry.register(ToolMeta("TaskCreateTool",  ToolCategory.PASSTHROUGH, is_passthrough=True))
registry.register(ToolMeta("TaskUpdateTool",  ToolCategory.PASSTHROUGH, is_passthrough=True))
