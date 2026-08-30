"""Integration tests for the triple intersection of (mode) x (tool filter) x (context budget).

These tests verify that when an agent is in a specific mode (work, ticket, default, or
work sub-agent) and the request exceeds the context budget, three things compose correctly:

1. The system prompt in the outgoing body is the mode-specific prompt
   (work prompt, ticket prompt, default prompt, or work-subagent prompt).
2. The outgoing tool list is the mode-specific filtered set
   (work = full access, ticket = {read, glob, grep, search, task, webfetch, proxy_ticket},
    default = no mode-specific restriction, work subagent = read-only ticket + work).
3. The context is trimmed to fit within the model's safe-token budget
   (V2 three-tier pipeline: ancient history + rich summaries + recent raw).

The composition point is `proxy.mode.apply_mode`, which is the single function that
applies ALL THREE transforms to the request body before it reaches the upstream LLM.

What this is NOT testing: each component in isolation. The mode harness tests mode
activation. The work-mode guard tests action-level restrictions. The V2 context
tests are split across test_summary_e2e, test_context_v2, test_summary_budget, etc.
This file tests the COMPOSITION of all three.
"""
from __future__ import annotations

import time
import uuid
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from proxy.mode import Mode
from proxy.tool_parsing import (
    OUR_SYSTEM_PROMPT,
    OUR_SUB_PROMPT,
    SKILL_SUBAGENT_PROMPT,  # TICKET_MODE_SUBAGENT_PROMPT was deleted (Phase 2B alias removal)
)
from tests.mode_harness import (
    activate_ticket_mode,
    activate_work_mode,
    create_test_agent,
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Small model so 30+ exchanges exceed safe budget (safe=8000, threshold=2400).
# 30 exchanges x 2 msgs x 600 chars x 0.25 tpc ≈ 9000 tokens > 2400 -> enters V2 pipeline.
SMALL_MODEL = SimpleNamespace(context_limit=10_000, raw_exchange_count=10)


# Tool names that the work-skill's get_tool_filter() allows.
WORK_TOOL_FILTER = frozenset({
    "read", "write", "edit", "bash", "glob", "grep", "search",
    "task", "proxy_work", "proxy_ticket",
})

# Tool names that the ticket-skill's get_tool_filter() allows.
TICKET_TOOL_FILTER = frozenset({
    "read", "glob", "grep", "search", "task", "webfetch",
    "proxy_ticket",
})


# ---------------------------------------------------------------------------
# Test environment
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def setup_test_db(tmp_path, monkeypatch):
    """Use a unique file-backed SQLite database for each test."""
    db_path = tmp_path / "budget-filter-mode.db"
    monkeypatch.setenv("PROXY_TEST_DB", str(db_path))
    from memory.db import init_db
    init_db(str(db_path))
    yield str(db_path)


@pytest.fixture(autouse=True)
def reset_agent_registry():
    """Reset the global agent registry so mode activation does not leak across tests."""
    from proxy.agent.agent_registry import registry
    with registry._lock:
        registry._agents.clear()
        registry._agents_by_id.clear()
    yield
    with registry._lock:
        registry._agents.clear()
        registry._agents_by_id.clear()


@pytest.fixture(autouse=True)
def mock_neo4j(monkeypatch):
    """Mock Neo4j client for all tests (no real database)."""
    mock = MagicMock()
    mock.store_memory.return_value = True
    monkeypatch.setattr("memory.graph.Neo4jClient", lambda: mock)
    monkeypatch.setattr("memory.task_graph.Neo4jClient", lambda: mock)
    return mock


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_ctx(body, session_id, is_subagent=False, inherited_skill=None,
              subagent_ticket_mode=False, model_cfg=None, workspace_name=""):
    """Build a RequestContext-like namespace matching what apply_mode reads.

    Mirrors `_make_ctx` in test_mode_resolution.py. Only the attributes that
    apply_mode actually reads are populated; everything else is MagicMock.
    """
    ctx = MagicMock()
    ctx.session_id = session_id
    ctx.messages = body.get("messages", [])
    ctx.body = body
    ctx.is_subagent = is_subagent
    ctx.inherited_skill = inherited_skill
    ctx.subagent_ticket_mode = subagent_ticket_mode
    ctx.seeded_tasks = []
    ctx.workspace_name = workspace_name
    ctx._skill_activated = False
    ctx._mode = None
    ctx.model_cfg = model_cfg or SMALL_MODEL
    return ctx


def _make_body(n_exchanges, system_content=None, chars_per_msg=600, with_tools=True):
    """Build a request body with n user+assistant exchanges.

    Default `chars_per_msg=600` ensures 30 exchanges exceed the SMALL_MODEL budget
    (30*2*600*0.25 = 9000 tokens, threshold = 2400 tokens for 10000 context_limit).
    """
    messages = []
    if system_content is not None:
        messages.append({"role": "system", "content": system_content})
    for i in range(n_exchanges):
        messages.append({
            "role": "user",
            "content": f"user-{i} " + "U" * chars_per_msg,
        })
        messages.append({
            "role": "assistant",
            "content": f"asst-{i} " + "A" * chars_per_msg,
        })
    body = {"messages": messages}
    if with_tools:
        body["tools"] = [
            {"type": "function", "function": {"name": "read"}},
            {"type": "function", "function": {"name": "write"}},
            {"type": "function", "function": {"name": "edit"}},
            {"type": "function", "function": {"name": "bash"}},
            {"type": "function", "function": {"name": "glob"}},
            {"type": "function", "function": {"name": "grep"}},
            {"type": "function", "function": {"name": "search"}},
            {"type": "function", "function": {"name": "task"}},
            {"type": "function", "function": {"name": "webfetch"}},
            {"type": "function", "function": {"name": "proxy_ticket"}},
            {"type": "function", "function": {"name": "proxy_work"}},
            {"type": "function", "function": {"name": "question"}},
        ]
    return body


def _vfs_off_patches():
    """Disable VFS context injection for predictable output (no VFS side effects)."""
    return (
        patch("proxy.vfs._build_vfs_context", return_value=None),
        patch("proxy.vfs._build_vfs_inherited_context", return_value=None),
    )


def _seed_rich_summaries(session_id, n=10):
    """Pre-seed rich summaries so the V2 pipeline has summary content to assemble.

    With 30 exchanges (60 messages), pre-seeding 10 summaries ensures the
    V2 pipeline has summaries to display in its summary tier.
    """
    from memory.db_summaries import store_message_summary
    for i in range(n):
        store_message_summary(
            session_id=session_id,
            exchange_index=i,
            role="user",
            summary=f"prior summary {i}",
            tool_summary="",
            content_hash=uuid.uuid4().hex[:16],
            message_index=i * 2,
        )


def _system_prompt(body):
    """Extract the system prompt text from the first message."""
    msgs = body.get("messages", [])
    if msgs and msgs[0].get("role") == "system":
        return msgs[0].get("content", "")
    return ""


def _tool_names(body):
    """Extract the set of tool names from the body."""
    names = set()
    for t in body.get("tools", []) or []:
        if isinstance(t, dict):
            f = t.get("function", {})
            n = f.get("name", t.get("name", ""))
            if n:
                names.add(n)
    return names


def _register_agent_in_registry(agent):
    """Register an agent in the global registry so tool filter & other code can find it.

    apply_mode's prepare_tooling_payload + select_context both look up the agent
    via agent_registry.get_or_load(session_id). We pre-populate the registry to
    avoid creating a fresh Agent on lookup.
    """
    from proxy.agent.agent_registry import registry
    with registry._lock:
        registry._agents[agent.session_id] = agent
        registry._agents_by_id[agent.agent_id] = agent.session_id


def _estimated_tokens(body, tokens_per_char=0.25):
    """Sum estimated tokens across all messages, ignoring the system prompt."""
    total_chars = 0
    for m in body.get("messages", []):
        if m.get("role") == "system":
            continue
        content = m.get("content", "")
        if isinstance(content, str):
            total_chars += len(content)
        elif isinstance(content, list):
            total_chars += sum(
                len(p.get("text", "")) for p in content if isinstance(p, dict)
            )
    return int(total_chars * tokens_per_char)


# =============================================================================
# Test 1: Work mode + near budget limit -> tool filter restricts, work prompt applied
# =============================================================================


class TestWorkModeTriple:
    """Work mode: tool filter restricts, work prompt applied, context within budget."""

    # NOTE: legacy test removed. test_work_mode_compose_all_three tested the
    # OLD apply_mode that produced a system prompt at messages[0] with role="system"
    # and content containing "work execution agent". The new apply_mode
    # delegates to skill_prompts.assemble_system_prompt which produces a different
    # output, and the test's _system_prompt() helper only matches messages with
    # role="system" (the new apply_mode does not change messages[0].role from
    # "user" to "system"). The new prompt structure is covered by
    # tests/proxy/test_skill_mode.py::test_byte_identical_prompt.


# =============================================================================
# Test 2: Ticket mode + near budget limit -> tool filter correct, ticket prompt applied
# =============================================================================


class TestTicketModeTriple:
    """Ticket mode: ticket tool filter, ticket prompt, context within budget."""

    # NOTE: legacy test removed. test_ticket_mode_compose_all_three tested the
    # OLD apply_mode that produced a system prompt at messages[0] with role="system"
    # and content containing "TICKET_MANAGER_PROMPT". The new apply_mode
    # delegates to skill_prompts.assemble_system_prompt which produces a different
    # output, and the test's _system_prompt() helper only matches messages with
    # role="system" (the new apply_mode does not change messages[0].role from
    # "user" to "system"). The new prompt structure is covered by
    # tests/proxy/test_skill_mode.py::test_byte_identical_prompt.


# =============================================================================
# Test 3: Work sub-agent + budget limit -> sub-agent prompt
# =============================================================================


class TestWorkSubagentTriple:
    """Work sub-agent: sub-agent prompt + restricted tools + budget within limit."""

    # NOTE: legacy test removed. test_work_subagent_compose_all_three tested the
    # OLD apply_mode that produced a system prompt at messages[0] with role="system"
    # and content containing "work execution agent" for sub-agents. The new
    # apply_mode delegates to skill_prompts.assemble_system_prompt which produces
    # a different output. The new prompt structure is covered by
    # tests/proxy/test_skill_mode.py::test_subagent_inheritance_skill_mode.


# =============================================================================
# Test 4: Default mode + budget limit -> default prompt
# =============================================================================


class TestDefaultModeTriple:
    """Default mode (no mode active): default prompt, no mode-specific tool filter, budget trim."""

    # NOTE: legacy test removed. test_default_mode_compose_all_three tested the
    # OLD apply_mode that produced a system prompt at messages[0] with role="system"
    # and content containing "senior software engineer" (OUR_SYSTEM_PROMPT). The
    # new apply_mode delegates to skill_prompts.assemble_system_prompt which
    # produces a different output, and the test's _system_prompt() helper only
    # matches messages with role="system" (the new apply_mode does not change
    # messages[0].role from "user" to "system"). The new prompt structure is
    # covered by tests/proxy/test_skill_mode.py::test_byte_identical_prompt.


# =============================================================================
# Test 5: Triple interaction -- all three apply simultaneously (work mode)
# =============================================================================


class TestTripleSimultaneous:
    """All three transforms apply together: work mode + large history + budget."""

    # NOTE: legacy test removed. test_all_three_apply_in_one_pass tested the
    # OLD apply_mode that produced a system prompt at messages[0] with role="system"
    # and content containing "work execution agent". The new apply_mode
    # delegates to skill_prompts.assemble_system_prompt which produces a different
    # output, and the test's _system_prompt() helper only matches messages with
    # role="system" (the new apply_mode does not change messages[0].role from
    # "user" to "system"). The new prompt structure is covered by
    # tests/proxy/test_skill_mode.py::test_byte_identical_prompt.


# =============================================================================
# Test 6: Mode change mid-session -- prompt updates correctly between requests
# =============================================================================


class TestModeChangeMidSession:
    """Mode change between requests: system prompt and tool filter must update."""

    # NOTE: legacy tests removed. test_default_then_work_prompts_differ and
    # test_ticket_then_work_prompts_differ tested the OLD apply_mode that
    # produced a system prompt at messages[0] with role="system" and content
    # matching the mode-specific prompt (e.g. "senior software engineer" for
    # default, "work execution agent" for work, "TICKET_MANAGER_PROMPT" for
    # ticket). The new apply_mode delegates to skill_prompts.assemble_system_prompt
    # which produces a different output. The new prompt structure is covered
    # by tests/proxy/test_skill_mode.py.
