"""Integration tests: mode gate → tool guard → action handler full chain.

Tests the boundary between mode activation, tool guard filtering, and
action dispatch. Each test exercises the real guard.check() →
dispatch_action() chain with a real Agent registered in the agent registry.
"""
from __future__ import annotations

import uuid
import pytest
from unittest.mock import MagicMock, patch

from proxy.tool_pipeline.tool_call import ToolCall
from proxy.tool_pipeline.tool_guard import MemoryToolGuard
from proxy.tool_pipeline.tool_pipeline import ToolContext
from proxy.tool_pipeline.tool_call_tracker import ToolCallTracker
from proxy.tool_pipeline.action_registry import dispatch_action
from memory.tools import proxy_ticket  # noqa: F401
from memory.tools import proxy_work  # noqa: F401
from tests.mode_harness import (
    create_test_agent,
    activate_ticket_mode,
    activate_work_mode,
)


def _tc(name: str, arguments: dict) -> ToolCall:
    """Create a ToolCall with a fresh id."""
    return ToolCall(id=f"call_{uuid.uuid4().hex[:24]}", name=name, arguments=arguments)


def _ctx(session_id: str, is_subagent: bool = False, inherited_skill: str = None,
         subagent_ticket_mode: bool = False) -> ToolContext:
    """Build a minimal ToolContext for guard checks."""
    ctx = MagicMock()
    ctx.session_id = session_id
    ctx.is_subagent = is_subagent
    ctx.inherited_skill = inherited_skill
    ctx.subagent_ticket_mode = subagent_ticket_mode
    return ctx


def _register_agent(agent, monkeypatch):
    """Patch agent_registry.get_or_load to return the given agent."""
    return patch(
        "proxy.agent.agent_registry.registry.get_or_load",
        return_value=agent,
    )


def _seed_hash(session_id: str, short_hash: str = "abc12345"):
    """Register a hash in the tracker so hash provenance check passes."""
    tracker = ToolCallTracker.get_instance(session_id)
    tracker.add_seen_hashes([short_hash])


def _mock_tg_for_list():
    """Build a mock TaskGraph that returns empty tasks for list_tasks."""
    mock_tg = MagicMock()
    mock_tg.get_actionable_tasks.return_value = []
    mock_tg.get_store_status.return_value = {}
    mock_tg._executor = MagicMock()
    return mock_tg


# =============================================================================
# Test 1: Ticket mode → proxy_ticket allowed, proxy_work blocked
# =============================================================================


class TestTicketModeGuardChain:
    """Ticket mode: proxy_ticket allowed, proxy_work blocked by mode allowlist."""

    def test_ticket_mode_allows_main_proxy_ticket_read_only(self, monkeypatch):
        """MAIN agent in ticket mode is ALLOWED proxy_ticket(list_tasks) (read-only).

        US-096: MAIN in ticket mode receives proxy_ticket read-only
        (list_tasks, get_task only). The read-only restriction is enforced at
        the action layer (_check_work_mode_actions), NOT by omitting the tool.
        """
        session_id = f"sess-ticket-{uuid.uuid4().hex[:8]}"
        agent = create_test_agent(session_id=session_id)
        activate_ticket_mode(agent)

        tc = _tc("proxy_ticket_list_tasks", {})
        ctx = _ctx(session_id)

        with _register_agent(agent, monkeypatch):
            guard = MemoryToolGuard()
            error = guard.check(tc, ctx)

        assert error is None, (
            f"MAIN agent in ticket mode should be allowed proxy_ticket(list_tasks), got: {error}"
        )

    def test_ticket_mode_blocks_main_from_proxy_ticket_writes(self, monkeypatch):
        """MAIN agent in ticket mode is BLOCKED from proxy_ticket write actions.

        US-096: Write actions (create_tasks, update_task, edit, set_parent,
        clear_parent, clear_blocked_by, set_priority, append_review_note) are
        blocked for MAIN in ticket mode. The block message names the action
        and the allowed read-only set. Sub-agents keep FULL proxy_ticket access.
        """
        session_id = f"sess-ticket-{uuid.uuid4().hex[:8]}"
        agent = create_test_agent(session_id=session_id)
        activate_ticket_mode(agent)

        tc = _tc("proxy_ticket_create_tasks", {
            "tasks": [{"name": "Test task", "priority": "high"}],
        })
        ctx = _ctx(session_id)

        with _register_agent(agent, monkeypatch):
            guard = MemoryToolGuard()
            error = guard.check(tc, ctx)

        assert error is not None, (
            "MAIN agent in ticket mode should be blocked from proxy_ticket(create_tasks)"
        )
        assert "create tasks" in error.lower() or "read-only" in error.lower()
        assert "list_tasks" in error or "delegate" in error.lower()

    def test_ticket_mode_blocks_main_from_proxy_work(self, monkeypatch):
        """MAIN agent in ticket mode is blocked from proxy_work and must delegate.

        Per 9f3b997, _check_ticket_mode_orchestrator blocks the MAIN agent
        from calling proxy_work directly; the MAIN agent must delegate via
        the `task` tool. Hash provenance is satisfied by seeding the tracker
        so the orchestrator check (which runs after provenance) is the
        actual blocker.
        """
        session_id = f"sess-ticket-{uuid.uuid4().hex[:8]}"
        agent = create_test_agent(session_id=session_id)
        activate_ticket_mode(agent)
        _seed_hash(session_id, "abc12345")

        tc = _tc("proxy_work", {"action": "start_work", "short_hash": "abc12345"})
        ctx = _ctx(session_id)

        with _register_agent(agent, monkeypatch):
            guard = MemoryToolGuard()
            error = guard.check(tc, ctx)

        assert error is not None, "Guard should block MAIN agent from proxy_work in ticket mode"
        assert "proxy_work is blocked in ticket mode" in error or "blocked" in error
        assert "task" in error.lower() or "delegate" in error.lower() or "blocked" in error.lower()

    def test_ticket_mode_blocks_proxy_ticket_at_entry_point(self, monkeypatch):
        """execute_ticket_tool works in ticket mode (no access_denied)."""
        from memory.tools.proxy_ticket import execute_ticket_tool

        session_id = f"sess-ticket-{uuid.uuid4().hex[:8]}"
        agent = create_test_agent(session_id=session_id)
        activate_ticket_mode(agent)

        mock_tg = _mock_tg_for_list()

        with _register_agent(agent, monkeypatch), \
             patch("proxy.context._get_task_graph_singleton", return_value=mock_tg), \
             patch("proxy.tool_pipeline.actions.list_tasks.get_recent_review_notes", return_value={}):
            result = execute_ticket_tool(
                "list_tasks", {}, session_id=session_id,
            )

        assert (result.get("error") or {}).get("code") != "access_denied", (
            f"Ticket mode should pass entry-point gate, got: {result}"
        )

    def test_ticket_mode_entry_point_blocks_proxy_work(self, monkeypatch):
        """C3: ticket mode does NOT unlock proxy_work. execute_proxy_work
        returns mode_inactive for ticket-mode agents.

        Pre-C3 ticket-mode MAIN agents could call proxy_work(stop_work) to
        close their own sessions (Bug #3 exemption). C3 deleted that path:
        proxy_work is work-mode-only. Default, ticket, and workspace mode
        callers are all rejected with mode_inactive.
        """
        from memory.tools.proxy_work import execute_proxy_work

        session_id = f"sess-ticket-{uuid.uuid4().hex[:8]}"
        agent = create_test_agent(session_id=session_id)
        activate_ticket_mode(agent)

        with _register_agent(agent, monkeypatch):
            result = execute_proxy_work(
                {"action": "stop_work", "short_hash": "abc12345", "ws_uuid": "ws-abc"},
                session_id=session_id,
            )

        # C3: ticket mode does NOT unlock proxy_work
        assert result.get("error", {}).get("code") == "mode_inactive", (
            f"C3 REGRESSION: ticket-mode execute_proxy_work should return mode_inactive; got: {result}"
        )
        assert "not in work mode" in result.get("error", {}).get("message", ""), (
            f"C3 REGRESSION: expected actionable 'not in work mode' message; got: {result}"
        )


# =============================================================================
# Test 2: Work mode → proxy_work allowed, proxy_ticket CRUD blocked
# =============================================================================


class TestWorkModeGuardChain:
    """Work mode: proxy_work allowed, proxy_ticket CRUD blocked by _check_work_mode_actions."""

    def test_work_mode_allows_proxy_work(self, monkeypatch):
        """proxy_work(start_work) passes guard in work mode with known hash."""
        session_id = f"sess-work-{uuid.uuid4().hex[:8]}"
        agent = create_test_agent(session_id=session_id)
        activate_work_mode(agent)
        _seed_hash(session_id, "abc12345")

        tc = _tc("proxy_work", {"action": "start_work", "short_hash": "abc12345"})
        ctx = _ctx(session_id)

        with _register_agent(agent, monkeypatch):
            guard = MemoryToolGuard()
            error = guard.check(tc, ctx)

        assert error is None, f"Guard should allow proxy_work in work mode, got: {error}"

    def test_work_mode_blocks_proxy_ticket_create_tasks(self, monkeypatch):
        """proxy_ticket(create_tasks) blocked in work mode by _check_work_mode_actions."""
        session_id = f"sess-work-{uuid.uuid4().hex[:8]}"
        agent = create_test_agent(session_id=session_id)
        activate_work_mode(agent)

        tc = _tc("proxy_ticket_create_tasks", {
            "tasks": [{"name": "Test task", "priority": "high"}],
        })
        ctx = _ctx(session_id)

        with _register_agent(agent, monkeypatch):
            guard = MemoryToolGuard()
            error = guard.check(tc, ctx)

        assert error is not None, "Work mode should block proxy_ticket(create_tasks)"
        assert "Work MAIN cannot create tasks" in error or "use proxy_work" in error.lower()
        assert "use proxy_work" in error.lower()

    def test_work_mode_allows_proxy_ticket_list_tasks(self, monkeypatch):
        """proxy_ticket(list_tasks) allowed in work mode (read-only)."""
        session_id = f"sess-work-{uuid.uuid4().hex[:8]}"
        agent = create_test_agent(session_id=session_id)
        activate_work_mode(agent)

        tc = _tc("proxy_ticket_list_tasks", {})
        ctx = _ctx(session_id)

        with _register_agent(agent, monkeypatch):
            guard = MemoryToolGuard()
            error = guard.check(tc, ctx)

        assert error is None, (
            f"Work mode should allow proxy_ticket(list_tasks), got: {error}"
        )

    def test_work_mode_blocks_proxy_ticket_update_task(self, monkeypatch):
        """proxy_ticket(update_task) blocked in work mode."""
        session_id = f"sess-work-{uuid.uuid4().hex[:8]}"
        agent = create_test_agent(session_id=session_id)
        activate_work_mode(agent)

        tc = _tc("proxy_ticket_update_task", {
            "short_hash": "abc12345",
            "new_status": "complete",
        })
        ctx = _ctx(session_id)

        with _register_agent(agent, monkeypatch):
            guard = MemoryToolGuard()
            error = guard.check(tc, ctx)

        assert error is not None, "Work mode should block proxy_ticket(update_task)"
        assert "Work MAIN cannot update tasks" in error

    def test_work_mode_entry_point_allows_proxy_work(self, monkeypatch):
        """execute_proxy_work succeeds in work mode."""
        from memory.tools.proxy_work import execute_proxy_work

        session_id = f"sess-work-{uuid.uuid4().hex[:8]}"
        agent = create_test_agent(session_id=session_id)
        activate_work_mode(agent)

        mock_tg = MagicMock()

        with _register_agent(agent, monkeypatch), \
             patch("proxy.context._get_task_graph_singleton", return_value=mock_tg), \
             patch("proxy.tool_pipeline.action_registry.dispatch_action", return_value={"ok": True}):
            result = execute_proxy_work(
                {"action": "start_work", "short_hash": "abc12345"},
                session_id=session_id,
            )

        assert (result.get("error") or {}).get("code") != "access_denied", (
            f"Work mode should pass proxy_work entry-point gate, got: {result}"
        )

    def test_work_mode_entry_point_allows_proxy_ticket_read(self, monkeypatch):
        """execute_ticket_tool succeeds for list_tasks in work mode."""
        from memory.tools.proxy_ticket import execute_ticket_tool

        session_id = f"sess-work-{uuid.uuid4().hex[:8]}"
        agent = create_test_agent(session_id=session_id)
        activate_work_mode(agent)

        mock_tg = _mock_tg_for_list()

        with _register_agent(agent, monkeypatch), \
             patch("proxy.context._get_task_graph_singleton", return_value=mock_tg), \
             patch("proxy.tool_pipeline.actions.list_tasks.get_recent_review_notes", return_value={}):
            result = execute_ticket_tool(
                "list_tasks", {}, session_id=session_id,
            )

        assert (result.get("error") or {}).get("code") != "access_denied", (
            f"Work mode should allow proxy_ticket(list_tasks), got: {result}"
        )


# =============================================================================
# Test 3: Default mode → both proxy_ticket and proxy_work blocked
# =============================================================================


class TestDefaultModeGuardChain:
    """Default mode (no mode active): both proxy tools blocked at entry-point."""

    def test_default_mode_blocks_proxy_ticket(self, monkeypatch):
        """proxy_ticket blocked at entry-point in default mode."""
        from memory.tools.proxy_ticket import execute_ticket_tool

        session_id = f"sess-default-{uuid.uuid4().hex[:8]}"
        agent = create_test_agent(session_id=session_id)
        # No mode activated — default mode

        with _register_agent(agent, monkeypatch):
            result = execute_ticket_tool(
                "list_tasks", {}, session_id=session_id,
            )

        assert result.get("error", {}).get("code") == "access_denied", (
            f"Default mode should block proxy_ticket, got: {result}"
        )
        assert "proxy_ticket is not available" in result.get("error", {}).get("message", "")

    def test_default_mode_blocks_proxy_work(self, monkeypatch):
        """proxy_work blocked at entry-point in default mode."""
        from memory.tools.proxy_work import execute_proxy_work

        session_id = f"sess-default-{uuid.uuid4().hex[:8]}"
        agent = create_test_agent(session_id=session_id)
        # No mode activated — default mode

        with _register_agent(agent, monkeypatch):
            result = execute_proxy_work(
                {"action": "start_work", "short_hash": "abc12345"},
                session_id=session_id,
            )

        assert result.get("error", {}).get("code") == "mode_inactive", (
            f"Default mode should block proxy_work, got: {result}"
        )
        assert "not in work mode" in result.get("error", {}).get("message", "")

    def test_default_mode_guard_allows_through(self, monkeypatch):
        """Guard itself doesn't block proxy_ticket in default mode (no mode = no allowlist restriction).

        The guard's _check_mode_allowlist returns None when there's no
        active mode (fail-open). The real blocking is at the entry-point.
        This test documents that design.
        """
        session_id = f"sess-default-{uuid.uuid4().hex[:8]}"
        agent = create_test_agent(session_id=session_id)
        # No mode activated

        tc = _tc("proxy_ticket_list_tasks", {})
        ctx = _ctx(session_id)

        with _register_agent(agent, monkeypatch):
            guard = MemoryToolGuard()
            error = guard.check(tc, ctx)

        # Guard is fail-open when no mode is active — entry-point is the blocker
        assert error is None, (
            f"Guard should not block in default mode (fail-open), got: {error}"
        )


# =============================================================================
# Test 4: Work sub-agent → proxy_ticket restricted, proxy_work allowed
# =============================================================================

# NOTE: legacy class removed. TestWorkSubAgentGuardChain tested the OLD
# _check_hierarchy body in tool_guard.py with the legacy "Work sub-agent"
# / "Sub-agent cannot use proxy_work" error messages and the legacy
# "inherited_skill='work'" → allowlist path. The NEW _check_hierarchy
# delegates to proxy.skill_guard.check_sub_agent_proxy_access, which uses
# the CAT-3 trust gate, ProxyAccessLevel (READ_ONLY/NEVER/FULL), and
# skill.check_action. The new contract is covered by
# tests/proxy/test_skill_guard.py and tests/integration/test_subagent_lifecycle.py
# (the new TestSubagentGuardChain tests).


# =============================================================================
# Test 5: Error messages propagate correctly
# =============================================================================


class TestErrorMessagePropagation:
    """Action handler errors propagate through the guard to the response."""

    def test_action_error_propagates_through_dispatch(self, monkeypatch):
        """Invalid create_tasks parameters produce an error from the action handler,
        not from the guard."""
        session_id = f"sess-err-{uuid.uuid4().hex[:8]}"
        agent = create_test_agent(session_id=session_id)
        activate_ticket_mode(agent)

        mock_tg = MagicMock()

        with _register_agent(agent, monkeypatch), \
             patch("proxy.context._get_task_graph_singleton", return_value=mock_tg):
            result = dispatch_action(
                "create_tasks",
                {"tasks": []},  # empty tasks → invalid_input
                session_id,
                mock_tg,
                tool="proxy_ticket",
            )

        assert result.get("error", {}).get("code") == "invalid_input", (
            f"Expected invalid_input from action handler, got: {result}"
        )
        msg = result.get("error", {}).get("message", "")
        assert "non-empty" in msg.lower() or "tasks" in msg.lower(), (
            f"Error message should mention the problem, got: {msg}"
        )

    def test_action_error_not_masked_by_guard(self, monkeypatch):
        """MAIN-in-ticket-mode guard block is distinct from action-level errors.

        The guard blocks the MAIN agent from calling proxy_ticket write
        actions in ticket mode (US-096, via _check_work_mode_actions). The
        action handler still validates parameters independently — we verify
        the action path still surfaces its own invalid_input error when
        called directly, so the two error sources remain distinct and
        unmaskable.
        """
        session_id = f"sess-err-{uuid.uuid4().hex[:8]}"
        agent = create_test_agent(session_id=session_id)
        activate_ticket_mode(agent)

        # Guard blocks the MAIN agent — confirm the blocking message
        tc = _tc("proxy_ticket_create_tasks", {
            "tasks": [],  # invalid — empty list
        })
        ctx = _ctx(session_id)

        with _register_agent(agent, monkeypatch):
            guard = MemoryToolGuard()
            guard_error = guard.check(tc, ctx)

        assert guard_error is not None, "Guard should block MAIN agent in ticket mode"
        assert "read-only" in guard_error.lower() or "delegate" in guard_error.lower()
        # Crucially: the guard error is NOT the action error
        assert "non-empty" not in guard_error.lower()

        # Action handler still surfaces its own error when invoked directly
        mock_tg = MagicMock()
        with _register_agent(agent, monkeypatch), \
             patch("proxy.context._get_task_graph_singleton", return_value=mock_tg):
            action_result = dispatch_action(
                "create_tasks",
                {"tasks": []},
                session_id,
                mock_tg,
                tool="proxy_ticket",
            )

        assert action_result.get("error") is not None, (
            "Action handler should return an error for empty tasks"
        )

    def test_unknown_action_error_propagates(self, monkeypatch):
        """Unknown action name produces a clear error from dispatch_action."""
        session_id = f"sess-err-{uuid.uuid4().hex[:8]}"
        agent = create_test_agent(session_id=session_id)
        activate_ticket_mode(agent)

        mock_tg = MagicMock()

        with _register_agent(agent, monkeypatch):
            result = dispatch_action(
                "nonexistent_action",
                {"action": "nonexistent_action"},
                session_id,
                mock_tg,
                tool="proxy_ticket",
            )

        assert result.get("error", {}).get("code") == "invalid_action", (
            f"Expected invalid_action, got: {result}"
        )
        assert "nonexistent_action" in result.get("error", {}).get("message", "")

    def test_guard_error_message_is_clear(self, monkeypatch):
        """Guard rejection messages are clear and actionable."""
        session_id = f"sess-err-{uuid.uuid4().hex[:8]}"
        agent = create_test_agent(session_id=session_id)
        activate_work_mode(agent)

        # Work mode blocks proxy_ticket CRUD — verify message is clear
        tc = _tc("proxy_ticket_create_tasks", {
            "tasks": [{"name": "X", "priority": "high"}],
        })
        ctx = _ctx(session_id)

        with _register_agent(agent, monkeypatch):
            guard = MemoryToolGuard()
            error = guard.check(tc, ctx)

        assert error is not None
        # Verify the error message contains actionable guidance
        assert "proxy_work" in error or "ticket mode" in error, (
            f"Error message should suggest alternatives, got: {error}"
        )

    def test_full_chain_ticket_mode_create_tasks_succeeds(self, monkeypatch):
        """Full chain: ticket sub-agent → guard passes → action executes create_tasks.

        Per 9f3b997, the MAIN agent is blocked from calling proxy_ticket in
        ticket mode, but a ticket sub-agent (is_subagent=True,
        subagent_ticket_mode=True) is allowed. This test exercises the
        full chain from a ticket sub-agent's perspective.
        """
        session_id = f"sess-chain-{uuid.uuid4().hex[:8]}"
        agent = create_test_agent(session_id=session_id)
        activate_ticket_mode(agent)

        mock_tg = MagicMock()
        mock_tg.create_tasks_batch.return_value = [{
            "short_hash": "abc12345",
            "name": "New task",
            "priority": "high",
            "status": "pending",
            "uuid": "ent-123",
        }]

        # Step 1: Guard check from a ticket sub-agent's context
        tc = _tc("proxy_ticket_create_tasks", {
            "tasks": [{"name": "New task", "priority": "high"}],
        })
        ctx = _ctx(
            session_id,
            is_subagent=True,
            inherited_skill="ticket",
            subagent_ticket_mode=True,
        )

        with _register_agent(agent, monkeypatch):
            guard = MemoryToolGuard()
            guard_error = guard.check(tc, ctx)

        assert guard_error is None, (
            f"Guard should pass for ticket sub-agent, got: {guard_error}"
        )

        # Step 2: Action dispatch
        with _register_agent(agent, monkeypatch), \
             patch("proxy.context._get_task_graph_singleton", return_value=mock_tg):
            result = dispatch_action(
                "create_tasks",
                {"tasks": [{"name": "New task", "priority": "high"}]},
                session_id,
                mock_tg,
                tool="proxy_ticket",
            )

        assert result.get("error") is None, f"Action should succeed, got: {result}"
        assert result["data"]["total_created"] == 1
        assert len(result["data"]["created"]) == 1
        assert result["data"]["created"][0]["name"] == "New task"

    def test_full_chain_work_mode_list_tasks_succeeds(self, monkeypatch):
        """Full chain: work mode → guard passes → action executes list_tasks."""
        session_id = f"sess-chain-{uuid.uuid4().hex[:8]}"
        agent = create_test_agent(session_id=session_id)
        activate_work_mode(agent)

        mock_tg = _mock_tg_for_list()
        mock_tg.get_actionable_tasks.return_value = [
            {"short_hash": "abc12345", "name": "Existing task", "priority": "medium", "status": "pending"},
        ]
        mock_tg.get_store_status.return_value = {"total": 1}

        # Step 1: Guard check
        tc = _tc("proxy_ticket_list_tasks", {})
        ctx = _ctx(session_id)

        with _register_agent(agent, monkeypatch):
            guard = MemoryToolGuard()
            guard_error = guard.check(tc, ctx)

        assert guard_error is None, f"Guard should allow list_tasks in work mode, got: {guard_error}"

        # Step 2: Action dispatch — mock get_recent_review_notes to avoid executor issues
        with _register_agent(agent, monkeypatch), \
             patch("proxy.context._get_task_graph_singleton", return_value=mock_tg), \
             patch("proxy.tool_pipeline.actions.list_tasks.get_recent_review_notes", return_value={}):
            result = dispatch_action(
                "list_tasks",
                {},
                session_id,
                mock_tg,
                tool="proxy_ticket",
            )

        assert result.get("error") is None, f"Action should succeed, got: {result}"
        assert result["data"]["total"] == 1
        assert len(result["data"]["tasks"]) == 1

    def test_full_chain_work_mode_create_tasks_blocked_by_guard(self, monkeypatch):
        """Full chain: work mode → guard blocks create_tasks → action never reached."""
        session_id = f"sess-chain-{uuid.uuid4().hex[:8]}"
        agent = create_test_agent(session_id=session_id)
        activate_work_mode(agent)

        # Step 1: Guard check
        tc = _tc("proxy_ticket_create_tasks", {
            "tasks": [{"name": "Should be blocked", "priority": "high"}],
        })
        ctx = _ctx(session_id)

        with _register_agent(agent, monkeypatch):
            guard = MemoryToolGuard()
            guard_error = guard.check(tc, ctx)

        assert guard_error is not None, "Guard should block create_tasks in work mode"

        # Step 2: Verify the error is a guard rejection, not an action error
        assert "Work MAIN cannot create tasks" in guard_error

        # Step 3: Verify that dispatch_action would succeed if called directly
        # (the guard is the blocker, not the action)
        mock_tg = MagicMock()
        mock_tg.create_tasks_batch.return_value = [{
            "short_hash": "abc12345",
            "name": "Should be blocked",
            "priority": "high",
            "status": "pending",
            "uuid": "ent-456",
        }]

        with _register_agent(agent, monkeypatch), \
             patch("proxy.context._get_task_graph_singleton", return_value=mock_tg):
            action_result = dispatch_action(
                "create_tasks",
                {"tasks": [{"name": "Should be blocked", "priority": "high"}]},
                session_id,
                mock_tg,
                tool="proxy_ticket",
            )

        # Action itself succeeds — the guard is the enforcement point
        assert action_result.get("error") is None, (
            f"Action should succeed when called directly (guard is the blocker), got: {action_result}"
        )
