"""Unit tests for WorkSkill (^^work command set)."""
from __future__ import annotations

import os
import pytest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from proxy.skills.skill_registry import SkillContext
from skills.work_skill import WorkSkill, WORK_MANAGER_PROMPT


# ----------------------------------------------------------------------------
# Fixtures
# ----------------------------------------------------------------------------


@pytest.fixture
def session_id():
    return "test-sess-work"


@pytest.fixture
def ctx(session_id):
    return SkillContext(
        session_id=session_id,
        agent_id=f"agent_{session_id}",
        messages=[],
        is_subagent=False,
        active_skill_name="work",
        raw_user_text="^^work abc12345",
        # Bug #2: the dispatcher stores the hash under "arg" (see
        # pipeline/dispatch.py:try_skill_activation and
        # skill_dispatcher.SkillDispatcher.dispatch). The pre-fix skill_state
        # key "task_hash" was wrong — WorkSkill.get_system_prompt now reads "arg".
        skill_state={"action": "task", "arg": "abc12345"},
        client_hint="kilocode",
    )


@pytest.fixture
def mock_tg():
    tg = MagicMock()
    tg.get_task_enriched.return_value = {
        "name": "Fix login bug",
        "priority": "high",
        "summary": "Auth flow returns 500 on invalid email.",
    }
    return tg


# ----------------------------------------------------------------------------
# Tests
# ----------------------------------------------------------------------------


class TestWorkSkillMetadata:
    """WorkSkill name, trigger, ttl, tool filter properties."""

    def test_name(self):
        assert WorkSkill().name == "work"

    def test_trigger(self):
        assert WorkSkill().trigger == "^^work"

    def test_ttl_seconds(self):
        assert WorkSkill().ttl_seconds == 300

    def test_get_meta_returns_skill_meta(self):
        from proxy.skills.skill_registry import SkillMeta
        meta = WorkSkill().get_meta()
        assert isinstance(meta, SkillMeta)
        assert meta.name == "work"
        assert meta.trigger == "^^work"
        assert meta.ttl_seconds == 300


class TestWorkSkillParseSubcommand:
    """parse_subcommand returns the right (action, arg) tuples."""

    def test_parse_with_task_hash(self):
        action, arg = WorkSkill().parse_subcommand("^^work abc12345")
        assert action == "task"
        assert arg == "abc12345"

    def test_parse_without_args_returns_list(self):
        action, arg = WorkSkill().parse_subcommand("^^work")
        assert action == "list"
        assert arg == ""

    def test_parse_non_work_command_returns_none(self):
        action, arg = WorkSkill().parse_subcommand("hello world")
        assert action is None
        assert arg == ""

    def test_parse_non_short_hash_returns_list(self):
        """An arg that isn't a short hash falls back to list."""
        action, arg = WorkSkill().parse_subcommand("^^work not-a-hash")
        assert action == "list"
        assert arg == ""


class TestWorkSkillActivation:
    """on_activate captures the task hash and rejects sub-agents.

    R4 CRIT-4 fix: on_activate MUST NOT mutate ctx.work_task_hash. The
    captured task hash is logged only; Phase 2 routes it through
    `proxy.work_task_store[session_id]`.
    """

    def test_on_activate_does_not_mutate_ctx(self, ctx):
        """R4 CRIT-4: on_activate MUST NOT set ctx.work_task_hash.

        The previous implementation did
        `try: ctx.work_task_hash = task_hash; except AttributeError: pass`,
        leaking an instance attribute onto the (potentially pooled)
        request context. The new implementation captures the hash into
        a local variable and does not touch ctx.
        """
        # Bug #10: on_activate now validates the hash exists in the task
        # graph. Mock the TaskGraph so the validation succeeds regardless
        # of whether a live Neo4j is reachable in the test env.
        mock_tg = MagicMock()
        mock_tg.is_available.return_value = True
        mock_tg.get_task_enriched.return_value = {
            "name": "Test", "priority": "medium", "summary": "",
        }
        ctx.skill_state = {"arg": "deadbeef"}
        # Baseline: ctx.work_task_hash is the dataclass default "".
        assert ctx.work_task_hash == ""
        with patch("proxy.context._get_task_graph_singleton", return_value=mock_tg), \
             patch("proxy.workspace_state.workspace_store") as mock_ws:
            mock_ws.get_active.return_value = "default"
            WorkSkill().on_activate(ctx)
        # Invariant: ctx.work_task_hash MUST remain the dataclass
        # default (no instance mutation).
        assert ctx.work_task_hash == "", (
            "R4 CRIT-4: WorkSkill.on_activate must not mutate ctx "
            "with work_task_hash (Phase 2 introduces work_task_store)."
        )

    def test_on_activate_rejects_subagent(self, ctx):
        ctx.is_subagent = True
        with pytest.raises(ValueError) as excinfo:
            WorkSkill().on_activate(ctx)
        assert "Sub-agents cannot activate work mode" in str(excinfo.value)

    def test_on_activate_without_task_hash(self, ctx):
        """R4 CRIT-4 follow-on: without a task hash in skill_state, raise ValueError.

        Previously the function silently set nothing on ctx and returned.
        Now it raises ValueError(US-092) — work mode is meaningless
        without a target task hash.
        """
        ctx.skill_state = {}
        with pytest.raises(ValueError, match="requires a task hash"):
            WorkSkill().on_activate(ctx)


class TestWorkSkillToolFilter:
    """Work mode has FULL tool access (read, write, edit, bash, etc.).

    Slice 8b: the monolithic `proxy_ticket` tool is decomposed into 10
    action-specific tools. Work MAIN sees the 3 read-only decomposed tools
    (`proxy_ticket_list_tasks`, `proxy_ticket_get_task`,
    `proxy_ticket_append_review_note`) plus `proxy_work` — the CRUD
    decomposed tools are NOT visible (visible=False in WORK_CAPABILITY_TABLE).
    """

    def test_get_tool_filter_contains_full_tool_set(self):
        tools = WorkSkill().get_tool_filter(SkillContext(
            session_id="x", agent_id="a", is_subagent=False,
        ))
        assert "read" in tools
        assert "write" in tools
        assert "edit" in tools
        assert "bash" in tools
        assert "glob" in tools
        assert "grep" in tools
        assert "task" in tools
        assert "proxy_work" in tools
        # Slice 8b: decomposed read-only ticket tools are visible for MAIN.
        assert "proxy_ticket_list_tasks" in tools
        assert "proxy_ticket_get_task" in tools
        assert "proxy_ticket_append_review_note" in tools
        # CRUD decomposed ticket tools are NOT visible (capability table).
        assert "proxy_ticket_create_tasks" not in tools
        assert "proxy_ticket_update_task" not in tools
        # Legacy monolithic name is gone — no tool is named just "proxy_ticket".
        assert "proxy_ticket" not in tools


class TestWorkSkillCanary:
    """get_canary returns an HMAC-signed canary (or unsigned if no secret)."""

    def test_get_canary_returns_string(self):
        with patch("proxy.canary._SESSION_SECRET", ""):
            canary = WorkSkill().get_canary(SkillContext(
                session_id="x", agent_id="a", is_subagent=False,
            ))
        assert isinstance(canary, str)
        assert "work" in canary
        assert canary.startswith("[[__SUBAGENT__:SKILL:work")

    def test_get_canary_unsigned_when_no_secret(self, monkeypatch):
        # `_get_session_secret()` is `os.environ.get(...) or _SESSION_SECRET`,
        # so BOTH the env var AND the module-level cache must be cleared —
        # setting only the env var leaves a cached fallback in place.
        monkeypatch.delenv("PROXY_SESSION_SECRET", raising=False)
        monkeypatch.setattr("proxy.canary._SESSION_SECRET", "")
        canary = WorkSkill().get_canary(SkillContext(
            session_id="x", agent_id="a", is_subagent=False,
        ))
        # No PROXY_SESSION_SECRET → unsigned format with 64-hex placeholder:
        # [[__SUBAGENT__:SKILL:work:000...000:timestamp]]
        # The "000...000" is the 64-char hex placeholder (not a real signature)
        assert "signature" not in canary
        # Verify format: starts with prefix, ends with ]]
        assert canary.startswith("[[__SUBAGENT__:SKILL:work:")
        assert canary.endswith("]]")
        # Extract the sig:ts part
        body = canary[len("[[__SUBAGENT__:SKILL:work:"):-2]
        # It should be "64-hex-placeholder:timestamp"
        parts = body.split(":")
        assert len(parts) == 2, f"Expected 'sig:ts' format, got: {body}"
        sig_part, ts_part = parts
        assert len(sig_part) == 64, f"Expected 64-char hex placeholder, got {len(sig_part)} chars"
        assert sig_part == "0" * 64, "Expected all zeros for unsigned canary placeholder"
        assert ts_part.isdigit(), f"Expected numeric timestamp, got: {ts_part}"

    def test_get_canary_signed_with_secret(self, monkeypatch):
        monkeypatch.setenv("PROXY_SESSION_SECRET", "test-secret")
        canary = WorkSkill().get_canary(SkillContext(
            session_id="x", agent_id="a", is_subagent=False,
            ))
        # When signed, the canary has format [[__SUBAGENT__:SKILL:work:sig:ts]]
        assert canary.startswith("[[__SUBAGENT__:SKILL:work:")
        assert canary != "[[__SUBAGENT__:SKILL:work]]"
        # Verify the signature is a valid sig:ts pair
        body = canary[len("[[__SUBAGENT__:SKILL:work:"):-2]  # strip prefix and "]]"
        assert ":" in body


class TestWorkSkillSystemPrompt:
    """get_system_prompt substitutes task context into the template."""

    def test_get_system_prompt_substitutes_task_context(self, ctx, mock_tg):
        with patch("proxy.context._get_task_graph_singleton", return_value=mock_tg), \
             patch("proxy.workspace_state.workspace_store") as mock_ws, \
             patch("proxy.state.make_workspace_id", return_value="sess:default"):
            mock_ws.get_active.return_value = "default"
            prompt = WorkSkill().get_system_prompt(ctx)

        assert "abc12345" in prompt
        assert "Fix login bug" in prompt
        assert "HIGH" in prompt
        assert "Auth flow returns 500" in prompt

    def test_get_system_prompt_fallback_when_no_task(self, mock_tg):
        """When no task_hash (free-form edit session), substitute safe defaults
        so the literal '<short_hash>' placeholder never leaks into the prompt.

        US-017c work sub-agents spawned without a claimed task exercise this
        path — the sub-agent prompt must not contain the unfilled template.
        """
        ctx = SkillContext(
            session_id="x", agent_id="a", is_subagent=False,
            work_task_hash="",
        )
        with patch("proxy.context._get_task_graph_singleton", return_value=mock_tg):
            prompt = WorkSkill().get_system_prompt(ctx)
        assert "<short_hash>" not in prompt, (
            "literal '<short_hash>' must NOT appear — placeholders are "
            "substituted with safe defaults when task_hash is empty"
        )
        assert "(ad-hoc edit session)" in prompt
        assert "(no specific task)" in prompt

    def test_get_system_prompt_fallback_on_taskgraph_failure(self, ctx):
        """When TaskGraph.get_task_enriched raises, return fallback substitution."""
        bad_tg = MagicMock()
        bad_tg.get_task_enriched.side_effect = Exception("Neo4j down")
        with patch("proxy.context._get_task_graph_singleton", return_value=bad_tg), \
             patch("proxy.workspace_state.workspace_store") as mock_ws, \
             patch("proxy.state.make_workspace_id", return_value="sess:default"):
            mock_ws.get_active.return_value = "default"
            prompt = WorkSkill().get_system_prompt(ctx)
        assert "abc12345" in prompt

    def test_get_system_prompt_reads_arg_key_not_task_hash(self, mock_tg):
        """Bug #2: get_system_prompt MUST read the hash from skill_state["arg"]
        (the key the dispatcher writes), not the legacy "task_hash" key.

        Reproduces the bug: a ctx with only skill_state={"task_hash": "abc12345"}
        (wrong key) MUST NOT substitute the hash; a ctx with
        skill_state={"arg": "abc12345"} (correct key) MUST substitute.
        """
        from proxy.skills.skill_definition import SkillRequestContext

        # Wrong key: hash must NOT be substituted (returns bare template).
        ctx_wrong = SkillRequestContext(
            session_id="sess-wrong-key",
            is_subagent=False,
            inherited_skill=None,
            subagent_ticket_mode=False,
            workspace_name="",
            client_hint=None,
            body={},
            seeded_tasks=[],
            mode_state=None,
        )
        ctx_wrong.skill_state = {"task_hash": "abc12345"}
        with patch("proxy.context._get_task_graph_singleton", return_value=mock_tg):
            prompt_wrong = WorkSkill().get_system_prompt(ctx_wrong)
        assert "abc12345" not in prompt_wrong, (
            "Bug #2 regression: get_system_prompt must NOT read the legacy "
            "'task_hash' key — only 'arg' (set by dispatch.py:411)."
        )

        # Correct key: hash MUST be substituted.
        ctx_right = SkillRequestContext(
            session_id="sess-right-key",
            is_subagent=False,
            inherited_skill=None,
            subagent_ticket_mode=False,
            workspace_name="",
            client_hint=None,
            body={},
            seeded_tasks=[],
            mode_state=None,
        )
        ctx_right.skill_state = {"action": "task", "arg": "abc12345"}
        with patch("proxy.context._get_task_graph_singleton", return_value=mock_tg), \
             patch("proxy.workspace_state.workspace_store") as mock_ws:
            mock_ws.get_active.return_value = "default"
            prompt_right = WorkSkill().get_system_prompt(ctx_right)
        assert "abc12345" in prompt_right
        assert "Fix login bug" in prompt_right  # name from mock_tg fixture

    def test_get_system_prompt_reads_modestate_skill_state(self, mock_tg):
        """Bug #2 primary path: get_system_prompt MUST read the hash from the
        active ModeState's skill_state (via agent.snapshot_mode()) — the value
        persisted by SkillDispatcher.activate at activation time. The
        SkillRequestContext built by assemble_system_prompt has no skill_state
        set dynamically, so this is the production read path.
        """
        from types import MappingProxyType
        from proxy.skills.skill_definition import ModeState, SkillRequestContext

        ctx = SkillRequestContext(
            session_id="sess-modestate",
            is_subagent=False,
            inherited_skill=None,
            subagent_ticket_mode=False,
            workspace_name="",
            client_hint=None,
            body={},
            seeded_tasks=[],
            mode_state=None,
        )
        # No skill_state on ctx — production assemble_system_prompt path.
        ctx.skill_state = {}

        # Install a work ModeState on the agent registry with the hash in "arg".
        ms = ModeState(
            name="work",
            activated_at=0.0,
            ttl_seconds=0,
            skill_state=MappingProxyType({"action": "task", "arg": "abc12345"}),
        )

        fake_agent = SimpleNamespace(snapshot_mode=lambda: ms)
        with patch("proxy.context._get_task_graph_singleton", return_value=mock_tg), \
             patch("proxy.workspace_state.workspace_store") as mock_ws, \
             patch("proxy.agent.agent_registry.registry.get_or_load",
                   return_value=fake_agent):
            mock_ws.get_active.return_value = "default"
            prompt = WorkSkill().get_system_prompt(ctx)

        assert "abc12345" in prompt
        assert "Fix login bug" in prompt

    def test_get_system_prompt_ignores_modestate_for_non_work_mode(self, mock_tg):
        """Bug #2 guard: when the active ModeState is NOT 'work' (e.g., ticket
        fallback installed after work-TTL expiry), get_system_prompt MUST NOT
        read its skill_state — work_skill.get_system_prompt is only called
        in work mode, but this guards against accidental cross-skill reads.
        """
        from types import MappingProxyType
        from proxy.skills.skill_definition import ModeState, SkillRequestContext

        ctx = SkillRequestContext(
            session_id="sess-ticket-modestate",
            is_subagent=False,
            inherited_skill=None,
            subagent_ticket_mode=False,
            workspace_name="",
            client_hint=None,
            body={},
            seeded_tasks=[],
            mode_state=None,
        )
        ctx.skill_state = {}

        # Ticket ModeState (not work) — must be ignored.
        ms = ModeState(
            name="ticket",
            activated_at=0.0,
            ttl_seconds=0,
            skill_state=MappingProxyType({"action": "task", "arg": "abc12345"}),
        )
        fake_agent = SimpleNamespace(snapshot_mode=lambda: ms)
        with patch("proxy.context._get_task_graph_singleton", return_value=mock_tg), \
             patch("proxy.agent.agent_registry.registry.get_or_load",
                   return_value=fake_agent):
            prompt = WorkSkill().get_system_prompt(ctx)

        # No work hash found → bare template returned, no substitution.
        assert "abc12345" not in prompt


class TestWorkSkillModeRestrictedActions:
    """Work mode agents have read-only proxy_ticket access (US-095).

    Slice 8b: WorkSkill's tool filter exposes the 3 read-only decomposed
    ticket tools (list_tasks, get_task, append_review_note) and proxy_work;
    the CRUD decomposed tools are NOT visible at all. Sub-agent access is
    enforced via WORK_CAPABILITY_TABLE (proxy_ticket READ_ONLY) and
    proxy_work action-allowlist (bug actions only, lifecycle excluded).
    """

    def test_tool_filter_excludes_proxy_ticket_crud_metadata(self):
        """Slice 8b: the tool filter exposes the decomposed read-only tools;
        the CRUD decomposed tools are NOT visible at all (visible=False in
        WORK_CAPABILITY_TABLE). Work subs have the same filter plus proxy_work.
        """
        tools = WorkSkill().get_tool_filter(SkillContext(
            session_id="x", agent_id="a", is_subagent=False,
        ))
        # Slice 8b decomposed read-only tools ARE visible.
        assert "proxy_ticket_list_tasks" in tools
        assert "proxy_ticket_get_task" in tools
        assert "proxy_ticket_append_review_note" in tools
        # CRUD decomposed tools are NOT visible (capability table marks them
        # visible=False for MAIN; the read-only trio is the only exposure).
        assert "proxy_ticket_create_tasks" not in tools
        assert "proxy_ticket_update_task" not in tools
        assert "proxy_ticket_edit" not in tools
        assert "proxy_ticket_set_parent" not in tools
        assert "proxy_ticket_clear_parent" not in tools
        assert "proxy_ticket_clear_blocked_by" not in tools
        assert "proxy_ticket_set_priority" not in tools
        # Legacy monolithic name no longer exists.
        assert "proxy_ticket" not in tools

    def test_get_sub_agent_tool_filter_contains_readonly_decomposed(self):
        """Work sub-agent filter mirrors MAIN minus `task`; same decomposed
        read-only ticket tools visible, CRUD still excluded."""
        tools = WorkSkill().get_sub_agent_tool_filter(SkillContext(
            session_id="x", agent_id="a", is_subagent=True,
        ))
        assert "proxy_ticket_list_tasks" in tools
        assert "proxy_ticket_get_task" in tools
        assert "proxy_ticket_append_review_note" in tools
        # proxy_work visible to sub-agents (bug actions only at runtime).
        assert "proxy_work" in tools
        # `task` is blocked for sub-agents.
        assert "task" not in tools
        # CRUD decomposed tools still excluded.
        assert "proxy_ticket_create_tasks" not in tools

    def test_work_manager_prompt_references_proxy_ticket_readonly(self):
        """The system prompt must state that proxy_ticket is read-only in work mode."""
        assert "list_tasks" in WORK_MANAGER_PROMPT
        assert "append_review_note" in WORK_MANAGER_PROMPT
        # MUST state that CRUD is not allowed
        assert "read-only" in WORK_MANAGER_PROMPT or "CANNOT" in WORK_MANAGER_PROMPT

    def test_work_sub_prompt_no_lifecycle_actions(self):
        """WORK_SUB_PROMPT must NOT tell sub-agents to call start_work or stop_work."""
        from proxy.tool_parsing.work_prompts import WORK_SUB_PROMPT
        # The "NOT allowed" prohibition list is OK, but the prompt must not
        # instruct subs to call start_work or stop_work as workflow steps.
        # Step 1 of the workflow must be "Do the actual work", not "Call start_work".
        assert "1. Call start_work" not in WORK_SUB_PROMPT, (
            "WORK_SUB_PROMPT step 1 must NOT be 'Call start_work' — "
            "sub-agents are blocked from lifecycle actions."
        )
        assert "stop_work with new_status" not in WORK_SUB_PROMPT, (
            "WORK_SUB_PROMPT must NOT instruct subs to call stop_work with new_status."
        )
        # But it MUST mention the allowed sub-agent actions
        assert "report_bug" in WORK_SUB_PROMPT
        assert "append_review_note" in WORK_SUB_PROMPT


class TestUS092WorkActivationDisplacesTicketMode:
    """US-092: ^^work activation MUST displace ticket mode when active.

    When the user/agent invokes ^^work <task_hash> while ticket mode is
    currently active, the active_mode MUST be replaced with work mode
    (work mode takes over).
    """

    def test_work_activation_displaces_ticket_mode(self):
        """After activating ticket mode, calling ^^work MUST replace
        active_mode with work mode.
        """
        from tests.mode_harness import create_test_agent, activate_ticket_mode, activate_work_mode

        agent = create_test_agent()

        # 1. Activate ticket mode
        activate_ticket_mode(agent, subcommand="plan", task_hash="abcd1234")
        assert agent.is_ticket_mode
        snap = agent.snapshot_mode()
        assert snap.name == "ticket"

        # 2. Activate work mode (simulating ^^work <task_hash>)
        activate_work_mode(agent, task_hash="abcd1234")

        # 3. Ticket mode MUST be displaced
        snap = agent.snapshot_mode()
        assert snap is not None
        assert snap.name == "work", (
            f"US-092: ^^work MUST displace ticket mode. "
            f"active_mode.name={snap.name!r}"
        )
        assert not agent.is_ticket_mode
        assert agent.is_skill_mode

    def test_work_activation_captures_task_hash(self):
        """US-092: ^^work <task_hash> MUST capture the task hash for
        system prompt anchoring.
        """
        from tests.mode_harness import create_test_agent, activate_work_mode

        agent = create_test_agent()
        mode = activate_work_mode(agent, task_hash="deadbeef")
        snap = agent.snapshot_mode()
        assert snap.name == "work"
        assert snap.skill_state.get("task_hash") == "deadbeef"
