"""C4 (Commit 4): mode-entry hygiene behavioral tests.

After C4:
- Mode banner inside MANDATORY_SYSTEM_INSTRUCTIONS so the model knows
  "this session is temporarily in ticket/work mode" (4.1).
- First-message duplication fixed: non-system messages[0] is NOT embedded
  in TASK_PAYLOAD (4.2).
- TASK_PAYLOAD framing softened to "CLIENT CONTEXT — background reference"
  (4.3).
- Rate-limit checks happen BEFORE dispatcher.dispatch() so a 429'd request
  doesn't leave a zombie mode (4.4).
- Tools without a matching client schema are SKIPPED (not emitted as a
  malformed bare-name stub) (4.5).

Each test has a corresponding revert-test in C4 STEP 4.
"""
import unittest
from unittest.mock import MagicMock, patch


def _build_ctx(*, session_id="sess-c4", is_subagent=False, messages=None):
    """Build a RequestContext-like MagicMock for skill_prompts / mode tests."""
    ctx = MagicMock()
    ctx.session_id = session_id
    ctx.is_subagent = is_subagent
    ctx.remote_addr = "127.0.0.1"
    ctx.workspace_name = ""
    ctx.client_hint = ""
    ctx.body = {}
    ctx.inherited_skill = None
    ctx.subagent_ticket_mode = False
    ctx._skill_activated = False
    ctx._mode = None
    ctx.messages = messages or []
    return ctx


class TestBannerPresentInTicketMode(unittest.TestCase):
    """C4 (a): ticket-mode prompt contains the [PROXY MODE: TICKET — one-shot] banner."""

    def test_banner_present_in_ticket_mode(self):
        from proxy.skill_prompts import assemble_system_prompt
        from skills.ticket_skill import TicketSkill
        from proxy.skills.skill_registry import SkillRegistry
        from tests.mode_harness import activate_ticket_mode, create_test_agent

        agent = create_test_agent(session_id="sess-c4-banner-ticket", agent_id="agent-c4-banner-ticket")
        activate_ticket_mode(agent)

        ctx = _build_ctx(
            session_id="sess-c4-banner-ticket",
            messages=[{"role": "user", "content": "hi"}],
        )
        body = {
            "messages": [
                {"role": "user", "content": "plan a thing"},
                {"role": "user", "content": "do it"},
            ],
            "tools": [],
            "model": "test-model",
        }

        prompt = assemble_system_prompt(agent=agent, ctx=ctx, body=body, registry=SkillRegistry())
        self.assertIn(
            "[PROXY MODE: TICKET — one-shot]",
            prompt,
            f"C4 REGRESSION: ticket-mode banner missing; prompt start: {prompt[:300]!r}",
        )


class TestBannerPresentInWorkMode(unittest.TestCase):
    """C4 (b): work-mode prompt contains the [PROXY MODE: WORK — one-shot] banner."""

    def test_banner_present_in_work_mode(self):
        from proxy.skill_prompts import assemble_system_prompt
        from skills.work_skill import WorkSkill
        from proxy.skills.skill_registry import SkillRegistry
        from tests.mode_harness import activate_work_mode, create_test_agent

        agent = create_test_agent(session_id="sess-c4-banner-work", agent_id="agent-c4-banner-work")
        activate_work_mode(agent, task_hash="deadbeef")

        ctx = _build_ctx(
            session_id="sess-c4-banner-work",
            messages=[{"role": "user", "content": "hi"}],
        )
        body = {
            "messages": [{"role": "user", "content": "do work"}],
            "tools": [],
            "model": "test-model",
        }

        prompt = assemble_system_prompt(agent=agent, ctx=ctx, body=body, registry=SkillRegistry())
        self.assertIn(
            "[PROXY MODE: WORK — one-shot]",
            prompt,
            f"C4 REGRESSION: work-mode banner missing; prompt start: {prompt[:300]!r}",
        )


class TestNonSystemMessages0NotDuplicated(unittest.TestCase):
    """C4 (c): non-system messages[0] appears exactly ONCE (not embedded in TASK_PAYLOAD)."""

    def test_non_system_messages0_not_duplicated(self):
        from proxy.skill_prompts import assemble_system_prompt
        from skills.ticket_skill import TicketSkill
        from proxy.skills.skill_registry import SkillRegistry
        from tests.mode_harness import activate_ticket_mode, create_test_agent

        agent = create_test_agent(session_id="sess-c4-dedup", agent_id="agent-c4-dedup")
        activate_ticket_mode(agent)

        unique_marker = "important question " + "x" * 32  # distinctive enough to grep
        ctx = _build_ctx(
            session_id="sess-c4-dedup",
            messages=[{"role": "user", "content": unique_marker}],
        )
        body = {
            "messages": [{"role": "user", "content": unique_marker}],
            "tools": [],
            "model": "test-model",
        }

        prompt = assemble_system_prompt(agent=agent, ctx=ctx, body=body, registry=SkillRegistry())

        # The marker should NOT appear in the system prompt (task_text = "" for
        # non-system messages). The marker IS still in body["messages"] (sent to
        # the LLM separately), but not embedded in TASK_PAYLOAD.
        # Revert-test 4.1: revert to task_text = raw and this FAILS (count >= 1).
        occurrences = prompt.count(unique_marker)
        self.assertEqual(
            occurrences, 0,
            f"C4 REGRESSION: non-system messages[0] embedded in system prompt {occurrences} times; expected 0",
        )
        # Verify the message is still in body["messages"] (sent to LLM separately).
        self.assertEqual(
            body["messages"][0]["content"], unique_marker,
            "Original message should still be in body['messages']",
        )


class TestTaskPayloadSaysClientContext(unittest.TestCase):
    """C4 (d): TASK_PAYLOAD says 'CLIENT CONTEXT', not 'untrusted user data'."""

    def test_task_payload_says_client_context(self):
        from proxy.skill_prompts import assemble_system_prompt
        from skills.ticket_skill import TicketSkill
        from proxy.skills.skill_registry import SkillRegistry
        from tests.mode_harness import activate_ticket_mode, create_test_agent

        agent = create_test_agent(session_id="sess-c4-client", agent_id="agent-c4-client")
        activate_ticket_mode(agent)

        # System message so extract_client_instructions runs and produces TASK_PAYLOAD.
        system_content = "client instructions " + "y" * 32
        ctx = _build_ctx(
            session_id="sess-c4-client",
            messages=[{"role": "system", "content": system_content}],
        )
        body = {
            "messages": [{"role": "system", "content": system_content}],
            "tools": [],
            "model": "test-model",
        }

        prompt = assemble_system_prompt(agent=agent, ctx=ctx, body=body, registry=SkillRegistry())

        # Revert-test 4.5: revert framing to "untrusted user data" and these FAIL.
        self.assertIn(
            "CLIENT CONTEXT",
            prompt,
            f"C4 REGRESSION: TASK_PAYLOAD does not say 'CLIENT CONTEXT'; got prompt tail: {prompt[-500:]!r}",
        )
        self.assertNotIn(
            "untrusted user data",
            prompt,
            f"C4 REGRESSION: TASK_PAYLOAD still says 'untrusted user data'",
        )


class TestRateLimitBeforeActivation(unittest.TestCase):
    """C4 (e): rate-limit runs BEFORE dispatcher.dispatch(), so 429 leaves mode un-set."""

    def test_rate_limit_before_activation(self):
        from proxy.pipeline.dispatch import try_skill_activation
        from tests.mode_harness import create_test_agent

        local_agent = create_test_agent(
            session_id="sess-c4-ratelimit",
            agent_id="agent-c4-ratelimit",
        )

        ctx = MagicMock()
        ctx.session_id = "sess-c4-ratelimit"
        ctx.is_subagent = False
        ctx.remote_addr = "127.0.0.1"
        ctx.messages = [{"role": "user", "content": "^^ticket plan a thing"}]
        ctx.body = {}
        ctx.inherited_skill = None

        # Mock the rate limiter to deny
        denied_result = MagicMock()
        denied_result.allowed = False
        denied_result.retry_after = 60

        # Mock detect_command to return a fake skill (so rate-limit runs)
        fake_skill = MagicMock()
        fake_skill.name = "ticket"
        mock_dispatcher_instance = MagicMock()
        mock_dispatcher_instance.detect_command.return_value = fake_skill

        features = {}

        with patch(
            "proxy.pipeline.dispatch.skill_rate_limiter"
        ) as mock_session_limiter, patch(
            "proxy.skill_dispatcher.SkillDispatcher"
        ) as mock_dispatcher_cls, patch(
            "proxy.pipeline.dispatch._get_agent", return_value=local_agent
        ):
            mock_session_limiter.check.return_value = denied_result
            mock_dispatcher_cls.return_value = mock_dispatcher_instance

            result = try_skill_activation(ctx, features)

        # Verify 429 response
        self.assertIsNotNone(result)
        self.assertEqual(result.status_code, 429, (
            f"C4 REGRESSION: rate-limit denial should produce 429; got status_code={result.status_code}"
        ))
        # Verify dispatcher.dispatch was NEVER called (rate-limit blocks it)
        mock_dispatcher_instance.dispatch.assert_not_called()
        # Verify the PRODUCTION agent (returned by _get_agent) is unmodified.
        # This is the real check — the local_agent IS the production agent because
        # we patched _get_agent to return it.
        self.assertIsNone(
            local_agent.snapshot_mode(),
            "C4.5 REGRESSION: production agent had mode installed despite rate-limit denial",
        )


class TestEmptySchemaToolSkipped(unittest.TestCase):
    """C4 (f): tools without a matching client schema are SKIPPED (not emitted as stub)."""

    def test_empty_schema_tool_skipped(self):
        from proxy.mode import Mode, apply_mode
        from skills.ticket_skill import TicketSkill
        from proxy.skills.skill_registry import SkillRegistry
        from tests.mode_harness import activate_ticket_mode, create_test_agent

        agent = create_test_agent(session_id="sess-c4-skip", agent_id="agent-c4-skip")
        activate_ticket_mode(agent)

        ctx = MagicMock()
        ctx.session_id = "sess-c4-skip"
        ctx.is_subagent = False
        ctx.remote_addr = ""
        ctx.workspace_name = ""
        ctx.client_hint = ""
        ctx.body = {}
        ctx.inherited_skill = None
        ctx.subagent_ticket_mode = False
        ctx._skill_activated = False
        ctx._mode = None
        ctx.messages = []
        ctx.model_name = "test-model"
        ctx.model_cfg = MagicMock()

        body = {
            "messages": [{"role": "user", "content": "hi"}],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "known_tool",
                        "description": "Known",
                        "parameters": {"type": "object", "properties": {}},
                    },
                },
                # No schema for "orphan_tool" — it should be skipped
            ],
            "model": "test-model",
        }

        # Mock select_tools to return a ToolMeta for orphan_tool (no matching schema)
        from proxy.tool_pipeline.tool_registry import ToolMeta
        from proxy.types import ToolingPayload
        orphan_meta = MagicMock(spec=ToolMeta)
        orphan_meta.name = "orphan_tool"
        known_meta = MagicMock(spec=ToolMeta)
        known_meta.name = "known_tool"

        # Bypass skill-mode tool filtering so the test body["tools"] schema
        # for "known_tool" survives into select_tools' tool_by_name map.
        no_augment = ToolingPayload(
            tools=[], tool_hash="", cached=False, augmented_tools=[],
        )

        with patch("proxy.skill_tools.select_tools", return_value=[known_meta, orphan_meta]), \
             patch("proxy.tool_cache.prepare_tooling_payload", return_value=no_augment), \
             patch("proxy.context._get_task_graph_singleton", return_value=MagicMock()):
            with self.assertLogs("proxy.mode", level="WARNING") as log_ctx:
                apply_mode(ctx, agent, body, Mode.SKILL)

        tool_names = [
            t.get("function", {}).get("name", "")
            for t in body.get("tools", [])
        ]
        # Revert-test 4.3: revert to bare-name stub and this FAILS (orphan_tool would be present).
        self.assertNotIn(
            "orphan_tool", tool_names,
            f"C4 REGRESSION: orphan_tool present in body['tools'] (should be skipped); got {tool_names}",
        )
        self.assertIn(
            "known_tool", tool_names,
            f"known_tool should be present; got {tool_names}",
        )
        # Verify the WARNING was logged
        orphan_warnings = [r for r in log_ctx.output if "orphan_tool" in r]
        self.assertTrue(
            len(orphan_warnings) >= 1,
            f"C4 REGRESSION: expected WARNING about orphan_tool; got warnings: {log_ctx.output}",
        )


if __name__ == "__main__":
    unittest.main()