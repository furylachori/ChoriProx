"""Reproduction test for the canary-not-inheriting-skill bug.

The user reports: when a parent in ticket mode spawns a sub-agent via the
`task` tool, the canary injected into the task prompt is the BARE form
([[__SUBAGENT__]]) instead of the skill-aware form
([[__SUBAGENT__:SKILL:ticket:HMAC:TS]]). This causes the sub-agent to
arrive at the proxy with subagent_ticket_mode=False, and the ticket-mode
tool filter (which includes proxy_ticket) is not applied.

This test reproduces the issue using the real (unmocked) SkillRegistry
singleton, which is the same code path the production proxy uses.

Wave 4: PostProcessStage.execute() replaces the direct apply_post_process
call. The 8 unrelated sub-methods are patched as no-ops so only the
canary-injection + lifecycle finalize paths run.
"""
import json
import os
import sys
import unittest
from unittest.mock import MagicMock, patch

# Ensure project root is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))

from contextlib import ExitStack


def _noop_pp_submethods_repro():
    """Patch the 8 PostProcessStage sub-methods to no-ops (canary repro helper)."""
    from proxy.pipeline.post_process import PostProcessStage
    stack = ExitStack()
    noop = lambda self, ctx: None
    stack.enter_context(patch.object(PostProcessStage, "_extract_minimax_reasoning", noop))
    stack.enter_context(patch.object(PostProcessStage, "_scan_write_results", noop))
    stack.enter_context(patch.object(PostProcessStage, "_sync_chat_history", noop))
    stack.enter_context(patch.object(PostProcessStage, "_persist_messages", noop))
    stack.enter_context(patch.object(PostProcessStage, "_populate_vfs_cache", noop))
    stack.enter_context(patch.object(PostProcessStage, "_capture_cache_metrics", noop))
    stack.enter_context(patch.object(PostProcessStage, "_handle_tool_interception", noop))
    stack.enter_context(patch.object(PostProcessStage, "_assign_model_name", noop))
    return stack


def _patch_get_or_load_repro(agent):
    """Return a patcher that makes agent_registry.get_or_load return the test's agent."""
    def _get_or_load(session_id):
        return agent
    return patch(
        "proxy.agent.agent_registry.registry.get_or_load",
        side_effect=_get_or_load,
    )


class TestCanaryInheritanceReproduction(unittest.TestCase):
    """Reproduce the bare-canary-when-parent-is-in-ticket-mode bug."""

    def setUp(self):
        """Set up the real SkillRegistry and a parent agent in ticket mode."""
        # Make the real skill canary signer work. Capture the original so
        # tearDown can restore it (avoids leaking test env into other tests).
        self._original_secret = os.environ.get("PROXY_SESSION_SECRET")
        os.environ.setdefault("PROXY_SESSION_SECRET", "test-secret-for-canary-sig")
        # Reset the SkillRegistry singleton so we can re-register and the
        # canary sig is reproducible.
        from proxy.skills.skill_registry import reset_default_registry
        reset_default_registry()
        from proxy.skills.skill_registry import SkillRegistry
        self.registry = SkillRegistry()
        # Confirm ticket skill is registered.
        self.ticket_skill = self.registry.get("ticket")
        self.assertIsNotNone(
            self.ticket_skill,
            "TicketSkill must be registered for this test to be meaningful",
        )
        self.work_skill = self.registry.get("work")
        self.assertIsNotNone(self.work_skill, "WorkSkill must be registered")

    def tearDown(self):
        """Restore env vars and reset singleton to avoid cross-test pollution."""
        if self._original_secret is not None:
            os.environ["PROXY_SESSION_SECRET"] = self._original_secret
        else:
            os.environ.pop("PROXY_SESSION_SECRET", None)
        try:
            from proxy.skills.skill_registry import reset_default_registry
            reset_default_registry()
        except Exception:
            pass

    def _build_parent_response(self):
        """A parent's response that includes a `task` tool call."""
        return {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "Delegating to a sub-agent.",
                        "tool_calls": [
                            {
                                "id": "call_task_repro",
                                "type": "function",
                                "function": {
                                    "name": "task",
                                    "arguments": json.dumps({
                                        "description": "Investigate",
                                        "prompt": "Look at the auth bug.",
                                    }),
                                },
                            }
                        ],
                    },
                    "finish_reason": "tool_calls",
                }
            ]
        }

    def _build_agent_in_ticket_mode(self):
        """Build a real Agent with ticket mode ACTIVE (not a MagicMock)."""
        from proxy.agent.agent import Agent
        from proxy.skills.skill_definition import ModeState

        # Create a fresh agent
        agent = Agent(agent_id="parent_repro", session_id="sess_repro")

        # Activate ticket mode with a real ModeState
        mode_state = ModeState(
            name="ticket",
            activated_at=__import__("time").time(),
            ttl_seconds=300,
            skill_state={},
        )
        ok = agent.set_active_mode(mode_state)
        self.assertTrue(ok, "Failed to set ticket mode on agent")
        return agent

    def test_real_ticket_mode_yields_skill_canary(self):
        """Parent in REAL ticket mode → injected canary must be SKILL-aware.

        The reproduction uses the real Agent class (not MagicMock) and the
        real SkillRegistry singleton (not a mocked one). This is the code
        path the production proxy uses.
        """
        from proxy.mode import Mode
        from proxy.pipeline.post_process import PostProcessStage

        agent = self._build_agent_in_ticket_mode()
        parent_response = self._build_parent_response()

        # Build a minimal ctx. Mode is SKILL (parent has ticket mode active).
        ctx = MagicMock()
        ctx.session_id = agent.session_id
        ctx.agent_id = agent.agent_id
        ctx.is_subagent = False
        ctx.subagent_ticket_mode = False
        ctx.seeded_tasks = []
        ctx._ticket_activated = False
        ctx.inherited_skill = None
        ctx.messages = parent_response["choices"][0]["message"].get("content", "")
        ctx.body = {"messages": [{"role": "user", "content": "Investigate the auth bug"}]}
        ctx._mode = Mode.SKILL
        ctx.model_route = MagicMock()
        ctx.model_route.is_remote = False
        ctx.model_route.model_cfg = None
        ctx.upstream = None
        ctx.swarm_policy = None
        ctx.llama_body = ctx.body
        ctx.request_id = "req_repro_ticket"
        ctx.client_hint = ""
        ctx.response_data = parent_response

        with _patch_get_or_load_repro(agent), _noop_pp_submethods_repro():
            result = PostProcessStage().execute(ctx)

        # Pull the task prompt back out
        task_args = json.loads(
            result["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"]
        )
        task_prompt = task_args["prompt"]
        print(f"\n[REPRO] Injected canary: {task_prompt[:100]!r}")

        # THE ASSERTION: canary must be skill-aware, not bare
        self.assertTrue(
            task_prompt.startswith("[[__SUBAGENT__:SKILL:ticket"),
            f"BUG REPRODUCED: ticket-mode parent injected BARE canary, "
            f"not skill-aware. Got prompt starts: {task_prompt[:80]!r}",
        )

    def test_real_work_mode_yields_skill_canary(self):
        """Parent in REAL work mode → injected canary must be SKILL:work:..."""
        from proxy.mode import Mode
        from proxy.pipeline.post_process import PostProcessStage
        from proxy.agent.agent import Agent
        from proxy.skills.skill_definition import ModeState

        agent = Agent(agent_id="parent_work_repro", session_id="sess_work_repro")
        mode_state = ModeState(
            name="work",
            activated_at=__import__("time").time(),
            ttl_seconds=300,
            skill_state={},
        )
        agent.set_active_mode(mode_state)

        parent_response = self._build_parent_response()
        ctx = MagicMock()
        ctx.session_id = agent.session_id
        ctx.agent_id = agent.agent_id
        ctx.is_subagent = False
        ctx.subagent_ticket_mode = False
        ctx.seeded_tasks = []
        ctx._ticket_activated = False
        ctx.inherited_skill = None
        ctx.messages = parent_response["choices"][0]["message"].get("content", "")
        ctx.body = {"messages": [{"role": "user", "content": "Investigate"}]}
        ctx._mode = Mode.SKILL
        ctx.model_route = MagicMock()
        ctx.model_route.is_remote = False
        ctx.model_route.model_cfg = None
        ctx.upstream = None
        ctx.swarm_policy = None
        ctx.llama_body = ctx.body
        ctx.request_id = "req_repro_work"
        ctx.client_hint = ""
        ctx.response_data = parent_response

        # Note: mode is SKILL because work mode is mapped to SKILL in resolve_mode
        # but the canary-injection check used to be gated on SKILL/TICKET/SKILL_SUBAGENT.
        # The post-Wave-4 inject_task_canaries fires for ALL modes (the bug fix
        # originally documented in the commit 4a7b47e comment).
        with _patch_get_or_load_repro(agent), _noop_pp_submethods_repro():
            result = PostProcessStage().execute(ctx)

        task_args = json.loads(
            result["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"]
        )
        task_prompt = task_args["prompt"]
        print(f"\n[REPRO] Work mode injected canary: {task_prompt[:100]!r}")

        self.assertTrue(
            task_prompt.startswith("[[__SUBAGENT__:SKILL:work"),
            f"BUG: work-mode parent injected non-work canary. "
            f"Got prompt starts: {task_prompt[:80]!r}",
        )


if __name__ == "__main__":
    unittest.main()
