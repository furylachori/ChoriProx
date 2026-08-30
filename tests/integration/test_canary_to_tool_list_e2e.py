#!/usr/bin/env python3
"""End-to-end test for canary → tool list flow.

This test reproduces the ACTUAL production bug: a parent injects a
WORKSPACE canary, the sub-agent receives it, and the proxy must
filter the tool list to NOT include proxy_ticket / proxy_work / or
any of the 10 decomposed proxy_ticket_* tools for sub-agents without
an active skill.

Previously, the WORKSPACE canary was rejected by parse_canary
because "workspace" isn't a registered skill. This test catches
that regression.

Slice 8b: the monolithic ``proxy_ticket`` is decomposed into 10
``proxy_ticket_*`` tools. Bare sub-agents (no skill) must not see
ANY of them; ticket sub-agents see the read-only subset.
"""
import os
import time
import sys
import unittest

sys.path.insert(0, "src")


class TestCanaryToToolList(unittest.TestCase):
    """End-to-end: sign canary → detect_subagent_ingress → prepare_tooling_payload.

    Uses the production sign_canary() to avoid format mismatches.
    """
    SECRET = "test-secret-1234"
    WORKSPACE = "choriprox"

    def setUp(self):
        from proxy import canary as canary_mod
        self._had_secret = "PROXY_SESSION_SECRET" in os.environ
        self._orig_secret = os.environ.pop("PROXY_SESSION_SECRET", None)
        os.environ["PROXY_SESSION_SECRET"] = self.SECRET
        self._orig_mod_secret = canary_mod._SESSION_SECRET
        canary_mod._SESSION_SECRET = self.SECRET

    def tearDown(self):
        from proxy import canary as canary_mod
        canary_mod._SESSION_SECRET = self._orig_mod_secret
        os.environ.pop("PROXY_SESSION_SECRET", None)
        if self._had_secret:
            os.environ["PROXY_SESSION_SECRET"] = self._orig_secret

    def _make_workspace_canary(self) -> str:
        """Sign a WORKSPACE canary with the production HMAC format.

        The parser at canary.py:410-413 builds the token as
        `workspace:<name>:<sig>:<ts>` and calls verify_canary with
        state_str="". The verifier computes message =
        `f"workspace:<name>:{state_str}:{ts}"` = `f"workspace:<name>::<ts>"`
        (with empty middle field). Sign accordingly.
        """
        import hashlib
        import hmac
        ts = int(time.time())
        # Production HMAC format: empty state_str produces "<skill>::<ts>"
        message = f"workspace:{self.WORKSPACE}::{ts}"
        sig = hmac.new(self.SECRET.encode(), message.encode(), hashlib.sha256).hexdigest()
        return f"[[__SUBAGENT__:WORKSPACE:{self.WORKSPACE}:{sig}:{ts}]]"

    def _make_skill_canary(self, skill: str) -> str:
        """Sign a SKILL canary with the production format.

        Production (proxy/mode.py:758-770) wraps sign_canary() output as
        `[[__SUBAGENT__:SKILL:<skill>:<sig:ts>]]`.
        """
        from proxy.canary import sign_canary
        sig_ts = sign_canary(skill)
        return f"[[__SUBAGENT__:SKILL:{skill}:{sig_ts}]]"

    def _make_body(self, canary: str) -> dict:
        """Build a body with the canary prepended to the first user message."""
        return {
            "messages": [
                {"role": "user", "content": f"{canary}\n\nDo something useful"}
            ],
            "tools": [
                {"type": "function", "function": {"name": "bash"}},
                {"type": "function", "function": {"name": "read"}},
                {"type": "function", "function": {"name": "proxy_ticket"}},
                {"type": "function", "function": {"name": "proxy_work"}},
                {"type": "function", "function": {"name": "task"}},
            ],
        }

    def test_workspace_canary_does_not_revert_to_bare(self):
        """WORKSPACE canary must extract workspace_name and not set fake skill name.

        The previous code set inherited_skill="workspace" (a fake name),
        which caused downstream skill_resolver to fail to activate any mode.
        Now the WORKSPACE canary sets inherited_skill=None and the
        workspace_name is extracted from the canary state.
        """
        from proxy.canary import detect_subagent_ingress

        canary = self._make_workspace_canary()
        body = self._make_body(canary)
        ingress = detect_subagent_ingress(body, session_id="sub-agent-session")

        self.assertTrue(ingress.is_subagent, "should be marked as sub-agent")
        self.assertEqual(
            ingress.workspace_name, self.WORKSPACE,
            "workspace_name must be extracted from WORKSPACE canary"
        )
        # WORKSPACE canary does NOT grant a skill mode — it only identifies
        # the workspace. Setting inherited_skill=None (default mode) keeps
        # the sub-agent out of the proxy_* tool list.
        self.assertIsNone(
            ingress.inherited_skill,
            f"WORKSPACE canary must not set fake skill, got {ingress.inherited_skill!r}"
        )

    def test_workspace_canary_tool_list_excludes_proxy_ticket(self):
        """WORKSPACE canary → bare sub-agent → proxy_ticket / decomposed tools NOT in tool list."""
        from proxy.tool_pipeline.tool_registry import _PROXY_ADMIN_TOOLS
        from proxy.canary import detect_subagent_ingress
        from proxy.tool_cache import prepare_tooling_payload

        canary = self._make_workspace_canary()
        body = self._make_body(canary)
        ingress = detect_subagent_ingress(body, session_id="sub-agent-session")

        result = prepare_tooling_payload(
            body=body, model_cfg=None, ticket_mode=False,
            is_subagent=True, skill_name=ingress.inherited_skill,
            session_id="sub-agent-session", agent_id="sub-agent-id",
        )
        tool_names = [t.get("function", {}).get("name", "?") for t in result.tools]

        leaked = _PROXY_ADMIN_TOOLS & set(tool_names)
        self.assertFalse(
            leaked,
            f"bare sub-agent must not see proxy admin tools (monolithic + decomposed); "
            f"got leaked: {sorted(leaked)}",
        )

    def test_skill_canary_ticket_includes_proxy_ticket(self):
        """SKILL:ticket canary → ticket sub-agent → decomposed proxy_ticket_* tools ARE in tool list.

        Slice 8b: the monolithic ``proxy_ticket`` is decomposed into 10
        ``proxy_ticket_*`` tools. A ticket sub-agent sees the read-only
        decomposed tools (the LLM-facing subset of ticket MAIN) but NOT
        the write decomposed tools (those are MAIN-only via inheritance).
        """
        from proxy.canary import detect_subagent_ingress
        from proxy.tool_cache import prepare_tooling_payload

        canary = self._make_skill_canary("ticket")
        body = self._make_body(canary)
        ingress = detect_subagent_ingress(body, session_id="sub-agent-session")
        self.assertEqual(ingress.inherited_skill, "ticket")

        result = prepare_tooling_payload(
            body=body, model_cfg=None, ticket_mode=True,
            is_subagent=True, skill_name=ingress.inherited_skill,
            session_id="sub-agent-session", agent_id="sub-agent-id",
        )
        tool_names = [t.get("function", {}).get("name", "?") for t in result.tools]

        self.assertIn(
            "proxy_ticket_list_tasks", tool_names,
            "ticket sub-agent must see proxy_ticket_list_tasks (decomposed read)",
        )
        self.assertIn(
            "proxy_ticket_get_task", tool_names,
            "ticket sub-agent must see proxy_ticket_get_task (decomposed read)",
        )
        # The monolithic ``proxy_ticket`` is admin and never appears in
        # the LLM-facing tool list.
        self.assertNotIn(
            "proxy_ticket", tool_names,
            "ticket sub-agent must NOT see the monolithic proxy_ticket "
            "(decomposed into 10 tools in Slice 8b)",
        )

    def test_bare_subagent_tool_list_excludes_proxy_ticket(self):
        """No canary → bare sub-agent → proxy_ticket and decomposed tools NOT in tool list."""
        from proxy.tool_pipeline.tool_registry import _PROXY_ADMIN_TOOLS
        from proxy.canary import detect_subagent_ingress
        from proxy.tool_cache import prepare_tooling_payload

        body = self._make_body("[[__SUBAGENT__]]\n\nDo something")
        ingress = detect_subagent_ingress(body, session_id="sub-agent-session")
        self.assertIsNone(ingress.inherited_skill, "bare canary should not set a skill")

        result = prepare_tooling_payload(
            body=body, model_cfg=None, ticket_mode=False,
            is_subagent=True, skill_name=ingress.inherited_skill,
            session_id="sub-agent-session", agent_id="sub-agent-id",
        )
        tool_names = [t.get("function", {}).get("name", "?") for t in result.tools]

        leaked = _PROXY_ADMIN_TOOLS & set(tool_names)
        self.assertFalse(
            leaked,
            f"bare sub-agent must not see proxy admin tools; got leaked: {sorted(leaked)}",
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
