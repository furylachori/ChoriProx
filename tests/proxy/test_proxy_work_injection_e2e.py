"""C1 (Commit 1) proxy-layer end-to-end tests for PROXY-MODES-PLAN.

C1 modifies `_get_injectable_memory_tools` in `src/proxy/tool_cache.py` to
branch on `skill_name == "work"` and return BOTH `PROXY_TICKET_TOOL_DEFINITION`
and `PROXY_WORK_TOOL_DEFINITION` (pre-fix it only returned
`PROXY_TICKET_TOOL_DEFINITION` for any skill_name).

This file pins the regression at the real call site — `apply_mode()` and
`prepare_tooling_payload()` — so a future refactor cannot silently
reintroduce the bug. Four tests:

1. `test_apply_mode_work_injects_proxy_work` — work mode MUST inject both
   proxy_work AND proxy_ticket (the regression target).
2. `test_apply_mode_ticket_does_not_inject_proxy_work` — ticket mode MUST
   inject proxy_ticket and MUST NOT inject proxy_work (work-only).
3. `test_apply_mode_default_mode_no_proxy_injection` — DEFAULT mode MUST
   inject neither (no skill = no proxy_*).
4. `test_cache_bleed_work_vs_ticket_same_body` — the cache key for
   `prepare_tooling_payload` must isolate per `skill_name`; priming with
   "work" must NOT bleed `proxy_work` into a later "ticket" call.

The pattern mirrors `tests/proxy/test_mode_resolution.py:209-268` — REAL
`WorkSkill` / `TicketSkill` instances fed via a MagicMock-backed
`SkillRegistry`, with `proxy.context.select_context` / `proxy.tool_cache.{get,
cache}_tool_schema` patched to keep the test hermetic.
"""
import unittest
from unittest.mock import MagicMock, patch


# ---------------------------------------------------------------------------
# Helpers (RequestContext-like factory — mirrors tests/proxy/test_mode_resolution.py:101-134)
# ---------------------------------------------------------------------------


def _make_proxy_body():
    """Minimal body with two tool schemas (read, bash) for filtering tests."""
    return {
        "messages": [{"role": "user", "content": "do work"}],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "read",
                    "description": "r",
                    "parameters": {"type": "object", "properties": {}},
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "bash",
                    "description": "b",
                    "parameters": {"type": "object", "properties": {}},
                },
            },
        ],
    }


def _make_proxy_ctx(*, session_id, body):
    """Build a RequestContext-like MagicMock for apply_mode tests.

    The fields match what `proxy.mode.apply_mode` and the downstream
    `prepare_tooling_payload` read — see tests/proxy/test_mode_resolution.py
    for the canonical field set.
    """
    ctx = MagicMock()
    ctx.body = body
    ctx.session_id = session_id
    ctx.is_subagent = False
    ctx.subagent_ticket_mode = False
    ctx.inherited_skill = None
    ctx.workspace_name = ""
    ctx.seeded_tasks = []
    ctx._skill_activated = False
    ctx._mode = None
    ctx.model_cfg = MagicMock()
    ctx.messages = body.get("messages", [])
    ctx.model_name = "test-model"
    ctx.client_hint = ""
    return ctx


# ---------------------------------------------------------------------------
# C1 tests — proxy_work injection in work mode
# ---------------------------------------------------------------------------


class TestProxyWorkInjectionE2E(unittest.TestCase):
    """End-to-end tests for C1: `apply_mode()` must inject `proxy_work` for
    work-mode agents, and the cache must isolate per `skill_name`.
    """

    @patch("proxy.context.select_context", return_value=None)
    @patch("proxy.tool_cache.cache_tool_schema")
    @patch("proxy.tool_cache.get_cached_tool_schema", return_value=None)
    @patch("proxy.skills.skill_registry.SkillRegistry")
    def test_apply_mode_work_injects_proxy_work(
        self, mock_registry_cls, mock_get_cached, mock_cache, mock_select
    ):
        """REGRESSION (C1): apply_mode() for a work-mode agent MUST inject
        proxy_work AND proxy_ticket in body["tools"]. Pre-fix the injection
        ignored skill_name and only injected proxy_ticket, so proxy_work was
        absent and the LLM had to hallucinate the calls.
        """
        from skills.work_skill import WorkSkill
        from tests.mode_harness import activate_work_mode, create_test_agent

        real_skill = WorkSkill()
        mock_registry = MagicMock()
        mock_registry.get.return_value = real_skill
        mock_registry_cls.return_value = mock_registry

        agent = create_test_agent(
            session_id="sess_c1_proxy_work",
            agent_id="agent_c1_proxy_work",
        )
        activate_work_mode(agent, task_hash="deadbeef")

        body = _make_proxy_body()
        ctx = _make_proxy_ctx(session_id="sess_c1_proxy_work", body=body)

        from proxy.mode import apply_mode, Mode
        apply_mode(ctx, agent, ctx.body, Mode.SKILL)

        tool_names = [
            t.get("function", {}).get("name", "")
            for t in ctx.body.get("tools", [])
        ]
        self.assertIn(
            "proxy_work",
            tool_names,
            f"C1 REGRESSION: apply_mode(work) did NOT inject proxy_work; "
            f"tools={tool_names}",
        )
        # Slice 8b: decomposed proxy_ticket_* tools (work MAIN sees 3 read/append).
        for tool_name in ("proxy_ticket_list_tasks", "proxy_ticket_get_task", "proxy_ticket_append_review_note"):
            self.assertIn(
                tool_name,
                tool_names,
                f"C1 REGRESSION: apply_mode(work) did NOT inject {tool_name}; "
                f"tools={tool_names}",
            )

    @patch("proxy.context.select_context", return_value=None)
    @patch("proxy.tool_cache.cache_tool_schema")
    @patch("proxy.tool_cache.get_cached_tool_schema", return_value=None)
    @patch("proxy.skills.skill_registry.SkillRegistry")
    def test_apply_mode_ticket_does_not_inject_proxy_work(
        self, mock_registry_cls, mock_get_cached, mock_cache, mock_select
    ):
        """C1: apply_mode() for a ticket-mode agent MUST inject proxy_ticket
        but NOT proxy_work. proxy_work is work-mode-only.
        """
        from skills.ticket_skill import TicketSkill
        from tests.mode_harness import activate_ticket_mode, create_test_agent

        real_skill = TicketSkill()
        mock_registry = MagicMock()
        mock_registry.get.return_value = real_skill
        mock_registry_cls.return_value = mock_registry

        agent = create_test_agent(
            session_id="sess_c1_proxy_ticket",
            agent_id="agent_c1_proxy_ticket",
        )
        activate_ticket_mode(agent)

        body = _make_proxy_body()
        ctx = _make_proxy_ctx(session_id="sess_c1_proxy_ticket", body=body)

        from proxy.mode import apply_mode, Mode
        apply_mode(ctx, agent, ctx.body, Mode.SKILL)

        tool_names = [
            t.get("function", {}).get("name", "")
            for t in ctx.body.get("tools", [])
        ]
        # Slice 8b: decomposed proxy_ticket_* tools (ticket MAIN sees 2 read-only).
        self.assertIn(
            "proxy_ticket_list_tasks",
            tool_names,
            f"C1 REGRESSION: apply_mode(ticket) did NOT inject "
            f"proxy_ticket_list_tasks; tools={tool_names}",
        )
        self.assertIn(
            "proxy_ticket_get_task",
            tool_names,
            f"C1 REGRESSION: apply_mode(ticket) did NOT inject "
            f"proxy_ticket_get_task; tools={tool_names}",
        )
        self.assertNotIn(
            "proxy_work",
            tool_names,
            f"C1 REGRESSION: apply_mode(ticket) MUST NOT inject proxy_work; "
            f"tools={tool_names}",
        )

    @patch("proxy.context.select_context", return_value=None)
    @patch("proxy.tool_cache.cache_tool_schema")
    @patch("proxy.tool_cache.get_cached_tool_schema", return_value=None)
    def test_apply_mode_default_mode_no_proxy_injection(
        self, mock_get_cached, mock_cache, mock_select
    ):
        """C1: apply_mode() for a DEFAULT-mode agent (no skill active) MUST
        inject neither proxy_work nor proxy_ticket. DEFAULT mode = no skill = no proxy_*.
        """
        from tests.mode_harness import create_test_agent

        agent = create_test_agent(
            session_id="sess_c1_default",
            agent_id="agent_c1_default",
        )
        # No mode activation — DEFAULT.
        body = _make_proxy_body()
        ctx = _make_proxy_ctx(session_id="sess_c1_default", body=body)

        from proxy.mode import apply_mode, Mode
        apply_mode(ctx, agent, ctx.body, Mode.DEFAULT)

        tool_names = [
            t.get("function", {}).get("name", "")
            for t in ctx.body.get("tools", [])
        ]
        self.assertNotIn(
            "proxy_work",
            tool_names,
            f"C1 REGRESSION: apply_mode(DEFAULT) MUST NOT inject proxy_work; "
            f"tools={tool_names}",
        )
        # Slice 8b: legacy monolithic proxy_ticket AND decomposed tools are
        # all skill-mode-only — DEFAULT mode injects none of them.
        self.assertNotIn(
            "proxy_ticket",
            tool_names,
            f"C1 REGRESSION: apply_mode(DEFAULT) MUST NOT inject proxy_ticket; "
            f"tools={tool_names}",
        )
        for tool_name in ("proxy_ticket_list_tasks", "proxy_ticket_get_task"):
            self.assertNotIn(
                tool_name,
                tool_names,
                f"C1 REGRESSION: apply_mode(DEFAULT) MUST NOT inject {tool_name}; "
                f"tools={tool_names}",
            )

    def test_cache_bleed_real_hash_real_cache(self):
        """C4.5: prime cache with work-mode call (real hash), then call
        ticket-mode with the SAME incoming body. The hashes MUST differ
        (skill_name in hash) and the ticket result must NOT contain proxy_work.

        Replaces the mocked cache-bleed tests with REAL cache verification.
        """
        from proxy.tool_cache import clear_tool_cache, prepare_tooling_payload
        clear_tool_cache()

        incoming = [
            {"type": "function", "function": {
                "name": "read", "description": "r",
                "parameters": {"type": "object", "properties": {}}
            }},
        ]

        # Work mode: real hash, real cache write
        result_work = prepare_tooling_payload(
            {"tools": list(incoming)}, None, skill_name="work",
            session_id="sess-c45-cache-work", agent_id="agent-c45-cache-work",
        )
        work_names = [t["function"]["name"] for t in result_work.tools]
        assert "proxy_work" in work_names, f"work mode must inject proxy_work; got {work_names}"
        assert result_work.tool_hash is not None, "real hash should be generated"

        # Ticket mode: same body, different skill → different hash → cache miss
        result_ticket = prepare_tooling_payload(
            {"tools": list(incoming)}, None, skill_name="ticket",
            session_id="sess-c45-cache-ticket", agent_id="agent-c45-cache-ticket",
        )
        ticket_names = [t["function"]["name"] for t in result_ticket.tools]
        assert "proxy_work" not in ticket_names, f"ticket mode must NOT inject proxy_work; got {ticket_names}"
        # Slice 8b: decomposed proxy_ticket_* tools (ticket MAIN sees 2 read-only).
        assert "proxy_ticket_list_tasks" in ticket_names, (
            f"ticket mode must inject proxy_ticket_list_tasks; got {ticket_names}"
        )
        assert "proxy_ticket_get_task" in ticket_names, (
            f"ticket mode must inject proxy_ticket_get_task; got {ticket_names}"
        )

        # The two hashes must differ (proves the skill_name suffix in the hash)
        assert result_work.tool_hash != result_ticket.tool_hash, (
            f"cache key collision: work hash {result_work.tool_hash} == ticket hash {result_ticket.tool_hash}"
        )

    def test_cache_bleed_work_vs_ticket_same_body(self):
        """C1 cache-key isolation: priming with skill_name='work' must NOT
        cause a later skill_name='ticket' call to return proxy_work.
        The cache key in `_generate_tool_hash_with_mode` includes
        `skill_name`, so the two calls must produce distinct cache entries.
        """
        from skills.ticket_skill import TicketSkill
        from skills.work_skill import WorkSkill
        from proxy.tool_cache import clear_tool_cache, prepare_tooling_payload

        clear_tool_cache()

        work_skill = WorkSkill()
        ticket_skill = TicketSkill()

        mock_registry = MagicMock()
        mock_registry.get.side_effect = (
            lambda name: work_skill if name == "work" else ticket_skill
        )

        body = _make_proxy_body()

        with patch("proxy.skills.skill_registry.SkillRegistry") as mock_sr_cls, \
             patch("proxy.context.select_context", return_value=None), \
             patch("proxy.tool_cache.get_cached_tool_schema") as mock_get_cached, \
             patch("proxy.tool_cache.cache_tool_schema"):
            mock_sr_cls.return_value = mock_registry
            # Always cache miss for the test.
            mock_get_cached.return_value = None

            with patch("proxy.tool_cache._generate_tool_hash_with_mode") as mock_hash, \
                 patch("proxy.tool_pipeline.tool_registry.registry") as mock_reg:
                # Provide distinct hash per call to exercise the skill_name append.
                mock_hash.side_effect = [
                    "hash_c1_proxy_work_prime",
                    "hash_c1_proxy_ticket_after",
                ]
                mock_reg._tools = {
                    "proxy_ticket_list_tasks": MagicMock(
                        available_in_skills=frozenset({"ticket", "work"})
                    ),
                    "proxy_ticket_get_task": MagicMock(
                        available_in_skills=frozenset({"ticket", "work"})
                    ),
                    "proxy_work": MagicMock(
                        available_in_skills=frozenset({"work"})
                    ),
                }

                work_payload = prepare_tooling_payload(
                    {"tools": list(body["tools"])},
                    None,
                    skill_name="work",
                    session_id="sess_c1_proxy_bleed_work",
                    agent_id="agent_c1_proxy_bleed_work",
                )
                work_names = [
                    t.get("function", {}).get("name", "")
                    for t in work_payload.tools
                ]
                self.assertIn(
                    "proxy_work",
                    work_names,
                    f"precondition: work prime must inject proxy_work; "
                    f"got {work_names}",
                )

                ticket_payload = prepare_tooling_payload(
                    {"tools": list(body["tools"])},
                    None,
                    skill_name="ticket",
                    session_id="sess_c1_proxy_bleed_ticket",
                    agent_id="agent_c1_proxy_bleed_ticket",
                )
                ticket_names = [
                    t.get("function", {}).get("name", "")
                    for t in ticket_payload.tools
                ]
                # Slice 8b: decomposed proxy_ticket_* tools (ticket MAIN sees 2 read-only).
                self.assertIn(
                    "proxy_ticket_list_tasks",
                    ticket_names,
                    f"ticket must inject proxy_ticket_list_tasks; got {ticket_names}",
                )
                self.assertIn(
                    "proxy_ticket_get_task",
                    ticket_names,
                    f"ticket must inject proxy_ticket_get_task; got {ticket_names}",
                )
                self.assertNotIn(
                    "proxy_work",
                    ticket_names,
                    f"CACHE BLEED: ticket call returned proxy_work from "
                    f"cache; cache key is not isolating per-skill_name. "
                    f"got {ticket_names}",
                )


if __name__ == "__main__":
    unittest.main()
