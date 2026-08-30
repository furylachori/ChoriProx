"""Tests for the subagent_mode override on the `task` tool.

US-017c extension: a parent agent in TICKET mode can request a WORK sub-agent
by passing `subagent_mode="work"` in a task tool call. The proxy escalates
the canary for that single task call from ticket to work, so the spawned
sub-agent inherits WORK mode (full coding tools) instead of the parent's
TICKET mode (read-only ticket tools).

Real `WorkSkill` and `TicketSkill` are used throughout — mocking them masked
prior bugs. The HMAC verifier is patched to accept any well-formed canary,
matching the pattern in test_canary.py and test_subagent_e2e.py.
"""
import json
import time
import unittest
from unittest.mock import MagicMock, patch

from proxy.canary import (
    ParsedCanary,
    detect_subagent_ingress,
    parse_canary,
    _inject_canary_into_task_calls,
)
from proxy.skills.skill_registry import SkillContext
from proxy.tool_parsing.normalize import (
    normalize_tool_arguments,
    provide_defaults_for_optional_params,
)


# R1-A test-fix (Edit A1 propagation + Edit A2 None-ownership guard):
# PostProcessStage.execute() reads `ctx.mode_snapshot_pre_upstream` via the
# class-level guard `hasattr(type(ctx), "mode_snapshot_pre_upstream")`. A
# plain MagicMock() does NOT declare this attribute on its class, so the
# guard returns False and `expected_mode` falls through to None — bypassing
# the lifecycle clear and silently dropping the test's per-turn contract.
#
# Declare the field on the MagicMock class so the production code's
# class-level guard accepts MagicMock ctx instances and the test can set
# the instance attribute to a real ModeState for the CAS.
MagicMock.mode_snapshot_pre_upstream = None


_FRESH_TS = str(int(time.time()))
_FAKE_SIG = "0" * 64  # 64 hex chars; verify_canary only checks length


def _make_pc(skill_name: str) -> ParsedCanary:
    return ParsedCanary(
        skill=skill_name,
        timestamp=int(_FRESH_TS),
        sig=_FAKE_SIG,
        is_dev_fallback=True,
    )


def _verify_accept_any(token: str, skill=None, **kwargs) -> ParsedCanary:
    """Stand-in for verify_canary that accepts any well-formed token.

    Accepts B-3/B-4 binding kwargs (state_str, session_id) without inspection
    — tests in this module are about MODE / TOOL FILTER behaviour, not HMAC
    binding. The binding semantics are covered by
    ``tests/proxy/test_canary.py::TestCanaryStateAndSessionBinding``.
    """
    if not token or ":" not in token:
        return None
    name = token.split(":", 1)[0]
    return _make_pc(name)


def _make_request_ctx(session_id="sess_subagent_mode_test") -> object:
    """Build a minimal RequestContext-like namespace."""
    ctx = MagicMock()
    ctx.session_id = session_id
    ctx.agent_id = "agent_subagent_mode_test"
    ctx.is_subagent = False
    ctx.request_id = "req_subagent_mode_test"
    ctx.client_hint = "test"
    ctx.messages = []
    return ctx


def _make_response(task_args: dict) -> dict:
    """Build a single-task-call response data dict."""
    return {
        "choices": [{
            "message": {
                "role": "assistant",
                "tool_calls": [{
                    "id": "call_test",
                    "type": "function",
                    "function": {
                        "name": "task",
                        "arguments": json.dumps(task_args),
                    },
                }],
            },
        }],
    }


def _make_post_process_ctx(session_id="sess_post_process", agent_id="agent_post_process"):
    """Build a RequestContext-like namespace suitable for PostProcessStage.

    Wave 4 migration: replaces direct apply_post_process(ctx, agent, data, mode)
    calls with PostProcessStage().execute(ctx). The ctx carries response_data,
    model_route, body, and all the attributes PostProcessStage touches.
    """
    body = {"messages": []}
    ctx = MagicMock()
    ctx.session_id = session_id
    ctx.agent_id = agent_id
    ctx.is_subagent = False
    ctx.subagent_ticket_mode = False
    ctx.seeded_tasks = []
    ctx.inherited_skill = None
    ctx._mode = None
    ctx.client_hint = "test"
    ctx.messages = []
    ctx.body = body
    ctx.request_id = "req_post_process_migration"
    ctx.model_route = MagicMock()
    ctx.model_route.is_remote = False
    ctx.model_route.model_cfg = None
    ctx.upstream = None
    ctx.swarm_policy = None
    ctx.llama_body = body
    return ctx


def _noop_postprocess_submethods():
    """Patch the 8 PostProcessStage sub-methods (extract / scan / sync / persist /
    populate / capture / intercept / assign) to no-ops. The three post-process
    ops under test (_apply_skill_response_transform, _inject_task_canaries,
    _finalize_skill_lifecycle_with_flag) still run.
    """
    from contextlib import ExitStack
    from proxy.pipeline.post_process import PostProcessStage as _Stage
    stack = ExitStack()
    noop = lambda self, ctx: None
    stack.enter_context(patch.object(_Stage, "_extract_minimax_reasoning", noop))
    stack.enter_context(patch.object(_Stage, "_scan_write_results", noop))
    stack.enter_context(patch.object(_Stage, "_sync_chat_history", noop))
    stack.enter_context(patch.object(_Stage, "_persist_messages", noop))
    stack.enter_context(patch.object(_Stage, "_populate_vfs_cache", noop))
    stack.enter_context(patch.object(_Stage, "_capture_cache_metrics", noop))
    stack.enter_context(patch.object(_Stage, "_handle_tool_interception", noop))
    stack.enter_context(patch.object(_Stage, "_assign_model_name", noop))
    return stack


def _patch_get_or_load(agent):
    """Return a patcher that makes agent_registry.get_or_load return the test's agent."""
    def _get_or_load(session_id):
        return agent
    return patch(
        "proxy.agent.agent_registry.registry.get_or_load",
        side_effect=_get_or_load,
    )


# ---------------------------------------------------------------------------
# Tool schema normalization
# ---------------------------------------------------------------------------


class TestSubagentModeParamNormalization(unittest.TestCase):
    """The `subagent_mode` arg must round-trip through normalization."""

    def test_subagent_mode_passes_through_unchanged(self):
        """normalize_tool_arguments keeps `subagent_mode` as-is (it's already canonical)."""
        args = {"prompt": "add a line", "subagent_mode": "work"}
        normalized = normalize_tool_arguments("task", args)
        self.assertEqual(normalized.get("subagent_mode"), "work")
        self.assertEqual(normalized.get("prompt"), "add a line")

    def test_provide_defaults_does_not_inject_subagent_mode(self):
        """provide_defaults_for_optional_params must not fill subagent_mode —
        leaving it absent means 'inherit parent mode' (per the spec).
        """
        args = {"prompt": "add a line"}
        defaults = provide_defaults_for_optional_params("task", args)
        self.assertNotIn(
            "subagent_mode", defaults,
            "subagent_mode must NOT be defaulted — absence means inherit parent mode",
        )


# ---------------------------------------------------------------------------
# _inject_canary_into_task_calls — subagent_mode dispatch
# ---------------------------------------------------------------------------


class TestInjectCanarySubagentMode(unittest.TestCase):
    """_inject_canary_into_task_calls honors per-task subagent_mode."""

    def setUp(self):
        # The parse_canary path calls verify_canary; we accept anything
        # well-formed so the work canary we generate is parseable on the way
        # back in. parse_canary ALSO looks up the skill in the registry,
        # so we patch that too — but the real WorkSkill / TicketSkill are
        # still used to PRODUCE the canary under test.
        self._verify_patcher = patch(
            "proxy.canary.verify_canary", side_effect=_verify_accept_any,
        )
        self._verify_patcher.start()
        self._registry_patcher = patch(
            "proxy.skills.skill_registry.SkillRegistry.get",
            side_effect=lambda name: MagicMock(name=name),
        )
        self._registry_patcher.start()
        # Isolate the escalation rate limiter per test. Several tests in
        # this class escalate ticket->work on a SHARED session_id
        # (`sess_subagent_mode_test`, from _make_request_ctx). The real
        # process-global `proxy.state.escalation_rate_limiter`
        # (max_requests=5, window_seconds=3600) accumulates tokens across
        # tests, so under xdist `--dist=loadfile` a later escalation can be
        # rate-limited (kind="rate_limited") and the test then sees a
        # ticket canary instead of a work canary — an intermittent flake.
        # Swapping in a fresh limiter per test guarantees an empty bucket
        # regardless of worker grouping or run order. This mirrors the
        # isolation used by TestEscalationRateLimit.
        from proxy.rate_limiter import RateLimiter
        self._limiter_patcher = patch(
            "proxy.state.escalation_rate_limiter",
            RateLimiter(max_requests=5, window_seconds=3600, persist=False),
        )
        self._limiter_patcher.start()

    def tearDown(self):
        self._limiter_patcher.stop()
        self._verify_patcher.stop()
        self._registry_patcher.stop()

    def _default_canary(self) -> str:
        """A real ticket canary produced by the real TicketSkill."""
        from skills.ticket_skill import TicketSkill
        ctx = SkillContext(
            session_id="sess_default_canary",
            agent_id="agent_default_canary",
            messages=[],
            skill_state={},
        )
        return TicketSkill().get_canary(ctx)

    def _prompt(self, data: dict) -> str:
        tc = data["choices"][0]["message"]["tool_calls"][0]
        return json.loads(tc["function"]["arguments"])["prompt"]

    def test_subagent_mode_work_in_ticket_mode_produces_work_canary(self):
        """Parent in TICKET mode + task(..., subagent_mode="work") → task prompt
        starts with a WORK canary, NOT the parent's ticket canary."""
        default_canary = self._default_canary()
        data = _make_response({
            "prompt": "add a line to README.md",
            "subagent_mode": "work",
        })

        result = _inject_canary_into_task_calls(
            data,
            default_canary,
            request_ctx=_make_request_ctx(),
            default_mode_name="ticket",
        )

        prompt = self._prompt(result)
        self.assertTrue(
            prompt.startswith("[[__SUBAGENT__:SKILL:work:"),
            f"expected WORK canary prefix, got: {prompt[:80]!r}",
        )
        self.assertNotIn(
            "[[__SUBAGENT__:SKILL:ticket:", prompt,
            "ticket canary must NOT appear — subagent_mode=work escalated",
        )

    def test_no_subagent_mode_uses_default_canary(self):
        """task(prompt=...) without subagent_mode → parent's default canary."""
        default_canary = self._default_canary()
        data = _make_response({"prompt": "list my tasks"})

        result = _inject_canary_into_task_calls(
            data,
            default_canary,
            request_ctx=_make_request_ctx(),
            default_mode_name="ticket",
        )

        prompt = self._prompt(result)
        self.assertTrue(
            prompt.startswith(default_canary),
            f"expected default canary prefix, got: {prompt[:80]!r}",
        )

    def test_subagent_mode_ticket_uses_default_canary(self):
        """Explicit subagent_mode="ticket" is a no-op — default canary used."""
        default_canary = self._default_canary()
        data = _make_response({
            "prompt": "list my tasks",
            "subagent_mode": "ticket",
        })

        result = _inject_canary_into_task_calls(
            data,
            default_canary,
            request_ctx=_make_request_ctx(),
            default_mode_name="ticket",
        )

        prompt = self._prompt(result)
        self.assertTrue(prompt.startswith(default_canary))

    def test_subagent_mode_invalid_logs_warning_and_uses_default(self):
        """Invalid subagent_mode value → WARNING logged, default canary used."""
        default_canary = self._default_canary()
        data = _make_response({
            "prompt": "list my tasks",
            "subagent_mode": "bogus-mode",
        })

        with self.assertLogs("proxy.canary", level="WARNING") as cm:
            result = _inject_canary_into_task_calls(
                data,
                default_canary,
                request_ctx=_make_request_ctx(),
                default_mode_name="ticket",
            )

        prompt = self._prompt(result)
        self.assertTrue(
            prompt.startswith(default_canary),
            "invalid subagent_mode must fall back to default canary",
        )
        warnings = [r.getMessage() for r in cm.records]
        self.assertTrue(
            any("subagent_mode" in m for m in warnings),
            f"expected a subagent_mode WARNING, got: {warnings}",
        )

    def test_subagent_mode_work_in_work_mode_is_no_op(self):
        """Parent already in WORK mode + subagent_mode="work" → work canary
        used (no change — both default and override are work). The escalation
        helper must NOT be called because no escalation is needed."""
        from skills.work_skill import WorkSkill
        default_ctx = SkillContext(
            session_id="sess_work_parent",
            agent_id="agent_work_parent",
            messages=[],
            skill_state={},
        )
        default_canary = WorkSkill().get_canary(default_ctx)

        data = _make_response({
            "prompt": "add a line to README.md",
            "subagent_mode": "work",
        })

        with patch("proxy.canary._build_work_canary_for_request") as mock_build:
            result = _inject_canary_into_task_calls(
                data,
                default_canary,
                request_ctx=_make_request_ctx(),
                default_mode_name="work",
            )

        mock_build.assert_not_called()

        prompt = self._prompt(result)
        self.assertTrue(
            prompt.startswith("[[__SUBAGENT__:SKILL:work:"),
            f"expected WORK canary, got: {prompt[:80]!r}",
        )

    def test_subagent_mode_work_in_ticket_mode_works_real_skill(self):
        """The fix: ticket parent + subagent_mode="work" → real WorkSkill.get_canary
        output is used. Confirms the REAL skill class is exercised (not mocked)."""
        default_canary = self._default_canary()
        from skills.work_skill import WorkSkill
        data = _make_response({
            "prompt": "add a line to README.md",
            "subagent_mode": "work",
        })

        result = _inject_canary_into_task_calls(
            data,
            default_canary,
            request_ctx=_make_request_ctx(),
            default_mode_name="ticket",
        )

        prompt = self._prompt(result)
        # Parse the canary and confirm it's a real work canary (signed by
        # sign_canary("work")), not a string-stitched facsimile.
        _p = parse_canary(prompt)
        skill_name = _p.skill_name if (_p and _p.verified) else None
        self.assertEqual(
            skill_name, "work",
            "the injected canary must parse as skill_name='work' (real WorkSkill output)",
        )
        # Sanity: WorkSkill().get_canary always returns a non-empty string
        # starting with the SKILL:work prefix. The parse above already
        # verified the canary is real and signed — no need to compare the
        # full canary string (which has a timestamp that drifts by 1s across
        # the second boundary between this call and the proxy call above,
        # making a full-string assertIn a ~10% flake). Compare the
        # SHA-256 sig only — stable across calls (HMAC of "work" with the
        # same dev-fallback secret). Canary format:
        #   [[__SUBAGENT__:SKILL:work:<64hex-sig>:<ts>]]
        #   split(":", 4) -> 5 parts: [marker, "SKILL", "work", sig, ts]]
        work_canary = WorkSkill().get_canary(
            SkillContext(session_id="x", agent_id="y", messages=[], skill_state={})
        )
        self.assertTrue(work_canary.startswith("[[__SUBAGENT__:SKILL:work:"))
        _parts = work_canary.split(":", 4)
        self.assertEqual(
            len(_parts), 5,
            f"work canary must have 5 colon-separated parts (no state); got {work_canary!r}",
        )
        _live_sig = _parts[3]
        self.assertTrue(
            len(_live_sig) == 64 and all(c in "0123456789abcdef" for c in _live_sig),
            f"live canary sig must be 64-hex; got {_live_sig!r}",
        )
        self.assertIn(
            _live_sig, prompt,
            f"prompt must contain the work canary sig the live skill produced. "
            f"live_sig={_live_sig!r} prompt[:120]={prompt[:120]!r}",
        )

    def test_subagent_mode_work_in_ticket_mode_logs_warning_with_request_id(self):
        """Escalation to work canary logs a WARNING with the request_id (audit trail).

        The request_id is propagated via the logging ``extra=`` kwarg so
        structured loggers (json_formatter) can correlate entries — the
        message string itself contains kind/subagent_mode but NOT the
        request_id (per AGENTS.md logging conventions)."""
        default_canary = self._default_canary()
        data = _make_response({
            "prompt": "add a line",
            "subagent_mode": "work",
        })
        ctx = _make_request_ctx()
        ctx.request_id = "req_audit_trace_abc"

        with self.assertLogs("proxy.canary", level="WARNING") as cm:
            _inject_canary_into_task_calls(
                data, default_canary,
                request_ctx=ctx, default_mode_name="ticket",
            )

        escalation = [
            r for r in cm.records
            if "canary_subagent_mode_escalation" in r.getMessage()
        ]
        self.assertEqual(len(escalation), 1, f"expected 1 escalation log, got {len(escalation)}: {[r.getMessage() for r in cm.records]}")
        record = escalation[0]
        msg = record.getMessage()
        self.assertIn("work", msg, "kind=work must be in the WARNING log")
        # request_id is on the record's __dict__ (set by extra= kwarg),
        # NOT in the formatted message.
        rid = record.__dict__.get("request_id")
        self.assertEqual(
            rid, "req_audit_trace_abc",
            f"request_id must propagate via extra= (got {rid!r})",
        )

    def test_escalation_failure_falls_back_to_default(self):
        """If WorkSkill.get_canary returns None/empty for any reason, the default
        canary is used (no crash)."""
        default_canary = self._default_canary()
        data = _make_response({
            "prompt": "add a line",
            "subagent_mode": "work",
        })

        with patch("skills.work_skill.WorkSkill.get_canary", return_value=None):
            result = _inject_canary_into_task_calls(
                data, default_canary,
                request_ctx=_make_request_ctx(),
                default_mode_name="ticket",
            )

        prompt = self._prompt(result)
        self.assertTrue(
            prompt.startswith(default_canary),
            "must fall back to default canary when WorkSkill.get_canary fails",
        )

    def test_request_ctx_none_works(self):
        """Function must not crash when request_ctx is None (no SkillContext info)."""
        default_canary = self._default_canary()
        data = _make_response({
            "prompt": "add a line",
            "subagent_mode": "work",
        })

        result = _inject_canary_into_task_calls(
            data, default_canary,
            request_ctx=None,
            default_mode_name="ticket",
        )

        prompt = self._prompt(result)
        self.assertTrue(prompt.startswith("[[__SUBAGENT__:SKILL:work:"))

    def test_subagent_mode_ticket_is_noop_for_work_parent(self):
        """Parent in WORK mode + subagent_mode="ticket" → no escalation.

        The orchestrator's `subagent_mode="ticket"` from a work-mode parent is a
        no-op — the work canary is used. There is no "downgrade" path. The LLM
        cannot force a work→ticket downgrade via this parameter. If a work-mode
        parent wants a read-only sub-agent, it should NOT use subagent_mode.
        """
        from skills.work_skill import WorkSkill
        default_ctx = SkillContext(
            session_id="sess_work_noop",
            agent_id="agent_work_noop",
            messages=[],
            skill_state={},
        )
        default_canary = WorkSkill().get_canary(default_ctx)

        data = _make_response({
            "prompt": "list my tasks",
            "subagent_mode": "ticket",
        })

        result = _inject_canary_into_task_calls(
            data,
            default_canary,
            request_ctx=_make_request_ctx(),
            default_mode_name="work",
        )

        prompt = self._prompt(result)
        self.assertTrue(
            prompt.startswith(default_canary),
            f"expected default work canary (ticket is a no-op), got: {prompt[:80]!r}",
        )

    def test_two_task_calls_in_one_response_use_correct_canaries(self):
        """A response with 2 task tool calls (one with subagent_mode='work',
        one without) → each task prompt gets the correct canary prefix.

        Ticket parent + 1 task(..., subagent_mode="work") + 1 task(...) →
        1 work canary + 1 default ticket canary, in prompt order."""
        default_canary = self._default_canary()
        data = {
            "choices": [{
                "message": {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "id": "call_a",
                            "type": "function",
                            "function": {
                                "name": "task",
                                "arguments": json.dumps({
                                    "prompt": "edit README.md",
                                    "subagent_mode": "work",
                                }),
                            },
                        },
                        {
                            "id": "call_b",
                            "type": "function",
                            "function": {
                                "name": "task",
                                "arguments": json.dumps({
                                    "prompt": "list my tasks",
                                }),
                            },
                        },
                    ],
                },
            }],
        }

        result = _inject_canary_into_task_calls(
            data,
            default_canary,
            request_ctx=_make_request_ctx(),
            default_mode_name="ticket",
        )

        tc_list = result["choices"][0]["message"]["tool_calls"]
        self.assertEqual(len(tc_list), 2, "both task calls must be preserved")
        prompt_a = json.loads(tc_list[0]["function"]["arguments"])["prompt"]
        prompt_b = json.loads(tc_list[1]["function"]["arguments"])["prompt"]

        # First call: work escalation.
        self.assertTrue(
            prompt_a.startswith("[[__SUBAGENT__:SKILL:work:"),
            f"first task call must have work canary, got: {prompt_a[:80]!r}",
        )
        self.assertIn(
            "edit README.md", prompt_a,
            "first task prompt content must be preserved after canary prefix",
        )

        # Second call: default ticket canary (no override → inherit).
        self.assertTrue(
            prompt_b.startswith(default_canary),
            f"second task call must have default ticket canary, got: {prompt_b[:80]!r}",
        )
        self.assertNotIn(
            "[[__SUBAGENT__:SKILL:work:", prompt_b,
            "second call must NOT be escalated to work",
        )
        self.assertIn(
            "list my tasks", prompt_b,
            "second task prompt content must be preserved after canary prefix",
        )


# ---------------------------------------------------------------------------
# WorkSkill.get_system_prompt — empty task_hash
# ---------------------------------------------------------------------------


class TestWorkSkillEmptyTaskHash(unittest.TestCase):
    """Work sub-agents without a claimed task must not see literal placeholders."""

    def _ctx(self, skill_state=None):
        ctx = MagicMock()
        ctx.session_id = "sess_empty_hash"
        ctx.skill_state = skill_state or {}
        ctx.client_hint = "test"
        ctx.work_task_hash = ""
        return ctx

    def test_no_task_hash_renders_without_literal_short_hash(self):
        """Empty task_hash → output must NOT contain literal '<short_hash>'."""
        from skills.work_skill import WorkSkill
        prompt = WorkSkill().get_system_prompt(self._ctx(skill_state={}))
        self.assertNotIn(
            "<short_hash>", prompt,
            "literal '<short_hash>' must not leak when task_hash is empty",
        )
        self.assertNotIn("<task_name>", prompt)
        self.assertNotIn("<priority>", prompt)
        self.assertNotIn("<summary>", prompt)

    def test_no_task_hash_substitutes_safe_defaults(self):
        """Substituted values are human-readable defaults, not placeholders."""
        from skills.work_skill import WorkSkill
        prompt = WorkSkill().get_system_prompt(self._ctx(skill_state={}))
        self.assertIn("(ad-hoc edit session)", prompt)
        self.assertIn("(no specific task)", prompt)
        self.assertIn("MEDIUM", prompt)
        self.assertIn("(no summary)", prompt)


# ---------------------------------------------------------------------------
# End-to-end: ticket parent + subagent_mode=work → work sub-agent ingress
# ---------------------------------------------------------------------------


class TestSubagentModeEndToEnd(unittest.TestCase):
    """Full chain: ticket parent's task call with subagent_mode=work → sub-agent
    request arrives with `inherited_skill="work"`."""

    def setUp(self):
        self._verify_patcher = patch(
            "proxy.canary.verify_canary", side_effect=_verify_accept_any,
        )
        self._verify_patcher.start()
        self._registry_patcher = patch(
            "proxy.skills.skill_registry.SkillRegistry.get",
            side_effect=lambda name: MagicMock(name=name),
        )
        self._registry_patcher.start()

    def tearDown(self):
        self._verify_patcher.stop()
        self._registry_patcher.stop()

    def test_ticket_parent_with_work_subagent_mode_yields_work_inherited_skill(self):
        """End-to-end: a parent in TICKET mode with task(..., subagent_mode='work')
        produces a task prompt whose canary parses as inherited_skill='work'."""
        from skills.ticket_skill import TicketSkill
        ticket_ctx = SkillContext(
            session_id="sess_e2e_parent",
            agent_id="agent_e2e_parent",
            messages=[],
            skill_state={},
        )
        ticket_canary = TicketSkill().get_canary(ticket_ctx)

        data = _make_response({
            "prompt": "add a line to README.md",
            "subagent_mode": "work",
        })

        result = _inject_canary_into_task_calls(
            data, ticket_canary,
            request_ctx=_make_request_ctx("sess_e2e_parent"),
            default_mode_name="ticket",
        )

        task_prompt = json.loads(
            result["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"]
        )["prompt"]

        # Simulate the sub-agent receiving this prompt as its first user message.
        sub_body = {"messages": [{"role": "user", "content": task_prompt}]}
        ingress = detect_subagent_ingress(sub_body)

        self.assertTrue(ingress.is_subagent)
        self.assertEqual(
            ingress.inherited_skill, "work",
            "sub-agent must inherit work skill — was ticket parent using "
            "subagent_mode='work' on the task call",
        )
        # And NOT ticket — the escalation must replace the parent's skill.
        self.assertFalse(ingress.subagent_ticket_mode)


# ---------------------------------------------------------------------------
# TICKET_MAIN_PROMPT — new subagent_mode section
# ---------------------------------------------------------------------------


class TestTicketMainPreambleSubagentMode(unittest.TestCase):
    """The MAIN ticket orchestrator prompt must document the new subagent_mode
    parameter so the LLM knows it can request a work sub-agent for edits."""

    def test_preamble_mentions_subagent_mode(self):
        from proxy.tool_parsing.ticket_prompts import TICKET_MAIN_PROMPT
        self.assertIn("subagent_mode", TICKET_MAIN_PROMPT)
        self.assertIn('"work"', TICKET_MAIN_PROMPT)
        self.assertIn("Sub-agent modes for edits", TICKET_MAIN_PROMPT)


# ---------------------------------------------------------------------------
# F-T1: Tool filter content assertion — the actual 3-times-today regression
# ---------------------------------------------------------------------------


class TestEscalatedWorkSubagentToolFilter(unittest.TestCase):
    """The escalated work sub-agent must receive write/edit/bash tools.

    This is the actual bug class behind the 3-times-today failure: the canary
    was injected but the sub-agent still inherited ticket's read-only tool
    filter. Asserting the canary text alone is insufficient — we must also
    assert that the skill lookup yields WorkSkill with the full tool set.
    """

    def setUp(self):
        self._verify_patcher = patch(
            "proxy.canary.verify_canary", side_effect=_verify_accept_any,
        )
        self._verify_patcher.start()
        self._registry_patcher = patch(
            "proxy.skills.skill_registry.SkillRegistry.get",
            side_effect=lambda name: MagicMock(name=name),
        )
        self._registry_patcher.start()

    def tearDown(self):
        self._verify_patcher.stop()
        self._registry_patcher.stop()

    def test_escalated_work_subagent_has_full_tool_filter(self):
        """Regression: ticket parent + subagent_mode=work → work sub-agent
        MUST receive write/edit/bash tools, not ticket's read-only set.
        This is the actual 3-times-today bug assertion."""
        from skills.ticket_skill import TicketSkill
        from skills.work_skill import WorkSkill

        ticket_ctx = SkillContext(
            session_id="sess_escalate_filter",
            agent_id="agent_escalate_filter",
            messages=[],
            skill_state={},
        )
        default_canary = TicketSkill().get_canary(ticket_ctx)

        data = _make_response({
            "prompt": "add a line to README.md",
            "subagent_mode": "work",
        })

        result = _inject_canary_into_task_calls(
            data, default_canary,
            request_ctx=_make_request_ctx("sess_escalate_filter"),
            default_mode_name="ticket",
        )

        task_prompt = json.loads(
            result["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"]
        )["prompt"]

        _p = parse_canary(task_prompt)
        skill_name = _p.skill_name if (_p and _p.verified) else None
        self.assertEqual(
            skill_name, "work",
            "injected canary must parse as skill_name='work'",
        )

        work_ctx = SkillContext(
            session_id="sess_escalate_filter",
            agent_id="agent_work_subagent",
            messages=[],
            skill_state={},
        )
        work_filter = WorkSkill().get_sub_agent_tool_filter(work_ctx)

        self.assertIn("write", work_filter)
        self.assertIn("edit", work_filter)
        self.assertIn("bash", work_filter)
        self.assertNotIn(
            "webfetch", work_filter,
            "work sub-agent filter must NOT include ticket-only webfetch",
        )

        ticket_ctx_for_filter = SkillContext(
            session_id="sess_escalate_filter",
            agent_id="agent_ticket_subagent",
            messages=[],
            skill_state={},
        )
        ticket_filter = TicketSkill().get_sub_agent_tool_filter(ticket_ctx_for_filter)
        self.assertIn(
            "webfetch", ticket_filter,
            "ticket sub-agent filter retains webfetch (sanity check that the "
            "assertion above is meaningful — we're not just claiming all "
            "filters are identical)",
        )
        self.assertNotIn(
            "write", ticket_filter,
            "ticket sub-agent filter must NOT include write (sanity check)",
        )


# ---------------------------------------------------------------------------
# F-T2: Workspace inheritance through escalated canary
# ---------------------------------------------------------------------------


class TestWorkSkillInheritanceWithoutTaskHash(unittest.TestCase):
    """WorkSkill.get_inheritance_state emits workspace_name even without task_hash.

    This is the work_skill.py change that made the escalation usable: a
    ticket parent with no claimed task can still escalate a work sub-agent
    that inherits the active workspace.
    """

    def test_work_skill_inheritance_state_emits_workspace_without_task_hash(self):
        """Empty task_hash but active workspace → state has workspace_name.

        This is the work_skill.py change that made the escalation usable.
        """
        from skills.work_skill import WorkSkill

        ctx = MagicMock()
        ctx.session_id = "sess_no_hash_with_workspace"
        ctx.client_hint = "test"
        ctx.skill_state = {}
        ctx.work_task_hash = ""

        with patch("proxy.workspace_state.workspace_store") as ws_mock:
            ws_mock.get_active.return_value = "choriprox"
            state = WorkSkill().get_inheritance_state(ctx)

        self.assertIsNotNone(state)
        self.assertEqual(state.get("workspace_name"), "choriprox")
        self.assertNotIn("task_hash", state)

    def test_work_skill_inheritance_state_logs_debug_on_workspace_failure(self):
        """workspace_store.get_active raising → logger.debug fires, state may still be set.

        Per F-Sec3: silent failure was a regression risk. Now a DEBUG log
        must be emitted when the workspace lookup fails.
        """
        from skills.work_skill import WorkSkill

        ctx = MagicMock()
        ctx.session_id = "sess_workspace_fail"
        ctx.client_hint = "test"
        ctx.skill_state = {}
        ctx.work_task_hash = ""

        with patch("proxy.workspace_state.workspace_store") as ws_mock:
            ws_mock.get_active.side_effect = RuntimeError("simulated db failure")
            with self.assertLogs("skills.work", level="DEBUG") as cm:
                state = WorkSkill().get_inheritance_state(ctx)

        debug_msgs = [r.getMessage() for r in cm.records]
        self.assertTrue(
            any("workspace lookup failed" in m for m in debug_msgs),
            f"expected DEBUG log for workspace failure, got: {debug_msgs}",
        )
        self.assertIsNone(
            state,
            "with no task_hash and workspace lookup failure, state should be None",
        )


# ---------------------------------------------------------------------------
# F-T3 / F-T4: _resolve_task_canary direct unit tests
# ---------------------------------------------------------------------------


class TestResolveTaskCanaryDirect(unittest.TestCase):
    """Direct unit tests for _resolve_task_canary (the pure helper).

    No response wrapping, no canary injection — call the helper directly to
    cover all the corner cases: missing/None arg, non-string types, ticket
    vs work parent, invalid values, build failure fallback.
    """

    def _resolve(self, args, default_mode_name="ticket", request_ctx=None):
        from proxy.canary import _resolve_task_canary
        return _resolve_task_canary(
            default_canary="DEFAULT_CANARY",
            args=args,
            default_mode_name=default_mode_name,
            request_ctx=request_ctx,
        )

    def test_resolve_task_canary_missing_arg(self):
        """No subagent_mode → default_canary, escalation=None."""
        canary, kind = self._resolve({"prompt": "hi"})
        self.assertEqual(canary, "DEFAULT_CANARY")
        self.assertIsNone(kind)

    def test_resolve_task_canary_none_arg(self):
        """subagent_mode=None → default_canary, escalation=None."""
        canary, kind = self._resolve({"subagent_mode": None})
        self.assertEqual(canary, "DEFAULT_CANARY")
        self.assertIsNone(kind)

    def test_resolve_task_canary_non_string_arg(self):
        """subagent_mode=[] / {} / 42 / True / "" → default_canary, escalation=None.

        F-T4: type guard rejects any non-string value (lists, dicts, ints,
        bools, empty strings). All fall back to the default canary.
        """
        for non_string in [[], ["work"], {"value": "work"}, 42, True, False, ""]:
            with self.subTest(non_string=non_string):
                canary, kind = self._resolve({"subagent_mode": non_string})
                self.assertEqual(
                    canary, "DEFAULT_CANARY",
                    f"non-string subagent_mode={non_string!r} must use default",
                )
                self.assertIsNone(
                    kind,
                    f"non-string subagent_mode={non_string!r} must not escalate",
                )

    def test_resolve_task_canary_ticket_in_ticket_parent(self):
        """subagent_mode=ticket, default_mode_name=ticket → default_canary, escalation=None."""
        canary, kind = self._resolve({"subagent_mode": "ticket"})
        self.assertEqual(canary, "DEFAULT_CANARY")
        self.assertIsNone(kind)

    def test_resolve_task_canary_work_in_ticket_parent(self):
        """subagent_mode=work, default_mode_name=ticket → work canary, escalation=work."""
        with patch(
            "proxy.canary._build_work_canary_for_request",
            return_value="WORK_CANARY",
        ):
            canary, kind = self._resolve({"subagent_mode": "work"})
        self.assertEqual(canary, "WORK_CANARY")
        self.assertEqual(kind, "work")

    def test_resolve_task_canary_work_in_work_parent(self):
        """subagent_mode=work, default_mode_name=work → default canary, escalation=None.

        The work parent already has the work canary — no escalation needed.
        _build_work_canary_for_request must NOT be called.
        """
        with patch("proxy.canary._build_work_canary_for_request") as mock_build:
            canary, kind = self._resolve(
                {"subagent_mode": "work"},
                default_mode_name="work",
            )
        self.assertEqual(canary, "DEFAULT_CANARY")
        self.assertIsNone(kind)
        mock_build.assert_not_called()

    def test_resolve_task_canary_invalid_value(self):
        """subagent_mode=garbage → default canary, escalation=invalid, WARNING logged."""
        with self.assertLogs("proxy.canary", level="WARNING") as cm:
            canary, kind = self._resolve({"subagent_mode": "garbage"})
        self.assertEqual(canary, "DEFAULT_CANARY")
        self.assertEqual(kind, "invalid")
        msgs = [r.getMessage() for r in cm.records]
        self.assertTrue(
            any("canary_subagent_mode_invalid" in m for m in msgs),
            f"expected invalid WARNING, got: {msgs}",
        )

    def test_resolve_task_canary_work_build_fails(self):
        """When _build_work_canary_for_request returns None, fall back to default."""
        with patch(
            "proxy.canary._build_work_canary_for_request",
            return_value=None,
        ):
            canary, kind = self._resolve({"subagent_mode": "work"})
        self.assertEqual(canary, "DEFAULT_CANARY")
        self.assertIsNone(
            kind,
            "when build returns None, must fall back silently (no escalation kind)",
        )

    def test_build_work_canary_for_request_swallows_work_skill_exceptions(self):
        """F-C6: WorkSkill.get_canary raising inside _build_work_canary_for_request
        must return None (caller falls back to default), not propagate."""
        from proxy.canary import _build_work_canary_for_request
        ctx = MagicMock()
        ctx.session_id = "sess_build_fails"
        ctx.client_hint = "test"
        ctx.skill_state = {}

        with patch(
            "skills.work_skill.WorkSkill.get_canary",
            side_effect=RuntimeError("simulated downstream failure"),
        ):
            result = _build_work_canary_for_request(ctx)
        self.assertIsNone(
            result,
            "_build_work_canary_for_request must swallow exceptions and return None",
        )


# ---------------------------------------------------------------------------
# F-T5: apply_post_process integration
# ---------------------------------------------------------------------------


class TestApplyPostProcessSubagentMode(unittest.TestCase):
    """End-to-end apply_post_process with subagent_mode escalation.

    Uses a real Agent in ticket mode, a real RequestContext, and a real
    `task` tool call with subagent_mode="work" in args. Verifies the
    final prompt carries a WORK canary.
    """

    def setUp(self):
        self._verify_patcher = patch(
            "proxy.canary.verify_canary", side_effect=_verify_accept_any,
        )
        self._verify_patcher.start()

    def tearDown(self):
        self._verify_patcher.stop()

    def test_ticket_parent_apply_post_process_work_subagent_mode_injects_work_canary(self):
        """Ticket parent + apply_post_process + task(..., subagent_mode='work') → work canary."""
        from proxy.mode import Mode
        from proxy.pipeline.post_process import PostProcessStage
        from proxy.agent.agent import Agent
        from proxy.skills.skill_definition import ModeState

        agent = Agent(
            agent_id="parent_t2_work",
            session_id="sess_t2_work",
        )
        mode_state = ModeState(
            name="ticket",
            activated_at=time.time(),
            ttl_seconds=300,
            skill_state={},
        )
        self.assertTrue(agent.set_active_mode(mode_state))

        ctx = _make_post_process_ctx(
            session_id=agent.session_id, agent_id=agent.agent_id,
        )
        ctx._mode = Mode.SKILL

        data = {
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [{
                        "id": "call_t2",
                        "type": "function",
                        "function": {
                            "name": "task",
                            "arguments": json.dumps({
                                "prompt": "edit README.md",
                                "subagent_mode": "work",
                            }),
                        },
                    }],
                },
                "finish_reason": "tool_calls",
            }]
        }
        ctx.response_data = data

        with _patch_get_or_load(agent), _noop_postprocess_submethods():
            result = PostProcessStage().execute(ctx)

        self.assertIsInstance(
            result, dict,
            "PostProcessStage.execute must return a real dict (not MagicMock)",
        )
        task_prompt = json.loads(
            result["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"]
        )["prompt"]
        _p = parse_canary(task_prompt)
        skill_name = _p.skill_name if (_p and _p.verified) else None
        self.assertEqual(
            skill_name, "work",
            "PostProcessStage.execute with subagent_mode=work must inject a work canary",
        )

    def test_escalated_work_subagent_e2e_has_full_tool_filter(self):
        """End-to-end: ticket parent + subagent_mode='work' via PostProcessStage.execute()
        must inject a work canary AND that canary must resolve to a sub-agent
        whose WorkSkill.sub_agent_tool_filter contains write/edit/bash.

        This is the actual 3-times-today regression assertion: the canary text
        alone is insufficient — the sub-agent must receive full coding tools.
        Composed of (a) PostProcessStage.execute() + (b) parse_canary + (c)
        WorkSkill.get_sub_agent_tool_filter on the inherited skill.
        """
        from proxy.mode import Mode
        from proxy.pipeline.post_process import PostProcessStage
        from proxy.agent.agent import Agent
        from proxy.skills.skill_definition import ModeState
        from skills.work_skill import WorkSkill
        from skills.ticket_skill import TicketSkill

        agent = Agent(
            agent_id="parent_t2_e2e_filter",
            session_id="sess_t2_e2e_filter",
        )
        mode_state = ModeState(
            name="ticket",
            activated_at=time.time(),
            ttl_seconds=300,
            skill_state={},
        )
        self.assertTrue(agent.set_active_mode(mode_state))

        ctx = _make_post_process_ctx(
            session_id=agent.session_id, agent_id=agent.agent_id,
        )
        ctx._mode = Mode.SKILL

        data = {
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [{
                        "id": "call_t2_e2e_filter",
                        "type": "function",
                        "function": {
                            "name": "task",
                            "arguments": json.dumps({
                                "prompt": "edit README.md",
                                "subagent_mode": "work",
                            }),
                        },
                    }],
                },
                "finish_reason": "tool_calls",
            }]
        }
        ctx.response_data = data

        with _patch_get_or_load(agent), _noop_postprocess_submethods():
            result = PostProcessStage().execute(ctx)

        self.assertIsInstance(
            result, dict,
            "PostProcessStage.execute must return a real dict (not MagicMock)",
        )
        task_prompt = json.loads(
            result["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"]
        )["prompt"]

        _p = parse_canary(task_prompt)
        skill_name = _p.skill_name if (_p and _p.verified) else None
        self.assertEqual(
            skill_name, "work",
            "PostProcessStage.execute with subagent_mode=work must inject a work canary; "
            f"got skill={skill_name!r} prompt_prefix={task_prompt[:80]!r}",
        )

        sub_ctx = SkillContext(
            session_id="sess_t2_e2e_filter_subagent",
            agent_id="agent_t2_e2e_filter_subagent",
            messages=[],
            skill_state={},
        )
        work_filter = WorkSkill().get_sub_agent_tool_filter(sub_ctx)
        self.assertIn("write", work_filter)
        self.assertIn("edit", work_filter)
        self.assertIn("bash", work_filter)
        self.assertNotIn(
            "webfetch", work_filter,
            "work sub-agent filter must NOT include ticket-only webfetch; "
            f"filter={sorted(work_filter)!r}",
        )

        ticket_filter = TicketSkill().get_sub_agent_tool_filter(sub_ctx)
        self.assertIn(
            "webfetch", ticket_filter,
            "ticket sub-agent filter retains webfetch (sanity check — proves "
            "the assertion above is meaningful, not a vacuous pass)",
        )
        self.assertNotIn(
            "write", ticket_filter,
            "ticket sub-agent filter must NOT include write (sanity check)",
        )

    def test_default_parent_apply_post_process_subagent_mode_yields_bare_or_workspace_canary(self):
        """Default parent (no skill) + PostProcessStage.execute() + task(..., subagent_mode='work')
        → bare / workspace canary, NOT an escalated work canary (no parent mode to escalate from).
        """
        from proxy.mode import Mode
        from proxy.pipeline.post_process import PostProcessStage
        from proxy.agent.agent import Agent

        agent = Agent(
            agent_id="parent_t2_default",
            session_id="sess_t2_default",
        )
        self.assertIsNone(
            agent.snapshot_mode(),
            "preconditions: parent has no active mode",
        )

        ctx = _make_post_process_ctx(
            session_id=agent.session_id, agent_id=agent.agent_id,
        )
        ctx._mode = Mode.DEFAULT

        data = {
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [{
                        "id": "call_t2_default",
                        "type": "function",
                        "function": {
                            "name": "task",
                            "arguments": json.dumps({
                                "prompt": "list my tasks",
                                "subagent_mode": "work",
                            }),
                        },
                    }],
                },
                "finish_reason": "tool_calls",
            }]
        }
        ctx.response_data = data

        with patch("proxy.workspace_state.workspace_store") as ws_mock, \
             _patch_get_or_load(agent), _noop_postprocess_submethods():
            ws_mock.get_active.return_value = "default"
            result = PostProcessStage().execute(ctx)

        self.assertIsInstance(
            result, dict,
            "PostProcessStage.execute must return a real dict (not MagicMock)",
        )
        task_prompt = json.loads(
            result["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"]
        )["prompt"]
        self.assertTrue(
            task_prompt.startswith("[[__SUBAGENT__]]\n"),
            f"expected bare canary for default workspace, got: {task_prompt[:80]!r}",
        )
        self.assertNotIn(
            "[[__SUBAGENT__:SKILL:work:", task_prompt,
            "default parent with subagent_mode=work must NOT escalate — "
            "there is no parent mode to escalate from; result was: "
            f"{task_prompt[:80]!r}",
        )


# ---------------------------------------------------------------------------
# B-1 + B-13: _build_work_canary_for_request parent ModeState.skill_state propagation
# ---------------------------------------------------------------------------


class TestEscalationParentSkillStatePropagation(unittest.TestCase):
    """Escalated work canary must inherit the parent's active ModeState.skill_state.

    Previous implementation read ``request_ctx.skill_state`` and ``ctx._mode.skill_state``
    — both dead code (RequestContext has no skill_state field; ctx._mode is a Mode enum,
    not a ModeState). These tests verify the new implementation correctly reads the
    parent's active ModeState via ``agent.snapshot_mode()`` and overlays its
    ``skill_state`` onto the SkillContext passed to ``WorkSkill.get_canary``.
    """

    def _make_real_agent_with_mode(self, session_id, arg_value="", extra=None):
        """Build a real Agent whose active ModeState has skill_state["arg"] = arg_value.

        work_skill.get_inheritance_state reads skill_state["arg"] (not "task_hash").
        Also overlays `extra` keys (e.g. custom metadata) into skill_state.
        """
        from proxy.agent.agent import Agent
        from proxy.skills.skill_definition import ModeState

        agent = Agent(
            agent_id="parent_esc_" + session_id[-8:],
            session_id=session_id,
        )
        ss = {}
        if arg_value:
            ss["arg"] = arg_value
        if extra:
            ss.update(extra)
        mode_state = ModeState(
            name="ticket",
            activated_at=time.time(),
            ttl_seconds=300,
            skill_state=ss,
        )
        self.assertTrue(agent.set_active_mode(mode_state))
        return agent

    def _ctx_with_real_agent(self, agent, messages=None, request_id="req_esc_test"):
        """Build a ctx MagicMock that explicitly exposes the real agent."""
        ctx = MagicMock()
        ctx.session_id = agent.session_id
        ctx.agent_id = agent.agent_id
        ctx.is_subagent = False
        ctx.request_id = request_id
        ctx.client_hint = "test"
        ctx.messages = messages if messages is not None else [{"role": "user", "content": "hi"}]
        ctx.body = {"messages": list(ctx.messages)}
        ctx.agent = agent
        return ctx

    def _extract_canary_state(self, canary_text):
        """Parse a canary string and return the (skill_name, state) tuple."""
        p = parse_canary(canary_text)
        if p is not None and p.kind == "skill" and p.verified:
            return p.skill_name, p.state
        return None, None

    def test_escalation_propagates_parent_skill_state_via_agent(self):
        """Parent's ModeState.skill_state must reach the canary's JSON state."""
        agent = self._make_real_agent_with_mode(
            session_id="sess_esc_inherit_01",
            arg_value="abc12345",
            extra={"foo": "bar"},
        )
        ctx = self._ctx_with_real_agent(agent, request_id="req_esc_inherit_01")

        with patch("proxy.canary.verify_canary", side_effect=_verify_accept_any):
            from proxy.canary import _build_work_canary_for_request
            canary = _build_work_canary_for_request(ctx)

        self.assertIsNotNone(canary, "must produce a canary string")
        skill_name, state = self._extract_canary_state(canary)
        self.assertEqual(skill_name, "work", "must produce a work canary")
        self.assertIsInstance(state, dict, "canary must carry JSON state")
        self.assertEqual(
            state.get("task_hash"), "abc12345",
            "parent ModeState.skill_state['arg'] must propagate into canary JSON as "
            "'task_hash' (work_skill.get_inheritance_state normalizes the key)",
        )

    def test_escalation_propagates_agent_mode_skill_state(self):
        """Agent ModeState.skill_state propagates via snapshot_mode()."""
        agent = self._make_real_agent_with_mode(
            session_id="sess_esc_override_02",
            arg_value="override_xyz",
            extra={"priority": "HIGH"},
        )
        ctx = self._ctx_with_real_agent(agent, request_id="req_esc_override_02")

        with patch("proxy.canary.verify_canary", side_effect=_verify_accept_any):
            from proxy.canary import _build_work_canary_for_request
            canary = _build_work_canary_for_request(ctx)

        skill_name, state = self._extract_canary_state(canary)
        self.assertEqual(skill_name, "work")
        self.assertIsInstance(state, dict)
        self.assertEqual(
            state.get("task_hash"), "override_xyz",
            "ModeState.skill_state['arg'] must override default empty skill_state",
        )

    def test_escalation_no_active_mode_yields_empty_skill_state(self):
        """Agent with no active ModeState → canary has empty/no JSON state."""
        from proxy.agent.agent import Agent

        agent = Agent(
            agent_id="parent_esc_no_mode_03",
            session_id="sess_esc_no_mode_03",
        )
        self.assertIsNone(agent.snapshot_mode(),
                          "precondition: agent has no active mode")
        ctx = self._ctx_with_real_agent(agent, request_id="req_esc_no_mode_03")

        with patch("proxy.canary.verify_canary", side_effect=_verify_accept_any):
            from proxy.canary import _build_work_canary_for_request
            canary = _build_work_canary_for_request(ctx)

        self.assertIsNotNone(canary, "must still produce a canary string")
        skill_name, state = self._extract_canary_state(canary)
        self.assertEqual(skill_name, "work")
        self.assertIsNone(
            state,
            "with no active ModeState, canary JSON state must be None "
            "(WorkSkill.get_inheritance_state returns None when no task_hash), "
            f"got state={state!r}",
        )

    def test_escalation_uses_canonical_skill_context_factory(self):
        """SkillContext built via from_request_context must carry canonical fields.

        Verifies the escalated SkillContext (the one fed to WorkSkill.get_canary)
        has the agent_id from the registry, the messages list from ctx, and the
        request_id propagated from ctx. Uses a real Agent so WorkSkill emits
        a real canary.
        """
        agent = self._make_real_agent_with_mode(
            session_id="sess_esc_canonical_04_unique",
            arg_value="canon1234",
        )
        ctx_messages = [
            {"role": "user", "content": "edit foo.py"},
            {"role": "assistant", "content": "OK"},
        ]
        ctx = self._ctx_with_real_agent(
            agent, messages=list(ctx_messages), request_id="req_esc_canon_04",
        )

        # Override conftest's MockAgent-returning get_or_load patch so
        # SkillContext.from_request_context sees our real Agent — this is
        # the production call shape (canary.py calls get_or_load(session_id)).
        from proxy.agent.agent_registry import registry as _canon_registry
        self._canon_patch = patch.object(
            _canon_registry, "get_or_load",
            side_effect=lambda sid: agent,
        )
        self._canon_patch.start()
        self.addCleanup(self._canon_patch.stop)

        # Patch WorkSkill.get_canary to capture the SkillContext it receives
        # and return a real canary so downstream parse works.
        captured = {}
        from skills.work_skill import WorkSkill
        real_canary = WorkSkill().get_canary(
            __import__("proxy.skills.skill_registry", fromlist=["SkillContext"]).SkillContext(
                session_id=agent.session_id,
                agent_id=agent.agent_id,
                messages=list(ctx_messages),
                skill_state={"task_hash": "canon1234"},
                request_id="req_esc_canon_04",
            )
        )

        def _record_and_return(skill_ctx):
            captured["skill_ctx"] = skill_ctx
            return real_canary

        with patch("skills.work_skill.WorkSkill.get_canary", side_effect=_record_and_return):
            from proxy.canary import _build_work_canary_for_request
            canary = _build_work_canary_for_request(ctx)

        self.assertIsNotNone(canary, "must produce a canary string")
        self.assertIn("skill_ctx", captured, "WorkSkill.get_canary must be called")
        sc = captured["skill_ctx"]
        self.assertEqual(sc.session_id, agent.session_id,
                         "session_id must come from ctx")
        self.assertEqual(
            sc.agent_id, agent.agent_id,
            "agent_id must come from agent_registry.get_or_load(session_id), "
            "not be the empty string",
        )
        self.assertEqual(
            sc.request_id, "req_esc_canon_04",
            "request_id must propagate from ctx to SkillContext",
        )
        self.assertEqual(
            list(sc.messages), ctx_messages,
            "messages must propagate from ctx to SkillContext",
        )
        self.assertEqual(
            sc.skill_state.get("arg"), "canon1234",
            "SkillContext.skill_state['arg'] must come from parent's ModeState.skill_state['arg']",
        )

    def test_escalation_propagates_parent_skill_state_via_registry_fallback(self):
        """Production path: ctx.agent is None → agent_registry.get_or_load() fallback.

        In production, ``RequestContext.agent`` is never set (verified by grep —
        no production code assigns ``ctx.agent = ...``). Therefore
        ``_build_work_canary_for_request`` MUST find the parent Agent via
        ``agent_registry.get_or_load(ctx.session_id)`` to read its active
        ``ModeState.skill_state``.

        This test uses a REAL ``Agent`` registered in ``agent_registry`` (not a
        mock — mocking masked the original bug class) and a ``SimpleNamespace``
        ctx where ``agent`` is genuinely absent (``getattr`` returns ``None``,
        no MagicMock auto-vivification).
        """
        from types import SimpleNamespace

        agent = self._make_real_agent_with_mode(
            session_id="sess_esc_registry_06_unique",
            arg_value="registry_path_hash",
            extra={"source": "registry_fallback"},
        )

        # SimpleNamespace ctx — `agent` is genuinely absent, mirroring production
        # reality where RequestContext.agent is never assigned. getattr(ctx,
        # "agent", None) returns None and triggers the registry fallback.
        ctx = SimpleNamespace(
            session_id=agent.session_id,
            agent_id=agent.agent_id,
            is_subagent=False,
            request_id="req_esc_registry_06",
            client_hint="test",
            messages=[{"role": "user", "content": "edit via registry path"}],
        )
        self.assertIsNone(
            getattr(ctx, "agent", None),
            "precondition: ctx.agent must be absent (production reality)",
        )

        # NOTE on test isolation: tests/conftest.py installs an autouse fixture
        # that patches proxy.agent.agent_registry.registry.get_or_load to return
        # a MockAgent. Our test needs canary.py's isinstance(agent, Agent) gate
        # to see a real Agent — MockAgent would fail isinstance and silently
        # skip the skill_state propagation, producing a canary without our
        # task_hash. We therefore override get_or_load with side_effect= that
        # returns our real Agent (this is the production call shape: canary.py
        # calls _agent_registry.get_or_load(session_id)). The conftest patch
        # is restored at test teardown via patch.object's context manager.
        from proxy.agent.agent_registry import registry as agent_registry

        with patch.object(
            agent_registry,
            "get_or_load",
            side_effect=lambda sid: agent,
        ):
            # Sanity: registry now returns our real agent (not MockAgent).
            self.assertIs(
                agent_registry.get_or_load(agent.session_id),
                agent,
                "registry.get_or_load(session_id) must return our real Agent, "
                "overriding the conftest MockAgent for this test",
            )

            with patch("proxy.canary.verify_canary", side_effect=_verify_accept_any):
                from proxy.canary import _build_work_canary_for_request
                canary = _build_work_canary_for_request(ctx)

        self.assertIsNotNone(canary, "must produce a canary string")
        skill_name, state = self._extract_canary_state(canary)
        self.assertEqual(skill_name, "work", "must produce a work canary")
        self.assertIsInstance(state, dict, "canary must carry JSON state")
        self.assertEqual(
            state.get("task_hash"), "registry_path_hash",
            "parent ModeState.skill_state['arg'] must propagate via "
            "agent_registry.get_or_load() fallback (ctx.agent was None)",
        )

    def test_escalation_propagates_request_id_to_log(self):
        """When WorkSkill.get_canary raises, the WARNING log includes request_id."""
        agent = self._make_real_agent_with_mode(
            session_id="sess_esc_log_05",
            arg_value="logtest9",
        )
        ctx = self._ctx_with_real_agent(agent, request_id="req_log_05_xyz")

        with patch(
            "skills.work_skill.WorkSkill.get_canary",
            side_effect=RuntimeError("simulated downstream failure"),
        ):
            with self.assertLogs("proxy.canary", level="WARNING") as cm:
                from proxy.canary import _build_work_canary_for_request
                result = _build_work_canary_for_request(ctx)

        self.assertIsNone(
            result,
            "_build_work_canary_for_request must swallow exceptions and return None",
        )
        msgs = [r.getMessage() for r in cm.records]
        self.assertTrue(
            any("WorkSkill.get_canary failed" in m for m in msgs),
            f"expected WorkSkill failure WARNING, got: {msgs}",
        )
        request_ids_in_records = [
            getattr(r, "request_id", None) for r in cm.records
            if hasattr(r, "request_id")
        ]
        self.assertIn(
            "req_log_05_xyz", request_ids_in_records,
            "request_id must be attached to WARNING log record via extra=",
        )


# ---------------------------------------------------------------------------
# B-5: Per-session rate limit on ticket->work escalation
# ---------------------------------------------------------------------------


class TestEscalationRateLimit(unittest.TestCase):
    """B-5: per-session rate limit on ticket→work escalation.

    Bounds the blast radius of prompt-injection that forces repeated
    ``subagent_mode="work"`` escalations. Default 5 escalations per
    1-hour sliding window per ``session_id``. Falls back to the default
    canary (no work tools) when the limit is exceeded.
    """

    def setUp(self):
        from proxy.state import escalation_rate_limiter
        self._limiter = escalation_rate_limiter
        self._limiter.reset()
        self._orig_max = self._limiter._max
        self._orig_window = self._limiter._window

    def tearDown(self):
        self._limiter.reset()
        self._limiter._max = self._orig_max
        self._limiter._window = self._orig_window

    def _resolve(self, args, default_mode_name="ticket", request_ctx=None):
        from proxy.canary import _resolve_task_canary
        return _resolve_task_canary(
            default_canary="DEFAULT_CANARY",
            args=args,
            default_mode_name=default_mode_name,
            request_ctx=request_ctx,
        )

    def _ctx(self, session_id):
        ctx = MagicMock()
        ctx.session_id = session_id
        ctx.agent_id = "agent_rl_" + session_id[-8:]
        ctx.request_id = "req_rl_" + session_id[-8:]
        ctx.is_subagent = False
        ctx.client_hint = "test"
        return ctx

    def _fresh_limiter(self, max_requests, window_seconds):
        from proxy.rate_limiter import RateLimiter
        return RateLimiter(
            max_requests=max_requests,
            window_seconds=window_seconds,
            name="escalation",
            persist=False,
        )

    def test_escalation_rate_limit_blocks_after_max(self):
        """After max_requests escalations, the N+1th is rate-limited."""
        test_limiter = self._fresh_limiter(max_requests=3, window_seconds=60)
        with patch("proxy.state.escalation_rate_limiter", test_limiter):
            ctx = self._ctx("sess_rl_max_01")
            args = {"subagent_mode": "work"}
            with patch(
                "proxy.canary._build_work_canary_for_request",
                return_value="WORK_CANARY",
            ):
                for i in range(3):
                    canary, kind = self._resolve(args, request_ctx=ctx)
                    self.assertEqual(
                        canary, "WORK_CANARY",
                        f"call {i+1}: expected work canary, got {canary!r}",
                    )
                    self.assertEqual(
                        kind, "work",
                        f"call {i+1}: expected kind=work, got {kind!r}",
                    )
                canary, kind = self._resolve(args, request_ctx=ctx)
                self.assertEqual(
                    canary, "DEFAULT_CANARY",
                    "4th call must fall back to default canary",
                )
                self.assertEqual(
                    kind, "rate_limited",
                    "4th call must be rate-limited",
                )

    def test_escalation_rate_limit_logs_warning(self):
        """Rate-limited escalation emits a WARNING log with 'rate-limited' text."""
        test_limiter = self._fresh_limiter(max_requests=1, window_seconds=60)
        with patch("proxy.state.escalation_rate_limiter", test_limiter):
            ctx = self._ctx("sess_rl_log_01")
            args = {"subagent_mode": "work"}
            with patch(
                "proxy.canary._build_work_canary_for_request",
                return_value="WORK_CANARY",
            ):
                canary, kind = self._resolve(args, request_ctx=ctx)
                self.assertEqual(kind, "work")
                with self.assertLogs("proxy.canary", level="WARNING") as cm:
                    canary, kind = self._resolve(args, request_ctx=ctx)
                self.assertEqual(kind, "rate_limited")
                msgs = [r.getMessage() for r in cm.records]
                self.assertTrue(
                    any("rate-limited" in m for m in msgs),
                    f"expected 'rate-limited' WARNING, got: {msgs}",
                )

    def test_escalation_rate_limit_different_sessions_independent(self):
        """Two different sessions do not share the same rate-limit bucket."""
        test_limiter = self._fresh_limiter(max_requests=2, window_seconds=60)
        with patch("proxy.state.escalation_rate_limiter", test_limiter):
            ctx_a = self._ctx("sess_rl_indep_a")
            ctx_b = self._ctx("sess_rl_indep_b")
            args = {"subagent_mode": "work"}
            with patch(
                "proxy.canary._build_work_canary_for_request",
                return_value="WORK_CANARY",
            ):
                for i in range(2):
                    canary, kind = self._resolve(args, request_ctx=ctx_a)
                    self.assertEqual(kind, "work", f"sess_a call {i+1} must succeed")
                canary, kind = self._resolve(args, request_ctx=ctx_a)
                self.assertEqual(
                    kind, "rate_limited",
                    "sess_a 3rd call must hit rate limit",
                )
                for i in range(2):
                    canary, kind = self._resolve(args, request_ctx=ctx_b)
                    self.assertEqual(
                        kind, "work",
                        f"sess_b call {i+1} must succeed (independent bucket)",
                    )
                canary, kind = self._resolve(args, request_ctx=ctx_b)
                self.assertEqual(
                    kind, "rate_limited",
                    "sess_b 3rd call must hit its own rate limit",
                )

    def test_escalation_rate_limit_no_session_id_bypasses(self):
        """When request_ctx is None or session_id is empty, the rate check is bypassed.

        We do NOT bucket anonymous requests together (they share no session_id),
        otherwise one anonymous request could rate-limit all others.
        """
        test_limiter = self._fresh_limiter(max_requests=1, window_seconds=60)
        with patch("proxy.state.escalation_rate_limiter", test_limiter):
            args = {"subagent_mode": "work"}
            with patch(
                "proxy.canary._build_work_canary_for_request",
                return_value="WORK_CANARY",
            ):
                canary, kind = self._resolve(args, request_ctx=None)
                self.assertEqual(kind, "work", "request_ctx=None must bypass rate limit")
                canary, kind = self._resolve(args, request_ctx=None)
                self.assertEqual(
                    kind, "work",
                    "repeated request_ctx=None must continue to bypass rate limit",
                )
                ctx_empty = self._ctx("")
                canary, kind = self._resolve(args, request_ctx=ctx_empty)
                self.assertEqual(
                    kind, "work",
                    "empty-string session_id must bypass rate limit",
                )
                canary, kind = self._resolve(args, request_ctx=ctx_empty)
                self.assertEqual(
                    kind, "work",
                    "repeated empty-string session_id must continue to bypass",
                )

    def test_escalation_rate_limit_default_parent_unchanged(self):
        """Parent in WORK mode never touches the rate limit (no escalation path).

        The rate limiter guards the ticket->work transition only. Parents
        already in work mode that pass ``subagent_mode="work"`` are a no-op
        and do not consume the bucket.
        """
        test_limiter = self._fresh_limiter(max_requests=1, window_seconds=60)
        with patch("proxy.state.escalation_rate_limiter", test_limiter):
            ctx = self._ctx("sess_rl_workparent_01")
            args = {"subagent_mode": "work"}
            with patch("proxy.canary._build_work_canary_for_request") as mock_build:
                for i in range(5):
                    canary, kind = self._resolve(
                        args, default_mode_name="work", request_ctx=ctx,
                    )
                    self.assertEqual(
                        canary, "DEFAULT_CANARY",
                        f"work parent call {i+1} must use default canary",
                    )
                    self.assertIsNone(
                        kind,
                        f"work parent call {i+1} must not escalate (kind={kind!r})",
                    )
                mock_build.assert_not_called()

    def test_escalation_rate_limit_window_resets(self):
        """After the window expires, escalation is allowed again."""
        test_limiter = self._fresh_limiter(max_requests=1, window_seconds=1)
        with patch("proxy.state.escalation_rate_limiter", test_limiter):
            ctx = self._ctx("sess_rl_window_01")
            args = {"subagent_mode": "work"}
            with patch(
                "proxy.canary._build_work_canary_for_request",
                return_value="WORK_CANARY",
            ):
                canary, kind = self._resolve(args, request_ctx=ctx)
                self.assertEqual(kind, "work", "1st call within window must succeed")
                canary, kind = self._resolve(args, request_ctx=ctx)
                self.assertEqual(
                    kind, "rate_limited",
                    "2nd call within window must be rate-limited",
                )
                time.sleep(1.2)
                canary, kind = self._resolve(args, request_ctx=ctx)
                self.assertEqual(
                    kind, "work",
                    "after window expires, escalation must be allowed again",
                )

    def test_escalation_rate_limit_real_session_id_extraction(self):
        """The rate limit key is the actual session_id pulled from request_ctx."""
        test_limiter = self._fresh_limiter(max_requests=2, window_seconds=60)
        with patch("proxy.state.escalation_rate_limiter", test_limiter):
            ctx = self._ctx("sess_abc_real_42")
            args = {"subagent_mode": "work"}
            with patch(
                "proxy.canary._build_work_canary_for_request",
                return_value="WORK_CANARY",
            ):
                for i in range(2):
                    canary, kind = self._resolve(args, request_ctx=ctx)
                    self.assertEqual(
                        kind, "work",
                        f"call {i+1} with real session_id must succeed",
                    )
                canary, kind = self._resolve(args, request_ctx=ctx)
                self.assertEqual(
                    kind, "rate_limited",
                    "3rd call with same real session_id must hit rate limit",
                )


# ---------------------------------------------------------------------------
# PER-TURN contract: apply_post_process clears MAIN-agent mode on stop
# (T1b) and the money test (T2) — a stop followed by a plain task call
# yields a BARE sub-agent, not a leaked ticket sub-agent.
# ---------------------------------------------------------------------------


class TestApplyPostProcessMainAgentStopClears(unittest.TestCase):
    """apply_post_process clears active mode when a MAIN agent emits stop.

    THE guard against wiring regressions: the per-turn contract must hold
    end-to-end through the real PostProcessStage.execute() + real Agent + real
    registry. Fails on the pre-fix code where TicketSkill.should_deactivate
    returned False for main agents.

    Wave 4: applies PostProcessStage.execute() (the post-process path) rather
    than calling apply_post_process directly. The unrelated sub-methods are
    patched to no-ops so only the 3 post-process ops + finalize run.
    """

    def setUp(self):
        self._verify_patcher = patch(
            "proxy.canary.verify_canary", side_effect=_verify_accept_any,
        )
        self._verify_patcher.start()

    def tearDown(self):
        self._verify_patcher.stop()

    def _make_ctx(self, agent):
        ctx = MagicMock()
        ctx.session_id = agent.session_id
        ctx.agent_id = agent.agent_id
        ctx.is_subagent = False
        ctx.client_hint = "test"
        ctx.messages = []
        ctx.body = {"messages": []}
        ctx.request_id = "req_pp_stop"
        ctx.model_route = MagicMock()
        ctx.model_route.is_remote = False
        ctx.model_route.model_cfg = None
        ctx.upstream = None
        ctx.swarm_policy = None
        ctx.llama_body = ctx.body
        return ctx

    def test_ticket_mode_cleared_on_main_stop(self):
        """Real Agent in ticket mode + PostProcessStage.execute(stop) → mode None."""
        from proxy.mode import Mode
        from proxy.pipeline.post_process import PostProcessStage
        from proxy.agent.agent import Agent
        from proxy.skills.skill_definition import ModeState

        agent = Agent(agent_id="main_pp_stop_ticket", session_id="sess_pp_stop_ticket")
        ms = ModeState(name="ticket", activated_at=time.time(), ttl_seconds=300, skill_state={})
        self.assertTrue(agent.set_active_mode(ms))

        ctx = self._make_ctx(agent)
        ctx._mode = Mode.SKILL
        data = {"choices": [{"message": {"role": "assistant", "content": "done"}, "finish_reason": "stop"}]}
        ctx.response_data = data

        # R1-A: pre-upstream snapshot must match the agent's active_mode so the
        # PostProcessStage CAS (clear_active_mode(expected_mode=...)) succeeds
        # against the COMPLETE turn disposition (finish_reason="stop").
        ctx.mode_snapshot_pre_upstream = ms
        with _patch_get_or_load(agent), _noop_postprocess_submethods():
            PostProcessStage().execute(ctx)
        self.assertIsNone(
            agent.snapshot_mode(),
            "ticket mode MUST clear on main-agent stop (per-turn contract)",
        )

    def test_custom_skill_mode_cleared_on_main_stop(self):
        """A custom (JSON) skill's mode clears on main-agent stop too."""
        import json
        import os
        import tempfile
        from proxy.mode import Mode
        from proxy.pipeline.post_process import PostProcessStage
        from proxy.agent.agent import Agent
        from proxy.skills.skill_definition import ModeState
        from proxy.skills.skill_registry import SkillRegistry, _skill_registry_new__

        # Build an isolated registry: builtins + a custom JSON skill.
        tmpdir = tempfile.mkdtemp(prefix="skills_pp_stop_")
        skill_path = os.path.join(tmpdir, "mytask.json")
        with open(skill_path, "w") as f:
            json.dump({
                "name": "mytask", "trigger": "^^mytask",
                "description": "do mytask",
                "prompt": "do it", "tool_whitelist": ["read"],
            }, f)

        old_env = os.environ.get("PROXY_SKILLS_DIR")
        os.environ["PROXY_SKILLS_DIR"] = tmpdir
        try:
            # Wave 1: populate_builtins now loads custom JSON atomically
            # inside the same call — no separate _load_custom_skills.
            reg = _skill_registry_new__(SkillRegistry)
            reg.populate_builtins()
        finally:
            if old_env is None:
                os.environ.pop("PROXY_SKILLS_DIR", None)
            else:
                os.environ["PROXY_SKILLS_DIR"] = old_env
        self.assertIsNotNone(reg.get("mytask"), "custom skill must load")

        agent = Agent(agent_id="main_pp_stop_custom", session_id="sess_pp_stop_custom")
        ms = ModeState(name="mytask", activated_at=time.time(), ttl_seconds=300, skill_state={})
        self.assertTrue(agent.set_active_mode(ms))

        ctx = self._make_ctx(agent)
        ctx._mode = Mode.SKILL
        data = {"choices": [{"message": {"role": "assistant", "content": "done"}, "finish_reason": "stop"}]}
        ctx.response_data = data

        # R1-A: pre-upstream snapshot must match the agent's active_mode so the
        # PostProcessStage CAS (clear_active_mode(expected_mode=...)) succeeds
        # against the COMPLETE turn disposition (finish_reason="stop").
        ctx.mode_snapshot_pre_upstream = ms
        with patch("proxy.skills.skill_registry.SkillRegistry", return_value=reg), \
             _patch_get_or_load(agent), _noop_postprocess_submethods():
            PostProcessStage().execute(ctx)
        self.assertIsNone(
            agent.snapshot_mode(),
            "custom skill mode MUST clear on main-agent stop (per-turn contract)",
        )


class TestTicketStopThenPlainPromptBareSubagent(unittest.TestCase):
    """THE money test: a ticket stop followed by a plain `task` call yields a
    BARE sub-agent, not a leaked ticket sub-agent.

    Sequence on ONE real Agent:
      1. ticket mode active
      2. PostProcessStage.execute(stop) → mode cleared (FAILS pre-fix)
      3. PostProcessStage.execute(tool_calls with task) → canary is BARE
      4. default tools contain edit/write/bash and NOT proxy_ticket
    """

    def setUp(self):
        self._verify_patcher = patch(
            "proxy.canary.verify_canary", side_effect=_verify_accept_any,
        )
        self._verify_patcher.start()
        # No active workspace → bare canary (not workspace canary) on turn 2.
        self._ws_patcher = patch("proxy.workspace_state.workspace_store")
        ws_mock = self._ws_patcher.start()
        ws_mock.get_active.return_value = "default"

    def tearDown(self):
        self._ws_patcher.stop()
        self._verify_patcher.stop()

    def test_stop_then_task_yields_bare_subagent(self):
        from proxy.mode import Mode
        from proxy.pipeline.post_process import PostProcessStage
        from proxy.agent.agent import Agent
        from proxy.skills.skill_definition import ModeState
        from proxy.tool_pipeline.tool_registry import registry as tool_registry

        agent = Agent(agent_id="parent_money", session_id="sess_money")
        ms = ModeState(name="ticket", activated_at=time.time(), ttl_seconds=300, skill_state={})
        self.assertTrue(agent.set_active_mode(ms))

        ctx = MagicMock()
        ctx.session_id = agent.session_id
        ctx.agent_id = agent.agent_id
        ctx.is_subagent = False
        ctx.client_hint = "test"
        ctx.messages = []
        ctx.body = {"messages": []}
        ctx.request_id = "req_money"
        ctx._mode = Mode.SKILL
        ctx.model_route = MagicMock()
        ctx.model_route.is_remote = False
        ctx.model_route.model_cfg = None
        ctx.upstream = None
        ctx.swarm_policy = None
        ctx.llama_body = ctx.body

        # --- Turn 1: main agent finishes ticket work with stop ---
        data_stop = {"choices": [{"message": {"role": "assistant", "content": "done"}, "finish_reason": "stop"}]}
        ctx.response_data = data_stop
        # R1-A: pre-upstream snapshot must match the agent's active_mode so the
        # PostProcessStage CAS (clear_active_mode(expected_mode=...)) succeeds
        # against the COMPLETE turn disposition (finish_reason="stop").
        ctx.mode_snapshot_pre_upstream = ms
        with _patch_get_or_load(agent), _noop_postprocess_submethods():
            PostProcessStage().execute(ctx)
        self.assertIsNone(
            agent.snapshot_mode(),
            "turn 1: ticket mode MUST clear on main-agent stop — this is the bug "
            "(pre-fix this assertion fails and turn 2 leaks a ticket canary)",
        )

        # --- Turn 2: a plain `task` call on a subsequent prompt ---
        data_task = {"choices": [{"message": {"role": "assistant", "content": None,
            "tool_calls": [{"id": "call_money", "type": "function",
                "function": {"name": "task",
                    "arguments": json.dumps({"prompt": "edit README.md"})}}]},
            "finish_reason": "tool_calls"}]}
        ctx.response_data = data_task
        # After turn 1 the agent has no active mode; the snapshot is None so the
        # finalize_skill_mode None-ownership guard (R2 #1) preserves the absence.
        ctx.mode_snapshot_pre_upstream = None
        with _patch_get_or_load(agent), _noop_postprocess_submethods():
            result = PostProcessStage().execute(ctx)

        self.assertIsInstance(result, dict)
        task_prompt = json.loads(
            result["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"]
        )["prompt"]

        # Canary MUST be bare, NOT SKILL:ticket.
        _p = parse_canary(task_prompt)
        skill_name = _p.skill_name if (_p and _p.verified) else None
        self.assertIsNone(
            skill_name,
            "turn 2: after ticket stop, the task call must carry a BARE canary; "
            f"got skill={skill_name!r} prompt_prefix={task_prompt[:80]!r}",
        )
        self.assertTrue(
            task_prompt.startswith("[[__SUBAGENT__]]"),
            f"turn 2: expected bare canary, got: {task_prompt[:80]!r}",
        )
        self.assertNotIn("[[__SUBAGENT__:SKILL:ticket:", task_prompt)

        # Default tool set: bare sub-agents get edit/write/bash, NOT proxy_ticket.
        defaults = tool_registry.get_default_tools()
        self.assertIn("edit", defaults)
        self.assertIn("write", defaults)
        self.assertIn("bash", defaults)
        self.assertNotIn(
            "proxy_ticket", defaults,
            "bare sub-agents must NOT see proxy_ticket (admin tool requires skill ctx)",
        )


if __name__ == "__main__":
    unittest.main()
