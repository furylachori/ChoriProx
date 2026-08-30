"""Phase 2 unit tests for canary parsing and injection.

Tests AC 2.16, 2.17, 2.19 for sub-agent canary handling per §11.1-§11.3
of the skill-system-spec.

All canaries in these tests carry a 64-character HMAC signature and a
fresh timestamp (or are mocked through ``verify_canary``) because the
production parser rejects bare/legacy formats.
"""
import json
import threading
import time
import unittest
from unittest.mock import patch, MagicMock

from proxy.canary import (
    ParsedCanary,
    parse_canary,
    verify_canary,
    _strip_existing_canaries,
    _inject_canary_into_task_calls,
    detect_subagent_ingress,
)
from proxy.mode import Mode


def _mock_skill_registry():
    """Create a mock SkillRegistry that returns valid skills."""
    mock_registry = MagicMock()
    mock_skill = MagicMock()
    mock_skill.name = "ticket"
    mock_registry.get.return_value = mock_skill
    return mock_registry


_FRESH_TS = str(int(time.time()))
_FAKE_SIG = "0" * 64  # 64 hex chars; verify_canary only checks length


def _make_post_process_ctx(session_id="sess_canary_pp", agent_id="agent_canary_pp"):
    """Build a RequestContext-like namespace suitable for PostProcessStage.execute().

    Wave 4 migration: replaces direct apply_post_process(ctx, agent, data, mode)
    calls. The ctx carries response_data (set by caller), model_route, body,
    and all the attributes PostProcessStage touches.
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
    ctx.client_hint = ""
    ctx.messages = []
    ctx.body = body
    ctx.request_id = "req_canary_pp"
    ctx.model_route = MagicMock()
    ctx.model_route.is_remote = False
    ctx.model_route.model_cfg = None
    ctx.upstream = None
    ctx.llama_body = body
    ctx.swarm_policy = None  # Explicit None — MagicMock auto-vivification would
                              # trigger malformed-policy fail-closed.
    return ctx


def _noop_canary_submethods():
    """Patch the 8 PostProcessStage sub-methods to no-ops for canary tests."""
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


def _patch_get_or_load_canary(agent):
    """Return a patcher that makes agent_registry.get_or_load return the test's agent."""
    def _get_or_load(session_id):
        return agent
    return patch(
        "proxy.agent.agent_registry.registry.get_or_load",
        side_effect=_get_or_load,
    )


def _sign_token(skill_name: str) -> str:
    """Build a no-secret canary token (skill:sig:ts) for tests."""
    return f"{skill_name}:{_FAKE_SIG}:{_FRESH_TS}"


def _make_pc(skill_name: str) -> ParsedCanary:
    """Build a ParsedCanary matching a no-secret verification result."""
    return ParsedCanary(
        skill=skill_name,
        timestamp=int(_FRESH_TS),
        sig=_FAKE_SIG,
        is_dev_fallback=True,
    )


class TestParseCanaryNewFormat(unittest.TestCase):
    """Tests for parse_canary with new format [[__SUBAGENT__:SKILL:...]]."""

    def setUp(self):
        """Mock SkillRegistry and verify_canary to allow canary parsing."""
        self.registry_patcher = patch('proxy.skills.skill_registry.SkillRegistry', return_value=_mock_skill_registry())
        self.registry_patcher.start()
        self.verify_patcher = patch(
            'proxy.canary.verify_canary',
            # Accept B-3/B-4 kwargs (state_str, session_id) without inspection —
            # the tests in this class are about PARSING, not HMAC binding.
            side_effect=lambda token, skill=None, **kwargs: _make_pc(token.split(":", 1)[0]),
        )
        self.verify_patcher.start()

    def tearDown(self):
        """Stop mocking SkillRegistry and verify_canary."""
        self.registry_patcher.stop()
        self.verify_patcher.stop()

    def test_parse_canary_new_format_with_state(self):
        """Parse [[__SUBAGENT__:SKILL:ticket:<sig>:<ts>|{json}]] → verified skill with state."""
        text = f'[[__SUBAGENT__:SKILL:ticket:{_FAKE_SIG}:{_FRESH_TS}|{{"skill_state": "val"}}]]'
        p = parse_canary(text)

        self.assertIsNotNone(p)
        self.assertEqual(p.kind, "skill")
        self.assertTrue(p.verified)
        self.assertEqual(p.skill_name, "ticket")
        self.assertEqual(p.state, {"skill_state": "val"})
        # remaining is everything after the closing ]]
        self.assertEqual(p.remaining, "")

    def test_parse_canary_with_trailing_double_brackets(self):
        """Regression: trailing text containing ']]' must not confuse the canary close.

        Before the fix, the parser used rfind(']]') to locate the canary
        close. If the trailing prompt text contained ']]' (e.g., a code snippet,
        markdown link, or wiki reference), rfind would land on the trailing ']]'
        and the JSON state would be lost (parsed as invalid → state=None).
        """
        # Trailing prompt contains ]] (e.g., wiki link or code snippet)
        text = f'[[__SUBAGENT__:SKILL:ticket:{_FAKE_SIG}:{_FRESH_TS}|{{"tasks": []}}]]\n\nSee doc]] for details'
        p = parse_canary(text)

        self.assertEqual(p.kind, "skill")
        self.assertTrue(p.verified)
        self.assertEqual(p.skill_name, "ticket")
        self.assertEqual(p.state, {"tasks": []})
        # Leading newlines are stripped (existing behavior); trailing ']]' is preserved.
        self.assertEqual(p.remaining, "See doc]] for details")

    def test_parse_canary_json_state_containing_close_brackets(self):
        """JSON state containing ']]' inside a string value must not break parsing.

        The JSON value ']]' inside a string is not a canary close — it is data.
        The parser must find the actual canary close at the end of the canary.
        """
        text = f'[[__SUBAGENT__:SKILL:ticket:{_FAKE_SIG}:{_FRESH_TS}|{{"note": "use ]] as sentinel"}}]]'
        p = parse_canary(text)

        self.assertEqual(p.kind, "skill")
        self.assertTrue(p.verified)
        self.assertEqual(p.skill_name, "ticket")
        self.assertEqual(p.state, {"note": "use ]] as sentinel"})
        self.assertEqual(p.remaining, "")

    def test_parse_canary_json_state_containing_pipe(self):
        """JSON state containing '|' inside a string value must not break parsing.

        The first '|' after the skill name is the JSON separator. Any '|' inside
        the JSON (e.g., a regex or URL) must not be mistaken for the separator.
        """
        text = f'[[__SUBAGENT__:SKILL:ticket:{_FAKE_SIG}:{_FRESH_TS}|{{"regex": "a|b", "url": "https://x.com|y"}}]]'
        p = parse_canary(text)

        self.assertEqual(p.kind, "skill")
        self.assertTrue(p.verified)
        self.assertEqual(p.skill_name, "ticket")
        self.assertEqual(p.state, {"regex": "a|b", "url": "https://x.com|y"})
        self.assertEqual(p.remaining, "")

    def test_parse_canary_new_format_no_state(self):
        """Parse [[__SUBAGENT__:SKILL:workspace:<sig>:<ts>]] → verified skill 'workspace'."""
        text = f'[[__SUBAGENT__:SKILL:workspace:{_FAKE_SIG}:{_FRESH_TS}]]'
        p = parse_canary(text)

        self.assertEqual(p.kind, "skill")
        self.assertTrue(p.verified)
        self.assertEqual(p.skill_name, "workspace")
        self.assertIsNone(p.state)
        self.assertEqual(p.remaining, "")

    def test_parse_canary_new_format_with_trailing_text(self):
        """Parse canary followed by text → remaining includes trailing content."""
        text = f'[[__SUBAGENT__:SKILL:ticket:{_FAKE_SIG}:{_FRESH_TS}]]\n\nHello world'
        p = parse_canary(text)

        self.assertEqual(p.kind, "skill")
        self.assertTrue(p.verified)
        self.assertEqual(p.skill_name, "ticket")
        self.assertEqual(p.remaining, "Hello world")

    def test_parse_canary_bare_no_sig_rejected(self):
        """Skill canary without sig:ts is structurally malformed → unverified."""
        text = '[[__SUBAGENT__:SKILL:ticket]]'
        p = parse_canary(text)

        self.assertEqual(p.kind, "skill")
        self.assertFalse(p.verified)
        self.assertIsNone(p.skill_name)
        self.assertEqual(p.remaining, text)
        self.assertIsNone(p.state)


class TestParseCanaryBareAndUnrecognized(unittest.TestCase):
    """Tests for bare [[__SUBAGENT__]] and unrecognized formats."""

    def test_parse_canary_bare_subagent(self):
        """Parse [[__SUBAGENT__]] → bare kind, no skill inheritance."""
        p = parse_canary('[[__SUBAGENT__]]')

        self.assertIsNotNone(p)
        self.assertEqual(p.kind, "bare")
        self.assertTrue(p.verified)
        self.assertIsNone(p.skill_name)
        self.assertEqual(p.remaining, "")
        self.assertIsNone(p.state)

    def test_parse_canary_unrecognized(self):
        """Parse [[__SUBAGENT__:UNKNOWN_FORMAT]] → no canary marker (None)."""
        p = parse_canary('[[__SUBAGENT__:UNKNOWN_FORMAT]]')

        # No recognised canary prefix at start → None
        self.assertIsNone(p)

    def test_parse_canary_malformed(self):
        """Parse \"not a canary at all\" → no canary marker (None)."""
        p = parse_canary("not a canary at all")

        self.assertIsNone(p)

    def test_parse_canary_malformed_no_closing_brackets(self):
        """Parse [[__SUBAGENT__:SKILL:open → structurally malformed skill (unverified)."""
        text = '[[__SUBAGENT__:SKILL:open'
        p = parse_canary(text)

        self.assertEqual(p.kind, "skill")
        self.assertFalse(p.verified)
        self.assertIsNone(p.skill_name)
        # No reliable close found → original text returned untouched.
        self.assertEqual(p.remaining, text)
        self.assertIsNone(p.state)

    def test_parse_canary_invalid_skill_name_numbers(self):
        """Parse [[__SUBAGENT__:SKILL:123invalid]] → unverified (invalid name)."""
        text = '[[__SUBAGENT__:SKILL:123invalid]]'
        p = parse_canary(text)

        self.assertEqual(p.kind, "skill")
        self.assertFalse(p.verified)
        # Skill names must start with lowercase letter
        self.assertIsNone(p.skill_name)
        self.assertIsNone(p.state)

    def test_parse_canary_invalid_dunder_name(self):
        """Parse [[__SUBAGENT__:SKILL:__init__]] → unverified (dunder rejected)."""
        text = '[[__SUBAGENT__:SKILL:__init__]]'
        p = parse_canary(text)

        self.assertEqual(p.kind, "skill")
        self.assertFalse(p.verified)
        # Dunder names are rejected per spec F-28
        self.assertIsNone(p.skill_name)
        self.assertIsNone(p.state)


class TestNoSessionSecretFailOpen(unittest.TestCase):
    """Dev-mode fallback: when PROXY_SESSION_SECRET is unset, canary verification
    must fail-OPEN (return ParsedCanary with is_dev_fallback=True) instead of
    fail-closed.

    Regression test for the bug where ticket-mode sub-agents spawned via the
    `task` tool were detected (`is_subagent=True`) but did NOT inherit the
    ticket skill (`subagent_ticket_mode=False`, `inherited_skill=None`).
    Root cause: `_sign_canary` returns `""` when the secret is unset, so the
    production HMAC path would reject every canary. With the no-secret
    fallback, `verify_canary` accepts a fresh 64-char sig + fresh timestamp.
    """

    def setUp(self):
        # Ensure no PROXY_SESSION_SECRET leak from prior tests (e.g.
        # test_canary_repro.py sets it via os.environ.setdefault with no
        # cleanup). The leaked env var can repopulate canary_mod._SESSION_SECRET
        # via the canary signer's env-var read, defeating the no-secret branch.
        import os
        self._had_proxy_session_secret = "PROXY_SESSION_SECRET" in os.environ
        self._original_proxy_session_secret = os.environ.pop(
            "PROXY_SESSION_SECRET", None,
        )
        # Mock the skill registry so parse_canary's "SEC-006" check passes.
        self.registry_patcher = patch(
            'proxy.skills.skill_registry.SkillRegistry',
            return_value=_mock_skill_registry(),
        )
        self.registry_patcher.start()
        # Clear the module-level secret so verification enters the no-secret
        # branch and accepts any well-formed 64-char sig + fresh timestamp.
        from proxy import canary as canary_mod
        self._original_secret = canary_mod._SESSION_SECRET
        canary_mod._SESSION_SECRET = ""

    def tearDown(self):
        self.registry_patcher.stop()
        from proxy import canary as canary_mod
        canary_mod._SESSION_SECRET = self._original_secret
        # Restore PROXY_SESSION_SECRET if it was present at setUp time.
        import os
        if self._had_proxy_session_secret:
            os.environ["PROXY_SESSION_SECRET"] = self._original_proxy_session_secret
        # If we removed it, leave it removed — matches the principle that this
        # test class wants the env var absent for the duration of the test.

    def test_parse_canary_ticket_when_no_secret(self):
        """PROXY_SESSION_SECRET="" → parse_canary should verify skill_name='ticket'."""
        text = f'[[__SUBAGENT__:SKILL:ticket:{_FAKE_SIG}:{_FRESH_TS}|{{"tasks": []}}]]'

        p = parse_canary(text)

        # Regression assertion: the bug was returning None here.
        self.assertEqual(p.kind, "skill")
        self.assertTrue(p.verified)
        self.assertEqual(p.skill_name, "ticket")
        self.assertEqual(p.state, {"tasks": []})
        self.assertEqual(p.remaining, "")

    def test_parse_canary_workspace_when_no_secret(self):
        """PROXY_SESSION_SECRET="" → workspace canary should also parse cleanly."""
        text = f'[[__SUBAGENT__:WORKSPACE:default:{_FAKE_SIG}:{_FRESH_TS}]]'

        p = parse_canary(text)

        self.assertEqual(p.kind, "workspace")
        self.assertTrue(p.verified)
        self.assertEqual(p.workspace_name, "default")
        self.assertEqual(p.remaining, "")

    def test_verify_canary_returns_parsedcanary_when_no_secret(self):
        """Direct unit: verify_canary returns ParsedCanary (fail-open) when secret unset."""
        from proxy import canary as canary_mod
        # Well-formed 3-part token (skill:sig:ts) verifies successfully in dev mode
        result = canary_mod.verify_canary(f"ticket:{_FAKE_SIG}:{_FRESH_TS}")
        self.assertIsNotNone(result)
        self.assertTrue(result.is_dev_fallback)
        self.assertEqual(result.skill, "ticket")

    def test_detect_subagent_ingress_ticket_inherits_when_no_secret(self):
        """End-to-end: detect_subagent_ingress sets subagent_ticket_mode=True
        when a ticket-skill canary is presented and the secret is unset.
        This is the user-facing regression: the sub-agent must inherit the
        ticket tool set (proxy_ticket, proxy_work) even without a secret.
        """
        from proxy.canary import detect_subagent_ingress
        body = {
            "messages": [
                {"role": "user", "content": f'[[__SUBAGENT__:SKILL:ticket:{_FAKE_SIG}:{_FRESH_TS}|{{"tasks": []}}]]\nHello'}
            ]
        }

        result = detect_subagent_ingress(body)

        # The canary is well-formed and the proxy minted it itself, so the
        # sub-agent MUST inherit ticket mode in dev (no-secret) environments.
        self.assertTrue(result.is_subagent)
        self.assertTrue(result.subagent_ticket_mode)
        self.assertEqual(result.inherited_skill, "ticket")
        # Canary must be stripped from the user message
        self.assertEqual(body["messages"][0]["content"], "Hello")


class TestStripCanaries(unittest.TestCase):
    """Tests for _strip_existing_canaries function."""

    def test_strip_canaries_strips_both_formats(self):
        """Strips [[__SUBAGENT__:SKILL:ticket:sig:ts]] and bare [[__SUBAGENT__]] markers."""
        text = (
            'prefix [[__SUBAGENT__]] middle '
            f'[[__SUBAGENT__:SKILL:ticket:{_FAKE_SIG}:{_FRESH_TS}]] '
            'suffix'
        )
        result = _strip_existing_canaries(text)

        self.assertNotIn("[[__SUBAGENT__:", result)
        self.assertNotIn("[[__SUBAGENT__]]", result)
        self.assertIn("prefix", result)
        self.assertIn("middle", result)
        self.assertIn("suffix", result)

    def test_strip_canaries_no_op_on_clean_text(self):
        """Stripping clean text returns the same text unchanged."""
        text = "This is clean text with no canaries."
        result = _strip_existing_canaries(text)

        self.assertEqual(result, text)

    def test_strip_canaries_removes_bare_canary(self):
        """Removes bare [[__SUBAGENT__]] markers."""
        text = 'Hello [[__SUBAGENT__]] world'
        result = _strip_existing_canaries(text)

        self.assertNotIn("[[__SUBAGENT__]]", result)
        self.assertIn("Hello", result)
        self.assertIn("world", result)

    def test_strip_canaries_removes_with_state(self):
        """Removes canary with JSON state (HMAC-signed format)."""
        text = f'[[__SUBAGENT__:SKILL:ticket:{_FAKE_SIG}:{_FRESH_TS}|{{"key":"val"}}]] and more'
        result = _strip_existing_canaries(text)

        self.assertNotIn("[[__SUBAGENT__:", result)
        self.assertIn("and more", result)


class TestStripCanariesWithJsonArrayState(unittest.TestCase):
    """Regression tests for the stripper when the canary state contains `]`.

    BUG: The previous regex-based stripper used `(?:\\|[^\\]]+)?` for the
    SKILL canary state, which cannot match states containing ']'. The
    production canary produced by `TicketSkill.get_canary()` includes a JSON
    state of the form `{"tasks": [{"short_hash": "..."}]}` — a JSON array
    whose closing `]` lives inside the canary. The old regex failed to
    recognise the full canary, so `_strip_existing_canaries` left it
    untouched, and `_inject_canary_into_task_calls` then prepended a fresh
    canary on top, yielding two canaries in the sub-agent's first user
    message. `detect_subagent_ingress` matched the bare branch first and
    the sub-agent lost its skill inheritance.
    """

    _PRODUCTION_STATE = {
        "tasks": [
            {
                "short_hash": "a1b2c3d4",
                "name": "Fix login bug",
                "priority": "high",
                "status": "in_progress",
                "summary": "...",
                "tag": "auth",
                "blocked_by": [],
                "is_blocked": False,
                "blocked_count": 0,
                "parent_short_hash": None,
                "parent_name": None,
            },
        ],
        "store_status": "connected",
    }
    _PRODUCTION_CANARY = (
        "[[__SUBAGENT__:SKILL:ticket:"
        + _FAKE_SIG
        + ":"
        + _FRESH_TS
        + "|"
        + json.dumps(_PRODUCTION_STATE)
        + "]]"
    )

    def test_strip_production_canary_with_array_state(self):
        """The full production canary (with a JSON-array state) must strip cleanly."""
        result = _strip_existing_canaries(self._PRODUCTION_CANARY)
        self.assertEqual(result, "")

    def test_strip_production_canary_preserves_trailing_prompt(self):
        """Stripping the production canary leaves the trailing prompt intact."""
        prompt = self._PRODUCTION_CANARY + "Look at the auth bug."
        result = _strip_existing_canaries(prompt)
        self.assertEqual(result, "Look at the auth bug.")

    def test_double_injection_yields_exactly_one_canary(self):
        """The user-facing symptom: two `_inject_canary_into_task_calls` runs
        on the same task prompt must leave EXACTLY ONE canary in the result.

        Before the fix, the first run produced
            [[__SUBAGENT__:SKILL:ticket:...|{...}]]Look at the auth bug.
        and the second run's stripper failed (state contains ']'), so the
        second run prepended a new canary on top, yielding
            [[__SUBAGENT__:SKILL:ticket:...|{...}]][[__SUBAGENT__:SKILL:ticket:...|{...}]]Look at the auth bug.
        — TWO canaries. The fix changes the stripper to a position-based
        parser that correctly identifies and removes the canary regardless
        of state content.
        """
        from proxy.canary import _inject_canary_into_task_calls

        # First injection — clean prompt.
        data = {
            "choices": [{
                "message": {
                    "tool_calls": [{
                        "function": {
                            "name": "task",
                            "arguments": json.dumps({
                                "prompt": "Look at the auth bug.",
                            }),
                        },
                    }],
                },
            }],
        }
        data1 = _inject_canary_into_task_calls(data, self._PRODUCTION_CANARY)
        prompt_after_first = json.loads(
            data1["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"]
        )["prompt"]

        # Second injection — simulates apply_post_process running on a parent
        # response whose tool-call prompt already carries the previous canary
        # (e.g. when the parent's prompt is regenerated through a re-query loop
        # or replayed from chat history).
        data2 = _inject_canary_into_task_calls(data1, self._PRODUCTION_CANARY)
        prompt_after_second = json.loads(
            data2["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"]
        )["prompt"]

        canary_count = prompt_after_second.count("[[__SUBAGENT__")
        self.assertEqual(
            canary_count, 1,
            f"Expected EXACTLY ONE canary after double injection, got "
            f"{canary_count}. prompt={prompt_after_second!r}",
        )
        # The single canary must be the production SKILL canary.
        self.assertTrue(
            prompt_after_second.startswith("[[__SUBAGENT__:SKILL:ticket:"),
            f"Prompt should start with the SKILL canary. Got: "
            f"{prompt_after_second[:80]!r}",
        )
        # No leading newline before the canary.
        self.assertFalse(
            prompt_after_second.startswith("\n[[__SUBAGENT__:SKILL:"),
            f"Prompt must NOT start with a newline before the canary. "
            f"Got: {prompt_after_second[:80]!r}",
        )
        # Original prompt text is preserved at the end.
        self.assertTrue(
            prompt_after_second.endswith("Look at the auth bug."),
            f"Original prompt must be preserved. Got end: "
            f"{prompt_after_second[-40:]!r}",
        )

    def test_apply_post_process_task_prompt_has_exactly_one_canary(self):
        """End-to-end: PostProcessStage.execute() on a parent response with a `task`
        tool call must produce a task prompt with EXACTLY ONE canary (the
        SKILL one) and no leading newline.

        This is the user-facing regression test for the bug. The parent is
        in TICKET mode so the canary resolution finds the SKILL canary.
        The task prompt must:
          - have exactly one canary
          - start with the SKILL canary (not with a newline, not with a
            bare canary followed by the SKILL one)
        """
        from proxy.mode import Mode
        from proxy.pipeline.post_process import PostProcessStage

        # Realistic production canary (state contains ']').
        canary = self._PRODUCTION_CANARY

        # Build a skill whose get_canary() returns the production canary.
        from skills.ticket_skill import TicketSkill
        from proxy.skills.skill_registry import SkillContext

        skill = TicketSkill()
        skill_ctx = SkillContext(
            session_id="sess_double_inject",
            agent_id="agent_double_inject",
            messages=[],
        )

        # Patch get_canary to return our production canary with [] in state.
        from unittest.mock import patch as _patch
        with _patch.object(skill, "get_canary", return_value=canary):
            mock_registry = MagicMock()
            mock_registry.get.return_value = skill
            mock_agent = MagicMock()
            mock_agent.snapshot_mode.return_value = MagicMock(
                name="ticket", skill_state={},
            )
            mock_agent.agent_id = "agent_double_inject"
            mock_agent.session_id = "sess_double_inject"
            ctx = _make_post_process_ctx(
                session_id="sess_double_inject",
                agent_id="agent_double_inject",
            )
            ctx._mode = Mode.SKILL
            parent_response = {
                "choices": [{
                    "message": {
                        "role": "assistant",
                        "content": "Delegating.",
                        "tool_calls": [{
                            "id": "call_task_e2e",
                            "type": "function",
                            "function": {
                                "name": "task",
                                "arguments": json.dumps({
                                    "description": "Investigate",
                                    "prompt": "Look at the auth bug.",
                                }),
                            },
                        }],
                    },
                    "finish_reason": "tool_calls",
                }],
            }
            ctx.response_data = parent_response
            with _patch(
                "proxy.skills.skill_registry.SkillRegistry",
                return_value=mock_registry,
            ), _patch_get_or_load_canary(mock_agent), _noop_canary_submethods():
                result = PostProcessStage().execute(ctx)

        task_args = json.loads(
            result["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"]
        )
        task_prompt = task_args["prompt"]
        canary_count = task_prompt.count("[[__SUBAGENT__")
        self.assertEqual(
            canary_count, 1,
            f"Expected EXACTLY ONE canary in the task prompt, got "
            f"{canary_count}. prompt={task_prompt!r}",
        )
        self.assertTrue(
            task_prompt.startswith("[[__SUBAGENT__:SKILL:ticket:"),
            f"Task prompt should start with the SKILL canary. Got: "
            f"{task_prompt[:80]!r}",
        )
        self.assertFalse(
            task_prompt.startswith("\n[[__SUBAGENT__:SKILL:"),
            f"Task prompt must NOT start with a newline before the canary. "
            f"Got: {task_prompt[:80]!r}",
        )


class TestInjectCanary(unittest.TestCase):
    """Tests for _inject_canary_into_task_calls function."""

    def _make_response(self, tool_calls):
        """Helper to create a mock response data dict."""
        return {
            "choices": [
                {
                    "message": {
                        "tool_calls": tool_calls
                    }
                }
            ]
        }

    def test_inject_canary_prepends_to_prompt(self):
        """Prepends canary to the prompt argument of task tool calls."""
        canary = f"[[__SUBAGENT__:SKILL:ticket:{_FAKE_SIG}:{_FRESH_TS}]]"
        data = self._make_response([
            {
                "function": {
                    "name": "task",
                    "arguments": json.dumps({"prompt": "Do something"})
                }
            }
        ])

        result = _inject_canary_into_task_calls(data, canary)

        tc = result["choices"][0]["message"]["tool_calls"][0]
        args = json.loads(tc["function"]["arguments"])
        self.assertTrue(args["prompt"].startswith(canary))
        self.assertIn("Do something", args["prompt"])

    def test_inject_canary_preserves_other_tool_calls(self):
        """Injection only modifies \"task\" calls, other tool calls untouched."""
        canary = f"[[__SUBAGENT__:SKILL:ticket:{_FAKE_SIG}:{_FRESH_TS}]]"
        data = self._make_response([
            {
                "function": {
                    "name": "read",
                    "arguments": json.dumps({"path": "/tmp/file.txt"})
                }
            },
            {
                "function": {
                    "name": "task",
                    "arguments": json.dumps({"prompt": "Do task"})
                }
            },
            {
                "function": {
                    "name": "write",
                    "arguments": json.dumps({"content": "hello"})
                }
            }
        ])

        result = _inject_canary_into_task_calls(data, canary)

        tool_calls = result["choices"][0]["message"]["tool_calls"]
        # read should be unchanged
        self.assertEqual(json.loads(tool_calls[0]["function"]["arguments"])["path"],
                         "/tmp/file.txt")
        # task should have canary prepended
        self.assertTrue(
            json.loads(tool_calls[1]["function"]["arguments"])["prompt"].startswith(canary)
        )
        # write should be unchanged
        self.assertEqual(json.loads(tool_calls[2]["function"]["arguments"])["content"],
                         "hello")

    def test_inject_canary_replaces_existing_canary(self):
        """Existing canary is stripped before injection to prevent stacking."""
        # Two distinct sig values so we can tell the old canary from the new one.
        old_sig = "a" * 64
        new_sig = "b" * 64
        old_canary = f"[[__SUBAGENT__:SKILL:ticket:{old_sig}:{_FRESH_TS}]]"
        new_canary = f"[[__SUBAGENT__:SKILL:ticket:{new_sig}:{_FRESH_TS}]]"
        data = self._make_response([
            {
                "function": {
                    "name": "task",
                    "arguments": json.dumps({
                        "prompt": f"{old_canary}\nOld prompt text"
                    })
                }
            }
        ])

        result = _inject_canary_into_task_calls(data, new_canary)

        tc = result["choices"][0]["message"]["tool_calls"][0]
        args = json.loads(tc["function"]["arguments"])
        # Old canary's distinct sig should be gone
        self.assertNotIn(old_sig, args["prompt"])
        # Exactly ONE canary remains (the new one)
        self.assertEqual(args["prompt"].count("[[__SUBAGENT__"), 1)
        # New canary should be present at the start
        self.assertTrue(args["prompt"].startswith(new_canary))
        # Original text should be preserved
        self.assertIn("Old prompt text", args["prompt"])

    def test_inject_canary_handles_empty_tool_calls(self):
        """Handles response with no tool_calls gracefully."""
        canary = f"[[__SUBAGENT__:SKILL:ticket:{_FAKE_SIG}:{_FRESH_TS}]]"
        data = {"choices": [{"message": {}}]}

        result = _inject_canary_into_task_calls(data, canary)

        self.assertEqual(result, data)

    def test_inject_canary_handles_missing_choices(self):
        """Handles response with missing choices gracefully."""
        canary = f"[[__SUBAGENT__:SKILL:ticket:{_FAKE_SIG}:{_FRESH_TS}]]"
        data = {}

        result = _inject_canary_into_task_calls(data, canary)

        self.assertEqual(result, data)

    def test_inject_canary_handles_invalid_json_arguments(self):
        """Handles tool call with invalid JSON arguments gracefully."""
        canary = f"[[__SUBAGENT__:SKILL:ticket:{_FAKE_SIG}:{_FRESH_TS}]]"
        data = self._make_response([
            {
                "function": {
                    "name": "task",
                    "arguments": "not valid json {"
                }
            }
        ])

        result = _inject_canary_into_task_calls(data, canary)

        # Should not crash, returns unchanged
        self.assertEqual(
            result["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"],
            "not valid json {"
        )


class TestTheory1RealWorldCanary(unittest.TestCase):
    """Theory 1: parse_canary fails on ACTUAL canary from TicketSkill.get_canary().

    The fix in commit 2aeca2b added json.JSONDecoder().raw_decode(). Hand-crafted
    tests pass. But does it work for the real canary produced by the production
    code path?
    """

    def setUp(self):
        """Mock SkillRegistry and verify_canary so we can parse production canaries."""
        self.registry_patcher = patch(
            'proxy.skills.skill_registry.SkillRegistry',
            return_value=_mock_skill_registry(),
        )
        self.registry_patcher.start()
        self.verify_patcher = patch(
            'proxy.canary.verify_canary',
            side_effect=lambda token, skill=None, **kwargs: _make_pc(token.split(":", 1)[0]),
        )
        self.verify_patcher.start()

    def tearDown(self):
        self.registry_patcher.stop()
        self.verify_patcher.stop()

    def _make_ctx(self) -> "SkillContext":
        """Build a realistic SkillContext with session_id, agent_id, messages."""
        from proxy.skills.skill_registry import SkillContext
        return SkillContext(
            session_id="sess_theory1_test_abc123",
            agent_id="agent_theory1_test_xyz789",
            messages=[
                {"role": "user", "content": "^^ticket fix the auth bug"},
                {"role": "assistant", "content": "Let me plan that."},
            ],
            active_workspace="default",
            client_hint="test",
        )

    def _make_mock_tg(self, tasks: list, *, available: bool = True, store_status=None):
        """Build a mock TaskGraph returning the given tasks."""
        mock_tg = MagicMock()
        mock_tg.is_available.return_value = available
        mock_tg.get_actionable_tasks.return_value = tasks
        if store_status is None:
            store_status = {"available": available, "status": "connected" if available else "unavailable"}
        mock_tg.get_store_status.return_value = store_status
        return mock_tg

    def test_theory1_production_canary_empty_tasks(self):
        """TicketSkill.get_canary() with NO tasks → canary should parse to skill_name='ticket'."""
        from skills.ticket_skill import TicketSkill

        skill = TicketSkill()
        ctx = self._make_ctx()
        mock_tg = self._make_mock_tg(tasks=[])

        with patch("skills.ticket_skill._get_task_graph_singleton", return_value=mock_tg):
            canary = skill.get_canary(ctx)

        print(f"\n[THEORY1] Empty tasks canary: {canary!r}")
        p = parse_canary(canary)
        skill_name = p.skill_name if (p and p.verified) else None
        state = p.state if p else None
        print(f"[THEORY1] parse_canary returned: skill_name={skill_name!r}, state_keys={list(state.keys()) if state else None}")
        self.assertEqual(
            skill_name, "ticket",
            f"BUG: parse_canary returned skill_name={skill_name!r} for production canary {canary!r}"
        )

    def test_theory1_production_canary_realistic_tasks(self):
        """TicketSkill.get_canary() with REALISTIC task data → canary should parse."""
        from skills.ticket_skill import TicketSkill

        realistic_tasks = [
            {
                "short_hash": "a1b2c3d4",
                "name": "Fix login bug",
                "priority": "high",
                "status": "in_progress",
                "summary": "User cannot login with valid credentials",
                "tag": "auth",
                "blockers": [],
                "is_blocked": False,
                "blocked_count": 0,
                "parent_short_hash": None,
                "parent_name": None,
            },
            {
                "short_hash": "e5f6g7h8",
                "name": "Add logout button",
                "priority": "medium",
                "status": "pending",
                "summary": "Add UI button for logout",
                "tag": "ui",
                "blockers": [],
                "is_blocked": False,
                "blocked_count": 0,
                "parent_short_hash": "a1b2c3d4",
                "parent_name": "Fix login bug",
            },
        ]

        skill = TicketSkill()
        ctx = self._make_ctx()
        mock_tg = self._make_mock_tg(tasks=realistic_tasks)

        with patch("skills.ticket_skill._get_task_graph_singleton", return_value=mock_tg):
            canary = skill.get_canary(ctx)

        print(f"\n[THEORY1] Realistic tasks canary: {canary!r}")
        p = parse_canary(canary)
        skill_name = p.skill_name if (p and p.verified) else None
        state = p.state if p else None
        print(f"[THEORY1] parse_canary returned: skill_name={skill_name!r}, "
              f"state_tasks_count={len(state.get('tasks', [])) if state else 0}")
        self.assertEqual(
            skill_name, "ticket",
            f"BUG: parse_canary returned skill_name={skill_name!r} for production canary {canary!r}"
        )
        self.assertIsNotNone(state, "state should not be None")
        self.assertEqual(len(state["tasks"]), 2, "should preserve both tasks")

    def test_theory1_production_canary_with_double_brackets_in_name(self):
        """Task name containing ']]' must not break the canary parser."""
        from skills.ticket_skill import TicketSkill

        edge_case_tasks = [
            {
                "short_hash": "aabbccdd",
                "name": "Fix [[wiki]] link in docs",  # contains "]]"
                "priority": "low",
                "status": "pending",
                "summary": "Standard summary",
                "tag": "docs",
                "blockers": [],
                "is_blocked": False,
                "blocked_count": 0,
                "parent_short_hash": None,
                "parent_name": None,
            },
        ]

        skill = TicketSkill()
        ctx = self._make_ctx()
        mock_tg = self._make_mock_tg(tasks=edge_case_tasks)

        with patch("skills.ticket_skill._get_task_graph_singleton", return_value=mock_tg):
            canary = skill.get_canary(ctx)

        print(f"\n[THEORY1] Edge-case canary (]] in name): {canary!r}")
        p = parse_canary(canary)
        skill_name = p.skill_name if (p and p.verified) else None
        print(f"[THEORY1] parse_canary returned: skill_name={skill_name!r}")
        self.assertEqual(
            skill_name, "ticket",
            f"BUG: ']]' in task name broke canary parser. Canary: {canary!r}"
        )

    def test_theory1_production_canary_with_pipe_in_summary(self):
        """Task summary containing '|' must not break the canary parser."""
        from skills.ticket_skill import TicketSkill

        edge_case_tasks = [
            {
                "short_hash": "11223344",
                "name": "Add config option",
                "priority": "medium",
                "status": "pending",
                "summary": "Use the | delimiter in the file",  # contains "|"
                "tag": "config",
                "blockers": [],
                "is_blocked": False,
                "blocked_count": 0,
                "parent_short_hash": None,
                "parent_name": None,
            },
        ]

        skill = TicketSkill()
        ctx = self._make_ctx()
        mock_tg = self._make_mock_tg(tasks=edge_case_tasks)

        with patch("skills.ticket_skill._get_task_graph_singleton", return_value=mock_tg):
            canary = skill.get_canary(ctx)

        print(f"\n[THEORY1] Edge-case canary (| in summary): {canary!r}")
        p = parse_canary(canary)
        skill_name = p.skill_name if (p and p.verified) else None
        print(f"[THEORY1] parse_canary returned: skill_name={skill_name!r}")
        self.assertEqual(
            skill_name, "ticket",
            f"BUG: '|' in task summary broke canary parser. Canary: {canary!r}"
        )

    def test_theory1_production_canary_unavailable_store(self):
        """When task store is unavailable, get_canary should still produce a parseable canary."""
        from skills.ticket_skill import TicketSkill

        skill = TicketSkill()
        ctx = self._make_ctx()
        mock_tg = self._make_mock_tg(tasks=[], available=False)

        with patch("skills.ticket_skill._get_task_graph_singleton", return_value=mock_tg):
            canary = skill.get_canary(ctx)

        print(f"\n[THEORY1] Unavailable-store canary: {canary!r}")
        p = parse_canary(canary)
        skill_name = p.skill_name if (p and p.verified) else None
        state = p.state if p else None
        print(f"[THEORY1] parse_canary returned: skill_name={skill_name!r}, state={state!r}")
        self.assertEqual(
            skill_name, "ticket",
            f"BUG: unavailable-store canary did not parse. Canary: {canary!r}"
        )


class TestTheory4ResolveModeInheritedSkill(unittest.TestCase):
    """Theory 4: resolve_mode() only activates the ticket skill when
    subagent_ticket_mode=True. If subagent_ticket_mode is False but a
    canary in the first user message still names the inherited skill
    ("ticket"), the resolver must still activate the skill and return
    Mode.SKILL_SUBAGENT.

    Result of investigation: Theory 4 is INCORRECT. The current code in
    src/proxy/mode.py:resolve_mode() handles ctx.inherited_skill
    INDEPENDENTLY of ctx.subagent_ticket_mode (line 131: `if _has_inherited_skill`
    branch runs even when subagent_ticket_mode is False, and the elif on
    line 154 only runs when inherited_skill is unset). _activate_skill is
    invoked, is_ticket is set to True, and the returned Mode is
    SKILL_SUBAGENT (or SKILL for main). apply_mode() then loads the active
    skill via agent.snapshot_mode() and prepare_tooling_payload injects
    proxy_ticket. These tests pin that behaviour so future refactors cannot
    regress it.
    """

    def setUp(self):
        """Mock SkillRegistry + _activate_skill so the test is hermetic."""
        from skills.ticket_skill import TicketSkill
        real_skill = TicketSkill()

        mock_registry = MagicMock()
        mock_registry.get.return_value = real_skill

        self.registry_patcher = patch(
            "proxy.skills.skill_registry.SkillRegistry",
            return_value=mock_registry,
        )
        self.registry_patcher.start()

        from proxy.skills.skill_definition import ModeState

        def _activate_side_effect(ag, skill, skill_ctx, ctx=None):
            import time as _t
            mode_state = ModeState(
                name=skill.name,
                activated_at=_t.time(),
                ttl_seconds=300,
                allowed_tool_names=frozenset(
                    {"read", "write", "bash", "proxy_ticket", "proxy_work"}
                ),
                skill_state={},
            )
            ag.set_active_mode(mode_state)
            if ctx is not None:
                ctx._skill_activated = True

        self.activate_patcher = patch(
            "proxy.agent.agent._inherit_skill",
            side_effect=_activate_side_effect,
        )
        self.activate_patcher.start()

    def tearDown(self):
        self.registry_patcher.stop()
        self.activate_patcher.stop()

    def _build_subagent_ctx(self):
        """Build a Theory 4 context: is_subagent=True, subagent_ticket_mode=False,
        inherited_skill='ticket' (set by the canary in messages).
        """
        ctx = MagicMock()
        ctx.session_id = "sess_theory4_canary"
        ctx.is_subagent = True
        ctx.subagent_ticket_mode = False      # <-- the bug scenario
        ctx.inherited_skill = "ticket"        # <-- canary-derived
        ctx.workspace_name = ""
        ctx.seeded_tasks = []
        ctx._skill_activated = False
        ctx._mode = None
        ctx.model_cfg = MagicMock()
        ctx.body = {
            "messages": [
                {"role": "user", "content": "do a subagent task"},
            ],
            "tools": [
                {"type": "function", "function": {
                    "name": "read", "description": "r",
                    "parameters": {"type": "object", "properties": {}}
                }},
                {"type": "function", "function": {
                    "name": "bash", "description": "b",
                    "parameters": {"type": "object", "properties": {}}
                }},
            ],
        }
        ctx.messages = ctx.body["messages"]
        return ctx

    def _build_agent(self):
        agent = MagicMock()
        agent.agent_id = "agent_theory4_canary"
        agent._mode_lock = threading.RLock()
        agent.active_mode = None
        agent.snapshot_mode.return_value = None
        agent.snapshot_mode_and_ticket_ts.return_value = (None, None)
        agent.set_active_mode = MagicMock(return_value=True)
        agent.clear_active_mode = MagicMock(return_value=True)
        return agent

    def test_theory4_resolve_mode_skill_subagent(self):
        """is_subagent + inherited_skill='ticket' + subagent_ticket_mode=False
        → resolve_mode returns Mode.SKILL_SUBAGENT and the skill is activated.
        """
        from proxy.mode import resolve_mode
        ctx = self._build_subagent_ctx()
        agent = self._build_agent()

        mode = resolve_mode(ctx, agent)

        self.assertEqual(
            mode, Mode.SKILL_SUBAGENT,
            f"BUG (Theory 4): resolve_mode() returned {mode!r}, expected "
            f"Mode.SKILL_SUBAGENT when inherited_skill='ticket' and "
            f"is_subagent=True, regardless of subagent_ticket_mode",
        )
        # The skill must be activated on the agent — proves the
        # inherited_skill branch ran in resolve_mode.
        self.assertTrue(
            agent.set_active_mode.called,
            "BUG (Theory 4): _inherit_skill was not called despite "
            "inherited_skill='ticket' being set. resolve_mode() only "
            "checks subagent_ticket_mode?",
        )
        # Inspect the ModeState that was set: name must be 'ticket'.
        call_args = agent.set_active_mode.call_args
        if call_args is not None:
            mode_state = call_args[0][0] if call_args[0] else call_args.kwargs.get("mode_state")
            self.assertIsNotNone(mode_state, "_inherit_skill called without a ModeState argument")
            self.assertEqual(
                mode_state.name, "ticket",
                f"BUG (Theory 4): activated skill name is {mode_state.name!r}, "
                f"expected 'ticket'",
            )

    def test_theory4_resolve_mode_skill_subagent_apply_injects_proxy_ticket(self):
        """End-to-end: resolve_mode() then apply_mode() injects decomposed
        proxy_ticket_* tools into body['tools'] for a sub-agent with
        inherited_skill='ticket' and subagent_ticket_mode=False.
        """
        from proxy.mode import resolve_mode, apply_mode
        from unittest.mock import patch as _patch

        ctx = self._build_subagent_ctx()
        agent = self._build_agent()

        mode = resolve_mode(ctx, agent)
        self.assertEqual(mode, Mode.SKILL_SUBAGENT)

        # Now apply_mode() must observe the activated skill via
        # agent.snapshot_mode() returning the ModeState.
        # Re-route snapshot_mode to return the active_mode we just set.
        with _patch.object(agent, "snapshot_mode", return_value=agent.active_mode), \
             _patch("proxy.context.select_context", return_value=None), \
             _patch("proxy.tool_cache.cache_tool_schema"), \
             _patch("proxy.tool_cache.get_cached_tool_schema", return_value=None):
            apply_mode(ctx, agent, ctx.body, mode)

        tool_names = [
            t.get("function", {}).get("name", "")
            for t in ctx.body.get("tools", [])
        ]
        # Slice 8b: decomposed proxy_ticket_* tools (ticket SUB sees all 10).
        self.assertIn(
            "proxy_ticket_list_tasks", tool_names,
            f"BUG (Theory 4): apply_mode() did NOT inject "
            f"proxy_ticket_list_tasks for a sub-agent with "
            f"inherited_skill='ticket' (subagent_ticket_mode=False). "
            f"tools={tool_names}",
        )
        self.assertIn(
            "proxy_ticket_create_tasks", tool_names,
            f"BUG (Theory 4): apply_mode() did NOT inject "
            f"proxy_ticket_create_tasks for a sub-agent with "
            f"inherited_skill='ticket' (subagent_ticket_mode=False). "
            f"tools={tool_names}",
        )
        self.assertNotIn(
            "bash", tool_names,
            f"BUG (Theory 4): bash should be filtered out in ticket mode "
            f"but it is present. tools={tool_names}",
        )


class TestCanaryHmacSecurity(unittest.TestCase):
    """Pin the HMAC security guarantees for signed canaries.

    These tests set ``PROXY_SESSION_SECRET`` so the production HMAC path is
    exercised (not the dev no-secret fallback). They verify:
      1. The signature is a full 256-bit (64 hex char) digest, not truncated.
      2. Tampered payloads and flipped signatures are rejected.
      3. A sign→verify roundtrip recovers the correct skill name.
    """

    _SECRET = "test-hmac-secret-key-for-canary-security-tests"

    def setUp(self):
        import os
        from proxy import canary as canary_mod

        # Save and clear any leaked PROXY_SESSION_SECRET from prior tests.
        self._had_proxy_session_secret = "PROXY_SESSION_SECRET" in os.environ
        self._original_proxy_session_secret = os.environ.pop(
            "PROXY_SESSION_SECRET", None,
        )
        # Set a known secret so the production HMAC path runs.
        os.environ["PROXY_SESSION_SECRET"] = self._SECRET
        self._original_secret = canary_mod._SESSION_SECRET
        canary_mod._SESSION_SECRET = self._SECRET
        # Ensure PROXY_ENV is dev so any fail-closed branch doesn't trip.
        self._original_proxy_env = os.environ.get("PROXY_ENV")
        os.environ.setdefault("PROXY_ENV", "dev")

    def tearDown(self):
        import os
        from proxy import canary as canary_mod

        canary_mod._SESSION_SECRET = self._original_secret
        os.environ.pop("PROXY_SESSION_SECRET", None)
        if self._had_proxy_session_secret:
            os.environ["PROXY_SESSION_SECRET"] = self._original_proxy_session_secret
        if self._original_proxy_env is not None:
            os.environ["PROXY_ENV"] = self._original_proxy_env

    def test_canary_hmac_full_256_bit_no_truncation(self):
        """The signature portion is exactly 64 hex chars (256-bit), not 12."""
        from proxy.canary import sign_canary

        sig_ts = sign_canary("ticket")
        sig, _ts = sig_ts.split(":", 1)

        self.assertEqual(len(sig), 64, f"sig must be 64 hex chars, got {len(sig)}: {sig!r}")
        self.assertNotEqual(len(sig), 12, "sig must NOT be the old 12-char truncated format")
        self.assertTrue(
            all(c in "0123456789abcdef" for c in sig),
            f"sig must be lowercase hex, got {sig!r}",
        )

    def test_canary_hmac_tampered_payload_rejected(self):
        """Tampering with the payload (skill name) keeps the sig but must fail
        verification. Flipping a bit in the signature itself must also fail.
        """
        from proxy.canary import sign_canary, verify_canary

        sig_ts = sign_canary("ticket")
        sig, ts = sig_ts.split(":", 1)

        # Tamper payload: change skill name, keep original sig.
        tampered_payload = f"workspace:{sig}:{ts}"
        self.assertIsNone(verify_canary(tampered_payload), "tampered payload must be rejected")

        # Flip a bit in the signature itself.
        flipped_char = "0" if sig[0] != "0" else "1"
        flipped_sig = flipped_char + sig[1:]
        flipped = f"ticket:{flipped_sig}:{ts}"
        self.assertIsNone(verify_canary(flipped), "flipped signature must be rejected")

    def test_canary_hmac_sign_and_verify_roundtrip(self):
        """Sign a canary, then verify it — the ParsedCanary must recover the
        correct skill name.
        """
        from proxy.canary import sign_canary, verify_canary

        sig_ts = sign_canary("ticket")
        token = f"ticket:{sig_ts}"
        result = verify_canary(token)

        self.assertIsNotNone(result, "roundtrip verification must succeed")
        self.assertEqual(result.skill, "ticket")
        self.assertFalse(result.is_dev_fallback, "must use the production HMAC path, not dev fallback")
        self.assertEqual(len(result.sig), 64)


class TestCanaryStateAndSessionBinding(unittest.TestCase):
    """B-3 regression tests: HMAC must bind state to prevent mutation.

    B-3 (state mutation): the legacy HMAC input was just
    ``f"{skill_name}:{timestamp}"`` so the JSON state appended AFTER the
    signed portion could be mutated (e.g. swapping ``workspace_name`` or
    ``task_hash``) without invalidating the signature.

    Note (BLOCKER fix): ``session_id`` is NOT bound to the HMAC. Parent
    and sub-agent have different session_ids by design (the sub-agent's
    first user message is the parent's tool-call prompt, which hashes to
    a different session_id), so binding to either one would break the
    legitimate production flow. Replay protection is via the 5-minute
    freshness window only. The four pinned invariants are:

      1. Mutating state_str fails verification (B-3).
      2. Verifying with the correct state succeeds.
      3. Parent and sub-agent with DIFFERENT session_ids can still verify
         the same canary (BLOCKER).
      4. Legacy canaries (no state) still verify (backward compat).
    """

    _SECRET = "test-hmac-secret-key-for-canary-binding-tests"

    def setUp(self):
        import os
        from proxy import canary as canary_mod

        # Save and clear any leaked PROXY_SESSION_SECRET from prior tests.
        self._had_proxy_session_secret = "PROXY_SESSION_SECRET" in os.environ
        self._original_proxy_session_secret = os.environ.pop(
            "PROXY_SESSION_SECRET", None,
        )
        os.environ["PROXY_SESSION_SECRET"] = self._SECRET
        self._original_secret = canary_mod._SESSION_SECRET
        canary_mod._SESSION_SECRET = self._SECRET
        self._original_proxy_env = os.environ.get("PROXY_ENV")
        os.environ.setdefault("PROXY_ENV", "dev")

    def tearDown(self):
        import os
        from proxy import canary as canary_mod

        canary_mod._SESSION_SECRET = self._original_secret
        os.environ.pop("PROXY_SESSION_SECRET", None)
        if self._had_proxy_session_secret:
            os.environ["PROXY_SESSION_SECRET"] = self._original_proxy_session_secret
        if self._original_proxy_env is not None:
            os.environ["PROXY_ENV"] = self._original_proxy_env

    # ----------------------------------------------------------------
    # Sign-and-verify helpers
    # ----------------------------------------------------------------

    def _sign_state_bound(self, skill_name, state_dict):
        """Sign a state-bound canary (production path: state_str is included)."""
        from proxy.canary import sign_canary
        state_str = ""
        if state_dict:
            state_str = f"|{json.dumps(state_dict, sort_keys=True)}"
        return sign_canary(skill_name, state_str=state_str)

    def _token(self, skill_name, sig_ts):
        return f"{skill_name}:{sig_ts}"

    # ----------------------------------------------------------------
    # B-3 — state mutation must fail verification
    # ----------------------------------------------------------------

    def test_state_mutation_fails_verification(self):
        """Sign a canary with state A, verify with mutated state B → must fail."""
        from proxy.canary import verify_canary

        state = {"workspace_name": "ws_alpha", "task_hash": "abc123"}
        sig_ts = self._sign_state_bound("work", state)
        token = self._token("work", sig_ts)

        mutated = {"workspace_name": "ws_beta", "task_hash": "abc123"}
        state_str = f"|{json.dumps(mutated, sort_keys=True)}"

        result = verify_canary(token, state_str=state_str)
        self.assertIsNone(
            result,
            f"BUG (B-3): mutated state must fail verification. "
            f"expected=None, got={result!r}",
        )

    def test_state_mutation_change_workspace_only(self):
        """Only changing workspace_name must fail (the simple B-3 attack)."""
        from proxy.canary import verify_canary

        state = {"workspace_name": "ws_alpha", "task_hash": "abc123"}
        sig_ts = self._sign_state_bound("work", state)
        token = self._token("work", sig_ts)

        mutated = {"workspace_name": "ws_OTHER", "task_hash": "abc123"}
        state_str = f"|{json.dumps(mutated, sort_keys=True)}"

        self.assertIsNone(verify_canary(token, state_str=state_str))

    def test_state_mutation_change_task_hash_only(self):
        """Only changing task_hash must fail (B-3)."""
        from proxy.canary import verify_canary

        state = {"workspace_name": "ws_alpha", "task_hash": "abc123"}
        sig_ts = self._sign_state_bound("work", state)
        token = self._token("work", sig_ts)

        mutated = {"workspace_name": "ws_alpha", "task_hash": "OTHER"}
        state_str = f"|{json.dumps(mutated, sort_keys=True)}"

        self.assertIsNone(verify_canary(token, state_str=state_str))

    # ----------------------------------------------------------------
    # BLOCKER fix: parent and sub-agent have different session_ids by
    # design; canary MUST still verify on both sides.
    # ----------------------------------------------------------------

    def test_subagent_canary_with_different_session_ids_still_verifies(self):
        """BLOCKER fix: parent signs with session_A, sub-agent verifies
        with session_B (different content → different session_id), canary
        MUST still verify. This is the legitimate production flow.

        Regression assertion: the previous B-4 session-binding code
        rejected this case and the sub-agent lost skill/workspace/task
        inheritance. Replay protection is via the 5-minute freshness
        window only — session_id is NOT bound to the HMAC.
        """
        from proxy.canary import sign_canary, verify_canary

        parent_sid = "sess_parent_aaa"
        sub_sid = "sess_sub_bbb"

        # Sign as parent with state (production path).
        state = {"workspace_name": "ws_alpha", "task_hash": "abc12345"}
        state_str = f"|{json.dumps(state, sort_keys=True)}"
        sig_ts = sign_canary("work", state_str=state_str)
        token = f"work:{sig_ts}"

        # Verify as parent (same session_id).
        result_parent = verify_canary(token, state_str=state_str)
        self.assertIsNotNone(
            result_parent,
            f"parent-side verify must succeed with matching state",
        )

        # Verify as sub-agent (DIFFERENT session_id — the BLOCKER case).
        result_sub = verify_canary(token, state_str=state_str)
        self.assertIsNotNone(
            result_sub,
            f"BLOCKER: sub-agent must verify the parent's canary despite "
            f"different session_id (parent={parent_sid!r}, sub={sub_sid!r}). "
            f"expected=ParsedCanary, got=None",
        )
        self.assertEqual(result_sub.skill, "work")
        self.assertFalse(result_sub.is_dev_fallback)

    def test_workspace_canary_different_session_ids_still_verifies(self):
        """BLOCKER fix on the workspace branch: a workspace canary minted
        by the parent MUST still verify on the sub-agent side regardless
        of session_id mismatch."""
        from proxy.canary import sign_canary, verify_canary

        sig_ts = sign_canary("workspace:admin")
        token = f"workspace:admin:{sig_ts}"

        result = verify_canary(token, state_str="")
        self.assertIsNotNone(
            result,
            "workspace canary must verify regardless of session_id",
        )
        self.assertEqual(result.skill, "workspace:admin")

    # ----------------------------------------------------------------
    # Combined B-3 + BLOCKER
    # ----------------------------------------------------------------

    def test_state_mutation_with_correct_state_fails(self):
        """Mutated state (with no session binding to rely on) → must fail (B-3)."""
        from proxy.canary import verify_canary

        state = {"workspace_name": "ws_alpha", "task_hash": "abc123"}
        sig_ts = self._sign_state_bound("work", state)
        token = self._token("work", sig_ts)

        mutated = {"workspace_name": "ws_beta", "task_hash": "abc123"}
        state_str = f"|{json.dumps(mutated, sort_keys=True)}"

        self.assertIsNone(verify_canary(token, state_str=state_str))

    def test_state_match_succeeds(self):
        """Correct state → verification must succeed (no session_id dependency)."""
        from proxy.canary import verify_canary

        state = {"workspace_name": "ws_alpha", "task_hash": "abc123"}
        sig_ts = self._sign_state_bound("work", state)
        state_str = f"|{json.dumps(state, sort_keys=True)}"
        token = self._token("work", sig_ts)

        result = verify_canary(token, state_str=state_str)
        self.assertIsNotNone(result)
        self.assertEqual(result.skill, "work")
        self.assertFalse(result.is_dev_fallback)

    def test_roundtrip_parse_canary_state_bound(self):
        """End-to-end: sign work canary with state, parse back, verify it roundtrips.

        Verifies the integration between parse_canary (which reconstructs
        state_str from the parsed JSON dict) and verify_canary.
        """
        from proxy.canary import sign_canary

        state = {"workspace_name": "ws_alpha", "task_hash": "abc123"}
        state_str = f"|{json.dumps(state, sort_keys=True)}"
        sig = sign_canary("work", state_str=state_str)
        canary = f"[[__SUBAGENT__:SKILL:work:{sig}{state_str}]]"

        # Register a mock work skill so parse_canary's registry check passes.
        mock_skill = MagicMock()
        mock_skill.name = "work"
        mock_registry = MagicMock()
        mock_registry.get.return_value = mock_skill

        with patch("proxy.skills.skill_registry.SkillRegistry", return_value=mock_registry):
            p = parse_canary(canary)

        self.assertEqual(p.kind, "skill")
        self.assertTrue(p.verified)
        self.assertEqual(p.skill_name, "work")
        self.assertEqual(p.state, state)

    # ----------------------------------------------------------------
    # Backward compat — legacy canaries still verify
    # ----------------------------------------------------------------

    def test_legacy_canary_no_state_still_verifies(self):
        """Canary signed without state_str → verify with empty state still
        succeeds (backward compat for legacy producers).

        This is the soft-rollout window: old signers produce a legacy HMAC
        over ``f"{skill_name}:{timestamp}"``, and old verifiers (which
        don't pass state_str) still accept them. The new code preserves
        this behaviour.
        """
        from proxy.canary import sign_canary, verify_canary

        sig_ts = sign_canary("ticket")
        token = f"ticket:{sig_ts}"
        result = verify_canary(token)
        self.assertIsNotNone(result)
        self.assertEqual(result.skill, "ticket")

    def test_legacy_canary_no_state_with_ignored_session_id_still_verifies(self):
        """A legacy canary (no state) MUST verify even if the caller passes
        a session_id — the session_id kwarg is now ignored (BLOCKER fix).

        Pinning the BLOCKER-fix contract: the previous verifier rejected
        legacy canaries when the caller passed session_id (because the
        HMAC inputs differed). Now session_id is ignored so legacy canaries
        continue to verify under the new contract.
        """
        from proxy.canary import sign_canary, verify_canary

        sig_ts = sign_canary("ticket")
        token = f"ticket:{sig_ts}"
        result = verify_canary(token, state_str="", session_id="sess_alice")
        self.assertIsNotNone(
            result,
            "legacy canary must verify even with session_id passed "
            "(session_id is now an IGNORED kwarg)",
        )
        self.assertEqual(result.skill, "ticket")

    # ----------------------------------------------------------------
    # Workspace canary backward compat — workspace_name is embedded in
    # skill_name, so state_str is empty; session_id is now ignored (BLOCKER).
    # ----------------------------------------------------------------

    def test_workspace_canary_no_state_still_verifies(self):
        """Workspace canary with no state_str verifies regardless of session_id.

        Pins the BLOCKER fix: workspace canaries do not need session
        binding — the 5-minute freshness window is the replay protection.
        """
        from proxy.canary import sign_canary, verify_canary

        sig_ts = sign_canary("workspace:admin")
        token = f"workspace:admin:{sig_ts}"
        result = verify_canary(token)
        self.assertIsNotNone(result)
        self.assertEqual(result.skill, "workspace:admin")

    # ----------------------------------------------------------------
    # Determinism — sort_keys=True required on both sides
    # ----------------------------------------------------------------

    def test_state_json_must_use_sort_keys(self):
        """If state is {"a":1, "b":2}, both sign and verify must use sort_keys=True
        so the HMAC inputs match regardless of dict insertion order.

        We exercise this by signing with one insertion order and verifying
        with the same dict but inserted in reverse — both paths must
        re-serialise with sort_keys=True for the HMACs to match.
        """
        from proxy.canary import sign_canary, verify_canary

        state_a_to_b = {"a": 1, "b": 2}
        state_b_to_a = {"b": 2, "a": 1}

        state_str_a = f"|{json.dumps(state_a_to_b, sort_keys=True)}"
        sig_ts = sign_canary("work", state_str=state_str_a)

        # Verify using the dict in the REVERSE order — re-serialised with
        # sort_keys=True the bytes are identical, so the HMAC matches.
        state_str_b = f"|{json.dumps(state_b_to_a, sort_keys=True)}"
        token = f"work:{sig_ts}"
        result = verify_canary(token, state_str=state_str_b)
        self.assertIsNotNone(
            result,
            "sort_keys=True must canonicalise dict insertion order so HMAC matches",
        )

    def test_state_json_without_sort_keys_mismatches(self):
        """A state re-serialised WITHOUT sort_keys=True would mismatch a sign
        done with sort_keys=True — pinning the determinism invariant.

        This test exercises a dict whose insertion order would change the
        unsorted serialisation: ``{"b": 2, "a": 1}`` (insertion order
        produces ``{"b": 2, "a": 1}`` unsorted, ``{"a": 1, "b": 2}``
        sorted). Signing with one order and verifying with the other must
        NOT match — pinning the contract that BOTH sides use sort_keys=True.
        """
        from proxy.canary import sign_canary, verify_canary

        # Build two dicts that are EQUAL by content but DIFFERENT by insertion order.
        state_unsorted = {"b": 2, "a": 1}  # unsorted: json.dumps → '{"b": 2, "a": 1}'
        state_sorted = {"a": 1, "b": 2}    # sorted:   json.dumps → '{"a": 1, "b": 2}'

        state_str_unsorted = f"|{json.dumps(state_unsorted, sort_keys=False)}"
        state_str_sorted = f"|{json.dumps(state_sorted, sort_keys=True)}"

        # Sanity: the two strings are byte-different (the invariant we're testing).
        self.assertNotEqual(
            state_str_unsorted, state_str_sorted,
            "test setup error: insertion order did not produce different byte sequences",
        )

        # Sign with the unsorted bytes (simulates a producer that didn't use sort_keys).
        sig_ts = sign_canary("work", state_str=state_str_unsorted)
        token = f"work:{sig_ts}"

        # Verify with the sorted bytes (the new canonical production path).
        result = verify_canary(token, state_str=state_str_sorted)
        self.assertIsNone(
            result,
            "sign with unsorted JSON + verify with sorted JSON must NOT match — "
            "pins the deterministic canonicalisation contract (sort_keys=True on BOTH sides)",
        )

    # ----------------------------------------------------------------
    # sign_canary vs verify_canary drift detection
    # ----------------------------------------------------------------

    def test_sign_canary_legacy_message_unchanged(self):
        """Pin that sign_canary() with no state/session produces the legacy
        HMAC input (2-field message). This is the backward-compat window:
        we MUST NOT silently change the canonical input for legacy callers.
        """
        from proxy.canary import sign_canary, verify_canary

        sig_ts_legacy = sign_canary("ticket")
        # Roundtrip via the legacy verify path (no kwargs) — must succeed.
        result = verify_canary(f"ticket:{sig_ts_legacy}")
        self.assertIsNotNone(result)
        self.assertEqual(result.skill, "ticket")

    def test_sign_canary_bound_message_differs_from_legacy(self):
        """sign_canary() with state MUST produce a DIFFERENT signature
        than the legacy path — pinning that the state binding is actually applied.

        Without this invariant, the binding code could be silently bypassed
        (e.g. by always defaulting to the legacy message). session_id is
        no longer part of the HMAC (BLOCKER fix); state_str alone must be
        sufficient to differentiate the bound signature.
        """
        from proxy.canary import sign_canary

        legacy_sig = sign_canary("work")
        bound_sig = sign_canary("work", state_str="|{}")

        legacy_hex, _ = legacy_sig.split(":", 1)
        bound_hex, _ = bound_sig.split(":", 1)
        self.assertNotEqual(
            legacy_hex, bound_hex,
            "bound canary signature must differ from legacy signature — "
            "otherwise the state binding is silently disabled",
        )

    # ----------------------------------------------------------------
    # parse_canary integration — round-trip with binding
    # ----------------------------------------------------------------

    def test_parse_canary_rejects_mutated_state(self):
        """End-to-end: parse a canary with a MUTATED state JSON → unverified.

        Production B-3 attack: attacker captures the canary, changes the
        JSON state suffix (workspace_name / task_hash), and presents the
        mutated canary. The HMAC must reject it.
        """
        from proxy.canary import sign_canary

        state = {"workspace_name": "ws_alpha", "task_hash": "abc123"}
        state_str = f"|{json.dumps(state, sort_keys=True)}"
        sig = sign_canary("work", state_str=state_str)

        # Mutate the JSON state appended AFTER the signature.
        mutated = {"workspace_name": "ws_beta", "task_hash": "abc123"}
        mutated_state_str = f"|{json.dumps(mutated, sort_keys=True)}"
        canary = f"[[__SUBAGENT__:SKILL:work:{sig}{mutated_state_str}]]"

        mock_skill = MagicMock()
        mock_skill.name = "work"
        mock_registry = MagicMock()
        mock_registry.get.return_value = mock_skill

        with patch("proxy.skills.skill_registry.SkillRegistry", return_value=mock_registry):
            p = parse_canary(canary)

        # Mutated state must fail HMAC → not verified, no skill_name.
        self.assertFalse(
            p.verified,
            f"BUG (B-3): parse_canary must reject mutated state. "
            f"expected verified=False, got verified={p.verified!r}",
        )
        self.assertIsNone(p.skill_name)

    def test_parse_canary_accepts_cross_session_canary(self):
        """BLOCKER fix end-to-end: parse_canary must succeed for a
        canary minted in a different session (the production sub-agent
        case where parent and sub-agent have different session_ids).

        This complements the BLOCKER unit test by exercising the full
        canary → parse path. The previous B-4 binding code rejected
        canaries whose session_id mismatched the producer's, which broke
        every sub-agent invocation in production.
        """
        from proxy.canary import sign_canary

        state = {"workspace_name": "ws_alpha", "task_hash": "abc123"}
        state_str = f"|{json.dumps(state, sort_keys=True)}"
        # Sign as parent (parent_sid is irrelevant — session_id is ignored now).
        sig = sign_canary("work", state_str=state_str)
        canary = f"[[__SUBAGENT__:SKILL:work:{sig}{state_str}]]"

        mock_skill = MagicMock()
        mock_skill.name = "work"
        mock_registry = MagicMock()
        mock_registry.get.return_value = mock_skill

        with patch("proxy.skills.skill_registry.SkillRegistry", return_value=mock_registry):
            # Parse in sub-agent context (different session_id would be
            # passed in production; here we just confirm no session_id
            # is required for parse to succeed).
            p = parse_canary(canary)

        self.assertTrue(p.verified)
        self.assertEqual(
            p.skill_name, "work",
            f"BLOCKER: parse_canary must succeed without session_id "
            f"(parent and sub-agent have different session_ids in production)",
        )
        self.assertEqual(p.state, state)


class TestCanaryDiagnosticsLogging(unittest.TestCase):
    """User-facing diagnostic: 3 log lines fire on canary flow.

    These tests pin the user diagnostic so the user can see exactly what
    the proxy is doing at each transformation step:

    - L1 (DEBUG) at `apply_post_process` entry: task_args_raw prompt_len=N prompt=<repr>
    - L2 (DEBUG) in `_inject_canary_into_task_calls`: prompt_before_strip / after_strip / canary / final_prompt
    - L3 (DEBUG) in `detect_subagent_ingress` entry: raw_msg_count / first_content_len / first_content

    H2: L1 and L3 were downgraded from INFO to DEBUG (they emit raw prompt
    content — PII/secrets leak at the production-default INFO level). These
    tests pin the DEBUG level so a regression (bumping back to INFO) fails.
    """

    def test_detect_ingress_logs_no_first_content(self):
        """L3 (detect_ingress first_content=) DEBUG line was REMOVED.

        Replaced with a negative-assertion test that pins the redaction:
        first_user content must NOT appear in any proxy.canary DEBUG record.
        """
        import logging
        secret_marker = "SECRET-MARKER-DETECT-12345"
        body = {"messages": [{"role": "user", "content": secret_marker + " extra"}]}
        with self.assertLogs("proxy.canary", level=logging.DEBUG) as cm:
            detect_subagent_ingress(body)
        msgs = [r.getMessage() for r in cm.records]
        joined = "\n".join(msgs)
        self.assertNotIn(
            secret_marker, joined,
            f"detect_ingress DEBUG log must not include first_user content. "
            f"Got: {msgs!r}",
        )

    def test_canary_inject_logs_no_prompt_content(self):
        """L2 must NOT include raw prompt/canary content in any log record.

        Replaces the previous positive-assertion test that pinned the
        prompt-disclosure DEBUG line. Per redaction policy, prompt content,
        the canary string, and signature fragments must NEVER appear in
        proxy.canary DEBUG records (or any other level).
        """
        import logging
        canary = "[[__SUBAGENT__:SKILL:ticket:deadbeefcafebabe1234:1700000000]]\n"
        secret_marker = "SECRET-MARKER-XYZ-12345"
        data = {
            "choices": [{
                "message": {
                    "role": "assistant",
                    "tool_calls": [{
                        "function": {
                            "name": "task",
                            "arguments": json.dumps({"prompt": secret_marker}),
                        }
                    }],
                },
            }],
        }
        with self.assertLogs("proxy.canary", level="DEBUG") as cm:
            _inject_canary_into_task_calls(data, canary)
        msgs = [r.getMessage() for r in cm.records]
        joined = "\n".join(msgs)
        self.assertNotIn(
            secret_marker, joined,
            f"L2 must not log the prompt content. Got: {msgs!r}",
        )
        self.assertNotIn(
            "[[__SUBAGENT__", joined,
            f"L2 must not log the canary marker. Got: {msgs!r}",
        )

    def test_detect_ingress_logs_first_content_length_only(self):
        """L3 (detect_ingress) redacted DEBUG line carries only structural
        fields (raw_msg_count, first_content_len) and NEVER the prompt
        content. Pin the structural field on the new redacted log line.
        """
        import logging
        long_text = "x" * 5000
        with self.assertLogs("proxy.canary", level=logging.DEBUG) as cm:
            detect_subagent_ingress({"messages": [{"role": "user", "content": long_text}]})
        msgs = [r.getMessage() for r in cm.records]
        matched = [m for m in msgs if "detect_ingress" in m and "first_content_len=" in m]
        self.assertEqual(
            len(matched), 1,
            f"Expected exactly 1 redacted detect_ingress DEBUG log, got "
            f"{len(matched)}: {matched!r}",
        )
        # The redacted line carries the length (5000) but NOT the content.
        self.assertIn(
            "first_content_len=5000", matched[0],
            f"Redacted log must carry the length. Got: {matched[0]!r}",
        )
        self.assertNotIn(
            "x" * 100, matched[0],
            f"Redacted log must not carry the prompt content. Got: {matched[0]!r}",
        )

    def test_detect_ingress_not_logged_at_info(self):
        """H2: L3 (detect_ingress first_content=) must NOT fire at INFO level.

        PII/secrets leak prevention — pinning the DEBUG level so a regression
        (someone bumping back to logger.info) fails this test.
        """
        import logging
        long_text = "x" * 5000
        with self.assertLogs("proxy.canary", level=logging.DEBUG) as cm:
            detect_subagent_ingress({"messages": [{"role": "user", "content": long_text}]})
        info_records = [
            r for r in cm.records
            if r.levelno == logging.INFO
            and "detect_ingress" in r.getMessage()
            and "first_content=" in r.getMessage()
        ]
        self.assertEqual(
            info_records, [],
            f"L3 must be DEBUG not INFO — found INFO records: {[r.getMessage() for r in info_records]}",
        )

    def test_task_args_raw_logs_no_prompt_content(self):
        """task_args_raw prompt= DEBUG log was REMOVED. Replaced with a
        negative-assertion test that pins the redaction: prompt content,
        canary markers, and signature fragments must NEVER appear in any
        proxy.canary DEBUG record produced by _inject_canary_into_task_calls.
        """
        import logging
        from proxy.canary import _inject_canary_into_task_calls

        secret_marker = "SECRET-MARKER-MODE-12345"
        long_prompt = "X" * 4096 + secret_marker + "X" * 100
        canary = "[[__SUBAGENT__:SKILL:ticket:deadbeefcafebabe1234567890abcdef:1700000000]]\n"
        data = {
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [{
                        "id": "call_redact",
                        "type": "function",
                        "function": {
                            "name": "task",
                            "arguments": json.dumps({
                                "description": "test",
                                "prompt": long_prompt,
                            }),
                        },
                    }],
                },
            }],
        }
        with self.assertLogs("proxy.canary", level=logging.DEBUG) as cm:
            _inject_canary_into_task_calls(data, canary)
        msgs = [r.getMessage() for r in cm.records]
        joined = "\n".join(msgs)
        # Markers that must NEVER appear in any log record at any level:
        for marker in (
            secret_marker,
            "[[__SUBAGENT__",
            "task_args_raw",
            "prompt=",
        ):
            self.assertNotIn(
                marker, joined,
                f"proxy.canary log must not contain {marker!r}. Got: {msgs!r}",
            )


class TestCanaryDiagnosticsDebugFull(unittest.TestCase):
    """DEBUG-level lines carry the full payload without truncation."""

    def test_task_args_raw_full_logs_no_prompt_content(self):
        """DEBUG-level task_args_raw_full line was REMOVED entirely. Pin the
        redaction: the marker must not appear in any DEBUG record (or any
        level). Asserted at the helper boundary (`_inject_canary_into_task_calls`),
        which is the actual production emit site.
        """
        import logging
        from proxy.canary import _inject_canary_into_task_calls

        secret_marker = "SECRET-MARKER-FULL-12345"
        long_prompt = "Y" * 8000 + secret_marker + "Y" * 100
        canary = "[[__SUBAGENT__:SKILL:ticket:0123456789abcdef0123456789abcdef:1700000000]]\n"
        data = {
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [{
                        "id": "call_untrunc",
                        "type": "function",
                        "function": {
                            "name": "task",
                            "arguments": json.dumps({
                                "description": "test",
                                "prompt": long_prompt,
                            }),
                        },
                    }],
                },
            }],
        }
        with self.assertLogs("proxy.canary", level=logging.DEBUG) as cm:
            _inject_canary_into_task_calls(data, canary)

        msgs = [r.getMessage() for r in cm.records]
        joined = "\n".join(msgs)
        for marker in (secret_marker, "task_args_raw"):
            self.assertNotIn(
                marker, joined,
                f"Removed DEBUG log must not appear. Got: {msgs!r}",
            )

    def test_detect_ingress_full_logs_no_first_content(self):
        """DEBUG-level detect_ingress_full line was REMOVED entirely. Pin the
        redaction: marker must not appear in any DEBUG record.
        """
        import logging
        from proxy.canary import detect_subagent_ingress

        secret_marker = "SECRET-MARKER-INGRESS-12345"
        long_text = "y" * 8000 + secret_marker + "y" * 100

        with self.assertLogs("proxy.canary", level=logging.DEBUG) as cm:
            detect_subagent_ingress({"messages": [{"role": "user", "content": long_text}]})

        msgs = [r.getMessage() for r in cm.records]
        joined = "\n".join(msgs)
        for marker in (secret_marker, "detect_ingress_full"):
            self.assertNotIn(
                marker, joined,
                f"Removed DEBUG log must not appear. Got: {msgs!r}",
            )


class TestCanaryRefactorPinning(unittest.TestCase):
    """Pinning tests for the canary parser collapse refactor.

    These exercise the PUBLIC surface (detect_subagent_ingress,
    verify_canary, _strip_existing_canaries) so they remain valid
    regression nets regardless of the internal parser implementation.

    Against the ORIGINAL (pre-refactor) code:
      - test_bare_then_workspace_canary_no_skill_inheritance is RED
        (the line-560 ``inherited_skill="workspace"`` bug).
      - the rest are GREEN (pinning current correct behaviour).
    """

    _SECRET = "test-secret-canary-refactor-pinning"

    def setUp(self):
        import os
        from proxy import canary as canary_mod

        self._had_secret = "PROXY_SESSION_SECRET" in os.environ
        self._orig_secret_env = os.environ.pop("PROXY_SESSION_SECRET", None)
        os.environ["PROXY_SESSION_SECRET"] = self._SECRET
        self._orig_mod_secret = canary_mod._SESSION_SECRET
        canary_mod._SESSION_SECRET = self._SECRET
        self._had_proxy_env = "PROXY_ENV" in os.environ
        self._orig_proxy_env = os.environ.pop("PROXY_ENV", None)
        os.environ["PROXY_ENV"] = self._orig_proxy_env or "dev"

        mock_registry = MagicMock()

        def _get(name):
            m = MagicMock()
            m.name = name
            return m

        mock_registry.get.side_effect = _get
        self._reg_patcher = patch(
            "proxy.skills.skill_registry.SkillRegistry", return_value=mock_registry
        )
        self._reg_patcher.start()

    def tearDown(self):
        import os
        from proxy import canary as canary_mod

        self._reg_patcher.stop()
        canary_mod._SESSION_SECRET = self._orig_mod_secret
        os.environ.pop("PROXY_SESSION_SECRET", None)
        if self._had_secret:
            os.environ["PROXY_SESSION_SECRET"] = self._orig_secret_env
        os.environ.pop("PROXY_ENV", None)
        if self._had_proxy_env:
            os.environ["PROXY_ENV"] = self._orig_proxy_env

    def _skill_canary(self, skill_name, state=None):
        from proxy.canary import sign_canary

        state_str = ""
        if state:
            state_str = f"|{json.dumps(state, sort_keys=True)}"
        sig = sign_canary(skill_name, state_str=state_str)
        return f"[[__SUBAGENT__:SKILL:{skill_name}:{sig}{state_str}]]"

    def _workspace_canary(self, ws_name):
        from proxy.canary import sign_canary

        sig = sign_canary(f"workspace:{ws_name}")
        return f"[[__SUBAGENT__:WORKSPACE:{ws_name}:{sig}]]"

    def test_bare_then_skill_canary_recovery(self):
        """[[__SUBAGENT__]]\n[[__SUBAGENT__:SKILL:ticket:...]] → inherit ticket."""
        from proxy.canary import detect_subagent_ingress

        canary = "[[__SUBAGENT__]]\n" + self._skill_canary("ticket")
        body = {"messages": [{"role": "user", "content": canary + "\nHello"}]}
        ingress = detect_subagent_ingress(body)

        self.assertTrue(ingress.is_subagent)
        self.assertEqual(ingress.inherited_skill, "ticket")
        self.assertTrue(ingress.subagent_ticket_mode)
        self.assertEqual(ingress.workspace_name, "")
        self.assertEqual(body["messages"][0]["content"], "Hello")

    def test_bare_then_workspace_canary_no_skill_inheritance(self):
        """THE LINE-560 BUG FIX.

        [[__SUBAGENT__]]\n[[__SUBAGENT__:WORKSPACE:ws:...]] must set
        workspace_name='ws' but inherited_skill=None (NOT 'workspace').
        Against the original code this is RED.
        """
        from proxy.canary import detect_subagent_ingress

        canary = "[[__SUBAGENT__]]\n" + self._workspace_canary("ws")
        body = {"messages": [{"role": "user", "content": canary + "\nHello"}]}
        ingress = detect_subagent_ingress(body)

        self.assertTrue(ingress.is_subagent)
        self.assertIsNone(ingress.inherited_skill)
        self.assertEqual(ingress.workspace_name, "ws")
        self.assertFalse(ingress.subagent_ticket_mode)
        self.assertEqual(body["messages"][0]["content"], "Hello")

    def test_bare_then_skill_canary_hmac_failure(self):
        """bare + skill canary with a bad signature → downgrade, no inheritance,
        and the exact stripped remainder is left in the message."""
        from proxy.canary import detect_subagent_ingress

        bad = f"[[__SUBAGENT__:SKILL:ticket:{'1' * 64}:{int(time.time())}]]"
        canary = "[[__SUBAGENT__]]\n" + bad
        body = {"messages": [{"role": "user", "content": canary + "\nHello"}]}
        ingress = detect_subagent_ingress(body)

        self.assertTrue(ingress.is_subagent)
        self.assertIsNone(ingress.inherited_skill)
        self.assertFalse(ingress.subagent_ticket_mode)
        self.assertEqual(body["messages"][0]["content"], "Hello")

    def test_verify_canary_rejects_expired_timestamp(self):
        """REAL verify_canary (no mock): old (>300s) and future (>30s skew)
        timestamps are rejected even with a correctly-computed HMAC. (C-3)"""
        import hashlib
        import hmac as _hmac
        from proxy.canary import verify_canary

        now = int(time.time())

        def _craft(ts):
            msg = f"ticket::{ts}"
            sig = _hmac.new(
                self._SECRET.encode(), msg.encode(), hashlib.sha256
            ).hexdigest()
            return f"ticket:{sig}:{ts}"

        self.assertIsNone(verify_canary(_craft(str(now - 400))))  # too old
        self.assertIsNone(verify_canary(_craft(str(now + 60))))  # future beyond skew

    def test_hmac_failure_remaining_text_skill(self):
        """Forged SKILL canary: exact stripped remainder, not just is_subagent."""
        from proxy.canary import detect_subagent_ingress

        bad = f"[[__SUBAGENT__:SKILL:ticket:{'1' * 64}:{int(time.time())}]]"
        body = {"messages": [{"role": "user", "content": bad + "\nHello skill"}]}
        ingress = detect_subagent_ingress(body)

        self.assertTrue(ingress.is_subagent)
        self.assertIsNone(ingress.inherited_skill)
        self.assertEqual(body["messages"][0]["content"], "Hello skill")

    def test_hmac_failure_remaining_text_workspace(self):
        """Forged WORKSPACE canary: exact stripped remainder, not just is_subagent."""
        from proxy.canary import detect_subagent_ingress

        bad = f"[[__SUBAGENT__:WORKSPACE:ws:{'1' * 64}:{int(time.time())}]]"
        body = {"messages": [{"role": "user", "content": bad + "\nHello ws"}]}
        ingress = detect_subagent_ingress(body)

        self.assertTrue(ingress.is_subagent)
        self.assertEqual(ingress.workspace_name, "")
        self.assertEqual(body["messages"][0]["content"], "Hello ws")

    def test_strip_two_adjacent_stacked_skill_canaries(self):
        """_strip_existing_canaries removes two adjacent skill canaries
        (state contains ']') and preserves trailing text."""
        from proxy.canary import _strip_existing_canaries

        c = self._skill_canary("ticket", state={"tasks": []})
        text = c + c + "trailing"
        result = _strip_existing_canaries(text)

        self.assertNotIn("[[__SUBAGENT__", result)
        # Exact equality: a regression stripping 1-of-2 + appending junk
        # would otherwise pass via assertIn("trailing", ...).
        self.assertEqual(result, "trailing")

    def test_task_hash_roundtrip_through_detect(self):
        """SKILL canary with {"task_hash":"abc"} → ingress.task_hash == 'abc'."""
        from proxy.canary import detect_subagent_ingress

        canary = self._skill_canary("work", state={"task_hash": "abc"})
        body = {"messages": [{"role": "user", "content": canary + "\nDo work"}]}
        ingress = detect_subagent_ingress(body)

        self.assertTrue(ingress.is_subagent)
        self.assertEqual(ingress.task_hash, "abc")

    def test_strip_workspace_canary(self):
        """_strip_existing_canaries on a WORKSPACE canary removes the marker
        and preserves surrounding text.

        WORKSPACE has 0% coverage on the stripper in earlier test suites
        (the SKILL-with-state case is well-pinned, but WORKSPACE has no
        state, only sig:ts). This pins the WORKSPACE branch.
        """
        from proxy.canary import _strip_existing_canaries

        c = self._workspace_canary("alpha")
        text = f"prefix {c} suffix"
        result = _strip_existing_canaries(text)

        self.assertNotIn("[[__SUBAGENT__", result)
        self.assertEqual(result, "prefix  suffix")

    def test_mid_message_canary_does_not_match(self):
        """A canary NOT at position 0 must NOT trigger sub-agent detection.

        Regression guard: _locate_canary uses text.find() which matches at
        any position. parse_canary must reject matches where loc.start != 0,
        restoring the old text.startswith() semantics. Without this guard,
        a mid-message canary would strip all preceding text from the user's
        message and silently flip is_subagent=True.
        """
        from proxy.canary import detect_subagent_ingress, parse_canary

        canary = self._skill_canary("ticket")
        content = f"I want to know more. {canary} Please help."

        # parse_canary must return None (canary not at start)
        p = parse_canary(content)
        self.assertIsNone(p)

        # detect_subagent_ingress must NOT trigger
        body = {"messages": [{"role": "user", "content": content}]}
        ingress = detect_subagent_ingress(body)
        self.assertFalse(ingress.is_subagent)
        # Content must be untouched
        self.assertEqual(body["messages"][0]["content"], content)



class TestCanaryVerifyCache(unittest.TestCase):
    """Canary verification cache tests (§5.4).

    The cache remembers verified canaries keyed by
    (session_id, sig_hash, state_hash, secret_hash) so long-running
    sub-agents (>5 min) don't lose their ticket mode when the HMAC
    freshness window expires.
    """

    _SECRET = "test-secret-canary-verify-cache"

    def setUp(self):
        import os
        from proxy import canary as canary_mod
        import proxy.state as state

        self._had_secret = "PROXY_SESSION_SECRET" in os.environ
        self._orig_secret_env = os.environ.pop("PROXY_SESSION_SECRET", None)
        os.environ["PROXY_SESSION_SECRET"] = self._SECRET
        self._orig_mod_secret = canary_mod._SESSION_SECRET
        canary_mod._SESSION_SECRET = self._SECRET
        self._had_proxy_env = "PROXY_ENV" in os.environ
        self._orig_proxy_env = os.environ.pop("PROXY_ENV", None)
        os.environ["PROXY_ENV"] = self._orig_proxy_env or "dev"

        mock_registry = MagicMock()

        def _get(name):
            m = MagicMock()
            m.name = name
            return m

        mock_registry.get.side_effect = _get
        self._reg_patcher = patch(
            "proxy.skills.skill_registry.SkillRegistry", return_value=mock_registry
        )
        self._reg_patcher.start()

        # Clear cache before each test
        with state.canary_verify_cache_lock:
            state.canary_verify_cache.clear()

    def tearDown(self):
        import os
        from proxy import canary as canary_mod
        import proxy.state as state

        self._reg_patcher.stop()
        canary_mod._SESSION_SECRET = self._orig_mod_secret
        os.environ.pop("PROXY_SESSION_SECRET", None)
        if self._had_secret:
            os.environ["PROXY_SESSION_SECRET"] = self._orig_secret_env
        os.environ.pop("PROXY_ENV", None)
        if self._had_proxy_env:
            os.environ["PROXY_ENV"] = self._orig_proxy_env

        with state.canary_verify_cache_lock:
            state.canary_verify_cache.clear()

    def _skill_canary(self, skill_name, state=None):
        from proxy.canary import sign_canary

        state_str = ""
        if state:
            state_str = f"|{json.dumps(state, sort_keys=True)}"
        sig = sign_canary(skill_name, state_str=state_str)
        return f"[[__SUBAGENT__:SKILL:{skill_name}:{sig}{state_str}]]"

    # ── Test 1: Cache hit/miss ────────────────────────────────────────

    def test_cache_hit_second_request_skips_freshness(self):
        """Sub-agent first request verifies + caches; second request (same
        canary) hits cache, skips freshness — assert no 'timestamp out of
        range' warning on the second call."""
        import logging
        import proxy.state as state

        canary = self._skill_canary("ticket")
        session_id = "sess_cache_hit"

        # First request: verify + cache
        p1 = parse_canary(canary, session_id=session_id)
        self.assertIsNotNone(p1)
        self.assertTrue(p1.verified)
        self.assertEqual(p1.skill_name, "ticket")

        # Cache must be populated
        self.assertEqual(len(state.canary_verify_cache), 1)

        # Second request: should hit cache, no freshness warning
        p2 = parse_canary(canary, session_id=session_id)

        self.assertIsNotNone(p2)
        self.assertTrue(p2.verified)
        self.assertEqual(p2.skill_name, "ticket")

    # ── Test 2: >5min simulation ──────────────────────────────────────

    def test_expired_timestamp_without_cache_fails(self):
        """Without cache (no session_id), a canary with timestamp >300s old
        must fail verification."""
        from proxy.canary import sign_canary

        state_str = ""
        sig = sign_canary("ticket", state_str=state_str)
        canary = f"[[__SUBAGENT__:SKILL:ticket:{sig}{state_str}]]"

        # Advance time by 400s so freshness check fails
        real_time = time.time
        try:
            time.time = lambda: real_time() + 400
            p = parse_canary(canary)  # no session_id → no cache
        finally:
            time.time = real_time

        self.assertIsNotNone(p)
        self.assertFalse(p.verified)

    def test_expired_timestamp_with_cache_succeeds(self):
        """With cache (same session_id), a canary with timestamp >300s old
        still verifies because the cache hit bypasses freshness.

        Strengthened (was: only checked verified+skill_name):
        - p2.state must equal p1.state (the cached state, NOT a fresh
          re-parse — catches F1 if the implementation falls through to
          ``loc.state`` on cache hit instead of returning the cached copy).
        - detect_subagent_ingress with a cached canary body must report
          subagent_ticket_mode=True (the production symptom: without
          the cache, a stale canary yields subagent_ticket_mode=False
          and the tool guard blocks proxy_ticket).
        """
        import proxy.state as state
        from proxy.canary import sign_canary

        state_orig = {"workspace_name": "ws_a", "task_hash": "abc"}
        state_str_orig = f"|{json.dumps(state_orig, sort_keys=True)}"
        sig = sign_canary("ticket", state_str=state_str_orig)
        canary = f"[[__SUBAGENT__:SKILL:ticket:{sig}{state_str_orig}]]"
        session_id = "sess_expired_cache"

        # First request at current time: verify + cache
        p1 = parse_canary(canary, session_id=session_id)
        self.assertTrue(p1.verified)
        self.assertEqual(p1.state, state_orig)

        # Advance time by 400s — past both HMAC freshness (300s) AND
        # any realistic cache TTL. Without the cache, verify_canary
        # would reject as expired.
        real_time = time.time
        try:
            time.time = lambda: real_time() + 400
            p2 = parse_canary(canary, session_id=session_id)
        finally:
            time.time = real_time

        self.assertIsNotNone(p2)
        self.assertTrue(
            p2.verified,
            "Cached canary must still verify at t+400s (cache hit "
            "skips the 300s HMAC freshness check).",
        )
        self.assertEqual(p2.skill_name, "ticket")

        # Cached state must round-trip verbatim (F1 invariant). If the
        # implementation returns ``loc.state`` on cache hit, this would
        # still match because the text is unchanged — but the new
        # test_cached_state_returned_verbatim below covers the
        # mutated-loc.state case via identity check.
        self.assertEqual(p2.state, state_orig)

        # Production symptom: subagent_ticket_mode=True via the same
        # path the request pipeline uses (detect_subagent_ingress).
        body = {"messages": [{"role": "user", "content": canary + "\nHi"}]}
        ingress = detect_subagent_ingress(body, session_id=session_id)
        self.assertTrue(ingress.is_subagent)
        self.assertTrue(
            ingress.subagent_ticket_mode,
            "Sub-agent with cached ticket canary must retain "
            "subagent_ticket_mode=True past the 300s freshness window. "
            "If False, the tool guard blocks proxy_ticket with the "
            "'MAIN agent in ticket mode' error.",
        )
        self.assertEqual(ingress.inherited_skill, "ticket")

    # ── Test 3: State mutation (C1) ───────────────────────────────────

    def test_state_mutation_causes_cache_miss_and_rejection(self):
        """Mutate the |{json} suffix of a cached canary → cache miss
        (different state_hash) → full HMAC verify → rejection. Assert
        verified=False and no cache poisoning."""
        import proxy.state as state
        from proxy.canary import sign_canary

        state_orig = {"workspace_name": "ws_a", "task_hash": "abc"}
        state_str_orig = f"|{json.dumps(state_orig, sort_keys=True)}"
        sig = sign_canary("work", state_str=state_str_orig)
        canary_orig = f"[[__SUBAGENT__:SKILL:work:{sig}{state_str_orig}]]"
        session_id = "sess_state_mut"

        # First request: verify + cache with original state
        p1 = parse_canary(canary_orig, session_id=session_id)
        self.assertTrue(p1.verified)
        self.assertEqual(len(state.canary_verify_cache), 1)

        # Mutate state — different state_hash → cache miss
        state_mut = {"workspace_name": "ws_b", "task_hash": "abc"}
        state_str_mut = f"|{json.dumps(state_mut, sort_keys=True)}"
        canary_mut = f"[[__SUBAGENT__:SKILL:work:{sig}{state_str_mut}]]"

        p2 = parse_canary(canary_mut, session_id=session_id)
        self.assertIsNotNone(p2)
        self.assertFalse(p2.verified)
        self.assertIsNone(p2.skill_name)

        # Cache must NOT be poisoned — still 1 entry (the original)
        self.assertEqual(len(state.canary_verify_cache), 1)

    # ── Test 4: Cross-session isolation ───────────────────────────────

    def test_cross_session_isolation(self):
        """Canary verified in session A; replay in session B (different
        session_id) → cache miss → full freshness check applies."""
        import proxy.state as state
        from proxy.canary import sign_canary

        state_str = ""
        sig = sign_canary("ticket", state_str=state_str)
        canary = f"[[__SUBAGENT__:SKILL:ticket:{sig}{state_str}]]"

        # Session A: verify + cache
        p_a = parse_canary(canary, session_id="sess_a")
        self.assertTrue(p_a.verified)
        self.assertEqual(len(state.canary_verify_cache), 1)

        # Session B: different session_id → cache miss
        # Advance time so freshness would fail if cache were shared
        real_time = time.time
        try:
            time.time = lambda: real_time() + 400
            p_b = parse_canary(canary, session_id="sess_b")
        finally:
            time.time = real_time

        # Session B should NOT hit session A's cache entry
        self.assertIsNotNone(p_b)
        self.assertFalse(p_b.verified)

        # Session A's entry still intact
        self.assertEqual(len(state.canary_verify_cache), 1)

    # ── Test 5: Dev-fallback caching (H1) ─────────────────────────────

    def test_dev_fallback_caching(self):
        """With no PROXY_SESSION_SECRET, first request caches
        is_dev_fallback=True; second request hits cache, returns
        is_dev_fallback=True (not hardcoded False)."""
        import os
        import proxy.state as state
        from proxy import canary as canary_mod

        # Remove secret to enter dev-fallback path
        os.environ.pop("PROXY_SESSION_SECRET", None)
        canary_mod._SESSION_SECRET = ""

        try:
            text = f'[[__SUBAGENT__:SKILL:ticket:{_FAKE_SIG}:{_FRESH_TS}]]'
            session_id = "sess_dev_fallback"

            p1 = parse_canary(text, session_id=session_id)
            self.assertIsNotNone(p1)
            self.assertTrue(p1.verified)
            self.assertEqual(p1.skill_name, "ticket")

            # Cache must be populated
            self.assertEqual(len(state.canary_verify_cache), 1)

            # Inspect cache entry for is_dev_fallback
            with state.canary_verify_cache_lock:
                entry = next(iter(state.canary_verify_cache.values()))
            _, _, _, cached_dev_fallback, _ = entry
            self.assertTrue(cached_dev_fallback)

            # Second request: cache hit
            p2 = parse_canary(text, session_id=session_id)
            self.assertIsNotNone(p2)
            self.assertTrue(p2.verified)
            self.assertEqual(p2.skill_name, "ticket")
        finally:
            os.environ["PROXY_SESSION_SECRET"] = self._SECRET
            canary_mod._SESSION_SECRET = self._SECRET

    # ── Test 6: Secret rotation (F3) ──────────────────────────────────

    def test_secret_rotation_causes_cache_miss(self):
        """Cache entry under secret X; rotate to secret Y (monkeypatch
        _get_session_secret) → cache miss → re-verify under new secret."""
        import proxy.state as state
        from proxy import canary as canary_mod
        from proxy.canary import sign_canary

        state_str = ""
        sig = sign_canary("ticket", state_str=state_str)
        canary = f"[[__SUBAGENT__:SKILL:ticket:{sig}{state_str}]]"
        session_id = "sess_secret_rot"

        # First request: verify + cache under secret X
        p1 = parse_canary(canary, session_id=session_id)
        self.assertTrue(p1.verified)
        self.assertEqual(len(state.canary_verify_cache), 1)

        # Rotate secret: change _get_session_secret to return a different value
        new_secret = "rotated-secret-value-xyz"
        orig_get_secret = canary_mod._get_session_secret
        try:
            canary_mod._get_session_secret = lambda: new_secret

            # Second request: different secret_hash → cache miss
            # The canary was signed with the OLD secret, so re-verify will fail
            p2 = parse_canary(canary, session_id=session_id)
        finally:
            canary_mod._get_session_secret = orig_get_secret

        # Cache miss → full HMAC verify with new secret → should fail
        self.assertIsNotNone(p2)
        self.assertFalse(p2.verified)

    # ── Test 7: Forged canary not cached ──────────────────────────────

    def test_forged_canary_not_cached(self):
        """Bad sig → verified=False → assert cache NOT populated
        (no negative caching)."""
        import proxy.state as state

        bad = f"[[__SUBAGENT__:SKILL:ticket:{'1' * 64}:{int(time.time())}]]"
        session_id = "sess_forged"

        p = parse_canary(bad, session_id=session_id)
        self.assertIsNotNone(p)
        self.assertFalse(p.verified)

        # Cache must be empty — no negative caching
        self.assertEqual(len(state.canary_verify_cache), 0)

    # ── Test 10: Bare canary excluded (N6) ────────────────────────────

    def test_bare_canary_never_enters_cache(self):
        """Bare [[__SUBAGENT__]] → assert cache unchanged (never enters
        _verify_canary)."""
        import proxy.state as state

        session_id = "sess_bare"
        p = parse_canary("[[__SUBAGENT__]]", session_id=session_id)
        self.assertIsNotNone(p)
        self.assertTrue(p.verified)
        self.assertEqual(p.kind, "bare")

        # Cache must be empty — bare canaries skip _verify_canary
        self.assertEqual(len(state.canary_verify_cache), 0)

    # ── Test 11: Tool guard integration ───────────────────────────────

    def test_tool_guard_allows_cached_ticket_subagent_past_5min(self):
        """End-to-end production symptom: a long-running ticket sub-agent
        (>5 min) must retain ``proxy_ticket`` access.

        Reproduction sequence (per spec §2.1):
        1. Build a real ticket canary via ``sign_canary``.
        2. Cache it via ``parse_canary`` (simulates the dispatch turn).
        3. Advance the clock 400s past the HMAC freshness window.
        4. Re-parse the canary → must still verify (cache hit).
        5. Call ``detect_subagent_ingress`` with the same canary body
           → ``subagent_ticket_mode=True`` (the production symptom
           bit: was ``False`` before the cache landed).
        6. Call the actual tool guard's ticket-mode orchestrator gate
           (``_check_ticket_mode_orchestrator``) with
           ``subagent_ticket_mode=True`` → must return ``None``
           (tool call allowed).
        7. Compare: with ``subagent_ticket_mode=False``, the same
           call returns the
           ``"Tool '...' is blocked for the MAIN agent in ticket mode"``
           error string. This delta proves the test catches the
           production symptom — a bare sub-agent or one with
           ``subagent_ticket_mode=False`` is blocked.

        NOTE: We exercise ``_check_ticket_mode_orchestrator`` (the gate
        that checks ``subagent_ticket_mode`` at tool_guard.py:495-498),
        NOT ``_check_hierarchy``. ``_check_hierarchy`` only inspects
        ``is_subagent`` (delegating to ``check_sub_agent_proxy_access``),
        so it cannot distinguish ``subagent_ticket_mode=True`` from
        ``False`` — using it would NOT catch the production symptom.
        """
        from unittest.mock import patch as _patch
        from proxy.canary import sign_canary as _sign
        from proxy.agent.agent import Agent as _Agent
        from proxy.skills.skill_definition import ModeState as _ModeState
        from proxy.tool_pipeline.tool_guard import MemoryToolGuard as _Guard
        from proxy.tool_pipeline.tool_call import ToolCall as _ToolCall
        from proxy.agent.agent_registry import registry as _agent_registry

        state_str = ""
        sig = _sign("ticket", state_str=state_str)
        canary = f"[[__SUBAGENT__:SKILL:ticket:{sig}{state_str}]]"
        session_id = "sess_tool_guard_e2e"

        # Cache the canary (simulates the sub-agent's dispatch turn).
        p1 = parse_canary(canary, session_id=session_id)
        self.assertTrue(p1.verified)
        self.assertEqual(p1.skill_name, "ticket")

        # Advance clock 400s past the HMAC freshness window.
        real_time = time.time
        try:
            time.time = lambda: real_time() + 400
            p2 = parse_canary(canary, session_id=session_id)
        finally:
            time.time = real_time

        # Cache hit skips freshness → still verified.
        self.assertIsNotNone(p2)
        self.assertTrue(
            p2.verified,
            "After 400s, cached canary must still verify (cache hit "
            "bypasses freshness — the production fix).",
        )
        self.assertEqual(p2.skill_name, "ticket")

        # detect_subagent_ingress with the cached canary → subagent_ticket_mode.
        body = {"messages": [{"role": "user", "content": canary + "\nHello"}]}
        ingress = detect_subagent_ingress(body, session_id=session_id)
        self.assertTrue(ingress.is_subagent)
        self.assertTrue(
            ingress.subagent_ticket_mode,
            "Sub-agent with cached ticket canary must report "
            "subagent_ticket_mode=True past 5min (else tool guard "
            "blocks proxy_ticket).",
        )
        self.assertEqual(ingress.inherited_skill, "ticket")

        # Reproduce the tool guard's production symptom gate.
        # The orchestrator's persistent ticket mode is the trigger that
        # causes _check_ticket_mode_orchestrator to block the call when
        # subagent_ticket_mode is False.
        orch_agent = _Agent(session_id=session_id, agent_id="orch_tg")
        ticket_mode = _ModeState(
            name="ticket",
            activated_at=time.time(),
            ttl_seconds=300,
            skill_state={},
        )
        self.assertTrue(orch_agent.set_active_mode(ticket_mode))

        # Tool call: the sub-agent invoking proxy_ticket.
        tc = _ToolCall(
            id="tg_1",
            name="proxy_ticket",
            arguments={"action": "list_tasks"},
            source="llm",
        )

        # Build a minimal ctx with the production-symptom flags.
        ctx_ok = MagicMock()
        ctx_ok.session_id = session_id
        ctx_ok.is_subagent = True
        ctx_ok.subagent_ticket_mode = True
        ctx_ok.agent_id = "sub_tg"

        ctx_bad = MagicMock()
        ctx_bad.session_id = session_id
        ctx_bad.is_subagent = True
        ctx_bad.subagent_ticket_mode = False  # simulates post-bug behavior
        ctx_bad.agent_id = "sub_tg"

        guard = _Guard()

        # Patch agent_registry.get_or_load to return our orch_agent
        # (which has persistent ticket mode). The source imports it
        # inside _check_ticket_mode_orchestrator, so patch the
        # canonical location.
        with _patch.object(_agent_registry, "get_or_load", return_value=orch_agent):
            # Production-correct path: subagent_ticket_mode=True → allowed.
            err_ok = guard._check_ticket_mode_orchestrator(tc, ctx_ok)
            self.assertIsNone(
                err_ok,
                "ticket sub-agent with subagent_ticket_mode=True must "
                "be allowed past 5min; got error: "
                f"{err_ok!r}",
            )

            # Production-symptom path: subagent_ticket_mode=False (the
            # pre-fix bug) → orchestrator's persistent ticket mode
            # blocks the call with the documented error.
            err_bad = guard._check_ticket_mode_orchestrator(tc, ctx_bad)
            self.assertIsNotNone(
                err_bad,
                "With subagent_ticket_mode=False, the orchestrator's "
                "persistent ticket mode must block the call (this is "
                "the production error).",
            )
            self.assertIn("MAIN agent in ticket mode", err_bad)

    # ── Test 12: Deactivation re-arm (N2) ─────────────────────────────

    def test_cached_canary_survives_clear_active_mode(self):
        """Cache re-arms the ticket skill after ``clear_active_mode``
        (spec §5.10 — INTENDED behavior, N2).

        Sequence:
        1. Build a real ticket canary; cache it via ``parse_canary``.
        2. Create an Agent in persistent ticket mode (mirrors the
           orchestrator's state inherited by the sub-agent registry
           entry).
        3. Call ``agent.clear_active_mode(expected_mode=ms)`` to
           simulate the production path:
           ``TicketSkill.should_deactivate`` returns True on
           ``finish_reason == 'stop'`` → ``mode.py`` calls
           ``agent.clear_active_mode()`` → ``active_mode = None``.
        4. Assert the agent's mode is cleared (sanity check).
        5. Advance the clock 400s past the HMAC freshness window.
        6. Re-parse the canary → must still verify (cache hit).
           The cache keeps the canary valid; the next request's
           ``skill_resolver.resolve_mode`` re-inherits the skill from
           ``ctx.inherited_skill='ticket'``, creating a fresh
           ``ModeState`` — this is the intended re-arm (§5.10).
        7. Assert the cache entry is unchanged (1 entry, same key).

        This is INTENDED per §5.10: the parent's canary is the
        authoritative statement of the sub-agent's role for the
        session; a re-invoked sub-agent retains its role. True
        revocation requires the parent to stop dispatching task calls.
        """
        import proxy.state as state
        from proxy.agent.agent import Agent as _Agent
        from proxy.skills.skill_definition import ModeState as _ModeState

        canary = self._skill_canary("ticket")
        session_id = "sess_deactivation_rearm"

        # 1. Cache the canary.
        p1 = parse_canary(canary, session_id=session_id)
        self.assertTrue(p1.verified)
        self.assertEqual(p1.skill_name, "ticket")
        self.assertEqual(len(state.canary_verify_cache), 1)

        # 2. Create an Agent in persistent ticket mode (simulates the
        # orchestrator's persistent ticket mode that the sub-agent's
        # agent-registry entry would inherit).
        agent = _Agent(session_id=session_id, agent_id="rearm_agent")
        mode_state = _ModeState(
            name="ticket",
            activated_at=time.time(),
            ttl_seconds=300,
            skill_state={},
        )
        self.assertTrue(agent.set_active_mode(mode_state))
        self.assertEqual(agent.get_mode_name(), "ticket")

        # 3. Simulate should_deactivate → clear_active_mode. Pass the
        # snapshotted mode as expected_mode to mirror the production
        # CAS-style call (prevents racing with a concurrent re-activate).
        current_mode = agent.snapshot_mode()
        self.assertIsNotNone(current_mode)
        cleared = agent.clear_active_mode(expected_mode=current_mode)
        self.assertTrue(
            cleared,
            "clear_active_mode with the current mode as expected_mode "
            "must succeed (production call pattern from mode.py:344).",
        )

        # 4. Mode is cleared.
        self.assertIsNone(
            agent.snapshot_mode(),
            "Agent's active_mode must be None after clear_active_mode.",
        )

        # Cache entry must be UNAFFECTED by the clear — the canary
        # cache is keyed by HMAC signature, not by agent mode state.
        self.assertEqual(
            len(state.canary_verify_cache), 1,
            "clear_active_mode must NOT evict the canary cache entry "
            "(per spec §5.10).",
        )

        # 5. Advance clock 400s past the HMAC freshness window.
        real_time = time.time
        try:
            time.time = lambda: real_time() + 400
            # 6. Re-parse the canary → cache hit → still verified.
            p2 = parse_canary(canary, session_id=session_id)
        finally:
            time.time = real_time

        # The cache kept the canary valid across the mode clear AND
        # the 400s clock advance — without it, the HMAC freshness
        # check at 218 would reject the timestamp.
        self.assertIsNotNone(p2)
        self.assertTrue(
            p2.verified,
            "Cached canary must still verify after clear_active_mode "
            "and a 400s clock advance — the next request's "
            "skill_resolver.resolve_mode re-inherits the skill "
            "from ctx.inherited_skill='ticket' (spec §5.10).",
        )
        self.assertEqual(p2.skill_name, "ticket")

        # 7. Cache state preserved.
        self.assertEqual(len(state.canary_verify_cache), 1)

    # ── Test 13: F1 — cached state returned verbatim (identity check) ──

    def test_cached_state_returned_verbatim_on_cache_hit(self):
        """F1 regression guard: on cache hit, ``CanaryParse.state`` must be
        the CACHED state object, NOT a freshly re-parsed ``loc.state``.

        Spec §5.4 C1: "On a cache hit this is returned VERBATIM; the
        canary text's state suffix is NOT re-parsed or trusted."
        Implementation invariant: the cache ``state`` field stores
        ``copy.deepcopy(loc.state)`` at populate time and returns
        that SAME object on hit.

        We verify via object identity:
        1. Cache a canary with state ``{"workspace_name": "ws-A"}``.
        2. Capture the cached entry's ``state`` field (deep-copied at
           populate time — a different object than the original
           ``loc.state``).
        3. Call ``parse_canary`` again with the same canary.
        4. Assert ``p2.state is cached_state_obj`` — same object.

        If the implementation regresses to ``state=loc.state`` on cache
        hit (the F1 bug), the returned state is a freshly parsed dict
        (different object identity) and the assertion fails.

        Equality (``==``) check is not sufficient because both the
        cached copy and a fresh re-parse have the same content;
        identity (``is``) is the load-bearing invariant.
        """
        import proxy.state as state

        canary = self._skill_canary(
            "ticket",
            state={"workspace_name": "ws-A", "task_hash": "h-A"},
        )
        session_id = "sess_cached_state_verbatim"

        # First call: verify + cache. p1.state is loc.state (a fresh parse).
        p1 = parse_canary(canary, session_id=session_id)
        self.assertTrue(p1.verified)
        self.assertEqual(
            p1.state, {"workspace_name": "ws-A", "task_hash": "h-A"},
        )

        # Capture the CACHED state object (the deepcopy stored in the
        # cache, distinct from p1.state which is the original loc.state).
        with state.canary_verify_cache_lock:
            cached_entry = next(iter(state.canary_verify_cache.values()))
        cached_state_obj = cached_entry[2]
        self.assertIsNotNone(cached_state_obj)
        # Sanity: the cached object is a deepcopy — distinct identity
        # from the original p1.state.
        self.assertIsNot(
            cached_state_obj, p1.state,
            "Test invariant: cached_state_obj must be a deepcopy of "
            "loc.state (different identity). If it's the same object, "
            "the implementation is not deep-copying on cache populate.",
        )

        # Second call: cache hit. The returned state must BE the cached
        # object (not a fresh re-parse).
        p2 = parse_canary(canary, session_id=session_id)
        self.assertIsNotNone(p2)
        self.assertTrue(p2.verified)
        self.assertEqual(p2.skill_name, "ticket")

        # F1 identity invariant: p2.state must BE the cached object.
        self.assertIs(
            p2.state, cached_state_obj,
            "F1 regression: cache hit returned a state object that is "
            "NOT the cached entry's state. The implementation is "
            "re-parsing loc.state on cache hit (or returning a fresh "
            "copy) instead of the cached copy. Per spec §5.4 C1 the "
            "cached state must be returned verbatim.",
        )

        # Sanity: p2.state must NOT be the same object as p1.state
        # (different parse iterations → different loc.state objects).
        # If the implementation returned p1.state, that would also
        # violate the "verbatim cached state" contract (p1.state is
        # loc.state from iteration 1, not the cached copy).
        self.assertIsNot(
            p2.state, p1.state,
            "F1 regression: cache hit returned loc.state from a "
            "previous parse instead of the cached deepcopy.",
        )


if __name__ == "__main__":
    unittest.main()
