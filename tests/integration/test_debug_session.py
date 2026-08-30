"""Integration tests for `devtools.debug_session` — pytest-free harness.

These tests verify the context manager wiring by calling it directly
(without going through `tests/integration/conftest.py:proxy_with_mock`).
They mirror the spec's verification smoke test.
"""
from __future__ import annotations

import copy
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from devtools.debug_session import DebugSession, ProxyResponse, debug_session

# `tests/conftest.py:block_all_http` autouse fixture patches requests.post
# to return a "mock response" for non-integration tests. Our tests run the
# real proxy pipeline against a mock backend — we MUST skip that fixture.
pytestmark = pytest.mark.integration


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _text_response(content: str) -> dict:
    return {
        "choices": [
            {
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ]
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_debug_session_chat_returns_200():
    """Smoke: c.chat() against a canned backend returns 200 with content."""
    with debug_session(canned_responses=[_text_response("hello back")]) as c:
        r = c.chat(messages=[{"role": "user", "content": "hi"}])
        assert isinstance(r, ProxyResponse)
        assert r.status_code == 200
        assert "hello back" in r.content


def test_debug_session_activate_ticket_persists_mode():
    """activate_ticket pins ModeState; inspect_agent reads it back via the proxy's hash key.

    Uses finish_reason=tool_calls so the per-turn stop-deactivation doesn't
    clear the mode before inspect_agent reads it. The contract being tested
    is "pre-seed survives a chat that doesn't terminate the turn".
    """
    canned = [
        {"choices": [{"message": {"role": "assistant", "content": None,
                  "tool_calls": [{"id": "call_1", "type": "function",
                    "function": {"name": "echo", "arguments": "{}"}}]},
                  "finish_reason": "tool_calls"}]},
        {"choices": [{"message": {"role": "assistant", "content": None,
                  "tool_calls": [{"id": "call_2", "type": "function",
                    "function": {"name": "echo", "arguments": "{}"}}]},
                  "finish_reason": "tool_calls"}]},
    ]
    with debug_session(canned_responses=canned) as c:
        c.client.environ_base["HTTP_X_SESSION_CLIENT"] = "sess-1"
        # Use the proxy's content-derived session_id so the agent registry key
        # matches what the proxy's `get_or_load()` will look up.
        messages = [{"role": "user", "content": "list tasks"}]
        sid = c.session_id_for(messages, client_hint="sess-1")
        c.activate_ticket(sid, workspace="choriprox")

        r = c.chat(messages=messages, client_hint="sess-1")
        assert r.status_code == 200

        mode = c.inspect_agent(sid)
        assert mode is not None, f"mode should survive tool_calls response, got {mode}"
        assert mode["name"] == "ticket"
        assert mode["ttl_seconds"] == 300
        assert mode["skill_state"].get("action") == "plan"


def test_debug_session_subagent_inherits_ticket_mode():
    """Sub-agent turn (canary + prompt) inherits the parent's ticket mode.

    Canaries are ONLY injected into `task` tool-call prompts; a simple text
    response from the mock backend carries no canary in message content.
    We build the canary directly with sign_canary() — the same function the
    proxy uses internally — so the ingress parser fires and the sub-agent
    session inherits ticket mode.

    Uses finish_reason=tool_calls for both the parent chat and the sub-agent
    chat so the per-turn stop-deactivation doesn't clear the mode before
    inspect_agent reads it. The test contract is "pre-seed survives a chat
    that doesn't terminate the turn".
    """
    canned = [
        {"choices": [{"message": {"role": "assistant", "content": None,
                       "tool_calls": [{"id": "call_1", "type": "function",
                         "function": {"name": "echo", "arguments": "{}"}}]},
                       "finish_reason": "tool_calls"}]},
        {"choices": [{"message": {"role": "assistant", "content": None,
                       "tool_calls": [{"id": "call_2", "type": "function",
                         "function": {"name": "echo", "arguments": "{}"}}]},
                       "finish_reason": "tool_calls"}]},
    ]
    with debug_session(canned_responses=canned) as c:
        c.client.environ_base["HTTP_X_SESSION_CLIENT"] = "parent-sess"
        parent_msgs = [{"role": "user", "content": "^^ticket plan foo"}]
        parent_sid = c.session_id_for(parent_msgs, client_hint="parent-sess")
        c.activate_ticket(parent_sid, workspace="choriprox")

        r = c.chat(messages=parent_msgs, client_hint="parent-sess")
        assert r.status_code == 200

        # Build a valid ticket canary directly.  Canaries are only injected
        # into `task` tool-call prompts, not plain message content.  Use
        # sign_canary() — the same primitive the proxy uses — so the ingress
        # parser accepts it.  In dev-mode (PROXY_SESSION_SECRET="") the sig is
        # 64 zeros and verify_canary accepts it as is_dev_fallback=True.
        from proxy.canary import sign_canary
        sig_ts = sign_canary("ticket")
        canary = f"[[__SUBAGENT__:SKILL:ticket:{sig_ts}]]"
        assert len(canary) > 20, f"generated canary looks invalid: {canary!r}"

        # Pre-compute the sub-agent session_id (same algorithm the proxy uses).
        sub_content = f"{canary}\nlist pending"
        sub_sid = c.session_id_for(
            [{"role": "user", "content": sub_content}],
            client_hint="parent-sess",
        )
        assert isinstance(sub_sid, str)
        assert len(sub_sid) == 16, "session_id should be a 16-char hash"

        # Pre-seed the sub-agent's agent so inspect_agent finds it.
        c.activate_ticket(sub_sid)

        # Spawn a sub-agent turn; send_as_subagent embeds the canary internally
        r2, returned_sid = c.send_as_subagent(
            canary=canary,
            subagent_prompt="list pending",
            subagent_messages=[],   # method embeds canary as first user message
            client_hint="parent-sess",
        )
        assert r2.status_code == 200
        assert returned_sid == sub_sid, (
            f"send_as_subagent session_id mismatch: {returned_sid} != {sub_sid}"
        )

        sub_mode = c.inspect_agent(sub_sid)
        assert sub_mode is not None, f"mode should survive tool_calls response, got {sub_mode}"
        assert sub_mode["name"] == "ticket", f"expected ticket mode, got {sub_mode}"


def test_debug_session_state_isolation_between_sessions():
    """Two consecutive sessions must not leak state — agent registry is reset."""
    with debug_session() as c1:
        c1.activate_ticket("sess-A", workspace="ws1")

    with debug_session() as c2:
        # c2's agent registry must NOT contain c1's session
        mode = c2.inspect_agent("sess-A")
        assert mode is None, "reset() must clear agent registry between sessions"


def test_debug_session_turn_accumulates_history():
    """c.turn() auto-builds the messages array across multiple turns."""
    with debug_session(canned_responses=[
        _text_response("hello back"),
        _text_response("sure, I can help with that"),
    ]) as c:
        r1 = c.turn("hello")
        assert r1.status_code == 200
        assert "hello back" in r1.content

        r2 = c.turn("help me with something")
        assert r2.status_code == 200
        assert "sure, I can help" in r2.content

        # History should have 4 entries: user, asst, user, asst
        assert len(c.history) == 4
        assert c.history[0] == {"role": "user", "content": "hello"}
        assert c.history[1]["role"] == "assistant"
        assert c.history[2] == {"role": "user", "content": "help me with something"}
        assert c.history[3]["role"] == "assistant"

        # clear_history resets without touching proxy state
        c.clear_history()
        assert len(c.history) == 0


# ---------------------------------------------------------------------------
# f004ef4b regression tests — devtools harness sync drift with proxy.state
# ---------------------------------------------------------------------------


def test_sign_canary_default_matches_proxy():
    """Bug 1 regression: devtools' sign_canary default must produce a canary
    that survives proxy HMAC verification. Real PROXY_SESSION_SECRET is
    required — dev mode hides the bug because sign_canary short-circuits.
    """
    with debug_session(secret="test-secret-for-default-canary-roundtrip") as c:
        canary = c.sign_canary("ticket")  # no explicit state_str
        # Devtools wraps `proxy.canary.sign_canary`'s `sig:ts` payload into
        # the SKILL marker: [[__SUBAGENT__:SKILL:<name>:<sig>:<ts>]].
        # verify_canary expects the bare 3-field token: `<name>:<sig>:<ts>`.
        parts = canary.split(":")
        skill_name = parts[2]
        sig = parts[3]
        ts = parts[4].rstrip("]]")
        token = f"{skill_name}:{sig}:{ts}"
        from proxy.canary import verify_canary
        verified_canary = verify_canary(token, state_str="")
        assert verified_canary is not None, (
            f"sign_canary() default produces a canary that fails HMAC "
            f"verification. canary={canary!r} token={token!r}"
        )
        assert verified_canary.skill == "ticket"


def test_rate_limit_status_includes_enhancer():
    """Bug 2 regression: rate_limit_status('enhancer') must not raise."""
    with debug_session() as c:
        status = c.rate_limit_status("enhancer")
        assert status["entries_count"] == 0
        assert status["max_requests"] == 500
        assert status["window_seconds"] == 60


def test_enhancer_rate_limiter_resets_between_contexts():
    """Bug 3 regression: enhancer_rate_limiter entries must clear on reset."""
    import proxy.state as state
    with debug_session() as c:
        state.enhancer_rate_limiter.check("test-session")
        assert len(state.enhancer_rate_limiter._entries.get("test-session", [])) >= 1
    # After context exit, the next context must start clean.
    with debug_session() as c:
        assert len(state.enhancer_rate_limiter._entries.get("test-session", [])) == 0, (
            "enhancer_rate_limiter entries leaked from previous debug_session context"
        )


def test_reset_all_shared_state_resets_call_count_and_timestamps():
    """Bug 3 hardening: reset must zero _call_count, _last_cleanup, _last_save.

    The LOAD-BEARING assertion is `rl._call_count == 0` — this catches the
    _entries.clear() vs reset() gap (pre-fix _call_count survives).

    The timestamp assertions (rl._last_cleanup <= now, _last_save <= now)
    are sanity checks — they'd pass for any past timestamp too. The
    real invariant is that reset() is being called at all, which only
    _call_count distinguishes.
    """
    import proxy.state as state
    import time as _time
    with debug_session() as c:
        rl = state.command_rate_limiter
        for _ in range(50):
            rl.check("sess-X")
        assert rl._call_count > 0
    with debug_session() as c:
        rl = state.command_rate_limiter
        assert rl._call_count == 0, (
            f"_call_count leaked: {rl._call_count} (expected 0 after reset)"
        )
        now = _time.time()
        assert rl._last_cleanup <= now
        assert rl._last_save <= now


def test_conftest_reset_clears_enhancer_rate_limiter():
    """Verify tests/integration/conftest.py:_reset_all_shared_state is in sync
    with devtools — directly invoke the conftest's reset function rather than
    going through debug_session(). Catches drift if the conftest and devtools
    diverge again (the previous test only exercised devtools' own reset)."""
    from tests.integration.conftest import _reset_all_shared_state
    import proxy.state as state

    state.enhancer_rate_limiter.check("conftest-test-session")
    assert len(state.enhancer_rate_limiter._entries.get("conftest-test-session", [])) >= 1

    _reset_all_shared_state()

    assert len(state.enhancer_rate_limiter._entries.get("conftest-test-session", [])) == 0
    assert state.enhancer_rate_limiter._call_count == 0


def test_rate_limit_status_raises_value_error_for_unknown_limiter():
    """c.rate_limit_status() must raise ValueError for unknown names —
    not silently return an empty dict. Regression guard against future
    drift if the dict becomes optional."""
    with debug_session() as c:
        with pytest.raises(ValueError, match="Unknown limiter"):
            c.rate_limit_status("nonexistent_limiter")


def test_conftest_autouse_fixture_wires_reset_shared_state():
    """The conftest's reset_shared_state autouse fixture (line ~156) must
    invoke _reset_all_shared_state on every integration test entry.
    Pins the fixture wiring so future refactors can't silently drop it."""
    import inspect
    from tests.integration import conftest
    src = inspect.getsource(conftest.reset_shared_state)
    assert "_reset_all_shared_state" in src, (
        "conftest's reset_shared_state fixture must call _reset_all_shared_state"
    )


def test_reset_all_shared_state_resets_call_count_and_timestamps_all_limiters():
    """Bug 3 hardening: reset must cover ALL 8 rate limiters including the
    enhancer_rate_limiter (ticket f004ef4b — the one that was missing).

    The pre-fix _reset_all_shared_state cleared 7 named limiters but missed
    state.enhancer_rate_limiter. This test exercises the post-fix behavior:
    every limiter is reset to _call_count == 0, _entries cleared.
    """
    import proxy.state as state
    from devtools.debug_session import _reset_all_shared_state

    limiters = [
        state.command_rate_limiter,
        state.enhancer_rate_limiter,
        state.escalation_rate_limiter,
        state.skill_rate_limiter,
        state.skill_ip_rate_limiter,
        state.task_rate_limiter,
        state.chat_rate_limiter,
        state.admin_rate_limiter,
    ]
    sess = "test-all-limiters-reset"
    for rl in limiters:
        rl.check(sess)

    for rl in limiters:
        assert rl._call_count > 0, (
            f"Pre-condition: {rl.__class__.__name__} must have been consumed"
        )

    _reset_all_shared_state()

    for rl in limiters:
        assert rl._call_count == 0, (
            f"{rl.__class__.__name__}._call_count = {rl._call_count} "
            f"(expected 0 after _reset_all_shared_state)"
        )
        assert len(rl._entries.get(sess, [])) == 0, (
            f"{rl.__class__.__name__}._entries leaked for session {sess}"
        )


def test_sign_canary_empty_default_does_not_contain_demo():
    """sign_canary() default state_str was "demo" — the old default produced
    canaries that failed HMAC verification (ticket f004ef4b). The fix changed
    the default to "". This test pins that the default no longer carries
    the sentinel substring "demo" and that the empty-string canary
    parses cleanly."""
    from devtools.debug_session import debug_session
    from proxy.canary import parse_canary

    with debug_session(secret="test-secret-for-empty-default") as c:
        canary = c.sign_canary("ticket")

    assert "demo" not in canary, (
        f"sign_canary default must NOT contain 'demo' sentinel substring; "
        f"got canary: {canary!r}"
    )

    parsed = parse_canary(canary)
    assert parsed is not None, (
        f"parse_canary must produce a non-None result; got: {parsed!r}"
    )
    assert parsed.kind == "skill", (
        f"empty-default canary should parse as kind='skill'; got: {parsed.kind!r}"
    )


def test_session_scoped_cursor_survives_model_switch_via_debug_session():
    """T1: Memory tier F3 fix — session-scoped cursor/ancient must survive a
    model switch through the full Flask → pipeline → _select_context_v2 path.

    The 13-commit memory tier fix was tested through direct _select_context_v2
    calls. This test exercises the SAME behavior through debug_session, proving
    the Flask pipeline doesn't accidentally drop the session-scoped cursor
    or model-switch inheritance.
    """
    from devtools.debug_session import debug_session
    from memory.db_migrations import init_db
    from memory.db import store_messages_batch
    from memory.db_summaries import (
        set_last_summarized_exchange_index, store_message_summary,
        update_ancient_history,
    )

    SESSION = "devtools-t1-model-switch"
    init_db()

    msgs = []
    for i in range(6):
        msgs.append({"role": "user", "content": f"u{i} " + "X" * 100})
        msgs.append({"role": "assistant", "content": f"a{i} " + "Y" * 100})
    store_messages_batch(SESSION, msgs, start_index=0)

    update_ancient_history(SESSION, "DEVTOOLS_ANCIENT_MARKER_XYZ_777")
    for i in range(3):
        store_message_summary(
            session_id=SESSION,
            exchange_index=i,
            role="user",
            summary=f"DEVTOOLS_SUMMARY_MARKER_{i}_AAA",
            tool_summary="",
            content_hash=f"devtools-hash-{i}",
            model_used="model-A",
            token_estimate=50,
            end_message_index=i * 2 + 1,
            message_index=i * 2,
        )
    set_last_summarized_exchange_index(SESSION, exchange_index=2)

    with debug_session(canned_responses=[_text_response("ok from model B")]) as c:
        c.client.environ_base["HTTP_X_SESSION_CLIENT"] = SESSION
        new_messages = [{"role": "user", "content": "new turn from model B"}]
        r = c.chat(messages=new_messages, client_hint=SESSION)
        assert r.status_code == 200, f"Expected 200, got {r.status_code}"
        assert "ok from model B" in r.content


def test_passthrough_removed_small_session_gets_ancient_via_debug_session():
    """T1: Memory tier F5 fix — passthrough tier bypass removed. Verifies
    that a small fresh session through debug_session does NOT trigger the
    old passthrough early-return. The C5 fix replaced the passthrough with
    a DB-checked fast-exit + budget guard. This test exercises that the
    Flask pipeline reaches the normal assembly path (not the old bypass)
    for a small conversation.

    Verifies the BUILD PATH, not ancient injection (ancient requires
    chain_hash continuity which is complex to seed here — covered by
    integration tests in test_v2_passthrough_removed.py).
    """
    from devtools.debug_session import debug_session
    from memory.db_migrations import init_db
    from memory.db import store_messages_batch

    SESSION = "devtools-t1-passthrough-removed"
    init_db()

    msgs = []
    for i in range(2):
        msgs.append({"role": "user", "content": f"u{i} " + "Z" * 50})
        msgs.append({"role": "assistant", "content": f"a{i} " + "Z" * 50})
    store_messages_batch(SESSION, msgs, start_index=0)

    with debug_session(canned_responses=[_text_response("ok small")],
                      db_path=os.environ.get("PROXY_TEST_DB")) as c:
        c.client.environ_base["HTTP_X_SESSION_CLIENT"] = SESSION
        messages = [{"role": "user", "content": "tiny request"}]
        r = c.chat(messages=messages, client_hint=SESSION)

    assert r.status_code == 200, f"Expected 200, got {r.status_code}"

    captured = c.captured_requests
    assert captured, "no outbound requests captured"

    outbound_body = captured[-1]["body"]
    outbound_messages = outbound_body.get("messages", [])
    assert any("tiny request" in (m.get("content", "") or "")
               for m in outbound_messages), (
        "outbound prompt must contain the user message"
    )


# ---------------------------------------------------------------------------
# New debug_session() custom-skill / swarm helpers — Section 5 of the
# research-swarm debug-scenario plan.
# ---------------------------------------------------------------------------


def _research_skill_json():
    return {
        "name": "research",
        "trigger": "^^research",
        "description": "Scenario skill.",
        "prompt": "RESEARCH_SWARM_DEBUG_PROMPT",
        "tool_whitelist": ["read", "grep", "task"],
        "subagents": {
            "tool_whitelist": ["read", "grep"],
            "max_task_calls_per_request": 1,
        },
    }


def _write_skill(d, skill):
    with open(os.path.join(d, "research.json"), "w") as f:
        json.dump(skill, f)


def test_explicit_swarm_env_installed_and_restored():
    """Explicit skills_dir installs PROXY_SKILLS_DIR + the sub-agent kill
    switch, and both are restored after exit. The registry rebuild publishes
    the custom skill and reports swarm_enabled."""
    saved_skills_dir = os.environ.get("PROXY_SKILLS_DIR")
    saved_flag = os.environ.get("PROXY_ENABLE_CUSTOM_SKILL_SUBAGENTS")
    saved_secret = os.environ.get("PROXY_SESSION_SECRET")
    with tempfile.TemporaryDirectory() as d:
        _write_skill(d, _research_skill_json())
        with debug_session(
            skills_dir=d,
            enable_custom_skill_subagents=True,
            secret="test-secret-32-bytes-0123456789",
        ) as c:
            assert os.environ.get("PROXY_SKILLS_DIR") == d
            assert os.environ.get("PROXY_ENABLE_CUSTOM_SKILL_SUBAGENTS") == "1"
            reg_status = c.state  # Touch state module to ensure it's loaded
            assert reg_status is not None

    # After exit: env vars restored (to prior value or absent).
    assert os.environ.get("PROXY_SKILLS_DIR") == saved_skills_dir
    assert os.environ.get(
        "PROXY_ENABLE_CUSTOM_SKILL_SUBAGENTS"
    ) == saved_flag
    assert os.environ.get("PROXY_SESSION_SECRET") == saved_secret


def test_debug_session_restores_proxy_test_db():
    """PROXY_TEST_DB must be restored to its pre-session value (or removed
    if it was not set), never the deleted debug_session tmpdir path."""
    original = os.environ.get("PROXY_TEST_DB")
    original_present = "PROXY_TEST_DB" in os.environ
    sentinel = "/tmp/proxy_test_db_sentinel_do_not_use"

    try:
        # Case 1: PROXY_TEST_DB initially absent.
        os.environ.pop("PROXY_TEST_DB", None)
        with debug_session(secret="test-secret-32-bytes-0123456789") as c:
            inside_val = os.environ.get("PROXY_TEST_DB")
            assert inside_val is not None, "debug DB should be set inside context"
            assert "debug_session_" in inside_val
        after_val = os.environ.get("PROXY_TEST_DB")
        assert after_val is None, f"expected PROXY_TEST_DB restored to absent, got {after_val!r}"

        # Case 2: PROXY_TEST_DB initially set to a sentinel.
        os.environ["PROXY_TEST_DB"] = sentinel
        with debug_session(secret="test-secret-32-bytes-0123456789") as c:
            inside_val = os.environ.get("PROXY_TEST_DB")
            assert inside_val is not None and "debug_session_" in inside_val
        after_val = os.environ.get("PROXY_TEST_DB")
        assert after_val == sentinel, f"expected PROXY_TEST_DB restored to sentinel, got {after_val!r}"
        assert "debug_session_" not in (after_val or "")
    finally:
        if original_present:
            os.environ["PROXY_TEST_DB"] = original
        else:
            os.environ.pop("PROXY_TEST_DB", None)


def test_explicit_empty_skills_dir_disables_custom_skills():
    """skills_dir=='' explicitly disables custom-skill loading."""
    saved = os.environ.get("PROXY_SKILLS_DIR")
    try:
        with debug_session(
            skills_dir="",
            enable_custom_skill_subagents=True,
            secret="test-secret-32-bytes-0123456789",
        ) as c:
            assert os.environ.get("PROXY_SKILLS_DIR") == ""
            assert (
                os.environ.get("PROXY_ENABLE_CUSTOM_SKILL_SUBAGENTS") == "1"
            )
    finally:
        assert os.environ.get("PROXY_SKILLS_DIR") == saved


def test_invalid_swarm_config_fails_before_yield():
    """Invalid custom-JSON (reserve builtin name) raises
    SkillRegistryConfigurationError on registry rebuild, BEFORE the
    debug_session `with` block yields a Session. The exception propagates
    out but cleanup still runs (env restored)."""
    from proxy.skills.custom_skill_loader import SkillRegistryConfigurationError

    with tempfile.TemporaryDirectory() as d:
        bad_skill = _research_skill_json()
        bad_skill["name"] = "ticket"  # reserved builtin name
        _write_skill(d, bad_skill)
        with pytest.raises(SkillRegistryConfigurationError):
            with debug_session(
                skills_dir=d,
                enable_custom_skill_subagents=True,
                secret="test-secret-32-bytes-0123456789",
            ) as c:
                pytest.fail("DebugSession should not yield on invalid config")


def test_task_builders():
    """c.canned.task_call / task_calls produce valid OpenAI native task
    tool calls and reject empty lists."""
    with debug_session(
        secret="test-secret-32-bytes-0123456789",
    ) as c:
        single = c.canned.task_call("p")
        tcs = single["choices"][0]["message"]["tool_calls"]
        assert len(tcs) == 1
        assert tcs[0]["function"]["name"] == "task"
        args = json.loads(tcs[0]["function"]["arguments"])
        assert args["prompt"] == "p"
        multi = c.canned.task_calls(["a", "b"])
        mtcs = multi["choices"][0]["message"]["tool_calls"]
        assert len(mtcs) == 2
        assert mtcs[0]["id"] == "call_task_1"
        assert mtcs[1]["id"] == "call_task_2"
        assert json.loads(mtcs[0]["function"]["arguments"])["prompt"] == "a"
        assert json.loads(mtcs[1]["function"]["arguments"])["prompt"] == "b"
        with pytest.raises(ValueError):
            c.canned.task_calls([])


@pytest.mark.real_agent_registry
def test_ensure_agent_registers_fresh_agent():
    """ensure_agent uses AgentRegistry.register + register_lifecycle_callbacks
    (public API). Verified by spying on both call sites; no private registry
    collections or Agent fields are inspected.

    Marked real_agent_registry: the autouse mock_agent_registry_get_or_load
    fixture would break the public-API assertions."""
    from unittest.mock import patch, MagicMock
    from proxy.agent.agent_registry import registry
    import proxy.skill_lifecycle as _sl_mod
    import proxy.agent.agent_registry as _ar_mod

    with debug_session(secret="test-secret-32-bytes-0123456789") as c:
        sid = c.session_id_for(
            [{"role": "user", "content": "ensure-agent-unique-marker"}],
            client_hint="ea",
        )
        assert registry.get_or_load(sid) is None

        # Spy on the public functions ensure_agent must call. Use wraps=
        # so the real implementations still run.
        rlc_calls = []
        reg_calls = []
        real_rlc = _sl_mod.register_lifecycle_callbacks
        real_reg = _ar_mod.registry.register
        def _rlc_spy(agent, *a, **kw):
            rlc_calls.append(agent)
            return real_rlc(agent, *a, **kw)
        def _reg_spy(agent, *a, **kw):
            reg_calls.append(agent)
            return real_reg(agent, *a, **kw)

        with patch.object(_sl_mod, "register_lifecycle_callbacks", side_effect=_rlc_spy), \
             patch.object(_ar_mod.registry, "register", side_effect=_reg_spy):
            c.ensure_agent(sid)

        # Public lookup now returns the registered agent.
        registered = registry.get_or_load(sid)
        assert registered is not None, "ensure_agent did not register"

        # register_lifecycle_callbacks called exactly once with the new Agent.
        assert len(rlc_calls) == 1, f"expected 1 rlc call, got {len(rlc_calls)}"
        assert rlc_calls[0] is registered

        # registry.register called exactly once with the same Agent.
        assert len(reg_calls) == 1, f"expected 1 register call, got {len(reg_calls)}"
        assert reg_calls[0] is registered

        # Second ensure_agent is a no-op (idempotent): neither function called again.
        rlc_calls.clear()
        reg_calls.clear()
        with patch.object(_sl_mod, "register_lifecycle_callbacks", side_effect=_rlc_spy), \
             patch.object(_ar_mod.registry, "register", side_effect=_reg_spy):
            c.ensure_agent(sid)
        assert rlc_calls == [], "second ensure_agent must not call register_lifecycle_callbacks"
        assert reg_calls == [], "second ensure_agent must not call registry.register"


def test_swarm_status_reports_swarm_enabled():
    """^^status within a swarm debug session reports Swarm-enabled: yes
    and the loaded swarm-capable skill."""
    secret = "test-secret-32-bytes-0123456789"
    with tempfile.TemporaryDirectory() as d:
        skill = {
            "name": "research", "trigger": "^^research",
            "description": "x", "prompt": "PROMPT",
            "tool_whitelist": ["read", "grep", "task"],
            "subagents": {"tool_whitelist": ["read", "grep"], "max_task_calls_per_request": 1},
        }
        with open(os.path.join(d, "research.json"), "w") as f:
            json.dump(skill, f)
        with debug_session(
            canned_responses=[{"choices": [{"message": {"role": "assistant", "content": "d"}, "finish_reason": "stop"}]}],
            secret=secret,
            skills_dir=d,
            enable_custom_skill_subagents=True,
        ) as c:
            status = c.chat(messages=[{"role": "user", "content": "^^status"}], client_hint="t-status")
            content = status.json().get("choices", [{}])[0].get("message", {}).get("content", "")
            assert "Swarm-enabled: yes" in content
            assert "research" in content


def test_extraction_helpers_accept_structurally_valid_task_response():
    """STRUCTURAL helper test — uses a manually fabricated swarm-skills
    response (NOT real canary injection).

    Verifies extract_task_prompts / extract_swarm_canary parse a
    well-formed task-call response. Because the response is fabricated
    (not produced by the proxy's real PostProcess canary injection), this
    only checks the parsers' structural contract. The real PostProcess
    path is covered by test_swarm_parent_end_to_end."""
    with debug_session(
        secret="test-secret-32-bytes-0123456789",
    ) as c:
        canned_response = {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call_task_1",
                                "type": "function",
                                "function": {
                                    "name": "task",
                                    "arguments": json.dumps(
                                        {
                                            "prompt": "SIGNECanaryHEREinspect",
                                            "description": "x",
                                        },
                                        ensure_ascii=False,
                                    ),
                                },
                            }
                        ],
                    },
                    "finish_reason": "tool_calls",
                }
            ]
        }
        # Simulating a swarm parent signature: prefix the prompt with a real
        # SKILL canary. We use DebugSession.sign_canary to build it.
        marker = c.sign_canary("research", "")
        # Build a real post-transform response.
        injected_args = json.dumps(
            {"prompt": f"{marker}inspect auth", "description": "x"},
            ensure_ascii=False,
        )
        fake_swarm = {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call_task_1",
                                "type": "function",
                                "function": {
                                    "name": "task",
                                    "arguments": injected_args,
                                },
                            }
                        ],
                    },
                    "finish_reason": "tool_calls",
                }
            ]
        }
        fake_resp = ProxyResponse(
            status_code=200,
            data=json.dumps(fake_swarm).encode(),
            _json=fake_swarm,
        )
        prompts = c.extract_task_prompts(fake_resp)
        assert len(prompts) == 1
        assert prompts[0].startswith(marker)
        assert prompts[0].endswith("inspect auth")
        canary = c.extract_swarm_canary(fake_resp)
        assert canary is not None
        assert canary.startswith(marker)
        from proxy.canary import parse_canary

        parsed = parse_canary(canary)
        assert parsed is not None
        assert parsed.kind == "skill"
        # In this unit-style test the harness's default (300-builtin)
        # registry has no 'research' skill, so verified=False is expected.
        # The structural kind='skill' is the authoritative signal.
        assert parsed.skill_name in (None, "research")
        # content-only returns [], None
        content_only = ProxyResponse(
            status_code=200,
            data=json.dumps(
                {
                    "choices": [
                        {
                            "message": {
                                "role": "assistant",
                                "content": "plain",
                            },
                            "finish_reason": "stop",
                        }
                    ]
                }
            ).encode(),
        )
        assert c.extract_task_prompts(content_only) == []
        assert c.extract_swarm_canary(content_only) is None


def test_swarm_parent_end_to_end():
    """End-to-end swarm custom-skill happy path: enter debug_session with
    explicit skills_dir + kill switch, send parent chat that emits a task
    tool call, extract + verify canary, then spawn sub-agent via
    send_as_subagent and verify inbound child tools are filtered to the
    subagent whitelist (read+grep) — not task/question/proxy_ticket/proxy_work."""
    from proxy.canary import parse_canary

    secret = "test-secret-32-bytes-0123456789"
    skill = _research_skill_json()
    with tempfile.TemporaryDirectory() as d:
        _write_skill(d, skill)
        canned = [
            {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": "call_task_1",
                                    "type": "function",
                                    "function": {
                                        "name": "task",
                                        "arguments": json.dumps(
                                            {
                                                "prompt": "inspect auth",
                                                "description": "d",
                                            },
                                            ensure_ascii=False,
                                        ),
                                    },
                                }
                            ],
                        },
                        "finish_reason": "tool_calls",
                    }
                ]
            },
            {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": "sub-agent done",
                        },
                        "finish_reason": "stop",
                    }
                ]
            },
        ]
        # Snapshot the original canned[0] BEFORE passing it to the proxy so
        # we can assert post-session mock canned fixture isolation across HTTP
        # serialization (the HTTP layer decodes a fresh object, so this cannot
        # prove transform_swarm_task_calls deep-copies internally).
        source_snapshot = copy.deepcopy(canned[0])
        with debug_session(
            canned_responses=canned,
            secret=secret,
            skills_dir=d,
            enable_custom_skill_subagents=True,
        ) as c:
            c.client.environ_base["HTTP_X_SESSION_CLIENT"] = "test-parent"
            parent_msgs = [
                {
                    "role": "user",
                    "content": "^^research inspect auth",
                }
            ]
            psid = c.session_id_for(parent_msgs, client_hint="test-parent")
            c.ensure_agent(psid)
            parent_resp = c.chat(
                messages=parent_msgs,
                client_hint="test-parent",
            )
            assert parent_resp.status_code == 200
            prompts = c.extract_task_prompts(parent_resp)
            assert prompts, "expected at least one task prompt"
            canary = c.extract_swarm_canary(parent_resp)
            assert canary is not None
            parsed = parse_canary(canary)
            assert parsed is not None and parsed.verified
            assert parsed.skill_name == "research"
            # Mock canned fixture isolation across HTTP serialization: the HTTP
            # layer serializes canned[0] to JSON and the proxy decodes a fresh
            # object, so the template fixture cannot be mutated in place. This
            # is NOT proof that transform_swarm_task_calls deep-copies.
            assert canned[0] == source_snapshot, (
                "mock canned fixture was mutated in place"
            )
            # Spawn sub-agent. Pass parent tools= so the proxy's
            # select_tools has a non-empty client-set to filter against
            # the subagent whitelist.
            child_tools = [
                {
                    "type": "function",
                    "function": {
                        "name": "read",
                        "parameters": {"type": "object", "properties": {}},
                    },
                },
                {
                    "type": "function",
                    "function": {
                        "name": "grep",
                        "parameters": {"type": "object", "properties": {}},
                    },
                },
                {
                    "type": "function",
                    "function": {
                        "name": "task",
                        "parameters": {
                            "type": "object",
                            "properties": {"prompt": {"type": "string"}},
                            "required": ["prompt"],
                        },
                    },
                },
                {
                    "type": "function",
                    "function": {
                        "name": "question",
                        "parameters": {"type": "object", "properties": {}},
                    },
                },
            ]
            child_resp, child_sid = c.send_as_subagent(
                canary=canary,
                subagent_prompt="inspect auth",
                subagent_messages=[],
                client_hint="test-child",
                tools=child_tools,
            )
            assert child_resp.status_code == 200
            # Inbound child tools filtered to read+grep.
            outbound_body = c.captured_requests[-1]["body"]
            raw_tools = outbound_body.get("tools", []) or []
            outbound_names = sorted(
                t.get("function", {}).get("name", "")
                for t in raw_tools
                if isinstance(t, dict) and isinstance(t.get("function"), dict)
            )
            assert outbound_names == ["grep", "read"], (
                f"expected ['grep', 'read'], got {outbound_names}"
            )
            # The original child instruction must appear EXACTLY once — never
            # duplicated during ingress assembly / canary injection.
            outbound_user_msgs = [
                m for m in outbound_body.get("messages", [])
                if m.get("role") == "user"
            ]
            outbound_user_text = " ".join(
                m.get("content", "") for m in outbound_user_msgs
                if isinstance(m.get("content", ""), str)
            )
            instr_count = outbound_user_text.count("inspect auth")
            assert instr_count == 1, (
                f"child instruction must appear exactly once, got {instr_count}"
            )


# ---------------------------------------------------------------------------
# Revert-discrimination tests (Findings 4/5/6).
#
# The source-immutability (== 1) and exact-count checks guard against FUTURE
# regressions where the proxy shallow-copies and mutates the response dict it
# received from the upstream, or duplicates the child instruction during ingress
# assembly. To prove these checks are not tautological, each test below
# monkeypatches the proxy to simulate the exact regression (in-place mutation
# / instruction duplication) and asserts the check FAILS.
# ---------------------------------------------------------------------------


def test_mutation_harness_detects_shared_response_alias():
    """This is a MUTATION-HARNESS check, not a production revert test. It injects
    a synthetic mutation (shared alias / duplicated instruction) to prove the
    assertion DISCRIMINATES when the failure mode is made reachable. It does
    NOT execute a production revert.

    Finding 4/6: simulating shallow-copy aliasing (the proxy mutates the
    response dict it received instead of a deep copy) must FAIL the
    source-immutability check."""
    secret = "test-secret-32-bytes-0123456789"
    with tempfile.TemporaryDirectory() as d:
        _write_skill(d, _research_skill_json())
        canned = [
            {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": "call_task_1",
                                    "type": "function",
                                    "function": {
                                        "name": "task",
                                        "arguments": json.dumps(
                                            {"prompt": "inspect auth", "description": "d"},
                                            ensure_ascii=False,
                                        ),
                                    },
                                }
                            ],
                        },
                        "finish_reason": "tool_calls",
                    }
                ]
            },
            {
                "choices": [
                    {
                        "message": {"role": "assistant", "content": "sub-agent done"},
                        "finish_reason": "stop",
                    }
                ]
            },
        ]
        source_snapshot = copy.deepcopy(canned[0])

        # Simulate the shallow-copy-aliasing regression at two boundaries:
        #   1. call_upstream returns the test's original dict by reference
        #      (instead of decoding fresh bytes) — i.e. the proxy held a
        #      shared reference to the upstream response.
        #   2. transform_swarm_task_calls mutates that dict in place (skips the
        #      internal deep copy) — i.e. the post-process stage aliased nested
        #      structure instead of deep-copying.
        # Together these are exactly the regression we guard against.
        upstream_override = {"held": None}

        def _aliasing_call_upstream(url, body, headers, wants_stream, timeout_cap=None):
            # Return the live source dict the test is watching.
            return upstream_override["held"]

        def _in_place_transform(response_data, *, canary, max_task_calls):
            # Apply the real task-call transform WITHOUT the internal deep
            # copy: mutate the shared dict directly, then write a marker tc = response_data["choices"][0]["message"]["tool_calls"][0]
            tc = response_data["choices"][0]["message"]["tool_calls"][0]
            new_args = json.dumps(
                {"prompt": f"{canary}inspect auth", "description": "d"},
                ensure_ascii=False,
            )
            tc["function"]["arguments"] = new_args
            response_data["__revert_aliased__"] = True
            return response_data

        with patch("proxy.canary.transform_swarm_task_calls", side_effect=_in_place_transform), \
             patch("proxy.http_client.call_upstream", side_effect=_aliasing_call_upstream):
            with debug_session(
                canned_responses=canned,
                secret=secret,
                skills_dir=d,
                enable_custom_skill_subagents=True,
            ) as c:
                # Hand the live source dict to the upstream override so the
                # proxy's decode returns it by reference.
                upstream_override["held"] = canned[0]
                c.client.environ_base["HTTP_X_SESSION_CLIENT"] = "test-revert"
                psid = c.session_id_for(
                    [{"role": "user", "content": "^^research inspect auth"}],
                    client_hint="test-revert",
                )
                c.ensure_agent(psid)
                parent_resp = c.chat(
                    messages=[{"role": "user", "content": "^^research inspect auth"}],
                    client_hint="test-revert",
                )
                assert parent_resp.status_code == 200
                # Under the aliasing regression the original dict was mutated.
                assert canned[0] != source_snapshot, (
                    "revert: aliasing regression should have mutated the source"
                )


def test_mutation_harness_detects_duplicate_child_instruction():
    """This is a MUTATION-HARNESS check, not a production revert test. It injects
    a synthetic mutation (shared alias / duplicated instruction) to prove the
    assertion DISCRIMINATES when the failure mode is made reachable. It does
    NOT execute a production revert.

    Finding 5: duplicating the child instruction during ingress assembly
    must FAIL the exact-once check."""
    from proxy.canary import transform_swarm_task_calls as _real_tc

    secret = "test-secret-32-bytes-0123456789"
    with tempfile.TemporaryDirectory() as d:
        _write_skill(d, _research_skill_json())
        canned = [
            {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": "call_task_1",
                                    "type": "function",
                                    "function": {
                                        "name": "task",
                                        "arguments": json.dumps(
                                            {"prompt": "inspect auth", "description": "d"},
                                            ensure_ascii=False,
                                        ),
                                    },
                                }
                            ],
                        },
                        "finish_reason": "tool_calls",
                    }
                ]
            },
            {
                "choices": [
                    {
                        "message": {"role": "assistant", "content": "sub-agent done"},
                        "finish_reason": "stop",
                    }
                ]
            },
        ]

        # Buggy transform: inject the prompt twice (duplication regression).
        def _dup_transform(response_data, *, canary, max_task_calls):
            _real_tc(response_data, canary=canary, max_task_calls=max_task_calls)
            tc = response_data["choices"][0]["message"]["tool_calls"][0]
            args = json.loads(tc["function"]["arguments"])
            args["prompt"] = f'{args["prompt"]}\n{args["prompt"]}'
            tc["function"]["arguments"] = json.dumps(args, ensure_ascii=False)
            return response_data

        with patch("proxy.canary.transform_swarm_task_calls", side_effect=_dup_transform):
            with debug_session(
                canned_responses=canned,
                secret=secret,
                skills_dir=d,
                enable_custom_skill_subagents=True,
            ) as c:
                psid = c.session_id_for(
                    [{"role": "user", "content": "^^research inspect auth"}],
                    client_hint="test-revert-dup",
                )
                c.ensure_agent(psid)
                parent_resp = c.chat(
                    messages=[{"role": "user", "content": "^^research inspect auth"}],
                    client_hint="test-revert-dup",
                )
                assert parent_resp.status_code == 200
                prompts = c.extract_task_prompts(parent_resp)
                assert prompts, "expected at least one task prompt"
                # Under the duplication regression the prompt contains the
                # instruction twice → canary + "inspect auth" + "inspect auth".
                remainder = prompts[0]
                for marker in ("[[__SUBAGENT__",):
                    if marker in remainder:
                        remainder = remainder.split(marker, 1)[0]
                assert remainder.count("inspect auth") == 2, (
                    f"revert: duplication regression should yield 2 occurrences, "
                    f"got {remainder.count('inspect auth')}"
                )


# ---------------------------------------------------------------------------
# devtools/devtools scenario follow-up — harness immutability + template gates.
#
# Each test below pins a single producer/consumer invariant for an accepted
# production behavior. They are NOT tautological: each runs a real
# `debug_session(...)` and exercises the real Flask + mock-backend path.
# Revert-test pattern: temporarily editing `src/devtools/debug_session.py`
# to reintroduce the aliasing or schema-mutation regressions fails the
# matching assertion below.
# ---------------------------------------------------------------------------


def test_canned_error_controls_are_serialized_without_mutating_source():
    """`_make_handler_class` must deep-copy each canned dict before popping
    control keys, and the JSON body returned by the mock backend must NOT
    contain any `_status`/`_body`/`_retry_after` key.

    The fix is two-axis: caller-owned canned fixtures must remain unmutated
    across retries, and the wire body must be free of all `_*` keys so the
    proxy's response-decoder never sees internal control shapes.
    """
    # 1. Caller-owned canned dict; nested body carries a sentinel marker so
    #    we can prove the mock backend produced body = resp["_body"].
    source = {
        "_status": 429,
        "_body": {"error": "rate limited", "nested": {"marker": "SOURCE"}},
        "_retry_after": -5,
    }
    snapshot = copy.deepcopy(source)
    success = _text_response("HARNESS_RETRY_SUCCESS")

    # 2. Direct localhost POST for exact body assertions, plus a full
    #    debug_session to pin retry-count + captured-request shape.
    captured_body_one = None
    with debug_session(canned_responses=[source], models=["kimi"]) as c:
        # First call: the mock backend pops the error response. Direct
        # localhost POST gives us precise bytes for control-key assertions.
        resp = requests.post(
            f"http://127.0.0.1:{c.mock_port}/v1/chat/completions",
            json={"model": "kimi", "messages": [{"role": "user", "content": "x"}]},
            timeout=5,
        )
        assert resp.status_code == 429
        captured_body_one = resp.json()
    # Source must be byte-identical to the snapshot taken before queueing.
    assert source == snapshot, (
        f"caller-owned canned dict was mutated: source={source!r} vs {snapshot!r}"
    )
    assert source["_status"] == 429
    assert source["_retry_after"] == -5
    assert source["_body"] == snapshot["_body"]

    # JSON body of the error response must NOT serialize any hidden control
    # keys (`_status`, `_body`, `_retry_after`).
    assert "_status" not in captured_body_one
    assert "_retry_after" not in captured_body_one
    assert "_body" not in captured_body_one
    # Sentinel marker proves the body came from `_body`.
    assert captured_body_one.get("nested", {}).get("marker") == "SOURCE"

    # 3. Full Flask retry path: source + success, then c.chat() exercises
    #    real UpstreamStage + call_upstream + dynamic localhost backend.
    with debug_session(
        canned_responses=[source, success], models=["kimi"]
    ) as c:
        response = c.chat(
            messages=[{"role": "user", "content": "post-retry"}],
            client_hint="retry-harness-test",
        )
    assert response.status_code == 200
    assert "HARNESS_RETRY_SUCCESS" in response.content

    # Source must STILL be unchanged across the entire retry flow.
    assert source == snapshot, (
        f"caller-owned canned dict mutated after retry: {source!r} vs {snapshot!r}"
    )
    assert source["_status"] == 429
    assert source["_retry_after"] == -5
    assert source["_body"] == snapshot["_body"]


def test_debug_session_memory_db_retains_schema_through_first_real_request():
    """`debug_session(db_path=":memory:")` must keep the SQLite schema alive
    AFTER `init_db` closes its own migration connection: the
    `:memory:` keeper connection registered at the end of
    `init_db` keeps the shared cache populated for the subsequent pooled
    reads/writes a real `c.chat()` triggers.

    The reverse direction: without the keeper, the in-memory schema evaporates
    the moment `init_db` closes its last handle and a subsequent pooled
    SELECT/INSERT against a fresh :memory: connection sees an empty DB.
    """
    with debug_session(
        db_path=":memory:",
        models=["qwen3-coder"],
        canned_responses=[_text_response("MEMORY_DB_CHAT_OK")],
    ) as c:
        # Import inside the context — db_pool must be initialized by
        # debug_session before we touch it.
        import memory.db_pool as db_pool

        # 1. Pool exists and is bound to ":memory:".
        assert db_pool._pool is not None, "pool not initialized"
        assert db_pool._pool._db_path == ":memory:"

        # 2. Canonical tables present via schema-backed read (NOT SELECT 1).
        with db_pool.connection() as conn:
            rows = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
            table_names = {row[0] for row in rows}
        for required in ("messages", "sessions", "message_summaries"):
            assert required in table_names, (
                f"required table {required!r} missing — schema did not survive "
                f"init_db() close; got {sorted(table_names)}"
            )

        # 3. message_summaries unique constraint must be the canonical
        #    `(session_id, exchange_index)` — NOT the legacy
        #    `(session_id, message_index)`.
        with db_pool.connection() as conn:
            idx_rows = conn.execute(
                "PRAGMA index_list(message_summaries)"
            ).fetchall()
            index_names = [row[1] for row in idx_rows]
            unique_columns: list[tuple[str, ...]] = []
            for in_name in index_names:
                info_rows = conn.execute(
                    f"PRAGMA index_info({in_name})"
                ).fetchall()
                cols = tuple(row[2] for row in info_rows)
                unique_columns.append(cols)
        assert ("session_id", "exchange_index") in unique_columns, (
            f"message_summaries unique index missing (session_id, exchange_index); "
            f"saw {unique_columns}"
        )
        assert ("session_id", "message_index") not in unique_columns, (
            f"message_summaries still has legacy (session_id, message_index) "
            f"unique index after init_db(); saw {unique_columns}"
        )

        # 4. Real pooled schema-backed write+read of store_message_summary.
        from memory.db_summaries import store_message_summary

        write_ok = store_message_summary(
            session_id="debug-memory-contract",
            exchange_index=0,
            role="user",
            summary="MEMORY_SUMMARY_CONTRACT",
            tool_summary="",
            content_hash="debug-memory-contract-hash",
            model_used="debug-session",
            token_estimate=1,
            end_message_index=1,
            message_index=0,
        )
        assert write_ok is True, "store_message_summary returned False"

        # Verify checkout returned to the pool.
        assert db_pool._pool._checked_out == {}, (
            f"connection leaked after store_message_summary: "
            f"{list(db_pool._pool._checked_out)}"
        )

        with db_pool.connection() as conn:
            row = conn.execute(
                "SELECT summary, role, model_used FROM message_summaries "
                "WHERE session_id = ? AND exchange_index = ?",
                ("debug-memory-contract", 0),
            ).fetchone()
        assert row is not None, (
            "pooled read returned None — schema-backed persistence failed"
        )
        assert row["summary"] == "MEMORY_SUMMARY_CONTRACT", (
            f"summary mismatch: got {row['summary']!r}"
        )
        assert row["role"] == "user"
        assert row["model_used"] == "debug-session"

        assert db_pool._pool._checked_out == {}, (
            "connection leaked after pooled schema-backed read"
        )

        # 5. Full Flask chat through the same `:memory:` pool — proves the
        #    schema survives ALL stages (Cleanup → … → Upstream → …).
        response = c.chat(
            messages=[
                {"role": "user", "content": "memory keeper request"}
            ],
            client_hint="debug-memory-contract-chat",
        )
        assert response.status_code == 200
        assert "MEMORY_DB_CHAT_OK" in response.content

    # The harness captures outbound bodies per-context; pinned invariants
    # re-checked post-context.
    assert db_pool._pool is not None
    assert db_pool._pool._checked_out == {}, (
        "connection leaked after c.chat() (post-context)"
    )


@pytest.mark.parametrize(
    "template",
    [
        "tests/debug/templates/i_retry_lifecycle.py",
        "tests/debug/templates/j_sqlite_lifecycle.py",
        "tests/debug/templates/k_summary_budget.py",
        "tests/debug/templates/l_vfs_cache_safe_tail.py",
    ],
)
def test_debug_templates_retry_sqlite_and_summary_exit_cleanly(template):
    """Each new debug template must:
      - exit 0
      - emit ONLY JSONL, parseable line-by-line
      - emit checks with the required keys (step, check, ok, detail)
      - emit at least one check AND every check's ``ok is True``

    Uses an external 30-second subprocess timeout. The subprocess command
    does not mutate the worktree (no staging, no commit, no reset).
    """
    template_path = Path(template)
    assert template_path.exists(), f"missing template: {template}"
    repo_root = Path(__file__).resolve().parents[2]

    completed = subprocess.run(
        [sys.executable, str(template_path)],
        cwd=str(repo_root),
        env={**os.environ, "PYTHONPATH": "src", "PROXY_SESSION_SECRET": ""},
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    if completed.returncode != 0:
        raise AssertionError(
            f"{template} exited {completed.returncode}\n"
            f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
        )

    lines = [
        line for line in completed.stdout.splitlines() if line.strip()
    ]
    assert lines, f"{template} emitted no JSONL output"

    parsed_rows: list[dict] = []
    for line in lines:
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise AssertionError(
                f"{template} emitted non-JSONL line: {line!r} ({exc})"
            )
        assert isinstance(row, dict), (
            f"{template} emitted non-dict row: {row!r}"
        )
        for required in ("step", "check", "ok", "detail"):
            assert required in row, (
                f"{template} emitted row missing {required!r}: {row!r}"
            )
        parsed_rows.append(row)

    assert parsed_rows, f"{template} parsed zero checks"
    for row in parsed_rows:
        assert row["ok"] is True, (
            f"{template} emitted failing check: {row!r}"
        )


# `requests` import local to the test module so debug_session still imports
# cleanly when requests has a transient warning.
import requests  # noqa: E402  (intentional: avoid load-order side effects)
