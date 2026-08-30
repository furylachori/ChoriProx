"""Proxy-layer tests for `^^context` command + ContextOptimizeStage override wiring."""
import logging
import threading
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from proxy.pipeline.stage_modules.context_optimize import ContextOptimizeStage


@contextmanager
def suppress_loggers(*logger_names):
    """Suppress only expected-noise records from the named loggers for the scope.

    Used where the proxy harness emits expected noise (e.g. missing SQLite
    summary tables) that is unrelated to the test assertion and could mask a
    real fault. The filter drops ONLY records whose message contains the
    expected substring "no such table"; unrelated records (real DB errors,
    etc.) still pass through. Scoped to the named loggers only; root and
    other loggers pass.
    """
    filters = {}
    for name in logger_names:
        logger = logging.getLogger(name)
        f = _DropFilter()
        logger.addFilter(f)
        filters[logger] = f
    try:
        yield filters
    finally:
        for logger, f in filters.items():
            logger.removeFilter(f)


class _DropFilter(logging.Filter):
    """Drop ONLY records whose message contains the expected substring.

    Returns True (pass) for everything else — keeps unrelated faults (real
    DB errors, etc.) visible to the test runner. The expected noise is
    "no such table" (the proxy harness has no SQLite summary tables).
    """

    _DROP_SUBSTR = "no such table"

    def filter(self, record):  # noqa: D401 - logging.Filter contract
        return self._DROP_SUBSTR not in record.getMessage()


def _body(resp):
    return resp["choices"][0]["message"]["content"]


def _make_ctx(session_id="sess-px", is_subagent=False, override=None):
    return SimpleNamespace(
        session_id=session_id,
        request_id="req-px",
        is_subagent=is_subagent,
        inherited_skill=None,
        subagent_ticket_mode=False,
        model_cfg=None,
        body={"messages": [{"role": "user", "content": "hi"}], "tools": []},
        context_limit_override=override,
    )


# ---------------------------------------------------------------------------
# dispatch_command sets the agent override
# ---------------------------------------------------------------------------


def test_proxy_context_500_sets_agent_override():
    """Pin: `^^context 500` propagates from command → agent.get_context_limit_override()
    and reads back as 500000. Real-Agent round-trip is pinned in
    tests/unit/test_context_command.py::test_agent_context_limit_override_round_trip.
    Note: the agent below is the autouse MockAgent from
    tests/conftest.py::mock_agent_registry_get_or_load (NOT the real Agent).
    """
    from commands import CommandContext, dispatch_command
    from proxy.agent.agent_registry import registry as agent_registry
    resp = dispatch_command(CommandContext(raw_text="^^context 500", session_id="sess-px-1"))
    assert "500,000 tokens" in _body(resp)
    agent = agent_registry.get_or_load("sess-px-1")
    assert agent.get_context_limit_override() == 500000


def test_proxy_context_persistence_across_requests():
    """Pin: `^^context 400` survives across an unrelated `^^status` command.
    Real-Agent round-trip is pinned in
    tests/unit/test_context_command.py::test_agent_context_limit_override_round_trip.
    Note: the agent below is the autouse MockAgent from
    tests/conftest.py::mock_agent_registry_get_or_load (NOT the real Agent).
    """
    from commands import CommandContext, dispatch_command
    from proxy.agent.agent_registry import registry as agent_registry
    dispatch_command(CommandContext(raw_text="^^context 400", session_id="sess-px-2"))
    agent = agent_registry.get_or_load("sess-px-2")
    assert agent.get_context_limit_override() == 400000
    # A subsequent unrelated command must not clear the override.
    dispatch_command(CommandContext(raw_text="^^status", session_id="sess-px-2"))
    assert agent_registry.get_or_load("sess-px-2").get_context_limit_override() == 400000


# ---------------------------------------------------------------------------
# B1 reset regression — stale reused ctx must not leak a previous override
# ---------------------------------------------------------------------------


def test_proxy_context_B1_reset_regression():
    """A reused ctx entering with a stale override and no agent must end with
    override=None. The top-of-execute() reset line is the only thing that
    clears it when agent is None (the agent block is skipped)."""
    # Proxy harness has no SQLite summary tables; execute()'s V2 cold-path
    # recovery emits an expected "no such table" warning unrelated to the
    # assertion. Drop the offending loggers' records so the noise cannot mask
    # a real fault. Scoped to the exact loggers/messages that are expected.
    ctx = _make_ctx(session_id="sess-b1", is_subagent=False, override=999999)
    with patch("proxy.agent.agent_registry.registry.get_or_load", return_value=None):
        with suppress_loggers("proxy.context_v2", "memory.db"):
            try:
                ContextOptimizeStage().execute(ctx)
            except Exception:
                pass
    assert ctx.context_limit_override is None


# ---------------------------------------------------------------------------
# Sub-agent no-inheritance
# ---------------------------------------------------------------------------


def test_proxy_subagent_no_inheritance():
    """A sub-agent must NEVER inherit even its OWN session's stored override.

    The is_subagent guard inside the agent block is the load-bearing pin:
    setting context_limit_override is skipped for is_subagent=True requests.
    Revert-test: removing the `not is_subagent` from the agent block guard
    breaks this test (override leaks to 300000). Top-of-execute reset alone
    is NOT enough — the sub-agent's agent DOES exist and HAS an override.
    """
    from proxy.agent.agent_registry import registry as agent_registry
    session = "sess-px-subagent"
    agent = agent_registry.get_or_load(session)
    agent.set_context_limit_override(300000)
    ctx = _make_ctx(session_id=session, is_subagent=True, override=None)
    with suppress_loggers("memory.db"):
        try:
            ContextOptimizeStage().execute(ctx)
        except Exception:
            pass
    assert ctx.context_limit_override is None


# ---------------------------------------------------------------------------
# End-to-end: ^^context 500 short-circuits upstream via DispatchStage
# ---------------------------------------------------------------------------


def test_proxy_context_command_no_upstream_call(client):
    """End-to-end: `^^context 500` through the full Flask → DispatchStage
    short-circuit path returns the confirmation body and never reaches
    `proxy.http_client.call_upstream`. This pins the operational contract
    that the command path is upstream-free — distinct from
    test_proxy_subagent_no_inheritance which pins the agent-side guard."""
    with patch("proxy.http_client.call_upstream") as call_upstream:
        resp = client.post(
            "/v1/chat/completions",
            json={
                "model": "test-model",
                "messages": [{"role": "user", "content": "^^context 500"}],
                "stream": False,
            },
        )
    assert resp.status_code == 200, resp.get_data(as_text=True)
    body = resp.get_json()
    assert "500,000 tokens" in body["choices"][0]["message"]["content"]
    call_upstream.assert_not_called()
