"""Unit tests for `^^context` command + context-budget override resolver."""
from types import SimpleNamespace

import pytest

from proxy.context import resolve_effective_context_limit
from proxy.model_registry import get_context_config, model_config_cache


class Cfg:
    context_limit = 128000
    budget_ratio = 0.75
    raw_exchange_count = 8


# ---------------------------------------------------------------------------
# resolve_effective_context_limit
# ---------------------------------------------------------------------------


def test_resolve_no_override_returns_real():
    assert resolve_effective_context_limit(Cfg, None) == 128000


def test_resolve_override_below_real():
    assert resolve_effective_context_limit(Cfg, 50000) == 50000


def test_resolve_override_above_real():
    assert resolve_effective_context_limit(Cfg, 200000) == 128000


def test_resolve_override_no_model_cfg():
    assert resolve_effective_context_limit(None, 300000) == 300000


@pytest.mark.parametrize("real", [0, None])
def test_resolve_override_no_real_limit(real):
    cfg = SimpleNamespace(context_limit=real)
    assert resolve_effective_context_limit(cfg, 300000) == 300000


# ---------------------------------------------------------------------------
# get_context_config with override
# ---------------------------------------------------------------------------


def test_get_context_config_override_reduces_safe_tokens():
    config = get_context_config(
        "model-override-reduce",
        model_cfg=Cfg,
        context_limit_override=50000,
    )
    assert config["safe_context_tokens"] == int(50000 * 0.75) == 37500


def test_get_context_config_override_no_cache_write():
    before = dict(model_config_cache)
    config = get_context_config(
        "model-override-no-cache",
        model_cfg=Cfg,
        context_limit_override=50000,
    )
    after = dict(model_config_cache)
    assert before == after
    assert config["safe_context_tokens"] == 37500


def test_get_context_config_none_override_caches():
    model_name = "model-none-override-caches"
    get_context_config(model_name, model_cfg=Cfg, context_limit_override=None)
    assert model_name in model_config_cache


def test_get_context_config_override_non_override_call_unaffected():
    model_name = "model-override-then-none"
    get_context_config(model_name, model_cfg=Cfg, context_limit_override=50000)
    config = get_context_config(model_name, model_cfg=Cfg, context_limit_override=None)
    assert config["safe_context_tokens"] == int(128000 * 0.75) == 96000


def test_get_context_config_override_above_real_clamps():
    """override > real must clamp at the real model limit, not the override.
    Pin: without resolve_effective_context_limit's min() the safe budget
    would be derived from 200000 instead of 128000."""
    Cfg = type("Cfg", (), {"context_limit": 128000, "budget_ratio": 0.75, "raw_exchange_count": 0})()
    cfg = get_context_config("clamp-model", model_cfg=Cfg, context_limit_override=200000)
    assert cfg["safe_context_tokens"] == int(128000 * 0.75)


# ---------------------------------------------------------------------------
# Agent accessors
# ---------------------------------------------------------------------------


def test_agent_context_limit_override_round_trip():
    from proxy.agent.agent import Agent
    agent = Agent(session_id="t1")
    assert agent.get_context_limit_override() is None
    agent.set_context_limit_override(500000)
    assert agent.get_context_limit_override() == 500000


def test_agent_context_limit_override_overwrite():
    from proxy.agent.agent import Agent
    agent = Agent(session_id="t2")
    agent.set_context_limit_override(500000)
    agent.set_context_limit_override(300000)
    assert agent.get_context_limit_override() == 300000


# ---------------------------------------------------------------------------
# cmd_context
# ---------------------------------------------------------------------------


def _body(resp):
    return resp["choices"][0]["message"]["content"]


def test_cmd_context_valid_500():
    from commands import CommandContext, dispatch_command
    resp = dispatch_command(CommandContext(raw_text="^^context 500", session_id="sess-1"))
    body = _body(resp)
    assert "500,000 tokens" in body


def test_cmd_context_non_numeric():
    from commands import CommandContext, dispatch_command
    resp = dispatch_command(CommandContext(raw_text="^^context abc", session_id="sess-2"))
    body = _body(resp)
    assert "Invalid" in body or "integer" in body


def test_cmd_context_too_small_199():
    from commands import CommandContext, dispatch_command
    resp = dispatch_command(CommandContext(raw_text="^^context 199", session_id="sess-3"))
    body = _body(resp)
    assert "out of range" in body
    assert "199" in body


def test_cmd_context_too_large_1001():
    from commands import CommandContext, dispatch_command
    resp = dispatch_command(CommandContext(raw_text="^^context 1001", session_id="sess-4"))
    body = _body(resp)
    assert "out of range" in body
    assert "1001" in body


def test_cmd_context_no_arg():
    from commands import CommandContext, dispatch_command
    resp = dispatch_command(CommandContext(raw_text="^^context", session_id="sess-5"))
    body = _body(resp)
    assert "model default" in body or "no override" in body


def test_cmd_context_extra_arg():
    from commands import CommandContext, dispatch_command
    resp = dispatch_command(CommandContext(raw_text="^^context 500 extra", session_id="sess-6"))
    body = _body(resp)
    assert "Usage" in body or "^^context" in body
