"""M6 #2: ^^imp rate limit gate.

Pre-fix: try_enhancer in src/proxy/pipeline/dispatch.py had no
rate limit. Misbehaving clients could flood SOTA with ^^imp
requests, bypassing the 5/hr budget on the canary escalation path
and saturating the SOTA circuit breaker.

Fix: add enhancer_rate_limiter (500/60s/persist) to state.py
mirroring command_rate_limiter. try_enhancer now gates with
check_only() (so floods without ^^imp don't consume slots) and
commits the slot only after is_imp_trigger matches.
"""

import json
from unittest.mock import MagicMock, patch

import pytest


pytestmark = pytest.mark.unit


def _make_ctx(session_id="sess-enhancer-rl", text="^^imp summarize this"):
    """Build a minimal context object for try_enhancer."""
    ctx = MagicMock()
    ctx.session_id = session_id
    ctx.wants_stream = False
    ctx.messages = [{"role": "user", "content": text}]
    return ctx


def _features(enhancer_enabled=True):
    f = MagicMock()
    f.enhancer_enabled = enhancer_enabled
    return f


def _patch_enhancer(monkeypatch):
    """Patch the enhancer module functions used inside try_enhancer."""
    import enhancer as _enh
    monkeypatch.setattr(_enh, "is_imp_trigger", lambda text: True)
    monkeypatch.setattr(_enh, "dispatch_imp", lambda sid, text: {"choices": [{"message": {"role": "assistant", "content": "ok"}}]})
    return _enh


def _patch_enhancer_no_trigger(monkeypatch):
    """Patch is_imp_trigger to return False (no ^^imp match)."""
    import enhancer as _enh
    monkeypatch.setattr(_enh, "is_imp_trigger", lambda text: False)
    monkeypatch.setattr(_enh, "dispatch_imp", lambda sid, text: None)
    return _enh


def test_enhancer_rate_limit_kicks_in_at_501(monkeypatch):
    """After 500 successful ^^imp dispatches in a window, the
    501st call returns 429."""
    import proxy.state as state
    state.enhancer_rate_limiter.reset()

    from proxy.pipeline.dispatch import try_enhancer
    _patch_enhancer(monkeypatch)

    ctx = _make_ctx(session_id="sess-rl-501")
    features = _features()

    for i in range(500):
        result = try_enhancer(ctx, features)
        assert result is not None, f"call {i} should return a DispatchResult"
        assert result.status_code == 200, f"call {i} should succeed (200), got {result.status_code}"

    result = try_enhancer(ctx, features)
    assert result is not None, "501st call should return a DispatchResult (not None)"
    assert result.status_code == 429, (
        f"501st call should be rate-limited (429), got {result.status_code}"
    )
    assert result.handled is True
    body = result.response
    assert body["error"]["type"] == "rate_limit_exceeded"
    assert "retry_after" in body["error"]


def test_enhancer_rate_limit_resets_after_reset(monkeypatch):
    """After reset(), the limiter accepts new dispatches again."""
    import proxy.state as state
    state.enhancer_rate_limiter.reset()

    from proxy.pipeline.dispatch import try_enhancer
    _patch_enhancer(monkeypatch)

    ctx = _make_ctx(session_id="sess-rl-reset")
    features = _features()

    for i in range(500):
        try_enhancer(ctx, features)
    result = try_enhancer(ctx, features)
    assert result.status_code == 429

    state.enhancer_rate_limiter.reset()
    result = try_enhancer(ctx, features)
    assert result.status_code == 200


def test_non_imp_text_does_not_consume_slot(monkeypatch):
    """A flood of non-^^imp messages should NOT consume rate-limit
    slots. The check_only() pattern in try_enhancer only commits the
    slot after is_imp_trigger() matches.
    """
    import proxy.state as state
    state.enhancer_rate_limiter.reset()

    from proxy.pipeline.dispatch import try_enhancer
    _patch_enhancer_no_trigger(monkeypatch)

    ctx = _make_ctx(text="just a normal message", session_id="sess-no-slot")
    features = _features()

    for _ in range(1000):
        result = try_enhancer(ctx, features)
        assert result is None, (
            "Non-^^imp text should return None (not a trigger), "
            "not consume a rate-limit slot."
        )

    _patch_enhancer(monkeypatch)
    ctx2 = _make_ctx(text="^^imp do something", session_id="sess-no-slot")
    result = try_enhancer(ctx2, features)
    assert result is not None
    assert result.status_code == 200, (
        f"After 1000 non-imp calls, a real ^^imp should still succeed. "
        f"Got status {result.status_code}. The pre-fix bug would have "
        f"consumed a slot per non-trigger call, exhausting the budget."
    )


def test_rate_limit_in_2xx_band_does_not_block_normal_path(monkeypatch):
    """Sanity: a small number of ^^imp dispatches (under 500) all succeed."""
    import proxy.state as state
    state.enhancer_rate_limiter.reset()

    from proxy.pipeline.dispatch import try_enhancer
    _patch_enhancer(monkeypatch)

    ctx = _make_ctx(session_id="sess-small")
    features = _features()

    for i in range(50):
        result = try_enhancer(ctx, features)
        assert result.status_code == 200, f"call {i} should succeed (200)"
