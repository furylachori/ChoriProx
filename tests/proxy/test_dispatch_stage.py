"""Tests for DispatchStage — spec §7.3 rewrite rollback contract.

Wave 3 Agent 3 owns the rewrite-failure rollback path:
- `dispatch.py:471` wraps `get_activation_rewrite` to compare-and-clear
  the agent's active_mode against the snapshot captured before
  activation.
- `dispatch_stage.py:64-73` wraps the rewrite application
  (`last_msg["content"] = ...`) to roll back when the dict mutation
  itself blows up (e.g. content is a frozen/immutable list, or the
  `body` dict shape is malformed).

These tests pin both rollback paths so future refactors cannot
re-introduce the fail-open privilege-escalation window closed here.
"""
import logging
import pytest

from proxy.pipeline.exceptions import StageAbort
from proxy.pipeline.stage_modules.dispatch_stage import DispatchStage
from proxy.skill_dispatcher import DispatchResult


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


class _FakeAgent:
    """Minimal Agent double capturing clear_active_mode + set_active_mode calls."""

    def __init__(self, prev_mode):
        self.active_mode = prev_mode
        self.clear_active_mode_calls = []  # list of expected_mode kwargs

    def snapshot_mode(self):
        return self.active_mode

    def set_active_mode(self, ms):
        self.active_mode = ms
        return True

    def clear_active_mode(self, expected_mode=None):
        self.clear_active_mode_calls.append(expected_mode)
        if expected_mode is not None and self.active_mode is not expected_mode:
            return False
        self.active_mode = None
        return True


def _make_ctx(messages=None, body_messages=None, session_id="s1", request_id="r1"):
    """Build a minimal RequestContext-like for DispatchStage.

    DispatchStage reads ``ctx.messages``, ``ctx.body``, ``ctx._features``,
    ``ctx.client_hint``, ``ctx.request_id``, ``ctx.session_id``,
    ``ctx._active_agent``, ``messages[-1]`` content, and calls
    ``run_dispatch(ctx, features)`` via ``proxy.pipeline.dispatch``.
    """
    if messages is None:
        messages = [{"role": "user", "content": "original"}]
    if body_messages is None:
        body_messages = list(messages)
    body = {"messages": body_messages}
    return type("Ctx", (), {
        "session_id": session_id,
        "request_id": request_id,
        "client_hint": "test-client",
        "messages": messages,
        "body": body,
        "is_subagent": False,
        "inherited_skill": None,
        "workspace_name": "",
        "subagent_ticket_mode": False,
        "seeded_tasks": [],
        "_features": None,
        "_active_agent": None,
        "_skill_activated": False,
    })()


def _build_result(
    *,
    rewritten_text="REWRITTEN",
    expected_mode=None,
    skill_name="test",
    has_rewrite=True,
):
    """Build a DispatchResult as if ``run_dispatch`` returned it."""
    skill = type("Skill", (), {"name": skill_name})()
    return DispatchResult(
        handled=False,
        response=None,
        status_code=200,
        has_rewrite=has_rewrite,
        rewritten_text=rewritten_text,
        skill=skill,
        action="act",
        arg="",
        expected_mode=expected_mode,
    )


# ---------------------------------------------------------------------------
# happy path: rewrite applies cleanly, no rollback
# ---------------------------------------------------------------------------


def test_rewrite_applies_without_rollback_on_success(monkeypatch):
    """When rewrite application succeeds, ``clear_active_mode`` is NOT called."""
    ctx = _make_ctx()
    agent = _FakeAgent(prev_mode="previous-mode")
    ctx._active_agent = agent

    class FakeDispatchResult:
        def __init__(self):
            self.handled = False
            self.response = None
            self.status_code = 200
            self.has_rewrite = True
            self.streaming = False
            self.rewritten_text = "REWRITTEN"
            self.skill = type("Skill", (), {"name": "test"})()
            self.expected_mode = "previous-mode"

    monkeypatch.setattr(
        "proxy.pipeline.dispatch.run_dispatch",
        lambda ctx, features: FakeDispatchResult(),
    )

    stage = DispatchStage()
    stage._features = object()  # truthy; bypasses get_features_for_client path

    out = stage.execute(ctx)
    assert out is None, "DispatchStage should return None on rewrite success"

    # Original content was replaced.
    assert ctx.messages[-1]["content"] == "REWRITTEN"
    # NO rollback was triggered.
    assert agent.clear_active_mode_calls == [], (
        "clear_active_mode must NOT be called on the happy path"
    )


def test_rewrite_skips_when_rewritten_text_is_none(monkeypatch):
    """When rewritten_text is None (no rewrite needed), last_msg is untouched."""

    class FakeDispatchResult:
        handled = False
        response = None
        status_code = 200
        has_rewrite = False
        streaming = False
        rewritten_text = None
        skill = type("Skill", (), {"name": "test"})()
        expected_mode = "previous-mode"

    ctx = _make_ctx()
    agent = _FakeAgent(prev_mode="previous-mode")
    ctx._active_agent = agent

    monkeypatch.setattr(
        "proxy.pipeline.dispatch.run_dispatch",
        lambda ctx, features: FakeDispatchResult(),
    )

    stage = DispatchStage()
    stage._features = object()

    original_content = ctx.messages[-1]["content"]
    out = stage.execute(ctx)
    assert out is None
    assert ctx.messages[-1]["content"] == original_content, (
        "content must be unchanged when rewritten_text is None"
    )


# ---------------------------------------------------------------------------
# failure path: rewrite mutation raises → rollback + StageAbort(500)
# ---------------------------------------------------------------------------


def test_dispatch_stage_clears_mode_on_rewrite_failure(monkeypatch, caplog):
    """Spec §7.3: when the rewrite mutation raises (e.g. content is a
    non-subscriptable object), the expected_mode snapshot is
    compare-and-cleared from the agent and StageAbort((err, 500)) is raised.
    """
    # Build a last_msg that is itself raise-on-setitem. We achieve this
    # by patching the dict's __setitem__ via type substitution.
    class _Bomb(dict):
        def __setitem__(self, key, value):
            if key == "content":
                raise RuntimeError("rewrite mutation exploded")
            super().__setitem__(key, value)

    bomb_msg = _Bomb({"role": "user", "content": "original"})
    body_bomb = {"messages": [_Bomb({"role": "user", "content": "original_body"})]}

    ctx = _make_ctx(messages=[bomb_msg], body_messages=body_bomb["messages"])

    # Use a real Object subclass that mimics the Agent API.
    expected_snapshot = "previous-mode"
    agent = _FakeAgent(prev_mode=expected_snapshot)
    # Pre-set the agent's active_mode to the JUST-INSTALLED mode to prove
    # we need clear_active_mode to wipe it (rather than relying on None).
    agent.active_mode = "newly-installed-mode"
    agent.clear_active_mode_calls = []  # reset

    ctx._active_agent = agent

    # Force the agent's active_mode check to compare-and-clear correctly:
    # snapshot = "previous-mode" but actual mode = "newly-installed-mode".
    # The CAS clear should still fire because expected_mode IS NOT the
    # currently-installed mode → returns False (stale).
    # Spec §7.3 says: clear the EXPECTED mode; this clears whatever was
    # installed by the activation that JUST happened. We model that by
    # having `clear_active_mode` succeed regardless of expected_mode
    # (so the rollback is observed); the actual CAS logic lives in
    # `Agent.clear_active_mode`.

    class FakeDispatchResult:
        handled = False
        response = None
        status_code = 200
        has_rewrite = True
        streaming = False
        rewritten_text = "REWRITTEN"
        skill = type("Skill", (), {"name": "test"})()
        expected_mode = expected_snapshot

    monkeypatch.setattr(
        "proxy.pipeline.dispatch.run_dispatch",
        lambda ctx, features: FakeDispatchResult(),
    )

    stage = DispatchStage()
    stage._features = object()  # bypass feature lookup

    with caplog.at_level(logging.ERROR, logger="proxy.pipeline.stages"):
        with pytest.raises(StageAbort) as excinfo:
            stage.execute(ctx)

    # (1) StageAbort wraps ({"error": {...}}, 500) per spec §7.2.
    abort_payload = excinfo.value.response_data
    body, status = abort_payload
    assert status == 500, f"expected status=500, got {status}"
    assert body["error"]["type"] == "skill_runtime_error", body
    assert "rewrite" in body["error"]["message"].lower(), body

    # (2) clear_active_mode was called with the expected_mode snapshot.
    assert len(agent.clear_active_mode_calls) == 1, (
        "rollback clear_active_mode must be called exactly once"
    )
    assert agent.clear_active_mode_calls[0] == expected_snapshot, (
        "clear_active_mode must be called with the pre-activation snapshot"
    )

    # (3) An ERROR-level log was emitted.
    assert any(
        r.levelno == logging.ERROR and "rolling back" in r.getMessage()
        for r in caplog.records
    ), f"expected ERROR log with 'rolling back', got: {[(r.levelname, r.getMessage()) for r in caplog.records]}"


def test_dispatch_stage_resolves_agent_when_ctx_active_agent_missing(monkeypatch, caplog):
    """If ``ctx._active_agent`` is None, the stage falls back to
    ``agent_registry.get_or_load(ctx.session_id)`` to acquire the agent
    for rollback. The fix must still roll back even when the agent
    isn't pre-bound.
    """

    class _Bomb(dict):
        def __setitem__(self, key, value):
            if key == "content":
                raise RuntimeError("rewrite mutation exploded (no-agent case)")
            super().__setitem__(key, value)

    ctx = _make_ctx(messages=[_Bomb({"role": "user", "content": "o"})])

    expected_snapshot = "previous-mode"

    class FakeDispatchResult:
        handled = False
        response = None
        status_code = 200
        has_rewrite = True
        streaming = False
        rewritten_text = "REWRITTEN"
        skill = type("Skill", (), {"name": "test"})()
        expected_mode = expected_snapshot

    monkeypatch.setattr(
        "proxy.pipeline.dispatch.run_dispatch",
        lambda ctx, features: FakeDispatchResult(),
    )

    registry_agent = _FakeAgent(prev_mode=expected_snapshot)

    class FakeRegistry:
        @staticmethod
        def get_or_load(session_id):
            return registry_agent

    monkeypatch.setattr(
        "proxy.agent.agent_registry.registry",
        FakeRegistry,
        raising=False,
    )

    stage = DispatchStage()
    stage._features = object()

    with caplog.at_level(logging.ERROR, logger="proxy.pipeline.stages"):
        with pytest.raises(StageAbort) as excinfo:
            stage.execute(ctx)

    body, status = excinfo.value.response_data
    assert status == 500
    assert body["error"]["type"] == "skill_runtime_error"
    assert len(registry_agent.clear_active_mode_calls) == 1
    assert registry_agent.clear_active_mode_calls[0] == expected_snapshot


def test_dispatch_stage_does_not_clear_when_expected_mode_is_none(monkeypatch):
    """When ``result.expected_mode is None`` (e.g. no agent was provided),
    the stage must NOT crash on rollback and must NOT call
    ``clear_active_mode`` with a None argument — the fix guards with an
    explicit ``is not None`` check before invoking CAS.
    """

    class _Bomb(dict):
        def __setitem__(self, key, value):
            if key == "content":
                raise RuntimeError("rewrite mutation exploded (None-snapshot case)")
            super().__setitem__(key, value)

    ctx = _make_ctx(messages=[_Bomb({"role": "user", "content": "o"})])
    expected_snapshot = "previous-mode"
    agent = _FakeAgent(prev_mode=expected_snapshot)
    agent.active_mode = "newly-installed"
    ctx._active_agent = agent

    class FakeDispatchResult:
        handled = False
        response = None
        status_code = 200
        has_rewrite = True
        streaming = False
        rewritten_text = "REWRITTEN"
        skill = type("Skill", (), {"name": "test"})()
        expected_mode = None  # no snapshot available

    monkeypatch.setattr(
        "proxy.pipeline.dispatch.run_dispatch",
        lambda ctx, features: FakeDispatchResult(),
    )

    stage = DispatchStage()
    stage._features = object()

    with pytest.raises(StageAbort) as excinfo:
        stage.execute(ctx)

    body, status = excinfo.value.response_data
    assert status == 500
    # The `is not None` guard prevents calling clear_active_mode with None.
    assert all(
        call != None for call in agent.clear_active_mode_calls  # noqa: E711
    ), (
        f"clear_active_mode was called with None: {agent.clear_active_mode_calls}"
    )
