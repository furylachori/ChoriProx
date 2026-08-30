"""Pin the ActiveSkillUnavailable 409 contract end-to-end.

Spec §7.2 acceptance criteria:
  "Active skill missing from registry returns 409 and never reaches upstream."

This file proves the contract through the FULL chat-completions handler
(POST /v1/chat/completions), not just the Stage class. The Flask response
must satisfy:

  - status == 409
  - envelope is {"error": {"type": "skill_state_conflict", "code": 409,
                            "message": "..."}}, no extra fields
  - the upstream LLM was NEVER called
  - the agent's active mode was cleared (compare-and-clear against the
    snapshot we installed)

We patch proxy.mode.resolve_mode to return Mode.SKILL so that the active
mode + apply_mode path is exercised, and proxy.mode.apply_mode to raise
ActiveSkillUnavailable so the Stage's 409 contract is exercised.
"""

from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _reset_registry_and_agents():
    """Reset skill + agent registries around every test for isolation."""
    from proxy.skills.skill_registry import reset_default_registry
    from proxy.agent.agent_registry import AgentRegistry

    reset_default_registry()
    try:
        with AgentRegistry._lock:
            AgentRegistry._agents.clear()
            AgentRegistry._agents_by_id.clear()
    except (ImportError, AttributeError):
        pass
    yield
    reset_default_registry()
    try:
        with AgentRegistry._lock:
            AgentRegistry._agents.clear()
            AgentRegistry._agents_by_id.clear()
    except (ImportError, AttributeError):
        pass


def _build_real_agent(active_skill_name: str = "defunct_review"):
    """Build a real `Agent` instance with active_mode set to a missing-skill ModeState."""
    from proxy.agent.agent import Agent
    from proxy.skills.skill_definition import ModeState

    agent = Agent(
        session_id="active_skill_unavailable_session",
        agent_id="active_skill_unavailable",
        is_subagent=False,
    )
    snapshot = ModeState(
        name=active_skill_name,
        activated_at=0.0,
        ttl_seconds=300,
        allowed_tool_names=frozenset({"read"}),
        skill_state={},
    )
    agent.set_active_mode(snapshot)
    return agent, snapshot


def _build_mock_agent(active_skill_name: str = "never_existed"):
    """Build an agent double that records any prohibited stage-local clear.

    `snapshot_mode()` returns a stable snapshot object. `clear_active_mode`
    records calls so direct ContextOptimizeStage tests can assert that Slice
    1B leaves critical-failure finalization exclusively to RequestPipeline.
    """
    import threading
    from proxy.skills.skill_definition import ModeState

    snapshot = ModeState(
        name=active_skill_name,
        activated_at=0.0,
        ttl_seconds=300,
        allowed_tool_names=frozenset({"read"}),
        skill_state={},
    )

    agent = MagicMock()
    agent.session_id = "stage_iso_session"
    agent.agent_id = "stage_iso_agent"
    agent._mode_lock = threading.RLock()
    agent._cleared_with = []
    agent.active_mode = snapshot
    agent.snapshot_mode.return_value = snapshot

    def _clear(expected_mode=None):
        if expected_mode is not None and agent.active_mode is not expected_mode:
            return False
        agent._cleared_with.append(expected_mode)
        agent.active_mode = None
        return True
    agent.clear_active_mode.side_effect = _clear

    return agent, snapshot


# ---------------------------------------------------------------------------
# Flask-level 409 contract tests (proxy layer)
# ---------------------------------------------------------------------------


class TestActiveSkillUnavailable409:
    """Pin the 409 envelope shape end-to-end through the Flask handler."""

    def test_active_skill_missing_returns_409_skill_state_conflict(
        self, client, mock_llama_response, monkeypatch
    ):
        """Active mode references a missing skill → HTTP 409 + envelope shape.

        The 409 must surface the JSON envelope with type=skill_state_conflict,
        code=409, and a generic message — internal exception text MUST NOT
        leak. Upstream must NOT be called.
        """
        agent, _ = _build_real_agent("defunct_review")

        # Override the autouse fixture so the agent we constructed is what
        # the pipeline sees.
        monkeypatch.setattr(
            "proxy.agent.agent_registry.registry.get_or_load",
            lambda session_id, **kw: agent,
        )
        # Force resolve_mode to return SKILL so apply_mode exercises the
        # registry check.
        monkeypatch.setattr("proxy.mode.resolve_mode", lambda ctx, ag: __import__(
            "proxy.mode", fromlist=["Mode"]
        ).Mode.SKILL)

        resp = client.post(
            "/v1/chat/completions",
            json={
                "model": "test-model",
                "messages": [{"role": "user", "content": "any text"}],
            },
        )

        assert resp.status_code == 409, (
            f"Expected 409 skill_state_conflict, got {resp.status_code}: "
            f"{resp.get_data(as_text=True)[:200]}"
        )
        data = resp.get_json()
        assert "error" in data
        err = data["error"]
        assert isinstance(err, dict)
        assert err.get("type") == "skill_state_conflict"
        # MED-1 (Wave 3): the redundant inner `code` field was removed
        # from the envelope (the tuple's second element carries the
        # status). Status is asserted above via `resp.status_code`.
        assert "code" not in err, (
            f"envelope must not carry a redundant `code` field; got {err}"
        )
        body_str = str(data)
        assert "Traceback" not in body_str
        assert "ActiveSkillUnavailable" not in body_str
        assert "RuntimeError" not in body_str

    def test_active_skill_missing_clears_mode_no_upstream_call(
        self, client, mock_llama_response, monkeypatch
    ):
        """After the 409, agent.active_mode is cleared AND upstream was never called."""
        agent, snapshot = _build_real_agent("ghost_skill")
        assert agent.snapshot_mode() is snapshot

        monkeypatch.setattr(
            "proxy.agent.agent_registry.registry.get_or_load",
            lambda session_id, **kw: agent,
        )
        from proxy.mode import Mode
        monkeypatch.setattr("proxy.mode.resolve_mode", lambda ctx, ag: Mode.SKILL)

        # Spy on the upstream HTTP call. block_all_http (autouse) already
        # patches _HTTP_SESSION.request; we count to confirm it was not hit.
        upstream_calls = []
        import proxy.state as state
        original_session_request = state._HTTP_SESSION.request

        def _counting_request(*args, **kwargs):
            upstream_calls.append((args, kwargs))
            return original_session_request(*args, **kwargs)

        monkeypatch.setattr(state._HTTP_SESSION, "request", _counting_request)

        resp = client.post(
            "/v1/chat/completions",
            json={
                "model": "test-model",
                "messages": [{"role": "user", "content": "any text"}],
            },
        )

        assert resp.status_code == 409
        assert upstream_calls == [], (
            f"Upstream LLM was invoked for an active-skill 409 — fail-closed "
            f"contract violation. Calls: {len(upstream_calls)}"
        )
        assert agent.snapshot_mode() is None, (
            "compare-and-clear MUST remove the active mode after 409; "
            "a retry would re-trigger the same ActiveSkillUnavailable"
        )

    def test_active_skill_missing_returns_no_choices(
        self, client, mock_llama_response, monkeypatch
    ):
        """409 envelope must NOT contain 'choices' (no upstream result)."""
        agent, _ = _build_real_agent("vanished_skill")
        monkeypatch.setattr(
            "proxy.agent.agent_registry.registry.get_or_load",
            lambda session_id, **kw: agent,
        )
        from proxy.mode import Mode
        monkeypatch.setattr("proxy.mode.resolve_mode", lambda ctx, ag: Mode.SKILL)

        resp = client.post(
            "/v1/chat/completions",
            json={
                "model": "test-model",
                "messages": [{"role": "user", "content": "any text"}],
            },
        )

        assert resp.status_code == 409
        data = resp.get_json()
        assert "choices" not in data, (
            f"409 envelope must not include 'choices'; got {data}"
        )
        assert "error" in data
        err = data["error"]
        assert set(err.keys()).issubset({"type", "code", "message"}), (
            f"409 envelope contains extra keys: {err.keys()}"
        )


# ---------------------------------------------------------------------------
# Stage-level isolation tests (faster, no Flask needed)
# ---------------------------------------------------------------------------


class TestActiveSkillUnavailableStageContract:
    """Direct ContextOptimizeStage tests for fail-closed error translation.

    These tests pin the StageAbort envelope and prove that the stage does not
    clear active mode locally. Critical-failure finalization is owned by
    RequestPipeline and is covered by the full-pipeline lifecycle tests.
    """

    def test_stage_returns_409_without_locally_clearing_mode(self):
        """The stage maps the missing skill to 409 without finalizing mode."""
        from proxy.pipeline.exceptions import StageAbort
        from proxy.pipeline.stage_modules.context_optimize import (
            ContextOptimizeStage,
        )
        from proxy.mode import ActiveSkillUnavailable, Mode

        agent, snapshot = _build_mock_agent("never_existed")

        ctx = MagicMock()
        ctx.body = {
            "messages": [{"role": "user", "content": "x"}],
            "tools": [],
            "model": "m",
        }
        ctx.session_id = "stage_iso_session"
        ctx.request_id = "r"
        ctx.is_subagent = False
        ctx.subagent_ticket_mode = False
        ctx.inherited_skill = None
        ctx.model_cfg = None
        ctx.model_name = "m"
        ctx.messages = ctx.body["messages"]

        with patch(
            "proxy.mode.apply_mode",
            side_effect=ActiveSkillUnavailable("never_existed"),
        ), patch(
            "proxy.mode.resolve_mode",
            return_value=Mode.SKILL,
        ), patch(
            "proxy.agent.agent_registry.registry.get_or_load",
            return_value=agent,
        ):
            stage = ContextOptimizeStage()
            with pytest.raises(StageAbort) as cm:
                stage.execute(ctx)

        response_data, status_code = cm.value.response_data
        assert status_code == 409
        error = response_data["error"]
        assert error["type"] == "skill_state_conflict"
        assert "code" not in error, (
            f"envelope must not carry a redundant `code` field; got {error}"
        )
        assert "ActiveSkillUnavailable" not in error["message"]

        agent.clear_active_mode.assert_not_called()
        assert agent._cleared_with == []
        assert agent.active_mode is snapshot

    def test_stage_returns_500_without_locally_clearing_mode(self):
        """The stage maps the skill callback error without finalizing mode."""
        from proxy.pipeline.exceptions import StageAbort
        from proxy.pipeline.stage_modules.context_optimize import (
            ContextOptimizeStage,
        )
        from proxy.mode import Mode, SkillRuntimeError

        agent, snapshot = _build_mock_agent("broken_skill")

        ctx = MagicMock()
        ctx.body = {
            "messages": [{"role": "user", "content": "x"}],
            "tools": [],
            "model": "m",
        }
        ctx.session_id = "stage_iso_session"
        ctx.request_id = "r"
        ctx.is_subagent = False
        ctx.subagent_ticket_mode = False
        ctx.inherited_skill = None
        ctx.model_cfg = None
        ctx.model_name = "m"
        ctx.messages = ctx.body["messages"]

        inner = ValueError("underlying callback crash")
        skill_error = SkillRuntimeError(
            "broken_skill",
            source="apply_mode",
            original=inner,
        )

        with patch(
            "proxy.mode.apply_mode",
            side_effect=skill_error,
        ), patch(
            "proxy.mode.resolve_mode",
            return_value=Mode.SKILL,
        ), patch(
            "proxy.agent.agent_registry.registry.get_or_load",
            return_value=agent,
        ):
            stage = ContextOptimizeStage()
            with pytest.raises(StageAbort) as cm:
                stage.execute(ctx)

        response_data, status_code = cm.value.response_data
        assert status_code == 500
        error = response_data["error"]
        assert error["type"] == "skill_runtime_error"
        assert "code" not in error, (
            f"envelope must not carry a redundant `code` field; got {error}"
        )
        assert "underlying callback crash" not in error["message"]
        assert "ValueError" not in error["message"]

        agent.clear_active_mode.assert_not_called()
        assert agent._cleared_with == []
        assert agent.active_mode is snapshot
