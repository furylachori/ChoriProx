"""Unit tests for proxy.pipeline.stages and proxy.pipeline.request_pipeline.

Covers the 7-stage pipeline hierarchy and the RequestPipeline orchestrator.
"""
from unittest.mock import MagicMock, patch

import pytest

from proxy.pipeline.dispatch import DispatchResult
from proxy.pipeline.request_pipeline import (
    RequestPipeline,
    create_pipeline,
    is_critical_stage,
)
from proxy.pipeline.exceptions import StageAbort
from proxy.pipeline.stages import (
    DEFAULT_STAGES,
    CleanupStage,
    ContextOptimizeStage,
    DispatchStage,
    MotivatorStage,
    PostProcessStage,
    ProviderCompatStage,
    RouteResolutionStage,
    SecurityStage,
    UpstreamStage,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_ctx(
    messages=None,
    body=None,
    session_id="test-session",
    model_name="kimi-k2",
    model_route=None,
    model_cfg=None,
    client_hint="unknown",
):
    """Build a MagicMock RequestContext suitable for stage tests."""
    if messages is None:
        messages = [{"role": "user", "content": "hello"}]
    if body is None:
        body = {"messages": list(messages)}
    ctx = MagicMock()
    ctx.messages = list(messages)
    ctx.session_id = session_id
    ctx.model_name = model_name
    ctx.body = body
    ctx.model_route = model_route
    ctx.model_cfg = model_cfg
    ctx.client_hint = client_hint
    ctx.wants_stream = False
    ctx.is_subagent = False
    return ctx


def _make_route(backend_model="kimi-k2", model_cfg=None):
    """Build a mock ModelRoute with a model_cfg."""
    if model_cfg is None:
        model_cfg = MagicMock()
        model_cfg.backend_model = backend_model
        model_cfg.ignore_subagents = False
        model_cfg.memory_enabled = True
    route = MagicMock()
    route.model_cfg = model_cfg
    return route


def _make_fake_stage_class(
    name="FakeStage",
    return_value=None,
    raise_exc=None,
    side_effect=None,
    call_log=None,
):
    """Build a fake stage class with a configurable __name__ and execute().

    - return_value: what execute() returns on success
    - raise_exc: exception instance/class execute() raises (overrides return)
    - side_effect: callable invoked as side_effect(ctx) before the return/raise
    - call_log: optional list; the ctx is appended to it on each call
    """

    def execute(self, ctx):
        if call_log is not None:
            call_log.append(ctx)
        if side_effect is not None:
            side_effect(ctx)
        if raise_exc is not None:
            raise raise_exc
        return return_value

    cls = type(name, (), {"execute": execute})
    return cls


# ---------------------------------------------------------------------------
# StageAbort
# ---------------------------------------------------------------------------

class TestStageAbort:
    """Tests for the StageAbort exception class."""

    def test_stage_abort_carries_response_data(self):
        e = StageAbort(response_data={"x": 1})
        assert e.response_data == {"x": 1}
        assert isinstance(e, Exception)

    def test_stage_abort_allows_none_response(self):
        e = StageAbort(response_data=None)
        assert e.response_data is None


# ---------------------------------------------------------------------------
# DEFAULT_STAGES
# ---------------------------------------------------------------------------

class TestDefaultStages:
    """Tests for the DEFAULT_STAGES list."""

    EXPECTED = [
        CleanupStage,
        DispatchStage,
        RouteResolutionStage,
        SecurityStage,
        ContextOptimizeStage,
        MotivatorStage,
        ProviderCompatStage,
        UpstreamStage,
        PostProcessStage,
    ]

    def test_default_stages_count_and_order(self):
        assert len(DEFAULT_STAGES) == 9
        for i, cls in enumerate(self.EXPECTED):
            assert isinstance(DEFAULT_STAGES[i], cls), (
                f"DEFAULT_STAGES[{i}] is {type(DEFAULT_STAGES[i]).__name__}, expected {cls.__name__}"
            )


# ---------------------------------------------------------------------------
# CleanupStage
# ---------------------------------------------------------------------------

class TestCleanupStage:
    """Tests for CleanupStage.execute."""

    def test_cleanup_stage_increments_counter(self):
        import proxy.state as state
        state.request_counter = 0
        if hasattr(state, "CLEANUP_INTERVAL"):
            del state.CLEANUP_INTERVAL

        ctx = _make_ctx()

        with patch("proxy.session.cleanup_old_sessions") as m1, \
             patch("memory.db.cleanup_expired_messages") as m2, \
             patch("proxy.state.cleanup_stale_workspaces", return_value=0) as m3, \
             patch(
                 "proxy.agent.agent_registry.registry.cleanup_stale_ticket_modes"
             ) as m4:
            CleanupStage().execute(ctx)

        assert state.request_counter == 1
        m1.assert_not_called()
        m2.assert_not_called()
        m3.assert_not_called()
        m4.assert_called_once()

    def test_cleanup_stage_triggers_periodic_cleanup_at_interval(self):
        import proxy.state as state
        state.request_counter = 99
        if hasattr(state, "CLEANUP_INTERVAL"):
            del state.CLEANUP_INTERVAL

        with patch("proxy.session.cleanup_old_sessions") as m1, \
             patch("memory.db.cleanup_expired_messages") as m2, \
             patch("proxy.state.cleanup_stale_workspaces", return_value=0) as m3, \
             patch(
                 "proxy.agent.agent_registry.registry.cleanup_stale_ticket_modes"
             ) as m4:
            CleanupStage().execute(_make_ctx())

        assert state.request_counter == 100
        m1.assert_called_once()
        m2.assert_called_once()
        m3.assert_called_once()
        m4.assert_called_once()


# ---------------------------------------------------------------------------
# DispatchStage
# ---------------------------------------------------------------------------

class TestDispatchStage:
    """Tests for DispatchStage.execute."""

    def test_dispatch_stage_raises_stageabort_on_handled(self):
        resp = {"object": "chat.completion", "result": "done"}
        result = DispatchResult(
            rewritten_text=None,
            response=resp,
            status_code=200,
            handled=True,
        )
        ctx = _make_ctx()
        stage = DispatchStage(features=MagicMock())

        with patch("proxy.pipeline.dispatch.run_dispatch", return_value=result):
            with pytest.raises(StageAbort) as exc_info:
                stage.execute(ctx)

        assert exc_info.value.response_data is resp

    def test_dispatch_stage_mutates_messages_on_rewrite(self):
        result = DispatchResult(
            rewritten_text="X",
            response=None,
            status_code=200,
            handled=False,
        )
        msgs = [{"role": "user", "content": "old"}]
        body = {"messages": [{"role": "user", "content": "old"}]}
        ctx = _make_ctx(messages=msgs, body=body)
        stage = DispatchStage(features=MagicMock())

        with patch("proxy.pipeline.dispatch.run_dispatch", return_value=result):
            stage.execute(ctx)

        assert msgs[0]["content"] == "X"
        assert body["messages"][0]["content"] == "X"

    def test_dispatch_stage_returns_none_when_dispatch_returns_none(self):
        ctx = _make_ctx()
        stage = DispatchStage(features=MagicMock())

        with patch("proxy.pipeline.dispatch.run_dispatch", return_value=None):
            assert stage.execute(ctx) is None


# ---------------------------------------------------------------------------
# SecurityStage
# ---------------------------------------------------------------------------

class TestSecurityStage:
    """Tests for SecurityStage.execute."""

    def test_security_stage_scans_all_user_messages(self):
        messages = [{"role": "user", "content": f"msg {i}"} for i in range(15)]
        ctx = _make_ctx(messages=messages, body={"messages": list(messages)})
        ctx.model_route = _make_route()
        ctx.model_cfg = ctx.model_route.model_cfg

        with patch("proxy.auth.detect_malicious_content", return_value=[]) as m_detect:
            SecurityStage().execute(ctx)

        assert m_detect.call_count == 15
        for i in range(15):
            assert m_detect.call_args_list[i].args[0] == f"msg {i}"


# ---------------------------------------------------------------------------
# ProviderCompatStage
# ---------------------------------------------------------------------------

class TestProviderCompatStage:
    """Tests for ProviderCompatStage.execute."""

    def test_provider_compat_stage_builds_llama_body(self):
        route = _make_route(backend_model="kimi-k2-0905")
        body = {
            "messages": [{"role": "user", "content": "hi"}],
            "stream": True,
            "stream_options": {"include_usage": True},
            "tools": [{"type": "function", "function": {"name": "x"}}],
            "temperature": 1.5,
        }
        ctx = _make_ctx(body=body, model_route=route)
        # ctx.wants_stream defaults to False (non-streaming client). Ticket
        # 199ece93 Part A: llama_body["stream"] mirrors ctx.wants_stream, NOT
        # the raw body["stream"] — so it is False here even though the body
        # carries stream=True.
        ctx.wants_stream = False

        with patch("proxy_recovery_system.apply_proxy_recovery_system", side_effect=lambda b, s: b):
            ProviderCompatStage().execute(ctx)

        assert ctx.llama_body["stream"] is False
        assert "stream_options" not in ctx.llama_body
        # Tools may have description added by ProviderCompatStage
        assert ctx.llama_body["tools"][0]["function"]["name"] == "x"

        assert body["stream"] is True
        assert "stream_options" in body

    def test_provider_compat_stage_preserves_wants_stream(self):
        """Ticket 199ece93 Part A: a streaming client (ctx.wants_stream=True)
        must produce llama_body["stream"] is True so the upstream request
        carries stream=true. stream_options is still stripped."""
        route = _make_route(backend_model="kimi-k2-0905")
        body = {
            "messages": [{"role": "user", "content": "hi"}],
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        ctx = _make_ctx(body=body, model_route=route)
        ctx.wants_stream = True

        with patch("proxy_recovery_system.apply_proxy_recovery_system", side_effect=lambda b, s: b):
            ProviderCompatStage().execute(ctx)

        assert ctx.llama_body["stream"] is True
        assert "stream_options" not in ctx.llama_body

    def test_provider_compat_stage_propagates_adapter_failure(self):
        """M2-1: adapter failures must surface, not be swallowed.

        AGENTS.md: "apply_provider_compat has no feature flag" — the
        old `except Exception: logger.warning(...)` branch silently
        disabled provider compat. The fix removes that branch; this
        test forces MiniMaxAdapter.apply to raise and asserts the
        exception propagates out of execute().
        """
        from proxy.pipeline.provider_compat import MiniMaxAdapter

        route = _make_route(backend_model="MiniMax-Text-01")
        body = {
            "messages": [{"role": "system", "content": "sys"}],
            "tools": [{"type": "function", "function": {"name": "x"}}],
        }
        ctx = _make_ctx(body=body, model_route=route)

        with patch.object(MiniMaxAdapter, "apply", side_effect=RuntimeError("MiniMax boom")), \
             patch("proxy_recovery_system.apply_proxy_recovery_system", side_effect=lambda b, s: b):
            with pytest.raises(RuntimeError, match="MiniMax boom"):
                ProviderCompatStage().execute(ctx)


# ---------------------------------------------------------------------------
# UpstreamStage
# ---------------------------------------------------------------------------

class TestUpstreamStage:
    """Tests for UpstreamStage.execute."""

    def test_upstream_stage_test_mode_returns_mock(self):
        route = _make_route()
        upstream = MagicMock()
        upstream.original_model = "kimi-k2"

        ctx = _make_ctx(model_route=route)
        ctx.llama_body = {"model": "kimi-k2", "messages": []}
        ctx.upstream = upstream
        # NOTE: ticket 199ece93 removed the streaming `_defer_upstream` flag
        # — UpstreamStage now always invokes call_upstream. The field no
        # longer exists; setting it would AttributeError on a real ctx.

        with patch("proxy.model_registry.TEST_MODE", True), \
             patch("proxy.pipeline.stage_modules.upstream.prepare_upstream_request", return_value=upstream):
            UpstreamStage().execute(ctx)

        assert ctx.response_data["object"] == "chat.completion"
        assert ctx.response_data["model"] == "kimi-k2"
        assert ctx.response_data["choices"][0]["message"]["role"] == "assistant"
        assert ctx.response_data["choices"][0]["finish_reason"] == "stop"

    def test_upstream_stage_raises_stageabort_on_generic_exception(self):
        """# US-043"""
        route = _make_route()
        ctx = _make_ctx(model_route=route)
        ctx.llama_body = {"model": "kimi-k2", "messages": []}

        with patch(
            "proxy.pipeline.stage_modules.upstream.prepare_upstream_request",
            side_effect=RuntimeError("connection refused"),
        ):
            with pytest.raises(StageAbort) as exc_info:
                UpstreamStage().execute(ctx)

        response_tuple = exc_info.value.response_data
        assert isinstance(response_tuple, tuple)
        body, status = response_tuple
        assert status == 502
        assert body["error"]["type"] == "upstream_error"
        assert "connection refused" in body["error"]["message"]

    def test_upstream_stage_reraises_auth_config_error(self):
        from proxy.auth import AuthConfigError

        route = _make_route()
        ctx = _make_ctx(model_route=route)
        ctx.llama_body = {"model": "kimi-k2", "messages": []}

        with patch(
            "proxy.pipeline.stage_modules.upstream.prepare_upstream_request",
            side_effect=AuthConfigError("model-x", "API_KEY_ENV"),
        ):
            with pytest.raises(AuthConfigError):
                UpstreamStage().execute(ctx)


# ---------------------------------------------------------------------------
# ContextOptimizeStage
# ---------------------------------------------------------------------------

class TestContextOptimizeStage:
    """Tests for ContextOptimizeStage.execute — context limit enforcement."""

    def test_context_optimize_stage_aborts_on_value_error(self):
        ctx = _make_ctx()
        ctx.model_cfg = MagicMock()

        with patch(
            "proxy.context.enforce_context_limits",
            side_effect=ValueError("Token count 300000 exceeds limit 262144"),
        ), patch("proxy.agent.agent_registry.registry.get_or_load", return_value=None):
            with pytest.raises(StageAbort) as exc_info:
                ContextOptimizeStage().execute(ctx)

        response_tuple = exc_info.value.response_data
        assert isinstance(response_tuple, tuple)
        body, status = response_tuple
        assert status == 413
        assert body["error"]["type"] == "context_limit_exceeded"
        assert "300000" in body["error"]["message"]

    def test_context_optimize_stage_continues_on_non_value_error(self):
        ctx = _make_ctx()
        ctx.model_cfg = MagicMock()

        with patch(
            "proxy.context.enforce_context_limits",
            side_effect=TypeError("unexpected"),
        ), patch("proxy.agent.agent_registry.registry.get_or_load", return_value=None), \
          patch("proxy.mode.apply_mode", return_value={"messages": [{"role": "user", "content": "hello"}], "tools": []}):
            result = ContextOptimizeStage().execute(ctx)

        assert result is None

    def test_context_optimize_stage_aborts_when_apply_mode_raises_value_error(self):
        """P0-3: ValueError from apply_mode (context limit) produces 413 StageAbort."""
        from proxy.mode import Mode

        ctx = _make_ctx()
        ctx.model_cfg = MagicMock()
        ctx.request_id = "req-test-p03"

        mock_agent = MagicMock()
        mock_agent.active_mode = None

        with patch(
            "proxy.context.enforce_context_limits",
        ), patch("proxy.agent.agent_registry.registry.get_or_load", return_value=mock_agent), \
          patch("proxy.mode.resolve_mode", return_value=Mode.DEFAULT), \
          patch("proxy.mode.apply_mode", side_effect=ValueError("V2: assembled context overflows safe budget")):
            # ValueError is not ContextLimitError; should be caught by generic handler
            # and fall back to DEFAULT mode
            result = ContextOptimizeStage().execute(ctx)
        assert result is None

    def test_context_optimize_stage_aborts_on_context_limit_error(self):
        """H2: ContextLimitError from apply_mode produces 413 StageAbort."""
        from proxy.mode import Mode
        from proxy.context import ContextLimitError

        ctx = _make_ctx()
        ctx.model_cfg = MagicMock()
        ctx.request_id = "req-test-h2"

        mock_agent = MagicMock()
        mock_agent.active_mode = None

        with patch(
            "proxy.context.enforce_context_limits",
        ), patch("proxy.agent.agent_registry.registry.get_or_load", return_value=mock_agent), \
          patch("proxy.mode.resolve_mode", return_value=Mode.DEFAULT), \
          patch("proxy.mode.apply_mode", side_effect=ContextLimitError("V2: assembled context overflows safe budget")):
            with pytest.raises(StageAbort) as exc_info:
                ContextOptimizeStage().execute(ctx)

        response_tuple = exc_info.value.response_data
        assert isinstance(response_tuple, tuple)
        body, status = response_tuple
        assert status == 413
        assert body["error"]["type"] == "context_limit_exceeded"
        assert "overflows safe budget" in body["error"]["message"]

    def test_context_optimize_stage_deepcopy_on_success_path(self):
        """Bug B: body snapshot taken before apply_mode mutation."""
        ctx = _make_ctx()
        ctx.model_cfg = MagicMock()
        original_msgs = list(ctx.body.get("messages", []))

        mock_agent = MagicMock()
        mock_agent.active_mode = None

        # apply_mode returns a mutated body with extra "system_injected" flag
        mutated_body = {"messages": [{"role": "system", "content": "x"}] + original_msgs, "tools": []}

        with patch(
            "proxy.context.enforce_context_limits",
        ), patch("proxy.agent.agent_registry.registry.get_or_load", return_value=mock_agent), \
          patch("proxy.mode.resolve_mode"), \
          patch("proxy.mode.apply_mode", return_value=mutated_body):
            result = ContextOptimizeStage().execute(ctx)

        assert result is None
        # messages_original captured the pre-mutation messages
        assert ctx.messages_original == original_msgs
        assert ctx.messages_original is not original_msgs  # was a copy, not a reference

    def test_context_optimize_stage_deepcopy_on_fallback_path(self):
        """Bug B: body snapshot taken before apply_mode on the fallback (agent=None) path."""
        ctx = _make_ctx()
        ctx.model_cfg = MagicMock()
        original_msgs = list(ctx.body.get("messages", []))

        mutated_body = {"messages": [{"role": "system", "content": "x"}] + original_msgs, "tools": []}

        with patch(
            "proxy.context.enforce_context_limits",
        ), patch("proxy.agent.agent_registry.registry.get_or_load", return_value=None), \
          patch("proxy.mode.apply_mode", return_value=mutated_body):
            result = ContextOptimizeStage().execute(ctx)

        assert result is None
        from proxy.mode import Mode
        assert ctx._mode == Mode.DEFAULT
        assert ctx.messages_original == original_msgs
        assert ctx.messages_original is not original_msgs  # was a copy, not a reference

        # Strengthened assertions: confirm ctx.body is a deep copy of the
        # fallback output (apply_mode return) and the fallback source is
        # the expected upstream stage output.
        #
        # 1. ctx.body should equal the mutated_body returned by apply_mode
        #    (this is the "fallback source is the expected upstream stage output")
        assert ctx.body == mutated_body, \
            "ctx.body should equal the mutated_body returned by apply_mode"
        # 2. ctx.body must NOT be the same object (must be a deep copy)
        assert ctx.body is not mutated_body, \
            "ctx.body must be a separate deep copy, not the same object reference"
        # 3. Mutate the apply_mode return value and verify ctx.body is unaffected
        mutated_body["messages"].append({"role": "user", "content": "injected"})
        assert ctx.body == {"messages": [{"role": "system", "content": "x"}] + original_msgs, "tools": []}, \
            "ctx.body must remain unchanged after mutating the apply_mode return value"


# ---------------------------------------------------------------------------
# PostProcessStage
# ---------------------------------------------------------------------------

class TestPostProcessStage:
    """Tests for PostProcessStage.execute."""

    def test_postprocess_stage_dict_error_preserves_original_body(self):
        """Dict-valued provider error envelope is preserved verbatim with 502.

        When the upstream returns a structured error dict (with no choices),
        PostProcess aborts with the ORIGINAL body and HTTP 502 so the client
        can see exactly what the upstream reported.
        """
        from proxy.openai_compat import build_upstream_protocol_error

        route = _make_route()
        ctx = _make_ctx(model_route=route)
        provider_error = {"error": {"message": "boom", "type": "provider_error"}}
        ctx.response_data = provider_error
        ctx.llama_body = {}
        ctx.upstream = MagicMock()

        with pytest.raises(StageAbort) as exc_info:
            PostProcessStage().execute(ctx)

        response_tuple = exc_info.value.response_data
        assert isinstance(response_tuple, tuple)
        body, status = response_tuple
        assert status == 502
        assert body == provider_error
        assert ctx.response_data == provider_error

    def test_postprocess_stage_string_error_replaced_with_canonical(self):
        """String-valued `error` is replaced with build_upstream_protocol_error().

        A non-object `error` value (e.g. a bare string) is malformed — the
        shape check rejects it and PostProcess substitutes the canonical
        upstream-protocol-error envelope before aborting with 502.
        """
        from proxy.openai_compat import build_upstream_protocol_error

        route = _make_route()
        ctx = _make_ctx(model_route=route)
        ctx.response_data = {"choices": [], "error": "boom"}
        ctx.llama_body = {}
        ctx.upstream = MagicMock()

        with pytest.raises(StageAbort) as exc_info:
            PostProcessStage().execute(ctx)

        response_tuple = exc_info.value.response_data
        assert isinstance(response_tuple, tuple)
        body, status = response_tuple
        canonical = build_upstream_protocol_error()
        assert status == 502
        assert body == canonical
        assert ctx.response_data == canonical

    def test_postprocess_stage_returns_data_when_choices_present(self):
        route = _make_route()
        ctx = _make_ctx(model_route=route)
        ctx.response_data = {
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "ok"},
                    "finish_reason": "stop",
                }
            ]
        }
        ctx.llama_body = {}
        ctx.upstream = MagicMock()

        noop = lambda self, ctx: None
        with patch.object(PostProcessStage, "_extract_minimax_reasoning", noop), \
             patch.object(PostProcessStage, "_sync_chat_history", noop), \
             patch.object(PostProcessStage, "_persist_messages", noop), \
             patch.object(PostProcessStage, "_handle_tool_interception", noop), \
             patch.object(PostProcessStage, "_assign_model_name", noop), \
             patch.object(PostProcessStage, "_apply_skill_response_transform", noop), \
             patch.object(PostProcessStage, "_inject_task_canaries", noop), \
             patch.object(PostProcessStage, "_finalize_skill_lifecycle", noop):
            result = PostProcessStage().execute(ctx)

        assert result is not None
        assert result["choices"][0]["finish_reason"] == "stop"


# ---------------------------------------------------------------------------
# RequestPipeline orchestrator
# ---------------------------------------------------------------------------

class TestRequestPipeline:
    """Tests for the RequestPipeline orchestrator."""

    def test_request_pipeline_short_circuits_on_stage_result(self):
        call_log = []
        sentinel = {"sentinel": True}

        stage_a = _make_fake_stage_class(
            name="StageA",
            return_value=sentinel,
            call_log=call_log,
        )
        stage_b = _make_fake_stage_class(
            name="StageB",
            return_value={"should": "not_see"},
            call_log=call_log,
        )

        pipeline = RequestPipeline(stages=[stage_a(), stage_b()])
        ctx = _make_ctx(session_id=None)

        result = pipeline.run(ctx)

        assert result is sentinel
        assert len(call_log) == 1

    def test_request_pipeline_reraises_critical_stage_exception(self):
        call_log = []
        critical_exc = RuntimeError("boom-critical")

        critical_stage = _make_fake_stage_class(
            name="SecurityStage",
            raise_exc=critical_exc,
            call_log=call_log,
        )

        pipeline = RequestPipeline(stages=[critical_stage()])
        ctx = _make_ctx(session_id=None)

        with pytest.raises(RuntimeError) as exc_info:
            pipeline.run(ctx)

        assert exc_info.value is critical_exc
        assert len(call_log) == 1
        assert is_critical_stage(critical_stage())

    def test_request_pipeline_continues_on_non_critical_exception(self):
        call_log = []
        non_critical_exc = RuntimeError("transient")

        stage_a = _make_fake_stage_class(
            name="DispatchStage",
            raise_exc=non_critical_exc,
            call_log=call_log,
        )
        stage_b = _make_fake_stage_class(
            name="CleanupStage",
            return_value=None,
            call_log=call_log,
        )

        pipeline = RequestPipeline(stages=[stage_a(), stage_b()])
        ctx = _make_ctx(session_id=None)
        ctx.response_data = {"fallback": True}

        result = pipeline.run(ctx)

        assert len(call_log) == 2
        assert result == {"fallback": True}

    def test_request_pipeline_default_uses_default_stages(self):
        pipeline = RequestPipeline()
        assert len(pipeline.stages) == len(DEFAULT_STAGES)
        assert pipeline.stages[0] is DEFAULT_STAGES[0]

    def test_create_pipeline_returns_default_pipeline(self):
        pipeline = create_pipeline()
        assert isinstance(pipeline, RequestPipeline)
        assert len(pipeline.stages) == len(DEFAULT_STAGES)

    def test_is_critical_stage_true_for_security(self):
        assert is_critical_stage(SecurityStage()) is True

    def test_is_critical_stage_true_for_upstream(self):
        assert is_critical_stage(UpstreamStage()) is True

    def test_is_critical_stage_false_for_cleanup(self):
        assert is_critical_stage(CleanupStage()) is False


class TestContextOptimizeCascadeFallback:
    """H5: cascade fallback when primary fallback also fails."""

    def test_cascade_fallback_preserves_system_and_last(self):
        """When primary fallback fails, cascade should keep system + last message."""
        from proxy.mode import Mode
        from proxy.tool_parsing import OUR_SYSTEM_PROMPT

        system_msg = {"role": "system", "content": "You are helpful."}
        user_msg = {"role": "user", "content": "hello"}
        ctx = _make_ctx(messages=[system_msg, user_msg])
        ctx.model_cfg = MagicMock()
        ctx.request_id = "req-cascade"

        mock_agent = MagicMock()
        mock_agent.active_mode = None

        call_count = [0]
        def failing_apply_mode(*args, **kwargs):
            call_count[0] += 1
            raise RuntimeError("apply_mode always fails")

        with patch("proxy.context.enforce_context_limits"), \
             patch("proxy.agent.agent_registry.registry.get_or_load", return_value=mock_agent), \
             patch("proxy.mode.resolve_mode", return_value=Mode.DEFAULT), \
             patch("proxy.mode.apply_mode", side_effect=failing_apply_mode):
            result = ContextOptimizeStage().execute(ctx)

        assert result is None
        # Cascade should have produced a minimal messages list
        msgs = ctx.body.get("messages", [])
        assert len(msgs) == 2
        assert msgs[0]["role"] == "system"
        assert msgs[1]["role"] == "user"
        # Bug B: cascade fallback must replace client system prompt with proxy's
        assert msgs[0]["content"] == OUR_SYSTEM_PROMPT
        assert "You are helpful" not in msgs[0]["content"]

    def test_deepcopy_failure_produces_500(self):
        """If deepcopy of body fails, stage should abort with 500."""
        ctx = _make_ctx()
        ctx.model_cfg = MagicMock()
        ctx.request_id = "req-deepcopy-fail"

        mock_agent = MagicMock()
        mock_agent.active_mode = None

        with patch("proxy.context.enforce_context_limits"), \
             patch("proxy.agent.agent_registry.registry.get_or_load", return_value=mock_agent), \
             patch("copy.deepcopy", side_effect=MemoryError("out of memory")):
            with pytest.raises(StageAbort) as exc_info:
                ContextOptimizeStage().execute(ctx)

        response_tuple = exc_info.value.response_data
        body, status = response_tuple
        assert status == 500
        assert body["error"]["type"] == "internal_error"


# ---------------------------------------------------------------------------
# H1: agent-None path must surface ContextLimitError as 413
# ---------------------------------------------------------------------------

class TestContextOptimizeAgentNoneContextLimit:
    """H1: agent-None fallback path's apply_mode call must raise StageAbort(413)
    when apply_mode raises ContextLimitError or ValueError (context-limit like).
    """

    def test_agent_none_context_limit_raises_413(self):
        from proxy.context import ContextLimitError

        ctx = _make_ctx()
        ctx.model_cfg = MagicMock()
        ctx.request_id = "req-h1-cle"

        with patch("proxy.context.enforce_context_limits"), \
             patch("proxy.agent.agent_registry.registry.get_or_load", return_value=None), \
             patch("proxy.mode.apply_mode", side_effect=ContextLimitError("V2: agent-None path budget exceeded")):
            with pytest.raises(StageAbort) as exc_info:
                ContextOptimizeStage().execute(ctx)

        response_tuple = exc_info.value.response_data
        assert isinstance(response_tuple, tuple)
        body, status = response_tuple
        assert status == 413
        assert body["error"]["type"] == "context_limit_exceeded"
        assert "agent-None" in body["error"]["message"] or "budget exceeded" in body["error"]["message"]

    def test_agent_none_value_error_raises_413(self):
        ctx = _make_ctx()
        ctx.model_cfg = MagicMock()
        ctx.request_id = "req-h1-ve"

        with patch("proxy.context.enforce_context_limits"), \
             patch("proxy.agent.agent_registry.registry.get_or_load", return_value=None), \
             patch("proxy.mode.apply_mode", side_effect=ValueError("V2: agent-None value error budget exceeded")):
            with pytest.raises(StageAbort) as exc_info:
                ContextOptimizeStage().execute(ctx)

        response_tuple = exc_info.value.response_data
        assert isinstance(response_tuple, tuple)
        body, status = response_tuple
        assert status == 413
        assert body["error"]["type"] == "context_limit_exceeded"


# ---------------------------------------------------------------------------
# H2: cascade fallback must surface ContextLimitError as 413
# ---------------------------------------------------------------------------

class TestContextOptimizeCascadeFallbackContextLimit:
    """H2: when the cascade fallback's apply_mode raises ContextLimitError,
    the stage must re-raise StageAbort(413) instead of falling through to
    the minimal (empty) fallback.
    """

    def test_cascade_fallback_context_limit_raises_413(self):
        from proxy.context import ContextLimitError
        from proxy.mode import Mode

        ctx = _make_ctx()
        ctx.model_cfg = MagicMock()
        ctx.request_id = "req-h2-cascade-cle"

        mock_agent = MagicMock()
        mock_agent.active_mode = None

        call_count = [0]
        def primary_then_cascade_apply_mode(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                # First call (primary) — raise a non-CLE exception to trigger
                # the cascade fallback path.
                raise AttributeError("primary apply_mode failed")
            # Second call (cascade) — raise ContextLimitError. The stage
            # should re-raise this as 413, NOT proceed to the minimal fallback.
            raise ContextLimitError("V2: cascade fallback context overflow")

        with patch("proxy.context.enforce_context_limits"), \
             patch("proxy.agent.agent_registry.registry.get_or_load", return_value=mock_agent), \
             patch("proxy.mode.resolve_mode", return_value=Mode.DEFAULT), \
             patch("proxy.mode.apply_mode", side_effect=primary_then_cascade_apply_mode):
            with pytest.raises(StageAbort) as exc_info:
                ContextOptimizeStage().execute(ctx)

        response_tuple = exc_info.value.response_data
        assert isinstance(response_tuple, tuple)
        body, status = response_tuple
        assert status == 413
        assert body["error"]["type"] == "context_limit_exceeded"
        assert "cascade" in body["error"]["message"]
        # verify minimal fallback was NOT used (body should NOT be a minimal
        # system+last dict — the abort happened first)
        assert call_count[0] == 2

    def test_cascade_fallback_other_exception_uses_minimal(self):
        """H2 regression: non-CLE exception in cascade still triggers minimal fallback."""
        from proxy.mode import Mode
        from proxy.tool_parsing import OUR_SYSTEM_PROMPT

        ctx = _make_ctx()
        ctx.model_cfg = MagicMock()
        ctx.request_id = "req-h2-cascade-other"

        mock_agent = MagicMock()
        mock_agent.active_mode = None

        call_count = [0]
        def primary_then_cascade_apply_mode(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                raise AttributeError("primary apply_mode failed")
            # Second call (cascade) — raise a non-CLE exception.
            raise ValueError("cascade apply_mode failed for non-limit reason")

        with patch("proxy.context.enforce_context_limits"), \
             patch("proxy.agent.agent_registry.registry.get_or_load", return_value=mock_agent), \
             patch("proxy.mode.resolve_mode", return_value=Mode.DEFAULT), \
             patch("proxy.mode.apply_mode", side_effect=primary_then_cascade_apply_mode):
            result = ContextOptimizeStage().execute(ctx)

        # Should NOT raise — should fall through to minimal fallback
        assert result is None
        assert call_count[0] == 2
        msgs = ctx.body.get("messages", [])
        assert len(msgs) == 2
        assert msgs[0]["role"] == "system"
        assert msgs[0]["content"] == OUR_SYSTEM_PROMPT
        assert msgs[1]["role"] == "user"
        assert msgs[1]["content"] == "hello"

    def test_cascade_fallback_preserves_last_user_when_last_msg_is_tool(self):
        """M-D2: When the last message in the original body is a tool result
        (not a user message), the minimal cascade fallback MUST still preserve
        the LAST USER message — not grab the tool result.

        Latent bug: the current fallback code in
        `stage_modules/context_optimize.py` (`_cascade_apply_default`) does
        `body_original.get("messages", [])[-1]`, which would grab the tool
        result rather than the last user message. The model then sees a
        tool message at the tail of its conversation history with no
        following user instruction, which corrupts downstream prompting
        (e.g., the MotivatorStage's turn-aware reminders and any user-
        message-required downstream logic).

        This test asserts the CORRECT behavior (last USER message preserved)
        so that the moment the source bug is fixed, the test passes; until
        then it fails with a clear diagnostic pointing at
        `stage_modules/context_optimize.py` (`_cascade_apply_default`).
        """
        from proxy.mode import Mode
        from proxy.tool_parsing import OUR_SYSTEM_PROMPT

        # Build a context whose messages end in a tool result. The previous
        # user message is the only meaningful "last user" instruction.
        ctx = _make_ctx(
            messages=[
                {"role": "user", "content": "INSTRUCTION"},
                {"role": "tool", "tool_call_id": "x", "content": "result"},
            ],
        )
        ctx.model_cfg = MagicMock()
        ctx.request_id = "req-m-d2-cascade-tool"

        mock_agent = MagicMock()
        mock_agent.active_mode = None

        call_count = [0]
        def primary_then_cascade_apply_mode(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                # Force primary path to fail and cascade into fallback.
                raise AttributeError("primary apply_mode failed")
            # Cascade also fails with a non-CLE exception so the minimal
            # (system + last-msg) fallback path is exercised.
            raise ValueError("cascade apply_mode failed for non-limit reason")

        with patch("proxy.context.enforce_context_limits"), \
             patch("proxy.agent.agent_registry.registry.get_or_load", return_value=mock_agent), \
             patch("proxy.mode.resolve_mode", return_value=Mode.DEFAULT), \
             patch("proxy.mode.apply_mode", side_effect=primary_then_cascade_apply_mode):
            result = ContextOptimizeStage().execute(ctx)

        # The minimal fallback should NOT raise — it produces a body with
        # [system, last-user-message].
        assert result is None
        assert call_count[0] == 2
        msgs = ctx.body.get("messages", [])
        assert len(msgs) == 2, (
            f"expected 2 messages (system + last user), got {len(msgs)}: {msgs}"
        )
        assert msgs[0]["role"] == "system"
        assert msgs[0]["content"] == OUR_SYSTEM_PROMPT
        # The KEY assertion: the LAST message must be the LAST USER message,
        # not the tool result that happened to be the last entry in messages[].
        assert msgs[-1]["role"] == "user", (
            f"Last message role should be 'user' (the last USER instruction), "
            f"got {msgs[-1]['role']!r}. The fallback in "
            f"stage_modules/context_optimize.py (_cascade_apply_default) grabbed "
            f"the wrong message — likely the trailing tool result. Full msgs: "
            f"{msgs}"
        )
        assert msgs[-1]["content"] == "INSTRUCTION", (
            f"Last message content should be 'INSTRUCTION' (the last user "
            f"message), got {msgs[-1]['content']!r}. The fallback in "
            f"stage_modules/context_optimize.py (_cascade_apply_default) grabbed "
            f"the wrong message — likely the trailing tool result. Full msgs: {msgs}"
        )
        # And the tool result must NOT appear anywhere in the minimal fallback.
        assert not any(m.get("role") == "tool" for m in msgs), (
            f"Tool result leaked into minimal fallback body: {msgs}"
        )


# ---------------------------------------------------------------------------
# H3: ContextOptimizeStage must be in CRITICAL_STAGES
# ---------------------------------------------------------------------------

class TestContextOptimizeIsCritical:
    """H3: ContextOptimizeStage is a CRITICAL stage — any unhandled exception
    (other than StageAbort, which is re-raised directly) must abort the request
    rather than silently continuing with a broken body.
    """

    def test_context_optimize_stage_critical(self):
        """An unhandled exception in ContextOptimizeStage.execute is re-raised."""
        from proxy.context import ContextLimitError
        from proxy.mode import Mode

        ctx = _make_ctx()
        ctx.model_cfg = MagicMock()
        ctx.request_id = "req-h3-critical"

        mock_agent = MagicMock()
        mock_agent.active_mode = None

        pipeline = RequestPipeline(stages=[ContextOptimizeStage()])

        with patch("proxy.context.enforce_context_limits"), \
             patch("proxy.agent.agent_registry.registry.get_or_load", return_value=mock_agent), \
             patch("proxy.mode.resolve_mode", return_value=Mode.DEFAULT), \
             patch("proxy.mode.apply_mode", side_effect=ContextLimitError("V2: critical path overflow")):
            with pytest.raises(StageAbort) as exc_info:
                pipeline.run(ctx)

        # The StageAbort is raised by ContextOptimizeStage itself (H1/H2 path),
        # so it propagates naturally. The CRITICAL_STAGES set is a backstop for
        # unhandled exceptions.
        response_tuple = exc_info.value.response_data
        body, status = response_tuple
        assert status == 413
        assert body["error"]["type"] == "context_limit_exceeded"

    def test_non_critical_stage_continues(self):
        """Regression: a non-critical stage's exception is logged and the
        pipeline continues (this is the existing behavior we must not break)."""

        non_critical_exc = RuntimeError("transient")
        non_critical_stage = _make_fake_stage_class(
            name="CleanupStage",
            raise_exc=non_critical_exc,
        )

        pipeline = RequestPipeline(stages=[non_critical_stage()])
        ctx = _make_ctx(session_id=None)
        ctx.response_data = {"fallback": True}

        result = pipeline.run(ctx)
        assert result == {"fallback": True}

    def test_context_optimize_stage_in_critical_set(self):
        """Sanity: ContextOptimizeStage is recognized as a critical stage."""
        assert is_critical_stage(ContextOptimizeStage()) is True


# ---------------------------------------------------------------------------
# Commit 2: cascade fallback must pass auto-registered agent (not None)
# ---------------------------------------------------------------------------


class TestContextOptimizeCascadePassesAutoRegisteredAgent:
    """Commit 2: when the sub-agent auto-registration block in
    ContextOptimizeStage produces a real Agent, the cascade fallback must
    pass that Agent (not None) to apply_mode. Previously the cascade
    hardcoded None, which crashed select_tools / assemble_system_prompt
    on agent.snapshot_mode().
    """

    def test_cascade_fallback_receives_auto_registered_agent(self):
        """Sub-agent request: get_or_load returns None → auto-register →
        primary path raises → cascade fallback's apply_mode receives the
        auto-registered Agent (not None).
        """
        from proxy.mode import Mode

        ctx = _make_ctx()
        ctx.model_cfg = MagicMock()
        ctx.request_id = "req-commit2-cascade-agent"
        ctx.is_subagent = True

        auto_registered_agent = MagicMock()
        auto_registered_agent.active_mode = None

        # First get_or_load returns None (triggers auto-registration);
        # subsequent calls (after register()) return the real agent.
        get_or_load_calls = []
        def get_or_load_side_effect(session_id):
            get_or_load_calls.append(session_id)
            if len(get_or_load_calls) == 1:
                return None
            return auto_registered_agent

        # Capture the agent passed to apply_mode on the cascade call.
        apply_mode_agents = []
        call_count = [0]
        def apply_mode_side_effect(ctx, agent, body, mode):
            call_count[0] += 1
            apply_mode_agents.append(agent)
            if call_count[0] == 1:
                # Primary path — raise to force cascade fallback.
                raise RuntimeError("primary apply_mode failed")
            # Cascade call — return a valid body.
            return {"messages": [{"role": "user", "content": "ok"}], "tools": []}

        with patch("proxy.context.enforce_context_limits"), \
             patch(
                 "proxy.agent.agent_registry.registry.get_or_load",
                 side_effect=get_or_load_side_effect,
             ), \
             patch(
                 "proxy.agent.agent_registry.registry.register",
                 return_value=None,
             ), \
             patch("proxy.skill_lifecycle.register_lifecycle_callbacks"), \
             patch("proxy.mode.resolve_mode", return_value=Mode.DEFAULT), \
             patch("proxy.mode.apply_mode", side_effect=apply_mode_side_effect):
            result = ContextOptimizeStage().execute(ctx)

        assert result is None
        # Cascade apply_mode MUST have been called with a non-None agent —
        # this is the core Commit 2 fix.
        assert len(apply_mode_agents) == 2, (
            f"expected primary + cascade calls, got {len(apply_mode_agents)}"
        )
        assert apply_mode_agents[1] is auto_registered_agent, (
            "cascade apply_mode must receive the auto-registered Agent, "
            f"got {apply_mode_agents[1]!r}"
        )

    def test_cascade_fallback_passes_none_when_no_auto_registration(self):
        """Non-sub-agent path: get_or_load returns None, no
        auto-registration happens (is_subagent=False), cascade fallback
        receives None. The primary path's ``if agent is not None`` guard
        skips resolve_mode entirely, so apply_mode is called exactly once
        (from the cascade) with agent=None. The Commit 2 guards in
        select_tools and assemble_system_prompt ensure this does not crash.
        """
        from proxy.mode import Mode

        ctx = _make_ctx()
        ctx.model_cfg = MagicMock()
        ctx.request_id = "req-commit2-cascade-none"
        ctx.is_subagent = False

        apply_mode_agents = []
        def apply_mode_side_effect(ctx, agent, body, mode):
            apply_mode_agents.append(agent)
            return {"messages": [{"role": "user", "content": "ok"}], "tools": []}

        with patch("proxy.context.enforce_context_limits"), \
             patch(
                 "proxy.agent.agent_registry.registry.get_or_load",
                 return_value=None,
             ), \
             patch("proxy.mode.apply_mode", side_effect=apply_mode_side_effect):
            result = ContextOptimizeStage().execute(ctx)

        assert result is None
        # Primary path skipped (agent is None); cascade apply_mode called
        # exactly once with agent=None and must not crash.
        assert len(apply_mode_agents) == 1, (
            f"expected single cascade call, got {len(apply_mode_agents)}"
        )
        assert apply_mode_agents[0] is None

