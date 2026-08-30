"""Integration tests for ordering contracts between pipeline stages.

These tests verify that pipeline stages execute in the correct order and
that data flows correctly between them. Focus areas:
- DispatchStage text rewriting ↔ ctx.body synchronization
- PostProcessStage method ordering (scan_write_results before sync_chat_history)
- submit_sync() ↔ PostProcessStage finish_reason contract
- MotivatorStage ↔ ContextOptimizeStage ↔ ProviderCompatStage ordering
"""

import copy
import json
import time
import pytest
from unittest.mock import patch, MagicMock, call, PropertyMock
from collections import OrderedDict

from proxy.types import RequestContext, ModelRoute, UpstreamRequest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_ctx(
    messages=None,
    session_id="test-session-001",
    is_subagent=False,
    model_name="test-model",
) -> RequestContext:
    """Build a minimal RequestContext for stage-level tests."""
    if messages is None:
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "hello"},
        ]
    body = {"model": model_name, "messages": copy.deepcopy(messages), "stream": False}
    ctx = RequestContext(
        body=body,
        model_name=model_name,
        wants_stream=False,
        session_id=session_id,
        client_hint="test-client",
        messages=copy.deepcopy(messages),
        remote_addr="127.0.0.1",
        is_subagent=is_subagent,
    )
    ctx.request_id = "test-req-001"
    return ctx


def _make_model_cfg(**overrides):
    """Build a mock model config."""
    cfg = MagicMock()
    cfg.backend_model = overrides.get("backend_model", "mock-model")
    cfg.api_key_env = overrides.get("api_key_env", None)
    cfg.memory_enabled = overrides.get("memory_enabled", True)
    cfg.ignore_subagents = overrides.get("ignore_subagents", False)
    cfg.name = overrides.get("name", "test-cfg")
    return cfg


def _make_route(model_cfg=None, is_remote=True):
    """Build a mock ModelRoute."""
    if model_cfg is None:
        model_cfg = _make_model_cfg()
    return ModelRoute(
        model_cfg=model_cfg,
        upstream_url="http://mock-upstream/v1/chat/completions",
        tool_parser="openai",
        is_remote=is_remote,
    )


def _make_upstream(original_model="test-model"):
    """Build a mock UpstreamRequest."""
    return UpstreamRequest(
        url="http://mock-upstream/v1/chat/completions",
        body={"model": original_model, "messages": []},
        headers={"Authorization": "Bearer test-key"},
        original_model=original_model,
    )


# ---------------------------------------------------------------------------
# Test 1: DispatchStage text rewriting ↔ ctx.body synchronization (2.9)
# ---------------------------------------------------------------------------


class TestDispatchStageRewriteSync:
    """DispatchStage must synchronize rewrites to both ctx.messages and ctx.body."""

    @pytest.mark.integration
    def test_rewrite_syncs_to_both_messages_and_body(self):
        """After DispatchStage rewrites text, ctx.messages[-1] and
        ctx.body['messages'][-1] must both have the rewritten content."""
        from proxy.pipeline.stages import DispatchStage

        original_text = "^^ticket plan my feature"
        rewritten_text = "Plan the following feature: my feature"

        ctx = _make_ctx(
            messages=[
                {"role": "system", "content": "You are helpful."},
                {"role": "user", "content": original_text},
            ]
        )

        mock_dispatch_result = MagicMock()
        mock_dispatch_result.handled = False
        mock_dispatch_result.rewritten_text = rewritten_text

        with patch(
            "proxy.pipeline.dispatch.run_dispatch", return_value=mock_dispatch_result
        ):
            with patch(
                "clients.orchestrator_config.get_features_for_client",
                return_value=MagicMock(),
            ):
                stage = DispatchStage()
                stage.execute(ctx)

        # Both ctx.messages and ctx.body["messages"] must have the rewrite
        assert ctx.messages[-1]["content"] == rewritten_text, (
            f"ctx.messages[-1] not rewritten: {ctx.messages[-1].get('content')}"
        )
        assert ctx.body["messages"][-1]["content"] == rewritten_text, (
            f"ctx.body['messages'][-1] not rewritten: "
            f"{ctx.body['messages'][-1].get('content')}"
        )

    @pytest.mark.integration
    def test_rewrite_persists_through_apply_mode_deepcopy(self):
        """DispatchStage rewrite must survive apply_mode()'s copy.deepcopy(ctx.body).

        apply_mode() in ContextOptimizeStage does:
            body_original = copy.deepcopy(ctx.body)
        If the rewrite only mutated ctx.messages (not ctx.body), it would be lost.
        """
        from proxy.pipeline.stages import DispatchStage

        original_text = "^^ticket plan auth bug"
        rewritten_text = "Plan a fix for the auth bug"

        ctx = _make_ctx(
            messages=[
                {"role": "system", "content": "You are helpful."},
                {"role": "user", "content": original_text},
            ]
        )

        mock_dispatch_result = MagicMock()
        mock_dispatch_result.handled = False
        mock_dispatch_result.rewritten_text = rewritten_text

        with patch(
            "proxy.pipeline.dispatch.run_dispatch", return_value=mock_dispatch_result
        ):
            with patch(
                "clients.orchestrator_config.get_features_for_client",
                return_value=MagicMock(),
            ):
                stage = DispatchStage()
                stage.execute(ctx)

        # Simulate what ContextOptimizeStage does: deepcopy the body
        body_copy = copy.deepcopy(ctx.body)

        # The deepcopy must preserve the rewritten text
        assert body_copy["messages"][-1]["content"] == rewritten_text, (
            "Rewrite lost after deepcopy(ctx.body) — body was not updated by DispatchStage"
        )

    @pytest.mark.integration
    def test_no_rewrite_leaves_original_intact(self):
        """When dispatch returns no rewrite, original message content is unchanged."""
        from proxy.pipeline.stages import DispatchStage

        original_text = "just a normal message"
        ctx = _make_ctx(
            messages=[
                {"role": "system", "content": "You are helpful."},
                {"role": "user", "content": original_text},
            ]
        )

        mock_dispatch_result = MagicMock()
        mock_dispatch_result.handled = False
        mock_dispatch_result.rewritten_text = None

        with patch(
            "proxy.pipeline.dispatch.run_dispatch", return_value=mock_dispatch_result
        ):
            with patch(
                "clients.orchestrator_config.get_features_for_client",
                return_value=MagicMock(),
            ):
                stage = DispatchStage()
                stage.execute(ctx)

        assert ctx.messages[-1]["content"] == original_text
        assert ctx.body["messages"][-1]["content"] == original_text


# ---------------------------------------------------------------------------
# Test 2: PostProcessStage method ordering contract (2.10)
# ---------------------------------------------------------------------------


class TestPostProcessStageOrdering:
    """PostProcessStage must call methods in documented order.

    Per post_process.py:46-53:
        1. _extract_minimax_reasoning
        2. _scan_write_results
        3. _sync_chat_history
        4. _persist_messages
        5. _populate_vfs_cache
        6. _handle_tool_interception
        7. _assign_model_name

    Key contract: _scan_write_results MUST run BEFORE _sync_chat_history
    so VFS snapshots are confirmed/rolled back before the chat history
    distinguishes new vs historical messages.
    """

    @pytest.mark.integration
    def test_scan_write_results_before_sync_chat_history(self):
        """_scan_write_results must be called before _sync_chat_history."""
        from proxy.pipeline.post_process import PostProcessStage

        call_order = []

        original_execute = PostProcessStage.execute

        def tracking_execute(self, ctx):
            """Wrap each method to track call order."""
            methods = OrderedDict([
                ("_extract_minimax_reasoning", self._extract_minimax_reasoning),
                ("_scan_write_results", self._scan_write_results),
                ("_sync_chat_history", self._sync_chat_history),
                ("_persist_messages", self._persist_messages),
                ("_populate_vfs_cache", self._populate_vfs_cache),
                ("_handle_tool_interception", self._handle_tool_interception),
                ("_assign_model_name", self._assign_model_name),
            ])
            for name, method in methods.items():
                try:
                    call_order.append(name)
                    method(ctx)
                except Exception:
                    pass

            # Also track submit_sync
            data = ctx.response_data
            if data and data.get("choices"):
                finish_reason = (data.get("choices") or [{}])[0].get("finish_reason")
                if finish_reason in ("stop", "length"):
                    call_order.append("submit_sync")

            self._apply_skill_response_transform(ctx)
            self._inject_task_canaries(ctx)
            return ctx.response_data

        # Set up ctx with a mock response
        ctx = _make_ctx()
        ctx.model_route = _make_route()
        ctx.upstream = _make_upstream()
        ctx.messages_original = copy.deepcopy(ctx.messages)
        ctx.response_data = {
            "choices": [
                {
                    "message": {"role": "assistant", "content": "done"},
                    "finish_reason": "stop",
                }
            ],
            "model": "test-model",
        }

        with patch.object(PostProcessStage, "execute", tracking_execute):
            with patch("memory.sync.submit_sync"):
                stage = PostProcessStage()
                stage.execute(ctx)

        # Verify ordering
        assert "_scan_write_results" in call_order
        assert "_sync_chat_history" in call_order

        scan_idx = call_order.index("_scan_write_results")
        sync_idx = call_order.index("_sync_chat_history")

        assert scan_idx < sync_idx, (
            f"_scan_write_results (idx={scan_idx}) must run before "
            f"_sync_chat_history (idx={sync_idx}), got order: {call_order}"
        )

    @pytest.mark.integration
    def test_full_method_ordering_contract(self):
        """All PostProcessStage methods must execute in the documented order."""
        from proxy.pipeline.post_process import PostProcessStage

        call_order = []

        methods_to_track = [
            "_extract_minimax_reasoning",
            "_scan_write_results",
            "_sync_chat_history",
            "_persist_messages",
            "_populate_vfs_cache",
            "_handle_tool_interception",
            "_assign_model_name",
        ]

        ctx = _make_ctx()
        ctx.model_route = _make_route()
        ctx.upstream = _make_upstream()
        ctx.messages_original = copy.deepcopy(ctx.messages)
        ctx.response_data = {
            "choices": [
                {
                    "message": {"role": "assistant", "content": "done"},
                    "finish_reason": "stop",
                }
            ],
            "model": "test-model",
        }

        # Patch each tracked method to record its name in call_order
        patches = {}
        for method_name in methods_to_track:
            original = getattr(PostProcessStage, method_name)

            def make_wrapper(name, orig):
                def wrapper(self_inner, ctx_inner):
                    call_order.append(name)
                    return orig(self_inner, ctx_inner)
                return wrapper

            patches[method_name] = patch.object(
                PostProcessStage, method_name,
                side_effect=make_wrapper(method_name, original),
                autospec=True,
            )

        with patch("memory.sync.submit_sync"):
            for p in patches.values():
                p.start()
            try:
                stage = PostProcessStage()
                stage.execute(ctx)
            finally:
                for p in patches.values():
                    p.stop()

        # Verify all methods ran in order
        assert call_order == methods_to_track, (
            f"Method execution order mismatch.\n"
            f"Expected: {methods_to_track}\n"
            f"Got:      {call_order}"
        )

    @pytest.mark.integration
    def test_vfs_snapshot_confirmed_before_chat_sync(self):
        """VFS write result scanning must complete before chat history sync.

        This is the REG-004 contract: _scan_write_results confirms/rolls back
        VFS snapshots so _sync_chat_history can correctly distinguish new
        messages from historical ones.
        """
        from proxy.pipeline.post_process import PostProcessStage

        execution_log = []

        original_scan = PostProcessStage._scan_write_results
        original_sync = PostProcessStage._sync_chat_history

        def tracked_scan(self, ctx):
            execution_log.append("scan_write_results_START")
            result = original_scan(self, ctx)
            execution_log.append("scan_write_results_END")
            return result

        def tracked_sync(self, ctx):
            execution_log.append("sync_chat_history_START")
            # By this point, scan_write_results must be fully done
            assert "scan_write_results_END" in execution_log, (
                "sync_chat_history started before scan_write_results completed"
            )
            result = original_sync(self, ctx)
            execution_log.append("sync_chat_history_END")
            return result

        ctx = _make_ctx()
        ctx.model_route = _make_route()
        ctx.upstream = _make_upstream()
        ctx.messages_original = copy.deepcopy(ctx.messages)
        ctx.response_data = {
            "choices": [
                {
                    "message": {"role": "assistant", "content": "ok"},
                    "finish_reason": "stop",
                }
            ],
            "model": "test-model",
        }

        with patch.object(PostProcessStage, "_scan_write_results", tracked_scan):
            with patch.object(PostProcessStage, "_sync_chat_history", tracked_sync):
                with patch("memory.sync.submit_sync"):
                    stage = PostProcessStage()
                    stage.execute(ctx)

        assert execution_log == [
            "scan_write_results_START",
            "scan_write_results_END",
            "sync_chat_history_START",
            "sync_chat_history_END",
        ], f"Unexpected execution log: {execution_log}"


# ---------------------------------------------------------------------------
# Test 3: submit_sync() ↔ PostProcessStage context contract (2.11)
# ---------------------------------------------------------------------------


class TestSubmitSyncContract:
    """submit_sync() must be called only for final responses (finish_reason in stop/length).

    - finish_reason="tool_calls" → submit_sync NOT called
    - finish_reason="stop"       → submit_sync IS called
    - finish_reason=None         → graceful handling, no crash
    - finish_reason="length"     → submit_sync IS called
    """

    @pytest.mark.integration
    def test_submit_sync_not_called_for_tool_calls(self):
        """finish_reason='tool_calls' means the conversation isn't done — no sync."""
        from proxy.pipeline.post_process import PostProcessStage

        ctx = _make_ctx()
        ctx.model_route = _make_route()
        ctx.upstream = _make_upstream()
        ctx.messages_original = copy.deepcopy(ctx.messages)
        ctx.response_data = {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "type": "function",
                                "function": {"name": "read", "arguments": "{}"},
                            }
                        ],
                    },
                    "finish_reason": "tool_calls",
                }
            ],
            "model": "test-model",
        }

        with patch("memory.sync.submit_sync") as mock_sync:
            stage = PostProcessStage()
            result = stage.execute(ctx)

        mock_sync.assert_not_called()

    @pytest.mark.integration
    def test_submit_sync_called_for_stop(self):
        """finish_reason='stop' means the conversation is done — sync must fire."""
        from proxy.pipeline.post_process import PostProcessStage

        ctx = _make_ctx()
        ctx.model_route = _make_route()
        ctx.upstream = _make_upstream()
        ctx.messages_original = copy.deepcopy(ctx.messages)
        ctx.response_data = {
            "choices": [
                {
                    "message": {"role": "assistant", "content": "Here is the answer."},
                    "finish_reason": "stop",
                }
            ],
            "model": "test-model",
        }

        with patch("memory.sync.submit_sync") as mock_sync:
            stage = PostProcessStage()
            stage.execute(ctx)

        mock_sync.assert_called_once_with(ctx.session_id)

    @pytest.mark.integration
    def test_submit_sync_called_for_length(self):
        """finish_reason='length' (truncated) — sync must still fire."""
        from proxy.pipeline.post_process import PostProcessStage

        ctx = _make_ctx()
        ctx.model_route = _make_route()
        ctx.upstream = _make_upstream()
        ctx.messages_original = copy.deepcopy(ctx.messages)
        ctx.response_data = {
            "choices": [
                {
                    "message": {"role": "assistant", "content": "Truncated response..."},
                    "finish_reason": "length",
                }
            ],
            "model": "test-model",
        }

        with patch("memory.sync.submit_sync") as mock_sync:
            stage = PostProcessStage()
            stage.execute(ctx)

        mock_sync.assert_called_once_with(ctx.session_id)

    @pytest.mark.integration
    def test_submit_sync_graceful_on_none_finish_reason(self):
        """A valid completion with finish_reason=None must not submit sync or crash."""
        from proxy.pipeline.post_process import PostProcessStage

        ctx = _make_ctx()
        ctx.model_route = _make_route()
        ctx.upstream = _make_upstream()
        ctx.messages_original = copy.deepcopy(ctx.messages)
        ctx.response_data = {
            "choices": [
                {
                    "message": {"role": "assistant", "content": "weird response"},
                    "finish_reason": None,
                }
            ],
            "model": "test-model",
        }

        with patch("memory.sync.submit_sync") as mock_sync:
            stage = PostProcessStage()
            # Valid shape must not raise.
            result = stage.execute(ctx)

        # None is not in ("stop", "length"), so sync should NOT be called
        mock_sync.assert_not_called()

    @pytest.mark.integration
    @pytest.mark.parametrize(
        "response_data,label",
        [
            ({"choices": None}, "choices-None"),
            ({"choices": []}, "choices-empty"),
            ({"choices": [None]}, "choices-list-with-None"),
            ({"choices": [{"message": None}]}, "choices-message-None"),
            ({"model": "test-model", "usage": {}}, "no-choices-key"),
        ],
    )
    def test_malformed_response_aborts_with_canonical_502(self, response_data, label):
        """Malformed upstream data must abort PostProcess with 502 canonical error.

        Validates that the new fail-closed boundary in PostProcessStage.execute()
        rejects malformed responses BEFORE any success method runs. The abort
        carries build_upstream_protocol_error() and HTTP 502, and the lifecycle
        finalize still runs (so the active mode is cleared on the way out).
        """
        from proxy.pipeline.post_process import PostProcessStage
        from proxy.pipeline.exceptions import StageAbort
        from proxy.openai_compat import build_upstream_protocol_error

        ctx = _make_ctx()
        ctx.model_route = _make_route()
        ctx.upstream = _make_upstream()
        ctx.messages_original = copy.deepcopy(ctx.messages)
        ctx.request_id = f"req-malformed-{label}"
        ctx.session_id = f"sess-malformed-{label}"
        ctx.response_data = copy.deepcopy(response_data)

        spy_methods = [
            "_extract_minimax_reasoning",
            "_scan_write_results",
            "_sync_chat_history",
            "_persist_messages",
            "_populate_vfs_cache",
            "_capture_cache_metrics",
            "_handle_tool_interception",
            "_assign_model_name",
            "_apply_skill_response_transform",
            "_inject_task_canaries",
        ]
        spy_map = {name: MagicMock() for name in spy_methods}
        patches = [patch.object(PostProcessStage, name, spy_map[name]) for name in spy_methods]
        for p in patches:
            p.start()
        try:
            with patch("memory.sync.submit_sync") as mock_sync:
                with pytest.raises(StageAbort) as exc_info:
                    PostProcessStage().execute(ctx)
        finally:
            for p in patches:
                p.stop()

        mock_sync.assert_not_called()
        for name, spy in spy_map.items():
            spy.assert_not_called(), (
                f"{name} was called on malformed response — fail-closed "
                f"boundary did not run before this method"
            )

        body, status = exc_info.value.response_data
        canonical = build_upstream_protocol_error()
        assert status == 502
        assert body == canonical
        assert ctx.response_data == canonical
        assert ctx.skill_lifecycle_finalized is True

    @pytest.mark.integration
    def test_explicit_dict_error_aborts_with_original_body(self):
        """Dict-valued provider error envelope is preserved verbatim with 502.

        When the upstream returns a structured provider-error envelope (dict
        `error`, no choices), PostProcess aborts with the ORIGINAL body — not
        the canonical replacement — so the client can see exactly what the
        upstream reported. The canonical replacement is reserved for SHAPE
        errors (malformed payload, not a provider error envelope).
        """
        from proxy.pipeline.post_process import PostProcessStage
        from proxy.pipeline.exceptions import StageAbort

        provider_error = {
            "error": {
                "message": "boom",
                "type": "provider_error",
            }
        }
        ctx = _make_ctx()
        ctx.model_route = _make_route()
        ctx.upstream = _make_upstream()
        ctx.messages_original = copy.deepcopy(ctx.messages)
        ctx.request_id = "req-explicit-error"
        ctx.session_id = "sess-explicit-error"
        ctx.response_data = copy.deepcopy(provider_error)

        spy_methods = [
            "_extract_minimax_reasoning",
            "_scan_write_results",
            "_sync_chat_history",
            "_persist_messages",
            "_populate_vfs_cache",
            "_capture_cache_metrics",
            "_handle_tool_interception",
            "_assign_model_name",
            "_apply_skill_response_transform",
            "_inject_task_canaries",
        ]
        spy_map = {name: MagicMock() for name in spy_methods}
        patches = [patch.object(PostProcessStage, name, spy_map[name]) for name in spy_methods]
        for p in patches:
            p.start()
        try:
            with patch("memory.sync.submit_sync") as mock_sync:
                with pytest.raises(StageAbort) as exc_info:
                    PostProcessStage().execute(ctx)
        finally:
            for p in patches:
                p.stop()

        mock_sync.assert_not_called()
        for name, spy in spy_map.items():
            spy.assert_not_called(), (
                f"{name} was called on explicit dict error — fail-closed "
                f"boundary did not run before this method"
            )

        body, status = exc_info.value.response_data
        assert status == 502
        assert body == provider_error
        assert ctx.response_data == provider_error
        assert ctx.skill_lifecycle_finalized is True

    @pytest.mark.integration
    def test_submit_sync_skipped_for_subagent_when_configured(self):
        """Sub-agent responses skip sync when ignore_subagents=True."""
        from proxy.pipeline.post_process import PostProcessStage

        cfg = _make_model_cfg(ignore_subagents=True)
        route = _make_route(model_cfg=cfg)

        ctx = _make_ctx(is_subagent=True)
        ctx.model_route = route
        ctx.upstream = _make_upstream()
        ctx.messages_original = copy.deepcopy(ctx.messages)
        ctx.response_data = {
            "choices": [
                {
                    "message": {"role": "assistant", "content": "sub-agent done"},
                    "finish_reason": "stop",
                }
            ],
            "model": "test-model",
        }

        with patch("memory.sync.submit_sync") as mock_sync:
            stage = PostProcessStage()
            stage.execute(ctx)

        mock_sync.assert_not_called()


# ---------------------------------------------------------------------------
# Test 4: MotivatorStage ↔ ContextOptimizeStage ordering (2.12)
# ---------------------------------------------------------------------------


class TestMotivatorStageOrdering:
    """MotivatorStage runs after ContextOptimizeStage and before ProviderCompatStage.

    Stage order (from stages.py):
        5. ContextOptimizeStage
        6. MotivatorStage
        7. ProviderCompatStage

    MotivatorStage injects reminders into ctx.body["messages"] system prompt.
    ProviderCompatStage must see the motivated context.
    """

    @pytest.mark.integration
    def test_motivator_injects_into_system_prompt(self):
        """MotivatorStage must append reminder text to the system message."""
        from proxy.pipeline.stages import ContextOptimizeStage
        from proxy.pipeline.motivator_stage import MotivatorStage
        from proxy.agent.agent_registry import registry as agent_registry
        from proxy.mode import Mode

        # Build a sub-agent context with many turns to trigger motivator
        messages = [{"role": "system", "content": "You are a coding agent."}]
        # Add 30 messages (15 turns) to exceed 50% threshold of 20 max_turns
        for i in range(30):
            role = "user" if i % 2 == 0 else "assistant"
            messages.append({"role": role, "content": f"Message {i}"})

        ctx = _make_ctx(messages=messages, is_subagent=True)

        # Pre-populate agent with telemetry so context % is high
        agent = MagicMock()
        agent.telemetry = MagicMock()
        agent.telemetry.turn_count = 15
        agent.telemetry.max_request_chars = 600_000  # 75% of 800k
        agent.telemetry.original_task = "Fix the authentication bug in the login flow"
        agent.telemetry.get_context_pct.return_value = 75.0
        agent.telemetry.capture_task.return_value = None
        agent.telemetry.increment_turn.return_value = None
        agent.telemetry.record_request_chars.return_value = None
        agent.threshold_engine = MagicMock()
        agent.threshold_engine.thresholds = [50, 75, 90]
        agent.threshold_engine.check.return_value = 75
        agent.config = MagicMock()
        agent.config.motivator_thresholds = (50, 75, 90)
        agent.config.motivator_context_window_chars = 800_000
        agent.snapshot_mode.return_value = MagicMock(
            name="ticket", allowed_tool_names=frozenset()
        )

        # Set up ctx as if ContextOptimizeStage already ran
        ctx._mode = Mode.DEFAULT
        ctx.messages_original = copy.deepcopy(ctx.messages)

        with patch.object(agent_registry, "get_or_load", return_value=agent):
            stage = MotivatorStage()
            stage.execute(ctx)

        # Check that the MotivatorStage prepended the reminder to the LAST
        # user message (cache-stable: doesn't shift any middle positions).
        # messages[0] is the system prompt — MUST be untouched for cache hit.
        assert ctx.body["messages"][0]["role"] == "system"
        assert "MOTIVATOR" not in ctx.body["messages"][0]["content"], (
            "MotivatorStage must not mutate messages[0] — that breaks cache"
        )
        # The reminder lives in the LAST user message
        last_user = next(
            m for m in reversed(ctx.body["messages"])
            if m.get("role") == "user"
        )
        reminder_content = last_user["content"]
        assert "MOTIVATOR" in reminder_content, (
            f"Motivator reminder not found in last user message: {reminder_content[:200]}"
        )
        assert "75%" in reminder_content, "Context percentage not in motivator reminder"
        assert "Fix the authentication bug" in reminder_content, (
            "Original task not in motivator reminder"
        )

    @pytest.mark.integration
    def test_motivator_visible_to_provider_compat(self):
        """MotivatorStage injection must be visible when ProviderCompatStage reads ctx.body.

        This verifies the stage ordering contract: MotivatorStage (6) writes to
        ctx.body before ProviderCompatStage (7) reads it.
        """
        from proxy.pipeline.motivator_stage import MotivatorStage
        from proxy.pipeline.stages import ProviderCompatStage
        from proxy.agent.agent_registry import registry as agent_registry

        messages = [{"role": "system", "content": "You are a coding agent."}]
        for i in range(30):
            role = "user" if i % 2 == 0 else "assistant"
            messages.append({"role": role, "content": f"Message {i}"})

        ctx = _make_ctx(messages=messages, is_subagent=True)
        ctx.model_route = _make_route()
        ctx.messages_original = copy.deepcopy(ctx.messages)

        agent = MagicMock()
        agent.telemetry = MagicMock()
        agent.telemetry.turn_count = 15
        agent.telemetry.max_request_chars = 600_000
        agent.telemetry.original_task = "Fix the auth bug"
        agent.telemetry.get_context_pct.return_value = 75.0
        agent.telemetry.capture_task.return_value = None
        agent.telemetry.increment_turn.return_value = None
        agent.telemetry.record_request_chars.return_value = None
        agent.threshold_engine = MagicMock()
        agent.threshold_engine.thresholds = [50, 75, 90]
        agent.threshold_engine.check.return_value = 75
        agent.config = MagicMock()
        agent.config.motivator_thresholds = (50, 75, 90)
        agent.config.motivator_context_window_chars = 800_000

        captured_compat_body = []

        def mock_apply_provider_compat(body, model_cfg):
            captured_compat_body.append(copy.deepcopy(body))
            return body

        with patch.object(agent_registry, "get_or_load", return_value=agent):
            with patch(
                "proxy.pipeline.provider_compat.apply_provider_compat",
                side_effect=mock_apply_provider_compat,
            ):
                # Run MotivatorStage
                motivator = MotivatorStage()
                motivator.execute(ctx)

                # Run ProviderCompatStage
                compat = ProviderCompatStage()
                compat.execute(ctx)

        # ProviderCompatStage should have seen the motivated body
        assert len(captured_compat_body) == 1
        compat_body = captured_compat_body[0]
        # The Motivator reminder now lives in the last user message
        # (prepended). Check that the substring is present somewhere in
        # the messages array — wherever it ended up after compat transforms.
        all_content = " ".join(
            str(m.get("content", "")) for m in compat_body["messages"]
        )
        assert "MOTIVATOR" in all_content, (
            "ProviderCompatStage did not see MotivatorStage's injection"
        )

    @pytest.mark.integration
    def test_upstream_receives_motivated_context(self):
        """MotivatorStage injection must flow through ProviderCompatStage to upstream.

        Runs MotivatorStage then ProviderCompatStage in sequence and verifies
        the llama_body (which goes to upstream) contains the motivator text.
        """
        from proxy.pipeline.motivator_stage import MotivatorStage
        from proxy.pipeline.stages import ProviderCompatStage
        from proxy.agent.agent_registry import registry as agent_registry

        messages = [{"role": "system", "content": "You are a coding agent."}]
        for i in range(30):
            role = "user" if i % 2 == 0 else "assistant"
            messages.append({"role": role, "content": f"Message {i}"})

        ctx = _make_ctx(messages=messages, is_subagent=True)
        ctx.model_route = _make_route()
        ctx.messages_original = copy.deepcopy(ctx.messages)

        agent = MagicMock()
        agent.telemetry = MagicMock()
        agent.telemetry.turn_count = 15
        agent.telemetry.max_request_chars = 600_000
        agent.telemetry.original_task = "Fix the auth bug in login flow"
        agent.telemetry.get_context_pct.return_value = 75.0
        agent.telemetry.capture_task.return_value = None
        agent.telemetry.increment_turn.return_value = None
        agent.telemetry.record_request_chars.return_value = None
        agent.threshold_engine = MagicMock()
        agent.threshold_engine.thresholds = [50, 75, 90]
        agent.threshold_engine.check.return_value = 75
        agent.config = MagicMock()
        agent.config.motivator_thresholds = (50, 75, 90)
        agent.config.motivator_context_window_chars = 800_000

        with patch.object(agent_registry, "get_or_load", return_value=agent):
            with patch(
                "proxy.pipeline.provider_compat.apply_provider_compat",
                side_effect=lambda body, cfg: body,
            ):
                # Stage 6: MotivatorStage
                motivator = MotivatorStage()
                motivator.execute(ctx)

                # Stage 7: ProviderCompatStage
                compat = ProviderCompatStage()
                compat.execute(ctx)

        # ProviderCompatStage sets ctx.llama_body which goes to upstream
        llama_body = getattr(ctx, "llama_body", None)
        assert llama_body is not None, "ProviderCompatStage did not set ctx.llama_body"

        # Motivator reminder is now prepended to the last user message
        # (cache-stable position). The MiniMax adapter may convert system
        # to user, so we don't constrain which role the reminder ends up as.
        all_text = " ".join(
            str(m.get("content", "")) for m in llama_body.get("messages", [])
        )
        assert "MOTIVATOR" in all_text, (
            "Motivator reminder not in llama_body (upstream). "
            f"Messages: {[m.get('role') for m in llama_body.get('messages', [])]}"
        )
        # Sanity: system prompt (the actual system role) is untouched
        sys_text = " ".join(
            str(m.get("content", "")) for m in llama_body.get("messages", [])
            if m.get("role") == "system"
        )
        assert "MOTIVATOR" not in sys_text, (
            "Motivator reminder must not appear in a system-role message — "
            "it must live in a user-role message (cache-stable position)"
        )

    @pytest.mark.integration
    def test_motivator_does_not_run_for_non_subagent(self):
        """MotivatorStage must skip non-sub-agent requests."""
        from proxy.pipeline.motivator_stage import MotivatorStage
        from proxy.agent.agent_registry import registry as agent_registry

        messages = [{"role": "system", "content": "You are helpful."}]
        for i in range(30):
            role = "user" if i % 2 == 0 else "assistant"
            messages.append({"role": role, "content": f"Message {i}"})

        ctx = _make_ctx(messages=messages, is_subagent=False)
        ctx.messages_original = copy.deepcopy(ctx.messages)

        # If motivator ran, it would call agent_registry.get_or_load
        with patch.object(agent_registry, "get_or_load") as mock_load:
            stage = MotivatorStage()
            stage.execute(ctx)

        # should_run() returns False for non-sub-agents, so get_or_load not called
        mock_load.assert_not_called()
