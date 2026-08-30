"""Tests for MotivatorStage — sub-agent turn-aware prompt reminders.
# US-065: Motivator Turn-Aware Injection
"""
import logging
from dataclasses import dataclass, field
from typing import Any, Optional
from unittest.mock import MagicMock, patch

import pytest

from proxy.pipeline.motivator_stage import MotivatorStage, THRESHOLD_MESSAGES
from proxy.pipeline.threshold_engine import ThresholdEngine
from proxy.pipeline.agent_telemetry import AgentTelemetry


@dataclass
class FakeCtx:
    is_subagent: bool = True
    messages_original: Optional[list] = field(
        default_factory=lambda: [{"role": "user", "content": "Find the bug"}]
    )
    session_id: str = "sess-abc123"
    request_id: str = "req-xyz"
    body: dict = field(default_factory=lambda: {
        "messages": [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Find the bug"},
        ]
    })
    messages: list = field(default_factory=lambda: [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "Find the bug"},
    ])


def _make_agent(task=None, turn_count=0, max_request_chars=0, thresholds=(50, 75, 90)):
    agent = MagicMock()
    telemetry = AgentTelemetry()
    if task is not None:
        telemetry.original_task = task
    telemetry.turn_count = turn_count
    telemetry.max_request_chars = max_request_chars
    agent.telemetry = telemetry
    agent.threshold_engine = ThresholdEngine(thresholds)
    agent.config = None
    return agent


def _make_ctx_with_size(char_count):
    """Build a FakeCtx whose request body has approximately char_count chars."""
    sys_msg = "x" * (char_count // 2)
    user_msg = "x" * (char_count - len(sys_msg))
    return FakeCtx(
        body={"messages": [
            {"role": "system", "content": sys_msg},
            {"role": "user", "content": user_msg},
        ]},
        messages=[
            {"role": "system", "content": sys_msg},
            {"role": "user", "content": user_msg},
        ]
    )


class TestShouldRun:
    def test_enabled_by_default(self):
        stage = MotivatorStage()
        ctx = FakeCtx()
        assert stage.should_run(ctx) is True

    def test_disabled_when_not_subagent(self):
        stage = MotivatorStage()
        ctx = FakeCtx(is_subagent=False)
        with patch("proxy.pipeline.motivator_stage.MOTIVATOR_ENABLED", True):
            assert stage.should_run(ctx) is False

    def test_disabled_when_no_messages_original(self):
        stage = MotivatorStage()
        ctx = FakeCtx(messages_original=None)
        with patch("proxy.pipeline.motivator_stage.MOTIVATOR_ENABLED", True):
            assert stage.should_run(ctx) is False

    def test_enabled_for_subagent_with_messages(self):
        stage = MotivatorStage()
        ctx = FakeCtx()
        with patch("proxy.pipeline.motivator_stage.MOTIVATOR_ENABLED", True):
            assert stage.should_run(ctx) is True


class TestExecute:
    def _run(self, agent, ctx=None, sync_task=True):
        stage = MotivatorStage()
        if ctx is None:
            ctx = FakeCtx()
        if sync_task and agent.telemetry and agent.telemetry.original_task:
            ctx.messages_original = [{"role": "user", "content": agent.telemetry.original_task}]
        with patch("proxy.pipeline.motivator_stage.MOTIVATOR_ENABLED", True):
            with patch("proxy.pipeline.motivator_stage.CONTEXT_WINDOW_CHARS", 800_000):
                with patch(
                    "proxy.agent.agent_registry.registry"
                ) as mock_reg:
                    mock_reg.get_or_load.return_value = agent
                    return stage.execute(ctx)

    def test_noop_when_agent_is_none(self):
        ctx = FakeCtx()
        stage = MotivatorStage()
        with patch("proxy.pipeline.motivator_stage.MOTIVATOR_ENABLED", True):
            with patch("proxy.agent.agent_registry.registry") as mock_reg:
                mock_reg.get_or_load.return_value = None
                result = stage.execute(ctx)
        assert result is None

    def test_fires_50_pct_at_50pct_context(self):
        agent = _make_agent(task="Fix the login bug")
        ctx = _make_ctx_with_size(400_000)
        self._run(agent, ctx)
        # The reminder is now prepended to the LAST user message
        # (cache-stable position — does not shift middle messages).
        last_user = next(
            m for m in reversed(ctx.body["messages"])
            if m.get("role") == "user"
        )
        assert "[MOTIVATOR" in last_user["content"]
        assert "50%" in last_user["content"]
        assert "Fix the login bug" in last_user["content"]
        # messages[0] should be untouched (system prompt unchanged)
        assert ctx.body["messages"][0]["role"] == "system"
        # System message content was not modified — only the LAST user
        # message content was prepended with the reminder.
        assert "MOTIVATOR" not in ctx.body["messages"][0]["content"]




    def test_fires_only_highest_on_jump(self):
        agent = _make_agent(task="Refactor module")
        # jump straight from 0% to 75% in one request (no prior cumulative)
        ctx = _make_ctx_with_size(600_000)
        self._run(agent, ctx)
        last_user = next(
            m for m in reversed(ctx.body["messages"])
            if m.get("role") == "user"
        )
        assert "75%" in last_user["content"]
        # ThresholdEngine fired 50 AND 75 (both crossed in the jump) but
        # only the highest (75) is what the message content shows.
        assert 50 in agent.threshold_engine.fired
        assert 75 in agent.threshold_engine.fired

    def test_injects_into_last_user_message(self):
        agent = _make_agent(task="Deploy service")
        ctx = _make_ctx_with_size(400_000)
        self._run(agent, ctx)
        # messages[0] is the system prompt — MUST NOT contain the reminder
        assert ctx.body["messages"][0]["role"] == "system"
        assert "MOTIVATOR" not in ctx.body["messages"][0]["content"]
        # The reminder is prepended to the last user message
        last_user = next(
            m for m in reversed(ctx.body["messages"])
            if m.get("role") == "user"
        )
        assert "[MOTIVATOR" in last_user["content"]
        assert "Deploy service" in last_user["content"]



    def test_handles_multimodal_user_message(self):
        agent = _make_agent(task="Write tests")
        ctx = FakeCtx()
        # Last user message is multimodal (list of parts).
        # Make the text part large enough to trigger the 50% threshold.
        large_text = "x" * 400_000
        ctx.body["messages"][1]["content"] = [
            {"type": "text", "text": large_text},
            {"type": "image_url", "image_url": {"url": "..."}},
        ]
        ctx.messages = list(ctx.body["messages"])
        self._run(agent, ctx)
        # The system prompt is untouched (no MOTIVATOR mutation)
        assert "MOTIVATOR" not in ctx.body["messages"][0]["content"]
        # The reminder was prepended to the multimodal user message as a new text part
        last_user = next(
            m for m in reversed(ctx.body["messages"])
            if m.get("role") == "user"
        )
        assert isinstance(last_user["content"], list)
        # The new part (with the reminder) is at position 0
        first_part_text = last_user["content"][0].get("text", "")
        assert "[MOTIVATOR" in first_part_text
        assert "Write tests" in first_part_text
        # Original image_url part is preserved (now shifted to position 2
        # because the new text part was prepended at position 0).
        types = [p.get("type") for p in last_user["content"]]
        assert "image_url" in types, f"image_url part missing; got types={types}"
        # The original large_text content is preserved somewhere in the parts
        all_text = " ".join(
            str(p.get("text", "")) for p in last_user["content"]
            if isinstance(p, dict) and p.get("type") == "text"
        )
        assert large_text in all_text

    def test_handles_missing_user_message(self):
        agent = _make_agent(task="Something")
        ctx = FakeCtx()
        # No user message at all — reminder cannot be injected
        ctx.body["messages"] = [{"role": "system", "content": "Only system"}]
        ctx.messages = list(ctx.body["messages"])
        self._run(agent, ctx)
        all_content = " ".join(
            str(m.get("content", "")) for m in ctx.body["messages"]
        )
        assert "[MOTIVATOR" not in all_content



    def test_never_raises_on_internal_error(self):
        agent = _make_agent()
        agent.telemetry = None  # Will cause AttributeError
        ctx = FakeCtx()
        result = self._run(agent, ctx)
        assert result is None

    def test_logs_threshold_with_request_id(self, caplog):
        agent = _make_agent(task="Debug crash")
        ctx = _make_ctx_with_size(400_000)
        with caplog.at_level(logging.INFO, logger="proxy.pipeline.motivator_stage"):
            self._run(agent, ctx)
        assert any("MOTIVATOR" in r.message for r in caplog.records)
        assert any("sess-abc123" in r.message for r in caplog.records)

    def test_no_injection_below_threshold(self):
        agent = _make_agent(task="Explore codebase")
        ctx = _make_ctx_with_size(100_000)
        self._run(agent, ctx)
        all_system_content = " ".join(
            m["content"] for m in ctx.body["messages"]
            if m.get("role") == "system" and isinstance(m.get("content"), str)
        )
        assert "[MOTIVATOR" not in all_system_content

    def test_captures_task_from_messages_original(self):
        agent = _make_agent(turn_count=9)
        ctx = FakeCtx()
        ctx.messages_original = [
            {"role": "system", "content": "System prompt"},
            {"role": "user", "content": "Implement feature X"},
        ]
        self._run(agent, ctx)
        assert agent.telemetry.original_task == "Implement feature X"

    def test_task_change_resets_threshold_engine(self):
        agent = _make_agent(task="Old task", turn_count=10)
        agent.threshold_engine.fired.add(50)
        ctx = FakeCtx()
        ctx.messages_original = [{"role": "user", "content": "New task description"}]
        self._run(agent, ctx, sync_task=False)
        assert agent.telemetry.original_task == "New task description"
        assert agent.telemetry.turn_count == 1
        assert len(agent.threshold_engine.fired) == 0


class TestHighWatermarkNotCumulative:
    """C5 (P7): Multi-request context pct matches the LATEST request size,
    not the cumulative sum across requests.

    Pre-C5: quadratic overcount fired 50/75/90% thresholds 3.5× too early.
    Post-C5: high-watermark tracks the actual current context size.
    """

    def test_six_requests_pct_matches_latest_not_sum(self):
        """Simulate 6 requests with linearly-growing history through real
        MotivatorStage + real AgentTelemetry. After request 6, context pct
        equals pct of request 6's actual size, NOT the sum.

        Revert-test: old code (cumulative +=) reports pct of 21K (sum of
        1K+2K+...+6K), crossing 90% early for a 10K window.
        """
        agent = _make_agent(task="Long running task")
        stage = MotivatorStage()
        context_window = 10_000  # small window so thresholds are testable

        for i in range(1, 7):
            ctx = _make_ctx_with_size(i * 1000)
            with patch("proxy.pipeline.motivator_stage.MOTIVATOR_ENABLED", True), \
                 patch("proxy.pipeline.motivator_stage.CONTEXT_WINDOW_CHARS", context_window), \
                 patch("proxy.agent.agent_registry.registry") as mock_reg:
                mock_reg.get_or_load.return_value = agent
                stage.execute(ctx)

        # After 6 requests: max_request_chars should be 6000 (the latest)
        assert agent.telemetry.max_request_chars == 6000
        # pct should be 60% (6000/10000), NOT 100% (21000/10000 capped)
        pct = agent.telemetry.get_context_pct(context_window)
        assert pct == 60.0
