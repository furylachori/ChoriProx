"""Tests for proxy.pipeline.agent_telemetry.AgentTelemetry."""
import threading

import pytest

from proxy.pipeline.agent_telemetry import AgentTelemetry


class TestCaptureTask:
    def test_extracts_first_user_message(self):
        t = AgentTelemetry()
        msgs = [
            {"role": "system", "content": "S"},
            {"role": "user", "content": "Build a REST API"},
        ]
        t.capture_task(msgs)
        assert t.original_task == "Build a REST API"

    def test_caps_at_500_chars(self):
        t = AgentTelemetry()
        msgs = [{"role": "user", "content": "x" * 1000}]
        t.capture_task(msgs)
        assert len(t.original_task) == 500

    def test_multimodal_content(self):
        t = AgentTelemetry()
        msgs = [{"role": "user", "content": [
            {"type": "text", "text": "Hello"},
            {"type": "image_url", "image_url": {"url": "http://img"}},
            {"type": "text", "text": "World"},
        ]}]
        t.capture_task(msgs)
        assert t.original_task == "Hello World"

    def test_write_once_same_task(self):
        t = AgentTelemetry()
        t.capture_task([{"role": "user", "content": "First task that is long enough"}])
        t.turn_count = 5
        t.capture_task([{"role": "user", "content": "First task that is long enough"}])
        assert t.original_task == "First task that is long enough"
        assert t.turn_count == 5

    def test_none_messages(self):
        t = AgentTelemetry()
        t.capture_task(None)
        assert t.original_task is None

    def test_empty_list(self):
        t = AgentTelemetry()
        t.capture_task([])
        assert t.original_task is None


class TestIncrementTurn:
    def test_thread_safety(self):
        t = AgentTelemetry()
        threads = []
        for _ in range(10):
            def worker():
                for _ in range(100):
                    t.increment_turn()
            th = threading.Thread(target=worker)
            threads.append(th)
            th.start()
        for th in threads:
            th.join()
        assert t.turn_count == 1000


class TestCaptureTaskEdgeCases:
    def test_capture_task_skips_none_content(self):
        t = AgentTelemetry()
        msgs = [{"role": "user", "content": None}]
        t.capture_task(msgs)
        assert t.original_task is None

    def test_capture_task_skips_empty_content(self):
        t = AgentTelemetry()
        msgs = [{"role": "user", "content": ""}]
        t.capture_task(msgs)
        assert t.original_task is None

    def test_capture_task_resets_on_change(self):
        t = AgentTelemetry()
        t.capture_task([{"role": "user", "content": "Task A long enough"}])
        assert t.original_task == "Task A long enough"
        t.turn_count = 10
        t.custom_state["key"] = "val"
        t.capture_task([{"role": "user", "content": "Task B long enough"}])
        assert t.original_task == "Task B long enough"
        assert t.turn_count == 0
        assert t.custom_state == {}


class TestRecordRequestChars:
    """C5 (P7): high-watermark semantics — latest request IS the cumulative context.

    Pre-C5: `cumulative_chars += n` summed across requests (quadratic).
    Post-C5: `max_request_chars = max(max_request_chars, n)` tracks the
    largest context the agent has operated in.
    """

    def test_high_watermark_not_cumulative(self):
        """After 6 requests of 1K-6K chars, max_request_chars == 6000
        (the latest), NOT 21000 (the cumulative sum)."""
        t = AgentTelemetry()
        for i in range(1, 7):
            t.record_request_chars(i * 1000)
        assert t.max_request_chars == 6000  # NOT 21000

    def test_decreasing_request_size_keeps_high_watermark(self):
        """If context shrinks (trimmed by select_context), the high-watermark
        keeps the peak — the motivator warns based on the LARGEST context seen."""
        t = AgentTelemetry()
        t.record_request_chars(5000)
        t.record_request_chars(3000)  # smaller (trimmed)
        assert t.max_request_chars == 5000  # NOT 3000

    def test_get_context_pct_uses_high_watermark(self):
        """get_context_pct returns pct of max_request_chars, not cumulative."""
        t = AgentTelemetry()
        for i in range(1, 7):
            t.record_request_chars(i * 1000)
        # 6K / 10K = 60%
        pct = t.get_context_pct(10_000)
        assert pct == 60.0

    def test_zero_request_chars(self):
        """Fresh agent: max_request_chars == 0 → pct == 0."""
        t = AgentTelemetry()
        assert t.get_context_pct(10_000) == 0.0

    def test_reset_clears_max_request_chars(self):
        """reset() clears max_request_chars (along with other state)."""
        t = AgentTelemetry()
        t.record_request_chars(5000)
        t.reset()
        assert t.max_request_chars == 0
        assert t.get_context_pct(10_000) == 0.0

    def test_neg_or_zero_request_chars_no_op(self):
        """Negative or zero request sizes are ignored (no max update)."""
        t = AgentTelemetry()
        t.record_request_chars(5000)
        t.record_request_chars(0)
        t.record_request_chars(-100)
        assert t.max_request_chars == 5000  # unchanged
