"""Per-agent state tracking for pipeline stages.

Lives on the Agent dataclass. Thread-safe via _lock.
Reusable for: motivator, budget tracking, quality gates, analytics.
"""
import threading
from typing import Dict, Optional

from proxy.lock_ordering import make_lock


class AgentTelemetry:
    """Thread-safe per-agent telemetry.

    Attached to Agent instances. Tracks turn count, original task,
    and arbitrary custom state for pipeline stages.

    Lock ordering position: #8 (per-agent; motivator threshold tracking).
    """

    def __init__(self):
        self._lock = make_lock(8, "AgentTelemetry._lock", threading.Lock)
        self.turn_count: int = 0
        self.original_task: Optional[str] = None
        self.max_request_chars: int = 0
        self.custom_state: Dict = {}

    def increment_turn(self) -> int:
        """Thread-safe turn counter increment."""
        with self._lock:
            self.turn_count += 1
            return self.turn_count

    def capture_task(self, messages_original: Optional[list], max_len: int = 500):
        """Extract first user message from pre-optimization messages.

        Updates if task changes significantly. Resets counters on change.
        Handles multimodal content (list of dicts).
        """
        if not messages_original:
            return
        for msg in messages_original:
            if not isinstance(msg, dict):
                continue
            if msg.get("role") == "user":
                content = msg.get("content", "")
                if isinstance(content, list):
                    content = " ".join(
                        item.get("text", "")
                        for item in content
                        if isinstance(item, dict) and item.get("type") == "text"
                    )
                if not content:
                    continue
                if len(content.strip()) < 10:
                    continue  # Skip greetings/preamble (matches _extract_task_from_messages)
                new_task = content[:max_len]
                with self._lock:
                    if self.original_task is None:
                        self.original_task = new_task
                    elif new_task != self.original_task:
                        self.turn_count = 0
                        self.max_request_chars = 0
                        self.custom_state.clear()
                        self.original_task = new_task
                return

    def record_request_chars(self, n: int) -> None:
        """Record the char count of the current request body. High-watermark semantics.

        The latest request body contains the full conversation
        history (every prior message is included). So the latest
        request IS the cumulative context — we track the LARGEST
        context the agent has operated in via max() rather than
        summing (which would double-count across requests).

        Rough heuristic: 1 token ≈ 4 chars (typical English text).
        """
        if n <= 0:
            return
        with self._lock:
            self.max_request_chars = max(self.max_request_chars, n)

    def get_context_pct(self, context_window_chars: int = 4_000_000) -> float:
        """Context usage as percentage of model context window. Capped at 100.

        Default window is 1M tokens ≈ 4M chars (matches motivator_stage.py and
        AgentConfig.motivator_context_window_chars). Override per-model if needed.
        """
        if context_window_chars <= 0:
            return 100.0
        with self._lock:
            chars = self.max_request_chars
        return min(100.0, (chars / context_window_chars) * 100)

    def reset(self) -> None:
        """Reset all telemetry state."""
        with self._lock:
            self.turn_count = 0
            self.original_task = None
            self.max_request_chars = 0
            self.custom_state.clear()
