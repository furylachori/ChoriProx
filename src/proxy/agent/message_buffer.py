"""Thread-safe message buffer with budget-aware trimming."""
import logging
import threading
from typing import Optional

logger = logging.getLogger("proxy.agent.message_buffer")


class MessageBuffer:
    """Thread-safe message buffer with budget-aware trimming.

    Replaces raw list operations in tool_loop.py with a thread-safe abstraction.
    All operations acquire self._lock before accessing _messages.
    """

    def __init__(self, messages: Optional[list[dict]] = None):
        self._messages: list[dict] = messages or []
        self._lock = threading.Lock()

    def append(self, message: dict) -> None:
        """Append a single message to the buffer."""
        with self._lock:
            self._messages.append(message)

    def extend(self, messages: list[dict]) -> None:
        """Extend the buffer with multiple messages."""
        with self._lock:
            self._messages.extend(messages)

    def to_list(self) -> list[dict]:
        """Return a defensive copy of all messages."""
        with self._lock:
            return list(self._messages)

    def __len__(self) -> int:
        """Return the number of messages in the buffer."""
        with self._lock:
            return len(self._messages)

    def _cleanup_orphaned_tool_results(self) -> None:
        """Remove tool results that have no matching tool_call in the message list.

        Called internally after trimming. Thread safety: assumes caller holds self._lock.

        Algorithm (from memory/tool_loop.py:_cleanup_orphaned_tool_results):
        1. Collect all tool_call_ids from assistant messages that have tool_calls
        2. Remove any message with role="tool" whose "tool_call_id" is NOT in seen_ids
        3. Removal done in reverse order to avoid index shifting
        """
        # Collect all valid tool_call_ids from assistant messages
        valid_tool_call_ids = set()
        for msg in self._messages:
            if msg.get("role") == "assistant" and isinstance(msg.get("tool_calls"), list):
                for tc in msg["tool_calls"]:
                    if tc.get("id"):
                        valid_tool_call_ids.add(tc.get("id"))

        # Remove orphaned tool results (in reverse to avoid index shifting).
        # M4 #2: sub-agent return messages carry a `_sub_agent_return` sentinel
        # and are never orphaned — they belong to the parent agent's
        # record of the child's output, not to any assistant tool_call.
        removed = 0
        for i in range(len(self._messages) - 1, -1, -1):
            msg = self._messages[i]
            if msg.get("role") == "tool":
                if msg.get("_sub_agent_return"):
                    continue
                tool_call_id = msg.get("tool_call_id")
                if tool_call_id and tool_call_id not in valid_tool_call_ids:
                    del self._messages[i]
                    removed += 1

        if removed:
            logger.debug(f"MessageBuffer: Removed {removed} orphaned tool result(s)")

    def cleanup_orphaned_tool_results(self) -> None:
        """Public API for removing orphaned tool results."""
        with self._lock:
            self._cleanup_orphaned_tool_results()
