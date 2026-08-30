"""Static methods for augmenting message lists.

Reusable for any feature that needs to inject text into the LLM prompt:
motivator reminders, stealth behavior injection, dynamic guardrails,
contextual hints, A/B testing prompt variations.
"""


class PromptInjector:
    """Stateless message list augmentation utilities."""

    @staticmethod
    def append_to_system(messages: list, text: str) -> bool:
        """Append text to the first system message's content.

        Handles both string content and list (multimodal) content.
        Returns True if injected, False if no valid system message found.
        """
        if not isinstance(messages, list):
            return False
        if not text:
            return False
        if not messages or not isinstance(messages[0], dict):
            return False
        if messages[0].get("role") != "system":
            return False
        content = messages[0].get("content", "")
        if isinstance(content, str):
            messages[0]["content"] = content + "\n\n" + text
            return True
        elif isinstance(content, list):
            messages[0]["content"].append({"type": "text", "text": "\n\n" + text})
            return True
        return False

    @staticmethod
    def insert_system(messages: list, text: str, position: int = 1) -> bool:
        """Insert a new system message at the given position.

        Default position=1 places it right after the first system message.
        Returns True if inserted.
        """
        if not isinstance(messages, list):
            return False
        if not text:
            return False
        if position < 0 or position > len(messages):
            return False
        messages.insert(position, {"role": "system", "content": text})
        return True

    @staticmethod
    def append_user_message(messages: list, text: str) -> bool:
        """Append a synthetic user message to the end of the message list.

        Returns True if appended.
        """
        if not isinstance(messages, list):
            return False
        if not text:
            return False
        messages.append({"role": "user", "content": text})
        return True

    @staticmethod
    def prepend_to_system(messages: list, text: str) -> bool:
        """Prepend text to the first system message's content.

        Returns True if prepended, False if no valid system message found.
        """
        if not isinstance(messages, list):
            return False
        if not text:
            return False
        if not messages or not isinstance(messages[0], dict):
            return False
        if messages[0].get("role") != "system":
            return False
        content = messages[0].get("content", "")
        if isinstance(content, str):
            messages[0]["content"] = text + "\n\n" + content
            return True
        elif isinstance(content, list):
            messages[0]["content"].insert(0, {"type": "text", "text": text + "\n\n"})
            return True
        return False

    @staticmethod
    def prepend_to_last_user_message(messages: list, text: str) -> bool:
        """Prepend text to the LAST user-role message in the array.

        This is the cache-stable injection point for per-turn nudges (e.g.,
        MotivatorStage reminders). Mutating the last user message does NOT
        shift any earlier message positions, so the cached prefix
        (messages[0..N-2]) remains byte-identical across turns. Only the
        current turn's user input changes — and that input is the uncached
        tail of the prompt by definition.

        Once added, the reminder becomes part of the prefix for the NEXT
        turn's cache lookup. The cache builds incrementally — the reminder
        doesn't shift anything in the middle of the array.

        Returns True if prepended, False if no user message found.
        """
        if not isinstance(messages, list):
            return False
        if not text:
            return False
        # Find the last user message (the current turn's input)
        last_user_idx = None
        for i in range(len(messages) - 1, -1, -1):
            if isinstance(messages[i], dict) and messages[i].get("role") == "user":
                last_user_idx = i
                break
        if last_user_idx is None:
            return False
        content = messages[last_user_idx].get("content", "")
        separator = "\n\n---\n\n"
        if isinstance(content, str):
            # If the original content is empty/missing, skip injection —
            # prepending would silently lose the user's empty message and
            # send only the reminder. Better to drop the reminder.
            if not content.strip():
                return False
            messages[last_user_idx]["content"] = f"{text}{separator}{content}"
            return True
        elif isinstance(content, list):
            # Multimodal content — prepend a text part at position 0.
            # The reminder lives in a NEW text part; existing parts
            # (including image_url) are preserved in their original order
            # starting from position 1.
            # Mirror the string-branch guard: if there is no meaningful
            # content (all text parts empty/whitespace AND no non-text parts
            # carrying real content like images), skip injection —
            # prepending would send only the reminder and silently drop
            # the user's empty message.
            has_text = any(
                isinstance(p, dict)
                and p.get("type") == "text"
                and p.get("text", "").strip()
                for p in content
            )
            has_other = any(
                isinstance(p, dict)
                and p.get("type") not in (None, "text")
                for p in content
            )
            if not has_text and not has_other:
                return False
            messages[last_user_idx]["content"] = [
                {"type": "text", "text": f"{text}{separator}"}
            ] + content
            return True
        return False
