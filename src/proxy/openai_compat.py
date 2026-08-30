"""OpenAI-compatible response builders for synthetic proxy responses."""

import time
import uuid
from typing import Any, Dict, Optional


def chat_completion_shape_error(data: Any) -> Optional[str]:
    """Return a stable reason when data is not a valid chat completion."""
    if not isinstance(data, dict):
        return "response is not an object"

    # Deliberately dict-valued errors only: a provider "error" envelope must be an
    # object to be preserved verbatim. Top-level string/null/list "error" bodies are
    # NOT trusted (no configured provider emits them) and fall through to the
    # malformed-shape checks below -> canonical upstream_protocol_error. See F6,
    # docs/_analysis/four-day-change-audit-2026-07-17.md.
    if isinstance(data.get("error"), dict) and not data.get("choices"):
        return None

    choices = data.get("choices")
    if not isinstance(choices, list):
        return "choices is not an array"
    if not choices:
        return "choices is empty"

    first = choices[0]
    if not isinstance(first, dict):
        return "choices[0] is not an object"

    message = first.get("message")
    if not isinstance(message, dict):
        return "choices[0].message is not an object"

    return None


def build_upstream_protocol_error() -> Dict:
    return {
        "error": {
            "message": "Upstream returned an invalid chat completion response",
            "type": "upstream_protocol_error",
            "code": "invalid_chat_completion",
        }
    }


def build_openai_response(
    content: str,
    model: str,
    *,
    role: str = "assistant",
    usage: Optional[Dict] = None,
    finish_reason: str = "stop",
    response_id: Optional[str] = None,
    created: Optional[int] = None,
) -> Dict:
    """Build an OpenAI-shape chat completion response.

    Returns a dict conforming to https://platform.openai.com/docs/api-reference/chat/object.
    Optional fields default to sensible values.
    """
    return {
        "id": response_id or f"chatcmpl-{uuid.uuid4().hex[:24]}",
        "object": "chat.completion",
        "created": created or int(time.time()),
        "model": model,
        "choices": [{
            "index": 0,
            "message": {"role": role, "content": content},
            "finish_reason": finish_reason,
        }],
        "usage": usage or {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    }
