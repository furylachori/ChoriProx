"""Server-Sent Events formatting and streaming."""
import json
import logging
import time

logger = logging.getLogger(__name__)


def make_sse_chunk(data):
    """Format a dict as an SSE data line."""
    return f"data: {json.dumps(data)}\n\n"


def stream_response(message, original_data, original_model=None):
    """
    Generator that yields SSE chunks mimicking the OpenAI streaming format.
    If the message contains tool_calls, we emit a single chunk with the tool call.
    Otherwise we stream the content token-by-token (or as a single chunk).
    """
    base = {
        "id": original_data.get("id", f"chatcmpl-{int(time.time())}"),
        "object": "chat.completion.chunk",
        "created": original_data.get("created", int(time.time())),
        "model": original_model or original_data.get("model", "default"),
    }

    # Read the real finish_reason from the in-memory response dict (the same
    # object post_process / _format() mutated). Skill deactivation and SOTA
    # sync read finish_reason from THIS dict in apply_post_process (pre-stream),
    # NOT from the terminal chunk we emit below — keep the two consistent.
    from proxy.tool_pipeline.tool_pipeline import TRUNCATION_REASONS
    _response_fr = (original_data.get("choices") or [{}])[0].get("finish_reason")

    tool_calls = message.get("tool_calls")
    reasoning_content = (
        message.get("reasoning_content") or message.get("reasoning") or ""
    )

    if tool_calls:
        chunk = {
            **base,
            "choices": [
                {
                    "index": 0,
                    "delta": {"role": "assistant", "content": ""},
                    "finish_reason": None,
                }
            ],
        }
        yield make_sse_chunk(chunk)

        if reasoning_content:
            chunk = {
                **base,
                "choices": [
                    {
                        "index": 0,
                        "delta": {"reasoning_content": reasoning_content},
                        "finish_reason": None,
                    }
                ],
            }
            yield make_sse_chunk(chunk)

        for idx, tc in enumerate(tool_calls):
            chunk = {
                **base,
                "choices": [
                    {
                        "index": 0,
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": idx,
                                    "id": tc["id"],
                                    "type": "function",
                                    "function": {
                                        "name": tc["function"]["name"],
                                        "arguments": tc["function"]["arguments"],
                                    },
                                }
                            ]
                        },
                        "finish_reason": None,
                    }
                ],
            }
            yield make_sse_chunk(chunk)

        chunk = {
            **base,
            "choices": [{"index": 0, "delta": {}, "finish_reason": _response_fr if _response_fr in TRUNCATION_REASONS else "tool_calls"}],
        }
        yield make_sse_chunk(chunk)
    else:
        content = message.get("content", "") or ""

        chunk = {
            **base,
            "choices": [
                {
                    "index": 0,
                    "delta": {"role": "assistant", "content": ""},
                    "finish_reason": None,
                }
            ],
        }
        yield make_sse_chunk(chunk)

        if reasoning_content:
            chunk = {
                **base,
                "choices": [
                    {
                        "index": 0,
                        "delta": {"reasoning_content": reasoning_content},
                        "finish_reason": None,
                    }
                ],
            }
            yield make_sse_chunk(chunk)

        if content:
            chunk = {
                **base,
                "choices": [
                    {"index": 0, "delta": {"content": content}, "finish_reason": None}
                ],
            }
            yield make_sse_chunk(chunk)

        chunk = {
            **base,
            "choices": [{"index": 0, "delta": {}, "finish_reason": _response_fr if _response_fr in TRUNCATION_REASONS else "stop"}],
        }
        yield make_sse_chunk(chunk)

    yield "data: [DONE]\n\n"