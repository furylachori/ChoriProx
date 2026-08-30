import json
from qwen_proxy import stream_response, make_sse_chunk


def parse_sse_chunks(sse_text):
    """Parse SSE text into list of JSON dicts."""
    chunks = []
    for line in sse_text.strip().split("\n\n"):
        if line.startswith("data: ") and line != "data: [DONE]":
            chunks.append(json.loads(line[6:]))
    return chunks


def test_reasoning_content_emitted_before_content():
    message = {
        "role": "assistant",
        "reasoning_content": "Let me think...",
        "content": "The answer is 42.",
    }
    original_data = {"id": "test", "created": 0, "model": "kimi"}
    sse = "".join(stream_response(message, original_data, "kimi"))
    chunks = parse_sse_chunks(sse)

    reasoning_idxs = [
        i
        for i, c in enumerate(chunks)
        if "reasoning_content" in c["choices"][0].get("delta", {})
    ]
    content_idxs = [
        i
        for i, c in enumerate(chunks)
        if "content" in c["choices"][0].get("delta", {})
        and c["choices"][0]["delta"]["content"]
    ]

    assert len(reasoning_idxs) == 1
    assert len(content_idxs) == 1
    assert reasoning_idxs[0] < content_idxs[0]


def test_no_reasoning_chunk_when_absent():
    message = {"role": "assistant", "content": "Hello."}
    original_data = {"id": "test", "created": 0, "model": "kimi"}
    sse = "".join(stream_response(message, original_data, "kimi"))
    chunks = parse_sse_chunks(sse)

    reasoning_chunks = [
        c for c in chunks if "reasoning_content" in c["choices"][0].get("delta", {})
    ]
    assert len(reasoning_chunks) == 0


def test_model_alias_in_all_chunks():
    message = {
        "role": "assistant",
        "reasoning_content": "thinking",
        "content": "result",
    }
    original_data = {"id": "test", "created": 0, "model": "opencode-go/kimi-k2.6"}
    sse = "".join(stream_response(message, original_data, "kimi"))
    chunks = parse_sse_chunks(sse)

    for chunk in chunks:
        assert chunk["model"] == "kimi"


def test_done_terminator_present():
    message = {"role": "assistant", "content": "ok"}
    original_data = {"id": "test", "created": 0, "model": "kimi"}
    sse = "".join(stream_response(message, original_data, "kimi"))
    assert "data: [DONE]" in sse


def test_reasoning_content_with_tool_calls():
    message = {
        "role": "assistant",
        "reasoning_content": "I need to use a tool",
        "tool_calls": [
            {
                "id": "call_123",
                "type": "function",
                "function": {"name": "bash", "arguments": '{"command": "ls"}'},
            }
        ],
    }
    original_data = {"id": "test", "created": 0, "model": "kimi"}
    sse = "".join(stream_response(message, original_data, "kimi"))
    chunks = parse_sse_chunks(sse)

    reasoning_idxs = [
        i
        for i, c in enumerate(chunks)
        if "reasoning_content" in c["choices"][0].get("delta", {})
    ]
    tool_idxs = [
        i
        for i, c in enumerate(chunks)
        if "tool_calls" in c["choices"][0].get("delta", {})
    ]

    assert len(reasoning_idxs) == 1
    assert len(tool_idxs) == 1
    assert reasoning_idxs[0] < tool_idxs[0]


def test_reasoning_content_uses_reasoning_field():
    message = {
        "role": "assistant",
        "reasoning": "Thinking via reasoning field",
        "content": "Result",
    }
    original_data = {"id": "test", "created": 0, "model": "kimi"}
    sse = "".join(stream_response(message, original_data, "kimi"))
    chunks = parse_sse_chunks(sse)

    reasoning_chunks = [
        c
        for c in chunks
        if c["choices"][0]["delta"].get("reasoning_content")
        == "Thinking via reasoning field"
    ]
    assert len(reasoning_chunks) == 1


def test_original_model_default():
    message = {"role": "assistant", "content": "Hello."}
    original_data = {"id": "test", "created": 0, "model": "default-model"}
    sse = "".join(stream_response(message, original_data))
    chunks = parse_sse_chunks(sse)

    for chunk in chunks:
        assert chunk["model"] == "default-model"
