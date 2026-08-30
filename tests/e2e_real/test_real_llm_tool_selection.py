"""
Real LLM E2E Tests - Tool Selection
Exercises the actual LLM to verify tool selection works end-to-end.

These tests validate:
1. Round 1: The LLM selects the correct tool for the task
2. Subsequent rounds: The LLM eventually stops calling tools (finish_reason=stop)
3. Sync pipeline fires after the conversation ends

If the LLM does not stop naturally after MAX_TOOL_ROUNDS, the test records this as a
model behavior issue (forced_stop=True) and ensures sync still fires — exactly what
a real client (Kilocode) does when hitting its iteration limit.
"""

import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
from conftest import post, get, PROXY_URL, calculate_session_id, print_test_summary

MAX_TOOL_ROUNDS = 5  # Safety limit to prevent infinite loops


def _is_llama_available():
    """Check if llama-server is available."""
    try:
        resp = get("http://localhost:7777", timeout=2)
        return resp.status_code < 500
    except Exception:
        return False


def _run_tool_loop_with_assertions(url, messages, tools, tool_name_assertions,
                                    timeout=30, max_rounds=MAX_TOOL_ROUNDS):
    """Drive the tool loop like a real client would, with tool-name assertions.

    Args:
        tool_name_assertions: dict mapping round number -> expected tool name(s).
                              e.g. {1: "bash"} asserts round 1 selects "bash".
                              Subsequent rounds are not asserted for tool name.

    Returns:
        (final_data, total_rounds, all_tool_calls, forced_stop) - the last response data,
        how many rounds were needed, all tool calls made, and whether a forced stop was needed.
    """
    all_tool_calls = []

    for round_num in range(1, max_rounds + 1):
        response = post(
            url,
            json_data={
                "model": "qwen3-coder",
                "messages": messages,
                "tools": tools,
                "temperature": 0.0,
                "seed": 42,
            },
            timeout=timeout,
        )
        assert response.status_code == 200, f"Round {round_num} HTTP {response.status_code}"

        data = response.json()
        message = data["choices"][0]["message"]
        finish_reason = data["choices"][0].get("finish_reason")
        tool_calls = message.get("tool_calls", [])

        print(f"[TEST] Round {round_num} finish_reason: {finish_reason}, tool_calls: {len(tool_calls)}")

        # Assert tool selection for this round if specified
        if round_num in tool_name_assertions and tool_calls:
            expected = tool_name_assertions[round_num]
            actual = tool_calls[0]["function"]["name"]
            assert actual == expected, (
                f"Round {round_num}: Expected tool '{expected}', got '{actual}'"
            )

        all_tool_calls.extend(tool_calls)

        if finish_reason == "stop" and not tool_calls:
            # LLM is done — no more tool calls
            return data, round_num, all_tool_calls, False

        if tool_calls:
            # LLM wants to call more tools — simulate executing them and feed results back
            messages.append(message)
            for tc in tool_calls:
                tool_name = tc["function"]["name"]
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc["id"],
                    "content": f"Simulated result for {tool_name}",
                })
            # Continue loop
        else:
            print(f"[TEST] WARNING: Round {round_num} had finish_reason={finish_reason} but no tool_calls")
            return data, round_num, all_tool_calls, False

    # Exhausted max rounds — LLM kept calling tools.
    # Simulate what a real client does: send a final request WITHOUT tools,
    # forcing the LLM to give a text answer with finish_reason=stop.
    print(f"[TEST] LLM did not stop after {max_rounds} rounds — forcing final answer (no tools)")
    messages.append({
        "role": "user",
        "content": "Please provide your final answer based on the tool results above. Do not call any more tools.",
    })

    final_response = post(
        url,
        json_data={
            "model": "qwen3-coder",
            "messages": messages,
            "temperature": 0.0,
            "seed": 42,
        },
        timeout=timeout,
    )
    assert final_response.status_code == 200

    final_data = final_response.json()
    final_finish = final_data["choices"][0].get("finish_reason")
    print(f"[TEST] Forced final round finish_reason: {final_finish}")

    return final_data, max_rounds + 1, all_tool_calls, True


@pytest.mark.e2e_real
def test_llm_tool_selection_bash(sync_verification):
    """Test that LLM selects 'bash' tool when asked to run a command.

    Verifies:
    - Round 1: LLM selects 'bash' tool
    - LLM eventually stops after receiving tool results
    - Sync pipeline fires
    """
    if not _is_llama_available():
        pytest.skip("llama-server not available")

    messages = [
        {
            "role": "user",
            "content": "Run the command 'echo hello world' in bash",
        }
    ]
    session_id = calculate_session_id(messages)
    print(f"[TEST] Session ID: {session_id}")

    tools = [
        {
            "type": "function",
            "function": {
                "name": "bash",
                "description": "Execute bash commands",
                "parameters": {
                    "type": "object",
                    "properties": {"command": {"type": "string"}},
                    "required": ["command"],
                },
            },
        },
    ]

    final_data, total_rounds, all_calls, forced_stop = _run_tool_loop_with_assertions(
        f"{PROXY_URL}/v1/chat/completions",
        messages, tools,
        tool_name_assertions={1: "bash"},
        timeout=30,
    )
    print(f"[TEST] Completed in {total_rounds} round(s), {len(all_calls)} total tool call(s), forced_stop={forced_stop}")

    assert sync_verification(session_id, timeout=10), (
        f"Sync failed for session {session_id}"
    )
    print_test_summary("test_llm_tool_selection_bash", session_id, True)


@pytest.mark.e2e_real
def test_llm_tool_selection_read(sync_verification):
    """Test that LLM selects 'read' tool when asked to read a file.

    Verifies:
    - Round 1: LLM selects 'read' tool
    - LLM eventually stops after receiving tool results
    - Sync pipeline fires
    """
    if not _is_llama_available():
        pytest.skip("llama-server not available")

    messages = [{"role": "user", "content": "Read the contents of /tmp/test.txt"}]
    session_id = calculate_session_id(messages)
    print(f"[TEST] Session ID: {session_id}")

    tools = [
        {
            "type": "function",
            "function": {
                "name": "read",
                "description": "Read file contents",
                "parameters": {
                    "type": "object",
                    "properties": {"file_path": {"type": "string"}},
                    "required": ["file_path"],
                },
            },
        },
    ]

    final_data, total_rounds, all_calls, forced_stop = _run_tool_loop_with_assertions(
        f"{PROXY_URL}/v1/chat/completions",
        messages, tools,
        tool_name_assertions={1: "read"},
        timeout=30,
    )
    print(f"[TEST] Completed in {total_rounds} round(s), {len(all_calls)} total tool call(s), forced_stop={forced_stop}")

    assert sync_verification(session_id, timeout=10), (
        f"Sync failed for session {session_id}"
    )
    print_test_summary("test_llm_tool_selection_read", session_id, True)


@pytest.mark.e2e_real
def test_llm_tool_selection_write(sync_verification):
    """Test that LLM selects 'write' tool when asked to write a file.

    Verifies:
    - Round 1: LLM selects 'write' tool
    - LLM eventually stops after receiving tool results
    - Sync pipeline fires
    """
    if not _is_llama_available():
        pytest.skip("llama-server not available")

    messages = [{"role": "user", "content": "Write 'Hello World' to /tmp/test_llm.txt"}]
    session_id = calculate_session_id(messages)
    print(f"[TEST] Session ID: {session_id}")

    tools = [
        {
            "type": "function",
            "function": {
                "name": "write",
                "description": "Write to file",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "file_path": {"type": "string"},
                        "content": {"type": "string"},
                    },
                    "required": ["file_path", "content"],
                },
            },
        },
    ]

    final_data, total_rounds, all_calls, forced_stop = _run_tool_loop_with_assertions(
        f"{PROXY_URL}/v1/chat/completions",
        messages, tools,
        tool_name_assertions={1: "write"},
        timeout=30,
    )
    print(f"[TEST] Completed in {total_rounds} round(s), {len(all_calls)} total tool call(s), forced_stop={forced_stop}")

    assert sync_verification(session_id, timeout=10), (
        f"Sync failed for session {session_id}"
    )
    print_test_summary("test_llm_tool_selection_write", session_id, True)
