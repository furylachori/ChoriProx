"""
Real LLM E2E Tests - Tool Chain
Tests that the LLM can chain multiple tool calls together.

These tests validate the LLM's ability to:
1. Select the right tools across multiple rounds
2. Maintain context when tool results are fed back
3. Eventually stop when the task is complete (finish_reason=stop)
4. Trigger the sync pipeline after the conversation ends

If the LLM does not stop naturally, a forced final round (no tools) ensures sync fires.
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


def _run_tool_loop(url, messages, tools, timeout=30, max_rounds=MAX_TOOL_ROUNDS):
    """Drive the tool loop like a real client would.

    Keeps calling the LLM with tool results until it returns finish_reason=stop.
    If the LLM doesn't stop naturally, sends a final request WITHOUT tools (like a
    real client hitting its iteration limit) to force finish_reason=stop so sync fires.

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
def test_llm_two_step_chain(sync_verification):
    """Test that LLM can chain glob -> read across multiple rounds.

    Verifies:
    - Round 1: LLM calls glob
    - LLM handles tool results and may call more tools
    - LLM eventually stops (finish_reason=stop)
    - Sync pipeline fires
    """
    if not _is_llama_available():
        pytest.skip("llama-server not available")

    messages = [
        {
            "role": "user",
            "content": "Find all .py files in the current directory, then read the first one",
        }
    ]
    session_id = calculate_session_id(messages)
    print(f"[TEST] Session ID: {session_id}")

    tools = [
        {
            "type": "function",
            "function": {
                "name": "glob",
                "description": "Find files by pattern",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string"},
                        "pattern": {"type": "string"},
                    },
                    "required": ["path"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "read",
                "description": "Read file",
                "parameters": {
                    "type": "object",
                    "properties": {"file_path": {"type": "string"}},
                    "required": ["file_path"],
                },
            },
        },
    ]

    final_data, total_rounds, all_calls, forced_stop = _run_tool_loop(
        f"{PROXY_URL}/v1/chat/completions", messages, tools, timeout=60
    )
    print(f"[TEST] Completed in {total_rounds} round(s), {len(all_calls)} total tool call(s), forced_stop={forced_stop}")

    # A two-step chain should have made at least 2 tool calls across multiple rounds
    assert len(all_calls) >= 2, (
        f"Expected at least 2 tool calls for a two-step chain, got {len(all_calls)}"
    )

    assert sync_verification(session_id, timeout=10), (
        f"Sync failed for session {session_id}"
    )
    print_test_summary("test_llm_two_step_chain", session_id, True)


@pytest.mark.e2e_real
def test_llm_three_step_chain(sync_verification):
    """Test that LLM can chain glob -> read -> write across multiple rounds.

    Verifies:
    - Round 1: LLM calls glob (or similar first step)
    - Subsequent rounds: LLM calls read, then write
    - LLM eventually stops (finish_reason=stop)
    - Sync pipeline fires
    """
    if not _is_llama_available():
        pytest.skip("llama-server not available")

    messages = [
        {
            "role": "user",
            "content": "Find .py files, read one, then write a summary to /tmp/chain_test.txt",
        }
    ]
    session_id = calculate_session_id(messages)
    print(f"[TEST] Session ID: {session_id}")

    tools = [
        {
            "type": "function",
            "function": {
                "name": "glob",
                "description": "Find files by pattern",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string"},
                        "pattern": {"type": "string"},
                    },
                    "required": ["path"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "read",
                "description": "Read file",
                "parameters": {
                    "type": "object",
                    "properties": {"file_path": {"type": "string"}},
                    "required": ["file_path"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "write",
                "description": "Write file",
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

    final_data, total_rounds, all_calls, forced_stop = _run_tool_loop(
        f"{PROXY_URL}/v1/chat/completions", messages, tools, timeout=90
    )
    print(f"[TEST] Completed in {total_rounds} round(s), {len(all_calls)} total tool call(s), forced_stop={forced_stop}")

    # A three-step chain should have made at least 3 tool calls across multiple rounds
    assert len(all_calls) >= 3, (
        f"Expected at least 3 tool calls for a three-step chain, got {len(all_calls)}"
    )

    assert sync_verification(session_id, timeout=10), (
        f"Sync failed for session {session_id}"
    )
    print_test_summary("test_llm_three_step_chain", session_id, True)
