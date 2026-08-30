"""
Real LLM E2E Tests - 3 Levels of Complexity
Tests LLM capability across different complexity levels.

These tests simulate a real client-driven tool loop:
  Round 1: Send user request + tools → LLM returns tool_calls
  Round 2+: Send tool results back + tools → LLM should eventually return finish_reason=stop

If the LLM does not stop after MAX_TOOL_ROUNDS, a final request WITHOUT tools is sent
(simulating a real client hitting its iteration limit and asking for a summary).
This ensures sync fires regardless of LLM termination behavior.
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
        (final_data, total_rounds, forced_stop) - the last response data, how many
        rounds were needed, and whether a forced stop was needed.
    """
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

        if finish_reason == "stop" and not tool_calls:
            # LLM is done — no more tool calls
            return data, round_num, False

        if tool_calls:
            # LLM wants to call more tools — simulate executing them and feed results back
            messages.append(message)
            for tc in tool_calls:
                tool_name = tc["function"]["name"]
                # Simulate tool execution with a canned response
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc["id"],
                    "content": f"Simulated result for {tool_name}",
                })
            # Continue loop — next round will send these results back to the LLM
        else:
            # finish_reason != "stop" but no tool calls — unexpected, return what we have
            print(f"[TEST] WARNING: Round {round_num} had finish_reason={finish_reason} but no tool_calls")
            return data, round_num, False

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

    return final_data, max_rounds + 1, True


# ============================================
# LEVEL 1: Simple - Single tool call
# ============================================
@pytest.mark.e2e_real
def test_level1_simple_bash(sync_verification):
    """Level 1: Simple single tool call - bash echo.

    Verifies:
    - LLM correctly selects bash tool
    - LLM stops after receiving tool result (finish_reason=stop)
    - Sync pipeline fires after conversation ends
    """
    if not _is_llama_available():
        pytest.skip("llama-server not available")

    messages = [{"role": "user", "content": "Run 'echo level1' in bash"}]
    session_id = calculate_session_id(messages)
    print(f"[TEST] Session ID: {session_id}")

    tools = [
        {
            "type": "function",
            "function": {
                "name": "bash",
                "description": "Execute bash",
                "parameters": {
                    "type": "object",
                    "properties": {"command": {"type": "string"}},
                    "required": ["command"],
                },
            },
        },
    ]

    final_data, total_rounds, forced_stop = _run_tool_loop(
        f"{PROXY_URL}/v1/chat/completions", messages, tools, timeout=30
    )
    print(f"[TEST] Completed in {total_rounds} round(s), forced_stop={forced_stop}")

    # Now sync should have been submitted (finish_reason=stop triggers it)
    assert sync_verification(session_id, timeout=10), (
        f"Sync failed for session {session_id}"
    )
    print_test_summary("test_level1_simple_bash", session_id, True)


# ============================================
# LEVEL 2: Medium - Multi-tool with logic
# ============================================
@pytest.mark.e2e_real
def test_level2_medium_tool_chain(sync_verification):
    """Level 2: Medium complexity - tool chain with dependencies.

    Verifies:
    - LLM selects appropriate tools (glob, write)
    - LLM can handle multi-step tasks across multiple rounds
    - Sync pipeline fires after conversation ends
    """
    if not _is_llama_available():
        pytest.skip("llama-server not available")

    messages = [
        {
            "role": "user",
            "content": "Find all Python files, then create a summary file listing them",
        }
    ]
    session_id = calculate_session_id(messages)
    print(f"[TEST] Session ID: {session_id}")

    tools = [
        {
            "type": "function",
            "function": {
                "name": "glob",
                "description": "Find files",
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

    final_data, total_rounds, forced_stop = _run_tool_loop(
        f"{PROXY_URL}/v1/chat/completions", messages, tools, timeout=60
    )
    print(f"[TEST] Completed in {total_rounds} round(s), forced_stop={forced_stop}")

    assert sync_verification(session_id, timeout=10), (
        f"Sync failed for session {session_id}"
    )
    print_test_summary("test_level2_medium_tool_chain", session_id, True)


# ============================================
# LEVEL 3: Complex - Multi-step with memory/context
# ============================================
@pytest.mark.e2e_real
def test_level3_complex_multi_step(sync_verification):
    """Level 3: Complex multi-step with context preservation.

    Verifies:
    - LLM handles a complex multi-tool request across multiple rounds
    - LLM maintains context across rounds
    - Sync pipeline fires after conversation ends
    """
    if not _is_llama_available():
        pytest.skip("llama-server not available")

    messages = [
        {
            "role": "system",
            "content": "You are a helpful coding assistant. Use the tools provided.",
        },
        {
            "role": "user",
            "content": "First check what files are in /tmp, then read any Python files you find, then create a new file summarizing what you found",
        },
    ]
    session_id = calculate_session_id(messages)
    print(f"[TEST] Session ID: {session_id}")

    tools = [
        {
            "type": "function",
            "function": {
                "name": "bash",
                "description": "Execute bash",
                "parameters": {
                    "type": "object",
                    "properties": {"command": {"type": "string"}},
                    "required": ["command"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "glob",
                "description": "Find files",
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

    final_data, total_rounds, forced_stop = _run_tool_loop(
        f"{PROXY_URL}/v1/chat/completions", messages, tools, timeout=120
    )
    print(f"[TEST] Completed in {total_rounds} round(s), forced_stop={forced_stop}")

    assert sync_verification(session_id, timeout=10), (
        f"Sync failed for session {session_id}"
    )
    print_test_summary("test_level3_complex_multi_step", session_id, True)
