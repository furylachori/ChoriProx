#!/usr/bin/env python3
"""
Test script to verify token calculation includes tool definitions.
"""

import json
import requests
import sys


def test_token_calculation_with_tools():
    """Test that token calculation correctly includes tool definitions."""

    # Standard tool set
    standard_tools = [
        {
            "type": "function",
            "function": {
                "name": "read_file",
                "description": "Read file contents",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string"},
                        "offset": {"type": "integer", "default": 1},
                        "limit": {"type": "integer", "default": 2000},
                    },
                    "required": ["path"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "write_file",
                "description": "Write content to a file (overwrites existing)",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string"},
                        "content": {"type": "string"},
                    },
                    "required": ["path", "content"],
                },
            },
        },
    ]

    print("=== TOKEN CALCULATION TEST WITH TOOLS ===")

    # Test payload with tools
    test_payload = {
        "model": "qwen3-coder",
        "messages": [{"role": "user", "content": "Please read the README.md file"}],
        "tools": standard_tools,
        "stream": False,
    }

    # Calculate sizes manually
    message_content = test_payload["messages"][0]["content"]
    tools_json = json.dumps(standard_tools, separators=(",", ":"))

    print(f"Message content: {len(message_content):,} chars")
    print(f"Tool definitions: {len(tools_json):,} chars")
    print(
        f"Total chars (message + tools): {len(message_content) + len(tools_json):,} chars"
    )
    print(
        f"Estimated tokens (0.25 ratio): {(len(message_content) + len(tools_json)) * 0.25:.0f} tokens"
    )

    # First request - should show accurate token count
    print("\n=== FIRST REQUEST (with tools) ===")
    try:
        response = requests.post(
            "http://localhost:8000/v1/chat/completions", json=test_payload, timeout=10
        )
        print(f"Status: {response.status_code}")
        if response.status_code != 200:
            print(f"Response: {response.text}")
    except Exception as e:
        print(f"Error: {e}")

    # Second request - should show cached tools token count
    print("\n=== SECOND REQUEST (cached tools) ===")
    try:
        response = requests.post(
            "http://localhost:8000/v1/chat/completions", json=test_payload, timeout=10
        )
        print(f"Status: {response.status_code}")
    except Exception as e:
        print(f"Error: {e}")

    # Test without tools for comparison
    print("\n=== REQUEST WITHOUT TOOLS ===")
    test_payload_no_tools = {
        "model": "qwen3-coder",
        "messages": [{"role": "user", "content": "Please read the README.md file"}],
        "stream": False,
    }

    try:
        response = requests.post(
            "http://localhost:8000/v1/chat/completions",
            json=test_payload_no_tools,
            timeout=10,
        )
        print(f"Status: {response.status_code}")
    except Exception as e:
        print(f"Error: {e}")

    print("\n=== TEST COMPLETE ===")
    print("Check the proxy logs to see if token calculations now include tools.")


if __name__ == "__main__":
    test_token_calculation_with_tools()
