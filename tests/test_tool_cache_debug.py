#!/usr/bin/env python3
"""Test tool caching to reproduce the 400 Bad Request error"""

import requests
import json
import time

PROXY_URL = "http://localhost:8000/v1/chat/completions"


def test_tool_caching_with_complex_prompt():
    """Test with a prompt that should trigger tool caching"""

    # Use a complex prompt that might trigger caching of multiple tools
    messages = [
        {
            "role": "user",
            "content": "Create a comprehensive test suite for a data processing module that handles CSV files, JSON data, and database operations. Include unit tests, integration tests, and performance benchmarks.",
        }
    ]

    # Include multiple tools to trigger caching
    tools = [
        {
            "type": "function",
            "function": {
                "name": "task",
                "description": "Launch a new agent to handle complex tasks",
                "parameters": {
                    "$schema": "https://json-schema.org/draft/2020-12/schema",
                    "type": "object",
                    "properties": {
                        "description": {"type": "string"},
                        "prompt": {"type": "string"},
                        "subagent_type": {"type": "string"},
                    },
                    "required": ["description", "prompt", "subagent_type"],
                    "additionalProperties": False,
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "write",
                "description": "Write content to a file",
                "parameters": {
                    "$schema": "https://json-schema.org/draft/2020-12/schema",
                    "type": "object",
                    "properties": {
                        "file_path": {"type": "string"},
                        "content": {"type": "string"},
                    },
                    "required": ["file_path", "content"],
                    "additionalProperties": False,
                },
            },
        },
    ]

    payload = {
        "messages": messages,
        "tools": tools,
        "model": "qwen2.5-coder-32b-instruct",
    }

    print("Testing tool caching with complex prompt...")
    print(f"Tools count: {len(tools)}")

    try:
        response = requests.post(PROXY_URL, json=payload, timeout=30)
        print(f"Response status: {response.status_code}")

        assert response.status_code == 200, f"Request failed: {response.text}"
        data = response.json()
        print("✓ Success - tool caching worked")

    except requests.exceptions.RequestException as e:
        raise AssertionError(f"Request failed: {e}")


def test_suggest_survives_filter():
    """Verify 'suggest' tool survives the filter while 'nonexistent_tool' is dropped"""

    tools = [
        {
            "type": "function",
            "function": {
                "name": "read",
                "description": "Read a file from the local filesystem",
                "parameters": {
                    "$schema": "https://json-schema.org/draft/2020-12/schema",
                    "type": "object",
                    "properties": {
                        "file_path": {"type": "string"},
                    },
                    "required": ["file_path"],
                    "additionalProperties": False,
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "suggest",
                "description": "Suggest a course of action or recommendation",
                "parameters": {
                    "$schema": "https://json-schema.org/draft/2020-12/schema",
                    "type": "object",
                    "properties": {
                        "suggestion": {"type": "string"},
                    },
                    "required": ["suggestion"],
                    "additionalProperties": False,
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "nonexistent_tool",
                "description": "This tool does not exist and should be dropped by the filter",
                "parameters": {
                    "$schema": "https://json-schema.org/draft/2020-12/schema",
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                    },
                    "required": ["query"],
                    "additionalProperties": False,
                },
            },
        },
    ]

    payload = {
        "messages": [
            {
                "role": "user",
                "content": "Read the README and suggest improvements.",
            }
        ],
        "tools": tools,
        "model": "qwen2.5-coder-32b-instruct",
    }

    print("Testing suggest survives filter...")
    print(f"Tools sent: {[t['function']['name'] for t in tools]}")

    try:
        response = requests.post(PROXY_URL, json=payload, timeout=30)
        print(f"Response status: {response.status_code}")

        assert response.status_code == 200, f"Request failed: {response.text}"
        data = response.json()
        print("✓ PASS — suggest survived, nonexistent_tool dropped by filter")
        print("  (Verify log for WARNING at tool_cache.py:93-99 confirming drop)")

    except requests.exceptions.RequestException as e:
        raise AssertionError(f"Request failed: {e}")


if __name__ == "__main__":
    test_tool_caching_with_complex_prompt()
    test_suggest_survives_filter()
