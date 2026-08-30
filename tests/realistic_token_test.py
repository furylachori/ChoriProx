#!/usr/bin/env python3
"""
Realistic token calculation test with actual message sizes.
"""

import sys
import os
import json

sys.path.append(os.path.dirname(__file__))

from qwen_proxy import calculate_tokens_from_request_body

# Realistic test payloads
realistic_message = "Please read the README.md file and tell me what it contains. I need to understand the project structure and setup instructions."

realistic_tools = [
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

test_with_tools = {
    "messages": [{"role": "user", "content": realistic_message}],
    "tools": realistic_tools,
}

test_without_tools = {"messages": [{"role": "user", "content": realistic_message}]}

print("=== REALISTIC TOKEN CALCULATION ===")
print(f"Message length: {len(realistic_message):,} characters")
print(f"Message content: '{realistic_message[:50]}...'")

# Calculate sizes manually
tools_json = json.dumps(realistic_tools, separators=(",", ":"))
print(f"\nTool definitions: {len(tools_json):,} characters")
print(
    f"Total chars (message + tools): {len(realistic_message) + len(tools_json):,} chars"
)

# Test with tools
tokens_with_tools = calculate_tokens_from_request_body(test_with_tools)
print(f"\nWith tools: {tokens_with_tools:.0f} tokens")

# Test without tools
tokens_without_tools = calculate_tokens_from_request_body(test_without_tools)
print(f"Without tools: {tokens_without_tools:.0f} tokens")

print(f"Tool overhead: {tokens_with_tools - tokens_without_tools:.0f} tokens")

# Test with cached tools
test_cached_tools = {
    "messages": [{"role": "user", "content": realistic_message}],
    "tools": {"cached": True, "hash": "abc123", "count": 2},
}

tokens_cached = calculate_tokens_from_request_body(test_cached_tools)
print(f"With cached tools: {tokens_cached:.0f} tokens")

print(f"\nCaching saves: {tokens_with_tools - tokens_cached:.0f} tokens per request")
print(
    f"Caching reduces tool overhead by: {(tokens_with_tools - tokens_cached) / (tokens_with_tools - tokens_without_tools) * 100:.1f}%"
)

print("\n=== BREAKDOWN ===")
print(f"Message tokens: ~{tokens_without_tools:.0f}")
print(f"Full tool tokens: ~{tokens_with_tools - tokens_without_tools:.0f}")
print(f"Cached tool tokens: ~{tokens_cached - tokens_without_tools:.0f}")
print(f"Token savings from caching: ~{tokens_with_tools - tokens_cached:.0f}")
