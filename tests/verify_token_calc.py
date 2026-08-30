#!/usr/bin/env python3
"""
Quick verification of token calculation with tools.
"""

import sys
import os

sys.path.append(os.path.dirname(__file__))

from qwen_proxy import calculate_tokens_from_request_body

# Test payloads
test_with_tools = {
    "messages": [{"role": "user", "content": "test"}],
    "tools": [
        {
            "type": "function",
            "function": {
                "name": "read_file",
                "description": "Read file contents",
                "parameters": {
                    "type": "object",
                    "properties": {"path": {"type": "string"}},
                    "required": ["path"],
                },
            },
        }
    ],
}

test_without_tools = {"messages": [{"role": "user", "content": "test"}]}

print("=== TOKEN CALCULATION VERIFICATION ===")

# Test with tools
tokens_with_tools = calculate_tokens_from_request_body(test_with_tools)
print(f"With tools: {tokens_with_tools:.0f} tokens")

# Test without tools
tokens_without_tools = calculate_tokens_from_request_body(test_without_tools)
print(f"Without tools: {tokens_without_tools:.0f} tokens")

print(f"Tool overhead: {tokens_with_tools - tokens_without_tools:.0f} tokens")

# Test with cached tools
test_cached_tools = {
    "messages": [{"role": "user", "content": "test"}],
    "tools": {"cached": True, "hash": "abc123", "count": 1},
}

tokens_cached = calculate_tokens_from_request_body(test_cached_tools)
print(f"With cached tools: {tokens_cached:.0f} tokens")

print("\n=== VERIFICATION COMPLETE ===")
