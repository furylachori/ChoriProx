#!/usr/bin/env python3
"""
Test oversized prompt handling in context optimization
"""

import sys
import os

sys.path.append(os.path.dirname(__file__))

from qwen_proxy import select_context, calculate_tokens_from_text


def test_oversized_prompt():
    """Test that oversized prompts trigger single message mode"""

    # Test data
    session_id = "test_session"

    # Normal conversation
    normal_messages = [
        {"role": "user", "content": "Hello, can you help me?"},
        {"role": "assistant", "content": "Sure, what do you need?"},
        {"role": "user", "content": "I need to write some code"},
    ]

    # Oversized prompt (simulate MEGA_TEST)
    oversized_content = "x" * 5000  # 5,000 chars = ~1,250 tokens
    oversized_messages = [
        {"role": "user", "content": "Previous message"},
        {"role": "assistant", "content": "Previous response"},
        {"role": "user", "content": oversized_content},
    ]

    print("=== TESTING OVERSIZED PROMPT HANDLING ===")

    # Test normal conversation
    print("\n1. Normal conversation:")
    result_normal = select_context(session_id, normal_messages)
    print(f"   Input messages: {len(normal_messages)}")
    print(f"   Output messages: {len(result_normal)}")

    # Test oversized prompt
    print("\n2. Oversized prompt:")
    last_message_tokens = calculate_tokens_from_text(oversized_content)
    print(f"   Last message tokens: {last_message_tokens:.0f}")
    result_oversized = select_context(session_id, oversized_messages)
    print(f"   Input messages: {len(oversized_messages)}")
    print(f"   Output messages: {len(result_oversized)}")

    # Verify oversized prompt triggers single message mode
    if len(result_oversized) == 1:
        print("✅ SUCCESS: Oversized prompt correctly triggers single message mode")
    else:
        print("❌ FAILURE: Oversized prompt should trigger single message mode")

    # Verify normal conversation preserves context
    if len(result_normal) > 1:
        print("✅ SUCCESS: Normal conversation preserves context")
    else:
        print("❌ FAILURE: Normal conversation should preserve context")


if __name__ == "__main__":
    test_oversized_prompt()
