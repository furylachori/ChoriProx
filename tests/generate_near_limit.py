#!/usr/bin/env python3
"""
generate_near_limit.py

Creates a text file whose size is ~99.99 % of the proxy safe‑token limit
(13 000 tokens → 52 000 characters → ~51 996 chars).

Each line is the literal string:
    ASGHFJGLKJL:JLJLKJLKJLKJLKJLKJLKJLKJ

The resulting file is named `near_limit.txt` and lives in the current directory.
"""

import math
import pathlib

# Configuration – matches the proxy's token calculation
SAFE_TOKENS = 13_000  # safe token limit for the default model
CHARS_PER_TOKEN = 4  # proxy uses 0.25 tokens per char → 4 chars per token
TARGET_RATIO = 0.9999  # 99.99 % of the safe limit

# Compute target character count
target_chars = int(SAFE_TOKENS * CHARS_PER_TOKEN * TARGET_RATIO)

# Line to repeat (including newline)
line = "ASGHFJGLKJL:JLJLKJLKJLKJLKJLKJLKJLKJ\n"
line_len = len(line)

# Number of repetitions needed (ceil to not exceed the target)
num_lines = math.ceil(target_chars / line_len)

# Verify final size (should be ≤ target_chars + line_len)
final_chars = num_lines * line_len
assert final_chars <= target_chars + line_len, "Logic error in size calculation"

# Write the file
output_path = pathlib.Path("near_limit.txt")
with output_path.open("w", encoding="utf-8") as f:
    for _ in range(num_lines):
        f.write(line)

# Report
print(f"Generated {output_path} with:")
print(f"  • Lines      : {num_lines:,}")
print(f"  • Total chars: {final_chars:,}")
print(f"  • Approx. tokens (chars/4): {final_chars / 4:,.1f}")
print(f"  • Ratio of safe limit: {(final_chars / (SAFE_TOKENS * CHARS_PER_TOKEN)):.5%}")
