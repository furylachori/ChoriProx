#!/usr/bin/env python3
"""Fetch raw messages and summary for a single exchange."""

import argparse
import json
import sqlite3
import sys
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "memory.db"


def get_exchange_range(cursor, session_id: str, msg_index: int):
    """Find next user message to determine exchange end, and check summary end_message_index."""
    # Find the next user message after msg_index
    cursor.execute(
        "SELECT MIN(message_index) FROM messages "
        "WHERE session_id = ? AND message_index > ? AND role = 'user'",
        (session_id, msg_index),
    )
    next_user = cursor.fetchone()[0]

    # Find max message_index in the session
    cursor.execute(
        "SELECT MAX(message_index) FROM messages WHERE session_id = ?",
        (session_id,),
    )
    max_idx = cursor.fetchone()[0]

    if next_user is not None:
        # Exchange goes from msg_index to next_user - 1 (inclusive)
        end_by_msg = next_user - 1
    else:
        end_by_msg = max_idx

    # Also check end_message_index from message_summaries
    cursor.execute(
        "SELECT end_message_index FROM message_summaries "
        "WHERE session_id = ? AND message_index = ?",
        (session_id, msg_index),
    )
    row = cursor.fetchone()
    summary_end = row[0] if row and row[0] else None

    if summary_end is not None and summary_end > end_by_msg:
        end_idx = summary_end
    else:
        end_idx = end_by_msg

    return end_idx


def format_tool_calls_json(tool_calls_json: str) -> list[str]:
    """Extract tool names from tool_calls_json."""
    try:
        calls = json.loads(tool_calls_json)
        return [c["function"]["name"] for c in calls]
    except (json.JSONDecodeError, KeyError, TypeError):
        return []


def main():
    parser = argparse.ArgumentParser(
        description="Fetch raw messages and summary for a single exchange."
    )
    parser.add_argument("session_id", help="Session ID")
    parser.add_argument("message_index", type=int, help="User message index that starts the exchange")
    args = parser.parse_args()

    if not DB_PATH.exists():
        print("Database not found")
        sys.exit(1)

    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    # Verify the user message exists
    cursor.execute(
        "SELECT id FROM messages WHERE session_id = ? AND message_index = ? AND role = 'user'",
        (args.session_id, args.message_index),
    )
    if cursor.fetchone() is None:
        print(f"Error: No user message found at session={args.session_id}, message_index={args.message_index}")
        conn.close()
        sys.exit(1)

    end_idx = get_exchange_range(cursor, args.session_id, args.message_index)

    # Fetch messages in the range
    cursor.execute(
        "SELECT message_index, role, content, tool_calls_json, tool_call_id, created_at "
        "FROM messages "
        "WHERE session_id = ? AND message_index >= ? AND message_index <= ? "
        "ORDER BY message_index",
        (args.session_id, args.message_index, end_idx),
    )
    messages = cursor.fetchall()

    # Fetch summary
    cursor.execute(
        "SELECT summary, tool_summary, model_used, token_estimate "
        "FROM message_summaries "
        "WHERE session_id = ? AND message_index = ?",
        (args.session_id, args.message_index),
    )
    summary_row = cursor.fetchone()

    # --- Output ---
    print(f"=== Exchange: Session {args.session_id}, User Message #{args.message_index} ===")
    print(f"Range: messages {args.message_index} → {end_idx}")
    print()
    print("--- Raw Messages ---")
    print()

    for msg in messages:
        print(f"[Msg #{msg['message_index']}] {msg['role']} ({msg['created_at']})")

        if msg["tool_calls_json"]:
            names = format_tool_calls_json(msg["tool_calls_json"])
            if names:
                print(f"Tool calls: {', '.join(names)}")

        if msg["tool_call_id"]:
            print(f"tool_call_id: {msg['tool_call_id']}")

        if msg["content"]:
            content = msg["content"]
            if len(content) > 500:
                content = content[:500] + "\n[...truncated]"
            print(content)
        else:
            print("[no content]")

        print()

    # --- Summary ---
    print("--- Summary ---")

    if summary_row is None:
        print("No summary found for this exchange.")
    else:
        if summary_row["model_used"]:
            print(f"model:  {summary_row['model_used']}")
        if summary_row["token_estimate"]:
            print(f"tokens: {summary_row['token_estimate']}")
        print()

        print(summary_row["summary"])
        print()

        if summary_row["tool_summary"]:
            print("Tool summary:")
            print(summary_row["tool_summary"])

    conn.close()


if __name__ == "__main__":
    main()
