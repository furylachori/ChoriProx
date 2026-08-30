#!/usr/bin/env python3
"""Fetch the last N per-exchange summaries for a given session."""

import argparse
import sqlite3
from pathlib import Path
from textwrap import dedent

DB_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "memory.db"


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch per-exchange summaries.")
    parser.add_argument("session_id", help="Session ID to query")
    parser.add_argument(
        "--last",
        type=int,
        default=5,
        help="Number of most recent summaries to return (default: 5)",
    )
    args = parser.parse_args()

    if not DB_PATH.exists():
        print("Database not found")
        return

    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row

    exists = conn.execute(
        "SELECT 1 FROM messages WHERE session_id = ? LIMIT 1", (args.session_id,)
    ).fetchone()
    if not exists:
        print(f"Session not found: {args.session_id}")
        conn.close()
        return

    summaries = conn.execute(
        """SELECT message_index, end_message_index, role, summary,
                  tool_summary, model_used, token_estimate, created_at
           FROM message_summaries
           WHERE session_id = ?
           ORDER BY message_index DESC
           LIMIT ?""",
        (args.session_id, args.last),
    ).fetchall()

    conn.close()

    if not summaries:
        print("No summaries found for this session.")
        return

    print(
        f"=== Last {args.last} Summaries for session {args.session_id} ==="
    )

    summaries.reverse()  # show oldest first for readability (they're in DESC order from db)

    for s in summaries:
        print()
        print(
            f"--- Exchange #{s['message_index']} "
            f"(msg {s['message_index']} → {s['end_message_index']}) ---"
        )
        print(f"message_index:     {s['message_index']}")
        print(f"end_message_index: {s['end_message_index']}")
        print(f"role:              {s['role']}")
        print(f"model:             {s['model_used'] or 'N/A'}")
        print(f"tokens:            {s['token_estimate'] or 0}")
        print(f"created:           {s['created_at']}")
        print()
        print("Summary:")
        print(s["summary"])
        if s["tool_summary"]:
            print()
            print("Tool summary:")
            print(s["tool_summary"])


if __name__ == "__main__":
    main()
