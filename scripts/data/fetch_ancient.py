#!/usr/bin/env python3
"""Fetch the ancient history summary for a given session."""

import argparse
import sqlite3
from pathlib import Path
from textwrap import dedent

DB_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "memory.db"


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch ancient history for a session.")
    parser.add_argument("session_id", help="Session ID to query")
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

    cs = conn.execute(
        """SELECT exchange_count, last_message_index, created_at, ancient_history
           FROM conversation_summaries
           WHERE session_id = ?""",
        (args.session_id,),
    ).fetchone()

    conn.close()

    if cs is None:
        print(f"No conversation summaries found for session: {args.session_id}")
        return

    if not cs["ancient_history"]:
        print("No ancient history for this session.")
        return

    print(
        f"=== Ancient History for session {args.session_id} ==="
    )
    print("Session info:")
    print(f"  exchanges:        {cs['exchange_count'] or 0}")
    print(f"  last msg index:   {cs['last_message_index']}")
    print(f"  created:          {cs['created_at']}")
    print()
    print("Ancient History:")
    print(cs["ancient_history"])


if __name__ == "__main__":
    main()
