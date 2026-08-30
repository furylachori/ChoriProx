#!/usr/bin/env python3
"""Print general statistics about all sessions in the database."""

import argparse
import sqlite3
from pathlib import Path
from textwrap import dedent

DB_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "memory.db"


def main() -> None:
    parser = argparse.ArgumentParser(description="Print session statistics.")
    parser.parse_args()

    if not DB_PATH.exists():
        print("Database not found")
        return

    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row

    sessions = conn.execute(
        "SELECT DISTINCT session_id FROM messages ORDER BY session_id"
    ).fetchall()

    if not sessions:
        print("No sessions found.")
        conn.close()
        return

    print("=== Session Statistics ===")
    print(f"Total sessions: {len(sessions)}")

    stats_rows = []
    for (sid,) in sessions:
        raw_total = conn.execute(
            "SELECT COUNT(*) FROM messages WHERE session_id = ?", (sid,)
        ).fetchone()[0]
        user_count = conn.execute(
            "SELECT COUNT(*) FROM messages WHERE session_id = ? AND role = 'user'",
            (sid,),
        ).fetchone()[0]
        asst_count = conn.execute(
            "SELECT COUNT(*) FROM messages WHERE session_id = ? AND role = 'assistant'",
            (sid,),
        ).fetchone()[0]
        tool_count = conn.execute(
            "SELECT COUNT(*) FROM messages WHERE session_id = ? AND role = 'tool'",
            (sid,),
        ).fetchone()[0]
        exch_count = conn.execute(
            "SELECT COUNT(*) FROM message_summaries WHERE session_id = ?",
            (sid,),
        ).fetchone()[0]
        sum_count = exch_count  # same as number of exchanges per spec
        cs = conn.execute(
            "SELECT last_message_index, ancient_history FROM conversation_summaries WHERE session_id = ?",
            (sid,),
        ).fetchone()
        last_idx = cs["last_message_index"] if cs else None
        ancient = "yes" if cs and cs["ancient_history"] else "no"
        ent_count = conn.execute(
            "SELECT COUNT(*) FROM entities WHERE session_id = ?", (sid,)
        ).fetchone()[0]
        total_tokens = conn.execute(
            "SELECT COALESCE(SUM(rendered_full_tokens), 0) FROM message_summaries WHERE session_id = ?",
            (sid,),
        ).fetchone()[0]

        stats_rows.append(
            (raw_total, sid, user_count, asst_count, tool_count, exch_count, sum_count, last_idx, ancient, ent_count, total_tokens)
        )

    conn.close()

    stats_rows.sort(key=lambda r: r[0], reverse=True)

    for row in stats_rows:
        (
            raw_total, sid, user_count, asst_count, tool_count,
            exch_count, sum_count, last_idx, ancient, ent_count, total_tokens,
        ) = row
        print()
        print(f"Session: {sid}")
        print(f"  Raw messages:    {raw_total}")
        print(f"    user:           {user_count}")
        print(f"    assistant:      {asst_count}")
        print(f"    tool:           {tool_count}")
        print(f"  Exchanges:        {exch_count}")
        print(f"  Summaries:        {sum_count}")
        print(f"  Last msg index:   {last_idx if last_idx is not None else 'N/A'}")
        print(f"  Ancient history:  {ancient}")
        print(f"  Entities:         {ent_count}")
        print(f"  Total tokens:     {total_tokens}")


if __name__ == "__main__":
    main()
