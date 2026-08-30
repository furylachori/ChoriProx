#!/usr/bin/env python3
"""Fetch all entities related to a given exchange."""

import argparse
import sqlite3
import sys
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "memory.db"


def main():
    parser = argparse.ArgumentParser(
        description="Fetch all entities related to a given exchange."
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

    # Find the extraction batch that covers this message_index
    cursor.execute(
        "SELECT start_message_index, end_message_index, status, created_at, updated_at "
        "FROM extraction_batches "
        "WHERE session_id = ? AND start_message_index <= ? AND end_message_index >= ? "
        "ORDER BY id",
        (args.session_id, args.message_index, args.message_index),
    )
    batch = cursor.fetchone()

    if batch is None:
        print(f"No extraction batch covers message #{args.message_index}. "
              "The exchange may not have been extracted yet.")
        conn.close()
        sys.exit(0)

    # Fetch all entities for this session
    cursor.execute(
        "SELECT id, name, entity_type, summary, extracted_at "
        "FROM entities "
        "WHERE session_id = ? "
        "ORDER BY entity_type, name",
        (args.session_id,),
    )
    entities = cursor.fetchall()

    # --- Output ---
    print(f"=== Entities for Session {args.session_id}, Exchange starting at Msg #{args.message_index} ===")
    print(f"Extraction batch: messages {batch['start_message_index']} → {batch['end_message_index']} "
          f"(status: {batch['status']}, extracted {batch['created_at']})")
    print()

    if not entities:
        print("No entities found for this session.")
        conn.close()
        sys.exit(0)

    # Pre-fetch all facts for this session's entities
    entity_ids = [e["id"] for e in entities]
    facts_by_entity_id: dict[int, list] = {eid: [] for eid in entity_ids}

    if entity_ids:
        placeholders = ",".join("?" for _ in entity_ids)
        cursor.execute(
            f"SELECT id, source_entity_id, target_entity_id, relationship_type, fact "
            f"FROM facts "
            f"WHERE session_id = ? AND (source_entity_id IN ({placeholders}) OR target_entity_id IN ({placeholders}))",
            [args.session_id] + entity_ids + entity_ids,
        )
        all_facts = cursor.fetchall()

        # Index entities by id for quick lookup
        entity_by_id = {e["id"]: e for e in entities}

        for f in all_facts:
            sid = f["source_entity_id"]
            tid = f["target_entity_id"]
            rtype = f["relationship_type"]

            # Add fact to source entity
            if sid in facts_by_entity_id:
                target_name = entity_by_id[tid]["name"] if tid in entity_by_id else f"entity#{tid}"
                target_type = entity_by_id[tid]["entity_type"] if tid in entity_by_id else "?"
                facts_by_entity_id[sid].append(
                    (rtype, target_name, target_type, True)
                )
            # Add fact to target entity
            if tid in facts_by_entity_id:
                source_name = entity_by_id[sid]["name"] if sid in entity_by_id else f"entity#{sid}"
                source_type = entity_by_id[sid]["entity_type"] if sid in entity_by_id else "?"
                facts_by_entity_id[tid].append(
                    (rtype, source_name, source_type, False)
                )

    # Display
    print(f"--- Entities ({len(entities)} total) ---")
    print()

    for entity in entities:
        print(f"Entity: {entity['name']}")
        print(f"  Type:     {entity['entity_type']}")
        if entity["summary"]:
            print(f"  Summary:  {entity['summary']}")

        e_facts = facts_by_entity_id.get(entity["id"], [])
        if e_facts:
            print("  Facts:")
            for rtype, other_name, other_type, is_outgoing in e_facts:
                if is_outgoing:
                    arrow = f"→ {rtype} {other_name} ({other_type})"
                else:
                    arrow = f"← {rtype} {other_name} ({other_type})"
                print(f"    {arrow}")
        print()

    conn.close()


if __name__ == "__main__":
    main()
