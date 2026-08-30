#!/usr/bin/env python3
"""Backfill: persist extracted entities/facts from extraction_batches to entities/facts tables.

Usage:
    python -m scripts.backfill_entities [--dry-run] [--session-id SID]

Safe to run multiple times — uses INSERT OR IGNORE for idempotency.
"""
import argparse
import json
import logging
import os
import sys
import time

# Ensure the src directory is on the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("backfill")

# Import after path setup so memory.db resolves
from memory.db import connection, store_entity, store_fact  # noqa: E402
from memory.sync import _store_entities_and_facts  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description="Backfill entity/fact persistence")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be done")
    parser.add_argument("--session-id", type=str, help="Limit to a single session")
    args = parser.parse_args()

    where = ""
    params = []
    if args.session_id:
        where = "AND session_id = ?"
        params.append(args.session_id)

    with connection(pooled=True) as conn:
        rows = conn.execute(
            f"SELECT id, session_id, extraction_json FROM extraction_batches "
            f"WHERE extraction_json IS NOT NULL AND status IN ('completed', 'extracted') "
            f"{where} "
            f"ORDER BY id",
            params,
        ).fetchall()

    logger.info("Found %d batches to backfill", len(rows))

    total_entities = 0
    total_facts = 0
    for row in rows:
        batch_id = row["id"]
        session_id = row["session_id"]
        try:
            data = json.loads(row["extraction_json"]) if isinstance(row["extraction_json"], str) else row["extraction_json"]
        except (json.JSONDecodeError, TypeError) as e:
            logger.warning("Skipping batch %d: invalid JSON (%s)", batch_id, e)
            continue

        # Only handle v2 format (entities + facts, no events)
        if "entities" not in data or "facts" not in data:
            logger.debug("Skipping batch %d: not v2 format", batch_id)
            continue

        if args.dry_run:
            logger.info("[DRY RUN] Batch %d session %s: %d entities, %d facts",
                       batch_id, session_id,
                       len(data.get("entities", []) or []),
                       len(data.get("facts", []) or []))
            continue

        result = _store_entities_and_facts(session_id, data)
        total_entities += result["entities_sqlite"]
        total_facts += result["facts_sqlite"]
        if result["errors"]:
            logger.warning("Batch %d had errors: %s", batch_id, result["errors"])

    logger.info("Backfill complete: %d entities, %d facts stored", total_entities, total_facts)


if __name__ == "__main__":
    main()
