"""Entity and fact storage for memory system.

Extracted from db.py for modularity. These functions handle entity/fact
CRUD operations in SQLite (ChoriProx Plus §5.6).
"""

import logging
import sqlite3

from proxy.utils import _now_iso_sqlite

logger = logging.getLogger("memory.db")

_sqlite_version = tuple(int(x) for x in sqlite3.sqlite_version.split("."))
HAS_RETURNING = _sqlite_version >= (3, 35, 0)


def store_entity(conn, name, entity_type, summary="", session_id="", extracted_at=None):
    """Store an entity in SQLite. Returns entity row id (int), or None on failure.

    Uses RETURNING on SQLite 3.35+ for single-query upsert. Falls back to
    INSERT OR IGNORE + SELECT on older versions. Caller must commit.
    """
    now = _now_iso_sqlite()
    if extracted_at is None:
        extracted_at = now
    try:
        if HAS_RETURNING:
            cursor = conn.execute(
                """INSERT INTO entities (name, entity_type, summary, session_id, extracted_at, created_at, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?)
                   ON CONFLICT(name, entity_type, session_id)
                   DO UPDATE SET updated_at = CURRENT_TIMESTAMP, summary = excluded.summary
                   RETURNING id""",
                (name, entity_type, summary, session_id, extracted_at, now, now),
            )
            row = cursor.fetchone()
            return row[0] if row else None
        else:
            cursor = conn.execute(
                """INSERT OR IGNORE INTO entities (name, entity_type, summary, session_id, extracted_at, created_at, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (name, entity_type, summary, session_id, extracted_at, now, now),
            )
            if cursor.lastrowid:
                return cursor.lastrowid
            row = conn.execute(
                "SELECT id FROM entities WHERE name = ? AND entity_type = ? AND session_id = ?",
                (name, entity_type, session_id),
            ).fetchone()
            return row["id"] if row else None
    except Exception as e:
        logger.error("Failed to store entity %s/%s: %s session_id=%s", name, entity_type, e, session_id)
        return None


def store_fact(conn, source_entity_id, target_entity_id, relationship_type, fact,
               valid_at=None, invalid_at=None, session_id="", extracted_at=None):
    """Store a fact (relationship) in SQLite. Returns fact row id, or None on failure.

    Uses RETURNING on SQLite 3.35+ for single-query upsert. Falls back to
    INSERT OR IGNORE + SELECT on older versions. Caller must commit.
    """
    now = _now_iso_sqlite()
    if extracted_at is None:
        extracted_at = now
    try:
        if HAS_RETURNING:
            cursor = conn.execute(
                """INSERT INTO facts (source_entity_id, target_entity_id, relationship_type,
                   fact, valid_at, invalid_at, session_id, extracted_at, created_at, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                   ON CONFLICT(source_entity_id, target_entity_id, relationship_type, fact)
                   DO UPDATE SET updated_at = CURRENT_TIMESTAMP
                   RETURNING id""",
                (source_entity_id, target_entity_id, relationship_type, fact,
                 valid_at, invalid_at, session_id, extracted_at, now, now),
            )
            row = cursor.fetchone()
            return row[0] if row else None
        else:
            cursor = conn.execute(
                """INSERT OR IGNORE INTO facts (source_entity_id, target_entity_id, relationship_type,
                   fact, valid_at, invalid_at, session_id, extracted_at, created_at, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (source_entity_id, target_entity_id, relationship_type, fact,
                 valid_at, invalid_at, session_id, extracted_at, now, now),
            )
            if cursor.lastrowid:
                return cursor.lastrowid
            row = conn.execute(
                """SELECT id FROM facts
                   WHERE source_entity_id = ? AND target_entity_id = ?
                     AND relationship_type = ? AND fact = ?""",
                (source_entity_id, target_entity_id, relationship_type, fact),
            ).fetchone()
            return row["id"] if row else None
    except Exception as e:
        logger.error("Failed to store fact: %s session_id=%s", e, session_id)
        return None


def store_entities_batch(conn, entities):
    """Store multiple entities in batch. Returns list of IDs (2 queries total).

    Uses INSERT OR IGNORE + batch SELECT to resolve IDs. Caller must commit.
    """
    if not entities:
        return []

    entities = [e for e in entities if e.get("name")]
    if not entities:
        return []

    now = _now_iso_sqlite()
    params_per_row = 7
    chunk_size = min(999 // params_per_row, 500)

    for i in range(0, len(entities), chunk_size):
        chunk = entities[i:i + chunk_size]
        try:
            conn.executemany(
                """INSERT OR IGNORE INTO entities
                   (name, entity_type, summary, session_id, extracted_at, created_at, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                [
                    (
                        e["name"],
                        e.get("entity_type", e.get("type", "Concept")),
                        e.get("summary", ""),
                        e.get("session_id", ""),
                        e.get("extracted_at", now),
                        now,
                        now,
                    )
                    for e in chunk
                ],
            )
        except sqlite3.Error as e:
            logger.warning("SQLite entities batch insert failed: %s", e)

    all_ids = []
    for i in range(0, len(entities), chunk_size):
        chunk = entities[i:i + chunk_size]
        placeholders = ",".join(["(?, ?, ?)"] * len(chunk))
        params = []
        for e in chunk:
            params.extend([
                e["name"],
                e.get("entity_type", e.get("type", "Concept")),
                e.get("session_id", ""),
            ])
        try:
            rows = conn.execute(
                f"""SELECT id, name, entity_type FROM entities
                    WHERE (name, entity_type, session_id) IN (VALUES {placeholders})""",
                params,
            ).fetchall()
            id_map = {(r["name"], r["entity_type"]): r["id"] for r in rows}
            for e in chunk:
                eid = id_map.get((e["name"], e.get("entity_type", e.get("type", "Concept"))))
                if eid:
                    all_ids.append(eid)
        except sqlite3.Error as e:
            logger.warning("SQLite entities batch select failed: %s", e)

    return all_ids


def store_facts_batch(conn, facts):
    """Store multiple facts in batch. Returns list of IDs (2 queries total).

    Uses INSERT OR IGNORE + batch SELECT to resolve IDs. Caller must commit.
    """
    if not facts:
        return []

    facts = [f for f in facts if f.get("source_entity_id") and f.get("target_entity_id")]
    if not facts:
        return []

    now = _now_iso_sqlite()
    params_per_row = 10
    chunk_size = min(999 // params_per_row, 500)

    for i in range(0, len(facts), chunk_size):
        chunk = facts[i:i + chunk_size]
        try:
            conn.executemany(
                """INSERT OR IGNORE INTO facts
                   (source_entity_id, target_entity_id, relationship_type,
                    fact, valid_at, invalid_at, session_id, extracted_at, created_at, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                [
                    (
                        f["source_entity_id"],
                        f["target_entity_id"],
                        f.get("relationship_type", "RELATED_TO"),
                        f.get("fact", ""),
                        f.get("valid_at"),
                        f.get("invalid_at"),
                        f.get("session_id", ""),
                        f.get("extracted_at", now),
                        now,
                        now,
                    )
                    for f in chunk
                ],
            )
        except sqlite3.Error as e:
            logger.warning("SQLite facts batch insert failed: %s", e)

    all_ids = []
    for i in range(0, len(facts), chunk_size):
        chunk = facts[i:i + chunk_size]
        placeholders = ",".join(["(?, ?, ?, ?)"] * len(chunk))
        params = []
        for f in chunk:
            params.extend([
                f["source_entity_id"],
                f["target_entity_id"],
                f.get("relationship_type", "RELATED_TO"),
                f.get("fact", ""),
            ])
        try:
            rows = conn.execute(
                f"""SELECT id, source_entity_id, target_entity_id, relationship_type, fact
                    FROM facts
                    WHERE (source_entity_id, target_entity_id, relationship_type, fact)
                    IN (VALUES {placeholders})""",
                params,
            ).fetchall()
            id_map = {
                (r["source_entity_id"], r["target_entity_id"], r["relationship_type"], r["fact"]): r["id"]
                for r in rows
            }
            for f in chunk:
                fid = id_map.get((
                    f["source_entity_id"],
                    f["target_entity_id"],
                    f.get("relationship_type", "RELATED_TO"),
                    f.get("fact", ""),
                ))
                if fid:
                    all_ids.append(fid)
        except sqlite3.Error as e:
            logger.warning("SQLite facts batch select failed: %s", e)

    return all_ids
