import logging
import uuid

from proxy.utils import _now_iso_sqlite
from memory._db import get_db as _db

logger = logging.getLogger("memory.sync")


def _get_graph():
    """Get shared Neo4jClient from TaskGraph singleton."""
    from proxy.context import _get_task_graph_singleton
    tg = _get_task_graph_singleton()
    return tg._neo4j


def _create_cooccurrence_relations(memory_id, content):
    """Create RELATES_TO edges between entities mentioned together in a memory.
    
    Secondary feature — must never crash the sync pipeline.
    TODO: Implement using a proper entity-search API when available.
    """
    pass


def _store_memories(session_id: str, memories: list[dict] = None, prefix: str = "",
                    events: list[dict] = None, instructions: list[dict] = None,
                    tasks: list[dict] = None) -> bool:
    """Store memories in Neo4j. Returns True if any were stored.

    Two call patterns:
    1. Legacy: _store_memories(session_id, [...], prefix="Recovery ")
       — used by recovery path, each dict already has memory_type set.
    2. New: _store_memories(session_id, events=[...], instructions=[...], tasks=[...])
       — used by main sync path, sets memory_type automatically.
    """
    g = _get_graph()
    now = _now_iso_sqlite()

    all_memories = list(memories) if memories else []

    if events:
        for ev in events:
            ev = dict(ev)
            ev["memory_type"] = "event"
            all_memories.append(ev)
    if instructions:
        for ins in instructions:
            ins = dict(ins)
            ins["memory_type"] = "instruction"
            all_memories.append(ins)
    if tasks:
        for tsk in tasks:
            tsk = dict(tsk)
            tsk["memory_type"] = "task"
            all_memories.append(tsk)

    any_stored = False
    entity_extraction_pairs: list[tuple[str, str]] = []

    for memory in all_memories:
        memory["session_id"] = session_id
        if not memory.get("id"):
            memory["id"] = str(uuid.uuid4())
        if "created_at" not in memory:
            memory["created_at"] = now
        if "valid_at" not in memory:
            memory["valid_at"] = now
        if "last_accessed_at" not in memory:
            memory["last_accessed_at"] = now
        if "access_count" not in memory:
            memory["access_count"] = 0

    topic_key_memories = [m for m in all_memories if m.get("topic_key")]
    simple_memories = [m for m in all_memories if not m.get("topic_key")]

    for memory in topic_key_memories:
        try:
            stored = g.store_with_supersession(memory)
        except Exception as e:
            logger.warning("Neo4j store_with_supersession failed: %s", e)
            stored = False
        if stored:
            any_stored = True
            memory_id = memory.get("id")
            content = memory.get("content", "")
            if memory_id and content:
                entity_extraction_pairs.append((memory_id, content))
                _create_cooccurrence_relations(memory_id, content)

    for memory in simple_memories:
        try:
            stored = g.store_memory(memory)
        except Exception as e:
            logger.warning("Neo4j store_memory failed: %s", e)
            stored = False
        if stored:
            any_stored = True
            memory_id = memory.get("id")
            content = memory.get("content", "")
            if memory_id and content:
                entity_extraction_pairs.append((memory_id, content))
                _create_cooccurrence_relations(memory_id, content)

    if all_memories:
        import sqlite3
        sqlite_now = _now_iso_sqlite()
        params_per_row = 7
        chunk_size = min(999 // params_per_row, 500)
        try:
            with _db().immediate_transaction(pooled=True) as conn:
                for i in range(0, len(all_memories), chunk_size):
                    chunk = all_memories[i:i + chunk_size]
                    try:
                        conn.executemany(
                            """INSERT OR IGNORE INTO memory_log
                               (memory_id, session_id, action, memory_type, topic_key, content, created_at)
                               VALUES (?, ?, ?, ?, ?, ?, ?)""",
                            [
                                (
                                    m.get("id"),
                                    session_id,
                                    "sync_store",
                                    m.get("memory_type", "event"),
                                    m.get("topic_key"),
                                    m.get("content", ""),
                                    sqlite_now,
                                )
                                for m in chunk
                            ],
                        )
                    except sqlite3.Error as e:
                        logger.warning("SQLite memory_log batch insert failed: %s", e)
        except Exception as sqle:
            logger.warning("[_store_memories] SQLite batch write failed: %s", sqle)

    if entity_extraction_pairs:
        g.store_entities_from_memories_batch(entity_extraction_pairs)

    return any_stored


def _store_entities_sqlite(conn, entities, session_id, now, result):
    """SQLite entity upsert. Returns entity_id_map."""
    entity_id_map = {}
    batch_entities = []
    for ent in entities:
        name = ent.get("name", "")
        etype = ent.get("entity_type", ent.get("type", "Concept"))
        summary = ent.get("summary", "")
        if not name:
            continue
        batch_entities.append({
            "name": name,
            "entity_type": etype,
            "summary": summary,
            "session_id": session_id,
            "extracted_at": now,
        })

    if batch_entities:
        ids = _db().store_entities_batch(conn, batch_entities)
        for i, eid in enumerate(ids):
            ent = batch_entities[i]
            entity_id_map[(ent["name"], ent["entity_type"])] = eid
            result["entities_sqlite"] += 1
    return entity_id_map


def _resolve_fact_entity_ids(conn, entity_id_map, facts, session_id, now, result):
    """Build reverse index, resolve source/target IDs for facts.

    Returns resolved_facts with the resolved entity_type (always non-None
    when source_id/target_id are non-None). Deterministic type resolution:
    entities from the current batch take priority; DB entities fall back
    via db_name_to_type_id, ordered by their stored entity_type ASC.
    """
    name_to_type_id = {}
    for (name, etype), eid in entity_id_map.items():
        name_to_type_id.setdefault(name, []).append((etype, eid))

    fact_entity_names = set()
    for fact in facts:
        sn = fact.get("source_entity_name", fact.get("source_entity", ""))
        tn = fact.get("target_entity_name", fact.get("target_entity", ""))
        if sn:
            fact_entity_names.add(sn)
        if tn:
            fact_entity_names.add(tn)
    already_mapped = {name for (name, _) in entity_id_map}
    names_to_lookup = fact_entity_names - already_mapped

    db_entity_lookup = {}
    db_name_to_type_id = {}
    if names_to_lookup:
        names_list = list(names_to_lookup)
        rows = []
        for chunk_start in range(0, len(names_list), 500):
            chunk = names_list[chunk_start:chunk_start + 500]
            placeholders = ",".join("?" * len(chunk))
            chunk_rows = conn.execute(
                f"SELECT id, name, entity_type FROM entities WHERE name IN ({placeholders}) AND session_id = ? ORDER BY id ASC",
                chunk + [session_id],
            ).fetchall()
            rows.extend(chunk_rows)
        for row in rows:
            nm = row["name"]
            et = row["entity_type"]
            eid = row["id"]
            db_entity_lookup[(nm, et)] = eid
            db_name_to_type_id.setdefault(nm, []).append((et, eid))
        for row in rows:
            nm = row["name"]
            if (nm, None) not in db_entity_lookup:
                db_entity_lookup[(nm, None)] = row["id"]

    resolved_facts = []
    for fact in facts:
        source_name = fact.get("source_entity_name", fact.get("source_entity", ""))
        target_name = fact.get("target_entity_name", fact.get("target_entity", ""))
        relationship_type = fact.get("relationship_type", "RELATED_TO")
        fact_text = fact.get("fact", "")
        valid_at = fact.get("valid_at") or now
        invalid_at = fact.get("invalid_at")

        if not source_name or not target_name or not fact_text:
            continue

        source_id = None
        target_id = None
        source_type = None
        target_type = None

        extracted_source_type = fact.get("source_entity_type") or fact.get("source_type")
        extracted_target_type = fact.get("target_entity_type") or fact.get("target_type")

        for etype, eid in name_to_type_id.get(source_name, []):
            if extracted_source_type is None or etype == extracted_source_type:
                source_type = etype
                source_id = eid
                break
        if source_id is None:
            for etype, eid in db_name_to_type_id.get(source_name, []):
                if extracted_source_type is None or etype == extracted_source_type:
                    source_type = etype
                    source_id = eid
                    break
        if source_id is None and extracted_source_type is not None:
            source_id = db_entity_lookup.get((source_name, extracted_source_type))
            if source_id is not None:
                source_type = extracted_source_type
        if source_id is None and extracted_source_type is None:
            source_id = db_entity_lookup.get((source_name, None))
            if source_id is not None:
                entries = db_name_to_type_id.get(source_name, [])
                source_type = entries[0][0] if entries else "Concept"

        for etype, eid in name_to_type_id.get(target_name, []):
            if extracted_target_type is None or etype == extracted_target_type:
                target_type = etype
                target_id = eid
                break
        if target_id is None:
            for etype, eid in db_name_to_type_id.get(target_name, []):
                if extracted_target_type is None or etype == extracted_target_type:
                    target_type = etype
                    target_id = eid
                    break
        if target_id is None and extracted_target_type is not None:
            target_id = db_entity_lookup.get((target_name, extracted_target_type))
            if target_id is not None:
                target_type = extracted_target_type
        if target_id is None and extracted_target_type is None:
            target_id = db_entity_lookup.get((target_name, None))
            if target_id is not None:
                entries = db_name_to_type_id.get(target_name, [])
                target_type = entries[0][0] if entries else "Concept"

        if source_id is None or target_id is None:
            missing_name = source_name if source_id is None else target_name
            result["errors"].append(
                f"Skipping fact: entity '{missing_name}' not found for fact '{fact_text[:80]}'"
            )
            continue

        resolved_facts.append({
            "source_id": source_id,
            "target_id": target_id,
            "source_name": source_name,
            "target_name": target_name,
            "source_type": source_type,
            "target_type": target_type,
            "relationship_type": relationship_type,
            "fact_text": fact_text,
            "valid_at": valid_at,
            "invalid_at": invalid_at,
        })
    return resolved_facts


def _store_facts_sqlite(conn, resolved_facts, session_id, now, result):
    """SQLite fact upsert."""
    if not resolved_facts:
        return
    batch_facts = [
        {
            "source_entity_id": rf["source_id"],
            "target_entity_id": rf["target_id"],
            "relationship_type": rf["relationship_type"],
            "fact": rf["fact_text"],
            "valid_at": rf["valid_at"],
            "invalid_at": rf["invalid_at"],
            "session_id": session_id,
            "extracted_at": now,
        }
        for rf in resolved_facts
    ]
    ids = _db().store_facts_batch(conn, batch_facts)
    result["facts_sqlite"] += len(ids)
    if len(ids) < len(resolved_facts):
        failed_count = len(resolved_facts) - len(ids)
        result["errors"].append(f"Failed to store {failed_count} fact(s)")


def _store_entities_neo4j(g, entities, resolved_facts, session_id, now, result):
    """Neo4j batch entity/relation writes.

    Uses resolved_facts (already resolved by _resolve_fact_entity_ids) for
    the Neo4j side. The resolved source_type/target_type are always non-None,
    so the Neo4j MATCH operates on the (name, entity_type) composite key,
    preventing Cartesian-product fan-out when multiple Entity nodes share a name.
    """
    if entities:
        neo_entities = [
            {
                "name": e.get("name", ""),
                "entity_type": e.get("entity_type", e.get("type", "Concept")),
                "session_id": session_id,
                "first_seen": e.get("extracted_at", now),
                "last_seen": e.get("extracted_at", now),
            }
            for e in entities
            if e.get("name")
        ]
        if neo_entities:
            ok = g.store_entities_batch(neo_entities, session_id=session_id)
            if ok:
                result["entities_neo4j"] = len(neo_entities)
            else:
                result["errors"].append(f"Neo4j entity batch failed for {len(neo_entities)} entities")

    if resolved_facts:
        neo_relations = [
            {
                "source_name": rf["source_name"],
                "target_name": rf["target_name"],
                "source_entity_type": rf["source_type"],
                "target_entity_type": rf["target_type"],
                "relation_type": rf["relationship_type"],
                "source_memory_id": None,
                "fact": rf["fact_text"],
                "created_at": now,
                "valid_at": rf["valid_at"],
                "invalid_at": rf["invalid_at"],
            }
            for rf in resolved_facts
        ]
        if neo_relations:
            ok = g.store_relations_batch(neo_relations, session_id=session_id)
            if ok:
                result["facts_neo4j"] = len(neo_relations)
            else:
                result["errors"].append(f"Neo4j relation batch failed for {len(neo_relations)} facts")


def _store_entities_and_facts(session_id: str, data: dict) -> dict:
    """Orchestrate entity/fact persistence to SQLite and Neo4j.

    Batch-stores entities then facts in a single SQLite transaction.
    Also writes to Neo4j (best-effort, failure logged but non-fatal).

    Returns:
        dict with counts: {'entities_sqlite': N, 'entities_neo4j': N,
                           'facts_sqlite': N, 'facts_neo4j': N,
                           'errors': [str, ...]}
    """
    from memory.db import transaction as db_transaction

    entities = data.get("entities", []) or []
    facts = data.get("facts", []) or []
    result = {
        "entities_sqlite": 0,
        "entities_neo4j": 0,
        "facts_sqlite": 0,
        "facts_neo4j": 0,
        "sqlite_ok": True,
        "errors": [],
    }

    if not entities and not facts:
        return result

    now = _now_iso_sqlite()
    g = _get_graph()

    try:
        with db_transaction() as conn:
            entity_id_map = _store_entities_sqlite(conn, entities, session_id, now, result)
            resolved_facts = _resolve_fact_entity_ids(conn, entity_id_map, facts, session_id, now, result)
            _store_facts_sqlite(conn, resolved_facts, session_id, now, result)
    except Exception as e:
        logger.error("SQLite entity/fact persistence failed: %s", e, exc_info=True)
        result["errors"].append(f"SQLite transaction failed: {e}")
        result["sqlite_ok"] = False

    sqlite_failed = not result.get("sqlite_ok", True)
    if sqlite_failed:
        logger.warning("Skipping Neo4j writes because SQLite persistence failed")
        return result

    try:
        _store_entities_neo4j(g, entities, resolved_facts, session_id, now, result)
    except Exception as e:
        logger.warning("Neo4j entity/fact persistence failed (SQLite OK): %s", e)
        result["errors"].append(f"Neo4j batch failed: {e}")

    return result
