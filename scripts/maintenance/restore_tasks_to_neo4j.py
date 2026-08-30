#!/usr/bin/env python3
import json
import sys
from datetime import datetime

WORKSPACE_ID = "choriprox"


def main():
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <tasks_backup.json>")
        return 2
    backup_path = sys.argv[1]
    with open(backup_path) as f:
        data = json.load(f)

    tasks = [t.get("all_props", {}) for t in data if t.get("uuid")]
    print(f"Tasks with uuid: {len(tasks)} (out of {len(data)} total records)")

    workspace_ids = set(t.get("workspace_id") for t in tasks)
    print(f"workspace_ids in backup: {workspace_ids}")

    from memory.query_executor import get_shared_executor_from_env
    qe = get_shared_executor_from_env()
    if qe is None:
        print("ERROR: Neo4j not configured")
        return 1

    now = datetime.utcnow().isoformat() + "Z"
    qe.execute_write(
        "MERGE (w:Workspace {workspace_id: $ws}) "
        "ON CREATE SET w.created_at = $now, w.name = $ws",
        {"ws": WORKSPACE_ID, "now": now}
    )
    print(f"Workspace '{WORKSPACE_ID}' ensured")

    inserted = 0
    skipped = 0
    for ap in tasks:
        uuid = ap.get("uuid")
        short_hash = ap.get("short_hash") or uuid[-8:]
        name = ap.get("name")
        if not name:
            skipped += 1
            continue
        ws = ap.get("workspace_id") or WORKSPACE_ID
        summary = ap.get("summary")
        if isinstance(summary, (dict, list)):
            summary = json.dumps(summary, default=str)
        params = {
            "uuid": uuid,
            "short_hash": short_hash,
            "name": name,
            "session_id": ap.get("session_id"),
            "priority": ap.get("priority") or "P2",
            "summary": summary if isinstance(summary, str) else (str(summary) if summary else None),
            "status": ap.get("status") or "pending",
            "tag": ap.get("tag"),
            "created_at": ap.get("created_at") or ap.get("first_seen") or now,
            "status_changed_at": ap.get("status_changed_at") or ap.get("created_at") or now,
            "deleted_at": ap.get("deleted_at"),
            "started_at": ap.get("started_at"),
            "completed_at": ap.get("completed_at"),
            "workspace_id": ws,
        }
        try:
            qe.execute_write_no_retry(
                "MATCH (w:Workspace {workspace_id: $workspace_id}) "
                "MERGE (t:Task {uuid: $uuid}) "
                "ON CREATE SET t:Entity "
                "SET t.short_hash = $short_hash, t.name = $name, "
                "    t.session_id = $session_id, t.priority = $priority, "
                "    t.summary = $summary, t.status = $status, t.tag = $tag, "
                "    t.created_at = $created_at, t.status_changed_at = $status_changed_at, "
                "    t.deleted_at = $deleted_at, t.started_at = $started_at, "
                "    t.completed_at = $completed_at, t.workspace_id = $workspace_id "
                "MERGE (t)-[:IN_WORKSPACE]->(w)",
                params
            )
            inserted += 1
        except Exception as e:
            print(f"FAILED {uuid}: {type(e).__name__}: {str(e)[:200]}")
            skipped += 1

    print(f"Inserted: {inserted}, Skipped: {skipped}")
    count = qe.execute_read(
        "MATCH (t:Task {workspace_id: $ws}) RETURN count(t) AS c",
        {"ws": WORKSPACE_ID}
    )
    print(f"Verified: {count[0]['c']} tasks in '{WORKSPACE_ID}'")

    pending = qe.execute_read(
        "MATCH (t:Task {workspace_id: $ws}) "
        "WHERE t.deleted_at IS NULL AND t.status IN ['pending', 'in_progress'] "
        "RETURN t.short_hash AS h, t.name AS name, t.priority AS p, t.status AS s "
        "ORDER BY CASE t.priority WHEN 'high' THEN 1 WHEN 'medium' THEN 2 WHEN 'low' THEN 3 ELSE 4 END, t.name",
        {"ws": WORKSPACE_ID}
    )
    print(f"\nActionable ({len(pending)}):")
    for r in pending:
        print(f"  [{r['p']}] {r['h']} {r['s']:12s} {r['name']}")

    # Create _Migration markers so the proxy doesn't re-run all migrations
    # on next startup (which is safe now but would be dangerous if a future
    # non-idempotent migration is added).
    for v in range(1, 11):
        qe.execute_write_no_retry(
            "MERGE (m:_Migration {version: $v}) "
            "SET m.status = 'completed', m.completed_at = datetime()",
            {"v": v}
        )
    print("Migration markers 1-10 ensured")
    return 0


if __name__ == "__main__":
    sys.exit(main())
