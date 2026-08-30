"""Workspace persistence layer — Neo4j reads/writes for workspace state.

v5 global model:
- :Workspace nodes keyed on BARE name (no session_id)
- :Session nodes track which session last activated which workspace
- Persist writes :Workspace, :Session, and :LAST_USED_WORKSPACE in one Cypher
- Recovery loads the global catalog + per-session last-used
- Cleanup deletes :Session + edges (NEVER :Workspace — global, may be used by other sessions)
"""
import logging
from functools import partial

from proxy.workspace_state import MAX_RECENT_WORKSPACES, DEFAULT_WORKSPACE

logger = logging.getLogger("memory.workspace_persistence")

CLEANUP_BATCH_SIZE = 1000

# Public Cypher constant for the catalog query
CATALOG_QUERY = """
MATCH (w:Workspace)
WHERE NOT w.workspace_id CONTAINS ':'
RETURN w.workspace_id AS name, w.last_active_at AS last_active_at
ORDER BY w.last_active_at DESC
LIMIT $limit
"""


def _get_executor():
    """Get shared executor or None. Uses query_executor.get_shared_executor_from_env()."""
    from memory.query_executor import get_shared_executor_from_env
    return get_shared_executor_from_env()


def _persist_workspace(workspace_name: str, session_id: str) -> None:
    """Persist a workspace activation to Neo4j.

    Writes three things in one query (or two if Neo4j rejects multi-MERGE):
      1. MERGE :Workspace {name: workspace_name} — global catalog entry
      2. MERGE :Session {id: session_id} — session record
      3. MERGE (s)-[:LAST_USED_WORKSPACE]->(w) — per-session last-used pointer

    ON CREATE for :Workspace records `created_by_session` (audit only).
    All persist operations are fire-and-forget; failures are logged.
    """
    executor = _get_executor()
    if executor is None:
        return

    try:
        executor.execute_write(
            """
            MERGE (w:Workspace {workspace_id: $workspace_id})
            ON CREATE SET w.name = $name,
                          w.created_at = datetime(),
                          w.created_by_session = $session_id
            SET w.last_active_at = datetime()
            WITH w, $session_id AS sid
            MERGE (s:Session {id: sid})
            MERGE (s)-[r:LAST_USED_WORKSPACE]->(w)
            SET r.last_used_at = datetime()
            """,
            {
                "workspace_id": workspace_name,
                "name": workspace_name,
                "session_id": session_id,
            },
        )
    except Exception:
        logger.warning(
            "Workspace persist failed for %s (session %s)",
            workspace_name, session_id[:8], exc_info=True
        )


def recover_workspaces(workspace_store, session_id: str) -> None:
    """Rebuild in-memory workspace state for a session from Neo4j.

    1. Populates the global catalog cache (all :Workspace names).
    2. Looks up this session's LAST_USED_WORKSPACE relationship.
    3. If found, atomically sets _active[session_id] via try_recover().
    """
    executor = _get_executor()
    if executor is None:
        return

    # Step 1: Load the global catalog (bounded)
    try:
        catalog_result = executor.execute_read(
            CATALOG_QUERY,
            {"limit": 100},
        )
    except Exception:
        logger.warning(
            "Workspace recovery Neo4j read failed for session %s",
            session_id[:8], exc_info=True
        )
        return

    # Filter out None names; ensure 'default' is always in the catalog
    catalog = [r["name"] for r in catalog_result if r.get("name")]
    if "default" not in catalog:
        catalog.append("default")
    workspace_store.set_catalog_cache(catalog)

    # Step 2: Find the session's last-used workspace
    try:
        last_used_result = executor.execute_read(
            """
            MATCH (s:Session {id: $session_id})-[:LAST_USED_WORKSPACE]->(w:Workspace)
            RETURN w.workspace_id AS name
            ORDER BY w.last_active_at DESC
            LIMIT 1
            """,
            {"session_id": session_id},
        )
    except Exception:
        # If this fails, session stays at default
        return

    if not last_used_result:
        return  # Cold session with no prior record

    active = last_used_result[0]["name"]
    if not active or not isinstance(active, str):
        return

    # Step 3: Atomic check-and-set via try_recover()
    if workspace_store.try_recover(session_id, active, catalog):
        logger.info(
            "Recovered %d workspaces for session %s, active=%s",
            len(catalog), session_id[:8], active
        )


def cleanup_neo4j_sessions(session_ids: list[str]) -> None:
    """Batch-delete :Session nodes and their :LAST_USED_WORKSPACE edges.

    MUST be called OUTSIDE AgentRegistry._lock and active_workspace_lock.
    Processes in CLEANUP_BATCH_SIZE chunks to bound transaction size.

    IMPORTANT: does NOT touch :Workspace nodes — they are global and may be
    used by other sessions. Only the per-session :Session + edge are removed.
    """
    if not session_ids:
        return

    executor = _get_executor()
    if executor is None:
        return

    for i in range(0, len(session_ids), CLEANUP_BATCH_SIZE):
        batch = session_ids[i:i + CLEANUP_BATCH_SIZE]
        try:
            executor.execute_write(
                """
                UNWIND $session_ids AS sid
                MATCH (s:Session {id: sid})
                OPTIONAL MATCH (s)-[r:LAST_USED_WORKSPACE]->(:Workspace)
                DELETE r, s
                """,
                {"session_ids": batch},
            )
        except Exception:
            logger.warning(
                "Neo4j session cleanup failed for batch starting at index %d (%d sessions)",
                i, len(batch), exc_info=True
            )
        else:
            logger.info(
                "Cleaned up %d :Session nodes (batch %d/%d)",
                len(batch), i // CLEANUP_BATCH_SIZE + 1,
                (len(session_ids) + CLEANUP_BATCH_SIZE - 1) // CLEANUP_BATCH_SIZE
            )


def get_global_catalog(executor=None, limit: int = 100) -> list[str]:
    """Return all :Workspace names sorted by last_active_at DESC.

    Used by the in-memory catalog cache. Filters out compound IDs (from old
    per-session data that hasn't been fully migrated).
    """
    if executor is None:
        executor = _get_executor()
    if executor is None:
        return ["default"]

    try:
        result = executor.execute_read(CATALOG_QUERY, {"limit": limit})
    except Exception:
        logger.warning("get_global_catalog failed", exc_info=True)
        return ["default"]

    names = [r["name"] for r in result if r.get("name")]
    if "default" not in names:
        names.append("default")
    return names


def create_persist_callback():
    """Return a persist callback, or None if Neo4j is unavailable.

    The callback signature is (workspace_name, session_id) — bare name + audit session.
    Fire-and-forget; failures are logged, never raised.
    """
    if _get_executor() is None:
        return None
    return _persist_workspace


CLEANUP_QUERY = """
MATCH (w:Workspace)
WHERE w.workspace_id <> $default
  AND NOT w.workspace_id IN $protect
  AND NOT EXISTS { MATCH (t:Task {workspace_id: w.workspace_id}) WHERE t.deleted_at IS NULL }
  AND NOT EXISTS { MATCH (b:Bug {workspace_id: w.workspace_id}) }
DETACH DELETE w
RETURN count(w) AS deleted
"""


def cleanup_empty_workspaces(protect: list[str]) -> tuple[int, list[str]]:
    """Delete :Workspace nodes that have no live tasks and no bugs.

    A workspace is eligible for deletion when:
      - it is not 'default'
      - it is not in `protect` (caller passes the active workspace + any
        explicitly-preserved names)
      - no :Task with deleted_at IS NULL references it
      - no :Bug references it

    :Session nodes and :LAST_USED_WORKSPACE edges are removed via
    DETACH DELETE. Returns (deleted_count, deleted_names). On Neo4j
    unavailability or error, returns (0, []).
    """
    executor = _get_executor()
    if executor is None:
        return 0, []

    protect_clean = [n for n in protect if n]
    try:
        names_result = executor.execute_read(
            """
            MATCH (w:Workspace)
            WHERE w.workspace_id <> $default
              AND NOT w.workspace_id IN $protect
              AND NOT EXISTS { MATCH (t:Task {workspace_id: w.workspace_id}) WHERE t.deleted_at IS NULL }
              AND NOT EXISTS { MATCH (b:Bug {workspace_id: w.workspace_id}) }
            RETURN w.workspace_id AS name
            """,
            {"default": DEFAULT_WORKSPACE, "protect": protect_clean or [""]},
        )
        names = [r["name"] for r in names_result if r.get("name")]
    except Exception:
        logger.warning("cleanup_empty_workspaces: eligibility read failed", exc_info=True)
        return 0, []

    if not names:
        return 0, []

    try:
        result = executor.execute_write(
            CLEANUP_QUERY,
            {"default": DEFAULT_WORKSPACE, "protect": protect_clean or [""]},
        )
        deleted = result[0]["deleted"] if result and "deleted" in result[0] else 0
    except Exception:
        logger.warning("cleanup_empty_workspaces: delete failed", exc_info=True)
        return 0, []

    return int(deleted), names


def create_recovery_fn(workspace_store):
    """Create a recovery function bound to the workspace store."""
    return partial(recover_workspaces, workspace_store)
