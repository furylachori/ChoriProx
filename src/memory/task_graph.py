"""
Neo4j task graph operations with workspace-scoped access control.
All query and mutation methods enforce workspace_id filtering when provided.
"""
import logging
import uuid
from typing import Any, Optional

from proxy.utils import _now_iso

from memory.graph import Neo4jClient
from memory.query_executor import QueryExecutor, InfrastructureError, ConstraintViolation, QueryTimeoutError
from memory.task_graph_batch import TaskGraphBatchMixin
from memory.task_graph_queries import TaskGraphQueryMixin
from memory.task_graph_common import (
    AmbiguousHashError, MAX_NAME_LEN, MAX_SUMMARY_LEN, MAX_TAG_LEN,
    MAX_REVIEW_NOTE_LEN, MAX_REVIEW_NOTES_COUNT, SHORT_HASH_RE
)

import threading
logger = logging.getLogger(__name__)


_constraint_ensure_lock = threading.Lock()


class TaskGraph(TaskGraphBatchMixin, TaskGraphQueryMixin):
    """Neo4j task graph operations.

    Note: Neo4jClient uses the neo4j driver's built-in connection pooling.
    Each TaskGraph() call is lightweight — the driver handles session reuse.
    """

    VALID_TRANSITIONS = {
        "pending": {"in_progress", "complete", "invalid"},
        "in_progress": {"complete", "pending"},
        "complete": {"invalid"},
        "invalid": {"pending"},
    }

    SHORT_HASH_RE = SHORT_HASH_RE

    _constraint_ensured = False

    def __init__(self):
        self._neo4j = Neo4jClient()
        self._executor = self._neo4j._executor
        
        if not TaskGraph._constraint_ensured:
            with _constraint_ensure_lock:
                if not TaskGraph._constraint_ensured:
                    self._ensure_workspace_constraint()
                    TaskGraph._constraint_ensured = True

    def _ensure_workspace_constraint(self):
        """Add database-level uniqueness constraint for task name + workspace."""
        try:
            self._executor.execute_write(
                "CREATE CONSTRAINT task_name_workspace_unique IF NOT EXISTS "
                "FOR (t:Task) REQUIRE (t.name, t.workspace_id) IS UNIQUE"
            )
        except ConstraintViolation as e:
            logger.debug("Workspace constraint skipped (already exists): %s", e)
        except InfrastructureError as e:
            logger.warning("Failed to create workspace constraint: %s", e)

    def close(self) -> None:
        """No-op. Shared driver is closed by close_shared_driver() at shutdown."""
        pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False

    def is_available(self) -> bool:
        return self._neo4j._is_available()

    def get_store_status(self) -> dict:
        """Return store availability info for list_tasks response."""
        available = self.is_available()
        return {
            "available": available,
            "status": "connected" if available else "unavailable",
        }

    def create_task(
        self,
        session_id: str,
        name: str,
        priority: str = "medium",
        summary: str = "",
        tag: str = "",
        blocked_by: Optional[list[str]] = None,
        parent_uuid: Optional[str] = None,
        status: str = "pending",
        workspace_id: Optional[str] = None,
    ) -> dict:
        name_len = len(name) if name else 0
        summary_len = len(summary) if summary else 0
        if name_len > MAX_NAME_LEN or summary_len > MAX_SUMMARY_LEN:
            return {"error": "invalid_input"}

        if tag and len(tag) > MAX_TAG_LEN:
            return {"error": "tag_too_long"}

        if not workspace_id or not workspace_id.strip():
            return {"error": "invalid_workspace", "ok": False}

        full_uuid = f"entity-{uuid.uuid4().hex}"
        short_hash = full_uuid[-8:]
        now = _now_iso()

        def _create_task_tx(tx):
            dedup_query = """
            MATCH (t:Task {name: $name, workspace_id: $workspace_id})
            WHERE t.deleted_at IS NULL
            RETURN t.uuid
            """
            dedup_result = tx.run(dedup_query, {"name": name, "workspace_id": workspace_id})
            if dedup_result.single() is not None:
                return {"error": "duplicate_name"}

            create_query = """
            CREATE (t:Entity:Task {
                uuid: $uuid,
                short_hash: $short_hash,
                name: $name,
                session_id: $session_id,
                priority: $priority,
                summary: $summary,
                status: $status,
                tag: $tag,
                created_at: $created_at,
                deleted_at: $deleted_at,
                status_changed_at: $status_changed_at,
                started_at: $started_at,
                completed_at: $completed_at,
                workspace_id: $workspace_id
            })
            """
            params = {
                "uuid": full_uuid,
                "short_hash": short_hash,
                "name": name,
                "session_id": session_id,
                "priority": priority,
                "summary": summary,
                "status": status,
                "tag": tag,
                "created_at": now,
                "deleted_at": None,
                "status_changed_at": now,
                "started_at": None,
                "completed_at": None,
                "workspace_id": workspace_id,
            }
            tx.run(create_query, params)

            if blocked_by:
                blocked_by = list(set(blocked_by))
                if full_uuid in blocked_by:
                    raise ValueError("Task cannot block itself")
                
                result = tx.run(
                    """UNWIND $blocker_uuids AS blocker_uuid
                       MATCH (blocker:Task {uuid: blocker_uuid, workspace_id: $workspace_id})
                       RETURN blocker.uuid AS uuid""",
                    {"blocker_uuids": blocked_by, "workspace_id": workspace_id}
                )
                existing_blockers = {record["uuid"] for record in result}
                missing = set(blocked_by) - existing_blockers
                if missing:
                    raise ValueError(f"Blockers not found: {missing}")
                
                tx.run(
                    """UNWIND $blocker_uuids AS blocker_uuid
                       MATCH (t:Task {uuid: $task_uuid})
                       MATCH (blocker:Task {uuid: blocker_uuid, workspace_id: $workspace_id})
                       MERGE (t)-[r:BLOCKED_BY]->(blocker)
                       ON CREATE SET r.created_at = $now""",
                    {"task_uuid": full_uuid, "blocker_uuids": blocked_by, "workspace_id": workspace_id, "now": now}
                )

            if parent_uuid:
                tx.run(
                    """MATCH (child:Task {uuid: $child_uuid})
                       MATCH (parent:Task {uuid: $parent_uuid})
                       MERGE (child)-[r:HAS_PARENT]->(parent)
                       ON CREATE SET r.created_at = $now""",
                    {"child_uuid": full_uuid, "parent_uuid": parent_uuid, "now": now}
                )

            return {
                "uuid": full_uuid,
                "short_hash": short_hash,
                "name": name,
                "priority": priority,
                "status": status,
            }

        try:
            return self._executor.execute_transaction_no_retry(_create_task_tx)
        except ConstraintViolation:
            return {"error": "duplicate_name"}
        except InfrastructureError as e:
            logger.exception("create_task failed for session %s: %s", session_id, e)
            return {"error": "store_unavailable"}

    def create_tasks_batch(self, tasks: list[dict], workspace_id: Optional[str] = None) -> list[dict]:
        return self.create_tasks_batch_atomic(tasks, workspace_id=workspace_id)

    def get_task_by_short_hash(self, short_hash: str, workspace_id: Optional[str] = None) -> Optional[dict]:
        """Look up a task by its short hash prefix."""
        short_hash = short_hash.strip().lower()
        if not self.SHORT_HASH_RE.match(short_hash):
            return None

        if workspace_id:
            # v5: workspaces are global
            workspace_where = "AND e.workspace_id = $workspace_id"
        else:
            workspace_where = ""

        query = f"""
        MATCH (e:Task)
        WHERE e.short_hash = $hash
          {workspace_where}
          AND e.deleted_at IS NULL
        RETURN properties(e) AS props
        """

        params = {"hash": short_hash}
        if workspace_id:
            params["workspace_id"] = workspace_id
        
        records = self._executor.execute_read(query, params)

        if len(records) > 1:
            raise AmbiguousHashError(len(records))

        if not records:
            return None

        return dict(records[0].get("props", {}))

    def get_tasks_by_short_hashes(self, hashes: list[str], workspace_id: Optional[str] = None) -> dict[str, dict]:
        """Batch lookup: returns {hash: task_dict} for all hashes in one query.

        Raises AmbiguousHashError if any hash matches multiple tasks.
        """
        if not hashes:
            return {}

        normalized = [h.strip().lower() for h in hashes if h and h.strip()]
        if not normalized:
            return {}

        if workspace_id:
            # v5: workspaces are global
            workspace_where = "AND e.workspace_id = $workspace_id"
        else:
            workspace_where = ""

        query = f"""
        UNWIND $hashes AS h
        MATCH (e:Task)
        WHERE e.short_hash = h
          AND e.deleted_at IS NULL
          {workspace_where}
        RETURN e.short_hash AS hash, properties(e) AS props
        """

        params: dict[str, Any] = {"hashes": normalized}
        if workspace_id:
            params["workspace_id"] = workspace_id
        
        records = self._executor.execute_read(query, params)

        from collections import defaultdict
        hash_groups: dict[str, list[dict]] = defaultdict(list)
        for rec in records:
            props = dict(rec.get("props", {}))
            h = rec["hash"]
            hash_groups[h].append(props)

        for h, group in hash_groups.items():
            if len(group) > 1:
                raise AmbiguousHashError(len(group))

        return {h: group[0] for h, group in hash_groups.items()}

    def update_task_status(
        self, short_hash: str, new_status: str, workspace_id: Optional[str] = None,
        caller_agent_id: Optional[str] = None,
    ) -> dict:
        """Return {"ok": bool, "status": str}. Idempotent: same-status = ok:True."""
        short_hash = short_hash.strip().lower() if short_hash else short_hash
        if not short_hash:
            return {"ok": False, "status": "not_found"}

        if not workspace_id or not workspace_id.strip():
            return {"ok": False, "status": "error", "detail": "workspace_id required for status update"}

        assignment_where = (
            "AND (t.assigned_agent_id IS NULL OR t.assigned_agent_id = $caller_agent_id)"
            if caller_agent_id
            else ""
        )

        # v5: workspaces are global
        workspace_where = "AND t.workspace_id = $workspace_id"
        count_query = f"""
        MATCH (t:Task)
        WHERE t.short_hash = $hash
          AND t.deleted_at IS NULL
          {workspace_where}
          {assignment_where}
        RETURN count(t) AS cnt, collect(t.status)[0] AS current_status
        """

        MAX_RETRIES = 3
        for attempt in range(MAX_RETRIES):
            try:
                params: dict[str, Any] = {"hash": short_hash}
                if workspace_id:
                    params["workspace_id"] = workspace_id
                if caller_agent_id:
                    params["caller_agent_id"] = caller_agent_id
                records = self._executor.execute_read(count_query, params, timeout=30)

                if not records:
                    return {"ok": False, "status": "error"}
                record = records[0]

                cnt = record.get("cnt", 0) if record else 0
                if cnt == 0:
                    return {"ok": False, "status": "not_found"}
                if cnt > 1:
                    raise AmbiguousHashError(cnt)

                current_status = record.get("current_status")
                if current_status == new_status:
                    return {"ok": True, "status": current_status, "changed": False}

                if current_status not in self.VALID_TRANSITIONS:
                    return {"ok": False, "status": current_status}
                allowed = self.VALID_TRANSITIONS.get(current_status, set())
                if new_status not in allowed:
                    return {"ok": False, "status": current_status}

                now = _now_iso()
                # SPEC US-086/US-087: bug transitions MUST be atomic with the
                # task status update. Merge them into a single query so the
                # bug→resolved/bug→dismissed transition happens in the same
                # transaction as the task status change.
                if new_status == "complete":
                    update_query = f"""
                    MATCH (t:Task)
                    WHERE t.short_hash = $hash
                      AND t.status = $expected_status
                      AND t.deleted_at IS NULL
                      {workspace_where}
                      {assignment_where}
                    SET t.status = $new_status,
                        t.status_changed_at = $now,
                        t.completed_at = $now
                    WITH t
                    OPTIONAL MATCH (b:Bug)-[:RESOLVED_BY]->(t)
                    SET b.status = 'resolved'
                    WITH t
                    RETURN count(t) AS updated
                    """
                elif new_status == "invalid":
                    update_query = f"""
                    MATCH (t:Task)
                    WHERE t.short_hash = $hash
                      AND t.status = $expected_status
                      AND t.deleted_at IS NULL
                      {workspace_where}
                      {assignment_where}
                    SET t.status = $new_status,
                        t.status_changed_at = $now,
                        t.completed_at = $now
                    WITH t
                    OPTIONAL MATCH (b:Bug)-[:RESOLVED_BY]->(t)
                    SET b.status = 'dismissed',
                        b.dismissal_reason = 'task_invalidated',
                        b.dismissed_at = datetime()
                    WITH t
                    RETURN count(t) AS updated
                    """
                else:
                    update_query = f"""
                    MATCH (t:Task)
                    WHERE t.short_hash = $hash
                      AND t.status = $expected_status
                      AND t.deleted_at IS NULL
                      {workspace_where}
                      {assignment_where}
                    SET t.status = $new_status, t.status_changed_at = $now
                    RETURN count(t) AS updated
                    """

                update_params: dict[str, Any] = {
                    "hash": short_hash, "new_status": new_status, "now": now, "expected_status": current_status,
                }
                if workspace_id:
                    update_params["workspace_id"] = workspace_id
                if caller_agent_id:
                    update_params["caller_agent_id"] = caller_agent_id
                update_records = self._executor.execute_write(update_query, update_params, timeout=10)

                if not update_records:
                    return {"ok": False, "status": "error"}
                update_record = update_records[0]

                updated = update_record.get("updated", 0) if update_record else 0
                if updated == 1:
                    return {"ok": True, "status": new_status}
                continue
            except AmbiguousHashError:
                raise
            except InfrastructureError:
                raise
            except ConstraintViolation as e:
                logger.warning("update_task_status constraint violation: %s", e)
                return {"ok": False, "status": "error", "error": "constraint_violation"}
            except QueryTimeoutError:
                raise
            except Exception as e:
                logger.warning("update_task_status failed: %s", e)
                return {"ok": False, "status": "error"}

        try:
            params: dict[str, Any] = {"hash": short_hash}
            if workspace_id:
                params["workspace_id"] = workspace_id
            if caller_agent_id:
                params["caller_agent_id"] = caller_agent_id
            records = self._executor.execute_read(count_query, params, timeout=30)
            if not records:
                return {"ok": False, "status": "error"}
            record = records[0]
            current_status = record.get("current_status") if record else None
            return {"ok": False, "status": current_status if current_status else "error"}
        except InfrastructureError:
            raise
        except Exception as e:
            logger.warning("update_task_status re-read failed: %s", e)
            return {"ok": False, "status": "error"}

    def update_task(
        self,
        short_hash: str,
        workspace_id: Optional[str] = None,
        name: Optional[str] = None,
        summary: Optional[str] = None,
        priority: Optional[str] = None,
        tag: Optional[str] = None,
        blocked_by: Optional[list[str]] = None,
        parent_uuid: Optional[str] = None,
        clear_parent: bool = False,
        planning_info: Optional[str] = None,
        new_status: Optional[str] = None,
        caller_agent_id: Optional[str] = None,
    ) -> bool:
        short_hash = short_hash.strip().lower() if short_hash else short_hash
        if not short_hash:
            return False

        if not workspace_id or not workspace_id.strip():
            logger.error("update_task called without workspace_id")
            return False

        if name is not None and len(name) > MAX_NAME_LEN:
            return False
        if summary is not None and len(summary) > MAX_SUMMARY_LEN:
            return False
        if tag is not None and len(tag) > MAX_TAG_LEN:
            return False

        if new_status is not None:
            result = self.update_task_status(short_hash, new_status, workspace_id=workspace_id, caller_agent_id=caller_agent_id)
            if not result.get("ok", False):
                return False
            has_other_updates = any(v is not None for v in [name, summary, priority, tag, blocked_by, parent_uuid, planning_info]) or clear_parent
            if not has_other_updates:
                return True

        # v5: workspaces are global
        workspace_where = "AND t.workspace_id = $workspace_id"

        assignment_where = (
            "AND (t.assigned_agent_id IS NULL OR t.assigned_agent_id = $caller_agent_id)"
            if caller_agent_id
            else ""
        )

        now = _now_iso()

        def _update_task_tx(tx):
            nonlocal blocked_by
            if name is not None:
                ambiguity_query = f"""
                MATCH (t:Task)
                WHERE t.short_hash = $hash
                  AND t.deleted_at IS NULL
                  {workspace_where}
                  {assignment_where}
                RETURN count(t) AS cnt
                """
                check_query = f"""
                MATCH (t:Task)
                WHERE t.name = $name
                  AND NOT t.short_hash = $hash
                  AND t.deleted_at IS NULL
                  {workspace_where}
                  {assignment_where}
                RETURN count(t) AS cnt
                """
                amb_params: dict[str, Any] = {"hash": short_hash}
                if workspace_id:
                    amb_params["workspace_id"] = workspace_id
                if caller_agent_id:
                    amb_params["caller_agent_id"] = caller_agent_id
                amb_records = tx.run(ambiguity_query, amb_params)
                amb_result = amb_records.single() if hasattr(amb_records, 'single') else list(amb_records)
                if hasattr(amb_records, 'single'):
                    amb_cnt = amb_result.get("cnt", 0) if amb_result else 0
                else:
                    amb_cnt = amb_result[0].get("cnt", 0) if amb_result else 0
                if amb_cnt == 0:
                    return False
                if amb_cnt > 1:
                    raise AmbiguousHashError(amb_cnt)

                dup_params: dict[str, Any] = {"name": name, "hash": short_hash}
                if workspace_id:
                    dup_params["workspace_id"] = workspace_id
                if caller_agent_id:
                    dup_params["caller_agent_id"] = caller_agent_id
                dup_records = tx.run(check_query, dup_params)
                dup_result = list(dup_records)
                cnt = dup_result[0].get("cnt", 0) if dup_result else 0
                if cnt > 0:
                    return False
            else:
                count_query = f"""
                MATCH (t:Task)
                WHERE t.short_hash = $hash
                  AND t.deleted_at IS NULL
                  {workspace_where}
                  {assignment_where}
                RETURN count(t) AS cnt
                LIMIT 2
                """
                else_count_params: dict[str, Any] = {"hash": short_hash}
                if workspace_id:
                    else_count_params["workspace_id"] = workspace_id
                if caller_agent_id:
                    else_count_params["caller_agent_id"] = caller_agent_id
                records = tx.run(count_query, else_count_params)
                record_list = list(records)
                cnt = record_list[0].get("cnt", 0) if record_list else 0
                if cnt == 0:
                    return False
                if cnt > 1:
                    raise AmbiguousHashError(cnt)

            set_clauses = []
            params: dict[str, Any] = {"hash": short_hash}
            if workspace_id:
                params["workspace_id"] = workspace_id
            if caller_agent_id:
                params["caller_agent_id"] = caller_agent_id

            if name is not None:
                set_clauses.append("t.name = $name")
                params["name"] = name
            if summary is not None:
                set_clauses.append("t.summary = $summary")
                params["summary"] = summary
            if priority is not None:
                set_clauses.append("t.priority = $priority")
                params["priority"] = priority
            if tag is not None:
                set_clauses.append("t.tag = $tag")
                params["tag"] = tag
            if planning_info is not None:
                set_clauses.append("t.planning_info = $planning_info")
                params["planning_info"] = planning_info

            if set_clauses:
                set_clause = ", ".join(set_clauses)
                query = f"""
                MATCH (t:Task {{short_hash: $hash}})
                WHERE t.deleted_at IS NULL
                  {workspace_where}
                  {assignment_where}
                SET {set_clause}
                RETURN count(t) AS updated
                """
                tx.run(query, params)

            if blocked_by is not None:
                blocked_by_params: dict[str, Any] = {"hash": short_hash, "workspace_id": workspace_id}
                if caller_agent_id:
                    blocked_by_params["caller_agent_id"] = caller_agent_id
                tx.run(
                    f"""MATCH (t:Task {{short_hash: $hash}})-[r:BLOCKED_BY]->()
                       WHERE t.deleted_at IS NULL
                         {workspace_where}
                         {assignment_where}
                       DELETE r""",
                    blocked_by_params
                )

                if blocked_by:
                    task_uuid = tx.run(
                        f"""MATCH (t:Task {{short_hash: $hash}})
                           WHERE t.deleted_at IS NULL
                             {workspace_where}
                             {assignment_where}
                           RETURN t.uuid AS uuid""",
                        blocked_by_params
                    ).single()["uuid"]

                    blocked_by = list(set(blocked_by))
                    if task_uuid in blocked_by:
                        raise ValueError("Task cannot block itself")

                    result = tx.run(
                        """UNWIND $blocker_uuids AS blocker_uuid
                           MATCH (blocker:Task {uuid: blocker_uuid, workspace_id: $workspace_id})
                           RETURN blocker.uuid AS uuid""",
                        {"blocker_uuids": blocked_by, "workspace_id": workspace_id}
                    )
                    existing_blockers = {record["uuid"] for record in result}
                    missing = set(blocked_by) - existing_blockers
                    if missing:
                        raise ValueError(f"Blockers not found: {missing}")

                    merge_blocked_by_params: dict[str, Any] = {
                        "hash": short_hash,
                        "blocker_uuids": blocked_by,
                        "workspace_id": workspace_id,
                        "now": now,
                    }
                    if caller_agent_id:
                        merge_blocked_by_params["caller_agent_id"] = caller_agent_id
                    tx.run(
                        f"""UNWIND $blocker_uuids AS blocker_uuid
                           MATCH (t:Task {{short_hash: $hash}})
                           WHERE t.deleted_at IS NULL
                             {workspace_where}
                             {assignment_where}
                           MATCH (blocker:Task {{uuid: blocker_uuid, workspace_id: $workspace_id}})
                           MERGE (t)-[r:BLOCKED_BY]->(blocker)
                           ON CREATE SET r.created_at = $now""",
                        merge_blocked_by_params
                    )

            if clear_parent:
                parent_params: dict[str, Any] = {"hash": short_hash, "workspace_id": workspace_id}
                if caller_agent_id:
                    parent_params["caller_agent_id"] = caller_agent_id
                tx.run(
                    f"""MATCH (t:Task {{short_hash: $hash}})-[r:HAS_PARENT]->()
                       WHERE t.deleted_at IS NULL
                         {workspace_where}
                         {assignment_where}
                       DELETE r""",
                    parent_params
                )
            elif parent_uuid is not None:
                parent_params: dict[str, Any] = {"hash": short_hash, "workspace_id": workspace_id}
                if caller_agent_id:
                    parent_params["caller_agent_id"] = caller_agent_id
                tx.run(
                    f"""MATCH (t:Task {{short_hash: $hash}})-[r:HAS_PARENT]->()
                       WHERE t.deleted_at IS NULL
                         {workspace_where}
                         {assignment_where}
                       DELETE r""",
                    parent_params
                )
                if parent_uuid:
                    merge_parent_params: dict[str, Any] = {
                        "hash": short_hash,
                        "parent_uuid": parent_uuid,
                        "now": now,
                        "workspace_id": workspace_id,
                    }
                    if caller_agent_id:
                        merge_parent_params["caller_agent_id"] = caller_agent_id
                    tx.run(
                        f"""MATCH (child:Task {{short_hash: $hash}})
                           WHERE child.deleted_at IS NULL
                            {workspace_where.replace('t.', 'child.')}
                            {assignment_where.replace('t.', 'child.')}
                           MATCH (parent:Task {{uuid: $parent_uuid}})
                           MERGE (child)-[r:HAS_PARENT]->(parent)
                           ON CREATE SET r.created_at = $now""",
                        merge_parent_params
                    )

            return True

        try:
            return self._executor.execute_transaction_no_retry(_update_task_tx)
        except AmbiguousHashError:
            raise
        except InfrastructureError:
            raise
        except ConstraintViolation as e:
            logger.warning("update_task constraint violation: %s", e)
            return False
        except QueryTimeoutError:
            raise
        except Exception as e:
            logger.warning("update_task failed: %s", e)
            return False

    def append_review_note(
        self,
        short_hash: str,
        note: str,
        session_id: str = "",
        workspace_id: Optional[str] = None,
    ) -> dict:
        """Append a timestamped review note to a task.

        Creates a (:ReviewNote) node connected via [:HAS_NOTE] relationship.
        Atomic operation — no read-modify-write, no pessimistic lock needed.
        Truncates to MAX_REVIEW_NOTES_COUNT (100) notes per task.

        Returns: {"ok": True, "note_count": int} or {"ok": False, "error": str}
        """
        short_hash = short_hash.strip().lower() if short_hash else short_hash
        if not short_hash:
            return {"ok": False, "error": "invalid_input"}
        if not workspace_id or not workspace_id.strip():
            return {"ok": False, "error": "invalid_workspace"}
        if not note or not note.strip():
            return {"ok": False, "error": "empty_note"}

        note_text = note.strip()[:MAX_REVIEW_NOTE_LEN]
        now = _now_iso()
        note_uuid = f"review-{uuid.uuid4().hex}"

        # v5: workspaces are global
        workspace_where = "AND t.workspace_id = $workspace_id"

        def _append_review_note_tx(tx):
            check_query = f"""
            MATCH (t:Task {{short_hash: $hash}})
            WHERE t.deleted_at IS NULL {workspace_where}
            RETURN count(t) AS c
            """
            check = tx.run(check_query, {"hash": short_hash, "workspace_id": workspace_id})
            c = list(check)[0].get("c", 0)
            if c == 0:
                return {"ok": False, "error": "not_found"}
            if c > 1:
                raise AmbiguousHashError(c)

            create_query = f"""
            MATCH (t:Task {{short_hash: $hash}})
            WHERE t.deleted_at IS NULL {workspace_where}
            CREATE (n:ReviewNote {{uuid: $uuid, ts: $ts, note: $note, author: $author}})
            CREATE (t)-[:HAS_NOTE {{created_at: $ts}}]->(n)
            RETURN count(n) AS created
            """
            result = tx.run(create_query, {
                "hash": short_hash, "uuid": note_uuid, "ts": now,
                "note": note_text, "author": session_id or "unknown",
                "workspace_id": workspace_id,
            })
            records = list(result)
            if not records or records[0].get("created", 0) != 1:
                return {"ok": False, "error": "write_failed"}

            truncate_query = f"""
            MATCH (t:Task {{short_hash: $hash}})-[:HAS_NOTE]->(n:ReviewNote)
            WHERE t.deleted_at IS NULL {workspace_where}
            WITH n ORDER BY n.ts DESC SKIP $max_count
            DETACH DELETE n
            RETURN count(*) AS deleted
            """
            tx.run(truncate_query, {
                "hash": short_hash, "max_count": MAX_REVIEW_NOTES_COUNT,
                "workspace_id": workspace_id,
            })

            count_query = f"""
            MATCH (t:Task {{short_hash: $hash}})-[:HAS_NOTE]->(n:ReviewNote)
            WHERE t.deleted_at IS NULL {workspace_where}
            RETURN count(n) AS note_count
            """
            count_result = tx.run(count_query, {
                "hash": short_hash, "workspace_id": workspace_id,
            })
            count_records = list(count_result)
            note_count = count_records[0].get("note_count", 0) if count_records else 0

            return {"ok": True, "note_count": note_count}

        try:
            return self._executor.execute_transaction_no_retry(_append_review_note_tx)
        except AmbiguousHashError:
            raise
        except InfrastructureError:
            raise
        except Exception as e:
            logger.warning("append_review_note failed: %s", e)
            return {"ok": False, "error": "internal_error"}

    def resolve_blocked_by_hashes(
        self,
        hashes: list[str],
        exclude_uuid: Optional[str] = None,
        workspace_id: Optional[str] = None,
    ) -> list[str]:
        """Resolve a list of short hashes to full UUIDs, validating each one.

        Raises ValueError for: task not found, ambiguous hash, self-blocking,
        or blocker that is closed (complete/invalid).
        """
        if not hashes:
            return []

        stripped = [h.strip() for h in hashes]
        for h in stripped:
            if not h:
                raise ValueError("blocker hash cannot be empty")

        try:
            task_map = self.get_tasks_by_short_hashes(stripped, workspace_id=workspace_id)
        except AmbiguousHashError as e:
            raise ValueError(f"Ambiguous hash: {e.count} tasks match") from e

        resolved_uuids: list[str] = []
        for h in stripped:
            task = task_map.get(h)
            if task is None:
                raise ValueError(f"Blocker not found: {h}")

            task_uuid = task.get("uuid", "")
            if exclude_uuid and task_uuid == exclude_uuid:
                raise ValueError("Task cannot block itself")

            task_status = task.get("status", "")
            if task_status in ("complete", "invalid"):
                raise ValueError("Closed tasks cannot be blockers")

            resolved_uuids.append(task_uuid)

        return resolved_uuids

    def resolve_parent_hash(
        self,
        parent_hash: str,
        child_uuid: Optional[str] = None,
        workspace_id: Optional[str] = None,
    ) -> str:
        """Resolve a parent short hash to its full UUID, validating it.

        Raises ValueError for: task not found, ambiguous hash, self-parenting,
        or parent that is closed (complete/invalid).
        """
        try:
            task = self.get_task_by_short_hash(parent_hash, workspace_id=workspace_id)
        except AmbiguousHashError as e:
            raise ValueError(f"Ambiguous hash: {e.count} tasks match") from e
        if task is None:
            raise ValueError(f"Parent not found: {parent_hash}")

        parent_uuid = task.get("uuid", "")
        if child_uuid and parent_uuid == child_uuid:
            raise ValueError("Task cannot be its own parent")

        parent_status = task.get("status", "")
        if parent_status in ("complete", "invalid"):
            raise ValueError("Closed tasks cannot be parents")

        if child_uuid:
            ancestor_query = """
            MATCH path = (proposed_parent:Task {uuid: $parent_uuid})-[:HAS_PARENT*1..20]->(ancestor:Task)
            WHERE ancestor.deleted_at IS NULL
              AND ancestor.uuid = $child_uuid
            RETURN count(path) AS cycle_count
            """
            result = self._executor.execute_read(ancestor_query, {
                "parent_uuid": parent_uuid,
                "child_uuid": child_uuid,
            })
            if result and result[0]["cycle_count"] > 0:
                raise ValueError("Parent assignment would create a cycle")

        return parent_uuid
