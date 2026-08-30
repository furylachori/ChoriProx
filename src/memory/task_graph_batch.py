"""Batch operations for task creation and status updates."""
import logging
import os
import uuid
from typing import Any, Optional

from memory.query_executor import InfrastructureError, ConstraintViolation
from memory.task_graph_common import AmbiguousHashError, MAX_NAME_LEN, MAX_SUMMARY_LEN, MAX_TAG_LEN

logger = logging.getLogger(__name__)


def _normalize_key(name: str, session_id: str) -> str:
    return f"{name.lower().strip()}::{session_id}"


def _fallback_errors(count: int, message: str) -> list:
    """Generate fallback error dicts for remaining tasks in a failed batch."""
    return [{"error": "batch_aborted", "message": message} for _ in range(count)]


class TaskGraphBatchMixin:
    """Batch operations for task creation and status updates."""

    def create_tasks_batch_atomic(self, task_list: list, workspace_id: Optional[str] = None) -> list:
        """Create multiple tasks in a single atomic Neo4j transaction.

        Unlike create_tasks_batch (which calls create_task individually),
        this method opens one session and runs one write transaction.
        If any task fails validation, the entire batch is rolled back.

        Each task dict MAY include an `assign_to` field (string, agent_id, or None)
        per SPEC line 374-397 for pre-assignment at creation time. None leaves
        the task unassigned.

        Returns a list of result dicts (one per input task, in order).
        On success: {"short_hash": ..., "name": ..., "priority": ..., "status": ...}
        On error:   {"error": "...", "message": "..."}
        """
        if not task_list:
            return []

        if not workspace_id or not workspace_id.strip():
            return [{"error": "invalid_workspace", "message": f"Invalid workspace for task {t.get('name', 'unknown')}", "short_hash": ""} for t in task_list]

        batch = []
        seen: set[str] = set()
        dup_indices: list[int] = []
        non_dup_indices: list[int] = []
        for i, t in enumerate(task_list):
            name = t.get("name", "")
            priority = t.get("priority", "medium")
            summary = t.get("summary", "")
            tag = t.get("tag", "")
            blocked_by = t.get("blocked_by")
            parent_uuid = t.get("parent_uuid")
            session_id = t.get("session_id", "")
            status = t.get("status", "pending")
            assign_to = t.get("assign_to")  # SPEC line 374-397: optional pre-assignment

            key = _normalize_key(name, session_id)
            if key in seen:
                dup_indices.append(i)
                continue
            seen.add(key)
            non_dup_indices.append(i)

            name_len = len(name) if name else 0
            summary_len = len(summary) if summary else 0

            if name_len > MAX_NAME_LEN:
                return [
                    {"error": "invalid_input", "message": f"Task {i}: name exceeds {MAX_NAME_LEN} chars"},
                    *_fallback_errors(len(task_list) - 1, f"Batch aborted: task {i} name too long"),
                ]
            if summary_len > MAX_SUMMARY_LEN:
                return [
                    {"error": "invalid_input", "message": f"Task {i}: summary exceeds {MAX_SUMMARY_LEN} chars"},
                    *_fallback_errors(len(task_list) - 1, f"Batch aborted: task {i} summary too long"),
                ]
            if tag and len(tag) > MAX_TAG_LEN:
                return [
                    {"error": "invalid_input", "message": f"Task {i}: tag exceeds {MAX_TAG_LEN} chars"},
                    *_fallback_errors(len(task_list) - 1, f"Batch aborted: task {i} tag too long"),
                ]
            if priority not in ("high", "medium", "low"):
                return [
                    {"error": "invalid_input", "message": f"Task {i}: invalid priority '{priority}'"},
                    *_fallback_errors(len(task_list) - 1, f"Batch aborted: task {i} invalid priority"),
                ]

            batch.append({
                "name": name,
                "priority": priority,
                "summary": summary,
                "session_id": session_id,
                "status": status,
                "tag": tag,
                "blocked_by": blocked_by,
                "parent_uuid": parent_uuid,
                "assign_to": assign_to,
            })

        pre_generated_uuids = [f"entity-{uuid.uuid4().hex}" for _ in batch]

        try:
            result = self._executor.execute_transaction_no_retry(
                lambda tx: self._atomic_create_tasks(tx, batch, workspace_id, pre_generated_uuids)
            )
            results = [None] * len(task_list)
            for dup_i in dup_indices:
                results[dup_i] = {"error": "duplicate_name", "message": f"Duplicate task name in this session"}
            for j, res in enumerate(result):
                results[non_dup_indices[j]] = res
            return results
        except ConstraintViolation:
            return [
                {"error": "duplicate_name", "message": "Duplicate task name"}
                for _ in task_list
            ]
        except InfrastructureError as e:
            logger.exception("[TASK_GRAPH] Atomic batch create failed: %s", e)
            return [
                {"error": "batch_failed", "message": "Atomic batch creation failed due to an internal error"}
                for _ in task_list
            ]

    def _atomic_create_tasks(self, tx, batch, workspace_id: Optional[str] = None, pre_generated_uuids: list[str] = None):
        """Execute a batch of task creations in a single write transaction.

        Args:
            tx: Neo4j managed transaction.
            batch: List of task dicts with keys: name, priority, summary, session_id,
                   status, tag, blocked_by, parent_uuid.
            workspace_id: Optional workspace ID to associate with all tasks in the batch.
            pre_generated_uuids: Pre-generated UUIDs for each task in the batch.

        Returns:
            List of result dicts.
        """
        from proxy.utils import _now_iso

        if not batch:
            return []

        chunk_size = 100
        all_results = []
        all_blocked_by_rels = []
        all_parent_rels = []

        for i in range(0, len(batch), chunk_size):
            chunk = batch[i:i + chunk_size]
            chunk_uuids = pre_generated_uuids[i:i + chunk_size]
            
            assert len(chunk_uuids) == len(chunk), "UUID list length mismatch"
            
            now = _now_iso()
            
            result = tx.run(
                """UNWIND $tasks AS task
                   MERGE (t:Entity:Task {uuid: task.uuid})
                   ON CREATE SET
                     t.short_hash = task.short_hash,
                     t.name = task.name,
                     t.session_id = task.session_id,
                     t.priority = task.priority,
                     t.status = task.status,
                     t.summary = task.summary,
                     t.tag = task.tag,
                     t.assigned_agent_id = task.assigned_agent_id,
                     t.workspace_id = $workspace_id,
                     t.created_at = $created_at,
                     t.deleted_at = $deleted_at,
                     t.status_changed_at = $status_changed_at
                   ON MATCH SET
                     t.name = task.name,
                     t.priority = task.priority,
                     t.summary = task.summary
                   RETURN t.uuid AS uuid, t.short_hash AS short_hash, t.name AS name,
                          t.priority AS priority, t.status AS status,
                          t.assigned_agent_id AS assigned_agent_id""",
                {
                    "tasks": [
                        {
                            "uuid": task_uuid,
                            "short_hash": task_uuid[-8:],
                            "name": task_data["name"],
                            "session_id": task_data["session_id"],
                            "priority": task_data.get("priority", "medium"),
                            "status": task_data.get("status", "pending"),
                            "summary": task_data.get("summary", ""),
                            "tag": task_data.get("tag", ""),
                            "assigned_agent_id": task_data.get("assign_to"),
                        }
                        for task_uuid, task_data in zip(chunk_uuids, chunk)
                    ],
                    "workspace_id": workspace_id,
                    "created_at": now,
                    "deleted_at": None,
                    "status_changed_at": now
                }
            )
            
            for record in result:
                all_results.append({
                    "uuid": record["uuid"],
                    "short_hash": record["short_hash"],
                    "name": record["name"],
                    "priority": record["priority"],
                    "status": record["status"],
                })

        for idx, task_data in enumerate(batch):
            full_uuid = pre_generated_uuids[idx]
            blocked_by = task_data["blocked_by"]
            parent_uuid = task_data["parent_uuid"]

            if blocked_by:
                for blocker_uuid in blocked_by:
                    all_blocked_by_rels.append({
                        "child_uuid": full_uuid,
                        "blocker_uuid": blocker_uuid,
                    })

            if parent_uuid:
                all_parent_rels.append({
                    "child_uuid": full_uuid,
                    "parent_uuid": parent_uuid,
                })

        if all_blocked_by_rels:
            now = _now_iso()
            tx.run(
                """UNWIND $rels AS rel
                   MATCH (child:Task {uuid: rel.child_uuid})
                   MATCH (blocker:Task {uuid: rel.blocker_uuid})
                   MERGE (child)-[r:BLOCKED_BY]->(blocker)
                   ON CREATE SET r.created_at = $now""",
                {"rels": all_blocked_by_rels, "now": now}
            )

        if all_parent_rels:
            now = _now_iso()
            tx.run(
                """UNWIND $rels AS rel
                   MATCH (child:Task {uuid: rel.child_uuid})
                   MATCH (parent:Task {uuid: rel.parent_uuid})
                   MERGE (child)-[r:HAS_PARENT]->(parent)
                   ON CREATE SET r.created_at = $now""",
                {"rels": all_parent_rels, "now": now}
            )

        return all_results

    def update_tasks_batch_status(
        self,
        hashes: list[str],
        new_status: str,
        workspace_id: Optional[str] = None,
    ) -> list[dict]:
        """Batch status update for N tasks. Exactly 2 Neo4j queries regardless of N.

        Transaction model: best-effort partial update. Ambiguity is a hard failure
        (raises AmbiguousHashError). All other per-hash failures are reported in
        results. Concurrent-writer TOCTOU is a known limitation.

        Args:
            hashes: List of short hashes (exactly 8 hex chars). Caller MUST deduplicate.
            new_status: Target status (e.g., "complete", "invalid").
            workspace_id: Optional workspace scope. Strongly recommended for write
                operations. If None during a mutation, returns per-hash no_workspace
                errors. None = global scope (read-only queries only).

        Returns:
            list of dicts, one per input hash in input order:
              - success:                 {"hash": "...", "ok": True,  "status": new_status,     "changed": True}
              - no-op:                   {"hash": "...", "ok": True,  "status": current_status, "changed": False}
              - not_found:               {"hash": "...", "ok": False, "status": "not_found",    "changed": None}
              - invalid_transition:      {"hash": "...", "ok": False, "status": current_status,
                                         "changed": None, "error_reason": "invalid_transition"}
              - concurrent_modification: {"hash": "...", "ok": False, "status": current_status,
                                         "changed": None, "error_reason": "concurrent_modification"}
              - no_workspace:            {"hash": "...", "ok": False, "status": "error",
                                         "changed": None, "error_reason": "no_workspace"}
              - db_error:                {"hash": "...", "ok": False, "status": "error",
                                         "changed": None, "error_reason": "db_error"}

        Raises:
            AmbiguousHashError: One or more hashes match multiple tasks.
                Contains `count` attribute with total ambiguous matches.
                The batch update is NOT applied when ambiguity is detected —
                round-trip 2 (the write query) is never executed.
        """
        MAX_BATCH_SIZE = int(os.environ.get("TASK_BATCH_MAX_SIZE", "100"))
        if len(hashes) > MAX_BATCH_SIZE:
            return [{"ok": False, "changed": False, "error": f"Batch size {len(hashes)} exceeds maximum {MAX_BATCH_SIZE}"}]

        if len(hashes) > 50:
            logger.warning(
                "update_tasks_batch_status: batch size %d may be slow "
                "(ENDS WITH cannot use indexes; consider passing full UUIDs)",
                len(hashes),
            )

        hashes = [h.strip().lower() for h in hashes if h and h.strip()]
        if not hashes:
            return []

        if not workspace_id or not workspace_id.strip():
            return [
                {"hash": h, "ok": False, "status": "error",
                 "changed": None, "error_reason": "no_workspace"}
                for h in hashes
            ]

        # v5: workspaces are global. bare_name == workspace_id.
        workspace_where = "AND t.workspace_id = $workspace_id"

        from proxy.utils import _now_iso

        def _batch_status_tx(tx):
            validate_query = f"""
            UNWIND $hashes AS h
            WITH DISTINCT h
            MATCH (t:Task)
            WHERE t.short_hash = h
              AND t.deleted_at IS NULL
              {workspace_where}
            WITH h AS hash, collect({{status: t.status, uuid: t.uuid}}) AS matches
            RETURN
              hash,
              [m IN matches WHERE m.status IS NOT NULL | m.status][0] AS current_status,
              size(matches) AS match_count
            """
            params: dict[str, Any] = {"hashes": hashes}
            if workspace_id:
                params["workspace_id"] = workspace_id
            result = tx.run(validate_query, params)
            records = list(result)
            result.consume()

            total_ambiguous = 0
            validate_map: dict[str, dict] = {}
            for rec in records:
                h = rec["hash"]
                cnt = rec["match_count"]
                if cnt > 1:
                    total_ambiguous += cnt
                validate_map[h] = {
                    "current_status": rec["current_status"],
                    "match_count": cnt,
                }

            if total_ambiguous > 0:
                raise AmbiguousHashError(total_ambiguous)

            now = _now_iso()
            tasks_payload = []
            for i, h in enumerate(hashes):
                v = validate_map.get(h)
                if v is None or v["match_count"] == 0:
                    continue
                current = v["current_status"]
                if current == new_status:
                    continue
                allowed_targets = self.VALID_TRANSITIONS.get(current, set())
                if new_status not in allowed_targets:
                    continue
                tasks_payload.append({
                    "hash": h,
                    "expected_status": current,
                    "new_status": new_status,
                    "now": now,
                    "input_index": i,
                })

            updated_map = {}
            if tasks_payload:
                update_query = f"""
    UNWIND $tasks AS task
                 WITH task ORDER BY task.input_index
                 WITH collect(task) AS ordered_tasks
                 UNWIND ordered_tasks AS task
                MATCH (t:Task)
                WHERE t.short_hash = task.hash
                  AND t.status = task.expected_status
                  AND t.deleted_at IS NULL
                  {workspace_where}
                SET t.status = task.new_status, t.status_changed_at = task.now
                RETURN task.hash AS hash, count(t) AS updated
                """
                update_params = {"tasks": tasks_payload}
                if workspace_id:
                    update_params["workspace_id"] = workspace_id
                result2 = tx.run(update_query, update_params)
                update_records = list(result2)
                result2.consume()
                updated_map = {r["hash"]: r["updated"] for r in update_records}

            stale_hashes = []
            for h in hashes:
                v = validate_map.get(h)
                if v is None or v["match_count"] == 0:
                    continue
                current = v["current_status"]
                if current is None or current == new_status:
                    continue
                allowed_targets = self.VALID_TRANSITIONS.get(current, set())
                if new_status not in allowed_targets:
                    continue
                was_updated = updated_map.get(h, 0) > 0
                if not was_updated:
                    stale_hashes.append(h)

            fresh_status_map = {}
            if stale_hashes:
                reread_query = f"""
                UNWIND $hashes AS h
                MATCH (t:Task)
                WHERE t.short_hash = h
                  AND t.deleted_at IS NULL
                  {workspace_where}
                RETURN t.short_hash AS hash, t.status AS status
                """
                reread_params: dict[str, Any] = {"hashes": stale_hashes}
                if workspace_id:
                    reread_params["workspace_id"] = workspace_id
                reread_result = tx.run(reread_query, reread_params)
                for rec in reread_result:
                    fresh_status_map[rec["hash"]] = rec["status"]
                reread_result.consume()

            results = []
            for h in hashes:
                v = validate_map.get(h)
                if v is None or v["match_count"] == 0:
                    results.append({
                        "hash": h, "ok": False, "status": "not_found", "changed": None,
                    })
                    continue
                current = v["current_status"]
                if current is None:
                    results.append({
                        "hash": h, "ok": False, "status": "error",
                        "changed": None, "error_reason": "invalid_data",
                    })
                    continue
                if current == new_status:
                    results.append({
                        "hash": h, "ok": True, "status": current, "changed": False,
                    })
                    continue
                allowed_targets = self.VALID_TRANSITIONS.get(current, set())
                if new_status not in allowed_targets:
                    results.append({
                        "hash": h, "ok": False, "status": current,
                        "changed": None, "error_reason": "invalid_transition",
                    })
                    continue
                was_updated = updated_map.get(h, 0) > 0
                if was_updated:
                    results.append({
                        "hash": h, "ok": True, "status": new_status, "changed": True,
                    })
                else:
                    fresh_status = fresh_status_map.get(h, current)
                    results.append({
                        "hash": h, "ok": False, "status": fresh_status,
                        "changed": None, "error_reason": "concurrent_modification",
                    })
            return results

        try:
            return self._executor.execute_transaction(_batch_status_tx)
        except AmbiguousHashError:
            raise
        except InfrastructureError:
            raise
        except Exception as e:
            logger.exception("update_tasks_batch_status: batch update failed for %d hashes, workspace=%s",
                             len(hashes), workspace_id)
            return [
                {"hash": h, "ok": False, "status": "error",
                 "changed": None, "error_reason": "db_error"}
                for h in hashes
            ]
