import logging
import re
import unicodedata
from typing import Any

from memory.task_graph import AmbiguousHashError
from memory.query_executor import InfrastructureError
from memory.task_graph_common import MAX_SUMMARY_LEN
from neo4j.exceptions import TransientError, ServiceUnavailable, SessionExpired

from proxy.tool_pipeline.action_envelope import error_envelope, partial_envelope, success_envelope
from proxy.tool_pipeline.actions._errors import log_and_emit_infra_error

logger = logging.getLogger("proxy.tool_pipeline.actions")

TOOL = "proxy_ticket"
OPERATION = "create_tasks"


class CreateTasksAction:
    """Batch-create tasks with full validation."""

    @staticmethod
    def execute(arguments: dict, session_id: str, tg: Any, workspace_id: str = None, caller_agent_id: str = None) -> dict:
        tasks = arguments.get("tasks", [])
        known_tasks = arguments.get("known_tasks")

        if not isinstance(tasks, list) or not tasks:
            return error_envelope(
                TOOL, OPERATION, "invalid_input",
                "create_tasks requires: tasks (non-empty array of {name (required, \u2264200 chars), priority (required: high/medium/low)})",
                session_id,
                field="tasks",
            )

        if len(tasks) > 100:
            return error_envelope(
                TOOL, OPERATION, "batch_too_large",
                f"Maximum 100 tasks per create_tasks call, got {len(tasks)}. Split into multiple batches.",
                session_id,
            )

        valid_priorities = {"high", "medium", "low"}
        batch = []
        task_refs = []
        for i, t in enumerate(tasks):
            if not isinstance(t, dict):
                return error_envelope(
                    TOOL, OPERATION, "invalid_input",
                    f"Task at index {i} must be an object",
                    session_id,
                    field="tasks",
                )
            name = t.get("name", "")
            priority = t.get("priority", "medium")
            summary = t.get("summary", "")
            tag = t.get("tag")
            blocked_by_hashes = t.get("blocked_by")
            parent_hash_val = t.get("parent_hash")
            assign_to = t.get("assign_to")  # SPEC line 374-397: optional pre-assignment

            if tag and not re.match(r'^[A-Z]$', tag.strip()):
                return error_envelope(
                    TOOL, OPERATION, "invalid_input",
                    f"Task '{name}': tag must be a single uppercase letter (A-Z)",
                    session_id,
                    field="tag",
                )

            if not isinstance(name, str) or not name.strip():
                return error_envelope(
                    TOOL, OPERATION, "invalid_input",
                    f"Task at index {i}: name is required and must be a non-empty string",
                    session_id,
                    field="name",
                )

            if blocked_by_hashes is not None and not isinstance(blocked_by_hashes, list):
                return error_envelope(
                    TOOL, OPERATION, "invalid_input",
                    f"Task '{name}': blocked_by must be an array",
                    session_id,
                    field="blocked_by",
                )

            normalized_name = unicodedata.normalize("NFKC", name.strip())
            if len(normalized_name) > 200:
                return error_envelope(
                    TOOL, OPERATION, "invalid_input",
                    f"Task '{name}': name must be ≤200 characters",
                    session_id,
                    field="name",
                )

            if normalized_name.startswith("__batch_name:"):
                return error_envelope(
                    TOOL, OPERATION, "invalid_input",
                    f"Task '{name}': name starts with reserved prefix '__batch_name:'",
                    session_id,
                    field="name",
                )

            if re.match(r'^(?:task|step|subtask|sub-task|item|todo)\s+\d+$', normalized_name, re.IGNORECASE):
                return error_envelope(
                    TOOL, OPERATION, "generic_task_name",
                    f"Task '{name}': name is a generic placeholder. Use descriptive names derived from the actual work.",
                    session_id,
                    field="name",
                )

            if priority not in valid_priorities:
                return error_envelope(
                    TOOL, OPERATION, "invalid_input",
                    f"Task '{name}': priority must be high/medium/low, got '{priority}'",
                    session_id,
                    field="priority",
                )
            if summary and len(summary) > MAX_SUMMARY_LEN:
                return error_envelope(
                    TOOL, OPERATION, "invalid_input",
                    f"Task '{name}': summary exceeds {MAX_SUMMARY_LEN} characters",
                    session_id,
                    field="summary",
                )
            if tag and len(tag) > 50:
                return error_envelope(
                    TOOL, OPERATION, "invalid_input",
                    f"Task '{name}': tag exceeds 50 characters",
                    session_id,
                    field="tag",
                )
            # SPEC line 395-396: Non-None assign_to must pass agent_registry validation.
            # None entries skip assignment (task stays unassigned).
            if assign_to is not None:
                if not isinstance(assign_to, str) or not assign_to.strip():
                    return error_envelope(
                        TOOL, OPERATION, "invalid_input",
                        f"Task '{name}': assign_to must be a non-empty string or null",
                        session_id,
                        field="assign_to",
                    )
                try:
                    from proxy.agent.agent_registry import registry as _agent_registry
                    _target = _agent_registry.get_or_load(assign_to, caller_session_id=None)
                    if _target is None:
                        return error_envelope(
                            TOOL, OPERATION, "assign_to_agent_not_found",
                            f"Task '{name}': assign_to agent '{assign_to}' not found in registry",
                            session_id,
                            field="assign_to",
                        )
                except InfrastructureError:
                    raise
                except Exception as e:
                    return error_envelope(
                        TOOL, OPERATION, "assign_to_lookup_failed",
                        f"Task '{name}': assign_to registry lookup failed for '{assign_to}': {e}",
                        session_id,
                        field="assign_to",
                    )

            task_dict = {
                "session_id": session_id or "",
                "name": name.strip(),
                "priority": priority,
                "summary": summary if summary else "",
                "tag": tag.strip() if tag else "",
                "assign_to": assign_to,
            }
            batch.append(task_dict)
            task_refs.append({
                "blocked_by": blocked_by_hashes,
                "parent_hash": parent_hash_val,
            })

        name_to_index = {}
        for i, task_dict in enumerate(batch):
            normalized = unicodedata.normalize("NFKC", task_dict["name"]).lower().strip()
            if normalized in name_to_index:
                return error_envelope(
                    TOOL, OPERATION, "duplicate_name",
                    f"Duplicate task name in batch: '{task_dict['name']}'. Task names must be unique within a batch.",
                    session_id,
                    field="name",
                )
            name_to_index[normalized] = i

        known_tasks_map = {}
        if known_tasks:
            for kt in known_tasks:
                kt_normalized = unicodedata.normalize("NFKC", kt["name"]).lower().strip()
                known_tasks_map[kt_normalized] = kt["short_hash"]

        deferred_wiring = []
        db_blocked_by = {}

        for i, refs in enumerate(task_refs):
            bb = refs.get("blocked_by")
            if bb is not None and isinstance(bb, list):
                db_refs = []
                for ref in bb:
                    ref_normalized = unicodedata.normalize("NFKC", str(ref)).lower().strip()
                    if ref_normalized in name_to_index:
                        dep_idx = name_to_index[ref_normalized]
                        if dep_idx >= i:
                            return error_envelope(
                                TOOL, OPERATION, "ordering_violation",
                                f"blocked_by reference '{ref}' (index {dep_idx}) must appear before task at index {i}",
                                session_id,
                                field="blocked_by",
                            )
                        deferred_wiring.append((i, "blocked_by", batch[dep_idx]["name"]))
                    elif ref_normalized in known_tasks_map:
                        try:
                            resolved = tg.resolve_parent_hash(
                                known_tasks_map[ref_normalized],
                                workspace_id=workspace_id
                            )
                            db_refs.append(resolved)
                        except ValueError as e:
                            return error_envelope(
                                TOOL, OPERATION, "invalid_input",
                                f"Task '{batch[i]['name']}': blocked_by '{ref}' {e}",
                                session_id,
                                field="blocked_by",
                            )
                        except (TransientError, ServiceUnavailable, SessionExpired) as e:
                            return log_and_emit_infra_error(logger, TOOL, OPERATION, e, session_id)
                    else:
                        try:
                            resolved = tg.resolve_parent_hash(str(ref), workspace_id=workspace_id)
                            db_refs.append(resolved)
                        except ValueError as e:
                            return error_envelope(
                                TOOL, OPERATION, "invalid_input",
                                f"Task '{batch[i]['name']}': blocked_by '{ref}' {e}",
                                session_id,
                                field="blocked_by",
                            )
                        except (TransientError, ServiceUnavailable, SessionExpired) as e:
                            return log_and_emit_infra_error(logger, TOOL, OPERATION, e, session_id)
                if db_refs:
                    db_blocked_by[i] = db_refs

            ph = refs.get("parent_hash")
            if ph is not None:
                ph_normalized = unicodedata.normalize("NFKC", str(ph)).lower().strip()
                if ph_normalized in name_to_index:
                    dep_idx = name_to_index[ph_normalized]
                    if dep_idx >= i:
                        return error_envelope(
                            TOOL, OPERATION, "ordering_violation",
                            f"parent_hash reference '{ph}' (index {dep_idx}) must appear before task at index {i}",
                            session_id,
                            field="parent_hash",
                        )
                    deferred_wiring.append((i, "parent_hash", batch[dep_idx]["name"]))
                elif ph_normalized in known_tasks_map:
                    try:
                        resolved_parent = tg.resolve_parent_hash(known_tasks_map[ph_normalized], workspace_id=workspace_id)
                        batch[i]["parent_uuid"] = resolved_parent
                    except ValueError as e:
                        return error_envelope(
                            TOOL, OPERATION, "invalid_input",
                            f"Task '{batch[i]['name']}': {e}",
                            session_id,
                            field="parent_hash",
                        )
                    except (TransientError, ServiceUnavailable, SessionExpired) as e:
                        return log_and_emit_infra_error(logger, TOOL, OPERATION, e, session_id)
                else:
                    try:
                        resolved = tg.resolve_parent_hash(ph, workspace_id=workspace_id)
                        batch[i]["parent_uuid"] = resolved
                    except ValueError as e:
                        return error_envelope(
                            TOOL, OPERATION, "invalid_input",
                            f"Task '{batch[i]['name']}': parent_hash '{ph}' {e}",
                            session_id,
                            field="parent_hash",
                        )
                    except (TransientError, ServiceUnavailable, SessionExpired) as e:
                        return log_and_emit_infra_error(logger, TOOL, OPERATION, e, session_id)

        if known_tasks:
            known_hashes = [kt.get("short_hash", "") for kt in known_tasks if kt.get("short_hash")]
            try:
                known_map = tg.get_tasks_by_short_hashes(known_hashes, workspace_id=workspace_id)
            except AmbiguousHashError as e:
                return error_envelope(
                    TOOL, OPERATION, "invalid_known_tasks",
                    f"known_tasks contains ambiguous hash: {e}",
                    session_id,
                )
            except (ValueError, KeyError) as e:
                return error_envelope(
                    TOOL, OPERATION, "invalid_known_tasks",
                    f"known_tasks lookup failed: {e}",
                    session_id,
                )
            except (TransientError, ServiceUnavailable, SessionExpired) as e:
                return log_and_emit_infra_error(logger, TOOL, OPERATION, e, session_id)
            for kt in known_tasks:
                kt_hash = kt.get("short_hash", "")
                kt_name = kt.get("name", "")
                if kt_hash and kt_hash not in known_map:
                    return error_envelope(
                        TOOL, OPERATION, "invalid_known_tasks",
                        f"known_tasks entry '{kt_name}' maps to non-existent hash '{kt_hash}'",
                        session_id,
                    )

        for i, db_blockers in db_blocked_by.items():
            batch[i]["blocked_by"] = db_blockers

        results = tg.create_tasks_batch(batch, workspace_id=workspace_id)

        if session_id:
            try:
                from proxy.tool_pipeline.tool_call_tracker import ToolCallTracker
                tracker = ToolCallTracker.get_instance(session_id)
                tracker.register_created_hashes([r["short_hash"] for r in results if "short_hash" in r])
            except Exception as e:
                logger.warning(f"Failed to register created hashes for provenance: {e}")

        name_to_uuid = {}
        for r in results:
            if "uuid" in r and "name" in r:
                normalized = unicodedata.normalize("NFKC", r["name"]).lower().strip()
                if normalized in name_to_uuid:
                    logger.warning(f"NFKC collision in create_tasks results: '{r['name']}' normalizes to same key as existing '{name_to_uuid[normalized]}'")
                    continue
                name_to_uuid[normalized] = r["uuid"]

        wiring_errors = []
        batch_blocked_by = {}

        for task_idx, field, target_name in deferred_wiring:
            target_normalized = unicodedata.normalize("NFKC", target_name).lower().strip()
            target_uuid = name_to_uuid.get(target_normalized)
            if not target_uuid:
                wiring_errors.append({
                    "task_index": task_idx,
                    "field": field,
                    "target_name": target_name,
                    "error": "target not found in batch results",
                })
                continue
            task_short_hash = results[task_idx].get("short_hash") if task_idx < len(results) else None
            if not task_short_hash:
                wiring_errors.append({
                    "task_index": task_idx,
                    "field": field,
                    "target_name": target_name,
                    "error": "task not found in batch results",
                })
                continue

            if field == "blocked_by":
                batch_blocked_by.setdefault(task_idx, []).append(target_uuid)
            elif field == "parent_hash":
                try:
                    success = tg.update_task(
                        task_short_hash,
                        workspace_id=workspace_id,
                        parent_uuid=target_uuid,
                        caller_agent_id=caller_agent_id,
                    )
                    if not success:
                        wiring_errors.append({
                            "task_index": task_idx,
                            "field": field,
                            "target_name": target_name,
                            "error": "parent wiring update failed",
                        })
                        logger.warning("Deferred parent wiring failed for task index %s", task_idx)
                except InfrastructureError:
                    raise
                except Exception as e:
                    logger.error("Deferred parent wiring exception for task index %s: %s", task_idx, e, exc_info=True)
                    wiring_errors.append({
                        "task_index": task_idx,
                        "field": field,
                        "target_name": target_name,
                        "error": "internal wiring failure",
                    })

        for task_idx, batch_blockers in batch_blocked_by.items():
            task_short_hash = results[task_idx].get("short_hash") if task_idx < len(results) else None
            if not task_short_hash:
                wiring_errors.append({
                    "task_index": task_idx,
                    "field": "blocked_by",
                    "target_name": "",
                    "error": "task not found in batch results",
                })
                continue

            all_blockers = list(db_blocked_by.get(task_idx, [])) + batch_blockers
            try:
                success = tg.update_task(
                    task_short_hash,
                    workspace_id=workspace_id,
                    blocked_by=all_blockers,
                    caller_agent_id=caller_agent_id,
                )
                if not success:
                    wiring_errors.append({
                        "task_index": task_idx,
                        "field": "blocked_by",
                        "target_name": "",
                        "error": "blocked_by wiring update failed",
                    })
                    logger.warning("Deferred blocked_by wiring failed for task index %s", task_idx)
            except InfrastructureError:
                raise
            except Exception as e:
                logger.error("Deferred blocked_by wiring exception for task index %s: %s", task_idx, e, exc_info=True)
                wiring_errors.append({
                    "task_index": task_idx,
                    "field": "blocked_by",
                    "target_name": "",
                    "error": "internal wiring failure",
                })

        created = []
        errors = []
        for i, r in enumerate(results):
            task = batch[i]
            if "error" in r:
                errors.append(
                    {
                        "name": task["name"],
                        "error": r["error"],
                        "message": r.get("message", ""),
                    }
                )
            else:
                created.append(
                    {
                        "short_hash": r.get("short_hash"),
                        "name": r.get("name"),
                        "priority": r.get("priority"),
                        "status": r.get("status"),
                    }
                )

        total_errors = len(errors) + len(wiring_errors)
        data = {
            "created": created,
            "errors": errors,
            "wiring_errors": wiring_errors,
            "total_created": len(created),
            "total_errors": total_errors,
        }
        if total_errors > 0:
            return partial_envelope(TOOL, OPERATION, data, session_id)
        return success_envelope(TOOL, OPERATION, data, session_id)
