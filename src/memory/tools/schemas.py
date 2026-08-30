"""Tool schema definitions for memory tools."""
import logging

logger = logging.getLogger("memory.tools")

MEMORY_TOOL_DEFINITIONS = []
"""Tool schema definitions for memory tools (DISABLED placeholder).

Originally populated by commit cdf040e (Apr 2026) with 7 schemas.
Emptied by commit 8d838185 (May 2026) when memory tools were disabled.
The consumer at src/proxy/tool_cache.py:110-118 still iterates this list
but finds nothing (the loop is currently a no-op). Phase 4 will re-populate
this with real schemas via the tool action registry (see
docs/specs/consolidation-research/phase-4-tool-action-registry.md:116-118).

KEEP THIS EMPTY LIST — do not delete. Deletion contradicts the
feature-disabled pattern and the Phase 4 spec. See
docs/specs/consolidation-research/18-phase-3-final.md §4.1 (external
review confirmed intent to populate later).
"""


PROXY_TICKET_LIST_TASKS_DEFINITION = {
    "type": "function",
    "function": {
        "name": "proxy_ticket_list_tasks",
        "description": "List current tasks. Returns short_hash, name, priority, status, summary, tag, blocked_by, is_blocked, blocked_count, parent_short_hash, parent_name.",
        "parameters": {
            "type": "object",
            "properties": {
                "status_filter": {
                    "type": "array",
                    "items": {"type": "string", "enum": ["pending", "in_progress", "complete", "invalid"]},
                    "description": "Filter by status values. Default: ['pending', 'in_progress']."
                }
            },
            "additionalProperties": False
        }
    }
}

PROXY_TICKET_GET_TASK_DEFINITION = {
    "type": "function",
    "function": {
        "name": "proxy_ticket_get_task",
        "description": "Fetch the full task record by short_hash (uuid, name, status, priority, summary, tag, planning_info, created_at, assigned_agent_id, blockers, is_blocked, blocked_count, parent_short_hash, parent_name, review_notes).",
        "parameters": {
            "type": "object",
            "properties": {
                "short_hash": {
                    "type": "string",
                    "description": "The short hash of the target task (e.g. 'a1b2c3d4')."
                }
            },
            "required": ["short_hash"],
            "additionalProperties": False
        }
    }
}

PROXY_TICKET_CREATE_TASKS_DEFINITION = {
    "type": "function",
    "function": {
        "name": "proxy_ticket_create_tasks",
        "description": "Create one or more tasks. Maximum 100 tasks per call. Split into sequential batches of ≤100.",
        "parameters": {
            "type": "object",
            "properties": {
                "tasks": {
                    "type": "array",
                    "description": "Array of task objects to create.",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string", "maxLength": 200, "description": "Short, descriptive task name (max 200 chars). Must be unique per session."},
                            "priority": {"type": "string", "enum": ["high", "medium", "low"], "description": "Task priority."},
                            "summary": {"type": "string", "maxLength": 5000, "description": "1-2 sentence description (max 5000 chars)."},
                            "tag": {"type": "string", "pattern": "^[A-Z]$", "description": "Parallelism group label (A, B, C...)."},
                            "blocked_by": {"type": "array", "items": {"type": "string", "minLength": 1}, "description": "Short hashes of blocking tasks."},
                            "parent_hash": {"type": "string", "minLength": 1, "description": "Short hash of parent task."},
                            "assign_to": {"type": ["string", "null"], "description": "Optional agent ID to assign the task to."}
                        },
                        "required": ["name", "priority"]
                    }
                },
                "known_tasks": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string", "description": "Task name from list_tasks result"},
                            "short_hash": {"type": "string", "description": "Corresponding short_hash"}
                        },
                        "required": ["name", "short_hash"]
                    },
                    "description": "Optional. Subset of list_tasks results for dedup-skipped reference resolution."
                }
            },
            "required": ["tasks"],
            "additionalProperties": False
        }
    }
}

PROXY_TICKET_UPDATE_TASK_DEFINITION = {
    "type": "function",
    "function": {
        "name": "proxy_ticket_update_task",
        "description": "Change a task's status. Prefer over edit for status-only changes.",
        "parameters": {
            "type": "object",
            "properties": {
                "short_hash": {"type": "string", "description": "The short hash of the target task."},
                "new_status": {"type": "string", "enum": ["pending", "in_progress", "complete", "invalid"], "description": "The new status for the task."}
            },
            "required": ["short_hash", "new_status"],
            "additionalProperties": False
        }
    }
}

PROXY_TICKET_EDIT_DEFINITION = {
    "type": "function",
    "function": {
        "name": "proxy_ticket_edit",
        "description": "Update task fields. blocked_by REPLACES all existing blockers. For selective removal, pass the remaining blockers.",
        "parameters": {
            "type": "object",
            "properties": {
                "short_hash": {"type": "string", "description": "The short hash of the target task."},
                "name": {"type": "string", "maxLength": 200, "description": "New name for the task."},
                "summary": {"type": "string", "maxLength": 5000, "description": "New summary for the task."},
                "priority": {"type": "string", "enum": ["high", "medium", "low"], "description": "New priority for the task."},
                "blocked_by": {"type": "array", "items": {"type": "string", "minLength": 1}, "description": "New blocked_by list (short hashes). Replaces all existing blockers."},
                "parent_hash": {"type": "string", "minLength": 1, "description": "Short hash of parent task."},
                "new_status": {"type": "string", "enum": ["pending", "in_progress", "complete", "invalid"], "description": "New status."},
                "tag": {"type": "string", "pattern": "^[A-Z]$", "description": "New tag (A, B, C...)."},
                "planning_info": {"type": "string", "maxLength": 8000, "description": "Planning context (max 8000 chars)."}
            },
            "required": ["short_hash"],
            "additionalProperties": False
        }
    }
}

PROXY_TICKET_SET_PARENT_DEFINITION = {
    "type": "function",
    "function": {
        "name": "proxy_ticket_set_parent",
        "description": "Assign a parent task to establish a parent/child relationship. Cannot be self-referencing.",
        "parameters": {
            "type": "object",
            "properties": {
                "short_hash": {"type": "string", "description": "Short hash of the child task."},
                "parent_hash": {"type": "string", "minLength": 1, "description": "Short hash of the parent task."}
            },
            "required": ["short_hash", "parent_hash"],
            "additionalProperties": False
        }
    }
}

PROXY_TICKET_CLEAR_PARENT_DEFINITION = {
    "type": "function",
    "function": {
        "name": "proxy_ticket_clear_parent",
        "description": "Remove the parent relationship from a task.",
        "parameters": {
            "type": "object",
            "properties": {
                "short_hash": {"type": "string", "description": "The short hash of the target task."}
            },
            "required": ["short_hash"],
            "additionalProperties": False
        }
    }
}

PROXY_TICKET_CLEAR_BLOCKED_BY_DEFINITION = {
    "type": "function",
    "function": {
        "name": "proxy_ticket_clear_blocked_by",
        "description": "Remove ALL blockers from a task. For selective removal, use edit(blocked_by=[remaining...]).",
        "parameters": {
            "type": "object",
            "properties": {
                "short_hash": {"type": "string", "description": "The short hash of the target task."}
            },
            "required": ["short_hash"],
            "additionalProperties": False
        }
    }
}

PROXY_TICKET_SET_PRIORITY_DEFINITION = {
    "type": "function",
    "function": {
        "name": "proxy_ticket_set_priority",
        "description": "Change a task's priority. Synonyms: urgent/critical → high, normal/standard → medium, deferrable/nice-to-have → low.",
        "parameters": {
            "type": "object",
            "properties": {
                "short_hash": {"type": "string", "description": "The short hash of the target task."},
                "priority": {"type": "string", "enum": ["high", "medium", "low"], "description": "New priority."}
            },
            "required": ["short_hash", "priority"],
            "additionalProperties": False
        }
    }
}

PROXY_TICKET_APPEND_REVIEW_NOTE_DEFINITION = {
    "type": "function",
    "function": {
        "name": "proxy_ticket_append_review_note",
        "description": "Add a review note to a task.",
        "parameters": {
            "type": "object",
            "properties": {
                "short_hash": {"type": "string", "description": "The short hash of the target task."},
                "note": {"type": "string", "maxLength": 500, "description": "A review note to append (max 500 chars)."}
            },
            "required": ["short_hash", "note"],
            "additionalProperties": False
        }
    }
}


TICKET_TOOL_SCHEMAS = {
    "proxy_ticket_list_tasks": PROXY_TICKET_LIST_TASKS_DEFINITION,
    "proxy_ticket_get_task": PROXY_TICKET_GET_TASK_DEFINITION,
    "proxy_ticket_create_tasks": PROXY_TICKET_CREATE_TASKS_DEFINITION,
    "proxy_ticket_update_task": PROXY_TICKET_UPDATE_TASK_DEFINITION,
    "proxy_ticket_edit": PROXY_TICKET_EDIT_DEFINITION,
    "proxy_ticket_set_parent": PROXY_TICKET_SET_PARENT_DEFINITION,
    "proxy_ticket_clear_parent": PROXY_TICKET_CLEAR_PARENT_DEFINITION,
    "proxy_ticket_clear_blocked_by": PROXY_TICKET_CLEAR_BLOCKED_BY_DEFINITION,
    "proxy_ticket_set_priority": PROXY_TICKET_SET_PRIORITY_DEFINITION,
    "proxy_ticket_append_review_note": PROXY_TICKET_APPEND_REVIEW_NOTE_DEFINITION,
}


PROXY_WORK_TOOL_DEFINITION = {
    "type": "function",
    "function": {
        "name": "proxy_work",
        "description": (
            "Execute work on a task. Use 'start_work' to claim and begin a session, "
            "'stop_work' to end the session, 'report_bug' to flag a bug, "
            "'review_bug' to submit a review, 'validate_bug' to synthesize reviews, "
            "'resolve_bug' to approve/dismiss a bug, or 'handoff_task' to transfer the task."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": [
                        "start_work", "stop_work", "report_bug", "review_bug",
                        "validate_bug", "resolve_bug", "handoff_task"
                    ],
                    "description": "The action to perform. One of: start_work, stop_work, report_bug, review_bug, validate_bug, resolve_bug, handoff_task."
                },
                "short_hash": {
                    "type": "string",
                    "description": "Required for start_work, stop_work, handoff_task. Short hash of the target task."
                },
                "ws_uuid": {
                    "type": "string",
                    "description": "Required for stop_work. UUID of the work session to close."
                },
                "transferred_from_agent": {
                    "type": "string",
                    "description": "Optional for start_work. Set by handoff to record the source agent."
                },
                "git_commit_sha": {
                    "type": "string",
                    "description": "Optional for stop_work. Git commit SHA at session end."
                },
                "new_status": {
                    "type": "string",
                    "enum": ["complete", "invalid"],
                    "description": "Optional for stop_work. If set, the task transitions to this status (must be 'complete' or 'invalid')."
                },
                "task_short_hash": {
                    "type": "string",
                    "description": "Required for report_bug. Short hash of the task the bug was found during."
                },
                "bug_uuid": {
                    "type": "string",
                    "description": "Required for review_bug, validate_bug, resolve_bug."
                },
                "title": {
                    "type": "string",
                    "maxLength": 200,
                    "description": "Required for report_bug. Title of the bug (max 200 chars)."
                },
                "description": {
                    "type": "string",
                    "maxLength": 5000,
                    "description": "Required for report_bug. Detailed description (max 5000 chars)."
                },
                "severity": {
                    "type": "string",
                    "enum": ["critical", "high", "medium", "low"],
                    "description": "Required for report_bug. Bug severity."
                },
                "confidence": {
                    "type": "number",
                    "minimum": 0.0,
                    "maximum": 1.0,
                    "description": "Required for review_bug. Reviewer confidence 0.0..1.0."
                },
                "verdict": {
                    "type": "string",
                    "enum": ["valid", "not_reproducible", "by_design", "already_fixed"],
                    "description": "Required for review_bug. Reviewer verdict."
                },
                "proposal": {
                    "type": "string",
                    "maxLength": 5000,
                    "description": "Required for review_bug. Suggested fix or reasoning (max 5000 chars)."
                },
                "resolution": {
                    "type": "string",
                    "enum": ["approved", "dismissed"],
                    "description": "Required for resolve_bug. Human resolution."
                },
                "reason": {
                    "type": "string",
                    "description": "Optional for resolve_bug. Reason for dismissal (or approval notes)."
                },
                "to_agent_id": {
                    "type": "string",
                    "description": "Required for handoff_task. Target agent_id."
                },
                "context": {
                    "type": "string",
                    "description": "Optional for handoff_task. Context for the receiving agent."
                }
            },
            "required": ["action"]
        }
    }
}

