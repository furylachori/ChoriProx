"""Action classes for proxy_ticket — extracted from the original monolithic handler.

Each file defines one action class that implements the TaskAction protocol
(execute(arguments, session_id, tg) -> dict).
"""
from .create_tasks import CreateTasksAction
from .update_task import UpdateTaskAction
from .list_tasks import ListTasksAction
from .get_task import GetTaskAction
from .edit import EditAction
from .set_parent import SetParentAction
from .clear_parent import ClearParentAction
from .clear_blocked_by import ClearBlockedByAction
from .set_priority import SetPriorityAction
from .append_review_note import AppendReviewNoteAction
from .start_work import StartWorkAction
from .stop_work import StopWorkAction
from .handoff_task import HandoffTaskAction
from .report_bug import ReportBugAction
from .review_bug import ReviewBugAction
from .validate_bug import ValidateBugAction
from .resolve_bug import ResolveBugAction

__all__ = [
    "CreateTasksAction",
    "UpdateTaskAction",
    "ListTasksAction",
    "GetTaskAction",
    "EditAction",
    "SetParentAction",
    "ClearParentAction",
    "ClearBlockedByAction",
    "SetPriorityAction",
    "AppendReviewNoteAction",
    "StartWorkAction",
    "StopWorkAction",
    "HandoffTaskAction",
    "ReportBugAction",
    "ReviewBugAction",
    "ValidateBugAction",
    "ResolveBugAction",
]
