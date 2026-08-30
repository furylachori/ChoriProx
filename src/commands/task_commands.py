from commands import register, make_command_response, CommandContext
from memory.task_graph import TaskGraph, AmbiguousHashError
from typing import Optional
import logging

logger = logging.getLogger(__name__)

def is_ticket_trigger(text: str) -> bool:
    """Returns True if text triggers ^^ticket planning."""
    t = text.strip()
    return t == "^^ticket" or t.startswith("^^ticket ")


def is_work_trigger(text: str) -> bool:
    """Returns True if text triggers ^^work work execution."""
    t = text.strip()
    return t == "^^work" or t.startswith("^^work ")


