"""US-017i: Imperative with Time-Based or Emotional Criteria.

WHEN the user says "kill anything pending > 2 weeks" or "close all tasks
that are frustrating" THEN the agent MUST interpret time-based or emotional
criteria and take action AND MUST NOT ask for clarification or require
structured input.
"""
import pytest
from unittest.mock import MagicMock, patch

from proxy.tool_pipeline.actions.create_tasks import CreateTasksAction
from skills.ticket_skill import TicketSkill
from proxy.skills.skill_registry import SkillContext

pytestmark = pytest.mark.unit


@pytest.fixture
def mock_tg():
    tg = MagicMock()
    tg.is_available.return_value = True
    tg.create_tasks_batch.return_value = [
        {
            "short_hash": "aabbccdd",
            "uuid": "entity-aabbccdd00000001",
            "name": "kill anything pending > 2 weeks",
            "priority": "medium",
            "status": "pending",
            "tag": "",
        },
    ]
    return tg


@pytest.fixture
def ticket_skill():
    return TicketSkill()


@pytest.fixture
def ctx():
    return SkillContext(
        session_id="test-sess-017i",
        agent_id="agent-017i",
        messages=[],
        is_subagent=False,
        active_skill_name="ticket",
        raw_user_text="",
    )


class TestCreateTasksAcceptsUnstructuredDescriptions:
    """US-017i: create_tasks must accept natural-language task descriptions."""

    def test_time_based_criteria_as_task_name(self, mock_tg):
        """A task name like 'kill anything pending > 2 weeks' must be accepted."""
        arguments = {
            "tasks": [
                {"name": "kill anything pending > 2 weeks", "priority": "high"},
            ]
        }
        result = CreateTasksAction.execute(arguments, "sid-017i", mock_tg)
        assert result.get("error") is None
        assert result["data"]["total_created"] == 1
        assert result["data"]["created"][0]["name"] == "kill anything pending > 2 weeks"

    def test_emotional_criteria_as_task_name(self, mock_tg):
        """A task name like 'close all tasks that are frustrating' must be accepted."""
        mock_tg.create_tasks_batch.return_value = [
            {
                "short_hash": "eeff0011",
                "uuid": "entity-eeff001100000001",
                "name": "close all tasks that are frustrating",
                "priority": "medium",
                "status": "pending",
                "tag": "",
            },
        ]
        arguments = {
            "tasks": [
                {"name": "close all tasks that are frustrating", "priority": "medium"},
            ]
        }
        result = CreateTasksAction.execute(arguments, "sid-017i", mock_tg)
        assert result.get("error") is None
        assert result["data"]["total_created"] == 1
        assert result["data"]["created"][0]["name"] == "close all tasks that are frustrating"

    def test_batch_of_unstructured_descriptions(self, mock_tg):
        """Multiple unstructured descriptions in one batch must all be accepted."""
        names = [
            "kill anything pending > 2 weeks",
            "close all tasks that are frustrating",
            "review items older than 30 days",
        ]
        mock_tg.create_tasks_batch.return_value = [
            {
                "short_hash": f"hash{i:04d}",
                "uuid": f"entity-hash{i:04d}00000001",
                "name": n,
                "priority": "medium",
                "status": "pending",
                "tag": "",
            }
            for i, n in enumerate(names)
        ]
        arguments = {
            "tasks": [{"name": n, "priority": "medium"} for n in names]
        }
        result = CreateTasksAction.execute(arguments, "sid-017i", mock_tg)
        assert result.get("error") is None
        assert result["data"]["total_created"] == 3


class TestTicketSkillParsesImperativeCommands:
    """US-017i: parse_subcommand must route natural-language to plan, not reject."""

    def test_time_based_command_routes_to_plan(self, ticket_skill):
        """'kill anything pending > 2 weeks' must parse as plan action."""
        action, arg = ticket_skill.parse_subcommand(
            "^^ticket kill anything pending > 2 weeks", "sess-017i"
        )
        assert action == "plan"
        assert "anything pending > 2 weeks" in arg

    def test_emotional_command_with_non_subcommand_verb(self, ticket_skill):
        """Natural-language commands not starting with a subcommand verb route to plan."""
        action, arg = ticket_skill.parse_subcommand(
            "^^ticket remove all tasks that are frustrating", "sess-017i"
        )
        assert action == "plan"
        assert "all tasks that are frustrating" in arg

    def test_close_keyword_recognized_as_done_subcommand(self, ticket_skill):
        """'close' is a recognized subcommand prefix — routes to done, not plan."""
        action, arg = ticket_skill.parse_subcommand(
            "^^ticket close abc12345", "sess-017i"
        )
        assert action == "done"
        assert arg == "abc12345"

    def test_handle_direct_returns_none_for_plan(self, ticket_skill, ctx):
        """handle_direct must return None for plan action (defers to activation)."""
        from proxy import state
        state.active_workspaces[ctx.session_id] = "test-ws"
        result = ticket_skill.handle_direct(ctx, "plan", "kill anything pending > 2 weeks")
        assert result is None
