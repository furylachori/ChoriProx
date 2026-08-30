"""US-017e: Context-Aware Task Decomposition from Conversation.

WHEN the user says "turn what we discussed into tasks" or "break this down
into steps" THEN the agent MUST use the conversation context to create
meaningful, specific tasks AND MUST NOT create generic placeholder tasks.

Testable contract:
1. The context pipeline preserves recent conversation messages so the LLM
   can derive meaningful task names from them.
2. CreateTasksAction rejects generic placeholder task names ("Task 1",
   "Step 2") as a guardrail against low-quality decomposition.
"""
import pytest
from unittest.mock import patch, MagicMock

pytestmark = pytest.mark.unit


def _make_conversation(topic="auth"):
    """Build a realistic conversation about a specific topic."""
    if topic == "auth":
        return [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "I need to add OAuth2 login to my app. It should support Google and GitHub providers."},
            {"role": "assistant", "content": "Sure, I can help with that. We'll need to set up OAuth2 client credentials, implement the callback handler, and add session management."},
            {"role": "user", "content": "Also need a forgot-password flow with email verification tokens."},
            {"role": "assistant", "content": "Good point. We'll add a password reset request endpoint, token generation, and a reset confirmation page."},
        ]
    return [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Break down the deployment pipeline into tasks."},
        {"role": "assistant", "content": "We need CI/CD setup, Docker image builds, staging deployment, and production rollout."},
    ]


class TestContextPreservesConversation:
    """US-017e: The context pipeline must preserve recent conversation
    messages so the LLM has content to derive meaningful task names from."""

    @patch("proxy.vfs._build_vfs_context", return_value=None)
    @patch("proxy.vfs._build_vfs_inherited_context", return_value=None)
    @patch("memory.db.get_rich_summaries", return_value=[])
    @patch("memory.db.get_ancient_history", return_value="")
    def test_recent_conversation_messages_preserved_in_context(
        self, _ancient, _summaries, _vfs_sub, _vfs_main
    ):
        """Recent user/assistant messages about OAuth must survive context selection."""
        from proxy.context import select_context

        messages = _make_conversation("auth")
        result = select_context("test-session", messages, model_name="default")

        roles = [m.get("role") for m in result]
        assert "user" in roles
        assert "assistant" in roles

        contents = " ".join(m.get("content", "") for m in result if m.get("role") != "system")
        assert "OAuth2" in contents or "oauth" in contents.lower()
        assert "password" in contents.lower()

    @patch("proxy.vfs._build_vfs_context", return_value=None)
    @patch("proxy.vfs._build_vfs_inherited_context", return_value=None)
    @patch("memory.db.get_rich_summaries", return_value=[])
    @patch("memory.db.get_ancient_history", return_value="")
    def test_system_prompt_present_alongside_conversation(
        self, _ancient, _summaries, _vfs_sub, _vfs_main
    ):
        """The system prompt must be in the output alongside conversation messages."""
        from proxy.context import select_context

        messages = _make_conversation("auth")
        result = select_context("test-session", messages, model_name="default")

        system_msgs = [m for m in result if m.get("role") == "system"]
        assert len(system_msgs) >= 1
        assert len(result) >= 2


class TestCreateTasksRejectsGenericNames:
    """US-017e: CreateTasksAction must reject generic placeholder task names
    that indicate the LLM did not use conversation context for decomposition."""

    def _make_tg_mock(self):
        tg = MagicMock()
        tg.create_tasks_batch.return_value = [
            {"short_hash": "aabbccdd", "name": t["name"], "priority": t.get("priority", "medium"), "status": "pending"}
            for t in []
        ]
        return tg

    def test_rejects_generic_task_n(self):
        """'Task 1', 'Task 2' are generic placeholders — must be rejected."""
        from proxy.tool_pipeline.actions.create_tasks import CreateTasksAction

        tg = MagicMock()
        result = CreateTasksAction.execute(
            {"tasks": [
                {"name": "Task 1", "priority": "high"},
                {"name": "Task 2", "priority": "medium"},
            ]},
            session_id="test-session",
            tg=tg,
            workspace_id="ws-1",
        )

        assert "error" in result
        assert "generic" in result.get("error", {}).get("message", "").lower() or "placeholder" in result.get("error", {}).get("message", "").lower()

    def test_rejects_generic_step_n(self):
        """'Step 1', 'Step 2' are generic placeholders — must be rejected."""
        from proxy.tool_pipeline.actions.create_tasks import CreateTasksAction

        tg = MagicMock()
        result = CreateTasksAction.execute(
            {"tasks": [
                {"name": "Step 1", "priority": "high"},
                {"name": "Step 2", "priority": "medium"},
            ]},
            session_id="test-session",
            tg=tg,
            workspace_id="ws-1",
        )

        assert "error" in result

    def test_accepts_meaningful_task_names(self):
        """Task names derived from conversation context must be accepted."""
        from proxy.tool_pipeline.actions.create_tasks import CreateTasksAction

        tg = MagicMock()
        tg.create_tasks_batch.return_value = [
            {"short_hash": "aabb1122", "name": "Implement OAuth2 callback handler", "priority": "high", "status": "pending"},
            {"short_hash": "ccdd3344", "name": "Add password reset email flow", "priority": "medium", "status": "pending"},
        ]

        result = CreateTasksAction.execute(
            {"tasks": [
                {"name": "Implement OAuth2 callback handler", "priority": "high"},
                {"name": "Add password reset email flow", "priority": "medium"},
            ]},
            session_id="test-session",
            tg=tg,
            workspace_id="ws-1",
        )

        assert result.get("error") is None
        assert result["data"]["total_created"] == 2
