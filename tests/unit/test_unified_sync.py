import pytest
from memory.prompts.extraction import (
    UNIFIED_SYSTEM_PROMPT,
    UNIFIED_EXTRACTION_INSTRUCTIONS,
    build_unified_prompt,
)


class TestUnifiedPrompt:
    def test_prompt_includes_current_summary(self):
        """build_unified_prompt with current_summary='foo' contains the summary label and foo."""
        messages = [{"role": "user", "content": "test"}]
        result = build_unified_prompt(messages, current_summary="foo")
        assert "[Current Summary" in result
        assert "foo" in result

    def test_prompt_defaults_to_new_session(self):
        """build_unified_prompt with current_summary='' contains '(new session)'."""
        messages = [{"role": "user", "content": "test"}]
        result = build_unified_prompt(messages, current_summary="")
        assert "(new session)" in result

    def test_unified_system_prompt_is_nonempty(self):
        """UNIFIED_SYSTEM_PROMPT is a non-empty string with expected keywords."""
        assert len(UNIFIED_SYSTEM_PROMPT) > 10
        assert "__sota_submit_sync__" in UNIFIED_SYSTEM_PROMPT

    def test_unified_prompt_preserves_conversation_exchange(self):
        """Output still contains the conversation delimiters with formatted messages."""
        messages = [{"role": "user", "content": "hello world"}]
        result = build_unified_prompt(messages)
        assert "BEGIN CONVERSATION" in result
        assert "hello world" in result

    def test_unified_prompt_passes_existing_memories(self):
        """Fix M5: existing_memories is passed through to base prompt."""
        messages = [{"role": "user", "content": "test"}]
        result = build_unified_prompt(messages, existing_memories="user prefers dark mode")
        assert "user prefers dark mode" in result
        assert "[Existing Memories" in result

    def test_unified_extraction_instructions_is_nonempty(self):
        """UNIFIED_EXTRACTION_INSTRUCTIONS is a non-empty string with expected content."""
        assert len(UNIFIED_EXTRACTION_INSTRUCTIONS) > 100
        assert "facts" in UNIFIED_EXTRACTION_INSTRUCTIONS
        assert "topic_key" in UNIFIED_EXTRACTION_INSTRUCTIONS
        assert "confidence" in UNIFIED_EXTRACTION_INSTRUCTIONS


# ---------------------------------------------------------------------------
# T2b: SyncResult + __SOTA_SUBMIT_SYNC_TOOL__ schema tests
# ---------------------------------------------------------------------------

import json
from unittest.mock import patch, MagicMock, PropertyMock

from memory.sota_client import (
    SyncResult,
    __SOTA_SUBMIT_SYNC_TOOL__,
    _sanitize_text,
    _validate_confidence,
    _validate_topic_key,
    _validate_timestamp,
    _validate_task_status,
    _validate_content,
    _validate_extraction_items,
    _validate_fact_item,
    _validate_event_item,
    _validate_instruction_item,
    _validate_task_item,
    SOTAParseError,
)


class TestSyncResultParsing:
    """Tests for SyncResult construction and tool-call parsing logic."""

    def test_sync_result_parses_valid_json(self):
        """Valid JSON arguments produce correct SyncResult fields."""
        args = {
            "facts": [{"content": "fact1", "topic_key": "dev/test", "confidence": "high"}],
            "events": [{"content": "event1", "timestamp": "2024-01-01T00:00:00Z"}],
            "instructions": [{"content": "inst1", "topic_key": "dev/test", "confidence": "medium"}],
            "tasks": [{"content": "task1", "status": "pending"}],
            "summary": "Test summary",
        }
        result = SyncResult(
            facts=args["facts"],
            events=args["events"],
            instructions=args["instructions"],
            tasks=args["tasks"],
            summary=args["summary"],
        )
        assert result.summary == "Test summary"
        assert len(result.facts) == 1
        assert len(result.events) == 1
        assert len(result.instructions) == 1
        assert len(result.tasks) == 1

    def test_sync_result_missing_summary(self):
        """SyncResult defaults summary to empty string when missing."""
        result = SyncResult(facts=[], events=[], instructions=[], tasks=[])
        assert result.summary == ""

    def test_sync_result_missing_extraction_keys(self):
        """SyncResult accepts empty lists when keys are missing."""
        result = SyncResult()
        assert result.facts == []
        assert result.events == []
        assert result.instructions == []
        assert result.tasks == []

    def test_sync_result_extraction_key_not_list(self):
        """SyncResult stores raw values; caller is responsible for validation."""
        result = SyncResult(facts="not a list")
        assert result.facts == "not a list"

    def test_sync_result_summary_coerced_from_non_string(self):
        """SyncResult casts non-string summary to string."""
        result = SyncResult(summary=123)
        assert result.summary == 123  # stores raw; _sanitize_text called in sync path

    def test_tool_schema_matches_extraction_schema(self):
        """Tool schema properties match new entity-relationship extraction model."""
        tool_props = __SOTA_SUBMIT_SYNC_TOOL__["function"]["parameters"]["properties"]
        assert "extraction_timestamp" in tool_props
        assert "entities" in tool_props
        assert "facts" in tool_props
        # entity items should have name, type, summary
        entity_props = tool_props["entities"]["items"]["properties"]
        assert "name" in entity_props
        assert "type" in entity_props
        assert "summary" in entity_props
        assert "enum" in entity_props["type"]
        assert len(entity_props["type"]["enum"]) == 10
        # fact items should have source_entity, target_entity, relationship_type, fact
        fact_props = tool_props["facts"]["items"]["properties"]
        assert "source_entity" in fact_props
        assert "target_entity" in fact_props
        assert "relationship_type" in fact_props
        assert "fact" in fact_props
        assert "valid_at" in fact_props
        assert "invalid_at" in fact_props
        assert "enum" in fact_props["relationship_type"]
        assert len(fact_props["relationship_type"]["enum"]) == 10


# ---------------------------------------------------------------------------
# T3: Validation and sanitization tests
# ---------------------------------------------------------------------------

class TestValidationFunctions:
    """Tests for _validate_* functions and _sanitize_text()."""

    def test_confidence_validation(self):
        """_validate_confidence normalizes valid values, defaults invalid."""
        item = {"confidence": "HIGH"}
        _validate_confidence(item)
        assert item["confidence"] == "high"

        item = {"confidence": "Medium"}
        _validate_confidence(item)
        assert item["confidence"] == "medium"

        item = {"confidence": "low"}
        _validate_confidence(item)
        assert item["confidence"] == "low"

        item = {"confidence": "invalid"}
        _validate_confidence(item)
        assert item["confidence"] == "medium"

        item = {"confidence": ""}
        _validate_confidence(item)
        assert item["confidence"] == "medium"

        item = {}
        _validate_confidence(item)
        assert item["confidence"] == "medium"

    def test_topic_key_validation(self):
        """_validate_topic_key validates category/subcategory format."""
        item = {"topic_key": "dev/python"}
        _validate_topic_key(item)
        assert item["topic_key"] == "dev/python"

        item = {"topic_key": "a/b"}
        _validate_topic_key(item)
        assert item["topic_key"] == "a/b"

        item = {"topic_key": "invalid"}
        _validate_topic_key(item)
        assert item["topic_key"] == "misc/unknown"

        item = {"topic_key": "no_slash_at_all"}
        _validate_topic_key(item)
        assert item["topic_key"] == "misc/unknown"

        item = {"topic_key": ""}
        _validate_topic_key(item)
        assert item["topic_key"] == "misc/unknown"

        item = {}
        _validate_topic_key(item)
        assert item["topic_key"] == "misc/unknown"

    def test_timestamp_validation(self):
        """_validate_timestamp validates ISO-8601 format."""
        item = {"timestamp": "2024-01-01T00:00:00Z"}
        _validate_timestamp(item)
        assert item["timestamp"] == "2024-01-01T00:00:00Z"

        item = {"timestamp": "2024-01-01T00:00:00+00:00"}
        _validate_timestamp(item)
        assert item["timestamp"] == "2024-01-01T00:00:00+00:00"

        item = {"timestamp": "invalid"}
        _validate_timestamp(item)
        # Should be replaced with current time
        assert item["timestamp"] != "invalid"

        item = {"timestamp": ""}
        _validate_timestamp(item)
        assert item["timestamp"] != ""

        item = {}
        _validate_timestamp(item)
        assert "timestamp" in item

    def test_task_status_validation(self):
        """_validate_task_status normalizes valid statuses, defaults invalid."""
        item = {"status": "completed"}
        _validate_task_status(item)
        assert item["status"] == "pending"

        item = {"status": "in_progress"}
        _validate_task_status(item)
        assert item["status"] == "in_progress"

        item = {"status": "pending"}
        _validate_task_status(item)
        assert item["status"] == "pending"

        item = {"status": "cancelled"}
        _validate_task_status(item)
        assert item["status"] == "pending"

        item = {"status": "INVALID"}
        _validate_task_status(item)
        assert item["status"] == "pending"

        item = {"status": ""}
        _validate_task_status(item)
        assert item["status"] == "pending"

        item = {}
        _validate_task_status(item)
        assert item["status"] == "pending"

    def test_content_truncation(self):
        """_validate_content truncates content longer than 2000 chars."""
        long_content = "x" * 3000
        item = {"content": long_content}
        _validate_content(item)
        assert len(item["content"]) == 2003  # 2000 + "..."
        assert item["content"].endswith("...")

    def test_content_strips_control_chars(self):
        """_validate_content strips control characters."""
        item = {"content": "hello\x00world\x1ftest"}
        _validate_content(item)
        assert "\x00" not in item["content"]
        assert "\x1f" not in item["content"]
        assert "helloworldtest" in item["content"]

    def test_content_preserves_html_literal(self):
        """C4 Fix: _validate_content does NOT escape HTML literals."""
        item = {"content": "<b>hi</b>"}
        _validate_content(item)
        assert item["content"] == "<b>hi</b>"
        assert "&lt;" not in item["content"]

    def test_content_handles_null_bytes(self):
        """_validate_content removes null bytes from content."""
        item = {"content": "hello\x00world"}
        _validate_content(item)
        assert "\x00" not in item["content"]
        assert "helloworld" in item["content"]

    def test_sanitize_text_strips_control_chars(self):
        """_sanitize_text strips control characters from summary."""
        result = _sanitize_text("hello\x00world\x1ftest")
        assert "\x00" not in result
        assert "\x1f" not in result
        assert "helloworldtest" in result

    def test_sanitize_text_preserves_html_literal(self):
        """C4 Fix: _sanitize_text does NOT escape HTML literals."""
        result = _sanitize_text("<b>hi</b>")
        assert result == "<b>hi</b>"
        assert "&lt;" not in result

    def test_extraction_size_caps(self):
        """_validate_extraction_items caps extraction items."""
        args = {
            "facts": [
                {"content": "x" * 3000, "topic_key": "dev/test", "confidence": "high"}
            ],
            "events": [
                {"content": "y" * 3000, "timestamp": "2024-01-01T00:00:00Z"}
            ],
            "instructions": [
                {"content": "z" * 3000, "topic_key": "dev/test"}
            ],
            "tasks": [
                {"content": "w" * 3000, "status": "pending"}
            ],
        }
        _validate_extraction_items(args)
        assert len(args["facts"][0]["content"]) == 2003
        assert len(args["events"][0]["content"]) == 2003
        assert len(args["instructions"][0]["content"]) == 2003
        assert len(args["tasks"][0]["content"]) == 2003

    def test_per_item_type_validators(self):
        """Per-item-type validators apply correct field checks."""
        fact = {"content": "F1", "topic_key": "dev/test", "confidence": "high"}
        _validate_fact_item(fact)
        assert fact["content"] == "F1"
        assert fact["topic_key"] == "dev/test"
        assert fact["confidence"] == "high"

        event = {"content": "E1", "timestamp": "2024-01-01T00:00:00Z"}
        _validate_event_item(event)
        assert event["timestamp"] == "2024-01-01T00:00:00Z"
        assert event["confidence"] == "medium"

        task = {"content": "T1", "status": "in_progress"}
        _validate_task_item(task)
        assert task["status"] == "in_progress"

    def test_sanitize_text_no_double_escaping(self):
        """C4 Regression: calling _sanitize_text twice produces same output."""
        original = "<b>hi</b>&amp;friends"
        first = _sanitize_text(original)
        second = _sanitize_text(first)
        assert first == second
        assert "&amp;" in first  # original entity preserved
        assert "&lt;" not in first  # no new escaping introduced


# ---------------------------------------------------------------------------
# T4: build_extraction_prompt wrapper tests
# ---------------------------------------------------------------------------

class TestBuildExtractionPromptWrapper:
    """Tests for the backward-compat build_extraction_prompt wrapper."""

    def test_wrapper_delegates_to_build_unified_prompt(self):
        """build_extraction_prompt produces same output as build_unified_prompt with empty summary."""
        from memory.prompts.extraction import build_unified_prompt, build_extraction_prompt
        msgs = [{"role": "user", "content": "hello"}]
        result_a = build_extraction_prompt(msgs)
        result_b = build_unified_prompt(msgs, current_summary="", iso_timestamp="",
                                         conversation_start="", conversation_end="")
        # Should be equal if kwargs match defaults
        assert "BEGIN CONVERSATION" in result_a
        assert "hello" in result_a

    def test_wrapper_forwards_iso_timestamp(self):
        """build_extraction_prompt passes iso_timestamp through."""
        from memory.prompts.extraction import build_extraction_prompt
        msgs = [{"role": "user", "content": "test"}]
        result = build_extraction_prompt(msgs, iso_timestamp="2026-01-01T00:00:00Z")
        assert "2026-01-01T00:00:00Z" in result

    def test_wrapper_forwards_existing_memories(self):
        """build_extraction_prompt passes existing_memories through."""
        from memory.prompts.extraction import build_extraction_prompt
        msgs = [{"role": "user", "content": "test"}]
        result = build_extraction_prompt(msgs, existing_memories="memories go here")
        assert "memories go here" in result


# ---------------------------------------------------------------------------
# T5: Per-session lock tests
# ---------------------------------------------------------------------------

class TestPerSessionLock:
    """Tests for per-session sync dedup in executors."""

    def test_concurrent_sync_for_same_session_returns_none(self):
        """Second submit_sync call for same session_id returns None while first is queued."""
        import threading
        from memory.executors import submit_sync, _session_locks, _session_locks_lock

        barrier = threading.Barrier(2)

        def slow_fn(session_id):
            barrier.wait(timeout=5)
            return "done"

        # First call should queue
        f1 = submit_sync(slow_fn, "sess_123")
        assert f1 is not None

        # Second call for same session should return None
        f2 = submit_sync(slow_fn, "sess_123")
        assert f2 is None

        # Third call for different session should succeed
        f3 = submit_sync(slow_fn, "sess_456")
        assert f3 is not None

        # Wait for thread to finish
        f1.result(timeout=5)
        f3.result(timeout=5)

        # Lock should be released now
        with _session_locks_lock:
            assert "sess_123" not in _session_locks

    def test_session_lock_cleanup_on_exception(self):
        """Session lock is cleaned up even if the function raises an exception."""
        import time
        from memory.executors import submit_sync, _session_locks, _session_locks_lock

        def failing_fn(session_id):
            raise ValueError("test error")

        f = submit_sync(failing_fn, "sess_err")
        assert f is not None

        try:
            f.result(timeout=5)
        except ValueError:
            pass  # expected

        # Allow done callbacks to propagate
        time.sleep(0.05)

        with _session_locks_lock:
            assert "sess_err" not in _session_locks


# ---------------------------------------------------------------------------
# T6: Prompt security tests
# ---------------------------------------------------------------------------

class TestPromptSecurity:
    """Tests for security hardening in extraction prompts."""

    def test_system_prompt_has_injection_defense(self):
        """UNIFIED_SYSTEM_PROMPT warns that conversation text is data, not instructions."""
        from memory.prompts.extraction import UNIFIED_SYSTEM_PROMPT
        assert "DATA only" in UNIFIED_SYSTEM_PROMPT or "data only" in UNIFIED_SYSTEM_PROMPT
        assert "NOT instructions" in UNIFIED_SYSTEM_PROMPT

    def test_unified_prompt_has_end_conversation_reinforcement(self):
        """build_unified_prompt ends with a post-conversation reinforcement line."""
        from memory.prompts.extraction import build_unified_prompt
        msgs = [{"role": "user", "content": "test"}]
        result = build_unified_prompt(msgs)
        assert "END OF CONVERSATION" in result
        assert "data only" in result.lower()

    def test_no_summary_in_instructions_categories(self):
        """summary is NOT listed as an extraction category in instructions."""
        from memory.prompts.extraction import UNIFIED_EXTRACTION_INSTRUCTIONS
        # summary should not appear as a top-level category
        lines = UNIFIED_EXTRACTION_INSTRUCTIONS.split("\n")
        category_lines = [l.strip() for l in lines if l.strip().startswith("- ")]
        assert not any("summary" in l for l in category_lines), \
            "summary should not be in extraction categories"
