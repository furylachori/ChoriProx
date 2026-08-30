"""Tests for proxy.pipeline.prompt_injector.PromptInjector."""
import pytest

from proxy.pipeline.prompt_injector import PromptInjector


class TestAppendToSystem:
    def test_str_content(self):
        msgs = [{"role": "system", "content": "You are helpful."}]
        assert PromptInjector.append_to_system(msgs, "Be concise.") is True
        assert msgs[0]["content"] == "You are helpful.\n\nBe concise."

    def test_multimodal_content(self):
        msgs = [{"role": "system", "content": [{"type": "text", "text": "Base"}]}]
        assert PromptInjector.append_to_system(msgs, "Extra") is True
        assert msgs[0]["content"][-1] == {"type": "text", "text": "\n\nExtra"}

    def test_empty_messages(self):
        assert PromptInjector.append_to_system([], "text") is False

    def test_no_system_message(self):
        msgs = [{"role": "user", "content": "hi"}]
        assert PromptInjector.append_to_system(msgs, "text") is False

    def test_none_content(self):
        msgs = [{"role": "system", "content": None}]
        assert PromptInjector.append_to_system(msgs, "text") is False

    def test_tuple_messages(self):
        msgs = ({"role": "system", "content": "Base"},)
        assert PromptInjector.append_to_system(msgs, "Extra") is False

    def test_empty_text(self):
        msgs = [{"role": "system", "content": "Base"}]
        assert PromptInjector.append_to_system(msgs, "") is False


class TestInsertSystem:
    def test_default_position(self):
        msgs = [{"role": "system", "content": "S"}, {"role": "user", "content": "U"}]
        assert PromptInjector.insert_system(msgs, "Injected") is True
        assert msgs[1] == {"role": "system", "content": "Injected"}
        assert len(msgs) == 3

    def test_invalid_position_negative(self):
        msgs = [{"role": "system", "content": "S"}]
        assert PromptInjector.insert_system(msgs, "X", position=-1) is False

    def test_invalid_position_beyond_length(self):
        msgs = [{"role": "system", "content": "S"}]
        assert PromptInjector.insert_system(msgs, "X", position=5) is False


class TestAppendUserMessage:
    def test_appends(self):
        msgs = [{"role": "system", "content": "S"}]
        assert PromptInjector.append_user_message(msgs, "Hello") is True
        assert msgs[-1] == {"role": "user", "content": "Hello"}


class TestPrependToSystem:
    def test_str_content(self):
        msgs = [{"role": "system", "content": "You are helpful."}]
        assert PromptInjector.prepend_to_system(msgs, "IMPORTANT:") is True
        assert msgs[0]["content"] == "IMPORTANT:\n\nYou are helpful."

    def test_multimodal_content(self):
        msgs = [{"role": "system", "content": [{"type": "text", "text": "Base"}]}]
        assert PromptInjector.prepend_to_system(msgs, "First") is True
        assert msgs[0]["content"][0] == {"type": "text", "text": "First\n\n"}

    def test_tuple_messages(self):
        msgs = ({"role": "system", "content": "Base"},)
        assert PromptInjector.prepend_to_system(msgs, "First") is False


class TestPrependToLastUserMessage:
    def test_str_content_injects(self):
        msgs = [
            {"role": "system", "content": "S"},
            {"role": "user", "content": "Fix the bug"},
        ]
        assert PromptInjector.prepend_to_last_user_message(msgs, "Reminder") is True
        assert msgs[-1]["content"] == "Reminder\n\n---\n\nFix the bug"

    def test_str_whitespace_skipped(self):
        msgs = [{"role": "user", "content": "   \n\t  "}]
        assert PromptInjector.prepend_to_last_user_message(msgs, "Reminder") is False
        assert msgs[-1]["content"] == "   \n\t  "

    def test_str_empty_skipped(self):
        msgs = [{"role": "user", "content": ""}]
        assert PromptInjector.prepend_to_last_user_message(msgs, "Reminder") is False
        assert msgs[-1]["content"] == ""

    def test_no_user_message(self):
        msgs = [{"role": "system", "content": "S"}]
        assert PromptInjector.prepend_to_last_user_message(msgs, "Reminder") is False

    def test_empty_text_skipped(self):
        assert PromptInjector.prepend_to_last_user_message(msgs := [], "") is False

    def test_multimodal_all_whitespace_text_skipped(self):
        msgs = [{"role": "user", "content": [
            {"type": "text", "text": "   "},
            {"type": "text", "text": "\n\t"},
        ]}]
        assert PromptInjector.prepend_to_last_user_message(msgs, "Reminder") is False
        assert len(msgs[-1]["content"]) == 2

    def test_multimodal_empty_text_skipped(self):
        msgs = [{"role": "user", "content": [
            {"type": "text", "text": ""},
            {"type": "text", "text": ""},
        ]}]
        assert PromptInjector.prepend_to_last_user_message(msgs, "Reminder") is False
        assert len(msgs[-1]["content"]) == 2

    def test_multimodal_image_plus_whitespace_text_injects(self):
        msgs = [{"role": "user", "content": [
            {"type": "text", "text": "   "},
            {"type": "image_url", "image_url": {"url": "http://img"}},
        ]}]
        assert PromptInjector.prepend_to_last_user_message(msgs, "Reminder") is True
        assert msgs[-1]["content"][0] == {"type": "text", "text": "Reminder\n\n---\n\n"}
        assert len(msgs[-1]["content"]) == 3

    def test_multimodal_real_text_injects(self):
        msgs = [{"role": "user", "content": [
            {"type": "text", "text": "Describe this"},
            {"type": "image_url", "image_url": {"url": "http://img"}},
        ]}]
        assert PromptInjector.prepend_to_last_user_message(msgs, "Reminder") is True
        assert msgs[-1]["content"][0] == {"type": "text", "text": "Reminder\n\n---\n\n"}
        assert len(msgs[-1]["content"]) == 3

    def test_multimodal_target_last_user_message(self):
        msgs = [
            {"role": "user", "content": "first"},
            {"role": "assistant", "content": "ok"},
            {"role": "user", "content": [{"type": "text", "text": "second"}]},
        ]
        assert PromptInjector.prepend_to_last_user_message(msgs, "Reminder") is True
        assert msgs[-1]["content"][0] == {"type": "text", "text": "Reminder\n\n---\n\n"}
        assert msgs[-1]["content"][1] == {"type": "text", "text": "second"}
