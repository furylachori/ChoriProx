"""Tests for skill_registry module."""
import pytest
from proxy.skills.skill_registry import error_response


class TestErrorResponse:
    """Test error_response utility."""

    def test_error_response_shape(self):
        """error_response returns a properly shaped OpenAI-compatible error dict."""
        result = error_response("task not found")
        assert "choices" in result
        assert len(result["choices"]) == 1
        assert result["choices"][0]["message"]["role"] == "assistant"
        assert "Error: task not found" in result["choices"][0]["message"]["content"]
        assert result["model"] == "skill"
        # _status_code key should be present
        assert "_status_code" in result
