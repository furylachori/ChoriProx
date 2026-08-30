import pytest
from unittest.mock import MagicMock, patch
from email.utils import formatdate
from datetime import datetime, timezone, timedelta

from memory.sota_client import (
    parse_retry_after, SOTARateLimitError, SOTAAPIError,
    SOTAClient, CompressionError,
)
from memory._sota_rate_limiter import _SOTA_MIN_INTERVAL

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# parse_retry_after
# ---------------------------------------------------------------------------
class TestParseRetryAfter:
    def test_float_string_returns_float(self):
        assert parse_retry_after("3.14") == 3.14

    def test_integer_seconds_returned(self):
        assert parse_retry_after("5") == 5.0

    def test_http_date_string_parsed_to_seconds(self):
        # Use a date 10 seconds in the future
        future = datetime.now(timezone.utc) + timedelta(seconds=10)
        date_str = formatdate(future.timestamp(), usegmt=True)
        result = parse_retry_after(date_str)
        assert 0 <= result <= 11

    def test_none_returns_default(self):
        assert parse_retry_after(None) == 2.5
        assert parse_retry_after(None, default=5.0) == 5.0

    def test_empty_string_returns_default(self):
        assert parse_retry_after("") == 2.5
        assert parse_retry_after("", default=10.0) == 10.0

    def test_invalid_string_returns_default(self):
        assert parse_retry_after("not-a-number") == 2.5
        assert parse_retry_after("garbage") == 2.5

    def test_future_date_returns_positive_seconds(self):
        future = datetime.now(timezone.utc) + timedelta(seconds=30)
        date_str = formatdate(future.timestamp(), usegmt=True)
        result = parse_retry_after(date_str)
        assert result > 0

    def test_past_date_returns_0(self):
        past = datetime.now(timezone.utc) - timedelta(seconds=60)
        date_str = formatdate(past.timestamp(), usegmt=True)
        assert parse_retry_after(date_str) == 2.5

    def test_caps_oversized_numeric_value(self):
        from memory.sota_channel import _RETRY_AFTER_CAP_SECONDS
        assert parse_retry_after("86400") == float(_RETRY_AFTER_CAP_SECONDS)

    def test_date_format_capped(self):
        future = datetime.now(timezone.utc) + timedelta(seconds=86400)
        date_str = formatdate(future.timestamp(), usegmt=True)
        assert parse_retry_after(date_str) == 60.0

    def test_cap_log_emitted_when_capped(self, caplog):
        import logging
        with caplog.at_level(logging.WARNING, logger="memory.sota_channel"):
            parse_retry_after("86400")
        assert any(
            "Retry-After capped" in r.message and r.name == "memory.sota_channel"
            for r in caplog.records
        )

    def test_env_override_via_subprocess(self):
        """PROXY_SOTA_RETRY_AFTER_CAP=5 caps at 5s (subprocess probe)."""
        import os
        import subprocess
        import sys
        env = os.environ.copy()
        env["PROXY_SOTA_RETRY_AFTER_CAP"] = "5"
        env["PYTHONPATH"] = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            "src",
        ) + os.pathsep + env.get("PYTHONPATH", "")
        result = subprocess.run(
            [sys.executable, "-c",
             "import sys; from memory.sota_channel import parse_retry_after; "
             "sys.stdout.write(str(parse_retry_after('86400')))"],
            env=env, capture_output=True, text=True, timeout=30,
        )
        assert result.returncode == 0, f"Subprocess failed: {result.stderr}"
        assert float(result.stdout.strip()) == 5.0

    def test_env_override_invalid_falls_back_to_default(self):
        """PROXY_SOTA_RETRY_AFTER_CAP=garbage → falls back to 60s."""
        import os
        import subprocess
        import sys
        env = os.environ.copy()
        env["PROXY_SOTA_RETRY_AFTER_CAP"] = "garbage"
        env["PYTHONPATH"] = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            "src",
        ) + os.pathsep + env.get("PYTHONPATH", "")
        result = subprocess.run(
            [sys.executable, "-c",
             "import sys; from memory.sota_channel import parse_retry_after; "
             "sys.stdout.write(str(parse_retry_after('86400')))"],
            env=env, capture_output=True, text=True, timeout=30,
        )
        assert result.returncode == 0, f"Subprocess failed: {result.stderr}"
        assert float(result.stdout.strip()) == 60.0

    def test_env_override_out_of_range_clamps(self):
        """PROXY_SOTA_RETRY_AFTER_CAP=9999 → clamps to 300s."""
        import os
        import subprocess
        import sys
        env = os.environ.copy()
        env["PROXY_SOTA_RETRY_AFTER_CAP"] = "9999"
        env["PYTHONPATH"] = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            "src",
        ) + os.pathsep + env.get("PYTHONPATH", "")
        result = subprocess.run(
            [sys.executable, "-c",
             "import sys; from memory.sota_channel import parse_retry_after; "
             "sys.stdout.write(str(parse_retry_after('86400')))"],
            env=env, capture_output=True, text=True, timeout=30,
        )
        assert result.returncode == 0, f"Subprocess failed: {result.stderr}"
        assert float(result.stdout.strip()) == 300.0


# ---------------------------------------------------------------------------
# SOTARateLimitError
# ---------------------------------------------------------------------------
class TestSOTARateLimitError:
    def test_is_subclass_of_sota_api_error(self):
        assert issubclass(SOTARateLimitError, SOTAAPIError)

    def test_has_status_code_429(self):
        exc = SOTARateLimitError("rate limited")
        assert exc.status_code == 429

    def test_has_retry_after_attribute(self):
        exc = SOTARateLimitError("rate limited")
        assert hasattr(exc, "retry_after")

    def test_default_retry_after_is_2_5(self):
        exc = SOTARateLimitError("rate limited")
        assert exc.retry_after == 2.5

    def test_custom_retry_after_stored(self):
        exc = SOTARateLimitError("rate limited", retry_after=10.0)
        assert exc.retry_after == 10.0


# ---------------------------------------------------------------------------
# _SOTA_MIN_INTERVAL
# ---------------------------------------------------------------------------
class TestSOTAMinInterval:
    def test_equals_one_third(self):
        assert _SOTA_MIN_INTERVAL == pytest.approx(1.0 / 3.0)


# ---------------------------------------------------------------------------
# CompressionError
# ---------------------------------------------------------------------------
class TestCompressionError:
    def test_is_exception_subclass(self):
        assert issubclass(CompressionError, Exception)

    def test_can_be_raised_and_caught(self):
        with pytest.raises(CompressionError) as exc_info:
            raise CompressionError("test message")
        assert str(exc_info.value) == "test message"


# ---------------------------------------------------------------------------
# summarize_message stub callable (no real API)
# ---------------------------------------------------------------------------
class TestSummarizeMessageStub:
    def test_sota_client_has_summarize_message_method(self):
        client = SOTAClient()
        assert callable(client.summarize_message)

    def test_returns_dict_with_required_keys(self):
        client = SOTAClient()
        mock_result = MagicMock()
        mock_result.raw = {"summary": "test", "tool_summary": "", "token_estimate": 1}
        mock_result.elapsed_ms = 100.0
        mock_result.attempts = 1
        with patch("memory.sota_channel.SOTAChannel.ask", return_value=mock_result):
            result = client.summarize_message("sess-1", 0, "Hello world", "user")
        assert isinstance(result, dict)
        for key in ("summary", "tool_summary", "token_estimate"):
            assert key in result


# ---------------------------------------------------------------------------
# summarize_message with mock
# ---------------------------------------------------------------------------
class TestSummarizeMessageWithMock:
    def test_mock_openai_returns_specific_json(self):
        """Patch SOTAChannel.ask to return a specific JSON response."""
        client = SOTAClient()

        mock_result = MagicMock()
        mock_result.raw = {
            "summary": "User said hello.",
            "tool_summary": "[tools: none]",
            "token_estimate": 12,
        }
        mock_result.elapsed_ms = 50.0
        mock_result.attempts = 1

        with patch("memory.sota_channel.SOTAChannel.ask", return_value=mock_result):
            result = client.summarize_message("sess-1", 0, "Hello world", "user")
            assert result["summary"] == "User said hello."
            assert result["tool_summary"] == "[tools: none]"
            assert result["token_estimate"] == 12

    def test_token_cap_applied_when_summary_exceeds_limit(self):
        """Verify 512-token cap is enforced on long summaries."""
        client = SOTAClient()

        # Create a summary with ~500 words (well over 512 tokens)
        long_summary = " ".join(["word"] * 500)
        mock_result = MagicMock()
        mock_result.raw = {
            "summary": long_summary,
            "tool_summary": "",
            "token_estimate": 500,
        }
        mock_result.elapsed_ms = 100.0
        mock_result.attempts = 1

        with patch("memory.sota_channel.SOTAChannel.ask", return_value=mock_result):
            result = client.summarize_message("sess-1", 0, "Hello world", "user")
            words = result["summary"].split()
            # Cap=512 tokens, conversion is len(words) / 0.75 > 512 → truncate to 512*0.75=384 words
            assert len(words) <= 385  # 384 words + "..." appended
            assert result["summary"].endswith("...")
