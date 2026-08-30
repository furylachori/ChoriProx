# test_sota_channel.py — Unit tests for SOTAChannel Phase A

import pytest
import json
from unittest.mock import Mock, patch, MagicMock

from memory.sota_channel import (
    SOTAChannel,
    SOTAResult,
    SOTAParseError,
    SOTATimeoutError,
    SOTAAPIError,
    SOTARateLimitError,
    CircuitBreakerOpen,
)
from memory.circuit_breaker import CircuitBreaker, CircuitState
from memory._sota_http import SOTAMode


# ── Mock response helpers ───────────────────────────────────────────────────

class MockMessage:
    def __init__(self, content=None, tool_calls=None):
        self.content = content
        self.tool_calls = tool_calls or []


class MockChoice:
    def __init__(self, message):
        self.message = message


class MockResponse:
    def __init__(self, choices):
        self.choices = choices
        self.usage = None


class MockAPIError(Exception):
    """Mock API error mimicking OpenAI's APIError for testing.

    Since the OpenAI SDK's APIError.__init__ signature varies across versions,
    we patch the exception handling in sota_channel to catch this mock instead.
    """
    def __init__(self, message, status_code=None, response=None):
        super().__init__(message)
        self.status_code = status_code
        self.response = response


# ── Fixture ─────────────────────────────────────────────────────────────────

@pytest.fixture
def mock_sota_client():
    """Mock the SOTA client singleton with a fake OpenAI client."""
    with patch("memory._sota_http.get_sota_client") as mock_get_client:
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client
        yield mock_client


@pytest.fixture
def channel_summarization():
    """Create a SOTAChannel in SUMMARIZATION mode (content-only)."""
    return SOTAChannel(
        mode=SOTAMode.SUMMARIZATION,
        system_prompt="Summarize the content.",
        parse_fn=json.loads,
    )


@pytest.fixture
def channel_extraction_with_tools():
    """Create a SOTAChannel in EXTRACTION mode with tools."""
    return SOTAChannel(
        mode=SOTAMode.EXTRACTION,
        system_prompt="Extract entities.",
        tools=[{"type": "function", "function": {"name": "extract", "parameters": {}}}],
        tool_choice={"type": "function", "function": {"name": "extract"}},
    )


@pytest.fixture
def channel_compression():
    """Create a SOTAChannel in COMPRESSION mode (content-only)."""
    return SOTAChannel(
        mode=SOTAMode.COMPRESSION,
        system_prompt="Compress the content.",
        parse_fn=json.loads,
    )


@pytest.fixture
def channel_planning():
    """Create a SOTAChannel in PLANNING mode (content-only)."""
    return SOTAChannel(
        mode=SOTAMode.PLANNING,
        system_prompt="Create a plan.",
    )


# ── Helper to build a content-only response ──────────────────────────────────

def make_content_response(text: str) -> MockResponse:
    return MockResponse(choices=[MockChoice(message=MockMessage(content=text))])


# ── Fixture to reset circuit breaker before each test ───────────────────────

@pytest.fixture(autouse=True)
def reset_sota_breaker():
    """Reset the module-level SOTA circuit breaker before each test."""
    import memory.sota_channel as sc
    sc._sota_breaker.force_closed()
    yield
    sc._sota_breaker.force_closed()


# ── Tests ────────────────────────────────────────────────────────────────────

def test_successful_content_mode(mock_sota_client, channel_summarization):
    """SUMMARIZATION mode returns parsed dict when content-only (no tools)."""
    mock_sota_client.chat.completions.create.return_value = make_content_response(
        '{"summary":"test","tool_summary":"","token_estimate":1}'
    )

    result = channel_summarization.ask("Some content to summarize")

    assert isinstance(result, SOTAResult)
    assert result.parsed["summary"] == "test"


def test_successful_tool_call_mode(mock_sota_client, channel_extraction_with_tools):
    """EXTRACTION mode with tools returns a dict when model provides tool_calls."""
    tool_args = '{"extraction_timestamp":"2025-05-10T12:00:00Z","entities":[],"facts":[],"instructions":[],"tasks":[]}'

    # Build a Function object mock for tool_calls
    mock_function = MagicMock()
    mock_function.arguments = tool_args

    mock_tool_call = MagicMock()
    mock_tool_call.function = mock_function

    message = MockMessage(content=None, tool_calls=[mock_tool_call])
    mock_sota_client.chat.completions.create.return_value = MockResponse(
        choices=[MockChoice(message=message)]
    )

    result = channel_extraction_with_tools.ask("Extract something")

    assert isinstance(result.parsed, dict)
    assert "extraction_timestamp" in result.parsed
    assert "entities" in result.parsed


def _make_parse_fn():
    """Factory to create a parse_fn that raises SOTAParseError for invalid JSON."""
    def parse_fn(raw):
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            raise SOTAParseError(f"Invalid JSON from model: {raw[:50]}...")
    return parse_fn


def test_parse_failure_retry(mock_sota_client):
    """Invalid JSON on first attempt triggers retry and succeeds on second."""
    channel = SOTAChannel(
        mode=SOTAMode.SUMMARIZATION,
        system_prompt="Summarize the content.",
        parse_fn=_make_parse_fn(),
    )
    # First call returns invalid JSON, second call succeeds
    mock_sota_client.chat.completions.create.side_effect = [
        make_content_response("not valid json at all"),
        make_content_response('{"summary":"recovered","tool_summary":"","token_estimate":1}'),
    ]

    result = channel.ask("Some content")

    assert result.attempts == 2
    assert result.parsed["summary"] == "recovered"


def test_429_backoff(mock_sota_client):
    """429 error with Retry-After header triggers backoff and retry succeeds."""
    error_response = MagicMock()
    error_response.headers = {"Retry-After": "1"}

    rate_limit_error = MockAPIError(
        message="Rate limited",
        status_code=429,
        response=error_response,
    )

    channel = SOTAChannel(
        mode=SOTAMode.COMPRESSION,
        system_prompt="Compress the content.",
    )

    # Patch the OpenAI exception references in sota_channel so MockAPIError is caught
    with patch("memory.sota_channel.OpenAIAPIError", MockAPIError):
        mock_sota_client.chat.completions.create.side_effect = [
            rate_limit_error,
            make_content_response("compressed result"),
        ]

        result = channel.ask("Compress this")

    # Verify two calls were made (initial + retry)
    assert mock_sota_client.chat.completions.create.call_count == 2
    assert isinstance(result, SOTAResult)


def test_circuit_breaker_open_rejection(mock_sota_client):
    """Open circuit breaker raises CircuitBreakerOpen immediately without API call."""
    # Trip the breaker: call on_failure 5 times (failure_threshold = 5)
    import memory.sota_channel as sc
    breaker = sc._sota_breaker
    breaker.force_closed()  # ensure clean state
    for _ in range(5):
        breaker.on_failure()

    assert breaker.state == CircuitState.OPEN

    channel = SOTAChannel(
        mode=SOTAMode.SUMMARIZATION,
        system_prompt="Summarize.",
    )

    with pytest.raises(CircuitBreakerOpen):
        channel.ask("test prompt")

    # Ensure no API call was made
    mock_sota_client.chat.completions.create.assert_not_called()


def test_circuit_breaker_half_open_recovery(mock_sota_client):
    """HALF_OPEN state allows one probe call which succeeds and resets circuit."""
    import memory.sota_channel as sc
    breaker = sc._sota_breaker
    breaker.force_closed()
    for _ in range(5):
        breaker.on_failure()

    assert breaker.state == CircuitState.OPEN

    # Manually transition to HALF_OPEN by setting state directly
    with breaker._lock:
        breaker._state = CircuitState.HALF_OPEN
        breaker._half_open_pending = False  # allow first probe

    mock_sota_client.chat.completions.create.return_value = make_content_response(
        '{"summary":"probe success","tool_summary":"","token_estimate":1}'
    )

    channel = SOTAChannel(
        mode=SOTAMode.SUMMARIZATION,
        system_prompt="Summarize.",
        parse_fn=json.loads,
    )

    result = channel.ask("probe request")

    # Probe should succeed
    assert isinstance(result, SOTAResult)
    assert result.parsed["summary"] == "probe success"

    # Circuit should now be CLOSED
    assert breaker.state == CircuitState.CLOSED
    assert breaker.failure_count == 0


def test_content_only_mode_returns_text(mock_sota_client, channel_planning):
    """PLANNING mode (no tools) returns raw string when no JSON parse fn provided."""
    mock_sota_client.chat.completions.create.return_value = make_content_response(
        "This is a plain text plan without JSON"
    )

    result = channel_planning.ask("Make a plan")

    # In content-only mode without JSON, parsed is the raw cleaned string
    assert isinstance(result.parsed, str)


def test_validate_input_rejects_none():
    """ask(None) raises ValueError."""
    channel = SOTAChannel(
        mode=SOTAMode.SUMMARIZATION,
        system_prompt="Summarize.",
    )
    with pytest.raises(ValueError, match="cannot be None"):
        channel.ask(None)


def test_validate_input_rejects_empty():
    """ask('') raises ValueError."""
    channel = SOTAChannel(
        mode=SOTAMode.SUMMARIZATION,
        system_prompt="Summarize.",
    )
    with pytest.raises(ValueError, match="cannot be empty"):
        channel.ask("")


def test_validate_input_rejects_whitespace():
    """ask('   ') raises ValueError."""
    channel = SOTAChannel(
        mode=SOTAMode.SUMMARIZATION,
        system_prompt="Summarize.",
    )
    with pytest.raises(ValueError, match="cannot be empty"):
        channel.ask("   ")


def test_validate_input_accepts_valid(mock_sota_client, channel_summarization):
    """ask('valid prompt') with mocked response does not raise."""
    mock_sota_client.chat.completions.create.return_value = make_content_response(
        '{"summary":"ok","tool_summary":"","token_estimate":1}'
    )

    result = channel_summarization.ask("valid prompt")
    assert isinstance(result, SOTAResult)


def test_tool_call_content_fallback_type_guard(mock_sota_client, channel_extraction_with_tools):
    """EXTRACTION mode with tools returns content containing primitive JSON -> SOTAParseError.

    When the model returns content (not a tool_call) that is valid JSON but not a dict,
    the code raises SOTAParseError on that attempt. This triggers a retry.
    With only one mock response, the retry exhausts all attempts and raises SOTAParseError.
    """
    # Model returns content with a primitive JSON string (not a dict)
    # The code parses the content as JSON (succeeds: "hello"), then checks if it's a dict (it's not) -> raises SOTAParseError
    channel = channel_extraction_with_tools

    mock_sota_client.chat.completions.create.return_value = MockResponse(
        choices=[MockChoice(message=MockMessage(content='"hello"'))]
    )

    # After attempt 0 raises SOTAParseError, retries happen with the same response.
    # Attempts 1-4 all raise, then attempt 4 propagates the exception.
    with pytest.raises(SOTAParseError):
        channel.ask("Extract entities")


@patch("time.sleep", return_value=None)
def test_502_retry_succeeds(mock_sleep, mock_sota_client):
    """502 server error retries with exponential backoff and succeeds on third attempt."""
    error_502 = MockAPIError(
        message="Bad Gateway",
        status_code=502,
        response=None,
    )

    channel = SOTAChannel(
        mode=SOTAMode.COMPRESSION,
        system_prompt="Compress the content.",
    )

    with patch("memory.sota_channel.OpenAIAPIError", MockAPIError):
        mock_sota_client.chat.completions.create.side_effect = [
            error_502,
            error_502,
            make_content_response("eventually succeeded"),
        ]

        result = channel.ask("Content to compress")

    assert mock_sota_client.chat.completions.create.call_count == 3
    assert result.attempts == 3
    assert isinstance(result, SOTAResult)
    assert result.raw == "eventually succeeded"


@patch("time.sleep", return_value=None)
def test_5xx_exhausts_retries(mock_sleep, mock_sota_client):
    """After 5 consecutive 503 server errors, raises SOTAAPIError with status_code."""
    error_503 = MockAPIError(
        message="Service Unavailable",
        status_code=503,
        response=None,
    )

    channel = SOTAChannel(
        mode=SOTAMode.COMPRESSION,
        system_prompt="Compress the content.",
    )

    import memory.sota_channel as sc
    real_breaker = CircuitBreaker(
        failure_threshold=5,
        recovery_timeout_seconds=0.0,
        name="sota-test-5xx",
    )
    real_breaker.force_closed()
    breaker_saved = sc._sota_breaker
    try:
        sc._sota_breaker = real_breaker

        with patch("memory.sota_channel.OpenAIAPIError", MockAPIError):
            mock_sota_client.chat.completions.create.side_effect = [error_503] * 5

            with pytest.raises(SOTAAPIError) as exc_info:
                channel.ask("Content that will fail")

        assert exc_info.value.status_code == 503
        assert "503" in str(exc_info.value)
        assert mock_sota_client.chat.completions.create.call_count == 5
        assert real_breaker.failure_count == 1
        assert real_breaker.state == CircuitState.CLOSED
        assert real_breaker.can_execute() is True
    finally:
        sc._sota_breaker = breaker_saved