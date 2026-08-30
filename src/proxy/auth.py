"""Authentication, security, and error sanitization."""
import re
import logging

from proxy.utils import extract_text_content

logger = logging.getLogger(__name__)


class AuthConfigError(Exception):
    """Raised when api_key_env is configured but the env var is missing or empty."""

    def __init__(self, model_name, env_var):
        self.model_name = model_name
        self.env_var = env_var
        super().__init__(
            f"API key not configured for model '{model_name}' (env var '{env_var}' is missing or empty)"
        )


_SENSITIVE_KEYS = frozenset(
    {
        "api_key",
        "apikey",
        "api-key",
        "token",
        "secret",
        "authorization",
        "access_token",
        "refresh_token",
        "client_secret",
        "password",
        "private_key",
        "x-api-key",
        "x_api_key",
    }
)


def _sanitize_error_body(body, _depth=0, _max_depth=10):
    """Remove sensitive fields from backend error responses before forwarding to client.
    Returns a sanitized COPY. Handles both dicts and lists of dicts."""
    if _depth > _max_depth:
        return body
    if isinstance(body, dict):
        result = {}
        for key, val in body.items():
            if key.lower() in _SENSITIVE_KEYS:
                continue
            result[key] = _sanitize_error_body(val, _depth + 1, _max_depth)
        return result
    elif isinstance(body, list):
        return [_sanitize_error_body(item, _depth + 1, _max_depth) for item in body]
    elif isinstance(body, str):
        body = re.sub(r"Bearer\s+[A-Za-z0-9_\-\.]{8,}", "Bearer [REDACTED]", body)
        body = re.sub(
            r'(api[_-]?key|token|secret|authorization)\s*[=:]\s*[^\s&"]{8,}',
            r"\1=[REDACTED]",
            body,
            flags=re.I,
        )
        return body
    return body


INJECTION_PATTERNS = [
    (
        re.compile(
            r"(ignore\s+(all\s+)?previous\s+instructions?"
            r"|system\s*:\s*you are"
            r"|admin\s*:\s*act as)"
            r".*"
            r"(unfiltered|no restrictions|bypass|jailbreak|DAN|developer mode"
            r"|no guidelines|no rules|pretend you|forget your|new role|do anything now)",
            re.I | re.DOTALL
        ),
        "prompt_injection",
    ),
]


def detect_malicious_content(content: str) -> list:
    """Detect potentially malicious patterns in content. Returns list of detected types. Log-only, non-blocking."""
    content = extract_text_content(content)
    if not content:
        return []
    return [tag for pattern, tag in INJECTION_PATTERNS if pattern.search(content)]