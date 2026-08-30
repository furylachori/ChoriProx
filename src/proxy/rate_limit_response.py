"""Rate-limit response shape helpers for consistent 429 responses."""


def rate_limited_response(retry_after: int):
    """Flat rate-limit response shape (admin/app layers)."""
    return (
        {"error": "rate_limited", "retry_after": retry_after},
        429,
    )


def rate_limited_error_response(message: str, retry_after: int):
    """Nested rate-limit response shape (dispatch layer)."""
    return (
        {"error": {"message": message, "type": "rate_limit_exceeded", "retry_after": retry_after}},
        429,
    )
