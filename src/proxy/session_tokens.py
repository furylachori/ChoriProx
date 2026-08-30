"""Signed session tokens for client identity verification.

Tokens bind a client_id (authentication) and a session_id (continuity) with
an HMAC signature, preventing clients from forging session IDs or
impersonating other clients' sessions. The embedded session_id is restored
verbatim on the next request so session continuity survives content changes.
"""
from __future__ import annotations

import hmac
import hashlib
import json
import time
import secrets
import os
import logging

logger = logging.getLogger(__name__)

_SESSION_SECRET = os.environ.get("PROXY_SESSION_SECRET", "")


def create_session_token(client_id: str, session_id: str) -> str:
    """Create a signed session token binding both client_id and session_id.

    Returns a string in the format '{payload_json}:{hex_signature}'.
    The payload contains client_id (authentication), session_id (continuity),
    a created_at timestamp, and a random nonce (token uniqueness).
    The signature is the first 16 hex chars of HMAC-SHA256.

    Returns an empty string if PROXY_SESSION_SECRET is not configured
    (tokens are opt-in via environment variable).
    """
    if not _SESSION_SECRET:
        return ""
    payload = json.dumps({
        "client_id": client_id,
        "session_id": session_id,
        "created_at": time.time(),
        "nonce": secrets.token_hex(8)
    }, separators=(',', ':'))
    sig = hmac.new(_SESSION_SECRET.encode(), payload.encode(), hashlib.sha256).hexdigest()[:16]
    return f"{payload}:{sig}"


def validate_session_token(token: str, max_age_seconds: int = 86400) -> dict | None:
    """Validate a signed session token and return the payload.
    
    Returns the parsed payload dict (with client_id, session_id, created_at,
    nonce) if the token is valid. Returns None if:
    - PROXY_SESSION_SECRET is not configured (fail closed)
    - The token format is invalid
    - The HMAC signature doesn't match
    - The token has expired (older than max_age_seconds)
    """
    if not _SESSION_SECRET:
        return None  # Fail closed if secret not configured
    if not token:
        return None
    parts = token.rsplit(":", 1)
    if len(parts) != 2:
        return None
    payload_str, sig = parts
    expected = hmac.new(_SESSION_SECRET.encode(), payload_str.encode(), hashlib.sha256).hexdigest()[:16]
    if not hmac.compare_digest(sig, expected):
        return None
    try:
        payload = json.loads(payload_str)
        created_at = payload.get("created_at", 0)
        if time.time() - created_at > max_age_seconds:
            logger.warning("Session token expired: age=%.0fs, max=%ds",
                           time.time() - created_at, max_age_seconds)
            return None
        if created_at > time.time() + 300:
            return None
        return payload
    except (json.JSONDecodeError, ValueError, TypeError):
        return None
