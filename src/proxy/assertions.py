"""Boot-time assertions for the ChoriProx proxy process."""
import os


def is_dev_env() -> bool:
    """Return True if PROXY_ENV indicates a development environment.

    Accepts ``dev`` or ``development`` (case-insensitive). When PROXY_ENV is
    unset, defaults to ``"production"`` (i.e., returns False).

    This is the SINGLE shared helper for all dev-vs-production gating in the
    codebase. Use it instead of reading ``PROXY_ENV`` directly so the accepted
    value set stays consistent everywhere (boot assertion, canary verify,
    rate-limiter bypass, etc.).
    """
    return os.environ.get("PROXY_ENV", "production").lower() in ("dev", "development")


def assert_session_secret_configured() -> None:
    """Raise RuntimeError if PROXY_SESSION_SECRET is unset in production.

    Per spec §13 CRIT-M.1: only PROXY_ENV=dev permits an empty secret.
    In production, the secret MUST be set BEFORE the proxy process starts
    (deploy runbook requirement).
    """
    if is_dev_env():
        return  # dev mode permits empty secret
    secret = os.environ.get("PROXY_SESSION_SECRET", "")
    if not secret:
        raise RuntimeError(
            "PROXY_SESSION_SECRET is required in production. "
            "Set it in the environment before starting the proxy. "
            "For dev, set PROXY_ENV=dev."
        )
