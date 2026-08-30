# SOTAChannel — HTTP Client Singleton

import os
from enum import Enum
from typing import Dict, Optional

from openai import OpenAI


# ── SOTA Mode enum — single source of truth for per-mode model names and defaults ──

class SOTAMode(Enum):
    EXTRACTION = "extraction"
    SUMMARIZATION = "summarization"
    COMPRESSION = "compression"
    PLANNING = "planning"


# ── Module-level client singleton ───────────────────────────────────────────

_SOTA_CLIENT: Optional[OpenAI] = None


def get_sota_client() -> OpenAI:
    """Lazy singleton returning the shared OpenAI client for SOTAChannel.

    Created once on first call. All subsequent calls return the same instance.
    The caller (SOTAChannel) owns retry logic; this client uses max_retries=0.
    """
    global _SOTA_CLIENT
    if _SOTA_CLIENT is None:
        api_key = os.environ.get("PROXY_SOTA_API_KEY", "")
        base_url = os.environ.get(
            "PROXY_SOTA_ENDPOINT", "https://api.stepfun.ai/step_plan/v1"
        )
        timeout = int(os.environ.get("PROXY_SOTA_TIMEOUT", "180"))
        timeout = max(timeout, 180)  # floor at 180

        _SOTA_CLIENT = OpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=timeout,
            max_retries=0,
        )
    return _SOTA_CLIENT


# ── Model names per mode ────────────────────────────────────────────────────

_MODE_MODEL: Dict[SOTAMode, str] = {
    SOTAMode.EXTRACTION: os.environ.get("PROXY_SOTA_MODEL_EXTRACTION", "step-3.7-flash"),
    SOTAMode.SUMMARIZATION: os.environ.get("PROXY_SOTA_MODEL_SUMMARIZATION", "step-3.7-flash"),
    SOTAMode.COMPRESSION: os.environ.get("PROXY_SOTA_MODEL_COMPRESSION", "step-3.7-flash"),
    SOTAMode.PLANNING: os.environ.get("PROXY_SOTA_MODEL_PLANNING", "step-3.7-flash"),
}


# ── Default generation params per mode ─────────────────────────────────────

_MODE_DEFAULTS: Dict[SOTAMode, dict] = {
    SOTAMode.EXTRACTION: {"temperature": 0.2, "max_tokens": 32768},
    SOTAMode.SUMMARIZATION: {"temperature": 0.1, "max_tokens": 32768},
    SOTAMode.COMPRESSION: {"temperature": 0.2, "max_tokens": 32768},
    SOTAMode.PLANNING: {"temperature": 0.2, "max_tokens": 32768},
}