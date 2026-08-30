"""SecurityStage — malicious content detection."""

import logging
from typing import Any, Optional

from proxy.pipeline.exceptions import StageAbort

logger = logging.getLogger("proxy.pipeline.stages")


class SecurityStage:
    """Malicious content detection.

    Scans recent messages for malicious patterns and blocks if detected.
    """

    def execute(self, ctx: Any) -> Optional[dict]:
        try:
            from proxy.auth import detect_malicious_content
        except ImportError:
            logger.exception("SecurityStage: Failed to import detect_malicious_content — security scanning DISABLED")
            return None

        if not ctx.messages:
            return None

        messages = ctx.body.get("messages", [])
        for msg in messages:
            if msg.get("role") == "user":
                content = msg.get("content", "")
                if content:
                    threats = detect_malicious_content(content)
                    if threats:
                        logger.warning(
                            "[SECURITY] Blocked patterns detected: %s for session=%s",
                            threats, ctx.session_id,
                            extra={"request_id": getattr(ctx, 'request_id', '?'), "session_id": ctx.session_id},
                        )
                        raise StageAbort({
                            "error": {
                                "message": f"Request blocked: malicious content detected ({', '.join(threats)})",
                                "type": "security_violation",
                                "code": 400,
                            }
                        })

        return None
