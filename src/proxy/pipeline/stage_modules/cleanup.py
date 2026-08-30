"""CleanupStage — session cleanup + request counter."""

import logging
from typing import Any, Optional

logger = logging.getLogger("proxy.pipeline.stages")


class CleanupStage:
    """Session cleanup + request counter.

    Algorithm:
    1. Increment request counter: state.request_counter += 1
    2. Call cleanup_old_sessions() if counter % 100 == 0
    3. Call cleanup_stale_ticket_modes() from session_state
    """

    def execute(self, ctx: Any) -> Optional[dict]:
        import proxy.state as state

        should_cleanup = False
        with state.request_counter_lock:
            state.request_counter += 1
            CLEANUP_INTERVAL = getattr(state, "CLEANUP_INTERVAL", 100)
            should_cleanup = state.request_counter % CLEANUP_INTERVAL == 0

        if should_cleanup:
            try:
                from proxy.session import cleanup_old_sessions

                cleanup_old_sessions()
            except ImportError:
                logger.exception("CleanupStage: Failed to import cleanup_old_sessions — periodic session cleanup DISABLED")
            try:
                from memory.db import cleanup_expired_messages

                cleanup_expired_messages()
            except ImportError:
                logger.exception("CleanupStage: Failed to import cleanup_expired_messages — message expiration cleanup DISABLED")
            try:
                from proxy.state import cleanup_stale_workspaces
                n = cleanup_stale_workspaces()
                if n > 0:
                    logger.info("CleanupStage: removed %d stale workspace sessions", n)
            except ImportError:
                logger.exception("CleanupStage: Failed to import cleanup_stale_workspaces — workspace cleanup DISABLED")

        try:
            from proxy.agent.agent_registry import registry as agent_registry

            agent_registry.cleanup_stale_ticket_modes()
        except ImportError:
            logger.exception("CleanupStage: Failed to import agent_registry cleanup methods — stale ticket mode cleanup DISABLED")

        return None
