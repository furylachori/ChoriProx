"""Base class for pipeline stages that should never crash the request."""
import logging

logger = logging.getLogger(__name__)


class ConditionalStage:
    """Pipeline stage with safety guards.

    Subclass and override should_run() + _execute().
    - should_run() returns False → stage is skipped (return None)
    - _execute() raises → caught, logged as warning, return None
    - _execute() returns None → pipeline continues
    - _execute() returns DispatchResult → pipeline short-circuits
    """

    def execute(self, ctx):
        try:
            if not self.should_run(ctx):
                return None
            return self._execute(ctx)
        except Exception as e:
            logger.warning(
                "[%s] failed (non-fatal): %s",
                self.__class__.__name__, e,
                extra={"request_id": getattr(ctx, "request_id", "?")},
            )
            return None

    def should_run(self, ctx) -> bool:
        return True

    def _execute(self, ctx):
        raise NotImplementedError
