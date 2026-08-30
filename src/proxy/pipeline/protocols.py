"""Pipeline stage protocol."""

from typing import Any, Optional, Protocol


class PipelineStage(Protocol):
    """Protocol that all pipeline stages must implement."""

    def execute(self, ctx: Any) -> Optional[Any]:
        """Execute the stage.

        Args:
            ctx: RequestContext with session, body, messages, etc.

        Returns:
            dict/Response to short-circuit and use as response,
            or None to continue to next stage.

        Raises:
            StageAbort: When a stage produces a Flask Response or other
                        non-serializable response object that must bypass
                        downstream JSON serialization.
        """
        ...
