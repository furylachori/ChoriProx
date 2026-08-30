"""Pipeline exceptions."""


class StageAbort(Exception):
    """Raised by a stage when it has produced a response and the pipeline should stop.

    Not an error — used for dispatch handlers that fully handle the request.
    """

    def __init__(self, response_data):
        self.response_data = response_data
        super().__init__(f"StageAbort({response_data!r})")
