# SOTAChannel — Slot Scheduler Singleton

import os
import time
import threading
from contextlib import contextmanager

SOTA_MAX_INPUT_CHARS = int(os.environ.get("SOTA_MODEL_CTX", "256000")) * 0.60 * 4

_sota_rate_limit_lock = threading.Lock()
_sota_next_slot = 0.0
_SOTA_MIN_INTERVAL = 1.0 / 3.0  # 3 calls/sec for StepFun
_SOTA_RETRY_DELAY = 2.5


@contextmanager
def claim_slot():
    """Slot scheduler: ensures call STARTS at least _SOTA_MIN_INTERVAL apart.

    Each caller claims a slot time, advances the next slot, then sleeps
    until its slot arrives. No connection lock — callers run concurrently
    once their slots arrive.
    """
    global _sota_next_slot
    now = time.monotonic()
    with _sota_rate_limit_lock:
        my_slot = max(now, _sota_next_slot)
        _sota_next_slot = my_slot + _SOTA_MIN_INTERVAL
    sleep_time = my_slot - time.monotonic()
    if sleep_time > 0:
        time.sleep(sleep_time)
    yield