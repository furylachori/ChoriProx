"""Percentage-based event trigger engine.

Fires once per threshold per lifetime. Call reset() to re-arm.
Reusable for: turn limits, token budgets, cost tracking, rate warnings.
"""
import threading
from typing import Optional, Tuple


class ThresholdEngine:
    """Tracks which percentage thresholds have been crossed.

    Each threshold fires at most once. check() returns the highest
    newly-crossed threshold, or None if no new threshold was crossed.
    """

    def __init__(self, thresholds: Tuple[int, ...] = (50, 75, 90)):
        self.thresholds = sorted(set(thresholds))
        self.fired: set = set()
        self._lock = threading.Lock()

    def check(self, current_pct: float) -> Optional[int]:
        """Return the highest uncrossed threshold that current_pct has reached.

        Returns None if no new threshold was crossed.
        Each threshold fires at most once until reset().
        All crossed thresholds are marked as fired to prevent lower ones from
        firing later when jumping across multiple thresholds.
        """
        with self._lock:
            crossed = [
                t for t in self.thresholds
                if current_pct >= t and t not in self.fired
            ]
            if crossed:
                highest = max(crossed)
                self.fired.update(crossed)
                return highest
            return None

    def reset(self):
        """Re-arm all thresholds."""
        with self._lock:
            self.fired.clear()

    @property
    def all_fired(self) -> bool:
        """True if every threshold has been crossed."""
        with self._lock:
            return len(self.thresholds) > 0 and len(self.fired) == len(self.thresholds)

    @property
    def highest_fired(self) -> Optional[int]:
        """The highest threshold that has fired, or None."""
        with self._lock:
            return max(self.fired) if self.fired else None
