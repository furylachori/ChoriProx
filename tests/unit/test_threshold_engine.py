"""Tests for proxy.pipeline.threshold_engine.ThresholdEngine."""
import pytest

from proxy.pipeline.threshold_engine import ThresholdEngine


class TestThresholdEngine:
    def test_fires_at_50(self):
        te = ThresholdEngine()
        assert te.check(55) == 50

    def test_fires_at_75(self):
        te = ThresholdEngine()
        te.check(55)
        assert te.check(80) == 75

    def test_fires_at_90(self):
        te = ThresholdEngine()
        te.check(55)
        te.check(80)
        assert te.check(95) == 90

    def test_fires_only_once_per_threshold(self):
        te = ThresholdEngine()
        assert te.check(55) == 50
        assert te.check(55) is None

    def test_jump_returns_highest_crossed(self):
        te = ThresholdEngine()
        assert te.check(85) == 75

    def test_reset_rearms(self):
        te = ThresholdEngine()
        te.check(95)
        te.check(95)
        te.reset()
        assert te.check(55) == 50

    def test_all_fired(self):
        te = ThresholdEngine()
        assert te.all_fired is False
        te.check(95)
        assert te.all_fired is True

    def test_highest_fired(self):
        te = ThresholdEngine()
        assert te.highest_fired is None
        te.check(55)
        assert te.highest_fired == 50
        te.check(80)
        assert te.highest_fired == 75

    def test_custom_thresholds(self):
        te = ThresholdEngine(thresholds=(25, 50))
        assert te.check(30) == 25
        assert te.check(55) == 50
        assert te.all_fired is True

    def test_thread_safety(self):
        """Concurrent check() calls don't double-fire thresholds."""
        import threading
        engine = ThresholdEngine(thresholds=(50,))
        results = []

        def check_50():
            result = engine.check(55)
            results.append(result)

        threads = [threading.Thread(target=check_50) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert results.count(50) == 1
        assert results.count(None) == 9

    def test_empty_thresholds(self):
        te = ThresholdEngine(thresholds=())
        assert te.all_fired is False
        assert te.check(100) is None
        assert te.highest_fired is None

    def test_duplicate_thresholds(self):
        te = ThresholdEngine(thresholds=(50, 50, 75))
        assert te.thresholds == [50, 75]
        assert te.check(55) == 50
        assert te.check(80) == 75
        assert te.all_fired is True
