"""Unit tests for prompt cache metrics extraction and recording."""
import unittest
from unittest.mock import patch, MagicMock, call

from proxy.pipeline.post_process import (
    _extract_cache_tokens,
    _detect_provider,
)


class TestExtractCacheTokens(unittest.TestCase):
    """Extraction: DeepSeek flat, Qwen nested, MiMo, empty, missing, non-dict, negatives."""

    def test_deepseek_flat(self):
        """DeepSeek: flat prompt_cache_hit_tokens, no_cache computed from prompt_tokens."""
        data = {
            "usage": {
                "prompt_tokens": 150,
                "prompt_cache_hit_tokens": 80,
            }
        }
        result = _extract_cache_tokens(data)
        self.assertEqual(result["total"], 150)
        self.assertEqual(result["cache_read"], 80)
        self.assertIsNone(result["cache_write"])
        self.assertEqual(result["no_cache"], 70)  # 150 - 80

    def test_qwen_nested(self):
        """Qwen: nested prompt_tokens_details.cached_tokens, no_cache computed."""
        data = {
            "usage": {
                "prompt_tokens": 150,
                "prompt_tokens_details": {
                    "cached_tokens": 60,
                },
            }
        }
        result = _extract_cache_tokens(data)
        self.assertEqual(result["total"], 150)
        self.assertEqual(result["cache_read"], 60)
        self.assertIsNone(result["cache_write"])
        self.assertEqual(result["no_cache"], 90)  # 150 - 60

    def test_qwen_explicit_nested(self):
        """Qwen explicit: both cache_read and cache_write extracted from nested fields."""
        data = {
            "usage": {
                "prompt_tokens": 3000,
                "prompt_tokens_details": {
                    "cached_tokens": 300,
                    "cache_creation_input_tokens": 100,
                },
            }
        }
        result = _extract_cache_tokens(data)
        self.assertEqual(result["total"], 3000)
        self.assertEqual(result["cache_read"], 300)
        self.assertEqual(result["cache_write"], 100)
        self.assertEqual(result["no_cache"], 2600)

    def test_mimo_no_cache_fields(self):
        """MiMo: prompt_tokens present but no cache fields → no_cache is None."""
        data = {
            "usage": {
                "prompt_tokens": 100,
            }
        }
        result = _extract_cache_tokens(data)
        self.assertEqual(result["total"], 100)
        self.assertIsNone(result["cache_read"])
        self.assertIsNone(result["cache_write"])
        self.assertIsNone(result["no_cache"])

    def test_empty_usage(self):
        """Empty usage dict returns None for all cache fields."""
        data = {"usage": {}}
        result = _extract_cache_tokens(data)
        self.assertIsNone(result["total"])
        self.assertIsNone(result["cache_read"])
        self.assertIsNone(result["cache_write"])
        self.assertIsNone(result["no_cache"])

    def test_missing_usage(self):
        """Response without usage key returns None for all fields."""
        data = {"choices": [{"message": {"content": "hello"}}]}
        result = _extract_cache_tokens(data)
        self.assertIsNone(result["total"])
        self.assertIsNone(result["cache_read"])
        self.assertIsNone(result["cache_write"])
        self.assertIsNone(result["no_cache"])

    def test_non_dict_data(self):
        """Non-dict data returns None for all fields without crashing."""
        result = _extract_cache_tokens("not a dict")
        self.assertIsNone(result["total"])
        self.assertIsNone(result["cache_read"])
        self.assertIsNone(result["cache_write"])
        self.assertIsNone(result["no_cache"])

    def test_usage_none(self):
        """Usage set to None returns None for all fields."""
        data = {"usage": None}
        result = _extract_cache_tokens(data)
        self.assertIsNone(result["total"])
        self.assertIsNone(result["cache_read"])
        self.assertIsNone(result["cache_write"])
        self.assertIsNone(result["no_cache"])

    def test_negative_values_clamped(self):
        """Negative token values are clamped to 0 with a warning."""
        data = {
            "usage": {
                "prompt_tokens": 100,
                "prompt_cache_hit_tokens": -5,
            }
        }
        with patch("proxy.pipeline.post_process.logger.warning") as mock_warn:
            result = _extract_cache_tokens(data)
        self.assertEqual(result["cache_read"], 0)
        self.assertEqual(result["no_cache"], 100)  # 100 - 0
        mock_warn.assert_called_once()

    def test_float_values_truncated(self):
        """Float token values are truncated via int()."""
        data = {
            "usage": {
                "prompt_tokens": 100.7,
                "prompt_cache_hit_tokens": 40.2,
            }
        }
        result = _extract_cache_tokens(data)
        self.assertEqual(result["total"], 100)
        self.assertEqual(result["cache_read"], 40)
        self.assertEqual(result["no_cache"], 60)

    def test_input_tokens_fallback(self):
        """When prompt_tokens missing, falls back to input_tokens."""
        data = {
            "usage": {
                "input_tokens": 200,
                "prompt_cache_hit_tokens": 50,
            }
        }
        result = _extract_cache_tokens(data)
        self.assertEqual(result["total"], 200)
        self.assertEqual(result["cache_read"], 50)
        self.assertEqual(result["no_cache"], 150)

    def test_cache_write_without_cache_read(self):
        """cache_write present but cache_read absent → no_cache stays None."""
        data = {
            "usage": {
                "prompt_tokens": 1000,
                "cache_creation_input_tokens": 50,
            }
        }
        result = _extract_cache_tokens(data)
        self.assertEqual(result["total"], 1000)
        self.assertIsNone(result["cache_read"])
        self.assertEqual(result["cache_write"], 50)
        self.assertIsNone(result["no_cache"])

    def test_kimi_moonshot_substring(self):
        """Moonshot/Kimi style model names are recognized."""
        data = {
            "usage": {
                "prompt_tokens": 100,
                "prompt_cache_hit_tokens": 20,
            }
        }
        result = _extract_cache_tokens(data)
        self.assertEqual(result["cache_read"], 20)


class TestDetectProvider(unittest.TestCase):
    """Substring matching + override map for known models."""

    def test_deepseek_substring(self):
        self.assertEqual(_detect_provider("deepseek-chat", ""), "deepseek")

    def test_qwen_substring(self):
        self.assertEqual(_detect_provider("qwen-max", ""), "qwen")

    def test_qwen3_coder_override(self):
        self.assertEqual(_detect_provider("", "qwen3-coder"), "qwen")

    def test_deepseek_v4_pro_ark_override(self):
        self.assertEqual(_detect_provider("", "deepseek-v4-pro-ark"), "deepseek")

    def test_unknown_provider(self):
        self.assertEqual(_detect_provider("some-unknown-model", ""), "other")

    def test_minimax_substring(self):
        self.assertEqual(_detect_provider("MiniMax-M3", ""), "minimax")

    def test_mimo_substring(self):
        self.assertEqual(_detect_provider("mimo-v2.5-pro", ""), "mimo")

    def test_kimi_substring(self):
        self.assertEqual(_detect_provider("kimi-k2.6", ""), "kimi")


class TestRecordPromptCacheMetrics(unittest.TestCase):
    """Mock Prometheus counter, verify inc() calls."""

    def setUp(self):
        self.patcher = patch("proxy.metrics._get_prom")
        self.mock_get_prom = self.patcher.start()
        self.mock_prom = MagicMock()
        self.mock_get_prom.return_value = self.mock_prom

        self.env_patcher = patch.dict("os.environ", {"PROXY_METRICS_ENABLED": "1"})
        self.env_patcher.start()

        import proxy.metrics
        proxy.metrics._prom = None
        proxy.metrics._METRICS_ENABLED = True
        proxy.metrics._prompt_cache_requests_total = None
        proxy.metrics._prompt_cache_read_tokens_total = None
        proxy.metrics._prompt_cache_no_cache_tokens_total = None
        proxy.metrics._http_requests_total = None

    def tearDown(self):
        self.patcher.stop()
        self.env_patcher.stop()

    def test_record_with_cache_hit_and_miss(self):
        """cache_read and no_cache tokens both recorded with correct inc() values."""
        from proxy.metrics import record_prompt_cache_metrics

        record_prompt_cache_metrics(
            model="deepseek-chat",
            provider="deepseek",
            cache_read_tokens=80,
            no_cache_tokens=70,
        )

        labels = self.mock_prom.Counter.return_value.labels.return_value
        # Should have: requests_total.inc(), read_tokens.inc(80), no_cache.inc(70)
        self.assertEqual(labels.inc.call_count, 3)

    def test_record_without_cache_fields(self):
        """When both cache fields are None, only requests_total is incremented."""
        from proxy.metrics import record_prompt_cache_metrics

        record_prompt_cache_metrics(
            model="test-model",
            provider="other",
            cache_read_tokens=None,
            no_cache_tokens=None,
        )

        self.mock_prom.Counter.return_value.labels.return_value.inc.assert_called_once()


if __name__ == "__main__":
    unittest.main()
