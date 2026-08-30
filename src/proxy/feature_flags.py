"""Feature flags for incremental architecture refactor rollout.

Flag defaults vary — see per-flag comments below. Most flags default to ON
(Phase 2B style). Set environment variable to "0" to disable a flag.
Each flag controls a single refactor phase.
"""
import os

# Phase 3: Dispatch extraction
USE_DISPATCH_EXTRACTION = os.environ.get("USE_DISPATCH_EXTRACTION", "1") == "1"

# Phase 0: Prompt cache metrics
CACHE_METRICS_ENABLED = os.environ.get("PROXY_CACHE_METRICS_ENABLED", "1") != "0"
