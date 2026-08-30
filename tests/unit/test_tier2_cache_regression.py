"""§9.9 #4 regression test — Tier-2 cache eviction-regime baseline gate.

This is a ONE-SIDED contract. The historical measurement at dc5e9d2 was
53.5% cache_read_pct in the eviction regime. The test pins a minimum
acceptance threshold of (53.5% - 3.0pp) = 50.5%. Improvements ABOVE 53.5%
are ACCEPTABLE; regressions more than 3pp BELOW 53.5% are rejected.

Uses the same capped-context model (context_limit=2560,
safe_context_tokens=1920) that forces V2 eviction within a few turns.

The workload includes a role="tool" message (turn 2) carrying a small
synthetic Tier-2 suffix. This engages the capture + pin + strip pipeline,
making the measurement non-tautological. Several anti-tautology guards
assert: schema-readiness (canonical SQLite tables present), capture fired
(both expected instruction paths recorded), pin emitted (system message
carrying CLIENT_SUB_INSTRUCTIONS with both sub- and parent-instruction
bodies), strip applied (tool message no longer carries the
<system-reminder> suffix while preserving the file body), eviction regime
engaged (collapse with subsequent recovery), and metric collection shape
(non-negative ints, cache_read <= total).

No client tools are advertised. An absent tool payload has zero rendered
schema cost, so the capped context reaches the real eviction regime
without an artificial tool definition.

This test is deterministic: CacheSimulatingBackendHandler is a prefix-
simulating mock with no randomness.
"""
import json
import os
import sys
import tempfile
import threading
from http.server import HTTPServer
from unittest.mock import MagicMock, patch

import pytest
import requests

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from tests.fixtures.cache_sim_backend import (
    CacheSimulatingBackendHandler,
    reset_cache_sim,
    steady_state_cache_read_pct,
)

# Historical dc5e9d2 measurement with the identical harness
# (context_limit=2560). 53.5% is the historical value; we accept any
# measured value that does not regress more than 3.0pp below it.
_HISTORICAL_CACHE_READ_PCT = 53.5
_ALLOWED_REGRESSION_PP = 3.0
_CONTEXT_LIMIT = 2560
_NUM_TURNS = 30

# Small synthetic Tier-2 payload (~200 bytes) — engages the feature without
# dominating the capped model's budget.
_TIER2_TOOL_CONTENT = (
    "<path>/repo/proj/sub/r.py</path><type>file</type><content>\n"
    "def r(): pass\n"
    "</content><system-reminder>\n"
    "Instructions from: /repo/proj/sub/AGENTS.md\n"
    "# sub convention\n"
    "Instructions from: /repo/proj/AGENTS.md\n"
    "# parent convention\n"
    "</system-reminder>"
)
_INSTRUCTION_SUFFIX = (
    "<system-reminder>\n"
    "Instructions from: /repo/proj/sub/AGENTS.md\n"
    "# sub convention\n"
    "Instructions from: /repo/proj/AGENTS.md\n"
    "# parent convention\n"
    "</system-reminder>"
)


@pytest.fixture(autouse=True)
def _clean_cache_sim():
    reset_cache_sim()
    yield
    reset_cache_sim()


def _make_user_message(turn_idx):
    return (
        f"Let us discuss the implementation details for turn {turn_idx}. "
        "We need to carefully consider the architectural implications of "
        "the caching subsystem and how it interacts with the context window "
        "management layer. The proxy must maintain coherent state across "
        "multiple concurrent sessions while ensuring that the eviction "
        "policy does not discard entries that are still actively referenced "
        "by in-flight requests. Please analyze the trade-offs between LRU "
        "and LFU eviction strategies in this context."
    )


def _make_proxy_models_config(backend_url):
    return {
        "models": {
            "tiny": {
                "backend": backend_url,
                "tool_parser": "openai",
                "context_limit": _CONTEXT_LIMIT,
                "name": "Tiny Capped Model",
                "api_key_env": "TEST_API_KEY",
                "backend_model": "test/tiny",
            },
        },
        "default": {
            "backend": backend_url,
            "tool_parser": "openai",
            "context_limit": _CONTEXT_LIMIT,
        },
    }


@pytest.mark.integration
class TestTier2CacheRegression:
    """§9.9 #4 ONE-SIDED contract: cache_read_pct must not fall more than
    3pp below the historical dc5e9d2 measurement (i.e., >=50.5%).
    Improvements above the 53.5% historical value are acceptable.
    """

    def test_feature_engaged_and_cache_within_tolerance(self):
        """Run the capped-context harness with a tooled workload.

        Asserts (in order):
        1. Schema-readiness (canonical SQLite tables present).
        2. Capture fired (BOTH expected instruction paths recorded).
        3. Pin emitted with sub- and parent-instruction bodies.
        4. Strip applied to tool message, preserving the file body.
        5. Metric collection shape (ints, cache_read <= total, full turn count).
         6. Eviction regime engaged (collapse with subsequent recovery).
        7. cache_read_pct >= historical (53.5% - 3pp) — one-sided gate.
        """


        saved_env = {}
        env_overrides = [
            ("PROXY_ENV", "dev"),
            ("PROXY_SESSION_SECRET", "baseline-secret"),
            ("PROXY_API_KEY", ""),
            ("PROXY_VFS_ENABLED", "0"),
            ("PROXY_LOG_FORMAT", "text"),
            ("PROXY_DEBUG", ""),
            ("TEST_API_KEY", "sk-test-secret-key-12345"),
            ("PROXY_SOTA_API_KEY", ""),
            ("PROXY_SOTA_ENDPOINT", ""),
            ("PROXY_SOTA_MODEL", ""),
        ]
        for key, value in env_overrides:
            saved_env[key] = os.environ.get(key)
            os.environ[key] = value

        _tmpdir = tempfile.mkdtemp(prefix="tier2_regression_")
        _db_path = os.path.join(_tmpdir, "memory.db")
        _proxy_models_path = os.path.join(_tmpdir, "proxy_models.json")
        saved_env["PROXY_TEST_DB"] = os.environ.get("PROXY_TEST_DB")
        os.environ["PROXY_TEST_DB"] = _db_path

        from memory.db import init_db as init_sqlite_db

        assert init_sqlite_db(_db_path) is True

        from memory.db import get_connection
        from memory.db_pool import return_connection

        conn = get_connection(_db_path)
        try:
            tables = {
                row[0]
                for row in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type = 'table'"
                ).fetchall()
            }
        finally:
            return_connection(conn)

        assert {
            "messages",
            "message_summaries",
            "conversation_summaries",
            "sessions",
        } <= tables, (
            "Schema-readiness check failed; the canonical tables required for "
            "PostProcessStage._persist_messages are missing."
        )

        reset_cache_sim()
        server = HTTPServer(("127.0.0.1", 0), CacheSimulatingBackendHandler)
        port = server.server_address[1]
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()

        backend_url = f"http://127.0.0.1:{port}/v1"
        cfg = _make_proxy_models_config(backend_url)
        with open(_proxy_models_path, "w") as f:
            json.dump(cfg, f)

        patches = []
        try:
            patches.append(patch("memory.graph.Neo4jClient", MagicMock()))
            patches.append(patch("memory.task_graph.Neo4jClient", MagicMock()))
            patches.append(patch("memory.sync.init_db", MagicMock(return_value=None)))
            patches.append(patch(
                "memory.query_executor.get_shared_executor_from_env",
                MagicMock(return_value=None),
            ))
            patches.append(patch(
                "memory.workspace_persistence._get_executor",
                MagicMock(return_value=None),
            ))
            for p in patches:
                p.start()

            import qwen_proxy
            from clients.model_registry import ModelRegistry

            reg = ModelRegistry(config_path="proxy_models.json", base_dir=_tmpdir)
            patches.append(patch("qwen_proxy._model_registry", reg))
            patches.append(patch("proxy.app._model_registry", reg))
            patches.append(patch("proxy.model_registry._model_registry", reg))
            patches.append(patch("qwen_proxy.USE_ADAPTERS", True))
            patches.append(patch("proxy.state._HTTP_SESSION", requests.Session()))
            patches.append(patch("qwen_proxy._HTTP_SESSION", requests.Session()))
            for p in patches[-6:]:
                p.start()

            qwen_proxy.app.config["TESTING"] = True
            client = qwen_proxy.app.test_client()

            history = []
            for i in range(1, _NUM_TURNS + 1):
                content = _make_user_message(i)
                history.append({"role": "user", "content": content})

                # Inject tooled exchange at turn 2 to engage the Tier-2 pipeline.
                if i == 2:
                    history.append({
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [{
                            "id": "tc-cache-tier2",
                            "type": "function",
                            "function": {
                                "name": "read",
                                "arguments": '{"path": "/repo/proj/sub/r.py"}',
                            },
                        }],
                    })
                    history.append({
                        "role": "tool",
                        "tool_call_id": "tc-cache-tier2",
                        "content": _TIER2_TOOL_CONTENT,
                    })

                body = {
                    "messages": list(history),
                    "model": "tiny",
                    "stream": False,
                }
                resp = client.post("/v1/chat/completions", json=body)
                assert resp.status_code == 200, (
                    f"turn {i} returned status {resp.status_code}: {resp.data.decode()[:200]}"
                )
                resp_body = resp.get_json()
                assert isinstance(resp_body, dict), (
                    f"turn {i} returned non-object JSON: {resp_body!r}"
                )
                choices = resp_body.get("choices")
                assert isinstance(choices, list) and choices, (
                    f"turn {i} returned no choices: {resp_body!r}"
                )
                message = choices[0].get("message")
                assert isinstance(message, dict), (
                    f"turn {i} returned no assistant message: {choices[0]!r}"
                )
                history.append(message)

            reads = CacheSimulatingBackendHandler._cache_reads
            totals = CacheSimulatingBackendHandler._totals

            assert len(reads) == _NUM_TURNS
            assert len(totals) == _NUM_TURNS
            assert len(reads) == len(totals)
            assert all(isinstance(value, int) and value >= 0 for value in reads)
            assert all(isinstance(value, int) and value > 0 for value in totals)
            assert all(
                cache_read <= total
                for cache_read, total in zip(reads, totals)
            )

            # === Engagement assertion #1: capture fired (BOTH expected paths) ===
            import proxy.state as state
            captured_paths = set()
            for _key, inner in state.instruction_files_by_workspace.items():
                if not isinstance(inner, dict):
                    continue
                captured_paths.update(inner.keys())

            assert captured_paths, (
                "Capture never fired — instruction_files_by_workspace is empty. "
                "The Tier-2 feature is not engaged; regression test is tautological."
            )
            assert "/repo/proj/sub/AGENTS.md" in captured_paths, (
                f"Capture missing expected sub-instruction path. "
                f"Captured: {sorted(captured_paths)!r}"
            )
            assert "/repo/proj/AGENTS.md" in captured_paths, (
                f"Capture missing expected parent-instruction path. "
                f"Captured: {sorted(captured_paths)!r}"
            )

            # === Engagement assertions #2 & #3: pin emitted + strip applied ===
            pin_emitted = False
            pin_contains_sub_instruction = False
            pin_contains_parent_instruction = False
            strip_applied = False
            stripped_tool_content_preserved = False
            for body_bytes in CacheSimulatingBackendHandler._captured_raw_bodies:
                try:
                    outbound = json.loads(body_bytes)
                except (json.JSONDecodeError, TypeError):
                    continue
                for msg in outbound.get("messages", []):
                    role = msg.get("role", "")
                    content = msg.get("content", "")
                    if role == "system" and "CLIENT_SUB_INSTRUCTIONS" in content:
                        pin_emitted = True
                        if "# sub convention" in content:
                            pin_contains_sub_instruction = True
                        if "# parent convention" in content:
                            pin_contains_parent_instruction = True
                    if role == "tool" and msg.get("tool_call_id") == "tc-cache-tier2":
                        if _INSTRUCTION_SUFFIX not in content:
                            strip_applied = True
                        if "def r(): pass" in content:
                            stripped_tool_content_preserved = True

            assert pin_emitted, (
                "Pin never emitted — no outbound request body contained "
                "<CLIENT_SUB_INSTRUCTIONS>."
            )
            assert pin_contains_sub_instruction, (
                "Pin emitted but missing the sub-folder instruction body "
                "('# sub convention')."
            )
            assert pin_contains_parent_instruction, (
                "Pin emitted but missing the parent instruction body "
                "('# parent convention')."
            )
            assert strip_applied, (
                "Strip never applied — tool message still carries the "
                "<system-reminder> suffix in all outbound bodies."
            )
            assert stripped_tool_content_preserved, (
                "Strip applied, but tool body lost the file content "
                "('def r(): pass' is no longer present). The strip must "
                "remove only the suffix, not the body."
            )

            # === Eviction regime check (collapse + recovery) ===
            turns_to_drop = 3

            collapse_indices = [
                idx
                for idx in range(turns_to_drop + 1, len(reads))
                if (
                    reads[idx] < reads[idx - 1]
                    and reads[idx] <= totals[idx - 1] * 0.5
                )
            ]
            assert collapse_indices, (
                "No material cache-read collapse occurred in the steady-state "
                "window; the capped-context workload did not exercise eviction."
            )
            assert any(
                any(reads[later] > reads[idx] for later in range(idx + 1, len(reads)))
                for idx in collapse_indices
            ), "Cache reuse did not recover after eviction."

            # === Gate (one-sided): cache_read_pct must not regress >3pp below
            # the historical dc5e9d2 measurement. Improvements above the
            # historical value are acceptable. ===
            pct = steady_state_cache_read_pct(turns_to_drop=turns_to_drop)
            minimum_acceptable_pct = (
                _HISTORICAL_CACHE_READ_PCT - _ALLOWED_REGRESSION_PP
            )
            assert pct >= minimum_acceptable_pct, (
                f"cache_read_pct={pct:.1f}% is below the minimum accepted "
                f"{minimum_acceptable_pct:.1f}% derived from historical "
                f"dc5e9d2 measurement {_HISTORICAL_CACHE_READ_PCT:.1f}% "
                f"with {_ALLOWED_REGRESSION_PP:.1f}pp allowed regression."
            )

            # === Anti-tautology guards are folded into the schema-readiness,
            # capture, pin-content, strip-body-preservation, metric-collection-
            # shape, and collapse+recovery assertions above. No historic
            # baseline series comparison remains. ===
        finally:
            server.shutdown()
            server.server_close()
            for p in reversed(patches):
                try:
                    p.stop()
                except RuntimeError:
                    pass
            for key, value in saved_env.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value
            import shutil
            shutil.rmtree(_tmpdir, ignore_errors=True)
