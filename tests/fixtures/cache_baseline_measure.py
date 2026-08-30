"""Wave 5 — REAL eviction-regime baseline capture for the Tier-2 cache gate.

Drives multi-turn traffic through the proxy with CacheSimulatingBackendHandler
as the upstream, using a capped-context model (context_limit=2560, safe=1920)
so V2 eviction engages within a few turns. Records the steady-state cache_read_pct.

The workload now includes a role="tool" message (turn 2) carrying a small
synthetic Tier-2 suffix so the capture pipeline fires, the pin emits, and the
strip engages. Without this, the measurement is tautological (feature never
engaged → delta is trivially zero).

Usage:
    PYTHONPATH=src python3 tests/fixtures/cache_baseline_measure.py

Env: neo4j_disabled=True, no API keys, no network. Deterministic.
"""
import json
import os
import sys
import tempfile
import threading
from contextlib import contextmanager
from http.server import HTTPServer

_REPO_ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
sys.path.insert(0, os.path.join(_REPO_ROOT, "src"))
sys.path.insert(0, _REPO_ROOT)

from tests.fixtures.cache_sim_backend import (
    CacheSimulatingBackendHandler,
    reset_cache_sim,
    steady_state_cache_read_pct,
)

import requests
from unittest.mock import MagicMock, patch


# Small synthetic Tier-2 payload (~200 bytes). Large enough to carry the
# boundary marker and instruction blocks, small enough to fit the capped model.
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

_INSTRUCTION_SUFFIX = "<system-reminder>\nInstructions from: /repo/proj/sub/AGENTS.md\n# sub convention\nInstructions from: /repo/proj/AGENTS.md\n# parent convention\n</system-reminder>"


def _make_proxy_models_config(backend_url):
    return {
        "models": {
            "tiny": {
                "backend": backend_url,
                "tool_parser": "openai",
                "context_limit": 2560,
                "name": "Tiny Capped Model",
                "api_key_env": "TEST_API_KEY",
                "backend_model": "test/tiny",
            },
        },
        "default": {
            "backend": backend_url,
            "tool_parser": "openai",
            "context_limit": 2560,
        },
    }


@contextmanager
def _baseline_session():
    """Set up proxy + mock backend, yield (session, captured_bodies)."""
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

    _tmpdir = tempfile.mkdtemp(prefix="tier2_baseline_")
    _db_path = os.path.join(_tmpdir, "memory.db")
    _proxy_models_path = os.path.join(_tmpdir, "proxy_models.json")
    saved_env["PROXY_TEST_DB"] = os.environ.get("PROXY_TEST_DB")
    os.environ["PROXY_TEST_DB"] = _db_path

    reset_cache_sim()
    _captured_bodies = []

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

        yield client, _captured_bodies
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


def main():
    with _baseline_session() as (client, captured):
        history = []
        num_turns = 30

        # Engagement tracking
        pin_emitted = False
        strip_applied = False

        for i in range(1, num_turns + 1):
            content = _make_user_message(i)
            history.append({"role": "user", "content": content})

            # Insert the tooled exchange at turn 2 (after the user message).
            # We inject an assistant message with a tool_call and a matching
            # role="tool" result carrying the Tier-2 suffix. This fires the
            # capture pipeline on turn 2 and every subsequent turn (opencode
            # re-sends full history).
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

            body = {"messages": list(history), "model": "tiny", "stream": False}
            resp = client.post("/v1/chat/completions", json=body)

            if resp.status_code != 200:
                print(f"ERROR: turn {i} returned status {resp.status_code}")
                print(resp.data.decode()[:500])
                return

            # Capture the outbound request body sent to the mock backend
            # by inspecting what the cache sim received.
            raw_resp = resp.get_json()
            if raw_resp and "choices" in raw_resp and raw_resp["choices"]:
                msg = raw_resp["choices"][0].get("message", {})
                if msg:
                    history.append(msg)

        # After the loop: check engagement.
        import proxy.state as state

        capture_fired = bool(state.instruction_files_by_workspace)
        print(f"\nENGAGEMENT CHECK:")
        print(f"  capture fired (instruction_files_by_workspace non-empty): {capture_fired}")
        if capture_fired:
            for key, inner in state.instruction_files_by_workspace.items():
                print(f"    workspace={key!r}: {list(inner.keys())}")

        # Check pin emission and strip by inspecting captured raw request bodies.
        raw_bodies = CacheSimulatingBackendHandler._captured_raw_bodies
        for body_bytes in raw_bodies:
            try:
                outbound = json.loads(body_bytes)
            except (json.JSONDecodeError, TypeError):
                continue
            outbound_msgs = outbound.get("messages", [])
            for msg in outbound_msgs:
                role = msg.get("role", "")
                content = msg.get("content", "")
                if role == "system" and "CLIENT_SUB_INSTRUCTIONS" in content:
                    pin_emitted = True
                if role == "tool" and msg.get("tool_call_id") == "tc-cache-tier2":
                    if _INSTRUCTION_SUFFIX not in content:
                        strip_applied = True
        print(f"  pin emitted (<CLIENT_SUB_INSTRUCTIONS> in outbound): {pin_emitted}")
        print(f"  strip applied (tool msg lacks <system-reminder> suffix): {strip_applied}")

        reads = CacheSimulatingBackendHandler._cache_reads
        totals = CacheSimulatingBackendHandler._totals

        print("\nPer-turn cache_read / total series:")
        for idx, (r, t) in enumerate(zip(reads, totals), start=1):
            pct = (r / t * 100.0) if t > 0 else 0.0
            print(f"  turn {idx:2d}: cache_read={r:6d}  total={t:6d}  pct={pct:5.1f}%")

        # Detect pass-through regime (tautology check)
        is_passthrough = True
        for idx in range(1, len(reads)):
            if reads[idx] != totals[idx - 1]:
                is_passthrough = False
                break

        if is_passthrough:
            print("\nWARNING: pass-through regime detected (cache_read[turnN] == total[turnN-1]).")
            print("This means V2 eviction never engaged. Baseline is TAUTOLOGICAL.")
            print("The §9.9 #4 gate requires the eviction regime.")
        else:
            print("\nEviction regime confirmed: cache_read != previous total on some turns.")

        pct = steady_state_cache_read_pct(turns_to_drop=3)
        steady_turns = len(reads) - 3
        print(f"\nRESULT: cache_read_pct={pct:.1f}% (steady-state, turns={steady_turns})")
        print(f"ENGAGEMENT: capture={capture_fired}, pin={pin_emitted}, strip={strip_applied}")

        import sys as _sys
        _sys.stdout.flush()
        os._exit(0)


if __name__ == "__main__":
    main()
