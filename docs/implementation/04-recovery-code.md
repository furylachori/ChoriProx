# Block: Recovery Code

> Part of [IMPLEMENTATION_SPEC](../IMPLEMENTATION_SPEC.md). See also [Spec](../spec/02-architecture.md)

## 5. Proxy Recovery Implementation

The recovery system was originally designed as four layers (Layer 1 keyword-based
tool filtering was never implemented). After Tier 2.1 (commit `3d64b13`)
deleted the malformed-call auto-fix path (`smart_proxy_validation.py`,
`session_retry_manager.py`, `handle_tool_validation_error`), the only surviving
implementation is Layer 4 — context-aware tool delivery — collapsed into a
single function in `proxy_recovery_system.py`.

### 5.1 — 5.3 Deleted (Tier 2.1, commit 3d64b13)

The following modules/functions were removed as dead code:

- `src/smart_proxy_validation.py` — Layer 2: malformed-call detection and auto-fix.
- `src/session_retry_manager.py` — Layer 3: progressive escalation, retry budget,
  circuit breaker, TTL eviction.
- `handle_tool_validation_error()` in `src/proxy_recovery_system.py` — the only
  caller of the two deleted modules.
- `auto_fix_missing_description()` in `src/tool_optimizer.py:198-214` — orphaned
  after `handle_tool_validation_error` was deleted (Tier 2.2).

### 5.3 Layer 4 (survivor): Context-Aware Tool Delivery

**File:** `src/tool_optimizer.py` (197 lines post-Tier-2.2)

```python
OPENTOOLS_TO_CANONICAL = {
    "bash": "run_shell_command", "read": "read_file", "write": "write_file",
    "edit": "edit_file", "glob": "file_glob", "webfetch": "web_fetch",
}

def generate_optimized_tool_description(tool_name: str) -> dict
```

Generates LLM-optimized tool descriptions (80-90% token reduction).
Maps open tool names to canonical names for description lookup.

### 5.4 Recovery System Orchestration

**File:** `src/proxy_recovery_system.py` (38 lines post-Tier-2.2)

```python
def apply_proxy_recovery_system(body: dict, session_id: str) -> dict
    # Walks body["tools"]. For each tool that lacks a `parameters` block,
    # replaces it with the optimized schema from generate_optimized_tool_description.
    # Tools with parameters (e.g. memory tools) pass through unchanged.
```

Called from `proxy.pipeline.stages.ProviderCompatStage.execute` at
`src/proxy/pipeline/stages.py:868-872`. Module-level imported by
`src/proxy/app.py:44` for legacy re-export (tests patch this attribute).

Behavioral coverage: `tests/unit/test_tool_optimizer_unit.py::TestApplyProxyRecoverySystem`
(restored in Tier 2.2).