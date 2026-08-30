# Feature Flags

## Overview

ChoriProx uses environment variables to gate a small number of features.
The full set is defined in `src/proxy/feature_flags.py`.

## Currently-defined flags

| Flag | Default | Purpose |
|------|---------|---------|
| `USE_DISPATCH_EXTRACTION` | `"1"` (ON) | Whether command/skill dispatch is handled by the extracted `proxy/pipeline/dispatch.py` module |
| `CACHE_METRICS_ENABLED`  | `"1"` (ON) | Whether prompt-cache metrics are captured in `PostProcessStage` |
| `PROXY_ENABLE_CUSTOM_SKILL_SUBAGENTS` | unset (OFF) | Deployment-wide kill switch for swarm-capable custom JSON skills. Must be exactly `"1"` for any custom skill that declares a nested `subagents` block to load; otherwise such skills are rejected at startup with a `swarm_*` code. Read once at startup (not per-request). When enabled, `PROXY_SESSION_SECRET` is also required — otherwise startup fails with `swarm_missing_secret`. Not authentication: operator-controlled deployment configuration. See internal documentation §3. |

## Historical flags (removed)

The following flags were removed in earlier refactors. They are listed here so
older log lines and external documentation can be interpreted correctly:

- `USE_PROVIDER_REGISTRY` — removed in commit `bef35bd` (cleanup of dead
  Phase 3 fallback). Provider compat now runs unconditionally via
  `apply_provider_compat()` in `pipeline/provider_compat.py:155`. Setting this
  env var has no effect.
- `USE_AGENT_ADAPTER` — never implemented; the flag was planned but the
  underlying feature shipped under a different mechanism.
- `USE_AGENT_LIFECYCLE` — same as above.

## Related documentation

- `CONFIGURATION.md` — current configuration knobs (env vars, rate limits)
- `docs/environment-variables.md` — every env var with its default and override surface
- Internal documentation — swarm-capable custom skill contract (kill switch semantics)
