# ============================================================================
# NOT THE CANONICAL LAUNCH PATH (F7, docs/_analysis/four-day-change-audit-2026-07-17.md).
# The checked-in entrypoint is Flask's threaded server:
#     src/qwen_proxy.py:57 -> app.run(host="0.0.0.0", port=6666, threaded=True)
# gunicorn is NOT in requirements.txt and is not used by start.sh. This file is
# retained only as a reference for a possible future gunicorn deployment.
#
# IF gunicorn is ever adopted: the sync worker below with timeout=300 will REAP
# streaming completions that exceed 300s, killing the completion-coordinator
# worker mid-request (upstream read timeout is 1800s x up to 3 attempts). Before
# any production use you MUST switch to a gthread/threaded worker and raise
# `timeout` above the maximum completion lifetime.
# ============================================================================
"""Gunicorn production config for ChoriProx.

Auto-loaded by `gunicorn qwen_proxy:app` (the default — gunicorn reads
``gunicorn.conf.py`` from the working directory). Wires the canonical
proxy startup so the rate-limiter state is hydrated and the skill
registry is eagerly initialized BEFORE workers accept requests.

Order of hooks (gunicorn boot sequence):

  on_starting        — master process, before workers fork. Hydrates
                       in-memory state in the master so workers inherit
                       the populated state via fork copy-on-write.
  when_ready         — master, after the listening socket is bound.
  pre_fork           — master, before each worker fork.
  post_fork          — worker, immediately after fork.
  post_worker_init   — worker, after the WSGI app is loaded.

The hydration is split across ``on_starting`` (master) and the existing
Flask shell's lazy registration so a deploy without ``--preload`` still
gets a hydrated in-memory state. With ``--preload`` (the recommended
flag) the master pre-populates and workers inherit.

See ``src/proxy/app.py:initialize_proxy_at_startup`` for the canonical
ordered startup sequence (eager registry → init_db → load_all_rate_limiter_state
→ cleanup stale sessions).
"""
import os


# Server socket — bound by the master process.
bind = os.environ.get("PROXY_BIND", "0.0.0.0:6666")

# Worker model.
#
# workers=1 is the safe default. ChoriProx's Flask app runs threaded
# internally (qwen_proxy.py uses ``app.run(threaded=True)`` for the CLI;
# under gunicorn, ``--threads N`` is the equivalent knob). Scaling to
# multiple workers requires sticky-session routing on the upstream
# client (opencode / kilocode) because rate-limit state and chat history
# are per-process in-memory.
workers = int(os.environ.get("PROXY_WORKERS", "1"))

worker_class = "sync"

# Request timeout — matches the long-tail SOTA sync and step-plan
# inference latencies. 5 minutes is the documented operator expectation.
timeout = int(os.environ.get("PROXY_GUNICORN_TIMEOUT", "300"))
graceful_timeout = 30
keepalive = 5


def on_starting(server):
    """Gunicorn master-process startup hook.

    Runs in the MASTER process BEFORE any workers fork. We use this to
    run ``proxy.app.initialize_proxy_at_startup`` exactly once, so every
    forked worker inherits:

      * the populated SkillRegistry singleton (eager-init)
      * the hydrated persistent RateLimiter state from SQLite
      * the initialized SQLite schema (``rate_limiter_state``,
        ``sessions``, etc.)

    Without this hook, a fresh ``gunicorn qwen_proxy:app`` boot would
    skip the canonical startup sequence and either:
      * emit "no such table: rate_limiter_state" warnings on the first
        request to each worker, or
      * construct the SkillRegistry lazily on first dispatch
        (functional but slower cold-start).

    Failures propagate. The CLI in ``src/qwen_proxy.py`` wraps
    ``_eager_init_registry()`` separately to convert
    ``SkillRegistryConfigurationError`` to ``sys.exit(1)``; this hook
    follows the WSGI-shape contract and lets exceptions surface as
    gunicorn boot failures (gunicorn exits with a stack trace, which is
    the right operator-actionable failure mode for a WSGI worker).
    """
    from proxy.app import initialize_proxy_at_startup
    initialize_proxy_at_startup()


def post_fork(server, worker):
    """Per-worker post-fork hook.

    The master has already hydrated persistent RateLimiter state in
    ``on_starting``; workers inherit the populated ``_entries`` dicts
    via fork copy-on-write. Each worker keeps its own ``_call_count``
    counter (a fresh process-local int after fork) and its own
    ``_last_cleanup`` / ``_last_save`` time anchors — these are
    intentionally NOT reset here because the master's anchors are
    irrelevant after fork (each worker has its own clock baseline).

    No-op today; placeholder for future per-worker reset needs (e.g.
    clearing a worker-local connection pool).
    """
    pass