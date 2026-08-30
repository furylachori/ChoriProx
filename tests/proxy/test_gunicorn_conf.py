"""US-HYDRATE-WSGI: Gunicorn WSGI startup wiring.

The canonical startup sequence ``proxy.app.initialize_proxy_at_startup``
was wired from CLI (``qwen_proxy.py``) but had NO gunicorn hook — the
docstrings referenced a fictional "gunicorn on_starting hook" that did
not exist in the codebase. This is a functional failure for production
WSGI deployments: rate-limiter hydration never runs, so every fresh
gunicorn worker process emits "no such table: rate_limiter_state"
warnings and serves with cold in-memory state.

The fix adds ``gunicorn.conf.py`` at the repo root, which gunicorn
auto-loads. This test pins two behaviors:

  1. ``gunicorn.conf.on_starting`` is a callable that, when invoked,
     successfully runs the canonical startup sequence against a fresh
     tmp DB without raising.
  2. ``gunicorn.conf.bind`` defaults to ``0.0.0.0:6666`` (the documented
     proxy port).

Layer: ``tests/proxy/`` — proxy-boot lifecycle tests live alongside the
existing ``test_proxy_startup_orchestration.py``. We intentionally
mirror its fixtures and patterns.

Revert-test protocol: temporarily comment out the body of ``on_starting``
in ``gunicorn.conf.py`` (replace the body with ``pass``) and confirm
``TestOnStartingHydratesRateLimiters`` FAILS — those tests use
``unittest.mock.patch`` on ``proxy.app.initialize_proxy_at_startup``
and assert it was called. If the body is empty, the mock is never
invoked and the assertion fails.
"""
import importlib.util
import os
from pathlib import Path
from unittest.mock import patch

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


_GUNICORN_CONF_PATH = Path(__file__).resolve().parents[2] / "gunicorn.conf.py"


def _load_gunicorn_conf():
    """Load gunicorn.conf.py as a module WITHOUT requiring gunicorn installed.

    Gunicorn is not a dev dependency (it's the WSGI runner in production),
    but the conf file itself does not import the gunicorn package — it only
    defines callable attributes (``on_starting``, ``post_fork``, ...) that
    gunicorn's runner reads when actually serving. ``importlib.util`` loads
    the file as a free-standing module so we can introspect its attributes
    in any environment.
    """
    spec = importlib.util.spec_from_file_location(
        "_gunicorn_conf_under_test", str(_GUNICORN_CONF_PATH)
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------
# 1. Static introspection of gunicorn.conf.py
# ---------------------------------------------------------------------------


class TestGunicornConfShape:
    """Pinned-behavior: gunicorn.conf.py exists, defines the documented
    callable hooks, and binds to the documented default port.
    """

    def test_gunicorn_conf_file_exists_at_repo_root(self):
        assert _GUNICORN_CONF_PATH.is_file(), (
            f"gunicorn.conf.py not found at {_GUNICORN_CONF_PATH}. "
            f"Gunicorn auto-loads this file from the working directory — "
            f"its absence means the canonical startup sequence is never "
            f"invoked in production WSGI deployments."
        )

    def test_on_starting_is_callable(self):
        mod = _load_gunicorn_conf()
        assert callable(mod.on_starting), (
            "gunicorn.conf.on_starting must be a callable — gunicorn "
            "invokes it as a hook during master-process startup."
        )

    def test_post_fork_is_callable(self):
        mod = _load_gunicorn_conf()
        assert callable(mod.post_fork), (
            "gunicorn.conf.post_fork must be a callable (currently a "
            "no-op, but the attribute is part of the documented contract)."
        )

    def test_bind_default_is_documented_port(self):
        """Default bind must be 0.0.0.0:6666 to match the Flask CLI port."""
        # Use a clean env (override PROXY_BIND so we test the module-level default).
        saved = os.environ.pop("PROXY_BIND", None)
        try:
            mod = _load_gunicorn_conf()
            assert mod.bind == "0.0.0.0:6666", (
                f"gunicorn.conf.bind default must be '0.0.0.0:6666' to "
                f"match the documented proxy port and the CLI "
                f"qwen_proxy.py:app.run(host='0.0.0.0', port=6666) "
                f"default. Got {mod.bind!r}."
            )
        finally:
            if saved is not None:
                os.environ["PROXY_BIND"] = saved

    def test_workers_default_is_one(self):
        """Default workers=1 — ChoriProx rate-limit state is per-process."""
        saved = os.environ.pop("PROXY_WORKERS", None)
        try:
            mod = _load_gunicorn_conf()
            assert mod.workers == 1, (
                f"gunicorn.conf.workers default must be 1 (ChoriProx "
                f"rate-limit state is per-process in-memory; multi-worker "
                f"deploys need sticky-session routing on the upstream "
                f"client). Got {mod.workers!r}."
            )
        finally:
            if saved is not None:
                os.environ["PROXY_WORKERS"] = saved


# ---------------------------------------------------------------------------
# 2. on_starting must invoke the canonical startup sequence
# ---------------------------------------------------------------------------


class TestOnStartingHydratesRateLimiters:
    """Pinned-behavior: invoking ``on_starting`` runs the canonical
    startup sequence.

    These tests assert the EFFECT of the hook (that it called
    ``initialize_proxy_at_startup``) rather than testing the full
    hydration side-effect in isolation. The hook is a one-line
    wrapper, so the right test is "did the wrapper call the real
    function?" — verified by patching
    ``proxy.app.initialize_proxy_at_startup`` and asserting it was
    called once.

    Revert-test: comment out the body of ``on_starting`` (replace with
    ``pass``) → these tests fail because the mock is never invoked.
    """

    def test_on_starting_invokes_initialize_proxy_at_startup(self):
        """The hook's job is to delegate to ``initialize_proxy_at_startup``.

        Patches the target so we don't pay the cost of the full schema
        initialization; what we care about is the call relationship.
        """
        mod = _load_gunicorn_conf()
        with patch("proxy.app.initialize_proxy_at_startup") as mock_startup:
            mod.on_starting(object())  # gunicorn passes the server; any sentinel works

        assert mock_startup.call_count == 1, (
            f"Expected gunicorn.conf.on_starting to call "
            f"proxy.app.initialize_proxy_at_startup exactly once, got "
            f"call_count={mock_startup.call_count}. If this fails, the "
            f"hook body is missing or broken — comment out the "
            f"``initialize_proxy_at_startup()`` call to verify "
            f"revert-test discipline."
        )

    def test_on_starting_invokes_initialize_proxy_at_startup_with_real_signature(
        self, tmp_path, monkeypatch
    ):
        """End-to-end smoke: hook calls the REAL function, which runs the
        real schema-init path against a tmp DB.

        This test exercises the full production code path (no mock on
        ``initialize_proxy_at_startup``). It uses the same
        ``real_init_db_via_memory_sync`` pattern as
        ``test_proxy_startup_orchestration.py`` to bypass the autouse
        mock on ``memory.sync.init_db``. The assertion is "no
        exception was raised during the full hydration path."
        """
        from memory.db import init_db as real_init_db
        monkeypatch.setattr("memory.sync.init_db", real_init_db)
        monkeypatch.setenv("PROXY_TEST_DB", str(tmp_path / "gunicorn_e2e.db"))

        mod = _load_gunicorn_conf()
        try:
            mod.on_starting(None)
        except Exception as exc:
            pytest.fail(
                f"gunicorn.conf.on_starting raised {type(exc).__name__}: "
                f"{exc}. The hook must run the canonical startup "
                f"sequence; failures should be debugged not silently "
                f"swallowed. Note: gunicorn boot failures should "
                f"surface as a stack trace, not silent exit."
            )

    def test_on_starting_is_idempotent(self):
        """Two consecutive on_starting calls must not raise.

        Idempotency is part of the contract:
        ``initialize_proxy_at_startup`` is documented as idempotent.
        """
        mod = _load_gunicorn_conf()
        with patch("proxy.app.initialize_proxy_at_startup"):
            mod.on_starting(None)
            mod.on_starting(None)  # second call: must not raise