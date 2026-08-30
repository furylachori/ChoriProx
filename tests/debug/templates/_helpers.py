"""Shared helpers for debug_session templates. Import _emit + _finish.

Used by all templates under tests/debug/templates/. The directory is on
sys.path via the `sys.path.insert(0, os.path.dirname(__file__))` line at the
top of each template (not via the Makefile's PYTHONPATH — that's just `src`).

Prints one JSONL line per check:
    {"step":N,"check":"<what>","ok":bool,"detail":"..."}

Exits 0 if all checks passed, 1 otherwise — this is the ONLY way
`make debug-templates` can detect regressions.
"""
import json
import os
import sys

_failures = []


def _emit(step, check, ok, detail=""):
    """Print one JSONL result line. Tracks failures for _finish()."""
    print(json.dumps({"step": step, "check": check, "ok": bool(ok), "detail": detail}))
    sys.stdout.flush()
    if not ok:
        _failures.append((step, check))


def _finish():
    """Exit 0 if all checks passed, 1 otherwise. Call at the END of every template."""
    sys.stdout.flush()
    os._exit(1 if _failures else 0)