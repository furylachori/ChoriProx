#!/usr/bin/env python3
"""Bug-class reproducer: lock-ordering violation is caught at runtime when
`PROXY_DEBUG_LOCK_ORDERING=1` is set (see `src/proxy/lock_ordering.py`).

The OrderedLock wrapper is a no-op in production. Enabling the env var BEFORE
the proxy imports `proxy.state` swaps in the wrapper. Acquiring a lower-numbered
lock while holding a higher-numbered one raises RuntimeError immediately
instead of silently deadlocking.

Per `src/proxy/state.py:197-215`, lock hierarchy is 0..12 — AgentRegistry._lock
(0) is coarsest; log_overflow_count_lock (12) is the leaf.

This script proves the mechanism works. A real lock-ordering bug in a proxy
stage would reproduce the same RuntimeError when the offending request fires.

Run with:
    PYTHONPATH=src python3 tests/debug/lock-ordering-repro.py
"""
import os

# MUST be set BEFORE importing qwen_proxy (which imports proxy.state and
# constructs the locks via make_lock()).
os.environ["PROXY_DEBUG_LOCK_ORDERING"] = "1"

import sys

from devtools.debug_session import debug_session  # noqa: E402


def main() -> int:
    with debug_session() as c:
        # Verify OrderedLock is actually in place (not the regular Lock)
        from proxy.state import (
            canary_verify_cache_lock, vfs_lock,
        )
        ctypes = (type(canary_verify_cache_lock).__name__,
                  type(vfs_lock).__name__)
        print(f"lock types: canary={ctypes[0]} vfs={ctypes[1]}")
        if "Ordered" not in ctypes[0] or "Ordered" not in ctypes[1]:
            print("FAIL: env var did not take effect")
            return 1

        # Trigger an out-of-order acquisition: hold #11, then acquire #9
        with canary_verify_cache_lock:  # #11 — ascending baseline OK
            try:
                with vfs_lock:  # #9 — VIOLATION (acquiring lower while holding higher)
                    pass
                print("FAIL: violation was not caught")
                return 1
            except RuntimeError as e:
                msg = str(e)
                print(f"OK: violation caught -> {msg}")
                return 0 if "ordering" in msg.lower() else 1


if __name__ == "__main__":
    rc = main()
    sys.stdout.flush()
    os._exit(rc)
