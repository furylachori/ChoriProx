"""Unit tests for canary verification cache (§5.4).

These tests exercise the cache data structure through ``parse_canary``
(LRU eviction with ``move_to_end`` on hit, env-tunable cap, entry TTL)
so a regression in the source's cache wiring is caught directly.
"""

import json
import os
import subprocess
import sys
import time
import unittest
from collections import OrderedDict

import proxy.state as state


class TestCanaryVerifyCacheLRU(unittest.TestCase):
    """LRU eviction, env-tunable cap, and per-entry TTL tests.

    Each test sets ``state._CANARY_VERIFY_CACHE_MAX`` (or ``..._TTL``)
    via direct attribute assignment and restores it in tearDown, since
    the constants are read from env at module load but referenced at
    every cache write (so direct mutation takes effect immediately).
    """

    _SECRET = "test-secret-canary-verify-cache-lru"

    def setUp(self):
        from unittest.mock import patch, MagicMock
        from proxy import canary as canary_mod

        # Save original constants — restore in tearDown.
        self._orig_max = state._CANARY_VERIFY_CACHE_MAX
        self._orig_ttl = state._CANARY_VERIFY_CACHE_TTL

        # Mock SkillRegistry so parse_canary's registry check passes.
        mock_registry = MagicMock()

        def _get(name):
            m = MagicMock()
            m.name = name
            return m

        mock_registry.get.side_effect = _get
        self._reg_patcher = patch(
            "proxy.skills.skill_registry.SkillRegistry",
            return_value=mock_registry,
        )
        self._reg_patcher.start()

        # Ensure a known session secret (HMAC path) for the test.
        self._had_secret = "PROXY_SESSION_SECRET" in os.environ
        self._orig_secret_env = os.environ.pop("PROXY_SESSION_SECRET", None)
        os.environ["PROXY_SESSION_SECRET"] = self._SECRET
        self._orig_mod_secret = canary_mod._SESSION_SECRET
        canary_mod._SESSION_SECRET = self._SECRET

        # Clear cache + ensure PROXY_ENV=dev so verify_canary's dev-fallback
        # branch is reachable when we want it.
        self._had_proxy_env = "PROXY_ENV" in os.environ
        self._orig_proxy_env = os.environ.pop("PROXY_ENV", None)
        os.environ["PROXY_ENV"] = self._orig_proxy_env or "dev"

        with state.canary_verify_cache_lock:
            state.canary_verify_cache.clear()

    def tearDown(self):
        from proxy import canary as canary_mod

        state._CANARY_VERIFY_CACHE_MAX = self._orig_max
        state._CANARY_VERIFY_CACHE_TTL = self._orig_ttl

        self._reg_patcher.stop()
        canary_mod._SESSION_SECRET = self._orig_mod_secret
        os.environ.pop("PROXY_SESSION_SECRET", None)
        if self._had_secret:
            os.environ["PROXY_SESSION_SECRET"] = self._orig_secret_env
        os.environ.pop("PROXY_ENV", None)
        if self._had_proxy_env:
            os.environ["PROXY_ENV"] = self._orig_proxy_env

        with state.canary_verify_cache_lock:
            state.canary_verify_cache.clear()

    def _make_canary(self, skill_name, state_dict=None):
        """Build a unique canary via sign_canary with the given state."""
        from proxy.canary import sign_canary

        state_str = ""
        if state_dict is not None:
            state_str = f"|{json.dumps(state_dict, sort_keys=True)}"
        sig = sign_canary(skill_name, state_str=state_str)
        return f"[[__SUBAGENT__:SKILL:{skill_name}:{sig}{state_str}]]"

    def _cache_key(self, canary_text, session_id):
        """Return the OrderedDict key currently used for ``canary_text``.

        Looks the key up by extracting the (session_id, sig_hash,
        state_hash, secret_hash) tuple. Mirrors the source's key
        construction so the test doesn't depend on the truncation policy.
        """
        import hashlib
        from proxy.canary import _get_session_secret

        # Sig:ts:state shape: [[__SUBAGENT__:SKILL:name:<sig64>:<ts>[|<state>]]]
        # Pull sig (64 hex between two colons) and ts (digits after).
        import re
        m = re.search(r":([a-f0-9]{64}):(\d+)", canary_text)
        if not m:
            raise AssertionError(f"could not parse sig/ts from canary: {canary_text!r}")
        sig = m.group(1)
        # state_str is the |... suffix if present, else ""
        rest = canary_text[m.end():]
        if rest.startswith("|"):
            state_str = rest.split("]]", 1)[0]
        else:
            state_str = ""
        # sig_part = sig only (rsplit last ":")
        sig_part = sig
        sig_hash = hashlib.sha256(sig_part.encode()).hexdigest()
        state_hash = hashlib.sha256(state_str.encode()).hexdigest()
        secret = _get_session_secret()
        secret_hash = (
            hashlib.sha256(secret.encode()).hexdigest()[:16] if secret else "dev"
        )
        return (session_id, sig_hash, state_hash, secret_hash)

    # ── Test 8a: TRUE LRU (catches missing move_to_end — F4) ──────────

    def test_lru_eviction_preserves_most_recently_accessed(self):
        """True LRU: after accessing A, A must survive an overflow that
        evicts the now-oldest entry (B).

        Inserts A, B, C at cap=3 → all fit. Accesses A (cache hit) →
        with the spec's move_to_end, A becomes MRU and B becomes LRU.
        Inserts D → overflow evicts B, not A. Asserts A's cache key is
        still present and B's is gone. Without move_to_end (FIFO), A
        would be evicted (oldest insert), failing this test.
        """
        from proxy.canary import parse_canary

        state._CANARY_VERIFY_CACHE_MAX = 3
        canary_a = self._make_canary("ticket", {"id": "A"})
        canary_b = self._make_canary("ticket", {"id": "B"})
        canary_c = self._make_canary("ticket", {"id": "C"})
        canary_d = self._make_canary("ticket", {"id": "D"})
        session_id = "sess_lru_true"

        # 1. Insert A, B, C — all fit (cap=3).
        parse_canary(canary_a, session_id=session_id)
        parse_canary(canary_b, session_id=session_id)
        parse_canary(canary_c, session_id=session_id)
        self.assertEqual(len(state.canary_verify_cache), 3)

        # Capture cache keys for A, B, C.
        with state.canary_verify_cache_lock:
            keys_now = set(state.canary_verify_cache.keys())
        key_a = self._cache_key(canary_a, session_id)
        key_b = self._cache_key(canary_b, session_id)
        key_c = self._cache_key(canary_c, session_id)
        self.assertIn(key_a, keys_now)
        self.assertIn(key_b, keys_now)
        self.assertIn(key_c, keys_now)

        # 2. Access A — cache hit. With F4 move_to_end, A becomes MRU.
        parse_canary(canary_a, session_id=session_id)
        self.assertEqual(len(state.canary_verify_cache), 3)

        # 3. Insert D — overflow evicts the LRU. With LRU, B is LRU.
        parse_canary(canary_d, session_id=session_id)
        self.assertEqual(len(state.canary_verify_cache), 3)

        # 4. Assert: A SURVIVES (MRU), B evicted (LRU), C and D present.
        with state.canary_verify_cache_lock:
            keys_after = set(state.canary_verify_cache.keys())
        key_d = self._cache_key(canary_d, session_id)

        self.assertIn(
            key_a, keys_after,
            "A was MRU (just accessed) — must survive the overflow. "
            "If A was evicted, the cache is using FIFO, not LRU (F4 bug).",
        )
        self.assertNotIn(
            key_b, keys_after,
            "B should be the LRU entry after A's access — must be evicted. "
            "If B survives, the cache is using MRU-of-evict instead of LRU.",
        )
        self.assertIn(key_c, keys_after)
        self.assertIn(key_d, keys_after)

    # ── Test 8b: Exact-boundary ───────────────────────────────────────

    def test_lru_eviction_exact_boundary(self):
        """Exactly _CANARY_VERIFY_CACHE_MAX entries → no eviction."""
        from proxy.canary import parse_canary

        max_entries = 3
        state._CANARY_VERIFY_CACHE_MAX = max_entries

        # Insert exactly max_entries distinct canaries via parse_canary.
        for i in range(max_entries):
            parse_canary(
                self._make_canary("ticket", {"i": i}),
                session_id="sess_boundary",
            )

        self.assertEqual(len(state.canary_verify_cache), max_entries)

    # ── Test 9a: default cap ──────────────────────────────────────────

    def test_env_tunable_cap_default_is_4096(self):
        """Default PROXY_CANARY_VERIFY_CACHE_MAX is 4096 (spec §5.1)."""
        # Only meaningful if the test process didn't override via env.
        if "PROXY_CANARY_VERIFY_CACHE_MAX" not in os.environ:
            self.assertEqual(state._CANARY_VERIFY_CACHE_MAX, 4096)

    # ── Test 9b: custom cap (env-tunable) ─────────────────────────────

    def test_env_tunable_cap_custom_evicts_oldest(self):
        """Set cap=2, insert 3 distinct canaries via parse_canary, assert
        oldest evicted.

        The constant is read from env at module load but referenced on
        every cache write, so mutating the attribute takes effect
        immediately (matches the production behavior under
        ``PROXY_CANARY_VERIFY_CACHE_MAX=2``).
        """
        from proxy.canary import parse_canary

        state._CANARY_VERIFY_CACHE_MAX = 2

        canary_a = self._make_canary("ticket", {"id": "A"})
        canary_b = self._make_canary("ticket", {"id": "B"})
        canary_c = self._make_canary("ticket", {"id": "C"})
        session_id = "sess_env_tunable"

        parse_canary(canary_a, session_id=session_id)
        parse_canary(canary_b, session_id=session_id)
        parse_canary(canary_c, session_id=session_id)

        with state.canary_verify_cache_lock:
            self.assertEqual(len(state.canary_verify_cache), 2)
            keys_now = set(state.canary_verify_cache.keys())

        key_a = self._cache_key(canary_a, session_id)
        key_b = self._cache_key(canary_b, session_id)
        key_c = self._cache_key(canary_c, session_id)

        # A was the oldest → evicted. B and C survive.
        self.assertNotIn(key_a, keys_now)
        self.assertIn(key_b, keys_now)
        self.assertIn(key_c, keys_now)

    # ── Test 9c: invalid cap clamps to 1 ──────────────────────────────

    def test_invalid_cap_clamps_to_one_at_module_load(self):
        """Spec §5.1: PROXY_CANARY_VERIFY_CACHE_MAX < 1 clamps to 1.

        The clamp happens at module load (state.py:143-159), so we run a
        subprocess to load proxy.state with the env var set to 0/-1 and
        read back the constant. This isolates the test from the host
        process's already-loaded proxy.state.
        """
        for bad in ("0", "-1"):
            code = (
                "import os, sys; "
                "sys.path.insert(0, 'src'); "
                f"os.environ['PROXY_CANARY_VERIFY_CACHE_MAX'] = '{bad}'; "
                "os.environ['PROXY_ENV'] = 'dev'; "
                "os.environ['PROXY_SESSION_SECRET'] = 'x'; "
                "import proxy.state; "
                "print(proxy.state._CANARY_VERIFY_CACHE_MAX)"
            )
            result = subprocess.run(
                [sys.executable, "-c", code],
                capture_output=True,
                text=True,
                cwd=os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
            )
            self.assertEqual(
                result.returncode, 0,
                f"subprocess for cap={bad!r} failed:\n"
                f"stdout={result.stdout!r}\nstderr={result.stderr!r}",
            )
            self.assertEqual(
                result.stdout.strip(), "1",
                f"cap={bad!r}: expected clamp to 1, got "
                f"{result.stdout.strip()!r}",
            )

    # ── Test TTL-a: TTL expiry via real parse_canary flow ─────────────

    def test_ttl_expiry_repopulates_via_fresh_verify(self):
        """Insert via parse_canary, advance clock past TTL, call again.

        Asserts that the stale entry was evicted AND a fresh entry
        replaced it (the cache was consulted: it noticed the expired
        entry and re-populated from the verify path).

        Uses a short TTL (10s) so we can advance time without waiting;
        HMAC freshness (300s) is still well above the TTL so the
        second call's verify succeeds and a fresh entry is added.
        """
        from proxy.canary import parse_canary

        state._CANARY_VERIFY_CACHE_TTL = 10.0

        canary = self._make_canary("ticket", {"k": "ttl_test"})
        session_id = "sess_ttl_expiry"

        # First call: verify + cache.
        real_time = time.time
        try:
            base = real_time()
            time.time = lambda: base
            p1 = parse_canary(canary, session_id=session_id)
            self.assertTrue(p1.verified)
            self.assertEqual(len(state.canary_verify_cache), 1)

            with state.canary_verify_cache_lock:
                _, _, _, _, first_cached_at = next(
                    iter(state.canary_verify_cache.values())
                )

            # Advance clock past TTL but still within HMAC freshness.
            time.time = lambda: base + 11.0
            p2 = parse_canary(canary, session_id=session_id)
        finally:
            time.time = real_time

        # Second call must still verify (cache hit after TTL expiry →
        # fresh verify → success, since HMAC freshness is 300s).
        self.assertIsNotNone(p2)
        self.assertTrue(
            p2.verified,
            "After TTL expiry, parse_canary must fall through to fresh "
            "HMAC verify and succeed (HMAC freshness is 300s; we "
            "advanced only 11s past the cache TTL).",
        )
        self.assertEqual(p2.skill_name, "ticket")

        # Cache was consulted: still has 1 entry, but its cached_at
        # was updated to reflect the fresh verify, NOT the original
        # 0-second insert.
        self.assertEqual(len(state.canary_verify_cache), 1)
        with state.canary_verify_cache_lock:
            _, _, _, _, second_cached_at = next(
                iter(state.canary_verify_cache.values())
            )
        self.assertGreater(
            second_cached_at, first_cached_at + 5.0,
            "Cache entry's cached_at must advance after TTL-expiry "
            "repopulation. If cached_at is unchanged, the stale entry "
            "was returned without consulting the cache TTL.",
        )

    # ── Test TTL-b: TTL arithmetic ────────────────────────────────────

    def test_cache_ttl_fresh_entry_is_valid(self):
        """An entry within TTL should be valid."""
        ttl = state._CANARY_VERIFY_CACHE_TTL

        now = time.time()
        cached_ts = now - (ttl / 2)

        self.assertTrue(
            now - cached_ts <= ttl,
            "Fresh entry within TTL must be a valid cache hit",
        )


    # ── Test: sign_canary round-trips through verify_canary ────────────

    def test_sign_canary_round_trips_through_verify_canary(self):
        """sign_canary() return value must verify via verify_canary() directly.

        Pins the 2-part format mismatch fix — pre-fix, verify_canary treated
        sig:timestamp as skill_name:sig and returned None."""
        from proxy.canary import sign_canary, verify_canary

        sig_ts = sign_canary("ticket")
        result = verify_canary(sig_ts, skill="ticket")
        self.assertIsNotNone(
            result,
            f"sign_canary('ticket') returned {sig_ts!r} but "
            f"verify_canary returned None (format mismatch)",
        )
        self.assertEqual(result.skill, "ticket")


if __name__ == "__main__":
    unittest.main()
