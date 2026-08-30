"""US-DEV-SESSION-2: per-instance isolation of ``ToolLimits._overrides``.

Regression test for the archeology flag on commit ``cb696f7`` (which added
``ToolLimits._overrides`` + ``set_limit()`` debug API). The concern was that
two ``ToolLimits`` instances might share state — either via a class-level
``_overrides = {}`` attribute, via a module-level dict, or via
``set_limit()`` mutating in place. If shared, calling ``set_limit()`` on
one instance would silently bleed into the other.

This test pins the invariant: each ``ToolLimits()`` instance owns a
distinct ``_overrides`` dict, and ``set_limit()`` mutations are confined
to the instance they were called on. Both ``_overrides`` (the dict
itself) and the downstream ``can_proxy_ticket`` / ``can_proxy_work``
properties (which read through it) MUST reflect per-instance state.

Invariants exercised:

1. ``__init__`` creates ``self._overrides`` per-instance (NOT a class-
   level attribute). ``a._overrides is not b._overrides``.
2. ``set_limit(field, n)`` mutates ``self._overrides`` in place; the
   mutation does not propagate to sibling instances.
3. The default fall-back (``MAX_PROXY_TICKET_CALLS``,
   ``MAX_PROXY_WORK_CALLS``, etc.) is unaffected by overrides on other
   instances.
4. The behavior observable via the public API
   (``can_proxy_ticket`` / ``can_proxy_work`` /
   ``try_count_proxy_ticket`` / ``try_count_proxy_work``) reflects
   per-instance state, not class- or module-level state.

If any of these invariants regress — e.g. someone changes ``__init__``
to share a class-level dict, or moves ``_overrides`` to a module-level
sentinel — the test fails with a clear diff.
"""
from __future__ import annotations

import pytest

from proxy.tool_pipeline.tool_limits import (
    MAX_PROXY_TICKET_CALLS,
    MAX_PROXY_WORK_CALLS,
    ToolLimits,
)


class TestSetLimitIsolation:
    """``set_limit()`` on one ``ToolLimits`` MUST NOT affect a sibling instance.

    This is the regression pin for commit ``cb696f7``: the ``_overrides``
    dict is the hot-path override for the proxy_ticket / proxy_work /
    report_bug / subagent limits. If two instances shared state, a debug
    session's ``set_tool_limit()`` call would silently leak into the
    production tracker's limits (or vice versa), changing behavior of
    unrelated requests.
    """

    def test_overrides_dict_is_per_instance(self):
        """``__init__`` MUST create a fresh ``_overrides`` per instance.

        If this assertion fails, someone moved ``_overrides`` to a
        class-level attribute (``_overrides: dict[str, int] = {}`` on
        the class body) or to a module-level dict. Both regressions
        break per-instance isolation regardless of how ``set_limit``
        is implemented.
        """
        a = ToolLimits()
        b = ToolLimits()

        assert a._overrides is not b._overrides, (
            "ToolLimits._overrides is SHARED across instances "
            "(class-level or module-level attribute). Per-instance "
            "isolation is BROKEN — set_limit() on one instance will "
            "bleed into the other."
        )

    def test_set_limit_on_a_does_not_appear_in_b_dict(self):
        """``a.set_limit("proxy_ticket", 999)`` MUST NOT mutate ``b._overrides``."""
        a = ToolLimits()
        b = ToolLimits()

        a.set_limit("proxy_ticket", 999)

        assert "proxy_ticket" not in b._overrides, (
            "set_limit() on instance `a` leaked into instance `b` "
            f"— b._overrides == {b._overrides!r}"
        )
        assert b._overrides == {}, (
            f"Expected b._overrides to be empty {{}}, got {b._overrides!r}"
        )
        assert a._overrides == {"proxy_ticket": 999}, (
            f"Expected a._overrides == {{'proxy_ticket': 999}}, got {a._overrides!r}"
        )

    def test_b_default_limit_unchanged_after_a_set_limit(self):
        """``b.can_proxy_ticket`` MUST reflect the default limit, not ``a``'s override.

        This is the behavioral assertion: the property ``can_proxy_ticket``
        reads through ``self._overrides`` and falls back to
        ``MAX_PROXY_TICKET_CALLS`` when no override is present. If
        ``a.set_limit("proxy_ticket", 999)`` had leaked into ``b``,
        ``b.can_proxy_ticket`` would still report True at count 998 —
        not at count ``MAX_PROXY_TICKET_CALLS``.
        """
        a = ToolLimits()
        b = ToolLimits()

        a.set_limit("proxy_ticket", 999)

        # Drive `b` to exactly the default limit. Each distinct call_id
        # consumes one slot. At MAX, can_proxy_ticket must report False
        # because b has no override — MAX_PROXY_TICKET_CALLS applies.
        for i in range(MAX_PROXY_TICKET_CALLS):
            assert b.try_count_proxy_ticket(f"b-call-{i}") is True, (
                f"b.try_count_proxy_ticket({i + 1}/{MAX_PROXY_TICKET_CALLS}) "
                f"was unexpectedly rejected — b may have inherited a's override."
            )

        assert b.can_proxy_ticket is False, (
            f"b.can_proxy_ticket is True after {MAX_PROXY_TICKET_CALLS} "
            f"calls; expected False because b's effective limit is the "
            f"default ({MAX_PROXY_TICKET_CALLS}). a's override (999) leaked."
        )

    def test_a_limit_honored_independently_of_b(self):
        """``a``'s override (999) MUST apply to ``a``, regardless of ``b``'s state."""
        a = ToolLimits()
        b = ToolLimits()

        a.set_limit("proxy_ticket", 999)

        # Drive `a` past MAX_PROXY_TICKET_CALLS — a's override is 999,
        # so it should still allow calls beyond the default cap.
        for i in range(MAX_PROXY_TICKET_CALLS + 50):
            assert a.try_count_proxy_ticket(f"a-call-{i}") is True, (
                f"a.try_count_proxy_ticket rejected call {i + 1}; "
                f"a's override (999) is not being honored."
            )

        assert a.can_proxy_ticket is True, (
            "a.can_proxy_ticket is False at 80 calls with override 999; "
            "the override was not applied to a."
        )

    def test_symmetric_isolation_b_set_limit_does_not_affect_a(self):
        """``b.set_limit("proxy_work", 7)`` MUST NOT mutate ``a._overrides``."""
        a = ToolLimits()
        b = ToolLimits()

        # First seed `a` with an override so we can verify it survives `b`'s mutation.
        a.set_limit("proxy_ticket", 999)

        b.set_limit("proxy_work", 7)

        assert "proxy_work" not in a._overrides, (
            "b.set_limit('proxy_work', 7) leaked into a._overrides "
            f"— a._overrides == {a._overrides!r}"
        )
        assert a._overrides == {"proxy_ticket": 999}, (
            f"a._overrides was corrupted by b's set_limit; "
            f"got {a._overrides!r}, expected {{'proxy_ticket': 999}}"
        )
        assert b._overrides == {"proxy_work": 7}, (
            f"b._overrides == {b._overrides!r}; expected {{'proxy_work': 7}}"
        )

    def test_proxy_work_isolation(self):
        """The same isolation invariant MUST hold for ``proxy_work``."""
        a = ToolLimits()
        b = ToolLimits()

        a.set_limit("proxy_work", 999)

        assert "proxy_work" not in b._overrides
        assert b._overrides == {}
        assert b.can_proxy_work is True, (
            "b.can_proxy_work is False at count 0; expected True "
            "(no override on b, so default MAX_PROXY_WORK_CALLS applies)."
        )
        assert a.can_proxy_work is True, (
            "a.can_proxy_work is False at count 0 with override 999; "
            "override should not flip can_proxy_work at zero calls."
        )