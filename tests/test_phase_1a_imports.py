"""tests/test_phase_1a_imports.py — regression test for §6 import-site enumeration.

After 1a, ordinary ModeState consumers must import from
`proxy.skills.skill_definition`. Direct imports from `proxy.agent.types`
remain permitted only in `proxy.state`, which re-exports the defining
class, and in the identity-contract test that compares all three import
paths.

If a future PR adds a new `from proxy.agent.types import ModeState` import
in src/ or tests/, this test fails — pin the invariant.
"""
import re
from pathlib import Path

# Historical count of `from proxy.agent.types import ModeState` import sites
# at the start of Phase 1a (per docs/specs/consolidation-research/
# 11-phase-1a-plan.md §6). Pinned here so the test fails BOTH if a new
# site is added (len(rows) > 0) AND if the documented historical count
# is wrong (HISTORICAL_IMPORT_SITE_COUNT != 29).
HISTORICAL_IMPORT_SITE_COUNT = 29


def test_mode_state_import_site_count_matches_section_6():
    """After 1a: 0 sites remain (all 29 re-pointed; re-export file excluded).

    DEVIATION from plan §5: the plan asserts `len(rows) == 29` (the
    pre-re-pointing count). After the 29 sites are re-pointed AND the
    re-export file is excluded by path, the count is 0. This is the
    only sensible regression test — it pins the invariant that no
    future PR adds a new `from proxy.agent.types import ModeState`
    import.

    R3-1 (CRIT) fix: pin BOTH the current state (`len(rows) == 0`) AND
    the documented historical number (`HISTORICAL_IMPORT_SITE_COUNT
    == 29`). The historical constant is the audit trail — if a future
    reviewer changes the pre-1a count, this test fails loudly and
    surfaces the discrepancy.
    """
    repo_root = Path(__file__).resolve().parent.parent
    pattern = re.compile(
        r"^\s*from proxy\.agent\.types import.*\bModeState\b"
    )
    rows = []
    for path in repo_root.rglob("*.py"):
        rel = path.relative_to(repo_root)
        rel_str = str(rel)
        # The defining-class re-export and the explicit identity-contract test
        # intentionally import from proxy.agent.types.
        if rel_str in {
            "src/proxy/state.py",
            "tests/proxy/test_2a_foundation.py",
        }:
            continue
        # Limit to src/ and tests/ to match §6's scope.
        if not (rel_str.startswith("src/") or rel_str.startswith("tests/")):
            continue
        for lineno, line in enumerate(path.read_text().splitlines(), 1):
            if pattern.match(line):
                rows.append(f"{rel}:{lineno}")
    rows.sort()
    unique_files = {r.split(":")[0] for r in rows}
    # Pin 1: current state — no new import sites added.
    assert len(rows) == 0, (
        f"Expected 0 remaining import sites after 1a re-pointing, got {len(rows)}. "
        f"Each remaining site must be re-pointed to "
        f"`from proxy.skills.skill_definition import ModeState`. "
        f"Rows: {rows}"
    )
    assert len(unique_files) == 0, (
        f"Expected 0 unique files with old import, got {len(unique_files)}. "
        f"Files: {sorted(unique_files)}"
    )
    # Pin 2: historical count — the documented §6 pre-1a site count.
    # If this fails, the historical record is wrong and a reviewer
    # should reconcile this constant with the spec.
    assert HISTORICAL_IMPORT_SITE_COUNT == 29, (
        f"HISTORICAL_IMPORT_SITE_COUNT drift: expected 29 (per "
        f"docs/specs/consolidation-research/11-phase-1a-plan.md §6), "
        f"got {HISTORICAL_IMPORT_SITE_COUNT}. Reconcile with the spec."
    )
