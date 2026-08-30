"""Pure validator + explicit regenerator for docs/US_COVERAGE.md.

Parses USER-STORIES.md, greps tests/ for each US-XXX, and asserts that
docs/US_COVERAGE.md matches the regenerated data. The test is
read-only: it does not write to any tracked file.

Regenerate explicitly (ordinary pytest does not invoke __main__):

    PYTHONPATH=src python3 tests/unit/test_user_stories_inventory.py

The test fails with a diagnostic that names this command when the
on-disk inventory is missing or stale.
"""
from __future__ import annotations

import datetime
import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
USER_STORIES = REPO_ROOT / "USER-STORIES.md"
TESTS_DIR = REPO_ROOT / "tests"
INVENTORY = REPO_ROOT / "docs" / "US_COVERAGE.md"

_US_HEADING_RE = re.compile(r"^###\s+(US-\d+[a-z]?)\s*:\s*(.*)$")
_SECTION_RE = re.compile(r"^##\s+(.*)$")
_SUPPRESSED_RE = re.compile(r"\((SUPERSEDED[^)]*|NOT YET IMPLEMENTED[^)]*|RESERVED[^)]*)\)", re.IGNORECASE)
_FILE_RECURSE = "**/*.py"
_GENERATED_LINE_RE = re.compile(r"^> \*\*Generated:\*\* \d{4}-\d{2}-\d{2}\s*$", re.MULTILINE)


def _strip_generated_date(text: str) -> str:
    """Strip the `> **Generated:** YYYY-MM-DD` line so the sync test is stable
    across day boundaries. The line is still present in the on-disk file (we
    just don't compare it).
    """
    return _GENERATED_LINE_RE.sub("> **Generated:** <stripped>", text)


def _parse_stories(user_stories_path: Path) -> list[dict]:
    """Return list of {id, title, section, suppressed, reason} for each `### US-XXX` heading."""
    text = user_stories_path.read_text(encoding="utf-8")
    stories: list[dict] = []
    current_section = "(untitled)"
    for raw_line in text.splitlines():
        section_match = _SECTION_RE.match(raw_line)
        if section_match:
            current_section = section_match.group(1).strip()
            continue
        heading_match = _US_HEADING_RE.match(raw_line)
        if not heading_match:
            continue
        us_id = heading_match.group(1)
        title = heading_match.group(2).strip()
        suppressed_match = _SUPPRESSED_RE.search(title)
        suppressed = bool(suppressed_match)
        reason = suppressed_match.group(1) if suppressed_match else ""
        stories.append({
            "id": us_id,
            "title": title,
            "section": current_section,
            "suppressed": suppressed,
            "reason": reason,
        })
    return stories


def _scan_tests_for_references(tests_dir: Path) -> dict[str, int]:
    """Grep tests/ for `US-XXX` literal strings. Return {us_id: file_count}."""
    counts: dict[str, int] = {}
    for path in tests_dir.rglob(_FILE_RECURSE):
        try:
            content = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        for match in set(re.findall(r"US-\d+[a-z]?", content)):
            counts[match] = counts.get(match, 0) + 1
    return counts


def _build_inventory(stories: list[dict], ref_counts: dict[str, int]) -> str:
    """Build the markdown body of docs/US_COVERAGE.md from parsed data."""
    total = len(stories)
    suppressed_stories = [s for s in stories if s["suppressed"]]
    active_stories = [s for s in stories if not s["suppressed"]]
    referenced = [s for s in active_stories if ref_counts.get(s["id"], 0) > 0]
    pct = round(100.0 * len(referenced) / len(active_stories)) if active_stories else 0

    today = datetime.date.today().isoformat()

    lines: list[str] = []
    lines.append("# USER-STORIES.md Coverage Inventory")
    lines.append("")
    lines.append(f"> **Generated:** {today}")
    lines.append(f"> **Total stories:** {total}")
    lines.append(f"> **Active stories (excluding superseded/unimplemented/reserved):** {len(active_stories)}")
    lines.append(f"> **Referenced by at least one test:** {len(referenced)} ({pct}%)")
    lines.append("")

    section_totals: dict[str, dict[str, int]] = {}
    for s in active_stories:
        section_totals.setdefault(s["section"], {"active": 0, "referenced": 0})
        section_totals[s["section"]]["active"] += 1
        if ref_counts.get(s["id"], 0) > 0:
            section_totals[s["section"]]["referenced"] += 1

    lines.append("## Coverage by section")
    lines.append("")
    lines.append("| Section | Active | Referenced | % |")
    lines.append("|---|---|---|---|")
    for section, totals in section_totals.items():
        active_n = totals["active"]
        ref_n = totals["referenced"]
        section_pct = round(100.0 * ref_n / active_n) if active_n else 0
        lines.append(f"| {section} | {active_n} | {ref_n} | {section_pct}% |")
    lines.append("")

    unreferenced = [s for s in active_stories if ref_counts.get(s["id"], 0) == 0]
    lines.append("## Unreferenced active stories")
    lines.append("")
    if unreferenced:
        lines.append("| Story | Title | Reason for gap |")
        lines.append("|---|---|---|")
        for s in unreferenced:
            reason = "No test in tests/ references this ID"
            lines.append(f"| {s['id']} | {s['title']} | {reason} |")
    else:
        lines.append("All active stories are referenced by at least one test.")
    lines.append("")

    if suppressed_stories:
        lines.append("## Suppressed (not in denominator)")
        lines.append("")
        lines.append("| Story | Reason |")
        lines.append("|---|---|")
        for s in suppressed_stories:
            lines.append(f"| {s['id']} | {s['reason']} |")
        lines.append("")

    return "\n".join(lines)


def _regenerate() -> tuple[list[dict], dict[str, int], str]:
    stories = _parse_stories(USER_STORIES)
    ref_counts = _scan_tests_for_references(TESTS_DIR)
    body = _build_inventory(stories, ref_counts)
    return stories, ref_counts, body


class TestUserStoriesInventory:
    """Pure validator + explicit regenerator for docs/US_COVERAGE.md."""

    def test_inventory_is_well_formed_and_in_sync(self):
        stories, ref_counts, body = _regenerate()

        assert stories, "USER-STORIES.md must contain at least one US-XXX heading"
        active = [s for s in stories if not s["suppressed"]]
        assert active, "There must be at least one active (non-suppressed) story"
        assert "## Coverage by section" in body
        assert "## Unreferenced active stories" in body
        assert "## Suppressed (not in denominator)" in body

        for s in stories:
            assert s["id"].startswith("US-"), f"Invalid story id: {s['id']!r}"

        if not INVENTORY.exists():
            pytest.fail(
                f"docs/US_COVERAGE.md is missing. "
                f"Regenerate with: PYTHONPATH=src python3 tests/unit/test_user_stories_inventory.py"
            )

        on_disk = INVENTORY.read_text(encoding="utf-8")
        expected = _strip_generated_date(body)
        actual = _strip_generated_date(on_disk)
        if actual != expected:
            pytest.fail(
                f"docs/US_COVERAGE.md is out of sync. "
                f"Regenerate with: PYTHONPATH=src python3 tests/unit/test_user_stories_inventory.py "
                f"(expected starts with: {expected[:80]!r}; actual starts with: {actual[:80]!r})"
            )

    def test_story_ids_are_unique(self):
        stories, _, _ = _regenerate()
        ids = [s["id"] for s in stories]
        duplicates = sorted({us_id for us_id in ids if ids.count(us_id) > 1})
        assert not duplicates, f"USER-STORIES.md has duplicate US-XXX heading IDs: {duplicates}"


if __name__ == "__main__":
    stories, ref_counts, body = _regenerate()
    if not INVENTORY.exists():
        INVENTORY.parent.mkdir(parents=True, exist_ok=True)
        INVENTORY.write_text(body, encoding="utf-8")
        print(f"Generated {INVENTORY}")
    else:
        INVENTORY.write_text(body, encoding="utf-8")
        print(f"Regenerated {INVENTORY}")
