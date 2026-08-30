import pytest

from proxy.skills.command_text_utils import extract_command_text, strip_environment_details


class TestStripEnvironmentDetails:
    def test_paired_tags_only(self):
        """Remove a complete <environment_details>...</environment_details> block."""
        text = "some text <environment_details>inner content</environment_details> trailing"
        assert strip_environment_details(text) == "some text  trailing"

    def test_orphaned_opening_tag(self):
        """Remove an <environment_details> tag that has no closing tag."""
        text = "do something <environment_details>Current time: 2026-05-27"
        assert strip_environment_details(text) == "do something"

    def test_no_xml_passthrough(self):
        """Return text unchanged when there is no environment_details XML."""
        text = "just a plain command with no XML"
        assert strip_environment_details(text) == text

    def test_command_with_xml(self):
        """Strip the XML while preserving the command portion."""
        text = (
            "approve 42\n"
            "<environment_details>\n"
            "Current time: 2026-05-27T08:00:00-06:00\n"
            "Working directory: /tmp\n"
            "</environment_details>"
        )
        assert strip_environment_details(text) == "approve 42"

    def test_multiple_blocks(self):
        """Strip multiple environment_details blocks."""
        text = (
            "<environment_details>a</environment_details>"
            " keep me "
            "<environment_details>b</environment_details>"
        )
        assert strip_environment_details(text) == "keep me"


class TestExtractCommandText:
    @pytest.mark.parametrize(
        "raw, expected",
        [
            (
                "  approve 7  \n<environment_details>foo</environment_details>",
                "approve 7",
            ),
            (
                "list all",
                "list all",
            ),
            (
                "  status  <environment_details>details</environment_details>  ",
                "status",
            ),
            (
                "<environment_details>only xml</environment_details>",
                "",
            ),
        ],
        ids=["command+xml", "plain", "extra-whitespace", "xml-only"],
    )
    def test_various_inputs(self, raw, expected):
        assert extract_command_text(raw) == expected

    def test_preserves_non_environment_xml(self):
        """Non-environment_details XML must survive the strip."""
        raw = 'search <query>foo bar</query> <environment_details>x</environment_details>'
        assert extract_command_text(raw) == "search <query>foo bar</query>"

    def test_nested_same_tag(self):
        """Nested <tag>same_text</tag> produces empty string."""
        # All environment_details tags (including outer) are stripped
        text = "<environment_details>before <environment_details>inner</environment_details> after</environment_details>"
        result = extract_command_text(text)
        assert result == ""


class TestStripSystemReminder:
    """<system-reminder> is opencode's mode-change/reminder block — same class
    as <environment_details>, handled by the same scoped cleaner."""

    def test_paired_reminder_stripped(self):
        text = "^^ws myname<system-reminder>mode change</system-reminder>"
        assert strip_environment_details(text) == "^^ws myname"

    def test_reminder_appended_to_command(self):
        # The exact bug: ^^ws <name> + appended reminder → reminder became the name.
        text = (
            "^^ws myname\n"
            "<system-reminder>\nYour operational mode has changed from plan to build.\n"
            "You are no longer in read-only mode.\n</system-reminder>"
        )
        assert strip_environment_details(text) == "^^ws myname"

    def test_reminder_only_becomes_empty(self):
        assert strip_environment_details("<system-reminder>mode</system-reminder>") == ""

    def test_reminder_nested_same_tag(self):
        text = "<system-reminder>outer <system-reminder>inner</system-reminder> after</system-reminder>keep"
        assert strip_environment_details(text) == "keep"

    def test_reminder_orphaned_open(self):
        text = "do <system-reminder>Current time: 2026-07-03"
        assert strip_environment_details(text) == "do"

    def test_both_tags_stripped_together(self):
        text = (
            "approve 42"
            "<environment_details>cwd=/tmp git=main</environment_details>"
            "<system-reminder>mode change</system-reminder>"
        )
        assert strip_environment_details(text) == "approve 42"

    def test_cross_tag_nesting_inner_swallowed(self):
        # A <system-reminder> INSIDE an <environment_details> span is consumed
        # as raw chars by the outer tag's skip — inner never re-enters matching.
        text = "<environment_details>x<system-reminder>y</system-reminder>z</environment_details>"
        assert strip_environment_details(text) == ""

    def test_mixed_case_tag_preserved(self):
        # Canary-safety boundary: only lowercase explicit tags match. A mixed-case
        # tag is NOT stripped (locks against the recurring ".lower() fallback"
        # regression — the normalize.py landmine class). If someone "helpfully"
        # lowercases the comparison, this test fails.
        text = "do <System-Reminder>mode</System-Reminder> now"
        assert strip_environment_details(text) == text


class TestCanaryPreservation:
    """Canary-safe invariant: the cleaner strips ONLY the explicit framework
    tags, never generic <tag> matching. Canary text (HMAC-signed, plain text)
    must round-trip byte-identical — recursing into it would wipe sub-agent
    mode detection. (The historical landmine.)"""

    def test_canary_text_preserved_with_framework_tag(self):
        # Real SKILL canary format (src/proxy/canary.py:6-7,
        # src/proxy/skills/skill_definition.py:821): the cleaner strips ONLY the
        # explicit framework tags, so the canary round-trips byte-identical.
        canary = "[[__SUBAGENT__:SKILL:ticket:9f2a1c4b8e7d6a5f3c2b1a09f8e7d6c5b4a3f2e1d0c9b8a7f6e5d4c3b2a1908f:1719900000]]"
        text = canary + "<system-reminder>mode change</system-reminder>"
        assert strip_environment_details(text) == canary

    def test_workspace_canary_preserved_with_framework_tag(self):
        # Workspace canary (the variant relevant to the ^^ws crash) round-trips
        # intact. Real WORKSPACE format per src/proxy/canary.py:8 /
        # src/proxy/mode.py:270.
        canary = "[[__SUBAGENT__:WORKSPACE:demo:9f2a1c4b8e7d6a5f3c2b1a09f8e7d6c5b4a3f2e1d0c9b8a7f6e5d4c3b2a1908f:1719900000]]"
        text = "^^ws<system-reminder>mode change</system-reminder>" + canary
        assert strip_environment_details(text) == "^^ws" + canary

    def test_canary_with_angle_brackets_not_touched(self):
        # A canary or user text containing unrelated <...> is NOT stripped —
        # only the explicit framework tag names match.
        text = "ctx[<mode>] = build<system-reminder>x</system-reminder>"
        assert strip_environment_details(text) == "ctx[<mode>] = build"

    def test_unknown_tag_preserved(self):
        # Generic XML the cleaner does NOT know about survives — confirms it is
        # scoped to the explicit tag set, not generic <tag> stripping.
        text = "do <unknown_tag>data</unknown_tag> now"
        assert strip_environment_details(text) == text
