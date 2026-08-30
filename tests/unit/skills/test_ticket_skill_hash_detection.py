"""Regression tests for parse_subcommand hash detection.

Bug: `^^ticket close the #4` was interpreted as a batch close on the
(non-existent) hash "the #4", producing a FAILED response. The parser
should fall through to plan mode (LLM) when the post-strip argument
contains no valid short hash tokens.
"""
from __future__ import annotations

import pytest

from skills.ticket_skill import TicketSkill


@pytest.fixture
def skill():
    return TicketSkill()


class TestHashListDetection:
    @pytest.mark.parametrize("verb,hashes,expected_arg", [
        ("done", "c9ac1101", "c9ac1101"),
        ("done", "c9ac1101,859f219f", "c9ac1101,859f219f"),
        ("close", "#c9ac1101", "#c9ac1101"),
        ("close", "c9ac1101", "c9ac1101"),
    ])
    def test_valid_hash_args_route_to_done(self, skill, verb, hashes, expected_arg):
        full = f"{verb} {hashes}"
        action, parsed = skill._parse_ticket_args(full)
        assert action == "done"
        assert parsed == expected_arg

    @pytest.mark.parametrize("verb,arg,expected_plan_arg", [
        ("close", "the #4", "close the #4"),
        ("done", "task 3", "done task 3"),
        ("invalidate", "the broken ones", "invalidate the broken ones"),
        ("reopen", "task 5", "reopen task 5"),
    ])
    def test_all_verbs_route_natural_language_to_plan(self, skill, verb, arg, expected_plan_arg):
        full = f"{verb} {arg}"
        action, plan_arg = skill._parse_ticket_args(full)
        assert action == "plan"
        assert plan_arg == expected_plan_arg

    def test_verb_preserved_in_plan_arg(self, skill):
        """The LLM must see the user's full intent — the verb is not stripped."""
        action, plan_arg = skill._parse_ticket_args("close the #4")
        assert action == "plan"
        assert plan_arg == "close the #4"

    def test_content_word_not_stripped(self, skill):
        """^^ticket option 1: the first word is content, not a planning verb.

        Regression for the "parser eats a word" bug — the everything-else
        plan fallback must NOT strip the leading word.
        """
        action, plan_arg = skill._parse_ticket_args("option 1")
        assert action == "plan"
        assert plan_arg == "option 1"

    def test_close_with_hash_after_verb_works(self, skill):
        action, arg = skill._parse_ticket_args("close c9ac1101")
        assert action == "done"
        assert arg == "c9ac1101"

    def test_invalidate_with_hash_works(self, skill):
        action, arg = skill._parse_ticket_args("invalidate c9ac1101")
        assert action == "invalidate"
        assert arg == "c9ac1101"

    def test_reopen_with_hash_works(self, skill):
        action, arg = skill._parse_ticket_args("reopen c9ac1101")
        assert action == "reopen"
        assert arg == "c9ac1101"

    def test_close_with_pound_prefix_stripped(self, skill):
        action, arg = skill._parse_ticket_args("close #c9ac1101")
        assert action == "done"
        assert arg == "#c9ac1101"

    def test_close_with_comma_separated_hashes(self, skill):
        action, arg = skill._parse_ticket_args("done c9ac1101,859f219f")
        assert action == "done"
        assert arg == "c9ac1101,859f219f"

    def test_close_with_hash_works(self, skill):
        action, arg = skill._parse_ticket_args("close c9ac1101")
        assert action == "done"
        assert arg == "c9ac1101"

    def test_invalidate_with_hash_works(self, skill):
        action, arg = skill._parse_ticket_args("invalidate c9ac1101")
        assert action == "invalidate"
        assert arg == "c9ac1101"

    def test_reopen_with_hash_works(self, skill):
        action, arg = skill._parse_ticket_args("reopen c9ac1101")
        assert action == "reopen"
        assert arg == "c9ac1101"
