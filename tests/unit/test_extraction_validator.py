import pytest
import logging
from memory.extraction_validator import (
    levenshtein_distance, text_similarity,
    normalize_entity_type, normalize_relationship_type,
    validate_entity_references, validate_no_self_loops,
    validate_extraction, should_supersede,
    CANONICAL_ENTITY_TYPES, DEPRECATED_ENTITY_TYPES,
    CANONICAL_RELATIONSHIP_TYPES, REMOVED_RELATIONSHIP_TYPES,
    INVERSE_RELATIONSHIP_TYPES,
)

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# levenshtein_distance
# ---------------------------------------------------------------------------

class TestLevenshteinDistance:
    def test_identical_strings(self):
        assert levenshtein_distance("hello", "hello") == 0
        assert levenshtein_distance("", "") == 0

    def test_completely_different(self):
        assert levenshtein_distance("abc", "xyz") == 3
        assert levenshtein_distance("kitten", "sitting") == 3

    def test_empty_strings(self):
        assert levenshtein_distance("", "abc") == 3
        assert levenshtein_distance("abc", "") == 3

    def test_single_char_differences(self):
        assert levenshtein_distance("a", "b") == 1
        assert levenshtein_distance("ab", "ac") == 1
        assert levenshtein_distance("abc", "ab") == 1
        assert levenshtein_distance("ab", "abc") == 1

    def test_substitution(self):
        assert levenshtein_distance("test", "tent") == 1

    def test_deletion(self):
        assert levenshtein_distance("delete", "delet") == 1

    def test_insertion(self):
        assert levenshtein_distance("insrt", "insert") == 1


# ---------------------------------------------------------------------------
# text_similarity
# ---------------------------------------------------------------------------

class TestTextSimilarity:
    def test_identical(self):
        assert text_similarity("hello world", "hello world") == 1.0

    def test_completely_different(self):
        result = text_similarity("abcdefghij", "klmnopqrst")
        assert result < 0.5

    def test_empty_strings(self):
        assert text_similarity("", "") == 1.0
        assert text_similarity("hello", "") == 0.0
        assert text_similarity("", "hello") == 0.0

    def test_case_insensitive(self):
        assert text_similarity("Hello World", "hello world") == 1.0
        assert text_similarity("UPPERCASE", "uppercase") == 1.0

    def test_whitespace_insensitive(self):
        assert text_similarity("  hello  ", "hello") == 1.0

    def test_partial_similarity(self):
        sim = text_similarity("hello world", "hello there")
        assert 0.4 < sim < 0.9

    def test_normalized_case_and_whitespace(self):
        assert text_similarity("  HeLLo  ", "hello") == 1.0


# ---------------------------------------------------------------------------
# normalize_entity_type
# ---------------------------------------------------------------------------

class TestNormalizeEntityType:
    def test_canonical_types_pass_through(self):
        for t in CANONICAL_ENTITY_TYPES:
            result, warning = normalize_entity_type(t)
            assert result == t
            assert warning is None

    def test_deprecated_types_map_correctly(self):
        assert normalize_entity_type("Platform") == ("Technology", None)
        assert normalize_entity_type("Service") == ("Technology", None)
        assert normalize_entity_type("API") == ("Technology", None)
        assert normalize_entity_type("Database") == ("Technology", None)
        assert normalize_entity_type("Tool") == ("Technology", None)
        assert normalize_entity_type("Document") == ("File", None)
        assert normalize_entity_type("Config") == ("File", None)
        assert normalize_entity_type("Schema") == ("File", None)
        assert normalize_entity_type("Template") == ("File", None)
        assert normalize_entity_type("Test") == ("File", None)
        assert normalize_entity_type("Pattern") == ("Concept", None)
        assert normalize_entity_type("Feature") == ("Concept", None)
        assert normalize_entity_type("Event") == ("Concept", None)
        assert normalize_entity_type("Issue") == ("Bug", None)

    def test_unknown_type_defaults_to_concept_with_warning(self, caplog):
        caplog.set_level(logging.WARNING)
        result, warning = normalize_entity_type("FakeType")
        assert result == "Concept"
        assert warning is not None
        assert "Unknown entity type 'FakeType'" in warning

    def test_unknown_type_defaults_to_concept(self):
        result, warning = normalize_entity_type("RandomGarbage")
        assert result == "Concept"
        assert warning == "Unknown entity type 'RandomGarbage', defaulting to Concept"


# ---------------------------------------------------------------------------
# normalize_relationship_type
# ---------------------------------------------------------------------------

class TestNormalizeRelationshipType:
    def test_canonical_pass_through(self):
        for rel in CANONICAL_RELATIONSHIP_TYPES:
            fact = {"relationship_type": rel, "source_entity": "A", "target_entity": "B"}
            result, warning = normalize_relationship_type(fact)
            assert result["relationship_type"] == rel
            assert result["source_entity"] == "A"
            assert result["target_entity"] == "B"
            assert warning is None

    def test_deprecated_maps_correctly(self):
        fact = {"relationship_type": "BUILT_WITH", "source_entity": "A", "target_entity": "B"}
        result, warning = normalize_relationship_type(fact)
        assert result["relationship_type"] == "USES"
        assert result["source_entity"] == "A"
        assert result["target_entity"] == "B"
        assert warning is None

    def test_deprecated_implements_maps_to_uses(self):
        fact = {"relationship_type": "IMPLEMENTS", "source_entity": "A", "target_entity": "B"}
        result, warning = normalize_relationship_type(fact)
        assert result["relationship_type"] == "USES"

    def test_deprecated_written_in_maps_to_uses(self):
        fact = {"relationship_type": "WRITTEN_IN", "source_entity": "A", "target_entity": "B"}
        result, warning = normalize_relationship_type(fact)
        assert result["relationship_type"] == "USES"

    def test_deprecated_part_of_maps_to_contains(self):
        fact = {"relationship_type": "PART_OF", "source_entity": "A", "target_entity": "B"}
        result, warning = normalize_relationship_type(fact)
        assert result["relationship_type"] == "CONTAINS"

    def test_deprecated_located_at_maps_to_contains(self):
        fact = {"relationship_type": "LOCATED_AT", "source_entity": "A", "target_entity": "B"}
        result, warning = normalize_relationship_type(fact)
        assert result["relationship_type"] == "CONTAINS"

    def test_deprecated_references_maps_to_documents(self):
        fact = {"relationship_type": "REFERENCES", "source_entity": "A", "target_entity": "B"}
        result, warning = normalize_relationship_type(fact)
        assert result["relationship_type"] == "DOCUMENTS"

    def test_inverse_swaps_source_and_target(self):
        fact = {"relationship_type": "USED_BY", "source_entity": "A", "target_entity": "B"}
        result, warning = normalize_relationship_type(fact)
        assert result["relationship_type"] == "USES"
        assert result["source_entity"] == "B"
        assert result["target_entity"] == "A"
        assert warning is None

    def test_inverse_caused_by_swaps(self):
        fact = {"relationship_type": "CAUSED_BY", "source_entity": "A", "target_entity": "B"}
        result, warning = normalize_relationship_type(fact)
        assert result["relationship_type"] == "CAUSES"
        assert result["source_entity"] == "B"
        assert result["target_entity"] == "A"

    def test_inverse_solved_by_swaps(self):
        fact = {"relationship_type": "SOLVED_BY", "source_entity": "A", "target_entity": "B"}
        result, warning = normalize_relationship_type(fact)
        assert result["relationship_type"] == "RESOLVES"
        assert result["source_entity"] == "B"
        assert result["target_entity"] == "A"

    def test_inverse_resolved_by_swaps(self):
        fact = {"relationship_type": "RESOLVED_BY", "source_entity": "A", "target_entity": "B"}
        result, warning = normalize_relationship_type(fact)
        assert result["relationship_type"] == "RESOLVES"
        assert result["source_entity"] == "B"
        assert result["target_entity"] == "A"

    def test_inverse_created_by_swaps(self):
        fact = {"relationship_type": "CREATED_BY", "source_entity": "A", "target_entity": "B"}
        result, warning = normalize_relationship_type(fact)
        assert result["relationship_type"] == "CREATES"
        assert result["source_entity"] == "B"
        assert result["target_entity"] == "A"

    def test_inverse_built_by_swaps(self):
        fact = {"relationship_type": "BUILT_BY", "source_entity": "A", "target_entity": "B"}
        result, warning = normalize_relationship_type(fact)
        assert result["relationship_type"] == "CREATES"
        assert result["source_entity"] == "B"
        assert result["target_entity"] == "A"

    def test_inverse_documented_by_swaps(self):
        fact = {"relationship_type": "DOCUMENTED_BY", "source_entity": "A", "target_entity": "B"}
        result, warning = normalize_relationship_type(fact)
        assert result["relationship_type"] == "DOCUMENTS"
        assert result["source_entity"] == "B"
        assert result["target_entity"] == "A"

    def test_inverse_referenced_by_swaps(self):
        fact = {"relationship_type": "REFERENCED_BY", "source_entity": "A", "target_entity": "B"}
        result, warning = normalize_relationship_type(fact)
        assert result["relationship_type"] == "DOCUMENTS"
        assert result["source_entity"] == "B"
        assert result["target_entity"] == "A"

    def test_inverse_contained_in_swaps(self):
        fact = {"relationship_type": "CONTAINED_IN", "source_entity": "A", "target_entity": "B"}
        result, warning = normalize_relationship_type(fact)
        assert result["relationship_type"] == "CONTAINS"
        assert result["source_entity"] == "B"
        assert result["target_entity"] == "A"

    def test_inverse_follows_swaps(self):
        fact = {"relationship_type": "FOLLOWS", "source_entity": "A", "target_entity": "B"}
        result, warning = normalize_relationship_type(fact)
        assert result["relationship_type"] == "PRECEDES"
        assert result["source_entity"] == "B"
        assert result["target_entity"] == "A"

    def test_inverse_preceded_by_swaps(self):
        fact = {"relationship_type": "PRECEDED_BY", "source_entity": "A", "target_entity": "B"}
        result, warning = normalize_relationship_type(fact)
        assert result["relationship_type"] == "PRECEDES"
        assert result["source_entity"] == "B"
        assert result["target_entity"] == "A"

    def test_removed_type_produces_error(self):
        fact = {"relationship_type": "VULNERABLE_TO", "source_entity": "A", "target_entity": "B"}
        result, warning = normalize_relationship_type(fact)
        assert "Removed relationship type: VULNERABLE_TO" in warning

    def test_removed_type_mitigates_produces_error(self):
        fact = {"relationship_type": "MITIGATES", "source_entity": "A", "target_entity": "B"}
        result, warning = normalize_relationship_type(fact)
        assert "Removed relationship type: MITIGATES" in warning

    def test_removed_type_compromises_produces_error(self):
        fact = {"relationship_type": "COMPROMISES", "source_entity": "A", "target_entity": "B"}
        result, warning = normalize_relationship_type(fact)
        assert "Removed relationship type: COMPROMISES" in warning

    def test_removed_type_runs_produces_error(self):
        fact = {"relationship_type": "RUNS", "source_entity": "A", "target_entity": "B"}
        result, warning = normalize_relationship_type(fact)
        assert "Removed relationship type: RUNS" in warning

    def test_unknown_type_produces_error(self):
        fact = {"relationship_type": "NONEXISTENT", "source_entity": "A", "target_entity": "B"}
        result, warning = normalize_relationship_type(fact)
        assert result["relationship_type"] == "USES"
        assert warning == "Unknown relationship type 'NONEXISTENT', defaulting to USES"

    def test_unknown_type_with_spaces_hyphens(self):
        fact = {"relationship_type": "some weird-thing", "source_entity": "A", "target_entity": "B"}
        result, warning = normalize_relationship_type(fact)
        assert result["relationship_type"] == "USES"
        assert "Unknown relationship type 'some weird-thing'" in warning

    def test_preserves_original_keys(self):
        fact = {"relationship_type": "BUILT_WITH", "source_entity": "A",
                "target_entity": "B", "fact": "some detail"}
        result, warning = normalize_relationship_type(fact)
        assert result["fact"] == "some detail"

    def test_does_not_mutate_input(self):
        fact = {"relationship_type": "USED_BY", "source_entity": "A", "target_entity": "B"}
        original = fact.copy()
        normalize_relationship_type(fact)
        assert fact == original


# ---------------------------------------------------------------------------
# validate_entity_references
# ---------------------------------------------------------------------------

class TestValidateEntityReferences:
    def test_valid_references_pass(self):
        extraction = {
            "entities": [{"name": "A"}, {"name": "B"}],
            "facts": [{"source_entity": "A", "target_entity": "B"}],
        }
        errors = validate_entity_references(extraction)
        assert errors == []

    def test_dangling_source_entity_caught(self):
        extraction = {
            "entities": [{"name": "A"}],
            "facts": [{"source_entity": "MISSING", "target_entity": "A"}],
        }
        errors = validate_entity_references(extraction)
        assert len(errors) == 1
        assert "source_entity 'MISSING'" in errors[0]

    def test_dangling_target_entity_caught(self):
        extraction = {
            "entities": [{"name": "A"}],
            "facts": [{"source_entity": "A", "target_entity": "MISSING"}],
        }
        errors = validate_entity_references(extraction)
        assert len(errors) == 1
        assert "target_entity 'MISSING'" in errors[0]

    def test_both_dangling_caught(self):
        extraction = {
            "entities": [{"name": "A"}],
            "facts": [{"source_entity": "X", "target_entity": "Y"}],
        }
        errors = validate_entity_references(extraction)
        assert len(errors) == 2

    def test_empty_entities_and_facts_pass(self):
        extraction = {"entities": [], "facts": []}
        errors = validate_entity_references(extraction)
        assert errors == []

    def test_multiple_facts_partial_failures(self):
        extraction = {
            "entities": [{"name": "A"}, {"name": "B"}],
            "facts": [
                {"source_entity": "A", "target_entity": "B"},
                {"source_entity": "X", "target_entity": "A"},
            ],
        }
        errors = validate_entity_references(extraction)
        assert len(errors) == 1
        assert "source_entity 'X'" in errors[0]

    def test_missing_entities_key(self):
        extraction = {"facts": [{"source_entity": "A", "target_entity": "B"}]}
        errors = validate_entity_references(extraction)
        assert len(errors) == 2  # both source and target missing from empty entity set

    def test_fact_index_in_error_message(self):
        extraction = {
            "entities": [{"name": "A"}],
            "facts": [
                {"source_entity": "A", "target_entity": "A"},
                {"source_entity": "BAD", "target_entity": "A"},
            ],
        }
        errors = validate_entity_references(extraction)
        assert len(errors) == 1
        assert "fact[1]" in errors[0]


# ---------------------------------------------------------------------------
# validate_no_self_loops
# ---------------------------------------------------------------------------

class TestValidateNoSelfLoops:
    def test_self_relationship_caught(self):
        extraction = {
            "entities": [{"name": "A"}],
            "facts": [{"source_entity": "A", "target_entity": "A"}],
        }
        errors = validate_no_self_loops(extraction)
        assert len(errors) == 1
        assert "self-relationship" in errors[0]
        assert "A" in errors[0]

    def test_normal_relationships_pass(self):
        extraction = {
            "entities": [{"name": "A"}, {"name": "B"}],
            "facts": [{"source_entity": "A", "target_entity": "B"}],
        }
        errors = validate_no_self_loops(extraction)
        assert errors == []

    def test_empty_facts_pass(self):
        extraction = {"entities": [], "facts": []}
        errors = validate_no_self_loops(extraction)
        assert errors == []

    def test_multiple_self_loops_all_caught(self):
        extraction = {
            "entities": [{"name": "X"}, {"name": "Y"}],
            "facts": [
                {"source_entity": "X", "target_entity": "X"},
                {"source_entity": "Y", "target_entity": "Y"},
            ],
        }
        errors = validate_no_self_loops(extraction)
        assert len(errors) == 2

    def test_mixed_facts_self_and_normal(self):
        extraction = {
            "entities": [{"name": "A"}, {"name": "B"}],
            "facts": [
                {"source_entity": "A", "target_entity": "B"},
                {"source_entity": "A", "target_entity": "A"},
            ],
        }
        errors = validate_no_self_loops(extraction)
        assert len(errors) == 1


# ---------------------------------------------------------------------------
# validate_extraction
# ---------------------------------------------------------------------------

class TestValidateExtraction:
    def _valid_extraction(self):
        return {
            "extraction_timestamp": "2026-01-01T00:00:00Z",
            "entities": [
                {"type": "Person", "name": "Alice", "summary": "A test person named Alice"},
                {"type": "Technology", "name": "Python", "summary": "A programming language used for testing"},
            ],
            "facts": [
                {
                    "source_entity": "Alice",
                    "target_entity": "Python",
                    "relationship_type": "USES",
                    "fact": "Alice uses Python for writing test automation scripts",
                },
            ],
        }

    def test_full_valid_extraction_passes(self):
        extraction = self._valid_extraction()
        result, errors = validate_extraction(extraction)
        assert errors == []
        assert result["entities"][0]["type"] == "Person"
        assert result["facts"][0]["relationship_type"] == "USES"

    def test_missing_required_keys_caught(self):
        extraction = {"entities": [], "facts": []}
        result, errors = validate_extraction(extraction)
        assert len(errors) == 1
        assert "extraction_timestamp" in errors[0]

    def test_missing_entities_key_caught(self):
        extraction = {"extraction_timestamp": "t", "facts": []}
        result, errors = validate_extraction(extraction)
        assert len(errors) == 1
        assert "entities" in errors[0]

    def test_missing_facts_key_caught(self):
        extraction = {"extraction_timestamp": "t", "entities": []}
        result, errors = validate_extraction(extraction)
        assert len(errors) == 1
        assert "facts" in errors[0]

    def test_missing_entity_fields_caught(self):
        extraction = {
            "extraction_timestamp": "t",
            "entities": [{"name": "A"}],
            "facts": [],
        }
        result, errors = validate_extraction(extraction)
        assert len(errors) >= 1
        assert "entity[0]" in errors[0]
        assert "missing required fields" in errors[0]

    def test_missing_entity_name_caught(self):
        extraction = {
            "extraction_timestamp": "t",
            "entities": [{"type": "Person"}],
            "facts": [],
        }
        result, errors = validate_extraction(extraction)
        assert len(errors) >= 1
        assert any("name" in e.lower() or "key" in e.lower() for e in errors)

    def test_missing_fact_fields_caught(self):
        extraction = {
            "extraction_timestamp": "t",
            "entities": [{"name": "AB", "type": "Person", "summary": "AB is a person in the test"}],
            "facts": [{"source_entity": "AB", "target_entity": "AB"}],
        }
        result, errors = validate_extraction(extraction)
        assert len(errors) >= 1
        assert "fact[0]" in errors[0]
        assert "missing required field" in errors[0]

    def test_entity_type_normalization_applied(self):
        extraction = {
            "extraction_timestamp": "t",
            "entities": [
                {"type": "Platform", "name": "Docker", "summary": "Container platform used in testing"},
                {"type": "Issue", "name": "Bug123", "summary": "A bug found during testing phase"},
            ],
            "facts": [],
        }
        result, errors = validate_extraction(extraction)
        assert result["entities"][0]["type"] == "Technology"
        assert result["entities"][1]["type"] == "Bug"
        assert errors == []  # deprecated types produce no warnings

    def test_relationship_normalization_applied(self):
        extraction = {
            "extraction_timestamp": "t",
            "entities": [
                {"name": "Aj", "type": "Person", "summary": "Aj is a person entity"},
                {"name": "Bj", "type": "Project", "summary": "Bj is a project entity"},
            ],
            "facts": [
                {
                    "source_entity": "Aj",
                    "target_entity": "Bj",
                    "relationship_type": "BUILT_WITH",
                    "fact": "Aj built Bj using various technology tools",
                },
            ],
        }
        result, errors = validate_extraction(extraction)
        assert errors == []
        assert result["facts"][0]["relationship_type"] == "USES"

    def test_inverse_relationship_swaps_source_target(self):
        extraction = {
            "extraction_timestamp": "t",
            "entities": [
                {"name": "Au", "type": "Person", "summary": "A user working on tests"},
                {"name": "Bu", "type": "Project", "summary": "B project being tested"},
            ],
            "facts": [
                {
                    "source_entity": "Au",
                    "target_entity": "Bu",
                    "relationship_type": "USED_BY",
                    "fact": "Bu is being used by Au during development",
                },
            ],
        }
        result, errors = validate_extraction(extraction)
        assert errors == []
        assert result["facts"][0]["relationship_type"] == "USES"
        assert result["facts"][0]["source_entity"] == "Bu"
        assert result["facts"][0]["target_entity"] == "Au"

    def test_entity_name_truncated(self):
        extraction = {
            "extraction_timestamp": "t",
            "entities": [
                {"name": "X" * 210, "type": "Concept", "summary": "This is a valid summary"},
            ],
            "facts": [],
        }
        result, errors = validate_extraction(extraction)
        assert len(result["entities"][0]["name"]) == 200
        assert any("name truncated" in e for e in errors)

    def test_entity_summary_truncated(self):
        extraction = {
            "extraction_timestamp": "t",
            "entities": [
                {"name": "ValidName", "type": "Concept", "summary": "S" * 2100},
            ],
            "facts": [],
        }
        result, errors = validate_extraction(extraction)
        assert len(result["entities"][0]["summary"]) == 2000
        assert any("summary truncated" in e for e in errors)

    def test_fact_truncated(self):
        extraction = {
            "extraction_timestamp": "t",
            "entities": [
                {"name": "A", "type": "Concept", "summary": "A is a concept for testing"},
                {"name": "B", "type": "Concept", "summary": "B is another concept for testing"},
            ],
            "facts": [
                {
                    "source_entity": "A",
                    "target_entity": "B",
                    "relationship_type": "USES",
                    "fact": "F" * 2100,
                },
            ],
        }
        result, errors = validate_extraction(extraction)
        assert len(result["facts"][0]["fact"]) == 2000
        assert any("fact truncated" in e for e in errors)

    def test_entity_name_too_short_error(self):
        extraction = {
            "extraction_timestamp": "t",
            "entities": [
                {"name": "X", "type": "Concept", "summary": "This is a valid summary"},
            ],
            "facts": [],
        }
        result, errors = validate_extraction(extraction)
        assert any("name too short" in e for e in errors)

    def test_entity_summary_too_short_error(self):
        extraction = {
            "extraction_timestamp": "t",
            "entities": [
                {"name": "ValidName", "type": "Concept", "summary": "short"},
            ],
            "facts": [],
        }
        result, errors = validate_extraction(extraction)
        assert any("summary too short" in e for e in errors)

    def test_fact_text_too_short_error(self):
        extraction = {
            "extraction_timestamp": "t",
            "entities": [
                {"name": "A", "type": "Concept", "summary": "A is a concept used for testing"},
                {"name": "B", "type": "Concept", "summary": "B is another concept for testing"},
            ],
            "facts": [
                {
                    "source_entity": "A",
                    "target_entity": "B",
                    "relationship_type": "USES",
                    "fact": "short",
                },
            ],
        }
        result, errors = validate_extraction(extraction)
        assert any("fact too short" in e for e in errors)

    def test_self_loops_caught_in_validate_extraction(self):
        extraction = {
            "extraction_timestamp": "t",
            "entities": [
                {"name": "A", "type": "Concept", "summary": "A is a concept that cycles"},
            ],
            "facts": [
                {
                    "source_entity": "A",
                    "target_entity": "A",
                    "relationship_type": "USES",
                    "fact": "A uses itself recursively for testing",
                },
            ],
        }
        result, errors = validate_extraction(extraction)
        assert any("self-relationship" in e for e in errors)

    def test_dangling_refs_caught_in_validate_extraction(self):
        extraction = {
            "extraction_timestamp": "t",
            "entities": [
                {"name": "A", "type": "Concept", "summary": "A is a concept"},
            ],
            "facts": [
                {
                    "source_entity": "A",
                    "target_entity": "GHOST",
                    "relationship_type": "USES",
                    "fact": "A uses something that does not exist in entities",
                },
            ],
        }
        result, errors = validate_extraction(extraction)
        assert any("target_entity 'GHOST'" in e for e in errors)

    def test_unknown_entity_type_warning(self):
        extraction = {
            "extraction_timestamp": "t",
            "entities": [
                {"name": "XValid", "type": "UnheardOf", "summary": "Something with an unknown type"},
            ],
            "facts": [],
        }
        result, errors = validate_extraction(extraction)
        assert result["entities"][0]["type"] == "Concept"
        assert any("Unknown entity type 'UnheardOf'" in e for e in errors)

    def test_removed_relationship_type_error(self):
        extraction = {
            "extraction_timestamp": "t",
            "entities": [
                {"name": "A", "type": "Concept", "summary": "A is a concept for testing"},
                {"name": "B", "type": "Concept", "summary": "B is another concept for testing"},
            ],
            "facts": [
                {
                    "source_entity": "A",
                    "target_entity": "B",
                    "relationship_type": "VULNERABLE_TO",
                    "fact": "A is vulnerable to B's behavior",
                },
            ],
        }
        result, errors = validate_extraction(extraction)
        assert any("Removed relationship type" in e for e in errors)

    def test_unknown_relationship_type_error(self):
        extraction = {
            "extraction_timestamp": "t",
            "entities": [
                {"name": "A", "type": "Concept", "summary": "A is a concept used in testing"},
                {"name": "B", "type": "Concept", "summary": "B is another concept for testing"},
            ],
            "facts": [
                {
                    "source_entity": "A",
                    "target_entity": "B",
                    "relationship_type": "EATS",
                    "fact": "A eats B in a metaphorical sense during the test",
                },
            ],
        }
        result, errors = validate_extraction(extraction)
        assert any("Unknown relationship type" in e for e in errors)

    def test_multiple_errors_collected(self):
        extraction = {
            "extraction_timestamp": "t",
            "entities": [{"name": "An", "type": "UnknownType", "summary": "ok"}],
            "facts": [
                {
                    "source_entity": "An",
                    "target_entity": "An",
                    "relationship_type": "EATS",
                    "fact": "short",
                },
            ],
        }
        result, errors = validate_extraction(extraction)
        assert len(errors) > 1

    def test_entity_without_summary_defaults_to_empty(self):
        extraction = {
            "extraction_timestamp": "t",
            "entities": [
                {"name": "AValidName", "type": "Concept"},
            ],
            "facts": [],
        }
        result, errors = validate_extraction(extraction)
        assert any("summary too short" in e for e in errors)

    def test_fact_without_fact_field_defaults_to_empty(self):
        extraction = {
            "extraction_timestamp": "t",
            "entities": [
                {"name": "A", "type": "Concept", "summary": "A is a concept for test"},
                {"name": "B", "type": "Concept", "summary": "B is another concept for test"},
            ],
            "facts": [
                {
                    "source_entity": "A",
                    "target_entity": "B",
                    "relationship_type": "USES",
                },
            ],
        }
        result, errors = validate_extraction(extraction)
        assert any("missing required field 'fact'" in e for e in errors)


# ---------------------------------------------------------------------------
# should_supersede
# ---------------------------------------------------------------------------

class TestShouldSupersede:
    def test_same_triple_high_similarity_true(self):
        old = {
            "source_entity": "A", "target_entity": "B",
            "relationship_type": "USES",
            "fact": "A uses B for testing purposes",
        }
        new = {
            "source_entity": "A", "target_entity": "B",
            "relationship_type": "USES",
            "fact": "A uses B for testing purposez",  # only 1 char diff
        }
        assert should_supersede(old, new) is True

    def test_different_triple_false(self):
        old = {
            "source_entity": "A", "target_entity": "B",
            "relationship_type": "USES", "fact": "A uses B",
        }
        new = {
            "source_entity": "X", "target_entity": "Y",
            "relationship_type": "USES", "fact": "A uses B",
        }
        assert should_supersede(old, new) is False

    def test_different_relationship_type_false(self):
        old = {
            "source_entity": "A", "target_entity": "B",
            "relationship_type": "USES", "fact": "A uses B",
        }
        new = {
            "source_entity": "A", "target_entity": "B",
            "relationship_type": "CONTAINS", "fact": "A uses B",
        }
        assert should_supersede(old, new) is False

    def test_same_triple_low_similarity_false(self):
        old = {
            "source_entity": "A", "target_entity": "B",
            "relationship_type": "USES",
            "fact": "A uses B for testing",
        }
        new = {
            "source_entity": "A", "target_entity": "B",
            "relationship_type": "USES",
            "fact": "Completely unrelated fact about something else entirely",
        }
        assert should_supersede(old, new) is False

    def test_exact_match_high_similarity(self):
        old = {
            "source_entity": "A", "target_entity": "B",
            "relationship_type": "DEPENDS_ON",
            "fact": "A depends on B",
        }
        new = {
            "source_entity": "A", "target_entity": "B",
            "relationship_type": "DEPENDS_ON",
            "fact": "A depends on B",  # identical
        }
        assert should_supersede(old, new) is True

    def test_threshold_boundary(self):
        old = {
            "source_entity": "A", "target_entity": "B",
            "relationship_type": "USES", "fact": "0123456789",
        }
        # similarity = 1.0 - (2/10) = 0.8  (changed 2 chars out of 10)
        new = {
            "source_entity": "A", "target_entity": "B",
            "relationship_type": "USES", "fact": "01234567XY",
        }
        assert should_supersede(old, new) is False  # 0.8 is not > 0.8


# ---------------------------------------------------------------------------
# Field length limit edge cases
# ---------------------------------------------------------------------------

class TestFieldLengthLimits:
    """Edge cases for field length validation inside validate_extraction."""

    @staticmethod
    def _make_extraction(name="OkName", summary="A valid enough summary",
                         fact="A valid enough fact text"):
        return {
            "extraction_timestamp": "t",
            "entities": [
                {"name": name, "type": "Concept", "summary": summary},
                {"name": "Bb", "type": "Concept", "summary": "Bb is a concept for testing"},
            ],
            "facts": [
                {
                    "source_entity": name,
                    "target_entity": "Bb",
                    "relationship_type": "USES",
                    "fact": fact,
                },
            ],
        }

    def test_name_exactly_200_chars_passes(self):
        extraction = self._make_extraction(name="A" * 200)
        result, errors = validate_extraction(extraction)
        assert not any("name truncated" in e for e in errors)

    def test_name_exactly_201_chars_truncated(self):
        extraction = self._make_extraction(name="A" * 201)
        result, errors = validate_extraction(extraction)
        assert len(result["entities"][0]["name"]) == 200
        assert any("name truncated" in e for e in errors)

    def test_name_exactly_2_chars_passes(self):
        extraction = self._make_extraction(name="AB")
        result, errors = validate_extraction(extraction)
        assert not any("name too short" in e for e in errors)

    def test_name_exactly_1_char_error(self):
        extraction = self._make_extraction(name="A")
        result, errors = validate_extraction(extraction)
        assert any("name too short" in e for e in errors)

    def test_summary_exactly_2000_chars_passes(self):
        extraction = self._make_extraction(summary="A" * 2000)
        result, errors = validate_extraction(extraction)
        assert not any("summary truncated" in e for e in errors)

    def test_summary_exactly_2001_chars_truncated(self):
        extraction = self._make_extraction(summary="A" * 2001)
        result, errors = validate_extraction(extraction)
        assert len(result["entities"][0]["summary"]) == 2000
        assert any("summary truncated" in e for e in errors)

    def test_summary_exactly_10_chars_passes(self):
        extraction = self._make_extraction(summary="A" * 10)
        result, errors = validate_extraction(extraction)
        assert not any("summary too short" in e for e in errors)

    def test_summary_exactly_9_chars_error(self):
        extraction = self._make_extraction(summary="A" * 9)
        result, errors = validate_extraction(extraction)
        assert any("summary too short" in e for e in errors)

    def test_fact_exactly_2000_chars_passes(self):
        extraction = self._make_extraction(fact="A" * 2000)
        result, errors = validate_extraction(extraction)
        assert not any("fact truncated" in e for e in errors)

    def test_fact_exactly_2001_chars_truncated(self):
        extraction = self._make_extraction(fact="A" * 2001)
        result, errors = validate_extraction(extraction)
        assert len(result["facts"][0]["fact"]) == 2000
        assert any("fact truncated" in e for e in errors)

    def test_fact_exactly_10_chars_passes(self):
        extraction = self._make_extraction(fact="A" * 10)
        result, errors = validate_extraction(extraction)
        assert not any("fact too short" in e for e in errors)

    def test_fact_exactly_9_chars_error(self):
        extraction = self._make_extraction(fact="A" * 9)
        result, errors = validate_extraction(extraction)
        assert any("fact too short" in e for e in errors)
