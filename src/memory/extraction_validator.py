import logging

logger = logging.getLogger("memory.extraction_validator")

CANONICAL_ENTITY_TYPES = {"Person", "Organization", "Project", "Technology",
                          "File", "Concept", "Objective", "Bug", "Fix", "Task"}

DEPRECATED_ENTITY_TYPES = {"Platform": "Technology", "Service": "Technology", "API": "Technology",
                           "Database": "Technology", "Tool": "Technology", "Document": "File",
                           "Config": "File", "Schema": "File", "Template": "File", "Test": "File",
                           "Pattern": "Concept", "Feature": "Concept", "Event": "Concept", "Issue": "Bug"}

CANONICAL_RELATIONSHIP_TYPES = {"USES", "CONTAINS", "CREATES", "SUPERSEDES",
                                "DOCUMENTS", "CAUSES", "RESOLVES", "DEPENDS_ON",
                                "PRECEDES", "ALTERNATIVE_TO"}

DEPRECATED_RELATIONSHIP_TYPES = {"BUILT_WITH": "USES", "IMPLEMENTS": "USES", "WRITTEN_IN": "USES",
                                 "SUPPORTS": "USES", "NOT_SUPPORTS": "USES", "PART_OF": "CONTAINS",
                                 "LOCATED_AT": "CONTAINS", "REFERENCES": "DOCUMENTS"}

REMOVED_RELATIONSHIP_TYPES = {"VULNERABLE_TO", "NOT_VULNERABLE_TO", "MITIGATES",
                              "COMPROMISES", "PREFERS", "DECIDED", "RUNS",
                              "EXECUTES", "HAS_STEP_RUN", "RUNS_STEP", "EXTRACTED_FROM"}

INVERSE_RELATIONSHIP_TYPES = {"USED_BY": "USES", "CAUSED_BY": "CAUSES", "SOLVED_BY": "RESOLVES",
                              "RESOLVED_BY": "RESOLVES", "CREATED_BY": "CREATES",
                              "BUILT_BY": "CREATES", "DOCUMENTED_BY": "DOCUMENTS",
                              "REFERENCED_BY": "DOCUMENTS", "CONTAINED_IN": "CONTAINS",
                              "FOLLOWS": "PRECEDES", "PRECEDED_BY": "PRECEDES"}


def levenshtein_distance(s1: str, s2: str) -> int:
    if s1 == s2:
        return 0
    if not s1:
        return len(s2)
    if not s2:
        return len(s1)

    prev = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        curr = [i + 1]
        for j, c2 in enumerate(s2):
            cost = 0 if c1 == c2 else 1
            curr.append(min(curr[j] + 1, prev[j + 1] + 1, prev[j] + cost))
        prev = curr
    return prev[-1]


def normalize_entity_type(type_str: str) -> tuple:
    """Returns (canonical_type, warning_or_None)."""
    if type_str in CANONICAL_ENTITY_TYPES:
        return type_str, None
    if type_str in DEPRECATED_ENTITY_TYPES:
        return DEPRECATED_ENTITY_TYPES[type_str], None
    return "Concept", f"Unknown entity type '{type_str}', defaulting to Concept"


def normalize_relationship_type(fact: dict) -> tuple:
    """Returns (normalized_fact, warning_or_None). Swaps source/target for inverses."""
    result = fact.copy()
    rel = result["relationship_type"].upper().replace("-", "_").replace(" ", "_")

    if rel in REMOVED_RELATIONSHIP_TYPES:
        return result, f"Removed relationship type: {rel}"

    if rel in INVERSE_RELATIONSHIP_TYPES:
        result["relationship_type"] = INVERSE_RELATIONSHIP_TYPES[rel]
        result["source_entity"], result["target_entity"] = result["target_entity"], result["source_entity"]
        return result, None

    if rel in DEPRECATED_RELATIONSHIP_TYPES:
        result["relationship_type"] = DEPRECATED_RELATIONSHIP_TYPES[rel]
        return result, None

    if rel in CANONICAL_RELATIONSHIP_TYPES:
        result["relationship_type"] = rel
        return result, None

    result["relationship_type"] = "USES"
    return result, f"Unknown relationship type '{fact['relationship_type']}', defaulting to USES"


def validate_entity_references(extraction: dict) -> list:
    """Return list of error messages. Empty = valid."""
    entity_names = {e.get("name", "") for e in extraction.get("entities", [])}
    errors = []
    for i, fact in enumerate(extraction.get("facts", [])):
        if fact["source_entity"] not in entity_names:
            errors.append(f"fact[{i}]: source_entity '{fact['source_entity']}' not in entities list")
        if fact["target_entity"] not in entity_names:
            errors.append(f"fact[{i}]: target_entity '{fact['target_entity']}' not in entities list")
    return errors


def validate_no_self_loops(extraction: dict) -> list:
    errors = []
    for i, fact in enumerate(extraction.get("facts", [])):
        if fact.get("source_entity") == fact.get("target_entity") and fact.get("source_entity") is not None:
            errors.append(f"fact[{i}]: self-relationship not allowed (source == target == '{fact.get('source_entity', '')}')")
    return errors


def validate_extraction(extraction: dict) -> tuple:
    """Returns (normalized_extraction, errors). Empty errors = valid."""
    errors = []

    for key in ("extraction_timestamp", "entities", "facts"):
        if key not in extraction:
            errors.append(f"Missing required key: {key}")
            return extraction, errors

    for i, entity in enumerate(extraction["entities"]):
        if "type" not in entity or "name" not in entity:
            errors.append(f"entity[{i}]: missing required fields (name, type)")
            continue
        canonical, err = normalize_entity_type(entity["type"])
        if err:
            logger.warning("entity[%d]: %s", i, err)
            errors.append(err)
        entity["type"] = canonical

    name_mapping = {}
    for i, entity in enumerate(extraction["entities"]):
        name = entity.get("name", "")
        original_name = name
        if len(name) > 200:
            entity["name"] = name[:200]
            name_mapping[original_name] = entity["name"]
            errors.append(f"entity[{i}]: name truncated to 200 chars")
        if len(name) < 2:
            errors.append(f"entity[{i}]: name too short ({len(name)} chars), minimum 2")
        summary = entity.get("summary", "")
        if len(summary) > 2000:
            entity["summary"] = summary[:2000]
            errors.append(f"entity[{i}]: summary truncated to 2000 chars")
        if len(summary) < 10:
            errors.append(f"entity[{i}]: summary too short ({len(summary)} chars), minimum 10")

    errors.extend(validate_entity_references(extraction))

    normalized_facts = []
    for i, fact in enumerate(extraction["facts"]):
        for field in ("source_entity", "target_entity", "relationship_type", "fact"):
            if field not in fact:
                errors.append(f"fact[{i}]: missing required field '{field}'")
                break
        else:
            normalized, err = normalize_relationship_type(fact)
            if err:
                logger.warning("fact[%d]: %s", i, err)
                errors.append(err)
                normalized_facts.append(normalized)  # keep fact with non-canonical type
            else:
                normalized_facts.append(normalized)

    for fact in normalized_facts:
        if fact["source_entity"] in name_mapping:
            fact["source_entity"] = name_mapping[fact["source_entity"]]
        if fact["target_entity"] in name_mapping:
            fact["target_entity"] = name_mapping[fact["target_entity"]]

    extraction["facts"] = normalized_facts
    errors.extend(validate_no_self_loops(extraction))

    for i, fact in enumerate(extraction["facts"]):
        fact_text = fact.get("fact", "")
        if len(fact_text) > 2000:
            fact["fact"] = fact_text[:2000]
            errors.append(f"fact[{i}]: fact truncated to 2000 chars")
        if len(fact_text) < 10:
            errors.append(f"fact[{i}]: fact too short ({len(fact_text)} chars), minimum 10")

    return extraction, errors


def text_similarity(a: str, b: str) -> float:
    """Normalized Levenshtein similarity. Returns 0.0 to 1.0."""
    if a == b:
        return 1.0
    if not a or not b:
        return 0.0
    a_norm = a.lower().strip()
    b_norm = b.lower().strip()
    if a_norm == b_norm:
        return 1.0
    max_len = max(len(a_norm), len(b_norm))
    if max_len == 0:
        return 1.0
    distance = levenshtein_distance(a_norm, b_norm)
    return 1.0 - (distance / max_len)


def should_supersede(old_fact: dict, new_fact: dict) -> bool:
    """Returns True if new_fact should replace old_fact."""
    if (old_fact["source_entity"], old_fact["target_entity"], old_fact["relationship_type"]) != \
       (new_fact["source_entity"], new_fact["target_entity"], new_fact["relationship_type"]):
        return False

    similarity = text_similarity(old_fact["fact"], new_fact["fact"])
    return similarity > 0.8
