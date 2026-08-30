EXTRACTION_SCHEMA = {
    "required": ["extraction_timestamp", "entities", "facts"],
    "entities": {
        "required": ["name", "type", "summary"],
        "optional": [],
        "valid_types": ["Person", "Organization", "Project", "Technology", "File", "Concept", "Objective", "Bug", "Fix", "Task"]
    },
    "facts": {
        "required": ["source_entity", "target_entity", "relationship_type", "fact"],
        "optional": ["valid_at", "invalid_at"],
        "valid_relationship_types": ["USES", "CONTAINS", "CREATES", "SUPERSEDES", "DOCUMENTS", "CAUSES", "RESOLVES", "DEPENDS_ON", "PRECEDES", "ALTERNATIVE_TO"]
    },
    "events": {
        "required": ["topic_key", "content"],
        "optional": ["timeline"],
    },
    "instructions": {
        "required": ["topic_key", "content"],
        "optional": [],
    },
    "tasks": {
        "required": ["topic_key", "content"],
        "optional": ["priority"],
    },
}

EXTRACT_FROM_SUMMARIES_PROMPT = r"""
You are a knowledge graph curator. Your job is to read conversation summaries and identify
the named entities involved and how they relate to one another. This builds a structured,
searchable memory that answers questions like "what was I working on last week and why?"

Each entity is a specific, named thing — a person, project, technology, file, concept,
objective, bug, fix, or task. Each relationship connects two entities: one uses another,
contains another, creates another, and so on. The types below are your vocabulary.
Use only these.

## Entity Types (10)
- Person: Named humans
- Organization: Companies, teams
- Project: Named software projects
- Technology: Frameworks, languages, tools (absorbs: Platform, Service, API, Database, Tool)
- File: Specific file paths (absorbs: Document, Config, Schema, Template, Test)
- Concept: Abstract ideas, patterns
- Objective: Auto-extracted goals (no status, no priority)
- Bug: Defects or incorrect behavior
- Fix: Concrete remediations
- Task: Manual lifecycle items (NOT auto-extracted — for reference only)

## Relationship Types (10)
- USES: A uses B (lightweight)
- CONTAINS: A contains B (structural)
- CREATES: A creates B (lifecycle)
- SUPERSEDES: A replaces B (lifecycle)
- DOCUMENTS: A documents B
- CAUSES: A causes B (causation)
- RESOLVES: A resolves B (Bug->Fix)
- DEPENDS_ON: A depends on B (structural requirement)
- PRECEDES: A precedes B (temporal/logical ordering)
- ALTERNATIVE_TO: A is alternative to B (mutually exclusive)

## Rules
1. Only extract named entities. Skip generic terms.
2. Every fact must reference two distinct entities from your entities list.
3. Source and target entity names must EXACTLY match your entities list.
4. No self-relationships (source != target).
5. Use extraction_timestamp for present-tense facts. Set invalid_at only if explicitly stated.
6. Objective vs Task: Objective = goal/outcome. Task = concrete action item.

## Output Format
Submit via the extraction tool function with entities and facts arrays.
Empty arrays for any category with no extractions are fine.
""".strip()
