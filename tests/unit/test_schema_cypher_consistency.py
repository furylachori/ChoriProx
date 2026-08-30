"""Schema validation tests — catch Neo4j label/rel-type warnings before production.

These tests verify that every Cypher query in the persistence layer references
labels and relationship types that are registered in init_schema(). Without this,
MATCH on a never-created label/rel-type produces UnknownLabelWarning /
UnknownRelationshipTypeWarning in production logs.
"""
import re
import inspect

import pytest

# ── helpers ──────────────────────────────────────────────────────────────────

_NEVER_REFERENCED = {"_t", "_dummy", "_schema_init", "workspace_id"}


def _extract_labels_reltypes(cypher_strings: list[str]) -> tuple[set[str], set[str]]:
    """Parse Cypher strings and return (labels, rel_types) sets."""
    labels = set()
    rel_types = set()
    for cypher in cypher_strings:
        # Labels: `(:Label)`, `(n:Label)`, or `(n:Label {props})`.
        # Match colon+word after opening paren, capturing the word.
        for m in re.finditer(r'\([a-z]?:(\w+)', cypher):
            lbl = m.group(1)
            if lbl not in _NEVER_REFERENCED and not lbl[0].isdigit():
                labels.add(lbl)
        # Rel-types: `-[:REL_TYPE]->` or `-[:REL_TYPE {prop:val}]->`
        for m in re.finditer(r'\[:(\w+)\]', cypher):
            rt = m.group(1)
            if rt not in _NEVER_REFERENCED:
                rel_types.add(rt)
    return labels, rel_types


def _get_persistence_cypher() -> list[str]:
    """Return all Cypher strings from workspace_persistence.py."""
    cypher_strings = []
    import memory.workspace_persistence as wp
    for _, obj in inspect.getmembers(wp):
        if inspect.isfunction(obj):
            try:
                src = inspect.getsource(obj)
            except OSError:
                continue
            matches = re.findall(r'"""(.+?)"""', src, re.DOTALL)
            cypher_strings.extend(matches)
        elif isinstance(obj, str) and not hasattr(wp, '__name__'):
            # Module-level constants
            if len(obj) > 20:
                cypher_strings.append(obj)
    # Also check module-level constants
    for name in ('CATALOG_QUERY',):
        val = getattr(wp, name, None)
        if isinstance(val, str):
            cypher_strings.append(val)
    return cypher_strings


def _get_schema_block() -> str:
    """Return the `queries = [...]` block from graph.py's init_schema.

    Uses bracket balancing to find the closing `]` of the list literal,
    so string literals containing `]` (e.g. ``[:LAST_USED_WORKSPACE]``)
    don't prematurely terminate the block.
    """
    src = open('src/memory/graph.py').read()
    start = src.find('queries = [')
    if start == -1:
        return ''
    i = start + len('queries = [')
    depth = 1
    in_str = False
    str_ch = ''
    while i < len(src) and depth > 0:
        ch = src[i]
        if in_str:
            if ch == '\\':
                i += 2
                continue
            if ch == str_ch:
                in_str = False
        else:
            if ch in ('"', "'"):
                in_str = True
                str_ch = ch
            elif ch == '[':
                depth += 1
            elif ch == ']':
                depth -= 1
        i += 1
    return src[start:i]


def _get_schema_labels() -> set[str]:
    """Return all labels registered in init_schema()."""
    labels = set()
    block = _get_schema_block()
    for m in re.finditer(r'(?:FOR|ON)\s*\([^)]*:(\w+)', block):
        labels.add(m.group(1))
    for m in re.finditer(r'CREATE\s*\([^)]*:(\w+)', block):
        labels.add(m.group(1))
    return labels


# ── tests ────────────────────────────────────────────────────────────────────

class TestSchemaCypherConsistency:
    """Every label/rel-type in persistence Cypher must be registered in init_schema()."""

    def test_all_persistence_labels_in_schema(self):
        """Catch UnknownLabelWarning: labels MATCH'd but never CREATE'd."""
        persistence_labels, _ = _extract_labels_reltypes(_get_persistence_cypher())
        schema_labels = _get_schema_labels()

        missing = persistence_labels - schema_labels
        assert not missing, (
            f"Labels used in workspace_persistence.py but NOT registered in "
            f"init_schema(): {sorted(missing)}. Add a CREATE + DELETE no-op "
            f"to init_schema() for each missing label, or these will produce "
            f"UnknownLabelWarning in production."
        )

    def test_all_persistence_relationships_in_schema(self):
        """Catch UnknownRelationshipTypeWarning: rel-types MATCH'd on but never created."""
        _, persistence_rels = _extract_labels_reltypes(_get_persistence_cypher())

        # Relationship types are registered via CREATE (s)-[:REL]->(w) in init_schema().
        # Read the init_schema queries block for any relationship type creation.
        block = _get_schema_block()
        schema_rels = set()
        for m in re.finditer(r'\[:(\w+)\]', block):
            schema_rels.add(m.group(1))

        missing = persistence_rels - schema_rels
        assert not missing, (
            f"Rel-types used in workspace_persistence.py but NOT created in "
            f"init_schema(): {sorted(missing)}. Add a CREATE + DELETE no-op "
            f"to init_schema() for each missing rel-type, or these will produce "
            f"UnknownRelationshipTypeWarning in production."
        )

    def test_persistence_cypher_not_empty(self):
        """Sanity — the Cypher extraction must actually find something."""
        cypher_strings = _get_persistence_cypher()
        assert len(cypher_strings) >= 3, (
            f"Expected at least 3 Cypher strings in workspace_persistence.py, "
            f"got {len(cypher_strings)}. The extraction logic may be broken."
        )

    def test_schema_labels_not_empty(self):
        """Sanity — the schema label extraction must actually find something."""
        labels = _get_schema_labels()
        assert len(labels) >= 3, (
            f"Expected at least 3 labels in init_schema(), got {len(labels)}. "
            f"The extraction logic may be broken."
        )
