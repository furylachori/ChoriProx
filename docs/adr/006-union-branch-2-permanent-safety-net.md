# ADR-006: UNION branch 2 as permanent safety net

## Status
Accepted (2026-06-02)

## Context

ChoriProx's task graph stores parent-child relationships between tasks using
a `parent_task_uuid` property on the child task node. The `get_child_tasks`
query in `task_graph_queries.py` retrieves children by matching this property.

However, the `parent_task_uuid` property was not part of the original schema.
Migration 001 (`001_backfill_parent_task_uuid.py`) was created to backfill
this property on existing task nodes by traversing the `CONTAINS`
relationships that were the original parent-child mechanism.

Two risks exist even after migration:

1. **New code paths creating tasks without `parent_task_uuid`** — If a future
   code path creates a child task via the `CONTAINS` relationship but forgets
   to set the `parent_task_uuid` property, the child becomes invisible to
   `get_child_tasks`.

2. **Migration partial failure** — If migration 001 fails partway through
   (e.g., Neo4j timeout on a large graph), some tasks retain only the
   `CONTAINS` relationship without the `parent_task_uuid` property.

Losing visibility of child tasks is a data-loss scenario: users cannot see
subtasks, blocker info is incomplete, and task dependency resolution fails
silently.

## Decision

Keep the `CONTAINS` fallback branch in `get_child_tasks` permanently as a
UNION query:

```cypher
MATCH (parent:Entity {short_hash: $hash})<-[:DEPENDS_ON]-(child:Entity)
WHERE parent.workspace_id = $wid
RETURN child

UNION

MATCH (parent:Entity {short_hash: $hash})-[:CONTAINS]->(child:Entity)
WHERE parent.workspace_id = $wid AND NOT exists(child.parent_task_uuid)
RETURN child
```

Branch 1 (DEPENDS_ON) is the primary path. Branch 2 (CONTAINS) catches
tasks that lack `parent_task_uuid`. The `NOT exists(child.parent_task_uuid)`
filter ensures branch 2 only returns tasks not already found by branch 1,
minimizing duplicate processing.

## Consequences

**Easier:**
- Child tasks are never silently lost, regardless of how they were created
  or whether migration completed fully.
- Future code changes that accidentally skip `parent_task_uuid` assignment
  are caught by the safety net rather than causing invisible data loss.
- Operational confidence: the query works correctly even on partially
  migrated databases.

**More difficult:**
- The UNION adds slight query execution overhead. Neo4j must evaluate both
  branches and deduplicate results. For typical task graphs (<100 children),
  this overhead is negligible (<1ms).
- The query is more complex to read and understand. Developers must know
  why branch 2 exists and that it's intentional, not dead code.
- Testing must cover both branches, including the edge case where a task
  appears in both branches.

## Alternatives Considered

1. **Remove branch 2 after migration** — Trust that migration 001 completed
   and all future code sets `parent_task_uuid`. Rejected because the risk
   of silent data loss outweighs the minimal performance cost.

2. **Application-level fallback** — Run the primary query first; if results
   seem incomplete, run a second query. Rejected because "seem incomplete"
   is impossible to determine without knowing the expected count, which
   requires the CONTAINS query anyway.

3. **Database trigger** — Neo4j trigger that auto-sets `parent_task_uuid`
   when a CONTAINS relationship is created. Rejected because Neo4j Community
   Edition doesn't support triggers, and APOC triggers add operational
   complexity.
