# Component: Memory System

> Part of [SPEC](../SPEC.md). See also [Implementation](../implementation/04-memory-system.md)

The memory system provides persistent recall, storage, and task management via SOTA-backed
tool calls. The proxy intercepts tool calls, routes memory tools to the SOTA extraction
pipeline, and passes non-memory tools through to the host client unchanged.

Memory tool set (`registry.get_memory_tool_names()`, `src/proxy/tool_pipeline/tool_registry.py`; the literal `MEMORY_TOOL_NAMES` was deleted in Tier 3 Phase 3, commit `93f5396`):
`recall`, `search`

Memory tools are classified at runtime — any tool call whose `name` is in `registry.get_memory_tool_names()`
enters the memory pipeline; all others are passthrough. SOTA (MiniMax) handles entity
extraction, supersession, and summary generation asynchronously. Dual storage uses SQLite
(metadata, full-text) and Neo4j (entity graph, CO_OCCURS edges). Summaries are injected into
context via SQLite read-through.

## Memory Known Gaps

Items below are **TODO** or **INTENTIONAL** deviations only (FIXED items omitted).

| # | Area | Classification | Gap | Action |
|---|------|---------------|-----|--------|
| 1 | `save_memory` IDs | INTENTIONAL + TODO | Uses `uuid4()` instead of SHA-256 content hash | Fix to content-addressed IDs |
| 4 | Full-text search | INTENTIONAL | SQL `CONTAINS` instead of FTS5 | Revisit at >10K records |
| 8 | Temporal resolution | TODO | `resolve_temporal_references()` never called; returns `"none"` | Integrate into `recall_for_context` |
| 9 | Rate limit handling | TODO | No `SOTARateLimitError`; generic catch-all | Add HTTP 429 detection |
| 10 | Client-side retry | INTENTIONAL | No retry; relies on MiniMax server-side retry | Add if server retry insufficient |
| 13 | Token budget | TODO | Hardcoded 200 tokens instead of 8% of context window | Calculate dynamically |
| 15 | Entity extraction | INTENTIONAL | Whitespace splitting; poor multi-word entities | Upgrade in Phase 4 (SOTA-assisted) |
| 16 | `RELATES_TO` edges | TODO | CO_OCCURS only; explicit RELATES_TO deferred | Add in Phase 4 |
| 17 | `search` graph | TODO | SQLite-only; no Neo4j dual-store search | Implement in Phase 4 |
| 18 | `manage_task` graph | TODO | Tasks not mirrored to Neo4j | Mirror in Phase 4 |
| 29 | Supersession atomicity | TODO | Two separate transactions (update old, insert new) | Wrap in single SQLite transaction |
