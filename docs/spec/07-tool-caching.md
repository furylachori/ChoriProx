# Component: Tool Caching

> Part of [SPEC](../SPEC.md). See also [Implementation](../implementation/07-tool-caching.md)

Caches static tool schemas in-memory and replaces full definitions with lightweight references on subsequent requests.

## `generate_tool_hash`

Creates a SHA-256 hash from canonical tool representation, sorts tools by name, produces a 16-character hash.

## Cache Management

- In-memory LRU cache, max 100 entries.
- `cache_tool_schema` / `get_cached_tool_schema` with hit/miss tracking.
- Automatic eviction when limit exceeded.

## Payload Optimization

`optimize_tool_payload` replaces tool schemas with `{ "$ref": "<hash>" }`. `restore_tool_payload` expands references back to full schemas before forwarding to LLM.

## Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/v1/tools/cache` | GET | Cache stats, hit ratios, stored hashes |
| `/v1/tools/cache/clear` | POST | Clear cache and reset statistics |
| `/v1/tools/cache/<hash>` | GET | Retrieve cached tool definitions |

## Performance

| Metric | Before | After |
|--------|--------|-------|
| Payload size | 1,572 B | 39 B (−97.5%) |
| Tokens/request | ~383 more | saved |
| Request time | baseline | −2.1 s |
| Usable context | 19,500 tok | 21,033 tok (+7.9%) |
