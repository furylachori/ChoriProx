# API Versioning

## Why Versioning Was Added

ChoriProx introduced API versioning to allow backward-incompatible changes to be introduced without breaking existing clients. The `/v1/` API is now deprecated in favor of `/v2/`.

## What Changed Between v1 and v2

Currently, `/v1/` and `/v2/` are **identical** in request and response format. The `/v2/` prefix exists as the forward-compatible path for future changes.

## How to Migrate

Replace `/v1/` with `/v2/` in all client requests:

```
# Before (deprecated)
POST /v1/chat/completions
GET  /v1/models
GET  /v1/context/config

# After (current)
POST /v2/chat/completions
GET  /v2/models
GET  /v2/context/config
```

The `/health` endpoint is **not versioned** and remains at `/health`.

## Deprecation Timeline

| Milestone | Date |
|-----------|------|
| v2 introduced | June 2026 |
| v1 sunset | November 1, 2026 |

After the sunset date, `/v1/` endpoints may return `410 Gone` or be removed entirely.

## Detecting Deprecation

All `/v1/` responses include the following headers:

```
Deprecation: true
Sunset: Sat, 01 Nov 2026 00:00:00 GMT
Link: </v2/...>; rel="successor-version"
```

The `Link` header points to the equivalent `/v2/` endpoint for the current request path.

Clients should check for the `Deprecation: true` header to detect deprecated endpoints programmatically.

## Versioned Endpoints

| Endpoint | Methods |
|----------|---------|
| `/v{1,2}/chat/completions` | POST |
| `/v{1,2}/models` | GET |
| `/v{1,2}/context/config` | GET |
| `/v{1,2}/context/config/toggle` | POST |
| `/v{1,2}/context/sessions` | GET |
| `/v{1,2}/context/session/<session_id>` | GET |
| `/v{1,2}/tools/cache` | GET |
| `/v{1,2}/tools/cache/clear` | POST |
| `/v{1,2}/tools/cache/<tool_hash>` | GET |
| `/v{1,2}/debug/agents` | GET |
| `/v{1,2}/debug/agents/<agent_id>` | GET |
| `/v{1,2}/debug/agents/<agent_id>/history` | GET |
| `/health` | GET (not versioned) |
