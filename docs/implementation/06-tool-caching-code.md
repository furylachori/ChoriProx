# Block: Tool Caching

> Part of [IMPLEMENTATION_SPEC](../IMPLEMENTATION_SPEC.md). See also [Spec](../spec/02-architecture.md)

## 7. Tool Caching

**File:** src/tool_optimizer.py (integrated with qwen_proxy.py:1254-1257)

### 7.1 Core Functions

- generate_tool_hash(): SHA-256 hash of canonical tool representation, 16-char output
- cache_tool_schema() / get_cached_tool_schema(): LRU cache (max 100 entries)
- optimize_tool_payload(): replaces full schemas with cached references
- restore_tool_payload(): restores full schemas before sending to LLM

### 7.2 Performance

- Payload reduction: 1,572 bytes -> 39 bytes (97.5%)
- Token savings: ~383 tokens per request
- Request time reduction: ~2.1 seconds per request
- Context capacity increase: +1,533 characters (7.9% more usable context)

### 7.3 API Endpoints

- GET /v1/tools/cache -- cache statistics
- POST /v1/tools/cache/clear -- clear cache
- GET /v1/tools/cache/<tool_hash> -- retrieve cached definitions

### 7.4 Integration

Called at qwen_proxy.py:1254-1257:
```python
if "tools" in llama_body:
    llama_body, tool_hash = optimize_tool_payload(llama_body)
```

Then recovery system runs at qwen_proxy.py:1260-1262.
