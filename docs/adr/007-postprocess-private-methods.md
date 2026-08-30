# ADR-007: PostProcessStage private methods vs separate stages

## Status
Accepted (2026-06-02)

## Context

ChoriProx's request pipeline is built on a `PipelineStage` protocol where
each stage's `execute(ctx)` method returns `Optional[dict]`:

- `None` — continue to the next stage.
- `dict` — short-circuit the pipeline and return this dict as the response.

This design works well for stages that either transform the request/response
or abort early (e.g., `MaliciousContentStage` returns an error dict to block
the request).

Post-processing, however, has different semantics. After the upstream LLM
responds, ChoriProx must:

1. Extract MiniMax reasoning blocks from the response.
2. Sync messages to the chat history cache.
3. Persist messages to SQLite.
4. Handle tool interception (parse tool calls, execute them via ToolPipeline,
   and potentially re-query the LLM).
5. Assign the model name to the response.
6. Run skill post-processing hooks.

Each of these steps modifies `ctx` in place and continues. They never
short-circuit (except tool interception, which may return a re-query result).
Making each a separate `PipelineStage` subclass would force them to return
`None` always, misusing the short-circuit protocol.

## Decision

Implement post-processing sub-stages as private methods on a single
`PostProcessStage` class (`src/proxy/pipeline/post_process.py`), not as
separate `PipelineStage` subclasses.

The `execute()` method iterates through private methods in order:

```python
methods = [
    self._extract_minimax_reasoning,
    self._sync_chat_history,
    self._persist_messages,
    self._handle_tool_interception,
    self._assign_model_name,
]
for method in methods:
    result = method(ctx)
    if result is not None and method == self._handle_tool_interception:
        return result  # Only tool interception can short-circuit
```

Each private method handles one concern and returns `None` on success or a
dict only when tool interception triggers a re-query.

Tool interception delegates to `ToolPipeline` as a black box, keeping the
tool execution logic isolated.

## Consequences

**Easier:**
- Post-processing logic is cohesive in one class, readable top-to-bottom.
- Adding a new post-processing step is a new private method + one line in
  the `methods` list.
- The `execute()` method clearly shows the processing order and which
  steps can short-circuit.
- Error handling is centralized: a failed sub-stage logs the error and
  continues, preventing one failure from blocking all post-processing.

**More difficult:**
- The pipeline architecture is less "pure" — `PostProcessStage` is a stage
  that contains sub-stages, which doesn't fit the flat stage list pattern.
- Tool interception's black-box delegation to `ToolPipeline` means the
  post-process stage has hidden complexity that isn't visible from the
  pipeline's perspective.
- Reordering sub-stages requires understanding their dependencies (e.g.,
  `_persist_messages` must run before `_handle_tool_interception` because
  tool results reference persisted message IDs).

## Alternatives Considered

1. **Separate PipelineStage subclasses** — Each sub-stage as its own class.
   Rejected because the `Optional[dict]` return protocol doesn't match
   post-processing semantics. Every sub-stage would return `None`, making
   the short-circuit mechanism dead code.

2. **Separate post-processing pipeline** — A second pipeline with a
   different stage protocol (no return value, only mutations). Rejected
   because it duplicates pipeline infrastructure for one use case and
   makes the overall request flow harder to follow.

3. **Event-based post-processing** — Emit events after upstream response;
   handlers process independently. Rejected because post-processing steps
   have ordering dependencies (persist before tool interception) and
   shared state (tool interception reads persisted messages).
