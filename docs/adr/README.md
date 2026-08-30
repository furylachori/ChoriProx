# Architecture Decision Records

This directory contains Architecture Decision Records (ADRs) for ChoriProx.
Each ADR documents a significant architectural decision, its context,
alternatives considered, and consequences.

| ADR | Title | Status | Date |
|-----|-------|--------|------|
| 001 | [QueryExecutor over per-method session management](001-queryexecutor-over-per-method-sessions.md) | Accepted | 2026-06-02 |
| 002 | [RateLimiter as standalone class vs inline](002-ratelimiter-as-standalone-class.md) | Accepted | 2026-06-02 |
| 003 | [WorkspaceStateStore with compound operations](003-workspacestatestore-compound-operations.md) | Accepted | 2026-06-02 |
| 004 | [Eviction coordinator as separate module](004-eviction-coordinator-separate-module.md) | Accepted | 2026-06-02 |
| 005 | [Migration MERGE for concurrency protection](005-migration-merge-concurrency-protection.md) | Accepted | 2026-06-02 |
| 006 | [UNION branch 2 as permanent safety net](006-union-branch-2-permanent-safety-net.md) | Accepted | 2026-06-02 |
| 007 | [PostProcessStage private methods vs separate stages](007-postprocess-private-methods.md) | Accepted | 2026-06-02 |
| 008 | [state.py as thin re-export layer](008-state-py-thin-reexport-layer.md) | Accepted | 2026-06-02 |
| 009 | [Eager skill registration over lazy loading](009-eager-skill-registration.md) | Accepted | 2026-06-02 |
| 010 | [IoC callbacks for deactivation](010-ioc-callbacks-for-deactivation.md) | Accepted | 2026-06-02 |
| 011 | [No hot reload for custom JSON skills](011-custom-skill-restart-only.md) | Accepted | 2026-07-11 |
