# USER-STORIES.md Coverage Inventory

> **Generated:** 2026-08-30
> **Total stories:** 138
> **Active stories (excluding superseded/unimplemented/reserved):** 136
> **Referenced by at least one test:** 108 (79%)

## Coverage by section

| Section | Active | Referenced | % |
|---|---|---|---|
| Circuit Breaker & Neo4j Isolation | 4 | 4 | 100% |
| Session Tokens & Canary System | 6 | 5 | 83% |
| Rate Limiting | 3 | 3 | 100% |
| Skill System & Mode Management | 14 | 13 | 93% |
| Workspace Skill Commands | 7 | 7 | 100% |
| Memory System & Injection | 4 | 4 | 100% |
| Tool Loop | 3 | 3 | 100% |
| Context Optimization | 3 | 3 | 100% |
| Grep Sandbox | 2 | 2 | 100% |
| Provider Compatibility | 3 | 3 | 100% |
| Tool-Call Normalization | 2 | 2 | 100% |
| Post-Processing | 2 | 2 | 100% |
| Migrations | 2 | 2 | 100% |
| HTTP Client | 3 | 3 | 100% |
| State Module | 2 | 2 | 100% |
| Pipeline Error Handling | 4 | 3 | 75% |
| CLI Agent Requirements | 3 | 3 | 100% |
| Task System | 1 | 0 | 0% |
| Security & Authentication | 2 | 2 | 100% |
| VFS (Virtual Filesystem Layer) | 7 | 7 | 100% |
| Streaming | 3 | 3 | 100% |
| Model Routing | 1 | 1 | 100% |
| Admin Endpoints | 2 | 2 | 100% |
| Agent Registry | 1 | 1 | 100% |
| Motivator | 1 | 1 | 100% |
| Ticket-Driven Environment: Task Assignment | 4 | 4 | 100% |
| Ticket-Driven Environment: Work Sessions | 4 | 4 | 100% |
| Ticket-Driven Environment: Bug Reporting | 8 | 8 | 100% |
| Ticket-Driven Environment: Handoff | 2 | 2 | 100% |
| Ticket-Driven Environment: Action Protocol | 1 | 1 | 100% |
| Ticket-Driven Environment: Work Mode | 8 | 8 | 100% |
| Workspace Model v5: Global Catalog (Replaces Per-Session US-066..US-072) | 24 | 0 | 0% |

## Unreferenced active stories

| Story | Title | Reason for gap |
|---|---|---|
| US-006 | Canary Verification Fail-Safe | No test in tests/ references this ID |
| US-017 | Ticket MAIN Tool Surface | No test in tests/ references this ID |
| US-044a | Malformed Chat Request — Structured 400 Envelope | No test in tests/ references this ID |
| US-050 | Fresh Task List for Agents | No test in tests/ references this ID |
| US-100 | Workspace ID is a Bare Name (No Session Prefix) | No test in tests/ references this ID |
| US-101 | Global Catalog of Workspaces | No test in tests/ references this ID |
| US-102 | Task Visibility is Per-Workspace, Not Per-Session | No test in tests/ references this ID |
| US-103 | Migration 010 Strips Prefixes from Existing Data | No test in tests/ references this ID |
| US-104 | Migration 010 Ordering — Code First, Then Restart | No test in tests/ references this ID |
| US-105 | Old Compound Workspace Nodes Kept as Soft Backup | No test in tests/ references this ID |
| US-106 | Per-Session Last-Used Workspace Tracked via `:LAST_USED_WORKSPACE` | No test in tests/ references this ID |
| US-107 | Recovery on Cold Start Restores Per-Session Last-Used | No test in tests/ references this ID |
| US-108 | Typo Guard Checks Global Catalog | No test in tests/ references this ID |
| US-109 | Workspace Creation is One-Shot Global, Not Per-Session | No test in tests/ references this ID |
| US-110 | Persist Callback is Retried on Every Call (Lazy Re-Wire) | No test in tests/ references this ID |
| US-111 | Cleanup Evicts Per-Session State, NOT Global Workspaces | No test in tests/ references this ID |
| US-112 | SESSION_TTL Bumped to 8 Hours for Per-Session In-Memory State | No test in tests/ references this ID |
| US-113 | Sub-Agent Inherits Parent's Workspace via Canary | No test in tests/ references this ID |
| US-114 | Cross-Session Task Sharing is the Default | No test in tests/ references this ID |
| US-115 | Cleanup Does NOT Delete Last-Used Edge for Other Sessions | No test in tests/ references this ID |
| US-116 | VFS Cache Key is Bare Workspace Name | No test in tests/ references this ID |
| US-117 | Migration 009 (Rewritten) Backfills Bare-Name Workspaces | No test in tests/ references this ID |
| US-118 | Cleanup Race is Eliminated (v4 Latent Bug Fixed) | No test in tests/ references this ID |
| US-119 | All 12 `make_workspace_id` Caller Sites Use Bare Name | No test in tests/ references this ID |
| US-120 | Cleanup is Batched in 1000-Session Chunks | No test in tests/ references this ID |
| US-121 | `^^workspace list` Catalog is Cached with TTL | No test in tests/ references this ID |
| US-122 | Recovery Does NOT Stampede Neo4j | No test in tests/ references this ID |
| US-123 | Failed Migration 010 Is Recoverable by Deleting `_Migration` Marker | No test in tests/ references this ID |

## Suppressed (not in denominator)

| Story | Reason |
|---|---|
| US-018 | SUPERSEDED by US-073 |
| US-049 | Reserved — Intentionally Skipped |
