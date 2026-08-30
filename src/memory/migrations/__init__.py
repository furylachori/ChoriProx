from .runner import MigrationRunner
from .base import Migration
import importlib

_mod1 = importlib.import_module(".001_backfill_parent_task_uuid", __package__)
_mod2 = importlib.import_module(".002_verify_backfill", __package__)
_mod3 = importlib.import_module(".003_remove_project_relationships", __package__)
_mod4 = importlib.import_module(".004_graph_native_tasks", __package__)
_mod5 = importlib.import_module(".005_ensure_graph_native", __package__)
_mod6 = importlib.import_module(".006_agent_task_assignment", __package__)
_mod7 = importlib.import_module(".007_work_sessions", __package__)
_mod8 = importlib.import_module(".008_bug_tracking", __package__)
_mod9 = importlib.import_module(".009_workspace_backfill", __package__)
_mod10 = importlib.import_module(".010_strip_workspace_prefix", __package__)
_mod11 = importlib.import_module(".011_entity_dedup_constraint", __package__)
BackfillParentTaskUuid = _mod1.BackfillParentTaskUuid
VerifyBackfill = _mod2.VerifyBackfill
RemoveProjectRelationships = _mod3.RemoveProjectRelationships
GraphNativeTasks = _mod4.GraphNativeTasks
EnsureGraphNative = _mod5.EnsureGraphNative
AgentTaskAssignmentMigration = _mod6.AgentTaskAssignmentMigration
WorkSessionMigration = _mod7.WorkSessionMigration
BugTrackingMigration = _mod8.BugTrackingMigration
WorkspaceBackfill = _mod9.WorkspaceBackfill
StripWorkspacePrefix = _mod10.StripWorkspacePrefix
EntityDedupConstraint = _mod11.EntityDedupConstraint
