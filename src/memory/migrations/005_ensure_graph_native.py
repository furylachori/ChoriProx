"""Migration 005: Ensure graph-native schema for planning_info and review notes.

Fixes issues from migration 004:
- Promotes planning_info from attributes JSON to top-level property
- Converts review_notes (list in attributes or top-level property) to ReviewNote nodes
- Removes stale review_notes property and cleans up attributes
"""
import json
import logging
import uuid as uuid_mod
from memory.migrations.base import Migration
from memory.query_executor import QueryExecutor

logger = logging.getLogger(__name__)


class EnsureGraphNative(Migration):
    version = 5
    description = "Promote planning_info and convert review_notes to ReviewNote nodes"

    def up(self, executor: QueryExecutor) -> None:
        self._step0_promote_planning_info(executor)
        self._step1_convert_review_notes_to_nodes(executor)
        self._step2_cleanup_attributes_and_properties(executor)

    # ------------------------------------------------------------------
    # Step 0: promote planning_info from attributes to top-level property
    # ------------------------------------------------------------------
    def _step0_promote_planning_info(self, executor: QueryExecutor) -> None:
        """Extract planning_info from attributes JSON and set as t.planning_info.

        Idempotent: skips nodes that already have t.planning_info set.
        """
        marker = "_m005_s0"
        total_promoted = 0

        while True:
            result = executor.execute_read(
                f"""MATCH (t:Task)
                   WHERE t.attributes IS NOT NULL
AND t.planning_info IS NULL
                    AND t.{marker} IS NULL
                    RETURN id(t) AS nid, t.attributes AS attrs
                    LIMIT 500"""
            )
            if not result:
                break

            updates = []
            all_nids = []
            for row in result:
                all_nids.append(row["nid"])
                try:
                    attrs = json.loads(row["attrs"]) if isinstance(row["attrs"], str) else {}
                except (json.JSONDecodeError, TypeError):
                    attrs = {}
                planning_info = attrs.get("planning_info")
                if planning_info:
                    updates.append({
                        "nid": row["nid"],
                        "planning_info": planning_info,
                    })

            if updates:
                executor.execute_write(
                    """UNWIND $updates AS u
                       MATCH (t:Task) WHERE id(t) = u.nid
                       SET t.planning_info = u.planning_info""",
                    {"updates": updates},
                )
                total_promoted += len(updates)

            executor.execute_write(
                f"""UNWIND $nids AS nid
                   MATCH (t:Task) WHERE id(t) = nid
                   SET t.{marker} = true""",
                {"nids": all_nids},
            )

            if len(result) < 500:
                break

        self._cleanup_marker(executor, marker)
        logger.info("Migration 005 step0: promoted planning_info on %d tasks", total_promoted)

    # ------------------------------------------------------------------
    # Step 1: convert review_notes to ReviewNote nodes
    # ------------------------------------------------------------------
    def _step1_convert_review_notes_to_nodes(self, executor: QueryExecutor) -> None:
        """Create (:ReviewNote) nodes from review_notes data.

        Sources (checked in order):
          1. attributes.review_notes  (JSON string, pre-migration-004)
          2. t.review_notes           (top-level property set by migration 004)

        Idempotent: skips tasks that already have [:HAS_NOTE] relationships.
        """
        marker = "_m005_s1"
        from proxy.utils import _now_iso
        now = _now_iso()
        total_notes = 0
        total_tasks = 0

        while True:
            result = executor.execute_read(
                f"""MATCH (t:Task)
                   WHERE (t.attributes IS NOT NULL OR t.review_notes IS NOT NULL)
AND t.{marker} IS NULL
                    AND NOT exists((t)-[:HAS_NOTE]->(:ReviewNote))
                    RETURN id(t) AS nid, t.attributes AS attrs, t.review_notes AS rn_prop
                    LIMIT 500"""
            )
            if not result:
                break

            all_nids = []
            note_creates = []
            for row in result:
                all_nids.append(row["nid"])
                review_notes = self._extract_review_notes(row.get("attrs"), row.get("rn_prop"))
                if not review_notes:
                    continue

                for rn in review_notes:
                    if not isinstance(rn, dict):
                        continue
                    note_creates.append({
                        "nid": row["nid"],
                        "uuid": f"review-{uuid_mod.uuid4().hex}",
                        "ts": rn.get("ts", now),
                        "note": rn.get("note", ""),
                        "author": rn.get("author", "unknown"),
                        "created_at": rn.get("ts", now),
                    })

            if note_creates:
                executor.execute_write(
                    """UNWIND $notes AS n
                       MATCH (t:Task) WHERE id(t) = n.nid
                       CREATE (rn:ReviewNote {uuid: n.uuid, ts: n.ts, note: n.note, author: n.author})
                       CREATE (t)-[:HAS_NOTE {created_at: n.created_at}]->(rn)""",
                    {"notes": note_creates},
                )
                total_notes += len(note_creates)
                total_tasks += len(set(n["nid"] for n in note_creates))

            executor.execute_write(
                f"""UNWIND $nids AS nid
                   MATCH (t:Task) WHERE id(t) = nid
                   SET t.{marker} = true""",
                {"nids": all_nids},
            )

            if len(result) < 500:
                break

        self._cleanup_marker(executor, marker)
        logger.info(
            "Migration 005 step1: created %d ReviewNote nodes across %d tasks",
            total_notes, total_tasks,
        )

    # ------------------------------------------------------------------
    # Step 2: clean up attributes and stale properties
    # ------------------------------------------------------------------
    def _step2_cleanup_attributes_and_properties(self, executor: QueryExecutor) -> None:
        """Remove planning_info and review_notes from attributes JSON.

        If attributes becomes empty, remove the property entirely.
        Also remove the stale t.review_notes property (set by migration 004).
        """
        marker = "_m005_s2"
        total_cleaned = 0
        total_removed = 0

        # --- 2a: strip planning_info and review_notes from attributes ---
        while True:
            result = executor.execute_read(
                f"""MATCH (t:Task)
                   WHERE t.attributes IS NOT NULL
AND t.{marker} IS NULL
                    RETURN id(t) AS nid, t.attributes AS attrs
                    LIMIT 500"""
            )
            if not result:
                break

            all_nids = []
            updates_with_attrs = []
            nids_to_remove_attrs = []
            for row in result:
                all_nids.append(row["nid"])
                try:
                    attrs = json.loads(row["attrs"]) if isinstance(row["attrs"], str) else {}
                except (json.JSONDecodeError, TypeError):
                    attrs = {}

                modified = False
                for key in ("planning_info", "review_notes"):
                    if key in attrs:
                        del attrs[key]
                        modified = True

                if modified:
                    if attrs:
                        updates_with_attrs.append({"nid": row["nid"], "attrs": json.dumps(attrs)})
                    else:
                        nids_to_remove_attrs.append(row["nid"])

            if updates_with_attrs:
                executor.execute_write(
                    """UNWIND $updates AS u
                       MATCH (t:Task) WHERE id(t) = u.nid
                       SET t.attributes = u.attrs""",
                    {"updates": updates_with_attrs},
                )
                total_cleaned += len(updates_with_attrs)

            if nids_to_remove_attrs:
                executor.execute_write(
                    """UNWIND $nids AS nid
                       MATCH (t:Task) WHERE id(t) = nid
                       REMOVE t.attributes""",
                    {"nids": nids_to_remove_attrs},
                )
                total_removed += len(nids_to_remove_attrs)

            executor.execute_write(
                f"""UNWIND $nids AS nid
                   MATCH (t:Task) WHERE id(t) = nid
                   SET t.{marker} = true""",
                {"nids": all_nids},
            )

            if len(result) < 500:
                break

        self._cleanup_marker(executor, marker)

        # --- 2b: remove stale t.review_notes property (set by migration 004) ---
        removed_prop = 0
        while True:
            r = executor.execute_write(
                """MATCH (t:Task) WHERE t.review_notes IS NOT NULL
                   WITH t LIMIT 500
                   REMOVE t.review_notes
                   RETURN count(*) AS cleaned"""
            )
            cleaned = r[0]["cleaned"] if r else 0
            removed_prop += cleaned
            if cleaned < 500:
                break

        logger.info(
            "Migration 005 step2: cleaned %d attributes, removed %d empty attributes, "
            "removed review_notes property from %d tasks",
            total_cleaned, total_removed, removed_prop,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _extract_review_notes(attrs, rn_prop):
        """Extract review_notes list from attributes JSON or top-level property.

        Returns a list of dicts or None.
        """
        # Prefer attributes source (pre-migration-004)
        if attrs:
            try:
                parsed = json.loads(attrs) if isinstance(attrs, str) else attrs
            except (json.JSONDecodeError, TypeError):
                parsed = {}
            notes = parsed.get("review_notes") if isinstance(parsed, dict) else None
            if notes and isinstance(notes, list):
                return notes

        # Fall back to top-level property (set by migration 004)
        if rn_prop is not None:
            if isinstance(rn_prop, list):
                return rn_prop
            if isinstance(rn_prop, str):
                try:
                    parsed = json.loads(rn_prop)
                    if isinstance(parsed, list):
                        return parsed
                except (json.JSONDecodeError, TypeError):
                    pass

        return None

    @staticmethod
    def _cleanup_marker(executor: QueryExecutor, marker: str) -> None:
        """Remove temporary marker property from all Task nodes."""
        while True:
            r = executor.execute_write(
                f"""MATCH (t:Task) WHERE t.{marker} IS NOT NULL
                   WITH t LIMIT 500
                   REMOVE t.{marker}
                   RETURN count(*) AS cleaned"""
            )
            cleaned = r[0]["cleaned"] if r else 0
            if cleaned < 500:
                break
