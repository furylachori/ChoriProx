"""Versioned migration runner with concurrent worker protection."""
import logging
import uuid
from memory.query_executor import QueryExecutor

logger = logging.getLogger(__name__)


class MigrationRunner:
    def __init__(self, executor: QueryExecutor):
        self._executor = executor
        self._migrations: list = []

    def register(self, migration) -> None:
        self._migrations.append(migration)
        self._migrations.sort(key=lambda m: m.version)

    def ensure_schema(self) -> None:
        self._executor.execute_write(
            "CREATE CONSTRAINT migration_version_unique IF NOT EXISTS "
            "FOR (m:_Migration) REQUIRE m.version IS UNIQUE"
        )

    def get_current_version(self) -> int:
        try:
            rows = self._executor.execute_read(
                "MATCH (m:_Migration) WHERE m.status = 'completed' "
                "RETURN max(m.version) AS version"
            )
            return rows[0]["version"] if rows and rows[0]["version"] else 0
        except Exception as e:
            logger.debug("get_current_version: Neo4j query failed: %s", e)
            return 0

    def run_pending(self) -> list[int]:
        self.ensure_schema()
        current = self.get_current_version()
        if current == 0:
            # Safety: if tasks exist but no _Migration markers, something is
            # wrong (likely a restore without markers). Refuse to run — a
            # re-run on already-migrated data is currently safe but would be
            # dangerous with a future non-idempotent migration.
            try:
                task_count = self._executor.execute_read(
                    "MATCH (t:Task) RETURN count(t) AS c"
                )
                if task_count and task_count[0].get("c", 0) > 0:
                    logger.critical(
                        "Migration version is 0 but %d tasks exist — refusing to run "
                        "migrations (possible restore without _Migration markers)",
                        task_count[0]["c"]
                    )
                    raise RuntimeError(
                        "Tasks exist but no _Migration markers. Run the restore "
                        "script to create markers, or manually MERGE _Migration nodes."
                    )
            except RuntimeError:
                raise
            except Exception:
                pass  # if the check fails, proceed (don't block on a read error)
        applied = []
        for m in self._migrations:
            if m.version <= current:
                continue
            worker_id = str(uuid.uuid4())
            lock_result = self._executor.execute_write(
                """MERGE (lock:MigrationLock {name: $migration_name})
                   ON CREATE SET lock.status = 'running', lock.worker_id = $worker_id, lock.started_at = datetime()
                   ON MATCH SET lock.status = CASE
                       WHEN lock.status = 'completed' THEN 'completed'
                       WHEN lock.status = 'running' AND lock.worker_id = $worker_id THEN 'running'
                       ELSE 'locked'
                   END
                   RETURN lock.status AS status""",
                {"migration_name": f"migration_{m.version}", "worker_id": worker_id}
            )
            lock_status = lock_result[0]["status"] if lock_result else "locked"
            if lock_status == "locked":
                logger.info("Migration %d already running on another worker, skipping", m.version)
                continue
            if lock_status == "completed":
                continue
            # Detect a previously-failed marker so we can flag the retry.
            # Must run BEFORE the marker is flipped to 'running' below.
            try:
                failed_rows = self._executor.execute_read(
                    "MATCH (m:_Migration {version: $v, status: 'failed'}) "
                    "RETURN count(m) AS c",
                    {"v": m.version}
                )
                if failed_rows and failed_rows[0].get("c", 0) > 0:
                    logger.warning(
                        "Retrying previously-failed migration %d: %s "
                        "(migrations are idempotent, retry is safe)",
                        m.version, m.description
                    )
            except Exception as e:
                logger.debug("Could not check failed status for migration %d: %s", m.version, e)
            logger.info("Running migration %d: %s", m.version, m.description)
            self._executor.execute_write(
                """MERGE (m:_Migration {version: $v})
                   ON CREATE SET m.status = 'running', m.started_at = datetime(), m.worker_id = $wid
                   ON MATCH SET m.status = 'running', m.started_at = datetime(), m.worker_id = $wid""",
                {"v": m.version, "wid": worker_id}
            )
            try:
                m.up(self._executor)
            except Exception as e:
                logger.error("Migration %d failed: %s", m.version, e, exc_info=True)
                try:
                    self._executor.execute_write(
                        "MATCH (m:_Migration {version: $v}) SET m.status = 'failed', m.error = $err",
                        {"v": m.version, "err": str(e)[:500]}
                    )
                except Exception as inner:
                    logger.debug("Could not write failure status for migration %d: %s", m.version, inner)
                raise
            self._executor.execute_write(
                "MATCH (m:_Migration {version: $v}) "
                "SET m.status = 'completed', m.completed_at = datetime(), m.error = null",
                {"v": m.version}
            )
            applied.append(m.version)
        return applied
