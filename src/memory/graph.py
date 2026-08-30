import contextlib
import logging
import os
import threading
from collections import OrderedDict
from typing import Any, Optional

from proxy.utils import _now_iso

try:
    import neo4j
    from neo4j import GraphDatabase
    from neo4j import Query
    from neo4j.exceptions import ClientError, ConstraintError
except ImportError:
    neo4j = None
    GraphDatabase = None
    Query = None
    ClientError = None
    ConstraintError = None

_SCHEMA_RACE_ERRORS = tuple(
    exc for exc in (ClientError, ConstraintError) if exc is not None
)

from memory.circuit_breaker import CircuitBreakerOpen, CircuitState
from memory.query_executor import QueryExecutor, InfrastructureError, ConstraintViolation

# Extend schema race detection to include our wrapped ConstraintViolation
# (mapped from neo4j ConstraintError by _map_neo4j_error). Without this,
# init_schema() treats dummy-node cleanup constraint violations as fatal.
_SCHEMA_RACE_ERRORS = _SCHEMA_RACE_ERRORS + (ConstraintViolation,)

logger = logging.getLogger("memory.graph")
logging.getLogger("neo4j.io").setLevel(logging.WARNING)
logging.getLogger("neo4j").setLevel(logging.WARNING)
logging.getLogger("neo4j.pool").setLevel(logging.WARNING)
logging.getLogger("neo4j.notifications").setLevel(logging.WARNING)

DEFAULT_NEO4J_TIMEOUT = 30  # seconds
DEFAULT_SEARCH_LIMIT = 50
MAX_ENTITIES_PER_CALL = 100

STOPWORDS = frozenset(
    {
        "the",
        "a",
        "is",
        "are",
        "was",
        "were",
        "in",
        "on",
        "at",
        "to",
        "for",
        "of",
        "with",
        "by",
        "from",
        "as",
        "an",
        "be",
        "have",
        "has",
        "had",
        "do",
        "does",
        "did",
        "will",
        "would",
        "could",
        "should",
        "may",
        "might",
        "can",
        "i",
        "we",
        "you",
        "he",
        "she",
        "it",
        "they",
        "my",
        "your",
        "his",
        "her",
        "its",
        "our",
        "their",
        "this",
        "that",
        "these",
        "those",
        "and",
        "or",
        "but",
        "not",
        "no",
        "yes",
        "what",
        "when",
        "where",
        "which",
        "who",
        "how",
        "why",
        "if",
        "then",
        "than",
        "so",
        "very",
        "just",
        "also",
        "about",
        "up",
        "out",
        "into",
        "some",
        "all",
        "any",
        "each",
        "every",
        "other",
        "more",
        "most",
        "much",
        "such",
        "only",
        "own",
        "same",
        "been",
        "being",
        "both",
        "few",
        "over",
        "here",
        "there",
        "hello",
        "hi",
        "hey",
        "ok",
        "okay",
        "please",
        "thanks",
        "thank",
        "see",
    }
)


class Neo4jClient:
    _supersession_locks_max: int = 1024
    _supersession_locks: OrderedDict[str, threading.Lock] = OrderedDict()
    _global_lock = threading.Lock()

    @staticmethod
    def _record_to_memory_dict(record):
        return {
            "id": record.get("id"),
            "memory_type": record.get("memory_type"),
            "content": record.get("content"),
            "topic_key": record.get("topic_key"),
            "session_id": record.get("session_id"),
            "created_at": record.get("created_at"),
            "valid_at": record.get("valid_at"),
            "invalid_at": record.get("invalid_at"),
            "superseded_by": record.get("superseded_by"),
            "confidence": record.get("confidence"),
            "last_accessed_at": record.get("last_accessed_at", ""),
            "access_count": record.get("access_count", 0),
        }

    @staticmethod
    def _memory_to_store_params(memory):
        return {
            "id": memory.get("id"),
            "memory_type": memory.get("memory_type"),
            "content": memory.get("content"),
            "topic_key": memory.get("topic_key"),
            "session_id": memory.get("session_id"),
            "created_at": memory.get("created_at"),
            "valid_at": memory.get("valid_at"),
            "invalid_at": memory.get("invalid_at"),
            "superseded_by": memory.get("superseded_by"),
            "confidence": memory.get("confidence"),
            "last_accessed_at": memory.get("last_accessed_at", ""),
            "access_count": memory.get("access_count", 0),
        }

    def __init__(
        self, bolt_url: Optional[str] = None, auth: Optional[tuple[str, str]] = None
    ) -> None:
        """Initialize Neo4j client with connection to graph database.

        Establishes a connection to Neo4j using the provided bolt URL and
        authentication credentials. Falls back to environment variables
        PROXY_NEO4J_BOLT_URL and NEO4J_AUTH if not provided.

        Args:
            bolt_url: Neo4j bolt URL (e.g., "bolt://localhost:7687").
                Defaults to PROXY_NEO4J_BOLT_URL env var or "bolt://localhost:7687".
            auth: Tuple of (username, password) for Neo4j authentication.
                Defaults to parsing NEO4J_AUTH env var (format: "user:password").
        """
        if neo4j is None:
            self._executor = QueryExecutor(None, None, None)
            self._driver = self._executor._driver
            return

        if bolt_url is None:
            bolt_url = os.environ.get("PROXY_NEO4J_BOLT_URL", "bolt://localhost:7687")

        if auth is None:
            neo4j_auth = os.environ.get("NEO4J_AUTH", "")
            if neo4j_auth:
                parts = neo4j_auth.split(":", 1)
                if len(parts) == 2:
                    auth = tuple(parts)
                else:
                    auth = (neo4j_auth, "")
            else:
                auth = ("neo4j", "")

        try:
            circuit_breaker = self._get_circuit_breaker()
        except Exception:
            logger.debug("Failed to get circuit breaker, proceeding without it", exc_info=True)
            circuit_breaker = None
        self._executor = QueryExecutor(bolt_url, auth, circuit_breaker)
        self._driver = self._executor._driver

    def close(self) -> None:
        """Close the Neo4j connection and release resources.

        Safely closes the underlying QueryExecutor and Neo4j driver.
        Idempotent — safe to call multiple times.
        """
        if hasattr(self, "_executor"):
            self._executor.close()

    def update_access_tracking(self, memory_ids: list[str]) -> None:
        """Update access tracking metadata for a batch of memories.

        Increments access_count and updates last_accessed_at timestamp for
        each memory ID. Used to track memory usage patterns for relevance
        scoring.

        Args:
            memory_ids: List of memory IDs to update. Empty list is a no-op.
        """
        if not memory_ids:
            return

        query = """
        MATCH (m:Memory) WHERE m.id IN $ids
        SET m.last_accessed_at = $now,
            m.access_count = COALESCE(m.access_count, 0) + 1
        """

        now = _now_iso()

        try:
            self._executor.execute_write(query, {"ids": memory_ids, "now": now}, timeout=5)
            logger.info("Access tracking updated for %d memories", len(memory_ids))
        except InfrastructureError as e:
            logger.error("Access tracking failed: %s", e)

    def _is_available(self) -> bool:
        if self._driver is None:
            return False
        circuit = self._get_circuit_breaker()
        if circuit.state == CircuitState.OPEN:
            logger.warning("Neo4j circuit breaker is OPEN — all operations will return sentinel values")
            return False
        return True

    @staticmethod
    def _get_circuit_breaker():
        import proxy.state as state

        return state.neo4j_circuit_breaker



    @classmethod
    def _get_topic_lock(cls, topic_key: str) -> threading.Lock:
        with cls._global_lock:
            if topic_key in cls._supersession_locks:
                cls._supersession_locks.move_to_end(topic_key)
                return cls._supersession_locks[topic_key]
            while len(cls._supersession_locks) >= cls._supersession_locks_max:
                oldest_key, oldest_lock = next(iter(cls._supersession_locks.items()))
                if oldest_lock.locked():
                    break
                del cls._supersession_locks[oldest_key]
            if len(cls._supersession_locks) >= cls._supersession_locks_max:
                for key, lk in list(cls._supersession_locks.items()):
                    if not lk.locked():
                        del cls._supersession_locks[key]
                        break
                else:
                    raise RuntimeError(
                        f"All {cls._supersession_locks_max} topic locks are held concurrently. "
                        "Increase NEO4J_TOPIC_LOCK_MAX or reduce concurrency."
                    )
            lock = threading.Lock()
            cls._supersession_locks[topic_key] = lock
            return lock

    def init_schema(self) -> None:
        """Initialize Neo4j schema with constraints and indexes.

        Creates uniqueness constraints and indexes for Memory, Entity, and
        Project nodes. Idempotent — safe to call multiple times. Handles
        race conditions when multiple instances initialize concurrently.

        Note: On a database with pre-existing duplicate (name, entity_type)
        Entity rows, the `entity_name_type_unique` constraint creation below
        raises a "data violation" ClientError. That error contains the word
        "constraint" and is swallowed by `_is_schema_race_error` below — the
        migration runner then creates the constraint on its next pass after
        deduplicating the offending rows via migration 011.

        Raises:
            InfrastructureError: If Neo4j is unavailable and error is not
                a schema race condition.
            ConstraintViolation: If schema creation fails for non-race reasons.
        """
        if not self._is_available():
            return

        queries = [
            "CREATE CONSTRAINT memory_id_unique IF NOT EXISTS FOR (m:Memory) REQUIRE m.id IS UNIQUE",
            "CREATE CONSTRAINT entity_name_type_unique IF NOT EXISTS "
            "FOR (e:Entity) REQUIRE (e.name, e.entity_type) IS UNIQUE",
            "DROP CONSTRAINT task_name_session_unique IF EXISTS",
            "CREATE INDEX memory_type_index IF NOT EXISTS FOR (m:Memory) ON (m.memory_type)",
            "CREATE INDEX memory_topic_key_index IF NOT EXISTS FOR (m:Memory) ON (m.topic_key)",
            "CREATE INDEX memory_session_id_index IF NOT EXISTS FOR (m:Memory) ON (m.session_id)",
            "CREATE INDEX entity_name_index IF NOT EXISTS FOR (e:Entity) ON (e.name)",
            "CREATE CONSTRAINT workspace_id_unique IF NOT EXISTS FOR (w:Workspace) REQUIRE w.workspace_id IS UNIQUE",
            "DROP INDEX workspace_session_idx IF EXISTS",
            "CREATE INDEX workspace_last_active_idx IF NOT EXISTS FOR (w:Workspace) ON (w.last_active_at)",
            # Register :Session label so recovery queries don't warn.
            # Create-and-delete is a no-op that registers the label.
            "CREATE (dummy:Session {_t: 1}) DELETE dummy",
            # Register :LAST_USED_WORKSPACE rel-type. Create dummy nodes,
            # create the relationship, then clean up. Registers both
            # the label and relationship type regardless of existing data.
            "CREATE (s:Session {_t: 2}) "
            "CREATE (w:Workspace {workspace_id: '_schema_init', _t: 2}) "
            "CREATE (s)-[:LAST_USED_WORKSPACE]->(w) "
            "DETACH DELETE s, w",
            # Register :Task and :Bug labels so workspace cleanup queries
            # (workspace_persistence.CLEANUP_QUERY) that reference them in
            # EXISTS subqueries don't emit UnknownLabelWarning. The Task/Bug
            # constraints live in task_graph.py; these are label-registration
            # no-ops only. Create-and-delete registers the label.
            "CREATE (dummy:Task {_t: 1}) DELETE dummy",
            "CREATE (dummy:Bug {_t: 1}) DELETE dummy",
        ]

        def _is_schema_race_error(e: Exception) -> bool:
            if not _SCHEMA_RACE_ERRORS or not isinstance(e, _SCHEMA_RACE_ERRORS):
                return False
            msg = str(e).lower()
            return "already exists" in msg or "constraint" in msg or "equivalent" in msg

        for query in queries:
            try:
                self._executor.execute_write(query)
            except (InfrastructureError, ConstraintViolation) as e:
                if _is_schema_race_error(e):
                    logger.debug("Schema query skipped (already exists): %s", query)
                else:
                    logger.error("Schema query failed: %s", query, exc_info=True)
                    raise

    def store_memory(self, memory: dict[str, Any]) -> bool:
        """Store or update a memory node in Neo4j.

        Uses MERGE to create or update a Memory node with the given ID.
        All memory attributes (type, content, topic_key, etc.) are set
        from the provided dictionary.

        Args:
            memory: Dictionary containing memory attributes:
                - id: Unique memory identifier (required)
                - memory_type: Type of memory (event, fact, etc.)
                - content: Memory content text
                - topic_key: Optional topic for grouping
                - session_id: Session that created this memory
                - created_at, valid_at, invalid_at: Timestamps
                - superseded_by: ID of superseding memory
                - confidence: Confidence score
                - last_accessed_at, access_count: Access tracking

        Returns:
            bool: True if stored successfully, False on InfrastructureError.
        """
        query = """
        MERGE (m:Memory {id: $id})
        SET m.memory_type = $memory_type,
            m.content = $content,
            m.topic_key = $topic_key,
            m.session_id = $session_id,
            m.created_at = $created_at,
            m.valid_at = $valid_at,
            m.invalid_at = $invalid_at,
            m.superseded_by = $superseded_by,
            m.confidence = $confidence,
            m.last_accessed_at = $last_accessed_at,
            m.access_count = $access_count
        RETURN m
        """

        params = Neo4jClient._memory_to_store_params(memory)

        try:
            self._executor.execute_write(query, params)
            logger.info(
                "Stored memory %s (type=%s) session_id=%s",
                memory.get("id"),
                memory.get("memory_type"),
                memory.get("session_id"),
            )
            return True
        except InfrastructureError as e:
            logger.error("Failed to store memory %s: %s session_id=%s", memory.get("id"), e, memory.get("session_id"))
            return False

    def store_with_supersession(self, memory: dict[str, Any]) -> bool:
        """Store a memory with automatic supersession tracking.

        If the memory has a topic_key, finds any existing memory with the
        same topic_key and memory_type that hasn't been superseded, marks
        it as superseded by the new memory, and sets its invalid_at timestamp.
        Uses per-topic locking to prevent race conditions.

        Args:
            memory: Dictionary containing memory attributes (see store_memory).
                Must include topic_key for supersession tracking.

        Returns:
            bool: True if stored successfully, False on InfrastructureError.
        """
        topic_key = memory.get("topic_key")
        if not topic_key:
            return self.store_memory(memory)

        lock = self._get_topic_lock(topic_key)
        with lock:
            def _tx(tx):
                result = tx.run(
                    """MATCH (m:Memory {topic_key: $topic_key, memory_type: $memory_type})
                       WHERE m.superseded_by IS NULL
                       RETURN m.id AS id LIMIT 1""",
                    {
                        "topic_key": topic_key,
                        "memory_type": memory.get("memory_type"),
                    },
                )
                records = list(result)
                result.consume()
                existing_id = records[0].get("id") if records else None

                params = Neo4jClient._memory_to_store_params(memory)
                params["topic_key"] = topic_key
                tx.run(
                    """MERGE (m:Memory {id: $id})
                       SET m.memory_type = $memory_type,
                           m.content = $content,
                           m.topic_key = $topic_key,
                           m.session_id = $session_id,
                           m.created_at = $created_at,
                           m.valid_at = $valid_at,
                           m.invalid_at = $invalid_at,
                           m.superseded_by = $superseded_by,
                           m.confidence = $confidence,
                           m.last_accessed_at = $last_accessed_at,
                           m.access_count = $access_count
                       RETURN m""",
                    params,
                )

                if existing_id:
                    now = memory.get("created_at")
                    if now is None:
                        now = _now_iso()
                    tx.run(
                        """MATCH (old:Memory {id: $old_id})
                           SET old.superseded_by = $new_id, old.invalid_at = $invalid_at""",
                        {
                            "old_id": existing_id,
                            "new_id": memory.get("id"),
                            "invalid_at": now,
                        },
                    )
                    logger.info(
                        "Superseded memory %s with %s (topic=%s) session_id=%s",
                        existing_id,
                        memory.get("id"),
                        topic_key,
                        memory.get("session_id"),
                    )

                logger.info(
                    "Stored memory %s (type=%s) session_id=%s",
                    memory.get("id"),
                    memory.get("memory_type"),
                    memory.get("session_id"),
                )
                return True

            try:
                return self._executor.execute_transaction(_tx)
            except InfrastructureError as e:
                logger.warning("store_with_supersession failed session_id=%s: %s", memory.get("session_id"), e)
                return False

    def store_entity(self, entity: dict[str, Any]) -> bool:
        """Store or update an entity node in Neo4j.

        Uses MERGE to create or update an Entity node identified by
        (name, entity_type) composite key. Updates first_seen/last_seen
        timestamps and session_id.

        Args:
            entity: Dictionary containing entity attributes:
                - name: Entity name (required)
                - entity_type: Type of entity (person, project, etc.)
                - session_id: Session that extracted this entity
                - first_seen: Timestamp when first observed
                - last_seen: Timestamp when last observed

        Returns:
            bool: True if stored successfully, False on InfrastructureError.
        """
        query = """
        MERGE (e:Entity {name: $name, entity_type: $entity_type})
        SET e.first_seen = $first_seen,
            e.last_seen = $last_seen,
            e.session_id = $session_id
        RETURN e
        """

        params = {
            "name": entity.get("name"),
            "entity_type": entity.get("entity_type"),
            "session_id": entity.get("session_id"),
            "first_seen": entity.get("first_seen"),
            "last_seen": entity.get("last_seen"),
        }

        try:
            self._executor.execute_write(query, params)
            return True
        except InfrastructureError:
            logger.warning("store_entity failed", exc_info=True)
            return False

    def store_entities_batch(self, entities: list[dict], session_id: Optional[str] = None) -> bool:
        """Store multiple entities in a single Neo4j transaction using UNWIND.

        Batch operation for efficient bulk entity creation. Uses UNWIND to
        process all entities in one transaction. Preserves existing first_seen
        timestamps while updating last_seen.

        Args:
            entities: List of entity dictionaries with keys: name, entity_type,
                session_id, first_seen, last_seen.
            session_id: Optional session identifier for logging. Defaults to None.

        Returns:
            bool: True if at least one entity was stored, False if empty input
                or on InfrastructureError.
        """
        if not entities:
            return False

        query = """
        UNWIND $entities AS entity
        MERGE (e:Entity {name: entity.name, entity_type: entity.entity_type})
        SET e.first_seen = CASE
                WHEN e.first_seen IS NULL THEN entity.first_seen
                ELSE e.first_seen
            END,
            e.last_seen = entity.last_seen,
            e.session_id = entity.session_id
        RETURN count(e) AS stored_count
        """

        params = {
            "entities": [
                {
                    "name": e.get("name"),
                    "entity_type": e.get("entity_type"),
                    "session_id": e.get("session_id"),
                    "first_seen": e.get("first_seen"),
                    "last_seen": e.get("last_seen"),
                }
                for e in entities
            ]
        }

        try:
            records = self._executor.execute_write(query, params)
            count = records[0]["stored_count"] if records else 0
            logger.info("store_entities_batch: stored %d entities session_id=%s", count, session_id)
            return count > 0
        except InfrastructureError:
            logger.warning("store_entities_batch failed session_id=%s", session_id, exc_info=True)
            return False

    def store_relations_batch(self, relations: list[dict], session_id: Optional[str] = None) -> bool:
        """Store multiple relations in a single Neo4j transaction using UNWIND and MERGE.

        Creates RELATES_TO relationships between existing Entity nodes. Matches
        entities by name and optional entity_type. Preserves existing source_memory
        and created_at while updating valid_at/invalid_at.

        Args:
            relations: List of relation dictionaries with keys:
                - source_name: Name of source entity (required)
                - target_name: Name of target entity (required)
                - source_entity_type: Optional type filter for source
                - target_entity_type: Optional type filter for target
                - relation_type: Type of relationship (required)
                - fact: Relationship description/fact text (required)
                - source_memory_id: ID of memory that extracted this relation
                - created_at, valid_at, invalid_at: Timestamps
            session_id: Optional session identifier for logging. Defaults to None.

        Returns:
            bool: True if at least one relation was stored, False if empty input
                or on InfrastructureError.
        """
        if not relations:
            return False

        query = """
        UNWIND $relations AS rel
        OPTIONAL MATCH (source:Entity {name: rel.source_name, entity_type: rel.source_entity_type})
        OPTIONAL MATCH (target:Entity {name: rel.target_name, entity_type: rel.target_entity_type})
        WITH rel, source, target
        WHERE source IS NOT NULL AND target IS NOT NULL
        MERGE (source)-[r:RELATES_TO {relation_type: rel.relation_type, fact: rel.fact}]->(target)
        ON CREATE SET r.source_memory = rel.source_memory_id,
                      r.created_at = rel.created_at,
                      r.valid_at = rel.valid_at,
                      r.invalid_at = rel.invalid_at
        ON MATCH SET r.source_memory = COALESCE(r.source_memory, rel.source_memory_id),
                     r.valid_at = rel.valid_at,
                     r.invalid_at = rel.invalid_at
        RETURN count(r) AS stored_count
        """

        logger.info(
            "store_relations_batch: attempting %d relations session_id=%s",
            len(relations), session_id
        )

        now = _now_iso()

        params = {
            "relations": [
                {
                    "source_name": r.get("source_name"),
                    "target_name": r.get("target_name"),
                    "source_entity_type": r.get("source_entity_type"),
                    "target_entity_type": r.get("target_entity_type"),
                    "relation_type": r.get("relation_type"),
                    "source_memory_id": r.get("source_memory_id"),
                    "fact": r.get("fact"),
                    "created_at": r.get("created_at", now),
                    "valid_at": r.get("valid_at", now),
                    "invalid_at": r.get("invalid_at"),
                }
                for r in relations
            ]
        }

        try:
            records = self._executor.execute_write(query, params)
            count = records[0]["stored_count"] if records else 0
            logger.info("store_relations_batch: stored %d/%d relations session_id=%s", count, len(relations), session_id)
            return count > 0
        except InfrastructureError:
            logger.warning("store_relations_batch failed session_id=%s", session_id, exc_info=True)
            return False

    def store_entities_from_memories_batch(self, memory_contents: list[tuple[str, str]]) -> int:
        """Extract entity mentions from a batch of memories in a single round-trip.

        Extracts entity names from all (memory_id, content) pairs, does a single
        Neo4j read to find existing entities, then a single UNWIND write to create
        all MENTIONS relationships.

        Args:
            memory_contents: List of (memory_id, content) tuples.

        Returns:
            Total number of MENTIONS relationships created/merged.
        """
        if not memory_contents:
            return 0

        all_entity_names: set[str] = set()
        for _, content in memory_contents:
            all_entity_names.update(
                w.strip() for w in content.split() if len(w.strip()) > 2
            )

        all_names_list = list(all_entity_names)
        if len(all_names_list) > MAX_ENTITIES_PER_CALL:
            logger.warning(
                "store_entities_from_memories_batch: %d entities exceeds limit %d, truncating",
                len(all_names_list), MAX_ENTITIES_PER_CALL,
            )
            all_names_list = all_names_list[:MAX_ENTITIES_PER_CALL]

        if not all_names_list:
            return 0

        try:
            records = self._executor.execute_read(
                "MATCH (e:Entity) WHERE e.name IN $names RETURN e.name AS name",
                {"names": all_names_list},
            )
            matched_names = {r.get("name") for r in records}

            if not matched_names:
                return 0

            created_at = _now_iso()
            mentions: list[dict] = []
            for memory_id, content in memory_contents:
                content_names = {w.strip() for w in content.split() if len(w.strip()) > 2}
                for name in content_names:
                    if name in matched_names:
                        mentions.append({
                            "memory_id": memory_id,
                            "entity_name": name,
                            "created_at": created_at,
                        })

            if not mentions:
                return 0

            batch_query = """
            UNWIND $mentions AS mention
            MATCH (m:Memory {id: mention.memory_id})
            MATCH (e:Entity {name: mention.entity_name})
            MERGE (m)-[r:MENTIONS]->(e)
            SET r.created_at = mention.created_at
            RETURN count(r) AS count
            """

            result = self._executor.execute_write(batch_query, {"mentions": mentions})
            count = result[0]["count"] if result else 0
            logger.info(
                "store_entities_from_memories_batch: %d MENTIONS for %d memories",
                count, len(memory_contents),
            )
            return count
        except InfrastructureError:
            logger.warning("store_entities_from_memories_batch failed", exc_info=True)
            return 0

    
