#!/usr/bin/env bash
# Apply Neo4j RBAC rules that prevent DETACH DELETE on :Task nodes.
#
# This is the strongest defence against the agent-via-bash data-loss vector:
# even if an LLM agent discovers the prod password and runs
#   cypher-shell -u neo4j -p <password> "MATCH (t:Task) DETACH DELETE t"
# the DENY rule blocks it — :Task nodes cannot be hard-deleted by any role.
# The proxy's normal operations (MERGE/SET/CREATE/REMOVE) are unaffected.
#
# Run once after starting or recreating the prod container:
#   bash scripts/neo4j-apply-rbac.sh
set -euo pipefail
CONTAINER="${1:-choriprox-neo4j-prod}"
AUTH="${NEO4J_AUTH:-neo4j:your-neo4j-password}"

echo "Applying RBAC DENY DELETE on :Task to $CONTAINER..."
docker exec "$CONTAINER" cypher-shell -u neo4j -p "${AUTH#*:}" \
  "DENY DELETE ON GRAPH * NODE Task TO admin, architect, editor, publisher, reader;" 2>&1 | tail -3

# Verify the rule is in place
echo "Verifying DENY rules:"
docker exec "$CONTAINER" cypher-shell -u neo4j -p "${AUTH#*:}" \
  "SHOW PRIVILEGES;" 2>&1 | grep -i "DENIED" | tail -10

echo "RBAC applied. :Task nodes are now protected from DETACH DELETE."