#!/usr/bin/env bash
# Snapshot and restore the production Neo4j graph (Docker).
#
# snapshot: dump the live neo4j-prod DB to backups/neo4j_dumps/ with a timestamp.
# restore:  load a dump file back into neo4j-prod (DESTRUCTIVE — replaces all data).
#
# Snapshots are full offline dumps (neo4j-admin database dump). Run a snapshot
# before any risky operation (migrations, bulk deletes, agent sessions that
# touch the DB).
#
# Usage:
#   ./scripts/neo4j-snapshot.sh                     # snapshot now
#   ./scripts/neo4j-snapshot.sh restore <file>       # restore from <file>
#   ./scripts/neo4j-snapshot.sh restore latest      # restore from newest dump
set -euo pipefail

DUMP_DIR="$(dirname "$0")/../backups/neo4j_dumps"
PROD_CONTAINER="choriprox-neo4j-prod"
DB_NAME="neo4j"
mkdir -p "$DUMP_DIR"

cmd="${1:-snapshot}"

case "$cmd" in
  snapshot)
    ts="$(date +%Y%m%dT%H%M%S)"
    out="$DUMP_DIR/neo4j-prod-$ts.dump"
    echo "Snapshotting $PROD_CONTAINER -> $out"
    echo "Stopping $PROD_CONTAINER for offline dump (consistent snapshot)..."
    docker compose -f "$(dirname "$0")/../docker-compose.yml" stop neo4j-prod
    vol=$(docker inspect -f '{{range .Mounts}}{{if eq .Destination "/data"}}{{.Name}}{{end}}{{end}}' "$PROD_CONTAINER")
    docker run --rm -e NEO4J_ACCEPT_LICENSE_AGREEMENT=yes -v "$vol:/data" \
      neo4j:2026.05.0-enterprise sh -c "mkdir -p /data/dumps && neo4j-admin database dump $DB_NAME --to-path=/data/dumps"
    docker cp "$PROD_CONTAINER:/data/dumps/$DB_NAME.dump" "$out" 2>/dev/null || \
      docker run --rm -v "$vol:/data" -v "$DUMP_DIR:/out" neo4j:2026.05.0-enterprise cp "/data/dumps/$DB_NAME.dump" "/out/$(basename "$out")"
    # Clean up in-container dump
    docker run --rm -v "$vol:/data" neo4j:2026.05.0-enterprise rm -f "/data/dumps/$DB_NAME.dump" 2>/dev/null || true
    echo "Restarting $PROD_CONTAINER..."
    docker compose -f "$(dirname "$0")/../docker-compose.yml" start neo4j-prod
    [ -s "$out" ] || { echo "ERROR: snapshot failed — no file produced at $out"; exit 1; }
    echo "Snapshot saved: $out ($(du -h "$out" | cut -f1))"
    ;;
  restore)
    file="${2:-}"
    if [ "$file" = "latest" ]; then
      file="$(ls -t "$DUMP_DIR"/neo4j-prod-*.dump "$DUMP_DIR"/neo4j.dump 2>/dev/null | head -1)"
      [ -n "$file" ] || { echo "No dump files found in $DUMP_DIR"; exit 1; }
    fi
    [ -n "$file" ] && [ -f "$file" ] || { echo "Usage: $0 restore <file|latest>"; exit 1; }
    echo "WARNING: restoring $file will REPLACE all data in $PROD_CONTAINER."
    read -r -p "Type 'yes' to proceed: " ans
    [ "$ans" = "yes" ] || { echo "Aborted."; exit 1; }
    trap 'docker compose -f "$(dirname "$0")/../docker-compose.yml" start neo4j-prod' EXIT
    echo "Stopping $PROD_CONTAINER for offline load..."
    docker compose -f "$(dirname "$0")/../docker-compose.yml" stop neo4j-prod
    vol=$(docker inspect -f '{{range .Mounts}}{{if eq .Destination "/data"}}{{.Name}}{{end}}{{end}}' "$PROD_CONTAINER")
    docker run --rm -e NEO4J_ACCEPT_LICENSE_AGREEMENT=yes -v "$vol:/data" -v "$(dirname "$file"):/dumps" \
      neo4j:2026.05.0-enterprise sh -c "cp /dumps/$(basename "$file") /data/$DB_NAME.dump && neo4j-admin database load $DB_NAME --from-path=/data --overwrite-destination && rm -f /data/$DB_NAME.dump"
    docker compose -f "$(dirname "$0")/../docker-compose.yml" start neo4j-prod
    trap - EXIT
    echo "Restored $(basename "$file") into $PROD_CONTAINER and restarted."
    ;;
  *)
    echo "Usage: $0 [snapshot|restore <file|latest>]"; exit 1
    ;;
esac