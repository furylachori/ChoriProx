// Defence-in-depth: block DETACH DELETE on :Task for all roles.
// The proxy only MERGEs/SETs/CREATEs tasks — it never needs to DELETE them.
// This prevents an agent with bash access from wiping tasks via cypher-shell,
// even if it discovers the prod password.
//
// Intentional B-7 gap: ONLY :Task is protected — do NOT extend to :Workspace,
// :Session, :Bug, :ReviewNote. The proxy itself runs as the admin role and
// legitimately hard-deletes these during workspace/session cleanup and
// migration 010 (STRIP phase). DENY-protecting them blocks the proxy's own
// operations. Tests/integration/test_neo4j_rbac.py::TestRbacGapDocumented
// asserts this gap is intentional.
DENY DELETE ON GRAPH * NODE Task TO admin, architect, editor, publisher, reader;