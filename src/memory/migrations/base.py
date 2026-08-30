"""Base class for Neo4j schema migrations."""
from memory.query_executor import QueryExecutor


class Migration:
    version: int = 0
    description: str = ""

    def up(self, executor: QueryExecutor) -> None:
        raise NotImplementedError

    def down(self, executor: QueryExecutor) -> None:
        pass
