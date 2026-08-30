"""Lazy-import cache for memory.db (breaks circular-import at module load).

The `memory.db` module imports from several submodules (`memory.db_summaries`,
`memory.db_entities`, ...). Those submodules re-export through `memory.db`,
which would create an ImportError if `from memory.db import X` were used at
their module level. The `get_db()` helper imports `memory.db` lazily on
first call — once cached, every caller shares the same module reference.
"""

_db_module = None


def get_db():
    """Return the memory.db module, importing it lazily on first use."""
    global _db_module
    if _db_module is None:
        import memory.db as _m
        _db_module = _m
    return _db_module
