import pytest


@pytest.fixture(autouse=True)
def _clear_chain_state():
    """Clear the shared _chain_state OrderedDict before/after each test.

    Required because _chain_state is module-global state in proxy.state.
    Without this fixture, test order would corrupt warm-path state across tests.
    """
    import proxy.state as state
    state._chain_state.clear()
    yield
    state._chain_state.clear()


@pytest.fixture(autouse=True)
def _reset_db_pool_manager_per_test():
    """Reset the global pool state before each unit test.

    Required because tests run in parallel and the new ``db_pool`` exposes
    manager-level state (``_terminal_shutdown``, recovery indexes, raw
    counts, unsafe paths). Without this fixture, a terminal-shutdown or
    pending recovery triggered by another test in the same run would
    leak into this one.
    """
    import memory.db_pool as db_pool

    def _reset():
        with db_pool._pool_condition:
            pool = db_pool._pool
            db_pool._pool = None

        if pool is not None:
            try:
                pool.shutdown(grace_period=0)
            except Exception:
                pass

        with db_pool._pool_condition:
            assert not db_pool._active_raw_connections
            assert db_pool._active_recovery_leaders == 0
            db_pool._recovery_attempts_by_source.clear()
            db_pool._recovering_paths.clear()
            db_pool._maintenance_paths.clear()
            db_pool._unsafe_raw_paths.clear()
            db_pool._terminal_shutdown = False
            db_pool._pool_condition.notify_all()

    _reset()
    yield
    _reset()
