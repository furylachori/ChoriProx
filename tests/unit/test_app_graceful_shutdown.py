import pytest

import proxy.app as app_module
from proxy.agent.agent_registry import registry
import memory.executors as executors
import memory.db as memory_db


@pytest.fixture(autouse=True)
def reset_shutdown_state():
    previous_shutdown_state = app_module._shutdown_in_progress
    app_module._shutdown_in_progress = False
    with registry._lock:
        registry._in_flight_requests.clear()
    try:
        yield
    finally:
        app_module._shutdown_in_progress = previous_shutdown_state
        with registry._lock:
            registry._in_flight_requests.clear()


def test_slice_3b_shutdown_orders_completion_memory_then_sqlite(monkeypatch):
    clock = {"now": 100.0}
    calls = []

    monkeypatch.setattr(
        app_module.time,
        "monotonic",
        lambda: clock["now"],
    )

    def completion_shutdown(*, grace_period):
        calls.append(("completion", grace_period))
        clock["now"] += 7.0
        return True

    def memory_shutdown(*, timeout):
        calls.append(("memory", timeout))
        clock["now"] += 4.0

    def close_sqlite():
        calls.append(("sqlite", None))

    monkeypatch.setattr(
        app_module,
        "_shutdown_completion_coordinator",
        completion_shutdown,
    )
    monkeypatch.setattr(executors, "_shutdown_thread_pool", memory_shutdown)
    monkeypatch.setattr(memory_db, "close_all_connections", close_sqlite)

    with registry._lock:
        registry._in_flight_requests.clear()

    app_module._graceful_shutdown()

    assert [entry[0] for entry in calls] == [
        "completion",
        "memory",
        "sqlite",
    ]
    assert calls[0][1] == pytest.approx(30.0)
    assert calls[1][1] == pytest.approx(23.0 / 4.0)
    assert app_module._shutdown_in_progress is True


def test_slice_3b_completion_timeout_preserves_memory_and_sqlite(monkeypatch):
    calls = []
    constant_time = 100.0

    monkeypatch.setattr(app_module.time, "monotonic", lambda: constant_time)

    def completion_shutdown(*, grace_period):
        calls.append("completion")
        return False

    def memory_shutdown(*, timeout):
        raise AssertionError(
            "must not run while completion workers remain active"
        )

    def close_sqlite():
        raise AssertionError(
            "must not run while completion workers remain active"
        )

    monkeypatch.setattr(
        app_module,
        "_shutdown_completion_coordinator",
        completion_shutdown,
    )
    monkeypatch.setattr(executors, "_shutdown_thread_pool", memory_shutdown)
    monkeypatch.setattr(memory_db, "close_all_connections", close_sqlite)

    app_module._graceful_shutdown()

    assert calls == ["completion"]


def test_slice_3b_inflight_timeout_preserves_memory_and_sqlite(monkeypatch):
    times = iter([100.0, 130.0])
    memory_calls = []
    sqlite_calls = []

    monkeypatch.setattr(app_module.time, "monotonic", lambda: next(times))
    monkeypatch.setattr(
        app_module,
        "_shutdown_completion_coordinator",
        lambda *, grace_period: True,
    )

    def memory_shutdown(*, timeout):
        memory_calls.append(timeout)
        raise AssertionError("must not run while requests remain active")

    def close_sqlite():
        sqlite_calls.append(True)
        raise AssertionError("must not run while requests remain active")

    monkeypatch.setattr(executors, "_shutdown_thread_pool", memory_shutdown)
    monkeypatch.setattr(memory_db, "close_all_connections", close_sqlite)

    with registry._lock:
        registry._in_flight_requests["session"] = 1

    app_module._graceful_shutdown()

    assert memory_calls == []
    assert sqlite_calls == []
