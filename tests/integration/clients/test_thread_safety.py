"""Thread safety integration tests.

Verify ClientRegistry and adapters are safe under concurrent access.
"""

import threading
import concurrent.futures
import pytest
import os

from clients.registry import ClientRegistry
from clients.kilocode import KiloCodeAdapter
from clients.opencode import OpenCodeAdapter


class TestThreadSafety:
    @pytest.fixture(autouse=True)
    def reset_singleton(self):
        """Reset global singleton after every test."""
        yield
        ClientRegistry.reset()

    @pytest.fixture
    def isolated_registry(self):
        return ClientRegistry._for_testing()

    def test_concurrent_register_does_not_corrupt(self, isolated_registry):
        adapters = [KiloCodeAdapter(), OpenCodeAdapter()]
        errors = []

        def register(adapter):
            try:
                isolated_registry.register(adapter)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=register, args=(a,)) for a in adapters]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
        assert set(isolated_registry.list_clients()) == {"kilocode", "opencode"}

    def test_concurrent_detect_with_same_tmp_dir(self, tmp_path, isolated_registry):
        (tmp_path / ".kilo").mkdir()
        isolated_registry.register(KiloCodeAdapter())

        results = []
        errors = []

        def detect():
            try:
                adapter = isolated_registry.detect(str(tmp_path))
                results.append(adapter.name)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=detect) for _ in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
        assert all(r == "kilocode" for r in results)
        assert isolated_registry.active().name == "kilocode"

    def test_concurrent_detect_with_different_tmp_dirs(
        self, tmp_path, isolated_registry
    ):
        dir_a = tmp_path / "a"
        dir_b = tmp_path / "b"
        dir_a.mkdir()
        dir_b.mkdir()
        (dir_a / ".kilo").mkdir()
        (dir_b / ".opencode").mkdir()

        isolated_registry.register(KiloCodeAdapter())
        isolated_registry.register(OpenCodeAdapter())

        results = []
        errors = []

        def detect(path, expected):
            try:
                adapter = isolated_registry.detect(path)
                results.append((expected, adapter.name))
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=detect, args=(str(dir_a), "kilocode"))
            for _ in range(10)
        ] + [
            threading.Thread(target=detect, args=(str(dir_b), "opencode"))
            for _ in range(10)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
        # Because registry._active is a single field, last writer wins;
        # we only care that no exception occurred and each call returned correctly.
        for expected, actual in results:
            assert actual == expected

    def test_concurrent_get_and_register(self, isolated_registry):
        isolated_registry.register(KiloCodeAdapter())
        errors = []

        def get():
            try:
                isolated_registry.get("kilocode")
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=get) for _ in range(50)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors

    def test_threadpool_detect_stress(self, tmp_path, isolated_registry):
        (tmp_path / ".kilo").mkdir()
        isolated_registry.register(KiloCodeAdapter())

        with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:
            futures = [
                executor.submit(isolated_registry.detect, str(tmp_path))
                for _ in range(100)
            ]
            results = [f.result() for f in concurrent.futures.as_completed(futures)]

        assert all(r.name == "kilocode" for r in results)

    def test_singleton_instance_thread_safe_creation(self):
        """Multiple threads racing to create the singleton should not duplicate."""
        ClientRegistry.reset()
        instances = []

        def get_instance():
            instances.append(ClientRegistry.instance())

        threads = [threading.Thread(target=get_instance) for _ in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(set(id(i) for i in instances)) == 1
        ClientRegistry.reset()

    def test_adapter_normalize_is_thread_safe(self):
        adapter = KiloCodeAdapter()
        args = {"filePath": "/tmp/test.py", "oldString": "a", "newString": "b"}

        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
            futures = [
                executor.submit(adapter.normalize_tool_args, "edit", dict(args))
                for _ in range(200)
            ]
            results = [f.result() for f in concurrent.futures.as_completed(futures)]

        for r in results:
            assert r == {"path": "/tmp/test.py", "old_string": "a", "new_string": "b"}
