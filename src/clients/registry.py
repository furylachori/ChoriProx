from typing import Dict, Optional
import threading
from clients.interface import ClientAdapter, RegistryError, DetectionError


class ClientRegistry:
    _instance: Optional["ClientRegistry"] = None
    _lock = threading.RLock()

    def __init__(self):
        self._clients: Dict[str, ClientAdapter] = {}
        self._active: Optional[str] = None

    @classmethod
    def instance(cls) -> "ClientRegistry":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    @classmethod
    def reset(cls) -> None:
        with cls._lock:
            cls._instance = None

    @classmethod
    def _for_testing(cls) -> "ClientRegistry":
        """Return a fresh isolated instance for tests. Does not affect singleton."""
        instance = cls.__new__(cls)
        instance._clients = {}
        instance._active = None
        return instance

    def register(self, adapter: ClientAdapter) -> None:
        with self._lock:
            self._clients[adapter.name] = adapter

    def get(self, name: str) -> ClientAdapter:
        with self._lock:
            if name not in self._clients:
                raise RegistryError(
                    f"Client '{name}' not registered. "
                    f"Available: {list(self._clients.keys())}"
                )
            return self._clients[name]

    def active(self) -> ClientAdapter:
        with self._lock:
            if self._active is None:
                raise RegistryError("No active client set. Call detect() first.")
            return self._clients[self._active]

    def set_active(self, name: str) -> None:
        with self._lock:
            if name not in self._clients:
                raise RegistryError(f"Client '{name}' not registered.")
            self._active = name

    def detect(self, project_root: str) -> ClientAdapter:
        import os

        if not project_root:
            raise DetectionError("project_root must be a non-empty string")
        # Sanitize: resolve real path to prevent path traversal
        real_root = os.path.realpath(project_root)
        if not os.path.isdir(real_root):
            raise DetectionError(f"project_root is not a valid directory: {real_root}")
        with self._lock:
            for name, adapter in self._clients.items():
                caps = adapter.get_capabilities()
                config_dir = os.path.join(real_root, caps.agent_config_dir)
                if os.path.isdir(config_dir):
                    self._active = name
                    return adapter
        raise DetectionError(
            f"No client config directory found in {real_root}. "
            f"Searched for: {[a.get_capabilities().agent_config_dir for a in self._clients.values()]}"
        )

    def list_clients(self) -> list[str]:
        with self._lock:
            return list(self._clients.keys())
