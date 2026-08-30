from clients.registry import ClientRegistry
from clients.interface import (
    ClientAdapter,
    CapabilitySet,
    ClientConfig,
    ClientsError,
    RegistryError,
    ConfigError,
    DetectionError,
    FormattedRequest,
    ToolCall,
)
from clients.orchestrator_config import ProxyFeatures, HookCapabilities

__all__ = [
    "register_client",
    "get_client",
    "detect_client",
    "active_client",
    "list_clients",
    "ClientAdapter",
    "CapabilitySet",
    "ClientConfig",
    "ProxyFeatures",
    "HookCapabilities",
    "ClientsError",
    "RegistryError",
    "ConfigError",
    "DetectionError",
    "FormattedRequest",
    "ToolCall",
]


def register_client(adapter: ClientAdapter) -> None:
    ClientRegistry.instance().register(adapter)


def get_client(name: str) -> ClientAdapter:
    return ClientRegistry.instance().get(name)


def detect_client(project_root: str) -> ClientAdapter:
    return ClientRegistry.instance().detect(project_root)


def active_client() -> ClientAdapter:
    return ClientRegistry.instance().active()


def list_clients() -> list[str]:
    return ClientRegistry.instance().list_clients()
