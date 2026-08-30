"""Import isolation tests for src/clients/.

Verify that clients modules can be imported without dragging in qwen_proxy.py.
This ensures the abstraction layer remains decoupled from the monolith.
"""

import subprocess
import sys
import os


PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
)
SRC = os.path.join(PROJECT_ROOT, "src")


class TestImportIsolation:
    """Run import checks in subprocess so sys.modules is clean."""

    @staticmethod
    def _run_snippet(code: str) -> subprocess.CompletedProcess:
        return subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            env={**os.environ, "PYTHONPATH": SRC},
        )

    def test_interface_import_does_not_load_proxy(self):
        snippet = (
            "import sys; "
            "from clients.interface import ClientAdapter, CapabilitySet, ClientConfig; "
            "assert 'qwen_proxy' not in sys.modules, 'qwen_proxy loaded unexpectedly'; "
            "print('ok')"
        )
        result = self._run_snippet(snippet)
        assert result.returncode == 0, (
            f"stdout: {result.stdout}\nstderr: {result.stderr}"
        )

    def test_registry_import_does_not_load_proxy(self):
        snippet = (
            "import sys; "
            "from clients.registry import ClientRegistry; "
            "assert 'qwen_proxy' not in sys.modules; "
            "print('ok')"
        )
        result = self._run_snippet(snippet)
        assert result.returncode == 0, (
            f"stdout: {result.stdout}\nstderr: {result.stderr}"
        )

    def test_capabilities_import_does_not_load_proxy(self):
        snippet = (
            "import sys; "
            "from clients.capabilities import KILOCODE_CAPABILITIES, OPENCODE_CAPABILITIES; "
            "assert 'qwen_proxy' not in sys.modules; "
            "print('ok')"
        )
        result = self._run_snippet(snippet)
        assert result.returncode == 0, (
            f"stdout: {result.stdout}\nstderr: {result.stderr}"
        )

    def test_kilocode_adapter_import_does_not_load_proxy(self):
        snippet = (
            "import sys; "
            "from clients.kilocode import KiloCodeAdapter; "
            "assert 'qwen_proxy' not in sys.modules; "
            "print('ok')"
        )
        result = self._run_snippet(snippet)
        assert result.returncode == 0, (
            f"stdout: {result.stdout}\nstderr: {result.stderr}"
        )

    def test_opencode_adapter_import_does_not_load_proxy(self):
        snippet = (
            "import sys; "
            "from clients.opencode import OpenCodeAdapter; "
            "assert 'qwen_proxy' not in sys.modules; "
            "print('ok')"
        )
        result = self._run_snippet(snippet)
        assert result.returncode == 0, (
            f"stdout: {result.stdout}\nstderr: {result.stderr}"
        )

    def test_model_registry_import_does_not_load_proxy(self):
        snippet = (
            "import sys; "
            "from clients.model_registry import ModelRegistry, ToolParserRegistry; "
            "assert 'qwen_proxy' not in sys.modules; "
            "print('ok')"
        )
        result = self._run_snippet(snippet)
        assert result.returncode == 0, (
            f"stdout: {result.stdout}\nstderr: {result.stderr}"
        )

    def test_parsers_qwen_xml_import_does_not_load_proxy(self):
        # M7 #2: stale qwen_xml fork removed. Test now exercises the
        # canonical parser location.
        snippet = (
            "import sys; "
            "from proxy.tool_parsing.parsers import parse_all_tools_from_content; "
            "assert 'qwen_proxy' not in sys.modules; "
            "print('ok')"
        )
        result = self._run_snippet(snippet)
        assert result.returncode == 0, (
            f"stdout: {result.stdout}\nstderr: {result.stderr}"
        )

    def test_public_api_import_does_not_load_proxy(self):
        snippet = (
            "import sys; "
            "from clients import register_client, get_client, detect_client, active_client, list_clients; "
            "assert 'qwen_proxy' not in sys.modules; "
            "print('ok')"
        )
        result = self._run_snippet(snippet)
        assert result.returncode == 0, (
            f"stdout: {result.stdout}\nstderr: {result.stderr}"
        )

    def test_orchestrator_config_import_does_not_load_proxy(self):
        snippet = (
            "import sys; "
            "from clients.orchestrator_config import ProxyFeatures, HookCapabilities; "
            "assert 'qwen_proxy' not in sys.modules; "
            "print('ok')"
        )
        result = self._run_snippet(snippet)
        assert result.returncode == 0, (
            f"stdout: {result.stdout}\nstderr: {result.stderr}"
        )
