"""Run the real startup script with a stand-in for its final proxy process.

Only synthetic credentials and an isolated directory reach the subprocess;
the Flask application, upstream APIs, and databases are never started.
"""

import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest


START_SCRIPT = Path(__file__).resolve().parents[1] / "start.sh"
REPORTED_CREDENTIALS = (
    "OPENCODE_API_KEY",
    "MIMO_API_KEY",
    "STEP_FUN_API_KEY",
    "ARK_MODEL_ARK_API_KEY",
    "BAILIAN_TOKEN_PLAN_API_KEY",
    "DASHSCOPE_INTL_API_KEY",
    "PROXY_API_KEY",
    "NEO4J_AUTH",
)


class TestStartScript(unittest.TestCase):
    def run_start(self, credentials, persisted_key=None):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            shutil.copy2(START_SCRIPT, root / "start.sh")
            stub_dir = root / "bin"
            stub_dir.mkdir()
            capture = root / "proxy-environment.json"
            stub = stub_dir / "python3"
            stub.write_text(
                f"#!{sys.executable}\n"
                "import json, os, pathlib, sys\n"
                f"names = {REPORTED_CREDENTIALS!r}\n"
                "captured = {name: os.environ.get(name) for name in names}\n"
                "captured['PROXY_SOTA_API_KEY'] = os.environ.get('PROXY_SOTA_API_KEY')\n"
                "captured['PYTHONPATH'] = os.environ.get('PYTHONPATH')\n"
                "captured['argv'] = sys.argv[1:]\n"
                "pathlib.Path(os.environ['START_TEST_CAPTURE']).write_text(json.dumps(captured))\n"
            )
            stub.chmod(0o700)
            if persisted_key is not None:
                (root / "data").mkdir()
                (root / "data" / "proxy_api_key.txt").write_text(persisted_key + "\n")
            env = {
                "PATH": f"{stub_dir}:/usr/bin:/bin",
                "START_TEST_CAPTURE": str(capture),
                **credentials,
            }
            result = subprocess.run(
                ["/bin/bash", str(root / "start.sh")],
                env=env,
                capture_output=True,
                text=True,
                timeout=10,
            )
            self.assertEqual(result.returncode, 0, "The isolated startup script failed")
            captured = json.loads(capture.read_text())
        self.assertEqual(captured.pop("argv"), ["src/qwen_proxy.py"])
        self.assertEqual(captured.pop("PYTHONPATH"), "src")
        return result.stdout + result.stderr, captured

    def assert_no_credentials_in_output(self, output, credentials):
        for name, value in credentials.items():
            if value:
                self.assertFalse(value in output, f"Startup disclosed {name}")
                self.assertFalse(value[:10] in output, f"Startup disclosed a prefix of {name}")

    def test_configured_credentials_are_forwarded_without_printing_values(self):
        for length in ("short", "long"):
            with self.subTest(length=length):
                credentials = {
                    name: f"q{index}x" if length == "short" else f"fixture-{index:02d}-credential-for-startup-testing"
                    for index, name in enumerate(REPORTED_CREDENTIALS)
                }
                credentials["NEO4J_AUTH"] = "neo4j:" + credentials["NEO4J_AUTH"]
                output, captured = self.run_start(credentials)
                self.assert_no_credentials_in_output(output, credentials)
                for name, value in credentials.items():
                    self.assertEqual(captured[name], value)
                    self.assertIn(f"  {name}: configured\n", output)
                self.assertEqual(captured["PROXY_SOTA_API_KEY"], credentials["STEP_FUN_API_KEY"])

    def test_empty_optional_credentials_are_reported_as_unconfigured(self):
        credentials = {
            "PROXY_API_KEY": "fixture-proxy-startup-key",
            "NEO4J_AUTH": "neo4j:fixture-startup-password",
            "STEP_FUN_API_KEY": "fixture-stepfun-startup-key",
        }
        output, captured = self.run_start(credentials)
        self.assert_no_credentials_in_output(output, credentials)
        for name in set(REPORTED_CREDENTIALS) - set(credentials):
            self.assertEqual(captured[name], "")
            self.assertIn(f"  {name}: not configured\n", output)

    def test_persisted_proxy_key_is_loaded_without_printing_it(self):
        persisted_key = "fixture-persisted-proxy-startup-key"
        output, captured = self.run_start({}, persisted_key=persisted_key)
        self.assertEqual(captured["PROXY_API_KEY"], persisted_key)
        self.assert_no_credentials_in_output(output, captured)
        self.assertIn("  PROXY_API_KEY: configured\n", output)


if __name__ == "__main__":
    unittest.main()
