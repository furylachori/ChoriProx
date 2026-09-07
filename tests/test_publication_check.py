"""Publication hygiene: exercise real Git objects without proxy/DB services.

Run with: python3 -m unittest tests.test_publication_check -v
Scanner cases require the real Gitleaks binary; no scanner is mocked.
"""

import importlib.util
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/check_publication.py"
spec = importlib.util.spec_from_file_location("publication_check", SCRIPT)
check = importlib.util.module_from_spec(spec)
spec.loader.exec_module(check)
SCANNER = shutil.which(os.environ.get("GITLEAKS_BIN", "gitleaks"))


def seeded_secret():
    # Intentionally assemble a nonfunctional PAT fixture, never a real credential.
    return "ghp_" + "a1B2c3D4e5F6g7H8i9J0k1L2m3N4o5P6q7R8"


def seeded_email():
    return "privacy.fixture" + "@" + "mailbox.invalid"


def isolated_git_environment():
    # Inherited author overrides or GIT_DIR must not escape this fixture repo.
    return {
        **{key: value for key, value in os.environ.items() if not key.startswith("GIT_")},
        "GIT_CONFIG_GLOBAL": os.devnull,
        "GIT_CONFIG_SYSTEM": os.devnull,
    }


class GitRepositoryTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="publication-test-")
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)
        self.repo = self.directory / "repo"
        self.repo.mkdir()
        self.git("init", "-q")
        self.git("config", "user.name", "Fixture Author")
        self.git("config", "user.email", "fixture@example.com")
        self.git("config", "commit.gpgsign", "false")
        self.git("config", "tag.gpgsign", "false")
        self.write("README.md", "Public fixture.\n")
        self.git("add", "README.md")
        self.git("commit", "-qm", "Initial public fixture")

    def git(self, *args):
        return subprocess.run(
            ["git", "-C", str(self.repo), *args], check=True, capture_output=True,
            env=isolated_git_environment(),
            timeout=15,
        ).stdout.decode().strip()

    def write(self, name, contents):
        path = self.repo / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(contents)

    def snapshot(self, staged=False):
        snapshot = self.directory / ("staged" if staged else "full")
        snapshot.mkdir(exist_ok=True)
        check.collect(self.repo, snapshot, staged)
        return snapshot

    def run_check(self, *args, scanner=None):
        environment = {**isolated_git_environment(), "GITLEAKS_BIN": scanner or SCANNER or "gitleaks"}
        return subprocess.run(
            [sys.executable, str(SCRIPT), "--repo", str(self.repo), *args],
            capture_output=True, text=True, env=environment, timeout=75,
        )


class PublicationCollectionTests(GitRepositoryTest):
    def test_index_contents_are_not_replaced_by_working_file(self):
        self.write("settings.txt", seeded_secret())
        self.git("add", "settings.txt")
        self.write("settings.txt", "Removed only from worktree.\n")
        snapshot = self.snapshot(staged=True)
        self.assertEqual(check.scan_path(snapshot, "index/settings.txt").read_text(), seeded_secret())
        self.assertNotIn("Removed only from worktree.\n", [p.read_text() for p in snapshot.iterdir()])

    def test_deleted_historical_content_and_raw_author_are_included(self):
        self.write("notes.txt", seeded_secret())
        self.git("add", "notes.txt")
        self.git("-c", "user.email=" + seeded_email(), "commit", "-qm", "Historical fixture")
        self.git("rm", "notes.txt")
        self.git("commit", "-qm", "Remove fixture")
        snapshot = self.snapshot()
        contents = [p.read_text() for p in snapshot.iterdir()]
        self.assertIn(seeded_secret(), contents)
        self.assertTrue(any(seeded_email() in content for content in contents))

    def test_reused_blob_preserves_every_historical_filename(self):
        private_path = seeded_email() + ".txt"
        self.git("mv", "README.md", private_path)
        self.git("commit", "-qm", "Rename unchanged blob")
        self.git("mv", private_path, "README.md")
        self.git("commit", "-qm", "Restore public filename")
        snapshot = self.snapshot()
        self.assertNotIn(private_path, check.scan_path(snapshot, "metadata/historical-paths.txt").read_text())
        self.assertIn(private_path, check.scan_path(snapshot, "metadata/paths.txt").read_text())

    def test_ignored_runtime_is_preserved_but_force_added_runtime_is_rejected(self):
        self.write(".gitignore", ".env\ndata/\n")
        self.write(".env", seeded_secret())
        self.write("data/capture.json", seeded_email())
        self.snapshot()
        self.assertEqual((self.repo / ".env").read_text(), seeded_secret())
        self.git("add", "-f", ".env")
        with self.assertRaisesRegex(check.CheckError, "Private runtime"):
            self.snapshot(staged=True)

    def test_untracked_unignored_content_is_included(self):
        self.write("new-notes.txt", seeded_email())
        snapshot = self.snapshot()
        self.assertEqual(check.scan_path(snapshot, "worktree/new-notes.txt").read_text(), seeded_email())

    def test_unknown_binary_is_rejected_and_exact_nul_fixture_is_validated(self):
        (self.repo / "sample.bin").write_bytes(b"private\x00binary")
        with self.assertRaisesRegex(check.CheckError, "binary file"):
            self.snapshot()
        (self.repo / "sample.bin").unlink()
        (self.repo / "large_test.txt").write_bytes(b"\0" * 204800)
        self.assertIn("Validated", check.scan_path(self.snapshot(), "worktree/large_test.txt").read_text())

    def test_capture_and_local_auth_artifacts_require_exclusion(self):
        for path in ("tests/chat_dumps/request.json", ".codex/auth.json", ".claude/settings.local.json"):
            with self.subTest(path=path):
                self.write(path, "{}")
                self.git("add", "-f", path)
                with self.assertRaisesRegex(check.CheckError, "Private runtime"):
                    self.snapshot(staged=True)
                self.git("rm", "-f", path)

    def test_oversized_and_lfs_inputs_fail_closed(self):
        oversized = self.repo / "oversized.txt"
        with oversized.open("wb") as stream:
            stream.truncate(check.MAX_BYTES + 1)
        with self.assertRaisesRegex(check.CheckError, "oversized file"):
            self.snapshot()
        oversized.unlink()
        self.write("asset.txt", "version https://git-lfs.github.com/spec/v1\n")
        with self.assertRaisesRegex(check.CheckError, "LFS payloads"):
            self.snapshot()

    def test_deleted_historical_symlink_requires_review(self):
        (self.repo / "link").symlink_to("README.md")
        self.git("add", "link")
        self.git("commit", "-qm", "Historical symlink")
        self.git("rm", "link")
        self.git("commit", "-qm", "Remove symlink")
        with self.assertRaisesRegex(check.CheckError, "Historical symlinks"):
            self.snapshot()

    def test_replacement_refs_and_shallow_history_fail_closed(self):
        first = self.git("rev-parse", "HEAD")
        self.write("README.md", "Second clean fixture.\n")
        self.git("add", "README.md")
        self.git("commit", "-qm", "Second fixture")
        self.git("replace", first, "HEAD")
        with self.assertRaisesRegex(check.CheckError, "replacement refs"):
            self.snapshot()
        self.git("replace", "-d", first)
        self.write(".git/shallow", first + "\n")
        with self.assertRaisesRegex(check.CheckError, "Complete Git history"):
            self.snapshot()

    def test_tag_pointing_to_tree_requires_review(self):
        self.git("tag", "tree-fixture", "HEAD^{tree}")
        with self.assertRaisesRegex(check.CheckError, "non-commit objects"):
            self.snapshot()

    def test_missing_scanner_is_an_error_without_clean_claim(self):
        result = self.run_check(scanner=str(self.directory / "missing-gitleaks"))
        self.assertEqual(result.returncode, 2)
        self.assertIn("install Gitleaks", result.stderr)
        self.assertNotIn("passed", result.stdout)

    def test_file_diagnostics_cannot_echo_credentials_in_filenames(self):
        token_name = "AKIA" + "9B2C3D4E5F6G7H8J"
        path = "settings/" + token_name + ".pem"
        self.write(path, "Fixture")
        self.git("add", path)
        with self.assertRaises(check.CheckError) as caught:
            self.snapshot(staged=True)
        self.assertNotIn(token_name, str(caught.exception))
        self.assertIn("input-", str(caught.exception))


@unittest.skipUnless(SCANNER, "Real Gitleaks unavailable; scanner validation was not run")
class PublicationScannerTests(GitRepositoryTest):
    def test_clean_reserved_examples_pass(self):
        self.write("examples.txt", "fixture@example.com uses /home/user/workspace\n")
        result = self.run_check()
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_staged_secret_hidden_by_clean_worktree_fails_redacted(self):
        self.write("settings.txt", seeded_secret())
        self.git("add", "settings.txt")
        self.write("settings.txt", "Clean working copy.\n")
        result = self.run_check("--staged")
        self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
        self.assertNotIn(seeded_secret(), result.stdout + result.stderr)
        self.assertIn("github-pat", result.stdout)

    def test_deleted_historical_secret_is_still_detected(self):
        self.write("notes.txt", seeded_secret())
        self.git("add", "notes.txt")
        self.git("commit", "-qm", "Historical fixture")
        self.git("rm", "notes.txt")
        self.git("commit", "-qm", "Remove fixture")
        result = self.run_check()
        self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
        self.assertNotIn(seeded_secret(), result.stdout + result.stderr)

    def test_raw_git_author_email_is_detected(self):
        self.git("-c", "user.email=" + seeded_email(), "commit", "--allow-empty", "-qm", "Private identity fixture")
        result = self.run_check()
        self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
        self.assertIn("personal-email", result.stdout)
        self.assertNotIn(seeded_email(), result.stdout + result.stderr)

    def test_generic_backend_key_still_fails_after_fixture_exceptions(self):
        value = "p9B6wQ3rT7mU2kL4" + "vN8xY5zA1cE0fG6h"
        self.write("backend.txt", 'PROXY_SOTA_API_KEY = "' + value + '"\n')
        result = self.run_check()
        self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
        self.assertIn("generic-api-key", result.stdout)
        self.assertNotIn(value, result.stdout + result.stderr)

    def test_private_home_accounts_are_detected_on_each_platform(self):
        account = "publication-private" + "-person"
        for prefix in ("/home/", "/Users/", "C:\\Users\\"):
            with self.subTest(platform=prefix):
                value = prefix + account
                self.write("notes.txt", value + "\n")
                result = self.run_check()
                self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
                self.assertIn("personal-home-path", result.stdout)
                self.assertNotIn(account, result.stdout + result.stderr)

    def test_scanner_path_exclusions_cannot_hide_untracked_or_staged_secrets(self):
        for path in ("image.svg", "text.bin", ".gitleaks.toml", "vendor/key.txt"):
            with self.subTest(path=path):
                self.write(path, seeded_secret())
                untracked = self.run_check()
                self.assertEqual(untracked.returncode, 1, untracked.stdout + untracked.stderr)
                self.git("add", path)
                staged = self.run_check("--staged")
                self.assertEqual(staged.returncode, 1, staged.stdout + staged.stderr)
                self.assertNotIn(seeded_secret(), staged.stdout + staged.stderr)
                self.git("rm", "-f", path)


if __name__ == "__main__":
    unittest.main()
