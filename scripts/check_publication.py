#!/usr/bin/env python3
"""Scan publishable Git contents with Gitleaks without exposing match values."""

import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile


CONFIG = Path(__file__).resolve().parents[1] / ".gitleaks.toml"
MAX_BYTES = 10 * 1024 * 1024
# tests/test_data/large_test.txt: exactly 204800 NUL bytes, no private data.
NUL_FIXTURE_SHA256 = "8eafc7bd411c1f02b9e972a83d2b0a4164eefc5ef51e6b63ad7acc78be4ad44f"


class CheckError(Exception):
    pass


def git(root, *args, input=None):
    result = subprocess.run(
        ["git", "-C", str(root), *args], input=input, capture_output=True,
        env={**os.environ, "GIT_NO_REPLACE_OBJECTS": "1"}, timeout=60,
    )
    if result.returncode:
        raise CheckError("Git could not read the requested publication contents.")
    return result.stdout


def safe_path(path):
    """File names can themselves contain a secret or a private identity."""
    return "input-" + hashlib.sha256(path.encode("utf-8", errors="replace")).hexdigest()[:12]


def scan_path(snapshot, name):
    # Gitleaks' default path exclusions must not hide staged .svg/vendor/config files.
    return snapshot / (hashlib.sha256(name.encode("utf-8")).hexdigest() + ".txt")


def check_artifact(path):
    """Do not publish local settings or runtime captures, even without a token hit."""
    name = Path(path).name.lower()
    if (
        path.split("/", 1)[0] in ("data", "backups", "logs")
        or set(Path(path).parts) & {"__pycache__", "chat_dumps", "debug_inbound", "debug_upstream_sse"}
        or path in (".claude/settings.local.json", ".codex/auth.json", ".codex/history.jsonl")
        or path.startswith((".codex/sessions/", ".codex/log/"))
        or name in ("proxy_config.json", "proxy_api_key.txt", ".netrc", ".npmrc", ".pypirc", ".git-credentials")
        or (name.startswith(".env") and name not in (".env.example", ".env.sample"))
        or re.search(r"\.(?:db|sqlite3?|log)(?:[.-].*)?$|\.(?:pem|key|p12|pfx|ppk|dump)$", name)
    ):
        raise CheckError(f"Private runtime/configuration artifact must be excluded: {safe_path(path)}")


def write_input(snapshot, name, content):
    if len(content) > MAX_BYTES:
        raise CheckError(f"Manual review required for file larger than 10 MiB: {safe_path(name)}")
    if content.startswith(b"version https://git-lfs.github.com/spec/v1"):
        raise CheckError("Git LFS payloads require a separate publication audit.")
    if hashlib.sha256(content).hexdigest() == NUL_FIXTURE_SHA256:
        content = b"Validated all-NUL synthetic bulk fixture.\n"
    try:
        content.decode("utf-8")
    except UnicodeDecodeError:
        raise CheckError(f"Manual review required for non-UTF-8 file: {safe_path(name)}") from None
    if b"\0" in content:
        raise CheckError(f"Manual review required for binary file: {safe_path(name)}")
    scan_path(snapshot, name).write_bytes(content)


def blobs(root, object_ids):
    """Read exact Git objects in a batch; never execute or check out their contents."""
    if not object_ids:
        return {}
    request = "\n".join(sorted(set(object_ids))).encode() + b"\n"
    sizes = git(root, "cat-file", "--batch-check", input=request)
    for line in sizes.splitlines():
        parts = line.split()
        if len(parts) != 3 or int(parts[2]) > MAX_BYTES:
            raise CheckError("A Git object is missing or larger than 10 MiB; manual review required.")
    stream = io.BytesIO(git(root, "cat-file", "--batch", input=request))
    result = {}
    while header := stream.readline():
        oid, kind, size = header.split()
        data = stream.read(int(size))
        if stream.read(1) != b"\n":
            raise CheckError("Incomplete Git object read.")
        result[oid.decode()] = (kind.decode(), data)
    return result


def collect(root, snapshot, staged):
    paths = []
    entries = []
    for entry in git(root, "ls-files", "--stage", "-z").split(b"\0"):
        if not entry:
            continue
        details, path = entry.split(b"\t", 1)
        mode, oid, stage = details.decode().split()
        if stage != "0":
            raise CheckError("Resolve the unmerged Git index before scanning.")
        if mode not in ("100644", "100755"):
            raise CheckError("Symlinks and submodules require a separate publication audit.")
        entries.append((oid, os.fsdecode(path)))
    objects = blobs(root, [oid for oid, _ in entries])
    for oid, path in entries:
        check_artifact(path)
        paths.append(path)
        write_input(snapshot, "index/" + path, objects[oid][1])
    if not staged:
        if git(root, "rev-parse", "--is-shallow-repository").strip() != b"false":
            raise CheckError("Complete Git history is required for a full publication check.")
        grafts = Path(os.fsdecode(git(root, "rev-parse", "--git-path", "info/grafts")).strip())
        if not grafts.is_absolute():
            grafts = root / grafts
        if grafts.exists() or git(root, "for-each-ref", "--format=%(refname)", "refs/replace").strip():
            raise CheckError("Remove Git grafts/replacement refs in a separate audit copy before scanning history.")
        for ref_type in git(root, "for-each-ref", "--format=%(objecttype)%09%(*objecttype)").splitlines():
            if ref_type.split() not in ([b"commit"], [b"tag", b"commit"]):
                raise CheckError("Refs pointing to non-commit objects require a separate publication audit.")
        for path_bytes in git(root, "ls-files", "--cached", "--others", "--exclude-standard", "-z").split(b"\0"):
            if not path_bytes:
                continue
            path = os.fsdecode(path_bytes)
            file = root / path
            if file.is_symlink():
                raise CheckError("Symlinks require a separate publication audit.")
            if not file.exists():
                continue  # Deleted worktree files remain covered by the index/history.
            if not file.resolve().is_relative_to(root.resolve()):
                raise CheckError("Files outside the repository require a separate publication audit.")
            check_artifact(path)
            if file.stat().st_size > MAX_BYTES:
                raise CheckError(f"Manual review required for oversized file: {safe_path(path)}")
            paths.append(path)
            write_input(snapshot, "worktree/" + path, file.read_bytes())
        revisions = ["--all"]
        if subprocess.run(["git", "-C", str(root), "rev-parse", "--verify", "HEAD"], capture_output=True, timeout=60).returncode == 0:
            revisions.append("HEAD")
        listing = git(root, "rev-list", "--objects", *revisions)
        ids = [line.split(b" ", 1)[0].decode() for line in listing.splitlines()]
        historical = blobs(root, ids)
        tree_ids = {data.splitlines()[0].split()[1].decode() for kind, data in historical.values() if kind == "commit"}
        for tree_id in tree_ids:
            for entry in git(root, "ls-tree", "-r", "-z", tree_id).split(b"\0"):
                if not entry:
                    continue
                details, path_bytes = entry.split(b"\t", 1)
                if details.split()[0] not in (b"100644", b"100755"):
                    raise CheckError("Historical symlinks/submodules require a separate publication audit.")
                path = os.fsdecode(path_bytes)
                check_artifact(path)
                paths.append(path)
        for oid, (kind, content) in historical.items():
            if kind in ("blob", "commit", "tag"):
                write_input(snapshot, f"history/{kind}/{oid}.txt", content)
        write_input(snapshot, "metadata/historical-paths.txt", listing)
        write_input(snapshot, "metadata/refs.txt", git(root, "for-each-ref", "--format=%(refname)"))
    write_input(snapshot, "metadata/paths.txt", "\n".join(paths).encode("utf-8"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--staged", action="store_true", help="scan the exact complete Git index only")
    parser.add_argument("--repo", type=Path, default=Path.cwd(), help="repository to inspect")
    args = parser.parse_args()
    scanner = shutil.which(os.environ.get("GITLEAKS_BIN", "gitleaks"))
    if not scanner:
        print("Publication check unavailable: install Gitleaks or set GITLEAKS_BIN.", file=sys.stderr)
        return 2
    try:
        root = Path(os.fsdecode(git(args.repo, "rev-parse", "--show-toplevel")).strip())
        with tempfile.TemporaryDirectory(prefix="chori-publication-") as temporary:
            private = Path(temporary)
            snapshot = private / "scan"
            snapshot.mkdir()
            collect(root, snapshot, args.staged)
            report = private / "report.json"
            ignore = private / "empty-ignore"
            ignore.write_text("")
            result = subprocess.run([
                scanner, "dir", str(snapshot), "--config", str(CONFIG),
                "--redact=100", "--no-banner", "--log-level=error", "--exit-code=3",
                "--report-format=json", "--report-path", str(report),
                "--ignore-gitleaks-allow", "--gitleaks-ignore-path", str(ignore),
                "--max-target-megabytes=0", "--max-decode-depth=2",
            ], capture_output=True, timeout=60)
            if result.returncode not in (0, 3) or not report.exists():
                raise CheckError("Gitleaks failed; verify the installed version and scan configuration.")
            findings = json.loads(report.read_text()) or []
            if findings or result.returncode == 3:
                print(f"Publication blocked: {len(findings)} secret/PII findings (values redacted).")
                for finding in findings[:25]:
                    path = str(finding["File"]).removeprefix(str(snapshot) + "/")
                    print(f"  {safe_path(path)}:{finding['StartLine']} [{finding['RuleID']}]")
                return 1
        scope = "exact staged contents" if args.staged else "index, working files, reachable history and Git metadata"
        print(f"Publication check passed: {scope}.")
        return 0
    except (CheckError, OSError, ValueError, KeyError, subprocess.TimeoutExpired) as error:
        # Only our own errors are safe to print: OS/JSON errors may contain data.
        message = str(error) if isinstance(error, CheckError) else "Unable to complete the publication check."
        print(f"Publication check failed: {message}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
