# Preparing a public repository

Install [Gitleaks](https://github.com/gitleaks/gitleaks#installing) from its official
release or a package manager, then run:

```sh
make check-publication
```

The check scans the exact Git index, current tracked and unignored files, every
reachable historical blob, commit/tag metadata, and file/ref names. It uses
Gitleaks' maintained default secret rules plus contact-email and workstation-path
checks in `.gitleaks.toml`. Reports print opaque input IDs, line numbers, and rule
names; matched values and original filenames are not printed. Scan snapshots are
private temporary files removed when the check
finishes. Set `GITLEAKS_BIN` to use a scanner outside `PATH`.

The check fails when Gitleaks is missing or cannot complete, history is shallow,
the index has conflicts, or content needs separate review: submodules, symlinks,
Git LFS payloads, non-commit refs, non-UTF-8/binary files, or files larger than
10 MiB. Replacement refs and grafts also require a separate audit copy. The existing
bulk-input fixture is allowed only when its bytes match the reviewed all-NUL
fixture exactly. Local runtime data, logs, databases, and private configuration
must stay excluded even if a scanner finds no recognizable token in them.

For an optional local commit hook, install
[pre-commit](https://pre-commit.com/#install) and run `pre-commit install`.
The checked-in hook runs `python3 scripts/check_publication.py --staged`, which
reads the complete index directly, including staged content hidden by later
working-file edits. The full publication check is still required before pushing.
No hooks are installed automatically.

Before publishing:

1. Review any finding locally. Rotate a real credential if it was committed or
   shared, even after removing it. Keep exceptions narrow and limited to verified
   synthetic examples or intentional public project contacts.
2. Inspect history, author/committer identities, tags, documentation, and fixtures
   for private information that pattern matching cannot reliably recognize,
   including names, conversations, customer data, and internal URLs.
3. If old history contains private content, clean that history in a separate copy
   or prepare a fresh repository from the reviewed files. Do not push the original
   refs until they pass the full check. Configure the intended public Git author
   name and no-reply email before making the publication commit.
4. Stage the reviewed files, run the full check again, and publish only those
   reviewed refs. GitHub's secret scanning and push protection provide another
   check when enabled; they do not replace review of personal information.

Ignored `data/`, logs, caches, local configuration, and `.git/` can still contain
private information. They are preserved locally and are not part of a normal Git
push. Create source archives with `git archive` from the verified commit; do not
upload a raw workspace directory or attach runtime captures to releases/issues.
The check does not inspect remote GitHub issues, releases, artifacts, unreachable
local Git objects, or external databases. A passing scan is evidence for the
scanned content, not a guarantee that every form of private information is absent.
