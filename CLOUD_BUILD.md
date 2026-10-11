# Cloud Build checks

These configs add an offline CI lane without changing the app's existing test
targets:

- [`cloudbuild.yaml`](cloudbuild.yaml) is the fast pull request check. It installs
  runtime and test dependencies, runs `make check`, then runs the unit suite with
  two workers, excluding the named 10,000-message hardware benchmark below.
- [`cloudbuild.full.yaml`](cloudbuild.full.yaml) is the separately runnable,
  fuller offline lane. It runs the same checks plus the unit, proxy, root, and
  mocked end-to-end targets with two workers.

Both configs use Python 3.12.14 on Debian Bookworm, an `E2_STANDARD_2` worker,
Cloud Logging only, and the existing `ci-pr-smoke` service account. Each config
is a single build step so dependency installation and checks share the same
pinned runtime image. Both steps install Debian's `sqlite3` command-line
utility, which the database-recovery unit tests invoke, and GNU Make, which the
existing test targets require. The full step also installs Git because 13
unskipped `PublicationCollectionTests` in `tests/test_publication_check.py`
create Git repositories during setup.
The Neo4j URL is explicitly set to loopback port 7688; neither config starts a
database container or targets the production Neo4j port.

## Coverage and exclusions

The fast lane covers the repository's mock-drift/import checks and `tests/unit`
except for the separately tracked hardware benchmark below.
The full lane adds `tests/proxy`, top-level `tests/test_*.py`, and mocked tests
under `tests/e2e`. The standard unit targets exclude only the hardware benchmark
named below. The Makefile targets preserve nonzero exit status on other failures.
In the full lane, seven `PublicationScannerTests` methods are explicitly
skipped by their `skipUnless(SCANNER)` guard because Gitleaks is not installed.
The fast lane does not run the root tests; Git-backed publication integrity
tests do run in the full lane.

The standard-runtime lanes explicitly exclude
`tests/unit/test_v2_concurrency.py::TestLargeSession::test_10k_messages_chain_and_v2_under_thresholds`.
It is a hardware-sensitive 10,000-message timing benchmark with a strict 200 ms
limit; it measured 640.6 ms and 729.7 ms in the local 2-vCPU environment. Keep
it in a separate, owner-selected performance lane. No specialized hardware
runner is configured by this change, so that benchmark remains unvalidated in
Cloud Build.

The offline lanes deliberately omit `tests/integration`, `tests/load`,
`tests/memory`, `tests/skills`, and `make test`: the integration target starts
the isolated Neo4j Enterprise test container and
removes its integration SQLite database files. They also omit
`make test-e2e-real`, which requires live LLM/Neo4j services and provider
credentials. Docker-backed Neo4j, live-provider, OS/hardware-specific, and other
specialized checks need separate infrastructure and are not represented as
covered here. Included tests normally block external HTTP; integration-marked
cases use loopback fixtures or SQLite and do not call live providers. Dependency
installation needs access to PyPI and Debian package repositories.

There is no Python dependency lock file. `requirements.txt` and the `[test]`
extras in `pyproject.toml` contain version ranges, so package resolution can
change over time. Pinning the runtime image does not make Python dependencies
reproducible.

## Local timing and Cloud Build estimate

The earlier temporary-venv measurements (fast 2m50s; full 4m55s) predated the
final benchmark exclusion and package installation. In the exact
`python:3.12.14-slim-bookworm` image, I ran the configured full sequence after
installing `git`, `make`, and `sqlite3`, then separately reran the
Git-dependent `PublicationCollectionTests`. The combined setup, full sequence,
and focused rerun took 346 seconds. `make check` passed. The full test targets
returned nonzero: unit reported 3 failed and 4,010 passed; proxy reported 8
failed, 695 passed, 1 skipped, 1 xfailed, and 7 subtests; root reported 2
failed, 157 passed, 7 skipped, 11 errors, and 5 subtests; mocked e2e reported
46 passed. Unit failures were three `OSError: could not get source code`
errors in database-recovery and migration tests. Proxy and root failures were
`TrustToken.for_trusted_caller()` caller-file mismatch errors. The focused
`PublicationCollectionTests` run passed all 13 tests in 2.84s, confirming the
Git package dependency. No test or application code changed.

The earlier fast baseline had three database-recovery tests fail because the
temporary virtual environment lacked the `sqlite3` command-line utility, and
the hardware benchmark below exceeded its limit. The final configs install
`sqlite3` and exclude only that named benchmark; the selected-image full run
shows that other failures remain nonzero as intended.
Cloud Build estimates use the standard `E2_STANDARD_2` rate
of USD 0.006 per build minute and include generous setup/runtime margin over the
measured local command timings. They exclude PR event volume, package download
egress, log storage/retention, and any free-tier or temporary credits.

| Config | Measured local duration | Conservative Cloud Build duration | Estimate per run |
| --- | ---: | ---: | ---: |
| Fast PR (`cloudbuild.yaml`) | 2m50s earlier baseline; 4,010 passed, 4 failed before final exclusions | 8 min | USD 0.048 |
| Full offline (`cloudbuild.full.yaml`) | 346s setup + full sequence + focused publication rerun; full sequence failed as detailed above | 15 min | USD 0.090 |

The fast baseline's hardware failure was not hidden:
`tests/unit/test_v2_concurrency.py::TestLargeSession::test_10k_messages_chain_and_v2_under_thresholds`
it exceeded its 200 ms limit (640.6 ms in the fast baseline; 729.7 ms in the
full baseline). It is excluded from standard runtime builds and remains a
separate performance-validation decision. These results are local evidence,
not Cloud Build results.

At the 15-minute estimate, two full runs per day for a 30-day month cost about
USD 5.40 in build minutes. The full config has a 60-minute hard timeout; at
that cap, two scheduled runs per day for a 30-day month would cost at most USD
21.60 for build minutes. This is a timeout ceiling for the scheduled lane, not
a forecast, and excludes the other charges listed above. The four repositories'
scheduled timeout ceilings total USD 68.40/month. At 60 PR runs per repository
using each fast lane's timeout ceiling, the combined build-minute ceiling is
USD 95.40 before logs, storage, and network charges. This is a planning
envelope, not an enforced cap; PR volume and ancillary charges need separate
ownership before the monthly target can be guaranteed.

## Project setup proposal (not applied)

Use project `project-33a976dd-e360-4c43-8d9` and region
`northamerica-south1`; the existing Cloud Build v2 connection is `mex`. Do not
submit this config until `ci-pr-smoke` no longer has the custom
`cloudbuild.builds.create` permission (or it has been replaced by a logs-only
runtime identity). The human or automation submitting the build must separately
have only the required build-submission permission and `iam.serviceAccounts.actAs`
on the exact runtime service account. A manual invocation of the fast config
would then be:

```sh
gcloud builds submit \
  --project=project-33a976dd-e360-4c43-8d9 \
  --region=northamerica-south1 \
  --config=cloudbuild.yaml .
```

For a pull request trigger, use the retained repository and `main` as its base,
then select the existing `ci-pr-smoke` service account when creating the
trigger only after its custom `cloudbuild.builds.create` permission is removed
from the runtime identity, or use a logs-only runtime identity. PR code runs as
the selected runtime account, so build-creation permission belongs on the
trigger submitter. Cloud Build ignores the config's `serviceAccount` field for
trigger builds, so the trigger itself must select the intended account. For a separate
scheduled lane, configure a trigger that uses `cloudbuild.full.yaml`; the
proposed Cloud Scheduler cadence is `0 6,18 * * *` in timezone
`America/Costa_Rica`. No trigger or schedule is created by this change.

The configs name
`projects/project-33a976dd-e360-4c43-8d9/serviceAccounts/ci-pr-smoke@project-33a976dd-e360-4c43-8d9.iam.gserviceaccount.com`.
That account currently has Logs Writer and the custom Cloud Build create
permission. No IAM change is part of this proposal. Keep permissions scoped to the actual build needs. If the person or
automation creating a trigger needs to select this identity, grant
`iam.serviceAccounts.actAs` only on this service account to that creator; do
not grant the build account permission to impersonate itself, and do not add
broad project roles such as Owner or Editor. No IAM or service-account changes
are part of this proposal.
