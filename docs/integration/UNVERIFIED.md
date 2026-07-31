# Unverified and incomplete items

The final decision remains **REJECTED — not eligible to replace master**.
The following items were not proven by this isolated session:

- PostgreSQL restore, first migration, second migration/checksum no-op, count
  preservation, foreign-key/orphan checks, lock duration, concurrency, restart
  recovery, provider cooldown persistence and scoped-ingestion failure cases.
  The repository had `psql` client tools but no server, `initdb`/`pg_ctl`, or
  usable Docker daemon. Guarded PostgreSQL tests therefore skipped.
- The pure policy/adapted-entry lane passed 71 tests. The refreshed synthetic
  fault lane passed 60 tests, including 22 validation-loop fairness/manifest
  checks after repairing the candidate-only scoring, broad-context and
  zero-budget guards. The adapted legacy lane remains 39 passed and 30
  failures; each is classified as obsolete or requiring a new scoped/granted
  fixture in [LEGACY_TEST_CLASSIFICATION.md](LEGACY_TEST_CLASSIFICATION.md).
  These contracts were not weakened to satisfy old tests.
- Real bubblewrap held-in, held-out and canary evaluator runs did pass, as did
  protected-write, network and missing-isolation-binary negative tests. This
  proves evaluator isolation for the earlier fixture run, not production
  database or backend execution. A post-fix rerun on this host preserved the
  candidate tree hash but was blocked because bwrap could not create its
  isolated network (`NETLINK_ROUTE: Operation not permitted`); mocked contract
  and missing-bwrap fallback checks still passed.
- No approved CPU, GPU/Colab, SSH or provider canary ran. The Colab CLI was
  inspected only for help/contract availability; no OAuth or remote secret was
  used. The evaluator's canary suite is not a hardware canary.
- No real OutcomeRecord sample, reviewer identity/signature, production API
  serving run or production rollback was executed. The exact `7d0b42a`
  detached worktree was clean and its isolated temp-SQLite startup rehearsal
  passed.
- No push, deployment, production database connection, production worktree
  mutation, remote-ref deletion or master merge occurred.

Evidence that is available is recorded in
[ACCEPTANCE_EVIDENCE.md](ACCEPTANCE_EVIDENCE.md) and the filled
[ISOLATED_CI_EVIDENCE_TEMPLATE.md](ISOLATED_CI_EVIDENCE_TEMPLATE.md).
