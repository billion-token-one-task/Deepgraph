# Unverified and incomplete items

The final decision remains **REJECTED — not eligible to replace master**.
The following items were not proven by this isolated session:

- A disposable PostgreSQL lane now passes restore-from-schema setup, first
  migration, second migration/checksum no-op, synthetic pre-existing count
  preservation, foreign-key/orphan/scope checks, multi-agenda reservation,
  durable compute restart recovery, Colab queue restart quarantine, and scoped
  ingestion lease/checkpoint/retry exhaustion. This was a synthetic schema
  restore because no production backup dump was supplied; real backup-row
  preservation and provider cooldown persistence remain unverified.
- The pure policy/adapted-entry lane passed 71 tests. The refreshed synthetic
  fault lane passed 60 tests, including 22 validation-loop fairness/manifest
  checks after repairing the candidate-only scoring, broad-context and
  zero-budget guards. The adapted legacy lane remains 39 passed and 30
  failures; each is classified as obsolete or requiring a new scoped/granted
  fixture in [LEGACY_TEST_CLASSIFICATION.md](LEGACY_TEST_CLASSIFICATION.md).
  These contracts were not weakened to satisfy old tests.
- Real bubblewrap held-in, held-out and canary evaluator runs passed, as did
  protected-write, network and missing-isolation-binary negative tests. The
  post-fix rerun preserved the candidate tree hash before/after. This proves
  evaluator isolation for the disposable fixture, not production database or
  backend execution.
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
