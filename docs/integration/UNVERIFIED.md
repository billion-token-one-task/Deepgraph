# Unverified and incomplete items

The final decision remains **REJECTED — not eligible to replace master**.
The following items were not proven by this isolated session:

- PostgreSQL restore, first migration, second migration/checksum no-op, count
  preservation, foreign-key/orphan checks, lock duration, concurrency, restart
  recovery, provider cooldown persistence and scoped-ingestion failure cases.
  The repository had `psql` client tools but no server, `initdb`/`pg_ctl`, or
  usable Docker daemon. Guarded PostgreSQL tests therefore skipped.
- The pure policy/adapted-entry lane passed 71 tests, but the broader synthetic
  fault collection had 55 passed and 5 stale validation-loop failures. The
  adapted legacy lane had 39 passed and 30 failures from removed APIs,
  mandatory grant arguments, role-routed calls and obsolete fallback
  expectations. These contracts were not weakened to satisfy old tests.
- Real bubblewrap held-in, held-out and canary evaluator runs did pass, as did
  protected-write, network and missing-isolation-binary negative tests. This
  proves evaluator isolation for the fixture, not production database or
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
