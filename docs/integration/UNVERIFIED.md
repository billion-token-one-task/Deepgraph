# Unverified and incomplete items

The final decision remains **REJECTED — not eligible to replace master**.
The following items were not proven by this isolated session:

- The physical backup PostgreSQL lane now passes isolated restore, first
  migration, second migration/checksum no-op, all 48 pre-existing table-count
  preservation (including `claims`), FK/orphan/scope checks, and the three
  PostgreSQL repository integration files (6/6). Provider cooldown persistence
  and external backend restart behavior remain unverified.
- The pure policy/adapted-entry lane passed 71 tests. The refreshed synthetic
  fault lane passed 60 tests, including 22 validation-loop fairness/manifest
  checks after repairing the candidate-only scoring, broad-context and
  zero-budget guards. The adapted legacy lane remains 39 passed and 30
  failures; each is classified as obsolete or requiring a new scoped/granted
  fixture in [LEGACY_TEST_CLASSIFICATION.md](LEGACY_TEST_CLASSIFICATION.md).
  These contracts were not weakened to satisfy old tests.
- Real bubblewrap held-in, held-out and canary evaluator runs passed again in a
  detached final-evidence worktree, as did the earlier protected-write, network
  and missing-isolation-binary negative tests. This proves evaluator isolation
  for a disposable fixture, not production database or backend execution.
- A disposable control-plane canary did run with CPU and the authorized A100
  target-1 SSH secret reference. It passed durable submit/settle,
  idempotency rejection, and submission-failure quarantine; evidence SHA-256
  is recorded in `ACCEPTANCE_EVIDENCE.md`. This does not prove a scientific
  benchmark, Colab/provider execution, or reviewer approval.
- The SSH transport reference repair and strict host-key pinning are unit-tested;
  the A100 control-plane probe/canary used a secret reference without logging
  its value. Full scientific GPU, Colab, and provider canaries remain open.
- A target-3 probe with the injected secret reference and strict known-hosts
  file timed out after 30 seconds. This is recorded as unavailable/unverified,
  not as a successful canary.
- A disposable CPU+A100 control-plane canary assembled a trusted OutcomeRecord,
  the count-only API status endpoint returned 200, and reviewer approval was
  independently verified and persisted. This is not production serving or a
  scientific benchmark. Real LLM provider execution/restart remains unverified.
- No push, deployment, production database connection, production worktree
  mutation, remote-ref deletion or master merge occurred.

Evidence that is available is recorded in
[ACCEPTANCE_EVIDENCE.md](ACCEPTANCE_EVIDENCE.md) and the filled
[ISOLATED_CI_EVIDENCE_TEMPLATE.md](ISOLATED_CI_EVIDENCE_TEMPLATE.md).
