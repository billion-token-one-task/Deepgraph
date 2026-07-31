# Validation matrix for isolated CI

Final-session record (2026-07-31): static audits passed; the isolated Python
3.13 policy lane passed 71 tests; the synthetic fault lane passed 60/60,
including the validation-loop fairness/manifest subset 22/22; and the prior
real bubblewrap held-in/held-out/canary plus protected-write, network and
missing-bwrap fallback negative tests passed. A post-fix real evaluator rerun
preserved the fixture tree hash but was blocked by the host bwrap
`NETLINK_ROUTE` permission error. The PostgreSQL lane was blocked because no
local server or usable Docker daemon was available. Adapted legacy tests were
39 passed/30 failed and are individually classified in
[LEGACY_TEST_CLASSIFICATION.md](LEGACY_TEST_CLASSIFICATION.md). No production
URL, provider credential, deployment, or database was used.

Record the isolated run in
[ISOLATED_CI_EVIDENCE_TEMPLATE.md](ISOLATED_CI_EVIDENCE_TEMPLATE.md). Empty or
false fields remain blockers; do not infer acceptance from test code alone.

## Lane 1: static source audit

Permitted on a clean source checkout with no application environment:

```bash
python3 scripts/meta_harness_static_audit.py
python3 scripts/meta_harness_scope_audit.py
python3 scripts/meta_harness_sql_audit.py
python3 scripts/meta_harness_migration.py
git diff --check
```

The static audit parses source AST, checks topic/plugin boundaries, scans the
additive migration, and checks examples for credential literals. It does not
prove runtime behavior.

## Lane 2: pure policy tests

Run in an isolated dependency-pinned CI image with all production database
variables unset:

```bash
env -u DEEPGRAPH_DATABASE_URL -u DATABASE_URL \
  pytest -q \
    tests/test_scientific_integrity_contract.py \
    tests/test_meta_harness_v1_contracts.py \
    tests/test_meta_harness_v1_routing.py \
    tests/test_meta_harness_v1_calibration.py \
    tests/test_meta_harness_v1_colab_contract.py \
    tests/test_meta_harness_v1_durable_queues.py \
    tests/test_harness_evaluator_runner.py \
    tests/test_frontier_source.py \
    tests/test_llm_role_boundaries.py \
    tests/test_benchmark_design_agent.py \
    tests/test_scientific_authority.py \
    tests/test_meta_learner.py \
    tests/test_outcome_assembly.py \
    tests/test_paradigm_prompt.py \
    tests/test_test_db_isolation.py
```

Use mocks only. No network, provider, subprocess compute, or production paths
may be available in this lane.

Run the adapted legacy integration units in a separate isolated process; they
may create temporary worktrees/SQLite fixtures but must still have production
URLs and provider credentials unset:

```bash
env -u DEEPGRAPH_DATABASE_URL -u DATABASE_URL \
  pytest -q \
    tests/test_experiment_forge.py \
    tests/test_experiment_repair.py \
    tests/test_validation_loop.py \
    tests/test_vnext_gpu_scheduler.py
```

## Lane 3: disposable PostgreSQL restore

Follow [MIGRATION_RUNBOOK.md](MIGRATION_RUNBOOK.md), then run:

```bash
DEEPGRAPH_ALLOW_ISOLATED_INTEGRATION_TESTS=1 \
DEEPGRAPH_ISOLATED_POSTGRES_URL='postgresql://.../deepgraph_ci_restore' \
META_HARNESS_CANDIDATE_COMMIT='<candidate-commit>' \
pytest -q tests/integration/test_meta_harness_postgres.py
```

After that migration/count test passes, run the durable compute test in a new
test process against the same disposable restore:

```bash
DEEPGRAPH_ALLOW_ISOLATED_INTEGRATION_TESTS=1 \
DEEPGRAPH_ISOLATED_POSTGRES_URL='postgresql://.../deepgraph_ci_restore' \
META_HARNESS_CANDIDATE_COMMIT='<candidate-commit>' \
pytest -q tests/integration/test_compute_repository_postgres.py
```

Then run the content-addressed evidence-state persistence test, also as its own
process:

```bash
DEEPGRAPH_ALLOW_ISOLATED_INTEGRATION_TESTS=1 \
DEEPGRAPH_ISOLATED_POSTGRES_URL='postgresql://.../deepgraph_ci_restore' \
META_HARNESS_CANDIDATE_COMMIT='<candidate-commit>' \
pytest -q tests/integration/test_evidence_repository_postgres.py
```

Do not reverse these commands on a fresh restore: the migration test expects
its first application to be new. Never combine either command with general
repository test discovery.

Add integration cases for:

- two agendas reserving concurrently without cross-scope reads;
- hard token/GPU caps and `max_concurrency`;
- explicit legacy import and default backlog exclusion;
- grant expiry/restart reconciliation;
- OutcomeRecord exact metering and idempotency;
- route/provider failure observations;
- persisted provider cooldown reload after process restart;
- trusted OutcomeRecord assembly and rejection of caller-supplied metrics;
- failed compute usage capture before OutcomeRecord settlement;
- compute timeout/heartbeat recovery beyond the durable repository cases.
- signed reviewer approval purpose/subject/expiry/key-rotation behavior;
- evidence-graph Frontier query-ref reproducibility and operator-evidence
  rejection;
- proposal-pending identity/grant/promotion idempotency.
- scoped-ingestion duplicate enqueue, lease expiry, checkpoint resume, retry
  exhaustion and grant expiry;
- Colab admission/bind crash windows, single-worker claim, restart
  `usage_unknown` quarantine and terminal artifact/usage settlement.

SQLite is not an acceptance substitute. Its legacy compatibility tests may
remain a fast lane, but meta-harness-v1 migration and concurrency authority are
PostgreSQL-only until explicitly proven otherwise.

## Lane 4: candidate isolation

Use a throwaway clone and database namespace and follow
[EVALUATOR_ISOLATION.md](EVALUATOR_ISOLATION.md). Verify:

- worktree is a dedicated child of `candidate_root`;
- production path and database name are rejected;
- environment strips database URLs, HOME, tokens, OAuth, SSH and API keys;
- evaluator, held-out, canary, budget, migration and policy paths are
  immutable;
- diff module/line limits come from a reviewed policy;
- held-in, held-out and canary are hash-pinned and all required;
- approval cannot be produced by the candidate process.
- the isolation binary is real (not a mock), network is unavailable, candidate
  and suite mounts reject writes, only the output mount is writable, and
  candidate tree hashes are identical before/after.

## Lane 5: fault injection and canary

With synthetic providers and backend transports:

- auth failure enters cooldown;
- transient failure retries only the configured route set;
- no independent evaluator routes to manual review;
- SSH/Colab timeout, heartbeat loss, stop failure and missing artifacts never
  become `succeeded`, `completed`, or `confirmed`;
- the same idempotency key does not duplicate a job;
- restart reconciles leases/grants without duplicate submission.

Only after lanes 1–5 pass may an approved small CPU and then small-GPU canary
be considered.
