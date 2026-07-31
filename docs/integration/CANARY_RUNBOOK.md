# Candidate and compute canary runbook

This runbook describes future isolated execution. No canary was launched in
this session.

## Admission

The canary requires:

- an approved candidate commit;
- a disposable PostgreSQL restore migrated twice;
- one paused/manual test agenda with a positive token cap;
- old backlog policy `explicit_import_only`;
- a persisted FrontierPacket that passed the Frontier Gate;
- a persisted promote/revisit IdeaDecisionPacket;
- a short-lived ResourceGrant for the exact stage/backend/artifacts;
- hash-pinned evaluator, held-out and policy inputs mounted read-only.

## Sequence

1. Create a detached child worktree under the configured candidate root.
2. Create a unique database/schema and artifact namespace with the
   `meta_harness_candidate_` prefix.
3. Run held-in and held-out in the isolated environment.
4. Issue a CPU-pilot grant and run one bounded job.
5. Audit raw metrics, environment/run manifest, claim ledger and usage.
6. Record an OutcomeRecord with measured—not inferred—token/GPU/time usage.
7. If CPU evidence permits, obtain separate approval and issue a small-GPU
   grant; do not reuse the pilot grant for a full benchmark.
8. Inject one provider failure and one backend failure; both must terminate
   truthfully without scientific promotion.
9. Run the canary suite and generate a RegressionReport.
10. Require reviewer/human approval before creating a HarnessArchive.

## Stop conditions

Stop immediately on scope mismatch, expired/missing grant, budget overrun,
unexpected production hostname/database, writable evaluator/holdout inputs,
missing artifacts, absent heartbeat, duplicate submission, or any generated
numeric/stronger manuscript claim.

## Acceptance evidence

Archive only hashes and non-sensitive summaries:

- candidate/base/tree/patch/policy/evaluator/holdout hashes;
- agenda, decision packet and grant IDs;
- backend state transitions and idempotency key;
- artifact manifest hashes;
- exact metered usage and OutcomeRecord ID;
- held-in/out/canary results;
- reviewer identity and approval timestamp.

Do not archive credentials, OAuth HOME, database URLs, row contents, dumps, or
backup material in the repository.
