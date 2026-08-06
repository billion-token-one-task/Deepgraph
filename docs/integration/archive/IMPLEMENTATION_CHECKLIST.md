# meta-harness-v1 detailed implementation checklist

Last reviewed: 2026-08-02 UTC

This is the authoritative, fine-grained checklist for the controlled B
integration. It complements the semantic [PORTING_LEDGER.md](PORTING_LEDGER.md)
and the release-gate [ACCEPTANCE_EVIDENCE.md](ACCEPTANCE_EVIDENCE.md).

Status legend:

- `[x]` confirmed by an allowed static/Git/text check in this workspace;
- `[~]` implementation or test material exists, but required isolated runtime
  verification has not happened;
- `[ ]` open or only partially integrated;
- `[-]` explicitly excluded or prohibited for this release/host.

An item marked `[~]` is not release evidence. Only the isolated CI/canary
artifacts named in its evidence column may promote it to `[x]`.

The original S-series checkpoint below is historical 2026-07-31 evidence. The
explicit, local-only 2026-08-02 database cutover is recorded separately at the
end of this checklist so it does not rewrite that prior audit record.

## S. Production safety and authority

- [x] **S-01** Production path fixed as `/home/billion-token/Deepgraph`.
  Evidence: [BASELINE.md](BASELINE.md).
- [x] **S-02** Production branch/HEAD rechecked as
  `local/snapshot-20260621@7d0b42a`.
- [x] **S-03** Production tracked worktree remained unchanged.
- [x] **S-04** Host load checked before both work periods and remained below 3.
- [x] **S-05** Disk and restricted process view checked before work.
- [x] **S-06** No checkout/switch/restore/pull/merge/rebase/reset/stash ran in
  the production worktree.
- [x] **S-07** No production configuration, `.env`, systemd unit or service was
  modified/restarted.
- [x] **S-08** No production database connection was opened.
- [~] **S-09** Isolated Python 3.13 dependency install, pytest, app
  import/startup and compile checks ran in temporary paths; PostgreSQL,
  production-path and CPU/GPU/SSH/Colab execution remained out of scope.
- [x] **S-10** No `.env.bak-*`, `*.bak-*`, dump, `backups/`, OAuth HOME, token
  or credential was read or added.
- [x] **S-11** No push, remote ref, branch deletion, master replacement,
  deployment or database migration occurred.
- [ ] **S-12** Single-operator/deployment-quiescence must be established before
  any future remote mutation. Evidence required: approved change window.

## L. Immutable lineage and development environment

- [x] **L-01** Independent clone exists at
  `/home/ec2-user/Deepgraph-meta-harness-v1`.
- [x] **L-02** Original candidate remains `integration/meta-harness-v1` at
  `77e8ac0`; frozen acceptance branch is
  `integration/master-candidate-20260731`.
- [x] **L-03** Candidate base is
  `origin/master@6048a9568c79b011074e0dba2662fd473cfab250`.
- [x] **L-04** Local archive ref for production resolves to
  `7d0b42af8e8f061c3c16800c44224c110f3b94a0`.
- [x] **L-05** Local archive ref for GitHub master resolves to
  `6048a9568c79b011074e0dba2662fd473cfab250`.
- [x] **L-06** Local archive ref for topic-gate resolves to
  `9d24d29c6a7d1017301ffa9c36ff9b4b3dfae88d`.
- [x] **L-07** Production and GitHub master have no merge-base.
- [x] **L-08** Production `7d0b42a` and topic-gate `9d24d29` are sibling
  commits with parent `4f78f82`; neither is in GitHub master history.
- [x] **L-09** No merge/cherry-pick of the large unrelated commits occurred.
- [ ] **L-10** Remote archive refs require explicit approval and immediate hash
  re-verification. Proposed commands only: [BASELINE.md](BASELINE.md).

## P0. Audit baseline

- [x] **P0-01** Porting ledger records source commit/path, target, behavior,
  tests, status and rejected behavior.
- [x] **P0-02** Cross-line state dictionary separates operational and
  scientific states.
- [x] **P0-03** Schema delta records production/GitHub differences.
- [x] **P0-04** Measured cross-line diff deviation is recorded.
- [x] **P0-05** UI, images, historical papers and submission assets are
  explicitly excluded from v1.
- [x] **P0-06** Root changelog records lineage and candidate status:
  [CHANGELOG.md](../../../CHANGELOG.md).

## P1. GitHub base scientific integrity and plugin boundary

- [x] **P1-01** Fixed VOC/idea8/run13 result completion was removed from
  `agents/paper_orchestra_pipeline.py`.
- [x] **P1-02** `_deemphasize_significance_caveats` and callers were removed.
- [x] **P1-03** Generic manuscript path no longer contains a CRPP deterministic
  scientific narrative fallback.
- [x] **P1-04** Generic figure path cannot generate topic-specific figures.
- [x] **P1-05** CGGR/idea8 runners, shard tools, result auditor, tests and docs
  are isolated under `plugins/examples/cggr`.
- [x] **P1-06** Generic agent registry no longer registers topic scripts.
- [x] **P1-07** Topic method aliases/ablations are absent from generic default
  configuration.
- [x] **P1-08** Topic example activation is explicit, disabled by default and
  labelled non-production.
- [~] **P1-09** `p=1` is non-significant. Evidence: integrity fixture/test;
  isolated unit test pending.
- [~] **P1-10** Missing p-value cannot claim significance.
- [~] **P1-11** `refuted` cannot become a positive claim even with low p.
- [~] **P1-12** Zero/missing baseline, missing metric and incomplete benchmark
  cannot confirm.
- [~] **P1-13** Presentation/refinement cannot introduce numbers or strengthen
  claims.
- [~] **P1-14** Manuscript generation requires
  `scientific_evidence_state=manuscript_allowed` and reviewer approval.
- [~] **P1-15** Existing validation verdict path invokes the unified scientific
  evidence contract; exhaustive legacy status audit remains P5-14.

## P2. Agenda, scope, backlog and hard budgets

- [x] **P2-01** `ResearchAgenda` requires a positive token hard cap; zero is
  never unlimited.
- [x] **P2-02** GPU budget zero means GPU disabled.
- [x] **P2-03** Agenda has token/GPU spent and reserved accounting.
- [x] **P2-04** Agenda has positive `max_concurrency`.
- [x] **P2-05** Agenda has backend allowlist.
- [x] **P2-06** Default backlog policy is `explicit_import_only`.
- [x] **P2-07** Direction intake/loader produces a validated agenda echo.
- [x] **P2-08** Selector reads only `deep_insights.agenda_id=<active agenda>`.
- [x] **P2-09** Multi-agenda cycle selects fairly from active agendas without a
  global unscoped fallback.
- [x] **P2-10** Queue insertion requires selection and insight to share
  `agenda_id`.
- [x] **P2-11** Existing unscoped auto job blocks implicit reuse.
- [x] **P2-12** Legacy import records actor, reason, entity, agenda and
  idempotency key.
- [x] **P2-13** Old rows receive nullable scope columns but are not
  automatically assigned.
- [~] **P2-14** Reserve-before-call and atomic ledger updates are implemented;
  disposable PostgreSQL multi-agenda/concurrency harness passed; real backup
  lane remains required.
- [~] **P2-15** Token/GPU cap overrun pauses the agenda; disposable cap and
  max-concurrency fault checks passed; provider/real-backup lane remains open.
- [~] **P2-16** Resume requires a new cap above spent+reserved; CI pending.
- [~] **P2-17** ResourceGrant issuance reserves agenda capacity atomically;
  disposable PostgreSQL issuance and cross-scope rejection passed.
- [~] **P2-18** Expired grants release reservations and block queued jobs after
  reconciliation; restart CI pending.
- [x] **P2-19** Scoped problem creation requires `agenda_id`.
- [x] **P2-20** Scoped idea storage requires `agenda_id`.
- [x] **P2-21** Scoped experimental negative-evidence edges require
  `agenda_id`.
- [x] **P2-22** Auto-research candidate query requires matching agenda, idea and
  active non-expired grant.
- [x] **P2-23** Experiment forge/run/artifact core insertions carry agenda and
  grant scope.
- [x] **P2-24** Validation, claim, manuscript and bundle core insertions carry
  agenda scope.
- [~] **P2-25** The AST audit finds 154 agenda-owned literal mutations and zero
  definite statements without `agenda_id`; disposable PostgreSQL cross-agenda
  and orphan checks passed.
- [x] **P2-26** Signal-outcome learning is agenda-local; shared ingestion
  signal counters are not mutated by experiment feedback.

## P2M. Additive PostgreSQL migration

- [x] **P2M-01** Migration file is additive and checksum-journaled.
- [x] **P2M-02** Guarded planner defaults to no database access.
- [x] **P2M-03** Apply requires an isolated-looking database name, separate URL,
  explicit acknowledgement and full candidate hash.
- [x] **P2M-04** Static plan reports no DROP/TRUNCATE/DELETE/rename/destructive
  type token.
- [x] **P2M-05** Migration creates production-missing `research_problems`.
- [x] **P2M-06** Migration creates production-missing
  `experimental_evidence_edges`.
- [x] **P2M-07** Migration creates production-missing
  `benchmark_harness_jobs`.
- [x] **P2M-08** Frontier/decision/grant/outcome/LLM/compute/harness tables are
  present in the migration.
- [x] **P2M-09** Legacy scope columns are nullable for add-only compatibility.
- [~] **P2M-10** Migration preserves synthetic pre-existing counts on the
  disposable schema restore; a real production backup count check is pending.
- [x] **P2M-11** Migration first run applied and second run returned a checksum
  no-op on the disposable PostgreSQL restore.
- [~] **P2M-12** Synthetic restored schema starts after migration; a real
  production backup restore was not supplied.
- [~] **P2M-13** Disposable foreign-key/orphan/scope checks passed; lock-duration
  and production-backup evidence remain open.

## P3. LLM routing

- [x] **P3-01** Roles are explicit: proposer/evaluator/reviewer.
- [x] **P3-02** Provider routes include provider, model, model family, prompt
  version, timeout/retry/cooldown.
- [x] **P3-03** Evaluator/reviewer route must differ from proposer provider or
  model family.
- [x] **P3-04** No eligible independent evaluator fails closed to manual review.
- [x] **P3-05** Auth and transient failures have separate cooldowns.
- [x] **P3-06** Actual token usage and failures create route observations.
- [x] **P3-07** LLM sub-reservations cannot exceed the parent ResourceGrant.
- [x] **P3-08** Request grant ID must match the persisted grant.
- [x] **P3-09** Synthetic and real provider failure-injection tests passed;
  cooldown and fail-closed behavior were verified in isolated PostgreSQL.
- [~] **P3-10** Forge scout/scaffold/repair, validation iteration/
  reproduction-repair, proposal method/experiment design, benchmark design,
  Tier-2 evaluator/reviewer debate, manuscript revision and plain final review
  plus ingestion/extraction/enrichment use granted role routes and fail closed.
  A persisted `proposal_pending` identity prevents fake pre-candidate IDs.
  Ingestion verifies its active persisted agenda/idea/stage grant before
  pipeline work. Evidence:
  [LLM_CALLER_INVENTORY.md](LLM_CALLER_INVENTORY.md).
- [x] **P3-11** Provider cooldowns are persisted in
  `llm_provider_cooldowns`, reloaded by reconstructed routers and extended
  monotonically; real provider restart evidence passed.
- [~] **P3-12** Explicit provider-returned cost fields are captured and
  persisted with route observations; unknown cost remains `NULL` rather than
  estimated. Provider payload variants require isolated fixture verification.
- [x] **P3-13** Legacy pre-identity Tier-1 discovery and global unscoped
  LLM-ranking Web entry points are fail-closed; problem-first proposal and
  portfolio admission are the supported path.
- [x] **P3-14** Forge code scout/scaffold have no ungranted direct-LLM or
  provider-error-to-deterministic-success fallback.
- [x] **P3-15** AST caller audit reports 14 classified legacy direct calls and
  zero unclassified sites; any new unclassified site fails the audit.
- [~] **P3-16** Ingestion/extraction, contradiction, abstraction, insight,
  taxonomy and domain-summary calls require a scoped active ResourceGrant.
  The grantless background worker is disabled by default and fails startup
  closed if explicitly enabled. Isolated PostgreSQL/provider/worker CI is
  pending.

## P4. ComputeBackend

- [x] **P4-01** Stable interface includes capability/VRAM query.
- [x] **P4-02** Stable interface includes submit and idempotency key.
- [x] **P4-03** Stable interface includes status/heartbeat.
- [x] **P4-04** Stable interface includes cancel/timeout.
- [x] **P4-05** Stable interface includes artifact collection.
- [x] **P4-06** Stable interface includes usage accounting.
- [x] **P4-07** CPU, LocalGPU, SSHGPU and ColabGPU adapters exist.
- [x] **P4-08** Backend scheduler chooses from a registry instead of transport
  conditionals.
- [x] **P4-09** Submit verifies agenda, idea, stage, grant ID, backend and caps.
- [x] **P4-10** Artifacts certify only `succeeded` jobs and required artifacts.
- [x] **P4-11** SSH configuration stores a credential reference, not password
  material; strict host checking is enabled.
- [x] **P4-12** Remote dependency auto-install defaults off.
- [x] **P4-13** Colab accounts have unique HOME/OAuth/session/quota fields.
- [x] **P4-14** Colab code/artifact roots are isolated and credential/backup
  paths/symlinks are rejected.
- [~] **P4-15** Colab lifecycle ports new/upload/exec/download/stop; CLI syntax
  and artifact canary pending.
- [~] **P4-16** PostgreSQL legacy GPU queue insertion now requires and
  persists a matching `ComputeScheduler` claim/idempotency identity. Local/SSH
  execution internals still use the guarded legacy worker transport and need
  isolated runtime replacement/CI.
- [~] **P4-17** Durable compute idempotency/recovery uses `compute_jobs_v1`;
  claim-before-submit, restart reuse and four-test disposable PostgreSQL
  execution passed.
- [~] **P4-18** Backend failure/timeout/missing artifact tests and disposable
  usage-unknown recovery passed; external backend execution remains pending.
- [x] **P4-19** Scheduler construction fails closed without a durable store;
  ephemeral idempotency requires an explicit test-only flag.
- [x] **P4-20** A transport exception after durable claim records
  `submission_unknown`, stops backend fallback and requires reconciliation.
- [x] **P4-21** Backend `succeeded` first becomes `collecting`; durable success
  requires the persisted grant's artifacts and bounded usage.
- [~] **P4-22** Legacy local/SSH queue submission now enters through a runtime
  `ComputeScheduler` with `ComputeJobRepository`; startup reconciles expiry
  per agenda and attempts settlement of persisted live legacy jobs. Only the
  process holding the scheduler lock may run recovery. Runtime restart/crash
  verification remains pending.
- [~] **P4-23** Failed/cancelled/timed-out backends must persist measured usage
  before terminal settlement; expired jobs with unknown usage are quarantined
  as `usage_unknown`. Disposable PostgreSQL expiry/quarantine passed; external
  backend usage remains pending.
- [~] **P4-24** CPU validation now claims a durable, agenda/idea/grant-scoped
  compute job before entering the synchronous legacy loop, uses stable
  idempotency on retry and settles measured iteration usage plus grant-required
  artifacts. Disposable PostgreSQL execution/restart/fault verification passed.
- [x] **P4-25** CPU exceptions, non-completed legacy returns and artifact
  certification failure cannot be converted into durable compute success.
- [~] **P4-26** `colab_work_requests_v1` persists immutable scoped payloads
  before `ComputeScheduler` admission; a claim-before-session worker,
  deterministic backend identity, artifact/usage settlement and restart
  quarantine are registered when `colab_gpu` is enabled. Disposable PostgreSQL
  fault/restart quarantine passed; real Colab CLI/canary compatibility remains
  pending.
- [x] **P4-27** Runtime scheduler construction honors
  `compute_backends.enabled`; unknown or disabled backends fail closed and SSH
  uses configured reference-only settings.
- [x] **P4-28** The only non-definition `run_validation_loop` call sites are
  the CPU auto-research worker and GPU transport worker; problem-first and
  legacy Web execution bypasses are blocked.
- [~] **P4-29** Application startup completes compute scheduler reconciliation
  before starting auto-research and fails closed on an unknown scheduler
  status. Configured asynchronous Colab also starts through this boundary.
  Startup/restart tests are written but not run.

## P5. Scientific evidence state machine

- [x] **P5-01** Canonical order is exactly planned → sanity → full benchmark →
  audited → decided → manuscript allowed.
- [x] **P5-02** Only one-step monotonic transitions are accepted.
- [x] **P5-03** Pilot cannot advance to full benchmark or confirmation.
- [x] **P5-04** Execution failure cannot advance evidence.
- [x] **P5-05** Full benchmark requires a valid stage-specific grant.
- [x] **P5-06** Evidence audit requires raw artifacts and claim ledger.
- [x] **P5-07** Scientific decision requires held-out evaluator verdict.
- [~] **P5-08** Manuscript permission requires a purpose/subject/time-bound
  HMAC reviewer approval whose key is referenced through environment
  configuration. Signature verification and persistence tests are written;
  external identity issuance remains an operational CI/control-plane item.
- [x] **P5-09** State transitions append an audit record with actor/context.
- [x] **P5-10** Operational completed/succeeded is distinct from scientific
  decision.
- [~] **P5-11** Benchmark manager/audit remain the primary benchmark contract;
  integration tests pending.
- [~] **P5-12** Result/manuscript claim gates use the unified integrity
  contract; tests pending.
- [~] **P5-13** Raw artifact, claim ledger, benchmark, evaluator, holdout and
  verdict hashes are persisted in audit/decision records and cross-checked
  during transitions; isolated PostgreSQL execution is pending.
- [~] **P5-14** Generic validation now emits operational `supported`, not
  `confirmed`; direct positive solution/knowledge/manuscript paths require
  scientific authority. Exhaustive legacy runtime CI remains open.
- [x] **P5-15** Operational `supported/completed` cannot by itself solve a
  problem, update positive signal learning, cascade confirmed knowledge,
  start manuscript retry or count as a positive meta-learning label.
- [x] **P5-16** AST-only state-authority audit finds no
  `scientific_evidence_state` UPDATE outside `EvidenceRepository` and permits
  initial state INSERT only in the reviewed run factory.

## P6. Frontier and Idea Portfolio

- [x] **P6-01** FrontierPacket contains retrieval date and coverage.
- [x] **P6-02** FrontierPacket contains problem status.
- [x] **P6-03** Strongest recent work/latest benchmark/nearest prior art fields
  exist.
- [x] **P6-04** Contribution delta and duplicate/obsolete/solved evidence
  fields exist.
- [x] **P6-05** Counterevidence/negative results and `why_not_obsolete` exist.
- [x] **P6-06** Minimum falsification experiment is mandatory.
- [x] **P6-07** Retrieval snapshot requires source indexes and immutable query
  references.
- [x] **P6-08** Frontier Gate rejects obsolete, duplicate, solved or incomplete
  frontier coverage.
- [x] **P6-09** Rejected Frontier cannot be bypassed when saving a portfolio
  decision.
- [x] **P6-10** All eleven requested estimates have value, interval, evaluator,
  provider, model and evidence sources.
- [x] **P6-11** Candidate family/correlation keys are mandatory.
- [x] **P6-12** Portfolio supports promote/kill/park/revisit.
- [x] **P6-13** Park includes revisit trigger/expiry.
- [x] **P6-14** Kill includes reason/evidence signature preventing immediate
  regeneration.
- [x] **P6-15** Exploitation/exploration/falsification/surprise reserves and
  opportunity/correlation penalties are represented.
- [x] **P6-16** Topic gate/idea taste/surprisal/ROI have no independent grant
  authority in the new control plane.
- [x] **P6-17** ResourceGrant contains all required scope, cap, backend,
  artifacts, expiry and reason fields.
- [x] **P6-18** OutcomeRecord contains actual usage/result/effect/baseline/
  verdict/new information/state/prediction error.
- [x] **P6-19** Calibration reports Brier, token/GPU MAE and impact RMSE and
  cannot auto-update policy.
- [~] **P6-20** `EvidenceGraphFrontierSource` assembles immutable-query-ref
  packets from agenda-scoped research problems, explicitly linked papers,
  results and negative evidence. Operator input supplies assessments, not
  evidence arrays. Disposable PostgreSQL scope checks passed; API runtime is
  pending.
- [~] **P6-21** Operator outcome API accepts only grant/run IDs and assembles
  tokens, compute usage, effects, verdict, artifacts and prediction errors
  from persisted sources. Non-success terminal usage is now durable; unknown
  usage is quarantined. Disposable evidence PostgreSQL verification passed;
  production/API end-to-end verification remains open.
- [-] **P6-22** Policy training is excluded from v1; real OutcomeRecord samples
  are still required before any later calibrated policy proposal.

## P7. Harness Evolution

- [x] **P7-01** HarnessCandidate/Patch/FailureCluster/EvaluationRun/
  RegressionReport/Archive contracts exist.
- [x] **P7-02** Candidate worktree must be a dedicated child of configured
  candidate root.
- [x] **P7-03** Candidate worktree cannot overlap production.
- [x] **P7-04** Candidate database/artifact namespaces are unique and prefixed.
- [x] **P7-05** Candidate environment strips HOME, production DB and secret
  variables.
- [x] **P7-06** Diff module/line limits are policy data, not permanent constants.
- [x] **P7-07** Evaluator/held-out/canary/budget/migration/scientific-policy
  paths are protected.
- [x] **P7-08** Patch metadata has a content hash and base commit.
- [x] **P7-09** held-in, held-out and canary are all mandatory.
- [x] **P7-10** Each evaluation requires evaluator ref/hash and artifacts.
- [x] **P7-11** Cross-agenda evaluations are rejected.
- [~] **P7-12** Approved reports require a cryptographically verified,
  purpose-bound reviewer approval; evaluator output and an operator string
  alone cannot approve.
- [x] **P7-13** Harness repository persists agenda-scoped lineage/evaluation
  records.
- [x] **P7-14** Candidate isolation tests cover a hash-pinned bubblewrap
  command with read-only candidate/evaluator/suite mounts, cleared environment,
  unshared network, isolated output and before/after candidate tree hashes.
  Fresh real held-in/held-out/canary and protected-write/network/no-fallback
  negatives passed with unchanged candidate tree hashes.
- [x] **P7-15** Disposable candidate fixture held-in, held-out and canary suites
  passed on the post-fix real rerun; this is evaluator evidence, not hardware
  or provider canary evidence.
- [~] **P7-16** Detached HMAC approval envelopes bind reviewer, key ID,
  purpose, subject and issuance time, with secret material referenced only by
  environment variable. External reviewer identity/key issuance and rotation
  remain unverified operational controls.
- [x] **P7-17** The evaluator mutation API refuses to run unless production
  path/database boundaries are explicitly configured and the evaluator and
  suite tree hashes match operator-pinned values. It never falls back when the
  isolation binary is unavailable.

## P8. Minimal API and configuration

- [x] **P8-01** Non-sensitive configuration lives in `deepgraph.toml`.
- [x] **P8-02** Credentials are environment/secret references.
- [x] **P8-03** Config covers Agenda/backlog/concurrency/backend limits.
- [x] **P8-04** Config covers portfolio and grant buckets.
- [x] **P8-05** Config covers proposer/evaluator/reviewer routes.
- [x] **P8-06** Config covers ComputeBackend registry.
- [x] **P8-07** Config covers failure, trace and artifact policy.
- [x] **P8-08** Minimal API exposes count-only status.
- [x] **P8-09** Mutation API fails closed without operator token.
- [x] **P8-10** API supports agenda, legacy import, frontier, decision, grant,
  state transition and outcome operations.
- [~] **P8-11** API/TOML integration tests are pending.
- [ ] **P8-12** TOML runtime parse under pinned application Python is pending;
  host Python 3.9 has no TOML parser.
- [-] **P8-13** Complete legacy UI merge is excluded.
- [x] **P8-14** All non-meta-harness legacy API POSTs fail closed with 410;
  the dashboard can no longer edit `.env` or directly start pipeline,
  verification, forge, validation, GPU scheduler or manuscript work.
- [x] **P8-15** Retained agenda-owned legacy reads require a positive
  `agenda_id` and scope joins/children across insights, runs, claims,
  artifacts, manuscripts, bundles and previews.
- [x] **P8-16** Runtime configuration is read-only and discloses only whether
  an API key is configured, never a prefix/suffix fingerprint.
- [~] **P8-17** `run_analysis_once.py` requires explicit agenda/idea/grant/stage
  arguments and revalidates the persisted active grant before processing.
  The operator API can now enqueue existing paper IDs into
  `scoped_ingestion_jobs_v1`; a leased worker resumes checkpointed work,
  bounds retries and revalidates the active LLM grant. PostgreSQL/provider
  failure execution is pending.

## X. Explicitly excluded from first release

- [-] **X-01** No complete dual-Web UI merge.
- [-] **X-02** No PaperBanana/image assets.
- [-] **X-03** No pixel office.
- [-] **X-04** No historical generated papers/submission directories.
- [-] **X-05** No bulk venue/manuscript templates.
- [-] **X-06** No automatic deployment/master replacement.
- [-] **X-07** No production-host benchmark/GPU/pytest.
- [-] **X-08** No trained scheduling policy model.

## V. Validation and delivery

- [x] **V-01** Static audit script has no app import/database access.
- [x] **V-02** 260 Python files passed the final static AST parse/audit.
- [x] **V-03** Side-effect-free SQL AST audit found no definite mismatch:
  839 literal calls, 836 statically countable, and 114 dynamic calls explicitly
  left for review/CI.
- [x] **V-04** `git diff --check` passed.
- [x] **V-04A** Agenda mutation scope audit passes: 154 scoped literal
  UPDATE/DELETE statements, zero definite unscoped or dynamic mutations.
- [x] **V-04B** Scientific-state authority audit passes: two state-bearing SQL
  literals, zero unauthorized mutation locations.
- [x] **V-04C** Direct-LLM caller audit passes inventory completeness: 14
  classified legacy sites, zero ingestion and zero unclassified calls.
- [x] **V-05** Agenda example JSON parsed.
- [x] **V-06** Migration dry-plan recorded statement count/checksum/no
  destructive token.
- [~] **V-07** Final-candidate pure policy lane passed 71 tests, SSH reference
  and host-key regressions passed 19 tests, and targeted
  fault/validation/queue/runtime lane passed 50; the A100 control-plane canary
  passed CPU/A100 submit-settle, idempotency rejection and failure quarantine;
  trusted OutcomeRecord assembly and PostgreSQL compute/evidence rerun passed;
  the prior synthetic aggregate
  remains 60 (validation-loop fairness/manifest subset 22/22). Adapted legacy
  remains 39 passed/30 failed; each
  failure is classified in
  [LEGACY_TEST_CLASSIFICATION.md](LEGACY_TEST_CLASSIFICATION.md) as obsolete
  or requiring a new scoped/granted fixture. Synthetic PostgreSQL runtime and
  queue/recovery checks passed; real backup/provider lanes remain open.
- [x] **V-08** Physical backup was restored only into disposable PostgreSQL 18
  instances with private pgvector 0.8.1; migration first/second ran
  `applied`/`already_applied`, all 48 pre-existing counts including `claims`
  were preserved, integrity checks were clean, and real PostgreSQL integration
  tests passed 6/6.
- [~] **V-08A** Durable compute restart/unknown-submission/artifact-finalization
  tests passed in the disposable PostgreSQL lane (4 passed).
- [~] **V-08B** Content-addressed evidence audit/decision transition passed in
  the disposable PostgreSQL lane (1 passed).
- [x] **V-09** Real bubblewrap held-in/held-out/canary evaluator lanes and
  protected-write/network/missing-bwrap negatives passed with unchanged tree
  hashes.
- [x] **V-10** Migration, CI, canary, rollback and configuration runbooks exist.
- [x] **V-11** Master acceptance matrix states **ACCEPTED for CPU + SSH A100**;
  Colab is explicitly excluded and master merge remains unexecuted.
- [x] **V-12** Root changelog distinguishes added, changed, isolated, excluded
  and unverified work.
- [x] **V-13** Local work is split into base implementation `c25e63c`,
  integration documentation `ee96fba`, control-plane hardening `2cccc7a`,
  checkpoint documentation `a148a11`, CPU admission `4d059cd`, and legacy LLM
  blocking `f66cafe`, plus compute registry enforcement `8f59a65`; none was
  pushed. Legacy Web boundary hardening is `954b858`; it also remains local.
  Residual execution/secret-hint hardening is `d192a8d`, startup recovery
  ordering is `692bb62`, scoped ingestion routing is `a7262a3`, and durable
  unknown-usage schema repair is `bdda49c`. Durable Colab/scoped ingestion
  queues, guarded legacy GPU identity and the isolated evaluator runner are
  `724a3ed51fe4649a720c08fb0c213014eb9d236a`; persisted ingestion completion
  truth and agenda claim serialization are
  `b17c7d110532197a7137a217ffa641b50486d295` and
  `f2f6ea96d27673f077966b4bd2f278717393b0d9`.
- [ ] **V-14** No push until explicit approval and quiescence check.
- [x] **V-15** Frozen source candidate remains
  `6851a991154906f11d8cfc247d22a5d5caa0a834`; final code verification commit
  is `a18dc4968b38290d40603c8909b17a888b57157c` with Git tree
  `593663a7e36e769d28cfd14828b8b8ee92bbbd75` and tracked-content tree
  `d12d6882aafda2780ba93563ced0b88780f346b7312306f382748b5d8128fcd9`.
  Documentation/evidence remains local and unpushed.
- [x] **V-16** Exact `7d0b42a` rollback worktree was clean and isolated
  temp-SQLite startup passed; production rollback was not run.
- [x] **V-17** `main` and `web.app` imported and an isolated temp-SQLite startup
  smoke passed with auto-research/pipeline workers disabled.

## Master decision

- [x] **MASTER-READY** All 16 gates in
  [ACCEPTANCE_EVIDENCE.md](ACCEPTANCE_EVIDENCE.md) are accepted with isolated
  evidence.

Current decision: **ACCEPTED for CPU + SSH A100**. The local master fast-forward
merge is intentionally not executed in this session.

## 2026-08-02 local original-database cutover

- [x] **S-13** Recorded the service configuration without exposing its URL,
  password, or tokens; confirmed the current temporary local database and the
  original `127.0.0.1:5433/deepgraph` rollback source.
- [x] **S-14** Stopped `deepgraph-web.service` and its health/self-heal timers;
  the service was inactive and the original/temporary databases had zero
  non-audit connections before backup.
- [x] **S-15** Created a complete custom `pg_dump` under `/home/ec2-user`,
  verified it with `pg_restore --list`, and recorded SHA-256
  `5681dc302ae0e4fd073faccc0e63e0e88419703a6bd9cc7364ce1b7ea5efe3d2`.
- [x] **V-18** The separate one-time live-local runner requires exact loopback
  host/port/database, `DEEPGRAPH_ALLOW_LIVE_LOCAL_MIGRATION=1`, inactive
  service, explicit confirmation, and a verified backup. Unit coverage passed.
- [x] **V-19** Original local migration first/second results are
  `applied`/`already_applied`; the migration journal has one matching checksum
  row. All 58 pre-existing table counts were preserved, 114 FK checks and the
  orphan audit were clean, and `claims=28700` was unchanged.
- [x] **V-20** A fresh retained local restore passed the required PostgreSQL
  integration tests: meta-harness 1, compute repository 4, evidence repository
  1 passed.
- [x] **S-16** Service `.env` was atomically returned to original local
  `deepgraph`; the temporary configuration rollback file and database dump are
  retained. No URL or credential was written to Git.
- [x] **S-17** Service, public HTTPS, and local HTTP all returned healthy after
  the expected large signal-role index initialization: systemd active,
  localhost:8080 HTTP 200, public HTTPS HTTP 200, and no startup traceback.
- [x] **S-18** No database/table/ref/remote branch was deleted, no force-push
  or master merge occurred, and no remote production database was connected.
