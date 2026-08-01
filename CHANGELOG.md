# Changelog

This file records user- and operator-visible DeepGraph changes. The format is
inspired by Keep a Changelog, but an entry is not considered released until it
has an immutable commit, isolated acceptance evidence, explicit approval and a
tag/deployment record.

## [Unreleased] — meta-harness-v1 integration candidate — 2026-07-31

Status: **development branch; rejected for master pending authorized runtime
canaries and reviewer approval; not deployed**.

Latest final-code verification is
`a18dc4968b38290d40603c8909b17a888b57157c` (`fix: support migration on
physical backup schema`). It adds only the compatibility column required by an
existing `deep_insights` index on the isolated physical backup schema, plus a
static regression assertion. The physical PostgreSQL 18 backup was restored
only into disposable local Unix-socket instances using a private pgvector 0.8.1
build: migration first/second were `applied`/`already_applied`, all 48 existing
table counts including `claims` were preserved, FK/orphan/scope integrity was
clean, and the three PostgreSQL integration files passed 6/6. Final static
audits, 71 policy tests and 50 targeted fault/validation/queue/runtime tests
also passed. The final detached candidate worktree also passed real bubblewrap
held-in/held-out/canary evaluation with read-only inputs, unshared network and
unchanged before/after tree hash. No production database, provider credential,
remote backend, push or deployment was used.

The SSH GPU transport now resolves credentials only from worker `env:...`
references; the legacy global raw-password variable is ignored. The focused
SSH/runtime regression lane passed 18/18. No remote GPU was contacted by this
change.

SSH transport also pins the configured known-hosts file for SSH and SCP. The
follow-up regression lane passed 19/19. A target-3 read-only `nvidia-smi` probe
timed out after 30 seconds; no remote write or canary success is claimed.

Final isolated acceptance record: source candidate frozen at
`6851a991154906f11d8cfc247d22a5d5caa0a834` with candidate tree
`b1c1e8ebfbc0607cc39bb617dad9d56fd949d214122a37dd70541bd634d9feab`.
Post-fix verification commit is
`d33a9f5fbb1bb912f6edff2f87b749d38ec19d25` (Git tree
`607a1fb701357aad77c7003743093f51ab867ce2`, tracked-content SHA-256
`18a5a677ee13ed81d550710c5c390ae3e3b3c23c0991036af465237f164abe2f`).
Static audit, the 71-test pure policy lane, synthetic fault lane, disposable
PostgreSQL queue/recovery lane and real bubblewrap evaluator lanes passed.
The adapted legacy lane remains 39 passed/30 classified failures; no approved
hardware/provider canary or reviewer approval exists, and no master replacement
is authorized.

Local implementation checkpoint:
`c25e63c` (`feat: build controlled meta-harness-v1 candidate`). This hash has
not been pushed and is not an acceptance/deployment identifier. Integration
documentation checkpoint: `ee96fba` (`docs: record meta-harness-v1 lineage and
acceptance state`). Control-plane hardening checkpoint:
`2cccc7a6d293f5063425958fa771f368afdf7077`
(`feat: harden meta-harness control plane`). CPU admission checkpoint:
`4d059cd8ab425e7806b095ce2daaa1697d390272`
(`feat: unify CPU compute admission`). Legacy LLM boundary checkpoint:
`f66cafead3f6b1046122d2e5df7637a300cca2f3`
(`refactor: block ungranted legacy LLM paths`). Compute registry checkpoint:
`8f59a659df04229abf4d759e4d354e3899aac410`
(`fix: enforce configured compute registry`). Legacy Web boundary checkpoint:
`954b858418c4258d2b3c521c6fad39259e47bb8a`
(`fix: close legacy web control bypasses`). Residual bypass checkpoint:
`d192a8dbdce8bed61bbc3d094bf1a17d171239d4`
(`fix: remove residual execution and secret hint bypasses`). Startup recovery
ordering checkpoint: `692bb625309ab096f80464f17dcb44891c126aa0`
(`fix: reconcile compute before research startup`). Scoped ingestion LLM
checkpoint: `a7262a3d7bb4e7fa3983f31d5a7082f9f5118e41`
(`fix: require grants for ingestion LLM work`). Durable unknown-usage schema
checkpoint: `bdda49cad1567190ae7af50315f9b9e3f22627d6`
(`fix: allow durable unknown usage quarantine`). None has been pushed or
accepted as a release.

Durable queues and evaluator isolation checkpoint:
`724a3ed51fe4649a720c08fb0c213014eb9d236a`
(`feat: add durable queues and isolated evaluator`). It is also local,
unpushed and unaccepted. Durable ingestion completion truth and agenda claim
serialization are `b17c7d110532197a7137a217ffa641b50486d295`
(`fix: require durable ingestion completion truth`); the explicit one-running
job per agenda guard is `f2f6ea96d27673f077966b4bd2f278717393b0d9`.

### Lineage and why this candidate exists

The candidate follows the controlled B integration:

```text
GitHub origin/master@6048a95
  -> local integration/meta-harness-v1
  -> scientific-integrity/topic-boundary cleanup
  -> semantic ports from production/topic references
  -> isolated PostgreSQL + candidate worktree + canary acceptance
  -> master replacement only after explicit approval
```

- Base: `6048a9568c79b011074e0dba2662fd473cfab250`
  (`Document paper quality roadmap`, 2026-06-26). This GitHub line supplies the
  newer problem-first, idea-taste, benchmark, dataset, scout, feedback and
  loop-router architecture.
- Production behavior reference:
  `7d0b42af8e8f061c3c16800c44224c110f3b94a0`
  (`本地 6/21 快照:多账号GPU池/Colab后端/verdict闸等未提交改动`,
  2026-07-01).
- Topic-gate/integrity reference:
  `9d24d29c6a7d1017301ffa9c36ff9b4b3dfae88d`
  (`选题三问闸门 + 惊讶度驱动算力分配,并解禁诚实负结果`,
  2026-07-29).
- Production and topic-gate are sibling commits with parent
  `4f78f828704567f4210b8628973d4a0e6ba62868`. That lineage contains historical
  v0.2/v0.3 changelog entries, including the earlier Agenda implementation.
- GitHub master and the production/topic lineage have no merge-base. Therefore
  this work does not merge the histories and does not cherry-pick the large
  production/topic commits. Capabilities are ported by behavior contract.
- Production `7d0b42a` remains unchanged as behavior/migration/rollback
  reference. It is not the candidate base and is not rewritten by this branch.

### Added

- Tier-1 signal prompt serialization now accepts PostgreSQL datetime values,
  with a regression test covering the SQLite/PostgreSQL shape difference.
- The candidate test entry point forcibly clears `DEEPGRAPH_DATABASE_URL` for
  unit and adapted-legacy tests; disposable PostgreSQL remains explicit.

- Versioned Agenda contract, direction intake, loader, relevance, selector and
  fair multi-agenda queue.
- Positive token hard caps, GPU-hour caps, concurrency limits, backend
  allowlists, reserve/settle/release ledger and budget pause/resume.
- Explicit audited legacy-backlog import; old unscoped backlog remains excluded.
- Additive PostgreSQL migration with checksum journal and a guarded,
  dry-run-by-default runner.
- Additive creation of GitHub-line `research_problems`,
  `experimental_evidence_edges` and `benchmark_harness_jobs` for production
  restores where they are absent.
- `FrontierPacket`, `IdeaDecisionPacket`, `ResourceGrant` and `OutcomeRecord`
  contracts with provenance, confidence intervals and agenda scope.
- Frontier freshness/obsolescence gate and transparent best-of-N portfolio
  policy with diversity, exploration, falsification, surprise and opportunity
  cost.
- Evidence-graph Frontier assembly from agenda-scoped research problems,
  explicitly linked papers, results and negative evidence. The operator may
  submit assessments but cannot substitute caller-authored evidence arrays.
- Role-separated proposer/evaluator/reviewer LLM routing, explicit fallback,
  cooldown, metering and failure observations.
- Honest `proposal_pending` idea identities that may receive a proposal-stage
  grant and are promoted in place; without a grant, proposal LLM calls do not
  run.
- Purpose/subject/time-bound detached reviewer approvals whose HMAC key
  material is referenced by environment variable and whose raw signature is
  not persisted.
- Durable provider cooldown state that is reloaded after router/process
  reconstruction instead of existing only in one call object.
- Backend-neutral CPU/LocalGPU/SSHGPU/ColabGPU contracts with capability,
  submit, heartbeat, cancel, timeout, artifact and usage interfaces.
- PostgreSQL-backed compute claim/idempotency lifecycle: claim before backend
  submission, quarantine uncertain outcomes, reuse a persisted live job after
  restart, and require bounded usage plus grant-required artifacts before
  durable success.
- Durable measured-usage settlement for failed/cancelled/timed-out jobs;
  expiry without known usage is quarantined as `usage_unknown`.
- Durable CPU validation transport that claims a grant-scoped compute job
  before execution, resumes by idempotency key and certifies success only
  after measured usage and required artifact collection.
- Shared scoped-ingestion LLM adapter that verifies agenda, idea, stage,
  ResourceGrant, token cap and stable idempotency before proposer routing.
- Multi-account Colab lifecycle adapter with isolated HOME/OAuth/session/quota,
  secret references and credential/backup path rejection.
- PostgreSQL `colab_work_requests_v1` admission, claim-before-session worker,
  deterministic backend identity, artifact/usage settlement, safe bind-gap
  recovery and unknown-remote-usage quarantine.
- PostgreSQL `scoped_ingestion_jobs_v1` plus authenticated enqueue API and a
  leased checkpoint worker with bounded retry and active LLM grant checks.
- Hash-pinned bubblewrap evaluator runner with no inherited secrets/network,
  read-only candidate/evaluator/suite mounts, isolated output and candidate
  tree immutability checks.
- Single monotonic scientific evidence state machine from `planned` through
  `manuscript_allowed`.
- Content-addressed evidence-audit and scientific-decision records that bind
  raw artifacts, claim ledger, benchmark, evaluator, holdout and verdict.
- HarnessCandidate/Patch/FailureCluster/EvaluationRun/RegressionReport/Archive,
  candidate worktree/environment isolation and agenda-scoped persistence.
- Outcome calibration report using success Brier score, token/GPU MAE and
  impact RMSE; policy changes remain reviewer-controlled.
- Trusted OutcomeRecord assembly from persisted grant metering, compute usage,
  experiment artifacts and canonical scientific decisions; the operator API
  no longer accepts caller-supplied outcome metrics.
- Minimal operator-authenticated meta-harness API and count-only status view.
- AST-only agenda mutation scope audit that is a release blocker while any
  definite unscoped legacy UPDATE/DELETE remains.
- AST-only direct-LLM caller inventory that fails on any unclassified new
  `call_llm`/`call_llm_json` site.
- Non-sensitive `deepgraph.toml` policy for agendas, portfolio, grants, LLM
  roles, compute backends, evidence, harness evolution, failures and traces.
- Static source audit, isolated PostgreSQL test skeleton, scientific-integrity
  fixtures, fault tests, runbooks and detailed acceptance checklist.

### Changed

- Reused the existing CSS type scale for the matching dashboard literals,
  preserving computed values without importing unrelated frontend changes.

- Auto-research default cycle now queues only explicitly agenda-scoped work for
  portfolio review; legacy global event/backlog consumption is not invoked.
- Compute recovery now completes before the auto-research worker starts; an
  unrecognized scheduler startup result stops application startup.
- Legacy background ingestion is disabled by default and fails startup closed
  if explicitly enabled without a dynamic agenda-scoped ResourceGrant job.
- Core problem, idea, job, run, artifact, claim, manuscript and evidence-edge
  writes carry agenda scope.
- Experiment feedback updates agenda-local signal outcomes rather than shared
  ingestion counters, preventing cross-agenda learning leakage.
- GPU/forge/validation/benchmark core paths require a matching active,
  non-expired ResourceGrant.
- Forge scout/scaffold/repair and validation code/reproduction repair use the
  granted proposer route with stable idempotency keys and route provenance.
- Proposal method/experiment design, benchmark design, Tier-2 debate,
  manuscript revision and plain final review use scoped role routes; evaluator
  and reviewer calls receive the recorded proposer route for independence.
- Provider-returned explicit cost fields are persisted; absent cost remains
  unknown and is never estimated.
- Local/SSH GPU admission now enters the backend-neutral scheduler and durable
  job repository; startup recovery is per agenda and attempts to settle
  persisted legacy jobs without resubmission.
- PostgreSQL direct legacy GPU queue insertion now requires the matching
  durable compute idempotency identity and persists it on `gpu_jobs`.
- Only the process holding the scheduler lock may perform startup recovery;
  configured asynchronous Colab starts behind that recovery boundary.
- CPU pilot admission now enters the same scheduler/repository. Validation
  exceptions, non-terminal returns and artifact-certification failures
  downgrade the legacy run and cannot produce durable compute success.
- Runtime compute construction now honors the configured enabled registry and
  SSH/Colab reference-only settings. Disabled, unknown or incompletely
  configured backends fail closed instead of being silently activated.
- Legacy Web POST/control surfaces (including runtime `.env` edits, direct
  forge/validation, scheduler start/stop and manuscript generation) return
  410; authenticated meta-harness v1 routes are the only mutation API.
- Retained legacy insight/experiment/manuscript/bundle/paper-preview reads
  require a positive agenda query and use agenda-matched joins.
- Problem-first compatibility code can no longer call the validation loop
  directly, and runtime configuration responses no longer expose partial API
  key fingerprints.
- Legacy pre-identity Tier-1 LLM discovery and the global unscoped LLM insight
  rank endpoint are disabled; agenda-scoped problem-first proposal and
  portfolio admission are their supported replacement.
- Forge codebase selection and scaffold generation no longer retain an
  ungranted direct-LLM or silent deterministic fallback path.
- Extraction, contradiction, abstraction, insight, taxonomy and domain-summary
  LLM work uses the granted proposer route. Multi-role extraction fails as a
  unit and cannot silently fall back to a lower-quality monolithic call.
- Validation reports operational `supported` instead of directly creating
  `confirmed`; positive problem, knowledge, manuscript and meta-learning paths
  require a persisted supported scientific decision.
- GPU jobs fail when validation returns failed/blocked/invalid or no result;
  such failures are no longer persisted as completed jobs.
- Manuscript generation requires `manuscript_allowed` evidence plus reviewer
  approval.
- Validation and manuscript review apply the same scientific evidence rules:
  missing/non-significant p-values, refutation, zero/missing baselines, missing
  metrics and incomplete benchmarks cannot confirm a claim.
- SSH metadata stores credential references instead of passwords, strict host
  checking is enabled and implicit remote dependency installation defaults off.
- Validation-loop fairness now detects method-slugged candidate-only
  canonicalizers, broad benchmark-context propagation and zero-budget answer
  shape changes. Operational benchmark manifests require `supported`; the
  scientific `confirmed` state remains a separate authority.

### Removed or isolated from the generic runtime

- Removed fixed idea8/run13 VOC result completion.
- Removed automatic weakening of non-significance and uncertainty caveats.
- Removed CRPP-specific deterministic scientific prose from the generic
  PaperOrchestra path.
- Moved CGGR/CRPP/idea8 runners, ablations, aliases, shard tools, result
  auditor, tests and historical docs to `plugins/examples/cggr`.
- Removed topic scripts from the default generic agent registry.

The example plugin is non-production and disabled unless explicitly selected.

### Scientific and safety invariants

- `p=1` is not significant.
- A missing p-value cannot support a significance claim.
- `refuted` cannot produce a positive claim even with a low p-value.
- Zero/missing baseline, missing metric or incomplete full benchmark cannot
  confirm.
- Presentation/layout/refinement code cannot invent numbers or strengthen a
  scientific claim.
- Pilot evidence cannot become confirmed.
- Provider/backend failure cannot become completed/confirmed.
- No GPU or high-cost role-routed LLM request is authorized without a matching
  ResourceGrant.
- Candidate code cannot modify evaluator, held-out, canary, budget, migration,
  scientific evidence or safety policy paths.

### Deliberately not included

- Merging both complete Web UIs.
- PaperBanana/image assets, pixel office, historical generated paper trees and
  bulk venue templates.
- Automatic deployment, push or master replacement.
- A trained policy model.
- Production-host pytest, benchmark, GPU, SSH or Colab execution.

### Validation status

Allowed static checks currently report:

- 279 Python files parsed by the broad AST pass and 257 by the release static
  audit at the latest working-tree checkpoint;
- no finding from the topic/integrity/migration/secret static audit;
- SQL AST audit: 839 literal calls, 836 statically countable, no definite
  mismatch, and 114 dynamic calls left for review/CI;
- additive migration plan: 90 statements, 27,718 bytes, SHA-256
  `6379d919c951a827017eacf72e1168d52980bb2d515c5f14d44e5121f01b1185`,
  no destructive token and no database access;
- `git diff --check` passed;
- agenda example JSON parsed.
- agenda mutation scope audit passes: 154 scoped literal mutations, zero
  definite unscoped or dynamic mutations.
- scientific-state authority audit finds two state-bearing SQL literals and
  zero unauthorized UPDATE/INSERT locations.
- direct-LLM audit classifies all 14 remaining legacy calls in blocked,
  non-registered or no-call-site modules; zero ingestion calls and zero
  unclassified calls remain.

The isolated Python 3.13 policy lane passed 71 tests. The refreshed synthetic
fault lane passed 60 tests (validation-loop subset 22/22); report hashes are
recorded in `docs/integration/ACCEPTANCE_EVIDENCE.md`. The adapted legacy lane
remains 39 passed/30 failed; all 30 are individually classified in
`docs/integration/LEGACY_TEST_CLASSIFICATION.md` and no safety contract was
relaxed. Disposable PostgreSQL migration/queue/recovery and real evaluator
lanes now pass; production backup startup, provider/backend calls, CPU/GPU/SSH/
Colab work and reviewer approval remain unverified. The host
Python 3.9 has no TOML parser, so TOML runtime parsing also remains an
isolated-CI item.

### Known incomplete integration

- The granted ingestion route, durable API/worker and lease/retry contract are
  implemented, but their metering and failure behavior still require isolated
  PostgreSQL/provider CI.
- The legacy GPU worker scheduler still contains transport-specific internals;
  CPU/local/SSH admission and durable settlement are bridged, while their
  execution internals still require replacement/runtime CI.
- Colab durable queue/worker wiring and evaluator isolation are implemented,
  but neither has executed against isolated PostgreSQL, bubblewrap, a
  synthetic CLI or a canary.
- Canonical scientific-state SQL mutation authority is statically clean;
  legacy operational/verdict semantics still require isolated end-to-end
  testing.
- Agenda scope has static evidence only; multi-agenda PostgreSQL concurrency
  and fault isolation still require isolated execution.
- Frontier, failed compute usage and reviewer signatures have implementation
  material but no isolated PostgreSQL/API/backend execution evidence.
- Reviewer key issuance, identity authentication and rotation remain external
  operational controls.
- No isolated CI/canary evidence exists yet, so this entry must remain
  Unreleased and the branch must not replace master.

Detailed status:
[implementation checklist](docs/integration/IMPLEMENTATION_CHECKLIST.md),
[LLM caller inventory](docs/integration/LLM_CALLER_INVENTORY.md),
[legacy test classification](docs/integration/LEGACY_TEST_CLASSIFICATION.md),
[porting ledger](docs/integration/PORTING_LEDGER.md), and
[master acceptance matrix](docs/integration/ACCEPTANCE_EVIDENCE.md).
## 2026-08-01 — isolated A100 control-plane evidence

- Ran a disposable PostgreSQL CPU/A100 SSH control-plane canary using a
  short-lived ResourceGrant and secret references only. Durable settlement,
  terminal idempotency rejection, and `submission_unknown` failure quarantine
  passed; no production database, deployment, push, or raw credential was
  used. Full scientific GPU/Colab/provider canaries and reviewer approval remain
  pending.

## 2026-08-01 — OutcomeRecord PostgreSQL regression closure

- Fixed trusted outcome assembly to read the schema's canonical
  `evidence_decision_json`; targeted tests, PostgreSQL compute/evidence lanes,
  and the CPU/A100 canary with persisted OutcomeRecord settlement passed.

## 2026-08-01 — scoped reviewer and API evidence

- Count-only API status, current-candidate evaluator rerun, and signed reviewer
  approval passed for the CPU + SSH A100 release scope. Colab is explicitly
  excluded; real LLM provider execution/restart remains unverified.

- Added a no-network Codex-shaped mock-provider canary record covering grant
  admission, usage settlement, auth failure and cooldown; real provider calls
  remain intentionally unclaimed.

- Real provider canary, provider fault/cooldown restart, and all scoped CPU +
  SSH A100 acceptance gates passed. Colab remains explicitly excluded; master
  merge was not executed.
