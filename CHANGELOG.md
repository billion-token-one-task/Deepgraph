# Changelog

## Unreleased — 2026-08-03 bounded-autonomy recovery

See `docs/runbooks/RECOVERY_2026-08-03.md` for the full record.

- Replaced the untracked self-heal watchdog with a source-owned, unit-tested
  policy (`orchestrator/selfheal_policy.py`, `scripts/deepgraph_selfheal.py`)
  that never restarts because research output is absent while autonomy is
  disabled, no work is admitted, or an Agenda is waiting on Frontier, portfolio,
  grant, reviewer, budget, or provider authority. Restart now requires a
  repeatedly failing health probe or a genuine stall with work admitted; every
  decision emits an operator-safe reason code and unknown signals do nothing.
- Ported the three-question topic gate and surprise-driven stage ladder from
  `9d24d29` onto the current contracts, without an LLM inside the gate and
  without an enable/disable switch: a candidate with no recorded prediction is
  parked, not silently passed. The gate is re-run at decision persistence, so
  legacy backlog and direct auto-research routes cannot buy resources around it.
- Added `FrontierEvaluationAuthority`: a single-use, agenda- and
  problem-scoped, token- and TTL-capped authority that can produce only a
  Frontier assessment. It resolves the Frontier bootstrap deadlock without
  weakening ResourceGrant rules, is independent of the proposer route, keeps an
  append-only usage ledger, and fails closed on provider, scope, expiry,
  malformed-output, and missing-evidence paths.
- Made compute backend capability explicit: a configured-but-unverified backend
  is `unknown` and is refused for scheduling; Colab without its account manifest
  and local GPU on a GPU-less host are `disabled`; the legacy
  `DEEPGRAPH_GPU_BACKEND` field is reported as a conflict and enables nothing.
- Hardened GPU grants: the 8-hour per-grant ceiling now applies even when an
  Agenda declares no policy (an Agenda may only tighten it), and grants must
  carry a short, unexpired TTL.
- Added additive migration `0002_topic_gate_and_frontier_authority`, a
  read-only production delta inventory, and a file-level deployment manifest
  with SHA256, owner/mode, health checks, and rollback artifacts.

## Unreleased

- Enforce Agenda-configured per-grant GPU-hour caps in addition to aggregate
  GPU-hour budgets and concurrency limits.
- Add bounded, read-only legacy-claim triage for creating a new, explicitly
  scoped self-improving-harness research problem without importing or mutating
  legacy claims or backlog.

## Unreleased — 2026-08-02 local PostgreSQL cutover

- Added a one-time, fail-closed live-local migration path. It accepts only
  `127.0.0.1:5433/deepgraph` and requires an explicit environment opt-in,
  separate confirmation, inactive web service, and a verified custom backup
  directly under `/home/ec2-user`.
- Upgraded the original local `deepgraph` schema with `0001_meta_harness_v1`.
  The first invocation applied it and the second verified checksum no-op; all
  58 pre-existing table counts, 114 FKs, orphan checks and `claims=28700` were
  preserved.
- Restored the service from the temporary local restore to original local
  `deepgraph`, retained both the database and `.env` rollback artifacts, and
  verified systemd, localhost HTTP and public HTTPS health. No remote database
  was contacted and no Git URL/credential was recorded.

## 0.2.0 — meta-harness-v1 (accepted scoped release)

Release scope: CPU + SSH A100. Colab is implemented but explicitly excluded
from this release. The accepted candidate has been fast-forwarded into the
local `master`; remote publication and deployment remain separate operator
actions.

### Why this release exists

DeepGraph already supported literature ingestion, idea generation and
experiment orchestration, but those paths did not share one durable authority
for research scope, resource limits, retries and scientific promotion. Version
0.2.0 introduces that authority without automatically assigning historical
backlog or treating operational completion as scientific confirmation.

For operators, the practical change is straightforward: select work inside a
`ResearchAgenda`, issue a bounded `ResourceGrant`, execute on an allowed
backend, certify measured usage and artifacts, and assemble a trusted
`OutcomeRecord`. Missing scope, expired grants, incomplete evidence and
ambiguous remote submissions fail closed.

### Added

- Agenda-scoped Frontier/Decision/ResourceGrant/Outcome contracts with hard
  token/GPU caps, explicit backlog import, and fail-closed legacy paths.
- Durable PostgreSQL migration, compute claims, idempotency, usage settlement,
  restart recovery, and unknown-outcome quarantine.
- CPU and SSH GPU backends with secret-reference credentials and strict SSH
  known-host pinning.
- Role-separated LLM routing with metering, durable cooldowns, retry policy,
  and failure observations.
- Hash-pinned bubblewrap held-in/held-out/canary evaluation and signed,
  subject-bound reviewer approval.
- Durable scoped ingestion and Colab queue contracts; Colab remains disabled
  in the 0.2.0 release scope.

### Changed

- Legacy global backlog consumption is replaced by agenda-local selection and
  explicit audited import.
- CPU and SSH GPU submissions are persisted before transport execution and
  reconciled after restart instead of being blindly resubmitted.
- Operational `completed`/`supported` states no longer authorize a positive
  scientific claim or manuscript by themselves.
- LLM proposer, evaluator and reviewer routes are separated, metered and
  recorded with prompt/model provenance.
- Operator mutations move to `/api/meta-harness/v1`; legacy control endpoints
  fail closed while the count-only status endpoint remains readable.

### Safety and compatibility

- PostgreSQL migration `0001_meta_harness_v1` is additive, checksum-journaled
  and repeatable. It was repaired for older backups whose `deep_insights`
  table lacked `research_problem_id`.
- Existing unscoped rows remain unscoped. The release does not silently attach
  historical work to a new Agenda.
- SSH credentials and reviewer keys are runtime secret references, not stored
  values. SSH execution requires strict known-host pinning.
- Thirty adapted-legacy test failures are retained as audited obsolete or
  replaced contracts; compatibility shims do not restore grantless, unscoped,
  password-bearing or unlimited behavior.

### Verification

- Physical disposable PostgreSQL restore: migration twice, 48 table counts
  preserved including `claims`, FK/orphan/scope checks, and repository tests.
- Policy/fault/evaluator lanes passed; adapted legacy failures remain explicitly
  classified without weakening grant or scope rules.
- CPU/API, SSH A100, real provider, provider cooldown restart, and reviewer
  approval passed in isolation. Those acceptance runs did not connect a
  production database or perform a deployment; publication and deployment are
  separately controlled operator actions.

### Release boundaries

- Accepted: CPU, SSH A100 control-plane/backend execution, PostgreSQL durable
  state, real LLM provider routing, cooldown restart, evaluator isolation,
  trusted OutcomeRecord assembly and signed reviewer approval.
- Excluded: Colab runtime and OAuth lifecycle. The contracts remain disabled
  until a later release completes its own acceptance.
- Not claimed: a scientific A100 benchmark result, production deployment,
  production database migration, or continued support for unsafe legacy
  control paths.

### Upgrade notes

1. Rehearse the migration twice on a disposable physical backup restore and
   preserve pre-existing table counts before touching any deployment target.
2. Start with `DEEPGRAPH_COMPUTE_BACKENDS=cpu`; enable `ssh_gpu` only after
   target, credential reference and known-hosts pinning are configured.
3. Create new work inside an Agenda. Import historical backlog only through the
   audited explicit-import path.
4. Treat merge, remote publication, database migration and deployment as four
   separate operator actions.

## 0.1.x — previous DeepGraph engine

Literature ingestion, evidence graph construction, opportunity discovery,
experiment orchestration, manuscript generation, and the original dashboard.
See Git history for the detailed pre-0.2.0 development record.
