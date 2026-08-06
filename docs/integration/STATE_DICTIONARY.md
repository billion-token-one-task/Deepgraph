# Cross-line state dictionary

meta-harness-v1 uses one scientific state and separate operational states.
Operational success must never imply scientific confirmation.

## Scientific evidence state

| Canonical state | Meaning | Legacy states mapped here | May advance when |
|---|---|---|---|
| `planned` | contract exists; no valid run evidence | `candidate`, `pending`, `queued`, `harness_required` | benchmark contract and valid ResourceGrant exist |
| `sanity_passed` | bounded pilot completed with required raw artifacts | `testing`, topic-gate pilot `escalate` | pilot artifact audit succeeds |
| `full_benchmark_complete` | preregistered full protocol completed | `completed` only when completeness is independently proven | all required datasets/models/baselines/seeds/metrics exist |
| `evidence_audited` | raw artifacts, claim ledger and statistics passed audit | `verified` only when audit-backed | evaluator records non-conflicting audit |
| `scientifically_decided` | explicit `supported`, `refuted`, `inconclusive`, or `invalid` verdict | `confirmed`, `failed`, `blocked` are not accepted without mapping evidence | held-out evaluator signs verdict |
| `manuscript_allowed` | decided evidence supports the exact bounded claims | `bundle_ready`, `ready` only after claim gate | reviewer approves claim ledger |

The transition order is monotonic. Pilot evidence can be `supported_at_pilot`
as an observation but cannot create `confirmed`, `scientifically_decided`, or
`manuscript_allowed`.

## Operational dimensions

These dimensions must be stored separately:

- task: `queued | leased | running | succeeded | failed | canceled | timed_out`
- backend: `submitted | provisioning | running | collecting | completed |
  failed | canceled | timed_out | lost`
- portfolio: `promote | kill | park | revisit`
- agenda: `active | paused_budget | paused_manual | closed`
- harness candidate: `draft | evaluating | rejected | awaiting_approval |
  approved | archived`
- durable compute: `submitting | submission_unknown | submitted | running |
  cancel_requested | collecting | succeeded | failed | cancelled | timed_out |
  usage_unknown`
- Colab work: `admitting | queued | running | succeeded | failed | timed_out |
  cancelled | manual_reconciliation`
- scoped ingestion: `queued | running | retryable | succeeded | failed |
  cancelled | manual_reconciliation`

`completed` is accepted as a backend state only after artifact collection and
usage accounting complete. A backend transport error can never map to
`succeeded`, and a task `succeeded` can never by itself advance scientific
evidence.

## Scientific verdict and claim-strength rules

| Evidence condition | Required verdict/claim behavior |
|---|---|
| p-value is 1 | not significant |
| p-value missing | no significance claim |
| evaluator verdict `refuted` | no positive claim even if p is low |
| zero/missing baseline | cannot confirm improvement |
| no metric | cannot confirm |
| incomplete benchmark | cannot confirm |
| compile/layout/polish pass | cannot add numeric facts or strengthen claims |
| negative/null result | preserve it in OutcomeRecord and calibration data |

## Backlog scope

Rows existing before migration receive no `agenda_id` and remain excluded.
They may enter an agenda only through an explicit import record that records
source row, destination agenda, actor, timestamp, reason, and idempotency key.
Keyword similarity alone is never authorization.

## Topic gate: `parked` and the missing-prediction rule

Verified against source 2026-08-06.

The topic gate is a pure function of the candidate row, the agenda and the
policy — no LLM, no database (`agents/topic_gate.py:387`, module docstring
`agents/topic_gate.py:37-39`). There is no `TOPIC_GATE_ENABLED` switch
anywhere in the tree; only `DEEPGRAPH_TOPIC_GATE_*` threshold variables
exist, so the gate cannot be turned off.

A candidate with no recorded prediction (or no confidence) is never passed
by eliciting one with an ungranted model call; it fails the gate with reason
code `topic_gate_prediction_missing` (`agents/topic_gate.py:396-411`, code
defined at `agents/topic_gate.py:72`). A failed gate produces a `kill` or
`park` portfolio decision: only `promote` and `revisit` — the decisions that
can buy resources — are re-checked against the gate before persisting;
killing or parking stays recordable with its reasons
(`meta_harness/repository.py:650-662`). `parked` therefore means "stopped
with an auditable reason, revisitable", never "passed quietly".

## Evidence-ladder refusal blockers (`meta_harness/evidence_state.py`)

Verified against source 2026-08-06. `advance()` allows exactly one forward
step (`meta_harness/evidence_state.py:67-68`); any blocker makes it raise
`EvidenceTransitionError` with the comma-joined, deduplicated blocker list
(`meta_harness/evidence_state.py:147-148`). A refusal is raised, not
persisted: no row is written for a refused transition.

Checked on every transition:

- `resource_grant_invalid` (`evidence_state.py:71`)
- `resource_grant_id_missing` (`evidence_state.py:73`)
- `execution_not_successful` (`evidence_state.py:75`)

Per target state:

- `sanity_passed`: `raw_artifacts_missing` (`evidence_state.py:78`),
  `raw_artifacts_hash_missing_or_invalid` (`evidence_state.py:79-83`)
- `full_benchmark_complete`: `pilot_cannot_complete_full_benchmark`
  (`evidence_state.py:86`), `full_benchmark_incomplete`
  (`evidence_state.py:88`), `benchmark_contract_hash_missing_or_invalid`
  (`evidence_state.py:89-93`)
- `evidence_audited`: `claim_ledger_missing` (`evidence_state.py:96`),
  `independent_evaluation_incomplete` (`evidence_state.py:98`), five
  `*_hash_missing_or_invalid` content-hash checks
  (`evidence_state.py:99-106`), `evaluator_ref_missing`
  (`evidence_state.py:108`), `holdout_ref_missing`
  (`evidence_state.py:110`)
- `scientifically_decided`: `scientific_verdict_missing`
  (`evidence_state.py:112-117`), `verdict_hash_missing_or_invalid`
  (`evidence_state.py:118-122`), the same five hash checks
  (`evidence_state.py:123-130`), `evaluator_ref_missing` /
  `holdout_ref_missing` (`evidence_state.py:131-134`),
  `positive_evidence_decision_failed` when a `supported` verdict lacks a
  passing `decide_evidence` result (`evidence_state.py:135-136`)
- `manuscript_allowed`: `positive_manuscript_requires_supported_verdict`
  (`evidence_state.py:138-139`), `reviewer_approval_required`
  (`evidence_state.py:140-141`), `verdict_hash_missing_or_invalid`
  (`evidence_state.py:142-146`)

The `*_missing_or_invalid` names are produced by `_require_content_hash`,
which accepts only a 64-hex sha256 (optionally `sha256:`-prefixed)
(`evidence_state.py:45-55`).

## `decide_evidence` fail-closed blockers (`contracts/scientific_evidence.py:74-146`)

Verified against source 2026-08-06. Applies the M1/M4 evidence rules; every
blocker is recorded in the returned decision, and `confirmation_allowed`
requires a `supported` verdict, a complete evidence set and significance
(`contracts/scientific_evidence.py:111-125`).

- `evaluator_refuted` / `evaluator_inconclusive` / `evaluator_invalid`
  (`scientific_evidence.py:79-84`)
- `metric_missing` (`scientific_evidence.py:87`)
- `baseline_missing` (`scientific_evidence.py:89`)
- `baseline_zero` (`scientific_evidence.py:90-91`)
- `full_benchmark_incomplete` (`scientific_evidence.py:93`)
- `raw_artifacts_incomplete` (`scientific_evidence.py:95`)
- `claim_ledger_incomplete` (`scientific_evidence.py:97`)
- `independent_evaluator_missing` (`scientific_evidence.py:99`)
- `p_value_missing` (`scientific_evidence.py:101`)
- `not_significant` — a present p-value that does not clear alpha under a
  `supported` verdict (`scientific_evidence.py:103-109`)

Claim strength caps at `bounded_supported` and only when confirmation is
allowed; a `supported` verdict with a metric but incomplete evidence caps at
`descriptive` (`scientific_evidence.py:127-136`).

## Self-heal actions and reason codes (`orchestrator/selfheal_policy.py:37-57`)

Verified against source 2026-08-06. Two actions only: `restart` and `hold`
(`selfheal_policy.py:37-38`). Reason codes are an operator contract — logged,
alerted on and asserted in tests (`selfheal_policy.py:40-41`).

Restart reasons (all three):

- `restart_health_probe_failed` (`selfheal_policy.py:48`)
- `restart_expected_output_stalled` (`selfheal_policy.py:49`)
- `restart_db_idle_in_transaction_stalled` (`selfheal_policy.py:50`)

Hold family (`selfheal_policy.py:42-47,51-57`):
`hold_process_not_running_systemd_owns_recovery`,
`hold_within_startup_grace`, `hold_maintenance_mode`,
`hold_health_probe_unavailable`, `hold_health_ok`,
`hold_health_failure_below_threshold`,
`hold_autonomy_disabled_no_output_expected`,
`hold_no_active_work_no_output_expected`, `hold_awaiting_authority`,
`hold_provider_or_credit_issue_restart_cannot_fix`,
`hold_output_freshness_unavailable`, `hold_output_fresh`,
`hold_restart_cooldown_active`.

## Backend capability tri-state (`meta_harness/backend_capability.py`)

Verified against source 2026-08-06. Exactly three states
(`backend_capability.py:30-32`), semantics in the module docstring
(`backend_capability.py:9-21`):

- `enabled` — listed, fully configured, and recorded as verified by an
  operator after a real canary (CPU is enabled once listed,
  `backend_capability.py:151-152`)
- `unknown` — listed and configured but never verified; usable only for a
  separately authorized canary, never for ordinary scheduling
  (`backend_capability.py:56-57,153-155`)
- `disabled` — not listed, missing configuration, or contradicted by the
  host (`backend_capability.py:146-150`)

There is no fallback path: an unavailable backend never silently becomes
another backend, and a legacy field never enables anything on its own
(`backend_capability.py:20-21,142-144`). Scheduling requires exactly
`enabled` and fails closed otherwise (`backend_capability.py:52-53,215-228`).

## ResourceGrant release reasons (`meta_harness/repository.py`)

Verified against source 2026-08-06. Reservation release reasons:

- `grant_expired` — expiry sweep releases reservations with
  `release_reason='grant_expired'` (`repository.py:1009,1018`) and parks the
  candidate at stage `resource_grant_expired` (`repository.py:1035`)
- `grant_revoked:<reason>` — revocation writes the operator reason,
  truncated to 200 chars (`repository.py:930,939`), and blocks the candidate
  at stage `resource_grant_revoked` (`repository.py:948`)

Both stages are withdrawal park positions, not terminal kills: withdrawn
candidates can be re-examined once a new grant exists
(`repository.py:1060-1061,1076`). Separately, `grants.authorize()` appends
the blocker `grant_expired` when a grant presented for execution is past its
expiry (`meta_harness/grants.py:42-43`).
