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
