# Frontier bootstrap runbook

How to get the first Frontier packet for an Agenda that has none, without
weakening any ResourceGrant rule.

## Why an authority exists

A ResourceGrant requires a persisted `promote`/`revisit` decision packet. A
decision packet requires a Frontier packet that passed the Frontier gate. A
Frontier packet requires an independent evaluator, and an evaluator that calls a
model requires authority to spend tokens. That is a deadlock.

`FrontierEvaluationAuthority` breaks it with strictly less power than a grant:

| Dimension | ResourceGrant | Frontier authority |
| --- | --- | --- |
| scope | agenda + idea | agenda + one research problem |
| operations | stage work | `frontier_assessment` only |
| backends | agenda allowlist | `llm` only, fixed |
| GPU | up to 8 h | 0, structurally |
| tokens | agenda budget | <= 20,000 hard ceiling |
| TTL | <= 24 h (GPU) / 72 h | <= 120 min hard ceiling |
| uses | until expiry | exactly one |
| can create | experiments, proposals, jobs | one Frontier assessment |

Budget is reserved from the same `agenda_resource_ledger` as every other spend,
so a bootstrap cannot escape the Agenda's token cap.

## Preconditions

1. Migration `0002_topic_gate_and_frontier_authority` applied.
2. The research problem exists, belongs to the Agenda, and has explicitly linked
   evidence papers. Without linked evidence the run fails closed before any
   model call.
3. An evaluator provider route is configured and is **independent of the
   proposer route** (different provider, or different model family).
4. `DEEPGRAPH_META_HARNESS_OPERATOR_TOKEN` is set for the web service.

## Issue one authority

```
curl -sS -X POST http://127.0.0.1:8080/api/meta-harness/v1/frontier/authority \
  -H "X-DeepGraph-Operator-Token: $TOKEN" \
  -H 'Content-Type: application/json' \
  -d '{
        "agenda_id": 5,
        "research_problem_id": 1,
        "token_cap": 8000,
        "ttl_minutes": 30,
        "idempotency_key": "agenda-5-problem-1-bootstrap-1",
        "provider": "<evaluator provider name>",
        "model": "<evaluator model id>",
        "model_family": "<evaluator model family>",
        "prompt_version": "frontier-bootstrap-v1",
        "evaluator": "frontier-bootstrap-evaluator",
        "issued_by": "operator:<name>",
        "issue_reason": "agenda 5 has no frontier packet yet"
      }'
```

Re-issuing with the same `idempotency_key` returns the same authority instead of
reserving budget twice.

## Run the one evaluation

```
curl -sS -X POST http://127.0.0.1:8080/api/meta-harness/v1/frontier/bootstrap \
  -H "X-DeepGraph-Operator-Token: $TOKEN" \
  -H 'Content-Type: application/json' \
  -d '{
        "frontier_evaluation_authority_id": <id>,
        "agenda_id": 5,
        "research_problem_id": 1,
        "proposer_provider": "<proposer provider name>",
        "proposer_model_family": "<proposer model family>"
      }'
```

Passing the proposer route is what enforces independence: if the authority names
the same provider *and* the same model family, the request is refused with
`evaluator_not_independent_of_proposer` before any spend.

HTTP 200 means the packet passed the Frontier gate; 202 means it was produced
but the gate rejected it (the reason codes are in the body, and no decision
packet can be built on it).

## Audit it

```
curl -sS "http://127.0.0.1:8080/api/meta-harness/v1/frontier/authority/<id>/audit?agenda_id=5" \
  -H "X-DeepGraph-Operator-Token: $TOKEN"
```

Returns the authority (scope, caps, route, status, close time), every usage row
(success or failure, tokens, cost, failure reason, produced packet id, the
content-addressed `evidence_query_ref`), and totals. No credential appears.

## Expected failures, all fail-closed

| Situation | Result |
| --- | --- |
| provider unavailable | `provider_unavailable:*`, authority revoked, budget settled, no fallback route tried |
| output is not the required JSON | `malformed_assessment:*`, no packet saved |
| research problem has no linked papers | `evidence_unavailable:*`, no model call at all |
| authority expired or already used | refused before any spend; a consumed authority replays its packet id |
| usage above the cap | `token_cap_exceeded`, no packet saved |
| evaluator route equals the proposer route | `evaluator_not_independent_of_proposer` |

A revoked authority is final. Issue a new one with a new idempotency key after
fixing the cause.

## After a passing packet

The packet is only an input. To reach a ResourceGrant you still need:

1. a portfolio decision (`POST /portfolio/decide`) that promotes the idea, which
   re-runs the topic gate against the candidate's recorded prediction, and
2. an explicit grant (`POST /grants`), which enforces the agenda's token and
   GPU budgets, max concurrency, the 8-hour per-grant GPU ceiling, the backend
   allowlist, and a short TTL.

Neither step can be skipped by having a Frontier packet.
