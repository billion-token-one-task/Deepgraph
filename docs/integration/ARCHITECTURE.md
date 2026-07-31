# meta-harness-v1 architecture

The first closed loop is:

```text
ingestion -> evidence graph -> agenda -> frontier/problem-first
          -> idea portfolio -> benchmark contract -> granted compute
          -> evidence audit -> outcome feedback -> harness evaluation
```

## Authority boundaries

- Agenda owns scope and hard resource ceilings.
- Frontier produces freshness, prior-art, contradiction, and obsolescence
  evidence. It cannot grant compute.
- Topic gate, idea taste, surprisal, novelty, and ROI are portfolio features.
- IdeaPortfolioManager is the only decision component that may propose a
  ResourceGrant.
- ResourceGrant admission is the only path to a GPU or high-cost LLM call.
- ComputeBackend executes an already-authorized request and reports truthful
  operational state; it cannot decide scientific status.
- The evidence state machine and held-out evaluator decide scientific state;
  operational `completed/supported` has no positive downstream authority.
- Harness candidates operate only in isolated worktrees and database
  namespaces. Evaluator, holdout, budget, safety policy, and production results
  are read-only and hash-pinned.
- Reviewer/human approval is required for a harness upgrade.

## Bitter Lesson placement

General search, explicit candidate lineage, OutcomeRecord feedback, and
calibrated portfolio decisions are the scalable core. Domain runners and
method aliases are opt-in example plugins. Additional hand-written agents do
not gain independent budget authority.

The v1 policy is a transparent best-of-N heuristic with logged inputs,
confidence intervals, correlations, opportunity cost, and reserved budget
buckets. It is intentionally replaceable by a constrained contextual bandit
or Thompson sampler after sufficient OutcomeRecord data exists.

## Isolation model

| Surface | Isolation |
|---|---|
| agenda queue/data | mandatory `agenda_id`; no global fallback |
| legacy backlog | excluded until explicit import |
| candidate source | dedicated Git worktree under configured candidate root |
| candidate database | unique PostgreSQL database/schema namespace |
| candidate artifacts | immutable candidate/evaluation prefix |
| Colab | per-account HOME, OAuth/session and quota; secret references only |
| SSH | backend adapter; no scheduler transport branches |
| policy inputs | content-addressed, candidate read-only |

## Implemented module map

| Authority | Module |
|---|---|
| Agenda scope and hard budget | `contracts/agenda.py`, `agents/agenda_repository.py` |
| Direction intake/selection | `agents/direction_intake.py`, `agenda_loader.py`, `agenda_selector.py`, `agenda_orchestrator.py` |
| Frontier construction/gate | `meta_harness/frontier_builder.py`, `frontier.py` |
| Decision and resource allocation | `meta_harness/portfolio.py`, `grants.py`, `repository.py` |
| Role-separated LLM | `meta_harness/llm_routing.py`, granted entry point in `agents/llm_client.py` |
| Backend-neutral execution | `meta_harness/compute.py`, `compute_repository.py`, `backends/colab_cli.py`, `backends/colab_durable.py` |
| Scientific state | `contracts/scientific_evidence.py`, `meta_harness/evidence_state.py` |
| Feedback/calibration | trusted OutcomeRecord assembly in `repository.py`, `meta_harness/calibration.py` |
| Harness evolution | `harness_evolution.py`, `candidate_workspace.py`, `evaluator_runner.py`, `harness_repository.py` |
| Scoped ingestion queue | `ingestion_queue.py`, `orchestrator/scoped_ingestion_worker.py` |
| Operator control/observation | `web/meta_harness_routes.py` |

The v1 APIs and repositories form an explicit control plane. The legacy
background scheduler is not yet the authoritative implementation of every
phase: legacy local/SSH execution branches and legacy state writes require
further adapters. Durable compute claims are committed before a backend call;
an uncertain response is quarantined for manual reconciliation, and backend
success remains `collecting` until required artifacts and bounded usage are
persisted. Colab and scoped ingestion now have durable claim workers, while
their PostgreSQL/provider/backend crash tests remain pending. Until that work
is complete, operator/API progression must remain fail-closed and the
candidate is not deployable.

## Resource reservation hierarchy

```text
ResearchAgenda hard cap
  -> agenda_resource_ledger reservation
    -> ResourceGrant (idea + stage + expiry + backends + artifacts)
      -> resource_grant_usage_reservations (metered LLM calls)
      -> compute job usage
    -> OutcomeRecord settles actual total usage
```

No layer may interpret unconsumed capacity as actual usage. Provider cooldown
and route observations are durable control-plane state. Expired grants release
reservations and block their queued job; they do not produce an OutcomeRecord
or scientific success. The operator outcome endpoint accepts only a persisted
grant/run identity and derives values from trusted stores.
