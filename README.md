# DeepGraph

DeepGraph is an open research engine with a control plane in front of it. It
ingests papers, extracts structured evidence into a knowledge graph, and
proposes research questions from that graph. What distinguishes it from a
literature-mining pipeline is the part that says *no*: nothing the system
produces becomes a scientific claim until an audited evidence ladder allows it,
and nothing spends a token or a GPU-hour without a bounded, expiring
authorization.

**The system has produced no scientific findings.** As of 2026-08-04,
`scientific_decisions_total` is 0. The engineering that would let a finding be
trusted exists and is tested; a finding does not. This README is written so that
distinction survives skimming.

> 中文导读：DeepGraph 是一个带控制平面的开放研究引擎。它读论文、抽取证据、
> 构建知识图谱并生成研究问题，但真正的重点在于"拒绝"：任何结论必须走完可审计的
> 证据阶梯才能被称为科学结论，任何花费必须先拿到有上限、会过期的授权。
> **截至 2026-08-04，系统还没有任何科学结论**（`scientific_decisions_total = 0`）。
> 下文中的论文数、想法数、实验运行数都属于"运行状态"，不是"科学成果"，两者在
> 数据库、API 和界面上始终分开显示。

---

## Read this first: two registers

Every run in DeepGraph carries two independent statuses. They are stored
separately, served separately, and rendered as two separate badges. They are
never merged, and neither substitutes for the other.

| Register | Question it answers | Vocabulary |
|---|---|---|
| **Operational** (`RUN`) | Did the job execute? | `planned`, `running`, `completed`, `failed`, `cancelled` |
| **Scientific** (`EVIDENCE` / `DECIDED`) | What does the audited evidence say? | `not assessed`, `sanity_passed`, `full_benchmark_complete`, `evidence_audited`, `decided: supported / refuted / inconclusive`, `manuscript_allowed` |

The rule that matters: a job whose operational status is `completed` is
scientifically `not assessed` by default. Finishing is not evidence. The UI
renders that pairing as a neutral grey badge, never a green tick
(`web/static/js/app.js:3170-3202`).

This discipline is the reason the rest of this document is careful about which
numbers it quotes and what it calls them.

---

## Status as of 2026-08-04

### Scientific register

| Measure | 2026-08-04 |
|---|---|
| Scientific decisions (`scientific_decisions_total`) | **0** |
| Runs that have reached `scientifically_decided` | **none** |
| Findings, supported claims, manuscripts allowed | **none** |

### Operational register

Counts of work processed. These are **inputs and process, not achievements**.
They say the machine ran; they say nothing about whether anything was learned.
Recorded 2026-08-04; the dashboard serves the same counts from `/api/stats`.

| Measure | 2026-08-04 |
|---|---|
| Source papers analyzed | 6,729 |
| Paper ideas generated | 97 |
| Experiment runs | 111 |

### Autonomy

The global autonomy switches are **off**. The deployed instance runs with
`DEEPGRAPH_AUTO_RESEARCH_ENABLED=false` and `DEEPGRAPH_AUTO_PIPELINE_ENABLED=0`
(verified 2026-08-03, `docs/runbooks/RECOVERY_2026-08-03.md`). Work is started
deliberately, one named candidate at a time, by an operator
(`scripts/run_bounded_pilot.py`).

Turning those switches on would not remove the gates. A grant is still required
before any spend, the topic gate still runs at decision persistence, and the
evidence ladder still refuses transitions whose evidence is missing. Autonomy
changes who presses start, not what is enforced.

### The first closed chain

On 2026-08-04 the authorization -> execution -> settlement chain closed end to
end for the first time, producing `OutcomeRecord` id=1:

```
2,612 tokens spent
execution_result = failed
verdict          = inconclusive
evidence ladder  = planned  (unchanged; the failure bought no promotion)
unused reservation refunded to the agenda budget
```

This is presented as evidence that **the accounting is honest**, not as a
result. The pilot failed, the system recorded that it failed, the ladder
correctly did not move, and the budget that was reserved but not spent went
back. A system that only ever settles successes cannot be trusted to settle
anything.

---

## How a question becomes evidence

```
research direction              submitted as a small config, or by an operator
  -> ResearchAgenda             scope + hard token/GPU budget + backend allowlist
  -> FrontierPacket             what is already known, prior art, why not obsolete
  -> topic gate                 written prediction, priced in bits of surprise
  -> IdeaDecisionPacket         promote / kill / park / revisit, with reason codes
  -> ResourceGrant              bounded, expiring authorization for ONE stage
  -> execution                  CPU or SSH GPU; measured usage, certified artifacts
  -> OutcomeRecord              exact settlement, including of failures
  -> evidence ladder            content-hash-gated, one step at a time
  -> reviewer approval          human HMAC signature, required for manuscripts
```

Each arrow is a place the chain can stop, and stopping is a first-class,
recorded event rather than an error to be retried away.

---

## Innovations

These are designed mechanisms, not marketing. Each entry names the files that
implement it so the claim can be checked.

### 1. An evidence ladder with content-hash gates

Six monotonic states, one forward step at a time, no skipping:

```
planned -> sanity_passed -> full_benchmark_complete
        -> evidence_audited -> scientifically_decided -> manuscript_allowed
```

Every transition is checked against the artifacts it depends on. Hashes must be
real sha256 digests of the actual outputs; a missing or malformed one produces a
named blocker (`raw_artifacts_missing`, `holdout_ref_missing`,
`benchmark_contract_hash_missing_or_invalid`,
`positive_evidence_decision_failed`, ...) and the transition is refused. A pilot
run cannot claim a completed full benchmark.

A `supported` verdict has to survive a second, independent check. The repository
does not take the caller's word for it: at `scientifically_decided` it re-derives
the decision from the stored audit record and refuses on any hash mismatch, and
the decision rule itself is fail-closed -- a missing metric, a missing or *zero*
baseline, an incomplete claim ledger, a missing independent evaluator, a missing
p-value, or a p-value that is not significant each block confirmation.
`manuscript_allowed` then requires a `supported` verdict **plus** a reviewer
approval record **plus** a matching verdict hash.

Accepted transitions are written to `evidence_state_transitions` with their full
context, and `scientifically_decided` additionally writes a
`scientific_decision_records` row, in one transaction. A refusal raises with its
named blockers and persists nothing: the run stays exactly where it was.

`contracts/meta_harness.py:22-29`, `meta_harness/evidence_state.py`,
`contracts/scientific_evidence.py:74-135`,
`meta_harness/repository.py:440-585`.

### 2. The two-register status model, carried all the way to the UI

The ladder is only worth having if a reader can see it. Most systems lose this
at the API boundary: the database knows the difference between "the job
finished" and "the hypothesis is supported", and the web layer renders one green
checkmark for both.

DeepGraph renders them as two components that cannot be substituted for each
other. `RUN: completed` and `EVIDENCE: not assessed` appear side by side.
Refuted renders in purple, because refutation is a valid scientific outcome; red
stays reserved for operational failure. Refused admission gates carry their
reason codes into the timeline, and failed jobs appear there with a truncated,
path-scrubbed failure reason -- both with the same standing as successes.

A read-only provenance API serves this: public agenda list, per-run and per-idea
ladder state rolled up with verdicts, selection rationale including *rejected*
candidates with their reason codes, and a merged chronological timeline built
from frontier packets, decisions, grants, compute jobs, evidence transitions and
outcomes. Every response passes an explicit field allowlist, absolute paths in
free text are redacted, and missing tables yield empty results rather than
errors.

`web/provenance_routes.py`, `web/static/js/app.js:3170-3222`,
`tests/test_provenance_web.py` (11 tests).

### 3. A topic gate that prices admission in bits

Before a candidate can buy compute it must answer three questions: is there a
written prediction with a stated confidence; would both outcomes lead to
different actions; is it already published. Passing is measured in **bits of
surprise** -- the information a result would carry -- not in whether the job
would run. A pilot passes by refuting a confident prediction, not by completing.

Two properties are deliberate and both are load-bearing:

- **No LLM inside the gate.** The gate is pure: no model call, no database
  read. An earlier version elicited the prediction with a model call and *passed
  the candidate when the provider was down*, which is both an ungranted LLM
  route and a silent fallback. A candidate with no recorded prediction is now
  parked with an auditable reason.
- **No enable/disable switch.** `TOPIC_GATE_ENABLED` does not exist. A kill
  switch on a gate is a bypass. Thresholds are configurable; the gate itself is
  not optional.

The gate is re-run at decision persistence, so legacy backlog, operator routes
and direct auto-research calls all converge on the same check and none of them
can buy tokens around it.

`agents/topic_gate.py`, `meta_harness/topic_gate_admission.py`,
`meta_harness/repository.py:638-660`, `agents/agenda_selector.py`,
`tests/test_topic_gate.py` (27 tests).

### 4. Frontier bootstrap authority: breaking a circular dependency without a hole

A ResourceGrant requires a portfolio decision, which requires a Frontier packet,
which requires an evaluator, which required a ResourceGrant. The usual fix is an
exception: let the first call through unauthorized. That is a hole, and holes
get reused.

`FrontierEvaluationAuthority` is instead a *strictly smaller* authority that
cannot become a ResourceGrant:

- binds to exactly one active agenda and one persisted research problem;
- hard token ceiling of 20,000 and TTL ceiling of 120 minutes, both of which
  configuration may lower but never raise;
- one pinned provider / model / model family / prompt version;
- may produce `frontier_assessment` and nothing else; `backend_allowlist` is
  `("llm",)` and `max_gpu_hours` is 0;
- reserves agenda budget through the same ledger as every other spend, with an
  append-only per-attempt usage ledger;
- provenance on the produced packet comes from the authority, never from the
  model's claim about itself.

It fails closed on every path -- unavailable provider, missing scope, expired
authority, malformed output, missing linked evidence, usage above the cap -- and
each failure settles the ledger and closes the authority rather than falling
back to an unscoped call. A consumed authority replays its packet id instead of
spending again.

`contracts/meta_harness.py:132-192`, `meta_harness/frontier_authority.py`,
`meta_harness/frontier_bootstrap.py`, `docs/runbooks/FRONTIER_BOOTSTRAP.md`,
`tests/test_frontier_bootstrap_authority.py` (26 tests).

### 5. ResourceGrant economics: reserve, measure, settle -- including failures

A grant authorizes one stage of one idea. It carries a token cap, a GPU class
and GPU-hour cap, a backend allowlist, required artifact outputs, an idempotency
key and an expiry. A grant that bounds nothing is rejected at construction; so
is one with an empty backend allowlist or no artifact requirements.

The lifecycle is reserve-then-settle. Issuing a grant reserves budget against
the agenda; execution records measured usage; settlement writes the actual spend
and *releases the difference*, with a reason recorded on the ledger row
(`grant_expired`, `grant_revoked`, ...). An expired or revoked grant is
reconciled rather than left holding budget, and its candidate is requeued rather
than stranded.

`OutcomeRecord` id=1 is the worked example: a failed execution, settled exactly,
2,612 tokens charged, the remainder refunded, the ladder unmoved.

`contracts/meta_harness.py:297-372`, `meta_harness/repository.py:778-1030`,
`tests/test_gpu_grant_enforcement.py` (18 tests).

### 6. Retrospective review: letting history onto the ladder without diluting it

Work done before the governance existed has no grant, no held-out split and no
authorization chain. Two bad options present themselves: pretend it qualifies,
or throw it away.

The retrospective path takes a third. A historical run can be walked onto the
ladder, but only through an explicit HMAC-signed reviewer approval that binds to
that exact run, and the resulting verdict is **capped at `inconclusive`** --
`supported` and `refuted` are unreachable by construction, because there is no
holdout to earn them. The whole walk happens in one audited transaction and the
cap is stated in the record itself.

`meta_harness/retrospective_review.py`, `meta_harness/reviewer_approval.py`,
`meta_harness/repository.py:846-860`, `scripts/retrospective_review.py`,
`tests/test_retrospective_review.py`.

### 7. Direction intake: a sentence becomes a budgeted, scoped agenda

A research direction submitted as a small config is parsed deterministically
into a `ResearchAgenda` with a hard token cap, a GPU-hour cap, a backend
allowlist, a max concurrency, and `backlog_policy: explicit_import_only` so it
cannot silently inherit historical work. A submission with no positive token
budget is rejected. The parse produces an echo for confirmation before anything
is created.

The submission UI posts to the existing token-gated operator endpoint; it adds
no new authentication mechanism and no execution controls. Submitting proposes;
it does not authorize compute.

`agents/direction_intake.py`, `web/static/js/app.js:4034-4110`,
`research_agendas/harness_edit_loop_study.v1.json` (worked example: one
falsifiable question, a fixed base model, a fixed task suite, a single editable
surface, a pre-registered held-in/held-out split, and negative results
explicitly in scope).

### 8. Operations discipline as part of the contract

- **Manifest-driven deployment.** Every deployment is a generated manifest with
  per-file source commit, sha256, size, target path, owner, mode, backup
  artifact, health check and acceptance criterion, grouped into batches. Files
  the runtime has locally modified are flagged `operator_diff_required` and must
  be diffed before being overwritten; no whole-tree convergence is attempted.
  `deploy/manifest/`, `scripts/build_deployment_manifest.py`,
  `scripts/deploy_manifest_batch.py`.

- **A self-heal watchdog that mostly refuses to act.** The predecessor restarted
  the service because research output was absent -- while autonomy was
  deliberately off. The replacement restarts only on a repeatedly failing health
  probe, on a genuine stall with work actually admitted, or on a stuck
  PostgreSQL `idle in transaction` session. Every other outcome is a hold with a
  named reason code (`hold_autonomy_disabled_no_output_expected`,
  `hold_awaiting_authority`, `hold_provider_or_credit_issue_restart_cannot_fix`,
  ...), and unknown signals do nothing.
  `orchestrator/selfheal_policy.py`, `scripts/deepgraph_selfheal.py`,
  `docs/runbooks/SELFHEAL.md`, `tests/test_selfheal_policy.py` (46 tests).

- **Backends are innocent until verified.** A configured-but-unverified backend
  is `unknown` and refused for scheduling, usable only for a separately
  authorized canary. Colab without its account manifest and local GPU on a
  GPU-less host are `disabled`. The legacy `DEEPGRAPH_GPU_BACKEND` field is
  reported as a conflict and enables nothing. There is no fallback path: an
  unavailable backend never silently becomes another one.
  `meta_harness/backend_capability.py`, `tests/test_backend_capability.py`
  (16 tests).

---

## Architecture

```
Papers (arXiv)
  |
  v
Grant-scoped ingestion  -- PDF parse -- LLM extraction
  |
  v
Knowledge graph (entities, relations, claims, evidence)
  |
  +--> Domain summaries and opportunity briefs
  |
  v
Signal harvester (SQL-based, zero LLM cost)
  |   cross-node overlap, convergent patterns,
  |   contradiction clusters, performance plateaus
  |
  +--> Tier 1: paradigm agent -- structural isomorphisms across subfields
  |
  +--> Tier 2: paper idea agent -- executable paper ideas
  |
  v
Meta-harness control plane
  |   agenda scope, topic gate, frontier packet, portfolio decision,
  |   ResourceGrant, execution, settlement, evidence ladder
  |
  v
Knowledge loop <-- meta-learner
```

Ingestion is agenda-scoped. The old unscoped paper worker is retired: its
`start()` returns `disabled_resource_grant_required` and it can no longer pull
work on its own (`orchestrator/paper_worker.py:108-112`). Ingestion now runs
through `orchestrator/scoped_ingestion_worker.py` against
`scoped_ingestion_jobs_v1`.

### Repository map

| Directory | Purpose |
|---|---|
| `contracts/` | Versioned record types and their validation rules |
| `meta_harness/` | Control plane: ladder, grants, authorities, repository |
| `ingestion/` | arXiv discovery and PDF parsing |
| `agents/` | Extraction, insight generation, gating, experiment orchestration |
| `db/` | Schema, migrations, taxonomy, evidence graph, entity resolution |
| `orchestrator/` | Scheduling, workers, self-heal policy, compute runtime |
| `web/` | Flask API, provenance API, dashboard |
| `deploy/` | Systemd units, Caddyfile, deployment manifests |
| `scripts/` | Operator CLIs, audits, migrations, manifest tooling |
| `tests/` | 90 test modules |

Code is organized into compatibility-first big-agent folders
(`agents/paper_extraction/`, `graph_construction/`, `idea_generation/`,
`experiment_planning/`, `experiment_execution/`, `manuscript_generation/`,
`orchestration/`). Existing module imports stay valid; new code should use these
folders as ownership boundaries. See `agents/agent_registry.py` and
`docs/agent_architecture.md` for the exact legacy module map.

---

## Quick start

Python 3.12 or newer is required (`pyproject.toml`).

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# Edit .env: at minimum DEEPGRAPH_LLM_API_KEY
export $(grep -v '^#' .env | xargs)
python3.12 main.py
```

Then open `http://localhost:8080`.

This starts the engine and dashboard. The meta-harness control plane
additionally needs PostgreSQL; see below.

## Meta-harness setup

CPU is the safe default backend. Never rehearse a migration against a
production URL.

1. Rehearse the additive migration on a disposable physical restore, following
   [MIGRATION_RUNBOOK.md](docs/integration/MIGRATION_RUNBOOK.md). Review the
   plan, then apply only to the approved database:

   ```bash
   python3.12 scripts/meta_harness_migration.py

   # Inject DEEPGRAPH_MIGRATION_DATABASE_URL from a secret store first.
   python3.12 scripts/meta_harness_migration.py \
     --apply \
     --confirm-isolated-restore I_UNDERSTAND_THIS_WRITES_AN_ISOLATED_RESTORE \
     --source-commit "$(git rev-parse HEAD)"
   ```

2. Start with CPU and a short-lived operator token from your secret manager:

   ```bash
   export DEEPGRAPH_COMPUTE_BACKENDS=cpu
   export DEEPGRAPH_META_HARNESS_OPERATOR_TOKEN='<secret-store-injected>'
   python3.12 main.py
   ```

3. Confirm the control plane can read its schema. This endpoint returns counts
   only, never business rows:

   ```bash
   curl http://localhost:8080/api/meta-harness/v1/status
   ```

4. To enable the SSH GPU backend, configure references and pinned host identity
   only -- never values:

   ```bash
   export DEEPGRAPH_COMPUTE_BACKENDS=cpu,ssh_gpu
   export DEEPGRAPH_GPU_MODE=ssh
   export DEEPGRAPH_GPU_REMOTE_SSH_HOST='<approved-host>'
   export DEEPGRAPH_GPU_REMOTE_SSH_USER='<approved-user>'
   export DEEPGRAPH_COMPUTE_SSH_TARGET_REF=env:DEEPGRAPH_SSH_TARGET
   export DEEPGRAPH_COMPUTE_SSH_CREDENTIAL_REF=env:DEEPGRAPH_SSH_CREDENTIAL
   export DEEPGRAPH_SSH_KNOWN_HOSTS='<reviewed-known-hosts-file>'
   ```

   A backend configured this way is `unknown`, not `enabled`. It becomes usable
   for ordinary scheduling only after an operator records a real canary in
   `DEEPGRAPH_COMPUTE_VERIFIED_BACKENDS`. Audit the current state read-only with
   `python3.12 scripts/meta_harness_backend_audit.py`.

5. Mutations require the `X-DeepGraph-Operator-Token` header on
   `/api/meta-harness/v1/*`. Missing scope, expired grants, unlimited budgets
   and unknown backend outcomes all fail closed. If the token env var is unset,
   the mutation API is disabled entirely.

To execute one authorized candidate without touching the autonomy flags:

```bash
python3.12 scripts/run_bounded_pilot.py \
  --agenda <id> --idea <id> --grant <id> --actor 'ops:<who>' --dry-run
```

Policy fields and secret-reference configuration:
[CONFIGURATION.md](docs/integration/CONFIGURATION.md). Evidence behind the
accepted CPU + SSH A100 release scope:
[ACCEPTANCE_EVIDENCE.md](docs/integration/archive/ACCEPTANCE_EVIDENCE.md).

## Configuration

Defaults live in `deepgraph.toml`. Environment variables and `.env` override
TOML, which keeps existing deployments and long-running jobs compatible.

| Variable | Description |
|---|---|
| `DEEPGRAPH_LLM_API_KEY` | Required. LLM API key for extraction and generation |
| `DEEPGRAPH_LLM_SECONDARY_*` | Optional second OpenAI-compatible route |
| `DEEPGRAPH_LLM_EXTRA_PROVIDERS_JSON` | Optional JSON list of additional routes |
| `DEEPGRAPH_META_HARNESS_OPERATOR_TOKEN` | Enables the operator mutation API; unset means disabled |
| `DEEPGRAPH_COMPUTE_BACKENDS` | Backends offered, e.g. `cpu,ssh_gpu` |
| `DEEPGRAPH_COMPUTE_VERIFIED_BACKENDS` | Backends an operator has canaried; only these schedule work |
| `DEEPGRAPH_AUTO_RESEARCH_ENABLED` | Global autonomy switch for discovery |
| `DEEPGRAPH_AUTO_PIPELINE_ENABLED` | Global autonomy switch for the ingestion pipeline (default off) |
| `DEEPGRAPH_TOPIC_GATE_*` | Gate thresholds (surprise bits, min expected bits, confidence ceiling). There is no on/off switch |
| `DEEPGRAPH_PROFILE` | `machine_learning` or `open_science` |
| `DEEPGRAPH_ROOT_NODE_ID` | Defaults to `ml` or `science` by profile |
| `DEEPGRAPH_ARXIV_CATEGORIES` | Optional comma-separated arXiv category override |
| `DEEPGRAPH_WEB_PORT` | Dashboard port (default 8080) |

The `open_science` profile spans mathematics and statistics, physics, chemistry
and materials, life sciences, medicine and health, earth and climate,
engineering, and computing and AI:

```bash
export DEEPGRAPH_PROFILE=open_science
export DEEPGRAPH_ROOT_NODE_ID=science
python3.12 main.py
```

The SciForge discovery pipeline has further tuning knobs, set through
`DEEPGRAPH_BULK_*` environment variables (the `DISCOVERY_BULK_*` settings in
[config.py](config.py)).

## Tests

```bash
python3.12 -m unittest discover -s tests
```

90 test modules. `tests/test_dataset_resolver.py`,
`tests/test_paperorchestra_briefing.py` and `tests/test_vnext_manuscript.py`
fail to collect on some machines for a pre-existing `scripts` import issue.

The full suite carries a stable set of 74 pre-existing failures, recorded
identically before and after the recovery and frontend workstreams
(`docs/runbooks/RECOVERY_2026-08-03.md`,
`docs/frontend/FRONTEND_MERGE_IMPLEMENTATION.md`; last recorded 2026-08-04).
Within that set, 30 adapted-legacy failures are individually classified as
audited obsolete or replaced contracts in
[LEGACY_TEST_CLASSIFICATION.md](docs/integration/archive/LEGACY_TEST_CLASSIFICATION.md).
Compatibility shims do not restore grantless, unscoped, password-bearing or
unlimited behavior.

Read-only audits, all currently passing (recorded 2026-08-03):

```bash
python3.12 scripts/meta_harness_scope_audit.py
python3.12 scripts/meta_harness_sql_audit.py
python3.12 scripts/meta_harness_static_audit.py
python3.12 scripts/meta_harness_state_authority_audit.py
python3.12 scripts/meta_harness_llm_caller_audit.py
```

## Data and security

- Large local artifacts (SQLite databases, WAL files, cached PDFs, logs) are
  excluded by `.gitignore`. No API key is hardcoded; credentials come from the
  environment only.
- SSH credentials and reviewer keys are runtime secret *references*, not stored
  values. SSH execution requires strict known-host pinning.
- Public API responses pass through an `after_request` scrubber that strips a
  denylist of path and log keys and redacts absolute-path substrings in string
  values; the SSE `/api/events` stream is scrubbed the same way
  (`web/app.py:84-106`). Content hashes are public; filesystem paths are not.
- Every parameterless GET route is walked by a leak-guard test asserting no
  `/home/` paths, `log_tail`, `workspace_root` or similar appear in any response
  body (`tests/test_provenance_web.py:282`).

## Packaging

```bash
python3.12 -m pip install build
python3.12 -m build
```

## Changelog

Release history lives in [CHANGELOG.md](CHANGELOG.md), which is maintained
separately. The README no longer carries a duplicate copy.

## Further reading

| Document | What it covers |
|---|---|
| [docs/runbooks/RECOVERY_2026-08-03.md](docs/runbooks/RECOVERY_2026-08-03.md) | Self-heal, topic gate, frontier authority, backend truth, deployment baseline |
| [docs/runbooks/FRONTIER_BOOTSTRAP.md](docs/runbooks/FRONTIER_BOOTSTRAP.md) | Issuing and auditing a bootstrap authority |
| [docs/runbooks/SELFHEAL.md](docs/runbooks/SELFHEAL.md) | Watchdog reason codes and operator response |
| [docs/frontend/FRONTEND_DISCOVERY_AND_RECOMMENDATION.md](docs/frontend/FRONTEND_DISCOVERY_AND_RECOMMENDATION.md) | Why the UI is shaped this way; open product decisions |
| [docs/frontend/FRONTEND_MERGE_IMPLEMENTATION.md](docs/frontend/FRONTEND_MERGE_IMPLEMENTATION.md) | What the frontend merge changed |
| [docs/integration/STATE_DICTIONARY.md](docs/integration/STATE_DICTIONARY.md) | Every state name and what it does and does not mean |
| [docs/integration/archive/UNVERIFIED.md](docs/integration/archive/UNVERIFIED.md) | Claims deliberately not made (2026-08 integration snapshot) |
| [docs/integration/ARCHITECTURE.md](docs/integration/ARCHITECTURE.md) | Control-plane architecture in detail |

## License

MIT
