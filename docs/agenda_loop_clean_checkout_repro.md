# Clean-checkout reproduction — agenda-driven research loop (issue #9)

This document is the human-readable counterpart to
`artifacts/agenda_loop_acceptance.json`. Any reviewer can follow the five
steps below on a clean machine and arrive at the same selection / experiment
/ gate / manuscript / review / revision-plan artifacts that the JSON refers
to. The end-to-end run takes about 15 seconds on a 2020 MacBook Air.

## 1. Install

```bash
git clone https://github.com/hitome0123/Deepgraph.git
cd Deepgraph
git checkout feat/issue-9-agenda-driven-research-loop
python3.12 -m venv .venv
. .venv/bin/activate
pip install -e .
```

The agenda loop only requires the runtime dependencies declared in
`pyproject.toml` (`flask`, `httpx`, `numpy`, `pydantic`, `pymupdf`,
`waitress`). No GPU or remote service is needed.

## 2. Initialise the database

```bash
DEEPGRAPH_DATABASE_URL="" DEEPGRAPH_DB_PATH=/tmp/agenda_loop_acceptance.db \
  python -c "from db import database as db; db.init_db()"
```

`init_db()` automatically applies `db/schema_agenda.sql` (and the Postgres
counterpart when a `DEEPGRAPH_DATABASE_URL` is set), including the new
`agenda_evidence_gates` table.

## 3. Run the targeted tests

```bash
pytest tests/test_agenda_contract.py tests/test_agenda_selector.py \
       tests/test_agenda_orchestrator.py tests/test_agenda_review_loop.py \
       tests/test_agenda_routes.py tests/test_evidence_gate.py \
       tests/test_evidence_gate_routes.py -q
```

Expected: `60 passed`. This is the focused suite that exercises the issue #9
surface end-to-end.

### Full repository suite

`pytest -q` reports `272 passed, 3 failed` on a clean checkout. The three
failures are pre-existing on `origin/main` and unrelated to the agenda loop:

| Test | Pre-existing? | Notes |
|---|---|---|
| `tests/test_evidence_graph.py::EvidenceGraphSummaryTests::test_merge_candidate_context_helpers_exist` | yes | evidence-graph helper signature drift |
| `tests/test_parallel_orchestration.py::AutoResearchSchedulingTests::test_process_candidate_blocks_underspecified_verification` | yes | auto_research scheduling assertion drift |
| `tests/test_validation_loop_metrics.py::ValidationMetricParsingTests::test_validation_benchmark_env_preserves_paper_grade_contract_budget` | yes | benchmark env default budget drift (`64` vs `128`) |

These are also enumerated under
`clean_checkout_repro.known_baseline_failures` in
`artifacts/agenda_loop_acceptance.json`.

## 4. Run the demonstration build

```bash
DEEPGRAPH_DATABASE_URL="" DEEPGRAPH_DB_PATH=/tmp/agenda_loop_acceptance.db \
  python -m scripts.build_agenda_loop_acceptance
```

What this does:

1. wipes `/tmp/agenda_loop_acceptance.db` and `/tmp/dg_agenda_real_exp/`
2. seeds an 8-insight demo set via `scripts.seed_agenda_demo`
3. activates `research_agendas/token_scale_v1.yaml`
4. runs the selector (`agents.agenda_selector.select_and_persist`)
5. runs the real CPU micro-benchmark
   (`agents.real_experiment_runner.run_real_experiment_for_selection`,
   softmax vs linear attention on the committed
   `agents/benchmarks/qkv_fixture_512_64.npz` fixture, seed=1729, repeats=3)
6. evaluates the evidence gate
   (`agents.evidence_gate.run_gate`, rule_set `agenda_v1_default`)
7. iff `gate.status == pass`, creates `manuscript_runs` +
   `submission_bundles` rows
8. runs the internal reviewer
   (`agents.reviewer_adapter.run_review`,
   reviewer id `internal_evidence_gate`)
9. derives the revision plan
   (`agents.revision_planner.build_revision_plan`)
10. issues three HTTP GETs through Flask's `test_client` (no live server
    required) to confirm the loop is observable from the dashboard API:
    - `GET /api/research_agenda/current`
    - `GET /api/research_agenda/selection/latest`
    - `GET /api/research_agenda/loop/<selection_id>`
11. computes SHA-256 of `experiment_result_packet.json` and writes the
    acceptance bundle to `artifacts/agenda_loop_acceptance.json`, together
    with `artifacts/review_<id>.json` and
    `artifacts/revision_plan_<id>.json`.

## 5. Inspect the loop

After step 4 you can either re-issue any of the API calls via the test
client or boot the live server:

```bash
DEEPGRAPH_DATABASE_URL="" DEEPGRAPH_DB_PATH=/tmp/agenda_loop_acceptance.db \
  python -m web.app
```

then in a second terminal:

```bash
curl http://127.0.0.1:8080/api/research_agenda/current
curl http://127.0.0.1:8080/api/research_agenda/selection/latest
curl http://127.0.0.1:8080/api/research_agenda/loop/1
```

Each endpoint must return HTTP 200 with a JSON body. The same status codes
are recorded under `api_evidence` in the acceptance bundle.

## Acceptance bundle reference

`artifacts/agenda_loop_acceptance.json` is the authoritative machine-readable
record for AI verification:

```text
commit                       — git HEAD at generation time
clean_checkout_repro         — exactly the steps above
agenda_config_path           — research_agendas/token_scale_v1.yaml
active_agenda                — id + name + version
selection.selection_id       — primary key in agenda_selections
selection.rejected_count     — non-zero; selector recorded reasons
experiment.run_id            — primary key in experiment_runs
experiment.result_packet_sha256 — sha256 of experiment_result_packet.json
evidence_gate.status         — pass | block
manuscript.bundle_id         — only present when gate=pass
review.review_id             — primary key in agenda_reviews
revision_plan.plan_id        — primary key in agenda_revision_plans
api_evidence                 — three status codes (must all be 200)
```

Re-running step 4 deterministically reproduces every field, with the
exception of timestamps and DB-assigned row ids.
