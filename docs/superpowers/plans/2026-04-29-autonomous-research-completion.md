# Autonomous Research Completion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Complete the existing DeepGraph/SciForge pipeline so a discovered `deep_insight` can be verified, executed as real experiments, reviewed by AI, and exported as a grounded manuscript package without fake interfaces or duplicate workflow layers.

**Architecture:** Reuse the current `deep_insights`, `experiment_runs`, `experiment_iterations`, `experimental_claims`, `discovery_track_record`, and Web API flow. Add only small, focused modules for artifact management, structured planning, manuscript generation, and AI review; surface them through the existing experiment detail and run workflow instead of creating a separate research-project API.

**Tech Stack:** Python 3.12, Flask, SQLite, existing DeepGraph DB helpers, existing LLM client, existing EvoScientist bridge where useful, local filesystem artifacts under each experiment `workdir`, existing vanilla JS dashboard.

---

## Non-Negotiable Constraints

- Do not create placeholder/fake APIs. Every endpoint or function added must be called by an existing workflow or UI action.
- Do not create a parallel `research_projects` system unless the existing tables cannot represent the state. This plan uses `experiment_runs` as the canonical execution object.
- Do not duplicate `/api/experiments/run_full`, `/api/experiments/<run_id>/run`, `/api/deep_insights/<id>/verify`, or `/api/deep_insights/<id>/research`.
- Do not let the LLM invent experimental results, citations, metrics, or review outcomes.
- Manuscript claims must be traceable to one of: stored paper evidence, novelty reports, or experiment artifacts.
- If experiments fail or refute the hypothesis, output a refutation/negative-result report instead of a positive paper.

## Current Components To Reuse

- `agents/experiment_forge.py`: keep as the entry point that turns a `deep_insight` into a runnable `experiment_run`.
- `agents/validation_loop.py`: keep as the executor; improve its metric handling, artifact writes, and state transitions.
- `agents/result_interpreter.py`: keep as post-run interpretation; expand it to consume structured artifacts.
- `agents/knowledge_loop.py`: keep as graph feedback after completed runs.
- `agents/meta_learner.py`: keep as hit-rate/meta analysis after completed runs.
- `agents/novelty_verifier.py`: keep the existing novelty-check workflow for `deep_insights`.
- `web/app.py`: keep existing experiment routes; extend existing responses and add only two action routes where there is no current equivalent.
- `web/static/js/app.js`: keep the existing Experiments tab and Deep Insights buttons.

## Minimal New Components

- Create `agents/artifact_manager.py`: single responsibility for workdir paths, manifest reads/writes, checksums, and artifact listing.
- Create `agents/experiment_plan_compiler.py`: converts a `deep_insight` plus forged scaffold into a validated `execution_plan.json`.
- Create `agents/manuscript_writer.py`: writes `paper.md`, `references.bib`, and `reproducibility.md` from DB evidence and run artifacts.
- Create `agents/ai_reviewer.py`: produces structured AI review JSON and a readable review report for a generated manuscript.
- Create tests for each new module and focused regression tests for modified workflows.

## API Policy

Reuse existing endpoints:

- `POST /api/deep_insights/<id>/verify`
- `GET /api/deep_insights/<id>/verify_status`
- `POST /api/experiments/forge`
- `POST /api/experiments/<run_id>/run`
- `POST /api/experiments/run_full`
- `GET /api/experiments`
- `GET /api/experiments/<run_id>`

Add only these routes, because no existing route performs these actions:

- `POST /api/experiments/<run_id>/manuscript`: generate manuscript package from a completed/refuted run.
- `POST /api/experiments/<run_id>/review`: review the generated manuscript package.

Do not add separate project CRUD endpoints. Artifact listing should be added to `GET /api/experiments/<run_id>` because that route already owns experiment detail display.

---

### Task 1: Artifact Manager

**Files:**
- Create: `agents/artifact_manager.py`
- Test: `tests/test_artifact_manager.py`

- [x] **Step 1: Define artifact manifest format**

Use a file named `artifact_manifest.json` inside each experiment `workdir`:

```json
{
  "schema_version": 1,
  "run_id": 1,
  "artifacts": [
    {
      "type": "metrics",
      "path": "artifacts/metrics.json",
      "sha256": "hex",
      "created_at": "2026-04-29T00:00:00Z",
      "metadata": {}
    }
  ]
}
```

- [x] **Step 2: Write tests first**

Create tests that verify:

- `ensure_artifact_dirs(workdir)` creates `artifacts/logs`, `artifacts/results`, `artifacts/figures`, `artifacts/tables`, `artifacts/manuscript`, and `artifacts/reviews`.
- `record_artifact(...)` stores relative paths only.
- `list_artifacts(...)` returns an empty list for a missing manifest.
- Checksums change when file content changes.

- [x] **Step 3: Implement `artifact_manager.py`**

Public functions:

```python
def ensure_artifact_dirs(workdir: Path) -> dict[str, Path]: ...
def artifact_path(workdir: Path, relative_path: str) -> Path: ...
def record_artifact(workdir: Path, run_id: int, artifact_type: str, path: Path, metadata: dict | None = None) -> dict: ...
def list_artifacts(workdir: Path) -> list[dict]: ...
def read_text_artifact(workdir: Path, relative_path: str, max_chars: int = 20000) -> str: ...
```

Reject paths that resolve outside `workdir`.

- [x] **Step 4: Verify**

Run:

```bash
python3.12 -m unittest tests.test_artifact_manager
```

Expected: all tests pass.

---

### Task 2: Compile Structured Execution Plans

**Files:**
- Create: `agents/experiment_plan_compiler.py`
- Modify: `agents/experiment_forge.py`
- Test: `tests/test_experiment_plan_compiler.py`

- [x] **Step 1: Write tests first**

Cover:

- Tier 2 `deep_insight` with `experimental_plan` JSON produces `execution_plan.json`.
- Missing `experimental_plan` falls back to scaffold `success_criteria.json`.
- Invalid LLM output is rejected and replaced with a deterministic minimal plan.
- The compiler does not call any new API endpoint.

- [x] **Step 2: Implement compiler**

Public function:

```python
def compile_execution_plan(insight: dict, workdir: Path, codebase: dict, scaffold: dict) -> dict:
    """Create and persist execution_plan.json for a forged experiment."""
```

Required output keys:

```json
{
  "schema_version": 1,
  "hypothesis": "...",
  "primary_metric": "...",
  "metric_direction": "higher",
  "stages": [
    {"name": "reproduction", "required": true},
    {"name": "hypothesis_test", "required": true},
    {"name": "ablation", "required": false}
  ],
  "success_criteria": {}
}
```

- [x] **Step 3: Integrate with `forge_experiment`**

After `generate_scaffold(...)` and before inserting `experiment_runs`, call `compile_execution_plan(...)`.

Record the generated `execution_plan.json` through `artifact_manager.record_artifact(...)`.

- [x] **Step 4: Verify**

Run:

```bash
python3.12 -m unittest tests.test_experiment_plan_compiler
python3.12 -m unittest tests.test_artifact_manager
```

Expected: all tests pass.

---

### Task 3: Make Validation Loop Produce Auditable Artifacts

**Files:**
- Modify: `agents/validation_loop.py`
- Test: `tests/test_validation_loop_artifacts.py`

- [x] **Step 1: Write tests first**

Use a temporary workdir with a tiny `code/train.py` that prints a metric. Verify:

- `run_validation_loop(...)` writes `artifacts/results/metrics.json`.
- Run logs are stored under `artifacts/logs/`.
- Baseline and best metric in DB match `metrics.json`.
- A failed baseline returns `verdict="failed"` and does not call manuscript generation.

- [x] **Step 2: Fix metric parsing**

Replace the current dummy `Path("/dev/stdin")` evaluation path with direct parsing of `eval_result.stdout`.

Accepted metric output:

```text
metric_value: 0.812
```

Also accept:

```json
{"metric_value": 0.812}
```

- [x] **Step 3: Write structured result artifacts**

At the end of the loop, write:

```text
artifacts/results/metrics.json
artifacts/results/iterations.json
artifacts/logs/run.log
```

Record each file through `artifact_manager.record_artifact(...)`.

- [x] **Step 4: Keep existing DB behavior**

Do not replace `experiment_runs` or `experiment_iterations`. The JSON artifacts supplement the DB; they do not become a second source of truth.

- [x] **Step 5: Verify**

Run:

```bash
python3.12 -m unittest tests.test_validation_loop_artifacts
```

Expected: all tests pass.

---

### Task 4: Strengthen Result Interpretation And Knowledge Feedback

**Files:**
- Modify: `agents/result_interpreter.py`
- Modify: `agents/knowledge_loop.py`
- Test: `tests/test_result_interpreter.py`

- [x] **Step 1: Write tests first**

Cover:

- Confirmed run creates at least one `experimental_claim`.
- Refuted run creates a claim with `verdict="refuted"`.
- Missing metrics produces `verdict="inconclusive"` and a clear reason.
- `process_completed_run(...)` remains idempotent.

- [x] **Step 2: Interpret from DB and artifacts**

`interpret_run(run_id)` should read:

- `experiment_runs`
- `experiment_iterations`
- `artifacts/results/metrics.json` when present

If DB and artifact values disagree, return an error status and do not cascade.

- [x] **Step 3: Preserve existing knowledge loop**

Keep `process_completed_run(run_id)` as the single post-run entry point. Do not add a parallel feedback function.

- [x] **Step 4: Verify**

Run:

```bash
python3.12 -m unittest tests.test_result_interpreter
```

Expected: all tests pass.

---

### Task 5: Manuscript Writer

**Files:**
- Create: `agents/manuscript_writer.py`
- Modify: `web/app.py`
- Test: `tests/test_manuscript_writer.py`

- [x] **Step 1: Write tests first**

Use a fake completed `experiment_run`, fake `deep_insight`, and fake paper evidence. Verify:

- `generate_manuscript(run_id)` refuses runs without a terminal verdict.
- Confirmed runs produce `paper.md`, `references.bib`, and `reproducibility.md`.
- Refuted runs produce `negative_result_report.md` instead of a positive paper.
- Generated manuscript contains no citation key absent from `references.bib`.
- The function records manuscript files in the artifact manifest.

- [x] **Step 2: Implement writer**

Public function:

```python
def generate_manuscript(run_id: int) -> dict:
    """Generate a grounded manuscript package for a completed experiment run."""
```

Inputs:

- `deep_insights` row
- `experiment_runs` row
- `experimental_claims`
- `experiment_iterations`
- artifact files from the run `workdir`
- related papers from existing DB fields only

Outputs under:

```text
artifacts/manuscript/paper.md
artifacts/manuscript/references.bib
artifacts/manuscript/reproducibility.md
```

For refuted runs:

```text
artifacts/manuscript/negative_result_report.md
artifacts/manuscript/reproducibility.md
```

- [x] **Step 3: Add minimal API action**

Add:

```python
@app.route("/api/experiments/<int:run_id>/manuscript", methods=["POST"])
```

This route calls only `generate_manuscript(run_id)`.

Do not add a manuscript list route. Extend `api_experiment_detail(run_id)` to include `artifacts: list_artifacts(workdir)`.

- [x] **Step 4: Verify**

Run:

```bash
python3.12 -m unittest tests.test_manuscript_writer
```

Expected: all tests pass.

---

### Task 6: AI Reviewer

**Files:**
- Create: `agents/ai_reviewer.py`
- Modify: `web/app.py`
- Test: `tests/test_ai_reviewer.py`

- [x] **Step 1: Write tests first**

Use a fake manuscript artifact and a fake LLM response. Verify:

- Review JSON validates against the expected schema.
- Review report is written to `artifacts/reviews/review.md`.
- Review JSON is written to `artifacts/reviews/review.json`.
- Missing manuscript returns an error and does not call the LLM.

- [x] **Step 2: Implement reviewer**

Public function:

```python
def review_manuscript(run_id: int) -> dict:
    """Run structured AI review over the generated manuscript package."""
```

Required review fields:

```json
{
  "overall_score": 1,
  "recommendation": "reject",
  "major_concerns": [],
  "minor_concerns": [],
  "required_experiments": [],
  "citation_risks": [],
  "reproducibility_risks": []
}
```

The reviewer may critique and request experiments. It must not edit metrics or invent new results.

- [x] **Step 3: Add minimal API action**

Add:

```python
@app.route("/api/experiments/<int:run_id>/review", methods=["POST"])
```

This route calls only `review_manuscript(run_id)`.

- [x] **Step 4: Verify**

Run:

```bash
python3.12 -m unittest tests.test_ai_reviewer
```

Expected: all tests pass.

---

### Task 7: Wire Existing Run-Full Flow To Optional Manuscript/Review

**Files:**
- Modify: `web/app.py`
- Modify: `web/static/js/app.js`
- Test: `tests/test_web_experiment_routes.py`

- [x] **Step 1: Write route tests first**

Cover:

- `GET /api/experiments/<run_id>` includes artifacts.
- `POST /api/experiments/<run_id>/manuscript` returns 404 for missing run.
- `POST /api/experiments/<run_id>/review` refuses review before manuscript exists.

- [x] **Step 2: Keep execution route unchanged by default**

`POST /api/experiments/run_full` remains:

```text
forge → validation_loop → knowledge_loop
```

Do not automatically generate paper/review unless the request JSON contains:

```json
{"generate_manuscript": true, "review": true}
```

This avoids surprising long-running LLM calls.

- [x] **Step 3: Extend dashboard buttons**

In the existing Experiments tab:

- Show artifacts returned by `GET /api/experiments/<run_id>`.
- Add `Generate Manuscript` button for completed/refuted runs.
- Add `AI Review` button only when a manuscript artifact exists.

Do not add a new top-level dashboard tab.

- [x] **Step 4: Verify**

Run:

```bash
python3.12 -m unittest tests.test_web_experiment_routes
```

Expected: all tests pass.

---

### Task 8: End-To-End Toy Pipeline Test

**Files:**
- Create: `tests/test_e2e_toy_research_pipeline.py`
- Modify: test helpers only if needed

- [x] **Step 1: Build a no-network fake LLM fixture**

Patch `agents.llm_client.call_llm` and `agents.llm_client.call_llm_json` so the test never touches remote APIs.

- [x] **Step 2: Create toy data**

Insert:

- one `deep_insight`
- one tiny forged workdir
- one `train.py` that prints a metric
- one `success_criteria.json`

- [x] **Step 3: Run flow**

Call:

```python
forge_experiment(insight_id)
run_validation_loop(run_id)
process_completed_run(run_id)
generate_manuscript(run_id)
review_manuscript(run_id)
```

- [x] **Step 4: Assert outputs**

Verify:

- `experiment_runs.status == "completed"`
- `artifacts/results/metrics.json` exists
- manuscript or negative-result report exists
- review JSON exists
- no generated citation key is missing from `references.bib`

- [x] **Step 5: Verify**

Run:

```bash
python3.12 -m unittest tests.test_e2e_toy_research_pipeline
python3.12 -m unittest discover -s tests
```

Expected: all tests pass.

---

### Task 9: Documentation Update

**Files:**
- Modify: `README.md`
- Modify: `HANDOFF.md`

- [x] **Step 1: Update README status**

Clarify that the completed workflow is:

```text
deep_insight → forge → validation → knowledge_loop → manuscript → AI review
```

Also clarify that real paper quality depends on real experimental success and human review.

- [x] **Step 2: Update handoff**

Replace the old “unfinished final step” with the new completed/remaining state:

- Completed: artifact manifest, structured execution plan, manuscript generation, AI review.
- Remaining optional work: SLURM backend, richer LaTeX templates, multi-round rebuttal automation.

- [x] **Step 3: Verify docs**

Run:

```bash
python3.12 -m unittest discover -s tests
```

Expected: all tests pass after docs changes.

---

## Implementation Notes

### Why This Plan Does Not Add `research_projects`

The existing `experiment_runs` table already represents the unit that moves from scaffolded experiment to validation result. Adding `research_projects` now would create two competing workflow objects. This plan keeps `experiment_runs` canonical and uses artifacts under `workdir` for paper/review outputs.

### Why Manuscript And Review Need Two New Routes

The existing routes can forge and run experiments, but none explicitly generate a manuscript or review one. Two action routes are enough; listing and display belong in the existing experiment detail endpoint.

### Failure Behavior

If baseline reproduction fails, the workflow stops with a failed experiment and writes diagnostic artifacts.

If the proposed method does not beat baseline, the workflow writes a negative-result report and may still run AI review on that report.

If citation grounding fails, manuscript generation returns an error and writes no final paper.

### Local Versus Remote LLM

The system continues to use the existing remote LLM client for semantic tasks. Tests must mock LLM calls. No new local model runtime is introduced.

## Final Verification Command

Run:

```bash
python3.12 -m unittest discover -s tests
```

Expected: all tests pass.

If Python 3.12 is not available on the current machine, use the repository's configured Python interpreter and record the exact command/output in the final handoff.

---

## Live Verification Notes - 2026-04-29

- [x] **Browser/API smoke test**

Restarted the local Flask service on `http://127.0.0.1:8080` and verified the dashboard loads.

- [x] **Failed experiment manuscript behavior**

Run `#1` is a terminal failed run: baseline reproduction produced no metric. Manuscript generation now treats `status='failed'` as terminal and writes `artifacts/manuscript/negative_result_report.md` instead of returning `run_not_terminal`.

- [x] **AI review on negative result**

AI review successfully ran against the negative result report and wrote `artifacts/reviews/review.json` plus `artifacts/reviews/review.md`. The reviewer correctly rejected the report because the experiment did not obtain a metric and therefore cannot support publishable claims.

- [x] **Regression coverage**

Added a regression test for failed runs without `hypothesis_verdict`. Verification command:

```bash
.\.venv\Scripts\python.exe -m unittest discover -s tests
```

Observed result: `Ran 54 tests ... OK`.

- [x] **Validation entrypoint correction**

Investigated why the dashboard showed reproduction iterations with `status='crash'`. The failed run's `program.md` specified a runnable command under `code/`, but the validation loop ignored that scaffold command and attempted `code/train.py`. The loop now reuses the existing `program.md` run command before falling back to `train.py`.

Verification command:

```bash
.\.venv\Scripts\python.exe -m unittest discover -s tests
```

Observed result after the correction: `Ran 55 tests ... OK`.

Remaining environment blocker for the existing Fairlearn run: after the entrypoint fix, direct execution reaches the selected Fairlearn script but fails because the virtual environment does not currently have `numpy` installed. This is an experiment environment dependency issue, not an evidence-volume issue.

- [x] **Execution contract audit**

Additional audit found and fixed several bottom-layer issues:

- The validation loop no longer treats a legacy evaluator fallback of `metric_value: 0.0` as a real baseline when the run log contains no metric.
- If `evaluate.py` cannot compute a metric but the run log already contains the configured metric, the log metric is preserved instead of being overwritten with `None`.
- Fallback scaffold generation no longer writes evaluators that invent `0.0` on failure.
- Scratch or empty scaffolds now receive a transparent bootstrap `train.py` instead of only logging a warning and later crashing.
- `confirmed` now requires a real kept improvement during hypothesis testing; baseline-only runs are `inconclusive`.
- Knowledge-loop cascade now skips unknown taxonomy node ids instead of crashing on a foreign-key failure.

Verification command:

```bash
.\.venv\Scripts\python.exe -m unittest discover -s tests
```

Observed result after these fixes: `Ran 62 tests ... OK`.

- [x] **Local successful run**

Created a synthetic, clearly labeled SciForge healthcheck run in the local database to verify the corrected end-to-end plumbing without relying on external datasets or package installs.

Observed result:

- Run `#3` completed with `hypothesis_verdict='confirmed'`.
- Baseline `accuracy=0.8`; best `accuracy=0.9`.
- Hypothesis-testing iterations: `1`; kept improvements: `1`.
- Manuscript artifacts and AI review artifacts were written under `sciforge_runs/exp_5_SciForge_synthetic_improvement_healthche`.
- Browser/API verification confirmed run `#3` displays as completed/confirmed with metrics, kept iteration, manuscript, reproducibility, and review artifacts.

Important interpretation: this is a plumbing healthcheck, not a publishable scientific result. The AI review artifact explicitly marks it as synthetic and requires replacement with a real benchmark before any scientific claim.

- [x] **Fairlearn run diagnosis**

Run `#1` did not fail because evidence/data in DeepGraph was too small. It failed because:

- The LLM-selected scaffold used a Fairlearn library module as the execution entrypoint rather than a benchmark/training script.
- The original validation loop ignored the scaffold command and tried `code/train.py`.
- After fixing entrypoint selection, direct Fairlearn execution reaches dependency import and fails because the current `.venv` lacks `numpy`, `pandas`, `scikit-learn`, `scipy`, and `narwhals`.

To continue that specific Fairlearn experiment honestly, install the Fairlearn runtime dependencies and replace the library-file scaffold with a real benchmark harness. Do not mark it successful until it obtains a real metric and keeps an actual improvement.

- [x] **Fairlearn proxy rerun and reviewer gate**

Installed the Fairlearn runtime dependencies in `.venv`: `numpy`, `pandas`, `scikit-learn`, `scipy`, and `narwhals`.

Fixed additional bottom-layer issues found during the real rerun:

- Non-scratch library checkouts with no runnable `train.py` now receive a transparent bootstrap harness instead of attempting to execute a library module.
- Fairlearn fallback scaffolds now use a small deterministic Fairlearn/sklearn proxy benchmark that reports `accuracy`, `demographic_parity_gap`, and `fairness_score`.
- Tier 1 insight context passed to the coding agent now includes `formal_structure`, `transformation`, and the experimental plan, not only `proposed_method`.
- Existing cloned git repositories now commit the generated scaffold as a local baseline before hypothesis testing, so `git reset` cannot delete `train.py`.
- Manuscript generation now includes metric definition, artifact paths, and a per-iteration audit trail, and labels confirmed runs as proxy-supported rather than submission-ready scientific confirmation.

Observed run:

- Run `#5` completed with `hypothesis_verdict='confirmed'`.
- Workdir: `sciforge_runs/exp_7_Fairlearn_preference_cone_proxy`.
- Primary metric: `fairness_score = accuracy - 0.45 * demographic_parity_gap`.
- Baseline `fairness_score=0.5007863219992292`; best `fairness_score=0.6157093943616044`.
- Effect: `+0.1149230723623752` / `+22.95%`.
- Hypothesis-testing iterations: `3`; kept improvements: `3`.
- Manuscript artifacts were generated under `artifacts/manuscript/`.
- AI review artifacts were generated under `artifacts/reviews/`.

Reviewer result:

- `overall_score=1`
- `recommendation='reject'`

Interpretation: the autonomous engine can now run a real local proxy benchmark, keep metric-improving code changes, generate a manuscript package, and run AI review. It still cannot honestly output a publishable paper for this insight because the evidence is only a small Fairlearn proxy benchmark; it does not prove the safe-RL/CMDP/social-choice claim, lacks multi-seed statistics, real constrained-RL environments, stronger baselines, and proper citation synthesis.

Verification command:

```bash
.\.venv\Scripts\python.exe -m unittest discover -s tests
```

Observed result after these fixes: `Ran 67 tests ... OK`.
