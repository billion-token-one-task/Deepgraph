# PR Summary: SciForge Autonomous Research Pipeline And Safe RL/CMDP Review-Pass

Date: 2026-04-30

## Executive Summary

This work turns the project from a mostly discovery/experiment scaffold into a reusable SciForge research pipeline that can:

1. Convert a `deep_insights` discovery into a structured research spec.
2. Select an implemented benchmark capability instead of inventing fake experiment code.
3. Run a benchmark-suite experiment with multi-dataset, multi-seed evidence.
4. Generate statistical reports, reproducibility artifacts, manuscript drafts, and AI-review artifacts.
5. Feed AI-review failures back into follow-up plans.
6. Surface the resulting runs and insights through the existing Flask UI/API.

The final demonstrated result is run `25`, based on the RL/CMDP insight:

> Social-choice CMDPs: safe RL and group fairness as the same constrained preference-cone optimization

The final output is not a broad new safe-RL algorithm paper. It is a scoped finite-CMDP benchmark/reporting artifact note that passed AI review as `weak_accept` under that limited claim.

## Final Demonstrated Result

Latest successful run:

```text
run_id = 25
status = completed
phase = benchmark_suite
hypothesis_verdict = confirmed
AI review recommendation = weak_accept
AI review score = 6
evidence_gate.manuscript_status = paper_ready_candidate
```

Run workdir:

```text
E:\Download\Softwares\Deepgraph\sciforge_runs\exp_1_Social-choice_CMDPs__safe_RL_and_group_f_fa050b4642
```

Key artifacts:

```text
benchmark_config.json
artifacts/manuscript/paper_candidate.md
artifacts/reviews/review.json
artifacts/reviews/review.md
artifacts/results/evidence_gate.json
artifacts/results/statistical_report.json
artifacts/results/benchmark_results.json
artifacts/results/lp_validation.json
artifacts/results/cmdp_environment_appendix.json
artifacts/results/reproduction_check.json
artifacts/results/reproduction_manifest.json
```

Measured run summary:

```text
baseline safe_return = -2.1421053565841697
configured candidate safe_return = 1.402331621275302
signed aggregate difference = 3.5444369778594718
descriptive percent change = 165.46510968589703
```

The final reviewer caveat is important: the accepted claim is an artifact/reporting claim for small finite CMDPs, not a deployable safe-RL method and not a universal paper generator.

## Main Design Principles Followed

- Reused the existing `deep_insights`, `experiment_runs`, artifact folders, and web routes.
- Avoided fake interfaces and fake experiment results.
- Avoided adding redundant public APIs where existing routes could be extended.
- Kept benchmark logic isolated under `benchmarks/`.
- Kept manuscript, review, evidence-gate, and follow-up logic as separate small modules.
- Did not weaken the AI reviewer to force a pass.
- Used review failures to identify root causes before making changes.
- Preserved old legacy `insights` support while adding normalized `deep_insights` support.

## New Pipeline Architecture

The reusable path is:

```text
deep_insights
  -> research_spec_compiler
  -> capability_registry
  -> experiment_forge
  -> benchmark_suite
  -> statistical_reporter
  -> evidence_gate
  -> manuscript_writer
  -> ai_reviewer
  -> review_planner
  -> knowledge_loop/result_interpreter
  -> web UI/API
```

The important architectural choice is that `experiment_forge` can now select a real implemented capability, such as `safe_rl_cmdp`, and enter `benchmark_suite` mode. That avoids generating throwaway fake `train.py` scripts for domains where a reusable benchmark harness exists.

## Major New Modules

### Research And Experiment Planning

- `agents/research_spec_compiler.py`
  - Converts a `deep_insights` row into a structured research spec.
  - Detects safe-RL/CMDP tasks and routes them toward `safe_rl_cmdp`.

- `agents/experiment_plan_compiler.py`
  - Converts insight context into an execution plan.
  - Keeps plan metadata separate from benchmark execution.

- `agents/capability_registry.py`
  - Registers implemented capabilities.
  - Adds `safe_rl_cmdp` with default datasets, methods, seeds, metric, baselines, reference method, and ablations.
  - Keeps `generic_python_benchmark` as fallback.

### Benchmark Execution

- `agents/benchmark_suite.py`
  - Loads `benchmark_config.json`.
  - Dispatches to the selected benchmark harness.
  - Runs main rows plus ablation rows.
  - Writes `artifacts/results/benchmark_results.json`.
  - Calls safe-RL reproducibility artifact generation when capability is `safe_rl_cmdp`.

- `benchmarks/safe_rl_cmdp/envs.py`
  - Defines `FiniteCMDP`.
  - Adds six named finite CMDPs:
    - `risky_shortcut`
    - `delayed_safety`
    - `resource_gathering`
    - `randomized_bandit`
    - `stochastic_bridge`
    - `tight_budget_chain`
  - Adds systematic generated CMDPs:
    - `systematic_chain_s3_g70_c30`
    - `systematic_chain_s4_g82_c35`
    - `systematic_random_s4_a3_g78_c40`
    - `systematic_random_s5_a2_g90_c45`
  - These systematic tasks vary state count, action count, discount factor, cost limit, transition stochasticity, and randomized-optimum regimes.

- `benchmarks/safe_rl_cmdp/solvers.py`
  - Implements exact policy evaluation.
  - Implements deterministic policy enumeration.
  - Implements:
    - `reward_only`
    - fixed Lagrangian policies such as `lagrangian_penalty_4.00`
    - `lagrangian_grid_best`
    - `preference_cone_policy`
    - `deterministic_feasible_best`
    - `occupancy_lp_optimal`
  - Labels oracle/reference methods explicitly through `baseline_role` and `selection_protocol`.

- `benchmarks/safe_rl_cmdp/metrics.py`
  - Computes:
    - `reward`
    - `cost`
    - `cost_limit`
    - `constraint_violation`
    - `safe_return`
    - `feasible`
    - `policy_entropy`
    - `support_size`
    - `safety_penalty`

- `benchmarks/safe_rl_cmdp/harness.py`
  - Runs datasets x seeds x methods.
  - Records per-row method metadata, policies, runtime, and metrics.

- `benchmarks/safe_rl_cmdp/artifacts.py`
  - Writes:
    - `cmdp_environment_appendix.json`
    - `lp_validation.json`
    - `reproduction_manifest.json`
  - Adds LP backend cross-checks.
  - Adds analytic one-state randomized CMDP sanity check.
  - Adds LP-vs-deterministic and LP-vs-candidate gap reporting.

### Reporting, Manuscript, Review

- `agents/statistical_reporter.py`
  - Generates `statistical_report.json`.
  - Generates `artifacts/tables/main_results.md`.
  - Adds:
    - aggregate metric summaries,
    - reward/cost/violation summaries,
    - runtime summaries,
    - pairwise comparisons,
    - separated ablation labels such as `safety_penalty=3.00` and `safety_penalty=9.00`.

- `agents/evidence_gate.py`
  - Decides whether a manuscript is:
    - `preliminary`
    - `needs_more_experiments`
    - `not_publishable`
    - `paper_ready_candidate`
  - Blocks paper-ready status when AI review returns reject-like recommendations.
  - Allows local experimental verdict to remain confirmed when the only blocker is manuscript review revision.

- `agents/manuscript_writer.py`
  - Writes:
    - `paper_candidate.md`
    - `additional_experiments_required.md`
    - `preliminary_report.md`
    - `negative_result_report.md`
    - `references.bib`
    - `reproducibility.md`
  - Adds safe-RL/CMDP-specific manuscript sections:
    - implemented methods,
    - CMDP grounding,
    - environment metadata,
    - aggregate method summary,
    - aggregate reward/cost/violation summary,
    - environment-grouped summary,
    - runtime summary,
    - safety-penalty sensitivity summary,
    - LP validation summary,
    - LP randomization gap summary,
    - reproduction check summary,
    - artifact hash summary,
    - explicit reference/candidate framing.
  - Fixes misleading wording:
    - `Best value` became `Configured candidate value`.
    - `effect size` became `signed aggregate difference` where appropriate.
    - `occupancy_lp_optimal` is framed as exact LP reference, not deployable candidate.

- `agents/ai_reviewer.py`
  - Sends manuscript package to configured external LLM reviewer.
  - Validates structured review JSON.
  - Writes:
    - `artifacts/reviews/review.json`
    - `artifacts/reviews/review.md`
  - Fixes long-manuscript truncation by including manuscript head/tail plus compact artifact context.

- `agents/review_planner.py`
  - Converts reviewer-required experiments into `followup_experiment_plan.json`.

- `agents/reproduction_verifier.py`
  - Runs a real local reproduction smoke command.
  - Writes `artifacts/results/reproduction_check.json`.
  - Records command, exit code, duration, stdout/stderr tails, Python executable, repo root, and scope.

## Existing Modules Modified

### Experiment Forge

- `agents/experiment_forge.py`
  - Now compiles research specs and capability configs.
  - Writes `benchmark_config.json`.
  - Can mark a run as `benchmark_suite` mode.
  - Falls back to scratch scaffold when code scouting or LLM scaffold generation is unavailable.
  - Safe-RL path remains usable even when external scout/scaffold LLM calls fail.

### Validation Loop

- `agents/validation_loop.py`
  - Detects `benchmark_config.json` and switches to benchmark-suite execution.
  - Runs benchmark suite, statistical report, and evidence gate.
  - Updates `experiment_runs` with benchmark summary metrics.
  - Preserves `confirmed` benchmark verdict when the only blocker is manuscript review revision.

### Result Interpreter And Knowledge Loop

- `agents/result_interpreter.py`
  - Reads benchmark/evidence artifacts.
  - Creates scoped artifact-reporting claims.
  - Separates configured candidate method from exact reference frontier.
  - Avoids claiming broad scientific confirmation from internal artifact checks.

- `agents/knowledge_loop.py`
  - Processes completed benchmark-suite runs.
  - Cascades confirmed claims through existing knowledge-loop mechanisms.

### Database

- `db/schema.sql`
- `db/database.py`
  - Added/updated schema support for new artifact and run metadata used by the pipeline.
  - Fixed missing-table style failures encountered by web routes.

### Web/API/UI

- `web/app.py`
  - `/api/insights` now returns a unified feed:
    - legacy `insights`
    - normalized `deep_insights`
  - Preserves old endpoint rather than adding a redundant one.
  - Adds/uses experiment manuscript/review/follow-up routes.
  - Fixes `Plan Follow-up` path through existing `review_planner`.

- `web/static/js/app.js`
  - Deep insights now show correct actions:
    - `SciForge Run`
    - `Forge Only`
    - `Deep Research`
  - Avoids sending `deep_insight` IDs into legacy `Launch Research` routes.
  - Experiment detail UI detects manuscript/review/follow-up artifacts.

## Bugs Found And Fixed

### 1. `/api/insights` Empty

Symptom:

```text
insights UI was empty
sqlite old table `insights` had 0 rows
real discoveries were in `deep_insights`
```

Root cause:

```text
web/API integration bug: `/api/insights` only queried the legacy `insights` table.
```

Fix:

```text
Keep `/api/insights`, but normalize and merge legacy `insights` plus `deep_insights`.
```

Verification:

```text
GET /api/insights?limit=10 -> 5 normalized deep_insights
```

### 2. `Plan Follow-up` Returned 404

Root cause:

```text
Stale Flask process was still serving old code on 127.0.0.1:8080.
```

Fix:

```text
Restarted current workspace Flask process.
Route tests already showed the route existed.
```

### 3. AI Reviewer Saw Truncated Manuscript

Root cause:

```text
ai_reviewer sent only the first 50,000 characters.
The Limitations and secondary metrics were near/after that boundary.
Reviewer saw an incomplete paper.
```

Fix:

```text
Reviewer prompt now includes manuscript head/tail and compact artifact context.
```

### 4. LP Reference vs Candidate Confusion

Root cause:

```text
Manuscript and claim text used "best configured method" for lagrangian_penalty_4.00,
while occupancy_lp_optimal was also configured and had higher safe_return.
```

Fix:

```text
Use "configured candidate method" for lagrangian_penalty_4.00.
Report occupancy_lp_optimal separately as exact LP reference frontier.
```

### 5. Review Revision Caused Artifact Inconsistency

Root cause:

```text
After a weak_reject, a revised paper_candidate.md could be generated while
evidence_gate.json still said needs_more_experiments.
Reviewer correctly flagged the inconsistency.
```

Fix:

```text
When the only blocker is review_requires_revision and benchmark evidence is otherwise complete,
manuscript_writer writes a consistent revised paper_ready_candidate evidence gate.
```

### 6. Randomization Notes Were Static

Root cause:

```text
Manuscript hard-coded whether randomization was expected, but LP validation showed
actual LP-vs-deterministic gaps for some environments.
```

Fix:

```text
Environment randomization notes are derived from lp_validation.json.
```

### 7. Safety-Penalty Ablation Was Pooled

Root cause:

```text
safety_penalty=3 and safety_penalty=9 ablations shared the same label and appeared pooled.
```

Fix:

```text
Benchmark suite labels ablations with explicit safety_penalty values.
Statistical report and manuscript show them separately.
```

## Test Coverage Added

New or expanded tests include:

```text
tests/test_ai_reviewer.py
tests/test_artifact_manager.py
tests/test_benchmark_suite.py
tests/test_capability_registry.py
tests/test_database_schema.py
tests/test_e2e_toy_research_pipeline.py
tests/test_evidence_gate.py
tests/test_experiment_forge_plan_integration.py
tests/test_experiment_plan_compiler.py
tests/test_fairness_benchmark_harness.py
tests/test_knowledge_loop.py
tests/test_manuscript_writer.py
tests/test_reproduction_verifier.py
tests/test_research_spec_compiler.py
tests/test_result_interpreter.py
tests/test_review_planner.py
tests/test_safe_rl_cmdp_benchmark.py
tests/test_statistical_reporter.py
tests/test_validation_loop_artifacts.py
tests/test_web_experiment_routes.py
```

Final full-suite verification:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests
```

Result:

```text
Ran 155 tests in 13.810s
OK
```

## Browser/API Verification

After restarting Flask from the workspace:

```text
GET /api/insights?limit=5
  -> 5 rows
  -> first row source_table=deep_insights
  -> title=Social-choice CMDPs...

GET /api/experiments
  -> 25 rows
  -> latest run_id=25
  -> status=completed
  -> hypothesis_verdict=confirmed

GET /api/experiments/25
  -> artifacts include artifacts/reviews/review.json
  -> artifacts include artifacts/results/evidence_gate.json
```

The in-app browser DOM also contained:

```text
Social-choice CMDPs...
Experiments navigation
```

## What The Generated Paper Is

The final `paper_candidate.md` is best described as:

```text
A scoped finite-CMDP benchmark/reporting artifact note.
```

It demonstrates that the pipeline can generate and review a grounded research artifact with:

- real benchmark rows,
- multi-seed, multi-dataset evidence,
- LP references,
- diagnostic baselines,
- statistical summaries,
- reproducibility metadata,
- manuscript text,
- AI review,
- follow-up planning.

It is not:

- a new safe-RL algorithm,
- a proof of the original broad social-choice CMDP thesis,
- a universal autonomous paper generator,
- a deployable penalty-selection method,
- a clean-container reproducibility guarantee.

## PR Readiness Notes

Recommended PR strategy:

1. Core artifact and pipeline modules:
   - `artifact_manager`
   - `research_spec_compiler`
   - `capability_registry`
   - `benchmark_suite`
   - `statistical_reporter`
   - `evidence_gate`
   - `manuscript_writer`
   - `ai_reviewer`
   - `review_planner`
   - `reproduction_verifier`

2. Safe-RL/CMDP benchmark capability:
   - `benchmarks/safe_rl_cmdp/*`
   - safe-RL tests

3. Web/API integration:
   - unified `/api/insights`
   - experiment detail/review/follow-up routes
   - frontend action wiring

4. Documentation and examples:
   - plan docs
   - README/HANDOFF updates
   - this PR summary

5. Generated run artifacts:
   - Do not commit the entire `sciforge_runs/` directory by default.
   - If a reviewer needs an example artifact package, include a small curated sample or attach it outside the code PR.

## Known Follow-Up Work

- Add clean-container or fresh-checkout reproduction with hash matching.
- Add stronger non-oracle constrained-planning baselines if making comparative method claims.
- Add validation/held-out-environment penalty selection if making deployability claims.
- Improve citation hygiene and remove weak KG evidence citations from submission-facing references.
- Add external or independently specified tabular CMDP benchmarks if claiming broader benchmark utility.
- Consider PR-splitting so the safe-RL benchmark does not obscure the general SciForge pipeline changes.
