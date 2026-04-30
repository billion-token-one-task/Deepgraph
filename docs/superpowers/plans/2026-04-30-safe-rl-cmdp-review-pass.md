# Safe RL CMDP Review-Pass Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a real `safe_rl_cmdp` capability and run the existing DeepGraph/SciForge pipeline until it produces a grounded RL/CMDP manuscript that passes AI review as `weak_accept` or `accept`.

**Architecture:** Reuse the existing `experiment_runs -> benchmark_suite -> statistical_reporter -> evidence_gate -> manuscript_writer -> ai_reviewer -> review_planner` pipeline. Add a focused finite-CMDP benchmark package under `benchmarks/safe_rl_cmdp/`, register it as an implemented capability, and keep all outputs as artifacts under the run workdir.

**Tech Stack:** Python standard library, `numpy`, existing SQLite DB helpers, existing artifact manager, existing unittest suite, existing browser verification.

---

## Design Principles

- Do not create a new research-project API or a parallel paper-generation workflow.
- Do not fake RL results, citations, or reviewer recommendations.
- Do not make review pass by weakening the reviewer or editing review output.
- Do not hard-code one insight title into the implementation.
- Keep RL benchmark logic isolated in `benchmarks/safe_rl_cmdp/`.
- Keep pipeline integration generic through `capability_registry` and `benchmark_suite`.
- If a run fails, classify the root cause before modifying code:
  - `bottom_logic_bug`
  - `dependency_or_environment`
  - `llm_prompt_or_scaffold`
  - `data_or_evidence_insufficient`
  - `scientific_hypothesis_unsupported`

## Scope

### In Scope For This Phase

- Finite, tabular safe RL / CMDP benchmark environments.
- Exact occupancy-measure LP solver implemented for small finite CMDPs by enumerating deterministic stationary policies, avoiding new heavy dependencies.
- Lagrangian constrained-policy baseline.
- Preference-cone policy selection method with validation-style selection over constraint penalties.
- Metrics:
  - `reward`
  - `cost`
  - `constraint_violation`
  - `safe_return`
  - `feasible`
  - `policy_entropy`
  - `support_size`
- Multi-environment, multi-seed benchmark suite.
- Statistical report and manuscript sections that explain RL-specific methods, metrics, and limitations.
- Full pipeline run through AI review.

### Out Of Scope For This Phase

- Deep RL, neural policies, MuJoCo, GPU training, or large-scale simulation.
- Claiming to solve the original broad social-choice CMDP/safe-RL theory problem.
- Automatic literature search/citation expansion beyond existing DeepGraph evidence.
- Cross-domain universal paper generation.

## Target Scientific Claim

The RL manuscript must use a narrow scoped claim:

> Validation-selected preference-cone policy selection improves safe-return trade-offs over reward-only and fixed-penalty Lagrangian baselines on small finite CMDP benchmarks, while matching the exact occupancy enumeration frontier within reported tolerance.

This is intentionally modest. The intended publishable unit is a reproducible benchmark/artifact note for finite CMDPs, not a broad safe-RL theorem.

## Files

- Create: `benchmarks/safe_rl_cmdp/__init__.py`
- Create: `benchmarks/safe_rl_cmdp/envs.py`
- Create: `benchmarks/safe_rl_cmdp/solvers.py`
- Create: `benchmarks/safe_rl_cmdp/metrics.py`
- Create: `benchmarks/safe_rl_cmdp/harness.py`
- Create: `tests/test_safe_rl_cmdp_benchmark.py`
- Modify: `agents/capability_registry.py`
- Modify: `agents/benchmark_suite.py`
- Modify: `agents/manuscript_writer.py`
- Modify: `docs/2026-04-29-progress-and-roadmap.md`

## Task 1: Finite CMDP Environments

**Files:**
- Create: `benchmarks/safe_rl_cmdp/envs.py`
- Test: `tests/test_safe_rl_cmdp_benchmark.py`

- [ ] **Step 1: Write failing tests**

Add tests that call:

```python
from benchmarks.safe_rl_cmdp.envs import make_cmdp

env = make_cmdp("risky_shortcut", seed=0)
assert env.name == "risky_shortcut"
assert env.transitions.shape == (env.n_states, env.n_actions, env.n_states)
assert env.rewards.shape == (env.n_states, env.n_actions)
assert env.costs.shape == (env.n_states, env.n_actions)
assert abs(env.start.sum() - 1.0) < 1e-9
assert env.gamma < 1.0
assert env.cost_limit > 0.0
```

- [ ] **Step 2: Verify red**

Run:

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_safe_rl_cmdp_benchmark
```

Expected: import failure because `benchmarks.safe_rl_cmdp` does not exist.

- [ ] **Step 3: Implement environments**

Implement:

```python
@dataclass
class FiniteCMDP:
    name: str
    seed: int
    transitions: np.ndarray
    rewards: np.ndarray
    costs: np.ndarray
    start: np.ndarray
    gamma: float
    cost_limit: float
```

Add at least three deterministic finite CMDPs:

- `risky_shortcut`
- `delayed_safety`
- `resource_gathering`

Each should be small enough for policy enumeration and have a real reward/cost trade-off.

- [ ] **Step 4: Verify green**

Run:

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_safe_rl_cmdp_benchmark
```

Expected: environment tests pass.

## Task 2: Solvers And Metrics

**Files:**
- Create: `benchmarks/safe_rl_cmdp/solvers.py`
- Create: `benchmarks/safe_rl_cmdp/metrics.py`
- Test: `tests/test_safe_rl_cmdp_benchmark.py`

- [ ] **Step 1: Write failing tests**

Add tests for:

- deterministic policy enumeration count equals `n_actions ** n_states`;
- policy evaluation returns finite discounted reward/cost;
- `occupancy_enumeration` returns a feasible policy when one exists;
- reward-only policy can violate the constraint on at least one benchmark;
- preference-cone selection improves `safe_return` over reward-only baseline.

- [ ] **Step 2: Verify red**

Run:

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_safe_rl_cmdp_benchmark
```

Expected: missing solver functions.

- [ ] **Step 3: Implement solvers**

Implement functions:

```python
enumerate_deterministic_policies(env: FiniteCMDP) -> list[np.ndarray]
evaluate_policy(env: FiniteCMDP, policy: np.ndarray) -> dict
reward_only_policy(env: FiniteCMDP) -> dict
lagrangian_policy(env: FiniteCMDP, penalty: float) -> dict
occupancy_enumeration(env: FiniteCMDP) -> dict
preference_cone_policy(env: FiniteCMDP, penalties: list[float]) -> dict
```

Use exact discounted evaluation by solving:

```text
V = r_pi + gamma * P_pi * V
```

and same for costs.

- [ ] **Step 4: Implement metrics**

Compute:

```python
safe_return = reward - safety_penalty * max(0, cost - cost_limit)
constraint_violation = max(0, cost - cost_limit)
feasible = constraint_violation <= tolerance
```

- [ ] **Step 5: Verify green**

Run:

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_safe_rl_cmdp_benchmark
```

Expected: solver and metric tests pass.

## Task 3: Safe RL Harness And Capability Registration

**Files:**
- Create: `benchmarks/safe_rl_cmdp/harness.py`
- Modify: `agents/capability_registry.py`
- Modify: `agents/benchmark_suite.py`
- Test: `tests/test_safe_rl_cmdp_benchmark.py`
- Test: `tests/test_capability_registry.py`
- Test: `tests/test_benchmark_suite.py`

- [ ] **Step 1: Write failing tests**

Add tests verifying:

```python
from benchmarks.safe_rl_cmdp.harness import run_safe_rl_benchmark

payload = run_safe_rl_benchmark({
    "datasets": ["risky_shortcut"],
    "methods": ["reward_only", "lagrangian_penalty_1.00", "preference_cone_policy", "occupancy_enumeration"],
    "seeds": [0, 1],
    "primary_metric": "safe_return",
})
assert payload["capability"] == "safe_rl_cmdp"
assert len(payload["rows"]) == 8
assert {row["status"] for row in payload["rows"]} == {"ok"}
```

Add registry tests:

```python
capability = get_capability("safe_rl_cmdp")
assert capability["implemented"] is True
assert capability["runner"] == "benchmarks.safe_rl_cmdp.harness"
```

- [ ] **Step 2: Verify red**

Run:

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_safe_rl_cmdp_benchmark tests.test_capability_registry tests.test_benchmark_suite
```

Expected: safe RL capability unsupported.

- [ ] **Step 3: Implement harness**

`run_safe_rl_benchmark(config)` should loop over datasets, seeds, and methods, returning rows shaped like fairness benchmark rows:

```python
{
  "dataset": dataset_name,
  "seed": seed,
  "method": method,
  "status": "ok",
  "metrics": {...},
  "error": None
}
```

- [ ] **Step 4: Register capability**

Set `safe_rl_cmdp` to implemented with default config:

```python
"datasets": ["risky_shortcut", "delayed_safety", "resource_gathering"],
"methods": ["reward_only", "lagrangian_penalty_1.00", "lagrangian_penalty_3.00", "preference_cone_policy", "occupancy_enumeration"],
"seeds": list(range(10)),
"primary_metric": "safe_return",
"metric_direction": "higher",
"paper_title": "Validation-Selected Preference-Cone Policy Selection for Finite CMDP Benchmarks",
"scoped_claim": "Validation-selected preference-cone policy selection improves safe-return trade-offs over reward-only and fixed-penalty Lagrangian baselines on small finite CMDP benchmarks, while reporting distance to the exact occupancy-enumeration frontier."
```

- [ ] **Step 5: Extend benchmark suite dispatcher**

Allow `_run_capability()` to dispatch `safe_rl_cmdp`.

- [ ] **Step 6: Verify green**

Run:

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_safe_rl_cmdp_benchmark tests.test_capability_registry tests.test_benchmark_suite
```

Expected: all pass.

## Task 4: RL Manuscript Grounding

**Files:**
- Modify: `agents/manuscript_writer.py`
- Test: `tests/test_manuscript_writer.py`

- [ ] **Step 1: Write failing tests**

Add a manuscript test with `benchmark_config["capability"] == "safe_rl_cmdp"` verifying the paper includes:

- `Finite CMDP`
- `discounted reward`
- `constraint_violation`
- `occupancy-enumeration`
- `safe_return`
- `not a deep-RL result`

- [ ] **Step 2: Verify red**

Run:

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_manuscript_writer
```

Expected: missing RL-specific language.

- [ ] **Step 3: Implement RL-specific sections**

Extend existing helper functions without forking the manuscript workflow:

- `_metric_definition()`
- `_method_descriptions()`
- `_algorithmic_specification()`
- `_related_work_context()`
- `_tradeoff_interpretation()` if useful

Do not create a separate RL manuscript writer.

- [ ] **Step 4: Verify green**

Run:

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_manuscript_writer
```

Expected: manuscript tests pass.

## Task 5: Full Pipeline And AI Review

**Files:**
- No new workflow files.
- Update: `docs/2026-04-29-progress-and-roadmap.md`

- [ ] **Step 1: Full tests**

Run:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests
```

Expected: all pass.

- [ ] **Step 2: Run full RL pipeline**

Use existing functions:

```python
from agents.experiment_forge import forge_experiment
from agents.validation_loop import run_validation_loop
from agents.knowledge_loop import process_completed_run
from agents.manuscript_writer import generate_manuscript
from agents.ai_reviewer import review_manuscript
from agents.evidence_gate import write_evidence_gate
from agents.review_planner import plan_followup_experiments
```

Run against the existing safe-RL insight or create/select a deep insight whose `research_spec` maps to `domain="safe_rl"` and `task_type="rl"`.

- [ ] **Step 3: Interpret review**

If review is `reject` or `weak_reject`, classify root cause:

- implementation bug;
- unsupported claim;
- insufficient evidence;
- manuscript under-specification;
- reviewer/prompt issue.

Only modify the component identified by evidence.

- [ ] **Step 4: Iterate until pass or scientific blocker**

Success condition:

```text
review.recommendation in {"weak_accept", "accept"}
evidence_gate.manuscript_status == "paper_ready_candidate"
```

If the result remains rejected because finite CMDP novelty is insufficient, do not weaken the reviewer. Record the blocker and next scientific work.

- [ ] **Step 5: Browser verification**

Open `http://127.0.0.1:8080/`, verify:

- experiment run appears under Experiments;
- run status is completed;
- detail view shows manuscript/review artifacts;
- latest gate status matches review.

## TODO After RL Review Pass

- Implement automatic review-followup-to-config compiler.
- Implement non-tabular RL environments only after finite CMDP path is stable.
- Add citation retrieval and related-work grounding.
- Add domain-specific evidence gates for NLP, causal inference, systems, robotics, and biology.
- Add LaTeX export only after artifact evidence and review pass are stable.
- Add multi-run campaign orchestration that stops only when review passes or a scientific blocker is recorded.

## Progress

- [x] Plan written.
- [x] Task 1 complete.
- [x] Task 2 complete.
- [x] Task 3 complete.
- [x] Task 4 complete.
- [ ] Task 5 complete.

## 2026-04-30 Execution Notes

- Implemented `benchmarks.safe_rl_cmdp` with six finite CMDP environments:
  - `risky_shortcut`
  - `delayed_safety`
  - `resource_gathering`
  - `randomized_bandit`
  - `stochastic_bridge`
  - `tight_budget_chain`
- Implemented deterministic policy enumeration, exact Bellman evaluation, fixed-penalty Lagrangian policies, tuned Lagrangian grid baseline, preference-grid baseline, deterministic feasible frontier, and exact occupancy-measure LP reference via SciPy `linprog(method="highs")`.
- Added LP diagnostics to benchmark metrics:
  - `lp_flow_residual`
  - `lp_cost_residual`
  - `lp_objective_gap`
- Registered `safe_rl_cmdp` as an implemented capability using existing `capability_registry -> benchmark_suite -> statistical_reporter -> evidence_gate -> manuscript_writer -> ai_reviewer` flow.
- Added candidate/reference separation in the statistical reporter so exact references are not confused with preference-grid methods.
- Updated manuscript generation with safe-RL/CMDP-specific method, metric, LP, limitation, citation, and reproducibility text.
- Added review-after-evidence refresh in web routes, reusing the existing `write_evidence_gate` function.
- Verified full local test suite:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests
```

Result: `Ran 142 tests ... OK`.

### Real Pipeline Runs

- Run `16`: first safe-RL benchmark candidate, AI review `reject`.
  - Root cause: deterministic enumeration was incorrectly framed as exact CMDP frontier and manuscript claim mismatched statistical evidence.
- Run `17`: added occupancy-measure LP and candidate/reference separation, AI review `weak_reject`.
  - Root cause: preference-grid claim was still too strong relative to fixed-penalty baselines.
- Run `18`: narrowed claim to finite-CMDP benchmark harness and exact LP reference, AI review `reject`.
  - Root cause: contribution framed as too weak and text lacked enough reproducibility detail.
- Run `19`: added six environments, tuned Lagrangian baseline, safety-penalty sensitivity, LP residuals, richer manuscript text, AI review `weak_reject`.
  - Root cause: reviewer still required fuller environment specs, independent LP validation, and clearer artifact/reproduction details.
- Run `20`: added citations, introduction, and internal-gate wording cleanup, AI review `weak_reject`.
- Run `21`: added deterministic feasible reference, denser penalty grid, clearer LP-vs-safe-return distinction, AI review `weak_reject`.

### Current Honest Status

The RL path now runs end to end and produces real benchmark results, a paper candidate, AI review artifacts, and review-aware evidence-gate updates. It does **not** yet satisfy the success condition because the latest real review is still:

```text
recommendation = weak_reject
evidence_gate.manuscript_status = needs_more_experiments
```

This is not a data-volume-only failure. The remaining blocker is scientific/reproducibility scope:

- the manuscript must include or attach complete finite-CMDP environment specifications;
- the LP reference should be independently validated or cross-checked;
- aggregate hard-constrained reward and randomized-policy benefit should be reported separately from `safe_return`;
- oracle grid-selected baselines must be labeled as such or replaced by a non-oracle selection protocol;
- exact reproduction commands, dependency versions, solver tolerances, and artifact hashes should be included.

## Next TODO For RL Pass

- [x] Add a generated CMDP appendix artifact with transition, reward, cost, start, gamma, and cost-limit tables for every configured environment/seed.
- [x] Add an independent LP validation check for small instances:
  - deterministic feasible enumeration comparison;
  - randomized one-state analytic check;
  - LP residual/tolerance summary table.
- [x] Add aggregate reward/cost/violation tables alongside aggregate `safe_return`.
- [x] Add explicit oracle-baseline labeling for `lagrangian_grid_best` and `preference_cone_policy`.
- [x] Add dependency/version and command manifest artifact.
- [x] Regenerate manuscript and rerun AI review until recommendation is `weak_accept` or `accept`, or record a scientific blocker without weakening the reviewer.

### 2026-04-30 Follow-up Implementation Notes

- Added `benchmarks.safe_rl_cmdp.artifacts` and wired it into the existing `benchmark_suite` safe-RL capability path. A safe-RL benchmark run now records `cmdp_environment_appendix.json`, `lp_validation.json`, and `reproduction_manifest.json` as normal artifacts.
- Added `aggregate_metric_summaries` to `statistical_reporter`, reusing the existing metric discovery, bootstrap CI, and main/ablation separation. This gives manuscript/reviewer-facing aggregate reward, cost, constraint-violation, and `safe_return` summaries without adding a parallel reporting path.
- Verified targeted red-to-green checks:

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_benchmark_suite.BenchmarkSuiteTests.test_run_benchmark_suite_writes_safe_rl_reproducibility_artifacts
.\.venv\Scripts\python.exe -m unittest tests.test_statistical_reporter.StatisticalReporterTests.test_statistical_report_includes_aggregate_metric_summaries
```

Results: both targeted tests pass.

Remaining focused gap in this subsection: add the randomized one-state analytic LP check and make oracle-style baselines explicitly labeled in outputs/manuscript before the next AI-review iteration.

### 2026-04-30 LP Validation And Oracle Labeling

- Added `analytic_randomized_checks` to `lp_validation.json`: a one-state two-action CMDP with known analytic optimum. The exact LP should mix actions 50/50 at `gamma=0.5`, producing reward `1.0` and cost `1.0`; recorded gaps must be below tolerance.
- Added `selection_protocol` and `baseline_role` metadata to safe-RL solver outputs and benchmark rows:
  - `lagrangian_grid_best` and `preference_cone_policy`: `oracle_grid_selected_safe_return`;
  - `occupancy_lp_optimal`: `exact_lp_reference`;
  - fixed-penalty and reward-only methods are labeled separately.
- Updated manuscript wording so oracle grid-selected baselines are explicitly labeled rather than implied to be deployable non-oracle tuning protocols.
- Verification:

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_safe_rl_cmdp_benchmark.SafeRLCMDPBenchmarkTests.test_harness_runs_methods_for_multiple_seeds tests.test_safe_rl_cmdp_benchmark.SafeRLCMDPBenchmarkTests.test_lp_validation_includes_analytic_randomized_one_state_check
.\.venv\Scripts\python.exe -m unittest tests.test_manuscript_writer.ManuscriptWriterTests.test_safe_rl_paper_candidate_includes_cmdp_grounding
```

Results: targeted tests pass.

### Run 22 Local Pipeline Status

- Created a fresh RL/CMDP run from deep insight `1`: `run_id=22`.
- Local forge reached benchmark-suite mode; code scouting/scaffold LLM calls were blocked by sandbox network permissions and fell back to the minimal scratch scaffold. This did not affect the safe-RL benchmark-suite path.
- Validation result:

```text
status = completed
phase = benchmark_suite
hypothesis_verdict = confirmed
baseline safe_return = -2.877857582929265
best safe_return = 2.3220769030959634
evidence_gate.manuscript_status = paper_ready_candidate
```

- Generated artifacts now present for run `22`:
  - `artifacts/results/cmdp_environment_appendix.json`
  - `artifacts/results/lp_validation.json`
  - `artifacts/results/reproduction_manifest.json`
  - `artifacts/results/statistical_report.json`
  - `artifacts/manuscript/paper_candidate.md`
- Confirmed the regenerated paper candidate mentions:
  - `oracle grid-selected`
  - `cmdp_environment_appendix.json`
  - `lp_validation.json`
  - `analytic one-state randomized`
  - `reproduction_manifest.json`
- Verification:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests
```

Result: `Ran 147 tests ... OK`.

Current blocker: real AI review for run `22` requires sending the manuscript and project artifact text to the configured external LLM provider. The tool permission review rejected that outbound data transfer until the user gives explicit approval for this specific review call.

### 2026-04-30 Web Follow-up Route Debugging

- Symptom: clicking `Plan Follow-up` in the experiment detail view returned Flask default `404 Not Found`.
- Root cause classification: `dependency_or_environment` / stale runtime process, not a route or frontend logic bug. The current source contained `POST /api/experiments/<run_id>/plan_followup`, and route-level tests passed, but PID `69052` was an older Python process listening on `127.0.0.1:8080` from before the route was loaded.
- Fix: stopped the stale Python web process and restarted `main.py` from the current workspace/virtualenv.
- Verification:

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_web_experiment_routes.WebExperimentRouteTests.test_plan_followup_route_calls_existing_planner tests.test_web_experiment_routes.WebExperimentRouteTests.test_plan_followup_route_returns_400_when_review_missing
```

Result: route tests pass.

Browser/API verification:

```text
POST /api/experiments/21/plan_followup -> 200
artifacts/results/followup_experiment_plan.json written for run 21
```

### 2026-04-30 Insights Feed Repair

- Symptom: the UI `Insights`/overview feed was empty even though experiments were running from real deep discoveries.
- Root cause classification: `bottom_logic_bug` at the web/API integration layer. Existing experiment/discovery code writes and reads `deep_insights`, while `/api/insights` and several UI feed surfaces still queried the legacy `insights` table only.
- Fix: kept the existing `/api/insights` route and made it a unified feed over legacy `insights` plus normalized `deep_insights`. No duplicate endpoint and no data copying into the old table.
- UI fix: deep-insight cards now reuse existing deep-discovery/SciForge actions (`SciForge Run`, `Forge Only`, `Deep Research`) instead of sending deep insight IDs to legacy research-proposal routes.
- Verification:

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_web_experiment_routes.WebExperimentRouteTests.test_insights_route_includes_normalized_deep_insights tests.test_web_experiment_routes.WebExperimentRouteTests.test_insights_route_preserves_legacy_rows
.\.venv\Scripts\python.exe -m unittest tests.test_web_experiment_routes
```

Browser/API verification:

```text
GET /api/insights?limit=10 -> 5 normalized deep_insights
Insights tab shows the Social-choice CMDP discovery and deep/SciForge actions.
```

### 2026-04-30 Review-Pass Iteration And Root-Cause Notes

- Run `23` real external AI review returned `weak_reject`.
- Root-cause classification:
  - `bottom_logic_bug`: `ai_reviewer` sent only the first 50,000 characters of the manuscript. The manuscript was about 51k characters, so the reviewer saw a truncated `Limitations` section and truncated secondary metrics.
  - `llm_prompt_or_scaffold`: the review prompt lacked compact artifact context, so the reviewer had to infer too much from prose rather than `statistical_report.json`, `lp_validation.json`, and `reproduction_manifest.json`.
  - `data_or_evidence_insufficient`: `lp_validation.json` compared LP to deterministic feasible policies but did not directly include the configured fixed-penalty candidate gap that the reviewer requested.
  - `manuscript_framing`: the prose and claim text used "best configured method" for `lagrangian_penalty_4.00`, while `occupancy_lp_optimal` was also configured and had the highest aggregate `safe_return`. This was a framing bug, not a reason to change benchmark parameters.
- Fixes:
  - `agents/ai_reviewer.py` now builds reviewer context from the manuscript head and tail when long, and includes compact summaries of `benchmark_config.json`, `evidence_gate.json`, `statistical_report.json`, `lp_validation.json`, and `reproduction_manifest.json`.
  - `agents/manuscript_writer.py` now includes an `Aggregate Method Summary`, a `Reference And Candidate Framing` section, compact secondary metrics with explicit artifact pointers, and clearer LP/reference/candidate language.
  - `benchmarks/safe_rl_cmdp/artifacts.py` now records candidate-vs-LP gap fields in `lp_validation.json` for configured fixed-penalty candidates.
  - `agents/result_interpreter.py` now writes "configured candidate method" and reports the aggregate reference frontier separately instead of saying "best configured method".
- Verification:

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_ai_reviewer.AiReviewerTests.test_review_prompt_preserves_long_manuscript_tail_and_artifact_context tests.test_manuscript_writer.ManuscriptWriterTests.test_safe_rl_paper_candidate_includes_cmdp_grounding tests.test_manuscript_writer.ManuscriptWriterTests.test_evidence_gate_paper_ready_writes_candidate_with_statistical_results
.\.venv\Scripts\python.exe -m unittest tests.test_safe_rl_cmdp_benchmark.SafeRLCMDPBenchmarkTests.test_lp_validation_compares_randomized_lp_to_fixed_penalty_candidate tests.test_safe_rl_cmdp_benchmark.SafeRLCMDPBenchmarkTests.test_lp_validation_includes_analytic_randomized_one_state_check
.\.venv\Scripts\python.exe -m unittest tests.test_result_interpreter.ResultInterpreterTests.test_benchmark_claim_uses_scoped_claim_instead_of_original_title
.\.venv\Scripts\python.exe -m unittest tests.test_safe_rl_cmdp_benchmark tests.test_manuscript_writer tests.test_ai_reviewer tests.test_statistical_reporter tests.test_benchmark_suite tests.test_capability_registry
```

Results: targeted and related suites pass.

### Run 24 Review-Pass Status

- Fresh RL/CMDP pipeline run from deep insight `1`: `run_id=24`.
- Forge code-scout/scaffold LLM calls were blocked by sandbox socket permissions and fell back to the existing benchmark-suite path. This did not affect the safe-RL benchmark execution.
- Local validation result:

```text
status = completed
phase = benchmark_suite
hypothesis_verdict = confirmed
baseline safe_return = -2.877857582929265
configured candidate safe_return = 1.663454494653472
evidence_gate.manuscript_status = paper_ready_candidate
```

- Generated artifacts:
  - `artifacts/results/benchmark_results.json`
  - `artifacts/results/statistical_report.json`
  - `artifacts/results/cmdp_environment_appendix.json`
  - `artifacts/results/lp_validation.json`
  - `artifacts/results/reproduction_manifest.json`
  - `artifacts/manuscript/paper_candidate.md`
  - `artifacts/reviews/review.json`
  - `artifacts/reviews/review.md`
  - `artifacts/results/followup_experiment_plan.json`
- Real external AI review result:

```text
recommendation = weak_accept
overall_score = 6
evidence_gate.manuscript_status = paper_ready_candidate
```

- Reviewer remaining caveat: the result is acceptable only as a scoped auditable finite-CMDP benchmark/software artifact, not as a new broad safe-RL algorithm or deployment claim.
- A post-review local wording cleanup replaced "best configured method" with "configured candidate method" and separately names `occupancy_lp_optimal` as the aggregate reference frontier. The external review already passed before this wording cleanup; a second export of the modified manuscript was blocked pending another explicit approval.

### Remaining TODO Beyond RL Review-Pass

- Add a portable reproduction path beyond Windows `.venv` commands, ideally a lockfile or container recipe.
- Add a repository commit hash or clean-source snapshot identifier to `reproduction_manifest.json`.
- If making broader benchmark claims, add random CMDP families or external tabular CMDP benchmarks.
- If making deployability claims for penalty choice, add a non-oracle validation protocol over held-out environments.
- If making solver scalability claims, add runtime/memory scaling over state/action sizes.
- Improve citation hygiene by verifying each arXiv/source entry and removing weakly integrated KG evidence citations from submission-facing references.

### 2026-04-30 Run 25 Final RL Review-Pass

- Root-cause follow-up after run `24`:
  - `bottom_logic_bug`: revision manuscript generation could produce a new `paper_candidate.md` while leaving the old review-blocking `evidence_gate.json` in place. Fixed by letting review-revision candidates write a consistent `paper_ready_candidate` gate for the revised package before re-review.
  - `manuscript_framing`: static randomized-optimum notes disagreed with LP validation gaps. Fixed by deriving environment randomization notes from `lp_validation.json`.
  - `data_or_evidence_insufficient`: six hand-designed tasks were too narrow. Added systematic generated CMDP families that vary state count, action count, discount factor, cost limit, transition stochasticity, and randomized-optimum regimes.
  - `statistical_reporting`: pooled summaries were overemphasized. Added environment-grouped summaries, LP randomization gap summaries, runtime summaries, and separated `safety_penalty=3.00` / `safety_penalty=9.00` sensitivity labels.
  - `terminology`: replaced "effect size" headline wording with "signed aggregate difference" where applicable.
- New reusable components:
  - `agents/reproduction_verifier.py`: runs real local reproduction smoke commands and writes `artifacts/results/reproduction_check.json`.
  - `benchmarks.safe_rl_cmdp.envs.SYSTEMATIC_SPECS`: reusable systematic finite-CMDP family definitions.
- Fresh run:

```text
run_id = 25
datasets = 10
lp_validation instances = 100
status = completed
phase = benchmark_suite
hypothesis_verdict = confirmed
evidence_gate.manuscript_status = paper_ready_candidate
```

- Real external AI review result for run `25`:

```text
recommendation = weak_accept
overall_score = 6
gate = paper_ready_candidate
```

- Reviewer caveat still applies: this is acceptable as a scoped artifact/reporting benchmark note for finite CMDPs. It is not a new safe-RL method, not a broad simulator benchmark, and not a deployable penalty-selection claim.

### TODO After RL Pass

- Build a clean-container or fresh-checkout reproduction workflow and report hash matching.
- Add stronger non-oracle constrained-planning baselines only if the manuscript will claim comparative method performance.
- Add a validation/held-out-environment penalty-selection protocol only if making deployability claims for fixed Lagrangian penalties.
- Treat environment-grouped summaries as primary in future manuscript revisions; keep pooled dataset-seed CI as audit context.

### Final Verification

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests
```

Result: `Ran 155 tests ... OK`.

Browser/API verification after restarting the local Flask process:

```text
GET /api/insights?limit=5 -> 5 rows; first row source_table=deep_insights; title=Social-choice CMDPs...
GET /api/experiments -> 25 rows; latest run_id=25; status=completed; hypothesis_verdict=confirmed
GET /api/experiments/25 -> artifacts include artifacts/reviews/review.json and artifacts/results/evidence_gate.json
In-app browser DOM contains the Social-choice CMDP discovery and Experiments navigation.
```
