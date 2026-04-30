# Generalized Research Execution Framework Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Generalize DeepGraph/SciForge so different research titles can be compiled into reusable research specs, matched to existing experiment capabilities, evaluated through evidence gates, and iterated after AI review without hard-coding one topic such as Fairlearn.

**Architecture:** Keep `experiment_runs` as the canonical execution object and keep artifacts under each run `workdir`. Add small focused modules for research spec compilation, capability selection, benchmark-suite execution, statistical reporting, evidence gating, and review-to-followup planning. Do not add a parallel `research_projects` API, do not create fake endpoints, and do not move existing manuscript/review actions out of the current experiment workflow.

**Tech Stack:** Python, Flask, SQLite, existing `db.database` helpers, existing LLM client, existing artifact manager, existing validation loop, unittest, local benchmark harnesses under repository code.

---

## Non-Negotiable Constraints

- Reuse existing tables: `deep_insights`, `experiment_runs`, `experiment_iterations`, `experimental_claims`, `discovery_track_record`.
- Reuse existing routes where possible:
  - `POST /api/experiments/forge`
  - `POST /api/experiments/<run_id>/run`
  - `POST /api/experiments/run_full`
  - `GET /api/experiments/<run_id>`
  - `POST /api/experiments/<run_id>/manuscript`
  - `POST /api/experiments/<run_id>/review`
- Do not create a new top-level research-project CRUD system.
- Do not add interfaces that are not called by the workflow or covered by tests.
- Do not let `hypothesis_verdict='confirmed'` alone imply publishability.
- Do not hard-code title-specific logic into `validation_loop.py` or `manuscript_writer.py`.
- Do not use remote datasets as a required default path. Network-backed datasets may be later capabilities, but first version must run offline with synthetic or bundled/procedural datasets.
- Any generated manuscript must clearly distinguish:
  - plumbing healthcheck
  - proxy benchmark
  - empirical benchmark suite
  - paper-ready candidate

## Debugging And Failure Discipline

Every failed run, failed test, rejected review, or missing artifact must be classified before changing code or configuration.

Required failure categories:

- `bottom_logic_bug`: broken code path, schema mismatch, artifact contract mismatch, entrypoint selection bug, reset/commit bug.
- `dependency_or_environment`: missing package, blocked network, unavailable browser/server, path issue, permission issue.
- `llm_prompt_or_scaffold`: LLM generated a library entrypoint, missing benchmark harness, invented metric, incomplete method context.
- `data_or_evidence_insufficient`: not enough seeds, datasets, baselines, ablations, statistical power, citation grounding.
- `scientific_hypothesis_unsupported`: experiment ran correctly but result does not support the claim.

Before modifying thresholds, seed counts, timeouts, success criteria, or reviewer gates, record why the existing value is wrong. Do not make a failed run pass by weakening the standard. Fix the smallest component that evidence identifies as the cause, then re-run the narrow test before widening verification.

---

## Target Data Contracts

### `research_spec.json`

Stored as an artifact:

```json
{
  "schema_version": 1,
  "run_id": 1,
  "insight_id": 1,
  "claim": "Short claim being tested",
  "domain": "fairness",
  "task_type": "classification",
  "evidence_level": "proxy",
  "required_evidence": [
    "baseline_comparison",
    "multi_seed",
    "statistical_test"
  ],
  "candidate_capabilities": [
    "fairness_classification"
  ],
  "primary_metrics": [
    "fairness_score"
  ],
  "secondary_metrics": [
    "accuracy",
    "demographic_parity_gap",
    "equalized_odds_gap"
  ],
  "constraints": {
    "offline_first": true,
    "max_runtime_seconds": 600
  },
  "source": {
    "title": "Original title",
    "tier": 1
  }
}
```

### Capability descriptor

Returned by code, not stored as a new DB table in first version:

```json
{
  "id": "fairness_classification",
  "domains": ["fairness"],
  "task_types": ["classification"],
  "required_packages": ["numpy", "sklearn", "fairlearn"],
  "runner": "benchmarks.fairness_classification.harness",
  "default_config": {
    "datasets": ["synthetic_grouped"],
    "methods": ["logistic_regression", "threshold_optimizer", "preference_cone_threshold"],
    "seeds": [0, 1, 2, 3, 4]
  }
}
```

### `benchmark_config.json`

Stored in run workdir:

```json
{
  "schema_version": 1,
  "capability": "fairness_classification",
  "datasets": ["synthetic_grouped"],
  "methods": ["logistic_regression", "exponentiated_gradient", "threshold_optimizer", "preference_cone_threshold"],
  "seeds": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
  "primary_metric": "fairness_score",
  "metric_direction": "higher",
  "timeout_seconds": 600
}
```

### `benchmark_results.json`

Stored in `artifacts/results/benchmark_results.json`:

```json
{
  "schema_version": 1,
  "run_id": 1,
  "capability": "fairness_classification",
  "rows": [
    {
      "dataset": "synthetic_grouped",
      "seed": 0,
      "method": "logistic_regression",
      "status": "ok",
      "metrics": {
        "accuracy": 0.8,
        "demographic_parity_gap": 0.4,
        "equalized_odds_gap": 0.3,
        "fairness_score": 0.62,
        "runtime_seconds": 0.2
      },
      "error": null
    }
  ]
}
```

### `statistical_report.json`

Stored in `artifacts/results/statistical_report.json`:

```json
{
  "schema_version": 1,
  "run_id": 1,
  "primary_metric": "fairness_score",
  "metric_direction": "higher",
  "baseline_method": "logistic_regression",
  "best_method": "preference_cone_threshold",
  "summary": [
    {
      "dataset": "synthetic_grouped",
      "method": "preference_cone_threshold",
      "metric": "fairness_score",
      "mean": 0.63,
      "std": 0.04,
      "ci_low": 0.60,
      "ci_high": 0.66,
      "n": 10
    }
  ],
  "comparisons": [
    {
      "dataset": "synthetic_grouped",
      "baseline": "logistic_regression",
      "candidate": "preference_cone_threshold",
      "metric": "fairness_score",
      "mean_delta": 0.08,
      "paired_sign_test_p": 0.02,
      "wins": 8,
      "losses": 2,
      "ties": 0
    }
  ]
}
```

### `evidence_gate.json`

Stored in `artifacts/results/evidence_gate.json`:

```json
{
  "schema_version": 1,
  "run_id": 1,
  "manuscript_status": "needs_more_experiments",
  "blocking_reasons": [
    "benchmark_suite_has_fewer_than_10_seeds",
    "missing_safe_rl_environment"
  ],
  "satisfied_requirements": [
    "has_baseline_comparison",
    "has_statistical_report"
  ],
  "next_required_experiments": [
    "Run at least 10 seeds for every configured dataset and method."
  ]
}
```

---

## File Structure

Create:

- `agents/research_spec_compiler.py`
  - Converts a `deep_insights` row plus optional run context into `research_spec.json`.
- `agents/capability_registry.py`
  - In-memory registry of real capabilities. No DB table in first version.
- `agents/benchmark_suite.py`
  - Generic runner that loads `benchmark_config.json`, calls the selected benchmark harness, writes `benchmark_results.json`.
- `agents/statistical_reporter.py`
  - Converts benchmark rows into statistical summaries and comparison tables.
- `agents/evidence_gate.py`
  - Determines manuscript status from artifacts and review state.
- `agents/review_planner.py`
  - Converts `review.json` into follow-up experiment plan artifacts.
- `benchmarks/__init__.py`
- `benchmarks/fairness_classification/__init__.py`
- `benchmarks/fairness_classification/datasets.py`
- `benchmarks/fairness_classification/methods.py`
- `benchmarks/fairness_classification/metrics.py`
- `benchmarks/fairness_classification/harness.py`

Modify:

- `agents/experiment_forge.py`
  - Compile `research_spec.json`.
  - Select capability via registry.
  - Write `benchmark_config.json` when capability supports a benchmark suite.
  - Keep existing single-script fallback.
- `agents/validation_loop.py`
  - Detect `benchmark_config.json`.
  - If present, execute `benchmark_suite.run_benchmark_suite(run_id)`.
  - Keep current single-script loop unchanged for non-suite runs.
- `agents/manuscript_writer.py`
  - Read `evidence_gate.json`.
  - Generate `preliminary_report.md`, `additional_experiments_required.md`, or `paper_candidate.md` based on gate status.
  - Keep negative report behavior for failed/refuted runs.
- `agents/ai_reviewer.py`
  - No schema change unless needed; keep review artifact shape stable.
- `web/app.py`
  - Include new artifacts in existing experiment detail response through existing manifest listing.
  - Add one action route only if needed: `POST /api/experiments/<run_id>/plan_followup`.
- `web/static/js/app.js`
  - Display `manuscript_status` and follow-up artifact if present in experiment details.
- `README.md`
- `HANDOFF.md`
- `docs/2026-04-29-progress-and-roadmap.md`

Tests:

- `tests/test_research_spec_compiler.py`
- `tests/test_capability_registry.py`
- `tests/test_fairness_benchmark_harness.py`
- `tests/test_benchmark_suite.py`
- `tests/test_statistical_reporter.py`
- `tests/test_evidence_gate.py`
- `tests/test_review_planner.py`
- Update `tests/test_experiment_forge_plan_integration.py`
- Update `tests/test_validation_loop_artifacts.py`
- Update `tests/test_manuscript_writer.py`
- Update `tests/test_web_experiment_routes.py`
- Add/update `tests/test_e2e_toy_research_pipeline.py`

---

## Task 1: Research Spec Compiler

**Files:**
- Create: `agents/research_spec_compiler.py`
- Test: `tests/test_research_spec_compiler.py`

- [x] **Step 1: Write tests for generic title/spec compilation**

Test cases:

```python
def test_fairness_title_compiles_to_fairness_classification_spec():
    insight = {
        "id": 1,
        "title": "Group fairness via constrained preference optimization",
        "tier": 1,
        "experimental_plan": '{"metrics":{"primary":"fairness_score","direction":"higher"}}',
        "formal_structure": "Preference constraints over grouped outcomes.",
    }
    spec = compile_research_spec(insight, run_id=7)
    assert spec["domain"] == "fairness"
    assert spec["task_type"] == "classification"
    assert "fairness_classification" in spec["candidate_capabilities"]
    assert "fairness_score" in spec["primary_metrics"]
```

```python
def test_unknown_title_falls_back_to_generic_python_spec():
    insight = {
        "id": 2,
        "title": "A strange new optimization heuristic",
        "tier": 1,
        "experimental_plan": "",
    }
    spec = compile_research_spec(insight, run_id=8)
    assert spec["domain"] == "unknown"
    assert spec["task_type"] == "benchmark"
    assert spec["candidate_capabilities"] == ["generic_python_benchmark"]
```

```python
def test_spec_is_written_and_recorded_as_artifact(tmp_path):
    insight = {"id": 3, "title": "Fair classification", "tier": 2}
    spec = compile_and_write_research_spec(insight, tmp_path, run_id=9)
    assert (tmp_path / "research_spec.json").exists()
    assert spec["run_id"] == 9
```

- [x] **Step 2: Run tests and verify failure**

Run:

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_research_spec_compiler
```

Expected: import failure because `agents.research_spec_compiler` does not exist.

- [x] **Step 3: Implement `agents/research_spec_compiler.py`**

Public functions:

```python
def compile_research_spec(insight: dict, run_id: int | None = None) -> dict:
    ...

def compile_and_write_research_spec(insight: dict, workdir: Path, run_id: int) -> dict:
    ...
```

Rules:

- Parse `experimental_plan` JSON when present.
- Use conservative keyword detection only for first version:
  - fairness/group/demographic/equalized -> `domain="fairness"`, `task_type="classification"`.
  - safe/RL/CMDP/MDP/policy -> `domain="safe_rl"`, `task_type="rl"`.
  - otherwise `domain="unknown"`, `task_type="benchmark"`.
- Choose candidate capabilities:
  - fairness classification -> `["fairness_classification", "generic_python_benchmark"]`
  - safe RL -> `["safe_rl_cmdp", "generic_python_benchmark"]`
  - unknown -> `["generic_python_benchmark"]`
- Set `evidence_level="proxy"` by default.
- Required evidence default:
  - `baseline_comparison`
  - `multi_seed`
  - `statistical_test`
  - `ablation`
- Write `research_spec.json` in workdir and record it via `artifact_manager.record_artifact`.

- [x] **Step 4: Verify**

Run:

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_research_spec_compiler
```

Expected: all tests pass.

---

## Task 2: Capability Registry

**Files:**
- Create: `agents/capability_registry.py`
- Test: `tests/test_capability_registry.py`

- [x] **Step 1: Write tests**

Test cases:

```python
def test_registry_matches_fairness_spec_to_fairness_capability():
    spec = {"domain": "fairness", "task_type": "classification", "candidate_capabilities": ["fairness_classification"]}
    capability = select_capability(spec)
    assert capability["id"] == "fairness_classification"
    assert capability["runner"] == "benchmarks.fairness_classification.harness"
```

```python
def test_registry_falls_back_to_generic_when_specific_missing():
    spec = {"domain": "unknown", "task_type": "benchmark", "candidate_capabilities": ["missing", "generic_python_benchmark"]}
    capability = select_capability(spec)
    assert capability["id"] == "generic_python_benchmark"
```

```python
def test_capability_reports_missing_dependencies(monkeypatch):
    monkeypatch.setattr("agents.capability_registry.importlib.util.find_spec", lambda name: None)
    capability = get_capability("fairness_classification")
    missing = missing_dependencies(capability)
    assert "numpy" in missing
```

- [x] **Step 2: Run tests and verify failure**

Run:

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_capability_registry
```

Expected: import failure.

- [x] **Step 3: Implement registry**

Public functions:

```python
def list_capabilities() -> list[dict]:
    ...

def get_capability(capability_id: str) -> dict | None:
    ...

def select_capability(research_spec: dict) -> dict:
    ...

def missing_dependencies(capability: dict) -> list[str]:
    ...
```

Initial capabilities:

- `fairness_classification`
  - Real runner: `benchmarks.fairness_classification.harness`
  - Required packages: `numpy`, `sklearn`, `fairlearn`
- `generic_python_benchmark`
  - Runner empty in first version.
  - Used only to preserve current single-script fallback.
- `safe_rl_cmdp`
  - Mark as `implemented=False`.
  - Do not select unless no runnable alternative exists; if selected, evidence gate must mark as not runnable.

- [x] **Step 4: Verify**

Run:

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_capability_registry
```

Expected: all tests pass.

---

## Task 3: Fairness Classification Benchmark Harness

**Files:**
- Create: `benchmarks/__init__.py`
- Create: `benchmarks/fairness_classification/__init__.py`
- Create: `benchmarks/fairness_classification/datasets.py`
- Create: `benchmarks/fairness_classification/metrics.py`
- Create: `benchmarks/fairness_classification/methods.py`
- Create: `benchmarks/fairness_classification/harness.py`
- Test: `tests/test_fairness_benchmark_harness.py`

- [x] **Step 1: Write dataset tests**

Test:

```python
def test_synthetic_grouped_dataset_is_deterministic():
    a = make_dataset("synthetic_grouped", seed=3)
    b = make_dataset("synthetic_grouped", seed=3)
    assert a.x.shape == b.x.shape
    assert (a.y == b.y).all()
    assert (a.sensitive == b.sensitive).all()
```

- [x] **Step 2: Write metric tests**

Test:

```python
def test_group_metrics_report_expected_keys():
    y_true = np.array([0, 1, 1, 0])
    y_pred = np.array([0, 1, 0, 0])
    sensitive = np.array([0, 0, 1, 1])
    metrics = compute_metrics(y_true, y_pred, sensitive)
    assert set(metrics) >= {"accuracy", "demographic_parity_gap", "equalized_odds_gap", "fairness_score"}
```

- [x] **Step 3: Write harness tests**

Test:

```python
def test_harness_runs_all_methods_for_seed():
    config = {
        "datasets": ["synthetic_grouped"],
        "methods": ["logistic_regression", "preference_cone_threshold"],
        "seeds": [0],
        "primary_metric": "fairness_score",
    }
    result = run_fairness_benchmark(config)
    rows = result["rows"]
    assert len(rows) == 2
    assert all(row["status"] == "ok" for row in rows)
```

- [x] **Step 4: Run tests and verify failure**

Run:

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_fairness_benchmark_harness
```

Expected: import failure.

- [x] **Step 5: Implement dataset module**

`datasets.py` should define:

```python
@dataclass
class Dataset:
    name: str
    seed: int
    x_train: np.ndarray
    x_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray
    sensitive_train: np.ndarray
    sensitive_test: np.ndarray

def make_dataset(name: str, seed: int) -> Dataset:
    ...
```

First version supports only `synthetic_grouped`.

- [x] **Step 6: Implement metrics module**

`metrics.py` should define:

```python
def demographic_parity_gap(y_pred: np.ndarray, sensitive: np.ndarray) -> float:
    ...

def equalized_odds_gap(y_true: np.ndarray, y_pred: np.ndarray, sensitive: np.ndarray) -> float:
    ...

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, sensitive: np.ndarray) -> dict:
    ...
```

`fairness_score = accuracy - 0.45 * demographic_parity_gap`.

- [x] **Step 7: Implement methods module**

`methods.py` should define:

```python
def fit_predict(method: str, dataset: Dataset) -> np.ndarray:
    ...
```

Supported methods:

- `logistic_regression`
- `exponentiated_gradient`
- `threshold_optimizer`
- `preference_cone_threshold`

If Fairlearn method import fails, return a row-level error from harness rather than crashing the full suite.

- [x] **Step 8: Implement harness**

`harness.py` should define:

```python
def run_fairness_benchmark(config: dict) -> dict:
    ...
```

It returns the `benchmark_results.json` shape. It does not write files; file writing belongs to `agents/benchmark_suite.py`.

- [x] **Step 9: Verify**

Run:

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_fairness_benchmark_harness
```

Expected: all tests pass.

---

## Task 4: Benchmark Suite Runner

**Files:**
- Create: `agents/benchmark_suite.py`
- Test: `tests/test_benchmark_suite.py`

- [x] **Step 1: Write tests**

Test:

```python
def test_run_benchmark_suite_writes_results_artifact(tmp_path):
    workdir = tmp_path
    (workdir / "benchmark_config.json").write_text(json.dumps({
        "schema_version": 1,
        "capability": "fairness_classification",
        "datasets": ["synthetic_grouped"],
        "methods": ["logistic_regression"],
        "seeds": [0],
        "primary_metric": "fairness_score",
        "metric_direction": "higher",
    }), encoding="utf-8")
    fake_db = FakeDb(run={"id": 12, "workdir": str(workdir)})
    with patch("agents.benchmark_suite.db", fake_db):
        result = run_benchmark_suite(12)
    assert result["status"] == "complete"
    assert (workdir / "artifacts" / "results" / "benchmark_results.json").exists()
```

- [x] **Step 2: Run tests and verify failure**

Run:

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_benchmark_suite
```

Expected: import failure.

- [x] **Step 3: Implement runner**

Public functions:

```python
def load_benchmark_config(workdir: Path) -> dict:
    ...

def run_benchmark_suite(run_id: int) -> dict:
    ...
```

Behavior:

- Fetch `experiment_runs` row by `run_id`.
- Load `benchmark_config.json`.
- Select capability from registry.
- Import the runner module declared by capability.
- Call `run_fairness_benchmark(config)` for first version.
- Write `artifacts/results/benchmark_results.json`.
- Record artifact through `artifact_manager.record_artifact`.
- Return `{"status": "complete", "run_id": run_id, "rows": count}`.

- [x] **Step 4: Verify**

Run:

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_benchmark_suite
```

Expected: all tests pass.

---

## Task 5: Statistical Reporter

**Files:**
- Create: `agents/statistical_reporter.py`
- Test: `tests/test_statistical_reporter.py`

- [x] **Step 1: Write tests**

Test:

```python
def test_statistical_report_selects_best_method_and_comparison():
    rows = [
        {"dataset": "synthetic_grouped", "seed": 0, "method": "logistic_regression", "status": "ok", "metrics": {"fairness_score": 0.50}},
        {"dataset": "synthetic_grouped", "seed": 1, "method": "logistic_regression", "status": "ok", "metrics": {"fairness_score": 0.51}},
        {"dataset": "synthetic_grouped", "seed": 0, "method": "preference_cone_threshold", "status": "ok", "metrics": {"fairness_score": 0.61}},
        {"dataset": "synthetic_grouped", "seed": 1, "method": "preference_cone_threshold", "status": "ok", "metrics": {"fairness_score": 0.60}},
    ]
    report = build_statistical_report(rows, "fairness_score", "higher", baseline_method="logistic_regression")
    assert report["best_method"] == "preference_cone_threshold"
    assert report["comparisons"][0]["wins"] == 2
```

- [x] **Step 2: Run tests and verify failure**

Run:

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_statistical_reporter
```

Expected: import failure.

- [x] **Step 3: Implement reporter**

Public functions:

```python
def build_statistical_report(rows: list[dict], primary_metric: str, direction: str, baseline_method: str) -> dict:
    ...

def write_statistical_report(run_id: int) -> dict:
    ...
```

Implementation:

- Ignore rows where `status != "ok"` or metric missing.
- Group by dataset/method/metric.
- Compute mean, sample std, simple bootstrap CI with deterministic seed.
- Paired comparison by dataset and seed.
- Use sign test as first version:
  - `wins`, `losses`, `ties`.
  - two-sided p-value from binomial tail. Implement directly with `math.comb`.
- Write:
  - `artifacts/results/statistical_report.json`
  - `artifacts/tables/main_results.md`

- [x] **Step 4: Verify**

Run:

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_statistical_reporter
```

Expected: all tests pass.

---

## Task 6: Evidence Gate

**Files:**
- Create: `agents/evidence_gate.py`
- Test: `tests/test_evidence_gate.py`

- [x] **Step 1: Write tests**

Test:

```python
def test_proxy_without_statistical_report_is_preliminary(tmp_path):
    gate = evaluate_evidence({
        "research_spec": {"evidence_level": "proxy", "required_evidence": ["statistical_test"]},
        "benchmark_results": None,
        "statistical_report": None,
        "review": None,
    })
    assert gate["manuscript_status"] == "preliminary"
    assert "missing_statistical_report" in gate["blocking_reasons"]
```

```python
def test_reviewer_reject_blocks_paper_ready():
    gate = evaluate_evidence({
        "research_spec": {"required_evidence": ["baseline_comparison"]},
        "benchmark_results": {"rows": [{"status": "ok"}]},
        "statistical_report": {"comparisons": [{"paired_sign_test_p": 0.02}]},
        "review": {"recommendation": "reject", "required_experiments": ["Run more seeds."]},
    })
    assert gate["manuscript_status"] == "not_publishable"
    assert "review_rejected" in gate["blocking_reasons"]
```

- [x] **Step 2: Run tests and verify failure**

Run:

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_evidence_gate
```

Expected: import failure.

- [x] **Step 3: Implement evidence gate**

Public functions:

```python
def evaluate_evidence(inputs: dict) -> dict:
    ...

def write_evidence_gate(run_id: int) -> dict:
    ...
```

Rules:

- Missing benchmark results -> `preliminary`.
- Missing statistical report when `statistical_test` required -> `preliminary`.
- Fewer than 10 unique seeds when `multi_seed` required -> `needs_more_experiments`.
- Reviewer `reject` -> `not_publishable`.
- Reviewer `major_revision` -> `needs_more_experiments`.
- Only if benchmark, statistical report, no blocking review, and required evidence satisfied -> `paper_ready_candidate`.
- Always write blocking reasons and next required experiments.

- [x] **Step 4: Verify**

Run:

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_evidence_gate
```

Expected: all tests pass.

---

## Task 7: Integrate Spec, Capability, Suite, and Stats Into Forge/Validation

**Files:**
- Modify: `agents/experiment_forge.py`
- Modify: `agents/validation_loop.py`
- Test: `tests/test_experiment_forge_plan_integration.py`
- Test: `tests/test_validation_loop_artifacts.py`

- [x] **Step 1: Add forge integration tests**

Test:

```python
def test_forge_writes_research_spec_and_benchmark_config_for_fairness_insight():
    insight = {
        "id": 31,
        "title": "Group fairness via constrained preference optimization",
        "tier": 1,
        "experimental_plan": '{"metrics":{"primary":"fairness_score","direction":"higher"}}',
    }
    with patched_forge_dependencies(insight) as workdir:
        result = forge_experiment(31)
    assert (workdir / "research_spec.json").exists()
    assert (workdir / "benchmark_config.json").exists()
    assert result["execution_mode"] == "benchmark_suite"
```

- [x] **Step 2: Add validation integration test**

Test:

```python
def test_validation_loop_uses_benchmark_suite_when_config_exists(tmp_path):
    workdir = tmp_path
    (workdir / "benchmark_config.json").write_text('{"capability":"fairness_classification"}', encoding="utf-8")
    fake_db = FakeValidationDb({"id": 40, "workdir": str(workdir), "deep_insight_id": 2})
    with patch("agents.validation_loop.db", fake_db), \
         patch("agents.validation_loop.run_benchmark_suite", return_value={"status": "complete"}), \
         patch("agents.validation_loop.write_statistical_report", return_value={"status": "complete"}), \
         patch("agents.validation_loop.write_evidence_gate", return_value={"manuscript_status": "needs_more_experiments"}):
        result = run_validation_loop(40)
    assert result["execution_mode"] == "benchmark_suite"
```

- [x] **Step 3: Run tests and verify failure**

Run:

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_experiment_forge_plan_integration tests.test_validation_loop_artifacts
```

Expected: new tests fail.

- [x] **Step 4: Modify `experiment_forge.py`**

Integration behavior:

- After scaffold generation and before inserting `experiment_runs`, call:
  - `compile_and_write_research_spec(...)`
  - `select_capability(spec)`
- If selected capability has `runner` and `implemented=True`, write `benchmark_config.json`.
- Record `benchmark_config.json` as artifact after run id exists.
- Include `execution_mode` in return dict:
  - `benchmark_suite` if config written.
  - `single_script` otherwise.

No new DB columns are required in first version; use artifact files and existing `program_md`/`success_criteria`.

- [x] **Step 5: Modify `validation_loop.py`**

At start of `run_validation_loop`:

- If `workdir / "benchmark_config.json"` exists:
  - set run status to `testing`.
  - call `run_benchmark_suite(run_id)`.
  - call `write_statistical_report(run_id)`.
  - call `write_evidence_gate(run_id)`.
  - update `experiment_runs` using summary:
    - `status='completed'`
    - `hypothesis_verdict='confirmed'` only if evidence gate is `paper_ready_candidate` or statistical report shows improvement.
    - `hypothesis_verdict='inconclusive'` if benchmark ran but gate says more experiments needed.
    - `hypothesis_verdict='failed'` is not a valid current value; use `status='failed'` only for runner crash.
  - return result with `execution_mode='benchmark_suite'`.
- Else keep existing single-script validation path.

- [x] **Step 6: Verify**

Run:

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_experiment_forge_plan_integration tests.test_validation_loop_artifacts
```

Expected: all tests pass.

---

## Task 8: Manuscript Writer Evidence Gate Integration

**Files:**
- Modify: `agents/manuscript_writer.py`
- Test: `tests/test_manuscript_writer.py`

- [x] **Step 1: Write tests**

Test:

```python
def test_manuscript_writer_outputs_additional_experiments_report_when_gate_blocks_paper(tmp_path):
    write_json(tmp_path / "artifacts/results/evidence_gate.json", {
        "manuscript_status": "needs_more_experiments",
        "blocking_reasons": ["benchmark_suite_has_fewer_than_10_seeds"],
        "next_required_experiments": ["Run 10 seeds."]
    })
    result = generate_manuscript(run_id)
    assert "additional_experiments_required.md" in result["outputs"][0]
```

```python
def test_manuscript_writer_outputs_paper_candidate_when_gate_allows(tmp_path):
    write_json(tmp_path / "artifacts/results/evidence_gate.json", {
        "manuscript_status": "paper_ready_candidate",
        "blocking_reasons": [],
        "next_required_experiments": []
    })
    result = generate_manuscript(run_id)
    assert "paper_candidate.md" in result["outputs"][0]
```

- [x] **Step 2: Run tests and verify failure**

Run:

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_manuscript_writer
```

Expected: new tests fail.

- [x] **Step 3: Modify writer**

Behavior:

- Failed/refuted run still writes `negative_result_report.md`.
- If `evidence_gate.json` missing:
  - confirmed/inconclusive writes `preliminary_report.md`.
- If gate status:
  - `not_publishable` -> `additional_experiments_required.md`
  - `needs_more_experiments` -> `additional_experiments_required.md`
  - `preliminary` -> `preliminary_report.md`
  - `paper_ready_candidate` -> `paper_candidate.md`
- Keep `references.bib` and `reproducibility.md` behavior.
- Record all outputs through artifact manager.

- [x] **Step 4: Verify**

Run:

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_manuscript_writer
```

Expected: all tests pass.

---

## Task 9: Review Planner

**Files:**
- Create: `agents/review_planner.py`
- Modify: `web/app.py` only if adding route is necessary
- Test: `tests/test_review_planner.py`
- Test: `tests/test_web_experiment_routes.py`

- [x] **Step 1: Write planner tests**

Test:

```python
def test_review_planner_turns_required_experiments_into_followup_plan(tmp_path):
    write_json(tmp_path / "artifacts/reviews/review.json", {
        "recommendation": "reject",
        "required_experiments": ["Run 10 seeds.", "Add ablation."],
        "major_concerns": ["No statistics."]
    })
    plan = plan_followup_experiments(run_id=3, workdir=tmp_path)
    assert plan["status"] == "needs_followup"
    assert len(plan["experiments"]) == 2
```

- [x] **Step 2: Run tests and verify failure**

Run:

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_review_planner
```

Expected: import failure.

- [x] **Step 3: Implement planner**

Public functions:

```python
def plan_followup_experiments(run_id: int, workdir: Path | None = None) -> dict:
    ...
```

Behavior:

- Read `artifacts/reviews/review.json`.
- If missing, return `{"status": "error", "reason": "review_not_found"}`.
- Convert required experiments to:

```json
{
  "schema_version": 1,
  "run_id": 3,
  "status": "needs_followup",
  "experiments": [
    {
      "source": "ai_review",
      "description": "Run 10 seeds.",
      "suggested_artifact": "benchmark_config.json"
    }
  ]
}
```

- Write `artifacts/results/followup_experiment_plan.json`.
- Record artifact.

- [x] **Step 4: Add route only if useful to UI**

Add:

```python
@app.route("/api/experiments/<int:run_id>/plan_followup", methods=["POST"])
def api_plan_followup(run_id):
    from agents.review_planner import plan_followup_experiments
    result = plan_followup_experiments(run_id)
    return jsonify(result), 200 if result.get("status") != "error" else 400
```

This route is allowed because no existing route performs review-to-followup planning. It must not create a new project object.

- [x] **Step 5: Verify**

Run:

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_review_planner tests.test_web_experiment_routes
```

Expected: all tests pass.

---

## Task 10: Dashboard Display

**Files:**
- Modify: `web/static/js/app.js`
- Test: `tests/test_web_experiment_routes.py`

- [x] **Step 1: Add route-level test for artifact visibility**

Test:

```python
def test_experiment_detail_includes_evidence_gate_and_followup_artifacts():
    response = client.get("/api/experiments/5")
    payload = response.get_json()
    paths = {artifact["path"] for artifact in payload["artifacts"]}
    assert "artifacts/results/evidence_gate.json" in paths
```

- [x] **Step 2: Update UI only from existing detail payload**

Display:

- `manuscript_status` if `evidence_gate.json` artifact exists.
- `followup_experiment_plan.json` artifact if exists.
- Keep current artifact list; do not add a new top-level tab.

- [x] **Step 3: Verify route tests**

Run:

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_web_experiment_routes
```

Expected: all tests pass.

- [x] **Step 4: Browser smoke test**

Start/reload app and verify:

- Experiment detail shows benchmark/statistical/evidence/review artifacts.
- No existing Experiments tab behavior regresses.

---

## Task 11: End-to-End Generic Pipeline Test

**Files:**
- Modify: `tests/test_e2e_toy_research_pipeline.py`

- [x] **Step 1: Add E2E test for fairness spec path**

Test flow:

```python
forge_experiment(insight_id)
run_validation_loop(run_id)
process_completed_run(run_id)
generate_manuscript(run_id)
review_manuscript(run_id)  # mocked LLM
plan_followup_experiments(run_id)
```

Assertions:

- `research_spec.json` exists.
- `benchmark_config.json` exists.
- `benchmark_results.json` exists.
- `statistical_report.json` exists.
- `evidence_gate.json` exists.
- manuscript output is not `paper_candidate.md` unless gate passes.
- review and follow-up plan artifacts exist.

- [x] **Step 2: Run E2E test**

Run:

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_e2e_toy_research_pipeline
```

Expected: all tests pass.

- [x] **Step 3: Run full test suite**

Run:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests
```

Expected: all tests pass.

---

## Task 12: Documentation Update

**Files:**
- Modify: `README.md`
- Modify: `HANDOFF.md`
- Modify: `docs/2026-04-29-progress-and-roadmap.md`

- [x] **Step 1: Update README**

Document current workflow:

```text
deep_insight
  -> research_spec
  -> capability selection
  -> forge
  -> single-script or benchmark-suite validation
  -> statistical report
  -> evidence gate
  -> manuscript
  -> AI review
  -> follow-up plan
```

Explicitly state:

- The system can generate reports and paper candidates.
- It must not be treated as automatically producing publishable papers.
- Evidence gate and human review are required.

- [x] **Step 2: Update HANDOFF**

Include:

- New modules.
- How to run tests.
- How to inspect artifacts for a run.
- Known limitation: safe RL/CMDP capability remains planned, not implemented.

- [x] **Step 3: Update progress roadmap**

Mark generalized framework plan as created and link this plan file.

- [x] **Step 4: Verify**

Run:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests
```

Expected: all tests pass.

---

## Execution Guidance

Implement in this order:

1. Research spec compiler.
2. Capability registry.
3. Fairness benchmark harness.
4. Benchmark suite runner.
5. Statistical reporter.
6. Evidence gate.
7. Forge/validation integration.
8. Manuscript integration.
9. Review planner.
10. Dashboard display.
11. E2E test.
12. Docs.

Stop and reassess if any of these happen:

- A module needs a DB table that is not already present.
- A planned endpoint would duplicate an existing route.
- A capability cannot be tested offline.
- A manuscript path would label a proxy result as publishable.
- A test requires real remote LLM calls.

## Final Verification

Required command:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests
```

Expected:

```text
OK
```

Manual/browser smoke test:

- Open `http://127.0.0.1:8080/`.
- Inspect the latest experiment detail.
- Confirm artifacts include spec/config/results/stats/evidence/manuscript/review/follow-up as applicable.
- Confirm UI does not imply `paper_ready_candidate` unless `evidence_gate.json` says so.
