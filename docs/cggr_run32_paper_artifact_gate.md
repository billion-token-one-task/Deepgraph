# CGGR Paper Artifact Gate

## Active Completed Artifact

- Core method shard A: `experiment_runs.id=45`, `gpu_jobs.id=38`, completed at 2026-05-10 07:04.
- Core method shard B: `experiment_runs.id=46`, `gpu_jobs.id=39`, completed at 2026-05-10 06:52.
- Interrupted top-venue baseline shard: `experiment_runs.id=48`, `gpu_jobs.id=40`, failed before writing `benchmark_summary.json`; not citable by itself.
- Continuation top-venue shards: `experiment_runs.id=50` and `experiment_runs.id=51`, completed missing seeds.
- Repaired top-venue shard: `experiment_runs.id=52`, completed with 7,680 raw rows and 0 duplicates dropped.
- Stale-output merge attempt: `experiment_runs.id=53`, failed because the target results directory was not empty; not evidence.
- Final strict merged artifact: `experiment_runs.id=54`, completed at 2026-05-10 14:13, phase `strict_top_venue_full_benchmark_merged`.
- Final workdir: `C:\Users\Kemal\deepgraph_ideas\idea_13\experiments\main\runs\run_54`.
- Materialized evidence bundle: `workspace/tmp/materialize_bundle_topvenue_repair_v2`.

## Repro Commands

- Strict audit:
  `python scripts/audit_paper_benchmark_artifacts.py C:\Users\Kemal\deepgraph_ideas\idea_13\experiments\main\runs\run_54 --require-full --require-top-venue-baselines`
- Strict materialization:
  `python scripts/materialize_audited_cggr_results.py C:\Users\Kemal\deepgraph_ideas\idea_13\experiments\main\runs\run_54 --out-dir workspace\tmp\materialize_bundle_topvenue_repair_v2 --require-top-venue-baselines`
- Failure triage, if a future artifact is rejected:
  `python scripts/triage_cggr_audit_failure.py <run_workdir> --require-full [--require-top-venue-baselines]`

## Claim Policy

No performance claim may enter the abstract, introduction, results, discussion, conclusion, or claim register unless it is tied to a completed merged artifact with `full_benchmark_completed=true`, complete raw prediction coverage, and a passing audit.

Smoke, sanity, bootstrap, reproduction-only, partial, failed, and interrupted runs are infrastructure evidence only. They may not populate the main table or headline claims.

If CGGR underperforms or lacks statistical support in a completed artifact, the pipeline must not overwrite that artifact or tune thresholds on the final benchmark. It must preserve the result, write failure triage where needed, and register a new preregistered v2 run before changing the method.

Broad top-venue/SOTA-style claims require adaptive-reasoning comparators plus a passing `--require-top-venue-baselines` audit. `run_54` satisfies the strict comparator coverage gate, but the materialized claim audit still reports `claim_support_decision=downgraded` because the paired permutation p-value is `0.0625`. Phrase claims as scoped audited utility improvements, not as unqualified state-of-the-art dominance.

Route-to-Reason-style joint model/strategy routing is nearby work that must be acknowledged. Direct superiority over any specific external system still requires a registered comparable artifact or an explicit limitation.

## Required Evidence Map

| Manuscript element | Required artifact evidence | Gate |
| --- | --- | --- |
| Abstract performance sentence | `benchmark_artifact_manifest.json`, `main_results_table.json`, `bootstrap_ci.json`, `claim_values.json` | Must cite completed run id and claim decision |
| Main results table | `main_results_table.tex`, `per_dataset_results.json`, `per_seed_results.json` | Must include all non-oracle deployable baselines |
| Dataset coverage sentence | `run_config.json`, dataset load markers, audit diagnostics | Must show MuSiQue-Ans, StrategyQA, 2WikiMultihopQA, derived stress split |
| Baseline fairness sentence | `run_config.json`, method list, prompt/method config | Same model, split, seed, and decode-budget policy |
| Cost/latency claim | `latency_tokens_table.json`, `cost_latency_table.tex`, raw token/latency fields | No cost claim from aggregate quality alone |
| Simple-vs-hard claim | `difficulty_breakdown_table.json`, `simple_case_degradation.json` | Must report simple degradation and hard-set gains separately |
| Routing mechanism claim | `routing_analysis.json`, `routing_decisions.jsonl` | Must include per-example trigger score, decision, and final method |
| Routing health audit | `routing_decisions.jsonl`, `raw_predictions.jsonl`, audit diagnostics | Blocks global CGGR zero-deliberation collapse; flags per-slice all-route/no-route and token-cap saturation |
| Ablation claim | `ablation_table.json`, `ablation_table.tex` | Must include CGGR ablations and identify removed component |
| Robustness/significance claim | `bootstrap_ci.json`, `significance_report.md`, seed variance table | Must report uncertainty beside the mean |
| Failure analysis | `failure_cases.jsonl`, `failure_analysis.md` | Failures must remain in denominator or be explicitly audited |
| Reproducibility statement | `environment_report.json`, `run_config.json`, `reproducibility_statement.md` | Must include GPU, CUDA, package versions, command, env vars |

## Current Result Summary

- `run_54` passed the strict full artifact audit with `raw_predictions_lines=38400`, `full_benchmark_completed=true`, and no blockers.
- Required datasets are present: MuSiQue-Ans, StrategyQA, 2WikiMultihopQA, and the stress split.
- Required methods are present: vanilla direct, Always-Reason CoT, Self-Consistency, Least-to-Most, Confidence Gate, Disagreement Routing, Random Budget-Matched Routing, CGGR, CGGR ablations, and the strict top-venue-style CAR/Self-Route/VOC baselines.
- CGGR utility is `0.3853536374162394`; Vanilla Direct Answering is `0.3065925870472178`; Confidence Gate is `0.3048047572495335`; CAR-style is `0.3064268813193493`; Self-Route-style and VOC-style are both `0.30012929284641354`.
- CGGR beats the locked and strict top-venue-style deployable baselines on the audited utility metric.
- `claim_values.json` records `claim_support_decision=downgraded`; the result is positive but should be written with guarded statistical language.

## Invalidated Or Non-Citable Runs

| Run | Status | Reason |
| --- | --- | --- |
| `run_32` / `gpu_jobs.id=28` | Invalid | Pre-repair CGGR prompt/gating path; do not cite for paper results. |
| `run_35` / `gpu_jobs.id=30` | Invalid | Manual shard copied from the pre-repair runner; do not merge into audited evidence. |
| `run_34` / `gpu_jobs.id=29` | Recipe blocked | Spider/SAMSum were loaded through unsupported GSM8K aliases; must not relaunch without explicit benchmark recipes. |
| `run_37` / `gpu_jobs.id=32` | Invalid | Automatic scheduler relaunched the shard with the default method subset and overwrote repaired shard artifacts. |
| `run_36` / `gpu_jobs.id=31` | Invalid | Prompt truncation/routing health issue after repair attempt; do not cite. |
| `run_38` / `gpu_jobs.id=33` | Invalid | Prompt truncation/routing health issue after repair attempt; do not merge. |
| `run_39` / `gpu_jobs.id=34` | Invalid | Chat-template `BatchEncoding` bug caused swallowed generation exceptions, empty predictions, and zero-token rows. |
| `run_40` / `gpu_jobs.id=35` | Invalid | Same empty-generation failure as run 39. |
| `run_42` | Invalid | Forge timed out before writing `train.py`. |
| `run_43` / `gpu_jobs.id=36` | Invalid | Runtime safety-capped to smoke scale because `DEEPGRAPH_BENCHMARK_FULL_RUN=1` was missing. |
| `run_44` / `gpu_jobs.id=37` | Invalid | Runtime safety-capped to smoke scale because `DEEPGRAPH_BENCHMARK_FULL_RUN=1` was missing. |
| `run_48` / `gpu_jobs.id=40` | Partial only | Interrupted top-venue shard; complete rows were reused only through the registered `run_52` repair path. |
| `run_53` | Failed merge target | Stale local output directory guard triggered; not evidence. |

## Completion Checklist

- [x] Full core shards completed.
- [x] Interrupted top-venue shard repaired through registered continuation shards.
- [x] Strict merged artifact has `full_benchmark_completed=true`.
- [x] `raw_predictions.jsonl` covers every required method x dataset x seed x 128-example cell.
- [x] No duplicate raw prediction rows in the merged artifact.
- [x] `scripts/audit_paper_benchmark_artifacts.py <run_54> --require-full --require-top-venue-baselines` exits successfully.
- [x] Materialization writes `results_section_snippet.tex`, `limitations_snippet.tex`, `main_results_table.tex`, `cost_latency_table.tex`, `ablation_table.tex`, `utility_comparison_figure.tex`, `significance_report.md`, `failure_analysis.md`, `reproducibility_statement.md`, `completion_audit.md`, and `claim_values.json`.
- [x] DB points the active auto-research job to `experiment_run_id=54`.
- [ ] Any final manuscript submission bundle must use the `run_54` materialized evidence files and keep the downgraded claim decision visible.
- [ ] Any new stronger claim requires a new preregistered run rather than post-hoc edits to `run_54`.
