# CGGR Paper Artifact Gate

## Active Repaired Benchmark

- Full-scale method shard A: `experiment_runs.id=45`, `gpu_jobs.id=38`, `ssh:proxy.cn-south-1.gpu-instance.ppinfra.com:gpu0`
- Full-scale method shard B: `experiment_runs.id=46`, `gpu_jobs.id=39`, `ssh:proxy.cn-south-1.gpu-instance.ppinfra.com:gpu1`
- Remote workdirs: `/root/deepgraph-remote-worker/runs/run_45`, `/root/deepgraph-remote-worker/runs/run_46`
- Local workdirs: `C:\Users\Kemal\deepgraph_ideas\idea_13\experiments\main\runs\run_45`, `C:\Users\Kemal\deepgraph_ideas\idea_13\experiments\main\runs\run_46`
- Merged artifact target: `C:\Users\Kemal\deepgraph_ideas\idea_13\experiments\main\runs\run_47`
- Manuscript draft: `workspace/tmp/materialize_bundle_smoke_v2/main.tex`
- Monitor logs: `logs/monitor-run45-fullscale-shard-a.out.log`, `logs/monitor-run46-fullscale-shard-b.out.log`
- Auto merge watcher logs: `logs/watch-merge-run45-run46-to-run47-v8.out.log`, `logs/watch-merge-run45-run46-to-run47-v8.err.log`
- Read-only health watcher logs: `logs/live-health-run45-run46.out.log`, `logs/live-health-run45-run46.err.log`
- Merge command after both shards complete: `python scripts/merge_cggr_method_shards.py --out-workdir C:\Users\Kemal\deepgraph_ideas\idea_13\experiments\main\runs\run_47 C:\Users\Kemal\deepgraph_ideas\idea_13\experiments\main\runs\run_45 C:\Users\Kemal\deepgraph_ideas\idea_13\experiments\main\runs\run_46`
- Audit command: `python scripts/audit_paper_benchmark_artifacts.py C:\Users\Kemal\deepgraph_ideas\idea_13\experiments\main\runs\run_47 --require-full`
- Materialize command: `python scripts/materialize_audited_cggr_results.py C:\Users\Kemal\deepgraph_ideas\idea_13\experiments\main\runs\run_47 --out-dir workspace\tmp\materialize_bundle_smoke_v2\audited_results`

## Claim Policy

No performance claim may enter the abstract, introduction, results, discussion, conclusion, or claim register until `benchmark_artifact_manifest.json` reports `full_benchmark_completed=true` and the evidence below is present.

Smoke, sanity, bootstrap, reproduction-only, and synthetic results are infrastructure evidence only. They may not populate the main table or headline claims.

Runs produced before the repaired selective-deliberation runner are invalid paper evidence. In particular, run 32 and run 35 used a CGGR prompt/gating path that could route low-budget CGGR branches through a reasoning prompt with an insufficient decode cap. Run 34 is recipe-blocked because unsupported Spider/SAMSum aliases were mapped through GSM8K loaders.

## Required Evidence Map

| Manuscript element | Required artifact evidence | Gate |
| --- | --- | --- |
| Abstract performance sentence | `benchmark_artifact_manifest.json`, `main_results_table.json`, `bootstrap_ci.json` | Must cite completed run id and CI |
| Main results table | `main_results_table.json`, `per_dataset_results.json`, `per_seed_results.json` | Must include all non-oracle deployable baselines |
| Dataset coverage sentence | `run_config.json`, `benchmark_manifest.json`, dataset load markers in `run.log` | Must show MuSiQue-Ans, StrategyQA, 2WikiMultihopQA, derived stress split |
| Baseline fairness sentence | `run_config.json`, method list in `run.log`, prompt/method config hashes if produced | Same model, split, seed, and decode budget policy |
| Cost/latency claim | `latency_tokens_table.json`, raw per-example token/latency fields | No cost claim from aggregate quality alone |
| Simple-vs-hard claim | `difficulty_breakdown_table.json`, derived stress split artifacts | Must report simple degradation and hard-set gains separately |
| Routing mechanism claim | `routing_analysis.json`, `routing_decisions.jsonl` | Must include per-example trigger score, decision, and final method |
| Routing health audit | `routing_decisions.jsonl`, `raw_predictions.jsonl`, audit diagnostics | Blocks CGGR zero-deliberation collapse, duplicate/overlapping raw cells, and flags all-route or token-cap saturation |
| Ablation claim | `ablation_table.json` | Must include CGGR ablations and identify removed component |
| Robustness/significance claim | `bootstrap_ci.json`, paired permutation output, seed variance table | Must report uncertainty beside the mean |
| Failure analysis | `failure_cases.jsonl`, parse failure counts | Failures must remain in denominator or be explicitly audited |
| Reproducibility statement | `environment_report.json`, `run_config.json`, `remote_job_manifest` fields if present | Must include GPU, CUDA, package versions, command, env vars |

## Current Run 45/46 Observations

- The remote log shows real benchmark target loading for `MuSiQue-Ans`, `StrategyQA`, and `2WikiMultihopQA`.
- The derived stress split is included in the declared target list and depends on loaded benchmark examples.
- `raw_predictions.jsonl` and `run_config.json` exist remotely while the run is active.
- Both shards use `DEEPGRAPH_BENCHMARK_FULL_RUN=1`; the remote `BENCHMARK_STAGE: start` lines show `max_examples=128` and `seed_values=[0,1,2,3,4]`.
- The runner uses the repaired chat-template encoding path, fail-fast generation errors, task-aware multihop difficulty prior, and low-budget selective direct-answer prompt.
- The manuscript draft has been sanitized so old smoke/reproduction utility numbers are no longer used as paper evidence.
- 2026-05-09 19:56 +08:00 live checkpoint: `run_45` has 2,981 raw rows and 417 CGGR-only failure rows with no error field; `run_46` has 6,180 raw rows and 0 failure rows. The latest partial health audit passed the bad-JSON, duplicate-key, empty-prediction, and zero-token checks for both shards. DB still shows `run_45=running_gpu`, `run_46=running_gpu`, and `run_47=pending/merge_pending`.

## Parallel Shard Policy

- Shard A methods: `Vanilla Direct Answering`, `Always-Reason Chain-of-Thought`, `Self-Consistency Reasoning`, `CGGR`.
- Shard B methods: `Least-to-Most Prompting`, `Confidence Gate`, `Disagreement Routing`, `Random Budget-Matched Routing`, `CGGR/no_counterfactual_delta`, `CGGR/no_lcb`, `CGGR/no_self_divergence_penalty`, `CGGR/no_qstruct_term`.
- Shard configs: `run_45\spec\shard_config.json`, `run_46\spec\shard_config.json`.
- Merge config: `run_47\spec\merge_config.json`.
- Merge script: `scripts/merge_cggr_method_shards.py`.
- Oracle routing is not part of the current locked run contract. Any future label-aware oracle diagnostic must be contract-registered before execution and separated from deployable-method claims.

Runs 45 and 46 are full-scale method shards. Neither shard is by itself a complete benchmark. A merged artifact may be used only after the merge script verifies complete method coverage and the full artifact audit passes.

The merge and audit gates reject duplicate `method + dataset + seed + example_id` rows and any method/dataset/seed cell with more than 128 rows. Extra shards may only be merged if they are non-overlapping at the raw example level.

## Invalidated Runs

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

## Completion Audit Checklist

- [ ] `scripts/audit_paper_benchmark_artifacts.py <run_workdir> --require-full` exits successfully for the final audited artifact package.
- [ ] `run_45` and `run_46` have no active remote `python train.py` process.
- [ ] `gpu_jobs.id=38` and `gpu_jobs.id=39` are `completed`.
- [ ] The merged artifact, not either shard alone, has `full_benchmark_completed=true`.
- [ ] `benchmark_artifact_manifest.json` exists locally and remotely.
- [ ] Manifest has `full_benchmark_completed=true`.
- [ ] All required result files in the evidence map exist and are non-empty.
- [ ] `raw_predictions.jsonl` covers every required method x dataset x seed x 128-example cell, not only aggregate summaries.
- [ ] `raw_predictions.jsonl` has no duplicate `method + dataset + seed + example_id` rows and no cell above the 128-example contract.
- [ ] `bootstrap_ci.json` includes candidate/baseline CIs and paired permutation p-value.
- [ ] `benchmark_summary.json` includes `per_method_std` for required methods.
- [ ] `simple_case_degradation.json` and `calibration_reliability.json` contain numeric diagnostics.
- [ ] Materialization writes `results_section_snippet.tex`, `limitations_snippet.tex`, `main_results_table.tex`, `cost_latency_table.tex`, `ablation_table.tex`, `utility_comparison_figure.tex`, `completion_audit.md`, and `claim_values.json` with any audit warnings; manuscript prose/tables/figures must use these guarded files instead of hand-copied JSON numbers.
- [ ] `workspace/tmp/materialize_bundle_smoke_v2/main.tex` compiles with missing audited files and automatically inputs the audited snippets/tables/figure after materialization; no placeholder comparison plot is included.
- [ ] Dataset names in result files match loaded benchmark names; no unsupported alias is mapped to GSM8K.
- [ ] `scripts/merge_cggr_method_shards.py` exits successfully and writes a merged manifest naming `run_45` and `run_46`.
- [ ] Main manuscript table is populated only from audited result files.
- [ ] Every empirical claim has a run id and artifact path.
- [ ] Limitations report any missing cells, failed examples, retry history, and cost constraints.
