# CGGR Paper-Grade Experiment Agent Playbook

This playbook is the operating contract for turning CGGR from smoke-test evidence into a paper-grade benchmark package. It is intentionally artifact-first: an agent may write prose only after the required artifacts exist and pass audit.

Current completion audit checkpoint: `docs/cggr_current_completion_audit.md`.

Top-venue novelty checkpoint: `docs/cggr_top_venue_novelty_gap.md`.

## Current Canonical Runs

| Role | Run | GPU job | Worker | Status | Use in paper |
| --- | --- | --- | --- | --- | --- |
| Full-scale method shard A | `run_45` | `gpu_jobs.id=38` | `ssh:proxy.cn-south-1.gpu-instance.ppinfra.com:gpu0` | running | Not directly citable; merge required |
| Full-scale method shard B | `run_46` | `gpu_jobs.id=39` | `ssh:proxy.cn-south-1.gpu-instance.ppinfra.com:gpu1` | running | Not directly citable; merge required |
| Merged artifact target | `run_47` | none | local | waiting | Only citable if merge and full audit pass |
| Canary only | `run_41` | none | `ssh:proxy.cn-south-1.gpu-instance.ppinfra.com:gpu0` | completed | Infrastructure evidence only |

Automatic merge watcher:

- Script: `scripts/watch_and_merge_cggr_shards.py --shard-run-id 45 --shard-run-id 46 --merged-run-id 47 --materialize-out-dir workspace\tmp\materialize_bundle_smoke_v2\audited_results --poll-seconds 300 --compile-tex workspace\tmp\materialize_bundle_smoke_v2\main.tex`
- Logs: `logs/watch-merge-run45-run46-to-run47-v13.out.log`, `logs/watch-merge-run45-run46-to-run47-v13.err.log`
- Contract: the watcher must merge only after both shard runs and both GPU jobs are completed, then run the full artifact audit before materializing manuscript tables and compiling the manuscript with audited snippets/tables.

Read-only live-health watcher:

- Script: `scripts/watch_cggr_live_health.py --worker-id ssh:proxy.cn-south-1.gpu-instance.ppinfra.com:gpu0 --run-id 45 --run-id 46 --poll-seconds 600`
- Logs: `logs/live-health-run45-run46.out.log`, `logs/live-health-run45-run46.err.log`
- Contract: samples GPU utilization, DB state, raw JSONL health, failure rows, remote live PID counts, stage tails, and error tails. It does not write DB state or benchmark artifacts, and it does not replace the final run47 audit.

Prompt gate update:

- `prompts/benchmark_orchestra/system_orchestrator.md`, `stats_audit.md`, and `manuscript_writer.md` now explicitly require a requirement-to-artifact completion audit before claim support, a failure-analysis gate before discussing failure modes, and a top-venue gate before any SOTA/general adaptive-reasoning claim.
- Green tests, completed manifests, successful verifier output, and substantial implementation effort are implementation evidence only; they are not paper-completion proxies unless the audit covers every claim requirement.

Top-venue prior-art gate:

- The current locked run can support claims against its registered deployable baselines only after `run_47` passes full audit.
- It cannot support broad first-method, state-of-the-art, or superiority-over-all-adaptive-reasoning claims.
- Before a top-venue submission, the paper must acknowledge rational metareasoning/value-of-computation, certainty-based adaptive routing, Self-Route-style mode routing, Route-to-Reason-style joint model/strategy routing, and cost-quality model routing; direct empirical superiority over those methods requires registered benchmark cells and audited artifacts.
- Optional runner support exists for the first three directly comparable adaptive-reasoning baselines. Set `DEEPGRAPH_BENCHMARK_INCLUDE_TOP_VENUE_BASELINES=1` or explicitly request the labels `CAR-Style Certainty Adaptive Routing`, `Self-Route-Style Mode Routing`, and `Rational-Metareasoning VOC Routing`; then audit the merged artifact with `--require-top-venue-baselines`.
- A local template can be prepared from the current benchmark contract with `scripts/prepare_cggr_top_venue_baseline_shard.py`; the current prepared template is `workspace/tmp/cggr_top_venue_baseline_shard_template`. The preparation script must sanitize stale learned-router wording, mark the active evidence as fixed proxy-gated, disable trained-estimator/router claims, and filter oracle upper-bound diagnostics out of deployable baseline requirements. This is a launch-ready template, not paper evidence until run, merged, and audited.

Invalid evidence:

| Run | Reason |
| --- | --- |
| `run_32` | Pre-repair CGGR prompt/gating path. |
| `run_35` | Manual shard copied from the pre-repair runner. |
| `run_34` | Recipe-blocked benchmark alias issue: Spider/SAMSum through GSM8K loaders. |
| `run_37` | Scheduler overwrote the repaired shard with a default method subset. |
| `run_36` / `run_38` | Invalidated after prompt truncation/routing health check. |
| `run_39` / `run_40` | Invalidated after audit found all predictions empty from chat-template `BatchEncoding` mishandling. |
| `run_42` | Forge timeout before `train.py` was written. |
| `run_43` / `run_44` | Invalidated because launch missed `DEEPGRAPH_BENCHMARK_FULL_RUN=1` and runtime safety-capped to smoke scale. |

## Agent Roles

### 1. Benchmark Contract Agent

Responsibility: freeze the benchmark contract before any result claim is allowed.

Inputs:
- `spec/benchmark_manifest.json`
- `spec/experiment_spec.json`
- `workspace/tmp/materialize_bundle_smoke_v2/claim_evidence_map.json`
- current DB rows for `experiment_runs` and `gpu_jobs`

Outputs:
- locked dataset/model/method/seed/metering contract
- documented invalid runs and forbidden evidence
- updated claim evidence map

Prompt:

```text
You are the Benchmark Contract Agent for CGGR. Freeze the benchmark contract for paper evidence. Verify dataset identities, splits, model checkpoint, seed policy, max examples, baselines, ablations, metrics, and artifact requirements. Mark smoke tests, recipe-blocked runs, pre-repair runs, and unmerged shards as forbidden evidence. Do not report performance numbers. Produce only contract updates, blockers, and required artifacts.
```

Done when:
- `docs/cggr_run32_paper_artifact_gate.md` names the active canonical run and invalid evidence.
- `claim_evidence_map.json` has every empirical claim blocked until audit.
- No unsupported alias such as Spider/SAMSum-through-GSM8K is active.

### 2. Remote Run Controller Agent

Responsibility: keep GPU execution real, monitored, and isolated from scheduler collisions.

Inputs:
- `gpu_jobs`, `gpu_workers`, `experiment_runs`
- remote process table under `/root/deepgraph-remote-worker/runs/`
- monitor logs under `logs/`

Outputs:
- two full-scale method shards on GPU0/GPU1
- a merged full benchmark package only after both shards complete
- no active invalid run processes
- DB state that matches remote reality

Prompt:

```text
You are the Remote Run Controller Agent for CGGR. Maintain exactly the intended benchmark processes on the SSH GPU host. Verify cwd, GPU id, method subset, run.log stage, raw row counts, and DB state. Kill and mark failed any recipe-blocked, pre-repair, scheduler-overwritten, or wrong-method run. Never create a queued gpu_job for a manual shard; insert the running job only after the remote process is verified. Record monitor paths and errors.
```

Done when:
- `run_45` and `run_46` have live remote `python train.py` processes in the expected cwd.
- `run_34`, `run_37`, and other invalid runs have no live process.
- monitors are running and DB rows match process state.

### 3. Runner Quality Agent

Responsibility: inspect the runner for benchmark validity, not just successful execution.

Inputs:
- generated `code/train.py`
- `raw_predictions.jsonl`
- `routing_decisions.jsonl`
- `run_config.json`
- `environment_report.json`

Outputs:
- runner repair patches
- audit checks for routing collapse, sharding, missing environment, and token truncation
- unit tests proving the checks catch known failure modes

Prompt:

```text
You are the Runner Quality Agent for CGGR. Audit the benchmark runner for paper-grade validity. Check that low-budget selective branches use the direct-answer prompt, multihop examples can route to deliberation, sharded runs cannot satisfy the full gate, environment reports are present, and routing diagnostics catch CGGR zero-deliberation collapse. Add focused tests for every failure mode you repair. Do not improve headline numbers by changing the benchmark contract after results are known.
```

Done when:
- `tests.test_experiment_forge`, `tests.test_paper_benchmark_audit`, and relevant scheduler tests pass.
- `scripts/audit_paper_benchmark_artifacts.py` blocks incomplete, sharded, missing-environment, duplicate/overlapping raw-cell, empty-generation, and CGGR-zero-routing artifacts.
- `scripts/merge_cggr_method_shards.py` can merge non-overlapping full-scale method shards into a single auditable package and rejects duplicate `method + dataset + seed + example_id` rows.
- the merged manifest lists only methods that were actually present in the locked run contract; oracle diagnostics require a future pre-registered contract.
- the merged package reports per-seed method variance, paired bootstrap CIs, paired permutation p-value, full method/dataset/seed raw coverage, and non-empty simple-case/calibration diagnostics.

### 4. Evidence Auditor Agent

Responsibility: decide whether artifacts are allowed to enter the manuscript.

Inputs:
- final run workdir
- `scripts/audit_paper_benchmark_artifacts.py`
- `scripts/materialize_audited_cggr_results.py`
- claim evidence map

Outputs:
- audit JSON
- manuscript-ready tables only after audit passes
- blockers list otherwise

Prompt:

```text
You are the Evidence Auditor Agent for CGGR. Run the artifact audit against the final benchmark package with --require-full. Verify every required file, dataset, method, ablation, seed count, example count, routing trace, environment report, and routing diagnostic. If any blocker exists, refuse to materialize manuscript tables. If all checks pass, materialize tables and claim values with source workdir and run ids.
```

Done when:
- `scripts/audit_paper_benchmark_artifacts.py <final_workdir> --require-full` exits 0.
- `scripts/materialize_audited_cggr_results.py <final_workdir> --out-dir ...` writes tables, significance report, failure analysis, reproducibility statement, claim evidence map, completion audit, claim values, guarded manuscript snippets, and the watchdog contract/audit JSON files (`problem_awareness.json`, `publication_evidence_contract.json`, `evidence_manifest.json`, `claim_evidence_matrix.json`, `reviewer_report.json`, `paper_quality_report.json`) into the audited results bundle.
- `main.tex` inputs the materialized main-result, cost/latency, ablation tables, and utility-comparison figure snippet only after those audited files exist.
- non-blocking audit warnings are carried into `claim_values.json` and the limitations snippet so routing saturation or token-cap diagnostics cannot be silently dropped.
- every numeric claim in the manuscript has an artifact path and run id.

### 5. Manuscript Agent

Responsibility: write only claims that the evidence auditor has admitted.

Inputs:
- `main.tex`
- audited table files
- `claim_values.json`
- `claim_evidence_map.json`
- `completion_audit.md`
- references

Outputs:
- updated abstract, experiments, results, discussion, and conclusion
- no smoke-log metrics
- no unsupported superiority claims

Prompt:

```text
You are the Manuscript Agent for CGGR. Write the paper in a top-venue style, but only use numbers from audited artifacts. Keep conceptual claims separate from empirical claims. Insert result tables, confidence intervals, cost/latency accounting, routing analysis, ablations, limitations, and failure analysis only when each claim maps to an artifact id. If audit is incomplete, keep result cells blank and state that empirical claims are evidence-pending.
```

Done when:
- `main.tex` compiles.
- no smoke, synthetic, recipe-blocked, pre-repair, or unmerged-shard number appears in abstract/results/discussion/conclusion.
- claim map and manuscript agree.

## Required Gates

1. Service gate: DeepGraph responds at `http://127.0.0.1:8080/`.
2. Remote GPU gate: active runs have live processes in the expected remote cwd.
3. Recipe gate: unsupported aliases are blocked before launch.
4. Runner gate: repaired CGGR prompt/gating logic is present in generated code.
5. Artifact gate: all required result files exist and audit passes with `--require-full`.
6. Shard gate: a shard cannot satisfy the full gate by itself.
7. Routing gate: CGGR zero-deliberation collapse blocks paper evidence; all-route and token-cap saturation require inspection.
8. Statistical gate: raw rows must cover every required method/dataset/seed/example cell exactly once, and `bootstrap_ci.json`, `per_method_std`, `simple_case_degradation.json`, and `calibration_reliability.json` must be populated before manuscript materialization.
9. Manuscript gate: materialization script refuses to write result tables unless audit passes.
10. Top-venue novelty gate: `docs/cggr_top_venue_novelty_gap.md` must be resolved before claiming SOTA, first adaptive reasoning, or general superiority over current adaptive-reasoning work.
11. Top-venue baseline gate: for SOTA/general-superiority claims, `scripts/audit_paper_benchmark_artifacts.py --require-full --require-top-venue-baselines` must pass on a merged artifact that includes the CAR-style, Self-Route-style, and VOC-style baseline rows.
12. Top-venue materialization gate: broad adaptive-reasoning superiority claims remain blocked in `claim_values.json` unless `scripts/materialize_audited_cggr_results.py` is run with `--require-top-venue-baselines` on an artifact that passes the stricter audit.
13. Submission watchdog gate: `orchestrator/manuscript_watchdog.py` must block any submission bundle that asserts top-venue/SOTA/general adaptive-reasoning superiority while root-level `claim_values.json` or `audited_results/claim_values.json` reports `top_venue_general_superiority_decision` as blocked, must accept the six required contract/audit JSON files from either the bundle root or `audited_results/`, must block adaptive-reasoning/routing manuscripts that omit CAR, Self-Route, Rational Metareasoning, Route-to-Reason, or RouteLLM from the nearby prior-art acknowledgement, and must block ready bundles that still contain evidence-pending prose or blank result placeholders.

## Operational Commands

```powershell
# Service
Invoke-WebRequest -UseBasicParsing -Uri 'http://127.0.0.1:8080/' -TimeoutSec 10

# Merge completed full-scale method shards, then audit the merged package
.\.venv\Scripts\python.exe scripts\merge_cggr_method_shards.py --out-workdir C:\Users\Kemal\deepgraph_ideas\idea_13\experiments\main\runs\run_47 C:\Users\Kemal\deepgraph_ideas\idea_13\experiments\main\runs\run_45 C:\Users\Kemal\deepgraph_ideas\idea_13\experiments\main\runs\run_46
.\.venv\Scripts\python.exe scripts\audit_paper_benchmark_artifacts.py C:\Users\Kemal\deepgraph_ideas\idea_13\experiments\main\runs\run_47 --require-full

# Stricter top-venue audit after the optional adaptive-reasoning baseline shard is merged
.\.venv\Scripts\python.exe scripts\audit_paper_benchmark_artifacts.py <merged_workdir> --require-full --require-top-venue-baselines

# Prepare a local top-venue baseline shard template from the current contract
.\.venv\Scripts\python.exe scripts\prepare_cggr_top_venue_baseline_shard.py --source-run C:\Users\Kemal\deepgraph_ideas\idea_13\experiments\main\runs\run_45 --out-workdir workspace\tmp\cggr_top_venue_baseline_shard_template --force

# Manuscript table materialization
.\.venv\Scripts\python.exe scripts\materialize_audited_cggr_results.py C:\Users\Kemal\deepgraph_ideas\idea_13\experiments\main\runs\run_47 --out-dir workspace\tmp\materialize_bundle_smoke_v2\audited_results

# Strict top-venue materialization, only after the top-venue baseline artifact passes the stricter audit
.\.venv\Scripts\python.exe scripts\materialize_audited_cggr_results.py <merged_workdir> --out-dir <audited_results_dir> --require-top-venue-baselines

# Focused verification
.\.venv\Scripts\python.exe -m unittest tests.test_vnext_gpu_scheduler tests.test_paper_benchmark_audit tests.test_cggr_shard_contract_audit tests.test_experiment_forge tests.test_merge_cggr_method_shards
.\.venv\Scripts\python.exe -m unittest tests.test_watch_and_merge_cggr_shards
```
