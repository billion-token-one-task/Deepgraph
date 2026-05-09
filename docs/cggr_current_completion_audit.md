# CGGR Current Completion Audit

This is a live audit checkpoint for the active goal. It is not a completion certificate.

## Objective Restated As Deliverables

1. DeepGraph service is running and reachable.
2. Cloud GPU execution is connected and running real benchmark jobs, not smoke tests.
3. The experiment uses a paper-grade benchmark contract with real datasets, locked methods, seeds, examples, and full artifact retention.
4. Experiment agents/prompts have clear responsibilities and forbid smoke-test evidence in claims.
5. The paper/manuscript path can only consume audited benchmark artifacts.
6. Final empirical claims are supported by merged, audited, materialized artifacts with raw coverage, uncertainty, cost, routing, ablation, and reproducibility evidence.

## Prompt-To-Artifact Checklist

| Requirement | Evidence inspected | Current status | Remaining blocker |
| --- | --- | --- | --- |
| Service started | `Invoke-WebRequest http://127.0.0.1:8080/` returned 200 in the latest service check. | Satisfied for current session | Recheck before final completion. |
| Cloud GPU connected | Remote SSH worker reports two NVIDIA L40S GPUs and live `python train.py` compute apps with cwd `/root/deepgraph-remote-worker/runs/run_45/code` and `/root/deepgraph-remote-worker/runs/run_46/code`. | Satisfied for current session | Recheck before final completion. |
| Real benchmark, not smoke | `run_45`/`run_46` launched with `DEEPGRAPH_BENCHMARK_FULL_RUN=1`, `max_examples=128`, `seed_values=[0,1,2,3,4]`, Qwen2.5-14B-Instruct, method shards. | In progress | Shards are not complete and cannot support claims individually. |
| Canonical run IDs | DB: `experiment_runs.id=45/46` are `running_gpu`; `gpu_jobs.id=38/39` are `running`; `run_47` is `merge_pending`. | In progress | Need jobs 38/39 completed and run47 merged. |
| Invalid evidence blocked | `docs/cggr_run32_paper_artifact_gate.md` and `claim_evidence_map.json` list invalid runs 32, 34-40, 42-44 and canary 41 as non-claim evidence. | Satisfied | Recheck no invalid process/artifact is merged. |
| Agent split and prompts | `prompts/benchmark_orchestra/*` defines Contract, Dataset/Baseline, Harness, Remote GPU, Method, Stats/Audit, Manuscript roles. The orchestrator, Stats/Audit, and Manuscript prompts require raw coverage, paired stats, snippets, claim decisions, and an explicit requirement-to-artifact audit rather than proxy completion signals. | Satisfied | Keep prompts aligned if gate changes again. |
| Merge automation | `scripts/watch_and_merge_cggr_shards.py` watcher is running with v8 logs and waits for run45/run46 plus GPU jobs before merge/audit/materialize/manuscript compile. | In progress | Watcher must complete successfully after shards finish. |
| Live health monitoring | `scripts/watch_cggr_live_health.py` is running read-only with logs at `logs/live-health-run45-run46.out.log` and `logs/live-health-run45-run46.err.log`, sampling GPU utilization, DB state, raw JSONL health, failure rows, live remote PIDs, stage tails, and error tails. | In progress | Continue until shards complete; health samples do not replace final run47 audit. |
| Merge artifact quality | `scripts/merge_cggr_method_shards.py` validates method/dataset/seed/example coverage, blocks duplicate or overlapping raw cells, recomputes per-method/per-seed/per-dataset stats, cost, routing, simple-case, calibration, ablation, bootstrap, and manifests. | Implemented and tested | Needs real run45/run46 inputs. |
| Artifact audit quality | `scripts/audit_paper_benchmark_artifacts.py --require-full` blocks sharded artifacts, missing required files, missing datasets/methods/ablations, low seed/example count, raw coverage gaps, duplicate/overlapping raw cells, empty generations, zero-token generations, generation failures, CGGR zero-routing, missing stats, and missing simple/calibration diagnostics. | Implemented and tested | Must pass on run47. |
| Manuscript materialization | `scripts/materialize_audited_cggr_results.py` refuses to write unless full audit passes; writes tables, significance report, reproducibility statement, claim evidence map, completion audit, claim values with audit warnings, result/limitations snippets, and an audited utility-comparison figure snippet. | Implemented and tested | Must run on audited run47. |
| Manuscript integration | `workspace/tmp/materialize_bundle_smoke_v2/main.tex` compiles now, auto-inputs audited result/limitations snippets, materialized main/cost/ablation tables, and the audited utility-comparison figure snippet if present. | Satisfied for current draft | Needs audited snippets/tables/figure generated after run47. |
| Verification suite | Focused suite passed: `tests.test_experiment_forge`, `tests.test_paper_benchmark_audit`, `tests.test_cggr_shard_contract_audit`, `tests.test_vnext_gpu_scheduler`, `tests.test_merge_cggr_method_shards`, `tests.test_watch_and_merge_cggr_shards` (39 tests). | Satisfied for code changes | Re-run after any script changes and after merge if needed. |

## Latest Verified Runtime Checkpoint

- Checkpoint time: 2026-05-09 19:56:37 +08:00.
- DeepGraph service check returned HTTP 200 with a 26,051-byte response.
- GPU host reports two active NVIDIA L40S devices with live utilization; the latest direct sample saw GPU0 at 13,837 MiB / 71% and GPU1 at 13,835 MiB / 70%.
- `run_45`: latest partial health audit saw 2,981 raw rows with no bad JSON, duplicate key, empty prediction, or zero-token row; `failure_cases.jsonl` has 417 rows, all sampled as `stage=<no_stage>`, `method=CGGR`, with no error field. The run completed seed 1, `StrategyQA`, `Self-Consistency Reasoning` and is advancing through `CGGR`.
- `run_46`: latest partial health audit saw 6,180 raw rows with no bad JSON, duplicate key, empty prediction, or zero-token row; `failure_cases.jsonl` has 0 rows. The run completed seed 1, `StrategyQA`, `CGGR/no_qstruct_term` and is advancing through seed 1, `2WikiMultihopQA`, `Least-to-Most Prompting`.
- Routing diagnostics remain non-silent: run45 CGGR has nonzero routing (`320/512` in the latest partial audit); run46 has expected warning-worthy saturation in some gate/ablation methods, which the audit/materializer path now carries into limitations via audit warnings.
- Remote error scan found no traceback, OOM, CUDA error, runtime exception, killed process, or NaN marker in the latest sampled log tails.
- DB state remains `run_45=running_gpu`, `run_46=running_gpu`, `run_47=pending/merge_pending`; `gpu_jobs.id=38/39` remain `running`.
- Watcher was restarted as v8 after materializer updates so it imports the current audited-table/figure/warning code. It completed a 2026-05-09 19:55 +08:00 poll with `run_45=running_gpu`, `run_46=running_gpu`, `gpu_jobs.id=38/39=running`; `logs/watch-merge-run45-run46-to-run47-v8.err.log` remains empty.
- Live local processes include the DeepGraph service (`python -u main.py`), both remote shard monitors, the v8 merge watcher with `--compile-tex workspace\tmp\materialize_bundle_smoke_v2\main.tex`, and the read-only live-health watcher.
- `scripts/watch_cggr_live_health.py` passed `python -m py_compile`, passed a one-shot sample, and then started as a background read-only logger. Its first logged sample at 2026-05-09 19:25 +08:00 had `remote.ok=true`, empty `error_tail`, zero bad JSON, zero duplicate keys, zero empty predictions, and zero zero-token rows for both active shards.
- Prompt gate update: `prompts/benchmark_orchestra/system_orchestrator.md`, `stats_audit.md`, and `manuscript_writer.md` now explicitly reject green tests, completed manifests, successful verifier output, or substantial effort as completion proxies unless a requirement-to-artifact audit covers the paper claim.
- Local verification after the manuscript/materializer update: `python -m py_compile scripts/materialize_audited_cggr_results.py`, `tests.test_merge_cggr_method_shards` (4 tests), LaTeX compile of `workspace/tmp/materialize_bundle_smoke_v2/main.tex`, 12 related tests, and the 39-test focused suite all passed. The latest 39-test focused suite pass included the new `completion_audit.md` materializer output.

## Not Complete

The goal is not complete until all of the following are true:

- `run_45` and `run_46` finish on the remote GPU host.
- `gpu_jobs.id=38` and `gpu_jobs.id=39` are marked `completed`.
- `run_47` contains the merged full benchmark artifact.
- `python scripts/audit_paper_benchmark_artifacts.py C:\Users\Kemal\deepgraph_ideas\idea_13\experiments\main\runs\run_47 --require-full` exits successfully.
- `python scripts/materialize_audited_cggr_results.py C:\Users\Kemal\deepgraph_ideas\idea_13\experiments\main\runs\run_47 --out-dir workspace\tmp\materialize_bundle_smoke_v2\audited_results` exits successfully.
- `workspace/tmp/materialize_bundle_smoke_v2/main.tex` compiles with audited snippets, materialized result/cost/ablation tables, and the audited utility-comparison figure snippet; `completion_audit.md` is present in the materialized evidence bundle.
- Every empirical manuscript claim is mapped to run47 artifacts and claim support decisions.

Until then, do not call `update_goal`.
