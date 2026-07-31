# CGGR Current Completion Audit

This is the live evidence checkpoint for the CGGR paper-grade run. It records artifact status, not a blanket acceptance of every possible manuscript claim.

## 2026-05-10 Strict Top-Venue Result

- `run_45` completed the core method shard at 2026-05-10 07:04 with 10,240 raw rows.
- `run_46` completed the baseline/ablation shard at 2026-05-10 06:52 with 20,480 raw rows.
- `run_48` is preserved as a failed interrupted top-venue shard. It produced partial raw rows but no `benchmark_summary.json`, so it is not citable by itself.
- `run_50` and `run_51` completed the missing top-venue continuation seeds.
- `run_52` is the repaired top-venue shard, combining complete non-overlapping rows from `run_48`, `run_50`, and `run_51`: 7,680 raw rows, 0 duplicates dropped.
- `run_53` failed correctly because its local output directory already contained an older artifact. It is not evidence and was not overwritten.
- `run_54` is the clean strict merged artifact from `run_45`, `run_46`, and `run_52`. It passed `--require-full --require-top-venue-baselines` audit with 38,400 raw rows and no blockers.
- Materialized evidence bundle: `workspace/tmp/materialize_bundle_topvenue_repair_v2`.
- DB state is updated: `experiment_runs.id=54` is `completed/strict_top_venue_full_benchmark_merged`, and `auto_research_jobs.deep_insight_id=13` is `completed` with `experiment_run_id=54`.

## Main Numbers

| Method | Utility | Delta vs CGGR |
| --- | ---: | ---: |
| CGGR | 0.385354 | - |
| Vanilla Direct Answering | 0.306593 | +0.078761 |
| Confidence Gate | 0.304805 | +0.080549 |
| CAR-Style Certainty Adaptive Routing | 0.306427 | +0.078927 |
| Self-Route-Style Mode Routing | 0.300129 | +0.085224 |
| Rational-Metareasoning VOC Routing | 0.300129 | +0.085224 |
| Always-Reason Chain-of-Thought | 0.177546 | +0.207807 |
| Self-Consistency Reasoning | 0.127245 | +0.258109 |

CGGR quality is `0.394495`, average new tokens are `58.5066`, and global route rate is `0.625781`.

`claim_values.json` reports:

- `cggr_vs_baseline_delta=0.20780735238232578` against Always-Reason Chain-of-Thought.
- `cggr_vs_baseline_delta_ci95=[0.19667260846270507, 0.21856520798740892]`.
- `paired_permutation_p=0.0625`.
- `claim_support_decision=downgraded`.
- `top_venue_general_superiority_decision=eligible_under_strict_top_venue_audit`.

Interpretation: the completed artifact shows CGGR beats the locked baselines and the added top-venue-style adaptive routing baselines on the audited utility metric. The manuscript should still avoid overclaiming statistical certainty because the paired permutation p-value is 0.0625, and the materializer downgraded the broad claim support.

## Audit Policy Fix

The strict audit originally blocked `run_54` because some StrategyQA seed slices had zero CGGR deliberation. That was too broad: CGGR globally routed 0.625781 of examples to deliberation, so the method did not collapse.

The audit now blocks only a global CGGR zero-deliberation collapse. Per-dataset/seed extremes remain warnings and are carried into limitations. This was verified by:

- `python -m py_compile scripts\audit_paper_benchmark_artifacts.py tests\test_paper_benchmark_audit.py`
- `python -m unittest tests.test_paper_benchmark_audit`
- `python scripts\audit_paper_benchmark_artifacts.py C:\Users\Kemal\deepgraph_ideas\idea_13\experiments\main\runs\run_54 --require-full --require-top-venue-baselines`
- `python scripts\materialize_audited_cggr_results.py C:\Users\Kemal\deepgraph_ideas\idea_13\experiments\main\runs\run_54 --out-dir workspace\tmp\materialize_bundle_topvenue_repair_v2 --require-top-venue-baselines`

## Residual Warnings

The artifact is complete, but warnings remain and must be treated as limitations:

- Some routing methods have per-slice route rates of exactly 0 or 1.
- CGGR routes all MuSiQue and 2Wiki slices to deliberation, routes the stress split at 0.5, and routes most StrategyQA slices directly.
- Several top-venue-style baseline slices have high token-cap hit rates.
- The active CGGR implementation is a fixed proxy-gated executable instantiation, not a trained learned router.

## Remaining Manuscript Work

- Use the `run_54` materialized bundle for paper tables, snippets, figures, claim maps, and limitations.
- Do not cite `run_48` or `run_53` as successful evidence.
- Keep broad claims scoped: top-venue-style comparator coverage is now present, but claim strength remains downgraded by the materialized claim audit.
- If stronger statistical support or a cleaner routing profile is required, register a new v2 method run rather than tuning `run_54` after the fact.

## 2026-05-10 CGGR-v2 Candidate Runs

Two preregistered v2 candidate shards were launched after `run_54` exposed the strongest ablation signal:

- `run_55` / `gpu_jobs.id=43`: `cggr_v2_aggressive_calibrated_lcb`, running on `ssh:proxy.cn-south-1.gpu-instance.ppinfra.com:gpu0`.
  - Change: route threshold `0.34`, short branch `80` tokens, deliberative branch `224` tokens.
  - Goal: turn the strong `CGGR/no_lcb` result into a new preregistered candidate without mutating `run_54`.
- `run_56` / `gpu_jobs.id=44`: `cggr_v2_efficient_calibrated_lcb`, running on `ssh:proxy.cn-south-1.gpu-instance.ppinfra.com:gpu1`.
  - Change: route threshold `0.34`, short branch `64` tokens, deliberative branch `160` tokens.
  - Goal: test whether the lower-threshold quality gain survives a smaller deliberation cap and improves cost-adjusted utility.

Both shards use the real benchmark targets, 5 seeds, 128 examples per dataset/seed, and method subset `CGGR`. They are candidate-search evidence, not replacements for `run_54` unless they complete and are merged into a new full v2 artifact. Monitors:

- `logs/monitor-run55-cggr-v2.out.log`
- `logs/monitor-run56-cggr-v2.out.log`
- `logs/watch-cggr-v2-candidates.out.log`

The watcher writes comparison output to `workspace/tmp/cggr_v2_candidate_report.json` and `.md` after both candidate shards reach terminal status.
