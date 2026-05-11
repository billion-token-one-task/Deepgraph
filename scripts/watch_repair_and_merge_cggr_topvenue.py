#!/usr/bin/env python3
"""Repair an interrupted top-venue shard, then run strict CGGR merge/audit."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from audit_paper_benchmark_artifacts import TOP_VENUE_BASELINE_METHODS, audit  # noqa: E402
from db import database as db  # noqa: E402
from materialize_audited_cggr_results import materialize  # noqa: E402
from merge_cggr_method_shards import (  # noqa: E402
    _aggregate_rows,
    _load_json,
    _write_json,
    _write_jsonl,
    merge,
)
from triage_cggr_audit_failure import write_triage_report  # noqa: E402


def _run(row_id: int) -> dict:
    row = db.fetchone("SELECT * FROM experiment_runs WHERE id=?", (int(row_id),))
    return dict(row) if row else {}


def _run_path(row_id: int) -> Path:
    run = _run(row_id)
    if not run or not run.get("workdir"):
        raise RuntimeError(f"run {row_id} is missing or has no workdir")
    return Path(str(run["workdir"]))


def _iter_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if isinstance(row, dict):
                rows.append(row)
    return rows


def _seed_value(row: dict) -> int | None:
    try:
        return int(row.get("seed"))
    except (TypeError, ValueError):
        return None


def _filter_rows(rows: list[dict], keep_seeds: set[int] | None) -> list[dict]:
    methods = set(TOP_VENUE_BASELINE_METHODS)
    filtered = []
    for row in rows:
        if str(row.get("method") or "") not in methods:
            continue
        seed = _seed_value(row)
        if keep_seeds is not None and seed not in keep_seeds:
            continue
        filtered.append(row)
    return filtered


def _dedupe_key(row: dict) -> tuple[str, str, int | None, str]:
    return (
        str(row.get("method") or ""),
        str(row.get("dataset") or ""),
        _seed_value(row),
        str(row.get("example_id") or ""),
    )


def _dedupe_raw(rows: list[dict]) -> tuple[list[dict], int]:
    seen: set[tuple[str, str, int | None, str]] = set()
    deduped: list[dict] = []
    duplicates = 0
    for row in rows:
        key = _dedupe_key(row)
        if key in seen:
            duplicates += 1
            continue
        seen.add(key)
        deduped.append(row)
    return deduped, duplicates


def _expected_from_config(config: dict) -> tuple[list[str], list[int], int]:
    datasets = [
        str(row.get("name") or row.get("hf_dataset") or "").strip()
        for row in config.get("targets") or []
        if isinstance(row, dict) and str(row.get("name") or row.get("hf_dataset") or "").strip()
    ]
    seed_values = config.get("seed_values")
    if isinstance(seed_values, list) and seed_values:
        seeds = [int(seed) for seed in seed_values]
    else:
        seeds = list(range(int(config.get("seeds") or 0)))
    examples = int(config.get("max_examples_per_dataset_seed") or 0)
    return datasets, seeds, examples


def _coverage_blockers(raw_rows: list[dict], config: dict) -> list[str]:
    datasets, seeds, examples = _expected_from_config(config)
    counts: dict[tuple[str, str, int], int] = defaultdict(int)
    for row in raw_rows:
        seed = _seed_value(row)
        if seed is None:
            continue
        counts[(str(row.get("method") or ""), str(row.get("dataset") or ""), seed)] += 1
    blockers = []
    for method in sorted(TOP_VENUE_BASELINE_METHODS):
        for dataset in datasets:
            for seed in seeds:
                count = counts[(method, dataset, seed)]
                if count != examples:
                    blockers.append(
                        f"top-venue cell coverage mismatch: method={method} dataset={dataset} "
                        f"seed={seed} count={count}/{examples}"
                    )
                    if len(blockers) >= 20:
                        blockers.append("additional coverage mismatches omitted")
                        return blockers
    return blockers


def _datasets_for_summary(config: dict) -> list[dict]:
    rows = []
    for row in config.get("targets") or []:
        if not isinstance(row, dict):
            continue
        rows.append(
            {
                "name": row.get("name"),
                "id": row.get("hf_dataset") or row.get("name"),
                "config": row.get("config") or "",
                "split": row.get("split") or "",
                "num_materialized_examples": 5 * 128,
                "license_or_source": row.get("hf_dataset") or row.get("name"),
                "preprocessing": "Answer normalization with exact/F1 scoring and task-specific numeric/boolean extraction.",
            }
        )
    return rows


def repair_topvenue_shard(
    *,
    interrupted_run_id: int,
    interrupted_keep_seeds: set[int],
    continuation_run_ids: list[int],
    repaired_run_id: int,
) -> dict:
    out_workdir = _run_path(repaired_run_id)
    out_results = out_workdir / "results"
    if out_results.exists() and any(out_results.iterdir()):
        return {"ok": False, "blockers": [f"output results directory is not empty: {out_results}"]}
    out_results.mkdir(parents=True, exist_ok=True)
    (out_workdir / "spec").mkdir(parents=True, exist_ok=True)

    source_runs = [interrupted_run_id, *continuation_run_ids]
    source_paths = [_run_path(run_id) for run_id in source_runs]
    base_config = _load_json(source_paths[0] / "results" / "run_config.json")
    if not isinstance(base_config, dict):
        return {"ok": False, "blockers": [f"missing run_config in run {interrupted_run_id}"]}
    base_config = dict(base_config)
    base_config["methods"] = sorted(TOP_VENUE_BASELINE_METHODS)
    base_config["seed_values"] = [0, 1, 2, 3, 4]
    base_config["seeds"] = 5
    base_config["sharded_run"] = False
    base_config["top_venue_seed_shard_repaired"] = True
    base_config["merged_from_seed_shards"] = [str(path) for path in source_paths]
    base_config["full_benchmark_completed"] = True

    raw_rows: list[dict] = []
    routing_rows: list[dict] = []
    failure_rows: list[dict] = []
    raw_rows.extend(
        _filter_rows(
            _iter_jsonl(source_paths[0] / "results" / "raw_predictions.jsonl"),
            interrupted_keep_seeds,
        )
    )
    routing_rows.extend(
        _filter_rows(
            _iter_jsonl(source_paths[0] / "results" / "routing_decisions.jsonl"),
            interrupted_keep_seeds,
        )
    )
    failure_rows.extend(
        _filter_rows(
            _iter_jsonl(source_paths[0] / "results" / "failure_cases.jsonl"),
            interrupted_keep_seeds,
        )
    )
    for path in source_paths[1:]:
        raw_rows.extend(_filter_rows(_iter_jsonl(path / "results" / "raw_predictions.jsonl"), None))
        routing_rows.extend(_filter_rows(_iter_jsonl(path / "results" / "routing_decisions.jsonl"), None))
        failure_rows.extend(_filter_rows(_iter_jsonl(path / "results" / "failure_cases.jsonl"), None))

    raw_rows, duplicate_count = _dedupe_raw(raw_rows)
    blockers = _coverage_blockers(raw_rows, base_config)
    generation_failures = [
        row for row in failure_rows if str(row.get("stage") or "").strip().lower() == "generation_or_scoring"
    ]
    if generation_failures:
        blockers.append("generation_or_scoring failures present in top-venue seed shards")
    if blockers:
        return {
            "ok": False,
            "blockers": blockers,
            "raw_predictions_lines": len(raw_rows),
            "duplicates_dropped": duplicate_count,
        }

    lambda_cost = float(base_config.get("cost_lambda") or 0.03)
    (
        per_method,
        per_method_std,
        per_seed_results,
        per_dataset_results,
        cost_table,
        difficulty_table,
        routing_analysis,
        simple_case_degradation,
        calibration_reliability,
    ) = _aggregate_rows(raw_rows, routing_rows, lambda_cost=lambda_cost)

    _write_json(out_results / "run_config.json", base_config)
    _write_jsonl(out_results / "raw_predictions.jsonl", raw_rows)
    _write_jsonl(out_results / "routing_decisions.jsonl", routing_rows)
    _write_jsonl(out_results / "failure_cases.jsonl", failure_rows)
    _write_json(out_results / "per_seed_results.json", per_seed_results)
    _write_json(out_results / "per_dataset_results.json", per_dataset_results)
    _write_json(out_results / "main_results_table.json", cost_table)
    _write_json(out_results / "cost_utility_tradeoff_table.json", cost_table)
    _write_json(out_results / "latency_tokens_table.json", cost_table)
    _write_json(out_results / "difficulty_breakdown_table.json", difficulty_table)
    _write_json(out_results / "routing_analysis.json", routing_analysis)
    _write_json(out_results / "ablation_table.json", [])
    _write_json(out_results / "simple_case_degradation.json", simple_case_degradation)
    _write_json(out_results / "calibration_reliability.json", calibration_reliability)
    _write_json(
        out_results / "bootstrap_ci.json",
        {
            "candidate_method": "CGGR",
            "baseline_method": "Always-Reason Chain-of-Thought",
            "candidate_ci95": [0.0, 0.0],
            "baseline_ci95": [0.0, 0.0],
            "delta_ci95": [0.0, 0.0],
            "observed_delta": 0.0,
            "paired_permutation_p": 1.0,
            "top_venue_repair_shard_only": True,
        },
    )
    environment_report = {
        "schema_version": "top_venue_repaired_environment_report_v1",
        "repaired_from_seed_shards": [str(path) for path in source_paths],
        "source_environment_reports": [
            _load_json(path / "results" / "environment_report.json")
            for path in source_paths
            if (path / "results" / "environment_report.json").exists()
        ],
    }
    _write_json(out_results / "environment_report.json", environment_report)

    datasets = _datasets_for_summary(base_config)
    summary = {
        "primary_metric": "cost_adjusted_accuracy",
        "metric_name": "cost_adjusted_accuracy",
        "candidate_method": "CGGR",
        "best_method": max(per_method.items(), key=lambda item: float(item[1].get("metric_value") or 0.0))[0],
        "per_method": per_method,
        "per_method_std": per_method_std,
        "seed_results": per_seed_results,
        "num_seeds": 5,
        "datasets": datasets,
        "dataset": datasets[0] if datasets else {},
        "model": {
            "id": base_config.get("model_id"),
            "backend": "transformers",
            "cuda": True,
            "hardware": "NVIDIA L40S",
        },
        "ablations": [],
        "ablation_results": [],
        "ablation_table": [],
        "cost_utility_tradeoff_table": cost_table,
        "difficulty_breakdown_table": difficulty_table,
        "routing_analysis": routing_analysis,
        "latency_tokens_table": cost_table,
        "simple_case_degradation": simple_case_degradation,
        "calibration_reliability": calibration_reliability,
        "bootstrap_ci": {},
        "load_failures": [],
        "budget": {
            "seeds": 5,
            "max_examples_per_dataset_seed": 128,
            "methods": sorted(TOP_VENUE_BASELINE_METHODS),
            "target_count": 4,
        },
        "method": "Top-venue adaptive-reasoning baselines",
        "duration_seconds": 0.0,
        "peak_vram_mb": 0.0,
        "hardware": "NVIDIA L40S",
        "full_benchmark_completed": True,
        "top_venue_seed_shard_repaired": True,
        "repaired_from_seed_shards": [str(path) for path in source_paths],
    }
    _write_json(out_results / "benchmark_summary.json", summary)
    manifest = {
        "full_benchmark_completed": True,
        "top_venue_seed_shard_repaired": True,
        "repaired_from_seed_shards": [str(path) for path in source_paths],
        "datasets": datasets,
        "methods": sorted(TOP_VENUE_BASELINE_METHODS),
        "model": summary["model"],
        "artifact_paths": {path.name: str(path) for path in out_results.iterdir() if path.is_file()},
    }
    _write_json(out_results / "artifact_manifest.json", manifest)
    _write_json(out_results / "benchmark_artifact_manifest.json", manifest)

    note = {
        "top_venue_seed_shard_repaired": True,
        "source_runs": source_runs,
        "raw_predictions_lines": len(raw_rows),
        "duplicates_dropped": duplicate_count,
        "methods": sorted(TOP_VENUE_BASELINE_METHODS),
    }
    db.execute(
        """UPDATE experiment_runs
           SET status='completed', phase='top_venue_seed_shard_repaired',
               best_metric_value=?, error_message=?, completed_at=CURRENT_TIMESTAMP
           WHERE id=?""",
        (
            max(float(row.get("metric_value") or 0.0) for row in per_method.values()),
            json.dumps(note, ensure_ascii=False),
            int(repaired_run_id),
        ),
    )
    db.commit()
    return {
        "ok": True,
        "run_id": repaired_run_id,
        "workdir": str(out_workdir),
        "raw_predictions_lines": len(raw_rows),
        "duplicates_dropped": duplicate_count,
        "methods": sorted(TOP_VENUE_BASELINE_METHODS),
    }


def _mark_failed(merged_run_id: int, reason: dict) -> None:
    db.execute(
        "UPDATE experiment_runs SET status='failed', phase='merge_failed', error_message=?, completed_at=CURRENT_TIMESTAMP WHERE id=?",
        (json.dumps(reason, ensure_ascii=False), int(merged_run_id)),
    )
    db.execute(
        "UPDATE auto_research_jobs SET status='failed', stage='merge_failed', last_error=?, last_note=? WHERE deep_insight_id=?",
        (
            json.dumps(reason, ensure_ascii=False),
            "CGGR strict top-venue repair/merge failed; manuscript claims remain blocked.",
            13,
        ),
    )
    db.commit()


def _claim_values_summary(materialize_out_dir: Path) -> dict:
    path = materialize_out_dir / "claim_values.json"
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    if not isinstance(payload, dict):
        return {}
    return {
        "cggr_utility": payload.get("cggr_utility"),
        "claim_support_decision": payload.get("claim_support_decision"),
        "top_venue_general_superiority_decision": payload.get("top_venue_general_superiority_decision"),
        "paired_permutation_p": payload.get("paired_permutation_p"),
    }


def _mark_completed(merged_run_id: int, materialize_result: dict, materialize_out_dir: Path, source_run_ids: list[int]) -> None:
    claim_summary = _claim_values_summary(materialize_out_dir)
    note = {
        "merged_from_method_shards": source_run_ids,
        "audit_ok": True,
        "materialized": materialize_result.get("written", []),
        "claim_values": claim_summary,
        "top_venue_seed_repair": True,
    }
    db.execute(
        """UPDATE experiment_runs
           SET status='completed', phase='strict_top_venue_full_benchmark_merged',
               best_metric_value=?, error_message=?, completed_at=CURRENT_TIMESTAMP
           WHERE id=?""",
        (claim_summary.get("cggr_utility"), json.dumps(note, ensure_ascii=False), int(merged_run_id)),
    )
    db.execute(
        "UPDATE auto_research_jobs SET status='completed', stage='strict_top_venue_full_benchmark_merged', experiment_run_id=?, last_error=NULL, last_note=? WHERE deep_insight_id=?",
        (int(merged_run_id), json.dumps(note, ensure_ascii=False), 13),
    )
    db.commit()


def _compile_latex(tex_path: Path | None) -> dict:
    if tex_path is None:
        return {"ok": True, "skipped": True}
    if not tex_path.exists():
        return {"ok": False, "error": f"tex file does not exist: {tex_path}"}
    proc = subprocess.run(
        ["latexmk", "-pdf", "-interaction=nonstopmode", "-halt-on-error", tex_path.name],
        cwd=str(tex_path.parent),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=300,
    )
    return {
        "ok": proc.returncode == 0,
        "returncode": proc.returncode,
        "stdout_tail": (proc.stdout or "")[-4000:],
        "stderr_tail": (proc.stderr or "")[-4000:],
    }


def watch_and_repair(
    *,
    core_run_ids: list[int],
    interrupted_run_id: int,
    interrupted_keep_seeds: set[int],
    continuation_run_ids: list[int],
    repaired_run_id: int,
    merged_run_id: int,
    materialize_out_dir: Path,
    poll_seconds: int,
    compile_tex: Path | None,
) -> dict:
    db.init_db()
    watched_run_ids = [*core_run_ids, *continuation_run_ids]
    while True:
        statuses = {run_id: _run(run_id).get("status") for run_id in watched_run_ids}
        print("[watch-topvenue-repair] " + json.dumps({"runs": statuses}, ensure_ascii=False), flush=True)
        failed = {run_id: status for run_id, status in statuses.items() if status == "failed"}
        if failed:
            reason = {"error": "source shard failed", "statuses": statuses}
            _mark_failed(merged_run_id, reason)
            return {"ok": False, **reason}
        if statuses and all(status == "completed" for status in statuses.values()):
            break
        time.sleep(max(30, int(poll_seconds)))

    repair_result = repair_topvenue_shard(
        interrupted_run_id=interrupted_run_id,
        interrupted_keep_seeds=interrupted_keep_seeds,
        continuation_run_ids=continuation_run_ids,
        repaired_run_id=repaired_run_id,
    )
    print("[watch-topvenue-repair] repair " + json.dumps(repair_result, ensure_ascii=False), flush=True)
    if not repair_result.get("ok"):
        _mark_failed(merged_run_id, {"error": "topvenue repair failed", "repair": repair_result})
        return {"ok": False, "repair": repair_result}

    merged_run = _run(merged_run_id)
    if not merged_run or not merged_run.get("workdir"):
        reason = {"error": f"merged run {merged_run_id} is missing or has no workdir"}
        _mark_failed(merged_run_id, reason)
        return {"ok": False, **reason}
    shard_paths = [_run_path(run_id) for run_id in [*core_run_ids, repaired_run_id]]
    merge_result = merge(shard_paths, Path(str(merged_run["workdir"])))
    print("[watch-topvenue-repair] merge " + json.dumps(merge_result, ensure_ascii=False), flush=True)
    if not merge_result.get("ok"):
        _mark_failed(merged_run_id, {"error": "merge failed", "merge": merge_result})
        return {"ok": False, "merge": merge_result}

    merged_workdir = Path(str(merged_run["workdir"]))
    audit_result = audit(merged_workdir, require_full=True, require_top_venue_baselines=True)
    print("[watch-topvenue-repair] audit " + json.dumps(audit_result, ensure_ascii=False), flush=True)
    if not audit_result.get("ok"):
        triage = write_triage_report(
            merged_workdir,
            audit_result=audit_result,
            require_full=True,
            require_top_venue_baselines=True,
        )
        _mark_failed(merged_run_id, {"error": "audit failed", "audit": audit_result, "triage": triage})
        return {"ok": False, "audit": audit_result, "triage": triage}

    materialize_result = materialize(
        merged_workdir,
        materialize_out_dir,
        require_top_venue_baselines=True,
    )
    print("[watch-topvenue-repair] materialize " + json.dumps(materialize_result, ensure_ascii=False), flush=True)
    if not materialize_result.get("ok"):
        triage = write_triage_report(
            merged_workdir,
            audit_result=materialize_result.get("audit"),
            require_full=True,
            require_top_venue_baselines=True,
        )
        _mark_failed(merged_run_id, {"error": "materialize failed", "materialize": materialize_result, "triage": triage})
        return {"ok": False, "materialize": materialize_result, "triage": triage}

    compile_result = _compile_latex(compile_tex)
    print("[watch-topvenue-repair] compile_tex " + json.dumps(compile_result, ensure_ascii=False), flush=True)
    if not compile_result.get("ok"):
        _mark_failed(merged_run_id, {"error": "latex compile failed", "compile_tex": compile_result})
        return {"ok": False, "compile_tex": compile_result}
    materialize_result["latex_compile"] = compile_result
    _mark_completed(
        merged_run_id,
        materialize_result,
        materialize_out_dir,
        [*core_run_ids, interrupted_run_id, *continuation_run_ids, repaired_run_id],
    )
    return {
        "ok": True,
        "merged_run_id": merged_run_id,
        "merged_workdir": str(merged_workdir),
        "repaired_run_id": repaired_run_id,
        "materialized": materialize_result.get("written", []),
    }


def _parse_int_set(value: str) -> set[int]:
    return {int(part.strip()) for part in str(value).split(",") if part.strip()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--core-run-id", action="append", type=int, required=True)
    parser.add_argument("--interrupted-run-id", type=int, required=True)
    parser.add_argument("--interrupted-keep-seeds", default="0,1,2")
    parser.add_argument("--continuation-run-id", action="append", type=int, required=True)
    parser.add_argument("--repaired-run-id", type=int, required=True)
    parser.add_argument("--merged-run-id", type=int, required=True)
    parser.add_argument("--materialize-out-dir", type=Path, required=True)
    parser.add_argument("--poll-seconds", type=int, default=300)
    parser.add_argument("--compile-tex", type=Path)
    args = parser.parse_args()
    result = watch_and_repair(
        core_run_ids=args.core_run_id,
        interrupted_run_id=args.interrupted_run_id,
        interrupted_keep_seeds=_parse_int_set(args.interrupted_keep_seeds),
        continuation_run_ids=args.continuation_run_id,
        repaired_run_id=args.repaired_run_id,
        merged_run_id=args.merged_run_id,
        materialize_out_dir=args.materialize_out_dir,
        poll_seconds=args.poll_seconds,
        compile_tex=args.compile_tex,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False), flush=True)
    raise SystemExit(0 if result.get("ok") else 1)


if __name__ == "__main__":
    main()

