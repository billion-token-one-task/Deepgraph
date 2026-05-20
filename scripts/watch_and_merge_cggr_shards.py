#!/usr/bin/env python3
"""Wait for CGGR method shards, then merge, audit, and materialize results."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from db import database as db
from audit_paper_benchmark_artifacts import audit
from materialize_audited_cggr_results import materialize
from merge_cggr_method_shards import merge
from triage_cggr_audit_failure import write_triage_report


def _run(row_id: int) -> dict:
    row = db.fetchone("SELECT * FROM experiment_runs WHERE id=?", (row_id,))
    return dict(row) if row else {}


def _jobs_for(run_ids: list[int]) -> list[dict]:
    placeholders = ",".join("?" for _ in run_ids)
    rows = db.fetchall(
        f"SELECT * FROM gpu_jobs WHERE experiment_run_id IN ({placeholders}) ORDER BY experiment_run_id, id",
        tuple(run_ids),
    )
    return [dict(row) for row in rows]


def _mark_merged_failed(merged_run_id: int, reason: dict) -> None:
    db.execute(
        "UPDATE experiment_runs SET status='failed', phase='merge_failed', error_message=?, completed_at=CURRENT_TIMESTAMP WHERE id=?",
        (json.dumps(reason, ensure_ascii=False), merged_run_id),
    )
    db.execute(
        "UPDATE auto_research_jobs SET status='failed', stage='merge_failed', last_error=?, last_note=? WHERE deep_insight_id=?",
        (
            json.dumps(reason, ensure_ascii=False),
            "CGGR full-scale method shards failed to merge or audit; paper claims remain blocked.",
            13,
        ),
    )
    db.commit()


def _claim_values_summary(materialize_out_dir: Path) -> dict:
    claim_values = materialize_out_dir / "claim_values.json"
    if not claim_values.exists():
        return {}
    try:
        payload = json.loads(claim_values.read_text(encoding="utf-8"))
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


def _mark_merged_completed(
    merged_run_id: int,
    materialize_result: dict,
    *,
    materialize_out_dir: Path,
    shard_run_ids: list[int],
) -> None:
    claim_summary = _claim_values_summary(materialize_out_dir)
    best_value = claim_summary.get("cggr_utility")
    note = {
        "merged_from_method_shards": shard_run_ids,
        "audit_ok": True,
        "materialized": materialize_result.get("written", []),
        "claim_values": claim_summary,
    }
    db.execute(
        """UPDATE experiment_runs
           SET status='completed', phase='full_benchmark_merged',
               best_metric_value=?, error_message=?, completed_at=CURRENT_TIMESTAMP
           WHERE id=?""",
        (best_value, json.dumps(note, ensure_ascii=False), merged_run_id),
    )
    db.execute(
        "UPDATE auto_research_jobs SET status='completed', stage='full_benchmark_merged', experiment_run_id=?, last_error=NULL, last_note=? WHERE deep_insight_id=?",
        (merged_run_id, json.dumps(note, ensure_ascii=False), 13),
    )
    db.commit()


def _compile_latex(tex_path: Path | None) -> dict:
    if tex_path is None:
        return {"ok": True, "skipped": True}
    if not tex_path.exists():
        return {"ok": False, "error": f"tex file does not exist: {tex_path}"}
    cmd = ["latexmk", "-pdf", "-interaction=nonstopmode", "-halt-on-error", tex_path.name]
    proc = subprocess.run(
        cmd,
        cwd=str(tex_path.parent),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=300,
    )
    return {
        "ok": proc.returncode == 0,
        "command": cmd,
        "cwd": str(tex_path.parent),
        "returncode": proc.returncode,
        "stdout_tail": (proc.stdout or "")[-4000:],
        "stderr_tail": (proc.stderr or "")[-4000:],
    }


def watch(
    *,
    shard_run_ids: list[int],
    merged_run_id: int,
    materialize_out_dir: Path,
    poll_seconds: int,
    compile_tex: Path | None = None,
    require_top_venue_baselines: bool = False,
) -> dict:
    db.init_db()
    while True:
        runs = [_run(run_id) for run_id in shard_run_ids]
        jobs = _jobs_for(shard_run_ids)
        statuses = {
            "runs": {run.get("id"): run.get("status") for run in runs},
            "jobs": {job.get("experiment_run_id"): job.get("status") for job in jobs},
        }
        print("[watch-merge] " + json.dumps(statuses, ensure_ascii=False), flush=True)
        failed = [run for run in runs if run.get("status") == "failed"] + [
            job for job in jobs if job.get("status") == "failed"
        ]
        if failed:
            reason = {"error": "source shard failed", "statuses": statuses}
            _mark_merged_failed(merged_run_id, reason)
            return {"ok": False, **reason}
        completed_runs = all(run.get("status") == "completed" for run in runs)
        completed_jobs = all(job.get("status") == "completed" for job in jobs) and len(jobs) >= len(shard_run_ids)
        if completed_runs and completed_jobs:
            break
        time.sleep(max(30, poll_seconds))

    merged_run = _run(merged_run_id)
    if not merged_run or not merged_run.get("workdir"):
        reason = {"error": f"merged run {merged_run_id} is missing or has no workdir"}
        _mark_merged_failed(merged_run_id, reason)
        return {"ok": False, **reason}
    shard_paths = [Path(run["workdir"]) for run in runs]
    merged_workdir = Path(merged_run["workdir"])
    merge_result = merge(shard_paths, merged_workdir)
    print("[watch-merge] merge " + json.dumps(merge_result, ensure_ascii=False), flush=True)
    if not merge_result.get("ok"):
        _mark_merged_failed(merged_run_id, {"error": "merge failed", "merge_result": merge_result})
        return {"ok": False, "merge_result": merge_result}

    audit_result = audit(
        merged_workdir,
        require_full=True,
        require_top_venue_baselines=require_top_venue_baselines,
    )
    print("[watch-merge] audit " + json.dumps(audit_result, ensure_ascii=False), flush=True)
    if not audit_result.get("ok"):
        triage_result = write_triage_report(
            merged_workdir,
            audit_result=audit_result,
            require_full=True,
            require_top_venue_baselines=require_top_venue_baselines,
        )
        _mark_merged_failed(
            merged_run_id,
            {"error": "audit failed", "audit": audit_result, "triage": triage_result},
        )
        return {"ok": False, "audit": audit_result, "triage": triage_result}

    materialize_result = materialize(
        merged_workdir,
        materialize_out_dir,
        require_top_venue_baselines=require_top_venue_baselines,
    )
    print("[watch-merge] materialize " + json.dumps(materialize_result, ensure_ascii=False), flush=True)
    if not materialize_result.get("ok"):
        triage_result = write_triage_report(
            merged_workdir,
            audit_result=materialize_result.get("audit"),
            require_full=True,
            require_top_venue_baselines=require_top_venue_baselines,
        )
        _mark_merged_failed(
            merged_run_id,
            {"error": "materialize failed", "materialize": materialize_result, "triage": triage_result},
        )
        return {"ok": False, "materialize": materialize_result, "triage": triage_result}

    compile_result = _compile_latex(compile_tex)
    print("[watch-merge] compile_tex " + json.dumps(compile_result, ensure_ascii=False), flush=True)
    if not compile_result.get("ok"):
        _mark_merged_failed(merged_run_id, {"error": "latex compile failed", "compile_tex": compile_result})
        return {"ok": False, "compile_tex": compile_result}
    materialize_result["latex_compile"] = compile_result

    _mark_merged_completed(
        merged_run_id,
        materialize_result,
        materialize_out_dir=materialize_out_dir,
        shard_run_ids=shard_run_ids,
    )
    return {
        "ok": True,
        "merged_run_id": merged_run_id,
        "merged_workdir": str(merged_workdir),
        "materialized": materialize_result.get("written", []),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard-run-id", action="append", type=int, required=True)
    parser.add_argument("--merged-run-id", type=int, required=True)
    parser.add_argument("--materialize-out-dir", type=Path, required=True)
    parser.add_argument("--poll-seconds", type=int, default=300)
    parser.add_argument("--compile-tex", type=Path)
    parser.add_argument("--require-top-venue-baselines", action="store_true")
    args = parser.parse_args()
    result = watch(
        shard_run_ids=args.shard_run_id,
        merged_run_id=args.merged_run_id,
        materialize_out_dir=args.materialize_out_dir,
        poll_seconds=args.poll_seconds,
        compile_tex=args.compile_tex,
        require_top_venue_baselines=args.require_top_venue_baselines,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False), flush=True)
    raise SystemExit(0 if result.get("ok") else 1)


if __name__ == "__main__":
    main()
