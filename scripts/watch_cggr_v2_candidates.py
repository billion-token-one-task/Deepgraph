#!/usr/bin/env python3
"""Watch CGGR-v2 candidate shards and compare them against the audited run54 baseline."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from db import database as db  # noqa: E402


TERMINAL = {"completed", "failed", "cancelled"}


def _load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _run_row(run_id: int) -> dict[str, Any]:
    row = db.fetchone("SELECT * FROM experiment_runs WHERE id=?", (int(run_id),))
    return dict(row or {})


def _metric_from_run(run_id: int) -> dict[str, Any]:
    row = _run_row(run_id)
    workdir = Path(str(row.get("workdir") or ""))
    summary = _load_json(workdir / "results" / "benchmark_summary.json")
    per_method = summary.get("per_method") if isinstance(summary.get("per_method"), dict) else {}
    cggr = per_method.get("CGGR") if isinstance(per_method.get("CGGR"), dict) else {}
    idea = _load_json(workdir / "spec" / "v2_idea.json")
    shard_config = _load_json(workdir / "spec" / "shard_config.json")
    return {
        "run_id": int(run_id),
        "status": row.get("status"),
        "phase": row.get("phase"),
        "workdir": str(workdir) if workdir else "",
        "idea_id": idea.get("idea_id") or shard_config.get("shard_name"),
        "metric_value": cggr.get("metric_value"),
        "score": cggr.get("score"),
        "avg_new_tokens": cggr.get("avg_new_tokens"),
        "avg_latency_seconds": cggr.get("avg_latency_seconds"),
        "route_rate": cggr.get("route_rate"),
        "raw_predictions_lines": _line_count(workdir / "results" / "raw_predictions.jsonl"),
        "summary_exists": bool(summary),
        "candidate_change": shard_config.get("candidate_change"),
    }


def _line_count(path: Path) -> int:
    try:
        with path.open("rb") as handle:
            return sum(1 for _ in handle)
    except OSError:
        return 0


def _baseline_metrics(run_id: int) -> dict[str, Any]:
    row = _run_row(run_id)
    workdir = Path(str(row.get("workdir") or ""))
    summary = _load_json(workdir / "results" / "benchmark_summary.json")
    per_method = summary.get("per_method") if isinstance(summary.get("per_method"), dict) else {}
    out = {"run_id": int(run_id), "workdir": str(workdir)}
    for method in ("CGGR", "CGGR/no_lcb", "Vanilla Direct Answering", "Confidence Gate"):
        metric = per_method.get(method) if isinstance(per_method.get(method), dict) else {}
        out[method] = {
            "metric_value": metric.get("metric_value"),
            "score": metric.get("score"),
            "avg_new_tokens": metric.get("avg_new_tokens"),
            "route_rate": metric.get("route_rate"),
        }
    return out


def _as_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _write_reports(report: dict[str, Any], out_json: Path) -> None:
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    lines = [
        "# CGGR v2 Candidate Report",
        "",
        f"- Baseline run: `{report.get('baseline_run_id')}`",
        f"- Baseline CGGR utility: `{report.get('baseline_cggr_utility')}`",
        f"- Baseline no_lcb utility: `{report.get('baseline_no_lcb_utility')}`",
        f"- Decision: `{report.get('decision')}`",
        "",
        "| Run | Idea | Status | Utility | Delta vs CGGR | Delta vs no_lcb | Tokens | Route Rate |",
        "| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in report.get("candidates", []):
        lines.append(
            "| {run_id} | {idea_id} | {status} | {utility} | {delta_cggr} | {delta_no_lcb} | {tokens} | {route} |".format(
                run_id=row.get("run_id"),
                idea_id=row.get("idea_id") or "",
                status=row.get("status") or "",
                utility=_fmt(row.get("metric_value")),
                delta_cggr=_fmt(row.get("delta_vs_run54_cggr")),
                delta_no_lcb=_fmt(row.get("delta_vs_run54_no_lcb")),
                tokens=_fmt(row.get("avg_new_tokens")),
                route=_fmt(row.get("route_rate")),
            )
        )
    lines.append("")
    out_json.with_suffix(".md").write_text("\n".join(lines), encoding="utf-8")


def _fmt(value: Any) -> str:
    number = _as_float(value)
    return "--" if number is None else f"{number:.6f}"


def build_report(baseline_run_id: int, candidate_run_ids: list[int]) -> dict[str, Any]:
    baseline = _baseline_metrics(baseline_run_id)
    baseline_cggr = _as_float((baseline.get("CGGR") or {}).get("metric_value"))
    baseline_no_lcb = _as_float((baseline.get("CGGR/no_lcb") or {}).get("metric_value"))
    candidates = []
    best = None
    for run_id in candidate_run_ids:
        row = _metric_from_run(run_id)
        value = _as_float(row.get("metric_value"))
        row["delta_vs_run54_cggr"] = None if value is None or baseline_cggr is None else value - baseline_cggr
        row["delta_vs_run54_no_lcb"] = None if value is None or baseline_no_lcb is None else value - baseline_no_lcb
        candidates.append(row)
        if value is not None and (best is None or value > float(best.get("metric_value"))):
            best = row
    best_value = _as_float((best or {}).get("metric_value"))
    if best_value is None:
        decision = "no_completed_candidate_metric"
    elif baseline_no_lcb is not None and best_value >= baseline_no_lcb:
        decision = "candidate_matches_or_beats_no_lcb_target"
    elif baseline_cggr is not None and best_value > baseline_cggr:
        decision = "candidate_beats_run54_cggr_but_not_no_lcb"
    else:
        decision = "candidate_does_not_beat_run54_cggr"
    return {
        "baseline_run_id": int(baseline_run_id),
        "baseline": baseline,
        "baseline_cggr_utility": baseline_cggr,
        "baseline_no_lcb_utility": baseline_no_lcb,
        "candidate_run_ids": [int(run_id) for run_id in candidate_run_ids],
        "candidates": candidates,
        "best_candidate": best,
        "decision": decision,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }


def _mark_note(report: dict[str, Any], insight_id: int) -> None:
    note = {
        "cggr_v2_candidate_report": report,
        "report_path": report.get("report_path"),
    }
    db.execute(
        "UPDATE auto_research_jobs SET last_note=?, last_error=NULL, stage='cggr_v2_candidates_compared' WHERE deep_insight_id=?",
        (json.dumps(note, ensure_ascii=False), int(insight_id)),
    )
    db.commit()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-run-id", type=int, default=54)
    parser.add_argument("--candidate-run-id", type=int, action="append", required=True)
    parser.add_argument("--out-json", type=Path, default=Path("workspace/tmp/cggr_v2_candidate_report.json"))
    parser.add_argument("--poll-seconds", type=int, default=300)
    parser.add_argument("--deep-insight-id", type=int, default=13)
    args = parser.parse_args()

    db.init_db()
    while True:
        rows = [_run_row(run_id) for run_id in args.candidate_run_id]
        statuses = {int(row.get("id") or 0): str(row.get("status") or "") for row in rows}
        print("[cggr-v2-watch] statuses=" + json.dumps(statuses, sort_keys=True), flush=True)
        if all(status in TERMINAL for status in statuses.values()):
            report = build_report(int(args.baseline_run_id), [int(run_id) for run_id in args.candidate_run_id])
            report["report_path"] = str(args.out_json)
            _write_reports(report, args.out_json)
            _mark_note(report, int(args.deep_insight_id))
            print("[cggr-v2-watch] completed " + json.dumps(report, ensure_ascii=False, sort_keys=True), flush=True)
            return
        time.sleep(max(30, int(args.poll_seconds)))


if __name__ == "__main__":
    main()
