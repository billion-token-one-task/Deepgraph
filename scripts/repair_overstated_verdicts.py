"""Recompute hypothesis_verdict from each run's own artifacts and p-value.

Until 2026-08-20 the runner set scientific_negative_result from the SIGN of
the difference alone (`candidate <= baseline`) and three separate places
turned that flag into the verdict "refuted", never consulting the p-value the
same run had just computed. Failing to show an improvement is not showing
harm, and "refuted" is a scientific claim carrying the same evidential burden
as "supported".

The code is fixed (the runner now writes one authoritative verdict). This
script repairs the records the broken rule produced. It changes nothing about
the measurements: it only re-reads each run's own final_results.json and
applies the corrected rule to it. Every proposed change is printed with the
numbers that justify it.

The system caught this itself: the cross-vendor evaluator refused to concur
on run 189 (0.685 -> 0.63, p = 0.220), saying a non-significant result cannot
be classified as refuted. That dissent is what surfaced the defect.

Read-only by default. Pass --apply to write.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from db import database as db
from meta_harness.evidence_audit import significance_aware_verdict


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()

    changes = []
    for run in db.fetchall(
        "SELECT id, workdir, hypothesis_verdict FROM experiment_runs"
        " WHERE workdir IS NOT NULL ORDER BY id"
    ):
        run_id = int(run["id"])
        recorded = str(run["hypothesis_verdict"] or "").strip()
        if not recorded:
            continue
        results = Path(str(run["workdir"])) / "results" / "final_results.json"
        if not results.exists():
            continue
        final = json.loads(results.read_text())
        # Ignore the run row's own verdict when recomputing, or the broken
        # value would simply be echoed back as authoritative.
        artifact_view = dict(final)
        artifact_view.pop("hypothesis_verdict", None)
        correct = significance_aware_verdict(artifact_view)
        if correct == recorded:
            continue
        tests = final.get("statistical_tests") or {}
        changes.append(
            {
                "run_id": run_id,
                "recorded": recorded,
                "correct": correct,
                "baseline": final.get("baseline_metric_value"),
                "candidate": final.get("metric_value"),
                "p_value": tests.get("paired_permutation_p"),
            }
        )

    if not changes:
        print("every recorded verdict already matches its own p-value")
        return 0

    print(f"{len(changes)} run(s) carry a verdict their numbers do not support:")
    for change in changes:
        print(
            f"  run {change['run_id']:>4}: "
            f"{change['baseline']} -> {change['candidate']} "
            f"p={change['p_value']}  "
            f"{change['recorded']} -> {change['correct']}"
        )
    if not args.apply:
        print("re-run with --apply to correct them")
        return 1

    for change in changes:
        db.execute(
            "UPDATE experiment_runs SET hypothesis_verdict=? WHERE id=?",
            (change["correct"], change["run_id"]),
        )
    db.commit()
    print(f"corrected {len(changes)} verdict(s)")

    for change in changes:
        now = db.fetchone(
            "SELECT hypothesis_verdict FROM experiment_runs WHERE id=?",
            (change["run_id"],),
        )["hypothesis_verdict"]
        assert now == change["correct"], (change["run_id"], now)
        print(f"  run {change['run_id']}: now {now}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
