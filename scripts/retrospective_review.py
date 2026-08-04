"""Operator CLI for the retrospective (legacy) review path.

  python3 scripts/retrospective_review.py list
  python3 scripts/retrospective_review.py packet --run-id 14
  python3 scripts/retrospective_review.py apply --run-id 14 \
      --reviewer <id> --key-id <key>

`apply` additionally requires:
  DEEPGRAPH_ALLOW_RETROSPECTIVE_REVIEW=1
  DEEPGRAPH_REVIEWER_APPROVAL_KEYS_JSON, e.g. {"<key>": "env:DEEPGRAPH_REVIEWER_SECRET"}
  the referenced secret exported in the environment

The verdict is always capped at 'inconclusive' (no holdout exists for
historical runs); the reviewer's signature attests eligibility and evidence
integrity, not scientific support.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from meta_harness import retrospective_review as rr


def cmd_list() -> int:
    rows = rr.eligible_run_rows()
    if not rows:
        print("no eligible runs")
        return 0
    print(f"{'run':>4} {'agenda':>6} {'idea':>5} {'baseline':>12} {'best':>12} "
          f"{'effect%':>8} informal_verdict")
    for row in rows:
        print(f"{row['id']:>4} {row['agenda_id']:>6} "
              f"{row['deep_insight_id'] or '-':>5} "
              f"{row['baseline_metric_value']:>12.6g} "
              f"{row['best_metric_value']:>12.6g} "
              f"{(row['effect_pct'] or 0):>8.2f} "
              f"{row['hypothesis_verdict'] or '-'}")
    print(f"total: {len(rows)}")
    return 0


def cmd_packet(run_id: int) -> int:
    print(json.dumps(rr.build_packet(run_id), indent=2, ensure_ascii=False,
                     default=str))
    return 0


def cmd_apply(run_id: int, reviewer: str, key_id: str) -> int:
    if os.environ.get("DEEPGRAPH_ALLOW_RETROSPECTIVE_REVIEW") != "1":
        print("refusing: apply requires DEEPGRAPH_ALLOW_RETROSPECTIVE_REVIEW=1",
              file=sys.stderr)
        return 2
    manifest = json.loads(
        os.environ.get("DEEPGRAPH_REVIEWER_APPROVAL_KEYS_JSON", "{}") or "{}"
    )
    reference = str(manifest.get(key_id, ""))
    if not reference.startswith("env:"):
        print(f"refusing: key '{key_id}' is not in the reviewer key manifest",
              file=sys.stderr)
        return 2
    secret = os.environ.get(reference[4:], "")
    if not secret:
        print("refusing: reviewer secret is not exported", file=sys.stderr)
        return 2

    packet = rr.build_packet(run_id)
    if packet["blockers"]:
        print("run not eligible: " + ", ".join(packet["blockers"]),
              file=sys.stderr)
        return 1
    approval = rr.sign_approval(
        reviewer_id=reviewer, key_id=key_id, subject=packet["subject"],
        secret=secret,
    )
    result = rr.apply_review(run_id=run_id, approval=approval)
    print(json.dumps({"applied": result}, indent=2, ensure_ascii=False))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("list")
    packet = sub.add_parser("packet")
    packet.add_argument("--run-id", type=int, required=True)
    apply_p = sub.add_parser("apply")
    apply_p.add_argument("--run-id", type=int, required=True)
    apply_p.add_argument("--reviewer", required=True)
    apply_p.add_argument("--key-id", required=True)
    args = parser.parse_args()
    if args.command == "list":
        return cmd_list()
    if args.command == "packet":
        return cmd_packet(args.run_id)
    return cmd_apply(args.run_id, args.reviewer, args.key_id)


if __name__ == "__main__":
    raise SystemExit(main())
