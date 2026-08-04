#!/usr/bin/env python3
"""Record one candidate's topic-gate answers from a reviewed JSON file.

Operator-invoked only, and deliberately file-driven: the prediction is a
pre-registration, so it should be reviewable as a diffable artifact before it
reaches the database rather than assembled out of shell arguments.

    python3 scripts/record_topic_gate_prediction.py --agenda 7 --idea 98 \
        --record docs/pre_registrations/agenda7_idea98.json \
        --actor operator:owner --dry-run

``--dry-run`` screens the candidate exactly as the stored record would be
screened and prints the verdict without writing anything.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from meta_harness.topic_gate_record import (  # noqa: E402
    TopicGateRecordError,
    evaluate_record,
    record_prediction,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--agenda", type=int, required=True)
    parser.add_argument("--idea", type=int, required=True)
    parser.add_argument(
        "--record",
        required=True,
        help="path to the reviewed topic-gate record JSON",
    )
    parser.add_argument(
        "--actor",
        required=True,
        help="who is accountable for this write; stamped into the provenance",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="screen the record and print the verdict; write nothing",
    )
    args = parser.parse_args()

    try:
        record = json.loads(Path(args.record).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        print(json.dumps({"status": "refused", "reason": f"unreadable record: {exc}"}, indent=2))
        return 1
    if not isinstance(record, dict):
        print(json.dumps({"status": "refused", "reason": "record must be a JSON object"}, indent=2))
        return 1

    try:
        if args.dry_run:
            decision, _ = evaluate_record(
                agenda_id=args.agenda, idea_id=args.idea, record=record
            )
            print(
                json.dumps(
                    {
                        "status": "dry_run",
                        "gate_passed": decision.passed,
                        "gate_reason_codes": list(decision.reason_codes),
                        "blockers": [dict(item) for item in decision.blockers],
                        "expected_bits": decision.expected_bits,
                        "refute_bits": decision.refute_bits,
                        "confidence": decision.confidence,
                    },
                    indent=2,
                )
            )
            return 0 if decision.passed else 1
        result = record_prediction(
            agenda_id=args.agenda,
            idea_id=args.idea,
            record=record,
            actor=args.actor,
        )
    except TopicGateRecordError as exc:
        print(json.dumps({"status": "refused", "reason": str(exc)}, indent=2))
        return 1
    except Exception as exc:
        print(
            json.dumps(
                {"status": "refused", "reason": f"{type(exc).__name__}: {exc}"},
                indent=2,
            )
        )
        return 1
    print(json.dumps(result, indent=2, default=str))
    return 0 if result.get("gate_passed") else 1


if __name__ == "__main__":
    raise SystemExit(main())
