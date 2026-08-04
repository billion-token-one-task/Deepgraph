#!/usr/bin/env python3
"""Run one portfolio-granted candidate on CPU/LLM. Operator-invoked only.

This is the deliberate, single-candidate alternative to flipping global
autonomy: it names the agenda, idea and ResourceGrant on the command line and
executes exactly that one, then settles an OutcomeRecord. It never reads or
changes the autonomy flags.

    python3 scripts/run_bounded_pilot.py --agenda 5 --idea 97 --grant 1 \
        --actor ops:recovery --dry-run

``--dry-run`` performs the admission checks and prints what would run without
claiming the job or spending anything.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from orchestrator.bounded_execution import (  # noqa: E402
    BOUNDED_BACKENDS,
    BoundedExecutionError,
    BoundedExecutionRequest,
    _authorize_bounded_grant,
    execute_granted_candidate,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--agenda", type=int, required=True)
    parser.add_argument("--idea", type=int, required=True)
    parser.add_argument("--grant", type=int, required=True)
    parser.add_argument(
        "--actor",
        required=True,
        help="who is accountable for this execution; recorded on the transition",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="check admission only; claim nothing and spend nothing",
    )
    args = parser.parse_args()

    request = BoundedExecutionRequest(
        agenda_id=args.agenda,
        idea_id=args.idea,
        resource_grant_id=args.grant,
    )
    try:
        request.validate()
        grant, _ = _authorize_bounded_grant(request)
    except BoundedExecutionError as exc:
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

    if args.dry_run:
        print(
            json.dumps(
                {
                    "status": "admissible",
                    "agenda_id": grant.agenda_id,
                    "idea_id": grant.idea_id,
                    "resource_grant_id": grant.grant_id,
                    "stage": grant.stage,
                    "token_cap": grant.token_cap,
                    "max_gpu_hours": grant.max_gpu_hours,
                    "backends": sorted(grant.backend_allowlist),
                    "bounded_backends": sorted(BOUNDED_BACKENDS),
                    "expires_at": grant.expires_at,
                },
                indent=2,
            )
        )
        return 0

    result = execute_granted_candidate(request, actor=args.actor)
    print(json.dumps(result.to_dict(), indent=2, default=str))
    return 0 if result.status == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
