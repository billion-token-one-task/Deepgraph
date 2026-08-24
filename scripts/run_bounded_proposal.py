#!/usr/bin/env python3
"""Realize one explicitly named proposal grant without global discovery."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from orchestrator.bounded_proposal import (  # noqa: E402
    BoundedProposalError,
    BoundedProposalRequest,
    authorize_bounded_proposal,
    execute_bounded_proposal,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job", type=int, required=True)
    parser.add_argument("--agenda", type=int, required=True)
    parser.add_argument("--idea", type=int, required=True)
    parser.add_argument("--grant", type=int, required=True)
    parser.add_argument("--actor", required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    request = BoundedProposalRequest(
        job_id=args.job,
        agenda_id=args.agenda,
        idea_id=args.idea,
        resource_grant_id=args.grant,
    )
    try:
        scope, already_completed = authorize_bounded_proposal(request)
        if args.dry_run:
            print(
                json.dumps(
                    {
                        "status": "already_completed" if already_completed else "admissible",
                        **request.__dict__,
                        "token_cap": int(scope.get("token_cap") or 0),
                        "expires_at": scope.get("expires_at"),
                    },
                    indent=2,
                    default=str,
                )
            )
            return 0
        result = execute_bounded_proposal(request, actor=args.actor)
    except Exception as exc:
        reason = str(exc) if isinstance(exc, BoundedProposalError) else f"{type(exc).__name__}: {exc}"
        print(json.dumps({"status": "refused", "reason": reason}, indent=2))
        return 1
    print(json.dumps(result.to_dict(), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
