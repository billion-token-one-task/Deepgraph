#!/usr/bin/env python3
"""Explicitly recover one supported historical run's manuscript gate.

Dry-run is the default.  ``--apply`` uses the reviewed repository API: it does
not reopen the agenda or mutate the consumed audit grant/outcome.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from db import database as db  # noqa: E402
from meta_harness.manuscript_gate import (  # noqa: E402
    MANUSCRIPT_REVIEWER_SECRET_ENV,
    run_manuscript_gate,
)
from meta_harness.repository import MetaHarnessRepository  # noqa: E402


def _plan(*, agenda_id: int, run_id: int, expected_verdict_hash: str) -> dict:
    row = db.fetchone(
        """
        SELECT er.id AS run_id, er.agenda_id, er.deep_insight_id AS idea_id,
               er.resource_grant_id AS prior_grant_id, er.status AS run_status,
               er.scientific_evidence_state, er.workdir,
               rg.stage AS prior_grant_stage, rg.status AS prior_grant_status,
               arl.status AS prior_ledger_status,
               ra.status AS agenda_status, ra.is_active, ra.token_budget,
               ra.token_spent, ra.token_reserved,
               sdr.verdict, sdr.verdict_hash
        FROM experiment_runs er
        JOIN resource_grants rg ON rg.id=er.resource_grant_id
        JOIN agenda_resource_ledger arl ON arl.id=rg.reservation_id
        JOIN research_agendas ra ON ra.id=er.agenda_id
        JOIN scientific_decision_records sdr
          ON sdr.agenda_id=er.agenda_id AND sdr.experiment_run_id=er.id
        WHERE er.id=? AND er.agenda_id=?
        ORDER BY sdr.id DESC LIMIT 1
        """,
        (int(run_id), int(agenda_id)),
    )
    if not row:
        raise SystemExit("historical manuscript recovery scope was not found")
    plan = dict(row)
    actual_hash = str(plan.get("verdict_hash") or "").lower().removeprefix(
        "sha256:"
    )
    expected_hash = str(expected_verdict_hash or "").lower().removeprefix(
        "sha256:"
    )
    if actual_hash != expected_hash:
        raise SystemExit("expected verdict hash does not match persisted truth")
    workdir = Path(str(plan.get("workdir") or ""))
    ledger = workdir / "results" / "claim_ledger.json"
    if not ledger.is_file():
        raise SystemExit(f"claim ledger is missing: {ledger}")
    plan.update(
        {
            "expected_verdict_hash": expected_hash,
            "claim_ledger": str(ledger),
            "claim_ledger_bytes": ledger.stat().st_size,
            "operation": "historical_manuscript_recovery",
        }
    )
    return plan


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--agenda-id", type=int, required=True)
    parser.add_argument("--run-id", type=int, required=True)
    parser.add_argument("--expected-verdict-hash", required=True)
    parser.add_argument("--token-cap", type=int, default=40000)
    parser.add_argument("--ttl-minutes", type=int, default=120)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    if args.token_cap <= 0 or not 1 <= args.ttl_minutes <= 72 * 60:
        parser.error("token cap and TTL must be positive and bounded")

    plan = _plan(
        agenda_id=args.agenda_id,
        run_id=args.run_id,
        expected_verdict_hash=args.expected_verdict_hash,
    )
    plan["token_cap"] = args.token_cap
    plan["ttl_minutes"] = args.ttl_minutes
    plan["apply"] = bool(args.apply)
    print(json.dumps(plan, ensure_ascii=False, sort_keys=True, default=str))
    if not args.apply:
        db.rollback()
        return 0

    secret = os.getenv(MANUSCRIPT_REVIEWER_SECRET_ENV, "").strip()
    if not secret:
        raise SystemExit(
            f"missing reviewer signing secret: {MANUSCRIPT_REVIEWER_SECRET_ENV}"
        )
    expires_at = (
        datetime.now(timezone.utc) + timedelta(minutes=args.ttl_minutes)
    ).isoformat()
    repo = MetaHarnessRepository()
    grant_id = repo.issue_historical_manuscript_grant(
        agenda_id=args.agenda_id,
        experiment_run_id=args.run_id,
        expected_verdict_hash=args.expected_verdict_hash,
        token_cap=args.token_cap,
        expires_at=expires_at,
    )
    run = db.fetchone("SELECT * FROM experiment_runs WHERE id=?", (args.run_id,))
    if not run or int(run.get("resource_grant_id") or 0) != grant_id:
        raise RuntimeError("historical manuscript grant did not bind to the run")
    status = run_manuscript_gate(dict(run), secret=secret)
    print(
        json.dumps(
            {
                "agenda_id": args.agenda_id,
                "run_id": args.run_id,
                "resource_grant_id": grant_id,
                "status": status,
            },
            sort_keys=True,
        )
    )
    return 0 if status in {"manuscript_allowed", "refused"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
