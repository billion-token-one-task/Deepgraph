"""Reconcile research_agendas.gpu_hours_reserved with the ledger it summarises.

The agenda column is a running total of open GPU reservations; the ledger is
the record they are derived from. The invariant is

    research_agendas.gpu_hours_reserved
      == SUM(max(gpu_hours_reserved - gpu_hours_used, 0))
         over agenda_resource_ledger rows still 'reserved'

Three terminal paths used to release a reservation's full cap even after
attempt-level settlement had already released the hours it burned, so the
column drifted negative (agenda 7: -1.644, agenda 10: -4.608 on 2026-08-19)
and a negative reservation fails ResearchAgenda.validate(). The code fix is in
meta_harness/repository.py and agents/agenda_repository.py; this script repairs
the totals those releases already corrupted.

Read-only by default. Pass --apply to write.
"""

from __future__ import annotations

import argparse

from db import database as db

INVARIANT_SQL = """
SELECT COALESCE(SUM(GREATEST(
    COALESCE(gpu_hours_reserved, 0) - COALESCE(gpu_hours_used, 0), 0)), 0) AS v
FROM agenda_resource_ledger
WHERE agenda_id=? AND status='reserved'
"""


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()

    drifted = []
    for agenda in db.fetchall(
        "SELECT id, gpu_hours_reserved FROM research_agendas ORDER BY id"
    ):
        agenda_id = int(agenda["id"])
        stored = float(agenda["gpu_hours_reserved"] or 0.0)
        expected = float(db.fetchone(INVARIANT_SQL, (agenda_id,))["v"])
        if abs(stored - expected) > 1e-9:
            drifted.append((agenda_id, stored, expected))
            print(
                f"agenda {agenda_id}: stored={stored:+.9f} "
                f"ledger={expected:.9f} drift={stored - expected:+.9f}"
            )

    if not drifted:
        print("all agendas already match the ledger invariant")
        return 0
    if not args.apply:
        print(f"{len(drifted)} agenda(s) drifted; re-run with --apply to repair")
        return 1

    for agenda_id, _stored, expected in drifted:
        db.execute(
            "UPDATE research_agendas SET gpu_hours_reserved=?,"
            " updated_at=CURRENT_TIMESTAMP WHERE id=?",
            (expected, agenda_id),
        )
    db.commit()
    print(f"repaired {len(drifted)} agenda(s)")

    for agenda_id, _stored, expected in drifted:
        now = float(
            db.fetchone(
                "SELECT gpu_hours_reserved FROM research_agendas WHERE id=?",
                (agenda_id,),
            )["gpu_hours_reserved"]
        )
        assert abs(now - expected) <= 1e-9, (agenda_id, now, expected)
        print(f"agenda {agenda_id}: now {now:.9f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
