"""Find active grants that were issued for a run that never came.

A grant is issued, then the forge dies before it creates the run -- at the
review gate, most often -- and nothing ever attaches, consumes or settles it.
It holds an agenda concurrency slot until its TTL, which for a pilot grant is
24 hours.

Measured 2026-08-20 23:5x UTC:

    agenda   cap  active  orphaned
      1       1      1       1        completely blocked, by one orphan
     14       4      4       3        M3's agenda, at cap
     11       4      3       3
     10       7      2       2

Nine orphans across four agendas, the oldest 931 minutes -- more than fifteen
hours of a slot held for work that had already failed.

This is not a new mechanism. `MetaHaronessRepository.revoke_grant` already
withdraws an unused grant and refunds it, and refuses outright if the grant has
metered any usage. What was missing was a caller: nothing looks for orphans, so
nothing ever revokes one.

That refusal is also why this is safe to be wrong about. If a grant judged
orphaned had in fact done work, revoke_grant declines and the script reports it
-- the safety is in the API, not in this script's judgement.

Read-only unless --apply. Run it and read the table before you pass --apply.

Usage:
    python scripts/repair_orphaned_grants.py [--agenda N] [--min-age-minutes 120] [--apply]
"""

from __future__ import annotations

import argparse

from db import database as db
from meta_harness.repository import MetaHarnessRepository

# Measured attachment latency for pilot grants that DID get a run: median 110s,
# p90 4353s (73 minutes). The p90 is the number that matters -- a threshold
# below it would revoke grants whose run was still coming. Two hours clears it
# with room, and every specimen found today was older than 154 minutes.
#
# The maximum observed was 41349s, and the minimum was NEGATIVE (a run created
# before its grant), which is what re-attachment looks like. Time alone is
# therefore not proof, which is why the run-state condition below is required
# as well.
DEFAULT_MIN_AGE_MINUTES = 120


def find_orphans(*, agenda_id: int | None, min_age_minutes: int) -> list[dict]:
    """Active grants with no run attached, old enough that none is coming."""
    clause = "AND g.agenda_id=?" if agenda_id else ""
    params: list = [min_age_minutes]
    if agenda_id:
        params.append(int(agenda_id))
    rows = db.fetchall(
        f"""
        SELECT g.id, g.agenda_id, g.idea_id, g.stage,
               EXTRACT(EPOCH FROM (now() - g.created_at))/60 AS age_minutes
        FROM resource_grants g
        WHERE g.status='active'
          AND EXTRACT(EPOCH FROM (now() - g.created_at))/60 >= ?
          AND NOT EXISTS (
            SELECT 1 FROM experiment_runs e WHERE e.resource_grant_id = g.id
          )
          {clause}
        ORDER BY age_minutes DESC
        """,
        tuple(params),
    )
    out = []
    for row in rows:
        record = dict(row)
        # A live run for the same idea means the work is still moving and a
        # grant may yet be attached to it. Only an idea whose every run has
        # already failed is unambiguously done with this grant.
        live = db.fetchone(
            """
            SELECT COUNT(*) AS c FROM experiment_runs
            WHERE deep_insight_id=? AND status NOT IN ('failed', 'archived', 'cancelled')
            """,
            (int(record["idea_id"]),),
        )
        record["live_runs"] = int(dict(live or {}).get("c") or 0)
        out.append(record)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--agenda", type=int, default=None)
    parser.add_argument("--min-age-minutes", type=int, default=DEFAULT_MIN_AGE_MINUTES)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()

    orphans = find_orphans(
        agenda_id=args.agenda, min_age_minutes=args.min_age_minutes
    )
    if not orphans:
        print("no orphaned grants found")
        return 0

    print(f"{len(orphans)} active grant(s) with no run attached:")
    print(f"{'grant':>6} {'agenda':>7} {'idea':>6} {'stage':<16} {'age(min)':>9} {'live runs':>10}")
    for row in orphans:
        print(
            f"{row['id']:>6} {row['agenda_id']:>7} {row['idea_id']:>6} "
            f"{str(row['stage']):<16} {float(row['age_minutes']):>9.0f} {row['live_runs']:>10}"
        )

    revocable = [row for row in orphans if row["live_runs"] == 0]
    held_back = len(orphans) - len(revocable)
    if held_back:
        print(
            f"\n{held_back} skipped: the idea still has a live run, so a grant "
            "may yet attach to it."
        )
    if not args.apply:
        print(f"\nread-only. {len(revocable)} would be revoked; pass --apply to act.")
        return 0

    repo = MetaHarnessRepository()
    revoked = refused = 0
    for row in revocable:
        try:
            ok = repo.revoke_grant(
                int(row["id"]),
                agenda_id=int(row["agenda_id"]),
                reason=(
                    f"orphaned: no run ever attached, idea {row['idea_id']} has no "
                    f"live run, age {float(row['age_minutes']):.0f}min"
                ),
            )
        except Exception as exc:
            print(f"  grant {row['id']}: refused by revoke_grant: {exc}")
            refused += 1
            continue
        if ok:
            revoked += 1
            print(f"  grant {row['id']}: revoked, slot returned")
        else:
            refused += 1
            print(f"  grant {row['id']}: revoke_grant declined")
    print(f"\nrevoked {revoked}, declined {refused}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
