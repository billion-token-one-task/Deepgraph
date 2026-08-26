#!/usr/bin/env python3
"""Show or set an agenda's compute backend allowlist.

Operator tool: the allowlist is resource governance (which compute pools an
agenda may draw grants against), not a scientific gate. Every change is
printed before/after so the ops log carries the rollback value.

    agenda_backends.py show
    agenda_backends.py set --agenda 7 --backends cpu,llm,colab_gpu
    agenda_backends.py activate --agenda 14
    agenda_backends.py deactivate --agenda 14
    agenda_backends.py create --spec new-agenda.json
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agents.agenda_repository import AgendaRepository  # noqa: E402
from db import database as db  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("show")
    setter = sub.add_parser("set")
    setter.add_argument("--agenda", type=int, required=True)
    setter.add_argument("--backends", required=True,
                        help="comma-separated, e.g. cpu,llm,colab_gpu")
    budget = sub.add_parser("budget")
    budget.add_argument("--agenda", type=int, required=True)
    budget.add_argument("--tokens", type=int, default=None)
    budget.add_argument("--gpu-hours", type=float, default=None)
    budget.add_argument("--max-concurrency", type=int, default=None)
    creator = sub.add_parser("create")
    creator.add_argument("--spec", required=True,
                         help="path to a JSON agenda spec; see docs/internal/AGENDA_SPEC.md")
    activate = sub.add_parser("activate")
    activate.add_argument("--agenda", type=int, required=True)
    deactivate = sub.add_parser("deactivate")
    deactivate.add_argument("--agenda", type=int, required=True)
    expire = sub.add_parser("expire-grant")
    expire.add_argument("--agenda", type=int, required=True)
    expire.add_argument("--grant", type=int, required=True)
    expire.add_argument("--reason", required=True)
    args = parser.parse_args()

    if args.cmd == "expire-grant":
        from meta_harness.repository import MetaHarnessRepository

        done = MetaHarnessRepository().expire_grant_now(
            args.grant, agenda_id=args.agenda, reason=args.reason
        )
        # The stage list stopped being the whole rule when expire_grant_now
        # widened to execution-stage grants with nothing live attached; an
        # operator reading "must be proposal-stage" concludes the tool is
        # wrong for their grant when the API simply found live work.
        print(
            f"grant {args.grant}: "
            + (
                "expired+reconciled"
                if done
                else "not eligible (must be active, and either a proposal or"
                " evidence_audit grant, or an execution grant with no"
                " unfinished run, compute job or colab request)"
            )
        )
        return 0 if done else 1

    if args.cmd == "create":
        import json
        from contracts.agenda import ResearchAgenda

        spec = json.loads(Path(args.spec).read_text(encoding="utf-8"))
        agenda = ResearchAgenda(**spec)
        agenda.validate()
        agenda_id = AgendaRepository().create(agenda)
        row = dict(db.fetchone(
            "SELECT id, name, status, is_active, token_budget, gpu_hours_budget,"
            " max_concurrency, backend_allowlist_json FROM research_agendas WHERE id=?",
            (agenda_id,)) or {})
        print(f"created agenda {agenda_id}")
        for key, value in row.items():
            print(f"  {key}: {value}")
        return 0

    if args.cmd in ("activate", "deactivate"):
        want_active = args.cmd == "activate"
        try:
            change = AgendaRepository().set_active(args.agenda, want_active)
        except ValueError as exc:
            print(f"agenda {args.agenda}: refused -- {exc}")
            return 1
        print(f"agenda {args.agenda} before: {change['before']}")
        print(f"agenda {args.agenda} after:  {change['after']}")
        return 0

    if args.cmd == "show":
        for row in db.fetchall(
            "SELECT id, status, backend_allowlist_json, token_budget,"
            " token_spent, token_reserved, gpu_hours_budget, gpu_hours_spent"
            " FROM research_agendas ORDER BY id"
        ):
            row = dict(row)
            print(row["id"], row["status"], row["backend_allowlist_json"],
                  "tok", row["token_budget"], "spent", row["token_spent"],
                  "resv", row["token_reserved"], "gpuh", row["gpu_hours_budget"])
        return 0

    if args.cmd == "budget":
        q = ("SELECT token_budget, gpu_hours_budget, max_concurrency"
             " FROM research_agendas WHERE id=?")
        before = dict(db.fetchone(q, (args.agenda,)) or {})
        print(f"agenda {args.agenda} before: {before}")
        AgendaRepository().set_budgets(
            args.agenda,
            token_budget=args.tokens,
            gpu_hours_budget=args.gpu_hours,
            max_concurrency=args.max_concurrency,
        )
        after = dict(db.fetchone(q, (args.agenda,)) or {})
        print(f"agenda {args.agenda} after:  {after}")
        return 0

    before = db.fetchone(
        "SELECT backend_allowlist_json FROM research_agendas WHERE id=?",
        (args.agenda,),
    )
    print(f"agenda {args.agenda} before: {dict(before or {}).get('backend_allowlist_json')}")
    AgendaRepository().set_backend_allowlist(
        args.agenda, [b for b in args.backends.split(",")]
    )
    after = db.fetchone(
        "SELECT backend_allowlist_json FROM research_agendas WHERE id=?",
        (args.agenda,),
    )
    print(f"agenda {args.agenda} after:  {dict(after or {}).get('backend_allowlist_json')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
