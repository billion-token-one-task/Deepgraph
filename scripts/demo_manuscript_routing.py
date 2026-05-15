"""End-to-end demo of the Manuscript Venue Routing epic (issues #11-#15).

Runs the full D1 → D2 → D3 → D4 surface against an in-memory SQLite DB and
prints a compact transcript so a reviewer can eyeball the contract without
booting the Flask app:

    1. List all 6 registered TemplateAdapters (D1 base + D2 4 venues).
    2. Route 3 fixture states (CV / NLP / ML) through the VenueRouter.
    3. Fire the LLM tiebreaker (deterministic fallback) on the ML state.
    4. Lint a happy-path source + a violating source through FormatLinter.
    5. Persist a lint run and read it back via the public API.
    6. Boot the Flask test client and hit /api/manuscript/* end-to-end.

Usage::

    DEEPGRAPH_DATABASE_URL="" DEEPGRAPH_DB_PATH=/tmp/demo_mr.db \
        python -m scripts.demo_manuscript_routing
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _header(title: str) -> None:
    bar = "=" * 70
    print(f"\n{bar}\n{title}\n{bar}")


def main() -> int:
    os.environ.setdefault("DEEPGRAPH_DATABASE_URL", "")
    os.environ.setdefault("DEEPGRAPH_DB_PATH", "/tmp/demo_mr.db")
    db_path = Path(os.environ["DEEPGRAPH_DB_PATH"])
    if db_path.exists():
        db_path.unlink()

    from db import database as db
    from agents.manuscript_templates import get_adapter, list_adapters
    from agents.venue_router import (
        evaluate_venues,
        load_venue_config,
        needs_tiebreak,
        tiebreak_with_llm,
    )
    from agents.format_linter import lint_manuscript, persist_lint_run, get_lint_run

    db.init_db()

    # ------------------------------------------------------------------
    _header("D1/D2 · Registered TemplateAdapters")
    for tid in list_adapters():
        a = get_adapter(tid)
        print(
            f"  - {a.template_id:>14}  "
            f"col={a.column_layout:>13}  "
            f"bib={a.bibstyle_name:<22}  max_pages={a.max_pages}"
        )

    # ------------------------------------------------------------------
    _header("D1/D2 · Venue Router on 3 domain fixtures")
    fixtures = {
        "CV": {
            "title": "Diffusion-based image detection at scale",
            "claim_type": "empirical",
            "domain": "vision",
            "has_real_data": True,
            "tier": 1,
            "page_count_estimate": 8,
        },
        "NLP": {
            "title": "Tokenization for low-resource language model translation",
            "claim_type": "empirical",
            "domain": "nlp",
            "has_real_data": True,
            "tier": 1,
            "page_count_estimate": 9,
        },
        "ML": {
            "title": "Stochastic optimization for deep learning generalization",
            "claim_type": "empirical",
            "domain": "ml",
            "has_real_data": True,
            "tier": 1,
            "page_count_estimate": 9,
        },
    }
    venues = load_venue_config()
    ml_scored = None
    for name, state in fixtures.items():
        ev = evaluate_venues(state, venues)
        chosen = ev["selected"]["venue"].template_id if ev["selected"] else None
        score = ev["selected"]["breakdown"]["score"] if ev["selected"] else None
        print(f"  - {name:>3}: chosen={chosen}  score={score}")
        if name == "ML":
            ml_scored = ev["all_scored"]

    # ------------------------------------------------------------------
    _header("D3 · LLM Tiebreaker (deterministic fallback)")
    print(f"  needs_tiebreak(ml_scored) = {needs_tiebreak(ml_scored)}")
    tb = tiebreak_with_llm(fixtures["ML"], ml_scored)
    print(f"  tiebreak (no llm_caller) = {json.dumps(tb, indent=2)}")

    # ------------------------------------------------------------------
    _header("D3 · FormatLinter happy vs violating source")
    iclr = get_adapter("iclr2026")
    happy = iclr.normalize_source(
        r"\documentclass{article}" "\n"
        r"\begin{document}" "\n"
        r"Body." "\n"
        r"\bibliography{refs}" "\n"
        r"\end{document}" "\n"
    )
    bad = r"\begin{document}body\end{document}"
    happy_result = lint_manuscript(happy, iclr, page_count=8)
    bad_result = lint_manuscript(bad, iclr, page_count=15)
    print(f"  happy: pass={happy_result['pass']}  summary={happy_result['summary']}")
    print(f"  bad:   pass={bad_result['pass']}  summary={bad_result['summary']}")

    # ------------------------------------------------------------------
    _header("D3 · Persist + read-back lint run")
    run_id = persist_lint_run(selection_id=12345, adapter=iclr, lint_result=happy_result)
    print(f"  inserted run_id={run_id}")
    readback = get_lint_run(run_id)
    print(
        f"  readback: template_id={readback['template_id']}  "
        f"selection_id={readback['selection_id']}  pass={readback['pass']}"
    )

    # ------------------------------------------------------------------
    _header("D4 · Flask /api/manuscript/* end-to-end")
    from web import app as app_module
    client = app_module.app.test_client()

    r1 = client.get("/api/manuscript/venues")
    print(f"  GET  /api/manuscript/venues             → {r1.status_code} ({len(r1.get_json()['venues'])} venues)")

    r2 = client.post("/api/manuscript/route", json=fixtures["CV"])
    print(
        f"  POST /api/manuscript/route (CV state)    → {r2.status_code}  "
        f"chosen={r2.get_json()['selected']['template_id']}"
    )

    r3 = client.post(
        "/api/manuscript/route",
        json={**fixtures["ML"], "include_tiebreak": True},
    )
    rj = r3.get_json()
    print(
        f"  POST /api/manuscript/route (ML +tb)      → {r3.status_code}  "
        f"needs_tiebreak={rj['needs_tiebreak']}  "
        f"tiebreak={rj['tiebreak']['chosen_template_id']}"
    )

    r4 = client.post(
        "/api/manuscript/lint",
        json={
            "template_id": "iclr2026",
            "source": r"\documentclass{article}" "\n"
                      r"\begin{document}body\bibliography{refs}\end{document}",
            "page_count": 8,
        },
    )
    print(
        f"  POST /api/manuscript/lint (preview)      → {r4.status_code}  "
        f"pass={r4.get_json()['lint']['pass']}"
    )

    r5 = client.post(
        "/api/manuscript/lint/77777",
        json={
            "template_id": "iclr2026",
            "source": r"\documentclass{article}\usepackage{graphicx}"
                      r"\usepackage{amsmath}\usepackage{hyperref}"
                      r"\begin{document}body"
                      r"\bibliographystyle{iclr2026_conference}\bibliography{refs}\end{document}",
            "page_count": 8,
            "normalize": False,
        },
    )
    persisted_run_id = r5.get_json()["run_id"]
    print(f"  POST /api/manuscript/lint/77777          → {r5.status_code}  run_id={persisted_run_id}")

    r6 = client.get(f"/api/manuscript/lint_run/{persisted_run_id}")
    print(f"  GET  /api/manuscript/lint_run/{persisted_run_id}  → {r6.status_code}  template_id={r6.get_json()['lint_run']['template_id']}")

    print("\n[ok] demo complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
