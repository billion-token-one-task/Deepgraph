"""Audit generated manuscript bundles for top-venue readiness."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from db import database as db


ICLR_REQUIRED_FILES = {
    "iclr2026_conference.sty",
    "iclr2026_conference.bst",
    "math_commands.tex",
    "natbib.sty",
    "fancyhdr.sty",
}
CONTRACT_FILES = {
    "problem_awareness.json",
    "publication_evidence_contract.json",
    "evidence_manifest.json",
    "claim_evidence_matrix.json",
    "reviewer_report.json",
    "paper_quality_report.json",
}


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""


def _issue(severity: str, text: str) -> dict[str, str]:
    return {"severity": severity, "issue": text}


def _load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _load_claim_values(root: Path) -> dict[str, Any]:
    for relative in ("claim_values.json", "audited_results/claim_values.json"):
        payload = _load_json(root / relative)
        if payload:
            return payload
    return {}


def _available_contract_files(root: Path, root_files: set[str]) -> set[str]:
    available = set(root_files)
    audited_results = root / "audited_results"
    if audited_results.is_dir():
        available.update(path.name for path in audited_results.iterdir() if path.is_file())
    return available


def _sentences(text: str) -> list[str]:
    return [part.strip() for part in re.split(r"(?<=[.!?])\s+", text) if part.strip()]


def _top_venue_overclaim_issues(root: Path, main_tex: str) -> list[dict[str, str]]:
    claim_values = _load_claim_values(root)
    decision = str(claim_values.get("top_venue_general_superiority_decision") or "").lower()
    if decision and not decision.startswith("blocked"):
        return []
    patterns = [
        re.compile(r"\b(state[- ]of[- ]the[- ]art|sota)\b.{0,120}\b(result|performance|achiev|outperform|superior|sets?)\b", re.I),
        re.compile(r"\b(outperform[s]?|beats?|surpass(?:es)?|is superior to)\b.{0,120}\b(adaptive|routing|state[- ]of[- ]the[- ]art|sota|current)\b", re.I),
        re.compile(r"\bfirst\b(?!\s*,).{0,100}\b(adaptive reasoning|selective deliberation|reasoning routing)\b", re.I),
    ]
    guarded_terms = re.compile(r"\b(not|cannot|blocked|requires?|unless|without|pending|only if|not by itself|must not)\b", re.I)
    issues: list[dict[str, str]] = []
    for sentence in _sentences(main_tex):
        if guarded_terms.search(sentence):
            continue
        if any(pattern.search(sentence) for pattern in patterns):
            issues.append(
                _issue(
                    "high",
                    "Manuscript asserts top-venue/SOTA/general adaptive-reasoning superiority while claim_values blocks that claim.",
                )
            )
            break
    return issues


def _adaptive_prior_art_issues(main_tex: str) -> list[dict[str, str]]:
    lowered = main_tex.lower()
    discusses_adaptive_reasoning = re.search(
        r"\b(adaptive[- ]reasoning|adaptive[- ]routing|reasoning routing|selective reasoning|selective deliberation|adaptive compute|adaptive-compute)\b",
        lowered,
    )
    if not discusses_adaptive_reasoning:
        return []
    required = {
        "CAR / certainty-based adaptive routing": (
            "lu2025car",
            "certainty-based adaptive",
            "car-style",
            "certainty adaptive routing",
        ),
        "Self-Route": ("he2025selfroute", "self-route", "self route"),
        "Rational Metareasoning": ("desabbata2025rational", "rational metareasoning"),
        "Route-to-Reason": ("pan2025rtr", "route-to-reason", "route to reason"),
        "RouteLLM": ("ong2024routellm", "routellm", "route llm"),
    }
    missing = [label for label, needles in required.items() if not any(needle in lowered for needle in needles)]
    if not missing:
        return []
    return [
        _issue(
            "high",
            "Adaptive-reasoning/routing manuscript is missing required nearby prior-art acknowledgement: "
            + ", ".join(missing)
            + ".",
        )
    ]


def _placeholder_result_issues(main_tex: str) -> list[dict[str, str]]:
    patterns = [
        r"no main-result number is reported until",
        r"values are intentionally blank",
        r"result cells? (?:is|are) blank",
        r"evidence[- ]pending",
        r"withholding any superiority claim until audited artifacts exist",
        r"empirical claim remains pending",
        r"---\s*&\s*---",
        r"\&\s*--\s*\&",
    ]
    lowered = main_tex.lower()
    if any(re.search(pattern, lowered, re.I) for pattern in patterns):
        return [
            _issue(
                "high",
                "Submission manuscript still contains evidence-pending or blank-result placeholders.",
            )
        ]
    return []


def audit_bundle_path(bundle_path: str | Path, *, bundle_format: str = "conference") -> dict[str, Any]:
    """Return a deterministic quality audit for one submission bundle path."""
    root = Path(bundle_path)
    issues: list[dict[str, str]] = []
    main_path = root / "main.tex"
    main_tex = _read_text(main_path)
    if not root.exists():
        issues.append(_issue("high", "Bundle directory is missing."))
        return {
            "bundle_path": str(root),
            "bundle_format": bundle_format,
            "status": "block",
            "issues": issues,
        }
    if (root / "MANUSCRIPT_BLOCKED.json").exists() or (root / "DO_NOT_SUBMIT.md").exists():
        issues.append(_issue("high", "Bundle is marked manuscript_blocked/stale and must not be submitted."))
    if not main_tex:
        issues.append(_issue("high", "main.tex is missing or empty."))
    files = {path.name for path in root.iterdir() if path.is_file()}
    first_kb = main_tex[:1200].lower()
    is_conference = str(bundle_format or "").lower() == "conference"

    if is_conference:
        if "iclr2026_conference" not in main_tex:
            issues.append(_issue("high", "Conference paper does not use the ICLR 2026 style."))
        missing_template = sorted(ICLR_REQUIRED_FILES - files)
        if missing_template:
            issues.append(_issue("high", "Conference bundle is missing ICLR template files: " + ", ".join(missing_template) + "."))
        if "\\documentclass{article}" in first_kb and "iclr2026_conference" not in main_tex:
            issues.append(_issue("high", "Conference paper still uses the generic article class."))

    missing_contracts = sorted(CONTRACT_FILES - _available_contract_files(root, files))
    if missing_contracts:
        issues.append(_issue("high", "Bundle is missing paper contract/audit files: " + ", ".join(missing_contracts) + "."))
    if re.search(r"\\author\{[^}]*your name", main_tex, re.IGNORECASE):
        issues.append(_issue("high", "Paper still contains a placeholder author."))
    if len(main_tex) < 12_000:
        issues.append(_issue("high", "main.tex is too short for a full top-conference submission."))
    if not (root / "main.pdf").exists():
        issues.append(_issue("high", "Compiled main.pdf is missing."))
    if len(re.findall(r"\\cite[t|p]?\{", main_tex)) < 8:
        issues.append(_issue("medium", "Citation count is below a full-paper target."))
    if len(re.findall(r"\\includegraphics", main_tex)) < 2:
        issues.append(_issue("medium", "Figure count is below a benchmark-paper target."))
    issues.extend(_top_venue_overclaim_issues(root, main_tex))
    issues.extend(_adaptive_prior_art_issues(main_tex))
    issues.extend(_placeholder_result_issues(main_tex))

    high_count = sum(1 for item in issues if item.get("severity") == "high")
    return {
        "bundle_path": str(root),
        "bundle_format": bundle_format,
        "status": "block" if high_count else "pass",
        "high_issue_count": high_count,
        "main_tex_chars": len(main_tex),
        "issues": issues,
    }


def audit_ready_submission_bundles(*, limit: int = 50, mark_stale: bool = True) -> dict[str, Any]:
    """Audit DB bundles currently marked ready and optionally mark weak ones stale."""
    db.init_db()
    rows = db.fetchall(
        """
        SELECT sb.*, mr.experiment_run_id, mr.deep_insight_id
        FROM submission_bundles sb
        LEFT JOIN manuscript_runs mr ON mr.id = sb.manuscript_run_id
        WHERE sb.status='ready'
        ORDER BY sb.created_at DESC
        LIMIT ?
        """,
        (limit,),
    )
    reports: list[dict[str, Any]] = []
    stale_marked = 0
    for row in rows:
        report = audit_bundle_path(row.get("bundle_path") or "", bundle_format=row.get("bundle_format") or "conference")
        report["submission_bundle_id"] = row.get("id")
        report["manuscript_run_id"] = row.get("manuscript_run_id")
        report["experiment_run_id"] = row.get("experiment_run_id")
        report["deep_insight_id"] = row.get("deep_insight_id")
        reports.append(report)
        if report.get("status") != "block" or not mark_stale:
            continue

        reason = "Manuscript watchdog marked bundle stale: " + "; ".join(
            item["issue"] for item in report.get("issues", []) if item.get("severity") == "high"
        )[:1000]
        bundle_id = int(row["id"])
        manuscript_run_id = int(row["manuscript_run_id"])
        db.execute("UPDATE submission_bundles SET status='stale' WHERE id=?", (bundle_id,))
        remaining = db.fetchone(
            "SELECT COUNT(*) AS c FROM submission_bundles WHERE manuscript_run_id=? AND id<>? AND status='ready'",
            (manuscript_run_id, bundle_id),
        )
        if not remaining or int(remaining.get("c") or 0) == 0:
            db.execute(
                "UPDATE manuscript_runs SET status='stale', updated_at=CURRENT_TIMESTAMP WHERE id=?",
                (manuscript_run_id,),
            )
            run_id = row.get("experiment_run_id")
            if run_id is not None:
                db.execute(
                    """
                    UPDATE experiment_runs
                    SET status=CASE WHEN status='bundle_ready' THEN 'completed' ELSE status END,
                        error_message=CASE
                            WHEN status IN ('bundle_ready', 'completed') THEN ?
                            ELSE error_message
                        END
                    WHERE id=?
                    """,
                    (reason, int(run_id)),
                )
            insight_id = row.get("deep_insight_id")
            if insight_id is not None:
                db.execute(
                    "UPDATE deep_insights SET submission_status='stale', updated_at=CURRENT_TIMESTAMP WHERE id=?",
                    (int(insight_id),),
                )
                db.execute(
                    """
                    UPDATE auto_research_jobs
                    SET status=CASE WHEN status='bundle_ready' THEN 'completed' ELSE status END,
                        stage='manuscript_stale',
                        last_error=?,
                        last_note='Latest ready manuscript bundle failed watchdog audit.',
                        updated_at=CURRENT_TIMESTAMP,
                        last_checked_at=CURRENT_TIMESTAMP
                    WHERE deep_insight_id=?
                    """,
                    (reason, int(insight_id)),
                )
        db.commit()
        stale_marked += 1

    reconciled = reconcile_stale_manuscript_jobs(limit=limit) if mark_stale else 0
    return {
        "audited": len(rows),
        "stale_marked": stale_marked,
        "auto_jobs_reconciled": reconciled,
        "reports": reports,
    }


def reconcile_stale_manuscript_jobs(*, limit: int = 50) -> int:
    """Keep Auto Research from advertising stale manuscripts as bundle-ready."""
    db.init_db()
    rows = db.fetchall(
        """
        SELECT arj.deep_insight_id, arj.experiment_run_id, mr.id AS manuscript_run_id
        FROM auto_research_jobs arj
        JOIN manuscript_runs mr
          ON mr.deep_insight_id = arj.deep_insight_id
         AND (arj.experiment_run_id IS NULL OR mr.experiment_run_id = arj.experiment_run_id)
        WHERE arj.status='bundle_ready'
          AND mr.status='stale'
        ORDER BY mr.updated_at DESC
        LIMIT ?
        """,
        (limit,),
    )
    for row in rows:
        note = "Latest manuscript bundle is stale; a new evidence-backed ICLR bundle is required."
        db.execute(
            """
            UPDATE auto_research_jobs
            SET status='completed',
                stage='manuscript_stale',
                last_error=?,
                last_note=?,
                updated_at=CURRENT_TIMESTAMP,
                last_checked_at=CURRENT_TIMESTAMP
            WHERE deep_insight_id=?
            """,
            (note, note, int(row["deep_insight_id"])),
        )
    if rows:
        db.commit()
    return len(rows)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Audit manuscript bundles for top-venue readiness.")
    parser.add_argument("--bundle-path", help="Audit one manuscript bundle directory.")
    parser.add_argument("--bundle-format", default="conference", help="Bundle format label; default: conference.")
    parser.add_argument("--audit-ready", action="store_true", help="Audit DB bundles currently marked ready.")
    parser.add_argument("--limit", type=int, default=50, help="Maximum ready bundles to audit.")
    parser.add_argument(
        "--no-mark-stale",
        action="store_true",
        help="Report ready-bundle failures without marking DB rows stale.",
    )
    args = parser.parse_args(argv)

    if args.bundle_path:
        report = audit_bundle_path(args.bundle_path, bundle_format=args.bundle_format)
        print(json.dumps(report, indent=2, sort_keys=True))
        return 1 if report.get("status") == "block" else 0

    if args.audit_ready:
        report = audit_ready_submission_bundles(limit=args.limit, mark_stale=not args.no_mark_stale)
        print(json.dumps(report, indent=2, sort_keys=True))
        return 1 if any(item.get("status") == "block" for item in report.get("reports", [])) else 0

    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
