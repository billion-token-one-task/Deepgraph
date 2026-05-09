"""Audit generated manuscript bundles for top-venue readiness."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

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

    missing_contracts = sorted(CONTRACT_FILES - files)
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
