#!/usr/bin/env python3
"""Import research artifacts from a legacy DeepGraph bundle into the current DB."""

from __future__ import annotations

import json
import re
import shutil
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path("/home/billion-token")
DB_PATH = ROOT / "Deepgraph" / "deepgraph.db"
LEGACY_RESEARCH_ROOT = Path("/tmp/deepgraph_old_bundle/research")
CURRENT_RESEARCH_ROOT = ROOT / "research"


@dataclass
class ImportResult:
    copied_dirs: list[str]
    skipped_dirs: list[str]
    inserted_insights: list[dict]
    updated_insights: list[dict]


def first_heading(text: str) -> str | None:
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("# "):
            return line[2:].strip()
    return None


def first_nonempty_paragraph(text: str) -> str:
    lines = text.splitlines()
    started = False
    paragraph: list[str] = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            if started and paragraph:
                break
            continue
        if stripped.startswith("#"):
            continue
        started = True
        paragraph.append(stripped)
    return " ".join(paragraph)


def section_text(text: str, heading: str) -> str | None:
    needle = heading.strip().lower()
    lines = text.splitlines()
    for i, raw in enumerate(lines):
        if raw.strip().lstrip("#").strip().lower() != needle:
            continue
        out: list[str] = []
        for nxt in lines[i + 1 :]:
            stripped = nxt.strip()
            if stripped.startswith("#"):
                break
            out.append(nxt)
        value = "\n".join(out).strip()
        return value or None
    return None


def infer_novelty_status(title: str, text: str) -> str:
    hay = f"{title}\n{text}".lower()
    if "partially supported" in hay or "closest precursor" in hay or "overlaps" in hay:
        return "partially_exists"
    if "still open" in hay or "does not appear to have been done yet" in hay or "no identified" in hay:
        return "novel"
    return "unchecked"


def build_deep_insight_payload(dir_name: str, report_path: Path, dest_dir: Path) -> dict | None:
    if not re.match(r"^insight_\d+_", dir_name):
        return None

    text = report_path.read_text(errors="ignore")
    title = first_heading(text) or dir_name
    summary = first_nonempty_paragraph(text)
    current_gap = (
        section_text(text, "Current gap")
        or section_text(text, "What remains missing")
        or section_text(text, "Problem statement")
    )
    novelty_status = infer_novelty_status(title, text)
    imported_at = datetime.now(timezone.utc).isoformat()

    return {
        "tier": 2,
        "status": "verified",
        "title": title,
        "problem_statement": summary or None,
        "existing_weakness": current_gap,
        "evidence_summary": summary or current_gap,
        "related_work_positioning": current_gap,
        "novelty_status": novelty_status,
        "novelty_report": json.dumps(
            {
                "source": "legacy_bundle_import",
                "legacy_dir": dir_name,
                "imported_at": imported_at,
                "report_path": str(report_path),
                "summary": summary,
            },
            ensure_ascii=True,
        ),
        "evoscientist_workdir": str(dest_dir),
    }


def copy_legacy_dirs() -> tuple[list[str], list[str]]:
    copied: list[str] = []
    skipped: list[str] = []
    CURRENT_RESEARCH_ROOT.mkdir(parents=True, exist_ok=True)
    for src in sorted(LEGACY_RESEARCH_ROOT.iterdir()):
        if not src.is_dir():
            continue
        dest = CURRENT_RESEARCH_ROOT / src.name
        if dest.exists():
            skipped.append(src.name)
            continue
        shutil.copytree(src, dest)
        copied.append(src.name)
    return copied, skipped


def import_deep_insights(conn: sqlite3.Connection) -> tuple[list[dict], list[dict]]:
    inserted: list[dict] = []
    updated: list[dict] = []

    for report_path in sorted(LEGACY_RESEARCH_ROOT.glob("*/final_report.md")):
        dir_name = report_path.parent.name
        dest_dir = CURRENT_RESEARCH_ROOT / dir_name
        payload = build_deep_insight_payload(dir_name, report_path, dest_dir)
        if not payload:
            continue

        existing = conn.execute(
            "SELECT id, evoscientist_workdir FROM deep_insights WHERE title = ?",
            (payload["title"],),
        ).fetchone()

        if existing:
            if not existing["evoscientist_workdir"]:
                conn.execute(
                    """UPDATE deep_insights
                       SET evoscientist_workdir = ?, updated_at = CURRENT_TIMESTAMP
                       WHERE id = ?""",
                    (payload["evoscientist_workdir"], existing["id"]),
                )
                updated.append({"id": existing["id"], "title": payload["title"]})
            continue

        cur = conn.execute(
            """INSERT INTO deep_insights (
                   tier, status, title,
                   problem_statement, existing_weakness,
                   related_work_positioning, evidence_summary,
                   novelty_status, novelty_report, evoscientist_workdir
               ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                payload["tier"],
                payload["status"],
                payload["title"],
                payload["problem_statement"],
                payload["existing_weakness"],
                payload["related_work_positioning"],
                payload["evidence_summary"],
                payload["novelty_status"],
                payload["novelty_report"],
                payload["evoscientist_workdir"],
            ),
        )
        insight_id = cur.lastrowid
        conn.execute(
            """INSERT INTO auto_research_jobs (
                   deep_insight_id, status, stage, research_workdir, last_note
               ) VALUES (?, ?, ?, ?, ?)
               ON CONFLICT(deep_insight_id) DO UPDATE SET
                   research_workdir=excluded.research_workdir,
                   last_note=excluded.last_note,
                   updated_at=CURRENT_TIMESTAMP""",
            (
                insight_id,
                "completed",
                "imported_legacy_research",
                str(dest_dir),
                "Imported final_report.md from legacy bundle.",
            ),
        )
        inserted.append({"id": insight_id, "title": payload["title"]})

    return inserted, updated


def main() -> None:
    if not LEGACY_RESEARCH_ROOT.exists():
        raise SystemExit(f"Missing legacy research directory: {LEGACY_RESEARCH_ROOT}")
    if not DB_PATH.exists():
        raise SystemExit(f"Missing database: {DB_PATH}")

    copied_dirs, skipped_dirs = copy_legacy_dirs()

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        inserted_insights, updated_insights = import_deep_insights(conn)
        conn.commit()
    finally:
        conn.close()

    result = ImportResult(
        copied_dirs=copied_dirs,
        skipped_dirs=skipped_dirs,
        inserted_insights=inserted_insights,
        updated_insights=updated_insights,
    )
    print(
        json.dumps(
            {
                "copied_dirs": result.copied_dirs,
                "skipped_dirs": result.skipped_dirs,
                "inserted_insights": result.inserted_insights,
                "updated_insights": result.updated_insights,
            },
            ensure_ascii=True,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
