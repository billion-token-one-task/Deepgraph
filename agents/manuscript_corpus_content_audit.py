"""Compare the generated manuscript's content profile against the actual
neighbor papers from ``deepgraph.db`` — not a static PDF folder.

Existing ``reference_corpus_audit`` only knew about
``REFERENCE_PDF_CORPUS_DIR`` (a fixed set of exemplar PDFs). The user asked
for a content-quality loop that compares against papers from *our own
library that are in the same taxonomy nodes or that this manuscript cites*.

Pipeline:

1. For a given ``deep_insight_id``, gather neighbor paper IDs from
   ``deep_insights.source_paper_ids`` ∪ ``deep_insights.supporting_papers``
   ∪ paper_ids found by joining ``paper_taxonomy`` on
   ``deep_insights.source_node_ids`` (capped to ``max_neighbors``).
2. For each neighbor paper, parse ``papers.full_text`` into a
   section skeleton (Section name → word count) using a tolerant heading
   detector that handles both ``1. INTRODUCTION`` and ``\\section{...}``
   styles depending on how the paper was extracted.
3. Compute corpus medians: word-count-per-section, section name family
   (Intro / Related / Method / Experiments / Discussion).
4. Diff against the current draft. Generated issues are added to the
   bundle's ``content_audit_report.json`` with structured remediation
   hints (which section to grow, which to add) that R3's quality-rewrite
   loop consumes.

This module is read-only; it never mutates ``main.tex``. The rewrite step
is R3.
"""

from __future__ import annotations

import json
import re
import sqlite3
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from agents.reference_corpus_audit import _tex_plain_text, _word_count

# Section-family canonical names. Each map entry is (family → patterns).
_SECTION_FAMILIES: dict[str, tuple[re.Pattern[str], ...]] = {
    "intro": (re.compile(r"^introduction\b|^1[.\s]+introduction", re.IGNORECASE),),
    "related_work": (
        re.compile(r"^related\s+work\b|^background\b|^prior\s+work\b", re.IGNORECASE),
    ),
    "method": (
        re.compile(
            r"^method(?:ology|s)?\b|^approach\b|^our\s+method\b|^proposed\s+method\b",
            re.IGNORECASE,
        ),
    ),
    "experiments": (
        re.compile(
            r"^experiment(?:s|al)?\b|^evaluation\b|^empirical\b|^results\b",
            re.IGNORECASE,
        ),
    ),
    "discussion": (
        re.compile(r"^discussion\b|^analysis\b|^ablation\b", re.IGNORECASE),
    ),
    "limitations": (
        re.compile(r"^limitation(?:s)?\b|^broader\s+impact\b", re.IGNORECASE),
    ),
    "conclusion": (re.compile(r"^conclusion(?:s)?\b|^summary\b", re.IGNORECASE),),
}


_HEADING_PAT = re.compile(
    r"""
    ^\s*
    (?:\\(?:sub)*section\*?\{(?P<latex_head>[^}]+)\}      # \section{...}
    |                                                        # OR
    (?P<num_head>\d+(?:\.\d+)*\.?\s+[A-Z][A-Z0-9 \-]+)$    # 1. INTRODUCTION
    )
    """,
    re.VERBOSE | re.MULTILINE,
)


# ─────────────────────────── data containers ─────────────────────────────


@dataclass
class SectionSkeleton:
    name: str
    family: str | None
    word_count: int


@dataclass
class NeighborPaperProfile:
    paper_id: str
    title: str
    sections: list[SectionSkeleton]
    total_words: int
    citation_marker_count: int

    @property
    def family_words(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for s in self.sections:
            if s.family:
                out[s.family] = out.get(s.family, 0) + s.word_count
        return out


@dataclass
class ContentAuditIssue:
    severity: str
    kind: str
    message: str
    section_family: str | None = None
    target_words: int | None = None


# ─────────────────────────── helpers ───────────────────────────────────


def _classify_section(name: str) -> str | None:
    # Numbered headings like "3. METHOD" need the "3. " trimmed before family
    # classification; otherwise the leading digit blocks "^method".
    name_l = re.sub(r"^\s*\d+(?:\.\d+)*\.?\s+", "", name.strip()).lower()
    for family, patterns in _SECTION_FAMILIES.items():
        if any(p.search(name_l) for p in patterns):
            return family
    return None


def parse_section_skeleton(text: str) -> list[SectionSkeleton]:
    """Split ``text`` on detected headings and return per-section word counts."""
    if not text:
        return []
    matches = list(_HEADING_PAT.finditer(text))
    if not matches:
        return []
    sections: list[SectionSkeleton] = []
    for i, m in enumerate(matches):
        name = (m.group("latex_head") or m.group("num_head") or "").strip()
        if not name:
            continue
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[start:end]
        # Strip references blocks etc.
        body = re.split(r"\bReferences\b|\bBibliography\b", body, maxsplit=1)[0]
        words = _word_count(body)
        sections.append(
            SectionSkeleton(
                name=name,
                family=_classify_section(name),
                word_count=words,
            )
        )
    return sections


# ─────────────────────────── DB neighbor pull ─────────────────────────


def _connect(db_path: Path) -> sqlite3.Connection:
    con = sqlite3.connect(db_path)
    con.row_factory = sqlite3.Row
    return con


def _safe_load_json_list(raw: str | None) -> list[str]:
    if not raw:
        return []
    try:
        loaded = json.loads(raw)
    except (TypeError, json.JSONDecodeError):
        return []
    if not isinstance(loaded, list):
        return []
    return [str(x) for x in loaded if x]


def _neighbor_paper_ids(
    con: sqlite3.Connection,
    *,
    deep_insight_id: int,
    max_neighbors: int,
) -> list[str]:
    cur = con.execute(
        "SELECT source_paper_ids, supporting_papers, source_node_ids "
        "FROM deep_insights WHERE id=?",
        (deep_insight_id,),
    )
    row = cur.fetchone()
    if row is None:
        return []
    seen: list[str] = []
    seen_set: set[str] = set()

    def _add(pid: str) -> None:
        if pid and pid not in seen_set:
            seen_set.add(pid)
            seen.append(pid)

    for pid in _safe_load_json_list(row["source_paper_ids"]):
        _add(pid)
    for pid in _safe_load_json_list(row["supporting_papers"]):
        _add(pid)
        if len(seen) >= max_neighbors:
            return seen[:max_neighbors]
    node_ids = _safe_load_json_list(row["source_node_ids"])
    if node_ids and len(seen) < max_neighbors:
        q_marks = ",".join("?" * len(node_ids))
        cur = con.execute(
            f"SELECT DISTINCT paper_id FROM paper_taxonomy WHERE node_id IN ({q_marks})",
            node_ids,
        )
        for r in cur.fetchall():
            _add(r[0])
            if len(seen) >= max_neighbors:
                break
    return seen[:max_neighbors]


def _load_paper_profile(
    con: sqlite3.Connection, paper_id: str
) -> NeighborPaperProfile | None:
    cur = con.execute(
        "SELECT id, title, full_text FROM papers WHERE id=?",
        (paper_id,),
    )
    row = cur.fetchone()
    if row is None or not row["full_text"]:
        return None
    sections = parse_section_skeleton(row["full_text"])
    if not sections:
        return None
    total = sum(s.word_count for s in sections)
    refs = len(re.findall(r"\[\d{1,3}\]|\bet al\.", row["full_text"]))
    return NeighborPaperProfile(
        paper_id=row["id"],
        title=row["title"] or "",
        sections=sections,
        total_words=total,
        citation_marker_count=refs,
    )


# ─────────────────────────── audit ────────────────────────────────────


def _median(values: Iterable[float]) -> float | None:
    nums = [float(v) for v in values if v is not None]
    return float(statistics.median(nums)) if nums else None


def diff_against_neighbors(
    main_tex: str,
    neighbors: list[NeighborPaperProfile],
) -> dict[str, Any]:
    plain = _tex_plain_text(main_tex)
    gen_total = _word_count(plain)
    # Map our own section families
    gen_sections = parse_section_skeleton(main_tex)
    gen_family_words: dict[str, int] = {}
    for s in gen_sections:
        if s.family:
            gen_family_words[s.family] = gen_family_words.get(s.family, 0) + s.word_count
    # Corpus medians per family
    corpus_family_words: dict[str, list[int]] = {}
    for nb in neighbors:
        for fam, w in nb.family_words.items():
            corpus_family_words.setdefault(fam, []).append(w)
    family_medians: dict[str, float] = {}
    for fam, vals in corpus_family_words.items():
        med = _median(vals)
        if med is not None:
            family_medians[fam] = med
    issues: list[ContentAuditIssue] = []
    # Missing required families
    for fam in ("intro", "method", "experiments"):
        if gen_family_words.get(fam, 0) < 100 and fam in family_medians:
            issues.append(
                ContentAuditIssue(
                    severity="high",
                    kind=f"missing_family_{fam}",
                    message=(
                        f"draft has <100 words mapped to family '{fam}' "
                        f"while corpus median is {int(family_medians[fam])}"
                    ),
                    section_family=fam,
                    target_words=int(family_medians[fam]),
                )
            )
    # Family thinness (>= 100 words but < 50% of corpus median)
    for fam, med in family_medians.items():
        ours = gen_family_words.get(fam, 0)
        if 0 < ours < max(150, 0.5 * med):
            issues.append(
                ContentAuditIssue(
                    severity="medium",
                    kind=f"thin_family_{fam}",
                    message=(
                        f"family '{fam}' has {ours} words vs corpus median "
                        f"{int(med)} (<50%)"
                    ),
                    section_family=fam,
                    target_words=int(med),
                )
            )
    # Total length sanity
    if neighbors:
        corpus_total = _median(nb.total_words for nb in neighbors)
        if corpus_total and gen_total < 0.4 * corpus_total:
            issues.append(
                ContentAuditIssue(
                    severity="medium",
                    kind="total_length_thin",
                    message=(
                        f"draft total {gen_total} words vs corpus median "
                        f"{int(corpus_total)} (<40%)"
                    ),
                    target_words=int(corpus_total),
                )
            )
    return {
        "generated": {
            "total_words": gen_total,
            "section_count": len(gen_sections),
            "family_words": gen_family_words,
            "sections": [
                {"name": s.name, "family": s.family, "word_count": s.word_count}
                for s in gen_sections
            ],
        },
        "corpus_family_medians": {k: int(v) for k, v in family_medians.items()},
        "neighbor_count": len(neighbors),
        "neighbor_titles": [nb.title for nb in neighbors],
        "issues": [
            {
                "severity": iss.severity,
                "kind": iss.kind,
                "message": iss.message,
                "section_family": iss.section_family,
                "target_words": iss.target_words,
            }
            for iss in issues
        ],
    }


def audit_against_db_corpus(
    *,
    main_tex: str,
    deep_insight_id: int,
    db_path: Path,
    max_neighbors: int = 12,
) -> dict[str, Any]:
    """Top-level entry point.  Returns a structured audit report."""
    if not db_path.exists():
        return {"available": False, "error": "db_missing"}
    con = _connect(db_path)
    try:
        paper_ids = _neighbor_paper_ids(
            con,
            deep_insight_id=deep_insight_id,
            max_neighbors=max_neighbors,
        )
        if not paper_ids:
            return {"available": False, "error": "no_neighbor_papers"}
        profiles: list[NeighborPaperProfile] = []
        for pid in paper_ids:
            p = _load_paper_profile(con, pid)
            if p is not None:
                profiles.append(p)
        if not profiles:
            return {"available": False, "error": "neighbors_have_no_full_text"}
        diff = diff_against_neighbors(main_tex, profiles)
        return {
            "available": True,
            "schema_version": "content_corpus_audit_v1",
            "deep_insight_id": deep_insight_id,
            "neighbor_paper_ids": [p.paper_id for p in profiles],
            **diff,
        }
    finally:
        con.close()


def write_content_audit(
    bundle_dir: Path,
    *,
    deep_insight_id: int,
    db_path: Path,
) -> dict[str, Any]:
    bundle_dir = Path(bundle_dir)
    main_tex = bundle_dir / "main.tex"
    if not main_tex.is_file():
        return {"available": False, "error": "main_tex_missing"}
    text = main_tex.read_text(encoding="utf-8", errors="replace")
    report = audit_against_db_corpus(
        main_tex=text,
        deep_insight_id=deep_insight_id,
        db_path=db_path,
    )
    (bundle_dir / "content_audit_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    return report
