"""Bibliography and citation-placement auditor for manuscripts."""

from __future__ import annotations

import re
from typing import Any


AUDITOR_VERSION = "deepgraph_reference_auditor_v1_2026_06_10"

REFERENCE_AUDIT_STANDARD_TEXT = """Reference standard:
- A full paper must include at least 30 real bibliography entries and at least 30 distinct entries cited in the main text; aim for roughly 50 when the topic has enough literature.
- Every cited key must exist in references.bib; no invented, placeholder, or dangling citation keys are allowed.
- Each bibliography entry must include title, year, and author/editor/organization metadata, plus publication venue or a DOI/arXiv/URL identifier whenever available.
- Citations must not appear in the Abstract.
- Citations must not be stuffed into the contribution bullets or contribution paragraph.
- The main citation load should be in Introduction, Related Work, and Method. Experiments may cite datasets and baselines, but cannot compensate for a thin Related Work.
- Related Work should cite by category using small citation clusters, not one large undifferentiated citation dump."""


CITE_RE = re.compile(r"\\cite[a-zA-Z*]*(?:\[[^\]]*\]){0,2}\{([^}]+)\}", re.IGNORECASE)
BIB_ENTRY_RE = re.compile(r"@\s*([A-Za-z]+)\s*\{\s*([^,\s]+)\s*,(.*?)(?=\n\s*@|\Z)", re.DOTALL)
FIELD_RE = re.compile(
    r"([A-Za-z][A-Za-z0-9_-]*)\s*=\s*(\{(?:[^{}]|\{[^{}]*\})*\}|\"[^\"]*\"|[^,\n]+)",
    re.DOTALL,
)
ABSTRACT_RE = re.compile(r"\\begin\{abstract\}(.*?)\\end\{abstract\}", re.IGNORECASE | re.DOTALL)
SECTION_RE = re.compile(r"\\section\*?\{([^}]+)\}", re.IGNORECASE)

PLACEHOLDER_TERMS = (
    "todo",
    "tbd",
    "placeholder",
    "unknown",
    "missing",
    "dummy",
    "lorem ipsum",
    "sample citation",
    "paper title",
)


def _strip_braces(value: str) -> str:
    value = (value or "").strip().strip(",")
    if (value.startswith("{") and value.endswith("}")) or (value.startswith('"') and value.endswith('"')):
        value = value[1:-1]
    value = re.sub(r"\s+", " ", value.replace("\n", " ")).strip()
    return value


def parse_bib_entries(bibtex: str) -> dict[str, dict[str, Any]]:
    entries: dict[str, dict[str, Any]] = {}
    for match in BIB_ENTRY_RE.finditer(bibtex or ""):
        entry_type = match.group(1).strip().lower()
        key = match.group(2).strip()
        body = match.group(3) or ""
        fields: dict[str, str] = {}
        for field_match in FIELD_RE.finditer(body):
            fields[field_match.group(1).strip().lower()] = _strip_braces(field_match.group(2))
        entries[key] = {"entry_type": entry_type, "fields": fields, "raw": match.group(0)}
    return entries


def cited_keys(tex: str) -> list[str]:
    ordered: list[str] = []
    for raw in CITE_RE.findall(tex or ""):
        for key in raw.split(","):
            cleaned = key.strip()
            if cleaned and cleaned not in ordered:
                ordered.append(cleaned)
    return ordered


def _abstract_body(tex: str) -> str:
    match = ABSTRACT_RE.search(tex or "")
    return match.group(1) if match else ""


def _top_level_sections(tex: str) -> list[dict[str, Any]]:
    matches = list(SECTION_RE.finditer(tex or ""))
    sections: list[dict[str, Any]] = []
    for idx, match in enumerate(matches):
        start = match.end()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(tex or "")
        sections.append({"title": (match.group(1) or "").strip(), "body": (tex or "")[start:end]})
    return sections


def _canonical_section(title: str) -> str:
    lower = (title or "").strip().lower()
    if "intro" in lower:
        return "introduction"
    if "related" in lower or "prior work" in lower or lower == "background":
        return "related_work"
    if any(token in lower for token in ("method", "approach", "framework", "model", "algorithm")):
        return "method"
    if any(token in lower for token in ("experiment", "evaluation", "result", "analysis")):
        return "experiments_results"
    if any(token in lower for token in ("discussion", "limitation", "conclusion")):
        return "discussion_limitations"
    return "other"


def _section_citation_distribution(tex: str) -> dict[str, Any]:
    distribution: dict[str, list[str]] = {
        "abstract": cited_keys(_abstract_body(tex)),
        "introduction": [],
        "related_work": [],
        "method": [],
        "experiments_results": [],
        "discussion_limitations": [],
        "other": [],
    }
    for section in _top_level_sections(tex):
        canonical = _canonical_section(str(section.get("title") or ""))
        for key in cited_keys(str(section.get("body") or "")):
            if key not in distribution[canonical]:
                distribution[canonical].append(key)
    return {
        name: {"unique_count": len(keys), "keys": keys}
        for name, keys in distribution.items()
    }


def _contribution_region(tex: str) -> str:
    intro = ""
    for section in _top_level_sections(tex):
        if _canonical_section(str(section.get("title") or "")) == "introduction":
            intro = str(section.get("body") or "")
            break
    if not intro:
        return ""
    match = re.search(r"(\\paragraph\*?\{Contributions?\}|Contributions?\b)(.*)", intro, flags=re.IGNORECASE | re.DOTALL)
    if not match:
        return ""
    region = match.group(0)
    stop = re.search(r"\\(?:sub)*section\*?\{(?!Contributions?)[^}]+\}", region, flags=re.IGNORECASE)
    return region[: stop.start()] if stop else region


def _norm_title(title: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", (title or "").lower()).strip()


def _entry_issues(key: str, entry: dict[str, Any], *, current_year: int) -> list[dict[str, str]]:
    fields = entry.get("fields") if isinstance(entry.get("fields"), dict) else {}
    title = str(fields.get("title") or "")
    year = str(fields.get("year") or "")
    authors = fields.get("author") or fields.get("editor") or fields.get("organization")
    venue = fields.get("journal") or fields.get("booktitle") or fields.get("venue") or fields.get("publisher")
    note = fields.get("note") or ""
    note_identifier = note if re.search(r"(?:arxiv|doi|https?://)", note, flags=re.IGNORECASE) else ""
    identifier = fields.get("doi") or fields.get("url") or fields.get("eprint") or fields.get("arxiv") or fields.get("archiveprefix") or note_identifier
    issues: list[dict[str, str]] = []
    if not title:
        issues.append(_issue("high", "Reference auditor / metadata", "Bibliography entry is missing a title.", key, "Replace with verified paper metadata."))
    if not authors:
        issues.append(_issue("high", "Reference auditor / metadata", "Bibliography entry is missing authors/editor/organization.", key, "Replace with verified paper metadata."))
    if not year:
        issues.append(_issue("high", "Reference auditor / metadata", "Bibliography entry is missing publication year.", key, "Replace with verified paper metadata."))
    elif year.isdigit() and int(year) > current_year + 1:
        issues.append(_issue("high", "Reference auditor / metadata", "Bibliography entry has an implausible future year.", f"{key}: year={year}", "Verify the source and correct the year."))
    if not venue and not identifier:
        issues.append(_issue("medium", "Reference auditor / metadata", "Bibliography entry lacks venue and DOI/arXiv/URL identifiers.", key, "Add venue plus DOI, arXiv ID, or URL when available."))
    raw = " ".join(str(x) for x in [key, title, authors, venue, identifier]).lower()
    if any(re.search(rf"(?<![a-z0-9]){re.escape(term)}(?![a-z0-9])", raw) for term in PLACEHOLDER_TERMS):
        issues.append(_issue("high", "Reference auditor / authenticity", "Bibliography entry looks like a placeholder or fabricated citation.", key, "Remove placeholder references and replace them with verified literature."))
    return issues


def _issue(severity: str, standard: str, issue: str, evidence: str = "", fix: str = "") -> dict[str, str]:
    out = {"severity": severity, "standard": standard, "issue": issue}
    if evidence:
        out["evidence"] = evidence
    if fix:
        out["fix"] = fix
    return out


def audit_references(
    *,
    main_tex: str,
    bibtex: str,
    min_references: int = 30,
    min_cited_references: int = 30,
    current_year: int = 2026,
) -> dict[str, Any]:
    entries = parse_bib_entries(bibtex)
    cited = cited_keys(main_tex)
    cited_set = set(cited)
    entry_keys = set(entries)
    issues: list[dict[str, str]] = []

    if len(entries) < min_references:
        issues.append(
            _issue(
                "high",
                "Reference auditor / bibliography size",
                "Bibliography is below the required full-paper reference count.",
                f"bibliography_entry_count={len(entries)}; required_min={min_references}",
                "Return to literature discovery and collect at least 30 verified, topic-relevant references before final writing; aim for 50 when available.",
            )
        )
    if len(cited_set) < min_cited_references:
        issues.append(
            _issue(
                "high",
                "Reference auditor / cited-reference size",
                "Main text cites fewer distinct references than the required full-paper minimum.",
                f"unique_cited_count={len(cited_set)}; required_min={min_cited_references}",
                "Distribute at least 30 verified citations across Introduction, Related Work, and Method; aim for 50 when available and do not pad Abstract or contribution bullets.",
            )
        )

    missing = sorted(cited_set - entry_keys)
    if missing:
        issues.append(
            _issue(
                "high",
                "Reference auditor / dangling citations",
                "Manuscript cites keys missing from references.bib.",
                ", ".join(missing[:20]),
                "Add verified BibTeX entries for missing cite keys or remove the unsupported citations.",
            )
        )

    for key, entry in entries.items():
        issues.extend(_entry_issues(key, entry, current_year=current_year))

    by_title: dict[str, list[str]] = {}
    for key, entry in entries.items():
        title = _norm_title((entry.get("fields") or {}).get("title") or "")
        if title:
            by_title.setdefault(title, []).append(key)
    duplicates = [keys for keys in by_title.values() if len(keys) > 1]
    if duplicates:
        issues.append(
            _issue(
                "medium",
                "Reference auditor / duplicate references",
                "Bibliography contains duplicate titles under multiple keys.",
                "; ".join(", ".join(keys) for keys in duplicates[:5]),
                "Deduplicate references and keep the most complete BibTeX entry.",
            )
        )

    distribution = _section_citation_distribution(main_tex)
    if distribution["abstract"]["unique_count"]:
        issues.append(
            _issue(
                "high",
                "Reference auditor / citation placement",
                "Abstract contains citations, which the writing standard forbids.",
                ", ".join(distribution["abstract"]["keys"][:10]),
                "Remove citations from the Abstract and move supporting citations to Introduction or Related Work.",
            )
        )

    contribution_cites = cited_keys(_contribution_region(main_tex))
    if contribution_cites:
        issues.append(
            _issue(
                "high",
                "Reference auditor / contribution placement",
                "Contribution paragraph or bullets contain citations.",
                ", ".join(contribution_cites[:10]),
                "Move these citations into the problem/background paragraphs, Related Work, or Method, and keep contributions as direct claims.",
            )
        )

    citation_floor_scale = max(0.0, min(1.0, float(min_cited_references or 30) / 30.0))
    placement_requirements = {
        "introduction": max(4, round(8 * citation_floor_scale)),
        "related_work": max(8, round(20 * citation_floor_scale)),
        "method": max(3, round(5 * citation_floor_scale)),
    }
    for section, required in placement_requirements.items():
        actual = int(distribution[section]["unique_count"])
        if actual < required:
            issues.append(
                _issue(
                    "high" if section == "related_work" else "medium",
                    "Reference auditor / citation distribution",
                    f"{section.replace('_', ' ').title()} has too few distinct citations for a full paper.",
                    f"{section}_unique_citations={actual}; required_min={required}",
                    "Move verified, relevant citations into Introduction, Related Work, and Method rather than concentrating citations elsewhere.",
                )
            )

    primary_keys: set[str] = set()
    for section in ("introduction", "related_work", "method"):
        primary_keys.update(distribution[section]["keys"])
    if cited_set and len(primary_keys) / max(1, len(cited_set)) < 0.7:
        issues.append(
            _issue(
                "medium",
                "Reference auditor / citation distribution",
                "Less than 70% of citations are placed in Introduction, Related Work, and Method.",
                f"primary_sections={len(primary_keys)}; total_cited={len(cited_set)}",
                "Relocate literature/context citations into the sections that establish problem, prior work, and method lineage.",
            )
        )

    large_clusters = []
    for raw in CITE_RE.findall(main_tex or ""):
        keys = [k.strip() for k in raw.split(",") if k.strip()]
        if len(keys) >= 6:
            large_clusters.append(raw)
    if large_clusters:
        issues.append(
            _issue(
                "medium",
                "Reference auditor / citation style",
                "Manuscript contains large undifferentiated citation clusters.",
                large_clusters[0][:200],
                "Split citation clusters by method category and explain why each group is relevant.",
            )
        )

    decision = "pass"
    if any(issue.get("severity") == "high" for issue in issues):
        decision = "fail"
    elif issues:
        decision = "needs_revision"

    return {
        "schema_version": AUDITOR_VERSION,
        "status": decision,
        "bibliography_entry_count": len(entries),
        "unique_cited_count": len(cited_set),
        "undefined_citations": missing,
        "unused_bibliography_entries": sorted(entry_keys - cited_set),
        "citation_distribution": distribution,
        "metadata_checked_count": len(entries),
        "truth_check_scope": "offline BibTeX metadata integrity, placeholder detection, duplicate detection, and cite-key consistency; online DOI/arXiv resolution can be layered on top when network is available.",
        "issues": issues,
        "next_actions": [issue.get("fix") or issue.get("issue") for issue in issues if issue.get("fix") or issue.get("issue")],
    }
