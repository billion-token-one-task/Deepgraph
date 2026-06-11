"""Dedicated length auditor for generated manuscripts."""

from __future__ import annotations

import re
from typing import Any

from agents.manuscript_length_policy import (
    POLICY_VERSION,
    SectionBudget,
    policy_for_submission_target,
)


AUDITOR_VERSION = "deepgraph_manuscript_length_auditor_v1_2026_06_10"

WORD_RE = re.compile(r"[A-Za-z][A-Za-z0-9\-']+")
COMMAND_RE = re.compile(r"\\[a-zA-Z]+\*?(?:\[[^\]]*\])?(?:\{[^{}]*\})?")
ABSTRACT_RE = re.compile(r"\\begin\{abstract\}(.*?)\\end\{abstract\}", re.IGNORECASE | re.DOTALL)
SECTION_RE = re.compile(r"\\section\*?\{([^}]+)\}", re.IGNORECASE)


SECTION_ALIASES: dict[str, tuple[str, ...]] = {
    "introduction": ("introduction",),
    "related_work": ("related work", "prior work", "background"),
    "method": ("method", "approach", "framework", "model", "algorithm"),
    "experiments_results": (
        "experiments",
        "experiment",
        "evaluation",
        "experimental setup",
        "results",
        "main results",
        "analysis",
    ),
    "discussion_limitations": (
        "discussion",
        "limitations",
        "limitation",
        "conclusion",
        "threats to validity",
    ),
}


def _strip_comments(tex: str) -> str:
    return re.sub(r"(?<!\\)%.*", " ", tex or "")


def _plain_text(tex: str) -> str:
    text = _strip_comments(tex or "")
    text = re.sub(r"\\bibliographystyle\{[^}]*\}.*", " ", text, flags=re.DOTALL)
    text = re.sub(r"\\bibliography\{[^}]*\}.*", " ", text, flags=re.DOTALL)
    text = re.sub(r"\\begin\{(table|figure|algorithm|equation|align|gather)\*?\}.*?\\end\{\1\*?\}", " ", text, flags=re.IGNORECASE | re.DOTALL)
    text = COMMAND_RE.sub(" ", text)
    text = re.sub(r"[{}$^_\\]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def word_count(tex: str) -> int:
    return len(WORD_RE.findall(_plain_text(tex or "")))


def _abstract_body(tex: str) -> str:
    match = ABSTRACT_RE.search(tex or "")
    return match.group(1) if match else ""


def _top_level_sections(tex: str) -> list[dict[str, Any]]:
    matches = list(SECTION_RE.finditer(tex or ""))
    sections: list[dict[str, Any]] = []
    for idx, match in enumerate(matches):
        start = match.end()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(tex or "")
        title = (match.group(1) or "").strip()
        sections.append({"title": title, "body": (tex or "")[start:end]})
    return sections


def _canonical_section_words(tex: str) -> dict[str, int]:
    words = {"abstract": word_count(_abstract_body(tex))}
    for canonical in SECTION_ALIASES:
        words[canonical] = 0
    for section in _top_level_sections(tex):
        title = str(section.get("title") or "").strip().lower()
        body = str(section.get("body") or "")
        for canonical, aliases in SECTION_ALIASES.items():
            if any(alias == title or alias in title for alias in aliases):
                words[canonical] += word_count(body)
                break
    return words


def _issue(severity: str, standard: str, issue: str, evidence: str = "", fix: str = "") -> dict[str, str]:
    out = {"severity": severity, "standard": standard, "issue": issue}
    if evidence:
        out["evidence"] = evidence
    if fix:
        out["fix"] = fix
    return out


def _budget_issue(
    *,
    name: str,
    count: int,
    budget: SectionBudget,
) -> dict[str, str] | None:
    label = name.replace("_", " ")
    if count <= 0 and budget.required:
        return _issue(
            "high",
            "Length auditor / section coverage",
            f"Required section group is missing or empty: {label}.",
            f"{name}_words={count}",
            f"Add a substantive {label} section within the target range {budget.target_min_words}-{budget.target_max_words} words.",
        )
    if count < budget.min_words:
        return _issue(
            "high",
            "Length auditor / section floor",
            f"{label} is below the hard floor for a complete paper.",
            f"{name}_words={count}; hard_floor={budget.min_words}; target={budget.target_min_words}-{budget.target_max_words}",
            f"Expand {label} with concrete problem framing, method detail, experiment protocol, analysis, limitations, or literature positioning as appropriate.",
        )
    if count < budget.target_min_words:
        return _issue(
            "medium",
            "Length auditor / section target",
            f"{label} is below the best-paper-calibrated target range.",
            f"{name}_words={count}; target={budget.target_min_words}-{budget.target_max_words}",
            f"Add depth to {label} until it reaches the target range without padding.",
        )
    if count > budget.max_words:
        return _issue(
            "medium",
            "Length auditor / section ceiling",
            f"{label} exceeds the hard ceiling for the venue-calibrated main text.",
            f"{name}_words={count}; hard_ceiling={budget.max_words}",
            f"Compress {label} and move overflow to appendix or remove repetition.",
        )
    return None


def audit_manuscript_length(
    *,
    main_tex: str,
    page_count: int | None,
    venue_target: Any | None = None,
    bibliography_entry_count: int = 0,
) -> dict[str, Any]:
    policy = policy_for_submission_target(venue_target)
    total_words = word_count(main_tex)
    section_words = _canonical_section_words(main_tex)
    issues: list[dict[str, str]] = []

    reference_page_allowance = max(0, (int(bibliography_entry_count or 0) + 29) // 30)
    effective_total_page_limit = (policy.official_main_page_limit + reference_page_allowance) if policy.official_main_page_limit else None
    if policy.official_main_page_limit and page_count is not None and effective_total_page_limit is not None and page_count > effective_total_page_limit:
        issues.append(
            _issue(
                "high",
                "Length auditor / official page limit",
                "Compiled PDF appears to exceed the selected venue page budget even after excluding an estimated bibliography allowance.",
                f"page_count={page_count}; official_main_page_limit={policy.official_main_page_limit}; bibliography_entry_count={bibliography_entry_count}; reference_page_allowance={reference_page_allowance}; effective_total_page_limit={effective_total_page_limit}; venue={policy.label}",
                "Shorten main text or move nonessential material to appendix before marking bundle_ready.",
            )
        )
    if page_count is not None and page_count < policy.complete_main_page_range[0]:
        issues.append(
            _issue(
                "high",
                "Length auditor / complete-paper page floor",
                "Compiled paper is shorter than the complete-paper range for the selected venue family.",
                f"page_count={page_count}; complete_range={policy.complete_main_page_range[0]}-{policy.complete_main_page_range[1]}; venue={policy.label}",
                "Return to manuscript drafting and expand the underfilled sections listed by the length auditor.",
            )
        )
    if total_words < policy.main_word_range[0]:
        issues.append(
            _issue(
                "high",
                "Length auditor / main-body word floor",
                "Main manuscript body is too short for the venue-calibrated complete-paper target.",
                f"word_count={total_words}; target={policy.main_word_range[0]}-{policy.main_word_range[1]}; venue={policy.label}",
                "Expand the manuscript with method mechanics, protocol detail, full related work positioning, result interpretation, and limitations.",
            )
        )
    if total_words > policy.main_word_range[1] and not (policy.official_main_page_limit and page_count and page_count <= policy.official_main_page_limit):
        issues.append(
            _issue(
                "medium",
                "Length auditor / main-body word ceiling",
                "Main manuscript body is longer than the venue-calibrated target range.",
                f"word_count={total_words}; target={policy.main_word_range[0]}-{policy.main_word_range[1]}; venue={policy.label}",
                "Compress repetition and move secondary analysis to appendix while preserving required evidence.",
            )
        )

    for name, budget in policy.section_budgets.items():
        issue = _budget_issue(name=name, count=int(section_words.get(name) or 0), budget=budget)
        if issue:
            issues.append(issue)

    decision = "pass"
    if any(issue.get("severity") == "high" for issue in issues):
        decision = "fail"
    elif issues:
        decision = "needs_revision"

    return {
        "schema_version": AUDITOR_VERSION,
        "policy_version": POLICY_VERSION,
        "status": decision,
        "venue_policy": policy.to_dict(),
        "word_count": total_words,
        "page_count": page_count,
        "bibliography_entry_count": int(bibliography_entry_count or 0),
        "reference_page_allowance": reference_page_allowance,
        "effective_total_page_limit": effective_total_page_limit,
        "section_words": section_words,
        "issues": issues,
        "next_actions": [issue.get("fix") or issue.get("issue") for issue in issues if issue.get("fix") or issue.get("issue")],
    }
