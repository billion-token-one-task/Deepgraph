"""Venue-aware manuscript length policy.

The values here are intentionally explicit rather than inferred from one global
"8 pages / 4500 words" heuristic.  A generated paper is judged against the
selected venue family, the official page budget, and a best-paper-calibrated
section profile.  The policy is conservative: it defines a complete-paper
target range, not a guarantee that a paper is competitive.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


POLICY_VERSION = "deepgraph_manuscript_length_policy_v1_2026_06_10"


@dataclass(frozen=True)
class SectionBudget:
    min_words: int
    target_min_words: int
    target_max_words: int
    max_words: int
    required: bool = True

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class VenueLengthPolicy:
    key: str
    family: str
    label: str
    official_main_page_limit: int | None
    complete_main_page_range: tuple[int, int]
    main_word_range: tuple[int, int]
    min_reference_count: int
    min_cited_reference_count: int
    section_budgets: dict[str, SectionBudget]
    official_sources: tuple[str, ...]
    calibration_sources: tuple[str, ...]
    notes: str

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["section_budgets"] = {k: v.to_dict() for k, v in self.section_budgets.items()}
        return payload


COMMON_OFFICIAL_SOURCES = (
    "https://iclr.cc/Conferences/2026/AuthorGuide",
    "https://neurips.cc/Conferences/2026/CallForPapers",
    "https://cvpr.thecvf.com/Conferences/2026/AuthorGuidelines",
    "https://2024.aclweb.org/calls/main_conference_papers/",
)

BEST_PAPER_CALIBRATION_SOURCES = (
    "ICLR, NeurIPS, ICML, ACL, and CVPR recent award/best-paper program pages and PDFs sampled from 2022-2025.",
    "Local reference PDF corpus when available under config.REFERENCE_PDF_CORPUS_DIR.",
)


def _budgets(
    *,
    abstract: tuple[int, int, int, int] = (140, 160, 240, 300),
    introduction: tuple[int, int, int, int] = (650, 800, 1250, 1500),
    related_work: tuple[int, int, int, int] = (550, 700, 1150, 1450),
    method: tuple[int, int, int, int] = (950, 1200, 1900, 2300),
    experiments_results: tuple[int, int, int, int] = (1250, 1550, 2500, 3100),
    discussion_limitations: tuple[int, int, int, int] = (260, 360, 850, 1100),
) -> dict[str, SectionBudget]:
    return {
        "abstract": SectionBudget(*abstract, required=True),
        "introduction": SectionBudget(*introduction, required=True),
        "related_work": SectionBudget(*related_work, required=True),
        "method": SectionBudget(*method, required=True),
        "experiments_results": SectionBudget(*experiments_results, required=True),
        "discussion_limitations": SectionBudget(*discussion_limitations, required=True),
    }


VENUE_LENGTH_POLICIES: dict[str, VenueLengthPolicy] = {
    "iclr": VenueLengthPolicy(
        key="iclr",
        family="iclr",
        label="ICLR-style main conference paper",
        official_main_page_limit=9,
        complete_main_page_range=(8, 9),
        main_word_range=(5200, 7800),
        min_reference_count=30,
        min_cited_reference_count=30,
        section_budgets=_budgets(),
        official_sources=COMMON_OFFICIAL_SOURCES[:1],
        calibration_sources=BEST_PAPER_CALIBRATION_SOURCES,
        notes="ICLR main submissions have a 9-page main-text budget; complete papers should occupy roughly 8-9 main pages before references.",
    ),
    "neurips": VenueLengthPolicy(
        key="neurips",
        family="neurips",
        label="NeurIPS-style main conference paper",
        official_main_page_limit=9,
        complete_main_page_range=(8, 9),
        main_word_range=(5200, 7800),
        min_reference_count=30,
        min_cited_reference_count=30,
        section_budgets=_budgets(),
        official_sources=COMMON_OFFICIAL_SOURCES[1:2],
        calibration_sources=BEST_PAPER_CALIBRATION_SOURCES,
        notes="NeurIPS main papers are calibrated to a 9-page main text and dense empirical sections.",
    ),
    "icml": VenueLengthPolicy(
        key="icml",
        family="icml",
        label="ICML-style main conference paper",
        official_main_page_limit=8,
        complete_main_page_range=(7, 8),
        main_word_range=(4700, 7000),
        min_reference_count=30,
        min_cited_reference_count=30,
        section_budgets=_budgets(
            introduction=(600, 760, 1150, 1450),
            related_work=(500, 620, 1000, 1300),
            method=(900, 1100, 1750, 2150),
            experiments_results=(1200, 1500, 2300, 2900),
            discussion_limitations=(220, 300, 760, 980),
        ),
        official_sources=COMMON_OFFICIAL_SOURCES,
        calibration_sources=BEST_PAPER_CALIBRATION_SOURCES,
        notes="ICML-style papers normally use a tighter 8-page main-text budget, so section ranges are slightly smaller than ICLR/NeurIPS.",
    ),
    "acl": VenueLengthPolicy(
        key="acl",
        family="acl",
        label="ACL-family long paper",
        official_main_page_limit=8,
        complete_main_page_range=(7, 8),
        main_word_range=(4700, 6900),
        min_reference_count=30,
        min_cited_reference_count=30,
        section_budgets=_budgets(
            introduction=(650, 800, 1250, 1550),
            related_work=(700, 900, 1500, 1900),
            method=(800, 1000, 1650, 2050),
            experiments_results=(1050, 1350, 2200, 2800),
            discussion_limitations=(220, 320, 800, 1050),
        ),
        official_sources=COMMON_OFFICIAL_SOURCES[3:4],
        calibration_sources=BEST_PAPER_CALIBRATION_SOURCES,
        notes="ACL-family papers often spend more space on related work and dataset/task positioning.",
    ),
    "cvpr": VenueLengthPolicy(
        key="cvpr",
        family="cvpr",
        label="CVPR-family main conference paper",
        official_main_page_limit=8,
        complete_main_page_range=(7, 8),
        main_word_range=(4300, 6600),
        min_reference_count=30,
        min_cited_reference_count=30,
        section_budgets=_budgets(
            introduction=(520, 680, 1050, 1350),
            related_work=(500, 650, 1100, 1450),
            method=(900, 1150, 1850, 2300),
            experiments_results=(1200, 1500, 2400, 3000),
            discussion_limitations=(180, 260, 700, 920),
        ),
        official_sources=COMMON_OFFICIAL_SOURCES[2:3],
        calibration_sources=BEST_PAPER_CALIBRATION_SOURCES,
        notes="CVPR-family papers have an 8-page main-paper norm and usually allocate more page area to figures.",
    ),
    "journal": VenueLengthPolicy(
        key="journal",
        family="journal",
        label="Journal/TMLR-style manuscript",
        official_main_page_limit=None,
        complete_main_page_range=(10, 25),
        main_word_range=(6500, 14000),
        min_reference_count=30,
        min_cited_reference_count=30,
        section_budgets=_budgets(
            introduction=(800, 1000, 1700, 2200),
            related_work=(900, 1100, 2200, 2800),
            method=(1200, 1600, 3200, 4200),
            experiments_results=(1600, 2200, 4300, 5600),
            discussion_limitations=(450, 650, 1600, 2200),
        ),
        official_sources=COMMON_OFFICIAL_SOURCES,
        calibration_sources=BEST_PAPER_CALIBRATION_SOURCES,
        notes="Journal-style targets are not capped by conference page budgets, but still require complete section coverage.",
    ),
    "technical_report": VenueLengthPolicy(
        key="technical_report",
        family="technical_report",
        label="Evidence-bounded technical report",
        official_main_page_limit=None,
        complete_main_page_range=(5, 14),
        main_word_range=(3000, 9000),
        min_reference_count=20,
        min_cited_reference_count=20,
        section_budgets=_budgets(
            abstract=(100, 120, 260, 320),
            introduction=(450, 600, 1300, 1700),
            related_work=(350, 450, 1200, 1700),
            method=(650, 800, 1900, 2600),
            experiments_results=(700, 900, 2400, 3300),
            discussion_limitations=(250, 320, 1200, 1700),
        ),
        official_sources=COMMON_OFFICIAL_SOURCES,
        calibration_sources=BEST_PAPER_CALIBRATION_SOURCES,
        notes="Technical reports are allowed for controlled/materialized evidence when claims are scoped honestly; they still require a complete manuscript and traceable references.",
    ),
}


def policy_for_family(family: str | None) -> VenueLengthPolicy:
    key = (family or "iclr").strip().lower()
    if key in VENUE_LENGTH_POLICIES:
        return VENUE_LENGTH_POLICIES[key]
    if key in {"emnlp", "naacl", "coling"}:
        return VENUE_LENGTH_POLICIES["acl"]
    if key in {"iccv", "eccv"}:
        return VENUE_LENGTH_POLICIES["cvpr"]
    return VENUE_LENGTH_POLICIES["iclr"]


def policy_for_submission_target(target: Any | None) -> VenueLengthPolicy:
    if target is None:
        return policy_for_family("iclr")
    if isinstance(target, dict):
        return policy_for_family(target.get("family") or target.get("key"))
    return policy_for_family(getattr(target, "family", None) or getattr(target, "key", None))


def build_length_standard_text(target: Any | None = None) -> str:
    policy = policy_for_submission_target(target)
    ranges = policy.section_budgets
    parts = [
        f"Length standard: {POLICY_VERSION}.",
        f"Target family: {policy.label}. Complete main text should be {policy.complete_main_page_range[0]}-{policy.complete_main_page_range[1]} pages"
        + (f" under the official {policy.official_main_page_limit}-page main-text limit." if policy.official_main_page_limit else "."),
        f"Main body target: {policy.main_word_range[0]}-{policy.main_word_range[1]} words before references.",
        "Section target ranges are best-paper-calibrated priors, not decorative suggestions:",
    ]
    for name, budget in ranges.items():
        parts.append(
            f"- {name}: target {budget.target_min_words}-{budget.target_max_words} words; hard floor {budget.min_words}; hard ceiling {budget.max_words}."
        )
    parts.append(
        f"References: at least {policy.min_reference_count} bibliography entries and {policy.min_cited_reference_count} distinct cited entries in the main text; aim for roughly 50 when enough topic-relevant literature is available."
    )
    return "\n".join(parts)
