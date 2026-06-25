"""Venue and journal routing policy for manuscript generation.

The manuscript pipeline must not silently force every conference paper into the
ICLR template.  This module keeps the routing decision explicit and serializable
so prompts, LaTeX assembly, bundle manifests, and watchdog audits use the same
target.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import re
from typing import Any


ICLR2026_TEMPLATE_FILES = [
    "iclr2026_conference.sty",
    "iclr2026_conference.bst",
    "math_commands.tex",
    "natbib.sty",
    "fancyhdr.sty",
]


@dataclass(frozen=True)
class SubmissionTarget:
    key: str
    label: str
    family: str
    template: str
    bibliography_style: str
    page_limit: str
    double_blind: bool
    required_files: tuple[str, ...]
    guideline_files: tuple[str, ...]
    style_marker: str
    route_reason: str
    guidelines: str

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["required_files"] = list(self.required_files)
        payload["guideline_files"] = list(self.guideline_files)
        return payload


def _target(
    *,
    key: str,
    label: str,
    family: str,
    template: str,
    bibliography_style: str,
    page_limit: str,
    double_blind: bool = True,
    required_files: tuple[str, ...] = (),
    guideline_files: tuple[str, ...] = (),
    style_marker: str = "",
    route_reason: str = "",
    guidelines: str = "",
) -> SubmissionTarget:
    return SubmissionTarget(
        key=key,
        label=label,
        family=family,
        template=template,
        bibliography_style=bibliography_style,
        page_limit=page_limit,
        double_blind=double_blind,
        required_files=required_files,
        guideline_files=guideline_files,
        style_marker=style_marker or template,
        route_reason=route_reason,
        guidelines=guidelines,
    )


VENUE_POLICIES: dict[str, SubmissionTarget] = {
    "iclr2026": _target(
        key="iclr2026",
        label="ICLR 2026 main conference",
        family="iclr",
        template="iclr2026",
        bibliography_style="iclr2026_conference",
        page_limit="9 pages for initial submission main text, unlimited references",
        required_files=tuple(ICLR2026_TEMPLATE_FILES),
        style_marker="iclr2026_conference",
        guidelines=(
            "Use the official ICLR 2026 LaTeX files. Main text is limited to "
            "9 pages for initial submission, with unlimited references. The "
            "paper is double blind and must not reveal author or operator identities."
        ),
    ),
    "neurips2026": _target(
        key="neurips2026",
        label="NeurIPS 2026 main conference",
        family="neurips",
        template="neurips2026_generic",
        guideline_files=("Formatting_Instructions_For_NeurIPS_2026 (2).pdf",),
        bibliography_style="plainnat",
        page_limit="9 pages main text for main-track submissions, references excluded/unlimited according to active NeurIPS instructions",
        style_marker="neurips2026",
        guidelines=(
            "Route to NeurIPS 2026 rather than ICLR. Use the active NeurIPS "
            "formatting instructions when an official style package is configured. Main text should be treated as a 9-page target/limit, with references outside the main-page count. "
            "Keep the submission anonymous, use a compact top-conference structure, "
            "and do not reuse ICLR-specific style files for a NeurIPS target."
        ),
    ),
    "neurips2026_position": _target(
        key="neurips2026_position",
        label="NeurIPS 2026 position paper",
        family="neurips",
        template="neurips2026_position_generic",
        guideline_files=("nips26positionpaper (4).pdf", "Formatting_Instructions_For_NeurIPS_2026 (2).pdf"),
        bibliography_style="plainnat",
        page_limit="follow the active NeurIPS 2026 position-paper page limit and formatting instructions",
        style_marker="neurips2026_position",
        guidelines=(
            "Route to the NeurIPS 2026 position-paper track. Emphasize thesis, "
            "evidence scope, implications, and limitations. Do not present the "
            "paper as an ICLR main-conference empirical submission."
        ),
    ),
    "acl_generic": _target(
        key="acl_generic",
        label="ACL/EMNLP-style conference submission",
        family="acl",
        template="acl_generic",
        bibliography_style="plainnat",
        page_limit="ACL-family long paper norm: 8 pages main text, references/limitations outside the main-page budget when allowed by the selected call",
        style_marker="acl_generic",
        guidelines=(
            "Route to an ACL-family NLP submission. Keep terminology, related "
            "work, datasets, and citation style appropriate for NLP readers. Treat long papers as 8-page main-text submissions unless a specific call overrides it."
        ),
    ),
    "cvpr_generic": _target(
        key="cvpr_generic",
        label="CVPR/ICCV/ECCV-style conference submission",
        family="cvpr",
        template="cvpr_generic",
        bibliography_style="plainnat",
        page_limit="CVPR/ICCV/ECCV norm: 8 pages main paper, references excluded",
        style_marker="cvpr_generic",
        guidelines=(
            "Route to a computer-vision conference submission. Emphasize visual "
            "benchmarks, figures, dataset protocol, and comparison to vision baselines. Treat the main paper as an 8-page submission with references excluded."
        ),
    ),
    "icml_generic": _target(
        key="icml_generic",
        label="ICML-style conference submission",
        family="icml",
        template="icml_generic",
        bibliography_style="plainnat",
        page_limit="ICML-style norm: 8 pages main text, references/appendix outside the main-page budget",
        style_marker="icml_generic",
        guidelines=(
            "Route to an ICML-style ML submission. Use ICML-appropriate framing "
            "and avoid ICLR-specific template assumptions unless explicitly configured. Treat the main text as an 8-page submission unless a specific call overrides it."
        ),
    ),
    "journal_generic": _target(
        key="journal_generic",
        label="journal or TMLR-style manuscript",
        family="journal",
        template="journal_generic",
        bibliography_style="plainnat",
        page_limit="journal-style length; target a complete 10-25 page manuscript unless the selected journal specifies otherwise",
        double_blind=False,
        style_marker="journal_generic",
        guidelines=(
            "Route to a journal-style manuscript. Prefer clearer sectioning, "
            "complete reproducibility detail, and a less space-constrained narrative."
        ),
    ),
    "technical_report": _target(
        key="technical_report",
        label="technical report",
        family="technical_report",
        template="technical_report_generic",
        bibliography_style="plainnat",
        page_limit="technical-report length; target a complete evidence-bounded manuscript rather than a top-conference submission",
        double_blind=False,
        style_marker="technical_report",
        guidelines=(
            "Route controlled or materialized-trace evidence here when it is useful "
            "but not strong enough for a main-conference empirical claim. Keep the "
            "paper complete, reproducible, and explicit about evidence scope, "
            "missing baselines, and limitations."
        ),
    ),
}


def target_from_key(key: str | None) -> SubmissionTarget | None:
    if not key:
        return None
    normalized = str(key).strip().lower().replace("-", "_").replace(" ", "_")
    return VENUE_POLICIES.get(normalized)


def _flatten_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, dict):
        return " ".join(_flatten_text(v) for v in value.values())
    if isinstance(value, (list, tuple, set)):
        return " ".join(_flatten_text(v) for v in value)
    return str(value)


def _candidate_target_text(state: dict[str, Any] | None, configured_template: str | None) -> str:
    state = state or {}
    paper_intent = state.get("paper_intent") if isinstance(state.get("paper_intent"), dict) else {}
    publication_contract = (
        state.get("publication_evidence_contract")
        if isinstance(state.get("publication_evidence_contract"), dict)
        else {}
    )
    pieces = [
        configured_template or "",
        state.get("submission_target"),
        state.get("target_venue"),
        state.get("venue"),
        paper_intent.get("submission_target"),
        paper_intent.get("target_venue"),
        paper_intent.get("venue"),
        publication_contract.get("submission_target"),
        publication_contract.get("target_venue"),
        publication_contract.get("venue"),
        state.get("title"),
        state.get("paper_type"),
    ]
    return " ".join(_flatten_text(piece) for piece in pieces if piece).lower()


def infer_submission_target(
    state: dict[str, Any] | None = None,
    *,
    bundle_format: str = "conference",
    configured_template: str | None = "auto",
) -> SubmissionTarget:
    """Infer a target venue/journal from paper intent, contract, and config."""
    state = state or {}
    configured = (configured_template or "auto").strip().lower()
    explicit = target_from_key(configured)
    if explicit and configured not in {"auto", "default"}:
        return _target_with_reason(explicit, f"configured template={configured_template}")

    text = _candidate_target_text(state, "" if configured in {"auto", "default"} else configured)
    publication_contract = (
        state.get("publication_evidence_contract")
        if isinstance((state or {}).get("publication_evidence_contract"), dict)
        else {}
    )
    result_packet = (
        state.get("result_packet")
        if isinstance((state or {}).get("result_packet"), dict)
        else {}
    )
    claim_route = (
        result_packet.get("claim_route")
        if isinstance(result_packet.get("claim_route"), dict)
        else publication_contract.get("claim_route")
        if isinstance(publication_contract.get("claim_route"), dict)
        else {}
    )
    if str(bundle_format or "").lower() == "journal":
        return _target_with_reason(VENUE_POLICIES["journal_generic"], "bundle_format=journal")
    if str(bundle_format or "").lower() in {"technical_report", "report", "preprint"}:
        return _target_with_reason(VENUE_POLICIES["technical_report"], f"bundle_format={bundle_format}")
    if re.search(r"\b(tmlr|journal|transactions|journal article)\b", text):
        return _target_with_reason(VENUE_POLICIES["journal_generic"], "paper intent names a journal target")
    if re.search(r"\b(technical report|technical_report|report|preprint)\b", text):
        return _target_with_reason(VENUE_POLICIES["technical_report"], "paper intent names a technical report")
    evidence_tier = " ".join(
        str(x or "")
        for x in (
            result_packet.get("evidence_tier"),
            publication_contract.get("evidence_tier"),
            claim_route.get("route"),
        )
    ).lower()
    if configured in {"auto", "default"} and any(token in evidence_tier for token in ("controlled_materialized", "materialized", "technical_report")):
        return _target_with_reason(VENUE_POLICIES["technical_report"], "controlled/materialized evidence routes to technical report")
    if re.search(r"\b(neurips|nips)\b", text) and re.search(r"\b(position|position-paper|position paper)\b", text):
        return _target_with_reason(VENUE_POLICIES["neurips2026_position"], "paper intent names NeurIPS position paper")
    if re.search(r"\b(neurips|nips)\b", text):
        return _target_with_reason(VENUE_POLICIES["neurips2026"], "paper intent names NeurIPS")
    if re.search(r"\b(acl|emnlp|naacl|coling)\b", text):
        return _target_with_reason(VENUE_POLICIES["acl_generic"], "paper intent names ACL-family venue")
    if re.search(r"\b(cvpr|iccv|eccv)\b", text):
        return _target_with_reason(VENUE_POLICIES["cvpr_generic"], "paper intent names vision venue")
    if re.search(r"\bicml\b", text):
        return _target_with_reason(VENUE_POLICIES["icml_generic"], "paper intent names ICML")
    if re.search(r"\biclr\b", text):
        return _target_with_reason(VENUE_POLICIES["iclr2026"], "paper intent names ICLR")
    return _target_with_reason(VENUE_POLICIES["iclr2026"], "auto fallback: top-tier ML conference with available official template")


def _target_with_reason(target: SubmissionTarget, reason: str) -> SubmissionTarget:
    return _target(
        key=target.key,
        label=target.label,
        family=target.family,
        template=target.template,
        bibliography_style=target.bibliography_style,
        page_limit=target.page_limit,
        double_blind=target.double_blind,
        required_files=target.required_files,
        guideline_files=target.guideline_files,
        style_marker=target.style_marker,
        route_reason=reason,
        guidelines=target.guidelines,
    )


def generic_template_tex(state: dict[str, Any], target: SubmissionTarget) -> str:
    """Return a generic anonymous LaTeX shell for non-ICLR targets."""
    title = (state.get("title") or "Title").replace("&", r"\&")
    author = "Anonymous authors\\\\Paper under double-blind review" if target.double_blind else "Authors"
    return rf"""\documentclass[10pt]{{article}}
\usepackage[margin=1in]{{geometry}}
\usepackage{{microtype}}
\usepackage{{graphicx}}
\usepackage{{booktabs}}
\usepackage{{array}}
\usepackage{{tabularx}}
\usepackage{{amsmath,amssymb}}
\usepackage{{natbib}}
\usepackage{{hyperref}}
\usepackage{{url}}
\title{{{title}}}
\author{{{author}}}
\date{{{target.label}}}
\begin{{document}}
\maketitle
\begin{{abstract}}
\end{{abstract}}
\section{{Introduction}}
\section{{Related Work}}
\section{{Method}}
\section{{Experiments}}
\section{{Discussion}}
\section{{Conclusion}}
\bibliographystyle{{{target.bibliography_style}}}
\bibliography{{references}}
\end{{document}}
"""
