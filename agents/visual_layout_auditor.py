"""Visual layout and figure-policy auditor for manuscript bundles."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any


AUDITOR_VERSION = "deepgraph_visual_layout_auditor_v3_2026_06_11"


VISUAL_LAYOUT_STANDARD_TEXT = """Visual layout standard:
- Figures must never appear before maketitle, title, authors, abstract, or the first substantive paper text.
- Motivation and overview figures are mandatory PaperBanana/Gemini post-writing figures, but they must not be forced into the first viewport and must not be the first object in the paper.
- LaTeX captions are the only figure captions. Do not add standalone "Figure 1:" paragraphs after a figure.
- Generated images must not contain internal captions, "Figure X" labels, panel numbering such as "1./2./3.", or long explanatory paragraphs.
- Motivation figures must not use a rigid three-column comparison layout. Prefer a compact tension map, central mechanism schematic, or one worked-example diagram with at most two small callouts.
- Concept figures may use short local labels, score tags, or symbols, but the caption and surrounding prose must carry the explanation.
- Algorithm and pseudocode blocks must use real LaTeX algorithm structure (algorithm+algpseudocode/algorithmic or algorithm2e) with a caption and line structure; do not fake algorithms with center/minipage/enumerate/textbf blocks.
- Experiment figures are separate from motivation/overview diagrams: require at least three artifact-backed experiment plots, at least three distinct chart families, and searched style-reference metadata from related papers."""


FIGURE_ENV_RE = re.compile(r"\\begin\{figure\*?\}.*?\\end\{figure\*?\}", re.DOTALL | re.IGNORECASE)
INCLUDE_RE = re.compile(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", re.IGNORECASE)
CAPTION_RE = re.compile(r"\\caption\{([\s\S]*?)\}", re.IGNORECASE)
ALGORITHM_ENV_RE = re.compile(r"\\begin\{algorithm\*?\}.*?\\end\{algorithm\*?\}", re.DOTALL | re.IGNORECASE)
ALGORITHMIC_ENV_RE = re.compile(r"\\begin\{(?:algorithmic|algorithmicx)\}", re.IGNORECASE)
PSEUDOCODE_COMMAND_RE = re.compile(
    r"\\(?:State|Require|Ensure|Return|For|EndFor|If|Else|EndIf|While|EndWhile|Repeat|Until|KwData|KwResult|KwIn|KwOut|KwRet|SetKwInput|SetKwFunction|SetAlgoLined|DontPrintSemicolon|tcp)\b",
    re.IGNORECASE,
)
MIN_EXPERIMENT_FIGURES = 3
MIN_EXPERIMENT_CHART_FAMILIES = 3


MANUAL_ALGORITHM_PATTERNS = (
    re.compile(
        r"\\begin\{(?:center|quote|minipage)\}[\s\S]{0,1800}?\\(?:textbf|paragraph)\{Algorithm\s*\d*[^}]*\}[\s\S]{0,1800}?\\begin\{(?:enumerate|itemize|list)\}",
        re.IGNORECASE,
    ),
    re.compile(
        r"\\(?:textbf|paragraph)\{Algorithm\s*\d*[^}]*\}[\s\S]{0,1400}?\\begin\{(?:enumerate|itemize|list)\}",
        re.IGNORECASE,
    ),
    re.compile(
        r"\\textbf\{(?:Input|Data)\s*:?\}[\s\S]{0,700}?\\textbf\{(?:Output|Result)\s*:?\}[\s\S]{0,900}?\\begin\{(?:enumerate|itemize|list)\}",
        re.IGNORECASE,
    ),
)


def _issue(severity: str, standard: str, issue: str, evidence: str = "", fix: str = "") -> dict[str, str]:
    out = {"severity": severity, "standard": standard, "issue": issue}
    if evidence:
        out["evidence"] = evidence
    if fix:
        out["fix"] = fix
    return out


def _index(tex: str, needle: str) -> int:
    idx = (tex or "").find(needle)
    return idx if idx >= 0 else -1


def _figure_blocks(tex: str) -> list[dict[str, Any]]:
    blocks: list[dict[str, Any]] = []
    for match in FIGURE_ENV_RE.finditer(tex or ""):
        body = match.group(0)
        includes = [raw.strip() for raw in INCLUDE_RE.findall(body)]
        captions = [re.sub(r"\s+", " ", raw).strip() for raw in CAPTION_RE.findall(body)]
        blocks.append(
            {
                "start": match.start(),
                "end": match.end(),
                "body": body,
                "includes": includes,
                "captions": captions,
                "stems": [Path(raw).stem for raw in includes],
            }
        )
    return blocks


def _algorithm_blocks(tex: str) -> list[dict[str, Any]]:
    blocks: list[dict[str, Any]] = []
    for match in ALGORITHM_ENV_RE.finditer(tex or ""):
        body = match.group(0)
        captions = [re.sub(r"\s+", " ", raw).strip() for raw in CAPTION_RE.findall(body)]
        blocks.append({"start": match.start(), "end": match.end(), "body": body, "captions": captions})
    return blocks


def _inside_intervals(position: int, intervals: list[tuple[int, int]]) -> bool:
    return any(start <= position <= end for start, end in intervals)


def _manual_algorithm_snippets(tex: str, algorithm_intervals: list[tuple[int, int]]) -> list[str]:
    matches: list[tuple[int, int, str]] = []
    for pattern in MANUAL_ALGORITHM_PATTERNS:
        for match in pattern.finditer(tex or ""):
            if _inside_intervals(match.start(), algorithm_intervals):
                continue
            snippet = re.sub(r"\s+", " ", match.group(0)).strip()
            matches.append((match.start(), match.end(), snippet[:260]))
    matches.sort(key=lambda item: (item[0], -(item[1] - item[0])))

    kept: list[tuple[int, int, str]] = []
    for start, end, snippet in matches:
        overlaps_existing = any(start < kept_end and end > kept_start for kept_start, kept_end, _ in kept)
        if overlaps_existing:
            continue
        kept.append((start, end, snippet))
    return [snippet for _, _, snippet in kept]


def _has_structured_algorithm_body(block_body: str, tex: str) -> bool:
    if ALGORITHMIC_ENV_RE.search(block_body):
        return True
    if PSEUDOCODE_COMMAND_RE.search(block_body):
        return True
    if "algorithm2e" in (tex or "") and re.search(r"(?:;\s*(?:%[^\n]*)?\n|\\BlankLine\b)", block_body):
        return True
    return False


def _standalone_caption_after_figure(tex: str) -> str | None:
    pattern = re.compile(
        r"\\end\{figure\*?\}\s*(?:\\noindent\s*)?(?:\\textbf\{)?Figure\s*\d+\}?[:.][^\n]*(?:\n(?!\s*\\(?:section|subsection|begin\{figure)).*){0,3}",
        re.IGNORECASE,
    )
    match = pattern.search(tex or "")
    if not match:
        return None
    snippet = re.sub(r"\s+", " ", match.group(0)).strip()
    return snippet[:240]


def _is_concept_figure_asset(asset: dict[str, Any]) -> bool:
    text = " ".join(
        str(asset.get(key) or "")
        for key in ("figure_id", "title", "objective", "caption", "kind", "stage")
    ).lower()
    return any(token in text for token in ("fig_motivation_symbolic", "fig_overview_symbolic", "motivation", "overview"))


def _is_valid_experiment_plot_asset(asset: dict[str, Any]) -> bool:
    if _is_concept_figure_asset(asset):
        return False
    if asset.get("kind") != "plot":
        return False
    if str(asset.get("notes") or "").startswith("blocked") or asset.get("kind") in {"blocked", "fallback"}:
        return False
    return bool(asset.get("path") or asset.get("svg_path") or asset.get("pdf_path"))


def _infer_experiment_chart_family(asset: dict[str, Any]) -> str:
    explicit = str(asset.get("chart_family") or "").strip()
    if explicit:
        return explicit
    text = " ".join(
        str(asset.get(key) or "")
        for key in ("figure_id", "title", "objective", "chart_type", "renderer", "notes", "path")
    ).lower()
    if any(token in text for token in ("heatmap", "matrix", "confusion")):
        return "matrix_family"
    if any(token in text for token in ("scatter", "tradeoff", "trade-off", "frontier", "tsne", "t-sne", "umap", "embedding")):
        return "distribution_family"
    if any(token in text for token in ("line", "curve", "rank", "cmc", "trend")):
        return "line_family"
    if any(token in text for token in ("bar", "grouped")):
        return "bar_family"
    if any(token in text for token in ("panel", "diagnostic")):
        return "multipanel_family"
    return "unknown_family"


def _has_style_reference_metadata(asset: dict[str, Any]) -> bool:
    return bool(asset.get("style_reference_keys") or asset.get("style_reference_titles"))


def _concept_asset_text(asset: dict[str, Any]) -> str:
    return " ".join(
        str(asset.get(key) or "")
        for key in ("figure_id", "title", "objective", "caption", "notes", "renderer", "layout", "aspect_ratio")
    )


def audit_visual_layout(
    *,
    main_tex: str,
    figure_assets: list[dict[str, Any]] | None = None,
    page_count: int | None = None,
) -> dict[str, Any]:
    tex = main_tex or ""
    issues: list[dict[str, str]] = []
    blocks = _figure_blocks(tex)
    algorithm_blocks = _algorithm_blocks(tex)
    maketitle_idx = _index(tex, r"\maketitle")
    abstract_begin_idx = _index(tex, r"\begin{abstract}")
    abstract_end_idx = _index(tex, r"\end{abstract}")
    intro_idx = tex.lower().find(r"\section{introduction}")

    for block in blocks:
        includes = ", ".join(block["includes"][:4])
        if maketitle_idx >= 0 and block["start"] < maketitle_idx:
            issues.append(
                _issue(
                    "high",
                    "Visual layout auditor / top matter",
                    "A figure appears before maketitle/title/authors.",
                    includes,
                    "Move the figure after the abstract and at least one substantive Introduction paragraph, or remove it.",
                )
            )
        elif abstract_begin_idx >= 0 and block["start"] < abstract_begin_idx:
            issues.append(
                _issue(
                    "high",
                    "Visual layout auditor / top matter",
                    "A figure appears before the abstract.",
                    includes,
                    "Figures must not be placed in the title/author/abstract area.",
                )
            )
        elif abstract_end_idx >= 0 and block["start"] < abstract_end_idx:
            issues.append(
                _issue(
                    "high",
                    "Visual layout auditor / abstract layout",
                    "A figure is placed inside or before the abstract is complete.",
                    includes,
                    "Move all figures out of the abstract/top matter.",
                )
            )
        if any("motivation" in stem.lower() or "overview" in stem.lower() for stem in block["stems"]):
            if intro_idx < 0 or block["start"] < intro_idx:
                issues.append(
                    _issue(
                        "high",
                        "Visual layout auditor / concept figure placement",
                        "A motivation/overview figure appears before the Introduction.",
                        includes,
                        "Concept figures are mandatory, but they must not occupy the first viewport before the paper title or Introduction text.",
                    )
                )
            intro_text_before = tex[intro_idx:block["start"]] if intro_idx >= 0 and block["start"] > intro_idx else ""
            intro_words = len(re.findall(r"[A-Za-z][A-Za-z0-9\-']+", re.sub(r"\\[a-zA-Z]+\*?(?:\{[^}]*\})?", " ", intro_text_before)))
            if 0 <= intro_idx < block["start"] and intro_words < 120:
                issues.append(
                    _issue(
                        "medium",
                        "Visual layout auditor / concept figure placement",
                        "A motivation/overview figure appears before enough Introduction prose.",
                        f"intro_words_before_figure={intro_words}; figure={includes}",
                        "Place mandatory concept figures after at least one substantive Introduction paragraph; do not use them as title/top-matter content.",
                    )
                )

    duplicate_caption = _standalone_caption_after_figure(tex)
    if duplicate_caption:
        issues.append(
            _issue(
                "high",
                "Visual layout auditor / duplicate captions",
                "Standalone Figure caption text appears outside the figure environment.",
                duplicate_caption,
                "Keep only the LaTeX \\caption{...}; remove extra 'Figure X:' prose blocks.",
            )
        )


    algorithm_intervals = [(block["start"], block["end"]) for block in algorithm_blocks]
    for snippet in _manual_algorithm_snippets(tex, algorithm_intervals):
        issues.append(
            _issue(
                "high",
                "Visual layout auditor / algorithm layout",
                "Algorithm pseudocode is manually formatted instead of using a real algorithm environment.",
                snippet,
                "Rewrite the block with \\begin{algorithm}, \\caption{...}, and algorithmic/algpseudocode commands such as \\Require, \\Ensure, and \\State; do not use center/minipage/enumerate/textbf as the algorithm layout.",
            )
        )

    for idx, block in enumerate(algorithm_blocks, start=1):
        evidence = block["captions"][0] if block["captions"] else re.sub(r"\s+", " ", block["body"][:220]).strip()
        if not block["captions"]:
            issues.append(
                _issue(
                    "high",
                    "Visual layout auditor / algorithm layout",
                    "Algorithm environment is missing a LaTeX caption.",
                    f"algorithm_index={idx}; {evidence[:180]}",
                    "Add a concise \\caption{...} inside the algorithm environment so numbering and references are handled by LaTeX.",
                )
            )
        if re.search(r"\\begin\{(?:enumerate|itemize|list)\}", block["body"], flags=re.IGNORECASE):
            issues.append(
                _issue(
                    "high",
                    "Visual layout auditor / algorithm layout",
                    "Algorithm environment uses list/enumerate formatting instead of pseudocode line structure.",
                    f"algorithm_index={idx}; {evidence[:180]}",
                    "Use algorithmic/algpseudocode or algorithm2e commands for numbered statements and indentation.",
                )
            )
        if not _has_structured_algorithm_body(block["body"], tex):
            issues.append(
                _issue(
                    "high",
                    "Visual layout auditor / algorithm layout",
                    "Algorithm environment lacks algorithmic, algpseudocode, or algorithm2e structure.",
                    f"algorithm_index={idx}; {evidence[:180]}",
                    "Format pseudocode with a real algorithm body, for example \\begin{algorithmic}[1] with \\Require, \\Ensure, \\State, \\If, and \\Return lines.",
                )
            )

    assets = [asset for asset in (figure_assets or []) if isinstance(asset, dict)]
    by_id = {str(asset.get("figure_id") or ""): asset for asset in assets}
    for required_id, label in (("fig_motivation_symbolic", "motivation"), ("fig_overview_symbolic", "overview")):
        asset = by_id.get(required_id)
        if not asset:
            issues.append(
                _issue(
                    "high",
                    "Visual layout auditor / required concept figures",
                    f"Required {label} figure is missing.",
                    required_id,
                    "Generate both mandatory motivation and overview figures with the PaperBanana/Gemini post-writing stage before bundle_ready.",
                )
            )
            continue
        if asset.get("kind") != "diagram" or asset.get("stage") != "postwriting_api_figures":
            issues.append(
                _issue(
                    "high",
                    "Visual layout auditor / required concept figures",
                    f"Required {label} figure was not produced by the post-writing PaperBanana/Gemini diagram stage.",
                    f"figure_id={required_id}; kind={asset.get('kind')}; stage={asset.get('stage')}",
                    "Regenerate the concept figure through run_postwriting_api_figure_stage; native or early placeholders are not accepted.",
                )
            )
        notes = str(asset.get("notes") or "").lower()
        if asset.get("kind") == "fallback" or not asset.get("path") or "paperbanana_failed" in notes or "paperbanana_error" in notes or "paperbanana_not_configured" in notes:
            issues.append(
                _issue(
                    "high",
                    "Visual layout auditor / required concept figures",
                    f"Required {label} figure generation did not produce a usable PaperBanana/Gemini asset.",
                    f"figure_id={required_id}; notes={asset.get('notes')}",
                    "Fix the PaperBanana/Gemini generation failure and rerun the post-writing figure stage.",
                )
            )

    experiment_assets = [asset for asset in assets if _is_valid_experiment_plot_asset(asset)]
    experiment_families = sorted({_infer_experiment_chart_family(asset) for asset in experiment_assets})
    if len(experiment_assets) < MIN_EXPERIMENT_FIGURES:
        issues.append(
            _issue(
                "high",
                "Visual layout auditor / experiment figure pack",
                "The manuscript has too few artifact-backed experiment figures separate from motivation/overview diagrams.",
                f"experiment_plot_assets={len(experiment_assets)}; required={MIN_EXPERIMENT_FIGURES}",
                "Run the experiment_plot_reference manager and plotting orchestra to produce at least three experiment plots beyond motivation/overview diagrams.",
            )
        )
    if len(experiment_families) < MIN_EXPERIMENT_CHART_FAMILIES:
        issues.append(
            _issue(
                "high",
                "Visual layout auditor / experiment figure diversity",
                "Experiment figures do not cover enough distinct chart families.",
                f"families={experiment_families}; required={MIN_EXPERIMENT_CHART_FAMILIES}",
                "Use a non-redundant experiment pack, e.g. grouped bars plus a cost/quality scatter or line figure plus a method-metric heatmap; field-specific plots such as t-SNE/CMC are allowed only when backed by real artifacts.",
            )
        )
    missing_style_refs = [str(asset.get("figure_id") or asset.get("path") or "experiment_plot") for asset in experiment_assets if not _has_style_reference_metadata(asset)]
    if missing_style_refs:
        issues.append(
            _issue(
                "high",
                "Visual layout auditor / experiment figure references",
                "Experiment figures are missing searched style-reference metadata from related papers.",
                ", ".join(missing_style_refs[:8]),
                "Run experiment_plot_reference with live literature search and carry style_reference_keys/style_reference_titles into every experiment-plot asset.",
            )
        )

    for asset in assets:
        text = _concept_asset_text(asset)
        lower = text.lower()
        if not any(token in lower for token in ("motivation", "overview", "concept", "diagram", "mechanism")):
            continue
        if re.search(r"\bfig(?:ure)?\.?\s*\d+\b", lower):
            issues.append(
                _issue(
                    "high",
                    "Visual layout auditor / in-image text policy",
                    "Concept-figure prompt or metadata asks for internal Figure numbering.",
                    str(asset.get("figure_id") or "")[:120],
                    "Remove any in-image Figure/Fig. labels; the LaTeX caption provides numbering.",
                )
            )
        if re.search(r"\b(?:1|2|3)\s*[\).]\s*[A-Za-z]", text):
            issues.append(
                _issue(
                    "medium",
                    "Visual layout auditor / in-image text policy",
                    "Concept-figure prompt or metadata encourages visible numbered panels.",
                    str(asset.get("figure_id") or "")[:120],
                    "Use unnumbered local labels or icons instead of visible 1/2/3 panel numbering.",
                )
            )
        if "three tidy comparison" in lower or "three comparison" in lower or "three-column" in lower or "three column" in lower:
            issues.append(
                _issue(
                    "high",
                    "Visual layout auditor / concept composition",
                    "Motivation figure prompt uses a forced three-column comparison layout.",
                    str(asset.get("figure_id") or "")[:120],
                    "Use a compact tension map, central mechanism schematic, or one worked-example diagram with at most two small callouts.",
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
        "page_count": page_count,
        "figure_count": len(blocks),
        "experiment_figure_count": len(experiment_assets),
        "experiment_chart_families": experiment_families,
        "algorithm_block_count": len(algorithm_blocks) + len(_manual_algorithm_snippets(tex, algorithm_intervals)),
        "issues": issues,
        "next_actions": [issue.get("fix") or issue.get("issue") for issue in issues if issue.get("fix") or issue.get("issue")],
    }
