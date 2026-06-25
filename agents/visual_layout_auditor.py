"""Visual layout and figure-policy auditor for manuscript bundles."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any


AUDITOR_VERSION = "deepgraph_visual_layout_auditor_v7_2026_06_14"


VISUAL_LAYOUT_STANDARD_TEXT = """Visual layout standard:
- Figures must never appear before maketitle, title, authors, abstract, or the first substantive paper text.
- Motivation and overview figures are mandatory gpt-image-2 post-writing figures, but they must not be forced into the first viewport and must not be the first object in the paper. Rich near-square concept diagrams should use a balanced single-column size, roughly 0.72--0.88\\linewidth with a height cap and keepaspectratio, so they remain information-rich without becoming a long empty slab.
- LaTeX captions are the only figure captions. Do not add standalone "Figure 1:" paragraphs after a figure.
- Generated images must not contain internal captions, "Figure X" labels, panel numbering such as "1./2./3.", or long explanatory paragraphs.
- Motivation figures must not use a rigid three-column comparison layout. Prefer a compact tension map, central mechanism schematic, or one worked-example diagram with at most two small callouts.
- Concept figures may use short local labels, score tags, or symbols, but the caption and surrounding prose must carry the explanation.
- Algorithm and pseudocode blocks must use real LaTeX algorithm structure (algorithm+algpseudocode/algorithmic or algorithm2e) with a caption and line structure; do not fake algorithms with center/minipage/enumerate/textbf blocks.
- Experiment figures are separate from motivation/overview diagrams: main results and ablation are required artifact-backed plots; hyperparameter/threshold sensitivity is generated when a verified artifact exists and must not block otherwise. The target pack is three plots with searched style-reference metadata from related papers and local user-provided experiment-figure examples.
- Experiment figures must be wide multi-panel plots, specifically 1x3 or 1x4 layouts in figure* / textwidth form; single-column 1x1 experiment plots, default quality-cost scatter plots, and patterned/hatch bars are not allowed. Axis-based plots should use a full four-sided frame rather than only left/bottom axes.
- Numeric tables should fill the available single-column or double-column width using tabularx, tabular*, or resizebox with \\linewidth/\\textwidth; narrow centered tabular blocks are not acceptable."""


FIGURE_ENV_RE = re.compile(r"\\begin\{figure\*?\}.*?\\end\{figure\*?\}", re.DOTALL | re.IGNORECASE)
INCLUDE_RE = re.compile(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", re.IGNORECASE)
INCLUDE_WITH_OPTIONS_RE = re.compile(r"\\includegraphics(?:\[(?P<opts>[^\]]*)\])?\{(?P<path>[^}]+)\}", re.IGNORECASE)
CAPTION_RE = re.compile(r"\\caption\{([\s\S]*?)\}", re.IGNORECASE)
TABLE_ENV_RE = re.compile(r"\\begin\{table\*?\}.*?\\end\{table\*?\}", re.DOTALL | re.IGNORECASE)
ALGORITHM_ENV_RE = re.compile(r"\\begin\{algorithm\*?\}.*?\\end\{algorithm\*?\}", re.DOTALL | re.IGNORECASE)
ALGORITHMIC_ENV_RE = re.compile(r"\\begin\{(?:algorithmic|algorithmicx)\}", re.IGNORECASE)
PSEUDOCODE_COMMAND_RE = re.compile(
    r"\\(?:State|Require|Ensure|Return|For|EndFor|If|Else|EndIf|While|EndWhile|Repeat|Until|KwData|KwResult|KwIn|KwOut|KwRet|SetKwInput|SetKwFunction|SetAlgoLined|DontPrintSemicolon|tcp)\b",
    re.IGNORECASE,
)
MIN_EXPERIMENT_FIGURES = 3
MIN_EXPERIMENT_CHART_FAMILIES = 2
REQUIRED_EXPERIMENT_ROLES = {
    "main_results": {"main_results_bar", "main_results_bar_1x2", "backend_grouped_bars", "backend_rank_lines_1x4"},
    "ablation": {"ablation_bar", "ablation_results"},
}
OPTIONAL_EXPERIMENT_ROLES = {
    "hyperparameter": {"hyperparameter_sweep", "threshold_sweep"},
}
DISALLOWED_DEFAULT_EXPERIMENT_CHARTS = {"quality_cost_tradeoff", "scatter", "method_metric_heatmap"}
FORBIDDEN_CONCEPT_ASPECT_RATIOS = {"16:9", "21:9", "4:1", "3:1", "5:1"}


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


def _include_specs(body: str) -> list[dict[str, str]]:
    specs: list[dict[str, str]] = []
    for match in INCLUDE_WITH_OPTIONS_RE.finditer(body or ""):
        path = (match.group("path") or "").strip()
        specs.append({"path": path, "stem": Path(path).stem, "options": (match.group("opts") or "").strip()})
    return specs


def _include_width_fraction(options: str) -> float | None:
    if not options:
        return None
    match = re.search(r"width\s*=\s*([0-9]*\.?[0-9]+)?\s*\\(?:line|text|column)width", options)
    if not match:
        return None
    raw = match.group(1)
    if raw in (None, ""):
        return 1.0
    try:
        return float(raw)
    except ValueError:
        return None


def _concept_figure_width_issues(block: dict[str, Any]) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    for spec in _include_specs(str(block.get("body") or "")):
        stem = spec.get("stem", "").lower()
        if "motivation" not in stem and "overview" not in stem:
            continue
        options = spec.get("options", "")
        width = _include_width_fraction(options)
        has_height_cap = "height" in options and "keepaspectratio" in options
        if width is None:
            out.append(
                _issue(
                    "medium",
                    "Visual layout auditor / concept figure width",
                    "Motivation/overview figure has no explicit size controller.",
                    f"figure={spec.get('path')}; options={options}",
                    "Typeset concept figures with a balanced size such as \\includegraphics[width=0.82\\linewidth,height=0.46\\textheight,keepaspectratio]{...}.",
                )
            )
        elif width < 0.68:
            out.append(
                _issue(
                    "medium",
                    "Visual layout auditor / concept figure width",
                    "Motivation/overview figure is too small to read comfortably.",
                    f"figure={spec.get('path')}; width_fraction={width:.2f}",
                    "Use roughly 0.72--0.88\\linewidth for concept figures, with a height cap for near-square images.",
                )
            )
        elif width > 0.92 and not has_height_cap:
            out.append(
                _issue(
                    "medium",
                    "Visual layout auditor / concept figure height",
                    "Motivation/overview figure fills the text width without a height cap and may become too tall.",
                    f"figure={spec.get('path')}; width_fraction={width:.2f}; options={options}",
                    "For rich near-square concept diagrams, use a balanced size such as width=0.82\\linewidth,height=0.46\\textheight,keepaspectratio.",
                )
            )
    return out


def _algorithm_blocks(tex: str) -> list[dict[str, Any]]:
    blocks: list[dict[str, Any]] = []
    for match in ALGORITHM_ENV_RE.finditer(tex or ""):
        body = match.group(0)
        captions = [re.sub(r"\s+", " ", raw).strip() for raw in CAPTION_RE.findall(body)]
        blocks.append({"start": match.start(), "end": match.end(), "body": body, "captions": captions})
    return blocks


def _table_blocks(tex: str) -> list[dict[str, Any]]:
    blocks: list[dict[str, Any]] = []
    for match in TABLE_ENV_RE.finditer(tex or ""):
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
    if any(token in text for token in ("line", "curve", "rank", "cmc", "trend", "threshold", "sweep", "hyperparameter")):
        return "line_family"
    if any(token in text for token in ("bar", "grouped", "ablation")):
        return "bar_family"
    if any(token in text for token in ("panel", "diagnostic")):
        return "multipanel_family"
    return "unknown_family"


def _experiment_chart_type(asset: dict[str, Any]) -> str:
    explicit = str(asset.get("chart_type") or "").strip().lower()
    if explicit:
        return explicit
    text = " ".join(
        str(asset.get(key) or "")
        for key in ("figure_id", "title", "objective", "renderer", "notes", "path")
    ).lower()
    if "ablation" in text:
        return "ablation_bar"
    if any(token in text for token in ("hyperparameter", "threshold", "sweep", "sensitivity")):
        return "hyperparameter_sweep"
    if any(token in text for token in ("main", "benchmark", "result")):
        return "main_results_bar"
    if any(token in text for token in ("scatter", "tradeoff", "frontier")):
        return "scatter"
    return "unknown"


def _experiment_roles_present(experiment_assets: list[dict[str, Any]]) -> set[str]:
    chart_types = {_experiment_chart_type(asset) for asset in experiment_assets}
    present: set[str] = set()
    role_sets = {**REQUIRED_EXPERIMENT_ROLES, **OPTIONAL_EXPERIMENT_ROLES}
    for role, allowed in role_sets.items():
        if chart_types.intersection(allowed):
            present.add(role)
    return present


def _is_disallowed_default_scatter(asset: dict[str, Any]) -> bool:
    chart_type = _experiment_chart_type(asset)
    text = " ".join(str(asset.get(key) or "") for key in ("figure_id", "title", "objective", "renderer", "notes", "path")).lower()
    if chart_type in DISALLOWED_DEFAULT_EXPERIMENT_CHARTS:
        return True
    if any(token in text for token in ("quality-cost", "quality cost", "cost tradeoff", "cost frontier")):
        return True
    if "scatter" in text and not any(token in text for token in ("t-sne", "tsne", "umap", "embedding artifact")):
        return True
    return False


def _uses_forbidden_bar_pattern(asset: dict[str, Any]) -> bool:
    if bool(asset.get("uses_hatch")):
        return True
    text = " ".join(str(asset.get(key) or "") for key in ("figure_id", "title", "objective", "renderer", "notes", "chart_type")).lower()
    return bool(re.search(r"\b(?:hatch|stipple|stripe|striped|dotted|dot pattern|pattern texture)\b", text))


def _has_style_reference_metadata(asset: dict[str, Any]) -> bool:
    return bool(asset.get("style_reference_keys") or asset.get("style_reference_titles"))


def _has_required_style_provenance(asset: dict[str, Any]) -> bool:
    sources = asset.get("style_reference_sources") or asset.get("style_sources") or []
    if isinstance(sources, str):
        sources_text = sources.lower()
    else:
        sources_text = " ".join(str(x) for x in sources).lower()
    has_local = bool(asset.get("local_style_reference_dir") or "local" in sources_text or "user" in sources_text or "实验图例子" in sources_text)
    has_literature = bool(asset.get("style_reference_keys") or asset.get("style_reference_titles") or "literature" in sources_text or "paper" in sources_text)
    return has_local and has_literature


def _experiment_asset_layout(asset: dict[str, Any]) -> str:
    return str(asset.get("layout") or asset.get("panel_layout") or "").strip().lower()


def _is_allowed_experiment_layout(asset: dict[str, Any]) -> bool:
    layout = _experiment_asset_layout(asset)
    aspect = str(asset.get("aspect_ratio") or "").strip().lower()
    chart_type = str(asset.get("chart_type") or "").strip().lower()
    figure_id = str(asset.get("figure_id") or "").strip().lower()
    text = " ".join([layout, aspect, chart_type, figure_id])
    if layout in {"1x3", "1x4"}:
        return True
    return bool(re.search(r"(?:^|[^0-9])1x[34](?:[^0-9]|$)", text))


def _is_wide_latex_figure_block(block: dict[str, Any]) -> bool:
    body = str(block.get("body") or "")
    if not re.search(r"\\begin\{figure\*\}", body):
        return False
    return bool(re.search(r"\\includegraphics\s*(?:\[[^\]]*(?:\\textwidth|width\s*=\s*(?:0?\.\d+)?\\textwidth)[^\]]*\])?", body))


def _table_fills_available_width(body: str) -> bool:
    body = body or ""
    if re.search(r"\\begin\{tabularx\}\s*\{\s*\\(?:line|text)width\s*\}", body):
        return True
    if re.search(r"\\begin\{tabular\*\}\s*\{\s*\\(?:line|text)width\s*\}", body):
        return True
    if re.search(r"\\resizebox\s*\{\s*\\(?:line|text)width\s*\}", body):
        return True
    return False


def _table_has_cramped_method_column(body: str) -> bool:
    body = body or ""
    return bool(re.search(r"\\begin\{tabularx\}\s*\{\s*\\textwidth\s*\}\s*\{\s*l\s*\*\s*\{\s*[4-9]", body))


def _table_data_row_count(body: str) -> int:
    rows = 0
    for raw in (body or "").splitlines():
        line = raw.strip()
        if not line or line.startswith("%"):
            continue
        if re.search(r"\\\\\s*(?:%.*)?$", line) and not re.search(r"\\(?:toprule|midrule|bottomrule|cmidrule|caption|label)\b", line):
            rows += 1
    return rows


def _table_column_count(body: str) -> int:
    matches = re.findall(r"^\s*([^%\\][^\n]*?)\\\\\s*(?:%.*)?$", body or "", flags=re.MULTILINE)
    if not matches:
        return 0
    counts = [row.count("&") + 1 for row in matches if "&" in row]
    return max(counts) if counts else 0


def _table_width_specs(body: str) -> list[str]:
    specs: list[str] = []
    for pattern in (
        r"\\begin\{tabularx\}\s*\{([^{}]+)\}",
        r"\\begin\{tabular\*\}\s*\{([^{}]+)\}",
        r"\\resizebox\s*\{([^{}]+)\}",
    ):
        specs.extend(match.strip() for match in re.findall(pattern, body or "", flags=re.IGNORECASE))
    return specs


def _latex_width_factor(spec: str) -> tuple[float | None, str]:
    compact = re.sub(r"\s+", "", spec or "")
    match = re.match(r"([0-9]*\.?[0-9]+)\\(textwidth|linewidth|columnwidth)$", compact)
    if match:
        return float(match.group(1)), match.group(2)
    macro = re.sub(r"[^A-Za-z]", "", compact).lower()
    if compact in {r"\textwidth", r"\linewidth", r"\columnwidth"}:
        return 1.0, macro
    return None, macro


def _table_width_issues(body: str, tex: str) -> list[dict[str, str]]:
    issues: list[dict[str, str]] = []
    evidence = _table_width_specs(body)
    is_star = bool(re.search(r"\\begin\{table\*\}", body or ""))
    two_column = bool(re.search(r"\\documentclass\[[^\]]*twocolumn|\\twocolumn\b", tex or "", flags=re.IGNORECASE))
    column_count = _table_column_count(body)
    if not evidence and column_count >= 5:
        issues.append(_issue("high", "Visual layout auditor / table width fit", "Wide numeric table has no explicit width controller and may overflow the text block.", f"columns={column_count}", "Wrap wide tables in tabularx/tabular* or resizebox at exactly \\linewidth for table or \\textwidth for table*."))
        return issues
    for spec in evidence:
        factor, macro = _latex_width_factor(spec)
        if factor is not None and factor > 1.01:
            issues.append(_issue("high", "Visual layout auditor / table overflow", "Table width is larger than the available text block.", f"width={spec}", "Use exactly \\linewidth for single-column table or \\textwidth for table*; never scale beyond 1.0 of the available width."))
        elif factor is not None and factor < 0.92:
            issues.append(_issue("medium", "Visual layout auditor / table width fit", "Table is substantially narrower than the available column width.", f"width={spec}", "Set the table target width to \\linewidth or \\textwidth so the numeric comparison fills the column cleanly."))
        if two_column and not is_star and macro == "textwidth":
            issues.append(_issue("high", "Visual layout auditor / table overflow", "Single-column table uses \\textwidth in a two-column layout and may spill into the neighboring column.", f"width={spec}", "Use \\linewidth inside table or promote the table to table* with \\textwidth."))
    return issues


def _table_spacing_issues(body: str) -> list[dict[str, str]]:
    issues: list[dict[str, str]] = []
    rows = _table_data_row_count(body)
    stretch_match = re.search(r"\\renewcommand\s*\{\\arraystretch\}\s*\{([0-9]*\.?[0-9]+)\}", body or "")
    stretch = float(stretch_match.group(1)) if stretch_match else None
    sep_match = re.search(r"\\setlength\s*\{\\tabcolsep\}\s*\{([0-9]*\.?[0-9]+)\s*pt\}", body or "")
    tabcolsep = float(sep_match.group(1)) if sep_match else None
    if stretch is not None and stretch < 0.98:
        issues.append(_issue("high", "Visual layout auditor / table row spacing", "Table rows are compressed below readable spacing.", f"arraystretch={stretch}", "Use arraystretch around 1.06--1.14 for dense experiment tables; avoid squeezing rows to save space."))
    elif rows >= 8 and (stretch is None or stretch < 1.04):
        issues.append(_issue("medium", "Visual layout auditor / table row spacing", "Long table does not reserve enough row spacing for readable review-scale PDF output.", f"rows={rows}; arraystretch={stretch}", "Set \\renewcommand{\\arraystretch}{1.08} or similar for long numeric tables."))
    if tabcolsep is not None and tabcolsep < 2.5:
        issues.append(_issue("high", "Visual layout auditor / table column spacing", "Table column separation is too small and likely makes numbers collide visually.", f"tabcolsep={tabcolsep}pt", "Use a moderate \\tabcolsep, usually 3.5--5pt, together with compact fixed-width numeric columns."))
    elif tabcolsep is not None and tabcolsep < 3.5:
        issues.append(_issue("medium", "Visual layout auditor / table column spacing", "Table column separation is very tight.", f"tabcolsep={tabcolsep}pt", "Increase \\tabcolsep to roughly 4pt unless the table is explicitly resized to fit."))
    if re.search(r"\\(?:tiny|scriptsize)\b", body or "") and rows >= 6:
        issues.append(_issue("high", "Visual layout auditor / table font size", "Dense table uses tiny/scriptsize text, which is usually unreadable in review PDFs.", f"rows={rows}", "Use \\small at most, widen the table, or split it into main and supplementary tables."))
    return issues


def _table_style_issues(body: str) -> list[dict[str, str]]:
    issues: list[dict[str, str]] = []
    rows = _table_data_row_count(body)
    columns = _table_column_count(body)
    if rows < 3 and columns < 4:
        return issues
    has_booktabs = all(rule in (body or "") for rule in (r"\toprule", r"\midrule", r"\bottomrule"))
    has_header_shading = bool(re.search(r"\\(?:rowcolor|cellcolor)\{(?:gray|black|blue|cyan|teal|yellow)!", body or ""))
    has_grouping = bool(re.search(r"\\(?:cmidrule|addlinespace|multicolumn)\b", body or "")) or len(re.findall(r"\\midrule", body or "")) >= 2
    has_ours = bool(re.search(r"\b(?:ours|proposed)\b", body or "", flags=re.IGNORECASE))
    ours_highlighted = bool(re.search(r"\\rowcolor\{[^}]+\}[^\n]*(?:ours|proposed)|(?:ours|proposed)[^\n]*\\textbf", body or "", flags=re.IGNORECASE))
    if not has_booktabs:
        issues.append(_issue("medium", "Visual layout auditor / table style polish", "Numeric table lacks complete booktabs rules.", f"rows={rows}; columns={columns}", "Use \\toprule, \\midrule, and \\bottomrule rather than plain grid-like tables."))
    if not has_header_shading and rows >= 4:
        issues.append(_issue("medium", "Visual layout auditor / table style polish", "Experiment table lacks a subtle shaded header or grouped header band.", f"rows={rows}; columns={columns}", "Add a light gray header band or grouped header rows to make the table scan like a polished comparison table."))
    if rows >= 10 and not has_grouping:
        issues.append(_issue("medium", "Visual layout auditor / table grouping", "Long comparison table has no visible group separation.", f"rows={rows}; columns={columns}", "Use grouped header rows, \\addlinespace, \\cmidrule, or section separators for long tables."))
    if has_ours and not ours_highlighted:
        issues.append(_issue("medium", "Visual layout auditor / table result emphasis", "Table includes an ours/proposed row without visual emphasis.", "ours/proposed row detected", "Highlight the final method row with a pale row color and bold the primary metric values."))
    return issues


def _concept_asset_text(asset: dict[str, Any]) -> str:
    return " ".join(
        str(asset.get(key) or "")
        for key in ("figure_id", "title", "objective", "caption", "notes", "renderer", "layout", "aspect_ratio")
    )


NEGATION_WINDOW_TERMS = ("do not", "don't", "never", "avoid", "forbid", "forbidden", "not", "no ", "must not", "should not", "without")


def _has_unnegated_phrase(text: str, phrases: tuple[str, ...], *, window: int = 72) -> bool:
    lower = text.lower()
    for phrase in phrases:
        start = 0
        needle = phrase.lower()
        while True:
            idx = lower.find(needle, start)
            if idx < 0:
                break
            prefix = lower[max(0, idx - window):idx]
            if not any(term in prefix for term in NEGATION_WINDOW_TERMS):
                return True
            start = idx + len(needle)
    return False


def _is_negated_sentence_context(text: str, start: int, end: int) -> bool:
    lower = text.lower()
    left = max(lower.rfind(".", 0, start), lower.rfind(";", 0, start), lower.rfind("\n", 0, start))
    right_candidates = [pos for pos in (lower.find(".", end), lower.find(";", end), lower.find("\n", end)) if pos >= 0]
    right = min(right_candidates) if right_candidates else min(len(lower), end + 180)
    sentence = lower[left + 1 : right]
    if any(term in sentence for term in NEGATION_WINDOW_TERMS):
        return True
    prefix = lower[max(0, start - 180):start]
    return any(term in prefix for term in NEGATION_WINDOW_TERMS)


def audit_visual_layout(
    *,
    main_tex: str,
    figure_assets: list[dict[str, Any]] | None = None,
    page_count: int | None = None,
    allow_deterministic_concept_fallback: bool = False,
) -> dict[str, Any]:
    tex = main_tex or ""
    issues: list[dict[str, str]] = []
    blocks = _figure_blocks(tex)
    table_blocks = _table_blocks(tex)
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
            issues.extend(_concept_figure_width_issues(block))

    for idx, table in enumerate(table_blocks, start=1):
        body = str(table.get("body") or "")
        if re.search(r"\\begin\{tabular\}", body) and not _table_fills_available_width(body):
            evidence = (table.get("captions") or [f"table_index={idx}"])[0]
            issues.append(
                _issue(
                    "high",
                    "Visual layout auditor / table width",
                    "Numeric table uses a narrow centered tabular instead of filling the available column width.",
                    str(evidence)[:180],
                    "Use tabularx/tabular* or resizebox with \\linewidth for single-column tables and \\textwidth for table* tables.",
                )
            )
        if _table_has_cramped_method_column(body):
            evidence = (table.get("captions") or [f"table_index={idx}"])[0]
            issues.append(
                _issue(
                    "high",
                    "Visual layout auditor / table column allocation",
                    "Wide numeric table uses a cramped left Method column, causing method names to wrap poorly and numbers to spread out.",
                    str(evidence)[:180],
                    "Use a flexible ragged-right Method column such as >{\\raggedright\\arraybackslash}X and compact fixed-width centered numeric columns.",
                )
            )

        issues.extend(_table_width_issues(body, tex))
        issues.extend(_table_spacing_issues(body))
        issues.extend(_table_style_issues(body))


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
                    "Generate both mandatory motivation and overview figures with the gpt-image-2 post-writing stage before bundle_ready.",
                )
            )
            continue
        accepted_concept_stages = {"postwriting_api_figures"}
        if allow_deterministic_concept_fallback:
            accepted_concept_stages.add("deterministic_concept_fallback")
        if asset.get("kind") != "diagram" or asset.get("stage") not in accepted_concept_stages:
            issues.append(
                _issue(
                    "high",
                    "Visual layout auditor / required concept figures",
                    f"Required {label} figure was not produced by an accepted post-writing diagram stage.",
                    f"figure_id={required_id}; kind={asset.get('kind')}; stage={asset.get('stage')}",
                    "Regenerate the concept figure through run_postwriting_api_figure_stage, or route controlled-scope reports through the deterministic concept fallback.",
                )
            )
        aspect = str(asset.get("aspect_ratio") or "").strip().lower().replace(" ", "")
        if aspect in FORBIDDEN_CONCEPT_ASPECT_RATIOS:
            issues.append(
                _issue(
                    "high",
                    "Visual layout auditor / concept aspect ratio",
                    f"Required {label} figure uses a horizontally stretched aspect ratio.",
                    f"figure_id={required_id}; aspect_ratio={asset.get('aspect_ratio')}",
                    "Regenerate motivation/overview figures as compact 4:3 academic schematics, not 16:9 or panoramic banners.",
                )
            )
        notes = str(asset.get("notes") or "").lower()
        if asset.get("kind") == "fallback" or not asset.get("path") or ((not allow_deterministic_concept_fallback) and ("paperbanana_failed" in notes or "paperbanana_error" in notes or "paperbanana_not_configured" in notes)):
            issues.append(
                _issue(
                    "high",
                    "Visual layout auditor / required concept figures",
                    f"Required {label} figure generation did not produce a usable gpt-image-2 asset.",
                    f"figure_id={required_id}; notes={asset.get('notes')}",
                    "Fix the gpt-image-2 generation failure and rerun the post-writing figure stage.",
                )
            )

    experiment_assets = [asset for asset in assets if _is_valid_experiment_plot_asset(asset)]
    experiment_families = sorted({_infer_experiment_chart_family(asset) for asset in experiment_assets})
    roles_present = _experiment_roles_present(experiment_assets)
    missing_roles = sorted(set(REQUIRED_EXPERIMENT_ROLES) - roles_present)
    has_required_experiment_core = len(experiment_assets) >= 2 and not missing_roles
    if len(experiment_assets) < MIN_EXPERIMENT_FIGURES:
        severity = "medium" if has_required_experiment_core else "high"
        issues.append(
            _issue(
                severity,
                "Visual layout auditor / experiment figure pack",
                "The manuscript is below the target of three artifact-backed experiment figures separate from motivation/overview diagrams.",
                f"experiment_plot_assets={len(experiment_assets)}; target={MIN_EXPERIMENT_FIGURES}; hard_minimum=main_results+ablation",
                "Add a hyperparameter/threshold or field-specific diagnostic figure when a verified artifact exists; do not block a paper that already has main results and ablation plots.",
            )
        )
    if len(experiment_families) < MIN_EXPERIMENT_CHART_FAMILIES:
        severity = "medium" if has_required_experiment_core else "high"
        issues.append(
            _issue(
                severity,
                "Visual layout auditor / experiment figure diversity",
                "Experiment figures do not cover the target number of distinct chart families.",
                f"families={experiment_families}; target={MIN_EXPERIMENT_CHART_FAMILIES}; hard_minimum=main_results+ablation",
                "Prefer a non-redundant pack with main results, ablation, and a sensitivity or field-specific diagnostic plot when such artifacts are available.",
            )
        )
    if missing_roles:
        issues.append(
            _issue(
                "high",
                "Visual layout auditor / experiment figure required roles",
                "Experiment figure pack is missing required plot roles.",
                f"missing_roles={missing_roles}; present_roles={sorted(roles_present)}",
                "Generate separate artifact-backed figures for main results and ablation before entering final writing; hyperparameter/threshold plots are optional when no verified sweep artifact exists.",
            )
        )
    disallowed_scatter_assets = [str(asset.get("figure_id") or asset.get("path") or "experiment_plot") for asset in experiment_assets if _is_disallowed_default_scatter(asset)]
    if disallowed_scatter_assets:
        issues.append(
            _issue(
                "high",
                "Visual layout auditor / experiment scatter policy",
                "Default experiment pack contains a quality-cost/scatter/heatmap-style plot that is not allowed for this paper.",
                ", ".join(disallowed_scatter_assets[:8]),
                "Replace default scatter or mixed heatmap panels with ablation and hyperparameter/threshold sensitivity figures; use field-specific scatter only as an approved extra with real artifacts.",
            )
        )
    patterned_assets = [str(asset.get("figure_id") or asset.get("path") or "experiment_plot") for asset in experiment_assets if _uses_forbidden_bar_pattern(asset)]
    if patterned_assets:
        issues.append(
            _issue(
                "high",
                "Visual layout auditor / experiment bar texture",
                "Experiment figures use hatch/dot/pattern textures in bars.",
                ", ".join(patterned_assets[:8]),
                "Render bars with solid low-saturation fills, dark edges, direct labels, and stable ordering; do not use hatch, dotted, stippled, or striped textures.",
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
    missing_combined_provenance = [str(asset.get("figure_id") or asset.get("path") or "experiment_plot") for asset in experiment_assets if not _has_required_style_provenance(asset)]
    if missing_combined_provenance:
        issues.append(
            _issue(
                "high",
                "Visual layout auditor / experiment figure provenance",
                "Experiment figures do not document both local user-provided style examples and searched related-paper style references.",
                ", ".join(missing_combined_provenance[:8]),
                "Carry style_reference_sources plus local_style_reference_dir and literature style_reference_keys/style_reference_titles into every experiment-plot asset.",
            )
        )
    bad_layout_assets = [str(asset.get("figure_id") or asset.get("path") or "experiment_plot") for asset in experiment_assets if not _is_allowed_experiment_layout(asset)]
    if bad_layout_assets:
        issues.append(
            _issue(
                "high",
                "Visual layout auditor / experiment panel layout",
                "Experiment figures include single-panel or non-1x3/1x4 layouts; single-column 1x1 experiment plots are not allowed.",
                ", ".join(bad_layout_assets[:8]),
                "Render experiment figures as wide multi-panel 1x3 or 1x4 layouts, using figure* and \textwidth in LaTeX.",
            )
        )
    experiment_by_stem: dict[str, dict[str, Any]] = {}
    for asset in experiment_assets:
        for key in ("path", "svg_path", "pdf_path"):
            raw = str(asset.get(key) or "")
            if raw:
                experiment_by_stem[Path(raw).stem] = asset
    narrow_blocks: list[str] = []
    for block in blocks:
        stems = [Path(raw).stem for raw in block.get("includes") or []]
        if any(stem in experiment_by_stem for stem in stems) and not _is_wide_latex_figure_block(block):
            narrow_blocks.extend(stem for stem in stems if stem in experiment_by_stem)
    if narrow_blocks:
        issues.append(
            _issue(
                "high",
                "Visual layout auditor / experiment figure placement",
                "Experiment figures are placed as single-column figures instead of wide multi-panel figure* blocks.",
                ", ".join(narrow_blocks[:8]),
                "Use \\begin{figure*} with \\includegraphics[width=\\textwidth]{...} for experiment figures.",
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
        numbered_panel_match = re.search(r"\b(?:1|2|3)\s*[\).]\s*[A-Za-z]", text)
        if numbered_panel_match and not _is_negated_sentence_context(text, numbered_panel_match.start(), numbered_panel_match.end()):
            issues.append(
                _issue(
                    "medium",
                    "Visual layout auditor / in-image text policy",
                    "Concept-figure prompt or metadata encourages visible numbered panels.",
                    str(asset.get("figure_id") or "")[:120],
                    "Use unnumbered local labels or icons instead of visible 1/2/3 panel numbering.",
                )
            )
        if _has_unnegated_phrase(
            lower,
            ("three tidy comparison", "three comparison", "three-column", "three column"),
        ):
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
        "experiment_panel_layouts": sorted({_experiment_asset_layout(asset) or "unknown" for asset in experiment_assets}),
        "table_count": len(table_blocks),
        "algorithm_block_count": len(algorithm_blocks) + len(_manual_algorithm_snippets(tex, algorithm_intervals)),
        "issues": issues,
        "next_actions": [issue.get("fix") or issue.get("issue") for issue in issues if issue.get("fix") or issue.get("issue")],
    }
