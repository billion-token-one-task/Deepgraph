"""Venue submission style: titles, figure captions, table labels (ICLR-like)."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

from agents.paper_orchestra_pipeline import INCLUDEGRAPHICS_RE, _latex_escape

# Plot IDs must never replace motivation/overview diagrams (bar charts, AUC plots, etc.).
_PLOT_FIGURE_STEMS = (
    "fig_benchmark_method_panel",
    "fig_main_results_comparison",
    "fig_main_comparison_bar_chart",
    "fig_per_dataset_breakdown",
    "fig_grounding_discrepancy_auc",
    "fig_main_results",
    "fig_ablation",
)

_CURATED_CAPTIONS: dict[str, str] = {
    "fig_motivation_gemini": (
        "VLMs often rely on language priors rather than grounded visual evidence, "
        "motivating internal discrepancy signals for perception-error detection."
    ),
    "fig_overview_gemini": (
        r"CPG keeps parallel seen and generated streams; grounding discrepancy "
        r"$\Delta$ feeds a verification head for calibrated error probability."
    ),
    "fig_benchmark_method_panel": (
        "Benchmark panel comparing methods on the reasoning suite (primary score)."
    ),
    "fig_grounding_discrepancy_auc": (
        "Grounding discrepancy versus verification AUC across the evaluated split."
    ),
    "fig_main_results_comparison": (
        "Primary score comparison across methods and datasets."
    ),
    "fig_per_dataset_breakdown": (
        "Per-dataset primary score breakdown for all compared methods."
    ),
}

_ABLATION_LABELS: dict[str, str] = {
    "no_lcb": "w/o LCB calibration",
    "no_counterfactual_delta": "w/o counterfactual $\\Delta$",
    "no_self_divergence_penalty": "w/o self-divergence penalty",
    "compute_matched_baseline": "Compute-matched baseline",
    "remove_Contrastive Perceptual Grounding (CPG)": "w/o full CPG stack",
    "disable_routing": "w/o selective routing",
    "remove_verifier": "w/o verification head",
    "disable_property_1__grounding_discrepancy___delt": "w/o grounding discrepancy $\\Delta$",
    "disable_property_2__contrastive_loss_creates_ind": "w/o contrastive grounding loss",
    "disable_property_3__verification_classifier_prov": "w/o verification classifier",
}


def use_gemini_submission_figures() -> bool:
    """When false, never force-copy stale gemini PNGs over figures already in main.tex."""
    return os.getenv("DEEPGRAPH_USE_GEMINI_FIGURES", "0").strip().lower() in {"1", "true", "yes"}


def allow_stale_diagram_restore() -> bool:
    return os.getenv("DEEPGRAPH_ALLOW_STALE_DIAGRAM_RESTORE", "0").strip().lower() in {
        "1",
        "true",
        "yes",
    }


_VLM_TITLE_KEYWORDS = (
    "vision-language",
    "vlm",
    "multimodal",
    "visual",
    "perception",
    "grounding",
    "self-verif",
    "self-consistency",
    "hallucination",
    "calibrat",
    "dual-stream",
    "contrastive",
)


def _registry_row_relevant(row: dict) -> bool:
    blob = f"{row.get('title') or ''} {row.get('abstract') or ''}".lower()
    return sum(1 for k in _VLM_TITLE_KEYWORDS if k in blob) >= 2


def _title_phrases_from_registry(registry: list[dict]) -> list[str]:
    phrases: list[str] = []
    for row in registry:
        if not isinstance(row, dict) or not _registry_row_relevant(row):
            continue
        title = str(row.get("title") or "").strip()
        if ":" in title:
            phrases.append(title.split(":", 1)[1].strip())
        for chunk in re.split(r"[,;]", title):
            chunk = chunk.strip()
            if 8 < len(chunk) < 80:
                phrases.append(chunk)
    return phrases


def synthesize_title_from_literature(
    title: str,
    state: dict | None = None,
    orchestrated: dict | None = None,
) -> str:
    """Build a venue-style title from problem awareness + relevant citation-registry themes."""
    state = state or {}
    orchestrated = orchestrated if isinstance(orchestrated, dict) else {}
    method = str(state.get("method_name") or "Contrastive Perceptual Grounding (CPG)").strip()
    if "CPG" not in method and "Contrastive Perceptual" in method:
        method = f"{method} (CPG)"

    awareness = state.get("problem_awareness") if isinstance(state.get("problem_awareness"), dict) else {}
    hook = str(awareness.get("central_question") or awareness.get("motivation") or "").strip()
    method_answer = str(awareness.get("method_answer") or "").strip()

    registry = orchestrated.get("citation_registry") or state.get("citation_registry") or []
    lit_phrases = _title_phrases_from_registry(registry if isinstance(registry, list) else [])

    subtitle = ""
    if "grounding discrepancy" in method_answer.lower() or "dual-stream" in method_answer.lower():
        subtitle = "Self-Detected Perception Errors via Dual-Stream Grounding Discrepancy"
    elif "self-detect" in hook.lower() or "self-verif" in hook.lower():
        subtitle = "Toward Self-Detected Perception Errors in Vision-Language Models"
    elif lit_phrases:
        subtitle = lit_phrases[0]
        if len(subtitle) > 72:
            subtitle = subtitle[:69] + "..."
    else:
        subtitle = "Internal Perception-Error Detection in Vision-Language Models"

    subtitle = re.sub(r"\b\d+\.?\d*\s*%[^.]*", "", subtitle, flags=re.I)
    subtitle = re.sub(r"\bfail\s+\d+[^.]*", "", subtitle, flags=re.I)
    subtitle = re.sub(r"\s{2,}", " ", subtitle).strip(" .:-")
    if not subtitle:
        subtitle = "Internal Perception-Error Detection in Vision-Language Models"

    head = method
    raw = str(title or state.get("title") or "").strip()
    if raw and ":" in raw:
        head = raw.split(":", 1)[0].strip()
        if re.search(r"\d+\.?\d*\s*%|fail\s+\d", raw, re.I):
            head = re.sub(r"\b\d+\.?\d*\s*%.*", "", head).strip()
    if "cpg" not in head.lower() and "contrastive perceptual" in head.lower():
        head = head if "(CPG)" in head else f"{head} (CPG)"

    return f"{head}: {subtitle}"[:200]


def sanitize_submission_title(
    title: str,
    state: dict | None = None,
    orchestrated: dict | None = None,
) -> str:
    """Strip headline stats; prefer literature-aware synthesis over bare method acronym."""
    return synthesize_title_from_literature(title, state, orchestrated)


def short_method_name(name: str) -> str:
    n = str(name or "").strip()
    if not n:
        return n
    if "vanilla" in n.lower():
        return "Vanilla"
    if "chain-of-thought" in n.lower() or n.lower().startswith("always-reason"):
        return "CoT"
    if "cpg" in n.lower() or "contrastive perceptual" in n.lower():
        return "CPG (ours)"
    return n if len(n) <= 36 else n[:33] + "..."


def humanize_ablation_variant(name: str) -> str:
    key = str(name or "").strip()
    if key in _ABLATION_LABELS:
        return _ABLATION_LABELS[key]
    if key.startswith("disable_property_"):
        tail = re.sub(r"^disable_property_\d+__?", "", key)
        tail = tail.replace("__", " ").replace("_", " ")
        tail = re.sub(r"\s+", " ", tail).strip()
        tail = re.sub(r"\bdelt\b", r"$\\Delta$", tail, flags=re.I)
        tail = re.sub(r"\bind\b", "", tail, flags=re.I).strip()
        tail = re.sub(r"\bprov\b", "", tail, flags=re.I).strip()
        if tail:
            return f"w/o {tail}"
    key = key.replace("__", " ").replace("_", " ")
    key = re.sub(r"\s+", " ", key).strip()
    if key.lower().startswith("remove "):
        return "w/o " + key[7:]
    if key.lower().startswith("disable "):
        return "w/o " + key[8:]
    if key.lower().startswith("no "):
        return "w/o " + key[3:]
    return key


def polish_figure_caption(caption: str, asset: dict) -> str:
    fid = str(asset.get("figure_id") or Path(str(asset.get("path") or "")).stem)
    if fid in _CURATED_CAPTIONS:
        return _CURATED_CAPTIONS[fid]
    text = str(caption or "").strip()
    if text.lower().startswith("create a symbolic") or "no in-image title" in text.lower():
        title = str(asset.get("title") or "").strip()
        if title and title.lower() not in {"motivation", "overview"}:
            return title
        if "motivation" in fid.lower():
            return _CURATED_CAPTIONS["fig_motivation_gemini"]
        if "overview" in fid.lower():
            return _CURATED_CAPTIONS["fig_overview_gemini"]
    return text[:400] if text else fid.replace("_", " ").title()


def _is_plot_asset(asset: dict) -> bool:
    fid = str(asset.get("figure_id") or Path(str(asset.get("path") or "")).stem).lower()
    if any(stem in fid for stem in _PLOT_FIGURE_STEMS):
        return True
    if str(asset.get("kind") or "").lower() == "plot":
        return True
    text = " ".join(str(asset.get(k) or "") for k in ("title", "objective", "plot_type")).lower()
    return any(t in text for t in ("bar chart", "grouped bar", "scatter plot", "primary score comparison"))


def _asset_mtime(asset: dict) -> float:
    for key in ("path", "pdf_path", "png_path"):
        path = Path(str(asset.get(key) or ""))
        if path.is_file():
            try:
                return path.stat().st_mtime
            except OSError:
                continue
    return 0.0


def _pick_diagram_asset(assets: list[dict], role: str) -> dict | None:
    role = role.lower()
    candidates: list[dict] = []
    for asset in assets:
        if not isinstance(asset, dict) or _is_plot_asset(asset):
            continue
        fid = str(asset.get("figure_id") or "").lower()
        path = str(asset.get("path") or "").lower()
        stage = str(asset.get("stage") or "").lower()
        if role in fid or role in path or str(asset.get("kind") or "").lower() == "diagram":
            candidates.append(asset)
        elif stage == "postwriting_api_figures" and role in fid:
            candidates.append(asset)
    if not candidates:
        for asset in assets:
            if not isinstance(asset, dict) or _is_plot_asset(asset):
                continue
            fid = str(asset.get("figure_id") or "").lower()
            if "gemini" in fid and not use_gemini_submission_figures():
                continue
            if role in fid or role in str(asset.get("path") or "").lower():
                candidates.append(asset)
    if not candidates:
        return None
    return max(candidates, key=_asset_mtime)


def _collect_plotting_assets(orchestrated: dict, bundle_dir: Path | None = None) -> list[dict]:
    assets: list[dict] = []
    plotting = orchestrated.get("plotting") if isinstance(orchestrated.get("plotting"), dict) else {}
    for row in plotting.get("assets") or plotting.get("figure_assets") or plotting.get("generated_assets") or []:
        if isinstance(row, dict):
            assets.append(row)
    executor = plotting.get("plotting_executor") if isinstance(plotting.get("plotting_executor"), dict) else {}
    for row in executor.get("assets") or []:
        if isinstance(row, dict):
            assets.append(row)
    for row in orchestrated.get("figure_assets") or []:
        if isinstance(row, dict):
            assets.append(row)
    if bundle_dir:
        for rel in ("figures/paperorchestra_plotting_meta.json", "paperorchestra_plotting_meta.json"):
            meta_path = bundle_dir / rel
            if not meta_path.is_file():
                continue
            try:
                payload = json.loads(meta_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            for row in payload.get("assets") or payload.get("generated_assets") or []:
                if isinstance(row, dict):
                    assets.append(row)
            executor = payload.get("plotting_executor") if isinstance(payload.get("plotting_executor"), dict) else {}
            for row in executor.get("assets") or []:
                if isinstance(row, dict):
                    assets.append(row)
    return assets


def select_motivation_overview_diagram_assets(
    orchestrated: dict, bundle_dir: Path | None = None
) -> tuple[dict | None, dict | None]:
    """Return (motivation, overview) diagram assets — never bar-chart result plots."""
    assets = _collect_plotting_assets(orchestrated, bundle_dir)
    motivation = _pick_diagram_asset(assets, "motivation")
    overview = _pick_diagram_asset(assets, "overview")
    if motivation and overview and motivation.get("figure_id") == overview.get("figure_id"):
        overview = _pick_diagram_asset(
            [a for a in assets if a is not motivation],
            "overview",
        )
    return motivation, overview


def replace_title_in_tex(main_tex: str, state: dict, orchestrated: dict | None = None) -> str:
    title = sanitize_submission_title(
        re.search(r"\\title\{([^}]*)\}", main_tex or "").group(1)
        if re.search(r"\\title\{([^}]*)\}", main_tex or "")
        else str(state.get("title") or ""),
        state,
        orchestrated,
    )
    escaped = _latex_escape(title)
    if re.search(r"\\title\{", main_tex or ""):
        return re.sub(r"\\title\{[^}]*\}", rf"\\title{{{escaped}}}", main_tex, count=1)
    return main_tex


def replace_figure_captions_in_tex(main_tex: str) -> str:
    """Replace LLM image-generation instructions used as \\caption text."""
    out = main_tex or ""
    for bad, good in (
        ("Motivation Gemini", _CURATED_CAPTIONS["fig_motivation_gemini"]),
        ("Overview Gemini", _CURATED_CAPTIONS["fig_overview_gemini"]),
    ):
        out = out.replace(rf"\caption{{{bad}}}", rf"\caption{{{_latex_escape(good)}}}")

    def _fix_caption(match: re.Match[str]) -> str:
        body = match.group(1)
        if "create a symbolic" not in body.lower() and "no in-image title" not in body.lower():
            return match.group(0)
        label_m = re.search(r"\\label\{([^}]+)\}", main_tex[match.end() : match.end() + 120])
        fid = label_m.group(1).replace("fig:", "").replace("fig_", "") if label_m else ""
        cap = _CURATED_CAPTIONS.get(fid) or polish_figure_caption("", {"figure_id": fid})
        return rf"\caption{{{_latex_escape(cap)}}}"

    return re.sub(r"\\caption\{([^}]*(?:\{[^}]*\}[^}]*)*)\}", _fix_caption, out, flags=re.DOTALL)


_PLOT_STEM_PATTERN = re.compile(
    r"fig_(?:benchmark_method_panel|main_results_comparison|main_comparison_bar_chart|per_dataset_breakdown|grounding_discrepancy_auc)",
    re.I,
)


def replace_plot_figure_paths_only(
    main_tex: str, bundle_dir: Path, orchestrated: dict
) -> tuple[str, list[str]]:
    """Only fix mistaken bar-chart paths in \\includegraphics; do not overwrite existing diagram PNGs."""
    import shutil

    out = main_tex or ""
    if not _PLOT_STEM_PATTERN.search(out):
        return out, []

    motivation, overview = select_motivation_overview_diagram_assets(orchestrated, bundle_dir)
    if not motivation and not overview:
        return out, []

    figures_dir = bundle_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    applied: list[str] = []

    def _apply(role: str, asset: dict | None, replace_stems: tuple[str, ...]) -> None:
        nonlocal out
        if not asset:
            return
        src = Path(str(asset.get("path") or asset.get("pdf_path") or ""))
        if not src.is_file():
            return
        dest = figures_dir / src.with_suffix(".png").name
        if not dest.exists() or dest.stat().st_mtime < src.stat().st_mtime:
            shutil.copy2(src, dest)
        for plot_stem in replace_stems:
            if plot_stem.lower() not in out.lower():
                continue
            out = re.sub(
                rf"(\\includegraphics(?:\[[^\]]*\])?\{{[^}}]*?){re.escape(plot_stem)}([^}}]*\}})",
                rf"\1{dest.stem}\2",
                out,
                count=1,
                flags=re.I,
            )
            applied.append(f"{plot_stem}->{dest.name}")

    _apply(
        "motivation",
        motivation,
        ("fig_benchmark_method_panel", "fig_grounding_discrepancy_auc"),
    )
    _apply(
        "overview",
        overview,
        ("fig_main_results_comparison", "fig_main_comparison_bar_chart", "fig_per_dataset_breakdown"),
    )
    return out, applied


def restore_diagram_figure_graphics(
    main_tex: str, bundle_dir: Path, orchestrated: dict
) -> tuple[str, list[str]]:
    """Legacy restore disabled by default to avoid rolling back to stale gemini PNGs."""
    if allow_stale_diagram_restore() or use_gemini_submission_figures():
        return replace_plot_figure_paths_only(main_tex, bundle_dir, orchestrated)
    return replace_plot_figure_paths_only(main_tex, bundle_dir, orchestrated)


def dedupe_marked_paragraphs(main_tex: str) -> str:
    """Remove duplicate \\paragraph blocks (deterministic fill without markers)."""
    seen: set[str] = set()
    lines = (main_tex or "").splitlines()
    out: list[str] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.strip().startswith(r"\paragraph{"):
            block = [line]
            i += 1
            while i < len(lines) and not lines[i].strip().startswith(r"\paragraph{") and not lines[i].strip().startswith(r"\section"):
                block.append(lines[i])
                i += 1
            key = "\n".join(block).strip()
            if key in seen:
                continue
            seen.add(key)
            out.extend(block)
            continue
        out.append(line)
        i += 1
    return "\n".join(out) + ("\n" if main_tex.endswith("\n") else "")


def strip_trailing_after_end_document(main_tex: str) -> str:
    m = re.search(r"\\end\{document\}", main_tex or "")
    if not m:
        return main_tex
    return main_tex[: m.end()]


def apply_submission_style_fixes(
    main_tex: str,
    *,
    state: dict,
    bundle_dir: Path | None = None,
    orchestrated: dict | None = None,
) -> tuple[str, dict[str, Any]]:
    """Title, captions, and optional native figure swap."""
    meta: dict[str, Any] = {}
    out = strip_trailing_after_end_document(main_tex)
    out = replace_title_in_tex(out, state, orchestrated)
    meta["title"] = sanitize_submission_title(
        str(state.get("title") or ""), state, orchestrated
    )
    out = replace_figure_captions_in_tex(out)
    out = dedupe_marked_paragraphs(out)
    if bundle_dir and orchestrated:
        out, swaps = replace_plot_figure_paths_only(out, bundle_dir, orchestrated)
        meta["figure_swaps"] = swaps
        meta["diagram_restore_policy"] = "plot_paths_only_no_stale_gemini_rollback"
    return out, meta
