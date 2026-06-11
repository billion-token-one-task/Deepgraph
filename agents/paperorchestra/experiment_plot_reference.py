"""Reference-grounded experiment plot planning for PaperOrchestra.

This agent is intentionally separate from motivation/overview diagram generation.
It performs real literature search for field-specific experimental figure styles,
then emits an artifact-backed experiment plotting plan with at least three
distinct visual families.
"""

from __future__ import annotations

import json
import re
from typing import Any

from agents.paperorchestra.semantic_scholar import paper_to_bibtex_key, paper_year, search_papers


EXPERIMENT_PLOT_REFERENCE_VERSION = "deepgraph_experiment_plot_reference_v1_2026_06_11"
DEFAULT_MIN_EXPERIMENT_FIGURES = 3
DEFAULT_MIN_STYLE_REFERENCES = 3
DEFAULT_MIN_DISTINCT_FAMILIES = 3

CHART_FAMILY_BY_TYPE = {
    "main_results_bar": "bar_family",
    "main_results_bar_1x2": "bar_family",
    "backend_grouped_bars": "bar_family",
    "quality_cost_tradeoff": "distribution_family",
    "method_metric_heatmap": "matrix_family",
    "backend_heatmap_single": "matrix_family",
    "backend_rank_lines_1x4": "line_family",
    "ablation_bar": "bar_family",
    "tsne_embedding": "distribution_family",
    "umap_embedding": "distribution_family",
    "cmc_curve": "line_family",
    "ranking_examples": "qualitative_family",
}

DOMAIN_RULES = [
    {
        "domain": "person_re_identification",
        "needles": ["re-id", "reid", "re identification", "person re-identification", "visible-infrared", "vi-reid"],
        "queries": [
            "person re-identification experimental results t-SNE visualization CMC mAP",
            "visible infrared person re-identification t-SNE CMC mAP ablation",
            "person re-identification retrieval ranking examples CMC curve experimental figure",
        ],
        "recommended_chart_types": ["tsne_embedding", "cmc_curve", "ranking_examples", "main_results_bar", "method_metric_heatmap"],
    },
    {
        "domain": "vision_classification_or_detection",
        "needles": ["image", "vision", "classification", "detection", "segmentation", "clip", "imagenet", "cifar"],
        "queries": [
            "computer vision experimental results t-SNE ablation heatmap robustness figure",
            "image classification paper experimental figures ablation heatmap calibration",
        ],
        "recommended_chart_types": ["main_results_bar", "method_metric_heatmap", "quality_cost_tradeoff", "tsne_embedding"],
    },
    {
        "domain": "graph_learning",
        "needles": ["graph", "node", "edge", "gnn", "network"],
        "queries": [
            "graph neural network experimental results ablation heatmap t-SNE visualization",
            "graph learning paper experimental figure node classification ablation robustness",
        ],
        "recommended_chart_types": ["main_results_bar", "method_metric_heatmap", "quality_cost_tradeoff", "tsne_embedding"],
    },
    {
        "domain": "llm_reasoning",
        "needles": ["llm", "large language model", "reasoning", "multi-agent", "self-consistency", "debate", "test-time", "inference-time"],
        "queries": [
            "large language model reasoning experiments accuracy cost ablation figure",
            "multi-agent LLM debate reasoning experiments cost accuracy ablation visualization",
            "test-time compute language models accuracy token cost frontier experiment figure",
        ],
        "recommended_chart_types": ["main_results_bar", "quality_cost_tradeoff", "method_metric_heatmap", "backend_rank_lines_1x4"],
    },
    {
        "domain": "retrieval_or_rag",
        "needles": ["retrieval", "rag", "ranking", "recall", "mrr", "ndcg"],
        "queries": [
            "retrieval augmented generation experimental results recall nDCG ablation heatmap figure",
            "information retrieval paper experimental figure ranking examples ablation",
        ],
        "recommended_chart_types": ["main_results_bar", "method_metric_heatmap", "quality_cost_tradeoff", "ranking_examples"],
    },
]

GENERIC_STYLE_QUERIES = [
    "machine learning paper experimental results ablation heatmap uncertainty figure",
    "neural network paper experiment figures main results ablation sensitivity heatmap",
    "benchmark paper experimental plots grouped bar heatmap ablation cost tradeoff",
]


class ExperimentPlotReferenceError(RuntimeError):
    """Raised when experiment figures cannot be grounded in searched style references."""

    def __init__(self, report: dict[str, Any]):
        self.report = report
        blockers = report.get("blockers") or []
        detail = "; ".join(str(x) for x in blockers[:4]) if blockers else "unknown blocker"
        super().__init__(f"Experiment plot reference manager blocked plotting: {detail}")


def _json_text(value: Any, limit: int = 8000) -> str:
    try:
        text = json.dumps(value, ensure_ascii=False, default=str)
    except Exception:
        text = str(value)
    return text[:limit]


def _state_text(outline: dict[str, Any], state: dict[str, Any], evidence_brief: dict[str, Any] | None) -> str:
    bits = [
        state.get("title"),
        state.get("method_name"),
        state.get("problem_statement"),
        state.get("baseline_metric_name"),
        _json_text(state.get("paper_intent") or {}, 1200),
        _json_text(state.get("problem_awareness") or {}, 1800),
        _json_text(state.get("result_packet") or state.get("benchmark_summary") or {}, 3200),
        _json_text(outline.get("section_plan") if isinstance(outline, dict) else {}, 1200),
        _json_text(evidence_brief or {}, 2400),
    ]
    return " ".join(str(x or "") for x in bits).lower()


def _detect_domains(text: str) -> list[dict[str, Any]]:
    hits: list[dict[str, Any]] = []
    for rule in DOMAIN_RULES:
        if any(needle in text for needle in rule["needles"]):
            hits.append(rule)
    if not hits:
        hits.append(
            {
                "domain": "generic_machine_learning",
                "needles": [],
                "queries": GENERIC_STYLE_QUERIES,
                "recommended_chart_types": ["main_results_bar", "quality_cost_tradeoff", "method_metric_heatmap"],
            }
        )
    return hits


def _query_pool(domains: list[dict[str, Any]], state: dict[str, Any], metric_name: str) -> list[str]:
    title = str(state.get("title") or state.get("method_name") or "").strip()
    metric = str(metric_name or state.get("baseline_metric_name") or "metric").strip()
    queries: list[str] = []
    for rule in domains:
        queries.extend(str(q) for q in rule.get("queries") or [])
    if title:
        queries.append(f"{title} experimental results figure ablation heatmap")
    if metric:
        queries.append(f"{metric} benchmark experimental figure ablation cost heatmap")
    queries.extend(GENERIC_STYLE_QUERIES[:2])
    deduped: list[str] = []
    seen: set[str] = set()
    for q in queries:
        q = re.sub(r"\s+", " ", q).strip()
        key = q.lower()
        if q and key not in seen:
            seen.add(key)
            deduped.append(q)
    return deduped[:10]


def _infer_chart_tags(paper: dict[str, Any], domains: list[dict[str, Any]]) -> list[str]:
    text = " ".join(str(paper.get(k) or "") for k in ("title", "abstract", "venue")).lower()
    tags: list[str] = []
    checks = [
        ("tsne_embedding", ["t-sne", "tsne", "embedding visualization", "feature visualization"]),
        ("umap_embedding", ["umap"]),
        ("cmc_curve", ["cmc", "cumulative matching", "rank-1", "rank 1"]),
        ("ranking_examples", ["ranking", "retrieval examples", "qualitative", "top-k", "top k"]),
        ("method_metric_heatmap", ["heatmap", "matrix", "correlation", "confusion"]),
        ("quality_cost_tradeoff", ["cost", "latency", "tokens", "efficiency", "trade-off", "tradeoff", "frontier"]),
        ("ablation_bar", ["ablation", "component", "sensitivity"]),
        ("main_results_bar", ["benchmark", "main results", "comparison", "accuracy", "map", "f1"]),
    ]
    for tag, needles in checks:
        if any(needle in text for needle in needles):
            tags.append(tag)
    for rule in domains:
        for tag in rule.get("recommended_chart_types") or []:
            if tag not in tags:
                tags.append(tag)
    return tags[:6]


def _style_reference_row(paper: dict[str, Any], domains: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "paper_id": paper.get("paperId") or "",
        "style_key": paper_to_bibtex_key(paper),
        "title": paper.get("title") or "Untitled",
        "year": paper_year(paper),
        "venue": paper.get("venue") or "",
        "citation_count": paper.get("citationCount") or 0,
        "chart_tags": _infer_chart_tags(paper, domains),
    }


def _dedupe_references(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in rows:
        key = str(row.get("paper_id") or row.get("title") or "").lower()
        key = re.sub(r"\s+", " ", key).strip()
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(row)
    out.sort(key=lambda r: (len(r.get("chart_tags") or []), int(r.get("citation_count") or 0), int(r.get("year") or 0)), reverse=True)
    return out


def _benchmark_summary(state: dict[str, Any]) -> dict[str, Any]:
    packet = state.get("result_packet") if isinstance(state.get("result_packet"), dict) else {}
    summary = packet.get("benchmark_summary") if isinstance(packet.get("benchmark_summary"), dict) else None
    if isinstance(summary, dict):
        return summary
    summary = state.get("benchmark_summary") if isinstance(state.get("benchmark_summary"), dict) else None
    return summary or {}


def _has_recursive_key(value: Any, needles: tuple[str, ...], depth: int = 0) -> bool:
    if depth > 5:
        return False
    if isinstance(value, dict):
        for key, item in value.items():
            key_l = str(key).lower()
            if any(needle in key_l for needle in needles):
                return True
            if _has_recursive_key(item, needles, depth + 1):
                return True
    elif isinstance(value, list):
        return any(_has_recursive_key(item, needles, depth + 1) for item in value[:40])
    return False


def _evidence_flags(state: dict[str, Any]) -> dict[str, bool]:
    summary = _benchmark_summary(state)
    per_method = summary.get("per_method") if isinstance(summary.get("per_method"), dict) else {}
    has_cost = False
    for row in per_method.values():
        if isinstance(row, dict) and any(k in row for k in ("avg_new_tokens", "avg_latency_seconds", "tokens", "latency", "cost")):
            has_cost = True
            break
    return {
        "per_method": bool(per_method),
        "cost_or_latency": has_cost,
        "ablation": bool(summary.get("ablation_table") or state.get("ablation_table")),
        "difficulty": bool(summary.get("difficulty_breakdown") or summary.get("per_difficulty") or summary.get("difficulty_table")),
        "backend_matrix": bool(summary.get("per_backend") or summary.get("backend_matrix") or summary.get("per_dataset_backend")),
        "embedding_artifacts": _has_recursive_key(state, ("embedding", "tsne", "t_sne", "umap")),
        "retrieval_examples": _has_recursive_key(state, ("retrieval_examples", "ranking_examples", "topk", "top_k", "cmc")),
    }


def _reference_keys_for(tags: list[str], refs: list[dict[str, Any]], limit: int = 4) -> tuple[list[str], list[str]]:
    keys: list[str] = []
    titles: list[str] = []
    tag_set = set(tags)
    for ref in refs:
        if tag_set and not tag_set.intersection(set(ref.get("chart_tags") or [])):
            continue
        key = str(ref.get("style_key") or "")
        if key and key not in keys:
            keys.append(key)
            titles.append(str(ref.get("title") or key))
        if len(keys) >= limit:
            break
    if not keys:
        for ref in refs[:limit]:
            key = str(ref.get("style_key") or "")
            if key and key not in keys:
                keys.append(key)
                titles.append(str(ref.get("title") or key))
    return keys, titles


def _plot_spec(
    *,
    figure_id: str,
    chart_type: str,
    title: str,
    objective: str,
    data_source: str,
    refs: list[dict[str, Any]],
    aspect_ratio: str,
    layout: str,
    placement: str = "double_column",
) -> dict[str, Any]:
    family = CHART_FAMILY_BY_TYPE.get(chart_type, "other_family")
    ref_keys, ref_titles = _reference_keys_for([chart_type, family], refs)
    return {
        "figure_id": figure_id,
        "plot_type": "plot",
        "role": "experiment_figure_pack",
        "chart_type": chart_type,
        "chart_family": family,
        "title": title,
        "objective": objective,
        "data_source": data_source,
        "aspect_ratio": aspect_ratio,
        "layout": layout,
        "placement": placement,
        "source_agent": "experiment_plot_reference_manager",
        "style_reference_keys": ref_keys,
        "style_reference_titles": ref_titles,
        "standard_version": EXPERIMENT_PLOT_REFERENCE_VERSION,
    }


def _build_plot_plan(metric_name: str, refs: list[dict[str, Any]], flags: dict[str, bool], domains: list[dict[str, Any]]) -> list[dict[str, Any]]:
    metric = str(metric_name or "metric")
    plan = [
        _plot_spec(
            figure_id="fig_main_results",
            chart_type="main_results_bar",
            title="Main Results",
            objective=f"Compare verified {metric}, token/cost, latency, and route-rate metrics across methods with seed uncertainty.",
            data_source="benchmark_summary.json:per_method",
            refs=refs,
            aspect_ratio="4:1",
            layout="1x4",
        ),
        _plot_spec(
            figure_id="fig_quality_cost_tradeoff",
            chart_type="quality_cost_tradeoff",
            title="Quality-Cost Tradeoff",
            objective=f"Plot {metric} against token/cost or latency to show the deployment tradeoff among baselines and the proposed method.",
            data_source="benchmark_summary.json:per_method.avg_new_tokens",
            refs=refs,
            aspect_ratio="4:3",
            layout="single",
            placement="single_column",
        ),
        _plot_spec(
            figure_id="fig_method_metric_heatmap",
            chart_type="method_metric_heatmap",
            title="Method-Metric Matrix",
            objective="Summarize the method-by-metric profile as a heatmap so accuracy, cost, latency, and routing behavior are not repeated as another bar chart.",
            data_source="benchmark_summary.json:per_method",
            refs=refs,
            aspect_ratio="4:3",
            layout="single",
            placement="single_column",
        ),
    ]
    if flags.get("backend_matrix"):
        plan = [
            _plot_spec(
                figure_id="fig_backend_grouped_bars",
                chart_type="backend_grouped_bars",
                title="Backend Results",
                objective=f"Compare {metric} across inference backends and methods with seed uncertainty.",
                data_source="benchmark_summary.json:backend_matrix",
                refs=refs,
                aspect_ratio="4:3",
                layout="single",
                placement="single_column",
            ),
            _plot_spec(
                figure_id="fig_backend_heatmap_single",
                chart_type="backend_heatmap_single",
                title="Backend Heatmap",
                objective=f"Show the method-by-backend {metric} matrix with seed standard deviation.",
                data_source="benchmark_summary.json:backend_matrix",
                refs=refs,
                aspect_ratio="1:1",
                layout="single",
                placement="single_column",
            ),
            _plot_spec(
                figure_id="fig_backend_rank_lines_1x4",
                chart_type="backend_rank_lines_1x4",
                title="Backend Rank Stability",
                objective="Track method ranks across inference backends and datasets with uncertainty bands.",
                data_source="benchmark_summary.json:per_dataset_backend",
                refs=refs,
                aspect_ratio="4:1",
                layout="1x4",
            ),
        ]
    domain_recommended = []
    for rule in domains:
        domain_recommended.extend(str(x) for x in rule.get("recommended_chart_types") or [])
    if any(x in domain_recommended for x in ("tsne_embedding", "umap_embedding")) and flags.get("embedding_artifacts"):
        plan.append(
            _plot_spec(
                figure_id="fig_embedding_tsne",
                chart_type="tsne_embedding",
                title="Embedding Visualization",
                objective="Visualize learned or selected representations with t-SNE/UMAP because related field papers commonly use embedding plots for this task.",
                data_source="embedding artifacts",
                refs=refs,
                aspect_ratio="4:3",
                layout="single",
                placement="single_column",
            )
        )
    return plan[: max(DEFAULT_MIN_EXPERIMENT_FIGURES, 3)]


def discover_experiment_plot_references_or_raise(
    outline: dict[str, Any],
    state: dict[str, Any],
    evidence_brief: dict[str, Any] | None,
    *,
    metric_name: str,
    api_key: str | None = None,
    min_figures: int = DEFAULT_MIN_EXPERIMENT_FIGURES,
    min_style_references: int = DEFAULT_MIN_STYLE_REFERENCES,
    min_distinct_families: int = DEFAULT_MIN_DISTINCT_FAMILIES,
    per_query_limit: int = 8,
) -> dict[str, Any]:
    text = _state_text(outline or {}, state or {}, evidence_brief or {})
    domains = _detect_domains(text)
    queries = _query_pool(domains, state or {}, metric_name)
    errors: list[str] = []
    papers: list[dict[str, Any]] = []
    for query in queries:
        try:
            papers.extend(search_papers(query, limit=per_query_limit, api_key=api_key))
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{query}: {exc}")
    style_refs = _dedupe_references([_style_reference_row(paper, domains) for paper in papers if isinstance(paper, dict)])
    flags = _evidence_flags(state or {})
    plan = _build_plot_plan(metric_name, style_refs, flags, domains)
    families = sorted({str(fig.get("chart_family") or CHART_FAMILY_BY_TYPE.get(str(fig.get("chart_type") or ""), "")) for fig in plan if fig.get("plot_type") == "plot"})
    families = [x for x in families if x]

    blockers: list[str] = []
    if len(style_refs) < min_style_references:
        blockers.append(f"Only {len(style_refs)}/{min_style_references} searched experiment-figure style references were found.")
    if len(plan) < min_figures:
        blockers.append(f"Only {len(plan)}/{min_figures} experiment figures were planned.")
    if len(families) < min_distinct_families:
        blockers.append(f"Only {len(families)}/{min_distinct_families} distinct experiment chart families were planned: {families}.")
    if not flags.get("per_method"):
        blockers.append("No benchmark_summary.per_method evidence is available for artifact-backed experiment figures.")
    for fig in plan[:min_figures]:
        if not fig.get("style_reference_keys"):
            blockers.append(f"{fig.get('figure_id')} has no searched style-reference keys.")

    report = {
        "schema_version": EXPERIMENT_PLOT_REFERENCE_VERSION,
        "status": "blocked" if blockers else "ok",
        "domains": [rule.get("domain") for rule in domains],
        "queries_used": queries,
        "search_errors": errors[:8],
        "style_reference_count": len(style_refs),
        "style_references": style_refs[:24],
        "evidence_flags": flags,
        "planned_experiment_figure_count": len(plan),
        "distinct_chart_families": families,
        "plotting_plan": plan,
        "domain_recommended_chart_types": sorted({x for rule in domains for x in (rule.get("recommended_chart_types") or [])}),
        "blockers": blockers,
    }
    if blockers:
        raise ExperimentPlotReferenceError(report)
    return report
