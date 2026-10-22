"""Reference-grounded experiment plot planning for PaperOrchestra.

This agent is intentionally separate from motivation/overview diagram generation.
It performs real literature search for field-specific experimental figure styles,
then emits an artifact-backed experiment plotting plan with required
main-results and ablation roles. Hyperparameter/sensitivity plots are added
only when verified sweep artifacts exist.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any



EXPERIMENT_PLOT_REFERENCE_VERSION = "deepgraph_experiment_plot_reference_v4_2026_06_13"
DEFAULT_MIN_EXPERIMENT_FIGURES = 3
DEFAULT_MIN_STYLE_REFERENCES = 3
DEFAULT_MIN_DISTINCT_FAMILIES = 2
LOCAL_EXPERIMENT_STYLE_REFERENCE_DIR = Path(__file__).resolve().parents[2] / "实验图例子"
REQUIRED_STYLE_REFERENCE_SOURCES = ["local_user_examples", "related_literature_search"]

CHART_FAMILY_BY_TYPE = {
    "main_results_bar": "bar_family",
    "main_results_bar_1x2": "bar_family",
    "backend_grouped_bars": "bar_family",
    "quality_cost_tradeoff": "distribution_family",
    "method_metric_heatmap": "matrix_family",
    "backend_heatmap_single": "matrix_family",
    "backend_rank_lines_1x4": "line_family",
    "ablation_bar": "bar_family",
    "ablation_results": "bar_family",
    "dataset_breakdown": "matrix_family",
    "hyperparameter_sweep": "line_family",
    "threshold_sweep": "line_family",
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
        "recommended_chart_types": ["main_results_bar", "ablation_bar", "hyperparameter_sweep", "tsne_embedding", "cmc_curve", "ranking_examples"],
    },
    {
        "domain": "vision_classification_or_detection",
        "needles": ["image", "vision", "classification", "detection", "segmentation", "clip", "imagenet", "cifar"],
        "queries": [
            "computer vision experimental results t-SNE ablation heatmap robustness figure",
            "image classification paper experimental figures ablation heatmap calibration",
        ],
        "recommended_chart_types": ["main_results_bar", "ablation_bar", "hyperparameter_sweep", "tsne_embedding"],
    },
    {
        "domain": "graph_learning",
        "needles": ["graph", "node", "edge", "gnn", "network"],
        "queries": [
            "graph neural network experimental results ablation heatmap t-SNE visualization",
            "graph learning paper experimental figure node classification ablation robustness",
        ],
        "recommended_chart_types": ["main_results_bar", "ablation_bar", "hyperparameter_sweep", "tsne_embedding"],
    },
    {
        "domain": "llm_reasoning",
        "needles": ["llm", "large language model", "reasoning", "multi-agent", "self-consistency", "debate", "test-time", "inference-time"],
        "queries": [
            "large language model reasoning experiments accuracy cost ablation figure",
            "multi-agent LLM debate reasoning experiments cost accuracy ablation visualization",
            "test-time compute language models accuracy token cost frontier experiment figure",
        ],
        "recommended_chart_types": ["main_results_bar", "ablation_bar", "hyperparameter_sweep", "backend_rank_lines_1x4"],
    },
    {
        "domain": "retrieval_or_rag",
        "needles": ["retrieval", "rag", "ranking", "recall", "mrr", "ndcg"],
        "queries": [
            "retrieval augmented generation experimental results recall nDCG ablation heatmap figure",
            "information retrieval paper experimental figure ranking examples ablation",
        ],
        "recommended_chart_types": ["main_results_bar", "ablation_bar", "hyperparameter_sweep", "ranking_examples"],
    },
]

GENERIC_STYLE_QUERIES = [
    "machine learning paper experimental results main results ablation sensitivity figure",
    "neural network paper experiment figures grouped bar ablation hyperparameter sensitivity",
    "benchmark paper experimental plots grouped bar ablation threshold sensitivity",
]


class ExperimentPlotReferenceError(RuntimeError):
    """Raised when experiment figures cannot be grounded in searched style references."""

    def __init__(self, report: dict[str, Any]):
        self.report = report
        blockers = report.get("blockers") or []
        detail = "; ".join(str(x) for x in blockers[:4]) if blockers else "unknown blocker"
        super().__init__(f"Experiment plot reference manager blocked plotting: {detail}")
