"""Reference-grounded experiment plot planning for PaperOrchestra.

This agent is intentionally separate from motivation/overview diagram generation.
It performs real literature search for field-specific experimental figure styles,
then emits an artifact-backed experiment plotting plan with required
main-results and ablation roles. Hyperparameter/sensitivity plots are added
only when verified sweep artifacts exist.
"""

from __future__ import annotations

from typing import Any

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


class ExperimentPlotReferenceError(RuntimeError):
    """Raised when experiment figures cannot be grounded in searched style references."""

    def __init__(self, report: dict[str, Any]):
        self.report = report
        blockers = report.get("blockers") or []
        detail = "; ".join(str(x) for x in blockers[:4]) if blockers else "unknown blocker"
        super().__init__(f"Experiment plot reference manager blocked plotting: {detail}")
