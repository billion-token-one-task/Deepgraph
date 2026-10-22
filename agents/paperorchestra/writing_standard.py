"""Binding manuscript-writing standards for PaperOrchestra outputs."""

from __future__ import annotations


def build_paper_contract(state: dict, venue_target: dict | None = None) -> dict:
    """Construct a compact binding paper contract from manuscript state."""
    paper_intent = state.get("paper_intent") if isinstance(state.get("paper_intent"), dict) else {}
    publication_contract = (
        state.get("publication_evidence_contract")
        if isinstance(state.get("publication_evidence_contract"), dict)
        else {}
    )
    result_packet = state.get("result_packet") if isinstance(state.get("result_packet"), dict) else {}
    benchmark_summary = result_packet.get("benchmark_summary") if isinstance(result_packet.get("benchmark_summary"), dict) else {}
    method_name = state.get("method_name") or benchmark_summary.get("candidate_method") or state.get("title")
    datasets = benchmark_summary.get("datasets") or []
    baselines = list((benchmark_summary.get("per_method") or {}).keys()) if isinstance(benchmark_summary.get("per_method"), dict) else []
    metric = benchmark_summary.get("primary_metric") or benchmark_summary.get("metric_name") or state.get("baseline_metric_name")
    evidence_scope = (
        publication_contract.get("evidence_tier")
        or result_packet.get("evidence_tier")
        or ("full benchmark" if result_packet.get("full_benchmark_completed") else "controlled setting")
    )
    glossary = {
        "method_name": method_name,
        "datasets": [row.get("name") if isinstance(row, dict) else str(row) for row in datasets],
        "baselines": baselines,
        "primary_metric": metric,
    }
    return {
        "schema_version": "deepgraph_paper_contract_v1",
        "paper_type": paper_intent.get("paper_type") or publication_contract.get("paper_type") or "method paper",
        "contribution_type": paper_intent.get("contribution_type") or publication_contract.get("contribution_type") or "new method",
        "target": venue_target or {},
        "evidence_scope": evidence_scope,
        "supported_claims": state.get("claim_evidence_matrix") or state.get("claims") or [],
        "primary_metric": metric,
        "glossary": glossary,
        "banned_expressions": [
            "training-free",
            "This is the first",
            "Comprehensive experiments show",
            "Extensive experiments demonstrate",
            "Universal",
            "General",
            "state-of-the-art",
            "SOTA",
        ],
        "consistency_requirements": [
            "Do not expand evidence scope in abstract or conclusion.",
            "Do not write unfinished or placeholder numbers as completed results.",
            "Use the same names for methods, datasets, baselines, metrics, and protocol across sections.",
        ],
    }
