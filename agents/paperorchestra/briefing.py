"""Compact evidence briefs for PaperOrchestra manuscript generation.

The full DeepGraph manuscript state can be very large and highly repetitive.
This module builds a small, loss-aware brief that keeps the numbers, claims,
constraints, and citation seeds needed by writing agents without replaying the
entire database-shaped state into every LLM call.
"""

from __future__ import annotations

import json
from typing import Any

from agents.paperorchestra.figure_standard import backend_plot_pack, default_plot_plan


def _clip(value: Any, limit: int = 800) -> str:
    text = str(value or "").strip()
    if len(text) <= limit:
        return text
    return text[: limit - 3].rstrip() + "..."


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, str):
        try:
            loaded = json.loads(value)
        except json.JSONDecodeError:
            return [value] if value.strip() else []
        return loaded if isinstance(loaded, list) else [loaded]
    return [value]


def _safe_number(value: Any) -> float | int | str | None:
    if value is None or value == "":
        return None
    if isinstance(value, (int, float)):
        return value
    try:
        number = float(value)
    except (TypeError, ValueError):
        return _clip(value, 120)
    return int(number) if number.is_integer() else number


def _state_benchmark_summary(state: dict[str, Any]) -> dict[str, Any]:
    candidates: list[Any] = [
        state.get("benchmark_summary"),
        state.get("result_packet"),
        (state.get("result_packet") or {}).get("benchmark_summary")
        if isinstance(state.get("result_packet"), dict)
        else None,
    ]
    for claim in state.get("claims") or []:
        if isinstance(claim, dict):
            supporting = claim.get("supporting_data")
            candidates.append(supporting)
            if isinstance(supporting, dict):
                candidates.append(supporting.get("result_packet"))
                candidates.append(supporting.get("benchmark_summary"))
    for item in candidates:
        if not isinstance(item, dict):
            continue
        summary = item.get("benchmark_summary") if isinstance(item.get("benchmark_summary"), dict) else item
        if isinstance(summary, dict) and (
            isinstance(summary.get("per_method"), dict)
            or isinstance(summary.get("per_method_backend"), dict)
            or isinstance(summary.get("per_dataset_backend"), dict)
            or isinstance(summary.get("by_backend"), dict)
            or isinstance(summary.get("backend_matrix"), dict)
            or isinstance(summary.get("method_backend_scores"), dict)
            or summary.get("datasets")
            or summary.get("latency_tokens_table")
        ):
            return summary
    return {}


def _compact_method_row(name: str, row: Any) -> dict[str, Any]:
    if not isinstance(row, dict):
        return {"method": name, "value": _safe_number(row)}
    keep = (
        "metric_value",
        "score",
        "accuracy",
        "utility",
        "avg_new_tokens",
        "avg_latency_seconds",
        "route_rate",
        "count",
        "std",
        "ci95",
        "p_value",
    )
    out = {"method": name}
    for key in keep:
        if key in row:
            out[key] = row.get(key)
    for key in ("budget_histogram", "difficulty_breakdown", "failure_modes"):
        if key in row:
            out[key] = row.get(key)
    return out


def _compact_claims(state: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for idx, claim in enumerate(state.get("claims") or [], start=1):
        if not isinstance(claim, dict):
            continue
        rows.append(
            {
                "id": str(claim.get("id") or claim.get("claim_id") or idx),
                "claim_text": _clip(claim.get("claim_text") or claim.get("text"), 360),
                "verdict": claim.get("verdict"),
                "allowed_sections": claim.get("allowed_sections") or claim.get("paper_sections"),
            }
        )
    if rows:
        return rows[:12]
    matrix = state.get("claim_evidence_matrix") or []
    for idx, row in enumerate(matrix[:12], start=1):
        if not isinstance(row, dict):
            continue
        rows.append(
            {
                "id": str(row.get("claim_id") or row.get("id") or idx),
                "claim_text": _clip(row.get("claim_text") or row.get("claim"), 360),
                "evidence": _clip(row.get("evidence") or row.get("support"), 300),
                "allowed_sections": row.get("allowed_sections"),
            }
        )
    return rows


def _compact_dataset_rows(summary: dict[str, Any]) -> list[dict[str, Any]]:
    datasets = summary.get("datasets")
    if not datasets and isinstance(summary.get("dataset"), dict):
        datasets = [summary.get("dataset")]
    out: list[dict[str, Any]] = []
    for item in _as_list(datasets)[:8]:
        if not isinstance(item, dict):
            out.append({"name": _clip(item, 160)})
            continue
        out.append(
            {
                "name": item.get("name") or item.get("dataset"),
                "split": item.get("split"),
                "num_test": item.get("num_test") or item.get("count") or item.get("n"),
                "source": item.get("license_or_source") or item.get("source"),
                "preprocessing": _clip(item.get("preprocessing"), 180),
            }
        )
    return [row for row in out if any(v not in (None, "", []) for v in row.values())]


def _has_backend_matrix(summary: dict[str, Any]) -> bool:
    return any(
        isinstance(summary.get(key), dict) and summary.get(key)
        for key in (
            "per_method_backend",
            "per_dataset_backend",
            "by_backend",
            "backend_matrix",
            "method_backend_scores",
        )
    )


def _backend_plot_pack(metric_name: str) -> list[dict[str, Any]]:
    return backend_plot_pack(metric_name)


def _default_plot_pack(metric_name: str) -> list[dict[str, Any]]:
    return default_plot_plan(metric_name)


def build_evidence_brief(
    state: dict[str, Any],
    literature_block: str,
    iterations: list[dict[str, Any]],
    *,
    paper_ids: list[str] | None = None,
    baseline: float | None = None,
    metric_name: str | None = None,
) -> dict[str, Any]:
    """Return a compact, serializable manuscript brief."""
    summary = _state_benchmark_summary(state)
    per_method_raw = summary.get("per_method") if isinstance(summary.get("per_method"), dict) else {}
    per_method = [_compact_method_row(str(name), row) for name, row in per_method_raw.items()]
    paper_intent = state.get("paper_intent") if isinstance(state.get("paper_intent"), dict) else {}
    problem_awareness = state.get("problem_awareness") if isinstance(state.get("problem_awareness"), dict) else {}
    evidence_manifest = state.get("evidence_manifest") if isinstance(state.get("evidence_manifest"), dict) else {}
    reviewer_report = state.get("reviewer_report") if isinstance(state.get("reviewer_report"), dict) else {}
    result_packet = state.get("result_packet") if isinstance(state.get("result_packet"), dict) else {}
    model = summary.get("model") if isinstance(summary.get("model"), dict) else {}

    seed_results = summary.get("seed_results") if isinstance(summary.get("seed_results"), list) else []
    brief = {
        "schema_version": "paperorchestra_evidence_brief_v1",
        "title": state.get("title"),
        "problem": {
            "statement": _clip(state.get("problem_statement"), 900),
            "central_question": _clip(problem_awareness.get("central_question"), 500),
            "motivation": _clip(problem_awareness.get("motivation") or state.get("existing_weakness"), 900),
            "method_answer": _clip(problem_awareness.get("method_answer") or state.get("method_summary"), 900),
            "result_claim": _clip(problem_awareness.get("result_claim"), 600),
            "limitation": _clip(problem_awareness.get("limitation") or state.get("limitations"), 700),
        },
        "method": {
            "name": state.get("method_name"),
            "summary": _clip(state.get("method_summary"), 1000),
            "contributions": [_clip(x, 260) for x in _as_list(state.get("contributions"))[:6]],
            "training_free": bool(
                state.get("training_free")
                or "training-free" in json.dumps(state, ensure_ascii=False).lower()
                or "no training" in json.dumps(state, ensure_ascii=False).lower()
            ),
            "constraints": {
                "no_gpu_training": True,
                "cpu_only_allowed": True,
                "evidence_boundary": _clip(
                    (state.get("publication_evidence_contract") or {}).get("evidence_tier")
                    if isinstance(state.get("publication_evidence_contract"), dict)
                    else result_packet.get("evidence_tier"),
                    160,
                ),
            },
        },
        "experiment": {
            "primary_metric": metric_name or summary.get("primary_metric") or summary.get("metric_name") or state.get("baseline_metric_name"),
            "baseline": baseline if baseline is not None else state.get("baseline_metric_value") or result_packet.get("baseline"),
            "best": state.get("best_metric_value") or result_packet.get("best"),
            "effect_pct": state.get("effect_pct") or result_packet.get("effect_pct"),
            "verdict": state.get("verdict") or result_packet.get("verdict"),
            "datasets": _compact_dataset_rows(summary),
            "model": {
                "id": model.get("id") or model.get("name") or summary.get("model_name"),
                "hardware": model.get("hardware") or summary.get("hardware") or evidence_manifest.get("hardware"),
                "backend": model.get("backend"),
                "decoding": model.get("decoding"),
            },
            "num_seeds": summary.get("num_seeds") or len(seed_results) or result_packet.get("minimum_seeds"),
            "per_method": per_method[:16],
            "latency_tokens_table": _as_list(summary.get("latency_tokens_table"))[:16],
            "token_cost": summary.get("token_cost"),
            "latency": summary.get("latency"),
            "ablation_table": _as_list(summary.get("ablation_table"))[:16],
            "per_method_backend": summary.get("per_method_backend")
            or summary.get("by_backend")
            or summary.get("backend_matrix")
            or summary.get("method_backend_scores"),
            "per_dataset_backend": summary.get("per_dataset_backend"),
            "backends": summary.get("backends"),
            "methods": summary.get("methods"),
            "statistical_tests": summary.get("bootstrap_ci")
            or {
                "p_value": result_packet.get("p_value"),
                "statistical_test": (state.get("publication_evidence_contract") or {}).get("statistical_test")
                if isinstance(state.get("publication_evidence_contract"), dict)
                else None,
            },
            "iterations": [
                {
                    "iteration": it.get("iteration_number"),
                    "phase": it.get("phase"),
                    "metric_value": _safe_number(it.get("metric_value")),
                    "status": it.get("status"),
                    "description": _clip(it.get("description"), 220),
                }
                for it in iterations[:12]
                if isinstance(it, dict)
            ],
        },
        "claims": _compact_claims(state),
        "gate": {
            "paper_generation_allowed": True,
            "quality_gates": state.get("quality_gates") or {},
            "required_evidence": state.get("required_evidence") or {},
            "reviewer_status": reviewer_report.get("status"),
            "reviewer_blockers": reviewer_report.get("blockers") or reviewer_report.get("major_concerns") or [],
        },
        "literature": {
            "seed_paper_ids": [str(x) for x in (paper_ids or state.get("citation_seed_paper_ids") or [])[:16]],
            "positioning": _clip(literature_block or state.get("related_work_positioning") or state.get("evidence_summary"), 1600),
        },
        "intent": {
            "target_venue": paper_intent.get("target_venue") or "ICLR-style conference submission",
            "audience": _clip(paper_intent.get("audience"), 300),
            "main_message": _clip(paper_intent.get("main_message") or paper_intent.get("thesis"), 600),
        },
    }
    return brief


def evidence_brief_markdown(brief: dict[str, Any], *, max_chars: int = 18000) -> str:
    """Markdown rendering designed for LLM prompts."""
    exp = brief.get("experiment") or {}
    problem = brief.get("problem") or {}
    method = brief.get("method") or {}
    lines = [
        f"# Evidence Brief: {brief.get('title') or 'Untitled'}",
        "",
        "## Problem",
        f"- Question: {problem.get('central_question') or problem.get('statement')}",
        f"- Motivation: {problem.get('motivation')}",
        f"- Limitation to state: {problem.get('limitation')}",
        "",
        "## Method",
        f"- Name: {method.get('name')}",
        f"- Summary: {method.get('summary')}",
        f"- Training-free / no GPU training: {method.get('training_free')}",
        "- Contributions:",
    ]
    lines.extend(f"  - {item}" for item in method.get("contributions") or [])
    lines.extend(
        [
            "",
            "## Experiment",
            f"- Primary metric: {exp.get('primary_metric')}",
            f"- Baseline: {exp.get('baseline')}",
            f"- Best: {exp.get('best')}",
            f"- Effect percent: {exp.get('effect_pct')}",
            f"- Verdict: {exp.get('verdict')}",
            f"- Seeds: {exp.get('num_seeds')}",
            f"- Datasets: {json.dumps(exp.get('datasets') or [], ensure_ascii=False)}",
            f"- Model/hardware: {json.dumps(exp.get('model') or {}, ensure_ascii=False)}",
            "",
            "### Main Results",
        ]
    )
    for row in exp.get("per_method") or []:
        lines.append(f"- {json.dumps(row, ensure_ascii=False)}")
    lines.extend(["", "### Ablations"])
    for row in exp.get("ablation_table") or []:
        lines.append(f"- {json.dumps(row, ensure_ascii=False)}")
    lines.extend(["", "### Latency and Cost"])
    for row in exp.get("latency_tokens_table") or []:
        lines.append(f"- {json.dumps(row, ensure_ascii=False)}")
    lines.extend(
        [
            f"- Token cost summary: {json.dumps(exp.get('token_cost') or {}, ensure_ascii=False)[:1000]}",
            f"- Statistical tests: {json.dumps(exp.get('statistical_tests') or {}, ensure_ascii=False)}",
            "",
            "## Claims",
        ]
    )
    for row in brief.get("claims") or []:
        lines.append(f"- {json.dumps(row, ensure_ascii=False)}")
    lit = brief.get("literature") or {}
    lines.extend(
        [
            "",
            "## Literature Seeds",
            f"- Seed paper ids: {', '.join(lit.get('seed_paper_ids') or [])}",
            f"- Positioning: {lit.get('positioning')}",
            "",
            "## Gate",
            json.dumps(brief.get("gate") or {}, ensure_ascii=False)[:2200],
        ]
    )
    text = "\n".join(lines)
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 160].rstrip() + "\n\n[Evidence brief truncated to prompt budget; full artifacts remain in bundle.]"


def build_deterministic_outline(
    state: dict[str, Any],
    brief: dict[str, Any],
    *,
    metric_name: str,
) -> dict[str, Any]:
    """Build the PaperOrchestra outline locally instead of using one giant JSON call."""
    method = brief.get("method") or {}
    experiment = brief.get("experiment") or {}
    method_name = method.get("name") or state.get("method_name") or "the proposed method"
    title = state.get("title") or brief.get("title") or f"{method_name}: Training-Free Multi-Agent Reasoning"
    plotting_plan = _backend_plot_pack(metric_name) if _has_backend_matrix(experiment) else _default_plot_pack(metric_name)
    core_query = " ".join(
        str(x or "")
        for x in [
            method_name,
            "training-free multi-agent LLM reasoning",
            "test-time compute",
            "self-consistency debate consensus verification",
        ]
    ).strip()
    return {
        "schema_version": "paperorchestra_deterministic_outline_v1",
        "title": title,
        "thesis": (brief.get("intent") or {}).get("main_message")
        or (brief.get("problem") or {}).get("method_answer")
        or method.get("summary"),
        "intro_related_work_plan": {
            "introduction_strategy": {
                "problem_first_sentence": (brief.get("problem") or {}).get("central_question")
                or (brief.get("problem") or {}).get("statement"),
                "motivation": (brief.get("problem") or {}).get("motivation"),
                "contribution_order": method.get("contributions") or [],
                "search_directions": [
                    core_query,
                    "training-free LLM test-time reasoning multi-agent debate",
                    "self-consistency verifier consensus large language models",
                    "adaptive test-time compute allocation reasoning",
                ],
            },
            "related_work_strategy": {
                "subsections": [
                    {
                        "title": "Training-free test-time reasoning",
                        "methodology_cluster": "self-consistency, tree search, deliberation, and adaptive compute allocation",
                        "limitation_search_queries": [
                            "self-consistency large language models reasoning",
                            "test-time compute allocation large language models reasoning",
                        ],
                    },
                    {
                        "title": "Multi-agent deliberation and verification",
                        "methodology_cluster": "multi-agent debate, consensus, verifier-guided selection, answer diversity",
                        "limitation_search_queries": [
                            "multi-agent debate large language models reasoning",
                            "LLM agent consensus verification reasoning",
                        ],
                    },
                    {
                        "title": "Budget-aware reliability",
                        "methodology_cluster": "latency, token cost, calibration, selective reasoning, abstention",
                        "limitation_search_queries": [
                            "selective reasoning budget latency token cost LLM",
                            "calibration confidence verification large language model reasoning",
                        ],
                    },
                ]
            },
        },
        "plotting_plan": plotting_plan,
        "section_plan": [
            {
                "section_title": "Introduction",
                "purpose": "State the problem, the training-free constraint, the proposed method, and the verified result.",
                "subsections": [
                    {"title": "Problem and motivation", "citation_hints": ["test-time reasoning and multi-agent LLM reliability"]},
                    {"title": "Contributions", "citation_hints": ["training-free multi-agent deliberation"]},
                ],
            },
            {
                "section_title": "Method",
                "purpose": "Define the training-free multi-agent procedure and all inference-time decisions.",
                "subsections": [
                    {"title": "Agent candidate generation", "citation_hints": ["self-consistency reasoning"]},
                    {"title": "Consensus and verification", "citation_hints": ["multi-agent debate verification"]},
                    {"title": "Budget policy and complexity", "citation_hints": ["test-time compute allocation"]},
                ],
            },
            {
                "section_title": "Experiments",
                "purpose": "Report datasets, baselines, metrics, seeds, ablations, statistical tests, latency, and token cost.",
                "subsections": [
                    {"title": "Setup", "citation_hints": [str(x.get("name")) for x in experiment.get("datasets") or [] if isinstance(x, dict)]},
                    {"title": "Main results", "citation_hints": ["baseline fairness and multi-seed evaluation"]},
                    {"title": "Ablations and cost", "citation_hints": ["latency token cost ablation"]},
                ],
            },
            {
                "section_title": "Discussion_Conclusion",
                "purpose": "Interpret the evidence, state limitations, and avoid unsupported claims.",
                "subsections": [
                    {"title": "Limitations", "citation_hints": ["training-free reasoning limitations"]},
                    {"title": "Conclusion", "citation_hints": []},
                ],
            },
        ],
    }
