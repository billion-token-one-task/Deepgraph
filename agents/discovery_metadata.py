"""Helpers for mechanism-first discovery metadata and resource triage."""

from __future__ import annotations

import json

from agents.evidence_planner import build_evidence_plan
from contracts import DeepInsightSpec, normalize_deep_insight_storage


GPU_HINT_KEYWORDS = {
    "llm", "gpt", "llama", "mistral", "multimodal", "vision-language",
    "diffusion", "7b", "13b", "70b", "pretrain", "gpu",
}


def _load_json(value: str | dict | None, default):
    if value is None:
        return default
    if isinstance(value, (dict, list)):
        return value
    try:
        return json.loads(value)
    except (json.JSONDecodeError, TypeError):
        return default


def _parse_gpu_hours(plan: dict) -> float | None:
    compute = plan.get("compute_budget", {}) if isinstance(plan, dict) else {}
    raw = (
        compute.get("total_gpu_hours")
        or compute.get("gpu_hours")
        or compute.get("gpu")
    )
    if raw is None:
        return None
    if isinstance(raw, (int, float)):
        return float(raw)
    text = str(raw).strip().lower()
    digits = []
    for ch in text:
        if ch.isdigit() or ch == ".":
            digits.append(ch)
        elif digits:
            break
    if not digits:
        return None
    try:
        return float("".join(digits))
    except ValueError:
        return None


def _coerce_spec(insight: dict | DeepInsightSpec) -> DeepInsightSpec:
    if isinstance(insight, DeepInsightSpec):
        return insight
    return DeepInsightSpec.from_raw(insight)


def infer_resource_class(insight: dict | DeepInsightSpec) -> str:
    spec = _coerce_spec(insight)
    plan = spec.experimental_plan
    method = spec.proposed_method
    gpu_hours = _parse_gpu_hours(plan)
    corpus = " ".join(
        str(part or "")
        for part in [
            spec.title,
            spec.problem_statement,
            spec.existing_weakness,
            method.get("name"),
            method.get("type"),
            method.get("one_line"),
            method.get("definition"),
            json.dumps(plan.get("datasets", [])),
            json.dumps(plan.get("baselines", [])),
        ]
    ).lower()

    if gpu_hours is not None and gpu_hours >= 40:
        return "gpu_large"
    if gpu_hours is not None and gpu_hours > 0.5:
        return "gpu_small"
    if any(keyword in corpus for keyword in GPU_HINT_KEYWORDS):
        return "gpu_small"
    return "cpu"


def infer_experimentability(insight: dict) -> str:
    spec = _coerce_spec(insight)
    resource_class = spec.resource_class or infer_resource_class(spec)
    if resource_class == "cpu":
        return "easy"
    if resource_class == "gpu_small":
        return "medium"
    return "hard"


def _sentences(text: str | None) -> list[str]:
    if not text:
        return []
    parts = [part.strip(" -") for part in str(text).replace("\n", " ").split(".")]
    return [part for part in parts if part]


def build_evidence_packet(
    *,
    signal_mix: list[str],
    evidence_summary: str | None,
    falsification: str | dict | None,
    structural_evidence: list[str] | None = None,
    non_numeric_evidence: list[str] | None = None,
) -> dict:
    nn = [item.strip() for item in (non_numeric_evidence or []) if item and item.strip()]
    if len(nn) < 2:
        nn.extend(_sentences(evidence_summary)[: max(0, 2 - len(nn))])
    while len(nn) < 2:
        nn.append("Mechanism-oriented supporting evidence requires follow-up review.")
    structural = [item.strip() for item in (structural_evidence or []) if item and item.strip()]
    if not structural and evidence_summary:
        structural = _sentences(evidence_summary)[:1]

    if isinstance(falsification, str):
        falsification_obj = _load_json(falsification, {"summary": falsification})
    else:
        falsification_obj = falsification or {}

    packet = {
        "signal_mix": sorted({item for item in signal_mix if item}),
        "non_numeric_evidence": nn[:4],
        "structural_evidence": structural[:3],
        "falsification": falsification_obj,
    }
    return packet


def build_deep_insight_spec(insight: dict | DeepInsightSpec) -> DeepInsightSpec:
    spec = _coerce_spec(insight)
    signal_mix = list(spec.signal_mix)
    if not signal_mix:
        signal_mix = []
        if spec.mechanism_type:
            signal_mix.append(spec.mechanism_type)
        if spec.tier == 1:
            signal_mix.append("entity_overlap")
        else:
            signal_mix.append("paper_idea")
    spec.signal_mix = sorted({item for item in signal_mix if item})
    spec.resource_class = spec.resource_class or infer_resource_class(spec)
    spec.experimentability = spec.experimentability or infer_experimentability(spec)
    spec.submission_status = spec.submission_status or "not_started"
    if not spec.evidence_packet:
        spec.evidence_packet = build_evidence_packet(
            signal_mix=spec.signal_mix,
            evidence_summary=spec.evidence_summary,
            falsification=spec.falsification,
        )
    evidence_plan = _load_json(spec.evidence_plan, None)
    if not isinstance(evidence_plan, dict):
        evidence_plan = build_evidence_plan(normalize_deep_insight_storage(spec))
    spec.evidence_plan = evidence_plan
    return spec


def enrich_deep_insight(insight: dict | DeepInsightSpec) -> dict:
    spec = build_deep_insight_spec(insight)
    return normalize_deep_insight_storage(spec)
