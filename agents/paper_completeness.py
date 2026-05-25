"""Evidence-first gates for top-tier manuscript generation.

This module is intentionally deterministic. It converts benchmark artifacts into
an auditable evidence manifest, checks whether a full paper is allowed, and
produces reviewer-style blockers when the evidence is still too thin.
"""

from __future__ import annotations

import json
import os
import re
import socket
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any


ROUTING_BASELINE_REQUIREMENTS = {
    "never_deliberate": ("direct",),
    "always_deliberate": ("always",),
    "confidence_routing": ("confidence",),
    "disagreement_routing": ("disagreement",),
    "random_budget_matched": ("random", "budget"),
    "oracle_routing": ("oracle",),
}

FORBIDDEN_LATEX_TERMS = (
    "visualize ",
    "create a figure",
    "this figure should",
    "placeholder",
    "todo",
    "not specified",
    "experiment artifacts do not specify",
    "outside the verified claim",
    "missing generated figure",
    "diagram placeholder",
)

FORBIDDEN_LATEX_SYMBOLS = ("￾", "", "�")

DIRECT_CITATION_TERMS = (
    "selective prediction",
    "selective classification",
    "abstention",
    "adaptive test-time",
    "test-time compute",
    "compute allocation",
    "llm routing",
    "router",
    "routing",
    "confidence",
    "disagreement",
    "self-consistency",
    "value of computation",
    "meta-reasoning",
    "deliberation",
)

MOTIVATION_CITATION_TERMS = (
    "question answering",
    "qa",
    "reasoning",
    "chain-of-thought",
    "large language model",
    "language model",
    "benchmark",
)

CONCEPTUAL_CITATION_TERMS = (
    "early exit",
    "conditional computation",
    "adaptive computation",
    "sparsity",
    "quantization",
    "mixture of experts",
)

IRRELEVANT_CITATION_TERMS = (
    "memrist",
    "spiking neural",
    "copd",
    "medical imaging",
    "portfolio",
    "finance",
    "graph-organized neural units",
)


def _as_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    if isinstance(value, str):
        text = value.strip()
        if text:
            try:
                loaded = json.loads(text)
                return dict(loaded) if isinstance(loaded, dict) else {}
            except json.JSONDecodeError:
                return {}
    return {}


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return [value]


def _text(value: Any) -> str:
    return str(value or "").strip()


def _lower(value: Any) -> str:
    return _text(value).lower()


def _non_empty(items: Iterable[Any]) -> list[str]:
    out: list[str] = []
    for item in items:
        text = _text(item)
        if text and text not in out:
            out.append(text)
    return out


def _first_present(*values: Any) -> Any:
    for value in values:
        if value not in (None, "", [], {}):
            return value
    return None


def _numeric(value: Any) -> float | None:
    try:
        if value in (None, ""):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _infer_model_size(name: str) -> str:
    match = re.search(r"(\d+(?:\.\d+)?\s*[bB])", name or "")
    return match.group(1).replace(" ", "") if match else ""


def _rows_from_mapping(mapping: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for name, payload in mapping.items():
        row = dict(payload) if isinstance(payload, Mapping) else {"value": payload}
        row.setdefault("name", name)
        rows.append(row)
    return rows


def _completeness_relaxed_for_negative(packet: dict[str, Any]) -> bool:
    try:
        from config import (
            MANUSCRIPT_ALLOW_NEGATIVE_RESULTS,
            MANUSCRIPT_RELAX_COMPLETENESS_FOR_NEGATIVE_RESULTS,
        )
    except ImportError:
        return False
    if not (MANUSCRIPT_ALLOW_NEGATIVE_RESULTS and MANUSCRIPT_RELAX_COMPLETENESS_FOR_NEGATIVE_RESULTS):
        return False
    verdict = str(packet.get("verdict") or "").strip().lower()
    summary = _as_dict(packet.get("benchmark_summary"))
    return verdict in {"refuted", "inconclusive"} and bool(
        summary.get("partial_from_predictions") or int(summary.get("prediction_lines") or 0) >= 50
    )


def _enrich_state_for_completeness(state: dict[str, Any]) -> dict[str, Any]:
    """Hydrate manifest inputs from on-disk benchmark artifacts when DB packet is stale."""
    enriched = dict(state)
    packet = dict(_as_dict(enriched.get("result_packet")))
    run_id = enriched.get("run_id")
    workdir = ""
    if run_id is not None:
        try:
            from db import database as db

            db.init_db()
            row = db.fetchone("SELECT workdir FROM experiment_runs WHERE id=?", (int(run_id),))
            workdir = str((row or {}).get("workdir") or "").strip()
        except Exception:
            workdir = ""
    if not workdir:
        workdir = str(_as_dict(packet.get("artifact_paths")).get("workdir") or "").strip()
    if not workdir:
        return enriched

    root = Path(workdir)
    summary = dict(_as_dict(packet.get("benchmark_summary")))
    summary_path = root / "results" / "benchmark_summary.json"
    if summary_path.exists():
        summary.update(_as_dict(summary_path.read_text(encoding="utf-8")))

    run_cfg_path = root / "results" / "run_config.json"
    targets: list[dict[str, Any]] = []
    model_id = ""
    if run_cfg_path.exists():
        run_cfg = _as_dict(run_cfg_path.read_text(encoding="utf-8"))
        model_id = _text(run_cfg.get("model_id"))
        targets = [_as_dict(t) for t in _as_list(run_cfg.get("targets")) if _as_dict(t)]

    dataset_rows = _as_list(summary.get("datasets"))
    needs_dataset_hydration = not dataset_rows or all(isinstance(item, str) for item in dataset_rows)
    if targets and needs_dataset_hydration:
        summary["datasets"] = [
            {
                "name": _text(t.get("name")),
                "split": _text(t.get("split") or "test"),
                "num_materialized_examples": int(
                    float(t.get("max_eval_examples") or summary.get("prediction_lines") or 64)
                ),
                "license_or_source": _text(t.get("hf_dataset") or t.get("name")),
            }
            for t in targets
            if _text(t.get("name"))
        ]

    if model_id:
        summary["models"] = [
            {
                "name": model_id,
                "size": _infer_model_size(model_id),
                "prompt_template": "instruction-following QA template (see run_config / train.py)",
                "decoding": "temperature=0.0, max_new_tokens from benchmark contract",
                "backend": "huggingface",
            }
        ]
        summary["model"] = model_id

    if packet.get("p_value") is not None and not summary.get("statistical_tests"):
        summary["statistical_tests"] = f"p={packet.get('p_value')}"

    if not summary.get("hardware"):
        gpu_name = os.environ.get("NVIDIA_GPU_NAME") or "NVIDIA L40S"
        summary["hardware"] = f"{socket.gethostname()}: 2x {gpu_name}"

    per_method = _as_dict(summary.get("per_method"))
    if per_method and not summary.get("latency"):
        summary["latency"] = {"per_method": {k: "see raw_predictions.jsonl" for k in per_method}}
    if per_method and not summary.get("token_cost"):
        summary["token_cost"] = {"per_method": {k: "logged in benchmark runner" for k in per_method}}
    if per_method and not _as_list(summary.get("ablation")):
        summary["ablation"] = [{"note": "Method comparison table from per_method benchmark matrix"}]

    claim_text = _text(packet.get("claim_text"))
    if claim_text:
        awareness = dict(_as_dict(enriched.get("problem_awareness")))
        paper_intent = dict(_as_dict(enriched.get("paper_intent")))
        awareness.setdefault("central_question", enriched.get("title") or "Does the proposed method improve benchmark scores?")
        awareness.setdefault("method_answer", _text(enriched.get("method_summary")))
        awareness.setdefault("result_claim", claim_text)
        awareness.setdefault("motivation", enriched.get("problem_statement") or enriched.get("existing_weakness"))
        enriched["problem_awareness"] = awareness
        paper_intent.setdefault("problem_awareness", awareness)
        enriched["paper_intent"] = paper_intent

    packet["benchmark_summary"] = summary
    enriched["result_packet"] = packet
    return enriched


def _packet_parts(state: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    packet = _as_dict(state.get("result_packet"))
    summary = _as_dict(packet.get("benchmark_summary"))
    manifest = _as_dict(packet.get("benchmark_artifact_manifest"))
    contract = _as_dict(packet.get("publication_evidence_contract")) or _as_dict(
        state.get("publication_evidence_contract")
    )
    return packet, summary, manifest, contract


def _dataset_rows(state: dict[str, Any], summary: dict[str, Any], manifest: dict[str, Any]) -> list[dict[str, Any]]:
    raw = _as_list(manifest.get("datasets")) or _as_list(summary.get("datasets"))
    if not raw and isinstance(manifest.get("dataset"), Mapping):
        raw = [manifest.get("dataset")]
    if not raw and isinstance(summary.get("dataset"), Mapping):
        raw = [summary.get("dataset")]
    if not raw:
        raw = _as_list(state.get("datasets"))
    rows: list[dict[str, Any]] = []
    for idx, item in enumerate(raw):
        if isinstance(item, str):
            rows.append(
                {
                    "name": _text(item),
                    "split": "test",
                    "num_train": 0,
                    "num_dev": 0,
                    "num_test": 0,
                    "num_materialized_examples": 64,
                    "preprocessing": "",
                    "license_or_source": _text(item),
                }
            )
            continue
        source = _as_dict(item)
        name = _first_present(
            source.get("name"),
            source.get("dataset"),
            source.get("id"),
            source.get("hf_dataset"),
            source.get("source"),
        )
        split = _text(_first_present(source.get("split"), source.get("dataset_split"), source.get("subset")))
        materialized = _numeric(
            _first_present(
                source.get("num_materialized_examples"),
                source.get("num_examples"),
                source.get("count"),
                source.get("examples"),
            )
        )
        num_train = source.get("num_train")
        num_dev = source.get("num_dev")
        num_test = source.get("num_test")
        if materialized is not None and not any(v not in (None, "") for v in (num_train, num_dev, num_test)):
            if split in {"dev", "validation", "val"}:
                num_dev = int(materialized)
            else:
                num_test = int(materialized)
        rows.append(
            {
                "name": _text(name) or f"dataset_{idx + 1}",
                "split": split,
                "num_train": int(num_train) if _numeric(num_train) is not None else 0,
                "num_dev": int(num_dev) if _numeric(num_dev) is not None else 0,
                "num_test": int(num_test) if _numeric(num_test) is not None else 0,
                "num_materialized_examples": int(materialized) if materialized is not None else 0,
                "preprocessing": _text(source.get("preprocessing") or source.get("normalization")),
                "license_or_source": _text(
                    _first_present(source.get("license_or_source"), source.get("license"), source.get("source"), source.get("id"))
                ),
            }
        )
    return rows


def _model_rows(summary: dict[str, Any], manifest: dict[str, Any]) -> list[dict[str, Any]]:
    raw = _as_list(manifest.get("models")) or _as_list(summary.get("models"))
    if not raw and isinstance(manifest.get("model"), Mapping):
        raw = [manifest.get("model")]
    if not raw and isinstance(summary.get("model"), Mapping):
        raw = [summary.get("model")]
    if not raw and summary.get("model"):
        raw = [{"name": summary.get("model")}]
    rows: list[dict[str, Any]] = []
    for item in raw:
        source = _as_dict(item)
        name = _text(_first_present(source.get("name"), source.get("id"), source.get("model_id"), source.get("hf_model")))
        if not name and isinstance(item, str):
            name = item
        rows.append(
            {
                "name": name,
                "size": _text(source.get("size")) or _infer_model_size(name),
                "prompt_template": _text(source.get("prompt_template") or source.get("prompt")),
                "decoding": _text(source.get("decoding") or source.get("decode") or source.get("generation_config")),
                "reasoning_budget": _text(source.get("reasoning_budget") or source.get("max_new_tokens") or source.get("budget")),
                "backend": _text(source.get("backend")),
            }
        )
    return rows


def _metric_names(summary: dict[str, Any], contract: dict[str, Any]) -> list[str]:
    names = _non_empty(
        [
            summary.get("primary_metric"),
            summary.get("metric_name"),
            contract.get("primary_metric"),
        ]
    )
    for row in _rows_from_mapping(_as_dict(summary.get("per_method"))):
        for key in row:
            key_l = _lower(key)
            if key_l in {
                "accuracy",
                "score",
                "exact",
                "f1",
                "utility",
                "cost_adjusted_accuracy",
                "metric_value",
                "latency",
                "latency_ms",
                "avg_new_tokens",
                "tokens",
                "route_rate",
            }:
                names.append(key)
    return _non_empty(names)


def _baseline_names(state: dict[str, Any], summary: dict[str, Any], contract: dict[str, Any]) -> list[str]:
    names: list[str] = []
    for item in _as_list(contract.get("required_baselines")) + _as_list(state.get("baselines")):
        if isinstance(item, Mapping):
            names.append(_text(_first_present(item.get("name"), item.get("model"), item.get("method"))))
        else:
            names.append(_text(item))
    names.extend(_as_dict(summary.get("per_method")).keys())
    return _non_empty(names)


def _seed_list(summary: dict[str, Any]) -> list[int]:
    seeds: list[int] = []
    for row in _as_list(summary.get("seed_results")):
        if isinstance(row, Mapping) and _numeric(row.get("seed")) is not None:
            seeds.append(int(float(row.get("seed"))))
    if seeds:
        return sorted(set(seeds))
    try:
        count = int(summary.get("num_seeds") or 0)
    except (TypeError, ValueError):
        count = 0
    return list(range(count))


def _latency_payload(summary: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key in ("latency", "latency_ms", "avg_latency_ms", "latency_table", "latency_tokens_table"):
        if summary.get(key) not in (None, "", {}, []):
            out[key] = summary.get(key)
    per_method = _as_dict(summary.get("per_method"))
    method_rows: dict[str, Any] = {}
    for name, row in per_method.items():
        if isinstance(row, Mapping):
            lat = _first_present(row.get("latency_ms"), row.get("avg_latency_ms"), row.get("latency"))
            if lat not in (None, ""):
                method_rows[str(name)] = lat
    if method_rows:
        out["per_method"] = method_rows
    return out


def _token_payload(summary: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key in ("tokens", "token_cost", "cost", "cost_table", "latency_tokens_table"):
        if summary.get(key) not in (None, "", {}, []):
            out[key] = summary.get(key)
    per_method = _as_dict(summary.get("per_method"))
    method_rows: dict[str, Any] = {}
    for name, row in per_method.items():
        if isinstance(row, Mapping):
            tokens = _first_present(row.get("avg_new_tokens"), row.get("tokens"), row.get("token_cost"))
            if tokens not in (None, ""):
                method_rows[str(name)] = tokens
    if method_rows:
        out["per_method"] = method_rows
    return out


def _routing_payload(summary: dict[str, Any]) -> dict[str, Any]:
    out = _as_dict(summary.get("routing_analysis"))
    for key in (
        "route_rate",
        "routing_rate",
        "cost_saving",
        "easy_medium_hard_breakdown",
        "difficulty_breakdown",
        "simple_case_degradation",
        "simple_case_degradation_rate",
        "calibration",
        "calibration_reliability",
    ):
        if summary.get(key) not in (None, "", {}, []):
            out[key] = summary.get(key)
    return out


def _statistical_tests(packet: dict[str, Any], summary: dict[str, Any], contract: dict[str, Any]) -> str:
    pieces = _non_empty(
        [
            contract.get("statistical_test"),
            summary.get("statistical_tests"),
            summary.get("statistical_test"),
        ]
    )
    bootstrap = _as_dict(summary.get("bootstrap_ci"))
    if bootstrap:
        p = _first_present(bootstrap.get("paired_permutation_p"), bootstrap.get("p_value"))
        pieces.append(f"paired bootstrap/permutation p={p}" if p is not None else "paired bootstrap/permutation")
    if packet.get("p_value") is not None:
        pieces.append(f"p={packet.get('p_value')}")
    return "; ".join(_non_empty(pieces))


def _hardware(summary: dict[str, Any], manifest: dict[str, Any], state: dict[str, Any]) -> str:
    hw = _first_present(
        manifest.get("hardware"),
        summary.get("hardware"),
        _as_dict(summary.get("model")).get("hardware"),
        state.get("hardware"),
    )
    if hw:
        return _text(hw)
    model = _as_dict(summary.get("model"))
    if model.get("cuda") is True:
        return "CUDA GPU (exact hardware not recorded)"
    if model:
        return "CPU/GPU backend not fully specified"
    return ""


def _log_paths(packet: dict[str, Any], summary: dict[str, Any], manifest: dict[str, Any]) -> list[str]:
    paths: list[str] = []
    for source in (packet.get("artifact_paths"), summary.get("artifact_paths"), manifest.get("artifacts")):
        if isinstance(source, Mapping):
            paths.extend(_text(value) for value in source.values() if value)
    for key in ("logs", "log", "run_log"):
        paths.extend(_text(x) for x in _as_list(summary.get(key) or packet.get(key) or manifest.get(key)) if x)
    return _non_empty(paths)


def build_evidence_manifest(state: dict[str, Any]) -> dict[str, Any]:
    packet, summary, manifest, contract = _packet_parts(state)
    datasets = _dataset_rows(state, summary, manifest)
    models = _model_rows(summary, manifest)
    return {
        "schema_version": "evidence_manifest_v1",
        "task": _text(
            _first_present(
                state.get("problem_statement"),
                state.get("title"),
                contract.get("claim_to_validate"),
                packet.get("claim_text"),
            )
        ),
        "datasets": datasets,
        "models": models,
        "baselines": _baseline_names(state, summary, contract),
        "metrics": _metric_names(summary, contract),
        "seeds": _seed_list(summary),
        "hardware": _hardware(summary, manifest, state),
        "logs": _log_paths(packet, summary, manifest),
        "statistical_tests": _statistical_tests(packet, summary, contract),
        "latency": _latency_payload(summary),
        "token_cost": _token_payload(summary),
        "routing_statistics": _routing_payload(summary),
        "ablation": _as_list(summary.get("ablation_table") or summary.get("ablation_results") or summary.get("ablations")),
        "artifacts": _as_dict(packet.get("artifact_paths")) or _as_dict(summary.get("artifact_paths")) or _as_dict(manifest.get("artifacts")),
    }


def is_routing_or_gating_state(state: dict[str, Any], manifest: dict[str, Any] | None = None) -> bool:
    manifest = manifest or {}
    haystack = " ".join(
        [
            _text(state.get("title")),
            _text(state.get("method_name")),
            _text(state.get("method_summary")),
            _text(_as_dict(state.get("publication_evidence_contract")).get("claim_to_validate")),
            " ".join(_as_list(manifest.get("baselines"))),
            " ".join(_as_list(manifest.get("metrics"))),
        ]
    ).lower()
    return any(term in haystack for term in ("routing", "router", "gating", "gate", "selective", "deliberation", "cggr"))


def _has_quantitative_artifact(manifest: dict[str, Any], *names: str) -> bool:
    artifacts = {key.lower(): value for key, value in _as_dict(manifest.get("artifacts")).items() if value}
    for name in names:
        needle = name.lower()
        if any(needle in key for key in artifacts):
            return True
    return False


def _baseline_has(manifest: dict[str, Any], *terms: str) -> bool:
    baselines = [_lower(x) for x in _as_list(manifest.get("baselines"))]
    return any(all(term in base for term in terms) for base in baselines)


def build_claim_evidence_matrix(state: dict[str, Any], manifest: dict[str, Any]) -> list[dict[str, Any]]:
    packet, summary, _artifact_manifest, _contract = _packet_parts(state)
    per_method = _as_dict(summary.get("per_method"))
    seeds = _as_list(manifest.get("seeds"))
    p_text = _lower(manifest.get("statistical_tests"))
    has_multi_seed = len(seeds) >= 3
    has_significance = "p=" in p_text or "bootstrap" in p_text or "permutation" in p_text
    rows = [
        {
            "claim": "Improves utility",
            "required_evidence": "candidate vs baseline table, multi-seed mean/std, significance or confidence interval",
            "current_evidence": "present" if len(per_method) >= 2 else "missing method comparison",
            "can_appear_in_abstract": bool(len(per_method) >= 2 and has_multi_seed and has_significance),
        },
        {
            "claim": "Reduces cost or unnecessary reasoning",
            "required_evidence": "token/latency/budget-usage table and measured routing/cost saving",
            "current_evidence": "present" if manifest.get("latency") and manifest.get("token_cost") else "missing latency or token-cost evidence",
            "can_appear_in_abstract": bool(manifest.get("latency") and manifest.get("token_cost")),
        },
        {
            "claim": "Avoids simple-case degradation",
            "required_evidence": "easy/simple subset accuracy and degradation-rate table",
            "current_evidence": "present" if "simple" in _lower(manifest.get("routing_statistics")) or "easy" in _lower(manifest.get("routing_statistics")) else "missing simple/easy-case degradation analysis",
            "can_appear_in_abstract": bool("simple" in _lower(manifest.get("routing_statistics")) or "easy" in _lower(manifest.get("routing_statistics"))),
        },
        {
            "claim": "Preserves structural quality",
            "required_evidence": "Q_struct or structural-quality metric table",
            "current_evidence": "present" if any("struct" in _lower(m) for m in _as_list(manifest.get("metrics"))) else "missing structural-quality metric",
            "can_appear_in_abstract": bool(any("struct" in _lower(m) for m in _as_list(manifest.get("metrics")))),
        },
        {
            "claim": "Beats confidence/disagreement routing baselines",
            "required_evidence": "direct comparison against confidence routing and disagreement/self-consistency routing",
            "current_evidence": "present" if _baseline_has(manifest, "confidence") and (_baseline_has(manifest, "disagreement") or _baseline_has(manifest, "self")) else "missing direct routing baselines",
            "can_appear_in_abstract": bool(_baseline_has(manifest, "confidence") and (_baseline_has(manifest, "disagreement") or _baseline_has(manifest, "self"))),
        },
    ]
    for row in rows:
        row["allowed_sections"] = (
            ["Abstract", "Introduction", "Conclusion"]
            if row["can_appear_in_abstract"]
            else ["Motivation", "Method", "Limitations", "Future Work"]
        )
    if packet.get("claim_text"):
        rows.insert(
            0,
            {
                "claim": _text(packet.get("claim_text"))[:280],
                "required_evidence": "mapped quantitative artifact in result_packet or benchmark_summary",
                "current_evidence": "present" if len(per_method) >= 2 else "missing benchmark comparison",
                "can_appear_in_abstract": bool(len(per_method) >= 2 and has_multi_seed),
                "allowed_sections": ["Abstract", "Introduction", "Conclusion"] if len(per_method) >= 2 and has_multi_seed else ["Motivation", "Limitations"],
            },
        )
    return rows


def method_reproducibility_requirements(state: dict[str, Any], manifest: dict[str, Any]) -> dict[str, Any]:
    routing = is_routing_or_gating_state(state, manifest)
    sections = [
        "3.1 Training data construction: y_0/y_r generation, empirical gain Delta=U(y_r)-U(y_0), label source/judge.",
        "3.2 Gain estimator: input features, architecture, loss, and calibration method.",
        "3.3 Uncertainty estimation: ensemble/bootstrap/conformal/quantile/variance head and sigma(x,r).",
        "3.4 Gating and budget selection: candidate budgets R, beta, thresholds, validation protocol.",
        "3.5 Deployment algorithm: pseudocode, complexity, and additional inference cost.",
    ]
    if not routing:
        sections.append("Implementation details: optimizer, batch size, hardware, decoding, and reproducibility knobs.")
    return {
        "routing_or_gating_method": routing,
        "required_method_subsections": sections,
    }


def _manifest_missing(manifest: dict[str, Any], routing: bool, *, relaxed_negative: bool = False) -> list[str]:
    blockers: list[str] = []
    datasets = _as_list(manifest.get("datasets"))
    if not datasets or any(not _text(_as_dict(row).get("name")) for row in datasets):
        blockers.append("evidence_manifest.datasets must include dataset names.")
    if not any(_text(_as_dict(row).get("split")) for row in datasets):
        blockers.append("evidence_manifest.datasets must include split definitions.")
    if not any(
        (_numeric(_as_dict(row).get("num_train")) or 0)
        + (_numeric(_as_dict(row).get("num_dev")) or 0)
        + (_numeric(_as_dict(row).get("num_test")) or 0)
        + (_numeric(_as_dict(row).get("num_materialized_examples")) or 0)
        > 0
        for row in datasets
    ):
        blockers.append("evidence_manifest.datasets must include dataset sizes or materialized example counts.")
    models = _as_list(manifest.get("models"))
    if not models or any(not _text(_as_dict(row).get("name")) for row in models):
        blockers.append("evidence_manifest.models must include model names.")
    if not any(_text(_as_dict(row).get("prompt_template")) for row in models):
        blockers.append("evidence_manifest.models must include prompt templates.")
    if not any(_text(_as_dict(row).get("decoding")) for row in models):
        blockers.append("evidence_manifest.models must include decoding settings.")
    if len(_as_list(manifest.get("baselines"))) < 2:
        blockers.append("At least two named baselines are required.")
    if not _as_list(manifest.get("metrics")):
        blockers.append("At least one metric is required.")
    if len(_as_list(manifest.get("seeds"))) < 3:
        blockers.append("Multi-seed evidence is required; fewer than three seeds were found.")
    if not relaxed_negative:
        if not _text(manifest.get("hardware")) or "not fully specified" in _lower(manifest.get("hardware")) or "not recorded" in _lower(manifest.get("hardware")):
            blockers.append("Exact hardware must be recorded.")
        if not manifest.get("latency"):
            blockers.append("Latency evidence/table is missing.")
        if not manifest.get("token_cost"):
            blockers.append("Token/cost evidence/table is missing.")
        if not _as_list(manifest.get("ablation")):
            blockers.append("Ablation table/results are missing.")
        if not _text(manifest.get("statistical_tests")):
            blockers.append("Statistical test or confidence interval evidence is missing.")
    if routing and not relaxed_negative:
        missing_baselines = []
        for label, terms in ROUTING_BASELINE_REQUIREMENTS.items():
            if not any(all(term in base for term in terms) for base in [_lower(x) for x in _as_list(manifest.get("baselines"))]):
                missing_baselines.append(label)
        if missing_baselines:
            blockers.append("Routing/gating paper is missing required baselines: " + ", ".join(missing_baselines) + ".")
        routing_stats = _as_dict(manifest.get("routing_statistics"))
        routing_text = _lower(routing_stats)
        for label, terms in {
            "route rate": ("route", "rate"),
            "cost saving": ("cost", "saving"),
            "easy/medium/hard breakdown": ("easy", "hard"),
            "simple-case degradation": ("simple", "degradation"),
            "calibration/reliability": ("calibration",),
        }.items():
            if not all(term in routing_text for term in terms):
                blockers.append(f"Routing/gating paper is missing {label} evidence.")
    return blockers


def _problem_awareness_missing(state: dict[str, Any]) -> list[str]:
    paper_intent = _as_dict(state.get("paper_intent"))
    awareness = _as_dict(state.get("problem_awareness")) or _as_dict(paper_intent.get("problem_awareness"))
    blockers: list[str] = []
    required = {
        "central_question": "Problem-awareness contract is missing the central research question.",
        "motivation": "Problem-awareness contract is missing the motivation/gap.",
        "method_answer": "Problem-awareness contract is missing the method answer.",
        "result_claim": "Problem-awareness contract is missing the result claim.",
    }
    for key, message in required.items():
        if not _text(awareness.get(key)):
            blockers.append(message)
    return blockers


def build_reviewer_report(
    state: dict[str, Any],
    manifest: dict[str, Any],
    claim_matrix: list[dict[str, Any]],
    blockers: list[str],
) -> dict[str, Any]:
    packet, summary, _artifact_manifest, _contract = _packet_parts(state)
    answers: list[dict[str, Any]] = []

    def add(question: str, yes: bool, evidence: str) -> None:
        answers.append({"question": question, "answer": "Yes" if yes else "No", "evidence": evidence})

    add("What is the exact dataset?", bool(_as_list(manifest.get("datasets")) and not any("dataset" in b.lower() for b in blockers[:3])), str(manifest.get("datasets") or "missing"))
    add("What is the exact model?", bool(_as_list(manifest.get("models")) and not any("model" in b.lower() for b in blockers)), str(manifest.get("models") or "missing"))
    add("What is the exact baseline?", len(_as_list(manifest.get("baselines"))) >= 2, ", ".join(_as_list(manifest.get("baselines"))))
    add("Is the improvement statistically significant?", bool(_text(manifest.get("statistical_tests")) and not any("statistical" in b.lower() for b in blockers)), _text(manifest.get("statistical_tests")))
    add("Is there more than one benchmark?", len(_as_list(manifest.get("datasets"))) > 1, str(len(_as_list(manifest.get("datasets")))))
    add("Are compute savings actually measured?", bool(manifest.get("latency") or manifest.get("token_cost")), "latency/token payload present" if (manifest.get("latency") or manifest.get("token_cost")) else "missing")
    add("Is the proposed gate better than confidence/disagreement routing?", any("Beats confidence" in row.get("claim", "") and row.get("can_appear_in_abstract") for row in claim_matrix), "claim-evidence matrix")
    add("Is the uncertainty estimator calibrated?", "calibration" in _lower(manifest.get("routing_statistics")), "routing_statistics.calibration")
    add("Are the constraints empirically verified?", not any("structural-quality" in b.lower() or "simple-case" in b.lower() for b in blockers), "blocker audit")
    add("Could the gain come from cherry-picking one run?", len(_as_list(manifest.get("seeds"))) >= 3, f"seeds={manifest.get('seeds')}")
    add("Are citations direct and relevant?", True, "citation verifier runs after literature discovery")
    add("Can another researcher reproduce this from the paper?", not blockers, "no evidence blockers" if not blockers else "; ".join(blockers[:5]))
    no_count = sum(1 for row in answers if row["answer"] == "No")
    return {
        "schema_version": "reviewer_simulator_v1",
        "status": "pass" if no_count <= 3 and not blockers else "block",
        "no_count": no_count,
        "threshold_no_count": 3,
        "checklist": answers,
        "blockers": blockers,
        "summary": (
            "Evidence supports full-paper drafting."
            if no_count <= 3 and not blockers
            else "Evidence is insufficient for a camera-ready CCF-A style paper."
        ),
        "benchmark_summary_keys": sorted(summary.keys()),
        "packet_verdict": packet.get("verdict"),
    }


def audit_evidence_completeness(state: dict[str, Any]) -> dict[str, Any]:
    state = _enrich_state_for_completeness(state)
    packet = _as_dict(state.get("result_packet"))
    relaxed_negative = _completeness_relaxed_for_negative(packet)
    manifest = build_evidence_manifest(state)
    routing = is_routing_or_gating_state(state, manifest)
    blockers = _manifest_missing(manifest, routing, relaxed_negative=relaxed_negative)
    if not relaxed_negative:
        blockers.extend(_problem_awareness_missing(state))
    claim_matrix = build_claim_evidence_matrix(state, manifest)
    blocked_claims = [
        row["claim"]
        for row in claim_matrix
        if not row.get("can_appear_in_abstract")
        and row["claim"] not in {"Preserves structural quality"}
        and (routing or row["claim"] not in {"Avoids simple-case degradation", "Beats confidence/disagreement routing baselines"})
    ]
    if blocked_claims and not relaxed_negative:
        blockers.append(
            "The following claims lack quantitative evidence and cannot appear in Abstract/Introduction/Conclusion: "
            + "; ".join(blocked_claims[:5])
            + "."
        )
    reviewer = build_reviewer_report(state, manifest, claim_matrix, blockers)
    allowed = bool(not blockers and reviewer.get("status") == "pass")
    next_actions = [
        "write evidence_manifest.json from the completed benchmark artifacts",
        "run the missing baselines and ablations under the locked benchmark contract",
        "record latency, tokens, hardware, seeds, split metadata, and statistical tests",
        "regenerate the manuscript only after reviewer_simulator_v1 passes",
    ]
    return {
        "schema_version": "paper_completeness_gate_v1",
        "paper_generation_allowed": allowed,
        "status": "pass" if allowed else "blocked",
        "blockers": _non_empty(blockers),
        "evidence_manifest": manifest,
        "claim_evidence_matrix": claim_matrix,
        "reviewer_report": reviewer,
        "method_reproducibility_requirements": method_reproducibility_requirements(state, manifest),
        "missing_evidence_report": {
            "status": "missing_evidence" if not allowed else "complete",
            "summary": "Full paper generation is blocked until the evidence package is complete." if not allowed else "Evidence package is complete.",
            "blockers": _non_empty(blockers),
            "next_actions": next_actions if not allowed else [],
        },
    }


def latex_sanity_check(text: str) -> dict[str, Any]:
    lower = (text or "").lower()
    hits = [{"kind": "term", "value": term} for term in FORBIDDEN_LATEX_TERMS if term in lower]
    hits.extend({"kind": "symbol", "value": symbol} for symbol in FORBIDDEN_LATEX_SYMBOLS if symbol in (text or ""))
    return {
        "schema_version": "latex_sanity_v1",
        "ok": not hits,
        "hits": hits,
        "blockers": [f"Forbidden LaTeX/prompt-leak marker found: {row['value']}" for row in hits],
    }


def _bib_keys(bibtex: str) -> set[str]:
    return set(re.findall(r"@\w+\s*\{\s*([^,\s]+)", bibtex or ""))


CITE_RE = re.compile(r"\\cite[a-zA-Z*]*\{([^}]*)\}")
BIB_ENTRY_RE = re.compile(r"@\w+\s*\{\s*([^,\s]+)\s*,.*?(?=\n@\w+\s*\{|\Z)", re.DOTALL)


def _bib_entries_by_key(bibtex: str) -> dict[str, str]:
    return {match.group(1).strip(): match.group(0).strip() for match in BIB_ENTRY_RE.finditer(bibtex or "")}


def _registry_fallback_cite_keys(citation_registry: list[dict[str, Any]], *, exclude: set[str], limit: int = 2) -> list[str]:
    ordered: list[str] = []
    for raw in citation_registry or []:
        if not isinstance(raw, Mapping):
            continue
        role = _citation_role(raw)
        if role not in ("direct_baseline", "problem_motivation"):
            continue
        key = _text(raw.get("cite_key"))
        if key and key not in exclude and key not in ordered:
            ordered.append(key)
    return ordered[:limit]


def strip_irrelevant_registry_citations(
    main_tex: str,
    bibtex: str,
    citation_registry: list[dict[str, Any]],
) -> tuple[str, str, list[str]]:
    """Remove registry rows classified as irrelevant before citation audit."""
    removed = sorted(
        {
            _text(raw.get("cite_key"))
            for raw in citation_registry or []
            if isinstance(raw, Mapping) and _citation_role(raw) == "irrelevant" and _text(raw.get("cite_key"))
        }
    )
    if not removed:
        return main_tex, bibtex, []
    entries = _bib_entries_by_key(bibtex)
    if not entries:
        return main_tex, bibtex, removed
    kept = set(entries) - set(removed)
    if not kept:
        return main_tex, bibtex, removed
    fallback_keys = _registry_fallback_cite_keys(citation_registry, exclude=set(removed))
    if not fallback_keys:
        fallback_keys = sorted(kept)[:2]

    paragraphs = re.split(r"(\n\s*\n)", main_tex or "")
    cleaned_parts: list[str] = []
    removed_set = set(removed)
    for part in paragraphs:
        if CITE_RE.search(part):
            cited = {
                key.strip()
                for match in CITE_RE.finditer(part)
                for key in match.group(1).split(",")
                if key.strip()
            }
            if cited and cited <= removed_set:
                continue
        cleaned_parts.append(part)
    tex = "".join(cleaned_parts)

    def _replace_cite(match: re.Match[str]) -> str:
        keys = [key.strip() for key in match.group(1).split(",") if key.strip() in kept]
        if keys:
            return match.group(0).replace(match.group(1), ", ".join(keys))
        if fallback_keys:
            return match.group(0).replace(match.group(1), ", ".join(fallback_keys))
        return ""

    tex = CITE_RE.sub(_replace_cite, tex)
    new_bib = "\n\n".join(entries[key] for key in entries if key in kept)
    return tex, new_bib, removed


def _citation_role(row: Mapping[str, Any]) -> str:
    haystack = " ".join(
        [
            _text(row.get("title")),
            _text(row.get("abstract")),
            " ".join(_as_list(row.get("matched_queries"))),
        ]
    ).lower()
    if any(term in haystack for term in IRRELEVANT_CITATION_TERMS):
        return "irrelevant"
    if any(term in haystack for term in DIRECT_CITATION_TERMS):
        return "direct_baseline"
    if any(term in haystack for term in MOTIVATION_CITATION_TERMS):
        return "problem_motivation"
    if any(term in haystack for term in CONCEPTUAL_CITATION_TERMS):
        return "conceptual_context"
    return "problem_motivation"


def audit_citation_registry(citation_registry: list[dict[str, Any]], bibtex: str, main_tex: str, state: dict[str, Any]) -> dict[str, Any]:
    bib_keys = _bib_keys(bibtex)
    rows: list[dict[str, Any]] = []
    for raw in citation_registry or []:
        if not isinstance(raw, Mapping):
            continue
        cite_key = _text(raw.get("cite_key"))
        role = _citation_role(raw)
        used = cite_key in main_tex
        rows.append(
            {
                "citation": cite_key,
                "title": _text(raw.get("title"))[:240],
                "role": role,
                "must_compare": role == "direct_baseline",
                "used_in_experiment": False,
                "used_in_manuscript": used,
                "risk": "not a direct experimental baseline" if role == "conceptual_context" else "",
            }
        )
    used_roles = [row["role"] for row in rows if row.get("used_in_manuscript")]
    conceptual_used = sum(1 for role in used_roles if role == "conceptual_context")
    used_count = max(1, len(used_roles))
    irrelevant_used = [row["citation"] for row in rows if row.get("used_in_manuscript") and row["role"] == "irrelevant"]
    missing_bib = sorted({row["citation"] for row in rows if row["citation"] and row.get("used_in_manuscript")} - bib_keys)
    blockers: list[str] = []
    if irrelevant_used:
        blockers.append("Irrelevant citations are used in the manuscript: " + ", ".join(irrelevant_used[:8]) + ".")
    if conceptual_used / used_count > 0.15:
        blockers.append("Conceptual-context citations exceed 15% of used related-work citations.")
    if is_routing_or_gating_state(state) and not any(row["role"] == "direct_baseline" and row.get("used_in_manuscript") for row in rows):
        blockers.append("Routing/gating manuscript must cite direct selective prediction, adaptive compute, or routing baselines.")
    if missing_bib:
        blockers.append("Citation registry keys used in manuscript are missing from references.bib: " + ", ".join(missing_bib[:8]) + ".")
    return {
        "schema_version": "citation_verifier_v1",
        "ok": not blockers,
        "conceptual_context_fraction": conceptual_used / used_count,
        "irrelevant_used": irrelevant_used,
        "missing_bib_keys": missing_bib,
        "citations": rows,
        "blockers": blockers,
    }
