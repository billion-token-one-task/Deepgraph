"""Generate manuscript bundles using PaperOrchestra as the only backend."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from config import MANUSCRIPT_ALLOW_NEGATIVE_RESULTS, SUBMISSION_BUNDLE_FORMATS
from contracts import ContractValidationError, ManuscriptInputState
from contracts.pipeline import _real_benchmark_summary_present
from agents.benchmark_audit import best_iteration_benchmark_summary
from agents.metric_parser import persist_main_results_table
from agents.paper_completeness import audit_evidence_completeness
from db import database as db


def _json_load(value, default):
    if value is None:
        return default
    if isinstance(value, (dict, list)):
        return value
    try:
        return json.loads(value)
    except (json.JSONDecodeError, TypeError):
        return default


def _json_list(value) -> list:
    loaded = _json_load(value, [])
    return loaded if isinstance(loaded, list) else []


def _json_dict(value) -> dict[str, Any]:
    loaded = _json_load(value, {})
    return loaded if isinstance(loaded, dict) else {}


def _dedupe(values: list[Any]) -> list[Any]:
    out: list[Any] = []
    seen = set()
    for value in values:
        key = json.dumps(value, sort_keys=True, ensure_ascii=False) if isinstance(value, (dict, list)) else str(value)
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(value)
    return out


def _ensure_dirs(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _best_iteration(iterations: list[dict]) -> dict | None:
    kept = [it for it in iterations if it.get("status") == "keep" and it.get("metric_value") is not None]
    if not kept:
        return None
    return max(kept, key=lambda item: item.get("metric_value") or 0)


def _as_float(value: Any) -> float | None:
    if value in (None, "", []):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _result_contribution(run: dict, result_packet: dict) -> str:
    metric_name = run.get("baseline_metric_name") or result_packet.get("metric_name") or "metric"
    baseline = _as_float(result_packet.get("baseline", run.get("baseline_metric_value")))
    best = _as_float(result_packet.get("best", run.get("best_metric_value")))
    direction = str(result_packet.get("metric_direction") or "higher").lower()
    effect = _as_float(result_packet.get("effect_size"))
    if effect is None and baseline is not None and best is not None:
        effect = best - baseline if direction == "higher" else baseline - best
    effect_pct = _as_float(result_packet.get("effect_pct", run.get("effect_pct")))
    if baseline is None or best is None or effect is None:
        return f"Evaluated {metric_name} with baseline {baseline} and best metric {best}."
    pct_text = f", {effect_pct:+.2f}%" if effect_pct is not None else ""
    if effect > 1e-12:
        return f"Best {metric_name}={best:.6f} exceeds baseline {baseline:.6f} by {effect:+.6f}{pct_text}."
    if effect < -1e-12:
        return f"Best {metric_name}={best:.6f} remains below baseline {baseline:.6f} by {effect:+.6f}{pct_text}."
    return f"Best {metric_name}={best:.6f} ties baseline {baseline:.6f}; no positive effect has been established."


def _apply_manuscript_packet_overrides(packet: dict[str, Any], run: dict, proxy: dict) -> dict[str, Any]:
    """Refresh stale bootstrap_probe / blocks_manuscript flags after real benchmarks finish."""
    summary = packet.get("benchmark_summary") if isinstance(packet.get("benchmark_summary"), dict) else {}
    publication_contract = proxy.get("publication_evidence_contract") or {}
    if not isinstance(publication_contract, dict):
        publication_contract = {}
    tier = str(
        packet.get("evidence_tier")
        or publication_contract.get("evidence_tier")
        or proxy.get("evidence_tier")
        or ""
    ).strip().lower()
    if tier in {"bootstrap_probe", "sanity_real_benchmark"} and _real_benchmark_summary_present(summary):
        packet["evidence_tier"] = "benchmark_plan"
    verdict = str(run.get("hypothesis_verdict") or packet.get("verdict") or "").strip().lower()
    packet["verdict"] = verdict or packet.get("verdict")
    if MANUSCRIPT_ALLOW_NEGATIVE_RESULTS and _real_benchmark_summary_present(summary) and verdict in {
        "refuted",
        "inconclusive",
        "confirmed",
        "supported",
    }:
        packet["blocks_manuscript"] = False
        if bool(proxy.get("formal_experiment")):
            packet["formal_experiment"] = True
    proxy_claim_route = proxy.get("claim_route") if isinstance(proxy.get("claim_route"), dict) else {}
    packet_claim_route = packet.get("claim_route") if isinstance(packet.get("claim_route"), dict) else {}
    proxy_route_value = str(proxy_claim_route.get("route") or "").strip().lower()
    packet_route_value = str(packet_claim_route.get("route") or "").strip().lower()
    if proxy_route_value == "full_paper":
        if packet_route_value in {"probe", "blocked"}:
            # Route demotion that loses ablation/baseline gates is the
            # silent failure mode that "30 iter null → refuted paper"
            # exploits. Log it loudly and surface it in the packet.
            print(
                f"[ROUTE] WARNING: claim_route demoted from full_paper to {packet_route_value!r} "
                "in result packet; restoring contract route. Audit gates would otherwise lapse.",
                flush=True,
            )
            packet.setdefault("route_demotion_warnings", []).append(
                {
                    "from": "full_paper",
                    "to": packet_route_value,
                    "restored_from": "proxy.claim_route",
                }
            )
        packet["claim_route"] = proxy_claim_route
    elif packet_route_value in {"", "probe", "blocked"}:
        pub_route = publication_contract.get("claim_route") if isinstance(publication_contract.get("claim_route"), dict) else {}
        if str(pub_route.get("route") or "").strip().lower() == "full_paper":
            packet["claim_route"] = pub_route
    return packet


def _load_result_packet(run: dict, claims: list[dict]) -> dict[str, Any]:
    workdir_raw = str(run.get("workdir") or "").strip()
    benchmark_summary: dict[str, Any] = {}
    benchmark_artifact_manifest: dict[str, Any] = {}
    artifact_paths: dict[str, Any] = {}
    if workdir_raw:
        workdir = Path(workdir_raw)
        summary_path = workdir / "results" / "benchmark_summary.json"
        manifest_path = workdir / "results" / "benchmark_artifact_manifest.json"
        results_dir = workdir / "results"
        if (results_dir / "raw_predictions.jsonl").is_file():
            from agents.benchmark_artifacts import materialize_deep_benchmark_artifacts

            contract = {}
            criteria_path = workdir / "spec" / "success_criteria.json"
            if criteria_path.exists():
                crit = _json_dict(criteria_path.read_text(encoding="utf-8"))
                contract = (
                    crit.get("publication_evidence_contract")
                    or crit.get("publication_evidence")
                    or {}
                )
            if not isinstance(contract, dict):
                contract = {}
            materialize_deep_benchmark_artifacts(
                results_dir,
                publication_contract=contract,
                metric_name=str(run.get("baseline_metric_name") or "primary_score"),
            )
        if summary_path.exists():
            benchmark_summary = _json_dict(summary_path.read_text(encoding="utf-8"))
            persist_main_results_table(results_dir, benchmark_summary)
        success_hint = _json_dict(run.get("success_criteria"))
        best_summary = best_iteration_benchmark_summary(
            workdir,
            best_metric=_as_float(run.get("best_metric_value")),
            direction=str(success_hint.get("metric_direction") or "higher"),
        )
        if best_summary:
            benchmark_summary = best_summary
        if manifest_path.exists():
            benchmark_artifact_manifest = _json_dict(manifest_path.read_text(encoding="utf-8"))
            artifact_paths["artifact_manifest"] = str(manifest_path)
        packet_path = workdir / "results" / "experiment_result_packet.json"
        if packet_path.exists():
            packet = _json_dict(packet_path.read_text(encoding="utf-8"))
            if benchmark_summary:
                packet["benchmark_summary"] = benchmark_summary
            if benchmark_artifact_manifest and not packet.get("benchmark_artifact_manifest"):
                packet["benchmark_artifact_manifest"] = benchmark_artifact_manifest
            if artifact_paths:
                packet["artifact_paths"] = {**_json_dict(packet.get("artifact_paths")), **artifact_paths}
            proxy = _json_dict(run.get("proxy_config"))
            return _apply_manuscript_packet_overrides(packet, run, proxy)
    for claim in claims:
        packet = _json_dict(_json_dict(claim.get("supporting_data")).get("result_packet"))
        if packet:
            return packet
    proxy = _json_dict(run.get("proxy_config"))
    success = _json_dict(run.get("success_criteria"))
    if not success and workdir_raw:
        criteria_path = Path(workdir_raw) / "spec" / "success_criteria.json"
        if criteria_path.exists():
            success = _json_dict(criteria_path.read_text(encoding="utf-8"))
    publication_contract = (
        success.get("publication_evidence_contract")
        or success.get("publication_evidence")
        or proxy.get("publication_evidence_contract")
        or {}
    )
    if not isinstance(publication_contract, dict):
        publication_contract = {}
    paper_intent = success.get("paper_intent") or publication_contract.get("paper_intent") or proxy.get("paper_intent") or {}
    if not isinstance(paper_intent, dict):
        paper_intent = {}
    quality_gates = success.get("quality_gates") or publication_contract.get("quality_gates") or {}
    if not isinstance(quality_gates, dict):
        quality_gates = {}
    claim_route = success.get("claim_route") or publication_contract.get("claim_route") or proxy.get("claim_route") or {}
    if not isinstance(claim_route, dict):
        claim_route = {}
    if str(claim_route.get("route") or "").strip().lower() in {"probe", "blocked", ""}:
        proxy_route = proxy.get("claim_route") if isinstance(proxy.get("claim_route"), dict) else {}
        if str(proxy_route.get("route") or "").strip().lower() == "full_paper":
            claim_route = proxy_route
    packet = {
        "run_id": run.get("id"),
        "deep_insight_id": run.get("deep_insight_id"),
        "formal_experiment": bool(proxy.get("formal_experiment")),
        "smoke_test_only": bool(proxy.get("smoke_test_only")),
        "metric_name": run.get("baseline_metric_name") or "metric",
        "verdict": run.get("hypothesis_verdict") or "inconclusive",
        "baseline": run.get("baseline_metric_value"),
        "best": run.get("best_metric_value"),
        "effect_pct": run.get("effect_pct"),
        "benchmark_summary": benchmark_summary,
        "benchmark_artifact_manifest": benchmark_artifact_manifest,
        "full_benchmark_completed": bool(benchmark_artifact_manifest.get("full_benchmark_completed")),
        "artifact_paths": artifact_paths,
        "evidence_tier": success.get("evidence_tier") or publication_contract.get("evidence_tier") or proxy.get("evidence_tier") or "",
        "publication_ready": success.get("publication_ready", publication_contract.get("publication_ready")),
        "blocks_manuscript": bool(
            success.get("blocks_manuscript")
            or publication_contract.get("blocks_manuscript")
            or proxy.get("blocks_manuscript")
        ),
        "publication_evidence_contract": publication_contract,
        "claim_route": claim_route,
        "paper_intent": paper_intent,
        "quality_gates": quality_gates,
        "reviewer_objections": success.get("reviewer_objections") or publication_contract.get("reviewer_objections") or [],
    }
    return _apply_manuscript_packet_overrides(packet, run, proxy)


def _publication_contract_from_inputs(result_packet: dict, plan: dict, run: dict) -> dict[str, Any]:
    contract = _json_dict(result_packet.get("publication_evidence_contract"))
    if not contract:
        contract = _json_dict(plan.get("publication_evidence_contract"))
    if not contract:
        proxy = _json_dict(run.get("proxy_config"))
        contract = _json_dict(proxy.get("publication_evidence_contract"))
    if contract:
        return contract

    def _names(rows) -> list[str]:
        values: list[str] = []
        for row in rows or []:
            if isinstance(row, dict):
                value = row.get("name") or row.get("model") or row.get("dataset")
            else:
                value = row
            text = str(value or "").strip()
            if text and text not in values:
                values.append(text)
        return values

    datasets = _names(plan.get("datasets"))
    baselines = _names(plan.get("baselines"))
    real_datasets = [
        name for name in datasets
        if not any(marker in name.lower() for marker in ("synthetic", "toy", "smoke", "probe", "dummy"))
    ]
    metrics = _json_dict(plan.get("metrics"))
    metric_name = result_packet.get("metric_name") or metrics.get("primary") or "metric"
    evidence_tier = result_packet.get("evidence_tier") or (
        "benchmark_plan" if real_datasets and len(baselines) >= 2 else "formal_proxy"
    )
    return {
        "claim_to_validate": result_packet.get("claim_text") or "",
        "evidence_tier": evidence_tier,
        "publication_ready": False,
        "blocks_manuscript": bool(result_packet.get("blocks_manuscript")),
        "minimum_seeds": 3,
        "required_datasets": datasets,
        "required_real_benchmarks": real_datasets,
        "required_baselines": baselines,
        "required_ablations": _names(plan.get("ablations")) or ["method_removed", "compute_matched_baseline"],
        "primary_metric": metric_name,
        "statistical_test": "paired bootstrap confidence interval plus paired permutation test across seeds/tasks",
        "required_artifacts": ["main_results_table", "ablation_table", "seed_variance_table", "raw_metrics_jsonl"],
        "quality_gates": {
            "has_real_benchmark": bool(real_datasets),
            "baseline_count": len(baselines),
            "minimum_seeds": 3,
            "requires_ablation_table": True,
            "requires_full_benchmark_package": bool(evidence_tier == "benchmark_plan"),
        },
        "reviewer_objections": [
            "Are baselines fair and current?",
            "Do ablations isolate the claimed mechanism?",
            "Are results stable across seeds with statistical uncertainty?",
            "Are proxy limitations clearly separated from benchmark claims?",
        ],
    }


def _named_rows(rows: Any, *, keys: tuple[str, ...] = ("name", "model", "dataset")) -> list[str]:
    values: list[str] = []
    for row in rows or []:
        if isinstance(row, dict):
            value = next((row.get(key) for key in keys if row.get(key)), "")
        else:
            value = row
        text = str(value or "").strip()
        if text and text not in values:
            values.append(text)
    return values


def _build_problem_awareness(
    *,
    insight: dict,
    method: dict,
    plan: dict,
    result_packet: dict,
    publication_contract: dict,
) -> dict[str, Any]:
    existing = (
        _json_dict(insight.get("problem_awareness"))
        or _json_dict(result_packet.get("problem_awareness"))
        or _json_dict(plan.get("problem_awareness"))
        or _json_dict(publication_contract.get("problem_awareness"))
    )
    benchmark_summary = _json_dict(result_packet.get("benchmark_summary"))
    datasets = _named_rows(
        publication_contract.get("required_real_benchmarks")
        or publication_contract.get("required_datasets")
        or benchmark_summary.get("datasets")
        or plan.get("datasets"),
        keys=("name", "dataset", "id"),
    )
    baselines = _named_rows(
        publication_contract.get("required_baselines")
        or benchmark_summary.get("per_method", {}).keys()
        or plan.get("baselines"),
        keys=("name", "model", "method"),
    )
    metric = (
        result_packet.get("metric_name")
        or _json_dict(plan.get("metrics")).get("primary")
        or publication_contract.get("primary_metric")
        or "metric"
    )
    central_question = (
        existing.get("central_question")
        or existing.get("question")
        or publication_contract.get("claim_to_validate")
        or insight.get("problem_statement")
        or insight.get("title")
        or ""
    )
    motivation = (
        existing.get("motivation")
        or insight.get("existing_weakness")
        or insight.get("evidence_summary")
        or "Prior results leave the claimed mechanism unresolved."
    )
    method_answer = (
        existing.get("method_answer")
        or existing.get("method")
        or method.get("mechanism_repair")
        or method.get("one_line")
        or method.get("definition")
        or f"{method.get('name') or insight.get('title') or 'The proposed method'} is the intervention evaluated by the benchmark."
        or ""
    )
    if result_packet.get("baseline") is not None or result_packet.get("best") is not None:
        result_claim = (
            f"{metric}: baseline={result_packet.get('baseline')}, "
            f"best={result_packet.get('best')}, effect_pct={result_packet.get('effect_pct')}, "
            f"verdict={result_packet.get('verdict')}."
        )
    else:
        result_claim = str(existing.get("result_claim") or publication_contract.get("claim_to_validate") or "")
    return {
        **existing,
        "central_question": central_question,
        "motivation": motivation,
        "method_answer": method_answer,
        "result_claim": existing.get("result_claim") or result_claim,
        "falsification_result": existing.get("falsification_result")
        or method.get("falsification_hook")
        or "The method fails if it cannot beat matched baselines under the required datasets, seeds, and statistical tests.",
        "benchmark_context": {
            "datasets": datasets,
            "baselines": baselines,
            "primary_metric": metric,
        },
        "required_story_order": ["problem", "motivation", "method", "result", "limitations"],
        "top_venue_questions": [
            "What exact failure mode do current papers leave unresolved?",
            "Why is the proposed mechanism necessary rather than a stronger baseline?",
            "Which completed result supports the central claim, with uncertainty?",
            "What result would falsify or sharply weaken the paper?",
        ],
    }


def _build_paper_intent(
    *,
    insight: dict,
    method: dict,
    plan: dict,
    result_packet: dict,
    claim_records: list[dict],
    publication_contract: dict,
    problem_awareness: dict[str, Any],
) -> dict[str, Any]:
    existing = _json_dict(result_packet.get("paper_intent")) or _json_dict(plan.get("paper_intent"))
    if not existing:
        existing = _json_dict(publication_contract.get("paper_intent"))
    method_name = method.get("name") or insight.get("title") or "DeepGraph Method"
    primary_metric = (
        result_packet.get("metric_name")
        or _json_dict(plan.get("metrics")).get("primary")
        or "metric"
    )
    claim_text = (
        publication_contract.get("claim_to_validate")
        or (claim_records[0].get("claim_text") if claim_records else "")
        or result_packet.get("claim_text")
        or insight.get("title")
        or ""
    )
    evidence_tier = result_packet.get("evidence_tier") or publication_contract.get("evidence_tier") or "unknown"
    narrative_spine = existing.get("narrative_spine") if isinstance(existing.get("narrative_spine"), list) else []
    if not narrative_spine:
        narrative_spine = [
            f"Gap: {insight.get('existing_weakness') or insight.get('problem_statement') or ''}",
            f"Mechanism: {method_name} is the proposed intervention.",
            f"Evidence: report baseline, proposed method, ablations, seed variance, and {primary_metric}.",
            f"Boundary: evidence tier is {evidence_tier}; do not overclaim beyond completed experiments.",
        ]
    return {
        **existing,
        "problem_awareness": problem_awareness,
        "central_question": existing.get("central_question") or problem_awareness.get("central_question"),
        "motivation": existing.get("motivation") or problem_awareness.get("motivation"),
        "method_answer": existing.get("method_answer") or problem_awareness.get("method_answer"),
        "result_claim": existing.get("result_claim") or problem_awareness.get("result_claim"),
        "central_claim": existing.get("central_claim") or claim_text,
        "claim_route": existing.get("claim_route")
        or _json_dict(publication_contract.get("claim_route")).get("route"),
        "claim_strength": existing.get("claim_strength")
        or publication_contract.get("claim_strength"),
        "target_venue": existing.get("target_venue") or "top-tier ML venue",
        "reader_takeaway": existing.get("reader_takeaway")
        or f"{method_name} should be accepted only if the completed evidence supports the stated mechanism.",
        "evidence_tier": evidence_tier,
        "narrative_spine": narrative_spine,
        "do_not_overclaim": True,
    }


def _claim_source_papers(claim: dict, fallback_ids: list[str]) -> list[str]:
    supporting_data = _json_dict(claim.get("supporting_data"))
    candidates: list[str] = []
    for raw in (
        claim.get("source_paper_ids"),
        supporting_data.get("source_paper_ids"),
        supporting_data.get("supporting_papers"),
        supporting_data.get("paper_ids"),
        fallback_ids,
    ):
        for item in _json_list(raw) if not isinstance(raw, list) else raw:
            if item:
                candidates.append(str(item))
    return _dedupe(candidates)


def _claim_source_nodes(claim: dict, fallback_ids: list[str]) -> list[str]:
    supporting_data = _json_dict(claim.get("supporting_data"))
    candidates: list[str] = []
    for raw in (
        claim.get("source_node_ids"),
        supporting_data.get("source_node_ids"),
        fallback_ids,
    ):
        for item in _json_list(raw) if not isinstance(raw, list) else raw:
            if item:
                candidates.append(str(item))
    return _dedupe(candidates)


def _build_claim_records(claims: list[dict], *, fallback_papers: list[str], fallback_nodes: list[str], evidence_summary: str) -> list[dict]:
    out: list[dict] = []
    for claim in claims:
        supporting_data = _json_dict(claim.get("supporting_data"))
        out.append(
            {
                "id": claim.get("id"),
                "claim_text": claim.get("claim_text") or "",
                "claim_type": claim.get("claim_type") or "experimental",
                "verdict": claim.get("verdict") or "inconclusive",
                "effect_size": claim.get("effect_size"),
                "confidence": claim.get("confidence"),
                "supporting_data": supporting_data,
                "source_paper_ids": _claim_source_papers(claim, fallback_papers),
                "source_node_ids": _claim_source_nodes(claim, fallback_nodes),
                "evidence_summary": supporting_data.get("evidence_summary") or evidence_summary,
            }
        )
    return out


def build_manuscript_input_state(run: dict, insight: dict, iterations: list[dict], claims: list[dict]) -> ManuscriptInputState:
    method = _json_dict(insight.get("proposed_method"))
    plan = _json_dict(insight.get("experimental_plan"))
    evidence_plan = _json_dict(insight.get("evidence_plan"))
    related_raw = _json_load(insight.get("related_work_positioning"), {})
    related = related_raw if isinstance(related_raw, dict) else {}
    supporting_papers = [str(x) for x in _json_list(insight.get("supporting_papers")) if x]
    source_paper_ids = [str(x) for x in _json_list(insight.get("source_paper_ids")) if x]
    source_node_ids = [str(x) for x in _json_list(insight.get("source_node_ids")) if x]
    citation_seed_paper_ids = _dedupe(supporting_papers + source_paper_ids)
    if not citation_seed_paper_ids:
        rows = db.fetchall(
            """
            SELECT id FROM papers
            WHERE status IN ('reasoned', 'graph_written', 'text_ready', 'extracted')
            ORDER BY updated_at DESC
            LIMIT 8
            """
        )
        citation_seed_paper_ids = [str(row["id"]) for row in rows if row.get("id")]
    evidence_packet = _json_dict(insight.get("evidence_packet"))
    evidence_summary = insight.get("evidence_summary") or insight.get("related_work_positioning") or ""
    best_iter = _best_iteration(iterations)
    result_packet = _load_result_packet(run, claims)

    contributions = [
        method.get("one_line") or "Mechanism-first insight generation and automated experiment loop.",
        _result_contribution(run, result_packet),
        f"Generated as a {insight.get('mechanism_type') or 'mechanism-first'} DeepGraph insight.",
    ]

    claim_records = _build_claim_records(
        claims,
        fallback_papers=citation_seed_paper_ids,
        fallback_nodes=source_node_ids,
        evidence_summary=str(evidence_summary),
    )
    publication_contract = _publication_contract_from_inputs(result_packet, plan, run)
    problem_awareness = _build_problem_awareness(
        insight=insight,
        method=method,
        plan=plan,
        result_packet=result_packet,
        publication_contract=publication_contract,
    )
    paper_intent = _build_paper_intent(
        insight=insight,
        method=method,
        plan=plan,
        result_packet=result_packet,
        claim_records=claim_records,
        publication_contract=publication_contract,
        problem_awareness=problem_awareness,
    )
    quality_gates = _json_dict(result_packet.get("quality_gates")) or _json_dict(publication_contract.get("quality_gates"))
    claim_route = (
        _json_dict(result_packet.get("claim_route"))
        or _json_dict(publication_contract.get("claim_route"))
    )
    required_evidence = {
        "datasets": publication_contract.get("required_datasets") or plan.get("datasets", []),
        "real_benchmarks": publication_contract.get("required_real_benchmarks") or [],
        "baselines": publication_contract.get("required_baselines") or plan.get("baselines", []),
        "ablations": publication_contract.get("required_ablations") or plan.get("ablations", []),
        "minimum_seeds": publication_contract.get("minimum_seeds"),
        "statistical_test": publication_contract.get("statistical_test"),
        "artifacts": publication_contract.get("required_artifacts") or [],
    }
    reviewer_objections = result_packet.get("reviewer_objections") or publication_contract.get("reviewer_objections") or []
    if not isinstance(reviewer_objections, list):
        reviewer_objections = []

    state = ManuscriptInputState(
        run_id=run.get("id"),
        deep_insight_id=run.get("deep_insight_id"),
        formal_experiment=bool(result_packet.get("formal_experiment")),
        smoke_test_only=bool(result_packet.get("smoke_test_only")),
        title=insight.get("title") or f"DeepGraph Run {run['id']}",
        problem_statement=insight.get("problem_statement") or insight.get("existing_weakness") or "",
        existing_weakness=insight.get("existing_weakness") or "",
        method_name=method.get("name") or insight.get("title") or "DeepGraph Method",
        method_summary=method.get("one_line") or method.get("definition") or "",
        method_payload=method,
        mechanism_type=insight.get("mechanism_type") or "mechanism_first",
        resource_class=run.get("resource_class") or insight.get("resource_class") or "cpu",
        baseline_metric_name=run.get("baseline_metric_name") or result_packet.get("metric_name") or "metric",
        baseline_metric_value=result_packet.get("baseline", run.get("baseline_metric_value")),
        best_metric_value=result_packet.get("best", run.get("best_metric_value")),
        effect_pct=result_packet.get("effect_pct", run.get("effect_pct")),
        verdict=result_packet.get("verdict", run.get("hypothesis_verdict") or "inconclusive"),
        claims=claim_records,
        iterations=iterations,
        best_iteration=best_iter or _json_dict(result_packet.get("best_iteration")),
        datasets=plan.get("datasets", []),
        baselines=plan.get("baselines", []),
        paper_outline=related,
        contributions=contributions,
        supporting_papers=supporting_papers,
        source_paper_ids=source_paper_ids,
        source_node_ids=source_node_ids,
        citation_seed_paper_ids=citation_seed_paper_ids,
        evidence_summary=str(evidence_summary),
        evidence_packet=evidence_packet,
        evidence_plan=evidence_plan,
        experimental_plan=plan,
        submission_keywords=related.get("submission_keywords", [insight.get("mechanism_type"), insight.get("resource_class")]),
        result_packet=result_packet,
        publication_evidence_contract=publication_contract,
        claim_route=claim_route,
        paper_intent=paper_intent,
        problem_awareness=problem_awareness,
        quality_gates=quality_gates,
        required_evidence=required_evidence,
        reviewer_objections=[str(x) for x in reviewer_objections if x],
    )
    completeness = audit_evidence_completeness(state.to_dict())
    state.evidence_manifest = completeness.get("evidence_manifest") or {}
    state.claim_evidence_matrix = completeness.get("claim_evidence_matrix") or []
    state.reviewer_report = completeness.get("reviewer_report") or {}
    state.method_reproducibility_requirements = completeness.get("method_reproducibility_requirements") or {}
    state.missing_evidence_report = completeness.get("missing_evidence_report") or {}
    return state


def _build_canonical_state(run: dict, insight: dict, iterations: list[dict], claims: list[dict]) -> dict:
    return build_manuscript_input_state(run, insight, iterations, claims).to_dict()


def _bundle_manifest(bundle_root: Path) -> dict:
    manifest = {"files": []}
    for path in sorted(bundle_root.rglob("*")):
        if path.is_file():
            manifest["files"].append({"path": str(path.relative_to(bundle_root)), "size": path.stat().st_size})
    return manifest


def _store_assets(manuscript_run_id: int, bundle_root: Path, bundle_format: str) -> int:
    for path in sorted(bundle_root.rglob("*")):
        if not path.is_file():
            continue
        asset_type = path.suffix.lstrip(".") or "text"
        if path.name == "cover_letter.md":
            asset_type = "cover_letter"
        elif path.name == "references.bib":
            asset_type = "bib"
        elif path.suffix == ".tex":
            asset_type = "tex"
        elif path.suffix in {".svg", ".pdf", ".png"}:
            asset_type = "figure"
        db.execute(
            """
            INSERT INTO manuscript_assets (manuscript_run_id, asset_type, label, path)
            VALUES (?, ?, ?, ?)
            """,
            (manuscript_run_id, asset_type, f"{bundle_format}:{path.name}", str(path)),
        )
    bid = db.insert_returning_id(
        """
        INSERT INTO submission_bundles (manuscript_run_id, bundle_format, status, bundle_path, manifest_path)
        VALUES (?, ?, 'ready', ?, ?)
        RETURNING id
        """,
        (manuscript_run_id, bundle_format, str(bundle_root), str(bundle_root / "artifact_manifest.json")),
    )
    db.commit()
    return bid


def generate_submission_bundle(run_id: int, bundle_formats: list[str] | None = None) -> dict:
    from agents.paper_orchestra_pipeline import generate_bundle_paper_orchestra

    return generate_bundle_paper_orchestra(run_id, bundle_formats=bundle_formats or list(SUBMISSION_BUNDLE_FORMATS))


def list_manuscripts(limit: int = 50) -> list[dict]:
    db.init_db()
    return db.fetchall(
        """
        SELECT mr.*, di.title AS insight_title, er.hypothesis_verdict
        FROM manuscript_runs mr
        LEFT JOIN deep_insights di ON di.id = mr.deep_insight_id
        LEFT JOIN experiment_runs er ON er.id = mr.experiment_run_id
        ORDER BY mr.updated_at DESC
        LIMIT ?
        """,
        (limit,),
    )
