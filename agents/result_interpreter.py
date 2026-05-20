"""Result Interpreter: parse experimental outcomes into structured verdicts.

After the validation loop completes:
1. Load all iteration data from experiment_iterations
2. Calculate effect size and statistical significance
3. Generate a structured verdict with confidence
4. Create experimental_claims for the Knowledge Loop
"""
import json
import math
import random
from pathlib import Path

from agents.benchmark_audit import (
    benchmark_diagnostic_notes,
    benchmark_fairness_warnings_from_diff,
    benchmark_semantic_warnings,
    best_iteration_benchmark_summary,
)
from contracts import ExperimentResultPacket
from db import database as db

REFUTE_MIN = 30


def _bootstrap_ci(values: list[float], n_resamples: int = 1000,
                  ci: float = 0.95) -> tuple[float, float, float]:
    """Compute bootstrap confidence interval for the mean.
    Returns (mean, lower_bound, upper_bound).
    """
    if not values:
        return 0.0, 0.0, 0.0
    if len(values) == 1:
        return values[0], values[0], values[0]

    means = []
    n = len(values)
    for _ in range(n_resamples):
        sample = [random.choice(values) for _ in range(n)]
        means.append(sum(sample) / n)

    means.sort()
    alpha = (1 - ci) / 2
    lo_idx = max(0, int(alpha * n_resamples))
    hi_idx = min(n_resamples - 1, int((1 - alpha) * n_resamples))

    return sum(values) / n, means[lo_idx], means[hi_idx]


def _compute_p_value(treatment: list[float], control: list[float]) -> float:
    """Approximate p-value using permutation test.
    Tests whether treatment mean differs from control mean.
    """
    if not treatment or not control:
        return 1.0

    observed_diff = abs(sum(treatment) / len(treatment) - sum(control) / len(control))
    combined = treatment + control
    n_treat = len(treatment)
    n_perms = min(5000, math.factorial(min(len(combined), 10)))

    count_extreme = 0
    for _ in range(5000):
        random.shuffle(combined)
        perm_treat = combined[:n_treat]
        perm_ctrl = combined[n_treat:]
        perm_diff = abs(sum(perm_treat) / len(perm_treat) - sum(perm_ctrl) / len(perm_ctrl))
        if perm_diff >= observed_diff:
            count_extreme += 1

    return count_extreme / 5000


def _iteration_evidence_label(
    metric_value: float | None,
    baseline: float,
    direction: str,
    status: str,
) -> dict:
    if metric_value is None:
        return {"baseline_effect": None, "beats_baseline": False, "evidence_label": "no_metric"}
    effect = metric_value - baseline if direction == "higher" else baseline - metric_value
    beats_baseline = effect > 1e-12
    if beats_baseline:
        label = "positive_effect"
    elif abs(effect) <= 1e-12:
        label = "baseline_tie"
    elif status == "keep":
        label = "partial_recovery"
    else:
        label = "negative_or_no_gain"
    return {
        "baseline_effect": effect,
        "beats_baseline": beats_baseline,
        "evidence_label": label,
    }


def _json_load(value, default):
    if value is None:
        return default
    if isinstance(value, (dict, list)):
        return value
    try:
        return json.loads(value)
    except (json.JSONDecodeError, TypeError):
        return default


def _iteration_record_warnings(row: dict) -> list[str]:
    warnings: list[str] = []
    description = _json_load(row.get("description"), {})
    if isinstance(description, dict):
        raw = description.get("benchmark_semantic_warnings") or []
        if isinstance(raw, list):
            warnings.extend(str(item) for item in raw if item)
    warnings.extend(benchmark_fairness_warnings_from_diff(row.get("code_diff")))
    out: list[str] = []
    for warning in warnings:
        text = str(warning or "").strip()
        if text and text not in out:
            out.append(text)
    return out


def _load_run_sidecar_json(run: dict, *relative_paths: str) -> dict:
    workdir = str(run.get("workdir") or "").strip()
    if not workdir:
        return {}
    base = Path(workdir)
    for rel in relative_paths:
        path = base / rel
        if not path.exists():
            continue
        try:
            return _json_load(path.read_text(encoding="utf-8"), {})
        except OSError:
            continue
    return {}


def _load_benchmark_summary(run: dict) -> dict:
    return _load_run_sidecar_json(run, "results/benchmark_summary.json")


def _load_benchmark_artifact_manifest(run: dict) -> dict:
    return _load_run_sidecar_json(run, "results/benchmark_artifact_manifest.json")


def _benchmark_package_complete(
    benchmark_summary: dict,
    artifact_manifest: dict,
    *,
    minimum_seeds: int = 3,
) -> bool:
    per_method = benchmark_summary.get("per_method") if isinstance(benchmark_summary.get("per_method"), dict) else {}
    seed_results = benchmark_summary.get("seed_results") if isinstance(benchmark_summary.get("seed_results"), list) else []
    try:
        num_seeds = int(benchmark_summary.get("num_seeds") or len(seed_results) or 0)
    except (TypeError, ValueError):
        num_seeds = 0
    return bool(
        artifact_manifest.get("full_benchmark_completed")
        and per_method
        and len(per_method) >= 2
        and num_seeds >= minimum_seeds
    )


def _record_result_packet(run_id: int, workdir: str | None, packet: ExperimentResultPacket) -> str | None:
    if not workdir:
        return None
    out_dir = Path(workdir) / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "experiment_result_packet.json"
    path.write_text(json.dumps(packet.to_dict(), indent=2), encoding="utf-8")
    db.execute(
        """
        INSERT INTO experiment_artifacts (run_id, artifact_type, path, metric_key, metric_value, metadata)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (
            run_id,
            "source_data",
            str(path),
            packet.metric_name,
            packet.best,
            json.dumps({"contract_type": "ExperimentResultPacket"}),
        ),
    )
    db.commit()
    return str(path)


def interpret_run(run_id: int) -> dict:
    """Interpret a completed experiment run and create experimental claims.

    Returns structured verdict with statistics.
    """
    run = db.fetchone("SELECT * FROM experiment_runs WHERE id=?", (run_id,))
    if not run:
        return {"error": f"Run {run_id} not found"}

    insight_id = run["deep_insight_id"]
    proxy = _json_load(run.get("proxy_config"), {}) or _load_run_sidecar_json(
        run, "spec/proxy_config.json", "proxy_config.json"
    )
    benchmark_summary = _load_benchmark_summary(run)
    benchmark_artifact_manifest = _load_benchmark_artifact_manifest(run)

    repro_iters = db.fetchall(
        """SELECT metric_value FROM experiment_iterations
           WHERE run_id=? AND phase='reproduction' AND metric_value IS NOT NULL
           ORDER BY iteration_number""",
        (run_id,))
    repro_values = [r["metric_value"] for r in repro_iters]

    test_iters = db.fetchall(
        """SELECT iteration_number, metric_value, status, description, code_diff
           FROM experiment_iterations
           WHERE run_id=? AND phase='hypothesis_testing'
           ORDER BY iteration_number""",
        (run_id,))

    kept_values = [t["metric_value"] for t in test_iters
                   if t["status"] == "keep" and t["metric_value"] is not None]
    all_test_values = [t["metric_value"] for t in test_iters
                       if t["metric_value"] is not None]

    baseline = run["baseline_metric_value"] or (sum(repro_values) / len(repro_values) if repro_values else 0)
    best = run["best_metric_value"] or (max(kept_values) if kept_values else baseline)

    criteria_raw = run.get("success_criteria", "{}")
    try:
        criteria = json.loads(criteria_raw) if isinstance(criteria_raw, str) else criteria_raw or {}
    except (json.JSONDecodeError, TypeError):
        criteria = {}
    if not criteria:
        criteria = _load_run_sidecar_json(run, "spec/success_criteria.json", "success_criteria.json")
    metric_name = run["baseline_metric_name"] or criteria.get("metric_name") or "metric"
    direction = criteria.get("metric_direction", "higher")
    best_summary = best_iteration_benchmark_summary(
        run.get("workdir"),
        best_metric=best,
        direction=direction,
    )
    if best_summary:
        benchmark_summary = best_summary
    publication_contract = (
        criteria.get("publication_evidence_contract")
        or criteria.get("publication_evidence")
        or proxy.get("publication_evidence_contract")
        or {}
    )
    if not isinstance(publication_contract, dict):
        publication_contract = {}
    evidence_tier = (
        criteria.get("evidence_tier")
        or publication_contract.get("evidence_tier")
        or proxy.get("evidence_tier")
        or ""
    )
    publication_ready = criteria.get("publication_ready")
    if publication_ready is None:
        publication_ready = publication_contract.get("publication_ready")
    blocks_manuscript = bool(
        criteria.get("blocks_manuscript")
        or publication_contract.get("blocks_manuscript")
        or proxy.get("blocks_manuscript")
    )
    paper_intent = (
        criteria.get("paper_intent")
        or publication_contract.get("paper_intent")
        or proxy.get("paper_intent")
        or {}
    )
    if not isinstance(paper_intent, dict):
        paper_intent = {}
    problem_awareness = (
        criteria.get("problem_awareness")
        or publication_contract.get("problem_awareness")
        or paper_intent.get("problem_awareness")
        or proxy.get("problem_awareness")
        or {}
    )
    if not isinstance(problem_awareness, dict):
        problem_awareness = {}
    quality_gates = criteria.get("quality_gates") or publication_contract.get("quality_gates") or {}
    if not isinstance(quality_gates, dict):
        quality_gates = {}
    claim_route = (
        criteria.get("claim_route")
        or publication_contract.get("claim_route")
        or proxy.get("claim_route")
        or {}
    )
    if not isinstance(claim_route, dict):
        claim_route = {}
    reviewer_objections = criteria.get("reviewer_objections") or publication_contract.get("reviewer_objections") or []
    if not isinstance(reviewer_objections, list):
        reviewer_objections = []
    try:
        minimum_seeds = int(
            publication_contract.get("minimum_seeds")
            or quality_gates.get("minimum_seeds")
            or 3
        )
    except (TypeError, ValueError):
        minimum_seeds = 3
    full_benchmark_completed = _benchmark_package_complete(
        benchmark_summary,
        benchmark_artifact_manifest,
        minimum_seeds=minimum_seeds,
    )
    semantic_warnings = benchmark_semantic_warnings(
        benchmark_summary,
        metric_name=metric_name,
        candidate_method=str(benchmark_summary.get("candidate_method") or ""),
        direction=direction,
    )
    diagnostic_notes = benchmark_diagnostic_notes(
        benchmark_summary,
        metric_name=metric_name,
        candidate_method=str(benchmark_summary.get("candidate_method") or ""),
        direction=direction,
    )
    for note in reversed(diagnostic_notes):
        objection = f"Benchmark diagnostic note: {note}"
        if objection not in reviewer_objections:
            reviewer_objections.insert(0, objection)
    for row in test_iters:
        if row.get("status") == "keep":
            for warning in _iteration_record_warnings(row):
                if warning not in semantic_warnings:
                    semantic_warnings.append(warning)
    if semantic_warnings:
        full_benchmark_completed = False
        blocks_manuscript = True
        for warning in reversed(semantic_warnings):
            objection = f"Benchmark semantic warning: {warning}"
            if objection not in reviewer_objections:
                reviewer_objections.insert(0, objection)

    effect = best - baseline if direction == "higher" else baseline - best
    effect_pct = (effect / abs(baseline) * 100) if baseline != 0 else 0

    p_value = _compute_p_value(kept_values, repro_values) if kept_values and repro_values else 1.0
    confidence = 1.0 - p_value

    _, repro_lo, repro_hi = _bootstrap_ci(repro_values) if repro_values else (0, 0, 0)
    _, kept_lo, kept_hi = _bootstrap_ci(kept_values) if kept_values else (0, 0, 0)

    verdict = run.get("hypothesis_verdict", "inconclusive")
    if verdict not in ("confirmed", "refuted", "inconclusive", "reproduced"):
        if effect > 0 and p_value < 0.05:
            verdict = "confirmed"
        elif effect <= 0 and len(test_iters) >= REFUTE_MIN:
            verdict = "refuted"
        else:
            verdict = "inconclusive"
    benchmark_required = bool(
        str(evidence_tier or "").strip().lower() == "benchmark_plan"
        or quality_gates.get("requires_full_benchmark_package")
        or publication_contract.get("required_real_benchmarks")
        or quality_gates.get("has_real_benchmark")
    )
    if benchmark_required and (verdict != "confirmed" or not full_benchmark_completed):
        blocks_manuscript = True
        if "Full benchmark artifact package is required before manuscript generation." not in reviewer_objections:
            reviewer_objections.insert(
                0,
                "Full benchmark artifact package is required before manuscript generation.",
            )

    total_iters = len(test_iters)
    crash_count = sum(1 for t in test_iters if t["status"] == "crash")
    kept_count = sum(1 for t in test_iters if t["status"] == "keep")

    best_diff = ""
    for t in reversed(test_iters):
        if t["status"] == "keep" and t.get("code_diff"):
            best_diff = t["code_diff"]
            break

    insight = db.fetchone(
        """
        SELECT title, tier, source_paper_ids, supporting_papers, source_node_ids
        FROM deep_insights WHERE id=?
        """,
        (insight_id,),
    )
    insight_title = insight["title"] if insight else f"Insight {insight_id}"

    if verdict == "confirmed":
        claim_text = (
            f"Experimental validation confirms: {insight_title}. "
            f"The proposed method achieved {metric_name}={best:.6f} vs baseline {baseline:.6f} "
            f"(effect: {effect:+.6f}, {effect_pct:+.2f}%, p={p_value:.4f}) "
            f"over {total_iters} iterations with {kept_count} improvements kept."
        )
    elif verdict == "reproduced":
        claim_text = (
            f"Formal reproduction completed for: {insight_title}. "
            f"The configured baseline executed successfully with {metric_name}={baseline:.6f}, "
            f"but no hypothesis-testing iterations were run, so this result should not be "
            f"interpreted as experimental confirmation."
        )
    elif verdict == "refuted":
        claim_text = (
            f"Experimental validation refutes: {insight_title}. "
            f"After {total_iters} iterations, the proposed method could not improve "
            f"beyond baseline {metric_name}={baseline:.6f}. "
            f"Best achieved: {best:.6f} (effect: {effect:+.6f}, {effect_pct:+.2f}%)."
        )
    else:
        claim_text = (
            f"Experimental validation inconclusive for: {insight_title}. "
            f"Baseline {metric_name}={baseline:.6f}, best={best:.6f} "
            f"(effect: {effect:+.6f}, p={p_value:.4f}). "
            f"Insufficient evidence after {total_iters} iterations."
        )

    supporting_data = {
        "baseline": baseline,
        "best": best,
        "effect_size": effect,
        "effect_pct": effect_pct,
        "p_value": p_value,
        "confidence": confidence,
        "total_iterations": total_iters,
        "kept_iterations": kept_count,
        "crash_count": crash_count,
        "repro_ci": [repro_lo, repro_hi],
        "kept_ci": [kept_lo, kept_hi],
        "best_code_diff": best_diff[:3000],
        "metric_name": metric_name,
        "direction": direction,
        "source_paper_ids": json.loads(insight.get("source_paper_ids") or "[]") if insight.get("source_paper_ids") else json.loads(insight.get("supporting_papers") or "[]") if insight.get("supporting_papers") else [],
        "source_node_ids": json.loads(insight.get("source_node_ids") or "[]") if insight.get("source_node_ids") else [],
        "benchmark_summary": benchmark_summary,
        "benchmark_artifact_manifest": benchmark_artifact_manifest,
        "evidence_tier": evidence_tier,
        "publication_ready": publication_ready,
        "blocks_manuscript": blocks_manuscript,
        "full_benchmark_completed": full_benchmark_completed,
        "publication_evidence_contract": publication_contract,
        "claim_route": claim_route,
        "paper_intent": paper_intent,
        "problem_awareness": problem_awareness,
        "quality_gates": quality_gates,
        "reviewer_objections": reviewer_objections,
        "benchmark_semantic_warnings": semantic_warnings,
        "benchmark_diagnostic_notes": diagnostic_notes,
    }
    result_packet = ExperimentResultPacket(
        run_id=run_id,
        deep_insight_id=insight_id,
        formal_experiment=bool(proxy.get("formal_experiment")),
        smoke_test_only=bool(proxy.get("smoke_test_only")),
        metric_name=metric_name,
        metric_direction=direction,
        verdict=verdict,
        baseline=baseline,
        best=best,
        effect_size=effect,
        effect_pct=effect_pct,
        p_value=p_value,
        confidence=confidence,
        total_iterations=total_iters,
        kept_iterations=kept_count,
        crash_count=crash_count,
        reproduction_metrics=repro_values,
        hypothesis_iterations=[
            {
                "iteration_number": row["iteration_number"],
                "metric_value": row["metric_value"],
                "status": row["status"],
                "description": row["description"],
                "benchmark_semantic_warnings": _iteration_record_warnings(row),
                **_iteration_evidence_label(
                    row["metric_value"],
                    baseline,
                    direction,
                    row["status"],
                ),
            }
            for row in test_iters
        ],
        best_iteration={
            "metric_value": best,
            "code_diff": best_diff[:3000],
        },
        claim_text=claim_text,
        source_paper_ids=supporting_data.get("source_paper_ids") or [],
        source_node_ids=supporting_data.get("source_node_ids") or [],
        benchmark_summary=benchmark_summary,
        evidence_tier=str(evidence_tier or ""),
        publication_ready=publication_ready if isinstance(publication_ready, bool) else None,
        blocks_manuscript=blocks_manuscript,
        full_benchmark_completed=full_benchmark_completed,
        benchmark_artifact_manifest=benchmark_artifact_manifest,
        publication_evidence_contract=publication_contract,
        claim_route=claim_route,
        paper_intent=paper_intent,
        problem_awareness=problem_awareness,
        quality_gates=quality_gates,
        reviewer_objections=[str(x) for x in reviewer_objections if x],
        benchmark_semantic_warnings=semantic_warnings,
        benchmark_diagnostic_notes=diagnostic_notes,
        artifact_paths={
            "artifact_manifest": str(Path(run.get("workdir") or "") / "results" / "benchmark_artifact_manifest.json")
            if benchmark_artifact_manifest and run.get("workdir")
            else ""
        },
    )
    packet_path = _record_result_packet(run_id, run.get("workdir"), result_packet)
    if packet_path:
        result_packet.artifact_paths["result_packet"] = packet_path
    supporting_data["result_packet"] = result_packet.to_dict()
    source_paper_ids = json.dumps(supporting_data.get("source_paper_ids") or [])
    source_node_ids = json.dumps(supporting_data.get("source_node_ids") or [])

    existing = db.fetchone(
        "SELECT id FROM experimental_claims WHERE run_id=? AND deep_insight_id=?",
        (run_id, insight_id))

    if existing:
        db.execute(
            """UPDATE experimental_claims
               SET claim_text=?, verdict=?, effect_size=?, confidence=?,
                   p_value=?, supporting_data=?, source_paper_ids=?, source_node_ids=?
               WHERE id=?""",
            (claim_text, verdict, effect, confidence, p_value,
             json.dumps(supporting_data), source_paper_ids, source_node_ids, existing["id"]))
    else:
        db.execute(
            """INSERT INTO experimental_claims
               (run_id, deep_insight_id, claim_text, claim_type, verdict,
                effect_size, confidence, p_value, supporting_data, source_paper_ids, source_node_ids)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (run_id, insight_id, claim_text, "experimental", verdict,
             effect, confidence, p_value, json.dumps(supporting_data), source_paper_ids, source_node_ids))
    db.commit()

    new_status = "experimentally_reproduced" if verdict == "reproduced" else f"experimentally_{verdict}"
    db.execute(
        "UPDATE deep_insights SET status=?, updated_at=CURRENT_TIMESTAMP WHERE id=?",
        (new_status, insight_id))
    db.execute(
        "UPDATE experiment_runs SET hypothesis_verdict=?, effect_size=?, effect_pct=? WHERE id=?",
        (verdict, effect, effect_pct, run_id))
    db.commit()

    print(f"[INTERPRET] Run {run_id}: {verdict} (effect={effect:+.6f}, p={p_value:.4f}, "
          f"confidence={confidence:.4f})", flush=True)

    return {
        "run_id": run_id,
        "insight_id": insight_id,
        "verdict": verdict,
        "claim_text": claim_text,
        "baseline": baseline,
        "best": best,
        "effect_size": effect,
        "effect_pct": effect_pct,
        "p_value": p_value,
        "confidence": confidence,
        "total_iterations": total_iters,
        "kept_count": kept_count,
        "result_packet": result_packet.to_dict(),
    }
