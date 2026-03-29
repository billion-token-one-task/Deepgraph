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


def interpret_run(run_id: int) -> dict:
    """Interpret a completed experiment run and create experimental claims.

    Returns structured verdict with statistics.
    """
    run = db.fetchone("SELECT * FROM experiment_runs WHERE id=?", (run_id,))
    if not run:
        return {"error": f"Run {run_id} not found"}

    insight_id = run["deep_insight_id"]
    metric_name = run["baseline_metric_name"] or "metric"

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
    direction = criteria.get("metric_direction", "higher")

    effect = best - baseline if direction == "higher" else baseline - best
    effect_pct = (effect / abs(baseline) * 100) if baseline != 0 else 0

    p_value = _compute_p_value(kept_values, repro_values) if kept_values and repro_values else 1.0
    confidence = 1.0 - p_value

    _, repro_lo, repro_hi = _bootstrap_ci(repro_values) if repro_values else (0, 0, 0)
    _, kept_lo, kept_hi = _bootstrap_ci(kept_values) if kept_values else (0, 0, 0)

    verdict = run.get("hypothesis_verdict", "inconclusive")
    if verdict not in ("confirmed", "refuted", "inconclusive"):
        if effect > 0 and p_value < 0.05:
            verdict = "confirmed"
        elif effect <= 0 and len(test_iters) >= REFUTE_MIN:
            verdict = "refuted"
        else:
            verdict = "inconclusive"

    total_iters = len(test_iters)
    crash_count = sum(1 for t in test_iters if t["status"] == "crash")
    kept_count = sum(1 for t in test_iters if t["status"] == "keep")

    best_diff = ""
    for t in reversed(test_iters):
        if t["status"] == "keep" and t.get("code_diff"):
            best_diff = t["code_diff"]
            break

    insight = db.fetchone("SELECT title, tier FROM deep_insights WHERE id=?", (insight_id,))
    insight_title = insight["title"] if insight else f"Insight {insight_id}"

    if verdict == "confirmed":
        claim_text = (
            f"Experimental validation confirms: {insight_title}. "
            f"The proposed method achieved {metric_name}={best:.6f} vs baseline {baseline:.6f} "
            f"(effect: {effect:+.6f}, {effect_pct:+.2f}%, p={p_value:.4f}) "
            f"over {total_iters} iterations with {kept_count} improvements kept."
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
    }

    existing = db.fetchone(
        "SELECT id FROM experimental_claims WHERE run_id=? AND deep_insight_id=?",
        (run_id, insight_id))

    if existing:
        db.execute(
            """UPDATE experimental_claims
               SET claim_text=?, verdict=?, effect_size=?, confidence=?,
                   p_value=?, supporting_data=?
               WHERE id=?""",
            (claim_text, verdict, effect, confidence, p_value,
             json.dumps(supporting_data), existing["id"]))
    else:
        db.execute(
            """INSERT INTO experimental_claims
               (run_id, deep_insight_id, claim_text, claim_type, verdict,
                effect_size, confidence, p_value, supporting_data)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (run_id, insight_id, claim_text, "experimental", verdict,
             effect, confidence, p_value, json.dumps(supporting_data)))
    db.commit()

    new_status = f"experimentally_{verdict}"
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
    }
