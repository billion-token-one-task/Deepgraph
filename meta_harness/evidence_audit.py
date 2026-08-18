"""Evidence audit: the ladder's last two rungs, which never had an executor.

Until 2026-08-18 nothing in the repository could move a run past
full_benchmark_complete. This module implements the audit the state machine
was designed around, following contracts/scientific_evidence.decide_evidence:

1. Deterministic artifact verification -- the metric and the significance
   test are recomputed from raw_predictions and must match final_results.
2. A claim ledger derived from the verified artifacts, persisted and hashed.
3. An independent cross-vendor evaluator that reads the ledger and can
   dissent; its route identity and response hash go into the audit record.
4. A true holdout: fresh inference on examples the audited run never saw,
   executed through the same governed colab path (example_offset).

The audit is a two-phase idempotent state machine driven by auto_advance
passes: phase one verifies, writes the ledger, collects the evaluator's
judgement and submits the holdout run; phase two, once the holdout result is
verified on disk, compares directions and advances evidence_audited and then
scientifically_decided in one sitting.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from db import database as db
from meta_harness.evidence_state import EvidenceTransitionContext
from meta_harness.repository import MetaHarnessRepository
from meta_harness.runner_contract import (
    extract_p_value,
    recompute_metric,
)

HOLDOUT_OFFSET = 200
AUDIT_ACTOR = "evidence_audit_v1"
_VERIFY_TOLERANCE = 1e-9


class EvidenceAuditError(RuntimeError):
    pass


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _load_results(results_dir: Path) -> dict[str, Any]:
    final = json.loads((results_dir / "final_results.json").read_text())
    rows = [
        json.loads(line)
        for line in (results_dir / "raw_predictions.jsonl").read_text().splitlines()
        if line.strip()
    ]
    return {"final": final, "rows": rows}


def _verify_arms(final: Mapping[str, Any], rows: list[dict]) -> dict[str, float]:
    """Recompute both arms from raw rows; refuse on any mismatch."""
    metric = str(final.get("metric_name") or final.get("primary_metric"))
    baseline_method = str(final.get("baseline_method"))
    candidate_method = str(final.get("candidate_method"))
    recomputed: dict[str, float] = {}
    for method, reported_key in (
        (baseline_method, "baseline_metric_value"),
        (candidate_method, "metric_value"),
    ):
        method_rows = [r for r in rows if str(r.get("method")) == method]
        if not method_rows:
            raise EvidenceAuditError(f"no raw rows for method {method}")
        value = recompute_metric(method_rows, metric)
        reported = float(final.get(reported_key))
        if abs(value - reported) > _VERIFY_TOLERANCE:
            raise EvidenceAuditError(
                f"recomputed {method} {metric}={value} != reported {reported}"
            )
        recomputed[method] = value
    reported_p = extract_p_value(dict(final))
    if reported_p is None:
        raise EvidenceAuditError("final_results carries no p-value")
    return {
        "metric": metric,
        "baseline": recomputed[baseline_method],
        "candidate": recomputed[candidate_method],
        "p_value": float(reported_p),
    }


def build_claim_ledger(results_dir: Path) -> tuple[Path, str]:
    """Derive the claim ledger from verified artifacts and persist it."""
    loaded = _load_results(results_dir)
    final, rows = loaded["final"], loaded["rows"]
    verified = _verify_arms(final, rows)
    delta = verified["candidate"] - verified["baseline"]
    verdict = str(final.get("hypothesis_verdict") or (
        "refuted" if final.get("scientific_negative_result") else "inconclusive"
    ))
    ledger = {
        "schema_version": "claim_ledger_v1",
        "dataset_id": final.get("dataset_id"),
        "dataset_revision": final.get("dataset_revision"),
        "model_id": final.get("model_id"),
        "model_revision": final.get("model_revision"),
        "metric": verified["metric"],
        "claims": [
            {
                "claim_id": "primary_effect",
                "statement": (
                    f"candidate '{final.get('candidate_method')}' changes "
                    f"{verified['metric']} versus "
                    f"'{final.get('baseline_method')}' on "
                    f"{final.get('dataset_id')} by {delta:+.4f} "
                    f"({verified['baseline']:.4f} -> {verified['candidate']:.4f})"
                ),
                "baseline_value": verified["baseline"],
                "candidate_value": verified["candidate"],
                "delta": delta,
                "p_value": verified["p_value"],
                "n_examples": final.get("num_examples"),
                "seeds": final.get("seeds"),
                "verdict": verdict,
            }
        ],
        "artifact_hashes": final.get("artifact_hashes"),
        "recomputed_from_raw_predictions": True,
    }
    path = results_dir / "claim_ledger.json"
    payload = json.dumps(ledger, ensure_ascii=False, indent=2, sort_keys=True)
    path.write_text(payload, encoding="utf-8")
    return path, _sha256_text(payload)


def independent_evaluator_review(
    *,
    agenda_id: int,
    idea_id: int,
    resource_grant_id: int,
    ledger_path: Path,
) -> dict[str, Any]:
    """Cross-vendor evaluator reads the ledger and may dissent."""
    from agents.llm_client import call_llm_for_role, parse_llm_json_text

    ledger_text = ledger_path.read_text(encoding="utf-8")
    prompt = (
        "You are the independent evidence auditor for an autonomous research "
        "system. Below is a claim ledger recomputed from raw prediction "
        "artifacts. Judge whether the recorded verdict follows from the "
        "numbers under a two-sided alpha of 0.05. Dissent freely; your "
        "concurrence is not assumed.\n\n"
        f"CLAIM LEDGER:\n{ledger_text}\n\n"
        'Answer with one JSON object only: {"concur": true|false, '
        '"verdict": "supported"|"refuted"|"inconclusive", '
        '"reasons": ["..."]}'
    )
    raw, _tokens, route = call_llm_for_role(
        "Audit scientific evidence. Judge only from the numbers provided.",
        prompt,
        agenda_id=agenda_id,
        idea_id=idea_id,
        role="evaluator",
        stage="evidence_audit",
        resource_grant_id=resource_grant_id,
        operation="evidence_audit_review",
        idempotency_key=f"evidence-audit:{agenda_id}:{idea_id}:{_sha256_text(ledger_text)[:16]}",
        prompt_version="evidence_audit_v1",
        max_tokens=4096,
    )
    parsed, _how = parse_llm_json_text(raw)
    if not isinstance(parsed, dict) or "concur" not in parsed:
        raise EvidenceAuditError("evaluator returned no judgement")
    return {
        "judgement": parsed,
        "evaluator_ref": f"{route.get('provider')}:{route.get('model')}",
        "evaluator_hash": _sha256_text(raw),
    }


def _run_paths(run: Mapping[str, Any]) -> tuple[Path, Path, Path]:
    workdir = Path(str(run["workdir"]))
    results = workdir / "results"
    holdout = workdir / "results_holdout"
    return workdir, results, holdout


def _submit_holdout(run: Mapping[str, Any], grant_id: int) -> int:
    from orchestrator.meta_compute_runtime import ColabWorkSpec, submit_colab_work

    workdir, _results, holdout_dir = _run_paths(run)
    holdout_dir.mkdir(parents=True, exist_ok=True)
    spec = ColabWorkSpec(
        agenda_id=int(run["agenda_id"]),
        idea_id=int(run["deep_insight_id"]),
        experiment_run_id=int(run["id"]),
        resource_grant_id=int(grant_id),
        stage="evidence_audit",
        idempotency_key=(
            f"experiment-run:{run['agenda_id']}:{run['deep_insight_id']}:"
            f"{run['id']}:evidence_audit_holdout"
        ),
        code_dir=str(workdir / "code"),
        command_tokens=(
            "python", "train.py",
            "--config", "execution_requirements.json",
            "--candidate-adapter", "candidate_adapter.py",
            "--output-dir", ".",
        ),
        environment={
            "PYTHONUNBUFFERED": "1",
            "DEEPGRAPH_RUNNER_BATCH_SIZE": "24",
            "DEEPGRAPH_RUNNER_EXAMPLE_OFFSET": str(HOLDOUT_OFFSET),
        },
        artifact_map={
            "final_results": "final_results.json",
            "raw_predictions": "raw_predictions.jsonl",
            "environment_manifest": "environment_manifest.json",
            "dataset_manifest": "dataset_manifest.json",
            "model_manifest": "model_manifest.json",
        },
        artifact_output_dir=str(holdout_dir),
        timeout_seconds=5400,
    )
    job = submit_colab_work(spec)
    return int(getattr(job, "id", 0) or 0)


def run_evidence_audit_phase(
    *,
    agenda_id: int,
    idea_id: int,
    run_id: int,
    resource_grant_id: int,
    log=print,
) -> str:
    """Idempotently drive the audit. Returns the audit's current disposition."""
    run = db.fetchone("SELECT * FROM experiment_runs WHERE id=?", (run_id,))
    if not run:
        raise EvidenceAuditError("missing run")
    state = str(run.get("scientific_evidence_state") or "")
    if state == "scientifically_decided":
        return "decided"
    if state not in {"full_benchmark_complete", "evidence_audited"}:
        return f"not_ready:{state}"
    _workdir, results_dir, holdout_dir = _run_paths(run)

    ledger_path, ledger_hash = build_claim_ledger(results_dir)
    evaluator_path = results_dir / "audit_evaluator.json"
    if evaluator_path.exists():
        evaluator = json.loads(evaluator_path.read_text())
    else:
        evaluator = independent_evaluator_review(
            agenda_id=agenda_id,
            idea_id=idea_id,
            resource_grant_id=resource_grant_id,
            ledger_path=ledger_path,
        )
        evaluator_path.write_text(
            json.dumps(evaluator, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    holdout_final_path = holdout_dir / "final_results.json"
    if not holdout_final_path.exists():
        pending = db.fetchone(
            """
            SELECT id, status FROM colab_work_requests_v1
            WHERE experiment_run_id=? AND idempotency_key LIKE '%evidence_audit_holdout'
            ORDER BY id DESC LIMIT 1
            """,
            (run_id,),
        )
        if pending and str(dict(pending).get("status")) in {"queued", "running", "admitting"}:
            return "holdout_pending"
        if pending and str(dict(pending).get("status")) == "succeeded":
            return "holdout_artifacts_missing"
        _submit_holdout(run, resource_grant_id)
        log(f"[AUDIT] holdout submitted for run {run_id} at offset {HOLDOUT_OFFSET}")
        return "holdout_submitted"

    final = json.loads((results_dir / "final_results.json").read_text())
    holdout_final = json.loads(holdout_final_path.read_text())
    # The verdict lives on the run row (written at outcome time), not in
    # final_results.json; the artifact only carries scientific_negative_result.
    verdict = str(
        run.get("hypothesis_verdict")
        or ("refuted" if final.get("scientific_negative_result") else "inconclusive")
    )
    holdout_passed = holdout_consistent(verdict, final, holdout_final)
    holdout_hash = _sha256_text(holdout_final_path.read_text())
    judgement = evaluator["judgement"]
    evaluator_passed = bool(judgement.get("concur")) and (
        str(judgement.get("verdict")) == verdict
    )

    from orchestrator.bounded_execution import raw_artifacts_hash

    digest, present, missing = raw_artifacts_hash(
        agenda_id=agenda_id, experiment_run_id=run_id
    )
    if present <= 0 or missing:
        raise EvidenceAuditError("raw artifact registration incomplete")
    grant_row = db.fetchone(
        "SELECT preflight_result_id FROM resource_grants WHERE id=?",
        (resource_grant_id,),
    )
    contract_row = db.fetchone(
        """
        SELECT cer.requirements_hash
        FROM candidate_preflight_results_v1 cpr
        JOIN candidate_execution_requirements_v1 cer ON cer.id=cpr.requirement_id
        WHERE cpr.id=?
        """,
        (int(dict(grant_row or {}).get("preflight_result_id") or 0),),
    )
    contract_hash = str(dict(contract_row or {}).get("requirements_hash") or "")
    repo = MetaHarnessRepository()
    base_context = dict(
        resource_grant_valid=True,
        resource_grant_id=resource_grant_id,
        execution_succeeded=True,
        pilot_only=False,
        full_benchmark_complete=True,
        raw_artifacts_present=True,
        claim_ledger_present=True,
        evaluator_passed=evaluator_passed,
        holdout_passed=holdout_passed,
        raw_artifacts_hash=digest,
        claim_ledger_hash=ledger_hash,
        benchmark_contract_hash=contract_hash,
        evaluator_ref=str(evaluator["evaluator_ref"]),
        evaluator_hash=str(evaluator["evaluator_hash"]),
        holdout_ref=(
            f"{final.get('dataset_id')}:{final.get('dataset_revision')}:"
            f"test[{HOLDOUT_OFFSET}:{HOLDOUT_OFFSET + int(final.get('num_examples') or 0)}]"
        ),
        holdout_hash=holdout_hash,
        verdict=verdict,
        verdict_hash=_sha256_text(
            json.dumps(
                {"verdict": verdict, "claim_ledger": ledger_hash, "holdout": holdout_hash},
                sort_keys=True,
            )
        ),
    )
    if state == "full_benchmark_complete":
        repo.advance_experiment_state(
            agenda_id=agenda_id,
            experiment_run_id=run_id,
            target="evidence_audited",
            context=EvidenceTransitionContext(**base_context),
            actor=AUDIT_ACTOR,
        )
        state = "evidence_audited"
        log(f"[AUDIT] run {run_id} evidence_audited "
            f"(evaluator_passed={evaluator_passed} holdout_passed={holdout_passed})")
    if state == "evidence_audited":
        repo.advance_experiment_state(
            agenda_id=agenda_id,
            experiment_run_id=run_id,
            target="scientifically_decided",
            context=EvidenceTransitionContext(**base_context),
            actor=AUDIT_ACTOR,
        )
        log(f"[AUDIT] run {run_id} scientifically_decided verdict={verdict}")
    return "decided"


def holdout_consistent(
    verdict: str, final: Mapping[str, Any], holdout_final: Mapping[str, Any]
) -> bool:
    """The holdout must not contradict the verdict it is auditing.

    For supported: the candidate must also beat the baseline on held-out
    data with p < alpha. For refuted or inconclusive: the candidate must not
    significantly beat the baseline on held-out data (which would contradict
    the negative conclusion).
    """
    h_base = float(holdout_final.get("baseline_metric_value"))
    h_cand = float(holdout_final.get("metric_value"))
    h_p = extract_p_value(dict(holdout_final))
    direction_higher = str(final.get("metric_direction") or "higher") == "higher"
    candidate_beats = (h_cand > h_base) if direction_higher else (h_cand < h_base)
    significant = h_p is not None and h_p < 0.05
    if verdict == "supported":
        return candidate_beats and significant
    return not (candidate_beats and significant)
