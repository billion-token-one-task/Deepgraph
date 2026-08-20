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
from meta_harness.failure_policy import is_transport_class_failure
from meta_harness.repository import MetaHarnessRepository
from meta_harness.runner_contract import (
    extract_p_value,
    recompute_metric,
)

HOLDOUT_OFFSET = 200
AUDIT_ACTOR = "evidence_audit_v1"
# Bumped when the evaluator prompt's semantics change; a cached judgement
# made under an older prompt is re-collected rather than trusted.
AUDIT_EVALUATOR_PROMPT_REF = "evidence_audit_evaluator_v2"
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


def significance_aware_verdict(final: Mapping[str, Any]) -> str:
    """The run's verdict, honouring its own p-value.

    "refuted" is a scientific claim and carries the same evidential burden as
    "supported". Derived from the direction alone it overstated runs 164
    (delta -0.03, p=0.506) and 180 (delta -0.06, p=0.071); the cross-vendor
    evaluator dissented on run 189 (p=0.220) for exactly this reason, which
    is the check working as designed.
    """
    recorded = str(final.get("hypothesis_verdict") or "").strip()
    if recorded in {"supported", "refuted", "inconclusive"}:
        return recorded
    p_value = extract_p_value(dict(final))
    if p_value is None or float(p_value) >= 0.05:
        return "inconclusive"
    return "refuted" if final.get("scientific_negative_result") else "supported"


def build_claim_ledger(results_dir: Path) -> tuple[Path, str]:
    """Derive the claim ledger from verified artifacts and persist it."""
    loaded = _load_results(results_dir)
    final, rows = loaded["final"], loaded["rows"]
    verified = _verify_arms(final, rows)
    delta = verified["candidate"] - verified["baseline"]
    verdict = significance_aware_verdict(final)
    direction = str(final.get("metric_direction") or "higher")
    ledger = {
        "schema_version": "claim_ledger_v1",
        "dataset_id": final.get("dataset_id"),
        "dataset_revision": final.get("dataset_revision"),
        "model_id": final.get("model_id"),
        "model_revision": final.get("model_revision"),
        "metric": verified["metric"],
        "metric_direction": direction,
        "hypothesis": (
            f"the candidate IMPROVES {verified['metric']} "
            f"({direction} is better) versus the baseline"
        ),
        "claims": [
            {
                "claim_id": "primary_effect",
                "statement": (
                    f"candidate '{final.get('candidate_method')}' changes "
                    f"{verified['metric']} versus "
                    f"'{final.get('baseline_method')}' on "
                    f"{final.get('dataset_id')} by {delta:+.4f} "
                    f"({verified['baseline']:.4f} -> {verified['candidate']:.4f}; "
                    f"{direction} is better)"
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


MAX_EVALUATOR_ATTEMPTS = 3


def _evaluator_attempt(resource_grant_id: int) -> int:
    """How many evaluator calls this grant has already paid for."""
    try:
        row = db.fetchone(
            """
            SELECT COUNT(*) AS n FROM resource_grant_usage_reservations
            WHERE resource_grant_id=? AND operation='evidence_audit_review'
            """,
            (int(resource_grant_id),),
        )
    except Exception:
        return 0
    return int((row or {}).get("n") or 0)


def independent_evaluator_review(
    *,
    agenda_id: int,
    idea_id: int,
    resource_grant_id: int,
    ledger_path: Path,
) -> dict[str, Any]:
    """Cross-vendor evaluator reads the ledger and may dissent."""
    from agents.llm_client import (
        call_llm_for_role,
        configured_role_prompt_version,
        parse_llm_json_text,
    )

    ledger_text = ledger_path.read_text(encoding="utf-8")
    prompt = (
        "You are the independent evidence auditor for an autonomous research "
        "system. Below is a claim ledger recomputed from raw prediction "
        "artifacts. The preregistered hypothesis is DIRECTIONAL: the "
        "candidate method is claimed to IMPROVE the metric in the stated "
        "metric_direction. Verdict semantics:\n"
        "- supported: the candidate is significantly BETTER than the "
        "baseline in the preferred direction (p < 0.05).\n"
        "- refuted: the improvement hypothesis is rejected -- the candidate "
        "is significantly WORSE, or the measured difference shows no "
        "improvement (a significant harm still means refuted, never "
        "supported).\n"
        "- inconclusive: the measurement cannot decide either way.\n"
        "Judge whether the recorded verdict follows from the numbers under "
        "a two-sided alpha of 0.05. Dissent freely; your concurrence is not "
        "assumed.\n\n"
        f"CLAIM LEDGER:\n{ledger_text}\n\n"
        'Answer with one JSON object only: {"concur": true|false, '
        '"verdict": "supported"|"refuted"|"inconclusive", '
        '"reasons": ["..."]}'
    )
    attempt = _evaluator_attempt(resource_grant_id)
    if attempt >= MAX_EVALUATOR_ATTEMPTS:
        raise EvidenceAuditError("evaluator attempts exhausted")
    raw, _tokens, route = call_llm_for_role(
        "Audit scientific evidence. Judge only from the numbers provided.",
        prompt,
        agenda_id=agenda_id,
        idea_id=idea_id,
        role="evaluator",
        stage="evidence_audit",
        resource_grant_id=resource_grant_id,
        operation="evidence_audit_review",
        # The key carries the attempt number. A response that settles its
        # reservation and then fails to parse used to strand the run for
        # good: run 191 spent 4913 tokens, raised "evaluator returned no
        # judgement" before anything was written, and every retry was then
        # refused with "idempotency key already exists with status settled"
        # (2026-08-20). Paying again is the honest cost of having lost the
        # first answer, and MAX_EVALUATOR_ATTEMPTS bounds it.
        idempotency_key=(
            f"evidence-audit:{agenda_id}:{idea_id}:"
            f"{_sha256_text(ledger_text)[:16]}:{attempt}"
        ),
        prompt_version=configured_role_prompt_version("evaluator"),
        max_tokens=4096,
    )
    parsed, _how = parse_llm_json_text(raw)
    if not isinstance(parsed, dict) or "concur" not in parsed:
        # Keep what was paid for, so the next attempt is diagnosable rather
        # than a second blind call into the same failure.
        try:
            (ledger_path.parent / f"audit_evaluator_unparsed_{attempt}.txt").write_text(
                str(raw), encoding="utf-8"
            )
        except Exception:
            pass
        raise EvidenceAuditError("evaluator returned no judgement")
    return {
        "judgement": parsed,
        "evaluator_ref": f"{route.get('provider')}:{route.get('model')}",
        "evaluator_hash": _sha256_text(raw),
        "prompt_ref": AUDIT_EVALUATOR_PROMPT_REF,
        # The judgement is about THIS ledger. "concur" is an answer to the
        # verdict the ledger carried at the time, so a cached judgement is
        # only reusable while the ledger it judged is unchanged.
        "ledger_hash": _sha256_text(ledger_text),
    }


def _run_paths(run: Mapping[str, Any]) -> tuple[Path, Path, Path]:
    workdir = Path(str(run["workdir"]))
    results = workdir / "results"
    holdout = workdir / "results_holdout"
    return workdir, results, holdout


MAX_HOLDOUT_ATTEMPTS = 3
# A transport death measured nothing, so it buys a separate, larger budget:
# the science retry cap stays 3, but infrastructure may fail more often than
# that without condemning the run.
MAX_TRANSPORT_RETRIES = 5
_transport_class_failure = is_transport_class_failure


def _raw_input_hashes(path: Path) -> set[str]:
    return {
        str(json.loads(line).get("input_sha256") or "")
        for line in path.read_text().splitlines()
        if line.strip()
    }


def holdout_provenance_problem(results_dir: Path, holdout_dir: Path) -> str:
    """Refuse a holdout that is not demonstrably disjoint from the audited run.

    The first holdout flight (request 14, 2026-08-18) reproduced the audited
    numbers bit for bit: the run's vendored runner snapshot predated example
    offset support, so the env knob was silently ignored and test[0:200] ran
    twice. A manifest field alone is a claim; the raw input hash sets are the
    evidence, so both are checked.
    """
    manifest_path = holdout_dir / "dataset_manifest.json"
    if not manifest_path.exists():
        return "holdout_dataset_manifest_missing"
    manifest = json.loads(manifest_path.read_text())
    offset = manifest.get("example_offset")
    if offset is None or int(offset) != HOLDOUT_OFFSET:
        return f"holdout_offset_not_applied:{offset}"
    raw_path = holdout_dir / "raw_predictions.jsonl"
    if not raw_path.exists():
        return "holdout_raw_predictions_missing"
    overlap = _raw_input_hashes(results_dir / "raw_predictions.jsonl") & _raw_input_hashes(
        raw_path
    )
    overlap.discard("")
    if overlap:
        return f"holdout_examples_overlap_audited_run:{len(overlap)}"
    return ""


def _prepare_holdout_code(workdir: Path) -> Path:
    """Copy the run's code with its vendored measurement layer refreshed.

    The method identity (candidate_adapter, execution_requirements) is copied
    untouched; only the vendored meta_harness snapshot is replaced with this
    release's files so the runner understands the example offset. The original
    code dir stays byte-identical for provenance.
    """
    import shutil

    code_dir = workdir / "code"
    holdout_code = workdir / "code_holdout"
    if holdout_code.exists():
        shutil.rmtree(holdout_code)
    shutil.copytree(
        code_dir, holdout_code, ignore=shutil.ignore_patterns("__pycache__")
    )
    release_root = Path(__file__).resolve().parents[1]
    vendored_root = holdout_code / "meta_harness"
    for vendored in sorted(vendored_root.rglob("*.py")):
        rel = vendored.relative_to(holdout_code)
        source = release_root / rel
        if not source.exists():
            raise EvidenceAuditError(f"no current source for vendored {rel}")
        vendored.write_text(source.read_text(encoding="utf-8"), encoding="utf-8")
    return holdout_code


def _holdout_timeout_seconds(results_dir: Path, grant_id: int) -> int:
    """Size the holdout window from evidence, bounded by its funding.

    The audited run's measured wall clock is the best estimate of the
    holdout's cost (same model, same n, same seed list); a fixed 3h constant
    exceeded the 2h grants the timer issues, so admission refused every
    holdout before it started (grant 111, 2026-08-19).
    """
    try:
        wall = float(
            json.loads((results_dir / "final_results.json").read_text()).get(
                "wall_seconds"
            )
            or 0
        )
    except Exception:
        wall = 0.0
    sized = max(3600, int(wall * 2.0) + 900) if wall else 10800
    grant = db.fetchone(
        "SELECT max_gpu_hours FROM resource_grants WHERE id=?", (int(grant_id),)
    )
    funded = int(float(dict(grant or {}).get("max_gpu_hours") or 0) * 3600)
    if funded > 600:
        sized = min(sized, funded - 300)
    return max(1800, min(10800, sized))


def _submit_holdout(run: Mapping[str, Any], grant_id: int, attempt: int) -> int:
    from orchestrator.meta_compute_runtime import ColabWorkSpec, submit_colab_work

    workdir, _results, holdout_dir = _run_paths(run)
    holdout_dir.mkdir(parents=True, exist_ok=True)
    holdout_code = _prepare_holdout_code(workdir)
    key_suffix = "evidence_audit_holdout" if attempt <= 1 else (
        f"evidence_audit_holdout{attempt}"
    )
    spec = ColabWorkSpec(
        agenda_id=int(run["agenda_id"]),
        idea_id=int(run["deep_insight_id"]),
        experiment_run_id=int(run["id"]),
        resource_grant_id=int(grant_id),
        stage="evidence_audit",
        idempotency_key=(
            f"experiment-run:{run['agenda_id']}:{run['deep_insight_id']}:"
            f"{run['id']}:{key_suffix}"
        ),
        code_dir=str(holdout_code),
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
        timeout_seconds=_holdout_timeout_seconds(_results, grant_id),
    )
    job = submit_colab_work(spec)
    return int(getattr(job, "id", 0) or 0)


def _settle_completed_grants(run: Mapping[str, Any], log=print) -> None:
    """Close upper-ladder grants whose mission the decided run has completed.

    Grants for earlier stages are consumed when the finalizer records their
    outcome; full_benchmark and evidence_audit grants had no closer, so they
    sat active and held agenda concurrency slots until natural expiry (seen
    2026-08-18: grants 80/81 blocked every new proposal for a day). The
    closer is the same operator path the finalizer uses: an OutcomeRecord
    assembled purely from persisted metering.
    """
    repo = MetaHarnessRepository()
    for grant in db.fetchall(
        """
        SELECT id FROM resource_grants
        WHERE agenda_id=? AND idea_id=? AND status='active'
          AND stage IN ('full_benchmark', 'evidence_audit')
          AND NOT EXISTS (
            SELECT 1 FROM colab_work_requests_v1 c
            WHERE c.resource_grant_id=resource_grants.id
              AND c.status IN ('queued', 'admitting', 'running')
          )
        ORDER BY id
        """,
        (int(run["agenda_id"]), int(run["deep_insight_id"])),
    ):
        grant_id = int(grant["id"])
        try:
            outcome_id = repo.assemble_and_record_outcome(
                resource_grant_id=grant_id,
                experiment_run_id=int(run["id"]),
            )
        except Exception as exc:
            db.rollback()
            log(f"[AUDIT] grant {grant_id} settlement deferred: "
                f"{type(exc).__name__}: {exc}")
            continue
        log(f"[AUDIT] grant {grant_id} settled and consumed "
            f"(outcome {outcome_id})")


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
        _settle_completed_grants(run, log=log)
        return "decided"
    if state not in {"full_benchmark_complete", "evidence_audited"}:
        return f"not_ready:{state}"
    _workdir, results_dir, holdout_dir = _run_paths(run)

    ledger_path, ledger_hash = build_claim_ledger(results_dir)
    evaluator_path = results_dir / "audit_evaluator.json"
    evaluator = None
    if evaluator_path.exists():
        cached = json.loads(evaluator_path.read_text())
        # A judgement made under an older prompt whose semantics differed is
        # re-collected, not trusted (v1 never stated the hypothesis is
        # directional and misread run 171's significant harm as support).
        #
        # It is also re-collected when the ledger itself has changed. The
        # evaluator's "concur" answers a question about the verdict the
        # ledger carried when it was asked: run 189's judgement said
        # concur=false because the ledger claimed "refuted" at p=0.220. Once
        # the verdict was corrected to "inconclusive" -- which is exactly
        # what the evaluator had argued for -- the stale concur=false kept
        # blocking the ladder, an objection to a claim no longer being made.
        stale_ledger = str(cached.get("ledger_hash") or "") != ledger_hash
        if (
            str(cached.get("prompt_ref") or "") == AUDIT_EVALUATOR_PROMPT_REF
            and not stale_ledger
        ):
            evaluator = cached
    if evaluator is None:
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
    if holdout_final_path.exists():
        problem = holdout_provenance_problem(results_dir, holdout_dir)
        if problem:
            # Quarantine the invalid flight for forensics; the audit must
            # never advance on a holdout that is not provably disjoint.
            quarantine = holdout_dir.with_name(
                f"{holdout_dir.name}_invalid_{_sha256_text(problem)[:8]}"
            )
            if not quarantine.exists():
                holdout_dir.rename(quarantine)
            log(f"[AUDIT] run {run_id} holdout rejected ({problem}); quarantined")
    if not holdout_final_path.exists():
        rows = db.fetchall(
            """
            SELECT id, status, failure_reason FROM colab_work_requests_v1
            WHERE experiment_run_id=? AND idempotency_key LIKE '%evidence_audit_holdout%'
            ORDER BY id DESC
            """,
            (run_id,),
        )
        if rows and str(rows[0].get("status")) in {"queued", "running", "admitting"}:
            return "holdout_pending"
        attempt = len(rows) + 1
        # A flight that died in transport never evaluated a single example, so
        # it must not spend the holdout's retry budget: run 180 burned all
        # three attempts on two lost notebook sessions and one refused
        # provision, and the cap then blocked the attempt that the dedicated
        # host would have completed in fourteen minutes (2026-08-19).
        scientific_failures = sum(
            1
            for row in rows
            if not _transport_class_failure(row.get("failure_reason"))
        )
        if scientific_failures >= MAX_HOLDOUT_ATTEMPTS:
            raise EvidenceAuditError("holdout attempts exhausted")
        if attempt > MAX_HOLDOUT_ATTEMPTS + MAX_TRANSPORT_RETRIES:
            raise EvidenceAuditError(
                "holdout transport retries exhausted; infrastructure is not "
                "delivering a measurement"
            )
        _submit_holdout(run, resource_grant_id, attempt)
        log(
            f"[AUDIT] holdout attempt {attempt} submitted for run {run_id} "
            f"at offset {HOLDOUT_OFFSET}"
        )
        return "holdout_submitted"

    final = json.loads((results_dir / "final_results.json").read_text())
    holdout_final = json.loads(holdout_final_path.read_text())
    # The verdict lives on the run row (written at outcome time), not in
    # final_results.json; the artifact only carries scientific_negative_result.
    # The run row's verdict is authoritative when it is a real verdict; a row
    # written before the significance rule existed falls through to the same
    # rule the ledger uses, so the two can never disagree.
    recorded = str(run.get("hypothesis_verdict") or "").strip()
    verdict = (
        recorded
        if recorded in {"supported", "refuted", "inconclusive"}
        else significance_aware_verdict(final)
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
    _settle_completed_grants(run, log=log)
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
