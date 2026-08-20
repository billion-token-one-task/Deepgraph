"""Evidence audit unit coverage: ledger derivation and holdout consistency.

The audit is the ladder's last executor; these tests pin the parts that are
pure functions of artifacts so a regression cannot silently change what
counts as audited evidence.
"""

import json

import pytest

from meta_harness.evidence_audit import (
    EvidenceAuditError,
    HOLDOUT_OFFSET,
    _verify_arms,
    build_claim_ledger,
    holdout_consistent,
    holdout_provenance_problem,
)


def _final(base=0.67, cand=0.64, p=0.506, **extra):
    payload = {
        "schema_version": "final_results_v1",
        "dataset_id": "openai/gsm8k",
        "dataset_revision": "740312add88f781978c0658806c59bc2815b9866",
        "model_id": "Qwen/Qwen2.5-1.5B-Instruct",
        "model_revision": "989aa7980e4cf806f80c7fef2b1adb7bc71aa306",
        "metric_name": "numeric_accuracy",
        "primary_metric": "numeric_accuracy",
        "metric_direction": "higher",
        "baseline_method": "unmodified_input_baseline",
        "candidate_method": "candidate",
        "baseline_metric_value": base,
        "metric_value": cand,
        "num_examples": 200,
        "seeds": [0],
        "scientific_negative_result": cand <= base,
        "statistical_tests": {"paired_permutation_p": p},
        "artifact_hashes": {"raw_predictions": "0" * 64},
    }
    payload.update(extra)
    return payload


def _rows(n_correct_base, n_correct_cand, n=10):
    rows = []
    for method, n_correct in (
        ("unmodified_input_baseline", n_correct_base),
        ("candidate", n_correct_cand),
    ):
        for i in range(n):
            rows.append(
                {
                    "method": method,
                    "sample_index": i,
                    "prediction": "42" if i < n_correct else "wrong",
                    "target": "42",
                }
            )
    return rows


def test_verify_arms_matches_reported():
    final = _final(base=0.7, cand=0.4)
    verified = _verify_arms(final, _rows(7, 4))
    assert verified["baseline"] == pytest.approx(0.7)
    assert verified["candidate"] == pytest.approx(0.4)
    assert verified["p_value"] == pytest.approx(0.506)


def test_verify_arms_refuses_mismatch():
    final = _final(base=0.9, cand=0.4)  # reported baseline inflated
    with pytest.raises(EvidenceAuditError):
        _verify_arms(final, _rows(7, 4))


def test_build_claim_ledger_writes_verdict_and_hash(tmp_path):
    # A large delta at p=0.506 is still inconclusive: "refuted" carries the
    # same evidential burden as "supported", and deriving it from the sign
    # alone overstated runs 164 and 180 (2026-08-20).
    (tmp_path / "final_results.json").write_text(
        json.dumps(_final(base=0.7, cand=0.4))
    )
    (tmp_path / "raw_predictions.jsonl").write_text(
        "\n".join(json.dumps(r) for r in _rows(7, 4))
    )
    path, digest = build_claim_ledger(tmp_path)
    ledger = json.loads(path.read_text())
    claim = ledger["claims"][0]
    assert claim["verdict"] == "inconclusive"
    assert claim["delta"] == pytest.approx(-0.3)
    assert len(digest) == 64


def test_build_claim_ledger_records_refuted_when_the_harm_is_significant(tmp_path):
    (tmp_path / "final_results.json").write_text(
        json.dumps(_final(base=0.7, cand=0.4, p=0.001))
    )
    (tmp_path / "raw_predictions.jsonl").write_text(
        "\n".join(json.dumps(r) for r in _rows(7, 4))
    )
    path, _digest = build_claim_ledger(tmp_path)
    claim = json.loads(path.read_text())["claims"][0]
    assert claim["verdict"] == "refuted"


def test_holdout_consistency_by_verdict():
    final = _final()
    beats = _final(base=0.6, cand=0.7, p=0.01)
    loses = _final(base=0.7, cand=0.6, p=0.01)
    insignificant = _final(base=0.6, cand=0.7, p=0.4)
    # supported needs the holdout to also beat significantly
    assert holdout_consistent("supported", final, beats)
    assert not holdout_consistent("supported", final, loses)
    assert not holdout_consistent("supported", final, insignificant)
    # a negative verdict is contradicted only by a significant holdout win
    assert holdout_consistent("refuted", final, loses)
    assert holdout_consistent("refuted", final, insignificant)
    assert not holdout_consistent("refuted", final, beats)
    assert holdout_consistent("inconclusive", final, insignificant)


def test_holdout_offset_clears_full_benchmark_window():
    # the audited run consumed test[0:200]; the holdout must start past it
    assert HOLDOUT_OFFSET >= 200


def _write_arm(dirpath, hashes):
    dirpath.mkdir(exist_ok=True)
    (dirpath / "raw_predictions.jsonl").write_text(
        "\n".join(json.dumps({"input_sha256": h}) for h in hashes)
    )


def test_holdout_provenance_rejects_missing_or_wrong_offset(tmp_path):
    results, holdout = tmp_path / "results", tmp_path / "holdout"
    _write_arm(results, ["a", "b"])
    _write_arm(holdout, ["c", "d"])
    # request 14's failure mode: manifest with no offset recorded
    (holdout / "dataset_manifest.json").write_text(json.dumps({"num_examples": 2}))
    assert "holdout_offset_not_applied" in holdout_provenance_problem(results, holdout)
    (holdout / "dataset_manifest.json").write_text(
        json.dumps({"example_offset": 0, "num_examples": 2})
    )
    assert "holdout_offset_not_applied" in holdout_provenance_problem(results, holdout)


def test_holdout_provenance_rejects_example_overlap(tmp_path):
    results, holdout = tmp_path / "results", tmp_path / "holdout"
    _write_arm(results, ["a", "b"])
    _write_arm(holdout, ["b", "c"])
    (holdout / "dataset_manifest.json").write_text(
        json.dumps({"example_offset": HOLDOUT_OFFSET})
    )
    assert "overlap" in holdout_provenance_problem(results, holdout)


def test_holdout_provenance_accepts_disjoint_offset_run(tmp_path):
    results, holdout = tmp_path / "results", tmp_path / "holdout"
    _write_arm(results, ["a", "b"])
    _write_arm(holdout, ["c", "d"])
    (holdout / "dataset_manifest.json").write_text(
        json.dumps({"example_offset": HOLDOUT_OFFSET})
    )
    assert holdout_provenance_problem(results, holdout) == ""


def test_transport_failures_do_not_spend_the_science_retry_budget():
    from meta_harness.evidence_audit import _transport_class_failure

    for reason in (
        "transport:ColabCLIError:Colab output omitted the return-code sentinel",
        "transport:ColabCLIError:colab provision failed: TooManyAssignments",
        "controller_lost",
        "admission_abandoned_grant_inactive",
    ):
        assert _transport_class_failure(reason) is True
    # a real measurement failure still counts against the cap
    for reason in ("experiment_exit_2", "required_artifacts_missing", "", None):
        assert _transport_class_failure(reason) is False


def test_a_lost_evaluator_answer_does_not_strand_the_run():
    """A settled reservation whose answer never parsed must not be terminal.

    Run 191's audit called the evaluator, settled 4913 tokens, then raised
    "evaluator returned no judgement" before anything was written. Every
    retry was refused with "idempotency key already exists with status
    settled", so the run could never reach the ladder (2026-08-20). The key
    now carries the attempt number, and the raw response is kept so the next
    attempt is diagnosable rather than a second blind call.
    """
    import inspect

    from meta_harness import evidence_audit

    source = inspect.getsource(evidence_audit.independent_evaluator_review)
    assert "{attempt}" in source
    assert "audit_evaluator_unparsed_" in source
    assert "MAX_EVALUATOR_ATTEMPTS" in source


def test_the_evaluator_retry_is_bounded():
    from meta_harness.evidence_audit import MAX_EVALUATOR_ATTEMPTS

    # paying again is honest, paying forever is not
    assert 1 < MAX_EVALUATOR_ATTEMPTS <= 5


def test_evaluator_attempt_counts_what_the_grant_already_paid_for():
    from unittest import mock

    from meta_harness import evidence_audit

    with mock.patch.object(evidence_audit.db, "fetchone", return_value={"n": 2}):
        assert evidence_audit._evaluator_attempt(135) == 2
    # an unreadable ledger must not block the first call
    with mock.patch.object(
        evidence_audit.db, "fetchone", side_effect=RuntimeError("no db")
    ):
        assert evidence_audit._evaluator_attempt(135) == 0
