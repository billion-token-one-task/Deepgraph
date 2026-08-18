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
    (tmp_path / "final_results.json").write_text(
        json.dumps(_final(base=0.7, cand=0.4))
    )
    (tmp_path / "raw_predictions.jsonl").write_text(
        "\n".join(json.dumps(r) for r in _rows(7, 4))
    )
    path, digest = build_claim_ledger(tmp_path)
    ledger = json.loads(path.read_text())
    claim = ledger["claims"][0]
    assert claim["verdict"] == "refuted"
    assert claim["delta"] == pytest.approx(-0.3)
    assert len(digest) == 64


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
