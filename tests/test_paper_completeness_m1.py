import json
from pathlib import Path

from agents.paper_completeness import (
    audit_evidence_completeness,
    build_claim_evidence_matrix,
    build_evidence_ledger,
    build_reviewer_report,
)


FIXTURES = Path(__file__).parent / "fixtures"


def _load_fixture(name: str) -> dict:
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


def _improves_row(matrix: list[dict]) -> dict:
    return next(row for row in matrix if row["claim"] == "Improves utility")


def _claim_row(matrix: list[dict], claim: str) -> dict:
    return next(row for row in matrix if row["claim"].startswith(claim))


def _significance_answer(report: dict) -> dict:
    return next(
        row
        for row in report["checklist"]
        if row["question"] == "Is the improvement statistically significant?"
    )


def test_m1_p_eq_one_is_not_significant_even_with_p_text():
    payload = _load_fixture("m1_p_eq_one.json")
    matrix = build_claim_evidence_matrix(payload["state"], payload["manifest"])

    assert _improves_row(matrix)["can_appear_in_abstract"] is False
    assert _claim_row(matrix, "The method improves utility")["can_appear_in_abstract"] is False


def test_m1_confirmed_p_below_default_alpha_can_appear():
    payload = _load_fixture("m1_significant.json")
    matrix = build_claim_evidence_matrix(payload["state"], payload["manifest"])

    assert _improves_row(matrix)["can_appear_in_abstract"] is True


def test_m1_refuted_verdict_blocks_abstract_claim_even_with_low_p():
    payload = _load_fixture("m1_refuted.json")
    matrix = build_claim_evidence_matrix(payload["state"], payload["manifest"])

    assert _improves_row(matrix)["can_appear_in_abstract"] is False
    assert _claim_row(matrix, "The method improves utility")["can_appear_in_abstract"] is False


def test_m1_alpha_env_override_controls_significance(monkeypatch):
    payload = _load_fixture("m1_significant.json")
    payload["state"]["result_packet"]["p_value"] = 0.049
    payload["manifest"]["statistical_tests"] = "p=0.049"
    monkeypatch.setenv("DEEPGRAPH_SIGNIFICANCE_ALPHA", "0.01")

    matrix = build_claim_evidence_matrix(payload["state"], payload["manifest"])

    assert _improves_row(matrix)["can_appear_in_abstract"] is False


def test_m1_missing_p_value_is_not_significant_and_does_not_crash():
    payload = _load_fixture("m1_significant.json")
    payload["state"]["result_packet"]["p_value"] = None
    payload["manifest"]["statistical_tests"] = ""

    matrix = build_claim_evidence_matrix(payload["state"], payload["manifest"])

    assert _improves_row(matrix)["can_appear_in_abstract"] is False


def test_m1_reviewer_significance_answer_uses_numeric_p_but_preserves_existence_gate():
    payload = _load_fixture("m1_p_eq_one.json")
    matrix = build_claim_evidence_matrix(payload["state"], payload["manifest"])
    report = build_reviewer_report(payload["state"], payload["manifest"], matrix, blockers=[])

    assert _significance_answer(report)["answer"] == "No"
    audit = audit_evidence_completeness(payload["state"])
    assert not any("Statistical test or confidence interval" in blocker for blocker in audit["blockers"])


def test_m1_build_evidence_ledger_minimal_schema():
    payload = _load_fixture("m1_significant.json")
    packet = payload["state"]["result_packet"]
    summary = packet["benchmark_summary"]

    ledger = build_evidence_ledger(
        packet,
        summary,
        alpha=0.01,
        provenance={"command": "pytest tests/test_paper_completeness_m1.py"},
    )

    assert ledger["schema_version"] == "1.0"
    assert ledger["alpha"] == 0.01
    assert ledger["verdict"] == "confirmed"
    assert ledger["p_value"] == 0.0123
    assert ledger["effect_size"] == 0.045
    assert ledger["confidence"] == 0.9877
    assert ledger["per_method"]["Candidate"]["exact_match"] == 0.705
    assert ledger["seed_variance"]["Candidate"]["per_seed"] == {"0": 0.71, "1": 0.7, "2": 0.705}
    assert ledger["seeds"] == [0, 1, 2]
    assert ledger["provenance"]["command"].startswith("pytest")
