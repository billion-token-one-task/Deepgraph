import json
from pathlib import Path

from agents.paper_completeness import assert_traceable, validate_evidence_ledger


FIXTURES = Path(__file__).parent / "fixtures"


def _tex(name: str) -> str:
    return (FIXTURES / name).read_text(encoding="utf-8")


def _ledger(name: str) -> dict:
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


def test_m4_unsourced_abstract_number_reports_violation_when_ledger_lacks_value():
    violations = assert_traceable(_tex("m4_unsourced_number.tex"), _ledger("m4_missing_p_ledger.json"))

    assert any(v["rule"] == "unsourced_number" and v["value"] == "0.01" for v in violations)


def test_m4_refuted_verdict_blocks_positive_abstract_and_conclusion_claims():
    violations = assert_traceable(
        _tex("m4_refuted_positive_claim.tex"),
        _ledger("m4_refuted_ledger.json"),
    )

    assert any(v["rule"] == "positive_claim_with_negative_verdict" for v in violations)
    assert {v["location"]["section"] for v in violations} >= {"abstract", "conclusion"}


def test_m4_clean_abstract_and_conclusion_numbers_are_traceable_with_percent_normalization():
    violations = assert_traceable(_tex("m4_clean.tex"), _ledger("m4_clean_ledger.json"))

    assert violations == []


def test_m4_numbers_outside_abstract_and_conclusion_are_allowed():
    violations = assert_traceable(_tex("m4_method_only.tex"), _ledger("m4_clean_ledger.json"))

    assert violations == []


def test_m4_schema_validation_reports_missing_required_field():
    ledger = _ledger("m4_clean_ledger.json")
    ledger.pop("verdict")

    violations = validate_evidence_ledger(ledger)

    assert any(v["rule"] == "schema_error" and v["value"] == "verdict" for v in violations)


def test_m4_table_figure_and_seed_counts_are_not_unsourced_numbers():
    violations = assert_traceable(_tex("m4_whitelist_numbers.tex"), _ledger("m4_clean_ledger.json"))

    assert violations == []
