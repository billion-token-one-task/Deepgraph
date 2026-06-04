from pathlib import Path

from agents.paper_completeness import latex_sanity_check


FIXTURES = Path(__file__).parent / "fixtures"


def _tex(name: str) -> str:
    return (FIXTURES / name).read_text(encoding="utf-8")


def _rules(report: dict) -> set[str]:
    return {hit.get("rule") or hit.get("kind") for hit in report.get("hits", [])}


def test_m2a_unresolved_reference_fails_with_line_location():
    report = latex_sanity_check(_tex("m2a_unresolved.tex"))

    assert report["ok"] is False
    assert "unresolved_reference" in _rules(report)
    assert any(hit.get("location", {}).get("line") for hit in report["hits"])


def test_m2a_single_question_and_verbatim_question_marks_pass():
    report = latex_sanity_check(_tex("m2a_clean.tex"))

    assert report["ok"] is True


def test_m2b_scaffold_placeholder_regex_fails_but_latex_braces_pass():
    bad = latex_sanity_check(_tex("m2b_scaffold.tex"))
    clean = latex_sanity_check(_tex("m2b_clean.tex"))

    assert bad["ok"] is False
    assert "template_placeholder" in _rules(bad)
    assert clean["ok"] is True


def test_m2b_existing_placeholder_forbidden_term_still_blocks():
    report = latex_sanity_check("This caption contains placeholder text.")

    assert report["ok"] is False
    assert any(hit.get("value") == "placeholder" for hit in report["hits"])


def test_m2c_cross_run_method_token_fails_by_set_membership():
    report = latex_sanity_check(_tex("m2c_contaminated.tex"), state={"method_name": "CGGR"})

    assert report["ok"] is False
    assert "cross_run_identity" in _rules(report)
    assert any(hit.get("value") == "OtherMethod" for hit in report["hits"])


def test_m2c_method_token_and_abbreviation_whitelist_pass():
    report = latex_sanity_check(_tex("m2c_clean.tex"), state={"method_name": "CGGR"})

    assert report["ok"] is True


def test_m2d_repeated_boilerplate_sentence_fails():
    report = latex_sanity_check(_tex("m2d_repeat.tex"))

    assert report["ok"] is False
    assert "boilerplate_repetition" in _rules(report)


def test_m2d_clean_sentences_pass():
    report = latex_sanity_check(_tex("m2d_clean.tex"))

    assert report["ok"] is True
