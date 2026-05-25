"""Submission title and table label style."""

from agents.manuscript_submission_style import (
    humanize_ablation_variant,
    sanitize_submission_title,
    short_method_name,
)
from agents.manuscript_publication_tables import build_publication_ablation_table_tex


def test_sanitize_title_strips_headline_stats() -> None:
    raw = (
        "Contrastive Perceptual Grounding (CPG): "
        "VLMs fail 86.9% of perception errors despite architectural claims"
    )
    state = {
        "method_name": "Contrastive Perceptual Grounding (CPG)",
        "problem_awareness": {
            "method_answer": "dual-stream grounding discrepancy for error detection",
        },
    }
    out = sanitize_submission_title(raw, state)
    assert "86.9" not in out
    assert ":" in out
    assert "CPG" in out
    assert len(out) > 40


def test_short_method_names() -> None:
    assert short_method_name("Vanilla Direct Answering") == "Vanilla"
    assert "CPG" in short_method_name("Contrastive Perceptual Grounding (CPG)")


def test_ablation_pending_table_not_fake_scores() -> None:
    state = {
        "baseline_metric_name": "primary_score",
        "benchmark_summary": {
            "ablations": {
                "no_lcb": {"primary_score": None, "executed": False, "delta_vs_full": None},
            }
        },
    }
    tex = build_publication_ablation_table_tex(state)
    assert r"\textsc{Pending}" in tex
    assert "0.2699" not in tex
    assert humanize_ablation_variant("no_lcb") in tex
