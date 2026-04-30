"""Structured AI review for generated manuscript artifacts."""
from __future__ import annotations

import json
from pathlib import Path

from agents.artifact_manager import artifact_path, ensure_artifact_dirs, record_artifact
from agents.llm_client import call_llm_json
from db import database as db


REVIEW_FIELDS = (
    "overall_score",
    "recommendation",
    "major_concerns",
    "minor_concerns",
    "required_experiments",
    "citation_risks",
    "reproducibility_risks",
)


REVIEW_SYSTEM = """You are a rigorous scientific reviewer. Review the manuscript for correctness, novelty risk, experiment adequacy, citation grounding, and reproducibility.

Return ONLY JSON with these fields:
{
  "overall_score": 1,
  "recommendation": "reject|weak_reject|borderline|weak_accept|accept",
  "major_concerns": [],
  "minor_concerns": [],
  "required_experiments": [],
  "citation_risks": [],
  "reproducibility_risks": []
}

Do not invent new experimental results. Do not change metrics. If evidence is missing, report it as a concern."""


MAX_REVIEW_MANUSCRIPT_CHARS = 60000
REVIEW_HEAD_CHARS = 42000
REVIEW_TAIL_CHARS = 14000


def _find_manuscript(workdir: Path) -> Path | None:
    candidates = [
        workdir / "artifacts" / "manuscript" / "paper_candidate.md",
        workdir / "artifacts" / "manuscript" / "paper.md",
        workdir / "artifacts" / "manuscript" / "additional_experiments_required.md",
        workdir / "artifacts" / "manuscript" / "preliminary_report.md",
        workdir / "artifacts" / "manuscript" / "negative_result_report.md",
    ]
    for path in candidates:
        if path.exists() and path.is_file():
            return path
    return None


def _load_json(path: Path, default):
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return default


def _compact_json(value, max_chars: int = 12000) -> str:
    text = json.dumps(value, indent=2, ensure_ascii=False, default=str)
    if len(text) <= max_chars:
        return text
    head = max_chars // 2
    tail = max_chars - head
    return (
        text[:head]
        + "\n... [middle omitted from reviewer context; full artifact remains on disk] ...\n"
        + text[-tail:]
    )


def _review_package_context(workdir: Path) -> str:
    """Summarize machine-readable artifacts so review is not based only on prose."""
    artifact_specs = [
        ("benchmark_config.json", workdir / "benchmark_config.json"),
        ("evidence_gate.json", workdir / "artifacts" / "results" / "evidence_gate.json"),
        ("statistical_report.json", workdir / "artifacts" / "results" / "statistical_report.json"),
        ("lp_validation.json", workdir / "artifacts" / "results" / "lp_validation.json"),
        ("reproduction_manifest.json", workdir / "artifacts" / "results" / "reproduction_manifest.json"),
    ]
    sections = ["# Review Package Artifact Context"]
    for label, path in artifact_specs:
        payload = _load_json(path, None)
        if payload is None:
            continue
        if label == "statistical_report.json" and isinstance(payload, dict):
            payload = {
                "primary_metric": payload.get("primary_metric"),
                "metric_direction": payload.get("metric_direction"),
                "baseline_method": payload.get("baseline_method"),
                "candidate_method": payload.get("candidate_method"),
                "best_method": payload.get("best_method"),
                "absolute_best_method": payload.get("absolute_best_method"),
                "aggregate_metric_summaries": payload.get("aggregate_metric_summaries"),
                "comparisons": payload.get("comparisons"),
                "pairwise_comparisons": payload.get("pairwise_comparisons"),
            }
        elif label == "lp_validation.json" and isinstance(payload, dict):
            payload = {
                "status": payload.get("status"),
                "tolerances": payload.get("tolerances"),
                "analytic_randomized_checks": payload.get("analytic_randomized_checks"),
                "deterministic_reference_comparisons": payload.get("deterministic_reference_comparisons"),
            }
        sections.extend([
            "",
            f"## {label}",
            "",
            "```json",
            _compact_json(payload),
            "```",
        ])
    return "\n".join(sections) + "\n"


def _manuscript_for_review(manuscript: str) -> str:
    """Keep the ending sections visible when manuscripts exceed the prompt budget."""
    if len(manuscript) <= MAX_REVIEW_MANUSCRIPT_CHARS:
        return manuscript
    return (
        manuscript[:REVIEW_HEAD_CHARS]
        + "\n\n... [middle of manuscript omitted from reviewer prompt; see artifact context above and full local manuscript] ...\n\n"
        + manuscript[-REVIEW_TAIL_CHARS:]
    )


def _list_field(value) -> list:
    if isinstance(value, list):
        return [str(item) for item in value]
    if value:
        return [str(value)]
    return []


def _validate_review(payload: dict) -> dict:
    if not isinstance(payload, dict):
        payload = {}
    score = payload.get("overall_score", 1)
    try:
        score = int(score)
    except (TypeError, ValueError):
        score = 1
    score = max(1, min(10, score))

    recommendation = str(payload.get("recommendation") or "reject")
    if recommendation not in {"reject", "weak_reject", "borderline", "weak_accept", "accept"}:
        recommendation = "reject"

    return {
        "overall_score": score,
        "recommendation": recommendation,
        "major_concerns": _list_field(payload.get("major_concerns")),
        "minor_concerns": _list_field(payload.get("minor_concerns")),
        "required_experiments": _list_field(payload.get("required_experiments")),
        "citation_risks": _list_field(payload.get("citation_risks")),
        "reproducibility_risks": _list_field(payload.get("reproducibility_risks")),
    }


def _review_markdown(review: dict) -> str:
    def section(title: str, items: list[str]) -> str:
        if not items:
            return f"## {title}\n\n- None reported.\n"
        return f"## {title}\n\n" + "\n".join(f"- {item}" for item in items) + "\n"

    return (
        "# AI Review\n\n"
        f"Overall score: `{review['overall_score']}`\n\n"
        f"Recommendation: `{review['recommendation']}`\n\n"
        + section("Major Concerns", review["major_concerns"])
        + "\n"
        + section("Minor Concerns", review["minor_concerns"])
        + "\n"
        + section("Required Experiments", review["required_experiments"])
        + "\n"
        + section("Citation Risks", review["citation_risks"])
        + "\n"
        + section("Reproducibility Risks", review["reproducibility_risks"])
    )


def _write_review_artifact(workdir: Path, run_id: int, relative_path: str,
                           artifact_type: str, text: str):
    path = artifact_path(workdir, relative_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    record_artifact(workdir, run_id, artifact_type, path)
    return str(path)


def review_manuscript(run_id: int) -> dict:
    """Run structured AI review over the generated manuscript package."""
    run = db.fetchone("SELECT * FROM experiment_runs WHERE id=?", (run_id,))
    if not run:
        return {"status": "error", "reason": "run_not_found", "run_id": run_id}

    workdir = Path(run.get("workdir") or "")
    manuscript_path = _find_manuscript(workdir)
    if manuscript_path is None:
        return {"status": "error", "reason": "manuscript_not_found", "run_id": run_id}

    ensure_artifact_dirs(workdir)
    manuscript = manuscript_path.read_text(encoding="utf-8", errors="replace")
    prompt = (
        f"Run ID: {run_id}\n"
        f"Verdict: {run.get('hypothesis_verdict')}\n\n"
        f"{_review_package_context(workdir)}\n"
        "# Manuscript\n"
        f"{_manuscript_for_review(manuscript)}"
    )
    raw_review, tokens = call_llm_json(REVIEW_SYSTEM, prompt)
    review = _validate_review(raw_review)
    review["tokens"] = tokens

    outputs = [
        _write_review_artifact(
            workdir,
            run_id,
            "artifacts/reviews/review.json",
            "review",
            json.dumps(review, indent=2),
        ),
        _write_review_artifact(
            workdir,
            run_id,
            "artifacts/reviews/review.md",
            "review",
            _review_markdown(review),
        ),
    ]

    return {
        "status": "complete",
        "run_id": run_id,
        "review": review,
        "outputs": outputs,
    }
