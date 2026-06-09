"""Plain final-review gate for generated manuscripts.

This reviewer intentionally asks the boring, direct question a human user would
ask after opening the PDF: is this paper actually deliverable, and what is wrong?
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

from agents.llm_client import call_llm_json


PLAIN_REVIEW_SCHEMA_VERSION = "plain_manuscript_reviewer_v1"


def _clip(text: str, limit: int) -> str:
    text = str(text or "").strip()
    if len(text) <= limit:
        return text
    return text[:limit].rstrip() + "\n...[truncated]..."


def _latex_to_plain_excerpt(tex: str) -> str:
    text = re.sub(r"%.*", " ", tex or "")
    text = re.sub(r"\\begin\{figure\*?\}[\s\S]*?\\end\{figure\*?\}", " [FIGURE] ", text)
    text = re.sub(r"\\begin\{table\*?\}", "\n[TABLE]\n", text)
    text = re.sub(r"\\end\{table\*?\}", "\n[/TABLE]\n", text)
    text = re.sub(r"\\(?:section|subsection|subsubsection|paragraph)\*?\{([^}]*)\}", r"\n\n## \1\n", text)
    text = re.sub(r"\\caption\{([^}]*)\}", r"\nCaption: \1\n", text)
    text = re.sub(r"\\cite[a-zA-Z*]*(?:\[[^\]]*\]){0,2}\{[^}]*\}", " [CITE] ", text)
    text = re.sub(r"\\[a-zA-Z]+(?:\[[^\]]*\])?(?:\{[^}]*\})?", " ", text)
    text = re.sub(r"[{}$&#_^~]", " ", text)
    return re.sub(r"[ \t]{2,}", " ", text).strip()


def _pdf_text_excerpt(pdf_path: Path, limit: int = 24000) -> str:
    if not pdf_path.exists():
        return ""
    try:
        import fitz  # type: ignore

        chunks: list[str] = []
        with fitz.open(str(pdf_path)) as doc:
            for page in doc:
                chunks.append(page.get_text("text"))
                if sum(len(x) for x in chunks) >= limit:
                    break
        return _clip("\n".join(chunks), limit)
    except Exception:
        return ""


def _as_issue(raw: Any) -> dict[str, str]:
    if isinstance(raw, str):
        return {"severity": "medium", "issue": raw}
    if not isinstance(raw, dict):
        return {"severity": "medium", "issue": str(raw)}
    severity = str(raw.get("severity") or raw.get("level") or "medium").lower()
    if severity not in {"high", "medium", "low"}:
        severity = "medium"
    issue = str(raw.get("issue") or raw.get("problem") or raw.get("comment") or "").strip()
    out = {"severity": severity, "issue": issue or "Reviewer raised an unspecified concern."}
    for key in ("area", "evidence", "fix"):
        value = raw.get(key)
        if value:
            out[key] = str(value)
    return out


def _normalise_review(parsed: Any, tokens: int) -> dict:
    if not isinstance(parsed, dict):
        parsed = {"issues": parsed if isinstance(parsed, list) else [str(parsed)]}
    issues = [_as_issue(x) for x in parsed.get("issues") or parsed.get("shortcomings") or []]
    score = parsed.get("score")
    try:
        score = float(score)
    except (TypeError, ValueError):
        score = None
    can_deliver = parsed.get("can_deliver")
    if can_deliver is None:
        verdict = str(parsed.get("verdict") or parsed.get("recommendation") or "").lower()
        can_deliver = bool(verdict in {"accept", "ready", "deliverable", "weak_accept"} and not any(x["severity"] == "high" for x in issues))
    can_deliver = bool(can_deliver) and not any(x["severity"] == "high" for x in issues)
    if score is not None and score < 6.0:
        can_deliver = False
    status = "pass" if can_deliver else "fail"
    return {
        "schema_version": PLAIN_REVIEW_SCHEMA_VERSION,
        "status": status,
        "can_deliver": can_deliver,
        "score": score,
        "summary": str(parsed.get("summary") or parsed.get("overall") or "").strip(),
        "recommendation": str(parsed.get("recommendation") or parsed.get("verdict") or ("ready" if can_deliver else "needs_revision")),
        "issues": issues,
        "tokens": tokens,
    }


def review_manuscript_plain(*, bundle_dir: Path, main_tex: str, quality_context: dict | None = None) -> dict:
    """Run a direct LLM reviewer after ``main.tex``/``main.pdf`` exist."""
    bundle_dir = Path(bundle_dir)
    if "unit_tests" in str(bundle_dir) or os.environ.get("DEEPGRAPH_DISABLE_PLAIN_REVIEWER") == "1":
        return {
            "schema_version": PLAIN_REVIEW_SCHEMA_VERSION,
            "status": "skipped",
            "can_deliver": True,
            "issues": [],
            "skip_reason": "disabled_for_test_or_env",
        }

    pdf_excerpt = _pdf_text_excerpt(bundle_dir / "main.pdf")
    tex_excerpt = _clip(_latex_to_plain_excerpt(main_tex), 28000)
    context = _clip(json.dumps(quality_context or {}, ensure_ascii=False, indent=2, default=str), 12000)

    system_prompt = (
        "You are a skeptical but ordinary ML paper reviewer. Do not rescue the authors. "
        "Judge the manuscript as a real submission draft."
    )
    user_prompt = (
        "How is the quality of this paper? Is it deliverable/submittable now? List the deficiencies.\n"
        "Pay special attention to: whether the proposed method beats simple cheap baselines; "
        "statistical significance, seed count, dataset/model scale; route/gate rate showing whether the method actually works; "
        "overclaiming words such as Certified or training-free; duplicated abstract/title/sections; raw unrounded floats; "
        "question-style contribution framing; one-sentence Results/Discussion; unresolved citations or ???; "
        "and repeated or same-type experiment figures.\n\n"
        "Return JSON only with keys: can_deliver boolean, score number from 1 to 10, recommendation string, "
        "summary string, issues array of objects {severity: high|medium|low, area, issue, evidence, fix}.\n\n"
        "--- quality_context.json ---\n"
        + context
        + "\n\n--- pdf_text_excerpt ---\n"
        + (pdf_excerpt or "[PDF text unavailable]")
        + "\n\n--- main_tex_plain_excerpt ---\n"
        + tex_excerpt
    )
    try:
        parsed, tokens = call_llm_json(system_prompt, user_prompt, temperature=0.0)
        review = _normalise_review(parsed, tokens)
    except Exception as exc:
        review = {
            "schema_version": PLAIN_REVIEW_SCHEMA_VERSION,
            "status": "fail",
            "can_deliver": False,
            "score": None,
            "recommendation": "needs_revision",
            "summary": "Plain final reviewer could not complete; manuscript cannot be marked ready without this gate.",
            "issues": [
                {
                    "severity": "high",
                    "area": "Plain final review",
                    "issue": "Plain manuscript reviewer failed or was unavailable.",
                    "evidence": str(exc)[:500],
                    "fix": "Re-run the final review gate after the LLM reviewer is available.",
                }
            ],
            "error": str(exc),
        }
    return review
