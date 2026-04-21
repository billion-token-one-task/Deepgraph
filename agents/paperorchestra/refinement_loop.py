"""Step 5 (PaperOrchestra §4): AgentReview-style accept/revert loop (scores from LLM-as-reviewer)."""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from typing import Any

from agents.llm_client import call_llm, call_llm_json


REVIEWER_SYSTEM = """You are an area-chair style reviewer (similar in spirit to AgentReview / venue rubrics).
Score the manuscript draft on JSON axes only. Be strict. Output JSON only."""


def score_manuscript_latex(tex: str) -> dict[str, Any]:
    """Simulated holistic scores (1–4 sub-axes, 1–10 overall) for accept/revert logic."""
    user = f"""Manuscript (LaTeX excerpt, may be truncated):\n```latex\n{tex[:24000]}\n```
Return JSON with keys:
  "originality", "quality", "clarity", "significance", "soundness", "presentation", "contribution" — each 1-4 integer,
  "overall" — 1-10 integer,
  "acceptance_likelihood" — 0-100 integer (simulated).
"""
    out, _ = call_llm_json(REVIEWER_SYSTEM, user, temperature=0.0)
    if not isinstance(out, dict):
        return {"overall": 5, "originality": 2, "quality": 2, "clarity": 2, "significance": 2, "soundness": 2, "presentation": 2, "contribution": 2}
    return out


def _overall(s: dict[str, Any]) -> int:
    try:
        return int(s.get("overall", 5))
    except (TypeError, ValueError):
        return 5


def _subaxis_sum(s: dict[str, Any]) -> int:
    keys = ("originality", "quality", "clarity", "significance", "soundness", "presentation", "contribution")
    t = 0
    for k in keys:
        try:
            t += int(s.get(k, 2))
        except (TypeError, ValueError):
            t += 2
    return t


def iterative_refine_with_agentreview(
    *,
    content_refinement_system_tex: str,
    build_refinement_user: Callable[..., str],
    initial_tex: str,
    max_iters: int = 4,
) -> tuple[str, list[dict[str, Any]]]:
    """
    §4: accept revision if overall score increases, or tie with non-negative net sub-axis gain; else revert & halt.
    ``build_refinement_user(prev_tex, review_json) -> str`` supplies the user block for the official refinement agent.
    """
    worklog: list[dict[str, Any]] = []
    current = initial_tex
    prev_scores = score_manuscript_latex(current)
    prev_overall = _overall(prev_scores)
    prev_sub = _subaxis_sum(prev_scores)

    for i in range(max_iters):
        ref_user = build_refinement_user(current, prev_scores)
        refined_text, _ = call_llm(content_refinement_system_tex, ref_user)
        # Prefer full ```latex ... ``` block if present (official output format)
        latex = _extract_latex_block(refined_text) or refined_text
        new_scores = score_manuscript_latex(latex)
        new_overall = _overall(new_scores)
        new_sub = _subaxis_sum(new_scores)

        entry = {
            "iter": i + 1,
            "before_overall": prev_overall,
            "after_overall": new_overall,
            "before_subaxis_sum": prev_sub,
            "after_subaxis_sum": new_sub,
        }
        worklog.append(entry)

        # Accept if overall increases
        if new_overall > prev_overall:
            current = latex
            prev_scores, prev_overall, prev_sub = new_scores, new_overall, new_sub
            continue
        # Tie: accept if sub-axis sum does not decrease (non-negative net gain)
        if new_overall == prev_overall and new_sub >= prev_sub:
            current = latex
            prev_scores, prev_overall, prev_sub = new_scores, new_overall, new_sub
            continue
        # Revert & halt
        break

    return current, worklog


def _extract_latex_block(text: str) -> str | None:
    m = re.search(r"```latex\s*([\s\S]*?)```", text, re.I)
    if m:
        return m.group(1).strip()
    return None


def parse_refinement_dual_output(text: str) -> tuple[dict | None, str | None]:
    """Parse worklog JSON + LaTeX from a response that follows the paper's two-block structure."""
    j = None
    m = re.search(r"```(?:json)?\s*(\{[\s\S]*?\})\s*```", text)
    if m:
        try:
            j = json.loads(m.group(1))
        except json.JSONDecodeError:
            j = None
    latex = _extract_latex_block(text)
    return j, latex
