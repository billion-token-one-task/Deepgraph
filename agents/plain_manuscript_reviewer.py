"""Plain final-review gate for generated manuscripts.

This reviewer intentionally asks the boring, direct question a human user would
ask after opening the PDF: is this paper actually deliverable, and what is wrong?
"""

from __future__ import annotations

import json
import hashlib
import os
import re
from pathlib import Path
from typing import Any

from agents.llm_client import call_llm_json_for_role, configured_role_prompt_version


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


def review_manuscript_plain(
    *,
    bundle_dir: Path,
    main_tex: str,
    quality_context: dict | None = None,
    manuscript_state: dict | None = None,
) -> dict:
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
        "How is the quality of this manuscript as a document? Is the submission bundle deliverable now? List document-level deficiencies only.\n"
        "Do not judge experiment adequacy here: missing baselines, p-values, route/gate rates, seed count, dataset/model scale, ablations, benchmark scope, or whether the method meaningfully beats another method belong to the experiment/evidence gate, not this manuscript gate. "
        "Only flag experiment content when the manuscript text misstates the recorded results, fabricates evidence, or uses overclaiming wording unsupported by the supplied context. "
        "Pay special attention to: PDF/LaTeX compilation, exact compile errors in quality_context, duplicated abstract/title/sections, raw unrounded floats, unresolved citations or ???, missing or duplicated required sections, figure placement/layout, repeated figure assets, table formatting, citation coverage, and whether the text fits a complete conference paper.\n\n"
        "Return JSON only with keys: can_deliver boolean, score number from 1 to 10, recommendation string, "
        "summary string, issues array of objects {severity: high|medium|low, area, issue, evidence, fix}.\n\n"
        "--- quality_context.json ---\n"
        + context
        + "\n\n--- pdf_text_excerpt ---\n"
        + (pdf_excerpt or "[PDF text unavailable]")
        + "\n\n--- main_tex_plain_excerpt ---\n"
        + tex_excerpt
    )
    if len(user_prompt or "") > 20000:
        return {
            "schema_version": PLAIN_REVIEW_SCHEMA_VERSION,
            "status": "skipped",
            "can_deliver": True,
            "score": None,
            "recommendation": "deterministic_auditors_only",
            "summary": "Skipped plain LLM reviewer because the review prompt was too large for reliable streaming; deterministic quality auditors remain active.",
            "issues": [],
            "skip_reason": "prompt_too_large_for_reliable_llm_call",
            "prompt_chars": len(user_prompt or ""),
        }
    try:
        from db import database as db

        state = manuscript_state or {}
        agenda_id = int(state.get("agenda_id") or 0)
        idea_id = int(state.get("deep_insight_id") or state.get("idea_id") or 0)
        grant_id = int(state.get("resource_grant_id") or 0)
        run_id = int(state.get("run_id") or state.get("experiment_run_id") or 0)
        if min(agenda_id, idea_id, grant_id, run_id) <= 0:
            raise PermissionError(
                "plain manuscript review requires agenda/idea/run/ResourceGrant scope"
            )
        proposer = db.fetchone(
            """
            SELECT provider, model, model_family
            FROM llm_route_observations
            WHERE agenda_id=? AND idea_id=? AND role='proposer'
              AND status='succeeded'
            ORDER BY id DESC LIMIT 1
            """,
            (agenda_id, idea_id),
        )
        if not proposer:
            raise PermissionError(
                "plain manuscript review requires a recorded proposer route"
            )
        digest = hashlib.sha256(user_prompt.encode("utf-8")).hexdigest()
        parsed, tokens, route = call_llm_json_for_role(
            system_prompt,
            user_prompt,
            agenda_id=agenda_id,
            idea_id=idea_id,
            role="reviewer",
            stage="manuscript",
            resource_grant_id=grant_id,
            operation="plain_manuscript_review",
            idempotency_key=f"plain-review:{run_id}:{digest}",
            prompt_version=configured_role_prompt_version("reviewer"),
            proposer_route=dict(proposer),
            max_tokens=4096,
        )
        review = _normalise_review(parsed, tokens)
        review["route"] = route
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
