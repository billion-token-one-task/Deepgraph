"""Exact conference page-budget enforcement (main text ends on target page)."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

from agents.llm_client import call_llm
from agents.manuscript_templates import get_adapter
from agents.paper_orchestra_prompts import build_conference_guidelines
from agents.manuscript_submission_enrichment import sanitize_latex_body_unicode, strip_llm_wrapper_markup
from agents.paper_orchestra_pipeline import _compile_main_pdf, _page_count_from_log

MAINBODY_END_LABEL = "mainbody:end"

# Paragraphs sharing this fraction of distinctive tokens are considered duplicates.
_DUPLICATE_JACCARD = 0.80
_MIN_TOKENS_FOR_DUPCHECK = 8


def _strip_leading_comment_lines(text: str) -> str:
    lines = (text or "").splitlines()
    while lines and lines[0].lstrip().startswith("%"):
        lines = lines[1:]
    return "\n".join(lines)


def _paragraph_token_set(text: str) -> set[str]:
    raw = _strip_leading_comment_lines(text or "")
    raw = re.sub(r"\\[A-Za-z@]+\*?(\[[^\]]*\])?(\{[^}]*\})?", " ", raw)
    raw = re.sub(r"[^A-Za-z0-9]+", " ", raw).lower()
    tokens = [t for t in raw.split() if len(t) >= 4]
    return set(tokens)


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0


def find_duplicate_paragraphs(main_tex: str) -> list[dict[str, Any]]:
    """Return list of duplicate-pair findings inside the manuscript body.

    Each finding is ``{"paragraph_a_excerpt", "paragraph_b_excerpt", "jaccard"}``.
    Used by ``page_budget_blockers`` to flag the "Related Theme 4..11 identical"
    pattern as a HARD FAIL even when the page count happens to match.
    """
    body_start_match = re.search(r"\\begin\{document\}", main_tex or "")
    body_end_match = re.search(r"\\end\{document\}", main_tex or "")
    body = (main_tex or "")[
        (body_start_match.end() if body_start_match else 0) : (
            body_end_match.start() if body_end_match else len(main_tex or "")
        )
    ]
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", body) if p.strip()]
    paragraphs = [p for p in paragraphs if _strip_leading_comment_lines(p).strip()]
    findings: list[dict[str, Any]] = []
    tokens = [_paragraph_token_set(p) for p in paragraphs]
    for i in range(len(paragraphs)):
        if len(tokens[i]) < _MIN_TOKENS_FOR_DUPCHECK:
            continue
        for j in range(i + 1, len(paragraphs)):
            if len(tokens[j]) < _MIN_TOKENS_FOR_DUPCHECK:
                continue
            score = _jaccard(tokens[i], tokens[j])
            if score >= _DUPLICATE_JACCARD:
                a_excerpt = _strip_leading_comment_lines(paragraphs[i]).strip()[:160]
                b_excerpt = _strip_leading_comment_lines(paragraphs[j]).strip()[:160]
                findings.append(
                    {
                        "paragraph_a_excerpt": a_excerpt,
                        "paragraph_b_excerpt": b_excerpt,
                        "jaccard": round(score, 3),
                    }
                )
                if len(findings) >= 12:
                    return findings
    return findings


def deduplicate_paragraphs(main_tex: str) -> tuple[str, int]:
    """Drop subsequent paragraphs whose token-set Jaccard >= threshold.

    Returns (new_tex, removed_count). The first occurrence is always kept.
    Citations and figures are left untouched (we only collapse plain prose).
    """
    body_start_match = re.search(r"\\begin\{document\}", main_tex or "")
    body_end_match = re.search(r"\\end\{document\}", main_tex or "")
    if not body_start_match or not body_end_match:
        return main_tex, 0
    head = main_tex[: body_start_match.end()]
    tail = main_tex[body_end_match.start() :]
    body = main_tex[body_start_match.end() : body_end_match.start()]
    parts = re.split(r"(\n\s*\n)", body)
    seen: list[tuple[set[str], str]] = []
    new_parts: list[str] = []
    removed = 0
    for part in parts:
        if not part.strip():
            new_parts.append(part)
            continue
        # Sanity: a "part" might be the blank-line separator itself, or pure
        # whitespace, in which case keep it untouched.
        stripped_no_comment = _strip_leading_comment_lines(part).strip()
        if not stripped_no_comment:
            new_parts.append(part)
            continue
        if stripped_no_comment.startswith("\\section") or stripped_no_comment.startswith("\\begin{") or stripped_no_comment.startswith("\\end{"):
            new_parts.append(part)
            continue
        tokens = _paragraph_token_set(part)
        if len(tokens) < _MIN_TOKENS_FOR_DUPCHECK:
            new_parts.append(part)
            continue
        dup = False
        for stored_tokens, _stored in seen:
            if _jaccard(stored_tokens, tokens) >= _DUPLICATE_JACCARD:
                dup = True
                break
        if dup:
            removed += 1
            continue
        seen.append((tokens, part))
        new_parts.append(part)
    return head + "".join(new_parts) + tail, removed


def target_page_count(template_id: str) -> int:
    return int(get_adapter(template_id).max_pages)


def ensure_mainbody_end_label(main_tex: str) -> str:
    """Label the page where References begin (main body ends on previous page)."""
    tex = main_tex or ""
    if rf"\label{{{MAINBODY_END_LABEL}}}" in tex:
        return tex
    marker = rf"\label{{{MAINBODY_END_LABEL}}}" + "\n" + r"\clearpage" + "\n"
    for pat in (
        r"(\\bibliographystyle\{)",
        r"(\\begin\{thebibliography\})",
        r"(\\printbibliography)",
    ):
        m = re.search(pat, tex)
        if m:
            return tex[: m.start()] + marker + tex[m.start() :]
    if r"\end{document}" in tex:
        return tex.replace(r"\end{document}", marker + "\n" + r"\end{document}", 1)
    return tex + "\n" + marker + "\n"


def _reference_start_page(bundle_dir: Path) -> int | None:
    aux_path = bundle_dir / "main.aux"
    if not aux_path.is_file():
        return None
    raw = aux_path.read_text(encoding="utf-8", errors="replace")
    m = re.search(
        rf"\\newlabel\{{{re.escape(MAINBODY_END_LABEL)}\}}\{{\{{(\d+)\}}",
        raw,
    )
    if m:
        return int(m.group(1))
    return None


def main_body_page_count(bundle_dir: Path) -> int | None:
    """Pages from Abstract through Conclusion (References excluded)."""
    ref_start = _reference_start_page(bundle_dir)
    if ref_start is not None and ref_start > 1:
        return ref_start - 1
    return _page_count_from_log(bundle_dir)


def audit_page_budget(bundle_dir: Path, *, template_id: str) -> dict[str, Any]:
    target = target_page_count(template_id)
    body_pages = main_body_page_count(bundle_dir)
    total_pages = _page_count_from_log(bundle_dir)
    ref_start = _reference_start_page(bundle_dir)
    if body_pages is None:
        return {
            "template_id": template_id,
            "target_pages": target,
            "page_count": None,
            "total_pages": total_pages,
            "reference_start_page": ref_start,
            "pass": False,
            "issue": "page_count_unknown",
        }
    delta = body_pages - target
    return {
        "template_id": template_id,
        "target_pages": target,
        "page_count": body_pages,
        "main_body_pages": body_pages,
        "total_pages": total_pages,
        "reference_start_page": ref_start,
        "delta": delta,
        "pass": delta == 0,
        "issue": None if delta == 0 else ("too_short" if delta < 0 else "too_long"),
    }


def _adjust_pages_with_llm(
    main_tex: str,
    *,
    template_id: str,
    state: dict,
    page_count: int,
    target_pages: int,
) -> str:
    direction = "expand" if page_count < target_pages else "trim"
    deficit = target_pages - page_count
    guidelines = build_conference_guidelines(template_id)
    system = (
        "You are a senior ML conference editor tuning LaTeX to an EXACT page budget. "
        f"The compiled PDF main body (Abstract through Conclusion, NOT References) must be exactly "
        f"{target_pages} pages: Conclusion must end on page {target_pages}; References start on page "
        f"{target_pages + 1}. "
        f"You must {direction} main-body content by approximately {abs(deficit)} page(s). "
        "Use only ASCII LaTeX math ($\\in$, $\\leq$); never Unicode math symbols. "
        "Escape underscores in prose; use \\texttt{...} for variant names in tables. "
        "Preserve all \\cite keys, tables (tab:main_results, tab:ablations), figures, and labels. "
        "Add substantive Related Work subsections, per-dataset experiment analysis, ablation discussion, "
        "and a fuller Conclusion when expanding—not filler phrases. "
        "When trimming, shorten Related Work and Discussion first, never delete ablation or main results tables. "
        "Return the full document starting with \\documentclass."
    )
    user = (
        f"--- guidelines ---\n{guidelines[:10000]}\n"
        f"--- current_main_body_pages ---\n{page_count}\n"
        f"--- target_main_body_pages ---\n{target_pages}\n"
        f"--- direction ---\n{direction}\n"
        f"--- title ---\n{state.get('title') or ''}\n"
        f"--- paper.tex ---\n{main_tex[:100000]}"
    )
    adjusted, _ = call_llm(system, user, temperature=0.15)
    text = strip_llm_wrapper_markup(adjusted or "")
    if "\\documentclass" in text[:3000]:
        return sanitize_latex_body_unicode(ensure_mainbody_end_label(text))
    return main_tex


def _page_distance(audit: dict[str, Any]) -> int:
    if not audit.get("compile_ok") or audit.get("page_count") is None:
        return 10_000
    return abs(int(audit["page_count"]) - int(audit["target_pages"]))


def tune_mainbody_linespread(
    main_tex: str,
    bundle_dir: Path,
    *,
    template_id: str,
    target_pages: int,
) -> tuple[str, dict[str, Any]]:
    """Last resort: modest \\linespread to reach exact main-body page count."""
    report: dict[str, Any] = {"target_pages": target_pages, "attempts": []}
    base = re.sub(r"\\linespread\{[^}]+\}\s*\\selectfont\s*", "", main_tex or "")
    for spread in (1.0, 1.06, 1.10, 1.14, 1.18, 1.22, 1.26, 1.30):
        probe = base
        if spread != 1.0:
            probe = probe.replace(
                r"\begin{document}",
                rf"\begin{{document}}\n\linespread{{{spread:.2f}}}\selectfont\n",
                1,
            )
        probe = ensure_mainbody_end_label(probe)
        (bundle_dir / "main.tex").write_text(probe, encoding="utf-8")
        _compile_main_pdf(bundle_dir)
        audit = audit_page_budget(bundle_dir, template_id=template_id)
        audit["linespread"] = spread
        report["attempts"].append(audit)
        if audit.get("pass"):
            report["pass"] = True
            report["final"] = audit
            report["linespread"] = spread
            return probe, report
    report["final"] = report["attempts"][-1] if report["attempts"] else {}
    return ensure_mainbody_end_label(base), report


def page_budget_blockers(report: dict[str, Any] | None, *, template_id: str = "iclr2026") -> list[str]:
    """Human-readable blockers when main-body page count != venue requirement (refs excluded)."""
    if not report:
        return [
            f"Main-body page budget: required exactly {target_page_count(template_id)} pages "
            "(Abstract--Conclusion, References excluded); audit missing."
        ]
    blockers: list[str] = []
    if not report.get("pass"):
        det = report.get("deterministic_fill") or {}
        final = (
            report.get("final")
            or report.get("best_effort")
            or det.get("final")
            or (report.get("attempts") or [{}])[-1]
            or {}
        )
        target = final.get("target_pages") or target_page_count(template_id)
        got = final.get("page_count") or final.get("main_body_pages")
        total = final.get("total_pages")
        ref_at = final.get("reference_start_page")
        issue = final.get("issue") or "page_mismatch"
        blockers.append(
            "Main-body page budget HARD FAIL: "
            f"required exactly {target} pages (Abstract through Conclusion, References excluded), "
            f"got main_body={got}, total_pdf={total}, references_start_page={ref_at}, issue={issue}."
        )
    dup_count = report.get("duplicate_paragraph_count")
    dup_findings = report.get("duplicate_paragraphs") or []
    if dup_count:
        first = dup_findings[0] if dup_findings else {}
        excerpt_a = (first.get("paragraph_a_excerpt") or "")[:80]
        blockers.append(
            "Duplicate-paragraph HARD FAIL: "
            f"{dup_count} body paragraph(s) exceed Jaccard {_DUPLICATE_JACCARD} (e.g. \"{excerpt_a}...\"). "
            "Either rewrite, or rely on deterministic_fill dedup pass."
        )
    return blockers


def apply_exact_page_budget(
    main_tex: str,
    bundle_dir: Path,
    *,
    template_id: str,
    state: dict,
    max_attempts: int = 5,
) -> tuple[str, dict[str, Any]]:
    """Compile loop: deterministic fill, then LLM adjust until main-body pages == venue max."""
    from agents.manuscript_deterministic_fill import apply_deterministic_page_fill

    strict = os.getenv("DEEPGRAPH_STRICT_PAGE_BUDGET", "1").strip().lower() not in {"0", "false", "no"}
    target = target_page_count(template_id)
    report: dict[str, Any] = {"target_pages": target, "strict": strict, "attempts": []}
    current = sanitize_latex_body_unicode(ensure_mainbody_end_label(main_tex))
    # First-pass dedupe of any duplicate paragraphs already present in the
    # writer's output (this is the "Related Theme 4..11 identical" case).
    current, dedup_removed = deduplicate_paragraphs(current)
    report["pre_dedupe_removed_paragraphs"] = dedup_removed
    current, det_report = apply_deterministic_page_fill(
        current, bundle_dir, template_id=template_id, state=state, target_pages=target
    )
    report["deterministic_fill"] = det_report
    if det_report.get("pass"):
        report["pass"] = True
        report["final"] = det_report.get("final") or {}
        dup_findings = find_duplicate_paragraphs(current)
        report["duplicate_paragraphs"] = dup_findings
        report["duplicate_paragraph_count"] = len(dup_findings)
        if dup_findings:
            current, post_removed = deduplicate_paragraphs(current)
            report["post_dedupe_removed_paragraphs"] = post_removed
            (bundle_dir / "main.tex").write_text(current, encoding="utf-8")
            _compile_main_pdf(bundle_dir)
            final = audit_page_budget(bundle_dir, template_id=template_id)
            report["final"] = final
            report["pass"] = bool(final.get("pass"))
        return current, report
    last_good_tex = current
    last_good_audit: dict[str, Any] | None = None
    best_distance = 10_000

    for attempt in range(1, max_attempts + 1):
        current = sanitize_latex_body_unicode(ensure_mainbody_end_label(current))
        (bundle_dir / "main.tex").write_text(current, encoding="utf-8")
        compile_result = _compile_main_pdf(bundle_dir)
        audit = audit_page_budget(bundle_dir, template_id=template_id)
        audit["compile_ok"] = bool(compile_result.get("ok"))
        audit["attempt"] = attempt
        report["attempts"].append(audit)

        if audit.get("compile_ok"):
            dist = _page_distance(audit)
            if dist <= best_distance:
                best_distance = dist
                last_good_tex = current
                last_good_audit = audit

        if audit.get("pass"):
            report["pass"] = True
            report["final"] = audit
            return current, report

        pages = audit.get("page_count")
        if pages is None:
            current = last_good_tex
            continue
        if not strict:
            break

        try:
            candidate = _adjust_pages_with_llm(
                current,
                template_id=template_id,
                state=state,
                page_count=int(pages),
                target_pages=target,
            )
        except RuntimeError as exc:
            report.setdefault("llm_page_adjust_errors", []).append(str(exc))
            candidate = current
        if candidate != current:
            probe = sanitize_latex_body_unicode(ensure_mainbody_end_label(candidate))
            (bundle_dir / "main.tex").write_text(probe, encoding="utf-8")
            probe_compile = _compile_main_pdf(bundle_dir)
            if probe_compile.get("ok"):
                current = probe
            else:
                current = last_good_tex

    report["pass"] = bool(report.get("attempts") and report["attempts"][-1].get("pass"))
    report["final"] = report["attempts"][-1] if report.get("attempts") else {}
    if not report["pass"] and last_good_audit:
        report["best_effort"] = last_good_audit
        current = last_good_tex
    if not report.get("pass"):
        current, spread_report = tune_mainbody_linespread(
            current, bundle_dir, template_id=template_id, target_pages=target
        )
        report["linespread_tune"] = spread_report
        if spread_report.get("pass"):
            report["pass"] = True
            report["final"] = spread_report.get("final") or {}
    (bundle_dir / "main.tex").write_text(current, encoding="utf-8")
    _compile_main_pdf(bundle_dir)
    final = audit_page_budget(bundle_dir, template_id=template_id)
    dup_findings = find_duplicate_paragraphs(current)
    report["duplicate_paragraphs"] = dup_findings
    report["duplicate_paragraph_count"] = len(dup_findings)
    if dup_findings:
        current, post_removed = deduplicate_paragraphs(current)
        report["post_dedupe_removed_paragraphs"] = post_removed
        (bundle_dir / "main.tex").write_text(current, encoding="utf-8")
        _compile_main_pdf(bundle_dir)
        final = audit_page_budget(bundle_dir, template_id=template_id)
        report["final"] = final
        report["pass"] = bool(final.get("pass"))
    return current, report
