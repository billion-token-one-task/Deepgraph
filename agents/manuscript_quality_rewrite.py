"""Closed loop on top of ``paper_quality_report``: when the audit reports
high/critical issues that *are not* the compile error itself, take direct
remedial action — drop missing citations, comment out missing figures,
reapply venue template, ask the LLM to rewrite a flagged section — then
recompile via the R1 loop. This is the missing channel between
``_paper_quality_report`` (diagnostic) and an actually-fixed bundle.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from agents.llm_client import call_llm
from agents.manuscript_compile_repair import repair_compile_loop
from agents.manuscript_submission_enrichment import strip_llm_wrapper_markup


_ISSUE_TO_HANDLER_KEY = {
    "PDF compile did not pass": "compile",
    "Referenced figures are missing or placeholder-rendered": "missing_figures",
    "The manuscript has citations that are absent from references.bib": "missing_cites",
    "Conference bundle is not using the ICLR 2026 template": "venue_template",
    "Venue section gate:": "venue_section",
    "Main-body page budget HARD FAIL": "page_budget",
    "Duplicate-paragraph HARD FAIL": "duplicate_paragraph",
    "Internal-audit wording remains": "internal_audit_wording",
    "Off-topic bibliography entries remain": "offtopic_bib",
}


_REWRITE_SECTION_SYSTEM = (
    "You are a senior ML editor fixing a specific quality issue in a "
    "conference-paper section. Output ONLY the rewritten section body (do "
    "NOT include the surrounding \\section header, packages, or other "
    "sections). Preserve every \\cite{...}, \\ref{...}, \\label{...}, and "
    "math command. Do not invent new results or numbers. Stay under 1.5x "
    "the original length."
)


@dataclass
class QualityFix:
    issue: dict[str, Any]
    handler: str
    changed: bool = False
    notes: list[str] = None  # type: ignore[assignment]
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "issue": self.issue,
            "handler": self.handler,
            "changed": self.changed,
            "notes": self.notes or [],
            "error": self.error,
        }


def _match_handler(issue_text: str) -> str | None:
    issue_text = (issue_text or "").strip()
    for prefix, key in _ISSUE_TO_HANDLER_KEY.items():
        if prefix.lower() in issue_text.lower():
            return key
    return None


# ─────────────────────────── handlers ─────────────────────────────────────


def _strip_missing_cite_keys(main_tex: str, bibtex: str) -> tuple[str, list[str]]:
    cite_keys = set(re.findall(r"\\cite[a-zA-Z]*\{([^}]+)\}", main_tex))
    flat_keys: set[str] = set()
    for blob in cite_keys:
        for k in blob.split(","):
            k = k.strip()
            if k:
                flat_keys.add(k)
    bib_keys = set(re.findall(r"@\w+\s*\{\s*([^,\s]+)\s*,", bibtex))
    missing = sorted(flat_keys - bib_keys)
    if not missing:
        return main_tex, []
    new = main_tex
    notes: list[str] = []
    for key in missing:
        # Remove only the missing key; keep other keys in the same \cite call.
        def _rewrite(match: re.Match[str]) -> str:
            inner = match.group(1)
            kept = [k.strip() for k in inner.split(",") if k.strip() and k.strip() != key]
            if not kept:
                return ""
            return match.group(0).replace(inner, ",".join(kept))

        new = re.sub(r"\\cite[a-zA-Z]*\{([^}]+)\}", _rewrite, new)
        notes.append(f"dropped missing cite key {key}")
    return new, notes


def _drop_missing_figure_includes(main_tex: str, bundle_dir: Path) -> tuple[str, list[str]]:
    pat = re.compile(r"\\begin\{figure\*?\}[\s\S]*?\\end\{figure\*?\}", re.MULTILINE)
    notes: list[str] = []
    out_blocks: list[str] = []
    last = 0
    for m in pat.finditer(main_tex):
        block = m.group(0)
        include = re.search(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", block)
        if include:
            raw = include.group(1)
            candidate = bundle_dir / raw
            if not candidate.exists() and not candidate.suffix:
                candidate = candidate.with_suffix(".png")
            if not candidate.exists():
                # Replace block with a single comment line so the surrounding flow
                # (e.g. \ref{fig:...}) stays predictable; do NOT delete \label,
                # so downstream R3 can decide whether to drop the \ref too.
                out_blocks.append(main_tex[last : m.start()])
                out_blocks.append(
                    f"% figure {raw} was removed because its asset is missing on disk\n"
                )
                last = m.end()
                notes.append(f"dropped figure environment for missing asset {raw}")
                continue
    if not notes:
        return main_tex, []
    out_blocks.append(main_tex[last:])
    return "".join(out_blocks), notes


def _drop_internal_audit_wording(main_tex: str) -> tuple[str, list[str]]:
    notes: list[str] = []
    patterns = (
        r"\b(?:our|the)?\s*available log[s]?\b",
        r"\bsupplied artifact[s]?\b",
        r"\bprovided material[s]?\b",
        r"\bexperiment artifacts?\b",
        r"\bfaithful report of the recorded evidence\b",
    )
    new = main_tex
    for pat in patterns:
        candidate = re.sub(pat, "the run results", new, flags=re.IGNORECASE)
        if candidate != new:
            notes.append(f"replaced internal-audit phrasing matching /{pat}/")
        new = candidate
    return new, notes


def _drop_offtopic_bib_entries(bibtex: str, off_terms: tuple[str, ...]) -> tuple[str, list[str]]:
    notes: list[str] = []
    if not bibtex:
        return bibtex, []
    entries = re.split(r"(?m)^(?=@)", bibtex)
    kept: list[str] = []
    for ent in entries:
        if not ent.strip():
            continue
        low = ent.lower()
        if any(term in low for term in off_terms):
            m = re.search(r"@\w+\s*\{\s*([^,]+)", ent)
            key = m.group(1).strip() if m else "?"
            notes.append(f"dropped off-topic bib entry {key}")
            continue
        kept.append(ent)
    return "".join(kept), notes


# ─────────────────────────── section rewrite ─────────────────────────────


def _extract_section_body(main_tex: str, section_name: str) -> tuple[str, int, int] | None:
    pat = re.compile(rf"\\section\*?\{{{re.escape(section_name)}\}}", re.IGNORECASE)
    m = pat.search(main_tex)
    if not m:
        return None
    start = m.end()
    next_section = re.search(r"\n\\section\*?\{", main_tex[start:])
    end = start + next_section.start() if next_section else len(main_tex)
    return main_tex[start:end], start, end


def _rewrite_section_with_llm(
    main_tex: str,
    *,
    section_name: str,
    feedback: str,
    target_words: int | None = None,
) -> tuple[str, dict[str, Any]]:
    extracted = _extract_section_body(main_tex, section_name)
    if not extracted:
        return main_tex, {"ok": False, "error": f"section_{section_name}_not_found"}
    body, start, end = extracted
    word_budget = target_words or max(180, min(700, len(body.split()) + 60))
    user = (
        f"--- quality issue ---\n{feedback}\n\n"
        f"--- section ({section_name}) body ---\n{body.strip()}\n\n"
        f"--- instructions ---\nRewrite this section body to address the "
        f"issue. Aim for approximately {word_budget} words. Preserve every "
        f"\\cite{{...}}, \\ref{{...}}, \\label{{...}}."
    )
    try:
        out, _ = call_llm(_REWRITE_SECTION_SYSTEM, user, max_tokens=3200, temperature=0.15)
    except Exception as exc:
        return main_tex, {"ok": False, "error": str(exc)}
    cleaned = strip_llm_wrapper_markup(out or "").strip()
    cleaned = re.sub(r"^```(?:latex|tex)?\s*\n?", "", cleaned)
    cleaned = re.sub(r"\n?```\s*$", "", cleaned).strip()
    if not cleaned or "\\documentclass" in cleaned[:200]:
        return main_tex, {"ok": False, "error": "empty_or_full_doc"}
    rebuilt = main_tex[:start] + "\n" + cleaned + "\n" + main_tex[end:]
    return rebuilt, {
        "ok": True,
        "section": section_name,
        "old_chars": len(body),
        "new_chars": len(cleaned),
    }


# ─────────────────────────── top-level loop ──────────────────────────────


CompileFn = Callable[[Path], dict]


def _compile_with_repair(bundle_dir: Path) -> dict:
    return repair_compile_loop(bundle_dir, compile_fn=None)


def _load_paper_quality_report(bundle_dir: Path) -> dict[str, Any]:
    path = bundle_dir / "paper_quality_report.json"
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def apply_quality_rewrite_loop(
    bundle_dir: Path,
    *,
    quality_report: dict | None = None,
    max_rounds: int = 2,
    compile_fn: CompileFn | None = None,
) -> dict[str, Any]:
    """Walk paper_quality_report issues (severity high/critical) and apply
    direct remediations. Recompile via ``repair_compile_loop`` (which itself
    will retry on LaTeX errors). Returns a report saved to
    ``quality_rewrite_log.json``.
    """
    bundle_dir = Path(bundle_dir)
    main_tex_path = bundle_dir / "main.tex"
    bibtex_path = bundle_dir / "references.bib"
    if not main_tex_path.is_file():
        return {"ok": False, "skipped": "main_tex_missing"}

    report = quality_report if isinstance(quality_report, dict) else _load_paper_quality_report(bundle_dir)
    if not report:
        return {"ok": True, "skipped": "no_quality_report"}

    all_fixes: list[dict[str, Any]] = []
    rounds: list[dict[str, Any]] = []
    compile_fn = compile_fn or _compile_with_repair
    last_compile: dict[str, Any] = {}

    OFF_TOPIC = (
        "asthma", "agriculture", "covid-19", "wildlife", "tumor", "cancer",
    )

    for round_idx in range(1, max_rounds + 1):
        round_record: dict[str, Any] = {"round": round_idx, "fixes": []}
        original_tex = main_tex_path.read_text(encoding="utf-8", errors="replace")
        bibtex = bibtex_path.read_text(encoding="utf-8", errors="replace") if bibtex_path.is_file() else ""
        current_tex = original_tex
        current_bib = bibtex
        changed_overall = False

        issues = report.get("issues") if isinstance(report.get("issues"), list) else []
        for issue in issues:
            if not isinstance(issue, dict):
                continue
            sev = str(issue.get("severity") or "").lower()
            if sev not in {"high", "critical"}:
                continue
            issue_text = str(issue.get("issue") or "")
            handler_key = _match_handler(issue_text)
            fix = QualityFix(issue=dict(issue), handler=handler_key or "unhandled")
            if not handler_key or handler_key == "compile":
                round_record["fixes"].append(fix.to_dict())
                continue
            try:
                if handler_key == "missing_cites":
                    current_tex, notes = _strip_missing_cite_keys(current_tex, current_bib)
                    fix.notes = notes
                    fix.changed = bool(notes)
                elif handler_key == "missing_figures":
                    current_tex, notes = _drop_missing_figure_includes(current_tex, bundle_dir)
                    fix.notes = notes
                    fix.changed = bool(notes)
                elif handler_key == "venue_template":
                    from agents.manuscript_templates import get_adapter
                    notes = get_adapter("iclr2026").copy_files(bundle_dir)
                    fix.notes = [f"reapplied template files: {len(notes)}"]
                    fix.changed = bool(notes)
                elif handler_key == "internal_audit_wording":
                    current_tex, notes = _drop_internal_audit_wording(current_tex)
                    fix.notes = notes
                    fix.changed = bool(notes)
                elif handler_key == "offtopic_bib":
                    current_bib, notes = _drop_offtopic_bib_entries(current_bib, OFF_TOPIC)
                    fix.notes = notes
                    fix.changed = bool(notes)
                elif handler_key == "venue_section":
                    section = re.search(
                        r"Venue section gate:\s*(.+?)(?:\.|$)",
                        issue_text,
                        flags=re.IGNORECASE,
                    )
                    section_label = section.group(1).strip() if section else "Discussion"
                    rebuilt, llm_meta = _rewrite_section_with_llm(
                        current_tex,
                        section_name=section_label,
                        feedback=issue_text,
                    )
                    fix.notes = [json.dumps(llm_meta, ensure_ascii=False, default=str)]
                    if llm_meta.get("ok"):
                        current_tex = rebuilt
                        fix.changed = True
                elif handler_key == "page_budget":
                    # Trigger a re-run of the budget loop on the next round
                    # (handled by R2). Here we just record the issue.
                    fix.notes = ["page_budget_handled_by_R2"]
                elif handler_key == "duplicate_paragraph":
                    from agents.manuscript_page_budget import deduplicate_paragraphs
                    rebuilt, n = deduplicate_paragraphs(current_tex)
                    if n:
                        current_tex = rebuilt
                        fix.notes = [f"deduplicated {n} paragraphs"]
                        fix.changed = True
                else:
                    fix.notes = ["no_handler"]
            except Exception as exc:
                fix.error = str(exc)
            round_record["fixes"].append(fix.to_dict())
            if fix.changed:
                changed_overall = True

        if current_tex != original_tex:
            backup = bundle_dir / f"main.tex.quality_rewrite_round_{round_idx}"
            backup.write_text(original_tex, encoding="utf-8")
            main_tex_path.write_text(current_tex, encoding="utf-8")
        if current_bib != bibtex and bibtex_path.is_file():
            backup = bundle_dir / f"references.bib.quality_rewrite_round_{round_idx}"
            backup.write_text(bibtex, encoding="utf-8")
            bibtex_path.write_text(current_bib, encoding="utf-8")
        if changed_overall:
            last_compile = compile_fn(bundle_dir)
            round_record["post_compile"] = {
                "ok": bool(last_compile.get("ok")),
            }
            all_fixes.extend(round_record["fixes"])
            rounds.append(round_record)
            if last_compile.get("ok"):
                # Refresh report from disk if R1 wrote a new one; otherwise stop.
                report = _load_paper_quality_report(bundle_dir) or report
                continue
            break
        else:
            round_record["post_compile"] = {"ok": None, "no_changes": True}
            rounds.append(round_record)
            break

    final = {
        "ok": bool(last_compile.get("ok")) if last_compile else True,
        "rounds": rounds,
        "total_fixes_applied": sum(1 for f in all_fixes if f.get("changed")),
    }
    (bundle_dir / "quality_rewrite_log.json").write_text(
        json.dumps(final, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    return final
