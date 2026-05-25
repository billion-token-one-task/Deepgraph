"""LaTeX compile failure recovery loop.

When ``_compile_main_pdf`` returns ``ok=False`` the pipeline previously just
wrote ``pdf_compile_status.json`` and moved on, leaving the bundle marked
``manuscript_blocked``. This module is the closed loop that:

1. Reads the pdflatex log,
2. Classifies the first fatal error,
3. Attempts a *surgical* (regex-only) fix for known patterns,
4. Falls back to an LLM rewrite of a small window around the error line,
5. Recompiles. Repeats up to ``max_rounds`` times.

Each repair attempt is logged into ``compile_repair_log.json`` so downstream
audit can see what was attempted, why, and whether it landed a green compile.

The module is deliberately conservative: it never touches preamble packages
unless an explicit "file not found" classification fires, never strips
``\\cite`` or ``\\ref`` keys, and never re-runs the writer agent — the latter
is the job of R3 (``manuscript_quality_rewrite``).
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from agents.llm_client import call_llm
from agents.manuscript_submission_enrichment import (
    sanitize_main_tex_for_compile,
    strip_llm_wrapper_markup,
)


_RE_LATEX_ERROR_LINE = re.compile(r"^l\.(\d+)\s*(.*)$", re.MULTILINE)
_RE_UNDEFINED_CS = re.compile(r"!\s*Undefined control sequence\.?\s*$", re.MULTILINE | re.IGNORECASE)
_RE_FILE_NOT_FOUND = re.compile(
    r"!\s*LaTeX Error:\s*File `([^']+)' not found", re.MULTILINE
)
_RE_ENV_UNDEFINED = re.compile(
    r"!\s*LaTeX Error:\s*Environment ([^\s]+) undefined", re.MULTILINE
)
_RE_MISSING_BRACE = re.compile(r"!\s*(Missing \}|Missing \\}|Extra \}|Missing \$).*?$", re.MULTILINE)
_RE_PACKAGE_ERROR = re.compile(r"!\s*Package ([^\s]+) Error:\s*(.+?)$", re.MULTILINE)
_RE_BAD_CHAR = re.compile(r"!\s*Misplaced alignment tab character &|!\s*Missing \$ inserted", re.MULTILINE)
_RE_RUNAWAY = re.compile(r"Runaway argument\?\s*$", re.MULTILINE)
_RE_FATAL_LINE = re.compile(r"^!\s+.*$", re.MULTILINE)


# Substitutions applied non-destructively for missing-package errors. The
# key is the package name pdflatex complains about; the value is either
# ``None`` (drop the ``\usepackage`` line) or a replacement command name.
_FALLBACK_PACKAGE_DROP = {
    "apacite",          # writer sometimes leaks apacite when natbib expected
    "fontawesome",      # decorative, never required
    "fontawesome5",
    "ulem",             # \sout etc; safe to drop
    "wasysym",
    "tipa",
    "subfigure",        # deprecated, conflicts with subcaption
}


@dataclass
class CompileErrorClassification:
    kind: str
    message: str
    error_line: int | None = None
    file: str | None = None
    missing_pkg: str | None = None
    missing_env: str | None = None
    raw_error_block: str = ""
    follow_up_hints: list[str] = field(default_factory=list)


def _read_log(bundle_dir: Path) -> str:
    log = bundle_dir / "latex_compile.log"
    if log.is_file():
        try:
            return log.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return ""
    return ""


def _slice_around(text: str, idx: int, *, before: int = 200, after: int = 600) -> str:
    start = max(0, idx - before)
    end = min(len(text), idx + after)
    return text[start:end]


def classify_log_error(log_text: str) -> CompileErrorClassification | None:
    """Return a single classification for the first fatal error in the log."""
    if not log_text:
        return None
    # ----- missing package files come first; they explain "Undefined cs" downstream.
    m = _RE_FILE_NOT_FOUND.search(log_text)
    if m:
        sty = m.group(1)
        pkg = Path(sty).stem
        return CompileErrorClassification(
            kind="missing_package_file",
            message=f"missing LaTeX package file: {sty}",
            missing_pkg=pkg,
            raw_error_block=_slice_around(log_text, m.start()),
        )
    m = _RE_PACKAGE_ERROR.search(log_text)
    if m:
        return CompileErrorClassification(
            kind=f"package_error:{m.group(1).lower()}",
            message=m.group(2).strip(),
            missing_pkg=m.group(1),
            raw_error_block=_slice_around(log_text, m.start()),
        )
    m = _RE_ENV_UNDEFINED.search(log_text)
    if m:
        return CompileErrorClassification(
            kind="missing_environment",
            message=f"environment {m.group(1)} undefined",
            missing_env=m.group(1),
            raw_error_block=_slice_around(log_text, m.start()),
        )
    m = _RE_UNDEFINED_CS.search(log_text)
    if m:
        line_m = _RE_LATEX_ERROR_LINE.search(log_text, m.end())
        line = int(line_m.group(1)) if line_m else None
        return CompileErrorClassification(
            kind="undefined_control_sequence",
            message="undefined control sequence",
            error_line=line,
            raw_error_block=_slice_around(log_text, m.start()),
        )
    m = _RE_MISSING_BRACE.search(log_text)
    if m:
        line_m = _RE_LATEX_ERROR_LINE.search(log_text, m.end())
        line = int(line_m.group(1)) if line_m else None
        return CompileErrorClassification(
            kind="brace_mismatch",
            message=m.group(0).strip(),
            error_line=line,
            raw_error_block=_slice_around(log_text, m.start()),
        )
    m = _RE_BAD_CHAR.search(log_text)
    if m:
        line_m = _RE_LATEX_ERROR_LINE.search(log_text, m.end())
        return CompileErrorClassification(
            kind="bad_char_or_alignment",
            message=m.group(0).strip(),
            error_line=int(line_m.group(1)) if line_m else None,
            raw_error_block=_slice_around(log_text, m.start()),
        )
    m = _RE_RUNAWAY.search(log_text)
    if m:
        line_m = _RE_LATEX_ERROR_LINE.search(log_text, m.end())
        return CompileErrorClassification(
            kind="runaway_argument",
            message="runaway argument (unterminated brace/quote)",
            error_line=int(line_m.group(1)) if line_m else None,
            raw_error_block=_slice_around(log_text, m.start()),
        )
    # Generic fallback: any `! ` line.
    m = _RE_FATAL_LINE.search(log_text)
    if m:
        line_m = _RE_LATEX_ERROR_LINE.search(log_text, m.end())
        return CompileErrorClassification(
            kind="generic_fatal",
            message=m.group(0).strip()[:240],
            error_line=int(line_m.group(1)) if line_m else None,
            raw_error_block=_slice_around(log_text, m.start()),
        )
    return None


# ─────────────────────────── surgical fixes ──────────────────────────────


def _drop_usepackage(main_tex: str, pkg: str) -> tuple[str, bool]:
    pat = re.compile(
        rf"\\usepackage(?:\[[^\]]*\])?\{{[^}}]*\b{re.escape(pkg)}\b[^}}]*\}}\s*\n?",
    )
    new = pat.sub("", main_tex)
    return new, new != main_tex


def _surrogate_apacite(main_tex: str) -> tuple[str, bool]:
    """Map apacite citation macros to natbib equivalents already used by ICLR."""
    changed = False
    new = main_tex
    for src, dst in (
        (r"\\citeNP\b", r"\\cite"),
        (r"\\citeA\b", r"\\cite"),
        (r"\\citeyearNP\b", r"\\citeyear"),
    ):
        candidate = re.sub(src, dst, new)
        if candidate != new:
            changed = True
        new = candidate
    return new, changed


def _comment_out_environment(main_tex: str, env: str) -> tuple[str, bool]:
    """Wrap an undefined environment in ``\\iffalse … \\fi`` to silence it."""
    open_pat = re.compile(rf"\\begin\{{{re.escape(env)}\}}", re.MULTILINE)
    close_pat = re.compile(rf"\\end\{{{re.escape(env)}\}}", re.MULTILINE)
    new = open_pat.sub(r"% \\begin{" + env + r"} (silenced by compile_repair)", main_tex)
    new = close_pat.sub(r"% \\end{" + env + r"} (silenced by compile_repair)", new)
    return new, new != main_tex


def _escape_stray_percent_amp(line: str) -> str:
    """Inside prose, ``%`` is a comment marker; ``&`` only legal in tabular."""
    out = []
    i = 0
    while i < len(line):
        ch = line[i]
        if ch == "%" and (i == 0 or line[i - 1] != "\\"):
            out.append(r"\%")
        elif ch == "&" and (i == 0 or line[i - 1] != "\\"):
            out.append(r"\&")
        elif ch == "_" and (i == 0 or line[i - 1] != "\\"):
            out.append(r"\_")
        else:
            out.append(ch)
        i += 1
    return "".join(out)


def _surgical_attempt(main_tex: str, cls: CompileErrorClassification) -> tuple[str, list[str]]:
    """Apply non-LLM repairs we are confident about."""
    notes: list[str] = []
    tex = main_tex
    if cls.kind == "missing_package_file" and cls.missing_pkg:
        pkg = cls.missing_pkg
        if pkg.lower() in _FALLBACK_PACKAGE_DROP:
            tex, changed = _drop_usepackage(tex, pkg)
            if changed:
                notes.append(f"dropped \\usepackage{{{pkg}}}")
            if pkg.lower() == "apacite":
                tex, swapped = _surrogate_apacite(tex)
                if swapped:
                    notes.append("rewrote apacite macros to natbib equivalents")
    if cls.kind == "missing_environment" and cls.missing_env:
        tex, changed = _comment_out_environment(tex, cls.missing_env)
        if changed:
            notes.append(f"silenced undefined environment {cls.missing_env}")
    if cls.kind == "bad_char_or_alignment" and cls.error_line:
        lines = tex.splitlines()
        idx = cls.error_line - 1
        if 0 <= idx < len(lines):
            patched = _escape_stray_percent_amp(lines[idx])
            if patched != lines[idx]:
                lines[idx] = patched
                tex = "\n".join(lines)
                notes.append(f"escaped stray %/&/_ at line {cls.error_line}")
    if cls.kind == "undefined_control_sequence":
        # The single most frequent leak: apacite \citeA when natbib is loaded.
        tex, swapped = _surrogate_apacite(tex)
        if swapped:
            notes.append("rewrote apacite macros to natbib equivalents")
    return tex, notes


# ─────────────────────────── LLM rewrite ────────────────────────────────


_REWRITE_SYSTEM = (
    "You are a senior LaTeX engineer fixing a pdflatex error. Rewrite ONLY "
    "the block below. Do NOT add new section headers. Preserve every "
    "\\cite{...}, \\ref{...}, \\label{...}, \\includegraphics, and \\bibliography "
    "key exactly. Do not introduce new packages. Output ONLY the rewritten "
    "block (no markdown, no explanation). The rewritten block must compile "
    "with pdflatex+natbib under the iclr2026_conference template."
)


def _llm_rewrite_block(
    main_tex: str,
    *,
    line: int,
    error_message: str,
    context_radius: int = 12,
) -> tuple[str, dict[str, Any]]:
    lines = main_tex.splitlines()
    if line < 1 or line > len(lines):
        return main_tex, {"attempted": False, "reason": "line_out_of_range"}
    start = max(0, line - 1 - context_radius)
    end = min(len(lines), line - 1 + context_radius + 1)
    block = "\n".join(lines[start:end])
    user = (
        f"--- pdflatex error ---\n{error_message}\n\n"
        f"--- offending block (lines {start + 1}..{end}) ---\n{block}\n"
    )
    try:
        out, _ = call_llm(_REWRITE_SYSTEM, user, max_tokens=4000, temperature=0.05)
    except Exception as exc:
        return main_tex, {"attempted": True, "ok": False, "error": str(exc)}
    cleaned = strip_llm_wrapper_markup(out or "").strip()
    cleaned = re.sub(r"^```(?:latex|tex)?\s*\n?", "", cleaned)
    cleaned = re.sub(r"\n?```\s*$", "", cleaned).strip()
    if not cleaned or "\\documentclass" in cleaned[:200]:
        return main_tex, {"attempted": True, "ok": False, "error": "empty_or_full_doc"}
    rebuilt = "\n".join(lines[:start] + cleaned.splitlines() + lines[end:])
    return rebuilt, {
        "attempted": True,
        "ok": True,
        "rewritten_lines": f"{start + 1}-{end}",
        "block_chars": len(block),
        "new_block_chars": len(cleaned),
    }


# ─────────────────────────── top-level loop ─────────────────────────────


CompileFn = Callable[[Path], dict]


def _default_compile(bundle_dir: Path) -> dict:
    from agents.paper_orchestra_pipeline import _compile_main_pdf as _impl

    return _impl(bundle_dir)


def repair_compile_loop(
    bundle_dir: Path,
    *,
    max_rounds: int | None = None,
    compile_fn: CompileFn | None = None,
) -> dict[str, Any]:
    """Run compile → diagnose → repair → recompile up to ``max_rounds`` times.

    Returns a structured report that callers persist to
    ``compile_repair_log.json``. ``compile_fn`` is injected for tests.
    """
    bundle_dir = Path(bundle_dir)
    main_tex_path = bundle_dir / "main.tex"
    rounds: list[dict[str, Any]] = []
    if max_rounds is None:
        try:
            max_rounds = int(os.getenv("DEEPGRAPH_COMPILE_REPAIR_ROUNDS", "3"))
        except ValueError:
            max_rounds = 3
    compile_fn = compile_fn or _default_compile
    last_compile = compile_fn(bundle_dir)
    if last_compile.get("ok"):
        return {
            "ok": True,
            "rounds": [],
            "initial_ok": True,
            "final_compile": last_compile,
        }
    for round_idx in range(1, max_rounds + 1):
        log_text = _read_log(bundle_dir)
        cls = classify_log_error(log_text)
        round_record: dict[str, Any] = {
            "round": round_idx,
            "classification": cls.__dict__ if cls else None,
            "surgical_notes": [],
            "llm_rewrite": None,
            "post_compile": None,
        }
        if cls is None:
            round_record["skipped"] = "no_classification"
            rounds.append(round_record)
            break
        original_tex = main_tex_path.read_text(encoding="utf-8", errors="replace")
        sanitized, sanitize_notes = sanitize_main_tex_for_compile(original_tex)
        # apply structural sanitizer first; then surgical fixes
        candidate, surgical_notes = _surgical_attempt(sanitized, cls)
        if surgical_notes or sanitize_notes:
            round_record["surgical_notes"] = surgical_notes
            round_record["sanitize_notes"] = sanitize_notes
        used_llm = False
        if candidate == original_tex and cls.error_line:
            llm_out, llm_meta = _llm_rewrite_block(
                candidate,
                line=cls.error_line,
                error_message=f"{cls.kind}: {cls.message}\n\n{cls.raw_error_block}",
            )
            round_record["llm_rewrite"] = llm_meta
            if llm_meta.get("ok"):
                candidate = llm_out
                used_llm = True
        if candidate == original_tex:
            round_record["skipped"] = "no_repair_applied"
            rounds.append(round_record)
            break
        backup_name = f"main.tex.compile_repair_round_{round_idx}"
        (bundle_dir / backup_name).write_text(original_tex, encoding="utf-8")
        main_tex_path.write_text(candidate, encoding="utf-8")
        post = compile_fn(bundle_dir)
        round_record["post_compile"] = {
            "ok": bool(post.get("ok")),
            "returncode": post.get("returncode"),
        }
        rounds.append(round_record)
        last_compile = post
        if post.get("ok"):
            break
    report = {
        "ok": bool(last_compile.get("ok")),
        "rounds": rounds,
        "initial_ok": False,
        "final_compile": last_compile,
    }
    (bundle_dir / "compile_repair_log.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    return report
