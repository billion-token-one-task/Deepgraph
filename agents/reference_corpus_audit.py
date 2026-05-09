"""Reference-corpus profile checks for generated conference manuscripts."""

from __future__ import annotations

import re
import statistics
from pathlib import Path
from typing import Any


SECTION_RE = re.compile(r"\\(?:sub)*section\*?\{([^}]+)\}", re.IGNORECASE)
CITE_RE = re.compile(r"\\cite[tpa]?\{([^}]+)\}", re.IGNORECASE)
COMMAND_RE = re.compile(r"\\[a-zA-Z]+\*?(?:\[[^\]]*\])?(?:\{[^}]*\})?")
WORD_RE = re.compile(r"[A-Za-z][A-Za-z0-9\-']+")


def _median(values: list[float]) -> float | None:
    clean = [float(v) for v in values if v is not None]
    if not clean:
        return None
    return float(statistics.median(clean))


def _percentile(values: list[float], q: float) -> float | None:
    clean = sorted(float(v) for v in values if v is not None)
    if not clean:
        return None
    if len(clean) == 1:
        return clean[0]
    idx = max(0, min(len(clean) - 1, int(round((len(clean) - 1) * q))))
    return float(clean[idx])


def _word_count(text: str) -> int:
    return len(WORD_RE.findall(text or ""))


def _tex_plain_text(main_tex: str) -> str:
    text = re.sub(r"%.*", " ", main_tex or "")
    text = COMMAND_RE.sub(" ", text)
    text = re.sub(r"[{}$^_\\]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def generated_manuscript_profile(
    *,
    main_tex: str,
    page_count: int | None,
    figure_count: int,
    bibliography_entry_count: int,
) -> dict[str, Any]:
    sections = [s.strip() for s in SECTION_RE.findall(main_tex or "") if s.strip()]
    plain = _tex_plain_text(main_tex)
    return {
        "page_count": page_count,
        "tex_word_count": _word_count(plain),
        "section_count": len(sections),
        "sections": sections,
        "citation_command_count": len(CITE_RE.findall(main_tex or "")),
        "bibliography_entry_count": int(bibliography_entry_count or 0),
        "figure_reference_count": int(figure_count or 0),
        "table_count": len(re.findall(r"\\begin\{table", main_tex or "", re.IGNORECASE)),
        "equation_count": len(re.findall(r"\\begin\{equation|\\\[", main_tex or "", re.IGNORECASE)),
        "has_problem_motivation_spine": all(
            token in (main_tex or "").lower()
            for token in ("problem", "motivation", "method", "result")
        ),
    }


def _pdf_text_signals(path: Path, *, max_pages: int) -> dict[str, Any]:
    try:
        import fitz  # PyMuPDF
    except Exception as exc:  # pragma: no cover - depends on optional runtime
        return {"path": str(path), "error": f"pymupdf_unavailable: {exc}"}

    try:
        doc = fitz.open(str(path))
    except Exception as exc:
        return {"path": str(path), "error": f"open_failed: {exc}"}
    try:
        page_count = int(doc.page_count)
        pages = min(max_pages, page_count)
        text_parts = []
        for idx in range(pages):
            try:
                text_parts.append(doc.load_page(idx).get_text("text") or "")
            except Exception:
                continue
        text = "\n".join(text_parts)
    finally:
        doc.close()

    lower = text.lower()
    return {
        "path": str(path),
        "page_count": page_count,
        "sample_pages": pages,
        "sample_word_count": _word_count(text),
        "has_abstract": "abstract" in lower[:4000],
        "has_introduction": "introduction" in lower,
        "has_experiments": any(token in lower for token in ("experiment", "evaluation", "empirical")),
        "has_limitations": "limitation" in lower,
        "reference_marker_count": len(re.findall(r"\[[0-9]{1,3}\]|\bet al\.", text)),
    }


def profile_reference_corpus(
    corpus_dir: Path,
    *,
    max_papers: int = 32,
    max_pages_per_paper: int = 12,
) -> dict[str, Any]:
    corpus_dir = Path(corpus_dir)
    if not corpus_dir.exists():
        return {"available": False, "corpus_dir": str(corpus_dir), "error": "missing_corpus_dir"}
    pdfs = [p for p in sorted(corpus_dir.glob("*.pdf")) if p.is_file() and p.stat().st_size > 0]
    sampled = pdfs[: max(1, int(max_papers))]
    profiles = [_pdf_text_signals(path, max_pages=max_pages_per_paper) for path in sampled]
    valid = [p for p in profiles if not p.get("error")]
    return {
        "available": bool(valid),
        "corpus_dir": str(corpus_dir),
        "pdf_count": len(pdfs),
        "sampled_pdf_count": len(sampled),
        "valid_profile_count": len(valid),
        "profiles": profiles,
        "median_page_count": _median([p.get("page_count") for p in valid]),
        "p25_page_count": _percentile([p.get("page_count") for p in valid], 0.25),
        "median_sample_word_count": _median([p.get("sample_word_count") for p in valid]),
        "median_reference_marker_count": _median([p.get("reference_marker_count") for p in valid]),
        "section_signal_rates": {
            "abstract": sum(1 for p in valid if p.get("has_abstract")) / max(1, len(valid)),
            "introduction": sum(1 for p in valid if p.get("has_introduction")) / max(1, len(valid)),
            "experiments": sum(1 for p in valid if p.get("has_experiments")) / max(1, len(valid)),
            "limitations": sum(1 for p in valid if p.get("has_limitations")) / max(1, len(valid)),
        },
    }


def audit_against_reference_corpus(
    *,
    main_tex: str,
    page_count: int | None,
    figure_count: int,
    bibliography_entry_count: int,
    corpus_dir: Path,
) -> dict[str, Any]:
    generated = generated_manuscript_profile(
        main_tex=main_tex,
        page_count=page_count,
        figure_count=figure_count,
        bibliography_entry_count=bibliography_entry_count,
    )
    corpus = profile_reference_corpus(corpus_dir)
    issues: list[dict[str, str]] = []
    if not corpus.get("available"):
        issues.append({"severity": "medium", "issue": "Reference PDF corpus could not be profiled."})
    else:
        median_pages = corpus.get("median_page_count")
        if median_pages and page_count is not None and page_count < max(6, 0.55 * float(median_pages)):
            issues.append({"severity": "medium", "issue": "Generated paper is much shorter than the profiled reference corpus."})
        median_words = corpus.get("median_sample_word_count")
        if median_words and generated["tex_word_count"] < max(2500, 0.45 * float(median_words)):
            issues.append({"severity": "medium", "issue": "Generated manuscript text density is low relative to reference papers."})
    if generated["bibliography_entry_count"] < 10:
        issues.append({"severity": "medium", "issue": "Bibliography is sparse relative to top-conference reference papers."})
    if generated["figure_reference_count"] < 3:
        issues.append({"severity": "medium", "issue": "Figure count is low for an empirical top-conference submission."})
    required_sections = ("introduction", "method", "experiment")
    section_text = " ".join(generated["sections"]).lower()
    missing = [name for name in required_sections if name not in section_text]
    if missing:
        issues.append({"severity": "high", "issue": "Generated paper is missing expected section signals: " + ", ".join(missing) + "."})
    if not generated["has_problem_motivation_spine"]:
        issues.append({"severity": "high", "issue": "Problem-motivation-method-result spine is not explicit in the manuscript text."})
    return {
        "schema_version": "reference_corpus_audit_v1",
        "generated_profile": generated,
        "reference_corpus_profile": corpus,
        "issues": issues,
        "pass": not any(issue.get("severity") == "high" for issue in issues),
    }
