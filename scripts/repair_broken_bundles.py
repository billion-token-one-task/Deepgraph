#!/usr/bin/env python3
"""Repair LaTeX-corrupted manuscript bundles (idea 21, 42, 44, 57, ...).

For each requested idea:
  * read ``papers/bundles/conference/main.tex``
  * run :func:`sanitize_main_tex_for_compile` (strips think-blocks, splices
    inner ``\\documentclass`` blocks, dedupes abstract + cleveref, relocates
    content trapped after ``\\end{document}``, etc).
  * back up the original to ``main.tex.broken``
  * write the repaired tex
  * compile with pdflatex/bibtex/pdflatex/pdflatex
  * report page count + size + first 25 lines of any compile error
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agents.manuscript_submission_enrichment import sanitize_main_tex_for_compile  # noqa: E402


def _bundle_dir(idea_id: int) -> Path:
    return Path(f"/root/deepgraph_ideas/idea_{idea_id}/papers/bundles/conference")


def _run(cmd: list[str], cwd: Path) -> tuple[int, str]:
    proc = subprocess.run(
        cmd,
        cwd=str(cwd),
        capture_output=True,
        text=True,
        timeout=300,
    )
    return proc.returncode, (proc.stdout or "") + (proc.stderr or "")


def _compile(bundle_dir: Path) -> tuple[bool, int | None, str]:
    """Return (ok, pages, tail-of-log)."""

    rc1, log1 = _run(
        ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "main.tex"],
        bundle_dir,
    )
    _run(["bibtex", "main"], bundle_dir)
    _run(["pdflatex", "-interaction=nonstopmode", "main.tex"], bundle_dir)
    rc3, log3 = _run(
        ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "main.tex"],
        bundle_dir,
    )
    pdf_path = bundle_dir / "main.pdf"
    if not pdf_path.is_file():
        tail = "\n".join((log3 or log1).splitlines()[-30:])
        return False, None, tail
    pages: int | None = None
    try:
        proc = subprocess.run(
            ["pdfinfo", str(pdf_path)], capture_output=True, text=True, timeout=30
        )
        for line in (proc.stdout or "").splitlines():
            if line.startswith("Pages:"):
                pages = int(line.split(":", 1)[1].strip())
                break
    except Exception:
        pass
    if pages is None:
        try:
            from pypdf import PdfReader  # type: ignore

            pages = len(PdfReader(str(pdf_path)).pages)
        except Exception:
            try:
                from PyPDF2 import PdfReader as _LegacyReader  # type: ignore

                pages = len(_LegacyReader(str(pdf_path)).pages)
            except Exception:
                pages = None
    return True, pages, "ok"


def repair(idea_id: int) -> dict:
    bundle = _bundle_dir(idea_id)
    main_tex = bundle / "main.tex"
    if not main_tex.is_file():
        return {"idea": idea_id, "status": "missing_main_tex"}

    original = main_tex.read_text(encoding="utf-8")
    repaired, notes = sanitize_main_tex_for_compile(original)
    if repaired == original:
        notes["no_changes_needed"] = True

    backup = bundle / "main.tex.broken"
    if not backup.is_file():
        backup.write_text(original, encoding="utf-8")
    main_tex.write_text(repaired, encoding="utf-8")

    ok, pages, tail = _compile(bundle)
    return {
        "idea": idea_id,
        "bundle": str(bundle),
        "notes": notes,
        "compiled": ok,
        "pages": pages,
        "log_tail": tail if not ok else "",
        "pdf": str(bundle / "main.pdf") if ok else "",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("ideas", nargs="*", type=int, default=[21, 42, 44, 57])
    args = parser.parse_args()
    results = [repair(i) for i in args.ideas]
    print(json.dumps(results, indent=2, ensure_ascii=False))
    return 0 if all(r.get("compiled") for r in results) else 2


if __name__ == "__main__":
    raise SystemExit(main())
