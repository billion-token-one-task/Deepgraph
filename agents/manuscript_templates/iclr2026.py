"""ICLR 2026 template adapter.

Mirrors byte-for-byte the legacy ``_copy_iclr2026_template_files`` /
``_ensure_iclr2026_preamble`` / ``normalize_latex_source(force_iclr2026=True)``
behaviour from ``agents.paper_orchestra_pipeline``. The pipeline now calls
this adapter; this module is the single source of truth for ICLR-flavoured
LaTeX normalisation.

The legacy functions in ``paper_orchestra_pipeline`` remain as thin shims
that delegate here so byte-level output of the ``conference`` bundle format
is identical to the pre-refactor build (verified via diff fixture in
``tests/test_template_adapter.py``).
"""

from __future__ import annotations

import re
import shutil
from pathlib import Path
from typing import List

from config import ICLR2026_TEMPLATE_DIR, ICLR2026_TEMPLATE_FILES
from agents.manuscript_templates import TemplateAdapter, register


@register("iclr2026")
class ICLR2026Adapter(TemplateAdapter):
    """Adapter for the official ICLR 2026 conference template."""

    @property
    def venue_label(self) -> str:
        return "iclr2026_conference"

    @property
    def bibstyle_name(self) -> str:
        return "iclr2026_conference"

    @property
    def max_pages(self) -> int:
        # ICLR main paper budget (excluding appendix); used by FormatLinter.
        return 9

    def copy_files(self, bundle_dir: Path) -> List[str]:
        copied: list[str] = []
        if not ICLR2026_TEMPLATE_DIR.exists():
            return copied
        for name in ICLR2026_TEMPLATE_FILES:
            src = ICLR2026_TEMPLATE_DIR / name
            if not src.exists():
                continue
            dst = bundle_dir / name
            shutil.copy2(src, dst)
            copied.append(name)
        return copied

    def inject_preamble(self, source: str) -> str:
        """Force an ICLR 2026 submission preamble without touching the body.

        Idempotent: if the document already carries the ICLR preamble, the
        ``"iclr2026_conference" not in preamble`` / ``"math_commands.tex"
        not in preamble`` / per-package guards short-circuit each insertion.
        """
        if "\\begin{document}" not in source:
            return source
        preamble, marker, body = source.partition(r"\begin{document}")
        if r"\documentclass" not in preamble:
            preamble = r"\documentclass{article}" + "\n" + preamble
        if "iclr2026_conference" not in preamble:
            preamble = re.sub(
                r"(\\documentclass(?:\[[^\]]*\])?\{[^}]+\}\s*)",
                r"\1\\usepackage{iclr2026_conference,times}" + "\n",
                preamble,
                count=1,
            )
        if "math_commands.tex" not in preamble:
            preamble = preamble.rstrip() + "\n" + r"\input{math_commands.tex}" + "\n"
        for package in ("graphicx", "booktabs", "amsmath,amssymb", "hyperref", "url"):
            first_pkg = package.split(",", 1)[0]
            if first_pkg not in preamble:
                preamble = preamble.rstrip() + "\n" + rf"\usepackage{{{package}}}" + "\n"
        if r"\author" not in preamble:
            preamble = preamble.rstrip() + "\n" + r"\author{Anonymous authors\\Paper under double-blind review}" + "\n"
        preamble = re.sub(r"\\usepackage(?:\[[^\]]*\])?\{geometry\}\s*", "", preamble)
        return preamble + marker + body

    def normalize_source(self, source: str) -> str:
        """Replicate ``normalize_latex_source(text, force_iclr2026=True)``."""
        text = (source or "").strip()
        if text.startswith("```"):
            lines = text.splitlines()
            if lines and lines[0].strip().startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            text = "\n".join(lines).strip()
        if "```" in text:
            text = text.replace("```latex", "").replace("```tex", "").replace("```", "").strip()
        text = self.inject_preamble(text)
        uses_iclr = "iclr2026_conference" in text
        if not uses_iclr:
            text = re.sub(r"\\documentclass\{article\}", r"\\documentclass[10pt]{article}", text, count=1)
        if "\\begin{document}" in text and "microtype" not in text and not uses_iclr:
            text = re.sub(
                r"(\\documentclass(?:\[[^\]]*\])?\{[^}]+\}\s*)",
                r"\1\n\\usepackage{microtype}\n",
                text,
                count=1,
            )
        if "\\begin{document}" in text and "geometry" not in text and not uses_iclr:
            text = re.sub(
                r"(\\documentclass(?:\[[^\]]*\])?\{[^}]+\}\s*)",
                r"\1\n\\usepackage[margin=1in]{geometry}\n",
                text,
                count=1,
            )
        preamble_probe, marker_probe, _body_probe = text.partition(r"\begin{document}")
        if marker_probe and r"\date" not in preamble_probe and not uses_iclr:
            text = preamble_probe.rstrip() + "\n\\date{}\n" + marker_probe + _body_probe
        text = re.sub(
            r"(\\maketitle\s*)\\section\{Abstract\}\s*(.*?)(?=\\section\{Introduction\})",
            r"\1\\begin{abstract}\n\2\n\\end{abstract}\n",
            text,
            count=1,
            flags=re.DOTALL,
        )
        if "\\bibliography{" in text and "\\bibliographystyle{" not in text:
            style = "iclr2026_conference" if uses_iclr else "plain"
            text = re.sub(
                r"(\s*)\\bibliography\{",
                rf"\1\\bibliographystyle{{{style}}}\1\\bibliography{{",
                text,
                count=1,
            )
        if uses_iclr:
            text = re.sub(
                r"\\bibliographystyle\{[^}]+\}",
                r"\\bibliographystyle{iclr2026_conference}",
                text,
            )
        preamble, marker, body = text.partition(r"\begin{document}")
        if marker:
            needs_cleveref = ("\\Cref" in body or "\\cref" in body) and "cleveref" not in preamble
            needs_ams = (
                any(cmd in body for cmd in ("\\mathbb", "\\operatorname", "\\text", "\\eqref"))
                and "amsmath" not in preamble
            )
            if needs_ams or needs_cleveref:
                if needs_ams and "cleveref" in preamble:
                    preamble = preamble.replace(
                        r"\usepackage{cleveref}",
                        r"\usepackage{amsmath,amssymb}" + "\n" + r"\usepackage{cleveref}",
                        1,
                    )
                    needs_ams = False
                additions = []
                if needs_ams:
                    additions.append(r"\usepackage{amsmath,amssymb}")
                if needs_cleveref:
                    additions.append(r"\usepackage{cleveref}")
                if additions:
                    preamble = preamble.rstrip() + "\n" + "\n".join(additions) + "\n"
                text = preamble + marker + body
            elif "cleveref" in preamble and "amsmath" in preamble:
                clever_idx = preamble.find("cleveref")
                ams_idx = preamble.find("amsmath")
                if clever_idx >= 0 and ams_idx >= 0 and clever_idx < ams_idx:
                    preamble = preamble.replace(r"\usepackage{cleveref}", "")
                    preamble = preamble.replace(
                        r"\usepackage{amsmath,amssymb}",
                        r"\usepackage{amsmath,amssymb}" + "\n" + r"\usepackage{cleveref}",
                    )
                    text = preamble + marker + body
        return text + ("\n" if text and not text.endswith("\n") else "")
