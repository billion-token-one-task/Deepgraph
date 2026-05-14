"""arXiv plain-article template adapter.

Pairs the ``\\documentclass{article}`` + plain bibstyle defaults used by
``paperorchestra_arxiv2604/`` for technical-report-style submissions where
no specific venue style file is required. This is the fall-through choice
for theory-only / position-paper insights that the VenueRouter rejects
from peer-reviewed venues.

Replicates the legacy ``normalize_latex_source(text, force_iclr2026=False)``
branch byte-for-byte so the ``bundle_format != "conference"`` output stays
identical to the pre-refactor build.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import List

from agents.manuscript_templates import TemplateAdapter, register


@register("arxiv_plain")
class ArxivPlainAdapter(TemplateAdapter):
    """Adapter for arXiv-style plain article submissions."""

    @property
    def venue_label(self) -> str:
        return "arxiv"

    @property
    def column_layout(self) -> str:
        # Standard ``article`` class with margin=1in geometry → single column.
        return "single_column"

    @property
    def bibstyle_name(self) -> str:
        return "plain"

    @property
    def max_pages(self) -> int:
        # arXiv has no hard page limit; use a generous soft target so the
        # FormatLinter only warns on truly absurd page counts.
        return 30

    def copy_files(self, bundle_dir: Path) -> List[str]:
        # arXiv plain documents do not ship a venue-specific style; the
        # ``article`` class plus ``plain`` bibstyle is in every TeX install.
        return []

    def inject_preamble(self, source: str, *, submission_mode: bool = True) -> str:
        """No-op for arXiv plain: only ensure a ``\\documentclass`` exists.

        Idempotent by construction. ``submission_mode`` is accepted for
        contract uniformity with venue adapters but has no effect — arXiv
        ships plain articles in a single mode.
        """
        del submission_mode  # arXiv plain has no review/camera-ready distinction
        if "\\begin{document}" not in source:
            return source
        preamble, marker, body = source.partition(r"\begin{document}")
        if r"\documentclass" not in preamble:
            preamble = r"\documentclass{article}" + "\n" + preamble
        return preamble + marker + body

    def normalize_source(self, source: str, *, submission_mode: bool = True) -> str:
        """Replicate ``normalize_latex_source(text, force_iclr2026=False)``.

        ``submission_mode`` accepted for contract uniformity; arXiv plain
        has no review/camera-ready distinction so it is intentionally unused.
        """
        del submission_mode
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
        # NOTE: arXiv plain branch does NOT call inject_preamble — this
        # matches the legacy ``if force_iclr2026: source = ...`` guard.
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
        # Standard packages a typical full paper relies on (tables/figures/
        # math/hyperlinks). Idempotent: only injected when the body uses the
        # corresponding macro and the package isn't already loaded.
        if "\\begin{document}" in text and not uses_iclr:
            preamble_chk, marker_chk, body_chk = text.partition(r"\begin{document}")
            extra_pkgs: list[str] = []
            if ("\\toprule" in body_chk or "\\midrule" in body_chk or "\\bottomrule" in body_chk) \
                    and "booktabs" not in preamble_chk:
                extra_pkgs.append("booktabs")
            if "\\includegraphics" in body_chk and "graphicx" not in preamble_chk:
                extra_pkgs.append("graphicx")
            if ("\\href" in body_chk or "\\url" in body_chk) and "hyperref" not in preamble_chk:
                extra_pkgs.append("hyperref")
            if extra_pkgs:
                preamble_chk = preamble_chk.rstrip() + "\n" + "\n".join(
                    rf"\usepackage{{{p}}}" for p in extra_pkgs
                ) + "\n"
                text = preamble_chk + marker_chk + body_chk
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
