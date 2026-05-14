"""Shared base for the D2 (#13) stub-style venue adapters.

NeurIPS / ICML / ACL-ARR / CVPR all follow the same minimal pattern in this
PR: ship a per-venue ``.sty`` stub plus a README documenting source URL +
license, then have the adapter expose four properties (``venue_label``,
``column_layout``, ``bibstyle_name``, ``max_pages``) and implement the
three required methods (``copy_files`` / ``inject_preamble`` /
``normalize_source``).

Factoring the shared body here keeps each individual adapter under ~50
lines and avoids drifting per-venue copies of the same loop.
"""

from __future__ import annotations

import re
import shutil
from pathlib import Path
from typing import Iterable, List

from agents.manuscript_templates import TemplateAdapter


class _StubVenueAdapter(TemplateAdapter):
    """Common machinery for venues that ship a stub ``.sty`` + README.

    Subclasses set ``_assets_dir`` and ``_asset_files`` at class level and
    override the four metadata properties. The three required methods are
    implemented here.
    """

    _assets_dir: Path = Path("/dev/null")  # overridden by subclass
    _asset_files: List[str] = []  # overridden by subclass
    _sty_basename: str = ""  # e.g. "neurips2024" → loads neurips2024.sty

    def copy_files(self, bundle_dir: Path) -> List[str]:
        copied: list[str] = []
        if not self._assets_dir.exists():
            return copied
        for name in self._asset_files:
            src = self._assets_dir / name
            if not src.exists():
                continue
            dst = bundle_dir / name
            shutil.copy2(src, dst)
            copied.append(name)
        return copied

    def inject_preamble(self, source: str) -> str:
        """Idempotent ``\\usepackage{<venue_sty>}`` insertion.

        Mirrors the ICLR adapter's contract: no-op once the venue package
        is already referenced; otherwise wedges it after ``\\documentclass``
        so subsequent normalisation passes have a stable preamble shape.

        For ``two_column`` venues the documentclass gets a ``twocolumn``
        option so the rendered PDF actually flips to two columns even when
        the shipped venue ``.sty`` is a stub that doesn't set it itself.
        """
        if "\\begin{document}" not in source:
            return source
        preamble, marker, body = source.partition(r"\begin{document}")
        wants_twocolumn = self.column_layout == "two_column"
        if r"\documentclass" not in preamble:
            opts = "[twocolumn]" if wants_twocolumn else ""
            preamble = rf"\documentclass{opts}{{article}}" + "\n" + preamble
        elif wants_twocolumn and "twocolumn" not in preamble.split(marker, 1)[0]:
            # Splice ``twocolumn`` into the existing options list (or add a
            # fresh option block) without disturbing the documentclass arg.
            def _add_twocolumn(m: "re.Match[str]") -> str:
                existing = m.group(1)
                cls_arg = m.group(2)
                if existing is None:
                    return rf"\documentclass[twocolumn]{{{cls_arg}}}"
                if "twocolumn" in existing:
                    return m.group(0)
                return rf"\documentclass[{existing},twocolumn]{{{cls_arg}}}"
            preamble = re.sub(
                r"\\documentclass(?:\[([^\]]*)\])?\{([^}]+)\}",
                _add_twocolumn,
                preamble,
                count=1,
            )
        sty = self._sty_basename
        if sty and sty not in preamble:
            preamble = re.sub(
                r"(\\documentclass(?:\[[^\]]*\])?\{[^}]+\}\s*)",
                rf"\1\\usepackage{{{sty}}}" + "\n",
                preamble,
                count=1,
            )
        # Standard math/graphics packages every venue body relies on.
        for package in ("graphicx", "booktabs", "amsmath,amssymb", "hyperref"):
            first_pkg = package.split(",", 1)[0]
            if first_pkg not in preamble:
                preamble = preamble.rstrip() + "\n" + rf"\usepackage{{{package}}}" + "\n"
        return preamble + marker + body

    def normalize_source(self, source: str) -> str:
        """Apply a minimal-but-deterministic venue normalisation.

        Steps (all idempotent):
          1. Strip ``` fences (markdown leak guard).
          2. Run ``inject_preamble`` so the venue ``.sty`` is referenced.
          3. Force the configured ``bibstyle_name`` on any ``\\bibliography``.
          4. Ensure a trailing newline.
        """
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
        bib = self.bibstyle_name
        if "\\bibliography{" in text and "\\bibliographystyle{" not in text:
            text = re.sub(
                r"(\s*)\\bibliography\{",
                rf"\1\\bibliographystyle{{{bib}}}\1\\bibliography{{",
                text,
                count=1,
            )
        # If a foreign bibstyle already sits in the document, rewrite it so
        # the venue-specific style wins.
        text = re.sub(
            r"\\bibliographystyle\{[^}]+\}",
            rf"\\bibliographystyle{{{bib}}}",
            text,
        )
        return text + ("\n" if text and not text.endswith("\n") else "")


def asset_iter(files: Iterable[str]) -> List[str]:
    """Helper for adapter modules to declare their asset list inline."""
    return list(files)
