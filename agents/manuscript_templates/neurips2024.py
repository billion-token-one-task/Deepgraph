"""NeurIPS 2024 template adapter (D2 #13).

Single-column 10pt body, 9-page main limit + unlimited references and
checklist, ``unsrtnat`` bibstyle. See ``third_party/neurips2024/README.md``
for upstream source URL + license.
"""

from __future__ import annotations

from config import NEURIPS2024_TEMPLATE_DIR, NEURIPS2024_TEMPLATE_FILES
from agents.manuscript_templates import register
from agents.manuscript_templates._stub_adapter import _StubVenueAdapter


@register("neurips2024")
class NeurIPS2024Adapter(_StubVenueAdapter):
    _assets_dir = NEURIPS2024_TEMPLATE_DIR
    _asset_files = NEURIPS2024_TEMPLATE_FILES
    _sty_basename = "neurips_2024"

    @property
    def venue_label(self) -> str:
        return "neurips_2024"

    @property
    def column_layout(self) -> str:
        return "single_column"

    @property
    def bibstyle_name(self) -> str:
        return "unsrtnat"

    @property
    def max_pages(self) -> int:
        # NeurIPS 2024: 9 content pages + unlimited references/appendix.
        return 9

    def inject_preamble(self, source: str, *, submission_mode: bool = True) -> str:
        """Override to expose the ``[final]`` camera-ready toggle.

        ``neurips_2024.sty`` declares ``[final]`` as a real package option
        that flips the document out of the line-numbered double-blind review
        layout into the un-line-numbered camera-ready layout. See the upstream
        sty header for the full list of options (``[preprint]``, ``[nonatbib]``
        etc. — only ``final`` is plumbed through here because it's the only
        one the demo / pipeline currently needs).
        """
        out = super().inject_preamble(source)
        if not submission_mode:
            out = out.replace(
                r"\usepackage{neurips_2024}",
                r"\usepackage[final]{neurips_2024}",
                1,
            )
        return out

    def normalize_source(self, source: str, *, submission_mode: bool = True) -> str:
        """Thread ``submission_mode`` through to :meth:`inject_preamble`.

        Re-implements the parent flow so the ``submission_mode`` kwarg reaches
        ``inject_preamble``. Behaviour with ``submission_mode=True`` (default)
        is byte-equivalent to the parent's ``normalize_source``.
        """
        import re

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
        text = self.inject_preamble(text, submission_mode=submission_mode)
        bib = self.bibstyle_name
        if "\\bibliography{" in text and "\\bibliographystyle{" not in text:
            text = re.sub(
                r"(\s*)\\bibliography\{",
                rf"\1\\bibliographystyle{{{bib}}}\1\\bibliography{{",
                text,
                count=1,
            )
        text = re.sub(
            r"\\bibliographystyle\{[^}]+\}",
            rf"\\bibliographystyle{{{bib}}}",
            text,
        )
        return text + ("\n" if text and not text.endswith("\n") else "")
