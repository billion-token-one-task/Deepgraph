"""CVPR 2024 template adapter (D2 #13).

Two-column body, 8-page main limit + unlimited references,
``ieeenat_fullname`` bibstyle (the natbib variant of IEEE's
``IEEEtran`` style used by CVPR's official author kit). CVPR is the
canonical computer-vision venue and pairs naturally with the
vision/recognition router triggers in ``venues_v1.yaml``.
"""

from __future__ import annotations

import re

from config import CVPR2024_TEMPLATE_DIR, CVPR2024_TEMPLATE_FILES
from agents.manuscript_templates import register
from agents.manuscript_templates._stub_adapter import _StubVenueAdapter


@register("cvpr2024")
class CVPR2024Adapter(_StubVenueAdapter):
    _assets_dir = CVPR2024_TEMPLATE_DIR
    _asset_files = CVPR2024_TEMPLATE_FILES
    _sty_basename = "cvpr"
    # CVPR's upstream .sty defaults to camera-ready (``\toggletrue{cvprfinal}``)
    # — flip into line-numbered double-blind review with ``[review]``.
    _submission_option = "review"

    @property
    def venue_label(self) -> str:
        return "cvpr_2024"

    @property
    def column_layout(self) -> str:
        return "two_column"

    @property
    def bibstyle_name(self) -> str:
        return "ieeenat_fullname"

    @property
    def max_pages(self) -> int:
        # CVPR 2024: 8 content pages + unlimited references.
        return 8

    def inject_preamble(self, source: str, *, submission_mode: bool = True) -> str:
        # ``[review]`` mode references ``\confName``/``\confYear``/``\paperID``
        # in the page-header box — undefined macros otherwise crash the build.
        text = super().inject_preamble(source, submission_mode=submission_mode)
        if submission_mode and "\\confName" not in text:
            # ``re.sub`` repl: each literal backslash needs ``\\\\`` so the
            # final emitted LaTeX is ``\def\confName{...}`` etc.
            macros = (
                "\\\\def\\\\confName{CVPR}\n"
                "\\\\def\\\\confYear{2024}\n"
                "\\\\def\\\\paperID{0000}\n"
            )
            text = re.sub(
                r"(\\usepackage\[review\]\{cvpr\}\s*\n)",
                rf"\1{macros}",
                text,
                count=1,
            )
        return text
