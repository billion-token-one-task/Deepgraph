"""ICML 2024 template adapter (D2 #13).

Two-column 10pt body (``icml2024.sty`` calls ``\\twocolumn`` itself),
8-page main limit + unlimited references, ``icml2024`` bibstyle from
the official upstream ``.bst`` shipped under ``third_party/icml2024/``.
"""

from __future__ import annotations

from config import ICML2024_TEMPLATE_DIR, ICML2024_TEMPLATE_FILES
from agents.manuscript_templates import register
from agents.manuscript_templates._stub_adapter import _StubVenueAdapter


@register("icml2024")
class ICML2024Adapter(_StubVenueAdapter):
    _assets_dir = ICML2024_TEMPLATE_DIR
    _asset_files = ICML2024_TEMPLATE_FILES
    _sty_basename = "icml2024"

    @property
    def venue_label(self) -> str:
        return "icml_2024"

    @property
    def column_layout(self) -> str:
        # icml2024.sty itself issues ``\twocolumn`` — declare it so the
        # FormatLinter (D3) expects ``\columnwidth`` figure widths.
        return "two_column"

    @property
    def bibstyle_name(self) -> str:
        return "icml2024"

    @property
    def max_pages(self) -> int:
        # ICML 2024: 8 content pages + unlimited references.
        return 8
