"""ICML 2024 template adapter (D2 #13).

Single-column 10pt body, 8-page main limit + unlimited references,
``icml2024`` bibstyle alias (we use ``plainnat`` as the substitution since
ICML's official ``.bst`` ships under the same name as the ``.sty``).
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
        return "single_column"

    @property
    def bibstyle_name(self) -> str:
        return "plainnat"

    @property
    def max_pages(self) -> int:
        # ICML 2024: 8 content pages + unlimited references.
        return 8
