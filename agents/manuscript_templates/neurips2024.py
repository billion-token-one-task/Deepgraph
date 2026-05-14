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
    _sty_basename = "neurips2024"

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
