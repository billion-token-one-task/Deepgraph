"""CVPR 2024 template adapter (D2 #13).

Two-column body, 8-page main limit + unlimited references,
``ieeenat_fullname`` bibstyle (the natbib variant of IEEE's
``IEEEtran`` style used by CVPR's official author kit). CVPR is the
canonical computer-vision venue and pairs naturally with the
vision/recognition router triggers in ``venues_v1.yaml``.
"""

from __future__ import annotations

from config import CVPR2024_TEMPLATE_DIR, CVPR2024_TEMPLATE_FILES
from agents.manuscript_templates import register
from agents.manuscript_templates._stub_adapter import _StubVenueAdapter


@register("cvpr2024")
class CVPR2024Adapter(_StubVenueAdapter):
    _assets_dir = CVPR2024_TEMPLATE_DIR
    _asset_files = CVPR2024_TEMPLATE_FILES
    _sty_basename = "cvpr"

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
