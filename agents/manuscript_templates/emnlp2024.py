"""EMNLP 2024 template adapter.

Uses the same ACL Anthology ``acl.sty`` assets as ACL ARR; differs in
``template_id``, routing keywords, and venue-specific style guide
(``prompts/venue_styles/emnlp2024.md``).
"""

from __future__ import annotations

from config import EMNLP2024_TEMPLATE_DIR, EMNLP2024_TEMPLATE_FILES
from agents.manuscript_templates import register
from agents.manuscript_templates._stub_adapter import _StubVenueAdapter


@register("emnlp2024")
class EMNLP2024Adapter(_StubVenueAdapter):
    _assets_dir = EMNLP2024_TEMPLATE_DIR
    _asset_files = EMNLP2024_TEMPLATE_FILES
    _sty_basename = "acl"
    _submission_option = "review"

    @property
    def venue_label(self) -> str:
        return "emnlp2024"

    @property
    def column_layout(self) -> str:
        return "two_column"

    @property
    def bibstyle_name(self) -> str:
        return "acl_natbib"

    @property
    def max_pages(self) -> int:
        return 8
