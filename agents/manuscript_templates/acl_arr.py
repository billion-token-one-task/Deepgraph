"""ACL Rolling Review (ACL ARR) template adapter (D2 #13).

Two-column body, 8-page short / 9-page long main limit + unlimited
references, ``acl_natbib`` bibstyle. ACL papers cover the NLP venues
NAACL / ACL / EMNLP / EACL via the shared rolling-review submission.
"""

from __future__ import annotations

from config import ACL_ARR_TEMPLATE_DIR, ACL_ARR_TEMPLATE_FILES
from agents.manuscript_templates import register
from agents.manuscript_templates._stub_adapter import _StubVenueAdapter


@register("acl_arr")
class ACLArrAdapter(_StubVenueAdapter):
    _assets_dir = ACL_ARR_TEMPLATE_DIR
    _asset_files = ACL_ARR_TEMPLATE_FILES
    _sty_basename = "acl"

    @property
    def venue_label(self) -> str:
        return "acl_arr"

    @property
    def column_layout(self) -> str:
        # ACL papers are two-column; FormatLinter (D3) will expect
        # \columnwidth for in-column figures and figure* for full spans.
        return "two_column"

    @property
    def bibstyle_name(self) -> str:
        return "acl_natbib"

    @property
    def max_pages(self) -> int:
        # ACL ARR long paper budget; short papers cap at 4.
        return 9
