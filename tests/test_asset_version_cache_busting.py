"""The static bundle's cache-busting token must track the deployed release.

Assets are served `Cache-Control: immutable, max-age=604800`, so the query
string is the only thing that makes a new bundle a new URL. A literal token in
the template goes stale without anyone noticing: it was last bumped in d953811
and six frontend commits shipped behind it, which served returning visitors new
HTML with a week-old script and stylesheet.
"""

import re
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
TEMPLATE = REPO / "web" / "templates" / "index.html"


class AssetVersionTemplateTest(unittest.TestCase):
    def test_no_hardcoded_version_token(self):
        markup = TEMPLATE.read_text(encoding="utf-8")
        literals = re.findall(r"\?v=(?!\{\{)([^\"'\s>]+)", markup)
        self.assertEqual(
            literals, [],
            "hardcoded cache-busting tokens go stale silently: %r" % (literals,))

    def test_every_bundled_asset_is_versioned(self):
        markup = TEMPLATE.read_text(encoding="utf-8")
        for asset in ("js/app.js", "js/i18n.js", "css/style.css"):
            for match in re.finditer(re.escape(asset) + r"([\"'?][^\"']*)", markup):
                self.assertIn(
                    "?v={{ asset_version }}", asset + match.group(1),
                    "%s is served immutable without a release-derived version" % asset)


class AssetVersionValueTest(unittest.TestCase):
    def test_release_marker_drives_the_version(self):
        from web.app import _asset_version

        marker = REPO / ".release-commit"
        if marker.exists():
            commit = marker.read_text(encoding="utf-8").strip()
            self.assertTrue(_asset_version() and commit.startswith(_asset_version()))
        else:
            # A development tree falls back to bundle mtime, never to a constant.
            self.assertTrue(_asset_version().startswith("dev-"))

    def test_version_is_url_safe(self):
        from web.app import _asset_version

        self.assertRegex(_asset_version(), r"^[A-Za-z0-9._-]+$")


if __name__ == "__main__":
    unittest.main()
