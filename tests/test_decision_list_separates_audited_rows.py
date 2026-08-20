"""The decisions LIST must separate audited findings from stamped ones too.

The headline card was corrected on 2026-08-20 to count evidence_state_transitions
instead of scientific_decision_records. That pass missed the endpoint the list
itself renders, so /api/scientific_decisions kept reporting the union:

    API returned      54 decisions    inconclusive 42 / refuted 12
    actually audited  20 decisions    inconclusive  8 / refuted 12

All 34 extra rows are `inconclusive` and none of them ever produced an
evidence_audit_v1 transition. The refuted count matched exactly, which is why
the discrepancy could sit behind a plausible-looking page: the half that was
wrong was the half nobody cross-checked.

Worse than the count, no field on a row let a reader tell the two apart, so a
stamped row and an audited one rendered identically.

The legacy rows are marked rather than dropped. They are real history, and a
number that quietly disappears is how the next person rediscovers the same gap
the hard way.
"""

import inspect
import unittest

from web import app as web_app


class DecisionListTests(unittest.TestCase):
    def _source(self):
        return inspect.getsource(web_app.api_scientific_decisions)

    def test_each_row_reports_whether_it_climbed_the_ladder(self):
        source = self._source()
        self.assertIn("walked_ladder", source)
        # From the transitions table, which is the only record of the climb --
        # not from the status column, which is what was wrong to begin with.
        self.assertIn("evidence_state_transitions", source)
        self.assertIn("actor = 'evidence_audit_v1'", source)
        self.assertIn("to_state = 'scientifically_decided'", source)

    def test_the_headline_counter_excludes_unaudited_rows(self):
        source = self._source()
        self.assertIn("legacy_counts_by_verdict", source)
        self.assertIn("legacy_total", source)
        # The split must happen in SQL, so the counter cannot drift from the
        # per-row flag the list renders.
        self.assertIn("GROUP BY verdict, walked_ladder", source)

    def test_legacy_rows_are_still_returned(self):
        # Marked, not dropped.
        source = self._source()
        self.assertNotIn("WHERE walked_ladder", source)
        self.assertNotIn("AND walked_ladder", source)


class DecisionListRenderingTests(unittest.TestCase):
    """The flag is worth nothing if the page does not draw it."""

    def _asset(self, path):
        from pathlib import Path

        root = Path(inspect.getfile(web_app)).resolve().parent
        return (root / "static" / path).read_text(encoding="utf-8")

    def test_the_list_marks_an_unaudited_row(self):
        js = self._asset("js/app.js")
        self.assertIn("d.walked_ladder", js)
        self.assertIn("decision-unaudited-flag", js)

    def test_the_marker_is_styled(self):
        css = self._asset("css/style.css")
        self.assertIn(".decision-unaudited-flag", css)
        # Uses a real theme token; an undefined variable renders as no colour.
        self.assertIn("var(--text-dim)", css.split(".decision-unaudited-flag")[1][:400])

    def test_the_marker_is_translated_both_ways(self):
        i18n = self._asset("js/i18n.js")
        self.assertEqual(i18n.count('"decisions.unaudited"'), 2)
        self.assertEqual(i18n.count('"decisions.unauditedHint"'), 2)


if __name__ == "__main__":
    unittest.main()
