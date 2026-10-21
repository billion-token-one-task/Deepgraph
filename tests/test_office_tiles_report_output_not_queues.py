"""A department tile is green only when something finished.

On 2026-08-29 the front page showed the paper department "Extracting" while
the ledger had recorded nothing new for a day. Nothing was lying on purpose:
the tile turned green whenever the department had rows, and rows are exactly
what a jam leaves behind. Five papers had been parked mid-stage since the
previous evening, so the dashboard rendered a stall as work.

That failure mode is worse than a blank panel. Someone watching this page to
decide whether the system is alive gets a confident yes, and the longer the
jam lasts the more convincing the display becomes, because the stuck rows
never go away.

So the rule is now: a tile turns green on measured output inside the last
hour, and a dark tile owes the reader a reason.
"""

from __future__ import annotations

import unittest

from web import app as web_app


class DepartmentStateTests(unittest.TestCase):
    def test_rows_alone_do_not_make_a_department_working(self):
        parked = [{"status": "processing", "title": "a paper stuck since yesterday"}]
        self.assertEqual(
            web_app._office_department_state(parked, service_running=False),
            "idle",
            "queued rows were treated as progress; that is the 2026-08-29 bug",
        )

    def test_measured_output_makes_it_working(self):
        self.assertEqual(
            web_app._office_department_state([], service_running=True), "working")

    def test_a_failing_row_still_reads_blocked(self):
        rows = [{"status": "failed", "title": "x"}]
        self.assertEqual(
            web_app._office_department_state(rows, service_running=True), "blocked")
        self.assertEqual(
            web_app._office_department_state(rows, service_running=False), "blocked")


class OfficePayloadTests(unittest.TestCase):
    """Every tile is computed from a counted window, and says why when dark."""

    def _source(self):
        import inspect

        return inspect.getsource(web_app.api_agent_office)

    def test_every_tile_is_counted_over_the_same_window(self):
        source = self._source()
        self.assertIn("HOUR = 3600", source)
        # Not a service flag: the two that used to read one are named here so
        # a future edit that reintroduces the flag fails.
        self.assertNotIn('bool(paper_worker_status.get("running"))', source)
        for key in ("paper_extraction", "graph_construction", "idea_generation",
                    "experiment_planning", "experiment_execution",
                    "manuscript_generation", "orchestration"):
            self.assertIn(f'"{key}"', source)

    def test_a_dark_tile_carries_a_reason(self):
        source = self._source()
        self.assertIn("idle_reason", source)
        self.assertIn("office.idle.noOutput", source)
        self.assertIn("office.idle.noBacklog", source)


class OfficeRenderingTests(unittest.TestCase):
    """The reason has to reach the screen, not just the payload."""

    def _asset(self):
        from pathlib import Path

        # web/app.py -> web/ -> web/static/js
        web_dir = Path(web_app.__file__).resolve().parent
        return (web_dir / "static" / "js" / "app.js").read_text(encoding="utf-8")

    def test_an_idle_tile_does_not_borrow_a_queued_row_s_verb(self):
        source = self._asset()
        marker = source.index("function officeLeadText")
        body = source[marker:marker + 1400]
        # The idle branch must come before the items branch, or a parked row
        # supplies the caption again.
        self.assertLess(body.index('dep.status === "idle"'), body.index("if (items.length)"))
        self.assertIn("idle_reason", body)

    def test_both_languages_carry_the_reasons(self):
        from pathlib import Path

        web_dir = Path(web_app.__file__).resolve().parent
        i18n = (web_dir / "static" / "js" / "i18n.js").read_text(encoding="utf-8")
        for key in ("office.idle.noOutput", "office.idle.noBacklog", "office.lead.waiting"):
            self.assertEqual(i18n.count(f'"{key}"'), 2, f"{key} is not in both tables")


if __name__ == "__main__":
    unittest.main()
