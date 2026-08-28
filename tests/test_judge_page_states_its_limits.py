"""The exhibit page may not claim more than the ledger says.

/judge exists to be shown to people deciding whether to believe this system,
which makes every overstatement on it expensive in a way a dashboard's is not.
Two failures are specifically guarded here because both already happened:

  * A ledger total was written into the page as a literal. "全库 166 条判决" was
    true for about an hour after it was typed and then became a false number on
    the one page whose entire argument is that its numbers can be checked.
  * A cross-reference pointed at the wrong exhibit, so the row claiming the
    system rejects contaminated holdouts cited the exhibit about statistical
    significance instead.

The third guard has never failed, and is here because it is the one that would
matter most: an operator-frozen candidate must never render without saying so.
Every candidate on this page was transcribed from a paper by a human, and a
reader who misses that is reading a reproduction as a discovery.
"""

from __future__ import annotations

import re
import unittest

from web import judge_demo_routes as judge


def _exhibit_stub(verdict, *, model_version, run_id=1, blockers=()):
    return {
        "run": {"id": run_id, "agenda_id": 14, "baseline_metric_name": "numeric_accuracy",
                "baseline_metric_value": 0.32, "best_metric_value": 0.63,
                "effect_pct": 96.875},
        "idea": {"id": 226},
        "title": "C01 algorithmic_filter_sum_3shot",
        "operator_frozen": model_version == judge.OPERATOR_FROZEN_MODEL_VERSION,
        "model_version": model_version,
        "grants": [
            {"id": 408, "stage": "pilot", "token_cap": 40000, "max_gpu_hours": 2.0,
             "status": "consumed", "gpu_class": "NVIDIA A10G"},
            {"id": 410, "stage": "evidence_audit", "token_cap": 40000,
             "max_gpu_hours": 2.0, "status": "active", "gpu_class": "NVIDIA A10G"},
        ],
        "accounts": [{"stage": "pilot", "account_ref": "colab-pro"}],
        "audit": {},
        "decision": {},
        "outcome": {},
        "verdict": verdict,
        "p_value": 0.000999000999000999,
        "alpha": 0.05,
        "metric_value": 0.63,
        "baseline_value": 0.32,
        "blockers": list(blockers),
        "significant": True,
        "ladder": [
            {"state": s, "label": label, "why": why, "reached": True,
             "actor": "evidence_audit_v1", "at": "2026-08-21 05:29:52"}
            for s, label, why in judge.LADDER
        ],
        "complete": True,
    }


class CapabilityLedgerTests(unittest.TestCase):
    """No ledger figure may be a literal in the capability table."""

    # Counts that come from the database are written as {placeholders}. A bare
    # run of digits in a note is either a hardcoded total (the bug) or a fact
    # about spend that does not change (agenda 10's 363万 token, V1). The
    # allowlist names the second kind explicitly so a new number has to be
    # argued for rather than slipped in.
    ALLOWED_LITERALS = {"363", "50", "0", "10", "1"}

    # Digits glued to letters are part of an identifier, not a count: the first
    # version of this rule read "256" out of "sha256" and demanded a
    # placeholder for it.
    STANDALONE_NUMBER = re.compile(r"(?<![A-Za-z0-9])\d+(?![A-Za-z0-9])")

    def test_notes_use_placeholders_for_anything_the_ledger_counts(self):
        for name, _done, note in judge.CAPABILITY_LEDGER:
            for number in self.STANDALONE_NUMBER.findall(note):
                self.assertIn(
                    number, self.ALLOWED_LITERALS,
                    f"capability row {name!r} hardcodes the ledger figure "
                    f"{number!r}; use a {{placeholder}} filled by "
                    f"_ledger_totals() so the page cannot go stale",
                )

    def test_every_placeholder_is_supplied_by_the_totals_query(self):
        supplied = {
            "counts", "total", "supported", "supported_operator_frozen",
            "llm_total", "llm_supported",
        }
        for name, _done, note in judge.CAPABILITY_LEDGER:
            for field in re.findall(r"\{(\w+)\}", note):
                self.assertIn(
                    field, supplied,
                    f"capability row {name!r} formats {{{field}}}, which "
                    f"_ledger_totals() does not return",
                )

    def test_exhibit_cross_references_name_an_exhibit_that_exists(self):
        # Exhibits are lettered A-D in the template. The contamination row
        # cited C (significance) for its first hour of life.
        letters = set()
        for _name, _done, note in judge.CAPABILITY_LEDGER:
            letters.update(re.findall(r"展品 ([A-Z])", note))
        self.assertTrue(letters, "no capability row cites an exhibit at all")
        self.assertLessEqual(letters, {"A", "B", "C", "D"})

    def test_the_roadmap_half_is_not_quietly_empty(self):
        # The page's credibility rests on the "not yet" column being real.
        not_done = [name for name, done, _ in judge.CAPABILITY_LEDGER if not done]
        self.assertIn("harness 自进化 / RSI", " | ".join(not_done))
        self.assertGreaterEqual(len(not_done), 3)


class ProvenanceRenderingTests(unittest.TestCase):
    """An operator-frozen candidate must say so where the verdict is shown."""

    def setUp(self):
        from web.app import app

        self.app = app
        self._exhibit = judge._exhibit
        self._totals = judge._ledger_totals
        judge._ledger_totals = lambda: {
            "counts": {"supported": 3, "refuted": 61, "inconclusive": 40, "invalid": 63},
            "total": 167, "supported": 3, "supported_operator_frozen": 3,
            "llm_total": 135, "llm_supported": 0,
        }

    def tearDown(self):
        judge._exhibit = self._exhibit
        judge._ledger_totals = self._totals

    def _render(self, exhibits):
        judge._exhibit = lambda run_id: exhibits.get(run_id)
        with self.app.test_client() as client:
            response = client.get("/judge")
        self.assertEqual(response.status_code, 200)
        return response.data.decode("utf-8")

    def test_a_frozen_candidate_is_labelled_wherever_its_verdict_appears(self):
        frozen = _exhibit_stub("supported",
                               model_version=judge.OPERATOR_FROZEN_MODEL_VERSION)
        page = self._render({235: frozen, 240: frozen})
        self.assertIn("operator-frozen", page)
        # And the label sits in the same block as the verdict, not in a footnote
        # further down the page where a reader scanning verdicts will miss it.
        # Anchored on the rendered badge: "v-supported" alone matches the
        # stylesheet rule near the top of the document first.
        block = page[page.index('class="verdict v-supported"'):]
        self.assertLess(block.index("operator-frozen"), block.index("证据阶梯"))

    def test_an_llm_authored_candidate_shows_the_model_that_wrote_it(self):
        authored = _exhibit_stub("supported", model_version="gemini-3.7-flash-high")
        page = self._render({235: authored, 240: authored})
        self.assertIn("gemini-3.7-flash-high", page)

    def test_a_refuted_exhibit_renders_why_it_was_blocked(self):
        refuted = _exhibit_stub("refuted",
                                model_version=judge.OPERATOR_FROZEN_MODEL_VERSION,
                                blockers=("evaluator_refuted",))
        page = self._render({235: refuted, 240: refuted})
        self.assertIn("evaluator_refuted", page)
        self.assertIn("v-refuted", page)

    def test_the_headline_counts_come_from_the_totals_query(self):
        frozen = _exhibit_stub("supported",
                               model_version=judge.OPERATOR_FROZEN_MODEL_VERSION)
        judge._ledger_totals = lambda: {
            "counts": {"supported": 7, "refuted": 11, "inconclusive": 2, "invalid": 1},
            "total": 21, "supported": 7, "supported_operator_frozen": 7,
            "llm_total": 4, "llm_supported": 1,
        }
        page = self._render({235: frozen, 240: frozen})
        self.assertIn("全库 21 条判决里 supported 只有 7 条", page)
        self.assertNotIn("167", page)

    def test_the_authorised_gpu_class_is_not_shown_as_where_it_ran(self):
        """grant.gpu_class is a request, not a receipt.

        The audit grant for run 274 asked for an A10G and the work ran on
        Colab. Rendering the class alone labelled a Colab flight "NVIDIA
        A10G" -- the kind of small false detail that costs a reader's trust in
        every other figure on the page.
        """
        frozen = _exhibit_stub("supported",
                               model_version=judge.OPERATOR_FROZEN_MODEL_VERSION)
        page = self._render({235: frozen, 240: frozen})
        self.assertIn("colab-pro", page)
        self.assertIn("实际执行后端", page)
        self.assertNotIn("NVIDIA A10G", page)

    def test_the_grant_chain_shows_every_stage_that_had_to_wait(self):
        frozen = _exhibit_stub("supported",
                               model_version=judge.OPERATOR_FROZEN_MODEL_VERSION)
        page = self._render({235: frozen, 240: frozen})
        self.assertIn("grant 408", page)
        self.assertIn("grant 410", page)

    def test_a_missing_run_renders_an_absence_rather_than_a_placeholder(self):
        page = self._render({})
        self.assertIn("页面不编造占位数据", page)


class ReadOnlyTests(unittest.TestCase):
    def test_the_blueprint_exposes_no_mutating_route(self):
        from web.app import app

        for rule in app.url_map.iter_rules():
            if rule.endpoint.startswith("judge_demo."):
                self.assertEqual(
                    rule.methods & {"POST", "PUT", "PATCH", "DELETE"}, set(),
                    f"{rule.endpoint} accepts a mutating method",
                )


if __name__ == "__main__":
    unittest.main()
