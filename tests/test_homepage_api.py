"""The public front page reads one endpoint, and it publishes only findings.

Two failures are guarded here. The first is the quiet one: the page shows a
number, the number is real, and it came from the wrong field. `deep_insights`,
`research_openings` and `contradictions` were all in the 240-270 range at the
time this page was designed, so mislabelling one as another produced a page
that looked correct and was wrong. Each assertion below pins a rendered value
to the exact stats field it claims to be.

The second is disclosure. The endpoints behind the old front page carried
scheduler tuning, grant bookkeeping, controller counters and the titles of
papers being read at that moment. None of that is a research finding, and all
of it was reachable from the open internet.
"""
import json
import unittest
from pathlib import Path
from unittest import mock

from web import app as web_app


STATS = {
    "papers_total": 24648,
    "papers_processed": 7028,
    "deep_insights_total": 269,
    "experiments_completed": 74,
    "experiment_runs_total": 264,
    "scientific_decisions_total": 30,
    "decisions_supported": 1,
    "decisions_refuted": 15,
    "decisions_inconclusive": 14,
    "graph_entities_total": 245731,
    "graph_relations_total": 711900,
    "contradictions_total": 238,
    "estimated_fields": ["graph_entities_total", "graph_relations_total"],
    # Present in /api/stats and deliberately absent from the page.
    "papers_error": 951,
    "papers_pending": 16668,
    "adjudication_candidates": 7,
}

EMPTY_MAP = {"center": None, "domains": [], "other": None, "domains_total": 0}


def _reset_cache():
    web_app._homepage_cache["payload"] = None
    web_app._homepage_cache["stamp"] = 0.0


class HomepagePayloadTest(unittest.TestCase):
    def setUp(self):
        _reset_cache()
        self.addCleanup(_reset_cache)
        self.client = web_app.app.test_client()

    def _payload(self, **overrides):
        patches = {
            "_homepage_map": EMPTY_MAP,
            "_homepage_latest_conclusion": None,
            "_homepage_runtime": {"state": "idle", "reason": "awaiting_grant"},
        }
        patches.update(overrides)
        with (
            mock.patch.object(web_app._stats_cache, "get", return_value=dict(STATS)),
            mock.patch.object(web_app, "_homepage_map", return_value=patches["_homepage_map"]),
            mock.patch.object(web_app, "_homepage_latest_conclusion",
                              return_value=patches["_homepage_latest_conclusion"]),
            mock.patch.object(web_app, "_homepage_runtime", return_value=patches["_homepage_runtime"]),
        ):
            response = self.client.get("/api/homepage")
        self.assertEqual(response.status_code, 200)
        return response.get_json()

    def test_each_chain_step_reads_the_field_it_claims_to(self):
        chain = self._payload()["chain"]
        # Read -> raise -> run -> conclude, each pinned to one stats field.
        self.assertEqual(chain["papers_total"], STATS["papers_total"])
        self.assertEqual(chain["papers_processed"], STATS["papers_processed"])
        self.assertEqual(chain["topics_total"], STATS["deep_insights_total"])
        self.assertEqual(chain["experiments_completed"], STATS["experiments_completed"])
        self.assertEqual(chain["conclusions_total"], STATS["scientific_decisions_total"])
        self.assertEqual(chain["conclusions_supported"], STATS["decisions_supported"])
        self.assertEqual(chain["conclusions_refuted"], STATS["decisions_refuted"])
        self.assertEqual(chain["conclusions_inconclusive"], STATS["decisions_inconclusive"])

    def test_topics_is_not_the_contradiction_or_run_counter(self):
        # These sat within thirty of each other; a swap is invisible on screen.
        chain = self._payload()["chain"]
        self.assertNotEqual(chain["topics_total"], STATS["contradictions_total"])
        self.assertNotEqual(chain["experiments_completed"], STATS["experiment_runs_total"])

    def test_completed_and_attempted_runs_are_reported_separately(self):
        payload = self._payload()
        self.assertEqual(payload["chain"]["experiments_completed"], 74)
        self.assertEqual(payload["counts"]["experiment_runs_total"], 264)

    def test_counts_block_matches_stats(self):
        counts = self._payload()["counts"]
        for key in ("papers_total", "graph_entities_total", "graph_relations_total",
                    "contradictions_total", "experiment_runs_total"):
            self.assertEqual(counts[key], STATS[key], key)

    def test_planner_estimates_stay_labelled(self):
        # These are reltuples figures. The page may show them; it may not call
        # them exact.
        counts = self._payload()["counts"]
        self.assertIn("graph_entities_total", counts["estimated"])
        self.assertIn("graph_relations_total", counts["estimated"])

    def test_operational_counters_never_reach_the_page(self):
        blob = json.dumps(self._payload(), ensure_ascii=False)
        for leaked in ("papers_error", "papers_pending", "adjudication_candidates",
                       "active_grants", "controller", "max_active", "halt_reason",
                       "stale_work_items", "interval_seconds"):
            self.assertNotIn(leaked, blob, leaked)

    def test_runtime_is_two_public_fields_only(self):
        runtime = self._payload()["runtime"]
        self.assertEqual(set(runtime), {"state", "reason"})

    def test_a_second_read_is_served_from_cache(self):
        with (
            mock.patch.object(web_app._stats_cache, "get", return_value=dict(STATS)),
            mock.patch.object(web_app, "_homepage_map", return_value=EMPTY_MAP) as mapper,
            mock.patch.object(web_app, "_homepage_latest_conclusion", return_value=None),
            mock.patch.object(web_app, "_homepage_runtime",
                              return_value={"state": "idle", "reason": "awaiting_grant"}),
        ):
            first = self.client.get("/api/homepage").get_json()
            second = self.client.get("/api/homepage").get_json()
        self.assertEqual(first, second)
        self.assertEqual(mapper.call_count, 1)


class RuntimeVocabularyTest(unittest.TestCase):
    def test_internal_states_collapse_to_a_public_pair(self):
        cases = {
            "running": ("running", "running"),
            "queued": ("running", "queued"),
            "authorized_idle": ("idle", "authorized_idle"),
            "idle_no_authorized_work": ("idle", "awaiting_grant"),
            "halted": ("attention", "halted"),
            "error": ("attention", "attention"),
        }
        for internal, (state, reason) in cases.items():
            with self.subTest(internal=internal):
                with mock.patch.object(web_app, "_research_runtime_snapshot",
                                       return_value={"state": internal}):
                    self.assertEqual(web_app._homepage_runtime(), {"state": state, "reason": reason})

    def test_an_idle_system_always_says_why(self):
        # "Idle" with no reason was the complaint that started this: a reader
        # cannot tell a system waiting for budget from a system that crashed.
        for internal in ("authorized_idle", "idle_no_authorized_work"):
            with mock.patch.object(web_app, "_research_runtime_snapshot",
                                   return_value={"state": internal}):
                runtime = web_app._homepage_runtime()
            self.assertEqual(runtime["state"], "idle")
            self.assertNotIn(runtime["reason"], ("", None, "idle"))

    def test_an_unreadable_runtime_is_unknown_rather_than_idle(self):
        with mock.patch.object(web_app, "_research_runtime_snapshot", side_effect=RuntimeError("db")):
            self.assertEqual(web_app._homepage_runtime(), {"state": "unknown", "reason": "unavailable"})


class TextCleaningTest(unittest.TestCase):
    def test_batch_tags_and_candidate_numbers_leave_the_title(self):
        raw = ("[m3-object-counting-literature-v3-20260820] C01 algorithmic_filter_sum_3shot "
               "for object counting [remove-double-wrap-eos-v1-20260821]")
        self.assertEqual(
            web_app._clean_insight_title(raw),
            "algorithmic_filter_sum_3shot for object counting",
        )

    def test_a_title_that_is_only_tags_becomes_nothing_rather_than_blank_text(self):
        self.assertIsNone(web_app._clean_insight_title("[batch-a] [batch-b]"))
        self.assertIsNone(web_app._clean_insight_title(""))
        self.assertIsNone(web_app._clean_insight_title(None))

    def test_the_hypothesis_drops_its_internal_batch_preamble(self):
        raw = ("Within m3-object-counting-literature-v3-20260820, test whether "
               "zero_shot_cot_reason_first improves counting.")
        self.assertEqual(
            web_app._clean_problem_statement(raw),
            "Test whether zero_shot_cot_reason_first improves counting.",
        )

    def test_dataset_label_keeps_the_corpus_not_the_slice(self):
        self.assertEqual(
            web_app._homepage_dataset_label("tasksource/bigbench:210c1567:test[200:400]"),
            "tasksource/bigbench",
        )
        self.assertIsNone(web_app._homepage_dataset_label(""))


class ConclusionHeadlineTest(unittest.TestCase):
    """The headline is a sentence; the identifier is what you fall back to."""

    def test_the_written_one_line_wins(self):
        row = {
            "proposed_method": json.dumps({
                "name": "algorithmic_filter_sum_3shot",
                "one_line": "Externalize category filtering, quantity normalization and summation.",
            }),
            "evidence_summary": "something else",
            "insight_title": "[batch-20260820] C01 algorithmic_filter_sum_3shot for object counting",
        }
        self.assertEqual(
            web_app._conclusion_headline(row),
            "Externalize category filtering, quantity normalization and summation.",
        )

    def test_a_dict_column_is_read_without_json_decoding(self):
        row = {"proposed_method": {"one_line": "Rephrase, then filter and sum."}}
        self.assertEqual(web_app._conclusion_headline(row), "Rephrase, then filter and sum.")

    def test_the_summary_covers_the_records_without_a_one_line(self):
        row = {
            "proposed_method": json.dumps({"name": "x"}),
            "evidence_summary": "Balanced demonstrations teach category and quantity mapping.",
            "insight_title": "[batch] C02 x for y",
        }
        self.assertEqual(
            web_app._conclusion_headline(row),
            "Balanced demonstrations teach category and quantity mapping.",
        )

    def test_a_malformed_method_column_does_not_lose_the_headline(self):
        row = {"proposed_method": "{not json", "evidence_summary": "A readable summary."}
        self.assertEqual(web_app._conclusion_headline(row), "A readable summary.")

    def test_the_identifier_is_the_last_resort_and_still_gets_cleaned(self):
        row = {
            "proposed_method": None,
            "evidence_summary": "",
            "insight_title": "[m3-batch-20260820] C01 algorithmic_filter_sum_3shot for object counting",
        }
        self.assertEqual(
            web_app._conclusion_headline(row),
            "algorithmic_filter_sum_3shot for object counting",
        )

    def test_nothing_readable_anywhere_yields_nothing_rather_than_a_tag(self):
        self.assertIsNone(web_app._conclusion_headline(
            {"proposed_method": None, "evidence_summary": None, "insight_title": "[batch-only]"}))


class MapTest(unittest.TestCase):
    ROWS = [
        {"id": "ml.dl", "name": "Deep Learning", "paper_count": 5186, "gap_count": 3},
        {"id": "ml.theory", "name": "ML Theory", "paper_count": 1891, "gap_count": 3},
        {"id": "ml.tiny", "name": "Tiny", "paper_count": 4, "gap_count": 0},
        {"id": "ml.test", "name": "Test Node", "paper_count": 0, "gap_count": 0},
    ]

    def _map(self, visible=2):
        with (
            mock.patch.object(web_app.db, "fetchall", return_value=[dict(r) for r in self.ROWS]),
            mock.patch.object(web_app.tax, "get_node_summary", return_value=None),
            mock.patch.object(web_app.tax, "get_node", return_value={"id": "ml", "name": "Machine Learning"}),
            mock.patch.object(web_app, "_HOMEPAGE_MAP_VISIBLE", visible),
        ):
            return web_app._homepage_map()

    def test_directions_with_no_papers_are_not_directions(self):
        # `ml.test` is schema noise. Excluding it by paper count rather than by
        # name means the next stray node needs no code change.
        labels = [d["label"] for d in self._map(visible=10)["domains"]]
        self.assertNotIn("Test Node", labels)
        self.assertEqual(self._map(visible=10)["domains_total"], 3)

    def test_directions_are_ordered_by_paper_count(self):
        papers = [d["papers"] for d in self._map(visible=10)["domains"]]
        self.assertEqual(papers, sorted(papers, reverse=True))

    def test_the_tail_is_summarised_rather_than_dropped(self):
        result = self._map(visible=2)
        self.assertEqual(len(result["domains"]), 2)
        self.assertEqual(result["other"], {"count": 1, "papers": 4})

    def test_no_tail_means_no_summary_node(self):
        self.assertIsNone(self._map(visible=10)["other"])

    def test_the_node_summary_can_raise_open_questions_above_the_matrix_count(self):
        with (
            mock.patch.object(web_app.db, "fetchall", return_value=[dict(self.ROWS[0])]),
            mock.patch.object(web_app.tax, "get_node_summary",
                              return_value={"current_gaps": [1, 2, 3, 4, 5]}),
            mock.patch.object(web_app.tax, "get_node", return_value=None),
        ):
            result = web_app._homepage_map()
        self.assertEqual(result["domains"][0]["open_questions"], 5)


class MapLayoutFileTest(unittest.TestCase):
    LAYOUT = Path(__file__).resolve().parent.parent / "web" / "static" / "data" / "research-map-layout.json"

    def setUp(self):
        self.layout = json.loads(self.LAYOUT.read_text(encoding="utf-8"))

    def test_there_is_a_seat_for_every_direction_the_endpoint_will_show(self):
        self.assertEqual(len(self.layout["slots"]), web_app._HOMEPAGE_MAP_VISIBLE)
        self.assertIn("other", self.layout)

    def test_every_node_fits_inside_the_viewbox_at_full_radius(self):
        _, _, width, height = (float(v) for v in self.layout["viewBox"].split())
        radius = self.layout["radius"]["max"]
        for point in self.layout["slots"] + [self.layout["other"]]:
            self.assertGreaterEqual(point["x"] - radius, 0)
            self.assertLessEqual(point["x"] + radius, width)
            self.assertGreaterEqual(point["y"] - radius, 0)
            self.assertLessEqual(point["y"] + radius, height)


class DisclosureTest(unittest.TestCase):
    def test_automation_drops_scheduler_tuning_and_operator_notes(self):
        snapshot = {
            "auto_research": {
                "total": 250, "completed": 92, "failed": 93,
                "max_active": 3, "max_parallel_repairs": 1,
                "max_parallel_reviews": 2, "interval_seconds": 300,
            },
            "current_work": {
                "experiment_plans": [
                    {"deep_insight_id": 121, "stage": "awaiting_portfolio_decision",
                     "status": "queued", "last_error": None,
                     "last_note": "agenda_selection:3432", "title": "T"},
                ],
                "papers": [{"id": "1", "status": "processing", "stage_last_error": "boom"}],
            },
        }
        public = web_app._public_automation(snapshot)
        blob = json.dumps(public)
        for leaked in ("max_active", "max_parallel_repairs", "max_parallel_reviews",
                       "interval_seconds", "last_error", "last_note",
                       "stage_last_error", "agenda_selection"):
            self.assertNotIn(leaked, blob, leaked)
        # Progress itself is still published.
        self.assertEqual(public["auto_research"]["completed"], 92)
        self.assertEqual(public["current_work"]["experiment_plans"][0]["stage"],
                         "awaiting_portfolio_decision")

    def test_agent_office_publishes_names_not_import_paths(self):
        client = web_app.app.test_client()
        response = client.get("/api/agent_office")
        if response.status_code != 200:
            self.skipTest("agent office snapshot unavailable in this environment")
        for department in response.get_json().get("departments", []):
            for agent in department.get("sub_agents", []):
                self.assertNotIn("path", agent)
                self.assertTrue(agent.get("name"))


if __name__ == "__main__":
    unittest.main()
