import unittest
from unittest.mock import patch

from db import opportunity_engine as engine
from db.opportunity_engine import (
    rebuild_opportunity_triage,
    score_coverage_imbalance,
    score_metric_diversity,
    triage_opportunity,
)


class OpportunityScoringTests(unittest.TestCase):
    def test_coverage_imbalance_grows_with_method_dataset_ratio(self):
        low = score_coverage_imbalance(method_count=4, dataset_count=3)
        high = score_coverage_imbalance(method_count=12, dataset_count=2)
        self.assertGreater(high, low)
        self.assertLessEqual(high, 5.0)

    def test_metric_diversity_penalizes_narrow_measurement(self):
        narrow = score_metric_diversity(method_count=8, metric_count=1)
        broad = score_metric_diversity(method_count=8, metric_count=5)
        self.assertGreater(narrow, broad)
        self.assertGreaterEqual(narrow, 1.0)

    def test_triage_opportunity_scores_priority_and_band(self):
        opportunity = {
            "id": 7,
            "node_id": "ml.cv",
            "opportunity_type": "evaluation_gap",
            "title": "Test transformer on new dataset",
            "description": "A reproducible evaluation gap",
            "why_now": "Fresh benchmark gap with supporting papers",
            "value_score": 4.5,
            "confidence": 0.9,
            "signal_counts": {"gap": 2},
            "evidence_paper_ids": ["p1", "p2"],
        }

        triaged = triage_opportunity(opportunity)

        self.assertEqual(triaged["opportunity_id"], 7)
        self.assertGreaterEqual(triaged["priority_score"], 1.0)
        self.assertIn(triaged["priority_band"], {"high", "medium", "watchlist"})
        self.assertEqual(triaged["status"], "ready")
        self.assertGreaterEqual(triaged["scientific_value"], 4.0)

    def test_rebuild_opportunity_triage_cleans_stale_rows(self):
        opportunities = [
            {"id": 1, "node_id": "ml.cv", "opportunity_type": "evaluation_gap", "title": "A", "description": "A", "value_score": 4.0, "confidence": 0.8, "signal_counts": {}, "evidence_paper_ids": []},
            {"id": 2, "node_id": "ml.cv", "opportunity_type": "open_question", "title": "B", "description": "B", "value_score": 3.0, "confidence": 0.7, "signal_counts": {}, "evidence_paper_ids": []},
        ]

        with patch.object(engine, "get_node_opportunities", return_value=opportunities), \
             patch.object(engine, "upsert_opportunity_triage", side_effect=lambda triage: {"priority_score": triage["priority_score"], "node_id": triage["node_id"], "opportunity_id": triage["opportunity_id"]}), \
             patch.object(engine.db, "execute") as execute_mock:
            result = rebuild_opportunity_triage("ml.cv")

        self.assertEqual([row["opportunity_id"] for row in result], [1, 2])
        delete_calls = [call for call in execute_mock.call_args_list if "DELETE FROM opportunity_triage" in str(call)]
        self.assertTrue(delete_calls)


if __name__ == "__main__":
    unittest.main()
