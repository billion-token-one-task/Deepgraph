import unittest

from db.opportunity_engine import score_coverage_imbalance, score_metric_diversity


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


if __name__ == "__main__":
    unittest.main()
