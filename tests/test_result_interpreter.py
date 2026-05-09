import unittest

from agents.result_interpreter import _iteration_evidence_label


class ResultInterpreterTests(unittest.TestCase):
    def test_iteration_evidence_label_marks_kept_below_baseline_partial(self):
        label = _iteration_evidence_label(
            metric_value=0.392,
            baseline=0.421,
            direction="higher",
            status="keep",
        )

        self.assertEqual(label["evidence_label"], "partial_recovery")
        self.assertFalse(label["beats_baseline"])
        self.assertLess(label["baseline_effect"], 0)

    def test_iteration_evidence_label_marks_positive_effect(self):
        label = _iteration_evidence_label(
            metric_value=0.44,
            baseline=0.421,
            direction="higher",
            status="keep",
        )

        self.assertEqual(label["evidence_label"], "positive_effect")
        self.assertTrue(label["beats_baseline"])
        self.assertGreater(label["baseline_effect"], 0)


if __name__ == "__main__":
    unittest.main()
