import unittest

from agents.result_interpreter import _bootstrap_ci, _compute_p_value, _iteration_evidence_label


class ResultInterpreterTests(unittest.TestCase):
    def test_p_value_is_exact_for_small_samples(self):
        # 20 ways to split six values 3/3; only the observed split and its
        # mirror are as extreme as the observed one.
        self.assertEqual(_compute_p_value([3.0, 4.0, 5.0], [0.0, 1.0, 2.0]), 0.1)
        self.assertEqual(_compute_p_value([1.0], [1.0, 1.0]), 1.0)

    def test_sampled_p_value_is_repeatable_and_never_zero(self):
        treatment = [10.0 + i / 100 for i in range(20)]
        control = [i / 100 for i in range(20)]

        p = _compute_p_value(treatment, control)

        self.assertEqual(p, 1 / 5001)
        self.assertEqual(_compute_p_value(treatment, control), p)

    def test_bootstrap_ci_is_repeatable(self):
        values = [0.1, 0.4, 0.35, 0.8, 0.5]
        self.assertEqual(_bootstrap_ci(values), _bootstrap_ci(values))

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
