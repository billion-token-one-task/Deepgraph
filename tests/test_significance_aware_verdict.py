"""'Refuted' is a scientific claim and carries the same burden as 'supported'.

The runner set scientific_negative_result from the SIGN of the difference
alone -- `candidate <= baseline` -- and three separate places turned that flag
into the verdict "refuted", with no reference to the p-value the same run had
just computed.

Two of the eight audited ladders were overstated as a result:

    run 164: 0.67 -> 0.64  (delta -0.03)  p = 0.506  recorded "refuted"
    run 180: 0.79 -> 0.73  (delta -0.06)  p = 0.071  recorded "refuted"

The system caught this itself, which is the part worth keeping: the
cross-vendor evaluator refused to concur on run 189 (0.685 -> 0.63, p = 0.220)
and said so plainly -- "a non-significant result cannot be classified as
refuted... the measurement is inconclusive". That dissent blocked the ladder,
which is exactly what an independent evaluator is for.

Failing to show an improvement is not the same as showing harm.
"""

import unittest

from meta_harness.evidence_audit import significance_aware_verdict


def _final(base, cand, p, **extra):
    payload = {
        "metric_name": "numeric_accuracy",
        "metric_direction": "higher",
        "baseline_metric_value": base,
        "metric_value": cand,
        "scientific_negative_result": cand <= base,
        "statistical_tests": {"paired_permutation_p": p},
    }
    payload.update(extra)
    return payload


class SignificanceAwareVerdictTests(unittest.TestCase):
    def test_run_164_is_inconclusive_not_refuted(self):
        self.assertEqual(
            significance_aware_verdict(_final(0.67, 0.64, 0.5064935064935064)),
            "inconclusive",
        )

    def test_run_180_is_inconclusive_not_refuted(self):
        self.assertEqual(
            significance_aware_verdict(_final(0.79, 0.73, 0.07092907092907093)),
            "inconclusive",
        )

    def test_run_189_matches_the_evaluators_dissent(self):
        self.assertEqual(
            significance_aware_verdict(_final(0.685, 0.63, 0.21978021978021978)),
            "inconclusive",
        )

    def test_a_significant_harm_is_still_refuted(self):
        # the five ladders that were right stay right
        for base, cand, p in (
            (0.79, 0.67, 0.001998001998001998),
            (0.79, 0.66, 0.002997002997002997),
            (0.67, 0.48, 0.000999000999000999),
            (0.79, 0.545, 0.000999000999000999),
            (0.685, 0.595, 0.012987012987012988),
        ):
            self.assertEqual(
                significance_aware_verdict(_final(base, cand, p)), "refuted"
            )

    def test_a_significant_improvement_is_supported(self):
        self.assertEqual(
            significance_aware_verdict(_final(0.60, 0.72, 0.004)), "supported"
        )

    def test_a_missing_p_value_can_never_reach_a_verdict(self):
        payload = _final(0.60, 0.40, None)
        payload["statistical_tests"] = {}
        self.assertEqual(significance_aware_verdict(payload), "inconclusive")

    def test_an_explicit_recorded_verdict_is_authoritative(self):
        payload = _final(0.60, 0.72, 0.004, hypothesis_verdict="inconclusive")
        self.assertEqual(significance_aware_verdict(payload), "inconclusive")

    def test_the_runner_now_writes_the_verdict_itself(self):
        import inspect

        from meta_harness.runners import generic_transformers

        source = inspect.getsource(generic_transformers)
        self.assertIn('"hypothesis_verdict": hypothesis_verdict', source)
        self.assertIn("_significant = _p is not None and float(_p) < 0.05", source)

    def test_the_worker_reads_it_rather_than_re_deriving(self):
        import inspect

        from orchestrator import colab_worker

        source = inspect.getsource(colab_worker)
        self.assertIn('payload.get("hypothesis_verdict")', source)


if __name__ == "__main__":
    unittest.main()
