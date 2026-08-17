"""A result that measured nothing must not be filed as a scientific negative.

Run 153, 2026-08-16, cost 15114 tokens and reported exact_match 0.0 against 0.0.
The pipeline recorded verdict=refuted. The forensic read of its artifacts showed
the hypothesis was never tested:

  * dataset openai/gsm8k, model Qwen2.5-0.5B-Instruct, metric exact_match
  * the target was GSM8K's full worked solution, including the '####' marker
    that holds the real answer; nothing extracted it
  * every one of the 24 predictions was a chain-of-thought cut off mid-sentence
    under the 64-token default -- 0 of 24 even contained the target string

exact_match could not have been anything but zero, for any model. A statistic on
that comparison is valid and useless: a permutation test on 0.0 against 0.0
returns p=1.0 and the run reads as "no effect" when the truth is "no
measurement". Both belong on the repair path, not in the evidence ladder.
"""

import unittest

from meta_harness.runner_contract import (
    RunnerContractError,
    degenerate_measurement,
    validate_final_results,
)


def _payload(baseline, candidate, metric="exact_match", **overrides):
    payload = {
        "task_protocol": "generative_qa",
        "dataset_id": "openai/gsm8k",
        "dataset_revision": "a" * 40,
        "model_id": "Qwen/Qwen2.5-0.5B-Instruct",
        "model_revision": "b" * 40,
        "seeds": [0, 1, 2],
        "num_examples": 4,
        "baseline_method": "unmodified_input_baseline",
        "candidate_method": "non_gaussian_spectral_step_decomposition",
        "metric_name": metric,
        "metric_direction": "higher",
        "label_fallback_used": False,
        "statistical_tests": {"paired_permutation_p": 1.0},
        "per_method": {
            "unmodified_input_baseline": {metric: baseline},
            "non_gaussian_spectral_step_decomposition": {metric: candidate},
        },
        "artifacts": {
            name: {"path": f"{name}.json"}
            for name in (
                "final_results", "raw_predictions", "environment_manifest",
                "dataset_manifest", "model_manifest",
            )
        },
        "artifact_hashes": {"raw_predictions": "c" * 64},
        "gpu_environment": {"available": False},
    }
    payload.update(overrides)
    return payload


class DegenerateMeasurementTests(unittest.TestCase):
    def test_the_run_153_shape_is_refused(self):
        with self.assertRaises(RunnerContractError) as caught:
            validate_final_results(_payload(0.0, 0.0))
        self.assertEqual(caught.exception.reason_code, "metric_degenerate")

    def test_the_refusal_says_what_was_wrong(self):
        reason = degenerate_measurement(_payload(0.0, 0.0))
        self.assertIn("exact_match", reason)
        self.assertIn("nothing was measured", reason)

    def test_a_shared_ceiling_is_the_same_failure(self):
        self.assertTrue(degenerate_measurement(_payload(1.0, 1.0)))
        self.assertTrue(degenerate_measurement(_payload(100.0, 100.0, metric="accuracy_pct")))

    def test_a_real_null_result_is_not_degenerate(self):
        """Equal but non-floor values are a finding, not a broken instrument."""
        self.assertEqual(degenerate_measurement(_payload(0.42, 0.42)), "")
        validate_final_results(_payload(0.42, 0.42))

    def test_a_real_difference_passes(self):
        self.assertEqual(degenerate_measurement(_payload(0.31, 0.47)), "")
        validate_final_results(_payload(0.31, 0.47))

    def test_a_zero_baseline_with_a_real_candidate_still_passes_here(self):
        """decide_evidence blocks baseline_zero; this guard is about both arms."""
        self.assertEqual(degenerate_measurement(_payload(0.0, 0.35)), "")


class FailureRoutingTests(unittest.TestCase):
    def _context(self, reason_code):
        from meta_harness.failure_policy import FailureContext

        return FailureContext(
            reason_code=reason_code, detail="", code_hash="d" * 64,
            environment_hash="e" * 64, remaining_gpu_seconds=0.0,
        )

    def test_both_codes_are_registered(self):
        from meta_harness.failure_policy import REASON_CODES

        for code in ("metric_degenerate", "generation_truncated"):
            with self.subTest(code=code):
                self.assertIn(code, REASON_CODES)

    def test_both_route_to_repair_not_a_bare_defer(self):
        from meta_harness.failure_policy import decide_recovery

        for code in ("metric_degenerate", "generation_truncated"):
            with self.subTest(code=code):
                decision = decide_recovery(self._context(code), fingerprint_seen=False)
                self.assertEqual(decision.action, "repair_code")


if __name__ == "__main__":
    unittest.main()
