"""A run without a significance test must fail at the runner, not at the gate.

contracts.scientific_evidence.decide_evidence marks a run ``not_significant``
when no p-value is present, so ``confirmation_allowed`` is False and a
``supported`` verdict is unreachable however good the metric is. The runner
contract used to demand only a metric, so a run could pass every check it was
given, spend its whole budget, and discover at the final gate that the verdict
it was working toward had never been available.

That gap was measured on 2026-08-17: 34 recorded decisions, all inconclusive,
and no run had ever reached ``manuscript_allowed``.
"""

import json
import math
import unittest
from pathlib import Path

from contracts.scientific_evidence import EvidenceDecisionInput, decide_evidence
from meta_harness.runner_contract import (
    P_VALUE_CONTAINERS,
    P_VALUE_KEYS,
    RunnerContractError,
    extract_p_value,
    paired_permutation_test,
    validate_final_results,
)


def _payload(**overrides):
    payload = {
        "task_protocol": "sequence_classification",
        "dataset_id": "opaque/data",
        "dataset_revision": "a" * 40,
        "model_id": "opaque/model",
        "model_revision": "b" * 40,
        "seeds": [0],
        "num_examples": 2,
        "baseline_method": "baseline",
        "candidate_method": "candidate",
        "metric_name": "accuracy",
        "metric_direction": "higher",
        "label_fallback_used": False,
        "per_method": {
            "baseline": {"accuracy": 0.5},
            "candidate": {"accuracy": 1.0},
        },
        "statistical_tests": {"paired_permutation_p": 0.02},
        "artifacts": {
            name: {"path": f"{name}.json"}
            for name in (
                "final_results",
                "raw_predictions",
                "environment_manifest",
                "dataset_manifest",
                "model_manifest",
            )
        },
        "artifact_hashes": {"raw_predictions": "c" * 64},
        "gpu_environment": {"available": False},
    }
    payload.update(overrides)
    return payload


class PValueContractTests(unittest.TestCase):
    def test_a_payload_without_a_p_value_is_refused(self):
        payload = _payload()
        payload.pop("statistical_tests")
        with self.assertRaises(RunnerContractError) as caught:
            validate_final_results(payload)
        self.assertEqual(caught.exception.reason_code, "p_value_missing")

    def test_the_refusal_names_where_a_p_value_may_live(self):
        payload = _payload()
        payload.pop("statistical_tests")
        with self.assertRaises(RunnerContractError) as caught:
            validate_final_results(payload)
        message = str(caught.exception)
        self.assertIn("paired_permutation_p", message)
        self.assertIn("statistical_tests", message)

    def test_every_documented_container_is_accepted(self):
        for container in P_VALUE_CONTAINERS:
            with self.subTest(container=container):
                payload = _payload()
                payload.pop("statistical_tests")
                payload[container] = {"p_value": 0.01}
                self.assertEqual(validate_final_results(payload)[container]["p_value"], 0.01)

    def test_every_documented_key_is_accepted_at_top_level(self):
        for key in P_VALUE_KEYS:
            with self.subTest(key=key):
                payload = _payload()
                payload.pop("statistical_tests")
                payload[key] = 0.01
                validate_final_results(payload)

    def test_an_out_of_range_p_value_is_refused(self):
        for bad in (-0.1, 1.5):
            with self.subTest(p=bad):
                with self.assertRaises(RunnerContractError) as caught:
                    validate_final_results(_payload(statistical_tests={"p_value": bad}))
                self.assertEqual(caught.exception.reason_code, "p_value_invalid")

    def test_the_requirement_can_be_waived_explicitly(self):
        payload = _payload()
        payload.pop("statistical_tests")
        validate_final_results(payload, require_p_value=False)

    def test_the_contract_and_the_gate_agree_on_what_is_missing(self):
        """What the runner now refuses is exactly what the gate would block on."""
        decision = decide_evidence(
            EvidenceDecisionInput(
                verdict="supported",
                p_value=None,
                metric_value=1.0,
                baseline_value=0.5,
                full_benchmark_complete=True,
                raw_artifacts_complete=True,
                claim_ledger_complete=True,
                evaluator_id="independent",
            )
        )
        self.assertIn("p_value_missing", decision.blockers)
        self.assertFalse(decision.confirmation_allowed)
        self.assertEqual(decision.max_claim_strength, "descriptive")


class PairedPermutationTests(unittest.TestCase):
    def _rows(self, method, outcomes):
        return [
            {
                "method": method,
                "seed": 0,
                "sample_index": index,
                "prediction": "yes" if hit else "no",
                "target": "yes",
            }
            for index, hit in enumerate(outcomes)
        ]

    def test_identical_arms_are_not_significant(self):
        outcomes = [True, False] * 20
        result = paired_permutation_test(
            self._rows("baseline", outcomes),
            self._rows("candidate", outcomes),
            "accuracy",
            permutations=200,
        )
        self.assertEqual(result["observed_difference"], 0.0)
        self.assertEqual(result["paired_permutation_p"], 1.0)

    def test_a_clean_separation_is_significant(self):
        n = 40
        result = paired_permutation_test(
            self._rows("baseline", [False] * n),
            self._rows("candidate", [True] * n),
            "accuracy",
            permutations=500,
        )
        self.assertAlmostEqual(result["observed_difference"], 1.0)
        self.assertLess(result["paired_permutation_p"], 0.05)

    def test_p_is_never_zero_and_never_above_one(self):
        n = 30
        result = paired_permutation_test(
            self._rows("baseline", [False] * n),
            self._rows("candidate", [True] * n),
            "accuracy",
            permutations=100,
        )
        self.assertGreater(result["paired_permutation_p"], 0.0)
        self.assertLessEqual(result["paired_permutation_p"], 1.0)
        # add-one correction, so the floor is 1/(permutations+1)
        self.assertAlmostEqual(result["paired_permutation_p"], 1 / 101)

    def test_the_test_is_deterministic_for_a_seed(self):
        args = (self._rows("baseline", [True, False] * 10), self._rows("candidate", [True, True] * 10), "accuracy")
        first = paired_permutation_test(*args, permutations=200, seed=7)
        second = paired_permutation_test(*args, permutations=200, seed=7)
        self.assertEqual(first, second)

    def test_pairing_uses_seed_and_sample_index(self):
        baseline = self._rows("baseline", [True, False])
        candidate = self._rows("candidate", [True, False])
        candidate[0]["sample_index"] = 99  # only one pair now lines up
        result = paired_permutation_test(baseline, candidate, "accuracy", permutations=50)
        self.assertEqual(result["n_pairs"], 1)

    def test_unpairable_arms_are_refused(self):
        with self.assertRaises(RunnerContractError) as caught:
            paired_permutation_test(
                self._rows("baseline", [True]),
                [],
                "accuracy",
                permutations=10,
            )
        self.assertEqual(caught.exception.reason_code, "permutation_contract_violation")

    def test_the_result_is_readable_by_the_shared_extractor(self):
        result = paired_permutation_test(
            self._rows("baseline", [False] * 10),
            self._rows("candidate", [True] * 10),
            "accuracy",
            permutations=50,
        )
        self.assertEqual(
            extract_p_value({"statistical_tests": result}),
            result["paired_permutation_p"],
        )

    def test_a_corpus_level_metric_is_supported(self):
        """macro-F1 does not decompose per example, so the test recomputes it."""
        result = paired_permutation_test(
            self._rows("baseline", [False] * 12),
            self._rows("candidate", [True] * 12),
            "macro_f1",
            permutations=100,
        )
        self.assertTrue(math.isfinite(result["observed_difference"]))
        self.assertGreater(result["n_pairs"], 0)


if __name__ == "__main__":
    unittest.main()


def _context(reason_code):
    from meta_harness.failure_policy import FailureContext

    return FailureContext(
        reason_code=reason_code,
        detail="",
        code_hash="d" * 64,
        environment_hash="e" * 64,
        remaining_gpu_seconds=0.0,
    )


class FailureRoutingTests(unittest.TestCase):
    """A missing statistic is a repairable code defect, not an unknown crash."""

    def test_the_new_codes_are_in_the_stable_vocabulary(self):
        from meta_harness.failure_policy import REASON_CODES

        for code in ("p_value_missing", "p_value_invalid", "permutation_contract_violation"):
            with self.subTest(code=code):
                self.assertIn(code, REASON_CODES)

    def test_they_route_to_repair_rather_than_a_bare_defer(self):
        from meta_harness.failure_policy import decide_recovery

        for code in ("p_value_missing", "p_value_invalid", "permutation_contract_violation"):
            with self.subTest(code=code):
                decision = decide_recovery(
                    _context(code), fingerprint_seen=False
                )
                self.assertEqual(decision.action, "repair_code")

    def test_an_unregistered_code_would_have_been_deferred_silently(self):
        """Why registration matters: the fallback loses the diagnosis."""
        from meta_harness.failure_policy import decide_recovery

        decision = decide_recovery(
            _context("not_a_real_code"), fingerprint_seen=False
        )
        self.assertEqual(decision.action, "defer")
        self.assertEqual(decision.reason_code, "unknown_execution_failure")
