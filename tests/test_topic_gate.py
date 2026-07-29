import unittest

from agents.topic_gate import (
    allocate_compute,
    binary_entropy,
    next_stage,
    pilot_verdict,
    route_outcome,
    screen_topic,
    surprisal_bits,
)


def _prediction(**overrides):
    base = {
        "predicted_outcome": "truncating the history raises per-step accuracy",
        "confidence": 0.6,
        "action_if_confirmed": "make truncation the harness default",
        "action_if_refuted": "keep full history and chase self-conditioning instead",
        "already_published": "no",
    }
    base.update(overrides)
    return base


class Gate1Tests(unittest.TestCase):
    def test_topic_with_a_real_prediction_passes(self):
        result = screen_topic({"novelty_status": "novel"}, prediction=_prediction())
        self.assertTrue(result["passed"], result["blockers"])
        self.assertGreater(result["expected_bits"], 0.9)

    def test_missing_prediction_blocks_before_any_compute(self):
        result = screen_topic({"novelty_status": "novel"}, prediction=None)
        self.assertFalse(result["passed"])
        self.assertEqual([b["question"] for b in result["blockers"]], ["prediction"])

    def test_textbook_certainty_blocks(self):
        result = screen_topic(
            {"novelty_status": "novel"},
            prediction=_prediction(confidence=0.97),
        )
        self.assertFalse(result["passed"])
        self.assertIn("prediction", [b["question"] for b in result["blockers"]])
        self.assertIn("already known", result["blockers"][0]["reason"])

    def test_same_action_either_way_blocks(self):
        result = screen_topic(
            {"novelty_status": "novel"},
            prediction=_prediction(
                action_if_confirmed="write it up",
                action_if_refuted="Write it up.",
            ),
        )
        self.assertFalse(result["passed"])
        self.assertEqual([b["question"] for b in result["blockers"]], ["decision_relevance"])

    def test_both_outcomes_no_op_blocks(self):
        result = screen_topic(
            {"novelty_status": "novel"},
            prediction=_prediction(action_if_confirmed="none", action_if_refuted="nothing"),
        )
        self.assertFalse(result["passed"])
        self.assertEqual([b["question"] for b in result["blockers"]], ["decision_relevance"])

    def test_prior_work_blocks_from_either_source(self):
        by_novelty = screen_topic({"novelty_status": "exists"}, prediction=_prediction())
        self.assertFalse(by_novelty["passed"])
        self.assertEqual([b["question"] for b in by_novelty["blockers"]], ["prior_work"])

        by_prediction = screen_topic(
            {"novelty_status": "novel"},
            prediction=_prediction(
                already_published="yes",
                already_published_evidence="arXiv:2509.09677",
            ),
        )
        self.assertFalse(by_prediction["passed"])
        self.assertIn("arXiv:2509.09677", by_prediction["blockers"][0]["reason"])

    def test_json_string_prediction_is_accepted(self):
        result = screen_topic(
            {"novelty_status": "novel"},
            prediction='{"predicted_outcome": "x", "confidence": 0.5, '
            '"action_if_confirmed": "ship", "action_if_refuted": "drop"}',
        )
        self.assertTrue(result["passed"], result["blockers"])
        self.assertAlmostEqual(result["expected_bits"], 1.0, places=3)


class Gate2Tests(unittest.TestCase):
    def test_running_is_not_passing(self):
        verdict = pilot_verdict(
            _prediction(confidence=0.6),
            {"ran": True, "outcome": "confirmed", "null_model_control": "passed"},
        )
        self.assertEqual(verdict["verdict"], "stop")
        self.assertLess(verdict["surprise_bits"], 1.0)

    def test_refuted_prediction_escalates(self):
        verdict = pilot_verdict(
            _prediction(confidence=0.6),
            {"ran": True, "outcome": "refuted", "null_model_control": "passed"},
        )
        self.assertEqual(verdict["verdict"], "escalate")
        self.assertGreater(verdict["surprise_bits"], 1.0)

    def test_null_model_control_failure_invalidates(self):
        verdict = pilot_verdict(
            _prediction(confidence=0.6),
            {"ran": True, "outcome": "refuted", "null_model_control": "failed"},
        )
        self.assertEqual(verdict["verdict"], "invalid")
        self.assertEqual(verdict["surprise_bits"], 0.0)

    def test_pilot_that_did_not_run_is_invalid(self):
        verdict = pilot_verdict(_prediction(), {"ran": False, "outcome": "refuted"})
        self.assertEqual(verdict["verdict"], "invalid")

    def test_inconclusive_pilot_earns_no_bits(self):
        verdict = pilot_verdict(
            _prediction(),
            {"ran": True, "outcome": "inconclusive", "null_model_control": "passed"},
        )
        self.assertEqual(verdict["verdict"], "inconclusive")
        self.assertEqual(verdict["surprise_bits"], 0.0)


class Gate3Tests(unittest.TestCase):
    RIGOROUS = {"seeds": 3, "null_model_control": "passed", "p_value": 0.01, "packet_complete": True}

    def test_surprising_and_rigorous_goes_to_the_case_page(self):
        verdict = {"verdict": "escalate", "surprise_bits": 1.32, "reasons": []}
        route = route_outcome(verdict, self.RIGOROUS)
        self.assertEqual(route["channel"], "case_page")
        self.assertEqual(route["blockers"], [])

    def test_unsurprising_and_rigorous_goes_to_client_delivery(self):
        verdict = {"verdict": "stop", "surprise_bits": 0.74, "reasons": []}
        route = route_outcome(verdict, self.RIGOROUS)
        self.assertEqual(route["channel"], "client_delivery")

    def test_case_page_requires_the_null_model_control(self):
        verdict = {"verdict": "escalate", "surprise_bits": 1.32, "reasons": []}
        rigor = dict(self.RIGOROUS, null_model_control="missing")
        route = route_outcome(verdict, rigor)
        self.assertEqual(route["channel"], "withhold")
        self.assertIn("null-model control must run before any public number", route["blockers"])

    def test_client_delivery_only_discloses_a_missing_control(self):
        verdict = {"verdict": "stop", "surprise_bits": 0.74, "reasons": []}
        route = route_outcome(verdict, dict(self.RIGOROUS, null_model_control="missing"))
        self.assertEqual(route["channel"], "client_delivery")
        self.assertIn("null-model control not run", route["required_disclosures"])

    def test_thin_evidence_is_withheld_from_both_channels(self):
        verdict = {"verdict": "escalate", "surprise_bits": 1.32, "reasons": []}
        route = route_outcome(verdict, dict(self.RIGOROUS, seeds=1))
        self.assertEqual(route["channel"], "withhold")
        self.assertIn("seeds=1 < 3", route["blockers"])

    def test_failed_control_is_withheld_even_when_surprising(self):
        verdict = {"verdict": "invalid", "surprise_bits": 0.0, "reasons": ["control did not drop"]}
        route = route_outcome(verdict, dict(self.RIGOROUS, null_model_control="failed"))
        self.assertEqual(route["channel"], "withhold")


class AllocationTests(unittest.TestCase):
    def test_gpu_large_plan_still_starts_on_a_pilot_lane(self):
        alloc = allocate_compute(stage="pilot", resource_class="gpu_large", expected_bits=1.0)
        self.assertEqual(alloc["resource_class"], "gpu_small")
        self.assertEqual(alloc["budget_fraction"], 0.10)

    def test_bits_buy_the_planned_lane(self):
        alloc = allocate_compute(stage="full", resource_class="gpu_large", surprise_bits=1.32)
        self.assertEqual(alloc["resource_class"], "gpu_large")
        self.assertEqual(alloc["budget_fraction"], 1.0)
        self.assertEqual(alloc["priority"], 1.32)

    def test_stage_ladder_terminates(self):
        self.assertEqual(next_stage("pilot"), "confirm")
        self.assertEqual(next_stage("confirm"), "full")
        self.assertIsNone(next_stage("full"))

    def test_information_measures(self):
        self.assertAlmostEqual(binary_entropy(0.5), 1.0, places=6)
        self.assertLess(binary_entropy(0.95), 0.3)
        self.assertAlmostEqual(surprisal_bits(0.5), 1.0, places=6)


if __name__ == "__main__":
    unittest.main()
