"""Topic gate: reject what cannot pay for itself, before anything is spent."""

from __future__ import annotations

import json
import unittest
from unittest import mock

from agents.agenda_selector import score_candidate, select_next
from agents.topic_gate import (
    REASON_ALREADY_PUBLISHED,
    REASON_ANSWER_KNOWN,
    REASON_DUPLICATE_OR_OBSOLETE,
    REASON_EXPECTED_INFORMATION_TOO_LOW,
    REASON_GENERIC,
    REASON_MISSING_PREDICTION,
    REASON_NO_CHEAP_DECISIVE_EXPERIMENT,
    REASON_NO_DECISION_RELEVANCE,
    REASON_NOT_FALSIFIABLE,
    REASON_REJECT_KEYWORD,
    REASON_SCOPE_MISMATCH,
    STAGE_BUDGET_FRACTION,
    TopicGateError,
    TopicGatePolicy,
    binary_entropy,
    escalation_verdict,
    next_stage,
    observed_surprise_bits,
    screen_candidate,
    stage_token_cap,
    surprisal_bits,
)
from contracts.agenda import ResearchAgenda
from meta_harness.topic_gate_admission import TopicGateAdmissionError, require_pass


def _agenda(**overrides) -> ResearchAgenda:
    values = {
        "agenda_id": 5,
        "name": "robustness",
        "focus": ["robustness", "attention"],
        "token_budget": 50_000,
        "backend_allowlist": ["cpu", "llm"],
    }
    values.update(overrides)
    return ResearchAgenda(**values)


def _gate_record(**overrides) -> dict:
    record = {
        "prediction": {
            "predicted_outcome": "sparse attention keeps 95% accuracy under noise",
            "confidence": 0.6,
            "action_if_confirmed": "scale the sparse variant to the full benchmark",
            "action_if_refuted": "drop the sparsity claim and test the dense baseline",
            "already_published": "no",
        },
        "minimum_falsification_experiment": {
            "metric": "accuracy under injected noise",
            "baseline": "dense attention at the same parameter count",
            "decisive_comparison": "paired run over 3 seeds, delta > 2 points",
            "estimated_cost": {"tokens": 8000, "gpu_hours": 0, "wall_hours": 2},
        },
    }
    record.update(overrides)
    return record


def _candidate(**overrides) -> dict:
    candidate = {
        "id": 41,
        "agenda_id": 5,
        "title": "sparse attention robustness",
        "problem_statement": (
            "Does sparse attention preserve robustness under input noise at the "
            "same parameter count as dense attention?"
        ),
        "resource_class": "cpu",
        "experimentability": "easy",
        "topic_gate_json": json.dumps(_gate_record()),
    }
    candidate.update(overrides)
    return candidate


class ThreeQuestionTests(unittest.TestCase):
    def test_complete_candidate_passes(self):
        decision = screen_candidate(_candidate(), _agenda())

        self.assertTrue(decision.passed, decision.reason_codes)
        self.assertEqual(decision.reason_codes, ())
        self.assertAlmostEqual(decision.expected_bits, round(binary_entropy(0.6), 4))

    def test_missing_prediction_is_parked_not_elicited(self):
        # The historical gate called an LLM here and passed the candidate when
        # the provider was down. That silent fallback is gone.
        decision = screen_candidate(_candidate(topic_gate_json=None), _agenda())

        self.assertFalse(decision.passed)
        self.assertIn(REASON_MISSING_PREDICTION, decision.reason_codes)

    def test_known_answer_is_rejected(self):
        record = _gate_record()
        record["prediction"]["confidence"] = 0.97
        decision = screen_candidate(
            _candidate(topic_gate_json=json.dumps(record)), _agenda()
        )

        self.assertFalse(decision.passed)
        self.assertIn(REASON_ANSWER_KNOWN, decision.reason_codes)

    def test_same_action_for_both_outcomes_is_rejected(self):
        record = _gate_record()
        record["prediction"]["action_if_refuted"] = record["prediction"][
            "action_if_confirmed"
        ]
        decision = screen_candidate(
            _candidate(topic_gate_json=json.dumps(record)), _agenda()
        )

        self.assertFalse(decision.passed)
        self.assertIn(REASON_NO_DECISION_RELEVANCE, decision.reason_codes)

    def test_undeclared_action_is_rejected(self):
        record = _gate_record()
        record["prediction"]["action_if_refuted"] = ""
        decision = screen_candidate(
            _candidate(topic_gate_json=json.dumps(record)), _agenda()
        )

        self.assertIn(REASON_NO_DECISION_RELEVANCE, decision.reason_codes)

    def test_already_published_is_rejected_with_its_citation(self):
        record = _gate_record()
        record["prediction"]["already_published"] = "yes"
        record["prediction"]["already_published_evidence"] = "arXiv:2401.00001"
        decision = screen_candidate(
            _candidate(topic_gate_json=json.dumps(record)), _agenda()
        )

        self.assertFalse(decision.passed)
        self.assertIn(REASON_ALREADY_PUBLISHED, decision.reason_codes)
        self.assertIn(
            "arXiv:2401.00001",
            " ".join(blocker["reason"] for blocker in decision.blockers),
        )


class NoveltyAndFalsifiabilityTests(unittest.TestCase):
    def test_duplicate_or_obsolete_candidate_is_rejected(self):
        for status in ("exists", "duplicate", "obsolete", "solved"):
            with self.subTest(status=status):
                decision = screen_candidate(
                    _candidate(novelty_status=status), _agenda()
                )
                self.assertFalse(decision.passed)
                self.assertIn(REASON_DUPLICATE_OR_OBSOLETE, decision.reason_codes)

    def test_non_falsifiable_candidate_is_rejected(self):
        record = _gate_record()
        record["minimum_falsification_experiment"] = {"metric": "accuracy"}
        decision = screen_candidate(
            _candidate(topic_gate_json=json.dumps(record)), _agenda()
        )

        self.assertFalse(decision.passed)
        self.assertIn(REASON_NOT_FALSIFIABLE, decision.reason_codes)

    def test_expensive_decisive_experiment_is_rejected(self):
        record = _gate_record()
        record["minimum_falsification_experiment"]["estimated_cost"] = {
            "tokens": 500_000,
            "gpu_hours": 40,
            "wall_hours": 400,
        }
        decision = screen_candidate(
            _candidate(topic_gate_json=json.dumps(record)), _agenda()
        )

        self.assertFalse(decision.passed)
        self.assertIn(REASON_NO_CHEAP_DECISIVE_EXPERIMENT, decision.reason_codes)

    def test_missing_cost_is_rejected(self):
        record = _gate_record()
        record["minimum_falsification_experiment"].pop("estimated_cost")
        decision = screen_candidate(
            _candidate(topic_gate_json=json.dumps(record)), _agenda()
        )

        self.assertIn(REASON_NO_CHEAP_DECISIVE_EXPERIMENT, decision.reason_codes)

    def test_generic_statement_is_rejected(self):
        decision = screen_candidate(
            _candidate(problem_statement="", title="attention research"),
            _agenda(),
        )

        self.assertFalse(decision.passed)
        self.assertIn(REASON_GENERIC, decision.reason_codes)

    def test_out_of_scope_and_rejected_keywords(self):
        self.assertIn(
            REASON_SCOPE_MISMATCH,
            screen_candidate(_candidate(agenda_id=9), _agenda()).reason_codes,
        )
        agenda = _agenda(reject={"keywords": ["sparse attention"]})
        self.assertIn(
            REASON_REJECT_KEYWORD,
            screen_candidate(_candidate(), agenda).reason_codes,
        )

    def test_low_expected_information_is_rejected(self):
        record = _gate_record()
        record["prediction"]["confidence"] = 0.985
        policy = TopicGatePolicy(max_confidence=0.99, min_expected_bits=0.25)
        decision = screen_candidate(
            _candidate(topic_gate_json=json.dumps(record)), _agenda(), policy=policy
        )

        self.assertIn(REASON_EXPECTED_INFORMATION_TOO_LOW, decision.reason_codes)


class NegativeResultsStayValidTests(unittest.TestCase):
    def test_predicting_a_negative_outcome_is_not_a_blocker(self):
        record = _gate_record()
        record["prediction"]["predicted_outcome"] = (
            "sparse attention will NOT preserve robustness under noise"
        )
        decision = screen_candidate(
            _candidate(topic_gate_json=json.dumps(record)), _agenda()
        )

        self.assertTrue(decision.passed, decision.reason_codes)

    def test_a_refuted_pilot_escalates_because_it_produced_bits(self):
        verdict = escalation_verdict(
            {"confidence": 0.6, "predicted_outcome": "improves"},
            {"ran": True, "outcome": "refuted", "attribution_control": "passed"},
        )

        self.assertEqual(verdict["verdict"], "escalate")
        self.assertGreaterEqual(verdict["surprise_bits"], 1.0)

    def test_a_confirmed_low_surprise_pilot_stops(self):
        verdict = escalation_verdict(
            {"confidence": 0.6, "predicted_outcome": "improves"},
            {"ran": True, "outcome": "confirmed", "attribution_control": "passed"},
        )

        self.assertEqual(verdict["verdict"], "stop")
        self.assertLess(verdict["surprise_bits"], 1.0)
        self.assertIn("running_is_not_passing", verdict["reason_codes"])

    def test_unattributable_or_unrun_pilot_is_invalid(self):
        self.assertEqual(
            escalation_verdict({}, {"ran": False})["verdict"],
            "invalid",
        )
        self.assertEqual(
            escalation_verdict(
                {"confidence": 0.6},
                {"ran": True, "outcome": "refuted", "attribution_control": "failed"},
            )["verdict"],
            "invalid",
        )

    def test_undecided_pilot_is_inconclusive(self):
        self.assertEqual(
            escalation_verdict(
                {"confidence": 0.6},
                {"ran": True, "outcome": "inconclusive"},
            )["verdict"],
            "inconclusive",
        )


class SurpriseDrivenComputeTests(unittest.TestCase):
    def test_pilot_gets_a_tenth_of_the_planned_budget(self):
        self.assertEqual(STAGE_BUDGET_FRACTION["pilot"], 0.10)
        self.assertEqual(stage_token_cap(50_000, stage="pilot"), 5_000)
        self.assertEqual(stage_token_cap(50_000, stage="confirm"), 17_500)
        self.assertEqual(stage_token_cap(50_000, stage="full"), 50_000)

    def test_stage_ladder_is_ordered_and_finite(self):
        self.assertEqual(next_stage("pilot"), "confirm")
        self.assertEqual(next_stage("confirm"), "full")
        self.assertIsNone(next_stage("full"))
        with self.assertRaises(TopicGateError):
            next_stage("unbounded")
        with self.assertRaises(TopicGateError):
            stage_token_cap(0, stage="pilot")

    def test_bits_are_information_theoretic(self):
        self.assertAlmostEqual(surprisal_bits(0.5), 1.0)
        self.assertAlmostEqual(binary_entropy(0.5), 1.0)
        self.assertAlmostEqual(
            observed_surprise_bits(confidence=0.6, outcome="refuted"),
            round(surprisal_bits(0.4), 4),
        )
        self.assertEqual(
            observed_surprise_bits(confidence=0.6, outcome="inconclusive"), 0.0
        )


class SelectorIntegrationTests(unittest.TestCase):
    def test_blocked_candidate_is_never_ranked(self):
        score, breakdown, blockers = score_candidate(
            _candidate(topic_gate_json=None), _agenda()
        )

        self.assertEqual(score, float("-inf"))
        self.assertIn(REASON_MISSING_PREDICTION, blockers)
        self.assertIn("topic_gate", breakdown)

    def test_expected_information_is_a_ranking_feature(self):
        low = _gate_record()
        low["prediction"]["confidence"] = 0.88
        high_score, high_parts, _ = score_candidate(_candidate(), _agenda())
        low_score, low_parts, _ = score_candidate(
            _candidate(topic_gate_json=json.dumps(low)), _agenda()
        )

        self.assertGreater(high_parts["expected_information"], low_parts["expected_information"])
        self.assertGreater(high_score, low_score)

    def test_selector_rejects_gate_failures_with_reasons(self):
        class FakeRepository:
            def get(self, agenda_id):
                return _agenda(agenda_id=agenda_id)

            def candidates(self, agenda_id, *, limit):
                return [_candidate(agenda_id=agenda_id, topic_gate_json=None)]

            def save_selection(self, selection):  # pragma: no cover - not reached
                raise AssertionError("a blocked candidate must not be selected")

            def queue_selected_insight(self, selection):  # pragma: no cover
                raise AssertionError("a blocked candidate must not be queued")

        self.assertIsNone(select_next(5, repository=FakeRepository()))


class AdmissionTests(unittest.TestCase):
    """The gate cannot be bypassed by a route that skips the selector."""

    def test_admission_reruns_the_gate_from_persisted_rows(self):
        agenda_row = {
            "id": 5,
            "name": "robustness",
            "focus_json": json.dumps(["robustness", "attention"]),
            "token_budget": 50_000,
            "backend_allowlist_json": json.dumps(["cpu", "llm"]),
            "max_concurrency": 1,
            "status": "active",
            "is_active": 1,
        }
        with mock.patch(
            "meta_harness.topic_gate_admission.db.fetchone",
            side_effect=[_candidate(), agenda_row],
        ):
            decision = require_pass(agenda_id=5, idea_id=41)
        self.assertTrue(decision.passed)

        with mock.patch(
            "meta_harness.topic_gate_admission.db.fetchone",
            side_effect=[_candidate(topic_gate_json=None), agenda_row],
        ):
            with self.assertRaisesRegex(
                TopicGateAdmissionError, REASON_MISSING_PREDICTION
            ):
                require_pass(agenda_id=5, idea_id=41)

    def test_unscoped_candidate_cannot_be_admitted(self):
        with mock.patch(
            "meta_harness.topic_gate_admission.db.fetchone", return_value=None
        ):
            with self.assertRaises(TopicGateAdmissionError):
                require_pass(agenda_id=5, idea_id=41)

    def test_promote_decision_cannot_bypass_the_gate(self):
        from contracts.meta_harness import Estimate, IdeaDecisionPacket
        from meta_harness.repository import (
            MetaHarnessPersistenceError,
            MetaHarnessRepository,
        )

        def _estimate(value: float) -> Estimate:
            return Estimate(
                value=value,
                lower=0.0,
                upper=1.0,
                evaluator="judge-v1",
                provider="provider-b",
                model="model-b",
                evidence_sources=["frontier:1"],
            )

        packet = IdeaDecisionPacket(
            agenda_id=5,
            idea_id=41,
            frontier_packet_id=3,
            expected_impact=_estimate(0.5),
            success_probability=_estimate(0.5),
            novelty=_estimate(0.5),
            obsolescence_probability=_estimate(0.1),
            falsification_value=_estimate(0.5),
            reuse_value=_estimate(0.5),
            expected_token_cost=_estimate(0.5),
            expected_gpu_cost=_estimate(0.0),
            time_to_feedback=_estimate(0.5),
            execution_risk=_estimate(0.2),
            information_value=_estimate(0.5),
            candidate_family="attention",
            correlation_keys=["sparse-attention"],
            decision="promote",
            reason_codes=["portfolio_score_selected"],
        )
        blocked = mock.Mock()
        blocked.passed = False
        blocked.reason_codes = (REASON_MISSING_PREDICTION,)
        with mock.patch(
            "meta_harness.repository.db.fetchone",
            return_value={"agenda_id": 5, "gate_allowed": 1},
        ), mock.patch(
            "meta_harness.repository.topic_gate_admission.evaluate",
            return_value=blocked,
        ), mock.patch(
            "meta_harness.repository.db.insert_returning_id"
        ) as insert:
            with self.assertRaisesRegex(
                MetaHarnessPersistenceError, REASON_MISSING_PREDICTION
            ):
                MetaHarnessRepository().save_decision(packet)
        insert.assert_not_called()


if __name__ == "__main__":
    unittest.main()
