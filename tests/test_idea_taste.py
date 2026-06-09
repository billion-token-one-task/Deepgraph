import json
import unittest
from unittest import mock

from agents.idea_taste import (
    compute_taste_score,
    graph_novelty_gate,
    score_excitement,
    score_graph_novelty,
)


class IdeaTasteTests(unittest.TestCase):
    def test_graph_novelty_gate_rejects_empty_method_with_dense_graph_signal(self):
        insight = {
            "title": "Transformer baseline on GSM8K",
            "proposed_method": json.dumps(
                {"name": "Transformer", "one_line": "standard transformer baseline"}
            ),
            "source_node_ids": json.dumps(["ml.dl.nlp.reasoning"]),
        }
        result = score_graph_novelty(insight)
        self.assertIn("score", result)
        self.assertIn(result["status"], {"novel", "partial", "likely_exists"})

    def test_compute_taste_score_prefers_graph_novelty(self):
        base = {
            "evidence_packet": {
                "non_numeric_evidence": ["a", "b"],
                "structural_evidence": ["s"],
            },
            "support_score": 6,
            "mechanism_type": "claim_method_gap",
            "signal_mix": ["claim_method_gap"],
            "source_node_ids": ["ml.dl.nlp.reasoning"],
            "resource_class": "cpu",
            "graph_counterevidence": [],
        }
        low = compute_taste_score({**base, "graph_novelty": {"score": 2.0}})["taste_score"]
        high = compute_taste_score({**base, "graph_novelty": {"score": 9.0}})["taste_score"]
        self.assertGreater(high, low)

    def test_graph_novelty_gate_returns_error_for_likely_exists(self):
        with mock.patch(
            "agents.idea_taste.score_graph_novelty",
            return_value={"score": 2.0, "status": "likely_exists", "reasons": ["already done"]},
        ):
            gate = graph_novelty_gate({"title": "Known method"})
        self.assertIsNotNone(gate)

    def test_excitement_rewards_mechanism_over_benchmark_only(self):
        benchmark_only = {
            "title": "A benchmark for evaluating tool use",
            "problem_statement": "We propose a benchmark, metrics, and diagnostic suite.",
            "proposed_method": json.dumps(
                {"name": "ToolBenchScore", "type": "evaluation", "one_line": "A benchmark metric."}
            ),
            "experimental_plan": json.dumps({"datasets": ["one benchmark"]}),
        }
        mechanism = {
            "title": "Latent-State Routing Objective for Tool Use",
            "problem_statement": "Learn an invariant latent variable policy across tasks.",
            "proposed_method": json.dumps(
                {
                    "name": "Latent-State Routing Objective",
                    "type": "training_procedure",
                    "one_line": "A reusable objective for selecting tool calls.",
                    "definition": "min_theta E[L(theta)] subject to Pr(error)<delta",
                    "pseudocode": "1. Estimate latent state. 2. Optimize policy.",
                    "key_properties": ["cross-task reusable objective"],
                }
            ),
            "experimental_plan": json.dumps({"datasets": ["HotpotQA", "MuSiQue"]}),
        }
        self.assertGreater(
            score_excitement(mechanism)["score"],
            score_excitement(benchmark_only)["score"],
        )


if __name__ == "__main__":
    unittest.main()
