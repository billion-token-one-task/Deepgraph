import tempfile
import unittest
from pathlib import Path
from unittest import mock

from agents.benchmark_design_agent import build_benchmark_design_contract
from agents.plain_manuscript_reviewer import review_manuscript_plain
from agents.tier2_review_refine import _call_role


class LLMRoleBoundaryTests(unittest.TestCase):
    def test_benchmark_design_without_scope_never_calls_llm(self):
        with mock.patch(
            "agents.benchmark_design_agent.call_llm_json_for_role"
        ) as routed:
            contract = build_benchmark_design_contract(
                {"title": "A legal classification claim"},
                {"name": "Candidate"},
                {"datasets": [{"name": "GSM8K"}]},
            )

        routed.assert_not_called()
        self.assertNotEqual(contract["status"], "resolved")

    def test_tier2_reviewer_uses_scoped_role_route(self):
        proposer = {
            "name": "provider-a",
            "model": "model-a",
            "model_family": "family-a",
        }
        insight = {
            "agenda_id": 11,
            "proposal_candidate_id": 22,
            "resource_grant_id": 33,
        }
        with mock.patch(
            "agents.tier2_review_refine.call_llm_json_for_role",
            return_value=({"reviewer": "A"}, 17, {"name": "provider-b"}),
        ) as routed:
            payload, tokens, _route = _call_role(
                "system",
                "prompt",
                insight=insight,
                role="evaluator",
                operation="tier2_reviewer_a_round_1",
                proposer_route=proposer,
            )

        self.assertEqual(payload["reviewer"], "A")
        self.assertEqual(tokens, 17)
        kwargs = routed.call_args.kwargs
        self.assertEqual(kwargs["agenda_id"], 11)
        self.assertEqual(kwargs["idea_id"], 22)
        self.assertEqual(kwargs["resource_grant_id"], 33)
        self.assertEqual(kwargs["role"], "evaluator")
        self.assertEqual(kwargs["proposer_route"], proposer)

    def test_plain_reviewer_requires_recorded_proposer_and_reviewer_route(self):
        with tempfile.TemporaryDirectory(prefix="plain-review-") as directory:
            with (
                mock.patch(
                    "db.database.fetchone",
                    return_value={
                        "provider": "provider-a",
                        "model": "model-a",
                        "model_family": "family-a",
                    },
                ),
                mock.patch(
                    "agents.plain_manuscript_reviewer.call_llm_json_for_role",
                    return_value=(
                        {
                            "can_deliver": False,
                            "score": 5,
                            "issues": [{"severity": "high", "issue": "incomplete"}],
                        },
                        19,
                        {"name": "provider-b", "model": "model-b"},
                    ),
                ) as routed,
            ):
                review = review_manuscript_plain(
                    bundle_dir=Path(directory),
                    main_tex="\\section{Introduction} draft",
                    manuscript_state={
                        "agenda_id": 11,
                        "deep_insight_id": 22,
                        "resource_grant_id": 33,
                        "run_id": 44,
                    },
                )

        self.assertEqual(review["status"], "fail")
        kwargs = routed.call_args.kwargs
        self.assertEqual(kwargs["role"], "reviewer")
        self.assertEqual(kwargs["stage"], "manuscript")
        self.assertEqual(kwargs["proposer_route"]["provider"], "provider-a")


if __name__ == "__main__":
    unittest.main()
