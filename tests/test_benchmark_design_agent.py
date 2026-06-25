import unittest
from unittest import mock

from agents.benchmark_design_agent import (
    DESIGN_STATUS_RESOLVED,
    build_benchmark_design_contract,
)
from agents import experiment_forge
from agents.experiment_review import review_experiment_candidate


class BenchmarkDesignAgentTests(unittest.TestCase):
    def test_legal_claim_blocks_generic_gsm8k_substitution(self):
        parsed = {
            "id": 4,
            "tier": 2,
            "title": "Selective FOIA Privilege Classification via Latent Threshold Envelopes",
            "problem_statement": "Privilege review needs legal text classification evidence.",
        }
        method = {"name": "Threshold Envelope", "definition": "Classify FOIA privilege with evidence locking."}
        plan = {"datasets": [{"name": "GSM8K"}], "metrics": {"primary": "selective_risk"}}

        llm_contract = {
            "status": "resolved",
            "domain": "legal_nlp",
            "task_family": "legal_text_classification",
            "domain_literature_rationale": ["LegalBench/ContractNLI are the relevant benchmark families."],
            "benchmark_set_rationale": "LegalBench covers broad legal reasoning tasks while ContractNLI covers contract entailment, so the pair covers suite and focused legal NLI axes.",
            "candidate_benchmarks": [
                {
                    "name": "LegalBench",
                    "task_type": "legal_nlp",
                    "requires_harness": True,
                    "official_url": "https://hazyresearch.stanford.edu/legalbench/",
                    "literature_sources": [{"title": "LegalBench", "year": 2023, "url": "https://arxiv.org/abs/2308.11462"}],
                },
                {
                    "name": "ContractNLI",
                    "task_type": "legal_nli",
                    "requires_harness": True,
                    "official_url": "https://stanfordnlp.github.io/contract-nli/",
                    "literature_sources": [{"title": "ContractNLI", "year": 2021, "url": "https://arxiv.org/abs/2110.01799"}],
                },
            ],
            "required_baselines": [{"name": "Legal-BERT classifier"}],
            "primary_metric": {"name": "macro_f1", "direction": "higher"},
            "blockers": [],
        }
        with mock.patch("agents.benchmark_design_agent.call_llm_json", return_value=(llm_contract, 32)):
            contract = build_benchmark_design_contract(parsed, method, plan)

        self.assertEqual(contract["domain"], "legal_nlp")
        self.assertEqual(contract["status"], DESIGN_STATUS_RESOLVED)
        self.assertEqual(contract["candidate_benchmarks"][0]["name"], "LegalBench")
        self.assertIn("GSM8K", " ".join(contract["warnings"]))

    def test_math_prm_claim_can_use_gsm8k(self):
        parsed = {
            "id": 91,
            "tier": 2,
            "title": "Process Reward Models as Bellman Factorizations for math reasoning",
            "problem_statement": "Reasoning trajectories need process rewards.",
        }
        method = {"name": "PRM Bellman", "definition": "Use process reward models over chain-of-thought traces."}
        plan = {"datasets": [{"name": "GSM8K"}], "metrics": {"primary": "reward"}}

        llm_contract = {
            "status": "resolved",
            "domain": "math_reasoning_prm",
            "task_family": "math_reasoning",
            "benchmark_set_rationale": "GSM8K gives grade-school reasoning while PRM800K checks the step-level process-supervision axis required for PRM transfer.",
            "candidate_benchmarks": [
                {
                    "name": "GSM8K",
                    "task_type": "math_qa",
                    "hf_dataset": "openai/gsm8k",
                    "requires_harness": False,
                    "official_url": "https://github.com/openai/grade-school-math",
                    "literature_sources": [{"title": "Training Verifiers to Solve Math Word Problems", "year": 2021, "url": "https://arxiv.org/abs/2110.14168"}],
                },
                {
                    "name": "PRM800K",
                    "task_type": "process_reward_evaluation",
                    "benchmark_axis": "process_reward_annotations",
                    "requires_harness": True,
                    "official_url": "https://github.com/openai/prm800k",
                    "literature_sources": [{"title": "Let us Verify Step by Step", "year": 2023, "url": "https://arxiv.org/abs/2305.20050"}],
                },
            ],
            "required_baselines": [{"name": "chain-of-thought baseline"}],
            "primary_metric": {"name": "exact_match", "direction": "higher"},
            "blockers": [],
        }
        with mock.patch("agents.benchmark_design_agent.call_llm_json", return_value=(llm_contract, 28)):
            contract = build_benchmark_design_contract(parsed, method, plan)

        self.assertEqual(contract["domain"], "math_reasoning_prm")
        self.assertEqual(contract["status"], DESIGN_STATUS_RESOLVED)

    def test_prm_design_blocks_answer_only_math_benchmarks(self):
        parsed = {
            "id": 91,
            "tier": 2,
            "title": "Process Reward Models as Bellman Factorizations",
            "problem_statement": "Process reward credit assignment needs step-level evidence.",
        }
        method = {"name": "PRM Bellman", "definition": "Use process reward models over trajectories."}
        llm_contract = {
            "status": "resolved",
            "domain": "math_reasoning_prm",
            "task_family": "math_reasoning",
            "benchmark_set_rationale": "GSM8K and MATH are answer-only math benchmarks.",
            "candidate_benchmarks": [
                {
                    "name": "GSM8K",
                    "task_type": "math_qa",
                    "hf_dataset": "openai/gsm8k",
                    "official_url": "https://github.com/openai/grade-school-math",
                    "literature_sources": [{"title": "Training Verifiers to Solve Math Word Problems"}],
                },
                {
                    "name": "MATH",
                    "task_type": "math_qa",
                    "requires_harness": True,
                    "official_url": "https://github.com/hendrycks/math",
                    "literature_sources": [{"title": "Measuring Mathematical Problem Solving With the MATH Dataset"}],
                },
            ],
            "required_baselines": [{"name": "chain-of-thought baseline"}],
            "primary_metric": {"name": "exact_match", "direction": "higher"},
            "blockers": [],
        }
        with mock.patch("agents.benchmark_design_agent.call_llm_json", return_value=(llm_contract, 28)):
            contract = build_benchmark_design_contract(parsed, method, {"datasets": [{"name": "GSM8K"}]})

        self.assertEqual(contract["status"], "literature_review_required")
        self.assertIn("required domain evidence axis", " ".join(contract["blockers"]))


    def test_llm_failure_keeps_benchmark_design_blocked(self):
        parsed = {
            "title": "Process Reward Models as Bellman Factorizations for math reasoning",
            "problem_statement": "Reasoning trajectories need process rewards.",
        }
        method = {"name": "PRM Bellman", "definition": "Use process rewards."}
        plan = {"datasets": [{"name": "GSM8K"}]}

        with mock.patch("agents.benchmark_design_agent.call_llm_json", side_effect=RuntimeError("no route")):
            contract = build_benchmark_design_contract(parsed, method, plan)

        self.assertEqual(contract["status"], "literature_review_required")
        self.assertIn("LLM", " ".join(contract["blockers"]))


    def test_resolved_design_requires_dataset_sources_and_count_rationale(self):
        parsed = {
            "title": "Access Refusal and Prompt-Pressure Safety",
            "problem_statement": "Safety claims span tool-use prompt injection and text refusal.",
        }
        method = {"name": "Authorization Automaton", "definition": "Model refusal state transitions."}
        llm_contract = {
            "status": "resolved",
            "domain": "llm_safety_refusal",
            "task_family": "safety_refusal_evaluation",
            "candidate_benchmarks": [
                {"name": "AgentDojo", "task_type": "agent_tool_safety", "requires_harness": True}
            ],
            "required_baselines": [{"name": "direct instruction baseline"}],
            "primary_metric": {"name": "attack_success_rate", "direction": "lower"},
            "blockers": [],
        }
        with mock.patch("agents.benchmark_design_agent.call_llm_json", return_value=(llm_contract, 12)):
            contract = build_benchmark_design_contract(parsed, method, {"datasets": [{"name": "AgentDojo"}]})

        self.assertEqual(contract["status"], "literature_review_required")
        joined = " ".join(contract["blockers"])
        self.assertIn("paper or official benchmark sources", joined)
        self.assertIn("dataset count", joined)


    def test_forge_blocks_robotics_claim_until_domain_harness(self):
        llm_contract = {
            "status": "resolved",
            "domain": "robotics_policy",
            "task_family": "robotics_control",
            "domain_literature_rationale": ["Robomimic/LIBERO are standard manipulation benchmark families."],
            "benchmark_set_rationale": "Robomimic covers imitation manipulation and LIBERO covers language-conditioned manipulation, matching the robotics policy claim axes.",
            "candidate_benchmarks": [
                {
                    "name": "Robomimic",
                    "task_type": "robot_manipulation",
                    "requires_harness": True,
                    "official_url": "https://robomimic.github.io/",
                    "literature_sources": [{"title": "robomimic", "year": 2021, "url": "https://arxiv.org/abs/2108.03298"}],
                },
                {
                    "name": "LIBERO",
                    "task_type": "robot_manipulation",
                    "requires_harness": True,
                    "official_url": "https://libero-project.github.io/",
                    "literature_sources": [{"title": "LIBERO", "year": 2023, "url": "https://arxiv.org/abs/2306.03310"}],
                },
            ],
            "required_baselines": [{"name": "Diffusion Policy official baseline"}],
            "primary_metric": {"name": "success_rate", "direction": "higher"},
            "blockers": [],
        }
        with mock.patch("agents.benchmark_design_agent.call_llm_json", return_value=(llm_contract, 41)):
            enriched = experiment_forge._autofill_experiment_contracts(
                {
                    "id": 97,
                    "tier": 2,
                    "title": "Diffusion-Policy RL Finetuning as Wasserstein Gradient Flow",
                    "proposed_method": {
                        "name": "Diffusion Policy RL",
                        "definition": "Finetune diffusion policies on robot manipulation rollouts.",
                    },
                    "experimental_plan": {
                        "datasets": [{"name": "GSM8K"}],
                        "baselines": [{"name": "Diffusion Policy"}, {"name": "SAC"}],
                        "metrics": {"primary": "success_rate"},
                    },
                }
            )

        plan = enriched["experimental_plan"]
        self.assertEqual(plan["benchmark_design_contract"]["domain"], "robotics_policy")
        self.assertEqual(plan["benchmark_design_status"], "resolved")
        self.assertFalse(plan["generated_runner_supported"])
        self.assertEqual(plan["benchmark_targets"][0]["name"], "Robomimic")

        judgement = review_experiment_candidate(
            enriched,
            codebase={"url": "scratch", "name": "generated"},
            entrypoint_available=False,
        )
        self.assertEqual(judgement.recommended_route, "blocked")
        self.assertTrue(judgement.environment_review["benchmark_harness_required"])


if __name__ == "__main__":
    unittest.main()
