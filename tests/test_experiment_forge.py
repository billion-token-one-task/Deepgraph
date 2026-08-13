import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from agents import experiment_forge
from agents.experiment_review import review_experiment_candidate


class GenerateScaffoldTests(unittest.TestCase):
    @staticmethod
    def _capability_requirements():
        return {
            "task_protocol": "generative_qa",
            "candidate_hook": "candidate_prompt",
            "dataset": {
                "repository_id": "openai/gsm8k",
                "revision": "dataset-revision",
                "split": "test",
                "field_mapping": {"input": "question", "target": "answer"},
            },
            "model": {
                "repository_id": "Qwen/Qwen3-4B-Instruct-2507",
                "revision": "model-revision",
                "framework": "transformers",
                "task": "causal_lm",
            },
            "metric": {"name": "exact_match", "direction": "higher"},
            "artifact_contract": ["final_results", "raw_predictions"],
            "preferred_backends": ["ssh_gpu"],
        }

    @staticmethod
    def _capability_insight():
        return {
            "resource_class": "gpu_large",
            "proposed_method": {
                "name": "Counterfactual Gain-Gated Reasoning",
                "type": "prompting",
                "definition": (
                    "Estimate whether another reasoning pass will improve the "
                    "answer, and request it only when estimated gain is positive."
                ),
            },
            "experimental_plan": {
                "baselines": ["Direct", "Always-CoT"],
                "datasets": [{"name": "GSM8K"}],
                "metrics": {"primary": "exact_match"},
                "procedure": "Compare equal-budget direct and gain-gated prompts.",
            },
        }

    def test_preflight_forge_parses_frozen_json_fields_without_autofill(self):
        plan = {"model_targets": [{"hf_model": "Qwen/Qwen3-4B-Instruct-2507"}]}
        method = {"name": "CGGR", "definition": "Route reasoning selectively."}
        evidence = {"main_table": {"enabled": True}}
        insight = {
            "id": 105,
            "agenda_id": 11,
            "experimental_plan": json.dumps(json.dumps(plan)),
            "proposed_method": json.dumps(json.dumps(method)),
            "evidence_plan": json.dumps(json.dumps(evidence)),
        }

        with mock.patch.object(
            experiment_forge,
            "_autofill_experiment_contracts",
            side_effect=AssertionError("passed preflight must keep the frozen plan"),
        ):
            parsed = experiment_forge._prepare_forge_insight(
                insight,
                preflight_row={"status": "passed"},
                llm_scope={"agenda_id": 11, "idea_id": 105},
            )

        self.assertEqual(parsed["experimental_plan"], plan)
        self.assertEqual(parsed["proposed_method"], method)
        self.assertEqual(parsed["evidence_plan"], evidence)
        publication = experiment_forge._publication_evidence_contract(
            parsed,
            parsed["experimental_plan"],
            evidence_plan=parsed["evidence_plan"],
        )
        self.assertEqual(
            publication["required_models"],
            ["Qwen/Qwen3-4B-Instruct-2507"],
        )

    def test_persist_enriched_insight_normalizes_serialized_json_fields(self):
        plan = {"model_targets": [{"hf_model": "Qwen/Qwen3-4B-Instruct-2507"}]}
        method = {"name": "CGGR"}
        evidence = {"main_table": {"enabled": True}}
        parsed = {
            "agenda_id": 11,
            "resource_class": "gpu_large",
            "experimental_plan": json.dumps(json.dumps(plan)),
            "proposed_method": json.dumps(method),
            "evidence_plan": json.dumps(evidence),
        }

        with (
            mock.patch.object(experiment_forge.db, "execute") as execute,
            mock.patch.object(experiment_forge.db, "commit") as commit,
        ):
            experiment_forge._persist_enriched_insight(105, parsed)

        params = execute.call_args.args[1]
        self.assertEqual(json.loads(params[0]), method)
        self.assertEqual(json.loads(params[1]), plan)
        self.assertEqual(json.loads(params[2]), evidence)
        self.assertEqual(params[3:], ("gpu_large", 105, 11))
        commit.assert_called_once_with()

    def test_resource_granted_forge_llm_uses_scoped_role_route(self):
        scope = {
            "agenda_id": 11,
            "idea_id": 22,
            "resource_grant_id": 33,
            "stage": "experiment_forge",
        }
        route = {
            "provider": "provider-a",
            "model": "model-a",
            "model_family": "family-a",
            "prompt_version": "forge-proposer-v1",
        }
        with (
            mock.patch.object(
                experiment_forge,
                "configured_role_prompt_version",
                return_value="forge-proposer-v1",
            ),
            mock.patch.object(
                experiment_forge,
                "call_llm_json_for_role",
                return_value=({"ok": True}, 19, route),
            ) as routed,
        ):
            result, tokens, actual_route = (
                experiment_forge._resource_granted_proposer_json(
                    "system",
                    "prompt",
                    llm_scope=scope,
                    operation="experiment_forge.test",
                    max_tokens=200,
                )
            )

        self.assertEqual(result, {"ok": True})
        self.assertEqual(tokens, 19)
        self.assertEqual(actual_route, route)
        kwargs = routed.call_args.kwargs
        self.assertEqual(kwargs["agenda_id"], 11)
        self.assertEqual(kwargs["idea_id"], 22)
        self.assertEqual(kwargs["resource_grant_id"], 33)
        self.assertEqual(kwargs["stage"], "experiment_forge")
        self.assertEqual(kwargs["role"], "proposer")
        self.assertEqual(kwargs["prompt_version"], "forge-proposer-v1")
        self.assertTrue(
            kwargs["idempotency_key"].startswith("experiment_forge.test:")
        )

    def test_scoped_code_scout_does_not_use_unrouted_agentic_fallback(self):
        scope = {
            "agenda_id": 11,
            "idea_id": 22,
            "resource_grant_id": 33,
            "stage": "experiment_forge",
        }
        with mock.patch.object(
            experiment_forge,
            "_scout_codebase_single_shot",
            side_effect=RuntimeError("injected routed provider outage"),
        ) as routed:
            with self.assertRaisesRegex(RuntimeError, "provider outage"):
                experiment_forge.scout_codebase({"id": 22}, llm_scope=scope)
        routed.assert_called_once_with({"id": 22}, llm_scope=scope)

    def test_capability_scaffold_repairs_only_missing_top_level_fields(self):
        initial = {
            "success_criteria": {
                "metric_name": "exact_match",
                "metric_direction": "higher",
                "exciting": 0.05,
                "solid": 0.02,
                "disappointing": 0.0,
            }
        }
        adapter = (
            "CANDIDATE_METHOD = 'counterfactual_gain_gate'\n\n"
            "def candidate_prompt(example, baseline_prompt):\n"
            "    return baseline_prompt + '\\nCheck whether one more derivation is useful.'\n"
        )
        repaired = {
            "program_md": "# Program\nNEVER STOP until interrupted.",
            "evaluate_py": "print(0.0)\n",
            "candidate_adapter_py": adapter,
        }
        route = {"provider": "test", "model": "test-model"}

        with tempfile.TemporaryDirectory() as tmpdir:
            with mock.patch.object(
                experiment_forge,
                "_resource_granted_proposer_json",
                side_effect=[(initial, 11, route), (repaired, 7, route)],
            ) as proposer:
                scaffold = experiment_forge.generate_scaffold(
                    self._capability_insight(),
                    {
                        "url": "scratch",
                        "name": "minimal",
                        "main_train_file": "train.py",
                        "main_eval_command": "python train.py",
                    },
                    Path(tmpdir),
                    llm_scope={
                        "agenda_id": 11,
                        "idea_id": 105,
                        "resource_grant_id": 31,
                        "stage": "pilot",
                    },
                    runner_capability_bound=True,
                    runner_requirements=self._capability_requirements(),
                )

        self.assertEqual(scaffold["tokens"], 18)
        self.assertEqual(scaffold["candidate_adapter_py"], adapter)
        self.assertEqual(
            scaffold["capability_scaffold_repaired_fields"],
            ["program_md", "evaluate_py", "candidate_adapter_py"],
        )
        self.assertEqual(proposer.call_count, 2)
        self.assertEqual(
            proposer.call_args_list[1].kwargs["operation"],
            "experiment_forge.capability_scaffold_repair",
        )
        self.assertIn(
            '"candidate_hook": "candidate_prompt"',
            proposer.call_args_list[1].args[1],
        )

    def test_capability_scaffold_repair_fails_closed_before_writing_specs(self):
        initial = {
            "program_md": "# Program",
            "evaluate_py": "print(0.0)",
            "success_criteria": {"metric_name": "exact_match"},
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            workdir = Path(tmpdir)
            with mock.patch.object(
                experiment_forge,
                "_resource_granted_proposer_json",
                side_effect=[(initial, 11, {}), ({}, 5, {})],
            ):
                with self.assertRaisesRegex(
                    experiment_forge.CapabilityScaffoldContractError,
                    "capability_scaffold_contract_missing:candidate_adapter_py",
                ):
                    experiment_forge.generate_scaffold(
                        self._capability_insight(),
                        {"url": "scratch", "name": "minimal"},
                        workdir,
                        llm_scope={
                            "agenda_id": 11,
                            "idea_id": 105,
                            "resource_grant_id": 31,
                            "stage": "pilot",
                        },
                        runner_capability_bound=True,
                        runner_requirements=self._capability_requirements(),
                    )
            self.assertFalse((workdir / "spec").exists())

    def test_capability_scaffold_requires_durable_runner_contract_before_llm(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with mock.patch.object(
                experiment_forge,
                "_resource_granted_proposer_json",
            ) as proposer:
                with self.assertRaisesRegex(
                    experiment_forge.CapabilityScaffoldContractError,
                    "capability_scaffold_runner_contract_required",
                ):
                    experiment_forge.generate_scaffold(
                        self._capability_insight(),
                        {"url": "scratch", "name": "minimal"},
                        Path(tmpdir),
                        llm_scope={
                            "agenda_id": 11,
                            "idea_id": 105,
                            "resource_grant_id": 31,
                            "stage": "pilot",
                        },
                        runner_capability_bound=True,
                    )
            proposer.assert_not_called()

    def test_capability_scaffold_empty_success_criteria_requires_repair(self):
        missing = experiment_forge._missing_capability_scaffold_fields(
            {
                "program_md": "# Program",
                "evaluate_py": "print(0.0)",
                "candidate_adapter_py": "CANDIDATE_METHOD = 'method'",
                "success_criteria": {},
            }
        )

        self.assertEqual(missing, ["success_criteria"])

    def test_capability_scaffold_resume_uses_only_focused_repair(self):
        initial = {
            "program_md": "",
            "evaluate_py": "",
            "candidate_adapter_py": "",
            "success_criteria": {
                "metric_name": "exact_match",
                "metric_direction": "higher",
                "exciting": 0.05,
                "solid": 0.02,
                "disappointing": 0.0,
            },
        }
        repaired = {
            "program_md": "# Program\nNEVER STOP until interrupted.",
            "evaluate_py": "print(0.0)",
            "candidate_adapter_py": (
                "CANDIDATE_METHOD = 'gain_gate'\n"
                "def candidate_prompt(example, baseline_prompt):\n"
                "    return baseline_prompt + '\\nVerify the derivation.'\n"
            ),
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            with mock.patch.object(
                experiment_forge,
                "_resource_granted_proposer_json",
                return_value=(repaired, 7, {}),
            ) as proposer:
                scaffold = experiment_forge.generate_scaffold(
                    self._capability_insight(),
                    {"url": "scratch", "name": "minimal"},
                    Path(tmpdir),
                    llm_scope={
                        "agenda_id": 11,
                        "idea_id": 105,
                        "resource_grant_id": 31,
                        "stage": "pilot",
                    },
                    runner_capability_bound=True,
                    runner_requirements=self._capability_requirements(),
                    initial_result=initial,
                )

        self.assertEqual(scaffold["tokens"], 7)
        proposer.assert_called_once()
        self.assertEqual(
            proposer.call_args.kwargs["operation"],
            "experiment_forge.capability_scaffold_repair",
        )

    def test_failed_capability_scaffold_loader_requires_exact_original_error(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            spec = root / "spec"
            spec.mkdir()
            (spec / "program.md").write_text("", encoding="utf-8")
            (spec / "evaluate.py").write_text("", encoding="utf-8")
            (spec / "success_criteria.json").write_text(
                json.dumps(
                    {
                        "metric_name": "exact_match",
                        "metric_direction": "higher",
                        "exciting": 0.05,
                        "solid": 0.02,
                        "disappointing": 0.0,
                    }
                ),
                encoding="utf-8",
            )
            row = {
                "id": 137,
                "status": "failed",
                "phase": "runner_materialization_failed",
                "error_message": "runner_contract_violation:candidate_adapter_required",
                "workdir": str(root),
            }
            with (
                mock.patch.object(experiment_forge.db, "fetchone", return_value=row),
                mock.patch.object(experiment_forge.db, "commit"),
            ):
                loaded, workdir, initial = (
                    experiment_forge._load_failed_capability_scaffold(
                        run_id=137,
                        agenda_id=11,
                        insight_id=105,
                        resource_grant_id=31,
                    )
                )
            self.assertEqual(loaded["id"], 137)
            self.assertEqual(workdir, root)
            self.assertEqual(initial["candidate_adapter_py"], "")

            row["error_message"] = "runner_contract_violation:other"
            with (
                mock.patch.object(experiment_forge.db, "fetchone", return_value=row),
                mock.patch.object(experiment_forge.db, "commit"),
            ):
                with self.assertRaisesRegex(
                    experiment_forge.CapabilityScaffoldContractError,
                    "capability_scaffold_resume_source_invalid",
                ):
                    experiment_forge._load_failed_capability_scaffold(
                        run_id=137,
                        agenda_id=11,
                        insight_id=105,
                        resource_grant_id=31,
                    )

    def test_autofill_experiment_contracts_fills_missing_review_fields(self):
        llm_gate = {"status": "literature_review_required", "blockers": ["needs domain benchmark review"]}
        with mock.patch("agents.benchmark_design_agent.call_llm_json_for_role", return_value=(llm_gate, 0, {})):
            enriched = experiment_forge._autofill_experiment_contracts(
                {
                    "id": 7,
                    "tier": 2,
                    "title": "Secure linguistic communication and linguistic steganography as measure-preserving coding",
                    "proposed_method": None,
                    "experimental_plan": {
                        "models": ["gpt2-medium", "roberta-base", "sentence-transformers/all-mpnet-base-v2"],
                        "datasets": ["WikiText-103", "CNN/DailyMail"],
                        "procedure": "Compare coders, measure BER and detector AUC, and report bits/token.",
                    },
                    "supporting_papers": ["ACF", "Discop"],
                },
                llm_scope={
                    "agenda_id": 1,
                    "idea_id": 7,
                    "resource_grant_id": 3,
                    "stage": "experiment_forge",
                },
            )

        self.assertTrue(enriched["proposed_method"]["definition"])
        self.assertGreaterEqual(len(enriched["experimental_plan"]["baselines"]), 2)
        self.assertEqual(enriched["experimental_plan"]["metrics"]["primary"], "bit_error_rate")
        self.assertEqual(enriched["experimental_plan"]["datasets"][0]["name"], "WikiText-103")
        self.assertIn("publication_evidence_contract", enriched["experimental_plan"])
        self.assertIn("paper_intent", enriched["experimental_plan"])

    def test_autofill_experiment_contracts_makes_gpu_plan_real_benchmark_reviewable(self):
        llm_gate = {"status": "literature_review_required", "blockers": ["needs domain benchmark review"]}
        with mock.patch("agents.benchmark_design_agent.call_llm_json_for_role", return_value=(llm_gate, 0, {})):
            enriched = experiment_forge._autofill_experiment_contracts(
                {
                    "id": 16,
                    "tier": 2,
                    "title": "SSH GPU Smoke Validation Experiment Auto Experiment Run",
                    "resource_class": "gpu_small",
                    "proposed_method": {
                        "name": "remote_gpu_smoke",
                        "type": "systems_validation",
                        "definition": "Run a short CUDA-backed tensor workload and report device/VRAM telemetry.",
                    },
                    "experimental_plan": {
                        "baselines": [{"name": "remote_cuda_probe"}],
                        "metrics": [{"name": "gpu_probe_score"}],
                        "compute_budget": {"gpu_hours": 0.01},
                    },
                },
                llm_scope={
                    "agenda_id": 1,
                    "idea_id": 16,
                    "resource_grant_id": 3,
                    "stage": "experiment_forge",
                },
            )

        self.assertGreaterEqual(len(enriched["experimental_plan"]["baselines"]), 2)
        self.assertEqual(enriched["experimental_plan"]["metrics"]["primary"], "gpu_probe_score")
        self.assertEqual(enriched["experimental_plan"]["compute_budget"]["total_gpu_hours"], 0.01)
        self.assertTrue(enriched["experimental_plan"]["real_benchmark_required"])
        self.assertFalse(enriched["experimental_plan"]["generated_runner_supported"])
        self.assertEqual(enriched["experimental_plan"]["benchmark_design_status"], "literature_review_required")
        self.assertIn("benchmark_design_blockers", enriched["experimental_plan"])

        judgement = review_experiment_candidate(
            enriched,
            codebase={"url": "https://github.com/example/repo", "name": "repo", "main_train_file": "train.py", "main_eval_command": "python train.py"},
            entrypoint_available=True,
        )
        self.assertEqual(judgement.recommended_route, "blocked")
        self.assertTrue(judgement.environment_review["benchmark_harness_required"])

    def test_benchmark_plan_blocks_manuscript_until_full_artifacts(self):
        contract = experiment_forge._publication_evidence_contract(
            {
                "title": "CGGR",
                "problem_statement": "Selective deliberation needs fair QA benchmarks.",
                "proposed_method": {
                    "name": "CGGR",
                    "definition": "Estimate counterfactual reasoning gain before spending extra inference budget.",
                },
            },
            {
                "datasets": [{"name": "GSM8K"}],
                "baselines": [{"name": "Direct"}, {"name": "Always-CoT"}, {"name": "Random Budget-Matched"}],
                "model_targets": [{"name": "Qwen/Qwen2.5-7B-Instruct"}],
                "metrics": {"primary": "cost_adjusted_accuracy"},
                "ablations": [{"name": "no_counterfactual_delta"}, {"name": "compute_matched_baseline"}],
                "minimum_seeds": 5,
                "real_benchmark_required": True,
            },
            codebase={"url": "scratch", "name": "minimal"},
            scaffold_kind="full_benchmark_compiled",
        )

        self.assertEqual(contract["evidence_tier"], "benchmark_plan")
        self.assertTrue(contract["claim_route"]["paper_allowed"])
        self.assertTrue(contract["quality_gates"]["requires_full_benchmark_package"])
        self.assertTrue(contract["blocks_manuscript"])
        self.assertFalse(contract["quality_gates"]["manuscript_allowed"])
        self.assertIn("full_benchmark_completed=true", contract["reviewer_objections"][0])

    def test_unknown_benchmark_target_does_not_fallback_to_gsm8k(self):
        target = experiment_forge._normalize_benchmark_target({"name": "Spider"})

        self.assertEqual(target["name"], "Spider")
        self.assertEqual(target["hf_candidates"], [])
        self.assertEqual(target["hf_dataset"], "")
        self.assertFalse(target["generated_runner_supported"])
        self.assertIn("no concrete Hugging Face dataset id", target["generated_runner_blocker"])

    def test_known_harness_benchmark_is_not_promoted_by_an_hf_id(self):
        target = experiment_forge._normalize_benchmark_target(
            {
                "name": "MATH-500",
                "hf_dataset": "HuggingFaceH4/MATH-500",
                "task_type": "math_qa",
            }
        )

        self.assertFalse(target["generated_runner_supported"])
        self.assertIn("dedicated domain benchmark harness", target["generated_runner_blocker"])

    def test_real_benchmark_defaults_require_explicit_targets_or_real_datasets(self):
        with self.assertRaisesRegex(ValueError, "explicit benchmark_targets or real datasets"):
            experiment_forge._real_benchmark_defaults({"datasets": [], "benchmark_targets": []})

    def test_generated_runner_refuses_cross_domain_gsm8k_fallback(self):
        train_py = experiment_forge._real_llm_benchmark_train_py(
            method_name="DomainMethod",
            metric_name="accuracy",
            plan={
                "benchmark_targets": [
                    {
                        "name": "StrategyQA",
                        "hf_dataset": "tasksource/strategy-qa",
                        "split": "validation",
                        "task_type": "boolean_qa",
                    }
                ],
                "model_targets": [{"name": "TinyLlama", "hf_model": "TinyLlama/TinyLlama-1.1B-Chat-v1.0"}],
            },
        )

        self.assertNotIn("materialize GSM8K fallback", train_py)
        self.assertIn("refusing cross-domain GSM8K fallback", train_py)

    def test_restricted_benchmark_targets_add_executable_probe_and_defer_formal_targets(self):
        llm_contract = {
            "status": "resolved",
            "domain": "legal_nlp",
            "task_family": "legal_privilege_review",
            "benchmark_set_rationale": "TREC Legal privilege review and FOIA-Ex5 cover legal privilege and public-records exemption axes.",
            "candidate_benchmarks": [
                {
                    "name": "TREC 2010 Legal Track - Privilege Task",
                    "task_type": "legal_review",
                    "requires_harness": True,
                    "official_url": "https://trec-legal.umiacs.umd.edu/",
                    "literature_sources": [{"title": "TREC Legal Track", "year": 2010, "url": "https://trec.nist.gov/pubs/trec19/papers/LEGAL.OVERVIEW.pdf"}],
                },
                {
                    "name": "FOIA-Ex5-Privilege",
                    "task_type": "legal_review",
                    "requires_harness": True,
                    "official_url": "https://www.foia.gov/",
                    "literature_sources": [{"title": "FOIA Exemption 5 privilege review benchmark", "year": 2024, "url": "https://www.foia.gov/"}],
                },
            ],
            "required_baselines": [{"name": "Legal-BERT classifier"}],
            "primary_metric": {"name": "selective_risk", "direction": "lower"},
            "blockers": [],
        }
        with mock.patch("agents.benchmark_design_agent.call_llm_json_for_role", return_value=(llm_contract, 33, {})):
            plan = experiment_forge._ensure_real_benchmark_plan(
                {
                    "title": "FOIA privilege selector",
                    "problem_statement": "Privilege review needs a legal benchmark.",
                },
                {
                    "name": "PrivilegeSelector",
                    "definition": "Select releasable documents under privilege constraints.",
                },
                {
                    "datasets": [{"name": "TREC 2010 Legal Track - Privilege Task"}, {"name": "FOIA-Ex5-Privilege"}],
                    "baselines": [{"name": "Direct"}, {"name": "Always-CoT"}],
                    "metrics": {"primary": "cost_adjusted_accuracy"},
                },
                "gpu_large",
                llm_scope={
                    "agenda_id": 1,
                    "idea_id": 2,
                    "resource_grant_id": 3,
                    "stage": "experiment_forge",
                },
            )

        self.assertFalse(plan["generated_runner_supported"])
        self.assertNotIn("benchmark_design_blockers", plan)
        self.assertEqual(plan["benchmark_design_status"], "resolved")
        self.assertEqual(plan["benchmark_targets"][0]["name"], "TREC 2010 Legal Track - Privilege Task")
        self.assertTrue(plan["benchmark_targets"][0]["requires_harness"])
        self.assertIn("dedicated domain benchmark harness", plan["benchmark_recipe_blockers"][0]["reason"])
        self.assertNotIn("benchmark_probe_added", plan)

    def test_partial_benchmark_support_runs_supported_subset(self):
        plan = experiment_forge._ensure_real_benchmark_plan(
            {
                "title": "Code repair selector",
                "problem_statement": "Code repair needs executable and deferred proof benchmarks.",
            },
            {
                "name": "RepairSelector",
                "definition": "Select repairs using verifier feedback.",
            },
            {
                "benchmark_design_status": "resolved",
                "benchmark_design_contract": {"status": "resolved", "source": "test_reviewed_contract"},
                "benchmark_targets": [
                    {
                        "name": "MBPP",
                        "hf_dataset": "google-research-datasets/mbpp",
                        "split": "test",
                        "task_type": "code_generation",
                    },
                    {
                        "name": "HumanEval",
                        "hf_dataset": "openai/openai_humaneval",
                        "split": "test",
                        "task_type": "python_code_generation_with_unit_tests",
                    },
                ],
                "baselines": [{"name": "Direct"}],
                "metrics": {"primary": "pass_at_1"},
            },
            "gpu_large",
        )

        self.assertTrue(plan["generated_runner_supported"])
        self.assertEqual([row["name"] for row in plan["benchmark_targets"]], ["MBPP"])
        self.assertEqual(plan["benchmark_targets"][0]["benchmark_role"], "executable_probe")
        self.assertEqual(plan["deferred_benchmark_targets"], ["HumanEval"])
        self.assertTrue(plan["benchmark_harness_deferred"])
        self.assertEqual(plan["benchmark_execution"]["deferred_target_count"], 1)

    def test_generate_scaffold_accepts_evidence_plan(self):
        insight = {
            "proposed_method": {
                "name": "CGGR",
                "type": "hybrid",
                "one_line": "Route extra reasoning only when gain is positive.",
                "definition": "Estimate the counterfactual gain of more reasoning.",
            },
            "experimental_plan": {
                "baselines": ["baseline-a"],
                "datasets": ["StrategyQA"],
                "metrics": {"primary": "accuracy"},
                "expected_results": {"delta": "+2"},
            },
            "evidence_plan": {
                "main_table": {"enabled": True, "priority": "required"},
                "visualization": {"enabled": False, "priority": "skip"},
            },
            "problem_statement": "Decide when extra reasoning is useful.",
            "existing_weakness": "Always-on reasoning wastes budget.",
        }
        codebase = {
            "url": "scratch",
            "name": "minimal",
            "main_train_file": "train.py",
            "main_eval_command": "python evaluate.py",
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            workdir = Path(tmpdir)
            captured = {}

            def _fake_call_llm_json(system: str, prompt: str):
                captured["prompt"] = prompt
                return (
                    {
                        "program_md": "# program",
                        "evaluate_py": "print('ok')",
                        "success_criteria": {"metric_name": "accuracy"},
                    },
                    17,
                )

            with mock.patch.object(
                experiment_forge, "call_llm_json", side_effect=_fake_call_llm_json
            ):
                scaffold = experiment_forge.generate_scaffold(
                    insight, codebase, workdir
                )

        self.assertEqual(scaffold["tokens"], 17)
        self.assertIn("Adaptive Evidence Plan", captured["prompt"])
        self.assertIn("Honor this plan", captured["prompt"])
        self.assertIn("Publication Evidence Contract", captured["prompt"])
        self.assertIn("Benchmark Manifest", captured["prompt"])
        self.assertIn("Role: Experiment Contract Architect", experiment_forge.SCAFFOLD_SYSTEM)
        self.assertIn("Role: Full Benchmark Compiler", experiment_forge.SCAFFOLD_SYSTEM)
        self.assertIn("publication_evidence_contract", scaffold["success_criteria"])
        self.assertIn("benchmark_manifest", scaffold["success_criteria"]["publication_evidence_contract"])
        self.assertIn("claim_route", scaffold["success_criteria"]["publication_evidence_contract"])
        self.assertIn("claim_route", scaffold["benchmark_manifest"])
        self.assertIn("full_benchmark_stage", scaffold["benchmark_manifest"])
        self.assertIn("required_ablations", scaffold["success_criteria"])

    def test_generate_scaffold_injects_real_benchmark_runner_for_gpu_route(self):
        insight = {
            "resource_class": "gpu_large",
            "proposed_method": {
                "name": "Large GPU Method",
                "type": "training",
                "definition": "Train a large CUDA-backed model.",
            },
            "experimental_plan": {
                "baselines": ["baseline-a"],
                "datasets": ["StrategyQA"],
                "metrics": {"primary": "gpu_score"},
                "compute_budget": {"total_gpu_hours": 50},
            },
        }
        codebase = {
            "url": "https://github.com/example/gpu-repo",
            "name": "gpu-repo",
            "main_train_file": "train.py",
            "main_eval_command": "python train.py",
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            workdir = Path(tmpdir)

            def _fake_call_llm_json(system: str, prompt: str):
                return (
                    {
                        "program_md": "# program",
                        "evaluate_py": "print('ok')",
                        "success_criteria": {"metric_name": "gpu_score"},
                        "train_py": "import numpy as np\nprint('gpu_score: 0.1')\n",
                    },
                    17,
                )

            with mock.patch.object(
                experiment_forge, "call_llm_json", side_effect=_fake_call_llm_json
            ):
                scaffold = experiment_forge.generate_scaffold(
                    insight, codebase, workdir
                )

            train_py = (workdir / "code" / "train.py").read_text(encoding="utf-8")

        self.assertEqual(scaffold["baseline_command_override"], "python train.py")
        self.assertIn("torch.cuda.is_available", train_py)
        self.assertIn("load_dataset", train_py)
        self.assertIn("AutoModelForCausalLM", train_py)
        self.assertIn("DEFAULT_REPAIR_MAX_EXAMPLES_CAP", train_py)
        self.assertIn("DEEPGRAPH_BENCHMARK_FULL_RUN", train_py)
        self.assertIn("BENCHMARK_STAGE: eval_method_done", train_py)
        self.assertIn("_method_specs_for_run", train_py)
        self.assertIn("peak_vram_mb", train_py)
        self.assertNotEqual(scaffold["success_criteria"]["evidence_tier"], "bootstrap_probe")
        self.assertEqual(scaffold["success_criteria"]["evidence_tier"], "benchmark_plan")
        self.assertTrue(scaffold["success_criteria"]["blocks_manuscript"])
        self.assertIn("claim_route", scaffold["success_criteria"])
        self.assertFalse(scaffold["benchmark_manifest"]["sanity_only"])

    def test_generic_method_without_runner_plugin_gets_blocker(self):
        insight = {
            "resource_class": "gpu_large",
            "proposed_method": {
                "name": "Generic Candidate",
                "type": "reasoning",
                "definition": "Estimate counterfactual reasoning gain before spending extra inference budget.",
            },
            "experimental_plan": {
                "baselines": ["Direct"],
                "datasets": ["StrategyQA"],
                "metrics": {"primary": "cost_adjusted_accuracy"},
                "compute_budget": {"total_gpu_hours": 50},
            },
        }
        codebase = {
            "url": "scratch",
            "name": "minimal",
            "main_train_file": "train.py",
            "main_eval_command": "python train.py",
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            workdir = Path(tmpdir)

            def _fake_role_call(*args, **kwargs):
                return (
                    {
                        "program_md": "# program",
                        "evaluate_py": "print('ok')",
                        "success_criteria": {"metric_name": "cost_adjusted_accuracy"},
                        "train_py": "print('unused')\n",
                    },
                    17,
                    {
                        "provider": "test-provider",
                        "model": "test-model",
                        "model_family": "test-family",
                        "prompt_version": "test-v1",
                    },
                )

            with mock.patch.object(
                experiment_forge,
                "call_llm_json_for_role",
                side_effect=_fake_role_call,
            ):
                experiment_forge.generate_scaffold(
                    insight,
                    codebase,
                    workdir,
                    llm_scope={
                        "agenda_id": 1,
                        "idea_id": 2,
                        "resource_grant_id": 3,
                        "stage": "pilot",
                    },
                )

            train_py = (workdir / "code" / "train.py").read_text(encoding="utf-8")

        self.assertIn("no explicit audited runner_plugin", train_py)
        self.assertIn("full_benchmark_completed", train_py)
        self.assertIn("sys.exit(2)", train_py)

    def test_executable_probe_uses_real_gsm8k_runner_without_label_fallback(self):
        plan = {
            "harness_recovery_fresh_forge": True,
            "benchmark_targets": [
                {
                    "name": "GSM8K",
                    "hf_dataset": "openai/gsm8k",
                    "config": "main",
                    "split": "test",
                    "task_type": "math_qa",
                    "benchmark_role": "executable_probe",
                    "generated_runner_supported": True,
                }
            ],
            "model_targets": [
                {"hf_model": "Qwen/Qwen3.5-4B", "requires_cuda": True}
            ],
            "metrics": {"primary": "exact_match"},
            "minimum_seeds": 1,
            "procedure": "Score intermediate reasoning before choosing an answer.",
        }

        train_py = experiment_forge._real_llm_benchmark_train_py(
            method_name="Process Reward Probe",
            metric_name="exact_match",
            plan=plan,
        )

        compile(train_py, "train.py", "exec")
        self.assertIn("openai/gsm8k", train_py)
        self.assertIn('CANDIDATE_METHOD = "process_guided_candidate"', train_py)
        self.assertIn('"per_method": per_method', train_py)
        self.assertIn('"candidate_method": CANDIDATE_METHOD', train_py)
        self.assertIn('"label_fallback_used": False', train_py)
        self.assertIn('"full_benchmark_completed": False', train_py)
        self.assertIn("is_torchvision_available = lambda: False", train_py)
        self.assertIn('os.environ.setdefault("HF_HUB_DISABLE_XET", "1")', train_py)
        self.assertIn('os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "300")', train_py)
        self.assertIn('os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "60")', train_py)
        self.assertNotIn("prediction = target", train_py)
        self.assertNotIn("extract_gsm8k_answer(row[\"answer\"])", train_py)

        proxy = experiment_forge.build_proxy_config(plan)
        self.assertEqual(proxy["reproduction_iterations"], 1)
        self.assertEqual(proxy["max_iterations"], 0)
        self.assertEqual(proxy["refute_min_iterations"], 0)
        self.assertEqual(proxy["benchmark_seeds"], 1)

    def test_resolved_benchmark_design_is_reused_without_llm_redesign(self):
        contract = {
            "status": "resolved",
            "candidate_benchmarks": [
                {
                    "name": "GSM8K",
                    "hf_dataset": "openai/gsm8k",
                    "task_type": "math_qa",
                    "literature_sources": [{"title": "GSM8K"}],
                }
            ],
            "benchmark_evidence": [{"name": "GSM8K"}],
            "required_baseline_families": ["Direct", "Process-guided"],
            "primary_metric_candidates": ["exact_match"],
            "blockers": [],
            "warnings": [],
        }
        parsed = {
            "title": "Process reward probe",
            "problem_statement": "Test process-guided math reasoning.",
            "resource_class": "gpu_large",
            "experimental_plan": {
                "benchmark_design_status": "resolved",
                "benchmark_design_contract": contract,
                "benchmark_targets": contract["candidate_benchmarks"],
                "datasets": [{"name": "GSM8K"}],
                "metrics": {"primary": "exact_match"},
            },
        }
        method = {"name": "Process Reward Probe", "type": "reasoning"}

        with mock.patch.object(
            experiment_forge,
            "build_benchmark_design_contract",
            side_effect=AssertionError("resolved design must not be called again"),
        ):
            enriched = experiment_forge._enrich_experimental_plan(
                parsed,
                method,
                llm_scope={
                    "agenda_id": 11,
                    "idea_id": 105,
                    "resource_grant_id": 17,
                    "stage": "pilot",
                },
            )

        self.assertEqual(enriched["benchmark_design_status"], "resolved")
        self.assertEqual(enriched["benchmark_design_contract"], contract)

    def test_gpu_plan_drops_stale_bootstrap_models(self):
        plan = {
            "harness_recovery_fresh_forge": True,
            "benchmark_targets": [
                {
                    "name": "GSM8K",
                    "hf_dataset": "openai/gsm8k",
                    "task_type": "math_qa",
                    "generated_runner_supported": True,
                }
            ],
            "model_targets": [
                {"hf_model": "Qwen/Qwen2.5-0.5B-Instruct", "requires_cuda": True},
                {"hf_model": "TinyLlama/TinyLlama-1.1B-Chat-v1.0", "requires_cuda": True},
            ],
            "metrics": {"primary": "exact_match"},
        }
        parsed = {
            "title": "Math reasoning",
            # The historical row says CPU, but executable probes are scheduled
            # on GPU and therefore must not inherit CPU bootstrap models.
            "resource_class": "cpu",
            "proposed_method": {"name": "Candidate", "type": "reasoning"},
        }

        with mock.patch.object(
            experiment_forge,
            "EXPERIMENT_REAL_LLM_MODEL",
            "Qwen/Qwen3.5-4B",
        ):
            normalized = experiment_forge._ensure_real_benchmark_plan(
                parsed,
                parsed["proposed_method"],
                plan,
                "cpu",
            )

        models = [row["hf_model"] for row in normalized["model_targets"]]
        self.assertEqual(models[0], "Qwen/Qwen3.5-4B")
        self.assertFalse(any("Qwen2.5" in name or "TinyLlama" in name for name in models))

    def test_setup_workspace_falls_back_to_archive_when_git_missing(self):
        codebase = {
            "url": "https://github.com/example/project",
            "name": "project",
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            workroot = Path(tmpdir)

            def _fake_archive(url: str, code_dir: Path) -> bool:
                (code_dir / "train.py").write_text("print('hello')", encoding="utf-8")
                return True

            with (
                mock.patch.object(
                    experiment_forge,
                    "ensure_run_workspace",
                    return_value={
                        "run_root": workroot / "idea_7" / "experiments" / "main" / "runs" / "run_70",
                        "code_root": workroot / "idea_7" / "experiments" / "main" / "runs" / "run_70" / "code",
                        "results_root": workroot / "idea_7" / "experiments" / "main" / "runs" / "run_70" / "results",
                        "spec_root": workroot / "idea_7" / "experiments" / "main" / "runs" / "run_70" / "spec",
                        "codex_root": workroot / "idea_7" / "experiments" / "main" / "runs" / "run_70" / "codex",
                    },
                ),
                mock.patch.object(experiment_forge, "_git_binary", return_value=None),
                mock.patch.object(
                    experiment_forge, "_download_repo_archive", side_effect=_fake_archive
                ) as download_archive,
            ):
                workdir = experiment_forge.setup_workspace(7, 70, codebase)
                self.assertTrue((workdir / "code" / "train.py").exists())
                self.assertTrue((workdir / "spec").exists())
                self.assertTrue((workdir / "codex").exists())
                download_archive.assert_called_once()

    def test_build_proxy_config_carries_repo_execution_hints(self):
        proxy = experiment_forge.build_proxy_config(
            {"compute_budget": {"total_gpu_hours": 12}},
            codebase={
                "main_train_file": "src/qa/inference.py",
                "main_eval_command": "python src/qa/inference.py --dataset strategyqa",
            },
        )

        self.assertEqual(proxy["main_train_file"], "src/qa/inference.py")
        self.assertEqual(
            proxy["baseline_command"],
            "python src/qa/inference.py --dataset strategyqa",
        )
        self.assertEqual(proxy["estimated_gpu_hours"], 12)
        self.assertIn("budget_policy", proxy)
        self.assertEqual(proxy["budget_policy"]["estimated_gpu_hours"], 12)
        self.assertIn("benchmark_model", proxy)
        self.assertIn("benchmark_seeds", proxy)
        self.assertIn("benchmark_max_examples_per_seed", proxy)

    def test_normalize_codebase_metadata_clears_placeholder_entrypoint_for_real_repo(self):
        normalized = experiment_forge._normalize_codebase_metadata(
            {
                "url": "https://github.com/example/project",
                "name": "project",
                "main_train_file": "scratch",
                "main_eval_command": "unknown",
            }
        )

        self.assertEqual(normalized["main_train_file"], "")
        self.assertEqual(normalized["main_eval_command"], "")

    def test_checkpoint_run_state_serializes_incremental_fields(self):
        with (
            mock.patch.object(experiment_forge.db, "execute") as execute,
            mock.patch.object(experiment_forge.db, "commit") as commit,
        ):
            experiment_forge._checkpoint_run_state(
                42,
                phase="review_decision_ready",
                workdir="/tmp/run_42",
                codebase={"url": "https://github.com/example/project", "name": "project"},
                proxy_config={"formal_experiment": True, "smoke_test_only": False},
                baseline_metric_name="accuracy",
            )

        sql, params = execute.call_args.args
        self.assertIn("phase=?", sql)
        self.assertEqual(params[-1], 42)
        self.assertIn("review_decision_ready", params)
        self.assertIn("/tmp/run_42", params)
        self.assertTrue(any("formal_experiment" in str(value) for value in params))
        commit.assert_called_once()

    def test_fallback_scaffold_produces_real_benchmark_train_py(self):
        scaffold = experiment_forge._fallback_scaffold(
            {"name": "CGGR", "definition": "Adaptive reasoning gate."},
            {
                "benchmark_design_status": "resolved",
                "benchmark_design_contract": {"status": "resolved", "source": "unit_test"},
                "benchmark_targets": [
                    {
                        "name": "StrategyQA",
                        "hf_dataset": "tasksource/strategy-qa",
                        "split": "validation",
                        "task_type": "boolean_qa",
                    }
                ],
                "metrics": {"primary": "cost_adjusted_utility"},
            },
            {"url": "scratch", "name": "minimal"},
        )

        self.assertIn("train_py", scaffold)
        self.assertIn("load_dataset", scaffold["train_py"])
        self.assertIn("AutoModelForCausalLM", scaffold["train_py"])
        self.assertIn("DEEPGRAPH_BENCHMARK_TARGET_NAMES", scaffold["train_py"])
        self.assertIn("DEEPGRAPH_BENCHMARK_SEED_OFFSET", scaffold["train_py"])
        self.assertIn("DEEPGRAPH_BENCHMARK_SEED_COUNT", scaffold["train_py"])
        self.assertIn('"sharded_run": sharded_run', scaffold["train_py"])
        self.assertIn('_difficulty_proxy(question_for_prompt, target.get("task_type") or "qa")', scaffold["train_py"])
        self.assertIn('score = max(score, 0.46)', scaffold["train_py"])
        self.assertIn('max_new_tokens <= 80', scaffold["train_py"])
        self.assertIn('prompt_kind = "direct" if kind == "direct" or (selective_kind and max_new_tokens <= 80) else kind', scaffold["train_py"])
        self.assertIn("apply_chat_template", scaffold["train_py"])
        self.assertIn("return_dict=True", scaffold["train_py"])
        self.assertIn("_coerce_tokenizer_encoding", scaffold["train_py"])
        self.assertIn("LLM generation returned zero new tokens", scaffold["train_py"])
        self.assertIn("DEEPGRAPH_BENCHMARK_CONTINUE_ON_ERROR", scaffold["train_py"])
        self.assertIn("use at most two concise reasoning sentences", scaffold["train_py"])
        self.assertIn("Do not repeat the final answer", scaffold["train_py"])
        self.assertNotIn("materialize GSM8K fallback", scaffold["train_py"])
        self.assertNotEqual(scaffold["success_criteria"]["evidence_tier"], "bootstrap_probe")
        self.assertTrue(scaffold["success_criteria"]["blocks_manuscript"])

    def test_codebase_entrypoint_check_requires_expected_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            code_dir = Path(tmpdir)
            (code_dir / "README.md").write_text("repo", encoding="utf-8")

            self.assertFalse(
                experiment_forge._codebase_has_expected_entrypoint(
                    code_dir, {"main_train_file": "src/qa/inference.py"}
                )
            )

    def test_review_blocks_formal_run_when_required_benchmark_is_deferred(self):
        judgement = review_experiment_candidate(
            {
                "id": 91,
                "tier": 2,
                "title": "PRM Bellman",
                "resource_class": "gpu_large",
                "proposed_method": {"name": "PRM Bellman", "definition": "Use process rewards over reasoning traces."},
                "experimental_plan": {
                    "benchmark_design_status": "resolved",
                    "benchmark_design_contract": {
                        "status": "resolved",
                        "minimum_benchmark_count": 2,
                        "benchmark_set_rationale": "GSM8K covers answer accuracy and PRM800K covers process supervision.",
                        "candidate_benchmarks": [
                            {"name": "GSM8K", "hf_dataset": "openai/gsm8k", "literature_sources": [{"title": "GSM8K"}]},
                            {"name": "PRM800K", "requires_harness": True, "literature_sources": [{"title": "PRM800K"}]},
                        ],
                        "benchmark_evidence": [{"name": "GSM8K"}, {"name": "PRM800K"}],
                    },
                    "baselines": [{"name": "Direct"}, {"name": "CoT"}],
                    "datasets": [{"name": "GSM8K"}],
                    "benchmark_targets": [{"name": "GSM8K", "hf_dataset": "openai/gsm8k"}],
                    "model_targets": [{"name": "Qwen/Qwen2.5-7B-Instruct"}],
                    "metrics": {"primary": "exact_match"},
                    "compute_budget": {"total_gpu_hours": 12},
                    "generated_runner_supported": True,
                    "benchmark_harness_deferred": True,
                    "deferred_benchmark_targets": ["PRM800K"],
                },
            },
            codebase={"url": "scratch", "name": "minimal"},
            entrypoint_available=False,
        )

        self.assertEqual(judgement.recommended_route, "blocked")
        self.assertTrue(judgement.environment_review["benchmark_harness_required"])
        self.assertIn("PRM800K", " ".join(judgement.blockers))


    def test_review_routes_generated_real_benchmark_runner_to_formal(self):
        judgement = review_experiment_candidate(
            {
                "id": 9,
                "tier": 2,
                "title": "CGGR",
                "resource_class": "gpu_large",
                "proposed_method": {"name": "CGGR", "definition": "Adaptive gate."},
                "experimental_plan": {
                    "baselines": [{"name": "A"}, {"name": "B"}],
                    "datasets": [{"name": "GSM8K"}],
                    "model_targets": [{"name": "Qwen/Qwen2.5-7B-Instruct"}],
                    "metrics": {"primary": "accuracy"},
                    "compute_budget": {"total_gpu_hours": 12},
                },
            },
            codebase={"url": "scratch", "name": "minimal"},
            entrypoint_available=False,
        )

        self.assertFalse(judgement.smoke_test_only)
        self.assertTrue(judgement.formal_experiment)


if __name__ == "__main__":
    unittest.main()
