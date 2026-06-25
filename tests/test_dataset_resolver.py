import sys
import unittest
from unittest import mock

from agents import dataset_resolver
from agents import experiment_forge
from scripts import run_deepgraph_new_idea_once


class DatasetResolverTests(unittest.TestCase):
    def test_known_public_dataset_resolves_without_network(self):
        recipe = dataset_resolver.resolve_dataset_name("GSM8K")

        self.assertTrue(recipe["resolved"])
        self.assertEqual(recipe["hf_dataset"], "openai/gsm8k")
        self.assertEqual(recipe["source"], "local_registry")
        self.assertIn("test", recipe["split"])

    def test_restricted_dataset_stays_unresolved(self):
        recipe = dataset_resolver.resolve_dataset_name("TREC 2010 Legal Track - Privilege Task")

        self.assertFalse(recipe["resolved"])
        self.assertIn("restricted", recipe["reason"])

    def test_huggingface_search_can_resolve_unknown_dataset(self):
        candidate = dataset_resolver.DatasetCandidate(
            dataset_id="example/custom-qa",
            score=0.72,
            downloads=100,
            likes=5,
            tags=("text2text-generation", "question-answering"),
        )
        with mock.patch.object(dataset_resolver, "search_huggingface_datasets", return_value=[candidate]):
            recipe = dataset_resolver.resolve_dataset_name("CustomQA")

        self.assertTrue(recipe["resolved"])
        self.assertEqual(recipe["hf_dataset"], "example/custom-qa")
        self.assertEqual(recipe["source"], "huggingface_search")
        self.assertTrue(recipe["generated_runner_supported"])

    def test_spider_uses_official_harness_recipe_not_name_search(self):
        with mock.patch.object(dataset_resolver, "search_huggingface_datasets") as search:
            recipe = dataset_resolver.resolve_dataset_name("Spider")

        search.assert_not_called()
        self.assertTrue(recipe["resolved"])
        self.assertEqual(recipe["name"], "Spider")
        self.assertTrue(recipe["requires_harness"])
        self.assertFalse(recipe["generated_runner_supported"])
        self.assertIn("yale-lily", recipe["official_url"])

    def test_longmemeval_uses_harness_recipe_not_generic_runner(self):
        recipe = dataset_resolver.resolve_known_dataset_recipe("LongMemEval")

        self.assertIsNotNone(recipe)
        self.assertEqual(recipe["name"], "LongMemEval")
        self.assertTrue(recipe["requires_harness"])
        self.assertFalse(recipe["generated_runner_supported"])
        self.assertIn("longmemeval", recipe["official_url"].lower())


    def test_plan_resolution_updates_runner_gate(self):
        candidate = dataset_resolver.DatasetCandidate(
            dataset_id="example/spider-text-to-sql",
            score=0.72,
            downloads=100,
            likes=5,
            tags=("question-answering",),
        )
        with mock.patch.object(dataset_resolver, "search_huggingface_datasets", return_value=[candidate]):
            plan = dataset_resolver.resolve_plan_datasets(
                {
                    "datasets": [{"name": "CustomQA"}],
                    "generated_runner_supported": False,
                    "benchmark_recipe_blockers": [{"name": "CustomQA", "reason": "missing recipe"}],
                }
            )

        self.assertTrue(plan["generated_runner_supported"])
        self.assertEqual(plan["dataset_resolution"]["status"], "resolved")
        self.assertEqual(plan["benchmark_targets"][0]["hf_dataset"], "example/spider-text-to-sql")
        self.assertNotIn("benchmark_recipe_blockers", plan)

    def test_literature_selected_benchmark_does_not_hf_search_by_name(self):
        with mock.patch.object(dataset_resolver, "search_huggingface_datasets") as search:
            plan = dataset_resolver.resolve_plan_datasets(
                {
                    "benchmark_targets": [
                        {
                            "name": "Robomimic",
                            "task_type": "robot_manipulation",
                            "requires_harness": True,
                            "dataset_selection_source": "llm_literature_design",
                            "resolver_policy": "validate_only",
                        }
                    ],
                    "datasets": [{"name": "Robomimic"}],
                    "generated_runner_supported": False,
                }
            )

        search.assert_not_called()
        self.assertFalse(plan["generated_runner_supported"])
        self.assertEqual(plan["dataset_resolution"]["status"], "unresolved")
        self.assertIn("resolver search is disabled", plan["benchmark_recipe_blockers"][0]["reason"])

    def test_partial_plan_resolution_keeps_deferred_targets(self):
        plan = dataset_resolver.resolve_plan_datasets(
            {
                "datasets": [
                    {"name": "GSM8K"},
                    {"name": "TREC 2010 Legal Track - Privilege Task"},
                ],
                "generated_runner_supported": False,
                "benchmark_recipe_blockers": [
                    {"name": "GSM8K", "reason": "missing recipe"},
                    {"name": "TREC 2010 Legal Track - Privilege Task", "reason": "restricted"},
                ],
            }
        )

        self.assertTrue(plan["generated_runner_supported"])
        self.assertEqual(plan["dataset_resolution"]["status"], "partial")
        self.assertEqual(plan["benchmark_targets"][0]["hf_dataset"], "openai/gsm8k")
        self.assertEqual(plan["deferred_benchmark_targets"], ["TREC 2010 Legal Track - Privilege Task"])
        self.assertTrue(plan["benchmark_harness_deferred"])
        self.assertEqual(
            [row["name"] for row in plan["benchmark_recipe_blockers"]],
            ["TREC 2010 Legal Track - Privilege Task"],
        )

    def test_forge_resolves_before_blocking(self):
        candidate = dataset_resolver.DatasetCandidate(
            dataset_id="example/spider-text-to-sql",
            score=0.72,
            downloads=100,
            likes=5,
            tags=("question-answering",),
        )
        with mock.patch.object(dataset_resolver, "search_huggingface_datasets", return_value=[candidate]):
            plan = experiment_forge._ensure_real_benchmark_plan(
                {
                    "title": "Text-to-SQL routing",
                    "problem_statement": "Selectors need executable SQL benchmarks.",
                },
                {
                    "name": "Selector",
                    "definition": "Route candidate answers using confidence and schema evidence.",
                },
                {
                    "benchmark_design_status": "resolved",
                    "benchmark_design_contract": {"status": "resolved", "source": "test_reviewed_contract"},
                    "datasets": [{"name": "CustomQA"}],
                    "baselines": [{"name": "Direct"}, {"name": "Majority"}],
                    "metrics": {"primary": "accuracy"},
                },
                "cpu",
            )

        self.assertTrue(plan["generated_runner_supported"])
        self.assertEqual(plan["benchmark_targets"][0]["hf_dataset"], "example/spider-text-to-sql")

    def test_run_script_resolves_before_execution_blocker(self):
        idea = {
            "title": "Executable QA idea",
            "proposed_method": {"name": "MathQA", "definition": "Answer grade-school math word problems."},
            "experimental_plan": {
                "benchmark_design_status": "resolved",
                "benchmark_design_contract": {"status": "resolved", "source": "test_reviewed_contract"},
                "datasets": [{"name": "GSM8K"}],
                "generated_runner_supported": False,
                "benchmark_recipe_blockers": [{"name": "GSM8K", "reason": "missing recipe"}],
            },
        }

        updated = run_deepgraph_new_idea_once._resolve_idea_datasets(idea)
        self.assertIsNone(run_deepgraph_new_idea_once._execution_blocker(updated))

    def test_run_script_stores_unexecutable_recipe_and_routes_to_harness(self):
        idea = {
            "title": "Needs harness",
            "experimental_plan": {
                "datasets": [{"name": "CustomBench"}],
                "generated_runner_supported": False,
                "benchmark_recipe_blockers": [{"name": "CustomBench", "reason": "missing recipe"}],
            },
        }

        with (
            mock.patch.object(sys, "argv", ["run_deepgraph_new_idea_once.py", "--skip-harvest", "--store-limit", "1"]),
            mock.patch.object(run_deepgraph_new_idea_once, "discover_paper_ideas", return_value=[idea]),
            mock.patch.object(run_deepgraph_new_idea_once, "_resolve_idea_datasets", side_effect=lambda row: row),
            mock.patch.object(run_deepgraph_new_idea_once, "_is_cggr_related", return_value=False),
            mock.patch.object(run_deepgraph_new_idea_once, "detect_compute_profile", return_value=object()),
            mock.patch.object(run_deepgraph_new_idea_once, "gpu_resource_allowed", return_value=(True, "ok")),
            mock.patch.object(run_deepgraph_new_idea_once, "store_deep_insight", return_value=42) as store,
            mock.patch.object(run_deepgraph_new_idea_once.auto_research, "_handle_experiment_review_blocked") as route,
            mock.patch.object(run_deepgraph_new_idea_once.auto_research, "_process_candidate") as process,
            mock.patch.object(run_deepgraph_new_idea_once.db, "fetchone", return_value={"id": 42, "title": "Needs harness"}),
            mock.patch.object(run_deepgraph_new_idea_once.db, "fetchall", return_value=[]),
        ):
            rc = run_deepgraph_new_idea_once.main()

        self.assertEqual(rc, 0)
        store.assert_called_once_with(idea)
        route.assert_called_once()
        self.assertEqual(route.call_args.args[0], 42)
        self.assertEqual(route.call_args.kwargs["source"], "idea_generation")
        self.assertEqual(route.call_args.args[1]["error"], "no_executable_benchmark_recipe")
        process.assert_not_called()


if __name__ == "__main__":
    unittest.main()
