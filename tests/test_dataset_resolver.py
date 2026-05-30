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
            dataset_id="example/spider-text-to-sql",
            score=0.72,
            downloads=100,
            likes=5,
            tags=("text2text-generation", "question-answering"),
        )
        with mock.patch.object(dataset_resolver, "search_huggingface_datasets", return_value=[candidate]):
            recipe = dataset_resolver.resolve_dataset_name("Spider")

        self.assertTrue(recipe["resolved"])
        self.assertEqual(recipe["hf_dataset"], "example/spider-text-to-sql")
        self.assertEqual(recipe["source"], "huggingface_search")
        self.assertTrue(recipe["generated_runner_supported"])

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
                    "datasets": [{"name": "Spider"}],
                    "generated_runner_supported": False,
                    "benchmark_recipe_blockers": [{"name": "Spider", "reason": "missing recipe"}],
                }
            )

        self.assertTrue(plan["generated_runner_supported"])
        self.assertEqual(plan["dataset_resolution"]["status"], "resolved")
        self.assertEqual(plan["benchmark_targets"][0]["hf_dataset"], "example/spider-text-to-sql")
        self.assertNotIn("benchmark_recipe_blockers", plan)

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
                    "datasets": [{"name": "Spider"}],
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
            "experimental_plan": {
                "datasets": [{"name": "GSM8K"}],
                "generated_runner_supported": False,
                "benchmark_recipe_blockers": [{"name": "GSM8K", "reason": "missing recipe"}],
            },
        }

        updated = run_deepgraph_new_idea_once._resolve_idea_datasets(idea)
        self.assertIsNone(run_deepgraph_new_idea_once._execution_blocker(updated))


if __name__ == "__main__":
    unittest.main()
