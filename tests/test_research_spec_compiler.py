import json
import unittest

from tests.temp_utils import temporary_workdir


class ResearchSpecCompilerTests(unittest.TestCase):
    def test_fairness_title_compiles_to_fairness_classification_spec(self):
        from agents.research_spec_compiler import compile_research_spec

        insight = {
            "id": 1,
            "title": "Group fairness via constrained preference optimization",
            "tier": 1,
            "experimental_plan": json.dumps({
                "metrics": {"primary": "fairness_score", "direction": "higher"}
            }),
            "formal_structure": "Preference constraints over grouped outcomes.",
        }

        spec = compile_research_spec(insight, run_id=7)

        self.assertEqual(spec["domain"], "fairness")
        self.assertEqual(spec["task_type"], "classification")
        self.assertIn("fairness_classification", spec["candidate_capabilities"])
        self.assertIn("fairness_score", spec["primary_metrics"])
        self.assertEqual(spec["run_id"], 7)
        self.assertEqual(spec["insight_id"], 1)

    def test_safe_rl_title_compiles_to_safe_rl_spec(self):
        from agents.research_spec_compiler import compile_research_spec

        insight = {
            "id": 2,
            "title": "Preference-cone CMDP policy optimization for safe RL",
            "tier": 1,
            "experimental_plan": json.dumps({
                "metrics": {"primary": "constraint_violation", "direction": "lower"}
            }),
        }

        spec = compile_research_spec(insight, run_id=8)

        self.assertEqual(spec["domain"], "safe_rl")
        self.assertEqual(spec["task_type"], "rl")
        self.assertIn("safe_rl_cmdp", spec["candidate_capabilities"])
        self.assertIn("constraint_violation", spec["primary_metrics"])

    def test_cmdp_safe_rl_terms_take_precedence_over_fairness_terms(self):
        from agents.research_spec_compiler import compile_research_spec

        insight = {
            "id": 22,
            "title": "Social-choice CMDP policy optimization for safe RL and fairness",
            "tier": 1,
            "formal_structure": "Safe RL and fairness share constrained occupancy measures.",
            "experimental_plan": "",
        }

        spec = compile_research_spec(insight, run_id=22)

        self.assertEqual(spec["domain"], "safe_rl")
        self.assertEqual(spec["task_type"], "rl")
        self.assertEqual(spec["primary_metrics"], ["safe_return"])
        self.assertIn("safe_rl_cmdp", spec["candidate_capabilities"])

    def test_unknown_title_falls_back_to_generic_python_spec(self):
        from agents.research_spec_compiler import compile_research_spec

        insight = {
            "id": 3,
            "title": "A strange new optimization heuristic",
            "tier": 1,
            "experimental_plan": "",
        }

        spec = compile_research_spec(insight, run_id=9)

        self.assertEqual(spec["domain"], "unknown")
        self.assertEqual(spec["task_type"], "benchmark")
        self.assertEqual(spec["candidate_capabilities"], ["generic_python_benchmark"])

    def test_spec_is_written_and_recorded_as_artifact(self):
        from agents.artifact_manager import list_artifacts
        from agents.research_spec_compiler import compile_and_write_research_spec

        insight = {"id": 4, "title": "Fair classification", "tier": 2}

        with temporary_workdir() as workdir:
            spec = compile_and_write_research_spec(insight, workdir, run_id=10)

            spec_path = workdir / "research_spec.json"
            self.assertTrue(spec_path.exists())
            self.assertEqual(spec["run_id"], 10)
            self.assertIn("research_spec.json", {item["path"] for item in list_artifacts(workdir)})


if __name__ == "__main__":
    unittest.main()
