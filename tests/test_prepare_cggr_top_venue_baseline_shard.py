import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from prepare_cggr_top_venue_baseline_shard import TOP_VENUE_METHODS, prepare


class PrepareCggrTopVenueBaselineShardTests(unittest.TestCase):
    def _write_source_run(self, root: Path) -> Path:
        source = root / "run_45"
        spec = source / "spec"
        spec.mkdir(parents=True)
        experiment_spec = {
            "resource_class": "gpu_large",
            "proposed_method": {
                "name": "Counterfactual Gain Gated Reasoning (CGGR)",
                "definition": "Estimate counterfactual gain before spending reasoning budget.",
            },
            "experimental_plan": {
                "baselines": [
                    {"name": "Vanilla Direct Answering"},
                    {"name": "Oracle Routing Upper Bound"},
                ],
                "datasets": [{"name": "MuSiQue-Ans"}],
                "model_targets": [{"name": "Qwen/Qwen2.5-14B-Instruct"}],
                "metrics": {"primary": "cost_adjusted_accuracy"},
                "publication_evidence_contract": {
                    "claim_to_validate": (
                        "CGGR learns a per-input stopping and routing policy by directly estimating the "
                        "counterfactual utility gain of additional reasoning relative to immediate answering, "
                        "and only allocates extra budget when a lower confidence bound on gain is positive."
                    ),
                    "required_baselines": [
                        "Vanilla Direct Answering",
                        "Oracle Routing Upper Bound",
                    ],
                    "paper_intent": {
                        "central_claim": (
                            "CGGR learns a per-input stopping and routing policy by directly estimating the "
                            "counterfactual utility gain of additional reasoning relative to immediate answering, "
                            "and only allocates extra budget when a lower confidence bound on gain is positive."
                        )
                    },
                },
                "paper_intent": {
                    "central_claim": (
                        "CGGR learns a per-input stopping and routing policy by directly estimating the "
                        "counterfactual utility gain of additional reasoning relative to immediate answering, "
                        "and only allocates extra budget when a lower confidence bound on gain is positive."
                    )
                },
            },
        }
        benchmark_manifest = {
            "full_benchmark_stage": {
                "datasets": [
                    "MuSiQue-Ans",
                    "StrategyQA",
                    "2WikiMultihopQA",
                    "Stress Test Split: Simple-vs-Hard Counterfactual Partition",
                ],
                "models": ["Qwen/Qwen2.5-14B-Instruct"],
                "baselines": ["Vanilla Direct Answering", "Oracle Routing Upper Bound"],
            }
        }
        (spec / "experiment_spec.json").write_text(json.dumps(experiment_spec), encoding="utf-8")
        (spec / "benchmark_manifest.json").write_text(json.dumps(benchmark_manifest), encoding="utf-8")
        (spec / "success_criteria.json").write_text("{}", encoding="utf-8")
        return source

    def test_prepare_writes_executable_top_venue_shard_template(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source = self._write_source_run(root)
            out = root / "top_venue_shard"

            result = prepare(source, out)

            self.assertTrue(result["ok"], result)
            train_py = (out / "code" / "train.py").read_text(encoding="utf-8")
            shard_config = json.loads((out / "spec" / "shard_config.json").read_text(encoding="utf-8"))
            plan = json.loads((out / "spec" / "top_venue_baseline_plan.json").read_text(encoding="utf-8"))
            for method in TOP_VENUE_METHODS:
                self.assertIn(method, train_py)
                self.assertIn(method, shard_config["method_subset"])
                self.assertIn(method, [row["name"] for row in plan["baselines"]])
            self.assertEqual(shard_config["env"]["DEEPGRAPH_BENCHMARK_INCLUDE_TOP_VENUE_BASELINES"], "1")
            self.assertEqual(plan["minimum_seeds"], 5)
            self.assertEqual(plan["max_eval_examples"], 128)
            serialized_plan = json.dumps(plan)
            self.assertNotIn("CGGR learns a per-input stopping and routing policy", serialized_plan)
            self.assertIn("fixed proxy-gated", plan["publication_evidence_contract"]["claim_to_validate"])
            self.assertFalse(plan["publication_evidence_contract"]["trained_estimator_claim_allowed"])
            self.assertFalse(plan["publication_evidence_contract"]["learned_router_claim_allowed"])
            self.assertFalse(plan["publication_evidence_contract"]["oracle_router_in_active_contract"])
            self.assertNotIn(
                "Oracle Routing Upper Bound",
                [row["name"] for row in plan["baselines"]],
            )
            self.assertNotIn(
                "Oracle Routing Upper Bound",
                plan["publication_evidence_contract"]["required_baselines"],
            )
            for method in TOP_VENUE_METHODS:
                self.assertIn(
                    method,
                    plan["publication_evidence_contract"]["required_top_venue_baselines"],
                )
                self.assertIn(method, plan["publication_evidence_contract"]["required_baselines"])


if __name__ == "__main__":
    unittest.main()
