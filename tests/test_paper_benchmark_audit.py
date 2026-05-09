import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from audit_paper_benchmark_artifacts import audit


REQUIRED_DATASETS = [
    "MuSiQue-Ans",
    "StrategyQA",
    "2WikiMultihopQA",
    "Stress Test Split: Simple-vs-Hard Counterfactual Partition",
]

REQUIRED_METHODS = [
    "Vanilla Direct Answering",
    "Always-Reason Chain-of-Thought",
    "Self-Consistency Reasoning",
    "Least-to-Most Prompting",
    "Confidence Gate",
    "Disagreement Routing",
    "Random Budget-Matched Routing",
    "CGGR",
]

REQUIRED_ABLATIONS = [
    "no_counterfactual_delta",
    "no_lcb",
    "no_self_divergence_penalty",
    "no_qstruct_term",
]


class PaperBenchmarkAuditTests(unittest.TestCase):
    def _write_package(self, workdir: Path, *, sharded: bool) -> None:
        results = workdir / "results"
        results.mkdir(parents=True)
        per_method = {method: {"metric_value": 0.5, "score": 0.6} for method in REQUIRED_METHODS}
        datasets = [{"name": name, "id": name, "split": "test"} for name in REQUIRED_DATASETS]
        summary = {
            "full_benchmark_completed": True,
            "num_seeds": 5,
            "datasets": datasets,
            "per_method": per_method,
            "per_method_std": {method: 0.01 for method in REQUIRED_METHODS},
            "ablation_table": [{"ablation": name, "metric_value": 0.4} for name in REQUIRED_ABLATIONS],
            "budget": {"seeds": 5, "max_examples_per_dataset_seed": 128},
            "load_failures": [],
        }
        run_config = {
            "targets": [{"name": name} for name in REQUIRED_DATASETS],
            "methods": REQUIRED_METHODS,
            "ablations": REQUIRED_ABLATIONS,
            "seeds": 5,
            "seed_values": [0, 1, 2, 3, 4],
            "max_examples_per_dataset_seed": 128,
            "sharded_run": sharded,
            "shard_axes": {"method": sharded, "target": False, "seed": False},
        }
        manifest = {
            "full_benchmark_completed": True,
            "datasets": datasets,
            "methods": REQUIRED_METHODS,
        }
        json_files = {
            "benchmark_summary.json": summary,
            "run_config.json": run_config,
            "artifact_manifest.json": manifest,
            "per_seed_results.json": [{"seed": seed} for seed in range(5)],
            "per_dataset_results.json": {name: {} for name in REQUIRED_DATASETS},
            "main_results_table.json": per_method,
            "cost_utility_tradeoff_table.json": [],
            "ablation_table.json": summary["ablation_table"],
            "latency_tokens_table.json": [{"method": "CGGR"}],
            "difficulty_breakdown_table.json": [
                {"method": "Vanilla Direct Answering", "difficulty": "simple", "accuracy": 0.5, "count": 128},
                {"method": "CGGR", "difficulty": "simple", "accuracy": 0.6, "route_rate": 0.5, "count": 128},
            ],
            "routing_analysis.json": {"CGGR": {}},
            "simple_case_degradation.json": {
                "baseline_accuracy": 0.5,
                "candidate_accuracy": 0.6,
                "degradation": 0.1,
                "candidate_route_rate": 0.5,
            },
            "calibration_reliability.json": [{"difficulty_bucket": "simple", "observed_gain_vs_direct": 0.1}],
            "bootstrap_ci.json": {
                "candidate_method": "CGGR",
                "baseline_method": "Always-Reason Chain-of-Thought",
                "candidate_ci95": [0.5, 0.7],
                "baseline_ci95": [0.4, 0.6],
                "paired_permutation_p": 0.5,
            },
            "environment_report.json": {"python": sys.version, "packages": {}},
        }
        for name, payload in json_files.items():
            (results / name).write_text(json.dumps(payload), encoding="utf-8")
        raw_lines = []
        routing_lines = []
        for seed in range(5):
            for dataset in REQUIRED_DATASETS:
                for method in REQUIRED_METHODS:
                    for idx in range(128):
                        example_id = f"{dataset}-{seed}-{idx}"
                        raw_lines.append(
                            json.dumps(
                                {
                                    "seed": seed,
                                    "dataset": dataset,
                                    "method": method,
                                    "example_id": example_id,
                                    "prediction": "answer",
                                    "new_tokens": 4,
                                }
                            )
                        )
                        if method == "CGGR":
                            routing_lines.append(
                                json.dumps(
                                    {
                                        "seed": seed,
                                        "dataset": dataset,
                                        "method": method,
                                        "example_id": example_id,
                                        "max_new_tokens": 64,
                                        "routed_to_deliberation": idx % 2 == 0,
                                    }
                                )
                            )
        (results / "raw_predictions.jsonl").write_text("\n".join(raw_lines) + "\n", encoding="utf-8")
        (results / "routing_decisions.jsonl").write_text("\n".join(routing_lines) + "\n", encoding="utf-8")
        (results / "failure_cases.jsonl").write_text("", encoding="utf-8")

    def test_require_full_blocks_sharded_run_from_run_config(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            workdir = Path(tmpdir)
            self._write_package(workdir, sharded=True)

            result = audit(workdir, require_full=True)

        self.assertFalse(result["ok"])
        self.assertIn("sharded_run artifact cannot satisfy require-full gate by itself", result["blockers"])

    def test_empty_failure_cases_file_is_valid_when_no_failures(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            workdir = Path(tmpdir)
            self._write_package(workdir, sharded=False)

            result = audit(workdir, require_full=True)

        self.assertTrue(result["ok"], result)
        self.assertEqual(result["blockers"], [])

    def test_cggr_zero_routing_is_a_blocker(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            workdir = Path(tmpdir)
            self._write_package(workdir, sharded=False)
            results = workdir / "results"
            raw_lines = []
            routing_lines = []
            for idx in range(20):
                row = {
                    "seed": 0,
                    "dataset": "MuSiQue-Ans",
                    "method": "CGGR",
                    "example_id": f"ex-{idx}",
                    "new_tokens": 64,
                }
                raw_lines.append(json.dumps(row))
                routing_lines.append(
                    json.dumps(
                        {
                            "seed": 0,
                            "dataset": "MuSiQue-Ans",
                            "method": "CGGR",
                            "example_id": f"ex-{idx}",
                            "max_new_tokens": 64,
                            "routed_to_deliberation": False,
                        }
                    )
                )
            (results / "raw_predictions.jsonl").write_text("\n".join(raw_lines) + "\n", encoding="utf-8")
            (results / "routing_decisions.jsonl").write_text("\n".join(routing_lines) + "\n", encoding="utf-8")

            result = audit(workdir, require_full=True)

        self.assertFalse(result["ok"])
        self.assertIn("CGGR routing collapsed to zero deliberation for MuSiQue-Ans seed=0", result["blockers"])

    def test_generation_failure_cases_are_blockers(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            workdir = Path(tmpdir)
            self._write_package(workdir, sharded=False)
            results = workdir / "results"
            (results / "failure_cases.jsonl").write_text(
                json.dumps(
                    {
                        "stage": "generation_or_scoring",
                        "seed": 0,
                        "dataset": "MuSiQue-Ans",
                        "method": "CGGR",
                        "example_id": "ex-0",
                        "error_type": "AttributeError",
                        "error_repr": "AttributeError()",
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            result = audit(workdir, require_full=True)

        self.assertFalse(result["ok"])
        self.assertTrue(
            any(blocker.startswith("generation_or_scoring failures present") for blocker in result["blockers"])
        )

    def test_empty_predictions_and_zero_tokens_are_blockers(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            workdir = Path(tmpdir)
            self._write_package(workdir, sharded=False)
            results = workdir / "results"
            (results / "raw_predictions.jsonl").write_text(
                '{"id": 1, "method": "CGGR", "prediction": "", "new_tokens": 0}\n',
                encoding="utf-8",
            )

            result = audit(workdir, require_full=True)

        self.assertFalse(result["ok"])
        self.assertIn("empty decoded predictions present: 1/1 (1.000)", result["blockers"])
        self.assertIn("zero-token generations present: 1/1 (1.000)", result["blockers"])


if __name__ == "__main__":
    unittest.main()
