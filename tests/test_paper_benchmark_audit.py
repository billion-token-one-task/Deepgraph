import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from audit_paper_benchmark_artifacts import TOP_VENUE_BASELINE_METHODS, audit


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
    def _write_package(self, workdir: Path, *, sharded: bool, top_venue: bool = False) -> None:
        results = workdir / "results"
        results.mkdir(parents=True)
        methods = REQUIRED_METHODS + (sorted(TOP_VENUE_BASELINE_METHODS) if top_venue else [])
        per_method = {method: {"metric_value": 0.5, "score": 0.6} for method in methods}
        datasets = [{"name": name, "id": name, "split": "test"} for name in REQUIRED_DATASETS]
        summary = {
            "full_benchmark_completed": True,
            "num_seeds": 5,
            "datasets": datasets,
            "per_method": per_method,
            "per_method_std": {method: 0.01 for method in methods},
            "ablation_table": [{"ablation": name, "metric_value": 0.4} for name in REQUIRED_ABLATIONS],
            "budget": {"seeds": 5, "max_examples_per_dataset_seed": 128},
            "load_failures": [],
        }
        run_config = {
            "targets": [{"name": name} for name in REQUIRED_DATASETS],
            "methods": methods,
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
            "methods": methods,
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
                for method in methods:
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
                        if method == "CGGR" or method in TOP_VENUE_BASELINE_METHODS:
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

    def test_top_venue_baseline_mode_requires_recent_adaptive_baselines(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            workdir = Path(tmpdir)
            self._write_package(workdir, sharded=False)

            result = audit(workdir, require_full=True, require_top_venue_baselines=True)

        self.assertFalse(result["ok"])
        self.assertTrue(
            any("required method missing: CAR-Style Certainty Adaptive Routing" == blocker for blocker in result["blockers"]),
            result,
        )

    def test_top_venue_baseline_mode_passes_when_extra_methods_have_full_coverage(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            workdir = Path(tmpdir)
            self._write_package(workdir, sharded=False, top_venue=True)

            result = audit(workdir, require_full=True, require_top_venue_baselines=True)

        self.assertTrue(result["ok"], result)
        self.assertTrue(result["require_top_venue_baselines"])

    def test_cggr_global_zero_routing_is_a_blocker(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            workdir = Path(tmpdir)
            self._write_package(workdir, sharded=False)
            results = workdir / "results"
            routing_lines = []
            for seed in range(5):
                for dataset in REQUIRED_DATASETS:
                    for idx in range(128):
                        routing_lines.append(
                            json.dumps(
                                {
                                    "seed": seed,
                                    "dataset": dataset,
                                    "method": "CGGR",
                                    "example_id": f"{dataset}-{seed}-{idx}",
                                    "max_new_tokens": 64,
                                    "routed_to_deliberation": False,
                                }
                            )
                        )
            (results / "routing_decisions.jsonl").write_text("\n".join(routing_lines) + "\n", encoding="utf-8")

            result = audit(workdir, require_full=True)

        self.assertFalse(result["ok"])
        self.assertIn("CGGR routing collapsed to zero deliberation across all audited cells", result["blockers"])

    def test_cggr_single_cell_zero_routing_is_a_warning(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            workdir = Path(tmpdir)
            self._write_package(workdir, sharded=False)
            results = workdir / "results"
            routing_lines = []
            for seed in range(5):
                for dataset in REQUIRED_DATASETS:
                    for idx in range(128):
                        routing_lines.append(
                            json.dumps(
                                {
                                    "seed": seed,
                                    "dataset": dataset,
                                    "method": "CGGR",
                                    "example_id": f"{dataset}-{seed}-{idx}",
                                    "max_new_tokens": 64,
                                    "routed_to_deliberation": False
                                    if seed == 0 and dataset == "MuSiQue-Ans"
                                    else idx % 2 == 0,
                                }
                            )
                        )
            (results / "routing_decisions.jsonl").write_text("\n".join(routing_lines) + "\n", encoding="utf-8")

            result = audit(workdir, require_full=True)

        self.assertTrue(result["ok"], result)
        self.assertIn("routing rate is 0.000 for CGGR|MuSiQue-Ans|seed=0", result["warnings"])

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

    def test_low_score_failure_analysis_rows_are_not_generation_blockers(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            workdir = Path(tmpdir)
            self._write_package(workdir, sharded=False)
            results = workdir / "results"
            (results / "failure_cases.jsonl").write_text(
                json.dumps(
                    {
                        "seed": 0,
                        "dataset": "MuSiQue-Ans",
                        "method": "CGGR",
                        "example_id": "ex-0",
                        "prediction_answer": "wrong",
                        "gold_answer": "right",
                        "primary_score": 0.0,
                        "exact": 0.0,
                        "f1": 0.0,
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            result = audit(workdir, require_full=True)

        self.assertTrue(result["ok"], result)
        self.assertEqual(result["failure_diagnostics"]["failure_cases_rows"], 1)
        self.assertEqual(result["failure_diagnostics"]["generation_or_scoring_failures"], 0)

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
