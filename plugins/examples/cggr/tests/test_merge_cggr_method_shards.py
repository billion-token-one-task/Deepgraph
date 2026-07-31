import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import merge_cggr_method_shards
from audit_paper_benchmark_artifacts import TOP_VENUE_BASELINE_METHODS, audit
from materialize_audited_cggr_results import materialize


PAPER_DATASETS = [
    "MuSiQue-Ans",
    "StrategyQA",
    "2WikiMultihopQA",
    "Stress Test Split: Simple-vs-Hard Counterfactual Partition",
]


class MergeCggrMethodShardsTests(unittest.TestCase):
    def _write_shard(
        self,
        root: Path,
        method: str,
        *,
        score_by_seed: dict[int, float] | None = None,
        datasets: list[str] | None = None,
    ) -> None:
        results = root / "results"
        results.mkdir(parents=True)
        dataset_names = datasets or ["D1", "D2", "D3", "D4"]
        targets = [{"name": name, "hf_dataset": name, "split": "validation"} for name in dataset_names]
        config = {
            "model_id": "Qwen/Qwen2.5-14B-Instruct",
            "targets": targets,
            "methods": [method],
            "seeds": 5,
            "seed_values": [0, 1, 2, 3, 4],
            "max_examples_per_dataset_seed": 128,
            "sharded_run": True,
            "shard_axes": {"method": True, "target": False, "seed": False},
        }
        (results / "run_config.json").write_text(json.dumps(config), encoding="utf-8")
        (results / "benchmark_summary.json").write_text(
            json.dumps({"duration_seconds": 1.0, "peak_vram_mb": 10.0}),
            encoding="utf-8",
        )
        (results / "environment_report.json").write_text(json.dumps({"python": sys.version}), encoding="utf-8")
        rows = []
        routing = []
        for seed in range(5):
            score = 1.0 if score_by_seed is None else score_by_seed[seed]
            for dataset in dataset_names:
                for idx in range(128):
                    example_id = f"{dataset}-{seed}-{idx}"
                    difficulty = 0.2 if idx < 64 else 0.8
                    rows.append(
                        {
                            "seed": seed,
                            "dataset": dataset,
                            "method": method,
                            "example_id": example_id,
                            "prediction": "answer",
                            "primary_score": score,
                            "exact": score,
                            "f1": score,
                            "new_tokens": 3,
                            "latency_seconds": 0.1,
                        }
                    )
                    if any(token in method.lower() for token in ("cggr", "gate", "routing")):
                        routing.append(
                            {
                                "seed": seed,
                                "dataset": dataset,
                                "method": method,
                                "example_id": example_id,
                                "difficulty": difficulty,
                                "max_new_tokens": 192 if idx % 2 == 0 else 64,
                                "routed_to_deliberation": idx % 2 == 0,
                            }
                        )
        (results / "raw_predictions.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in rows),
            encoding="utf-8",
        )
        (results / "routing_decisions.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in routing),
            encoding="utf-8",
        )
        (results / "failure_cases.jsonl").write_text("", encoding="utf-8")

    def test_merges_complete_method_axis_and_clears_shard_flag(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            shard_a = root / "shard_a"
            shard_b = root / "shard_b"
            out = root / "merged"
            self._write_shard(shard_a, "Method A")
            self._write_shard(shard_b, "Method B")

            with mock.patch.object(merge_cggr_method_shards, "REQUIRED_METHODS", {"Method A", "Method B"}), mock.patch.object(
                merge_cggr_method_shards, "REQUIRED_ABLATIONS", set()
            ):
                result = merge_cggr_method_shards.merge([shard_a, shard_b], out)

            self.assertTrue(result["ok"], result)
            self.assertEqual(result["raw_predictions_lines"], 5120)
            merged_config = json.loads((out / "results" / "run_config.json").read_text(encoding="utf-8"))
            self.assertFalse(merged_config["sharded_run"])
            merged_summary = json.loads((out / "results" / "benchmark_summary.json").read_text(encoding="utf-8"))
            self.assertTrue(merged_summary["full_benchmark_completed"])

    def test_merge_computes_seed_std_cost_adjusted_metric_and_paired_stats(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            shard_a = root / "shard_a"
            shard_b = root / "shard_b"
            out = root / "merged"
            self._write_shard(
                shard_a,
                "Always-Reason Chain-of-Thought",
                score_by_seed={0: 0.40, 1: 0.42, 2: 0.44, 3: 0.46, 4: 0.48},
            )
            self._write_shard(
                shard_b,
                "CGGR",
                score_by_seed={0: 0.50, 1: 0.55, 2: 0.60, 3: 0.65, 4: 0.70},
            )

            with mock.patch.object(
                merge_cggr_method_shards,
                "REQUIRED_METHODS",
                {"Always-Reason Chain-of-Thought", "CGGR"},
            ), mock.patch.object(merge_cggr_method_shards, "REQUIRED_ABLATIONS", set()):
                result = merge_cggr_method_shards.merge([shard_a, shard_b], out)

            self.assertTrue(result["ok"], result)
            merged_summary = json.loads((out / "results" / "benchmark_summary.json").read_text(encoding="utf-8"))
            self.assertGreater(merged_summary["per_method_std"]["CGGR"], 0.0)
            self.assertLess(merged_summary["per_method"]["CGGR"]["metric_value"], merged_summary["per_method"]["CGGR"]["score"])
            bootstrap = merged_summary["bootstrap_ci"]
            self.assertGreater(bootstrap["observed_delta"], 0.0)
            self.assertIn("delta_ci95", bootstrap)
            self.assertLessEqual(bootstrap["paired_permutation_p"], 1.0)

    def test_merge_blocks_overlapping_duplicate_cells(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            shard_a = root / "shard_a"
            shard_b = root / "shard_b"
            out = root / "merged"
            self._write_shard(shard_a, "Method A")
            self._write_shard(shard_b, "Method A")

            with mock.patch.object(merge_cggr_method_shards, "REQUIRED_METHODS", {"Method A"}), mock.patch.object(
                merge_cggr_method_shards, "REQUIRED_ABLATIONS", set()
            ):
                result = merge_cggr_method_shards.merge([shard_a, shard_b], out)

            self.assertFalse(result["ok"], result)
            self.assertTrue(
                any("duplicate raw prediction rows" in blocker or "above paper gate" in blocker for blocker in result["blockers"]),
                result,
            )

    def test_merge_allows_low_score_failure_analysis_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            shard_a = root / "shard_a"
            shard_b = root / "shard_b"
            out = root / "merged"
            self._write_shard(shard_a, "Always-Reason Chain-of-Thought")
            self._write_shard(shard_b, "CGGR")
            failure_row = {
                "seed": 0,
                "dataset": "D1",
                "method": "CGGR",
                "example_id": "D1-0-0",
                "prediction_answer": "wrong",
                "gold_answer": "right",
                "primary_score": 0.0,
                "exact": 0.0,
                "f1": 0.0,
            }
            (shard_b / "results" / "failure_cases.jsonl").write_text(
                json.dumps(failure_row) + "\n",
                encoding="utf-8",
            )

            with mock.patch.object(
                merge_cggr_method_shards,
                "REQUIRED_METHODS",
                {"Always-Reason Chain-of-Thought", "CGGR"},
            ), mock.patch.object(merge_cggr_method_shards, "REQUIRED_ABLATIONS", set()):
                result = merge_cggr_method_shards.merge([shard_a, shard_b], out)

            self.assertTrue(result["ok"], result)
            merged_failures = (out / "results" / "failure_cases.jsonl").read_text(encoding="utf-8")
            self.assertIn('"prediction_answer": "wrong"', merged_failures)

    def test_merge_blocks_generation_or_scoring_failure_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            shard_a = root / "shard_a"
            shard_b = root / "shard_b"
            out = root / "merged"
            self._write_shard(shard_a, "Always-Reason Chain-of-Thought")
            self._write_shard(shard_b, "CGGR")
            failure_row = {
                "stage": "generation_or_scoring",
                "seed": 0,
                "dataset": "D1",
                "method": "CGGR",
                "example_id": "D1-0-0",
                "error_type": "RuntimeError",
                "error_repr": "RuntimeError('empty generation')",
            }
            (shard_b / "results" / "failure_cases.jsonl").write_text(
                json.dumps(failure_row) + "\n",
                encoding="utf-8",
            )

            with mock.patch.object(
                merge_cggr_method_shards,
                "REQUIRED_METHODS",
                {"Always-Reason Chain-of-Thought", "CGGR"},
            ), mock.patch.object(merge_cggr_method_shards, "REQUIRED_ABLATIONS", set()):
                result = merge_cggr_method_shards.merge([shard_a, shard_b], out)

            self.assertFalse(result["ok"], result)
            self.assertIn("generation_or_scoring failures present in shard failure_cases.jsonl", result["blockers"])

    def test_merge_preserves_extra_registered_baseline_methods(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            shard_a = root / "shard_a"
            shard_b = root / "shard_b"
            out = root / "merged"
            self._write_shard(shard_a, "Method A")
            self._write_shard(shard_b, "CAR-Style Certainty Adaptive Routing")

            with mock.patch.object(merge_cggr_method_shards, "REQUIRED_METHODS", {"Method A"}), mock.patch.object(
                merge_cggr_method_shards, "REQUIRED_ABLATIONS", set()
            ):
                result = merge_cggr_method_shards.merge([shard_a, shard_b], out)

            self.assertTrue(result["ok"], result)
            merged_config = json.loads((out / "results" / "run_config.json").read_text(encoding="utf-8"))
            manifest = json.loads((out / "results" / "artifact_manifest.json").read_text(encoding="utf-8"))
            self.assertIn("CAR-Style Certainty Adaptive Routing", merged_config["methods"])
            self.assertIn("CAR-Style Certainty Adaptive Routing", manifest["methods"])

    def test_merge_copies_claim_scope_override_into_results(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            shard_a = root / "shard_a"
            shard_b = root / "shard_b"
            out = root / "merged"
            self._write_shard(shard_a, "Method A")
            self._write_shard(shard_b, "Method B")
            spec_dir = out / "spec"
            spec_dir.mkdir(parents=True)
            (spec_dir / "merge_config.json").write_text(
                json.dumps({"claim_scope_override": "spec/active_claim_scope_override.json"}),
                encoding="utf-8",
            )
            override = {
                "schema_version": "cggr_active_claim_scope_override_v1",
                "active_method_claim": "fixed proxy-gated executable instantiation",
                "learned_router_claim_allowed": False,
                "broad_top_venue_or_sota_superiority_claim_allowed": False,
            }
            (spec_dir / "active_claim_scope_override.json").write_text(json.dumps(override), encoding="utf-8")

            with mock.patch.object(merge_cggr_method_shards, "REQUIRED_METHODS", {"Method A", "Method B"}), mock.patch.object(
                merge_cggr_method_shards, "REQUIRED_ABLATIONS", set()
            ):
                result = merge_cggr_method_shards.merge([shard_a, shard_b], out)

            self.assertTrue(result["ok"], result)
            copied = json.loads((out / "results" / "active_claim_scope_override.json").read_text(encoding="utf-8"))
            self.assertEqual(copied["active_method_claim"], "fixed proxy-gated executable instantiation")
            merged_config = json.loads((out / "results" / "run_config.json").read_text(encoding="utf-8"))
            self.assertEqual(merged_config["claim_scope_override"], "results/active_claim_scope_override.json")
            merged_summary = json.loads((out / "results" / "benchmark_summary.json").read_text(encoding="utf-8"))
            self.assertEqual(merged_summary["claim_scope_override"]["schema_version"], "cggr_active_claim_scope_override_v1")
            manifest = json.loads((out / "results" / "artifact_manifest.json").read_text(encoding="utf-8"))
            self.assertIn("active_claim_scope_override.json", manifest["artifact_paths"])

    def test_merged_required_methods_pass_full_audit_contract(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            shards = []
            for method in sorted(merge_cggr_method_shards.REQUIRED_METHODS):
                shard = root / method.replace("/", "_").replace(" ", "_")
                self._write_shard(shard, method, datasets=PAPER_DATASETS)
                shards.append(shard)
            out = root / "merged"
            spec_dir = out / "spec"
            spec_dir.mkdir(parents=True)
            (spec_dir / "merge_config.json").write_text(
                json.dumps({"claim_scope_override": "spec/active_claim_scope_override.json"}),
                encoding="utf-8",
            )
            (spec_dir / "active_claim_scope_override.json").write_text(
                json.dumps(
                    {
                        "schema_version": "cggr_active_claim_scope_override_v1",
                        "active_method_claim": "fixed proxy-gated executable instantiation",
                        "learned_router_claim_allowed": False,
                        "trained_estimator_claim_allowed": False,
                        "oracle_router_in_active_contract": False,
                        "broad_top_venue_or_sota_superiority_claim_allowed": False,
                    }
                ),
                encoding="utf-8",
            )

            result = merge_cggr_method_shards.merge(shards, out)

            self.assertTrue(result["ok"], result)
            audit_result = audit(out, require_full=True)
            self.assertTrue(audit_result["ok"], audit_result)
            materialized = materialize(out, root / "materialized")
            self.assertTrue(materialized["ok"], materialized)
            written_names = {Path(path).name for path in materialized["written"]}
            self.assertIn("significance_report.md", written_names)
            self.assertIn("reproducibility_statement.md", written_names)
            self.assertIn("claim_evidence_map.md", written_names)
            self.assertIn("completion_audit.md", written_names)
            self.assertIn("failure_analysis.md", written_names)
            self.assertIn("results_section_snippet.tex", written_names)
            self.assertIn("limitations_snippet.tex", written_names)
            self.assertIn("utility_comparison_figure.tex", written_names)
            self.assertIn("problem_awareness.json", written_names)
            self.assertIn("publication_evidence_contract.json", written_names)
            self.assertIn("evidence_manifest.json", written_names)
            self.assertIn("claim_evidence_matrix.json", written_names)
            self.assertIn("reviewer_report.json", written_names)
            self.assertIn("paper_quality_report.json", written_names)
            failure_analysis = (root / "materialized" / "failure_analysis.md").read_text(encoding="utf-8")
            self.assertIn("Failure Analysis", failure_analysis)
            self.assertIn("Failure rows", failure_analysis)
            reproducibility = (root / "materialized" / "reproducibility_statement.md").read_text(encoding="utf-8")
            self.assertIn("fixed proxy-gated executable instantiation", reproducibility)
            self.assertIn("Trained estimator: `False`", reproducibility)
            self.assertIn("Claim Scope Override", reproducibility)
            self.assertIn("Learned-router claim allowed: `False`", reproducibility)
            completion_audit = (root / "materialized" / "completion_audit.md").read_text(encoding="utf-8")
            self.assertIn("Prompt-To-Artifact Checklist", completion_audit)
            self.assertIn("claim_support_decision", completion_audit)
            self.assertIn("Top-venue general-superiority gate", completion_audit)
            self.assertIn("Claim scope override", completion_audit)
            claim_map = (root / "materialized" / "claim_evidence_map.md").read_text(encoding="utf-8")
            self.assertIn("Broad adaptive-reasoning superiority", claim_map)
            self.assertIn("Claim scope", claim_map)
            utility_figure = (root / "materialized" / "utility_comparison_figure.tex").read_text(encoding="utf-8")
            self.assertIn("Audited utility comparison", utility_figure)
            self.assertIn("CGGR", utility_figure)
            claim_values = json.loads((root / "materialized" / "claim_values.json").read_text(encoding="utf-8"))
            self.assertIn("paired_permutation_p", claim_values)
            self.assertIn("cggr_vs_baseline_delta_ci95", claim_values)
            self.assertIn("claim_support_decision", claim_values)
            self.assertEqual(claim_values["cggr_implementation_type"], "fixed proxy-gated executable instantiation")
            self.assertFalse(claim_values["cggr_trained_estimator"])
            self.assertFalse(claim_values["learned_router_claim_allowed"])
            self.assertFalse(claim_values["trained_estimator_claim_allowed"])
            self.assertFalse(claim_values["oracle_router_in_active_contract"])
            self.assertFalse(claim_values["broad_top_venue_or_sota_superiority_claim_allowed"])
            self.assertEqual(
                claim_values["claim_scope_override"]["schema_version"],
                "cggr_active_claim_scope_override_v1",
            )
            self.assertEqual(
                claim_values["top_venue_general_superiority_decision"],
                "blocked_missing_strict_top_venue_baseline_audit",
            )
            self.assertIn("audit_warnings", claim_values)
            publication_contract = json.loads(
                (root / "materialized" / "publication_evidence_contract.json").read_text(encoding="utf-8")
            )
            self.assertEqual(publication_contract["evidence_tier"], "audited_full_benchmark")
            self.assertEqual(
                publication_contract["quality_gates"]["top_venue_general_superiority_decision"],
                "blocked_missing_strict_top_venue_baseline_audit",
            )
            claim_matrix = json.loads((root / "materialized" / "claim_evidence_matrix.json").read_text(encoding="utf-8"))
            self.assertTrue(any(row["claim_id"] == "C4_top_venue_general_superiority" for row in claim_matrix))

    def test_merge_plus_top_venue_methods_passes_strict_top_venue_audit(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            shards = []
            methods = sorted(merge_cggr_method_shards.REQUIRED_METHODS | TOP_VENUE_BASELINE_METHODS)
            for method in methods:
                shard = root / method.replace("/", "_").replace(" ", "_").replace("-", "_")
                self._write_shard(shard, method, datasets=PAPER_DATASETS)
                shards.append(shard)
            out = root / "merged"

            result = merge_cggr_method_shards.merge(shards, out)

            self.assertTrue(result["ok"], result)
            audit_result = audit(out, require_full=True, require_top_venue_baselines=True)
            self.assertTrue(audit_result["ok"], audit_result)
            materialized = materialize(out, root / "materialized_top", require_top_venue_baselines=True)
            self.assertTrue(materialized["ok"], materialized)
            claim_values = json.loads((root / "materialized_top" / "claim_values.json").read_text(encoding="utf-8"))
            self.assertEqual(
                claim_values["top_venue_general_superiority_decision"],
                "eligible_under_strict_top_venue_audit",
            )
            merged_config = json.loads((out / "results" / "run_config.json").read_text(encoding="utf-8"))
            for method in TOP_VENUE_BASELINE_METHODS:
                self.assertIn(method, merged_config["methods"])


if __name__ == "__main__":
    unittest.main()
