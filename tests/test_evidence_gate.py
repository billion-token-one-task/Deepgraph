import json
import unittest
from unittest.mock import patch

from tests.temp_utils import temporary_workdir


class FakeEvidenceDb:
    def __init__(self, run: dict):
        self.run = run

    def fetchone(self, sql, params=()):
        if "FROM experiment_runs" in sql:
            return self.run
        return None


class EvidenceGateTests(unittest.TestCase):
    def test_proxy_without_statistical_report_is_preliminary(self):
        from agents.evidence_gate import evaluate_evidence

        gate = evaluate_evidence({
            "research_spec": {
                "evidence_level": "proxy",
                "required_evidence": ["statistical_test"],
            },
            "benchmark_results": None,
            "statistical_report": None,
            "review": None,
        })

        self.assertEqual(gate["manuscript_status"], "preliminary")
        self.assertIn("missing_statistical_report", gate["blocking_reasons"])

    def test_reviewer_reject_blocks_paper_ready(self):
        from agents.evidence_gate import evaluate_evidence

        gate = evaluate_evidence({
            "research_spec": {"required_evidence": ["baseline_comparison"]},
            "benchmark_results": {"rows": [{"status": "ok", "seed": 0}]},
            "statistical_report": {"comparisons": [{"paired_sign_test_p": 0.02}]},
            "review": {
                "recommendation": "reject",
                "required_experiments": ["Run more seeds."],
            },
        })

        self.assertEqual(gate["manuscript_status"], "not_publishable")
        self.assertIn("review_rejected", gate["blocking_reasons"])

    def test_reviewer_weak_reject_blocks_paper_ready(self):
        from agents.evidence_gate import evaluate_evidence

        gate = evaluate_evidence({
            "research_spec": {"required_evidence": []},
            "benchmark_results": {"rows": [{"status": "ok", "dataset": "d", "seed": 0}]},
            "statistical_report": {},
            "review": {"recommendation": "weak_reject", "required_experiments": ["Add baselines."]},
        })

        self.assertEqual(gate["manuscript_status"], "needs_more_experiments")
        self.assertIn("review_requires_revision", gate["blocking_reasons"])
        self.assertIn("Add baselines.", gate["next_required_experiments"])

    def test_multiseed_requirement_blocks_small_seed_count(self):
        from agents.evidence_gate import evaluate_evidence

        gate = evaluate_evidence({
            "research_spec": {"required_evidence": ["multi_seed", "statistical_test"]},
            "benchmark_results": {"rows": [
                {"status": "ok", "seed": 0},
                {"status": "ok", "seed": 1},
            ]},
            "statistical_report": {"comparisons": [{"paired_sign_test_p": 0.02}]},
            "review": None,
        })

        self.assertEqual(gate["manuscript_status"], "needs_more_experiments")
        self.assertIn("benchmark_suite_has_fewer_than_10_seeds", gate["blocking_reasons"])

    def test_sufficient_evidence_without_reject_is_paper_ready_candidate(self):
        from agents.evidence_gate import evaluate_evidence

        rows = [
            {"status": "ok", "seed": seed, "dataset": dataset}
            for dataset in ("synthetic_grouped", "sklearn_breast_cancer_grouped")
            for seed in range(10)
        ]
        gate = evaluate_evidence({
            "research_spec": {"required_evidence": ["multi_seed", "multi_dataset", "statistical_test", "baseline_comparison"]},
            "benchmark_results": {"rows": rows},
            "statistical_report": {"comparisons": [
                {"dataset": "synthetic_grouped", "mean_delta": 0.05, "paired_sign_test_p": 0.02, "wins": 8, "losses": 2},
                {"dataset": "sklearn_breast_cancer_grouped", "mean_delta": 0.03, "paired_sign_test_p": 0.04, "wins": 7, "losses": 3},
            ]},
            "review": {"recommendation": "weak_accept", "required_experiments": []},
        })

        self.assertEqual(gate["manuscript_status"], "paper_ready_candidate")
        self.assertFalse(gate["blocking_reasons"])

    def test_multidataset_requirement_blocks_single_dataset(self):
        from agents.evidence_gate import evaluate_evidence

        rows = [
            {"status": "ok", "seed": seed, "dataset": "synthetic_grouped"}
            for seed in range(10)
        ]
        gate = evaluate_evidence({
            "research_spec": {"required_evidence": ["multi_dataset", "multi_seed", "statistical_test"]},
            "benchmark_results": {"rows": rows},
            "statistical_report": {"comparisons": [{"dataset": "synthetic_grouped", "paired_sign_test_p": 0.02}]},
            "review": None,
        })

        self.assertEqual(gate["manuscript_status"], "needs_more_experiments")
        self.assertIn("benchmark_suite_has_fewer_than_2_datasets", gate["blocking_reasons"])

    def test_ablation_requirement_blocks_missing_ablation_rows(self):
        from agents.evidence_gate import evaluate_evidence

        rows = [
            {"status": "ok", "seed": seed, "dataset": dataset}
            for dataset in ("synthetic_grouped", "sklearn_breast_cancer_grouped")
            for seed in range(10)
        ]
        gate = evaluate_evidence({
            "research_spec": {"required_evidence": ["ablation", "multi_dataset", "multi_seed", "statistical_test"]},
            "benchmark_results": {"rows": rows},
            "statistical_report": {"comparisons": [{"dataset": "synthetic_grouped", "paired_sign_test_p": 0.02}]},
            "review": None,
        })

        self.assertEqual(gate["manuscript_status"], "needs_more_experiments")
        self.assertIn("missing_ablation_rows", gate["blocking_reasons"])

    def test_ablation_requirement_accepts_ablation_rows(self):
        from agents.evidence_gate import evaluate_evidence

        rows = [
            {"status": "ok", "seed": seed, "dataset": dataset}
            for dataset in ("synthetic_grouped", "sklearn_breast_cancer_grouped")
            for seed in range(10)
        ]
        rows.append({
            "status": "ok",
            "seed": 0,
            "dataset": "synthetic_grouped",
            "analysis_type": "ablation",
            "ablation": "preference_cone_penalty_sweep",
        })
        gate = evaluate_evidence({
            "research_spec": {"required_evidence": ["ablation", "multi_dataset", "multi_seed", "statistical_test"]},
            "benchmark_results": {"rows": rows},
            "statistical_report": {"comparisons": [{"dataset": "synthetic_grouped", "paired_sign_test_p": 0.02}]},
            "review": None,
        })

        self.assertIn("has_ablation", gate["satisfied_requirements"])
        self.assertNotIn("missing_ablation_rows", gate["blocking_reasons"])

    def test_failed_benchmark_rows_block_paper_ready(self):
        from agents.evidence_gate import evaluate_evidence

        rows = [
            {"status": "ok", "seed": seed, "dataset": dataset}
            for dataset in ("synthetic_grouped", "sklearn_breast_cancer_grouped")
            for seed in range(10)
        ]
        rows.append({
            "status": "error",
            "seed": 0,
            "dataset": "synthetic_grouped",
            "method": "threshold_optimizer",
            "error": "No module named 'fairlearn'",
        })
        rows.append({
            "status": "ok",
            "seed": 0,
            "dataset": "synthetic_grouped",
            "analysis_type": "ablation",
        })
        gate = evaluate_evidence({
            "research_spec": {"required_evidence": ["ablation", "multi_dataset", "multi_seed", "statistical_test"]},
            "benchmark_results": {"rows": rows},
            "statistical_report": {"comparisons": [{"dataset": "synthetic_grouped", "mean_delta": 0.1, "paired_sign_test_p": 0.02, "wins": 10, "losses": 0}]},
            "review": None,
        })

        self.assertEqual(gate["manuscript_status"], "needs_more_experiments")
        self.assertIn("benchmark_suite_has_error_rows", gate["blocking_reasons"])

    def test_unsupported_dataset_comparisons_block_paper_ready(self):
        from agents.evidence_gate import evaluate_evidence

        rows = [
            {"status": "ok", "seed": seed, "dataset": dataset}
            for dataset in ("synthetic_grouped", "sklearn_breast_cancer_grouped")
            for seed in range(10)
        ]
        rows.append({
            "status": "ok",
            "seed": 0,
            "dataset": "synthetic_grouped",
            "analysis_type": "ablation",
        })
        gate = evaluate_evidence({
            "research_spec": {"required_evidence": ["ablation", "multi_dataset", "multi_seed", "statistical_test"]},
            "benchmark_results": {"rows": rows},
            "statistical_report": {"comparisons": [
                {"dataset": "synthetic_grouped", "mean_delta": 0.1, "paired_sign_test_p": 0.02, "wins": 10, "losses": 0},
                {"dataset": "sklearn_breast_cancer_grouped", "mean_delta": -0.02, "paired_sign_test_p": 0.02, "wins": 1, "losses": 9},
            ]},
            "review": None,
        })

        self.assertEqual(gate["manuscript_status"], "needs_more_experiments")
        self.assertIn("statistical_comparisons_do_not_support_claim", gate["blocking_reasons"])

    def test_write_evidence_gate_writes_artifact(self):
        from agents.artifact_manager import list_artifacts
        from agents.evidence_gate import write_evidence_gate

        with temporary_workdir() as workdir:
            (workdir / "research_spec.json").write_text(json.dumps({
                "required_evidence": ["statistical_test"],
            }), encoding="utf-8")
            results_dir = workdir / "artifacts" / "results"
            results_dir.mkdir(parents=True)
            (results_dir / "benchmark_results.json").write_text(json.dumps({
                "rows": [{"status": "ok", "seed": 0}],
            }), encoding="utf-8")
            (results_dir / "statistical_report.json").write_text(json.dumps({
                "comparisons": [{"paired_sign_test_p": 0.02}],
            }), encoding="utf-8")
            fake_db = FakeEvidenceDb({"id": 15, "workdir": str(workdir)})

            with patch("agents.evidence_gate.db", fake_db):
                gate = write_evidence_gate(15)

            self.assertIn("manuscript_status", gate)
            self.assertTrue((results_dir / "evidence_gate.json").exists())
            self.assertIn(
                "artifacts/results/evidence_gate.json",
                {item["path"] for item in list_artifacts(workdir)},
            )


if __name__ == "__main__":
    unittest.main()
