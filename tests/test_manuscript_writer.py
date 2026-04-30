import json
import re
import unittest
from pathlib import Path
from unittest.mock import patch

from agents.artifact_manager import record_artifact
from agents.manuscript_writer import generate_manuscript
from tests.temp_utils import temporary_workdir


class FakeManuscriptDb:
    def __init__(self, run: dict, insight: dict, claims: list[dict] | None = None):
        self.run = run
        self.insight = insight
        self.claims = claims or []

    def fetchone(self, sql, params=()):
        if "FROM experiment_runs" in sql:
            return self.run
        if "FROM deep_insights" in sql:
            return self.insight
        return None

    def fetchall(self, sql, params=()):
        if "FROM experimental_claims" in sql:
            return self.claims
        return []


def _run(workdir: Path, verdict: str | None, status: str | None = None) -> dict:
    return {
        "id": 9,
        "deep_insight_id": 4,
        "workdir": str(workdir),
        "status": status or ("completed" if verdict else "testing"),
        "hypothesis_verdict": verdict,
        "baseline_metric_name": "accuracy",
        "baseline_metric_value": 0.7,
        "best_metric_value": 0.8,
        "effect_size": 0.1,
        "effect_pct": 14.28,
        "error_message": None,
    }


def _insight() -> dict:
    return {
        "id": 4,
        "title": "Calibrated Adapters Improve Small Models",
        "problem_statement": "Small models are poorly calibrated.",
        "hypothesis": "Adapters improve calibration without increasing inference cost.",
        "supporting_papers": json.dumps([
            {"id": "2301.00001", "title": "Calibration for Small Models", "authors": ["A. Author"]},
            {"id": "2301.00002", "title": "Adapter Methods", "authors": ["B. Author"]},
        ]),
    }


def _claims() -> list[dict]:
    return [{
        "claim_text": "Experimental validation confirms the hypothesis.",
        "verdict": "confirmed",
        "supporting_data": json.dumps({"baseline": 0.7, "best": 0.8}),
    }]


class ManuscriptWriterTests(unittest.TestCase):
    def test_refuses_run_without_terminal_verdict(self):
        with temporary_workdir() as tmp:
            fake_db = FakeManuscriptDb(_run(tmp, None), _insight())

            with patch("agents.manuscript_writer.db", fake_db):
                result = generate_manuscript(9)

            self.assertEqual(result["status"], "error")
            self.assertEqual(result["reason"], "run_not_terminal")

    def test_confirmed_run_without_evidence_gate_produces_preliminary_report(self):
        with temporary_workdir() as workdir:
            fake_db = FakeManuscriptDb(_run(workdir, "confirmed"), _insight(), _claims())

            with patch("agents.manuscript_writer.db", fake_db):
                result = generate_manuscript(9)

            manuscript_dir = workdir / "artifacts" / "manuscript"
            self.assertEqual(result["status"], "complete")
            self.assertEqual(result["manuscript_status"], "preliminary")
            self.assertTrue((manuscript_dir / "preliminary_report.md").exists())
            self.assertFalse((manuscript_dir / "paper.md").exists())
            self.assertTrue((manuscript_dir / "references.bib").exists())
            self.assertTrue((manuscript_dir / "reproducibility.md").exists())
            manifest = json.loads((workdir / "artifact_manifest.json").read_text(encoding="utf-8"))
            self.assertIn("artifacts/manuscript/preliminary_report.md", {item["path"] for item in manifest["artifacts"]})

    def test_refuted_run_produces_negative_result_report(self):
        with temporary_workdir() as workdir:
            fake_db = FakeManuscriptDb(_run(workdir, "refuted"), _insight(), [{
                "claim_text": "Experimental validation refutes the hypothesis.",
                "verdict": "refuted",
                "supporting_data": "{}",
            }])

            with patch("agents.manuscript_writer.db", fake_db):
                result = generate_manuscript(9)

            manuscript_dir = workdir / "artifacts" / "manuscript"
            self.assertEqual(result["status"], "complete")
            self.assertTrue((manuscript_dir / "negative_result_report.md").exists())
            self.assertFalse((manuscript_dir / "paper.md").exists())

    def test_failed_run_without_verdict_produces_negative_result_report(self):
        with temporary_workdir() as workdir:
            run = _run(workdir, None, status="failed")
            run["error_message"] = "reproduction failed: no metric obtained"
            fake_db = FakeManuscriptDb(run, _insight(), [{
                "claim_text": "No baseline metric could be reproduced.",
                "verdict": "inconclusive",
                "supporting_data": "{}",
            }])

            with patch("agents.manuscript_writer.db", fake_db):
                result = generate_manuscript(9)

            manuscript_dir = workdir / "artifacts" / "manuscript"
            report = manuscript_dir / "negative_result_report.md"
            self.assertEqual(result["status"], "complete")
            self.assertEqual(result["verdict"], "inconclusive")
            self.assertTrue(report.exists())
            self.assertFalse((manuscript_dir / "paper.md").exists())
            self.assertIn("reproduction failed: no metric obtained", report.read_text(encoding="utf-8"))

    def test_generated_citations_exist_in_bib(self):
        with temporary_workdir() as workdir:
            result_dir = workdir / "artifacts" / "results"
            result_dir.mkdir(parents=True)
            (result_dir / "evidence_gate.json").write_text(json.dumps({
                "manuscript_status": "paper_ready_candidate",
                "blocking_reasons": [],
                "satisfied_requirements": ["has_benchmark_results", "has_statistical_report"],
                "next_required_experiments": [],
            }), encoding="utf-8")
            fake_db = FakeManuscriptDb(_run(workdir, "confirmed"), _insight(), _claims())

            with patch("agents.manuscript_writer.db", fake_db):
                generate_manuscript(9)

            paper = (workdir / "artifacts" / "manuscript" / "paper_candidate.md").read_text(encoding="utf-8")
            bib = (workdir / "artifacts" / "manuscript" / "references.bib").read_text(encoding="utf-8")
            cited = set(re.findall(r"@([A-Za-z0-9_:-]+)", paper))
            bib_keys = set(re.findall(r"@\w+\{([^,]+),", bib))

            self.assertTrue(cited)
            self.assertTrue(cited.issubset(bib_keys))

    def test_confirmed_manuscript_includes_audit_context_and_qualified_claim(self):
        with temporary_workdir() as workdir:
            result_dir = workdir / "artifacts" / "results"
            result_dir.mkdir(parents=True)
            (result_dir / "evidence_gate.json").write_text(json.dumps({
                "manuscript_status": "paper_ready_candidate",
                "blocking_reasons": [],
                "satisfied_requirements": ["has_benchmark_results", "has_statistical_report"],
                "next_required_experiments": [],
            }), encoding="utf-8")
            (result_dir / "metrics.json").write_text(json.dumps({
                "metric_name": "fairness_score",
                "metric_direction": "higher",
                "baseline": 0.5,
                "best_value": 0.62,
                "effect_pct": 24.0,
                "iterations_kept": 2,
            }), encoding="utf-8")
            (result_dir / "iterations.json").write_text(json.dumps([
                {"iteration_number": 1, "phase": "reproduction", "metric_value": 0.5, "status": "ok"},
                {"iteration_number": 2, "phase": "hypothesis_testing", "metric_value": 0.62, "status": "keep"},
            ]), encoding="utf-8")
            fake_db = FakeManuscriptDb(_run(workdir, "confirmed"), _insight(), _claims())

            with patch("agents.manuscript_writer.db", fake_db):
                generate_manuscript(9)

            paper = (workdir / "artifacts" / "manuscript" / "paper_candidate.md").read_text(encoding="utf-8")
            self.assertIn("Evidence-Gated Benchmark Result", paper)
            self.assertIn("Metric Definition", paper)
            self.assertIn("Per-Iteration Audit Trail", paper)
            self.assertIn("artifacts/results/metrics.json", paper)
            self.assertNotIn("top-venue", paper.lower())

    def test_evidence_gate_needs_more_experiments_writes_required_experiments_report(self):
        with temporary_workdir() as workdir:
            result_dir = workdir / "artifacts" / "results"
            result_dir.mkdir(parents=True)
            (result_dir / "evidence_gate.json").write_text(json.dumps({
                "manuscript_status": "needs_more_experiments",
                "blocking_reasons": ["benchmark_suite_has_fewer_than_10_seeds"],
                "satisfied_requirements": ["has_baseline_comparison"],
                "next_required_experiments": [
                    "Run at least 10 seeds for every configured dataset and method."
                ],
            }), encoding="utf-8")
            fake_db = FakeManuscriptDb(_run(workdir, "inconclusive"), _insight(), _claims())

            with patch("agents.manuscript_writer.db", fake_db):
                result = generate_manuscript(9)

            manuscript_dir = workdir / "artifacts" / "manuscript"
            report = manuscript_dir / "additional_experiments_required.md"
            self.assertEqual(result["status"], "complete")
            self.assertEqual(result["manuscript_status"], "needs_more_experiments")
            self.assertTrue(report.exists())
            self.assertFalse((manuscript_dir / "paper.md").exists())
            text = report.read_text(encoding="utf-8")
            self.assertIn("benchmark_suite_has_fewer_than_10_seeds", text)
            self.assertIn("Run at least 10 seeds", text)

    def test_evidence_gate_paper_ready_writes_candidate_with_statistical_results(self):
        with temporary_workdir() as workdir:
            result_dir = workdir / "artifacts" / "results"
            result_dir.mkdir(parents=True)
            (result_dir / "evidence_gate.json").write_text(json.dumps({
                "manuscript_status": "paper_ready_candidate",
                "blocking_reasons": [],
                "satisfied_requirements": ["has_benchmark_results", "has_statistical_report"],
                "next_required_experiments": [],
            }), encoding="utf-8")
            (result_dir / "statistical_report.json").write_text(json.dumps({
                "primary_metric": "fairness_score",
                "fairness_penalty": 0.7,
                "baseline_method": "logistic_regression",
                "best_method": "preference_cone_threshold",
                "summary": [{
                    "dataset": "synthetic_grouped",
                    "method": "preference_cone_threshold",
                    "metric": "fairness_score",
                    "mean": 0.63,
                    "std": 0.04,
                    "ci_low": 0.60,
                    "ci_high": 0.66,
                    "n": 10,
                }],
                "comparisons": [{
                    "dataset": "synthetic_grouped",
                    "baseline": "logistic_regression",
                    "candidate": "preference_cone_threshold",
                    "metric": "fairness_score",
                    "mean_delta": 0.08,
                    "paired_sign_test_p": 0.02,
                    "wins": 8,
                    "losses": 2,
                    "ties": 0,
                }],
                "pairwise_comparisons": [{
                    "dataset": "synthetic_grouped",
                    "baseline": "exponentiated_gradient",
                    "candidate": "preference_cone_threshold",
                    "metric": "fairness_score",
                    "mean_delta": 0.03,
                    "paired_sign_test_p": 0.04,
                    "wins": 7,
                    "losses": 3,
                    "ties": 0,
                }],
                "metric_summaries": [
                    {
                        "dataset": "synthetic_grouped",
                        "method": "preference_cone_threshold",
                        "metric": "accuracy",
                        "mean": 0.82,
                        "std": 0.02,
                        "ci_low": 0.80,
                        "ci_high": 0.84,
                        "n": 10,
                    },
                    {
                        "dataset": "synthetic_grouped",
                        "method": "preference_cone_threshold",
                        "metric": "demographic_parity_gap",
                        "mean": 0.10,
                        "std": 0.01,
                        "ci_low": 0.09,
                        "ci_high": 0.11,
                        "n": 10,
                    },
                    {
                        "dataset": "synthetic_grouped",
                        "method": "preference_cone_threshold",
                        "metric": "equalized_odds_gap",
                        "mean": 0.13,
                        "std": 0.02,
                        "ci_low": 0.11,
                        "ci_high": 0.15,
                        "n": 10,
                    },
                ],
                "ablation_summary": [{
                    "ablation": "preference_cone_penalty_sweep",
                    "dataset": "synthetic_grouped",
                    "method": "preference_cone_threshold_penalty_0.20",
                    "metric": "fairness_score",
                    "mean": 0.61,
                    "std": 0.03,
                    "ci_low": 0.58,
                    "ci_high": 0.64,
                    "n": 10,
                }],
            }), encoding="utf-8")
            (workdir / "benchmark_config.json").write_text(json.dumps({
                "paper_title": "Preference-Cone Thresholding for Grouped Fairness Classification Benchmarks",
                "scoped_claim": "Preference-cone thresholding improves a fairness-weighted objective across offline grouped classification benchmarks.",
                "methods": ["logistic_regression", "preference_cone_threshold"],
            }), encoding="utf-8")
            fake_db = FakeManuscriptDb(_run(workdir, "confirmed"), _insight(), _claims())

            with patch("agents.manuscript_writer.db", fake_db):
                result = generate_manuscript(9)

            manuscript_dir = workdir / "artifacts" / "manuscript"
            paper = manuscript_dir / "paper_candidate.md"
            self.assertEqual(result["status"], "complete")
            self.assertEqual(result["manuscript_status"], "paper_ready_candidate")
            self.assertTrue(paper.exists())
            self.assertFalse((manuscript_dir / "paper.md").exists())
            text = paper.read_text(encoding="utf-8")
            self.assertIn("Statistical Evidence", text)
            self.assertIn("preference_cone_threshold", text)
            self.assertIn("Aggregate Method Summary", text)
            self.assertIn("paired sign test", text.lower())
            self.assertIn("Ablation And Sensitivity", text)
            self.assertIn("preference_cone_penalty_sweep", text)
            self.assertNotIn("not a submission-ready paper", text)
            self.assertIn("Preference-Cone Thresholding for Grouped Fairness Classification Benchmarks", text)
            self.assertIn("Preference-cone thresholding improves a fairness-weighted objective", text)
            self.assertIn("Implemented Methods", text)
            self.assertIn("accuracy - 0.7 * demographic_parity_gap", text)
            self.assertIn("Secondary Metrics", text)
            self.assertIn("accuracy", text)
            self.assertIn("demographic_parity_gap", text)
            self.assertIn("equalized_odds_gap", text)
            self.assertIn("exponentiated_gradient", text)
            self.assertIn("Algorithmic Specification", text)
            self.assertIn("argmax", text)
            self.assertIn("Train/Validation/Test Protocol", text)
            self.assertIn("test labels are not used for model selection", text)
            self.assertIn("Related Work Context", text)
            self.assertIn("Statistical Procedure", text)
            self.assertIn("Trade-off Interpretation", text)
            self.assertIn("Dataset And Environment Details", text)
            self.assertNotIn("machine-generated", text.lower())

    def test_safe_rl_paper_candidate_includes_cmdp_grounding(self):
        with temporary_workdir() as workdir:
            result_dir = workdir / "artifacts" / "results"
            result_dir.mkdir(parents=True)
            (result_dir / "evidence_gate.json").write_text(json.dumps({
                "manuscript_status": "paper_ready_candidate",
                "blocking_reasons": [],
                "satisfied_requirements": ["has_benchmark_results", "has_statistical_report"],
                "next_required_experiments": [],
            }), encoding="utf-8")
            (result_dir / "statistical_report.json").write_text(json.dumps({
                "primary_metric": "safe_return",
                "metric_direction": "higher",
                "baseline_method": "reward_only",
                "best_method": "preference_cone_policy",
                "candidate_method": "preference_cone_policy",
                "absolute_best_method": "occupancy_lp_optimal",
                "summary": [{
                    "dataset": "risky_shortcut",
                    "method": "preference_cone_policy",
                    "metric": "safe_return",
                    "mean": 1.2,
                    "std": 0.1,
                    "ci_low": 1.1,
                    "ci_high": 1.3,
                    "n": 10,
                }],
                "aggregate_metric_summaries": [
                    {
                        "method": "preference_cone_policy",
                        "metric": "safe_return",
                        "mean": 1.2,
                        "std": 0.1,
                        "ci_low": 1.1,
                        "ci_high": 1.3,
                        "n": 10,
                    },
                    {
                        "method": "occupancy_lp_optimal",
                        "metric": "safe_return",
                        "mean": 1.5,
                        "std": 0.0,
                        "ci_low": 1.5,
                        "ci_high": 1.5,
                        "n": 10,
                    },
                    {
                        "method": "preference_cone_policy",
                        "metric": "reward",
                        "mean": 1.3,
                        "std": 0.1,
                        "ci_low": 1.2,
                        "ci_high": 1.4,
                        "n": 10,
                    },
                    {
                        "method": "preference_cone_policy",
                        "metric": "cost",
                        "mean": 0.4,
                        "std": 0.1,
                        "ci_low": 0.3,
                        "ci_high": 0.5,
                        "n": 10,
                    },
                    {
                        "method": "preference_cone_policy",
                        "metric": "constraint_violation",
                        "mean": 0.0,
                        "std": 0.0,
                        "ci_low": 0.0,
                        "ci_high": 0.0,
                        "n": 10,
                    },
                    {
                        "method": "preference_cone_policy",
                        "metric": "runtime_seconds",
                        "mean": 0.001,
                        "std": 0.0,
                        "ci_low": 0.001,
                        "ci_high": 0.001,
                        "n": 10,
                    },
                ],
                "comparisons": [{
                    "dataset": "risky_shortcut",
                    "baseline": "reward_only",
                    "candidate": "preference_cone_policy",
                    "metric": "safe_return",
                    "mean_delta": 0.4,
                    "paired_sign_test_p": 0.01,
                    "wins": 10,
                    "losses": 0,
                    "ties": 0,
                }],
                "metric_summaries": [{
                    "dataset": "risky_shortcut",
                    "method": "preference_cone_policy",
                    "metric": "constraint_violation",
                    "mean": 0.0,
                    "std": 0.0,
                    "ci_low": 0.0,
                    "ci_high": 0.0,
                    "n": 10,
                }],
                "ablation_summary": [
                    {
                        "ablation": "safe_return_safety_penalty_sensitivity:safety_penalty=3.00",
                        "dataset": "risky_shortcut",
                        "method": "occupancy_lp_optimal",
                        "metric": "safe_return",
                        "mean": 1.0,
                        "std": 0.0,
                        "ci_low": 1.0,
                        "ci_high": 1.0,
                        "n": 10,
                    },
                    {
                        "ablation": "safe_return_safety_penalty_sensitivity:safety_penalty=9.00",
                        "dataset": "risky_shortcut",
                        "method": "occupancy_lp_optimal",
                        "metric": "safe_return",
                        "mean": 0.8,
                        "std": 0.0,
                        "ci_low": 0.8,
                        "ci_high": 0.8,
                        "n": 10,
                    },
                ],
            }), encoding="utf-8")
            (result_dir / "lp_validation.json").write_text(json.dumps({
                "status": "ok",
                "solver_backend_cross_checks": [{"max_objective_gap": 0.0}],
                "analytic_randomized_checks": [{"status": "ok"}],
                "deterministic_reference_comparisons": [{
                    "dataset": "risky_shortcut",
                    "seed": 0,
                    "lp_flow_residual": 0.0,
                    "lp_cost_residual": 0.0,
                    "lp_objective_gap": 0.0,
                    "lp_vs_candidate_reward_gap": 0.1,
                }],
            }), encoding="utf-8")
            (result_dir / "reproduction_check.json").write_text(json.dumps({
                "status": "ok",
                "scope": "local checkout smoke reproduction; not a clean container proof",
                "checks": [{
                    "command": ["python", "-m", "unittest"],
                    "exit_code": 0,
                    "duration_seconds": 1.0,
                }],
            }), encoding="utf-8")
            benchmark_config_path = workdir / "benchmark_config.json"
            benchmark_config_path.write_text(json.dumps({
                "capability": "safe_rl_cmdp",
                "paper_title": "Validation-Selected Preference-Cone Policy Selection for Finite CMDP Benchmarks",
                "scoped_claim": "Preference-cone policy selection improves safe-return on finite CMDP benchmarks.",
                "datasets": ["risky_shortcut"],
                "methods": ["reward_only", "preference_cone_policy", "occupancy_enumeration"],
                "seeds": list(range(10)),
            }), encoding="utf-8")
            record_artifact(workdir, 9, "benchmark_config", benchmark_config_path)
            record_artifact(workdir, 9, "statistical_report", result_dir / "statistical_report.json")
            run = _run(workdir, "confirmed")
            run["baseline_metric_name"] = "safe_return"
            fake_db = FakeManuscriptDb(run, _insight(), _claims())

            with patch("agents.manuscript_writer.db", fake_db):
                generate_manuscript(9)

            paper = (workdir / "artifacts" / "manuscript" / "paper_candidate.md").read_text(encoding="utf-8")
            self.assertIn("Finite CMDP", paper)
            self.assertIn("discounted reward", paper)
            self.assertIn("constraint_violation", paper)
            self.assertIn("occupancy-enumeration", paper)
            self.assertIn("safe_return", paper)
            self.assertIn("not a deep-RL result", paper)
            self.assertIn("oracle grid-selected", paper)
            self.assertIn("cmdp_environment_appendix.json", paper)
            self.assertIn("lp_validation.json", paper)
            self.assertIn("analytic one-state randomized", paper)
            self.assertIn("reproduction_manifest.json", paper)
            self.assertIn("Artifact Hash Summary", paper)
            self.assertIn("sha256", paper)
            self.assertIn("Aggregate Method Summary", paper)
            self.assertIn("Environment-Grouped Summary", paper)
            self.assertIn("Reference And Candidate Framing", paper)
            self.assertIn("exact LP reference, not the deployable candidate", paper)
            self.assertIn("Aggregate Reward Cost And Violation Summary", paper)
            self.assertIn("Environment Metadata Summary", paper)
            self.assertIn("LP randomized improvement observed", paper)
            self.assertIn("LP Validation Summary", paper)
            self.assertIn("LP Randomization Gap Summary", paper)
            self.assertIn("Runtime Summary", paper)
            self.assertIn("Safety Penalty Sensitivity Summary", paper)
            self.assertIn("safety_penalty=9.00", paper)
            self.assertIn("Reproduction Check Summary", paper)
            self.assertIn("Configured candidate value", paper)
            self.assertNotIn("Best value:", paper)

    def test_review_revision_gate_still_writes_revised_candidate(self):
        with temporary_workdir() as workdir:
            result_dir = workdir / "artifacts" / "results"
            result_dir.mkdir(parents=True)
            (result_dir / "evidence_gate.json").write_text(json.dumps({
                "manuscript_status": "needs_more_experiments",
                "blocking_reasons": ["review_requires_revision"],
                "satisfied_requirements": [
                    "has_benchmark_results",
                    "has_baseline_comparison",
                    "has_statistical_report",
                    "has_multi_seed",
                    "has_multi_dataset",
                    "has_ablation",
                ],
                "next_required_experiments": ["Clarify text."],
            }), encoding="utf-8")
            (result_dir / "statistical_report.json").write_text(json.dumps({
                "primary_metric": "safe_return",
                "metric_direction": "higher",
                "baseline_method": "reward_only",
                "best_method": "lagrangian_penalty_4.00",
                "candidate_method": "lagrangian_penalty_4.00",
                "absolute_best_method": "occupancy_lp_optimal",
                "summary": [],
                "aggregate_metric_summaries": [],
                "comparisons": [],
            }), encoding="utf-8")
            (workdir / "benchmark_config.json").write_text(json.dumps({
                "capability": "safe_rl_cmdp",
                "paper_title": "Revised Finite CMDP Artifact Note",
                "scoped_claim": "A revised scoped claim.",
                "datasets": ["risky_shortcut"],
                "methods": ["reward_only", "lagrangian_penalty_4.00", "occupancy_lp_optimal"],
                "seeds": list(range(10)),
            }), encoding="utf-8")
            fake_db = FakeManuscriptDb(_run(workdir, "confirmed"), _insight(), _claims())

            with patch("agents.manuscript_writer.db", fake_db):
                result = generate_manuscript(9)

            manuscript_dir = workdir / "artifacts" / "manuscript"
            self.assertEqual(result["manuscript_status"], "paper_ready_candidate")
            self.assertTrue((manuscript_dir / "paper_candidate.md").exists())
            gate = json.loads((result_dir / "evidence_gate.json").read_text(encoding="utf-8"))
            self.assertEqual(gate["manuscript_status"], "paper_ready_candidate")
            self.assertEqual(gate["blocking_reasons"], [])


if __name__ == "__main__":
    unittest.main()
