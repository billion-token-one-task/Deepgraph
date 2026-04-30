import json
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from agents.experiment_forge import forge_experiment
from agents.validation_loop import run_validation_loop
from agents.knowledge_loop import process_completed_run
from agents.manuscript_writer import generate_manuscript
from agents.ai_reviewer import review_manuscript
from agents.review_planner import plan_followup_experiments
from tests.temp_utils import temporary_workdir


class ToyResearchDb:
    def __init__(self, insight):
        self.insight = dict(insight)
        self.run = None
        self.iterations = []
        self.claims = []

    def fetchone(self, sql, params=()):
        if "FROM deep_insights" in sql:
            return self.insight
        if "FROM experiment_runs" in sql:
            return self.run
        if "SELECT id FROM experimental_claims WHERE run_id" in sql:
            run_id, insight_id = params
            for claim in self.claims:
                if claim["run_id"] == run_id and claim["deep_insight_id"] == insight_id:
                    return {"id": claim["id"]}
            return None
        return None

    def fetchall(self, sql, params=()):
        if "FROM experiment_iterations" in sql and "phase='reproduction'" in sql:
            return [
                {"metric_value": item["metric_value"]}
                for item in self.iterations
                if item["phase"] == "reproduction" and item["metric_value"] is not None
            ]
        if "FROM experiment_iterations" in sql and "phase='hypothesis_testing'" in sql:
            return [
                {
                    "iteration_number": item["iteration_number"],
                    "metric_value": item["metric_value"],
                    "status": item["status"],
                    "description": item["description"],
                    "code_diff": item.get("code_diff", ""),
                }
                for item in self.iterations
                if item["phase"] == "hypothesis_testing"
            ]
        if "FROM experimental_claims WHERE run_id" in sql:
            run_id = params[0]
            if "SELECT id" in sql:
                return [{"id": c["id"]} for c in self.claims if c["run_id"] == run_id and not c.get("cascaded")]
            return [c for c in self.claims if c["run_id"] == run_id]
        return []

    def execute(self, sql, params=()):
        stripped = sql.strip()
        if stripped.startswith("INSERT INTO experiment_runs"):
            self.run = {
                "id": 501,
                "deep_insight_id": params[0],
                "status": params[1],
                "phase": params[2],
                "workdir": params[3],
                "codebase_url": params[4],
                "codebase_ref": params[5],
                "program_md": params[6],
                "proxy_config": params[7],
                "success_criteria": params[8],
                "baseline_metric_name": params[9],
                "baseline_metric_value": None,
                "best_metric_value": None,
                "hypothesis_verdict": None,
                "effect_size": None,
                "effect_pct": None,
                "iterations_total": 0,
                "iterations_kept": 0,
            }
            return SimpleNamespace(lastrowid=501)
        if stripped.startswith("INSERT INTO experiment_iterations"):
            if params[2] == "hypothesis_testing":
                self.iterations.append({
                    "run_id": params[0],
                    "iteration_number": params[1],
                    "phase": params[2],
                    "code_diff": params[3],
                    "commit_hash": params[4],
                    "metric_value": params[5],
                    "metric_name": params[6],
                    "peak_memory_mb": params[7],
                    "duration_seconds": params[8],
                    "status": params[9],
                    "description": params[10],
                })
            else:
                self.iterations.append({
                    "run_id": params[0],
                    "iteration_number": params[1],
                    "phase": params[2],
                    "metric_value": params[3],
                    "metric_name": params[4],
                    "peak_memory_mb": params[5],
                    "duration_seconds": params[6],
                    "status": params[7],
                    "description": params[8],
                    "code_diff": "",
                })
            return SimpleNamespace(lastrowid=len(self.iterations))
        if stripped.startswith("INSERT INTO experimental_claims"):
            self.claims.append({
                "id": len(self.claims) + 1,
                "run_id": params[0],
                "deep_insight_id": params[1],
                "claim_text": params[2],
                "claim_type": params[3],
                "verdict": params[4],
                "effect_size": params[5],
                "confidence": params[6],
                "p_value": params[7],
                "supporting_data": params[8],
                "cascaded": 0,
            })
            return SimpleNamespace(lastrowid=len(self.claims))
        if stripped.startswith("UPDATE experiment_runs SET status='reproducing'"):
            self.run["status"] = "reproducing"
            self.run["phase"] = "reproduction"
        elif stripped.startswith("UPDATE experiment_runs") and "status='completed'" in sql:
            self.run["status"] = "completed"
            self.run["phase"] = "benchmark_suite" if "benchmark_suite" in sql else self.run.get("phase")
            self.run["hypothesis_verdict"] = params[0]
            if "baseline_metric_value" in sql and len(params) >= 6:
                self.run["baseline_metric_value"] = params[1]
                self.run["best_metric_value"] = params[2]
                self.run["effect_size"] = params[3]
                self.run["effect_pct"] = params[4]
            else:
                self.run["effect_size"] = params[1]
                self.run["effect_pct"] = params[2]
        elif "baseline_metric_value" in sql and "best_metric_value" in sql:
            self.run["baseline_metric_value"] = params[0]
            self.run["best_metric_value"] = params[1]
            self.run["phase"] = "hypothesis_testing"
            self.run["status"] = "testing"
        elif "iterations_total" in sql and "iterations_kept" in sql:
            self.run["iterations_total"] = params[0]
            self.run["iterations_kept"] = params[1]
            self.run["best_metric_value"] = params[2]
            self.run["effect_size"] = params[3]
            self.run["effect_pct"] = params[4]
        elif stripped.startswith("UPDATE experiment_runs SET hypothesis_verdict"):
            self.run["hypothesis_verdict"] = params[0]
            self.run["effect_size"] = params[1]
            self.run["effect_pct"] = params[2]
        elif stripped.startswith("UPDATE deep_insights"):
            if params:
                self.insight["status"] = params[0]
        return SimpleNamespace(lastrowid=None)

    def commit(self):
        return None


class E2EToyResearchPipelineTests(unittest.TestCase):
    def test_toy_pipeline_reaches_review_without_network(self):
        insight = {
            "id": 77,
            "tier": 2,
            "title": "Toy Calibration Insight",
            "hypothesis": "A tiny deterministic baseline can be validated.",
            "problem_statement": "Toy models need auditable validation.",
            "experimental_plan": json.dumps({"metrics": {"primary": "accuracy", "direction": "higher"}}),
            "supporting_papers": json.dumps([{"id": "toy-paper", "title": "Toy Paper", "authors": ["T. Author"]}]),
        }
        toy_db = ToyResearchDb(insight)

        with temporary_workdir() as workdir:
            code_dir = workdir / "code"
            code_dir.mkdir()
            (code_dir / "train.py").write_text("print('accuracy: 0.8')\n", encoding="utf-8")

            def scaffold(_insight, _codebase, _workdir):
                (_workdir / "success_criteria.json").write_text(json.dumps({
                    "metric_name": "accuracy",
                    "metric_direction": "higher",
                    "solid": 0.75,
                }), encoding="utf-8")
                return {
                    "program_md": "Run toy validation.",
                    "success_criteria": {"metric_name": "accuracy", "metric_direction": "higher", "solid": 0.75},
                    "tokens": 0,
                }

            def improve_toy(_workdir, _code_dir, *_args):
                (_code_dir / "train.py").write_text("print('accuracy: 0.9')\n", encoding="utf-8")
                return "Improved toy baseline"

            with patch("agents.experiment_forge.db", toy_db), \
                 patch("agents.validation_loop.db", toy_db), \
                 patch("agents.result_interpreter.db", toy_db), \
                 patch("agents.knowledge_loop.db", toy_db), \
                 patch("agents.manuscript_writer.db", toy_db), \
                 patch("agents.ai_reviewer.db", toy_db), \
                 patch("agents.experiment_forge.scout_codebase", return_value={"url": "scratch", "name": "toy"}), \
                 patch("agents.experiment_forge.setup_workspace", return_value=workdir), \
                 patch("agents.experiment_forge.generate_scaffold", side_effect=scaffold), \
                 patch("agents.experiment_forge.build_proxy_config", return_value={
                     "time_budget_seconds": 10,
                     "max_iterations": 1,
                     "reproduction_iterations": 1,
                     "refute_min_iterations": 1,
                 }), \
                 patch("agents.validation_loop._launch_coding_agent", side_effect=improve_toy), \
                 patch("agents.knowledge_loop.cascade_from_claim"), \
                 patch("agents.knowledge_loop.update_track_record"), \
                 patch("agents.ai_reviewer.call_llm_json", return_value=({
                     "overall_score": 7,
                     "recommendation": "borderline",
                     "major_concerns": [],
                     "minor_concerns": [],
                     "required_experiments": [],
                     "citation_risks": [],
                     "reproducibility_risks": [],
                 }, 42)):
                forge_result = forge_experiment(77)
                run_id = forge_result["run_id"]
                loop_result = run_validation_loop(run_id)
                processed = process_completed_run(run_id)
                manuscript = generate_manuscript(run_id)
                review = review_manuscript(run_id)

            self.assertEqual(loop_result["verdict"], "confirmed")
            self.assertEqual(processed["verdict"], "confirmed")
            self.assertEqual(manuscript["status"], "complete")
            self.assertEqual(review["status"], "complete")
            self.assertTrue((workdir / "artifacts" / "results" / "metrics.json").exists())
            self.assertTrue((workdir / "artifacts" / "manuscript" / "preliminary_report.md").exists())
            self.assertTrue((workdir / "artifacts" / "reviews" / "review.json").exists())

    def test_fairness_spec_pipeline_writes_benchmark_evidence_and_followup_plan(self):
        insight = {
            "id": 78,
            "tier": 1,
            "title": "Preference-cone fairness improves grouped classification",
            "hypothesis": "Preference-cone thresholding improves fairness score on grouped classification.",
            "problem_statement": "Group fairness classifiers need auditable multi-seed benchmarks.",
            "experimental_plan": json.dumps({"metrics": {"primary": "fairness_score", "direction": "higher"}}),
            "supporting_papers": json.dumps([{
                "id": "fairness-paper",
                "title": "Fairness Paper",
                "authors": ["F. Author"],
            }]),
        }
        toy_db = ToyResearchDb(insight)

        with temporary_workdir() as workdir:
            def scaffold(_insight, _codebase, _workdir):
                (_workdir / "success_criteria.json").write_text(json.dumps({
                    "metric_name": "fairness_score",
                    "metric_direction": "higher",
                    "solid": 0.01,
                }), encoding="utf-8")
                return {
                    "program_md": "Run benchmark suite.",
                    "success_criteria": {"metric_name": "fairness_score", "metric_direction": "higher", "solid": 0.01},
                    "tokens": 0,
                }

            with patch("agents.experiment_forge.db", toy_db), \
                 patch("agents.validation_loop.db", toy_db), \
                 patch("agents.benchmark_suite.db", toy_db), \
                 patch("agents.statistical_reporter.db", toy_db), \
                 patch("agents.evidence_gate.db", toy_db), \
                 patch("agents.result_interpreter.db", toy_db), \
                 patch("agents.knowledge_loop.db", toy_db), \
                 patch("agents.manuscript_writer.db", toy_db), \
                 patch("agents.ai_reviewer.db", toy_db), \
                 patch("agents.review_planner.db", toy_db), \
                 patch("agents.experiment_forge.scout_codebase", return_value={"url": "scratch", "name": "fairness"}), \
                 patch("agents.experiment_forge.setup_workspace", return_value=workdir), \
                 patch("agents.experiment_forge.generate_scaffold", side_effect=scaffold), \
                 patch("agents.experiment_forge.build_proxy_config", return_value={
                     "time_budget_seconds": 10,
                     "max_iterations": 1,
                     "reproduction_iterations": 1,
                     "refute_min_iterations": 1,
                 }), \
                 patch("agents.knowledge_loop.cascade_from_claim"), \
                 patch("agents.knowledge_loop.update_track_record"), \
                 patch("agents.ai_reviewer.call_llm_json", return_value=({
                     "overall_score": 2,
                     "recommendation": "reject",
                     "major_concerns": ["Only a synthetic grouped benchmark is present."],
                     "minor_concerns": [],
                     "required_experiments": ["Add a second offline grouped dataset."],
                     "citation_risks": [],
                     "reproducibility_risks": [],
                 }, 42)):
                forge_result = forge_experiment(78)
                run_id = forge_result["run_id"]
                loop_result = run_validation_loop(run_id)
                processed = process_completed_run(run_id)
                manuscript = generate_manuscript(run_id)
                review = review_manuscript(run_id)
                followup = plan_followup_experiments(run_id)

            self.assertEqual(forge_result["execution_mode"], "benchmark_suite")
            self.assertIn(loop_result["verdict"], {"confirmed", "inconclusive"})
            self.assertEqual(processed["verdict"], loop_result["verdict"])
            self.assertEqual(manuscript["status"], "complete")
            self.assertEqual(review["status"], "complete")
            self.assertEqual(followup["status"], "needs_followup")
            self.assertTrue((workdir / "research_spec.json").exists())
            self.assertTrue((workdir / "benchmark_config.json").exists())
            self.assertTrue((workdir / "artifacts" / "results" / "benchmark_results.json").exists())
            self.assertTrue((workdir / "artifacts" / "results" / "statistical_report.json").exists())
            self.assertTrue((workdir / "artifacts" / "results" / "evidence_gate.json").exists())
            self.assertTrue((workdir / "artifacts" / "reviews" / "review.json").exists())
            self.assertTrue((workdir / "artifacts" / "results" / "followup_experiment_plan.json").exists())


if __name__ == "__main__":
    unittest.main()
