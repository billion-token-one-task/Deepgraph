import json
import unittest
from pathlib import Path
from unittest.mock import patch

from agents.result_interpreter import interpret_run
from agents.knowledge_loop import process_completed_run
from tests.temp_utils import temporary_workdir


class FakeResultDb:
    def __init__(self, run: dict, insight: dict | None = None,
                 repro: list[dict] | None = None, tests: list[dict] | None = None):
        self.run = run
        self.insight = insight or {"id": run["deep_insight_id"], "title": "Toy insight", "tier": 2}
        self.repro = repro or []
        self.tests = tests or []
        self.claims = []
        self.deep_status = None

    def fetchone(self, sql, params=()):
        if "FROM experiment_runs" in sql:
            return self.run
        if "FROM deep_insights" in sql:
            return self.insight
        if "FROM experimental_claims WHERE run_id" in sql:
            run_id, insight_id = params
            for claim in self.claims:
                if claim["run_id"] == run_id and claim["deep_insight_id"] == insight_id:
                    return {"id": claim["id"]}
            return None
        if "FROM experimental_claims WHERE id" in sql:
            claim_id = params[0]
            for claim in self.claims:
                if claim["id"] == claim_id:
                    return claim
            return None
        return None

    def fetchall(self, sql, params=()):
        if "FROM experiment_iterations" in sql and "phase='reproduction'" in sql:
            return self.repro
        if "FROM experiment_iterations" in sql and "phase='hypothesis_testing'" in sql:
            return self.tests
        if "FROM experimental_claims WHERE run_id" in sql:
            run_id = params[0]
            return [{"id": c["id"]} for c in self.claims if c["run_id"] == run_id and not c.get("cascaded")]
        return []

    def execute(self, sql, params=()):
        stripped = sql.strip()
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
        elif stripped.startswith("UPDATE experimental_claims"):
            claim_id = params[-1]
            for claim in self.claims:
                if claim["id"] == claim_id:
                    claim.update({
                        "claim_text": params[0],
                        "verdict": params[1],
                        "effect_size": params[2],
                        "confidence": params[3],
                        "p_value": params[4],
                        "supporting_data": params[5],
                    })
        elif stripped.startswith("UPDATE deep_insights"):
            self.deep_status = params[0]
        elif stripped.startswith("UPDATE experiment_runs"):
            self.run["hypothesis_verdict"] = params[0]
            self.run["effect_size"] = params[1]
            self.run["effect_pct"] = params[2]
        return None

    def commit(self):
        return None


def _run(workdir: Path, verdict: str | None = "confirmed") -> dict:
    return {
        "id": 1,
        "deep_insight_id": 2,
        "workdir": str(workdir),
        "baseline_metric_name": "accuracy",
        "baseline_metric_value": 0.7,
        "best_metric_value": 0.8 if verdict != "refuted" else 0.6,
        "hypothesis_verdict": verdict,
        "success_criteria": json.dumps({"metric_direction": "higher"}),
    }


class ResultInterpreterTests(unittest.TestCase):
    def test_confirmed_run_creates_experimental_claim(self):
        with temporary_workdir() as tmp:
            fake_db = FakeResultDb(
                _run(tmp, "confirmed"),
                repro=[{"metric_value": 0.7}],
                tests=[{"iteration_number": 2, "metric_value": 0.8, "status": "keep", "description": "improved", "code_diff": ""}],
            )

            with patch("agents.result_interpreter.db", fake_db):
                result = interpret_run(1)

            self.assertEqual(result["verdict"], "confirmed")
            self.assertEqual(len(fake_db.claims), 1)
            self.assertEqual(fake_db.claims[0]["verdict"], "confirmed")

    def test_refuted_run_creates_refuted_claim(self):
        with temporary_workdir() as tmp:
            fake_db = FakeResultDb(
                _run(tmp, "refuted"),
                repro=[{"metric_value": 0.7}],
                tests=[{"iteration_number": 2, "metric_value": 0.6, "status": "discard", "description": "worse", "code_diff": ""}],
            )

            with patch("agents.result_interpreter.db", fake_db):
                result = interpret_run(1)

            self.assertEqual(result["verdict"], "refuted")
            self.assertEqual(fake_db.claims[0]["verdict"], "refuted")

    def test_missing_metrics_returns_inconclusive_reason(self):
        with temporary_workdir() as tmp:
            run = _run(tmp, None)
            run["baseline_metric_value"] = None
            run["best_metric_value"] = None
            fake_db = FakeResultDb(run)

            with patch("agents.result_interpreter.db", fake_db):
                result = interpret_run(1)

            self.assertEqual(result["verdict"], "inconclusive")
            self.assertEqual(result["reason"], "missing_metrics")

    def test_confirmed_without_kept_improvement_is_downgraded(self):
        with temporary_workdir() as tmp:
            fake_db = FakeResultDb(
                _run(tmp, "confirmed"),
                repro=[{"metric_value": 0.8}],
                tests=[],
            )
            fake_db.run["baseline_metric_value"] = 0.8
            fake_db.run["best_metric_value"] = 0.8

            with patch("agents.result_interpreter.db", fake_db):
                result = interpret_run(1)

            self.assertEqual(result["verdict"], "inconclusive")
            self.assertEqual(fake_db.claims[0]["verdict"], "inconclusive")

    def test_artifact_db_mismatch_returns_error_without_claim(self):
        with temporary_workdir() as workdir:
            metrics_dir = workdir / "artifacts" / "results"
            metrics_dir.mkdir(parents=True)
            (metrics_dir / "metrics.json").write_text(json.dumps({
                "baseline": 0.1,
                "best_value": 0.2,
                "verdict": "confirmed",
            }), encoding="utf-8")
            fake_db = FakeResultDb(_run(workdir, "confirmed"), repro=[{"metric_value": 0.7}])

            with patch("agents.result_interpreter.db", fake_db):
                result = interpret_run(1)

            self.assertEqual(result["status"], "error")
            self.assertEqual(result["reason"], "artifact_db_mismatch")
            self.assertEqual(fake_db.claims, [])

    def test_process_completed_run_is_idempotent(self):
        with temporary_workdir() as tmp:
            fake_db = FakeResultDb(
                _run(tmp, "confirmed"),
                repro=[{"metric_value": 0.7}],
                tests=[{"iteration_number": 2, "metric_value": 0.8, "status": "keep", "description": "improved", "code_diff": ""}],
            )

            with patch("agents.result_interpreter.db", fake_db), \
                 patch("agents.knowledge_loop.db", fake_db), \
                 patch("agents.knowledge_loop.cascade_from_claim"), \
                 patch("agents.knowledge_loop.update_track_record"):
                first = process_completed_run(1)
                second = process_completed_run(1)

            self.assertEqual(first["verdict"], "confirmed")
            self.assertEqual(second["verdict"], "confirmed")
            self.assertEqual(len(fake_db.claims), 1)

    def test_benchmark_claim_uses_scoped_claim_instead_of_original_title(self):
        with temporary_workdir() as workdir:
            result_dir = workdir / "artifacts" / "results"
            result_dir.mkdir(parents=True)
            (result_dir / "evidence_gate.json").write_text(json.dumps({
                "manuscript_status": "paper_ready_candidate",
                "blocking_reasons": [],
            }), encoding="utf-8")
            (result_dir / "statistical_report.json").write_text(json.dumps({
                "baseline_method": "logistic_regression",
                "best_method": "preference_cone_threshold",
                "comparisons": [{
                    "dataset": "synthetic_grouped",
                    "candidate": "preference_cone_threshold",
                    "paired_sign_test_p": 0.01,
                }],
            }), encoding="utf-8")
            (workdir / "benchmark_config.json").write_text(json.dumps({
                "scoped_claim": "Preference-cone thresholding improves a fairness-weighted objective across grouped classification benchmarks.",
            }), encoding="utf-8")
            fake_db = FakeResultDb(
                _run(workdir, "confirmed"),
                insight={"id": 2, "title": "Social-choice CMDPs: safe RL and group fairness", "tier": 2},
                repro=[{"metric_value": 0.7}],
                tests=[],
            )

            with patch("agents.result_interpreter.db", fake_db):
                result = interpret_run(1)

            self.assertEqual(result["verdict"], "confirmed")
            self.assertIn("Preference-cone thresholding improves", result["claim_text"])
            self.assertIn("configured candidate method", result["claim_text"])
            self.assertNotIn("best configured method", result["claim_text"])
            self.assertNotIn("Social-choice CMDPs", result["claim_text"])
            self.assertNotIn("paired sign test p=", result["claim_text"])

    def test_benchmark_interpretation_uses_statistical_report_metric_direction(self):
        with temporary_workdir() as workdir:
            result_dir = workdir / "artifacts" / "results"
            result_dir.mkdir(parents=True)
            (result_dir / "evidence_gate.json").write_text(json.dumps({
                "manuscript_status": "paper_ready_candidate",
                "blocking_reasons": [],
            }), encoding="utf-8")
            (result_dir / "statistical_report.json").write_text(json.dumps({
                "primary_metric": "fairness_score",
                "metric_direction": "higher",
                "baseline_method": "logistic_regression",
                "best_method": "preference_cone_threshold",
                "comparisons": [{
                    "dataset": "synthetic_grouped",
                    "candidate": "preference_cone_threshold",
                    "paired_sign_test_p": 0.01,
                }],
            }), encoding="utf-8")
            fake_db = FakeResultDb(
                _run(workdir, "confirmed"),
                repro=[],
                tests=[],
            )
            fake_db.run["baseline_metric_name"] = "social_choice_score"
            fake_db.run["baseline_metric_value"] = 0.60
            fake_db.run["best_metric_value"] = 0.75
            fake_db.run["success_criteria"] = json.dumps({"metric_direction": "lower"})

            with patch("agents.result_interpreter.db", fake_db):
                result = interpret_run(1)

            self.assertEqual(result["verdict"], "confirmed")
            self.assertGreater(result["effect_size"], 0)
            self.assertEqual(fake_db.claims[0]["verdict"], "confirmed")


if __name__ == "__main__":
    unittest.main()
