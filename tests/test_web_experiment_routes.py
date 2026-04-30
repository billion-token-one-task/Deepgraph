import unittest
from pathlib import Path
from unittest.mock import patch

from agents.artifact_manager import artifact_path, record_artifact
from tests.temp_utils import temporary_workdir
from web.app import app


class FakeWebDb:
    def __init__(self, run):
        self.run = run

    def fetchone(self, sql, params=()):
        if "FROM experiment_runs" in sql:
            return self.run
        return None

    def fetchall(self, sql, params=()):
        return []


class FakeInsightsDb:
    def __init__(self, legacy_rows=None, deep_rows=None):
        self.legacy_rows = legacy_rows or []
        self.deep_rows = deep_rows or []

    def fetchall(self, sql, params=()):
        if "FROM insights" in sql:
            return self.legacy_rows
        if "FROM deep_insights" in sql:
            return self.deep_rows
        return []

    def fetchone(self, sql, params=()):
        return None


class ImmediateThread:
    def __init__(self, target, args=(), daemon=None):
        self.target = target
        self.args = args

    def start(self):
        self.target(*self.args)


class WebExperimentRouteTests(unittest.TestCase):
    def setUp(self):
        app.config["TESTING"] = True
        self.client = app.test_client()

    def test_manuscript_route_calls_existing_writer(self):
        with patch("agents.manuscript_writer.generate_manuscript", return_value={
            "status": "complete",
            "run_id": 12,
        }) as writer:
            response = self.client.post("/api/experiments/12/manuscript")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json()["status"], "complete")
        writer.assert_called_once_with(12)

    def test_manuscript_route_returns_404_for_missing_run(self):
        with patch("agents.manuscript_writer.generate_manuscript", return_value={
            "status": "error",
            "reason": "run_not_found",
        }):
            response = self.client.post("/api/experiments/999/manuscript")

        self.assertEqual(response.status_code, 404)

    def test_review_route_calls_existing_reviewer(self):
        with patch("agents.ai_reviewer.review_manuscript", return_value={
            "status": "complete",
            "run_id": 12,
        }) as reviewer:
            response = self.client.post("/api/experiments/12/review")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json()["status"], "complete")
        reviewer.assert_called_once_with(12)

    def test_review_route_refreshes_evidence_gate_after_successful_review(self):
        with patch("agents.ai_reviewer.review_manuscript", return_value={
            "status": "complete",
            "run_id": 12,
        }) as reviewer, \
             patch("agents.evidence_gate.write_evidence_gate", return_value={
                 "manuscript_status": "paper_ready_candidate",
             }) as gate:
            response = self.client.post("/api/experiments/12/review")

        self.assertEqual(response.status_code, 200)
        reviewer.assert_called_once_with(12)
        gate.assert_called_once_with(12)

    def test_review_route_refuses_when_manuscript_missing(self):
        with patch("agents.ai_reviewer.review_manuscript", return_value={
            "status": "error",
            "reason": "manuscript_not_found",
        }):
            response = self.client.post("/api/experiments/12/review")

        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.get_json()["reason"], "manuscript_not_found")

    def test_plan_followup_route_calls_existing_planner(self):
        with patch("agents.review_planner.plan_followup_experiments", return_value={
            "status": "needs_followup",
            "run_id": 12,
            "experiments": [],
        }) as planner:
            response = self.client.post("/api/experiments/12/plan_followup")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json()["status"], "needs_followup")
        planner.assert_called_once_with(12)

    def test_plan_followup_route_returns_400_when_review_missing(self):
        with patch("agents.review_planner.plan_followup_experiments", return_value={
            "status": "error",
            "reason": "review_not_found",
        }):
            response = self.client.post("/api/experiments/12/plan_followup")

        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.get_json()["reason"], "review_not_found")

    def test_experiment_detail_includes_artifacts(self):
        with temporary_workdir() as workdir:
            metrics = artifact_path(workdir, "artifacts/results/metrics.json")
            metrics.parent.mkdir(parents=True)
            metrics.write_text('{"baseline": 0.7}', encoding="utf-8")
            record_artifact(workdir, 12, "metrics", metrics)
            fake_db = FakeWebDb({
                "id": 12,
                "deep_insight_id": 4,
                "workdir": str(workdir),
                "insight_title": "Toy",
                "insight_tier": 2,
            })

            with patch("web.app.db", fake_db):
                response = self.client.get("/api/experiments/12")

        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertIn("artifacts", payload)
        self.assertEqual(payload["artifacts"][0]["path"], "artifacts/results/metrics.json")

    def test_experiment_detail_includes_evidence_gate_and_followup_artifacts(self):
        with temporary_workdir() as workdir:
            evidence = artifact_path(workdir, "artifacts/results/evidence_gate.json")
            followup = artifact_path(workdir, "artifacts/results/followup_experiment_plan.json")
            evidence.parent.mkdir(parents=True)
            evidence.write_text('{"manuscript_status":"needs_more_experiments"}', encoding="utf-8")
            followup.write_text('{"status":"needs_followup"}', encoding="utf-8")
            record_artifact(workdir, 12, "evidence_gate", evidence, {
                "manuscript_status": "needs_more_experiments",
            })
            record_artifact(workdir, 12, "followup_experiment_plan", followup, {
                "status": "needs_followup",
                "experiment_count": 2,
            })
            fake_db = FakeWebDb({
                "id": 12,
                "deep_insight_id": 4,
                "workdir": str(workdir),
                "insight_title": "Toy",
                "insight_tier": 2,
            })

            with patch("web.app.db", fake_db):
                response = self.client.get("/api/experiments/12")

        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        artifacts = {item["path"]: item for item in payload["artifacts"]}
        self.assertIn("artifacts/results/evidence_gate.json", artifacts)
        self.assertIn("artifacts/results/followup_experiment_plan.json", artifacts)
        self.assertEqual(
            artifacts["artifacts/results/evidence_gate.json"]["metadata"]["manuscript_status"],
            "needs_more_experiments",
        )

    def test_run_full_does_not_generate_manuscript_or_review_by_default(self):
        with patch("web.app.threading.Thread", ImmediateThread), \
             patch("agents.experiment_forge.forge_experiment", return_value={"run_id": 88}), \
             patch("agents.validation_loop.run_validation_loop", return_value={"verdict": "confirmed"}), \
             patch("agents.knowledge_loop.process_completed_run", return_value={"verdict": "confirmed"}), \
             patch("agents.manuscript_writer.generate_manuscript") as writer, \
             patch("agents.ai_reviewer.review_manuscript") as reviewer:
            response = self.client.post("/api/experiments/run_full", json={"insight_id": 5})

        self.assertEqual(response.status_code, 200)
        writer.assert_not_called()
        reviewer.assert_not_called()

    def test_run_full_optionally_generates_manuscript_review_and_refreshes_gate(self):
        with patch("web.app.threading.Thread", ImmediateThread), \
             patch("agents.experiment_forge.forge_experiment", return_value={"run_id": 89}), \
             patch("agents.validation_loop.run_validation_loop", return_value={"verdict": "confirmed"}), \
             patch("agents.knowledge_loop.process_completed_run", return_value={"verdict": "confirmed"}), \
             patch("agents.manuscript_writer.generate_manuscript", return_value={"status": "complete"}) as writer, \
             patch("agents.ai_reviewer.review_manuscript", return_value={"status": "complete"}) as reviewer, \
             patch("agents.evidence_gate.write_evidence_gate", return_value={
                 "manuscript_status": "paper_ready_candidate",
             }) as gate:
            response = self.client.post(
                "/api/experiments/run_full",
                json={"insight_id": 5, "generate_manuscript": True, "review": True},
            )

        self.assertEqual(response.status_code, 200)
        writer.assert_called_once_with(89)
        reviewer.assert_called_once_with(89)
        gate.assert_called_once_with(89)

    def test_insights_route_includes_normalized_deep_insights(self):
        fake_db = FakeInsightsDb(deep_rows=[{
            "id": 7,
            "tier": 1,
            "status": "experimentally_confirmed",
            "title": "Unified CMDP fairness insight",
            "formal_structure": "Finite CMDP with preference cone constraints.",
            "transformation": "Map fairness constraints to CMDP costs.",
            "predictions": '["policy randomization improves feasible reward"]',
            "falsification": "No feasible reward gap exists.",
            "adversarial_score": 8.0,
            "adversarial_critique": "",
            "supporting_papers": '["2401.00001"]',
            "source_node_ids": '["rl.safe"]',
            "evidence_summary": "Safe RL and fairness share constrained optimization structure.",
            "novelty_status": "novel",
            "created_at": "2026-04-29 10:33:22",
            "updated_at": "2026-04-30 09:00:00",
        }])

        with patch("web.app.db", fake_db):
            response = self.client.get("/api/insights?limit=10")

        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertEqual(len(payload), 1)
        self.assertEqual(payload[0]["source_table"], "deep_insights")
        self.assertEqual(payload[0]["deep_insight_id"], 7)
        self.assertEqual(payload[0]["insight_type"], "paradigm_discovery")
        self.assertEqual(payload[0]["node_id"], "rl.safe")
        self.assertIn("Finite CMDP", payload[0]["hypothesis"])

    def test_insights_route_preserves_legacy_rows(self):
        legacy = {
            "id": 3,
            "node_id": "ml",
            "insight_type": "method_transfer",
            "title": "Legacy insight",
            "hypothesis": "Transfer method.",
            "evidence": "Evidence",
            "experiment": "Experiment",
            "impact": "Impact",
            "novelty_score": 4,
            "feasibility_score": 3,
            "supporting_papers": "[]",
            "created_at": "2026-04-29 10:00:00",
            "updated_at": "2026-04-29 10:00:00",
        }

        with patch("web.app.db", FakeInsightsDb(legacy_rows=[legacy])):
            response = self.client.get("/api/insights?limit=10")

        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertEqual(payload[0]["source_table"], "insights")
        self.assertEqual(payload[0]["id"], 3)


if __name__ == "__main__":
    unittest.main()
