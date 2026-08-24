"""Regression coverage for the exact, CPU-only bounded pilot lifecycle."""

from __future__ import annotations

import ast
import sys
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest import mock

from orchestrator import bounded_execution
from orchestrator.bounded_execution import (
    BoundedExecutionError,
    BoundedExecutionRequest,
    execute_granted_candidate,
)
from scripts import run_bounded_pilot


NOW = datetime.now(timezone.utc)
AGENDA_ID = 5
IDEA_ID = 97
GRANT_ID = 1
JOB_ID = 42
RUN_ID = 7


def _grant_row(**overrides) -> dict:
    row = {
        "id": GRANT_ID,
        "agenda_id": AGENDA_ID,
        "idea_id": IDEA_ID,
        "decision_packet_id": 1,
        "stage": "pilot",
        "token_cap": 5000,
        "gpu_class": "none",
        "max_gpu_hours": 0.0,
        "backend_allowlist_json": '["cpu", "llm"]',
        "artifact_requirements_json": '["final_results", "metrics"]',
        "expires_at": (NOW + timedelta(hours=6)).isoformat(),
        "grant_reason": "portfolio_score_selected",
        "idempotency_key": "grant:agenda-5-idea-97-pilot-1",
        "status": "active",
        "reservation_id": 2,
    }
    row.update(overrides)
    return row


def _job_row(**overrides) -> dict:
    row = {
        "id": JOB_ID,
        "agenda_id": AGENDA_ID,
        "deep_insight_id": IDEA_ID,
        "status": "queued",
        "stage": "portfolio_granted",
        "resource_grant_id": GRANT_ID,
        "experiment_run_id": None,
    }
    row.update(overrides)
    return row


def _run_row(**overrides) -> dict:
    row = {
        "id": RUN_ID,
        "agenda_id": AGENDA_ID,
        "deep_insight_id": IDEA_ID,
        "status": "scaffolding",
        "resource_grant_id": GRANT_ID,
        "scientific_evidence_state": "planned",
        "resource_class": "cpu",
        "hypothesis_verdict": "inconclusive",
    }
    row.update(overrides)
    return row


def _outcome_row(**overrides) -> dict:
    row = {
        "id": 11,
        "agenda_id": AGENDA_ID,
        "idea_id": IDEA_ID,
        "resource_grant_id": GRANT_ID,
        "experiment_run_id": RUN_ID,
        "execution_result": "completed",
        "verdict": "inconclusive",
        "state_decision": "sanity_passed",
    }
    row.update(overrides)
    return row


def _artifact(*, path: str | None = None, artifact_type: str = "final_results") -> dict:
    return {
        "id": 1,
        "artifact_type": artifact_type,
        "path": path or str(Path(__file__)),
        "metric_key": "pass_rate",
        "metric_value": 0.41,
    }


class FakeCursor:
    def __init__(self, rowcount: int):
        self.rowcount = rowcount


class FakeDb:
    """Stateful enough to exercise the lifecycle and its CAS predicates."""

    def __init__(
        self,
        *,
        grant=None,
        job=None,
        run=None,
        outcome=None,
        artifacts=None,
        claim_rows=1,
    ):
        self.grant = grant
        self.job = job
        self.run = run
        self.outcome = outcome
        self.artifacts = list(artifacts or [])
        self.claim_rows = claim_rows
        self.statements: list[tuple[str, tuple]] = []
        self.commits = 0
        self.rollbacks = 0

    def fetchone(self, sql, params=()):
        text = " ".join(sql.split()).lower()
        if "from resource_grants" in text:
            return dict(self.grant) if self.grant else None
        if "from auto_research_jobs" in text:
            return dict(self.job) if self.job else None
        if "from outcome_records" in text:
            return dict(self.outcome) if self.outcome else None
        if "select id from experiment_runs" in text:
            return {"id": self.run["id"]} if self.run else None
        if "from experiment_runs" in text:
            return dict(self.run) if self.run else None
        raise AssertionError(f"unexpected fetchone: {text}")

    def fetchall(self, sql, params=()):
        text = " ".join(sql.split()).lower()
        if "from experiment_artifacts" not in text:
            raise AssertionError(f"unexpected fetchall: {text}")
        rows = self.artifacts
        if "artifact_type='final_results'" in text:
            rows = [row for row in rows if row.get("artifact_type") == "final_results"]
        return [dict(row) for row in rows]

    def execute(self, sql, params=()):
        normalized = " ".join(sql.split())
        lower = normalized.lower()
        params = tuple(params)
        self.statements.append((normalized, params))
        if "update auto_research_jobs" not in lower:
            return FakeCursor(1)
        if "status='queued'" in lower:
            if self.claim_rows != 1:
                return FakeCursor(self.claim_rows)
            self.job.update(
                status="running_experiment",
                stage=params[0],
            )
            return FakeCursor(1)
        if "experiment_run_id is null" in lower:
            if self.job.get("experiment_run_id") is not None:
                return FakeCursor(0)
            self.job["experiment_run_id"] = params[0]
            return FakeCursor(1)
        if "set status='completed'" in lower:
            self.job.update(
                status="completed",
                stage=params[0],
                experiment_run_id=params[1],
                last_error=None,
            )
            return FakeCursor(1)
        if "set status=?, stage=?" in lower:
            self.job.update(
                status=params[0],
                stage=params[1],
                last_error=params[2],
                experiment_run_id=params[3],
            )
            return FakeCursor(1)
        return FakeCursor(1)

    def commit(self):
        self.commits += 1

    def rollback(self):
        self.rollbacks += 1


class FakeRepository:
    def __init__(
        self,
        fake_db: FakeDb,
        *,
        advance_error=None,
        assembly_error=None,
        revoke_error=None,
        revoke_result=True,
    ):
        self.db = fake_db
        self.advance_error = advance_error
        self.assembly_error = assembly_error
        self.revoke_error = revoke_error
        self.revoke_result = revoke_result
        self.advanced = []
        self.assembled = []
        self.revoked = []

    def advance_experiment_state(self, **kwargs):
        if self.advance_error:
            raise self.advance_error
        self.advanced.append(kwargs)
        self.db.run["scientific_evidence_state"] = kwargs["target"]
        return kwargs["target"]

    def assemble_and_record_outcome(self, *, resource_grant_id, experiment_run_id):
        if self.assembly_error:
            raise self.assembly_error
        self.assembled.append((resource_grant_id, experiment_run_id))
        self.db.outcome = _outcome_row(
            execution_result=self.db.run["status"],
            state_decision=self.db.run["scientific_evidence_state"],
        )
        self.db.grant["status"] = "consumed"
        return 11

    def revoke_grant(self, grant_id, *, agenda_id, reason):
        if self.revoke_error:
            raise self.revoke_error
        self.revoked.append((grant_id, agenda_id, reason))
        if self.revoke_result:
            self.db.grant["status"] = "revoked"
            self.db.job.update(status="blocked", stage="resource_grant_revoked")
        return self.revoke_result


def _request() -> BoundedExecutionRequest:
    return BoundedExecutionRequest(
        agenda_id=AGENDA_ID,
        idea_id=IDEA_ID,
        resource_grant_id=GRANT_ID,
        job_id=JOB_ID,
    )


def _execute(fake_db, repo, *, forge=None, validate=None):
    def default_forge(idea_id, grant_id):
        fake_db.run = _run_row()
        return {"run_id": RUN_ID}

    def default_validate(run_id):
        fake_db.run.update(status="completed")
        return {
            "verdict": "inconclusive",
            "baseline": 0.4,
            "best_value": 0.41,
        }

    with mock.patch.object(bounded_execution, "db", fake_db):
        return execute_granted_candidate(
            _request(),
            actor="ops:recovery",
            repository=repo,
            forge=forge or default_forge,
            validate=validate or default_validate,
        )


class ExactClaimAndBoundsTests(unittest.TestCase):
    def test_operator_cli_requires_exact_job(self):
        argv = [
            "run_bounded_pilot.py",
            "--agenda",
            str(AGENDA_ID),
            "--idea",
            str(IDEA_ID),
            "--grant",
            str(GRANT_ID),
            "--actor",
            "ops:test",
            "--dry-run",
        ]
        with mock.patch.object(sys, "argv", argv), self.assertRaises(SystemExit) as exc:
            run_bounded_pilot.main()
        self.assertEqual(exc.exception.code, 2)

    def test_request_requires_the_exact_job_id(self):
        with self.assertRaisesRegex(BoundedExecutionError, "job_id"):
            BoundedExecutionRequest(
                agenda_id=AGENDA_ID,
                idea_id=IDEA_ID,
                resource_grant_id=GRANT_ID,
            ).validate()

    def test_claim_is_scoped_to_job_agenda_idea_and_grant(self):
        fake = FakeDb(grant=_grant_row(), job=_job_row())
        with mock.patch.object(bounded_execution, "db", fake):
            claimed = bounded_execution._claim_job(_request())
        self.assertEqual(claimed["id"], JOB_ID)
        sql, params = fake.statements[0]
        self.assertIn("id=? AND agenda_id=? AND deep_insight_id=?", sql)
        self.assertIn("resource_grant_id=?", sql)
        self.assertEqual(params[2:6], (JOB_ID, AGENDA_ID, IDEA_ID, GRANT_ID))

    def test_claim_cas_failure_is_not_silently_replayed(self):
        fake = FakeDb(grant=_grant_row(), job=_job_row(), claim_rows=0)
        with mock.patch.object(bounded_execution, "db", fake):
            with self.assertRaisesRegex(BoundedExecutionError, "already_claimed"):
                bounded_execution._claim_job(_request())
        self.assertEqual(fake.rollbacks, 1)

    def test_non_replayable_job_stage_is_refused(self):
        fake = FakeDb(
            grant=_grant_row(),
            job=_job_row(stage="awaiting_portfolio_decision"),
        )
        with mock.patch.object(bounded_execution, "db", fake):
            with self.assertRaisesRegex(BoundedExecutionError, "non-replayable"):
                bounded_execution._claim_job(_request())

    def _authorize(self, grant):
        fake = FakeDb(grant=grant, job=_job_row())
        with mock.patch.object(bounded_execution, "db", fake):
            return bounded_execution._authorize_bounded_grant(_request())

    def test_gpu_backend_and_gpu_hours_are_refused(self):
        with self.assertRaisesRegex(BoundedExecutionError, "ssh_gpu"):
            self._authorize(
                _grant_row(backend_allowlist_json='["cpu", "ssh_gpu"]')
            )
        with self.assertRaisesRegex(BoundedExecutionError, "GPU-hour"):
            self._authorize(_grant_row(max_gpu_hours=0.1, gpu_class="a100"))
        with self.assertRaisesRegex(BoundedExecutionError, "GPU-class"):
            self._authorize(_grant_row(gpu_class="a100"))
        with self.assertRaisesRegex(BoundedExecutionError, "exactly cpu/llm"):
            self._authorize(_grant_row(backend_allowlist_json='["llm"]'))
        with self.assertRaisesRegex(BoundedExecutionError, "require final_results"):
            self._authorize(
                _grant_row(artifact_requirements_json='["logs", "metrics"]')
            )

    def test_non_pilot_expired_revoked_and_cross_idea_grants_are_refused(self):
        with self.assertRaisesRegex(BoundedExecutionError, "'pilot' grant"):
            self._authorize(_grant_row(stage="full_benchmark"))
        with self.assertRaisesRegex(Exception, "grant_expired"):
            self._authorize(
                _grant_row(expires_at=(NOW - timedelta(seconds=1)).isoformat())
            )
        with self.assertRaisesRegex(Exception, "grant_revoked"):
            self._authorize(_grant_row(status="revoked"))
        with self.assertRaisesRegex(Exception, "idea_scope_mismatch"):
            self._authorize(_grant_row(idea_id=IDEA_ID + 1))

    def test_exact_path_has_no_global_autonomy_or_pipeline_log_dependency(self):
        source = Path(bounded_execution.__file__).read_text("utf-8")
        tree = ast.parse(source)
        names = {
            node.id for node in ast.walk(tree) if isinstance(node, ast.Name)
        } | {node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)}
        self.assertFalse(
            names
            & {
                "AUTO_RESEARCH_ENABLED",
                "AUTO_PIPELINE_ENABLED",
                "DEEPGRAPH_AUTO_RESEARCH_ENABLED",
                "DEEPGRAPH_AUTO_PIPELINE_ENABLED",
                "log_event",
            }
        )
        imports = {
            node.module
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom) and node.module
        }
        self.assertNotIn("orchestrator.pipeline", imports)


class LifecycleTruthTests(unittest.TestCase):
    def test_success_requires_cpu_completed_run_real_final_results_and_evidence(self):
        fake = FakeDb(
            grant=_grant_row(),
            job=_job_row(),
            artifacts=[_artifact()],
        )
        repo = FakeRepository(fake)

        result = _execute(fake, repo)

        self.assertEqual(result.status, "completed")
        self.assertEqual(result.evidence_state, "sanity_passed")
        self.assertEqual(result.outcome_record_id, 11)
        self.assertEqual(fake.job["status"], "completed")
        self.assertEqual(fake.job["stage"], bounded_execution.DONE_STAGE)
        self.assertEqual(repo.assembled, [(GRANT_ID, RUN_ID)])
        self.assertEqual(repo.revoked, [])
        self.assertEqual(repo.advanced[0]["target"], "sanity_passed")
        self.assertTrue(repo.advanced[0]["context"].pilot_only)

    def test_missing_real_artifact_settles_usage_but_never_completes(self):
        fake = FakeDb(
            grant=_grant_row(),
            job=_job_row(),
            artifacts=[_artifact(path="/nonexistent/final-results.json")],
        )
        repo = FakeRepository(fake)

        result = _execute(fake, repo)

        self.assertEqual(result.status, "settled_failed")
        self.assertEqual(result.reason, "durable_success_criteria_not_met")
        self.assertEqual(fake.job["status"], "failed")
        self.assertEqual(fake.job["stage"], bounded_execution.FAILED_STAGE)
        self.assertEqual(repo.assembled, [(GRANT_ID, RUN_ID)])
        self.assertEqual(repo.advanced, [])

    def test_failed_run_is_settled_but_never_completed(self):
        fake = FakeDb(
            grant=_grant_row(),
            job=_job_row(),
            artifacts=[_artifact()],
        )
        repo = FakeRepository(fake)

        def failed_validation(run_id):
            fake.run.update(status="failed")
            return {"verdict": "blocked", "reason": "grant_required"}

        result = _execute(fake, repo, validate=failed_validation)

        self.assertEqual(result.status, "settled_failed")
        self.assertEqual(fake.job["status"], "failed")
        self.assertEqual(fake.job["stage"], bounded_execution.FAILED_STAGE)
        self.assertEqual(repo.assembled, [(GRANT_ID, RUN_ID)])
        self.assertEqual(repo.advanced, [])
        self.assertIn("blocked", result.details["validation_error"])

    def test_forge_rejection_after_run_is_settled_not_refunded(self):
        fake = FakeDb(
            grant=_grant_row(),
            job=_job_row(),
            artifacts=[_artifact(artifact_type="log")],
        )
        repo = FakeRepository(fake)

        def rejected_forge(idea_id, grant_id):
            fake.run = _run_row(status="failed")
            return {"error": "blocked: review issues", "run_id": RUN_ID}

        result = _execute(fake, repo, forge=rejected_forge)

        self.assertEqual(result.status, "settled_failed")
        self.assertEqual(repo.assembled, [(GRANT_ID, RUN_ID)])
        self.assertEqual(repo.revoked, [])
        self.assertIn("review issues", result.details["forge_error"])

    def test_forge_failure_before_run_refunds_and_fails_exact_job(self):
        fake = FakeDb(grant=_grant_row(), job=_job_row())
        repo = FakeRepository(fake)

        result = _execute(
            fake,
            repo,
            forge=lambda idea_id, grant_id: {"error": "scout unavailable"},
        )

        self.assertEqual(result.status, "failed")
        self.assertEqual(result.details["grant"], "revoked_and_refunded")
        self.assertEqual(fake.job["status"], "blocked")
        self.assertEqual(fake.job["stage"], "resource_grant_revoked")
        self.assertEqual(repo.assembled, [])

    def test_metered_no_run_that_cannot_refund_is_fail_closed(self):
        fake = FakeDb(grant=_grant_row(), job=_job_row())
        repo = FakeRepository(
            fake,
            revoke_error=RuntimeError("grant already metered usage"),
        )

        result = _execute(
            fake,
            repo,
            forge=lambda idea_id, grant_id: {"error": "boom"},
        )

        self.assertEqual(result.status, "settlement_required")
        self.assertEqual(fake.job["status"], "blocked")
        self.assertEqual(fake.job["stage"], bounded_execution.SETTLEMENT_REQUIRED_STAGE)
        self.assertIn("already metered", result.details["grant"])

    def test_settlement_required_replay_retries_release_but_never_executes(self):
        fake = FakeDb(
            grant=_grant_row(),
            job=_job_row(
                status="blocked",
                stage=bounded_execution.SETTLEMENT_REQUIRED_STAGE,
            ),
        )
        repo = FakeRepository(fake)
        forge = mock.Mock()
        validate = mock.Mock()

        result = _execute(fake, repo, forge=forge, validate=validate)

        forge.assert_not_called()
        validate.assert_not_called()
        self.assertEqual(result.status, "failed")
        self.assertEqual(result.details["grant"], "revoked_and_refunded")
        self.assertEqual(fake.job["status"], "blocked")
        self.assertEqual(fake.job["stage"], "resource_grant_revoked")

    def test_formally_revoked_terminal_job_replays_as_failed_without_work(self):
        fake = FakeDb(
            grant=_grant_row(status="revoked"),
            job=_job_row(status="blocked", stage="resource_grant_revoked"),
        )
        repo = FakeRepository(fake)
        forge = mock.Mock()
        validate = mock.Mock()

        result = _execute(fake, repo, forge=forge, validate=validate)

        forge.assert_not_called()
        validate.assert_not_called()
        self.assertEqual(result.status, "failed")
        self.assertEqual(result.details["grant"], "already_revoked")
        self.assertEqual(repo.revoked, [])

    def test_release_race_to_consumed_without_outcome_is_not_hidden_as_refund(self):
        fake = FakeDb(grant=_grant_row(), job=_job_row())
        repo = FakeRepository(fake, revoke_result=False)

        def raced_revoke(grant_id, *, agenda_id, reason):
            fake.grant["status"] = "consumed"
            return False

        repo.revoke_grant = raced_revoke
        result = _execute(
            fake,
            repo,
            forge=lambda idea_id, grant_id: {"error": "boom"},
        )

        self.assertEqual(result.status, "settlement_required")
        self.assertIn("consumed_during_unused_release", result.reason)
        self.assertEqual(fake.job["status"], "blocked")

    def test_outcome_assembly_failure_is_durable_settlement_required_not_refund(self):
        fake = FakeDb(
            grant=_grant_row(),
            job=_job_row(),
            artifacts=[_artifact()],
        )
        repo = FakeRepository(
            fake,
            assembly_error=RuntimeError("open LLM reservation"),
        )

        result = _execute(fake, repo)

        self.assertEqual(result.status, "settlement_required")
        self.assertEqual(fake.job["status"], "blocked")
        self.assertIn("open LLM reservation", result.reason)
        self.assertEqual(repo.revoked, [])

    def test_evidence_transition_failure_still_settles_metered_run_as_failure(self):
        fake = FakeDb(
            grant=_grant_row(),
            job=_job_row(),
            artifacts=[_artifact()],
        )
        repo = FakeRepository(
            fake,
            advance_error=RuntimeError("grant expired before evidence transition"),
        )

        result = _execute(fake, repo)

        self.assertEqual(result.status, "settled_failed")
        self.assertEqual(repo.assembled, [(GRANT_ID, RUN_ID)])
        self.assertEqual(fake.grant["status"], "consumed")
        self.assertEqual(fake.job["status"], "failed")
        self.assertIn("grant expired", result.details["advance_error"])

    def test_non_cpu_materialized_run_never_enters_validation_or_completes(self):
        fake = FakeDb(
            grant=_grant_row(),
            job=_job_row(),
            artifacts=[_artifact()],
        )
        repo = FakeRepository(fake)
        validation = mock.Mock()

        def gpu_forge(idea_id, grant_id):
            fake.run = _run_row(status="scaffolding", resource_class="gpu_small")
            return {"run_id": RUN_ID}

        result = _execute(fake, repo, forge=gpu_forge, validate=validation)

        validation.assert_not_called()
        self.assertEqual(result.status, "settled_failed")
        self.assertEqual(result.details["validation_skipped"], "bounded_path_refuses_non_cpu_run")
        self.assertEqual(fake.job["status"], "failed")


class CrashReplayTests(unittest.TestCase):
    def test_existing_run_is_settled_without_forge_or_validation(self):
        fake = FakeDb(
            grant=_grant_row(),
            job=_job_row(
                status="running_experiment",
                stage=bounded_execution.RUNNING_STAGE,
                experiment_run_id=RUN_ID,
            ),
            run=_run_row(
                status="completed",
                scientific_evidence_state="sanity_passed",
            ),
            artifacts=[_artifact()],
        )
        repo = FakeRepository(fake)
        forge = mock.Mock()
        validate = mock.Mock()

        result = _execute(fake, repo, forge=forge, validate=validate)

        forge.assert_not_called()
        validate.assert_not_called()
        self.assertEqual(result.status, "completed")
        self.assertEqual(result.details["replay"], "existing_run")
        self.assertEqual(repo.assembled, [(GRANT_ID, RUN_ID)])

    def test_consumed_grant_existing_outcome_repairs_job_without_any_rerun(self):
        fake = FakeDb(
            grant=_grant_row(status="consumed"),
            job=_job_row(
                status="running_experiment",
                stage=bounded_execution.RUNNING_STAGE,
                experiment_run_id=RUN_ID,
            ),
            run=_run_row(
                status="completed",
                scientific_evidence_state="sanity_passed",
            ),
            outcome=_outcome_row(),
            artifacts=[_artifact()],
        )
        repo = FakeRepository(fake)
        forge = mock.Mock()
        validate = mock.Mock()

        result = _execute(fake, repo, forge=forge, validate=validate)

        forge.assert_not_called()
        validate.assert_not_called()
        self.assertEqual(repo.assembled, [])
        self.assertEqual(result.status, "completed")
        self.assertEqual(result.details["replay"], "existing_outcome")
        self.assertEqual(fake.job["status"], "completed")

    def test_existing_failure_outcome_is_terminal_failed_not_completed(self):
        fake = FakeDb(
            grant=_grant_row(status="consumed"),
            job=_job_row(
                status="running_experiment",
                stage=bounded_execution.RUNNING_STAGE,
                experiment_run_id=RUN_ID,
            ),
            run=_run_row(status="failed"),
            outcome=_outcome_row(
                execution_result="failed",
                verdict="invalid",
                state_decision="planned",
            ),
            artifacts=[_artifact()],
        )
        repo = FakeRepository(fake)

        result = _execute(fake, repo, forge=mock.Mock(), validate=mock.Mock())

        self.assertEqual(result.status, "settled_failed")
        self.assertEqual(fake.job["status"], "failed")

    def test_invalid_outcome_verdict_is_never_reported_as_scientific_success(self):
        fake = FakeDb(
            grant=_grant_row(status="consumed"),
            job=_job_row(
                status="running_experiment",
                stage=bounded_execution.RUNNING_STAGE,
                experiment_run_id=RUN_ID,
            ),
            run=_run_row(
                status="completed",
                scientific_evidence_state="sanity_passed",
                hypothesis_verdict="invalid",
            ),
            outcome=_outcome_row(verdict="invalid"),
            artifacts=[_artifact()],
        )
        repo = FakeRepository(fake)

        result = _execute(fake, repo)

        self.assertEqual(result.status, "settled_failed")
        self.assertEqual(fake.job["status"], "failed")

    def test_existing_success_outcome_without_real_file_is_not_reported_complete(self):
        fake = FakeDb(
            grant=_grant_row(status="consumed"),
            job=_job_row(
                status="running_experiment",
                stage=bounded_execution.RUNNING_STAGE,
                experiment_run_id=RUN_ID,
            ),
            run=_run_row(
                status="completed",
                scientific_evidence_state="sanity_passed",
            ),
            outcome=_outcome_row(),
            artifacts=[_artifact(path="/nonexistent/final-results.json")],
        )
        repo = FakeRepository(fake)

        result = _execute(fake, repo)

        self.assertEqual(result.status, "settled_failed")
        self.assertEqual(fake.job["status"], "failed")


class RawArtifactHashTests(unittest.TestCase):
    def test_hash_covers_file_bytes_and_query_is_agenda_scoped(self):
        fake = FakeDb(artifacts=[_artifact()])
        with mock.patch.object(bounded_execution, "db", fake):
            with_bytes, present, missing = bounded_execution.raw_artifacts_hash(
                agenda_id=AGENDA_ID,
                experiment_run_id=RUN_ID,
            )
        fake.artifacts = [_artifact(path="/nonexistent/final-results.json")]
        with mock.patch.object(bounded_execution, "db", fake):
            without_bytes, absent_present, absent_missing = (
                bounded_execution.raw_artifacts_hash(
                    agenda_id=AGENDA_ID,
                    experiment_run_id=RUN_ID,
                )
            )

        self.assertEqual((present, missing), (1, 0))
        self.assertEqual((absent_present, absent_missing), (0, 1))
        self.assertNotEqual(with_bytes, without_bytes)
        self.assertRegex(with_bytes, r"^[0-9a-f]{64}$")
        first_sql, first_params = fake.statements[0] if fake.statements else ("", ())
        del first_sql, first_params  # reads are asserted below with a focused mock.
        with mock.patch.object(bounded_execution, "db") as mocked:
            mocked.fetchall.return_value = []
            bounded_execution.raw_artifacts_hash(
                agenda_id=AGENDA_ID,
                experiment_run_id=RUN_ID,
            )
        sql = " ".join(mocked.fetchall.call_args.args[0].split()).lower()
        self.assertIn("agenda_id=?", sql)
        self.assertEqual(mocked.fetchall.call_args.args[1], (AGENDA_ID, RUN_ID))


if __name__ == "__main__":
    unittest.main()
