"""The portfolio_granted authorization must actually be consumed.

Before this path existed, ``portfolio_granted`` had one writer and no reader
anywhere in the codebase: the meta-harness could authorize a candidate and
nothing would ever run it. Enabling global autonomy did not help, because no
loop looked at that stage either. These tests pin the wiring that closes the
gap, and the bounds that keep it from becoming a back door into autonomy.
"""

from __future__ import annotations

import ast
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


NOW = datetime.now(timezone.utc)
AGENDA_ID = 5
IDEA_ID = 97
GRANT_ID = 1
JOB_ID = 42
RUN_ID = 7


def _grant_row(**overrides) -> dict:
    values = {
        "id": GRANT_ID,
        "agenda_id": AGENDA_ID,
        "idea_id": IDEA_ID,
        "decision_packet_id": 1,
        "stage": "pilot",
        "token_cap": 5000,
        "gpu_class": "none",
        "max_gpu_hours": 0.0,
        "backend_allowlist_json": '["cpu", "llm"]',
        "artifact_requirements_json": '["logs", "metrics"]',
        "expires_at": (NOW + timedelta(hours=6)).isoformat(),
        "grant_reason": "portfolio_score_selected",
        "idempotency_key": "grant:agenda-5-idea-97-pilot-1",
        "status": "active",
        "reservation_id": 2,
    }
    values.update(overrides)
    return values


def _job_row(**overrides) -> dict:
    values = {
        "id": JOB_ID,
        "agenda_id": AGENDA_ID,
        "deep_insight_id": IDEA_ID,
        "status": "queued",
        "stage": "portfolio_granted",
        "resource_grant_id": GRANT_ID,
        "experiment_run_id": None,
    }
    values.update(overrides)
    return values


def _run_row(**overrides) -> dict:
    values = {
        "id": RUN_ID,
        "agenda_id": AGENDA_ID,
        "deep_insight_id": IDEA_ID,
        "status": "completed",
        "resource_grant_id": GRANT_ID,
        "scientific_evidence_state": "planned",
    }
    values.update(overrides)
    return values


class FakeCursor:
    def __init__(self, rowcount: int):
        self.rowcount = rowcount


class FakeDb:
    """Just enough of db.database to exercise the wiring, not the schema."""

    def __init__(self, *, grant=None, job=None, run=None, artifacts=None, claim_rows=1):
        self.grant = grant
        self.job = job
        self.run = run
        self.artifacts = artifacts if artifacts is not None else []
        self.claim_rows = claim_rows
        self.statements: list[str] = []
        self.commits = 0
        self.rollbacks = 0

    def fetchone(self, sql, params=()):
        text = " ".join(sql.split()).lower()
        if "from resource_grants" in text:
            return dict(self.grant) if self.grant else None
        if "from auto_research_jobs" in text:
            return dict(self.job) if self.job else None
        if "select id from experiment_runs" in text:
            return {"id": self.run["id"]} if self.run else None
        if "from experiment_runs" in text:
            return dict(self.run) if self.run else None
        raise AssertionError(f"unexpected fetchone: {text}")

    def fetchall(self, sql, params=()):
        text = " ".join(sql.split()).lower()
        if "from experiment_artifacts" in text:
            return [dict(row) for row in self.artifacts]
        raise AssertionError(f"unexpected fetchall: {text}")

    def execute(self, sql, params=()):
        text = " ".join(sql.split())
        self.statements.append(text)
        if "UPDATE auto_research_jobs" in text and "stage='portfolio_granted'" in text.replace(
            "stage=?", "stage=?"
        ):
            return FakeCursor(self.claim_rows)
        if "UPDATE auto_research_jobs" in text and "status='queued'" in text:
            return FakeCursor(self.claim_rows)
        return FakeCursor(1)

    def commit(self):
        self.commits += 1

    def rollback(self):
        self.rollbacks += 1


class FakeRepository:
    def __init__(self, *, outcome_id=11, advance_error=None):
        self.outcome_id = outcome_id
        self.advance_error = advance_error
        self.advanced = []
        self.assembled = []
        self.revoked = []

    def advance_experiment_state(self, **kwargs):
        if self.advance_error:
            raise self.advance_error
        self.advanced.append(kwargs)
        return kwargs["target"]

    def assemble_and_record_outcome(self, *, resource_grant_id, experiment_run_id):
        self.assembled.append((resource_grant_id, experiment_run_id))
        return self.outcome_id

    def revoke_grant(self, grant_id, *, agenda_id, reason):
        self.revoked.append((grant_id, agenda_id, reason))
        return True


def _request() -> BoundedExecutionRequest:
    return BoundedExecutionRequest(
        agenda_id=AGENDA_ID, idea_id=IDEA_ID, resource_grant_id=GRANT_ID
    )


def _run(fake_db, repo, *, forge=None, validate=None, artifact_file=None):
    forge = forge or (lambda idea_id, grant_id: {"run_id": RUN_ID})
    validate = validate or (
        lambda run_id: {"verdict": "inconclusive", "baseline": 0.4, "best_value": 0.41}
    )
    with mock.patch.object(bounded_execution, "db", fake_db):
        return execute_granted_candidate(
            _request(),
            actor="ops:recovery",
            repository=repo,
            forge=forge,
            validate=validate,
        )


class GrantedStageIsConsumedTests(unittest.TestCase):
    def test_the_granted_stage_is_read_and_claimed(self):
        """The whole point: portfolio_granted finally has a reader."""
        fake_db = FakeDb(grant=_grant_row(), job=_job_row(), run=_run_row())
        repo = FakeRepository()

        with mock.patch.object(bounded_execution, "db", fake_db):
            claimed = bounded_execution._claim_job(_request())

        self.assertEqual(claimed["id"], JOB_ID)
        claim = next(
            sql for sql in fake_db.statements if "UPDATE auto_research_jobs" in sql
        )
        self.assertIn("stage=?", claim)
        self.assertIn("status='queued'", claim)
        # Scoped: the audit rule is that every scoped-table mutation names the agenda.
        self.assertIn("agenda_id=?", claim)
        self.assertEqual(repo.advanced, [])

    def test_a_second_caller_cannot_claim_the_same_candidate(self):
        fake_db = FakeDb(grant=_grant_row(), job=_job_row(), claim_rows=0)

        with mock.patch.object(bounded_execution, "db", fake_db):
            with self.assertRaisesRegex(
                BoundedExecutionError, "granted_job_already_claimed"
            ):
                bounded_execution._claim_job(_request())

        self.assertEqual(fake_db.rollbacks, 1)
        self.assertEqual(fake_db.commits, 0)

    def test_a_job_at_another_stage_is_refused(self):
        fake_db = FakeDb(
            grant=_grant_row(), job=_job_row(stage="awaiting_portfolio_decision")
        )

        with mock.patch.object(bounded_execution, "db", fake_db):
            with self.assertRaisesRegex(BoundedExecutionError, "portfolio_granted"):
                bounded_execution._claim_job(_request())


class BoundsTests(unittest.TestCase):
    """Bounded means bounded: cpu/llm, one pilot grant, one named candidate."""

    def _authorize(self, grant_row):
        fake_db = FakeDb(grant=grant_row)
        with mock.patch.object(bounded_execution, "db", fake_db):
            return bounded_execution._authorize_bounded_grant(_request())

    def test_a_gpu_backend_is_refused_even_if_the_grant_allows_it(self):
        with self.assertRaisesRegex(BoundedExecutionError, "ssh_gpu"):
            self._authorize(
                _grant_row(backend_allowlist_json='["cpu", "llm", "ssh_gpu"]')
            )

    def test_a_gpu_hour_grant_is_refused(self):
        with self.assertRaisesRegex(BoundedExecutionError, "GPU-hour"):
            self._authorize(_grant_row(max_gpu_hours=4.0, gpu_class="a100"))

    def test_a_non_pilot_grant_is_refused(self):
        with self.assertRaisesRegex(BoundedExecutionError, "'pilot' grant"):
            self._authorize(_grant_row(stage="full_benchmark"))

    def test_an_expired_grant_is_refused_with_the_shared_reason_code(self):
        from meta_harness.grants import GrantDeniedError

        expired = _grant_row(expires_at=(NOW - timedelta(minutes=1)).isoformat())
        with self.assertRaisesRegex(GrantDeniedError, "grant_expired"):
            self._authorize(expired)

    def test_a_revoked_grant_is_refused(self):
        from meta_harness.grants import GrantDeniedError

        with self.assertRaisesRegex(GrantDeniedError, "grant_revoked"):
            self._authorize(_grant_row(status="revoked"))

    def test_a_grant_for_another_idea_is_refused(self):
        from meta_harness.grants import GrantDeniedError

        with self.assertRaisesRegex(GrantDeniedError, "idea_scope_mismatch"):
            self._authorize(_grant_row(idea_id=IDEA_ID + 1))

    def test_the_request_names_exactly_one_candidate(self):
        for field in ("agenda_id", "idea_id", "resource_grant_id"):
            with self.assertRaises(BoundedExecutionError):
                BoundedExecutionRequest(
                    **{
                        "agenda_id": AGENDA_ID,
                        "idea_id": IDEA_ID,
                        "resource_grant_id": GRANT_ID,
                        field: 0,
                    }
                ).validate()

    def test_the_path_never_consults_the_autonomy_flags(self):
        """This must not become a back door into global autonomy.

        Checked against the AST, not the text, so the module can still name the
        flags in prose while never referencing them as code.
        """
        source = Path(bounded_execution.__file__).with_suffix(".py").read_text("utf-8")
        tree = ast.parse(source)
        flags = {
            "AUTO_RESEARCH_ENABLED",
            "AUTO_PIPELINE_ENABLED",
            "DEEPGRAPH_AUTO_RESEARCH_ENABLED",
            "DEEPGRAPH_AUTO_PIPELINE_ENABLED",
        }
        referenced = {
            node.id for node in ast.walk(tree) if isinstance(node, ast.Name)
        } | {node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)}
        self.assertEqual(flags & referenced, set())
        literals = {
            node.value
            for node in ast.walk(tree)
            if isinstance(node, ast.Constant) and isinstance(node.value, str)
        }
        self.assertEqual(flags & literals, set())
        modules = {
            node.module
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom) and node.module
        }
        self.assertNotIn("orchestrator.auto_research", modules)
        self.assertNotIn("config", modules)


class FirstChainTests(unittest.TestCase):
    def test_a_successful_pilot_advances_to_sanity_passed_and_settles(self):
        artifact = Path(__file__)
        fake_db = FakeDb(
            grant=_grant_row(),
            job=_job_row(),
            run=_run_row(),
            artifacts=[
                {
                    "id": 1,
                    "artifact_type": "metrics",
                    "path": str(artifact),
                    "metric_key": "pass_rate",
                    "metric_value": 0.41,
                }
            ],
        )
        repo = FakeRepository()

        result = _run(fake_db, repo)

        self.assertEqual(result.status, "completed")
        self.assertEqual(result.evidence_state, "sanity_passed")
        self.assertEqual(result.outcome_record_id, 11)
        self.assertEqual(repo.assembled, [(GRANT_ID, RUN_ID)])
        self.assertEqual(repo.revoked, [])
        advanced = repo.advanced[0]
        self.assertEqual(advanced["target"], "sanity_passed")
        self.assertEqual(advanced["actor"], "ops:recovery")
        # A pilot may never claim a full benchmark; the state machine relies on it.
        self.assertTrue(advanced["context"].pilot_only)
        self.assertEqual(advanced["context"].resource_grant_id, GRANT_ID)
        self.assertRegex(advanced["context"].raw_artifacts_hash, r"^[0-9a-f]{64}$")
        settle = next(
            sql
            for sql in fake_db.statements
            if "UPDATE auto_research_jobs" in sql and "status='completed'" in sql
        )
        self.assertIn("agenda_id=?", settle)

    def test_a_run_without_artifact_files_still_records_an_outcome(self):
        """An unsettled grant would strand the agenda's reservation."""
        fake_db = FakeDb(
            grant=_grant_row(),
            job=_job_row(),
            run=_run_row(),
            artifacts=[
                {
                    "id": 1,
                    "artifact_type": "metrics",
                    "path": "/nonexistent/metrics.json",
                    "metric_key": "pass_rate",
                    "metric_value": None,
                }
            ],
        )
        repo = FakeRepository()

        result = _run(fake_db, repo)

        self.assertEqual(result.status, "completed")
        self.assertEqual(repo.advanced, [])
        self.assertEqual(result.evidence_state, "planned")
        self.assertEqual(result.details["not_advanced"], "no_artifact_files")
        self.assertEqual(repo.assembled, [(GRANT_ID, RUN_ID)])

    def test_an_incomplete_execution_is_recorded_not_promoted(self):
        fake_db = FakeDb(
            grant=_grant_row(),
            job=_job_row(),
            run=_run_row(status="failed"),
            artifacts=[{"id": 1, "artifact_type": "log", "path": str(Path(__file__))}],
        )
        repo = FakeRepository()

        result = _run(fake_db, repo)

        self.assertEqual(repo.advanced, [])
        self.assertEqual(result.details["not_advanced"], "execution_incomplete")
        self.assertEqual(repo.assembled, [(GRANT_ID, RUN_ID)])

    def test_a_forge_that_metered_spend_is_settled_not_revoked(self):
        """The forge creates the run before the gates that can reject it.

        A rejected experiment still holds real metered tokens, and a grant with
        metered usage cannot be revoked as unused -- so without settling it, the
        agenda's reservation is stranded with no way back.
        """
        fake_db = FakeDb(
            grant=_grant_row(),
            job=_job_row(),
            run=_run_row(status="failed"),
            artifacts=[{"id": 1, "artifact_type": "log", "path": str(Path(__file__))}],
        )
        repo = FakeRepository()

        result = _run(
            fake_db,
            repo,
            forge=lambda i, g: {"error": "blocked: 8 blocking review issues"},
        )

        self.assertEqual(result.status, "settled_without_result")
        self.assertEqual(result.experiment_run_id, RUN_ID)
        self.assertIn("8 blocking review issues", result.details["forge_error"])
        self.assertEqual(result.details["not_advanced"], "forge_rejected_the_experiment")
        # Settled, not revoked, and never promoted up the evidence ladder.
        self.assertEqual(repo.assembled, [(GRANT_ID, RUN_ID)])
        self.assertEqual(repo.advanced, [])
        self.assertEqual(repo.revoked, [])

    def test_a_forge_failure_with_no_run_blocks_the_job_and_refunds_the_grant(self):
        fake_db = FakeDb(grant=_grant_row(), job=_job_row())
        repo = FakeRepository()

        result = _run(fake_db, repo, forge=lambda i, g: {"error": "scout unavailable"})

        self.assertEqual(result.status, "failed")
        self.assertIn("scout unavailable", result.reason)
        self.assertEqual(repo.assembled, [])
        self.assertEqual(result.details["grant"], "revoked_and_refunded")
        self.assertEqual(repo.revoked[0][0], GRANT_ID)
        self.assertEqual(repo.revoked[0][1], AGENDA_ID)
        blocked = next(
            sql
            for sql in fake_db.statements
            if "UPDATE auto_research_jobs" in sql and "status='blocked'" in sql
        )
        self.assertIn("agenda_id=?", blocked)

    def test_a_blocked_validation_does_not_settle_an_outcome(self):
        fake_db = FakeDb(grant=_grant_row(), job=_job_row(), run=_run_row())
        repo = FakeRepository()

        result = _run(
            fake_db,
            repo,
            validate=lambda run_id: {"verdict": "blocked", "reason": "grant_required"},
        )

        self.assertEqual(result.status, "failed")
        self.assertIn("grant_required", result.reason)
        self.assertEqual(repo.assembled, [])

    def test_a_run_bound_to_a_different_grant_is_refused(self):
        fake_db = FakeDb(
            grant=_grant_row(), job=_job_row(), run=_run_row(resource_grant_id=99)
        )
        repo = FakeRepository()

        result = _run(fake_db, repo)

        self.assertEqual(result.status, "failed")
        self.assertIn("run_not_bound_to_grant", result.reason)
        self.assertEqual(repo.assembled, [])

    def test_a_metered_grant_that_cannot_be_revoked_is_reported_not_hidden(self):
        fake_db = FakeDb(grant=_grant_row(), job=_job_row())
        repo = FakeRepository()
        repo.revoke_grant = mock.Mock(
            side_effect=RuntimeError("grant already metered usage")
        )

        result = _run(fake_db, repo, forge=lambda i, g: {"error": "boom"})

        self.assertEqual(result.status, "failed")
        self.assertIn("not_revoked", result.details["grant"])
        self.assertIn("already metered", result.details["grant"])


class RawArtifactHashTests(unittest.TestCase):
    def test_the_hash_covers_file_bytes_not_just_row_metadata(self):
        with mock.patch.object(bounded_execution, "db") as fake:
            fake.fetchall.return_value = [
                {
                    "id": 1,
                    "artifact_type": "metrics",
                    "path": str(Path(__file__)),
                    "metric_key": "pass_rate",
                    "metric_value": 0.41,
                }
            ]
            with_bytes, present, missing = bounded_execution.raw_artifacts_hash(
                agenda_id=AGENDA_ID, experiment_run_id=RUN_ID
            )
            fake.fetchall.return_value = [
                {
                    "id": 1,
                    "artifact_type": "metrics",
                    "path": "/nonexistent/metrics.json",
                    "metric_key": "pass_rate",
                    "metric_value": 0.41,
                }
            ]
            without_bytes, absent_present, absent_missing = (
                bounded_execution.raw_artifacts_hash(
                    agenda_id=AGENDA_ID, experiment_run_id=RUN_ID
                )
            )

        self.assertEqual((present, missing), (1, 0))
        self.assertEqual((absent_present, absent_missing), (0, 1))
        self.assertNotEqual(with_bytes, without_bytes)
        self.assertRegex(with_bytes, r"^[0-9a-f]{64}$")

    def test_the_query_is_agenda_scoped(self):
        with mock.patch.object(bounded_execution, "db") as fake:
            fake.fetchall.return_value = []
            bounded_execution.raw_artifacts_hash(
                agenda_id=AGENDA_ID, experiment_run_id=RUN_ID
            )

        sql = " ".join(fake.fetchall.call_args.args[0].split()).lower()
        self.assertIn("agenda_id=?", sql)
        self.assertEqual(fake.fetchall.call_args.args[1], (AGENDA_ID, RUN_ID))


if __name__ == "__main__":
    unittest.main()
