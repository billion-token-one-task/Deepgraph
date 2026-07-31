"""Agenda-scoped persistence for Harness Evolution evaluation artifacts."""

from __future__ import annotations

import json
from typing import Any

from db import database as db
from meta_harness.harness_evolution import (
    EvaluationRun,
    FailureCluster,
    HarnessArchive,
    HarnessCandidate,
    HarnessPatch,
    HarnessPolicy,
    RegressionReport,
    validate_candidate,
    validate_patch,
)
from meta_harness.reviewer_approval import (
    ReviewerApprovalVerifier,
    harness_candidate_subject,
)


class HarnessPersistenceError(RuntimeError):
    pass


def _dump(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)


class HarnessRepository:
    def save_candidate(
        self,
        candidate: HarnessCandidate,
        *,
        policy: HarnessPolicy,
        production_path: str,
        production_database_namespace: str,
        parent_archive_id: int | None = None,
    ) -> int:
        validate_candidate(
            candidate,
            policy=policy,
            production_path=production_path,
            production_database_namespace=production_database_namespace,
        )
        if parent_archive_id is not None:
            archive = db.fetchone(
                "SELECT agenda_id FROM harness_archives WHERE id=?",
                (int(parent_archive_id),),
            )
            if (
                not archive
                or int(archive.get("agenda_id") or 0) != candidate.agenda_id
            ):
                raise HarnessPersistenceError("parent archive scope mismatch")
        existing = db.fetchone(
            """
            SELECT * FROM harness_candidates
            WHERE candidate_ref=?
            """,
            (candidate.candidate_ref,),
        )
        if existing:
            same = (
                int(existing.get("agenda_id") or 0) == candidate.agenda_id
                and existing.get("base_commit") == candidate.base_commit
                and existing.get("worktree_path") == candidate.worktree_path
                and existing.get("database_namespace")
                == candidate.database_namespace
                and existing.get("artifact_namespace")
                == candidate.artifact_namespace
            )
            if not same:
                raise HarnessPersistenceError(
                    "candidate_ref already exists with different isolation metadata"
                )
            return int(existing["id"])
        candidate_id = db.insert_returning_id(
            """
            INSERT INTO harness_candidates
                (agenda_id, parent_archive_id, candidate_ref, base_commit,
                 worktree_path, database_namespace, artifact_namespace, status)
            VALUES (?, ?, ?, ?, ?, ?, ?, 'draft')
            RETURNING id
            """,
            (
                candidate.agenda_id,
                parent_archive_id,
                candidate.candidate_ref,
                candidate.base_commit,
                candidate.worktree_path,
                candidate.database_namespace,
                candidate.artifact_namespace,
            ),
        )
        db.commit()
        return candidate_id

    def save_patch(
        self,
        patch: HarnessPatch,
        *,
        candidate_id: int,
        policy: HarnessPolicy,
    ) -> int:
        validate_patch(patch, policy=policy)
        candidate = db.fetchone(
            """
            SELECT agenda_id, candidate_ref, base_commit
            FROM harness_candidates
            WHERE id=?
            """,
            (int(candidate_id),),
        )
        if (
            not candidate
            or int(candidate.get("agenda_id") or 0) != patch.agenda_id
            or candidate.get("candidate_ref") != patch.candidate_ref
            or candidate.get("base_commit") != patch.base_commit
        ):
            raise HarnessPersistenceError("patch/candidate scope or lineage mismatch")
        existing = db.fetchone(
            """
            SELECT id FROM harness_patches
            WHERE candidate_id=? AND patch_hash=?
            """,
            (int(candidate_id), patch.patch_hash),
        )
        if existing:
            return int(existing["id"])
        patch_id = db.insert_returning_id(
            """
            INSERT INTO harness_patches
                (agenda_id, candidate_id, base_commit, patch_hash,
                 changed_modules_json, added_lines, deleted_lines,
                 policy_version)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            RETURNING id
            """,
            (
                patch.agenda_id,
                candidate_id,
                patch.base_commit,
                patch.patch_hash,
                _dump(patch.changed_paths),
                patch.added_lines,
                patch.deleted_lines,
                policy.version,
            ),
        )
        db.commit()
        return patch_id

    def save_failure_cluster(self, cluster: FailureCluster) -> int:
        cluster.validate()
        existing = db.fetchone(
            """
            SELECT id FROM failure_clusters
            WHERE agenda_id=? AND cluster_key=?
            """,
            (cluster.agenda_id, cluster.cluster_key),
        )
        if existing:
            return int(existing["id"])
        cluster_id = db.insert_returning_id(
            """
            INSERT INTO failure_clusters
                (agenda_id, cluster_key, signature_json, evidence_json,
                 occurrence_count)
            VALUES (?, ?, ?, ?, ?)
            RETURNING id
            """,
            (
                cluster.agenda_id,
                cluster.cluster_key,
                _dump(cluster.signatures),
                _dump(cluster.evidence_refs),
                cluster.occurrence_count,
            ),
        )
        db.commit()
        return cluster_id

    def save_evaluation(
        self,
        evaluation: EvaluationRun,
        *,
        candidate_id: int,
        patch_id: int,
    ) -> int:
        if evaluation.agenda_id <= 0:
            raise HarnessPersistenceError("EvaluationRun requires agenda_id")
        if evaluation.suite not in {"held_in", "held_out", "canary"}:
            raise HarnessPersistenceError("invalid evaluation suite")
        if evaluation.status not in {"passed", "failed"}:
            raise HarnessPersistenceError("invalid evaluation status")
        if evaluation.status == "failed" and not evaluation.failure_reason:
            raise HarnessPersistenceError(
                "failed evaluation requires a failure reason"
            )
        if not evaluation.evaluator_ref or not evaluation.evaluator_hash:
            raise HarnessPersistenceError("evaluator ref/hash are required")
        if not evaluation.artifact_manifest:
            raise HarnessPersistenceError("evaluation artifacts are required")
        scope = db.fetchone(
            """
            SELECT hc.agenda_id, hp.agenda_id AS patch_agenda_id
            FROM harness_candidates hc
            JOIN harness_patches hp ON hp.candidate_id=hc.id
            WHERE hc.id=? AND hp.id=?
            """,
            (int(candidate_id), int(patch_id)),
        )
        if (
            not scope
            or int(scope.get("agenda_id") or 0) != evaluation.agenda_id
            or int(scope.get("patch_agenda_id") or 0) != evaluation.agenda_id
        ):
            raise HarnessPersistenceError("evaluation scope mismatch")
        existing = db.fetchone(
            """
            SELECT id FROM harness_evaluation_runs
            WHERE candidate_id=? AND patch_id=? AND suite=?
            """,
            (candidate_id, patch_id, evaluation.suite),
        )
        if existing:
            raise HarnessPersistenceError(
                "evaluation suite is immutable; create a new patch/candidate"
            )
        evaluation_id = db.insert_returning_id(
            """
            INSERT INTO harness_evaluation_runs
                (agenda_id, candidate_id, patch_id, suite, evaluator_ref,
                 evaluator_hash, status, result_json, artifact_manifest_json,
                 completed_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            RETURNING id
            """,
            (
                evaluation.agenda_id,
                candidate_id,
                patch_id,
                evaluation.suite,
                evaluation.evaluator_ref,
                evaluation.evaluator_hash,
                evaluation.status,
                _dump({"failure_reason": evaluation.failure_reason}),
                _dump(evaluation.artifact_manifest),
            ),
        )
        db.commit()
        return evaluation_id

    def save_regression_report(
        self,
        report: RegressionReport,
        *,
        candidate_id: int,
        held_in_run_id: int,
        held_out_run_id: int,
        canary_run_id: int,
    ) -> int:
        if report.agenda_id <= 0:
            raise HarnessPersistenceError("RegressionReport requires agenda_id")
        run_ids = (held_in_run_id, held_out_run_id, canary_run_id)
        placeholders = ",".join("?" for _ in run_ids)
        rows = db.fetchall(
            f"""
            SELECT her.id, her.agenda_id, her.candidate_id, her.patch_id,
                   her.suite, her.status, hp.patch_hash
            FROM harness_evaluation_runs AS her
            JOIN harness_patches AS hp ON hp.id=her.patch_id
            WHERE her.id IN ({placeholders})
            """,
            tuple(int(value) for value in run_ids),
        )
        by_suite = {str(row["suite"]): row for row in rows}
        if set(by_suite) != {"held_in", "held_out", "canary"}:
            raise HarnessPersistenceError("all three evaluation suites are required")
        if any(
            int(row.get("agenda_id") or 0) != report.agenda_id
            or int(row.get("candidate_id") or 0) != int(candidate_id)
            or row.get("status") != "passed"
            for row in rows
        ):
            raise HarnessPersistenceError(
                "regression report requires three passed scoped evaluations"
            )
        if report.decision == "approved" and (
            not report.reviewer_approved
            or not report.reviewer
            or report.reviewer_approval is None
        ):
            raise HarnessPersistenceError(
                "approved report requires signed reviewer approval"
            )
        patch_hashes = {str(row.get("patch_hash") or "") for row in rows}
        if len(patch_hashes) != 1 or "" in patch_hashes:
            raise HarnessPersistenceError(
                "regression report evaluations must share one patch"
            )
        approval = None
        approval_record = None
        if report.decision == "approved":
            patch_hash = next(iter(patch_hashes))
            approval = ReviewerApprovalVerifier.from_environment().verify(
                report.reviewer_approval,
                purpose="harness_upgrade",
                subject=harness_candidate_subject(
                    agenda_id=report.agenda_id,
                    candidate_id=int(candidate_id),
                    patch_hash=patch_hash,
                ),
            )
            if approval.reviewer_id != report.reviewer:
                raise HarnessPersistenceError(
                    "report reviewer does not match signed reviewer"
                )
            approval_record = approval.public_record()
        report_id = db.insert_returning_id(
            """
            INSERT INTO harness_regression_reports
                (agenda_id, candidate_id, held_in_run_id, held_out_run_id,
                 canary_run_id, decision, reviewer, approved_at, report_json)
            VALUES (?, ?, ?, ?, ?, ?, ?,
                    CASE WHEN ?='approved' THEN CURRENT_TIMESTAMP ELSE NULL END,
                    ?)
            RETURNING id
            """,
            (
                report.agenda_id,
                candidate_id,
                held_in_run_id,
                held_out_run_id,
                canary_run_id,
                report.decision,
                report.reviewer,
                report.decision,
                _dump(
                    {
                        "blockers": report.blockers,
                        "reviewer_approved": report.reviewer_approved,
                        "reviewer_approval": approval_record,
                    }
                ),
            ),
        )
        if approval_record is not None:
            db.execute(
                """
                INSERT INTO reviewer_approval_records
                    (agenda_id, purpose, subject, reviewer_id, key_id,
                     issued_at, signature_hash)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    report.agenda_id,
                    approval_record["purpose"],
                    approval_record["subject"],
                    approval_record["reviewer_id"],
                    approval_record["key_id"],
                    approval_record["issued_at"],
                    approval_record["signature_hash"],
                ),
            )
        db.commit()
        return report_id

    def save_archive(self, archive: HarnessArchive) -> int:
        if archive.agenda_id <= 0:
            raise HarnessPersistenceError("HarnessArchive requires agenda_id")
        hashes = (
            archive.source_commit,
            archive.source_tree_hash,
            archive.policy_hash,
            archive.evaluator_hash,
            archive.holdout_hash,
        )
        if any(not str(value).strip() for value in hashes):
            raise HarnessPersistenceError("HarnessArchive hashes are required")
        existing = db.fetchone(
            """
            SELECT id FROM harness_archives
            WHERE agenda_id=? AND source_commit=? AND policy_hash=?
              AND evaluator_hash=? AND holdout_hash=?
            """,
            (
                archive.agenda_id,
                archive.source_commit,
                archive.policy_hash,
                archive.evaluator_hash,
                archive.holdout_hash,
            ),
        )
        if existing:
            return int(existing["id"])
        archive_id = db.insert_returning_id(
            """
            INSERT INTO harness_archives
                (agenda_id, source_commit, source_tree_hash, policy_hash,
                 evaluator_hash, holdout_hash)
            VALUES (?, ?, ?, ?, ?, ?)
            RETURNING id
            """,
            (
                archive.agenda_id,
                archive.source_commit,
                archive.source_tree_hash,
                archive.policy_hash,
                archive.evaluator_hash,
                archive.holdout_hash,
            ),
        )
        db.commit()
        return archive_id
