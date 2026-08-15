"""Durable candidate requirement declarations and preflight results."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from db import database as db
from meta_harness.backend_capability import reports_from_config
from meta_harness.runner_capability import (
    CapabilityContractError,
    ExperimentRequirements,
    PreflightEngine,
    PreflightEnvironment,
    PreflightResult,
    local_preflight_environment,
    requirements_from_plan,
)


class PreflightPersistenceError(RuntimeError):
    pass


def _dump(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)


def _hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            default=str,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def runtime_preflight_environment() -> PreflightEnvironment:
    reports = reports_from_config()
    enabled = tuple(
        kind
        for kind, report in reports.items()
        if report.usable_for_scheduling
    )
    backend_vram: dict[str, float] = {"cpu": 0.0}
    if db._use_pg():  # noqa: SLF001
        for kind, predicate in (
            ("ssh_gpu", "metadata LIKE ?"),
            ("local_gpu", "(metadata IS NULL OR metadata NOT LIKE ?)"),
        ):
            row = db.fetchone(
                f"""
                SELECT COALESCE(MAX(total_mem_gb), 0) AS max_vram_gb
                FROM gpu_workers WHERE {predicate}
                """,
                ('%"backend": "ssh"%',),
            ) or {}
            backend_vram[kind] = float(row.get("max_vram_gb") or 0.0)
        db.commit()
    # Colab hardware is still not assumed from an account manifest. What counts
    # is a canary that actually reached the accelerator and recorded it as a
    # live worker row; absent that the zero VRAM makes preflight defer rather
    # than guess, exactly as before.
    if db._use_pg():  # noqa: SLF001
        row = db.fetchone(
            """
            SELECT COALESCE(MAX(total_mem_gb), 0) AS max_vram_gb
            FROM gpu_workers
            WHERE metadata LIKE ? AND status <> 'retired'
            """,
            ('%"backend": "colab"%',),
        ) or {}
        db.commit()
        backend_vram["colab_gpu"] = float(row.get("max_vram_gb") or 0.0)
    backend_vram.setdefault("colab_gpu", 0.0)
    return local_preflight_environment(
        enabled_backends=enabled,
        backend_vram_gb=backend_vram,
        network_available=True,
        path=Path.cwd(),
    )


class CandidatePreflightRepository:
    def declare(
        self,
        *,
        agenda_id: int,
        idea_id: int,
        requirements: ExperimentRequirements,
        source_plan_hash: str,
    ) -> int:
        requirements.validate()
        try:
            existing = db.fetchone(
                """
                SELECT id FROM candidate_execution_requirements_v1
                WHERE agenda_id=? AND idea_id=? AND requirements_hash=?
                """,
                (agenda_id, idea_id, requirements.canonical_hash()),
            )
            if existing:
                db.commit()
                return int(existing["id"])
            db.execute(
                """
                UPDATE candidate_execution_requirements_v1
                SET status='superseded'
                WHERE agenda_id=? AND idea_id=? AND status='declared'
                """,
                (agenda_id, idea_id),
            )
            requirement_id = db.insert_returning_id(
                """
                INSERT INTO candidate_execution_requirements_v1
                    (agenda_id, idea_id, schema_version, source_plan_hash,
                     requirements_hash, requirements_json, status)
                VALUES (?, ?, ?, ?, ?, ?, 'declared') RETURNING id
                """,
                (
                    agenda_id,
                    idea_id,
                    requirements.schema_version,
                    source_plan_hash,
                    requirements.canonical_hash(),
                    _dump(requirements.to_dict()),
                ),
            )
            db.commit()
            return int(requirement_id)
        except Exception:
            db.rollback()
            raise

    def declare_invalid(
        self,
        *,
        agenda_id: int,
        idea_id: int,
        plan: Mapping[str, Any],
    ) -> int:
        plan_hash = _hash(plan)
        try:
            existing = db.fetchone(
                """
                SELECT id FROM candidate_execution_requirements_v1
                WHERE agenda_id=? AND idea_id=? AND requirements_hash=?
                """,
                (agenda_id, idea_id, plan_hash),
            )
            if existing:
                db.commit()
                return int(existing["id"])
            requirement_id = db.insert_returning_id(
                """
                INSERT INTO candidate_execution_requirements_v1
                    (agenda_id, idea_id, schema_version, source_plan_hash,
                     requirements_hash, requirements_json, status)
                VALUES (?, ?, 'invalid_candidate_plan_v1', ?, ?, ?, 'declared')
                RETURNING id
                """,
                (agenda_id, idea_id, plan_hash, plan_hash, _dump(plan)),
            )
            db.commit()
            return int(requirement_id)
        except Exception:
            db.rollback()
            raise

    def record(
        self,
        *,
        agenda_id: int,
        idea_id: int,
        requirement_id: int,
        result: PreflightResult,
        environment: PreflightEnvironment,
        idempotency_key: str,
    ) -> int:
        try:
            existing = db.fetchone(
                """
                SELECT id FROM candidate_preflight_results_v1
                WHERE requirement_id=? AND idempotency_key=?
                """,
                (requirement_id, idempotency_key),
            )
            if existing:
                db.commit()
                return int(existing["id"])
            result_id = db.insert_returning_id(
                """
                INSERT INTO candidate_preflight_results_v1
                    (agenda_id, idea_id, requirement_id, adapter_id,
                     adapter_version, selected_backend, status,
                     reason_codes_json, checks_json, environment_json,
                     dataset_revision, model_revision, idempotency_key)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?) RETURNING id
                """,
                (
                    agenda_id,
                    idea_id,
                    requirement_id,
                    result.adapter_id,
                    result.adapter_version,
                    result.selected_backend,
                    result.status,
                    _dump(list(result.reason_codes)),
                    _dump(dict(result.checks)),
                    _dump(
                        {
                            "enabled_backends": list(environment.enabled_backends),
                            "backend_vram_gb": dict(environment.backend_vram_gb),
                            "network_available": environment.network_available,
                            "disk_free_gb": environment.disk_free_gb,
                        }
                    ),
                    result.dataset_revision,
                    result.model_revision,
                    idempotency_key,
                ),
            )
            db.commit()
            return int(result_id)
        except Exception:
            db.rollback()
            raise

    def run_candidate(
        self,
        *,
        agenda_id: int,
        idea_id: int,
        engine: PreflightEngine | None = None,
        environment: PreflightEnvironment | None = None,
        idempotency_key: str | None = None,
    ) -> PreflightResult:
        row = db.fetchone(
            """
            SELECT * FROM deep_insights
            WHERE id=? AND agenda_id=?
            """,
            (idea_id, agenda_id),
        )
        db.commit()
        if not row:
            raise PreflightPersistenceError("candidate_not_found")
        raw_plan = str(row.get("experimental_plan") or "{}")
        try:
            plan = json.loads(raw_plan)
        except (TypeError, json.JSONDecodeError):
            plan = {}
        if not isinstance(plan, dict):
            plan = {}
        environment = environment or runtime_preflight_environment()
        selected_engine = engine or PreflightEngine()
        try:
            requirements = requirements_from_plan(plan)
            requirement_id = self.declare(
                agenda_id=agenda_id,
                idea_id=idea_id,
                requirements=requirements,
                source_plan_hash=_hash(plan),
            )
            result = selected_engine.run(requirements, environment)
        except CapabilityContractError as exc:
            requirement_id = self.declare_invalid(
                agenda_id=agenda_id,
                idea_id=idea_id,
                plan=plan,
            )
            result = PreflightResult(
                status="failed",
                reason_codes=(str(exc),),
                checks={"candidate_plan": "invalid"},
            )
        attempt_key = idempotency_key or (
            f"preflight:{requirement_id}:"
            f"{datetime.now(timezone.utc).strftime('%Y%m%d%H')}"
        )
        result_id = self.record(
            agenda_id=agenda_id,
            idea_id=idea_id,
            requirement_id=requirement_id,
            result=result,
            environment=environment,
            idempotency_key=attempt_key,
        )
        if tuple(result.reason_codes) == ("model_task_mismatch",):
            # Import lazily: experiment_forge imports this repository for the
            # post-preflight forge binding.  The renegotiator is deterministic
            # and returns only an in-memory plan; it cannot grant or persist.
            from agents.experiment_forge import renegotiate_stale_model_requirement

            try:
                candidate = renegotiate_stale_model_requirement(
                    dict(row),
                    reason_codes=result.reason_codes,
                    checks=dict(result.checks),
                )
            except Exception:
                # Renegotiation is an optional, fail-closed recovery path.  A
                # malformed historical plan must retain its original audited
                # preflight result rather than breaking candidate processing.
                candidate = None
            if candidate:
                revised_plan = candidate["plan"]
                try:
                    revised_requirements = requirements_from_plan(revised_plan)
                    revised_result = selected_engine.run(
                        revised_requirements,
                        environment,
                    )
                except Exception:
                    revised_result = None
                if revised_result is not None and revised_result.passed:
                    revised_checks = dict(revised_result.checks)
                    revised_checks["model_requirement_renegotiated"] = {
                        "trigger_reason_codes": list(result.reason_codes),
                        "observed_model_task": candidate["observed_model_task"],
                        "previous_model": candidate["previous_model"],
                        "replacement_model": candidate["replacement_model"],
                    }
                    revised_result = PreflightResult(
                        status=revised_result.status,
                        reason_codes=revised_result.reason_codes,
                        checks=revised_checks,
                        adapter_id=revised_result.adapter_id,
                        adapter_version=revised_result.adapter_version,
                        selected_backend=revised_result.selected_backend,
                        dataset_revision=revised_result.dataset_revision,
                        model_revision=revised_result.model_revision,
                    )
                    updated = db.execute(
                        """
                        UPDATE deep_insights
                        SET experimental_plan=?, updated_at=CURRENT_TIMESTAMP
                        WHERE id=? AND agenda_id=? AND experimental_plan=?
                        """,
                        (
                            _dump(revised_plan),
                            idea_id,
                            agenda_id,
                            raw_plan,
                        ),
                    )
                    if int(updated.rowcount or 0) == 1:
                        db.commit()
                        revised_requirement_id = self.declare(
                            agenda_id=agenda_id,
                            idea_id=idea_id,
                            requirements=revised_requirements,
                            source_plan_hash=_hash(revised_plan),
                        )
                        revised_attempt_key = idempotency_key or (
                            f"preflight:{revised_requirement_id}:"
                            f"{datetime.now(timezone.utc).strftime('%Y%m%d%H')}"
                        )
                        revised_result_id = self.record(
                            agenda_id=agenda_id,
                            idea_id=idea_id,
                            requirement_id=revised_requirement_id,
                            result=revised_result,
                            environment=environment,
                            idempotency_key=revised_attempt_key,
                        )
                        return PreflightResult(
                            status=revised_result.status,
                            reason_codes=revised_result.reason_codes,
                            checks=revised_result.checks,
                            adapter_id=revised_result.adapter_id,
                            adapter_version=revised_result.adapter_version,
                            selected_backend=revised_result.selected_backend,
                            dataset_revision=revised_result.dataset_revision,
                            model_revision=revised_result.model_revision,
                            preflight_result_id=revised_result_id,
                        )
                    db.rollback()
        return PreflightResult(
            status=result.status,
            reason_codes=result.reason_codes,
            checks=result.checks,
            adapter_id=result.adapter_id,
            adapter_version=result.adapter_version,
            selected_backend=result.selected_backend,
            dataset_revision=result.dataset_revision,
            model_revision=result.model_revision,
            preflight_result_id=result_id,
        )

    def require_passed(
        self,
        *,
        preflight_result_id: int,
        agenda_id: int,
        idea_id: int,
        allowed_backends: tuple[str, ...],
        required_artifacts: tuple[str, ...] = (),
    ) -> dict[str, Any]:
        row = db.fetchone(
            """
            SELECT p.*, r.status AS requirement_status, r.requirements_json
            FROM candidate_preflight_results_v1 p
            JOIN candidate_execution_requirements_v1 r ON r.id=p.requirement_id
            WHERE p.id=? AND p.agenda_id=? AND p.idea_id=?
            """,
            (preflight_result_id, agenda_id, idea_id),
        )
        try:
            requirements_payload = json.loads(
                str((row or {}).get("requirements_json") or "{}")
            )
        except (TypeError, json.JSONDecodeError):
            requirements_payload = {}
        declared_artifacts = set(
            requirements_payload.get("artifact_contract") or []
        )
        if (
            not row
            or str(row.get("status") or "") != "passed"
            or str(row.get("requirement_status") or "") != "declared"
            or str(row.get("selected_backend") or "") not in allowed_backends
            or not row.get("dataset_revision")
            or not row.get("model_revision")
            or not set(required_artifacts).issubset(declared_artifacts)
        ):
            db.rollback()
            raise PreflightPersistenceError("passed_candidate_preflight_required")
        return row
