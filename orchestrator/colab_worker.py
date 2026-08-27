"""Single-process worker for durable Colab compute requests."""

from __future__ import annotations

import hashlib
import json
import os
import socket
import threading
from datetime import datetime, timedelta
from pathlib import Path

from config import COMPUTE_COLAB_POLL_SECONDS
from db import database as db
from meta_harness.backends.colab_durable import (
    ColabWorkRepository,
    execution_request_from_row,
    grant_from_row,
)
from meta_harness.compute import ComputeBackendError, ComputeJob
from meta_harness.evidence_state import EvidenceTransitionContext
from meta_harness.repository import MetaHarnessRepository
from meta_harness.runner_contract import validate_final_results, verify_metric_from_artifacts


_thread: threading.Thread | None = None
_threads: list[threading.Thread] = []
_lock = threading.Lock()
_stop = threading.Event()
_last_status: dict = {"status": "not_started"}


def _worker_id() -> str:
    # Thread-qualified: with two accounts the worker runs one thread per
    # lane, and claim/requeue bookkeeping must not mix their claims.
    return (
        f"{socket.gethostname()}:{os.getpid()}:"
        f"{threading.current_thread().name}:colab"
    )


def _record_terminal_run_failure(row: dict, result, observed) -> None:
    """Make a terminal Colab result terminal for its owning experiment run.

    Durable compute state alone is not a scheduler consumer: an auto-research
    job remains ``queued_gpu`` until its experiment run changes state.  Without
    this handoff, a failed Colab request holds the only execution slot forever.
    The bounded Colab result payload remains the detailed diagnostic record.
    """

    status = str(getattr(observed, "status", "") or "")
    if status not in {"failed", "timed_out", "cancelled"}:
        return
    reason = str(
        getattr(result, "failure_reason", None)
        or getattr(observed, "failure_reason", None)
        or f"colab_returncode_{getattr(result, 'returncode', 'unknown')}"
    )
    db.execute(
        """
        UPDATE experiment_runs
        SET status='failed', phase='colab_compute_failed', error_message=?,
            completed_at=COALESCE(completed_at, CURRENT_TIMESTAMP)
        WHERE id=? AND agenda_id=? AND deep_insight_id=?
          AND status NOT IN ('completed', 'bundle_ready', 'cancelled', 'superseded', 'archived')
        """,
        (
            f"colab_compute_{status}:{reason}"[:4000],
            int(row["experiment_run_id"]),
            int(row["agenda_id"]),
            int(row["idea_id"]),
        ),
    )
    db.commit()


def _artifact_path_component(value: str) -> str:
    """Return a readable, collision-resistant directory component."""

    cleaned = "".join(
        character if character.isalnum() or character in {"-", "_", "."} else "_"
        for character in str(value)
    ).strip("._")
    digest = hashlib.sha256(str(value).encode("utf-8")).hexdigest()[:12]
    return f"{(cleaned or 'artifact')[:48]}-{digest}"


def _artifact_snapshot_path(
    *,
    workdir: Path,
    artifact_stage: str,
    artifact_type: str,
    source_path: Path,
    content_sha256: str,
) -> Path:
    return (
        workdir.resolve()
        / ".artifact-history-v1"
        / _artifact_path_component(artifact_stage)
        / _artifact_path_component(artifact_type)
        / content_sha256
        / source_path.name
    )


def _immutable_artifact_snapshot(
    *,
    workdir: Path,
    artifact_stage: str,
    artifact_type: str,
    source_path: Path,
    content: bytes,
    content_sha256: str,
) -> Path:
    """Persist one content-addressed artifact without ever replacing bytes."""

    snapshot = _artifact_snapshot_path(
        workdir=workdir,
        artifact_stage=artifact_stage,
        artifact_type=artifact_type,
        source_path=source_path,
        content_sha256=content_sha256,
    )
    snapshot.parent.mkdir(parents=True, exist_ok=True)
    try:
        with snapshot.open("xb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        snapshot.chmod(0o444)
    except FileExistsError:
        pass
    if snapshot.is_symlink() or not snapshot.is_file():
        raise ComputeBackendError(f"artifact_snapshot_invalid:{artifact_type}")
    snapshot_hash = hashlib.sha256(snapshot.read_bytes()).hexdigest()
    if snapshot_hash != content_sha256:
        raise ComputeBackendError(f"artifact_snapshot_hash_mismatch:{artifact_type}")
    return snapshot.resolve()


def _register_artifact_version(
    *,
    run: dict,
    resource_grant_id: int,
    artifact_stage: str,
    artifact_type: str,
    source_path: Path,
    content: bytes,
    content_sha256: str,
    metric_key: str | None,
    metric_value: float | None,
    legacy_unresolved: list[dict] | None = None,
) -> int:
    """Append an immutable stage version, idempotently for identical bytes."""

    stage = str(artifact_stage or "").strip().lower()
    if not stage:
        raise ComputeBackendError("artifact_stage_missing")
    actual_hash = hashlib.sha256(content).hexdigest()
    if actual_hash != content_sha256:
        raise ComputeBackendError(f"artifact_hash_mismatch:{artifact_type}")
    snapshot = _immutable_artifact_snapshot(
        workdir=Path(str(run.get("workdir") or "")),
        artifact_stage=stage,
        artifact_type=artifact_type,
        source_path=source_path,
        content=content,
        content_sha256=content_sha256,
    )
    existing = db.fetchone(
        """
        SELECT id, path FROM experiment_artifacts
        WHERE agenda_id=? AND run_id=? AND artifact_type=?
          AND artifact_stage=? AND content_sha256=?
        ORDER BY id DESC
        LIMIT 1
        """,
        (
            int(run["agenda_id"]),
            int(run["id"]),
            artifact_type,
            stage,
            content_sha256,
        ),
    )
    if existing:
        recorded_path = Path(str(existing.get("path") or ""))
        if recorded_path.resolve() != snapshot:
            raise ComputeBackendError(f"artifact_snapshot_path_mismatch:{artifact_type}")
        return int(existing["id"])

    previous = db.fetchone(
        """
        SELECT id, COALESCE(artifact_version, 1) AS artifact_version
        FROM experiment_artifacts
        WHERE agenda_id=? AND run_id=? AND artifact_type=?
        ORDER BY COALESCE(artifact_version, 1) DESC, id DESC
        LIMIT 1
        """,
        (int(run["agenda_id"]), int(run["id"]), artifact_type),
    )
    artifact_version = int((previous or {}).get("artifact_version") or 0) + 1
    metadata = {
        "artifact_stage": stage,
        "artifact_version": artifact_version,
        "contract_type": "RunnerArtifact",
        "immutable_snapshot": True,
        "resource_grant_id": int(resource_grant_id),
        "sha256": content_sha256,
        "source_path": str(source_path),
        "supersedes_artifact_id": int(previous["id"]) if previous else None,
        "verified_by": "colab_terminal_handoff_v1",
    }
    if legacy_unresolved:
        metadata["legacy_unresolved"] = list(legacy_unresolved)
    db.execute(
        """
        INSERT INTO experiment_artifacts
            (agenda_id, run_id, artifact_type, path, artifact_stage,
             artifact_version, content_sha256, metric_key, metric_value, metadata)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT DO NOTHING
        """,
        (
            int(run["agenda_id"]),
            int(run["id"]),
            artifact_type,
            str(snapshot),
            stage,
            artifact_version,
            content_sha256,
            metric_key,
            metric_value,
            json.dumps(metadata, sort_keys=True),
        ),
    )
    inserted = db.fetchone(
        """
        SELECT id FROM experiment_artifacts
        WHERE agenda_id=? AND run_id=? AND artifact_type=?
          AND artifact_stage=? AND content_sha256=?
        ORDER BY id DESC
        LIMIT 1
        """,
        (
            int(run["agenda_id"]),
            int(run["id"]),
            artifact_type,
            stage,
            content_sha256,
        ),
    )
    if not inserted:
        raise ComputeBackendError(f"artifact_registration_failed:{artifact_type}")
    return int(inserted["id"])


def _request_artifact_output_dir(run: dict, request: dict) -> Path:
    """Resolve the durable request's output directory inside its run workdir."""

    raw_workdir = str(run.get("workdir") or "").strip()
    raw_output = str(request.get("artifact_output_dir") or "").strip()
    if not raw_workdir or not raw_output:
        raise ComputeBackendError("artifact_output_dir_missing")
    output = Path(raw_output)
    if not output.is_absolute():
        raise ComputeBackendError("artifact_output_dir_not_absolute")
    workdir = Path(raw_workdir).resolve()
    output = output.resolve()
    if output == workdir or workdir not in output.parents:
        raise ComputeBackendError("artifact_output_dir_outside_run_workdir")
    return output


def _verified_request_artifacts(run: dict, request: dict):
    """Read and verify the artifacts named by one durable Colab request."""

    results_dir = _request_artifact_output_dir(run, request)
    final_path = results_dir / "final_results.json"
    payload = validate_final_results(
        json.loads(final_path.read_text(encoding="utf-8"))
    )
    verification = verify_metric_from_artifacts(final_path)
    artifacts: list[dict] = []
    for artifact_type, reference in payload["artifacts"].items():
        relative_path = str((reference or {}).get("path") or "")
        artifact_path = (results_dir / relative_path).resolve()
        if (
            not relative_path
            or (artifact_path != results_dir and results_dir not in artifact_path.parents)
            or not artifact_path.is_file()
        ):
            raise ComputeBackendError(f"artifact_contract_violation:{artifact_type}")
        expected_hash = str(payload["artifact_hashes"].get(artifact_type) or "")
        content = artifact_path.read_bytes()
        actual_hash = hashlib.sha256(content).hexdigest()
        if expected_hash and actual_hash != expected_hash:
            raise ComputeBackendError(f"artifact_hash_mismatch:{artifact_type}")
        artifacts.append(
            {
                "artifact_type": str(artifact_type),
                "source_path": artifact_path,
                "content": content,
                "content_sha256": actual_hash,
            }
        )
    return payload, verification, artifacts


def _record_terminal_run_success(row: dict, observed) -> None:
    """Promote a verified Colab result into the owning run's durable evidence.

    The Colab backend stores its own artifact manifest, but an outcome can only
    be assembled after the runner artifacts have been hash-verified and
    registered against the experiment run.  Keep this handoff here, adjacent to
    the failure handoff, so a terminal compute request can never strand a
    scheduler job in ``queued_gpu``.
    """

    if str(getattr(observed, "status", "") or "") != "succeeded":
        return
    run = db.fetchone(
        """
        SELECT id, agenda_id, deep_insight_id, resource_grant_id, workdir,
               scientific_evidence_state
        FROM experiment_runs
        WHERE id=? AND agenda_id=? AND deep_insight_id=?
        """,
        (int(row["experiment_run_id"]), int(row["agenda_id"]), int(row["idea_id"])),
    )
    if not run or int(run.get("resource_grant_id") or 0) != int(
        row["resource_grant_id"]
    ):
        return
    grant_row = db.fetchone(
        "SELECT stage, preflight_result_id FROM resource_grants WHERE id=?",
        (int(row["resource_grant_id"]),),
    )
    artifact_stage = str((grant_row or {}).get("stage") or "").strip().lower()
    if not artifact_stage:
        raise ComputeBackendError("artifact_stage_missing")
    request_stage = str(row.get("stage") or "").strip().lower()
    if request_stage and request_stage != artifact_stage:
        raise ComputeBackendError("artifact_stage_grant_mismatch")
    payload, verification, artifacts = _verified_request_artifacts(dict(run), row)
    for artifact in artifacts:
        artifact_type = str(artifact["artifact_type"])
        _register_artifact_version(
            run=dict(run),
            resource_grant_id=int(row["resource_grant_id"]),
            artifact_stage=artifact_stage,
            artifact_type=artifact_type,
            source_path=Path(artifact["source_path"]),
            content=artifact["content"],
            content_sha256=str(artifact["content_sha256"]),
            metric_key=verification.metric_name,
            metric_value=(
                verification.candidate_value
                if artifact_type == "final_results"
                else None
            ),
        )
    effect = (
        verification.candidate_value - verification.baseline_value
        if verification.direction == "higher"
        else verification.baseline_value - verification.candidate_value
    )
    effect_pct = (
        (effect / abs(verification.baseline_value)) * 100.0
        if verification.baseline_value != 0
        else None
    )
    # Read the runner's authoritative verdict; fall back to the significance
    # rule only for artifacts written before it existed. Deriving "refuted"
    # from the direction alone overstated runs 164 (p=0.506) and 180
    # (p=0.071): failing to show an improvement is not showing harm.
    verdict = str(payload.get("hypothesis_verdict") or "").strip()
    if verdict not in {"supported", "refuted", "inconclusive"}:
        _tests = payload.get("statistical_tests") or {}
        _p = _tests.get("paired_permutation_p") if isinstance(_tests, dict) else None
        if _p is None or float(_p) >= 0.05:
            verdict = "inconclusive"
        else:
            verdict = (
                "refuted"
                if payload.get("scientific_negative_result") is True
                else "supported"
            )
    # Holdout evidence is a second measurement, not a replacement for the
    # run's primary full-benchmark metrics. Before output-dir provenance was
    # honored this branch re-read results/ and happened to write the same
    # values back; reading results_holdout must not change that business fact.
    if artifact_stage != "evidence_audit":
        db.execute(
            """
            UPDATE experiment_runs
            SET status='completed', phase='colab_result_verified',
                baseline_metric_name=?, baseline_metric_value=?, best_metric_value=?,
                effect_size=?, effect_pct=?, hypothesis_verdict=?, error_message=NULL,
                completed_at=COALESCE(completed_at, CURRENT_TIMESTAMP)
            WHERE id=? AND agenda_id=? AND deep_insight_id=?
              AND status NOT IN ('failed', 'cancelled', 'superseded', 'archived')
            """,
            (
                verification.metric_name,
                verification.baseline_value,
                verification.candidate_value,
                effect,
                effect_pct,
                verdict,
                int(run["id"]),
                int(run["agenda_id"]),
                int(run["deep_insight_id"]),
            ),
        )
    db.commit()

    # The evidence transition used to live here, and again for ssh in
    # meta_compute_runtime, and not at all for cpu -- three guard sets for one
    # rule. It is now _advance_settled_evidence_state in the outcome finalizer,
    # which asks only what matters: did the compute job succeed and are the
    # runner's artifacts registered. A GPU reached over Colab and one reached
    # over ssh are the same resource to the science.


def _reconcile_succeeded_runs() -> int:
    """Recover success handoffs interrupted after durable Colab settlement."""

    rows = db.fetchall(
        """
        SELECT cwr.experiment_run_id, cwr.agenda_id, cwr.idea_id,
               cwr.resource_grant_id, cwr.stage, cwr.artifact_output_dir
        FROM colab_work_requests_v1 AS cwr
        JOIN compute_jobs_v1 AS cj ON cj.id=cwr.compute_job_id
        JOIN experiment_runs AS er ON er.id=cwr.experiment_run_id
        WHERE cwr.status='succeeded' AND cj.status='succeeded'
          AND er.status NOT IN ('failed', 'cancelled', 'superseded', 'archived')
          AND (
              er.status <> 'completed'
              OR COALESCE(er.scientific_evidence_state, 'planned') = 'planned'
              -- A full-benchmark run whose handoff ran under code that only
              -- knew the sanity rung still owes its ladder advance.
              OR (
                  COALESCE(er.scientific_evidence_state, 'planned') = 'sanity_passed'
                  AND EXISTS (
                      SELECT 1 FROM resource_grants g
                      WHERE g.id=cwr.resource_grant_id
                        AND g.stage='full_benchmark'
                  )
              )
          )
        ORDER BY cwr.completed_at ASC, cwr.id ASC
        LIMIT 20
        """
    )
    for row in rows:
        _record_terminal_run_success(
            dict(row), type("Observed", (), {"status": "succeeded"})()
        )
    return len(rows)


def recover_succeeded_run(*, experiment_run_id: int, resource_grant_id: int) -> bool:
    """Recover one explicitly named successful request without claiming work.

    This operator-safe entry point is intentionally narrower than ``run_one``:
    it cannot inspect, claim, submit, or execute any other queued Colab work.
    It only performs the verified terminal handoff for the supplied run/grant
    pair, which makes it suitable for recovering a controller interruption.
    """

    row = db.fetchone(
        """
        SELECT cwr.experiment_run_id, cwr.agenda_id, cwr.idea_id,
               cwr.resource_grant_id, cwr.stage, cwr.artifact_output_dir
        FROM colab_work_requests_v1 AS cwr
        JOIN compute_jobs_v1 AS cj ON cj.id=cwr.compute_job_id
        WHERE cwr.experiment_run_id=? AND cwr.resource_grant_id=?
          AND cwr.status='succeeded' AND cj.status='succeeded'
        ORDER BY cwr.completed_at DESC, cwr.id DESC
        LIMIT 1
        """,
        (int(experiment_run_id), int(resource_grant_id)),
    )
    if not row:
        return False
    _record_terminal_run_success(
        dict(row), type("Observed", (), {"status": "succeeded"})()
    )
    return True


def run_one() -> dict:
    """Claim and settle at most one request; safe for a scheduler loop or CI."""
    from meta_harness.attempt_gpu_usage import GrantGPUUsageControl
    from orchestrator.meta_compute_runtime import (
        build_scheduler,
        settle_colab_request,
    )

    for pending_request_id in (
        GrantGPUUsageControl().reconcile_terminal_colab_attempts()
    ):
        # One unsettleable request must not take the whole worker (and, at
        # startup, the whole web service) down with it: request 8's over-cap
        # settlement crash-looped the dashboard on 2026-08-17.
        try:
            settle_colab_request(pending_request_id)
        except Exception as exc:
            print(
                f"[COLAB] settlement failed for request {pending_request_id}: "
                f"{type(exc).__name__}: {exc}",
                flush=True,
            )
    reconciled_successes = _reconcile_succeeded_runs()

    repository = ColabWorkRepository()
    # A request the worker failed on its own defect never reached Colab and
    # cannot be recreated, because its idempotency key is derived from the run.
    # Give those back to the queue before claiming.
    repository.requeue_control_lost()
    # Ask whether any lane is free BEFORE claiming. The pool is only consulted
    # deep inside the executor, so a full pool used to surface as an exception
    # after the claim -- and the worker treats any exception as losing control,
    # so it failed the request and requeued it straight back into the same full
    # pool. Request 65 burned ten attempts in three minutes that way on
    # 2026-08-19 and run 189's audit could not proceed. Waiting for capacity
    # must cost the request nothing.
    # The pool lives on the CLI executor; the durable backend's own .accounts
    # is the raw tuple. Reaching for the wrong one raised AttributeError into
    # a bare `except`, which turned this whole guard into a silent no-op --
    # the same fail-open-and-hide-it shape the guard exists to stop. Report
    # the mistake instead of swallowing it.
    pool = None
    try:
        # ColabGPUBackend -> DurableColabTransport -> ColabCLIExecutor.accounts
        # is the pool. The backend's and the transport's own .accounts are both
        # the raw tuple; only the executor holds the ColabAccountPool.
        _backend = build_scheduler().configured_backend("colab_gpu")
        pool = _backend._transport.executor.accounts
    except Exception as exc:
        print(
            f"[COLAB] capacity probe unavailable: {type(exc).__name__}: {exc}",
            flush=True,
        )
    if pool is not None and not pool.has_capacity():
        return {
            "status": "no_capacity",
            "reconciled_succeeded_runs": reconciled_successes,
        }
    # Work declaring a dependency only a provisioned lane carries can run
    # nowhere else. Claiming it while no such lane is free means claiming,
    # failing and requeueing every poll: request 101 did that 43 times in
    # under four minutes while a full benchmark held the lane. Look before
    # claiming instead.
    if pool is not None and not pool.has_dedicated_capacity():
        try:
            from meta_harness.backends.colab_cli import (
                _requires_provisioned_runtime,
            )

            head = db.fetchone(
                """
                SELECT code_dir FROM colab_work_requests_v1
                WHERE status='queued'
                ORDER BY created_at, id
                LIMIT 1
                """
            )
            if head and _requires_provisioned_runtime(Path(str(head["code_dir"]))):
                return {
                    "status": "waiting_for_dedicated_lane",
                    "reconciled_succeeded_runs": reconciled_successes,
                }
        except Exception as exc:
            print(
                f"[COLAB] dedicated-lane peek failed: {type(exc).__name__}: {exc}",
                flush=True,
            )
    row = repository.claim_next(worker_id=_worker_id())
    if not row:
        return {"status": "idle", "reconciled_succeeded_runs": reconciled_successes}
    worker_id = _worker_id()
    try:
        scheduler = build_scheduler()
        backend = scheduler.configured_backend("colab_gpu")
        grant_row = db.fetchone(
            """
            SELECT rg.*, rg.status AS grant_status,
                   rg.idempotency_key AS grant_idempotency_key,
                   -- grant_from_row reads the grant id under the name the
                   -- work-request rows use; resource_grants calls it "id", so
                   -- selecting rg.* alone left the mapper with a KeyError and
                   -- every claimed Colab request was quarantined as
                   -- colab_worker_control_lost before reaching the executor.
                   rg.id AS resource_grant_id
            FROM resource_grants AS rg
            WHERE rg.id=? AND rg.agenda_id=? AND rg.idea_id=?
              AND rg.status='active' AND rg.expires_at > CURRENT_TIMESTAMP
            """,
            (
                int(row["resource_grant_id"]),
                int(row["agenda_id"]),
                int(row["idea_id"]),
            ),
        )
        if not grant_row:
            raise ComputeBackendError(
                "Colab work ResourceGrant expired after claim"
            )
        transport = getattr(backend, "_transport", None)
        if transport is None or not hasattr(transport, "executor"):
            raise ComputeBackendError(
                "configured Colab backend has no durable executor"
            )
        result = transport.executor.run_request(
            execution_request_from_row(row),
            grant=grant_from_row(grant_row),
        )
        repository.save_result(int(row["id"]), result=result)
        persisted = db.fetchone(
            """
            SELECT cwr.completed_at, cwr.started_at, cj.gpu_attempt_reservation_id
            FROM colab_work_requests_v1 cwr
            JOIN compute_jobs_v1 cj ON cj.id=cwr.compute_job_id
            WHERE cwr.id=?
            """,
            (int(row["id"]),),
        ) or {}
        db.commit()
        reason_code = {
            "succeeded": "attempt_completed",
            "timed_out": "attempt_timed_out",
        }.get(result.status, "attempt_failed")
        started_at = persisted.get("started_at")
        if isinstance(started_at, str):
            started_at = datetime.fromisoformat(started_at.replace("Z", "+00:00"))
        completed_at = persisted.get("completed_at")
        if (
            isinstance(started_at, datetime)
            and float(result.wall_seconds or 0.0) > 0.0
        ):
            # The executor duration is the accelerator wall time.  The
            # request's completed_at is only when the controller persisted the
            # return value and can be milliseconds after claim.
            completed_at = started_at + timedelta(seconds=float(result.wall_seconds))
        GrantGPUUsageControl().settle_attempt(
            int(persisted.get("gpu_attempt_reservation_id") or 0),
            completed_at=completed_at,
            reason_code=reason_code,
        )
        observed = scheduler.refresh_and_settle(
            ComputeJob(
                backend_kind="colab_gpu",
                backend_job_id=str(row["backend_job_id"]),
                idempotency_key=str(row["idempotency_key"]),
                status="running",
                heartbeat_at=str(row.get("started_at") or "") or None,
            ),
            requirements=tuple(
                str(value)
                for value in __import__("json").loads(
                    grant_row.get("artifact_requirements_json") or "[]"
                )
            ),
        )
        _record_terminal_run_failure(row, result, observed)
        _record_terminal_run_success(row, observed)
    except Exception as exc:
        try:
            db.rollback()
        except Exception:
            pass
        persisted = db.fetchone(
            "SELECT status FROM colab_work_requests_v1 WHERE id=?",
            (int(row["id"]),),
        ) or {}
        if str(persisted.get("status") or "") == "running":
            repository.quarantine_claim(
                int(row["id"]),
                worker_id=worker_id,
                # Keep the message, not just the type: "ColabCLIError" alone
                # could not distinguish a full pool from a real control loss.
                reason=f"colab_worker_control_lost:{type(exc).__name__}:{exc}"[:400],
            )
        raise
    return {
        "status": observed.status,
        "colab_work_request_id": int(row["id"]),
        "compute_job_id": int(row["compute_job_id"]),
    }


def _loop() -> None:
    global _last_status
    while not _stop.is_set():
        try:
            _last_status = run_one()
        except Exception as exc:  # pragma: no cover - defensive worker guard
            try:
                db.rollback()
            except Exception:
                pass
            _last_status = {
                "status": "worker_error",
                "error": f"{type(exc).__name__}:{exc}",
            }
        _stop.wait(max(1, int(COMPUTE_COLAB_POLL_SECONDS)))


def start() -> dict:
    global _thread, _threads
    with _lock:
        alive = [t for t in _threads if t.is_alive()]
        if alive:
            return {"status": "already_running", **_last_status}
        _stop.clear()
        # One lane per configured account (bounded by the env knob): the
        # shared account pool arbitrates so each lane holds a distinct
        # account, and the durable claim layer row-locks the queue.
        try:
            lanes = max(1, int(os.environ.get(
                "DEEPGRAPH_COLAB_WORKER_THREADS", "1") or 1))
        except ValueError:
            lanes = 1
        _threads = []
        for index in range(lanes):
            worker = threading.Thread(
                target=_loop,
                daemon=True,
                name=f"deepgraph-colab-worker-{index}",
            )
            worker.start()
            _threads.append(worker)
        _thread = _threads[0]
    return {"status": "started", "lanes": lanes}


def stop() -> dict:
    _stop.set()
    return {"status": "stopping"}


def get_status() -> dict:
    with _lock:
        alive = [t for t in _threads if t.is_alive()]
        running = bool(alive) or bool(_thread and _thread.is_alive())
    return {"running": running, "lanes": len(alive), **_last_status}
