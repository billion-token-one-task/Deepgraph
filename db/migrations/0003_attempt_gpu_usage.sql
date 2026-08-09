-- Canonical per-attempt GPU reservation and wall-clock settlement.
--
-- A ResourceGrant reserves the agenda-level ceiling.  These rows subdivide
-- that ceiling atomically between attempts, so admission has exactly one
-- source for settled usage and live reservations.

CREATE TABLE IF NOT EXISTS experiment_attempt_gpu_reservations_v1 (
    id BIGSERIAL PRIMARY KEY,
    agenda_id BIGINT NOT NULL REFERENCES research_agendas(id),
    idea_id BIGINT NOT NULL,
    resource_grant_id BIGINT NOT NULL REFERENCES resource_grants(id),
    compute_job_id BIGINT REFERENCES compute_jobs_v1(id),
    experiment_run_id BIGINT REFERENCES experiment_runs(id),
    gpu_job_id BIGINT REFERENCES gpu_jobs(id),
    attempt_key TEXT NOT NULL,
    backend_kind TEXT NOT NULL
        CHECK (backend_kind IN ('local_gpu', 'ssh_gpu', 'colab_gpu')),
    gpu_count INTEGER NOT NULL DEFAULT 1 CHECK (gpu_count > 0),
    reserved_gpu_seconds DOUBLE PRECISION NOT NULL
        CHECK (reserved_gpu_seconds > 0),
    timeout_seconds INTEGER NOT NULL CHECK (timeout_seconds > 0),
    status TEXT NOT NULL
        CHECK (status IN ('reserved', 'running', 'settled', 'released')),
    started_at TIMESTAMPTZ,
    completed_at TIMESTAMPTZ,
    actual_gpu_seconds DOUBLE PRECISION
        CHECK (actual_gpu_seconds IS NULL OR actual_gpu_seconds >= 0),
    reason_code TEXT,
    lease_expires_at TIMESTAMPTZ NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE (resource_grant_id, attempt_key),
    UNIQUE (compute_job_id),
    UNIQUE (gpu_job_id)
);

CREATE INDEX IF NOT EXISTS idx_attempt_gpu_reservations_grant_active
    ON experiment_attempt_gpu_reservations_v1(resource_grant_id, status, id);
CREATE INDEX IF NOT EXISTS idx_attempt_gpu_reservations_recovery
    ON experiment_attempt_gpu_reservations_v1(status, lease_expires_at, id);

ALTER TABLE IF EXISTS compute_jobs_v1
    ADD COLUMN IF NOT EXISTS gpu_attempt_reservation_id BIGINT
        REFERENCES experiment_attempt_gpu_reservations_v1(id);
ALTER TABLE IF EXISTS gpu_jobs
    ADD COLUMN IF NOT EXISTS gpu_attempt_reservation_id BIGINT
        REFERENCES experiment_attempt_gpu_reservations_v1(id);
-- Existing ledger constraints intentionally keep gpu_hours_used <= the grant
-- reservation.  Exceptional controller overruns are recorded separately so
-- measured usage is never discarded while normal acceptance can require zero.
ALTER TABLE IF EXISTS agenda_resource_ledger
    ADD COLUMN IF NOT EXISTS gpu_hours_overrun DOUBLE PRECISION NOT NULL DEFAULT 0
        CHECK (gpu_hours_overrun >= 0);

CREATE UNIQUE INDEX IF NOT EXISTS idx_compute_jobs_gpu_attempt_reservation
    ON compute_jobs_v1(gpu_attempt_reservation_id)
    WHERE gpu_attempt_reservation_id IS NOT NULL;
CREATE UNIQUE INDEX IF NOT EXISTS idx_gpu_jobs_gpu_attempt_reservation
    ON gpu_jobs(gpu_attempt_reservation_id)
    WHERE gpu_attempt_reservation_id IS NOT NULL;
