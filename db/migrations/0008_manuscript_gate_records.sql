-- A logical attempt is recorded before provider selection or token
-- sub-reservation.  It deliberately has no usage foreign key: a failure before
-- reservation spent nothing and must never be made to look metered.

CREATE TABLE IF NOT EXISTS manuscript_gate_attempts_v1 (
    id BIGSERIAL PRIMARY KEY,
    agenda_id BIGINT NOT NULL REFERENCES research_agendas(id),
    idea_id BIGINT NOT NULL REFERENCES deep_insights(id),
    experiment_run_id BIGINT NOT NULL REFERENCES experiment_runs(id),
    resource_grant_id BIGINT NOT NULL REFERENCES resource_grants(id),
    verdict_hash TEXT NOT NULL,
    attempt_number INTEGER NOT NULL CHECK (attempt_number BETWEEN 1 AND 2),
    idempotency_key TEXT NOT NULL,
    prompt_ref TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE (resource_grant_id, attempt_number),
    UNIQUE (resource_grant_id, idempotency_key)
);

CREATE INDEX IF NOT EXISTS idx_manuscript_gate_attempt_scope_v1
    ON manuscript_gate_attempts_v1(
        agenda_id, experiment_run_id, resource_grant_id, attempt_number
    );

-- Immutable terminal decisions for the manuscript reviewer.  Reviewer route
-- observations prove provider usage, but a successful provider call alone
-- cannot distinguish an explicit refusal from an unparseable answer.

CREATE TABLE IF NOT EXISTS manuscript_gate_records_v1 (
    id BIGSERIAL PRIMARY KEY,
    agenda_id BIGINT NOT NULL REFERENCES research_agendas(id),
    idea_id BIGINT NOT NULL REFERENCES deep_insights(id),
    experiment_run_id BIGINT NOT NULL REFERENCES experiment_runs(id),
    resource_grant_id BIGINT NOT NULL REFERENCES resource_grants(id),
    verdict_hash TEXT NOT NULL,
    disposition TEXT NOT NULL CHECK (
        disposition IN ('approved', 'refused', 'technical_failed')
    ),
    prompt_ref TEXT NOT NULL,
    judgement_json TEXT NOT NULL DEFAULT '{}',
    grant_usage_reservation_id BIGINT
        REFERENCES resource_grant_usage_reservations(id),
    reviewer_ref TEXT,
    reviewer_response_hash TEXT,
    failure_reason TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE (agenda_id, experiment_run_id, verdict_hash),
    UNIQUE (resource_grant_id),
    CHECK (
        (disposition IN ('approved', 'refused')
         AND grant_usage_reservation_id IS NOT NULL
         AND reviewer_ref IS NOT NULL
         AND reviewer_response_hash IS NOT NULL
         AND failure_reason IS NULL)
        OR
        (disposition = 'technical_failed'
         AND failure_reason IS NOT NULL)
    )
);

CREATE INDEX IF NOT EXISTS idx_manuscript_gate_scope_v1
    ON manuscript_gate_records_v1(
        agenda_id, experiment_run_id, disposition, created_at, id
    );
