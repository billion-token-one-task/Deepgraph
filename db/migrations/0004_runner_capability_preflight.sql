-- Candidate execution requirements and cheap grant-before-compute preflight.
-- Repository identities remain opaque data; authorization binds to protocols,
-- schemas, metrics, resources, revisions, and runner adapter contracts.

CREATE TABLE IF NOT EXISTS candidate_execution_requirements_v1 (
    id BIGSERIAL PRIMARY KEY,
    agenda_id BIGINT NOT NULL REFERENCES research_agendas(id),
    idea_id BIGINT NOT NULL REFERENCES deep_insights(id),
    schema_version TEXT NOT NULL,
    source_plan_hash TEXT NOT NULL,
    requirements_hash TEXT NOT NULL,
    requirements_json TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'declared'
        CHECK (status IN ('declared', 'superseded')),
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE (agenda_id, idea_id, requirements_hash)
);

CREATE INDEX IF NOT EXISTS idx_candidate_execution_requirements_active
    ON candidate_execution_requirements_v1(agenda_id, idea_id, status, id DESC);

CREATE TABLE IF NOT EXISTS candidate_preflight_results_v1 (
    id BIGSERIAL PRIMARY KEY,
    agenda_id BIGINT NOT NULL REFERENCES research_agendas(id),
    idea_id BIGINT NOT NULL REFERENCES deep_insights(id),
    requirement_id BIGINT NOT NULL REFERENCES candidate_execution_requirements_v1(id),
    adapter_id TEXT,
    adapter_version TEXT,
    selected_backend TEXT,
    status TEXT NOT NULL CHECK (status IN ('passed', 'deferred', 'failed')),
    reason_codes_json TEXT NOT NULL,
    checks_json TEXT NOT NULL,
    environment_json TEXT NOT NULL,
    dataset_revision TEXT,
    model_revision TEXT,
    idempotency_key TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE (requirement_id, idempotency_key)
);

CREATE INDEX IF NOT EXISTS idx_candidate_preflight_latest
    ON candidate_preflight_results_v1(agenda_id, idea_id, created_at DESC, id DESC);

ALTER TABLE IF EXISTS resource_grants
    ADD COLUMN IF NOT EXISTS preflight_result_id BIGINT
        REFERENCES candidate_preflight_results_v1(id);

CREATE INDEX IF NOT EXISTS idx_resource_grants_preflight
    ON resource_grants(preflight_result_id)
    WHERE preflight_result_id IS NOT NULL;
