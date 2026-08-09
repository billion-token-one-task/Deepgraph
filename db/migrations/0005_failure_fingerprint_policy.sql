-- Stable execution failure fingerprints and recovery decisions.

CREATE TABLE IF NOT EXISTS experiment_failure_fingerprints_v1 (
    id BIGSERIAL PRIMARY KEY,
    agenda_id BIGINT NOT NULL REFERENCES research_agendas(id),
    idea_id BIGINT NOT NULL REFERENCES deep_insights(id),
    experiment_run_id BIGINT NOT NULL REFERENCES experiment_runs(id),
    resource_grant_id BIGINT REFERENCES resource_grants(id),
    reason_code TEXT NOT NULL,
    fingerprint TEXT NOT NULL,
    code_hash TEXT NOT NULL,
    environment_hash TEXT NOT NULL,
    detail TEXT NOT NULL,
    recovery_action TEXT NOT NULL,
    recovery_json TEXT NOT NULL,
    occurrences INTEGER NOT NULL DEFAULT 1 CHECK (occurrences > 0),
    first_seen_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    last_seen_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE (agenda_id, idea_id, fingerprint)
);

CREATE INDEX IF NOT EXISTS idx_failure_fingerprint_run
    ON experiment_failure_fingerprints_v1(experiment_run_id, last_seen_at DESC);
