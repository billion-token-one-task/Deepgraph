-- Proposal generation and experiment execution have different falsification
-- contracts. Keep both immutable records even though deep_insights exposes
-- only the currently active gate snapshot.

CREATE TABLE IF NOT EXISTS candidate_stage_gate_records_v1 (
    id BIGSERIAL PRIMARY KEY,
    agenda_id BIGINT NOT NULL REFERENCES research_agendas(id),
    idea_id BIGINT NOT NULL REFERENCES deep_insights(id),
    stage TEXT NOT NULL CHECK (stage IN ('proposal','experiment')),
    record_json TEXT NOT NULL,
    content_hash TEXT NOT NULL,
    actor TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE (agenda_id, idea_id, stage, content_hash)
);

CREATE INDEX IF NOT EXISTS idx_candidate_stage_gate_scope
    ON candidate_stage_gate_records_v1(agenda_id, idea_id, stage, created_at, id);
