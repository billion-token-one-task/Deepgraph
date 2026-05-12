-- PostgreSQL-compatible schema for issue #9 (agenda-driven research loop).
-- Equivalent to schema_agenda.sql, but uses BIGSERIAL/TIMESTAMP/BOOLEAN.
-- Loaded by db/database.py when DEEPGRAPH_DATABASE_URL is set.

CREATE TABLE IF NOT EXISTS research_agendas (
    id BIGSERIAL PRIMARY KEY,
    version TEXT NOT NULL DEFAULT 'v1',
    name TEXT NOT NULL,
    description TEXT,
    focus_json TEXT NOT NULL DEFAULT '[]',
    prefer_json TEXT NOT NULL DEFAULT '{}',
    reject_json TEXT NOT NULL DEFAULT '{}',
    required_output_json TEXT NOT NULL DEFAULT '{}',
    raw_config_json TEXT,
    is_active INTEGER NOT NULL DEFAULT 1,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_research_agendas_active ON research_agendas(is_active, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_research_agendas_name ON research_agendas(name);

CREATE TABLE IF NOT EXISTS agenda_selections (
    id BIGSERIAL PRIMARY KEY,
    agenda_id INTEGER NOT NULL,
    selected_insight_id INTEGER,
    score REAL,
    rationale TEXT,
    rejected_candidates_json TEXT NOT NULL DEFAULT '[]',
    scoring_breakdown_json TEXT NOT NULL DEFAULT '{}',
    status TEXT NOT NULL DEFAULT 'pending',
    auto_research_job_id INTEGER,
    experiment_run_id INTEGER,
    manuscript_run_id INTEGER,
    submission_bundle_id INTEGER,
    error_message TEXT,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_agenda_selections_agenda ON agenda_selections(agenda_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_agenda_selections_insight ON agenda_selections(selected_insight_id);
CREATE INDEX IF NOT EXISTS idx_agenda_selections_status ON agenda_selections(status);

CREATE TABLE IF NOT EXISTS agenda_reviews (
    id BIGSERIAL PRIMARY KEY,
    selection_id INTEGER NOT NULL,
    submission_bundle_id INTEGER,
    manuscript_run_id INTEGER,
    reviewer TEXT NOT NULL DEFAULT 'internal_evidence_gate',
    recommendation TEXT NOT NULL,
    confidence REAL,
    strengths_json TEXT NOT NULL DEFAULT '[]',
    weaknesses_json TEXT NOT NULL DEFAULT '[]',
    required_revisions_json TEXT NOT NULL DEFAULT '[]',
    evidence_blockers_json TEXT NOT NULL DEFAULT '[]',
    raw_review_json TEXT,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_agenda_reviews_selection ON agenda_reviews(selection_id, created_at DESC);

CREATE TABLE IF NOT EXISTS agenda_revision_plans (
    id BIGSERIAL PRIMARY KEY,
    selection_id INTEGER NOT NULL,
    review_id INTEGER NOT NULL,
    rationale TEXT,
    next_experiments_json TEXT NOT NULL DEFAULT '[]',
    status TEXT NOT NULL DEFAULT 'proposed',
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_agenda_revision_plans_selection ON agenda_revision_plans(selection_id, created_at DESC);

CREATE TABLE IF NOT EXISTS agenda_evidence_gates (
    id BIGSERIAL PRIMARY KEY,
    selection_id INTEGER NOT NULL,
    experiment_run_id INTEGER,
    status TEXT NOT NULL,
    blockers_json TEXT NOT NULL DEFAULT '[]',
    metrics_summary_json TEXT NOT NULL DEFAULT '{}',
    packet_path TEXT,
    rule_set TEXT NOT NULL DEFAULT 'agenda_v1_default',
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_agenda_evidence_gates_selection
    ON agenda_evidence_gates(selection_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_agenda_evidence_gates_status
    ON agenda_evidence_gates(status);
