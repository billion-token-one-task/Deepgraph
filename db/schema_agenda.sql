-- Schema for issue #9: Agenda-driven autonomous research loop
-- Adds configurable research agenda + selection + review + revision tables
-- on top of existing schema_v2 (deep_insights, experiment_runs, manuscript_runs, submission_bundles).

CREATE TABLE IF NOT EXISTS research_agendas (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    version TEXT NOT NULL DEFAULT 'v1',
    name TEXT NOT NULL,
    description TEXT,
    focus_json TEXT NOT NULL DEFAULT '[]',
    prefer_json TEXT NOT NULL DEFAULT '{}',
    reject_json TEXT NOT NULL DEFAULT '{}',
    required_output_json TEXT NOT NULL DEFAULT '{}',
    raw_config_json TEXT,
    is_active INTEGER NOT NULL DEFAULT 1,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_research_agendas_active ON research_agendas(is_active, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_research_agendas_name ON research_agendas(name);

CREATE TABLE IF NOT EXISTS agenda_selections (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
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
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (agenda_id) REFERENCES research_agendas(id),
    FOREIGN KEY (selected_insight_id) REFERENCES deep_insights(id)
);

CREATE INDEX IF NOT EXISTS idx_agenda_selections_agenda ON agenda_selections(agenda_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_agenda_selections_insight ON agenda_selections(selected_insight_id);
CREATE INDEX IF NOT EXISTS idx_agenda_selections_status ON agenda_selections(status);

CREATE TABLE IF NOT EXISTS agenda_reviews (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
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
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (selection_id) REFERENCES agenda_selections(id)
);

CREATE INDEX IF NOT EXISTS idx_agenda_reviews_selection ON agenda_reviews(selection_id, created_at DESC);

CREATE TABLE IF NOT EXISTS agenda_revision_plans (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    selection_id INTEGER NOT NULL,
    review_id INTEGER NOT NULL,
    rationale TEXT,
    next_experiments_json TEXT NOT NULL DEFAULT '[]',
    status TEXT NOT NULL DEFAULT 'proposed',
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (selection_id) REFERENCES agenda_selections(id),
    FOREIGN KEY (review_id) REFERENCES agenda_reviews(id)
);

CREATE INDEX IF NOT EXISTS idx_agenda_revision_plans_selection ON agenda_revision_plans(selection_id, created_at DESC);
