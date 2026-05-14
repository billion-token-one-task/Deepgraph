-- Issue #11/#12 (D1): venue routing decisions for manuscript bundles.
-- A row is written by ``agents.venue_router.route_and_persist`` per
-- (selection_id, rule_set) pair. The history is preserved so dashboard
-- audits and the FormatLinter evidence-gate hook (D3) can replay the
-- chain of decisions for a given selection.

CREATE TABLE IF NOT EXISTS manuscript_venue_selections (
    id                       INTEGER PRIMARY KEY AUTOINCREMENT,
    selection_id             INTEGER NOT NULL,
    chosen_template_id       TEXT,                          -- NULL when all venues blocked
    score                    REAL,
    rationale                TEXT,
    rejected_venues_json     TEXT NOT NULL DEFAULT '[]',
    scoring_breakdown_json   TEXT NOT NULL DEFAULT '{}',
    rule_set                 TEXT NOT NULL DEFAULT 'venues_v1',
    status                   TEXT NOT NULL DEFAULT 'selected',
    created_at               TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_manuscript_venue_selections_selection
    ON manuscript_venue_selections(selection_id);

CREATE INDEX IF NOT EXISTS idx_manuscript_venue_selections_template
    ON manuscript_venue_selections(chosen_template_id);
