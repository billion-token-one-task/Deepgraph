-- Issue #11/#14 (D3): persist FormatLinter results per manuscript bundle.
-- One row per (selection_id, template_id, rule_set) lint pass. The history
-- is preserved so the dashboard can replay every lint run for a given
-- venue selection (alongside ``manuscript_venue_selections``).

CREATE TABLE IF NOT EXISTS format_lint_runs (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    selection_id      INTEGER,                           -- nullable: ad-hoc lint runs allowed
    template_id       TEXT NOT NULL,
    rule_set          TEXT NOT NULL DEFAULT 'format_linter_v1',
    pass              INTEGER NOT NULL DEFAULT 0,        -- 0/1; SQLite has no bool
    error_count       INTEGER NOT NULL DEFAULT 0,
    warning_count     INTEGER NOT NULL DEFAULT 0,
    info_count        INTEGER NOT NULL DEFAULT 0,
    checks_json       TEXT NOT NULL DEFAULT '[]',
    summary_json      TEXT NOT NULL DEFAULT '{}',
    created_at        TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_format_lint_runs_selection
    ON format_lint_runs(selection_id);

CREATE INDEX IF NOT EXISTS idx_format_lint_runs_template
    ON format_lint_runs(template_id);
