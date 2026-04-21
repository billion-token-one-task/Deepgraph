-- Insight feedback loop: provenance, outcomes, event log, signal harvester telemetry

CREATE TABLE IF NOT EXISTS insight_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    scope TEXT NOT NULL,                     -- insights | deep_insights
    insight_id INTEGER NOT NULL,
    from_outcome TEXT,
    to_outcome TEXT NOT NULL,
    reason TEXT,
    triggered_by TEXT DEFAULT 'system',    -- system | pipeline | user | api | verifier | experiment
    meta_json TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_insight_events_scope_insight ON insight_events(scope, insight_id);
CREATE INDEX IF NOT EXISTS idx_insight_events_to_outcome ON insight_events(to_outcome);

CREATE TABLE IF NOT EXISTS harvester_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    pattern_name TEXT NOT NULL,
    query_hash TEXT,
    node_id TEXT,
    candidate_count INTEGER DEFAULT 0,
    execution_time_ms INTEGER,
    meta_json TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_harvester_runs_pattern ON harvester_runs(pattern_name);
