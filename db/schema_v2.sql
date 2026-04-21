-- DeepGraph v2: Deep Insight Discovery Tables

-- Main output table for paradigm-level and paper-ready insights
CREATE TABLE IF NOT EXISTS deep_insights (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    tier INTEGER NOT NULL,                    -- 1 = paradigm, 2 = paper-ready
    status TEXT DEFAULT 'candidate',          -- candidate|verified|refined|exists|failed
    title TEXT NOT NULL,

    -- Tier 1 (Paradigm Discovery) fields
    formal_structure TEXT,                    -- mathematical/structural description
    field_a TEXT,                             -- JSON {node_id, key_entities, papers}
    field_b TEXT,                             -- JSON {node_id, key_entities, papers}
    transformation TEXT,                      -- formal mapping A → B
    predictions TEXT,                         -- JSON array of testable predictions
    falsification TEXT,                       -- what would disprove this
    adversarial_score REAL,                   -- 0-10 from adversarial challenge
    adversarial_critique TEXT,

    -- Tier 2 (Paper-Ready Ideas) fields
    problem_statement TEXT,
    existing_weakness TEXT,
    proposed_method TEXT,                     -- JSON {name, type, definition, complexity, key_properties}
    experimental_plan TEXT,                   -- JSON {baselines, datasets, metrics, ablations, expected_results, compute_budget}
    related_work_positioning TEXT,

    -- Shared fields
    supporting_papers TEXT DEFAULT '[]',      -- JSON array of paper IDs
    source_node_ids TEXT DEFAULT '[]',        -- JSON array of node IDs
    evidence_summary TEXT,
    signal_mix TEXT,                          -- JSON array of originating signal families
    mechanism_type TEXT,                      -- protocol_artifact|mechanism_mismatch|hidden_variable_bridge|...
    evidence_packet TEXT,                     -- JSON with numeric + non-numeric evidence bundle
    evidence_plan TEXT,                       -- JSON adaptive evidence plan (table/ablation/visualization by claim)
    experimentability TEXT,                   -- easy|medium|hard
    resource_class TEXT DEFAULT 'cpu',        -- cpu|gpu_small|gpu_large
    submission_status TEXT DEFAULT 'not_started',
    novelty_status TEXT,                      -- novel|partially_exists|exists|unchecked
    novelty_report TEXT,                      -- JSON from EvoScientist verification
    generation_tokens INTEGER DEFAULT 0,
    llm_calls INTEGER DEFAULT 0,
    evoscientist_workdir TEXT,                -- path to EvoScientist run
    generation_run_id TEXT,
    source_signal_ids TEXT,
    source_paper_ids TEXT,
    prompt_version TEXT,
    model_version TEXT,
    exemplars_used TEXT,
    token_cost_usd REAL,
    wall_clock_seconds REAL,
    outcome TEXT DEFAULT 'pending',
    outcome_reason TEXT,
    outcome_updated_at TIMESTAMP,
    human_score REAL,
    human_notes TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Signal harvester: cross-node entity overlap
CREATE TABLE IF NOT EXISTS node_entity_overlap (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    node_a_id TEXT NOT NULL,
    node_b_id TEXT NOT NULL,
    shared_entity_count INTEGER NOT NULL,
    shared_entity_ids TEXT,                   -- JSON array (top 20)
    shared_entity_types TEXT,                 -- JSON {type: count}
    taxonomic_distance INTEGER DEFAULT 0,     -- hops in taxonomy tree
    overlap_score REAL DEFAULT 0,             -- normalized overlap
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(node_a_id, node_b_id)
);

-- Signal harvester: convergent pattern matches
CREATE TABLE IF NOT EXISTS pattern_matches (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    pattern_a_id INTEGER NOT NULL,
    pattern_b_id INTEGER NOT NULL,
    similarity_score REAL NOT NULL,
    node_a_id TEXT,
    node_b_id TEXT,
    shared_tokens TEXT,                       -- JSON array of overlapping keywords
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(pattern_a_id, pattern_b_id)
);

-- Signal harvester: contradiction clusters
CREATE TABLE IF NOT EXISTS contradiction_clusters (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    theme TEXT NOT NULL,
    contradiction_ids TEXT NOT NULL,          -- JSON array
    shared_entities TEXT,                     -- JSON array of entity names
    cluster_size INTEGER DEFAULT 0,
    node_ids TEXT,                            -- JSON array of affected nodes
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Signal harvester: performance plateaus
CREATE TABLE IF NOT EXISTS performance_plateaus (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    node_id TEXT NOT NULL,
    dataset_name TEXT NOT NULL,
    metric_name TEXT NOT NULL,
    top_methods TEXT NOT NULL,                -- JSON array of {method, value}
    spread REAL NOT NULL,                     -- max - min among top methods
    spread_pct REAL NOT NULL,                 -- spread as % of max
    method_count INTEGER DEFAULT 0,
    paper_count INTEGER DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(node_id, dataset_name, metric_name)
);

-- ===== SciForge: Experiment Validation Tables =====

-- Experiment runs launched from deep_insights
CREATE TABLE IF NOT EXISTS experiment_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    deep_insight_id INTEGER NOT NULL REFERENCES deep_insights(id),
    status TEXT DEFAULT 'pending',             -- pending|scaffolding|reproducing|testing|completed|failed
    phase TEXT DEFAULT 'setup',                -- setup|reproduction|hypothesis_testing
    workdir TEXT,
    codebase_url TEXT,                         -- GitHub repo used as baseline
    codebase_ref TEXT,                         -- branch/commit of baseline
    program_md TEXT,                           -- generated program.md content
    proxy_config TEXT,                         -- JSON {data_fraction, max_epochs, early_stop_threshold, time_budget_seconds}
    success_criteria TEXT,                     -- JSON {exciting, solid, disappointing} metric thresholds
    iterations_total INTEGER DEFAULT 0,
    iterations_kept INTEGER DEFAULT 0,
    baseline_metric_name TEXT,
    baseline_metric_value REAL,
    best_metric_value REAL,
    hypothesis_verdict TEXT,                   -- confirmed|refuted|inconclusive|NULL
    effect_size REAL,                          -- best - baseline
    effect_pct REAL,                           -- effect_size / baseline * 100
    resource_class TEXT DEFAULT 'cpu',         -- cpu|gpu_small|gpu_large
    submission_bundle_id INTEGER,
    error_message TEXT,
    started_at TIMESTAMP,
    completed_at TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Individual experiment iterations within a run
CREATE TABLE IF NOT EXISTS experiment_iterations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id INTEGER NOT NULL REFERENCES experiment_runs(id),
    iteration_number INTEGER NOT NULL,
    phase TEXT NOT NULL,                       -- reproduction|hypothesis_testing
    code_diff TEXT,                            -- git diff of the change
    commit_hash TEXT,                          -- short git commit hash
    metric_value REAL,
    metric_name TEXT,
    peak_memory_mb REAL,
    duration_seconds REAL,
    status TEXT DEFAULT 'keep',                -- keep|discard|crash
    description TEXT,                          -- what the agent tried
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Knowledge graph feedback from experimental evidence
CREATE TABLE IF NOT EXISTS experimental_claims (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id INTEGER NOT NULL REFERENCES experiment_runs(id),
    deep_insight_id INTEGER NOT NULL REFERENCES deep_insights(id),
    claim_text TEXT NOT NULL,
    claim_type TEXT DEFAULT 'experimental',    -- experimental|reproduction|refutation
    verdict TEXT NOT NULL,                     -- confirmed|refuted|inconclusive
    effect_size REAL,
    confidence REAL,                           -- 0-1 statistical confidence
    p_value REAL,
    supporting_data TEXT,                      -- JSON with full experimental details
    source_paper_ids TEXT,                     -- JSON array of cited/source paper ids
    source_node_ids TEXT,                      -- JSON array of supporting node ids
    cascaded INTEGER DEFAULT 0,               -- 1 if cascade reasoning has processed this
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Meta-learning: track record of hypothesis quality by signal type
CREATE TABLE IF NOT EXISTS discovery_track_record (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    signal_type TEXT NOT NULL UNIQUE,          -- entity_overlap|pattern_match|contradiction_cluster|plateau|insight_derived
    hypothesis_count INTEGER DEFAULT 0,
    confirmed_count INTEGER DEFAULT 0,
    refuted_count INTEGER DEFAULT 0,
    inconclusive_count INTEGER DEFAULT 0,
    avg_effect_size REAL DEFAULT 0,
    avg_adversarial_score REAL DEFAULT 0,
    hit_rate REAL DEFAULT 0,                   -- confirmed / (confirmed + refuted)
    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Auto Research queue/state machine for background closed-loop execution
CREATE TABLE IF NOT EXISTS auto_research_jobs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    deep_insight_id INTEGER NOT NULL UNIQUE REFERENCES deep_insights(id),
    status TEXT DEFAULT 'queued',            -- queued|verifying|researching|eligible|running_experiment|completed|blocked|failed
    stage TEXT DEFAULT 'queued',             -- fine-grained sub-stage for dashboard
    cpu_eligible INTEGER,                    -- 1 eligible, 0 ineligible, NULL unchecked
    cpu_reason TEXT,
    resource_class TEXT DEFAULT 'cpu',
    scheduler_priority INTEGER DEFAULT 0,
    assigned_worker TEXT,
    artifact_bundle_id INTEGER,
    experiment_run_id INTEGER REFERENCES experiment_runs(id),
    research_workdir TEXT,
    last_note TEXT,
    last_error TEXT,
    last_checked_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Indexes
CREATE INDEX IF NOT EXISTS idx_deep_insights_tier ON deep_insights(tier);
CREATE INDEX IF NOT EXISTS idx_deep_insights_status ON deep_insights(status);
CREATE INDEX IF NOT EXISTS idx_deep_insights_novelty ON deep_insights(novelty_status);
CREATE INDEX IF NOT EXISTS idx_node_entity_overlap_score ON node_entity_overlap(overlap_score DESC);
CREATE INDEX IF NOT EXISTS idx_pattern_matches_score ON pattern_matches(similarity_score DESC);
CREATE INDEX IF NOT EXISTS idx_contradiction_clusters_size ON contradiction_clusters(cluster_size DESC);
CREATE INDEX IF NOT EXISTS idx_performance_plateaus_node ON performance_plateaus(node_id);
CREATE INDEX IF NOT EXISTS idx_experiment_runs_insight ON experiment_runs(deep_insight_id);
CREATE INDEX IF NOT EXISTS idx_experiment_runs_status ON experiment_runs(status);
CREATE INDEX IF NOT EXISTS idx_experiment_iterations_run ON experiment_iterations(run_id);
CREATE INDEX IF NOT EXISTS idx_experimental_claims_run ON experimental_claims(run_id);
CREATE INDEX IF NOT EXISTS idx_experimental_claims_insight ON experimental_claims(deep_insight_id);
CREATE INDEX IF NOT EXISTS idx_auto_research_jobs_status ON auto_research_jobs(status);

-- Mechanism-first discovery signal tables
CREATE TABLE IF NOT EXISTS mechanism_mismatches (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    node_id TEXT REFERENCES taxonomy_nodes(id),
    theme TEXT NOT NULL,
    explanation_variants TEXT,              -- JSON array
    paper_ids TEXT,                         -- JSON array
    support_count INTEGER DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS protocol_artifacts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    node_id TEXT REFERENCES taxonomy_nodes(id),
    artifact_type TEXT NOT NULL,
    summary TEXT NOT NULL,
    paper_ids TEXT,                         -- JSON array
    support_count INTEGER DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS negative_space_gaps (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    node_id TEXT REFERENCES taxonomy_nodes(id),
    gap_type TEXT NOT NULL,
    summary TEXT NOT NULL,
    paper_ids TEXT,                         -- JSON array
    support_count INTEGER DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS hidden_variable_bridges (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    node_a_id TEXT NOT NULL REFERENCES taxonomy_nodes(id),
    node_b_id TEXT NOT NULL REFERENCES taxonomy_nodes(id),
    shared_factor TEXT NOT NULL,
    paper_ids TEXT,                         -- JSON array
    score REAL DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(node_a_id, node_b_id, shared_factor)
);

CREATE TABLE IF NOT EXISTS claim_method_gaps (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    node_id TEXT REFERENCES taxonomy_nodes(id),
    summary TEXT NOT NULL,
    paper_ids TEXT,                         -- JSON array
    support_count INTEGER DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- GPU orchestration and artifact tracking
CREATE TABLE IF NOT EXISTS gpu_workers (
    id TEXT PRIMARY KEY,
    hostname TEXT,
    gpu_index INTEGER DEFAULT 0,
    gpu_model TEXT,
    total_mem_gb REAL DEFAULT 0,
    status TEXT DEFAULT 'idle',             -- idle|busy|offline
    heartbeat_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    metadata TEXT
);

CREATE TABLE IF NOT EXISTS gpu_jobs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    deep_insight_id INTEGER REFERENCES deep_insights(id),
    experiment_run_id INTEGER REFERENCES experiment_runs(id),
    resource_class TEXT DEFAULT 'gpu_small',
    gpu_count INTEGER DEFAULT 1,
    vram_required_gb REAL DEFAULT 0,
    timeout_s INTEGER DEFAULT 0,
    priority INTEGER DEFAULT 0,
    status TEXT DEFAULT 'queued',           -- queued|running|completed|failed|canceled
    assigned_worker TEXT REFERENCES gpu_workers(id),
    artifact_uri TEXT,
    error_message TEXT,
    started_at TIMESTAMP,
    completed_at TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS experiment_artifacts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id INTEGER NOT NULL REFERENCES experiment_runs(id),
    artifact_type TEXT NOT NULL,            -- log|metric|plot|bundle|source_data
    path TEXT NOT NULL,
    metric_key TEXT,
    metric_value REAL,
    metadata TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Manuscript and submission bundles
CREATE TABLE IF NOT EXISTS manuscript_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    experiment_run_id INTEGER REFERENCES experiment_runs(id),
    deep_insight_id INTEGER REFERENCES deep_insights(id),
    status TEXT DEFAULT 'drafting',         -- drafting|bundle_ready|failed
    canonical_state TEXT,
    workdir TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS manuscript_assets (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    manuscript_run_id INTEGER NOT NULL REFERENCES manuscript_runs(id),
    asset_type TEXT NOT NULL,               -- tex|figure|bib|cover_letter|metadata
    label TEXT,
    path TEXT NOT NULL,
    metadata TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS submission_bundles (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    manuscript_run_id INTEGER NOT NULL REFERENCES manuscript_runs(id),
    bundle_format TEXT NOT NULL,            -- conference|journal
    status TEXT DEFAULT 'ready',
    bundle_path TEXT NOT NULL,
    manifest_path TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_protocol_artifacts_node ON protocol_artifacts(node_id);
CREATE INDEX IF NOT EXISTS idx_negative_space_node ON negative_space_gaps(node_id);
CREATE INDEX IF NOT EXISTS idx_hidden_bridges_score ON hidden_variable_bridges(score DESC);
CREATE INDEX IF NOT EXISTS idx_claim_method_gaps_node ON claim_method_gaps(node_id);
CREATE INDEX IF NOT EXISTS idx_gpu_workers_status ON gpu_workers(status);
CREATE INDEX IF NOT EXISTS idx_gpu_jobs_status ON gpu_jobs(status);
CREATE INDEX IF NOT EXISTS idx_experiment_artifacts_run ON experiment_artifacts(run_id);
CREATE INDEX IF NOT EXISTS idx_manuscript_runs_experiment ON manuscript_runs(experiment_run_id);
CREATE INDEX IF NOT EXISTS idx_submission_bundles_run ON submission_bundles(manuscript_run_id);
