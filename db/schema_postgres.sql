CREATE EXTENSION IF NOT EXISTS vector;

-- DeepGraph Database Schema

CREATE TABLE IF NOT EXISTS papers (
    id TEXT PRIMARY KEY,              -- arXiv ID e.g. "2301.07041"
    title TEXT NOT NULL,
    authors TEXT,                      -- JSON array
    abstract TEXT,
    categories TEXT,                   -- JSON array
    published_date TEXT,
    pdf_url TEXT,
    full_text TEXT,                    -- extracted text
    appendix_text TEXT,                -- extracted appendix/supplement text
    status TEXT DEFAULT 'ingested',    -- ingested|extracted|abstracted|reasoned|error
    error_msg TEXT,
    token_cost INTEGER DEFAULT 0,
    processing_stage TEXT DEFAULT 'ingested',
    arxiv_base_id TEXT,
    processing_attempts INTEGER DEFAULT 0,
    stage_last_error TEXT,
    stage_locked_by TEXT,
    stage_started_at TIMESTAMP,
    stage_completed_at TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS claims (
    id BIGSERIAL PRIMARY KEY,
    paper_id TEXT NOT NULL REFERENCES papers(id),
    claim_text TEXT NOT NULL,
    claim_type TEXT,                   -- performance|method|finding|limitation
    method_name TEXT,
    dataset_name TEXT,
    metric_name TEXT,
    metric_value DOUBLE PRECISION,
    evidence_location TEXT,            -- "Table 2, row 3"
    conditions TEXT,                   -- JSON: experimental conditions
    embedding TEXT,                    -- JSON: vector for similarity search
    source_quote TEXT,                 -- verbatim grounding snippet from full_text
    char_start INTEGER,                -- span in the text region named by grounding_source_field
    char_end INTEGER,
    grounding_status TEXT,             -- verified|weak|unverified|no_quote
    grounding_score DOUBLE PRECISION,
    grounding_source_field TEXT,       -- full_text|appendix_text
    source_paper_ids TEXT,
    source_node_ids TEXT,
    claim_key TEXT,
    embedding_vector VECTOR(1536),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS methods (
    id BIGSERIAL PRIMARY KEY,
    name TEXT NOT NULL UNIQUE,
    category TEXT,
    description TEXT,
    key_innovation TEXT,
    first_paper_id TEXT REFERENCES papers(id),
    builds_on TEXT,                    -- JSON array of method names
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS patterns (
    id BIGSERIAL PRIMARY KEY,
    pattern_text TEXT NOT NULL,        -- e.g. "automation -> skill atrophy"
    pattern_type TEXT,                 -- problem|solution|phenomenon
    abstraction_level TEXT,
    domain_count INTEGER DEFAULT 1,
    domains TEXT,                      -- JSON array: ["education", "aviation", ...]
    claim_ids TEXT,                    -- JSON array of claim IDs
    node_id TEXT REFERENCES taxonomy_nodes(id),
    source_claims TEXT,                -- JSON array of claim IDs
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS contradictions (
    id BIGSERIAL PRIMARY KEY,
    claim_a_id INTEGER REFERENCES claims(id),
    claim_b_id INTEGER REFERENCES claims(id),
    description TEXT NOT NULL,
    condition_diff TEXT,               -- what differs between the two claims
    hypothesis TEXT,                   -- generated hypothesis to resolve
    severity TEXT DEFAULT 'medium',    -- low|medium|high
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS gaps (
    id BIGSERIAL PRIMARY KEY,
    problem_pattern_id INTEGER REFERENCES patterns(id),
    solution_pattern_id INTEGER REFERENCES patterns(id),
    gap_description TEXT NOT NULL,
    missing_domain TEXT,               -- which domain lacks this solution
    evidence_papers TEXT,              -- JSON array of paper IDs supporting the gap
    research_proposal TEXT,            -- auto-generated proposal
    value_score DOUBLE PRECISION,                  -- 0-5
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS stats (
    id BIGSERIAL PRIMARY KEY,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    papers_total INTEGER,
    papers_processed INTEGER,
    claims_total INTEGER,
    patterns_total INTEGER,
    contradictions_total INTEGER,
    gaps_total INTEGER,
    tokens_consumed INTEGER
);

-- ===== NEW: Taxonomy + Results Tables =====

CREATE TABLE IF NOT EXISTS taxonomy_nodes (
    id TEXT PRIMARY KEY,                  -- e.g. "dl.cv.detection"
    name TEXT NOT NULL,                   -- e.g. "Object Detection"
    parent_id TEXT REFERENCES taxonomy_nodes(id),
    depth INTEGER NOT NULL DEFAULT 0,
    description TEXT,
    sort_order INTEGER DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS paper_taxonomy (
    paper_id TEXT NOT NULL REFERENCES papers(id),
    node_id TEXT NOT NULL REFERENCES taxonomy_nodes(id),
    confidence DOUBLE PRECISION DEFAULT 1.0,          -- 0-1 confidence of classification
    PRIMARY KEY (paper_id, node_id)
);

CREATE TABLE IF NOT EXISTS results (
    id BIGSERIAL PRIMARY KEY,
    paper_id TEXT NOT NULL REFERENCES papers(id),
    node_id TEXT REFERENCES taxonomy_nodes(id),
    method_name TEXT NOT NULL,
    dataset_name TEXT NOT NULL,
    metric_name TEXT NOT NULL,
    metric_value DOUBLE PRECISION,
    metric_unit TEXT,                      -- e.g. "%", "ms", "FLOPs"
    is_sota INTEGER DEFAULT 0,            -- 1 if claimed SOTA
    evidence_location TEXT,
    conditions TEXT,                       -- JSON
    source_quote TEXT,
    char_start INTEGER,
    char_end INTEGER,
    grounding_status TEXT,
    grounding_score DOUBLE PRECISION,
    grounding_source_field TEXT,       -- full_text|appendix_text
    result_key TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS result_taxonomy (
    result_id INTEGER NOT NULL REFERENCES results(id),
    node_id TEXT NOT NULL REFERENCES taxonomy_nodes(id),
    PRIMARY KEY (result_id, node_id)
);

CREATE TABLE IF NOT EXISTS matrix_gaps (
    id BIGSERIAL PRIMARY KEY,
    node_id TEXT NOT NULL REFERENCES taxonomy_nodes(id),
    method_name TEXT NOT NULL,
    dataset_name TEXT NOT NULL,
    metric_name TEXT,
    gap_description TEXT NOT NULL,
    research_proposal TEXT,
    value_score DOUBLE PRECISION,                      -- 0-5
    evidence_paper_ids TEXT,               -- JSON array
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS paper_insights (
    paper_id TEXT PRIMARY KEY REFERENCES papers(id),
    plain_summary TEXT,
    problem_statement TEXT,
    approach_summary TEXT,
    work_type TEXT,
    key_findings TEXT,                     -- JSON array
    limitations TEXT,                      -- JSON array
    open_questions TEXT,                   -- JSON array
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS insights (
    id BIGSERIAL PRIMARY KEY,
    node_id TEXT NOT NULL REFERENCES taxonomy_nodes(id),
    insight_type TEXT NOT NULL,
    title TEXT NOT NULL,
    hypothesis TEXT NOT NULL,
    evidence TEXT,
    experiment TEXT,
    impact TEXT,
    novelty_score DOUBLE PRECISION DEFAULT 0,
    feasibility_score DOUBLE PRECISION DEFAULT 0,
    supporting_papers TEXT,
    generation_run_id TEXT,
    source_signal_ids TEXT,
    source_paper_ids TEXT,
    source_node_ids TEXT,
    prompt_version TEXT,
    model_version TEXT,
    exemplars_used TEXT,
    token_cost_usd DOUBLE PRECISION,
    wall_clock_seconds DOUBLE PRECISION,
    outcome TEXT DEFAULT 'pending',
    outcome_reason TEXT,
    outcome_updated_at TIMESTAMP,
    human_score DOUBLE PRECISION,
    human_notes TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS node_summaries (
    node_id TEXT PRIMARY KEY REFERENCES taxonomy_nodes(id),
    audience TEXT DEFAULT 'general',
    overview TEXT,
    why_it_matters TEXT,
    what_people_are_building TEXT,         -- JSON array of objects
    common_patterns TEXT,                  -- JSON array of strings
    common_methods TEXT,                   -- JSON array of strings
    common_datasets TEXT,                  -- JSON array of strings
    current_gaps TEXT,                     -- JSON array of objects
    starter_questions TEXT,                -- JSON array of strings
    generated_from_papers TEXT,            -- JSON array of paper IDs
    paper_count INTEGER DEFAULT 0,
    result_count INTEGER DEFAULT 0,
    generated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS graph_entities (
    id TEXT PRIMARY KEY,
    canonical_name TEXT NOT NULL,
    entity_type TEXT NOT NULL,
    normalized_name TEXT NOT NULL,
    description TEXT,
    aliases TEXT,                       -- JSON array
    metadata TEXT,                      -- JSON object
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS paper_entity_mentions (
    id BIGSERIAL PRIMARY KEY,
    paper_id TEXT NOT NULL REFERENCES papers(id),
    node_id TEXT REFERENCES taxonomy_nodes(id),
    entity_id TEXT NOT NULL REFERENCES graph_entities(id),
    mention_text TEXT,
    mention_role TEXT,
    confidence DOUBLE PRECISION DEFAULT 1.0,
    evidence_location TEXT,
    source_text TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS graph_relations (
    id BIGSERIAL PRIMARY KEY,
    paper_id TEXT NOT NULL REFERENCES papers(id),
    node_id TEXT REFERENCES taxonomy_nodes(id),
    subject_entity_id TEXT NOT NULL REFERENCES graph_entities(id),
    predicate TEXT NOT NULL,
    object_entity_id TEXT NOT NULL REFERENCES graph_entities(id),
    confidence DOUBLE PRECISION DEFAULT 1.0,
    evidence_location TEXT,
    source_text TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS node_graph_summaries (
    node_id TEXT PRIMARY KEY REFERENCES taxonomy_nodes(id),
    top_entities TEXT,                  -- JSON array of objects
    top_relations TEXT,                 -- JSON array of objects
    top_entity_types TEXT,              -- JSON array of objects
    generated_from_papers TEXT,         -- JSON array of paper IDs
    paper_count INTEGER DEFAULT 0,
    entity_count INTEGER DEFAULT 0,
    relation_count INTEGER DEFAULT 0,
    generated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS node_opportunities (
    id BIGSERIAL PRIMARY KEY,
    node_id TEXT NOT NULL REFERENCES taxonomy_nodes(id),
    opportunity_type TEXT NOT NULL,
    title TEXT NOT NULL,
    description TEXT NOT NULL,
    why_now TEXT,
    value_score DOUBLE PRECISION DEFAULT 0,
    confidence DOUBLE PRECISION DEFAULT 0,
    signal_counts TEXT,                  -- JSON object
    evidence_paper_ids TEXT,             -- JSON array
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS entity_resolutions (
    entity_id TEXT PRIMARY KEY REFERENCES graph_entities(id),
    canonical_entity_id TEXT NOT NULL REFERENCES graph_entities(id),
    status TEXT DEFAULT 'canonical',      -- canonical|merged
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS entity_merge_candidates (
    id BIGSERIAL PRIMARY KEY,
    entity_type TEXT NOT NULL,
    primary_entity_id TEXT NOT NULL REFERENCES graph_entities(id),
    candidate_entity_id TEXT NOT NULL REFERENCES graph_entities(id),
    similarity_score DOUBLE PRECISION NOT NULL,
    rationale TEXT,
    status TEXT DEFAULT 'pending',        -- pending|accepted|rejected
    generated_by TEXT DEFAULT 'heuristic',
    decision_note TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(primary_entity_id, candidate_entity_id)
);

-- Indexes
CREATE INDEX IF NOT EXISTS idx_papers_status ON papers(status);
CREATE INDEX IF NOT EXISTS idx_papers_arxiv_base ON papers(arxiv_base_id);
CREATE INDEX IF NOT EXISTS idx_papers_processing_stage ON papers(processing_stage);
CREATE INDEX IF NOT EXISTS idx_claims_paper ON claims(paper_id);
CREATE INDEX IF NOT EXISTS idx_claims_type ON claims(claim_type);
CREATE INDEX IF NOT EXISTS idx_claims_method ON claims(method_name);
CREATE UNIQUE INDEX IF NOT EXISTS idx_claims_claim_key ON claims(claim_key);
CREATE INDEX IF NOT EXISTS idx_claims_embedding_vector ON claims USING hnsw (embedding_vector vector_cosine_ops);
CREATE INDEX IF NOT EXISTS idx_patterns_type ON patterns(pattern_type);
CREATE INDEX IF NOT EXISTS idx_patterns_node ON patterns(node_id);
CREATE INDEX IF NOT EXISTS idx_insights_node ON insights(node_id);
CREATE INDEX IF NOT EXISTS idx_insights_type ON insights(insight_type);
CREATE INDEX IF NOT EXISTS idx_taxonomy_parent ON taxonomy_nodes(parent_id);
CREATE INDEX IF NOT EXISTS idx_paper_taxonomy_node ON paper_taxonomy(node_id);
CREATE INDEX IF NOT EXISTS idx_paper_taxonomy_paper ON paper_taxonomy(paper_id);
CREATE INDEX IF NOT EXISTS idx_results_paper ON results(paper_id);
CREATE INDEX IF NOT EXISTS idx_results_node ON results(node_id);
CREATE UNIQUE INDEX IF NOT EXISTS idx_results_result_key ON results(result_key);
CREATE INDEX IF NOT EXISTS idx_result_taxonomy_node ON result_taxonomy(node_id);
CREATE INDEX IF NOT EXISTS idx_result_taxonomy_result ON result_taxonomy(result_id);
CREATE INDEX IF NOT EXISTS idx_results_method ON results(method_name);
CREATE INDEX IF NOT EXISTS idx_results_dataset ON results(dataset_name);
CREATE INDEX IF NOT EXISTS idx_matrix_gaps_node ON matrix_gaps(node_id);
CREATE INDEX IF NOT EXISTS idx_paper_insights_work_type ON paper_insights(work_type);
CREATE UNIQUE INDEX IF NOT EXISTS idx_graph_entities_type_name ON graph_entities(entity_type, normalized_name);
CREATE INDEX IF NOT EXISTS idx_paper_entity_mentions_paper ON paper_entity_mentions(paper_id);
CREATE INDEX IF NOT EXISTS idx_paper_entity_mentions_node ON paper_entity_mentions(node_id);
CREATE INDEX IF NOT EXISTS idx_paper_entity_mentions_entity ON paper_entity_mentions(entity_id);
CREATE INDEX IF NOT EXISTS idx_graph_relations_paper ON graph_relations(paper_id);
CREATE INDEX IF NOT EXISTS idx_graph_relations_node ON graph_relations(node_id);
CREATE INDEX IF NOT EXISTS idx_graph_relations_subject ON graph_relations(subject_entity_id);
CREATE INDEX IF NOT EXISTS idx_graph_relations_object ON graph_relations(object_entity_id);
CREATE INDEX IF NOT EXISTS idx_node_opportunities_node ON node_opportunities(node_id);
CREATE INDEX IF NOT EXISTS idx_node_opportunities_type ON node_opportunities(opportunity_type);
CREATE INDEX IF NOT EXISTS idx_entity_resolutions_canonical ON entity_resolutions(canonical_entity_id);
CREATE INDEX IF NOT EXISTS idx_entity_merge_candidates_status ON entity_merge_candidates(status);
CREATE INDEX IF NOT EXISTS idx_entity_merge_candidates_type ON entity_merge_candidates(entity_type);
-- DeepGraph v2: Deep Insight Discovery Tables

-- Main output table for paradigm-level and paper-ready insights
CREATE TABLE IF NOT EXISTS deep_insights (
    id BIGSERIAL PRIMARY KEY,
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
    adversarial_score DOUBLE PRECISION,                   -- 0-10 from adversarial challenge
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
    token_cost_usd DOUBLE PRECISION,
    wall_clock_seconds DOUBLE PRECISION,
    outcome TEXT DEFAULT 'pending',
    outcome_reason TEXT,
    outcome_updated_at TIMESTAMP,
    human_score DOUBLE PRECISION,
    human_notes TEXT,
    workspace_root TEXT,
    experiment_root TEXT,
    plan_root TEXT,
    paper_root TEXT,
    canonical_run_id INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Signal harvester: cross-node entity overlap
CREATE TABLE IF NOT EXISTS node_entity_overlap (
    id BIGSERIAL PRIMARY KEY,
    node_a_id TEXT NOT NULL,
    node_b_id TEXT NOT NULL,
    shared_entity_count INTEGER NOT NULL,
    shared_entity_ids TEXT,                   -- JSON array (top 20)
    shared_entity_types TEXT,                 -- JSON {type: count}
    taxonomic_distance INTEGER DEFAULT 0,     -- hops in taxonomy tree
    overlap_score DOUBLE PRECISION DEFAULT 0,             -- normalized overlap
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(node_a_id, node_b_id)
);

-- Signal harvester: convergent pattern matches
CREATE TABLE IF NOT EXISTS pattern_matches (
    id BIGSERIAL PRIMARY KEY,
    pattern_a_id INTEGER NOT NULL,
    pattern_b_id INTEGER NOT NULL,
    similarity_score DOUBLE PRECISION NOT NULL,
    node_a_id TEXT,
    node_b_id TEXT,
    shared_tokens TEXT,                       -- JSON array of overlapping keywords
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(pattern_a_id, pattern_b_id)
);

-- Signal harvester: contradiction clusters
CREATE TABLE IF NOT EXISTS contradiction_clusters (
    id BIGSERIAL PRIMARY KEY,
    theme TEXT NOT NULL,
    contradiction_ids TEXT NOT NULL,          -- JSON array
    shared_entities TEXT,                     -- JSON array of entity names
    cluster_size INTEGER DEFAULT 0,
    node_ids TEXT,                            -- JSON array of affected nodes
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Signal harvester: performance plateaus
CREATE TABLE IF NOT EXISTS performance_plateaus (
    id BIGSERIAL PRIMARY KEY,
    node_id TEXT NOT NULL,
    dataset_name TEXT NOT NULL,
    metric_name TEXT NOT NULL,
    top_methods TEXT NOT NULL,                -- JSON array of {method, value}
    spread DOUBLE PRECISION NOT NULL,                     -- max - min among top methods
    spread_pct DOUBLE PRECISION NOT NULL,                 -- spread as % of max
    method_count INTEGER DEFAULT 0,
    paper_count INTEGER DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(node_id, dataset_name, metric_name)
);

-- ===== SciForge: Experiment Validation Tables =====

-- Experiment runs launched from deep_insights
CREATE TABLE IF NOT EXISTS experiment_runs (
    id BIGSERIAL PRIMARY KEY,
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
    baseline_metric_value DOUBLE PRECISION,
    best_metric_value DOUBLE PRECISION,
    hypothesis_verdict TEXT,                   -- confirmed|refuted|inconclusive|NULL
    effect_size DOUBLE PRECISION,                          -- best - baseline
    effect_pct DOUBLE PRECISION,                           -- effect_size / baseline * 100
    resource_class TEXT DEFAULT 'cpu',         -- cpu|gpu_small|gpu_large
    submission_bundle_id INTEGER,
    error_message TEXT,
    started_at TIMESTAMP,
    completed_at TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Individual experiment iterations within a run
CREATE TABLE IF NOT EXISTS experiment_iterations (
    id BIGSERIAL PRIMARY KEY,
    run_id INTEGER NOT NULL REFERENCES experiment_runs(id),
    iteration_number INTEGER NOT NULL,
    phase TEXT NOT NULL,                       -- reproduction|hypothesis_testing
    code_diff TEXT,                            -- git diff of the change
    commit_hash TEXT,                          -- short git commit hash
    metric_value DOUBLE PRECISION,
    metric_name TEXT,
    peak_memory_mb DOUBLE PRECISION,
    duration_seconds DOUBLE PRECISION,
    status TEXT DEFAULT 'keep',                -- keep|discard|crash
    description TEXT,                          -- what the agent tried
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Knowledge graph feedback from experimental evidence
CREATE TABLE IF NOT EXISTS experimental_claims (
    id BIGSERIAL PRIMARY KEY,
    run_id INTEGER NOT NULL REFERENCES experiment_runs(id),
    deep_insight_id INTEGER NOT NULL REFERENCES deep_insights(id),
    claim_text TEXT NOT NULL,
    claim_type TEXT DEFAULT 'experimental',    -- experimental|reproduction|refutation
    verdict TEXT NOT NULL,                     -- confirmed|refuted|inconclusive
    effect_size DOUBLE PRECISION,
    confidence DOUBLE PRECISION,                           -- 0-1 statistical confidence
    p_value DOUBLE PRECISION,
    supporting_data TEXT,                      -- JSON with full experimental details
    source_paper_ids TEXT,
    source_node_ids TEXT,
    cascaded INTEGER DEFAULT 0,               -- 1 if cascade reasoning has processed this
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Meta-learning: track record of hypothesis quality by signal type
CREATE TABLE IF NOT EXISTS discovery_track_record (
    id BIGSERIAL PRIMARY KEY,
    signal_type TEXT NOT NULL UNIQUE,          -- entity_overlap|pattern_match|contradiction_cluster|plateau|insight_derived
    hypothesis_count INTEGER DEFAULT 0,
    confirmed_count INTEGER DEFAULT 0,
    refuted_count INTEGER DEFAULT 0,
    inconclusive_count INTEGER DEFAULT 0,
    avg_effect_size DOUBLE PRECISION DEFAULT 0,
    avg_adversarial_score DOUBLE PRECISION DEFAULT 0,
    hit_rate DOUBLE PRECISION DEFAULT 0,                   -- confirmed / (confirmed + refuted)
    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Auto Research queue/state machine for background closed-loop execution
CREATE TABLE IF NOT EXISTS auto_research_jobs (
    id BIGSERIAL PRIMARY KEY,
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
    id BIGSERIAL PRIMARY KEY,
    node_id TEXT REFERENCES taxonomy_nodes(id),
    theme TEXT NOT NULL,
    explanation_variants TEXT,              -- JSON array
    paper_ids TEXT,                         -- JSON array
    support_count INTEGER DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS protocol_artifacts (
    id BIGSERIAL PRIMARY KEY,
    node_id TEXT REFERENCES taxonomy_nodes(id),
    artifact_type TEXT NOT NULL,
    summary TEXT NOT NULL,
    paper_ids TEXT,                         -- JSON array
    support_count INTEGER DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS negative_space_gaps (
    id BIGSERIAL PRIMARY KEY,
    node_id TEXT REFERENCES taxonomy_nodes(id),
    gap_type TEXT NOT NULL,
    summary TEXT NOT NULL,
    paper_ids TEXT,                         -- JSON array
    support_count INTEGER DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS hidden_variable_bridges (
    id BIGSERIAL PRIMARY KEY,
    node_a_id TEXT NOT NULL REFERENCES taxonomy_nodes(id),
    node_b_id TEXT NOT NULL REFERENCES taxonomy_nodes(id),
    shared_factor TEXT NOT NULL,
    paper_ids TEXT,                         -- JSON array
    score DOUBLE PRECISION DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(node_a_id, node_b_id, shared_factor)
);

CREATE TABLE IF NOT EXISTS claim_method_gaps (
    id BIGSERIAL PRIMARY KEY,
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
    total_mem_gb DOUBLE PRECISION DEFAULT 0,
    status TEXT DEFAULT 'idle',             -- idle|busy|offline
    heartbeat_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    metadata TEXT
);

CREATE TABLE IF NOT EXISTS gpu_jobs (
    id BIGSERIAL PRIMARY KEY,
    deep_insight_id INTEGER REFERENCES deep_insights(id),
    experiment_run_id INTEGER REFERENCES experiment_runs(id),
    resource_class TEXT DEFAULT 'gpu_small',
    gpu_count INTEGER DEFAULT 1,
    vram_required_gb DOUBLE PRECISION DEFAULT 0,
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
    id BIGSERIAL PRIMARY KEY,
    run_id INTEGER NOT NULL REFERENCES experiment_runs(id),
    artifact_type TEXT NOT NULL,            -- log|metric|plot|bundle|source_data
    path TEXT NOT NULL,
    metric_key TEXT,
    metric_value DOUBLE PRECISION,
    metadata TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Manuscript and submission bundles
CREATE TABLE IF NOT EXISTS manuscript_runs (
    id BIGSERIAL PRIMARY KEY,
    experiment_run_id INTEGER REFERENCES experiment_runs(id),
    deep_insight_id INTEGER REFERENCES deep_insights(id),
    status TEXT DEFAULT 'drafting',         -- drafting|bundle_ready|failed
    canonical_state TEXT,
    workdir TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS manuscript_assets (
    id BIGSERIAL PRIMARY KEY,
    manuscript_run_id INTEGER NOT NULL REFERENCES manuscript_runs(id),
    asset_type TEXT NOT NULL,               -- tex|figure|bib|cover_letter|metadata
    label TEXT,
    path TEXT NOT NULL,
    metadata TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS submission_bundles (
    id BIGSERIAL PRIMARY KEY,
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
-- Insight feedback loop: provenance, outcomes, event log, signal harvester telemetry

CREATE TABLE IF NOT EXISTS insight_events (
    id BIGSERIAL PRIMARY KEY,
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
    id BIGSERIAL PRIMARY KEY,
    pattern_name TEXT NOT NULL,
    query_hash TEXT,
    node_id TEXT,
    candidate_count INTEGER DEFAULT 0,
    execution_time_ms INTEGER,
    meta_json TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_harvester_runs_pattern ON harvester_runs(pattern_name);

-- Checkpoint + arXiv dedup (papers)
ALTER TABLE papers ADD COLUMN IF NOT EXISTS processing_stage TEXT DEFAULT 'ingested';
ALTER TABLE papers ADD COLUMN IF NOT EXISTS arxiv_base_id TEXT;
CREATE INDEX IF NOT EXISTS idx_papers_arxiv_base ON papers(arxiv_base_id);
ALTER TABLE papers ADD COLUMN IF NOT EXISTS processing_attempts INTEGER DEFAULT 0;
ALTER TABLE papers ADD COLUMN IF NOT EXISTS stage_last_error TEXT;
ALTER TABLE papers ADD COLUMN IF NOT EXISTS stage_locked_by TEXT;
ALTER TABLE papers ADD COLUMN IF NOT EXISTS stage_started_at TIMESTAMP;
ALTER TABLE papers ADD COLUMN IF NOT EXISTS stage_completed_at TIMESTAMP;
ALTER TABLE claims ADD COLUMN IF NOT EXISTS source_paper_ids TEXT;
ALTER TABLE claims ADD COLUMN IF NOT EXISTS source_node_ids TEXT;
ALTER TABLE claims ADD COLUMN IF NOT EXISTS claim_key TEXT;
ALTER TABLE claims ADD COLUMN IF NOT EXISTS embedding_vector VECTOR(1536);
ALTER TABLE results ADD COLUMN IF NOT EXISTS result_key TEXT;
ALTER TABLE experimental_claims ADD COLUMN IF NOT EXISTS source_paper_ids TEXT;
ALTER TABLE experimental_claims ADD COLUMN IF NOT EXISTS source_node_ids TEXT;

CREATE TABLE IF NOT EXISTS pipeline_events (
    id BIGSERIAL PRIMARY KEY,
    event_type TEXT NOT NULL,
    entity_type TEXT,
    entity_id TEXT,
    dedupe_key TEXT,
    payload TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS pipeline_event_consumers (
    consumer_name TEXT PRIMARY KEY,
    last_event_id BIGINT DEFAULT 0,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS paper_stage_checkpoints (
    paper_id TEXT NOT NULL REFERENCES papers(id),
    stage TEXT NOT NULL,
    payload TEXT,
    error_message TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (paper_id, stage)
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_pipeline_events_dedupe ON pipeline_events(dedupe_key);
CREATE INDEX IF NOT EXISTS idx_pipeline_events_type_id ON pipeline_events(event_type, id);
CREATE INDEX IF NOT EXISTS idx_paper_stage_checkpoints_stage ON paper_stage_checkpoints(stage);
CREATE UNIQUE INDEX IF NOT EXISTS idx_results_result_key ON results(result_key);
