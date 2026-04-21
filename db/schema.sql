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
    processing_stage TEXT DEFAULT 'ingested',  -- pipeline checkpoint: text_ready|extracted|graph_stored|reasoned
    arxiv_base_id TEXT,                -- id without vN suffix (dedup)
    processing_attempts INTEGER DEFAULT 0,
    stage_last_error TEXT,
    stage_locked_by TEXT,
    stage_started_at TIMESTAMP,
    stage_completed_at TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS claims (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    paper_id TEXT NOT NULL REFERENCES papers(id),
    claim_text TEXT NOT NULL,
    claim_type TEXT,                   -- performance|method|finding|limitation
    method_name TEXT,
    dataset_name TEXT,
    metric_name TEXT,
    metric_value REAL,
    evidence_location TEXT,            -- "Table 2, row 3"
    conditions TEXT,                   -- JSON: experimental conditions
    embedding TEXT,                    -- JSON: vector for similarity search
    source_quote TEXT,                 -- verbatim grounding snippet from full_text
    char_start INTEGER,                -- span in the text region named by grounding_source_field
    char_end INTEGER,
    grounding_status TEXT,             -- verified|weak|unverified|no_quote
    grounding_score REAL,
    grounding_source_field TEXT,       -- full_text|appendix_text
    source_paper_ids TEXT,             -- JSON array of source paper ids
    source_node_ids TEXT,              -- JSON array of source node ids
    claim_key TEXT,                    -- stable upsert/dedup key
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS methods (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL UNIQUE,
    category TEXT,
    description TEXT,
    key_innovation TEXT,
    first_paper_id TEXT REFERENCES papers(id),
    builds_on TEXT,                    -- JSON array of method names
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS patterns (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
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
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    claim_a_id INTEGER REFERENCES claims(id),
    claim_b_id INTEGER REFERENCES claims(id),
    description TEXT NOT NULL,
    condition_diff TEXT,               -- what differs between the two claims
    hypothesis TEXT,                   -- generated hypothesis to resolve
    severity TEXT DEFAULT 'medium',    -- low|medium|high
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS gaps (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    problem_pattern_id INTEGER REFERENCES patterns(id),
    solution_pattern_id INTEGER REFERENCES patterns(id),
    gap_description TEXT NOT NULL,
    missing_domain TEXT,               -- which domain lacks this solution
    evidence_papers TEXT,              -- JSON array of paper IDs supporting the gap
    research_proposal TEXT,            -- auto-generated proposal
    value_score REAL,                  -- 0-5
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS stats (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
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
    confidence REAL DEFAULT 1.0,          -- 0-1 confidence of classification
    PRIMARY KEY (paper_id, node_id)
);

CREATE TABLE IF NOT EXISTS results (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    paper_id TEXT NOT NULL REFERENCES papers(id),
    node_id TEXT REFERENCES taxonomy_nodes(id),
    method_name TEXT NOT NULL,
    dataset_name TEXT NOT NULL,
    metric_name TEXT NOT NULL,
    metric_value REAL,
    metric_unit TEXT,                      -- e.g. "%", "ms", "FLOPs"
    is_sota INTEGER DEFAULT 0,            -- 1 if claimed SOTA
    evidence_location TEXT,
    conditions TEXT,                       -- JSON
    source_quote TEXT,
    char_start INTEGER,
    char_end INTEGER,
    grounding_status TEXT,
    grounding_score REAL,
    grounding_source_field TEXT,           -- full_text|appendix_text
    result_key TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS result_taxonomy (
    result_id INTEGER NOT NULL REFERENCES results(id),
    node_id TEXT NOT NULL REFERENCES taxonomy_nodes(id),
    PRIMARY KEY (result_id, node_id)
);

CREATE TABLE IF NOT EXISTS matrix_gaps (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    node_id TEXT NOT NULL REFERENCES taxonomy_nodes(id),
    method_name TEXT NOT NULL,
    dataset_name TEXT NOT NULL,
    metric_name TEXT,
    gap_description TEXT NOT NULL,
    research_proposal TEXT,
    value_score REAL,                      -- 0-5
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
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    node_id TEXT NOT NULL REFERENCES taxonomy_nodes(id),
    insight_type TEXT NOT NULL,
    title TEXT NOT NULL,
    hypothesis TEXT NOT NULL,
    evidence TEXT,
    experiment TEXT,
    impact TEXT,
    novelty_score REAL DEFAULT 0,
    feasibility_score REAL DEFAULT 0,
    supporting_papers TEXT,
    generation_run_id TEXT,
    source_signal_ids TEXT,
    source_paper_ids TEXT,
    source_node_ids TEXT,
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
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    paper_id TEXT NOT NULL REFERENCES papers(id),
    node_id TEXT REFERENCES taxonomy_nodes(id),
    entity_id TEXT NOT NULL REFERENCES graph_entities(id),
    mention_text TEXT,
    mention_role TEXT,
    confidence REAL DEFAULT 1.0,
    evidence_location TEXT,
    source_text TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS graph_relations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    paper_id TEXT NOT NULL REFERENCES papers(id),
    node_id TEXT REFERENCES taxonomy_nodes(id),
    subject_entity_id TEXT NOT NULL REFERENCES graph_entities(id),
    predicate TEXT NOT NULL,
    object_entity_id TEXT NOT NULL REFERENCES graph_entities(id),
    confidence REAL DEFAULT 1.0,
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
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    node_id TEXT NOT NULL REFERENCES taxonomy_nodes(id),
    opportunity_type TEXT NOT NULL,
    title TEXT NOT NULL,
    description TEXT NOT NULL,
    why_now TEXT,
    value_score REAL DEFAULT 0,
    confidence REAL DEFAULT 0,
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
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    entity_type TEXT NOT NULL,
    primary_entity_id TEXT NOT NULL REFERENCES graph_entities(id),
    candidate_entity_id TEXT NOT NULL REFERENCES graph_entities(id),
    similarity_score REAL NOT NULL,
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
CREATE INDEX IF NOT EXISTS idx_claims_paper ON claims(paper_id);
CREATE INDEX IF NOT EXISTS idx_claims_type ON claims(claim_type);
CREATE INDEX IF NOT EXISTS idx_claims_method ON claims(method_name);
CREATE UNIQUE INDEX IF NOT EXISTS idx_claims_claim_key ON claims(claim_key);
CREATE INDEX IF NOT EXISTS idx_papers_arxiv_base ON papers(arxiv_base_id);
CREATE INDEX IF NOT EXISTS idx_papers_processing_stage ON papers(processing_stage);
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

CREATE TABLE IF NOT EXISTS pipeline_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    event_type TEXT NOT NULL,
    entity_type TEXT,
    entity_id TEXT,
    dedupe_key TEXT,
    payload TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS pipeline_event_consumers (
    consumer_name TEXT PRIMARY KEY,
    last_event_id INTEGER DEFAULT 0,
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
