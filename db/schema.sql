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
    status TEXT DEFAULT 'ingested',    -- ingested|extracted|abstracted|reasoned|error
    error_msg TEXT,
    token_cost INTEGER DEFAULT 0,
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
    domain_count INTEGER DEFAULT 1,
    domains TEXT,                      -- JSON array: ["education", "aviation", ...]
    claim_ids TEXT,                    -- JSON array of claim IDs
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

-- Indexes
CREATE INDEX IF NOT EXISTS idx_papers_status ON papers(status);
CREATE INDEX IF NOT EXISTS idx_claims_paper ON claims(paper_id);
CREATE INDEX IF NOT EXISTS idx_claims_type ON claims(claim_type);
CREATE INDEX IF NOT EXISTS idx_claims_method ON claims(method_name);
CREATE INDEX IF NOT EXISTS idx_patterns_type ON patterns(pattern_type);
CREATE INDEX IF NOT EXISTS idx_taxonomy_parent ON taxonomy_nodes(parent_id);
CREATE INDEX IF NOT EXISTS idx_paper_taxonomy_node ON paper_taxonomy(node_id);
CREATE INDEX IF NOT EXISTS idx_paper_taxonomy_paper ON paper_taxonomy(paper_id);
CREATE INDEX IF NOT EXISTS idx_results_paper ON results(paper_id);
CREATE INDEX IF NOT EXISTS idx_results_node ON results(node_id);
CREATE INDEX IF NOT EXISTS idx_result_taxonomy_node ON result_taxonomy(node_id);
CREATE INDEX IF NOT EXISTS idx_result_taxonomy_result ON result_taxonomy(result_id);
CREATE INDEX IF NOT EXISTS idx_results_method ON results(method_name);
CREATE INDEX IF NOT EXISTS idx_results_dataset ON results(dataset_name);
CREATE INDEX IF NOT EXISTS idx_matrix_gaps_node ON matrix_gaps(node_id);
CREATE INDEX IF NOT EXISTS idx_paper_insights_work_type ON paper_insights(work_type);
