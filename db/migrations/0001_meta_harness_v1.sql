-- meta-harness-v1 additive PostgreSQL migration.
-- DO NOT run on production directly. Validate twice against an isolated restore.
-- No destructive table operation, destructive type change, or implicit backlog
-- assignment.

CREATE TABLE IF NOT EXISTS deepgraph_schema_migrations (
    migration_key TEXT PRIMARY KEY,
    source_commit TEXT NOT NULL,
    checksum_sha256 TEXT NOT NULL,
    applied_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);
ALTER TABLE deepgraph_schema_migrations
    ADD COLUMN IF NOT EXISTS checksum_sha256 TEXT;

CREATE TABLE IF NOT EXISTS research_agendas (
    id BIGSERIAL PRIMARY KEY,
    version TEXT NOT NULL DEFAULT 'v1',
    name TEXT NOT NULL,
    description TEXT,
    focus_json TEXT NOT NULL DEFAULT '[]',
    prefer_json TEXT NOT NULL DEFAULT '{}',
    reject_json TEXT NOT NULL DEFAULT '{}',
    required_output_json TEXT NOT NULL DEFAULT '{}',
    raw_config_json TEXT NOT NULL DEFAULT '{}',
    is_active INTEGER NOT NULL DEFAULT 1,
    submitter TEXT,
    token_budget BIGINT,
    token_spent BIGINT NOT NULL DEFAULT 0,
    status TEXT NOT NULL DEFAULT 'paused_manual',
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);

-- Older physical backups may contain this table without a primary/unique key.
-- Add a compatible unique index before any new foreign key references it.
CREATE UNIQUE INDEX IF NOT EXISTS idx_research_agendas_id_unique
    ON research_agendas(id);

-- The legacy deep_insights table may likewise lack a declared primary key.
CREATE UNIQUE INDEX IF NOT EXISTS idx_deep_insights_id_unique
    ON deep_insights(id);

-- Existing production agendas may have NULL/zero budgets. Do not activate them
-- until an operator assigns a positive hard cap.
ALTER TABLE research_agendas ADD COLUMN IF NOT EXISTS token_reserved BIGINT NOT NULL DEFAULT 0;
ALTER TABLE research_agendas ADD COLUMN IF NOT EXISTS gpu_hours_budget DOUBLE PRECISION NOT NULL DEFAULT 0;
ALTER TABLE research_agendas ADD COLUMN IF NOT EXISTS gpu_hours_spent DOUBLE PRECISION NOT NULL DEFAULT 0;
ALTER TABLE research_agendas ADD COLUMN IF NOT EXISTS gpu_hours_reserved DOUBLE PRECISION NOT NULL DEFAULT 0;
ALTER TABLE research_agendas ADD COLUMN IF NOT EXISTS max_concurrency INTEGER NOT NULL DEFAULT 1;
ALTER TABLE research_agendas ADD COLUMN IF NOT EXISTS backend_allowlist_json TEXT NOT NULL DEFAULT '["cpu","llm"]';
ALTER TABLE research_agendas ADD COLUMN IF NOT EXISTS backlog_policy TEXT NOT NULL DEFAULT 'explicit_import_only';

CREATE INDEX IF NOT EXISTS idx_research_agendas_active
    ON research_agendas(is_active, status, updated_at, id);

CREATE TABLE IF NOT EXISTS agenda_selections (
    id BIGSERIAL PRIMARY KEY,
    agenda_id BIGINT NOT NULL REFERENCES research_agendas(id),
    selected_insight_id BIGINT,
    score DOUBLE PRECISION,
    rationale TEXT,
    rejected_candidates_json TEXT NOT NULL DEFAULT '[]',
    scoring_breakdown_json TEXT NOT NULL DEFAULT '{}',
    status TEXT NOT NULL DEFAULT 'pending',
    auto_research_job_id BIGINT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_agenda_selections_scope
    ON agenda_selections(agenda_id, status, created_at, id);

CREATE TABLE IF NOT EXISTS agenda_resource_ledger (
    id BIGSERIAL PRIMARY KEY,
    agenda_id BIGINT NOT NULL REFERENCES research_agendas(id),
    operation TEXT NOT NULL,
    idempotency_key TEXT NOT NULL,
    token_reserved BIGINT NOT NULL DEFAULT 0 CHECK (token_reserved >= 0),
    gpu_hours_reserved DOUBLE PRECISION NOT NULL DEFAULT 0 CHECK (gpu_hours_reserved >= 0),
    tokens_used BIGINT CHECK (tokens_used >= 0),
    gpu_hours_used DOUBLE PRECISION CHECK (gpu_hours_used >= 0),
    cost_usd DOUBLE PRECISION CHECK (cost_usd IS NULL OR cost_usd >= 0),
    status TEXT NOT NULL DEFAULT 'reserved'
        CHECK (status IN ('reserved', 'settled', 'released')),
    release_reason TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    settled_at TIMESTAMPTZ,
    UNIQUE (agenda_id, idempotency_key),
    CHECK (token_reserved > 0 OR gpu_hours_reserved > 0),
    CHECK (tokens_used IS NULL OR tokens_used <= token_reserved),
    CHECK (gpu_hours_used IS NULL OR gpu_hours_used <= gpu_hours_reserved)
);

CREATE INDEX IF NOT EXISTS idx_agenda_resource_ledger_scope
    ON agenda_resource_ledger(agenda_id, status, created_at, id);

-- Compatibility ledger retained from production Agenda. New code writes actual
-- settled LLM usage here only after reservation settlement.
CREATE TABLE IF NOT EXISTS agenda_token_ledger (
    id BIGSERIAL PRIMARY KEY,
    agenda_id BIGINT NOT NULL REFERENCES research_agendas(id),
    operation TEXT NOT NULL,
    tokens BIGINT NOT NULL DEFAULT 0,
    cost_usd DOUBLE PRECISION,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);
ALTER TABLE agenda_token_ledger ADD COLUMN IF NOT EXISTS provider TEXT;
ALTER TABLE agenda_token_ledger ADD COLUMN IF NOT EXISTS model TEXT;
ALTER TABLE agenda_token_ledger ADD COLUMN IF NOT EXISTS prompt_version TEXT;
ALTER TABLE agenda_token_ledger ADD COLUMN IF NOT EXISTS route_observation_id BIGINT;
CREATE INDEX IF NOT EXISTS idx_agenda_token_ledger_scope
    ON agenda_token_ledger(agenda_id, created_at, id);

CREATE TABLE IF NOT EXISTS legacy_scope_imports (
    id BIGSERIAL PRIMARY KEY,
    agenda_id BIGINT NOT NULL REFERENCES research_agendas(id),
    entity_type TEXT NOT NULL,
    entity_id BIGINT NOT NULL,
    actor TEXT NOT NULL,
    reason TEXT NOT NULL,
    idempotency_key TEXT NOT NULL,
    imported_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE (agenda_id, idempotency_key),
    UNIQUE (entity_type, entity_id)
);

CREATE TABLE IF NOT EXISTS agenda_signal_outcomes (
    id BIGSERIAL PRIMARY KEY,
    agenda_id BIGINT NOT NULL REFERENCES research_agendas(id),
    run_id BIGINT,
    experimental_claim_id BIGINT,
    signal_table TEXT NOT NULL,
    signal_content_hash TEXT NOT NULL,
    verdict TEXT NOT NULL
        CHECK (verdict IN ('supported', 'refuted', 'inconclusive')),
    effect_size DOUBLE PRECISION,
    p_value DOUBLE PRECISION,
    conditions_json TEXT NOT NULL DEFAULT '{}',
    idempotency_key TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE (agenda_id, idempotency_key)
);
CREATE INDEX IF NOT EXISTS idx_agenda_signal_outcome_scope
    ON agenda_signal_outcomes(
        agenda_id, signal_table, signal_content_hash, created_at, id
    );

-- These GitHub-line tables are absent from the production snapshot. Creating
-- them here lets an isolated production restore receive the problem-first and
-- benchmark contracts without running application startup as a migration.
CREATE TABLE IF NOT EXISTS research_problems (
    id BIGSERIAL PRIMARY KEY,
    agenda_id BIGINT REFERENCES research_agendas(id),
    problem_statement TEXT,
    source_signal_ref TEXT,
    node_ids TEXT,
    paper_ids TEXT,
    problem_quality_score DOUBLE PRECISION,
    status TEXT DEFAULT 'open',
    attempts_count INTEGER DEFAULT 0,
    ruled_out_approaches TEXT DEFAULT '[]',
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS experimental_evidence_edges (
    id BIGSERIAL PRIMARY KEY,
    agenda_id BIGINT REFERENCES research_agendas(id),
    experimental_claim_id BIGINT,
    run_id BIGINT,
    deep_insight_id BIGINT,
    research_problem_id BIGINT,
    empirical_entity_id TEXT,
    target_kind TEXT,
    target_id TEXT,
    relation TEXT,
    verdict TEXT,
    effect_size DOUBLE PRECISION,
    conditions TEXT,
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS benchmark_harness_jobs (
    id BIGSERIAL PRIMARY KEY,
    agenda_id BIGINT REFERENCES research_agendas(id),
    deep_insight_id BIGINT NOT NULL UNIQUE REFERENCES deep_insights(id),
    status TEXT DEFAULT 'harness_required',
    harness_kind TEXT DEFAULT 'custom_benchmark_harness',
    benchmark_name TEXT,
    dataset_refs TEXT,
    baseline_refs TEXT,
    required_capabilities TEXT,
    task_plan TEXT,
    artifact_uri TEXT,
    last_error TEXT,
    last_note TEXT,
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);

-- Add nullable scope to legacy rows. Existing backlog remains NULL/excluded.
ALTER TABLE IF EXISTS research_problems ADD COLUMN IF NOT EXISTS agenda_id BIGINT REFERENCES research_agendas(id);
ALTER TABLE IF EXISTS experimental_evidence_edges ADD COLUMN IF NOT EXISTS agenda_id BIGINT REFERENCES research_agendas(id);
ALTER TABLE IF EXISTS deep_insights ADD COLUMN IF NOT EXISTS agenda_id BIGINT REFERENCES research_agendas(id);
ALTER TABLE IF EXISTS deep_insights ADD COLUMN IF NOT EXISTS research_problem_id BIGINT;
ALTER TABLE IF EXISTS auto_research_jobs ADD COLUMN IF NOT EXISTS agenda_id BIGINT REFERENCES research_agendas(id);
ALTER TABLE IF EXISTS experiment_runs ADD COLUMN IF NOT EXISTS agenda_id BIGINT REFERENCES research_agendas(id);
ALTER TABLE IF EXISTS experiment_iterations ADD COLUMN IF NOT EXISTS agenda_id BIGINT REFERENCES research_agendas(id);
ALTER TABLE IF EXISTS experimental_claims ADD COLUMN IF NOT EXISTS agenda_id BIGINT REFERENCES research_agendas(id);
ALTER TABLE IF EXISTS experiment_artifacts ADD COLUMN IF NOT EXISTS agenda_id BIGINT REFERENCES research_agendas(id);
ALTER TABLE IF EXISTS gpu_jobs ADD COLUMN IF NOT EXISTS agenda_id BIGINT REFERENCES research_agendas(id);
ALTER TABLE IF EXISTS manuscript_runs ADD COLUMN IF NOT EXISTS agenda_id BIGINT REFERENCES research_agendas(id);
ALTER TABLE IF EXISTS manuscript_assets ADD COLUMN IF NOT EXISTS agenda_id BIGINT REFERENCES research_agendas(id);
ALTER TABLE IF EXISTS submission_bundles ADD COLUMN IF NOT EXISTS agenda_id BIGINT REFERENCES research_agendas(id);
ALTER TABLE IF EXISTS benchmark_harness_jobs ADD COLUMN IF NOT EXISTS agenda_id BIGINT REFERENCES research_agendas(id);

-- Preserve foreign-key compatibility with legacy experiment-run tables.
CREATE UNIQUE INDEX IF NOT EXISTS idx_experiment_runs_id_unique
    ON experiment_runs(id);

-- The application schema creates indexes on these legacy signal tables during
-- startup. Physical backups from older releases may lack the hash column;
-- add it before startup schema/index reconciliation runs.
ALTER TABLE IF EXISTS node_entity_overlap ADD COLUMN IF NOT EXISTS content_hash TEXT;
ALTER TABLE IF EXISTS pattern_matches ADD COLUMN IF NOT EXISTS content_hash TEXT;
ALTER TABLE IF EXISTS contradiction_clusters ADD COLUMN IF NOT EXISTS content_hash TEXT;
ALTER TABLE IF EXISTS performance_plateaus ADD COLUMN IF NOT EXISTS content_hash TEXT;
ALTER TABLE IF EXISTS mechanism_mismatches ADD COLUMN IF NOT EXISTS content_hash TEXT;
ALTER TABLE IF EXISTS protocol_artifacts ADD COLUMN IF NOT EXISTS content_hash TEXT;
ALTER TABLE IF EXISTS negative_space_gaps ADD COLUMN IF NOT EXISTS content_hash TEXT;
ALTER TABLE IF EXISTS hidden_variable_bridges ADD COLUMN IF NOT EXISTS content_hash TEXT;
ALTER TABLE IF EXISTS claim_method_gaps ADD COLUMN IF NOT EXISTS content_hash TEXT;

CREATE INDEX IF NOT EXISTS idx_research_problems_agenda ON research_problems(agenda_id, status, id);
CREATE INDEX IF NOT EXISTS idx_experimental_evidence_edges_agenda
    ON experimental_evidence_edges(agenda_id, run_id, id);
CREATE INDEX IF NOT EXISTS idx_deep_insights_agenda ON deep_insights(agenda_id, status, id);
CREATE UNIQUE INDEX IF NOT EXISTS idx_deep_insights_pending_proposal
    ON deep_insights(agenda_id, research_problem_id)
    WHERE research_problem_id IS NOT NULL AND status='proposal_pending';
CREATE INDEX IF NOT EXISTS idx_auto_research_jobs_agenda ON auto_research_jobs(agenda_id, status, id);
CREATE INDEX IF NOT EXISTS idx_experiment_runs_agenda ON experiment_runs(agenda_id, status, id);
CREATE INDEX IF NOT EXISTS idx_experiment_iterations_agenda ON experiment_iterations(agenda_id, run_id, id);
CREATE INDEX IF NOT EXISTS idx_experimental_claims_agenda ON experimental_claims(agenda_id, run_id, id);
CREATE INDEX IF NOT EXISTS idx_experiment_artifacts_agenda ON experiment_artifacts(agenda_id, run_id, id);
CREATE INDEX IF NOT EXISTS idx_gpu_jobs_agenda ON gpu_jobs(agenda_id, status, id);

CREATE TABLE IF NOT EXISTS frontier_packets (
    id BIGSERIAL PRIMARY KEY,
    agenda_id BIGINT NOT NULL REFERENCES research_agendas(id),
    research_problem_id BIGINT,
    retrieved_at TIMESTAMPTZ NOT NULL,
    coverage_json TEXT NOT NULL,
    problem_status TEXT NOT NULL,
    strongest_recent_work_json TEXT NOT NULL DEFAULT '[]',
    latest_benchmarks_json TEXT NOT NULL DEFAULT '[]',
    nearest_prior_art_json TEXT NOT NULL DEFAULT '[]',
    contribution_delta_json TEXT NOT NULL DEFAULT '{}',
    obsolete_evidence_json TEXT NOT NULL DEFAULT '[]',
    counterevidence_json TEXT NOT NULL DEFAULT '[]',
    why_not_obsolete TEXT NOT NULL,
    minimum_falsification_experiment_json TEXT NOT NULL,
    evaluator_route_observation_id BIGINT,
    gate_allowed INTEGER NOT NULL DEFAULT 0,
    gate_reason_codes_json TEXT NOT NULL DEFAULT '[]',
    content_hash TEXT NOT NULL UNIQUE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS idea_decision_packets (
    id BIGSERIAL PRIMARY KEY,
    agenda_id BIGINT NOT NULL REFERENCES research_agendas(id),
    idea_id BIGINT NOT NULL,
    frontier_packet_id BIGINT NOT NULL REFERENCES frontier_packets(id),
    decision TEXT NOT NULL CHECK (decision IN ('promote', 'kill', 'park', 'revisit')),
    estimates_json TEXT NOT NULL,
    candidate_family TEXT NOT NULL,
    correlation_keys_json TEXT NOT NULL DEFAULT '[]',
    reason_codes_json TEXT NOT NULL DEFAULT '[]',
    revisit_condition_json TEXT,
    revisit_after TIMESTAMPTZ,
    policy_version TEXT NOT NULL,
    decided_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CHECK (decision <> 'park' OR revisit_condition_json IS NOT NULL OR revisit_after IS NOT NULL)
);

CREATE INDEX IF NOT EXISTS idx_idea_decisions_scope
    ON idea_decision_packets(agenda_id, decision, decided_at, id);

CREATE TABLE IF NOT EXISTS resource_grants (
    id BIGSERIAL PRIMARY KEY,
    agenda_id BIGINT NOT NULL REFERENCES research_agendas(id),
    idea_id BIGINT NOT NULL,
    decision_packet_id BIGINT NOT NULL REFERENCES idea_decision_packets(id),
    stage TEXT NOT NULL,
    token_cap BIGINT NOT NULL CHECK (token_cap >= 0),
    gpu_class TEXT,
    max_gpu_hours DOUBLE PRECISION NOT NULL DEFAULT 0 CHECK (max_gpu_hours >= 0),
    backend_allowlist_json TEXT NOT NULL,
    artifact_requirements_json TEXT NOT NULL,
    expires_at TIMESTAMPTZ NOT NULL,
    grant_reason TEXT NOT NULL,
    reservation_id BIGINT NOT NULL REFERENCES agenda_resource_ledger(id),
    status TEXT NOT NULL DEFAULT 'active'
        CHECK (status IN ('active', 'consumed', 'expired', 'revoked')),
    idempotency_key TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE (agenda_id, idempotency_key)
);

CREATE INDEX IF NOT EXISTS idx_resource_grants_admission
    ON resource_grants(agenda_id, idea_id, status, expires_at, id);

CREATE TABLE IF NOT EXISTS resource_grant_usage_reservations (
    id BIGSERIAL PRIMARY KEY,
    agenda_id BIGINT NOT NULL REFERENCES research_agendas(id),
    resource_grant_id BIGINT NOT NULL REFERENCES resource_grants(id),
    operation TEXT NOT NULL,
    idempotency_key TEXT NOT NULL,
    token_reserved BIGINT NOT NULL CHECK (token_reserved > 0),
    tokens_used BIGINT CHECK (tokens_used >= 0 AND tokens_used <= token_reserved),
    cost_usd DOUBLE PRECISION CHECK (cost_usd IS NULL OR cost_usd >= 0),
    status TEXT NOT NULL DEFAULT 'reserved'
        CHECK (status IN ('reserved', 'settled', 'released')),
    release_reason TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    settled_at TIMESTAMPTZ,
    UNIQUE (resource_grant_id, idempotency_key)
);
CREATE INDEX IF NOT EXISTS idx_grant_usage_scope
    ON resource_grant_usage_reservations(
        agenda_id, resource_grant_id, status, created_at, id
    );

ALTER TABLE IF EXISTS auto_research_jobs
    ADD COLUMN IF NOT EXISTS resource_grant_id BIGINT REFERENCES resource_grants(id);
ALTER TABLE IF EXISTS experiment_runs
    ADD COLUMN IF NOT EXISTS resource_grant_id BIGINT REFERENCES resource_grants(id);
ALTER TABLE IF EXISTS gpu_jobs
    ADD COLUMN IF NOT EXISTS resource_grant_id BIGINT REFERENCES resource_grants(id);
ALTER TABLE IF EXISTS gpu_jobs
    ADD COLUMN IF NOT EXISTS meta_harness_idempotency_key TEXT;
ALTER TABLE IF EXISTS benchmark_harness_jobs
    ADD COLUMN IF NOT EXISTS resource_grant_id BIGINT REFERENCES resource_grants(id);
CREATE UNIQUE INDEX IF NOT EXISTS idx_gpu_jobs_meta_harness_identity
    ON gpu_jobs(agenda_id, meta_harness_idempotency_key)
    WHERE meta_harness_idempotency_key IS NOT NULL;

CREATE TABLE IF NOT EXISTS outcome_records (
    id BIGSERIAL PRIMARY KEY,
    agenda_id BIGINT NOT NULL REFERENCES research_agendas(id),
    idea_id BIGINT NOT NULL,
    resource_grant_id BIGINT NOT NULL REFERENCES resource_grants(id),
    experiment_run_id BIGINT REFERENCES experiment_runs(id),
    actual_tokens BIGINT NOT NULL DEFAULT 0 CHECK (actual_tokens >= 0),
    actual_gpu_hours DOUBLE PRECISION NOT NULL DEFAULT 0 CHECK (actual_gpu_hours >= 0),
    wall_seconds DOUBLE PRECISION NOT NULL DEFAULT 0 CHECK (wall_seconds >= 0),
    execution_result TEXT NOT NULL,
    effect DOUBLE PRECISION,
    baseline DOUBLE PRECISION,
    verdict TEXT NOT NULL CHECK (verdict IN ('supported', 'refuted', 'inconclusive', 'invalid')),
    new_information_json TEXT NOT NULL DEFAULT '{}',
    state_decision TEXT NOT NULL,
    prediction_error_json TEXT NOT NULL DEFAULT '{}',
    artifact_manifest_json TEXT NOT NULL,
    recorded_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE (resource_grant_id)
);

CREATE INDEX IF NOT EXISTS idx_outcome_records_calibration
    ON outcome_records(agenda_id, recorded_at, id);

CREATE TABLE IF NOT EXISTS llm_route_observations (
    id BIGSERIAL PRIMARY KEY,
    agenda_id BIGINT NOT NULL REFERENCES research_agendas(id),
    idea_id BIGINT NOT NULL,
    role TEXT NOT NULL CHECK (role IN ('proposer', 'evaluator', 'reviewer')),
    provider TEXT NOT NULL,
    model TEXT NOT NULL,
    model_family TEXT NOT NULL,
    prompt_version TEXT NOT NULL,
    input_tokens BIGINT NOT NULL DEFAULT 0,
    output_tokens BIGINT NOT NULL DEFAULT 0,
    cost_usd DOUBLE PRECISION,
    status TEXT NOT NULL,
    failure_reason TEXT,
    grant_usage_reservation_id BIGINT
        REFERENCES resource_grant_usage_reservations(id),
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS llm_provider_cooldowns (
    route_id TEXT PRIMARY KEY,
    provider TEXT NOT NULL,
    model TEXT NOT NULL,
    failure_category TEXT NOT NULL
        CHECK (failure_category IN ('auth', 'transient', 'provider_error')),
    cooldown_until TIMESTAMPTZ NOT NULL,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_llm_provider_cooldowns_active
    ON llm_provider_cooldowns(cooldown_until, route_id);

CREATE TABLE IF NOT EXISTS compute_jobs_v1 (
    id BIGSERIAL PRIMARY KEY,
    agenda_id BIGINT NOT NULL REFERENCES research_agendas(id),
    idea_id BIGINT NOT NULL,
    resource_grant_id BIGINT NOT NULL REFERENCES resource_grants(id),
    stage TEXT NOT NULL,
    backend_kind TEXT NOT NULL CHECK (backend_kind IN ('cpu', 'local_gpu', 'ssh_gpu', 'colab_gpu')),
    backend_job_id TEXT,
    backend_account_ref TEXT,
    idempotency_key TEXT NOT NULL,
    command_ref TEXT NOT NULL,
    artifact_namespace TEXT NOT NULL,
    requested_gpu_hours DOUBLE PRECISION NOT NULL DEFAULT 0
        CHECK (requested_gpu_hours >= 0),
    timeout_seconds INTEGER NOT NULL CHECK (timeout_seconds > 0),
    status TEXT NOT NULL CHECK (
        status IN (
            'submitting', 'submission_unknown', 'submitted', 'running',
            'cancel_requested', 'collecting', 'succeeded', 'failed',
            'cancelled', 'timed_out', 'usage_unknown'
        )
    ),
    heartbeat_at TIMESTAMPTZ,
    timeout_at TIMESTAMPTZ NOT NULL,
    artifact_manifest_json TEXT,
    usage_json TEXT,
    failure_reason TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE (agenda_id, idempotency_key),
    UNIQUE (backend_kind, backend_job_id)
);
CREATE INDEX IF NOT EXISTS idx_compute_jobs_v1_active
    ON compute_jobs_v1(agenda_id, status, timeout_at, id);

CREATE TABLE IF NOT EXISTS colab_work_requests_v1 (
    id BIGSERIAL PRIMARY KEY,
    agenda_id BIGINT NOT NULL REFERENCES research_agendas(id),
    idea_id BIGINT NOT NULL,
    experiment_run_id BIGINT NOT NULL REFERENCES experiment_runs(id),
    resource_grant_id BIGINT NOT NULL REFERENCES resource_grants(id),
    compute_job_id BIGINT REFERENCES compute_jobs_v1(id),
    stage TEXT NOT NULL,
    idempotency_key TEXT NOT NULL,
    code_dir TEXT NOT NULL,
    command_tokens_json TEXT NOT NULL,
    environment_json TEXT NOT NULL DEFAULT '{}',
    artifact_map_json TEXT NOT NULL,
    artifact_output_dir TEXT NOT NULL,
    timeout_seconds INTEGER NOT NULL CHECK (timeout_seconds > 0),
    status TEXT NOT NULL CHECK (
        status IN (
            'admitting', 'queued', 'running', 'succeeded', 'failed',
            'timed_out', 'cancelled', 'manual_reconciliation'
        )
    ),
    worker_id TEXT,
    account_ref TEXT,
    session_ref TEXT,
    attempt_count INTEGER NOT NULL DEFAULT 0 CHECK (attempt_count >= 0),
    result_json TEXT,
    artifact_manifest_json TEXT,
    wall_seconds DOUBLE PRECISION CHECK (wall_seconds IS NULL OR wall_seconds >= 0),
    failure_reason TEXT,
    started_at TIMESTAMPTZ,
    completed_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE (agenda_id, idempotency_key),
    UNIQUE (compute_job_id)
);
CREATE INDEX IF NOT EXISTS idx_colab_work_requests_v1_queue
    ON colab_work_requests_v1(status, created_at, id);

CREATE TABLE IF NOT EXISTS scoped_ingestion_jobs_v1 (
    id BIGSERIAL PRIMARY KEY,
    agenda_id BIGINT NOT NULL REFERENCES research_agendas(id),
    idea_id BIGINT NOT NULL,
    resource_grant_id BIGINT NOT NULL REFERENCES resource_grants(id),
    stage TEXT NOT NULL,
    idempotency_key TEXT NOT NULL,
    paper_ids_json TEXT NOT NULL,
    status TEXT NOT NULL CHECK (
        status IN (
            'queued', 'running', 'retryable', 'succeeded', 'failed',
            'manual_reconciliation', 'cancelled'
        )
    ),
    max_attempts INTEGER NOT NULL DEFAULT 3 CHECK (max_attempts > 0),
    attempt_count INTEGER NOT NULL DEFAULT 0 CHECK (attempt_count >= 0),
    lease_owner TEXT,
    lease_expires_at TIMESTAMPTZ,
    result_json TEXT,
    failure_reason TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    started_at TIMESTAMPTZ,
    completed_at TIMESTAMPTZ,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE (agenda_id, idempotency_key)
);
CREATE INDEX IF NOT EXISTS idx_scoped_ingestion_jobs_v1_queue
    ON scoped_ingestion_jobs_v1(status, created_at, id);

CREATE TABLE IF NOT EXISTS harness_candidates (
    id BIGSERIAL PRIMARY KEY,
    agenda_id BIGINT NOT NULL REFERENCES research_agendas(id),
    parent_archive_id BIGINT,
    candidate_ref TEXT NOT NULL,
    base_commit TEXT NOT NULL,
    worktree_path TEXT NOT NULL,
    database_namespace TEXT NOT NULL,
    artifact_namespace TEXT NOT NULL,
    status TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE (candidate_ref),
    UNIQUE (worktree_path),
    UNIQUE (database_namespace),
    UNIQUE (artifact_namespace)
);

CREATE TABLE IF NOT EXISTS harness_patches (
    id BIGSERIAL PRIMARY KEY,
    agenda_id BIGINT NOT NULL REFERENCES research_agendas(id),
    candidate_id BIGINT NOT NULL REFERENCES harness_candidates(id),
    base_commit TEXT NOT NULL,
    patch_hash TEXT NOT NULL,
    changed_modules_json TEXT NOT NULL,
    added_lines INTEGER NOT NULL CHECK (added_lines >= 0),
    deleted_lines INTEGER NOT NULL CHECK (deleted_lines >= 0),
    policy_version TEXT NOT NULL,
    UNIQUE (candidate_id, patch_hash)
);

CREATE TABLE IF NOT EXISTS failure_clusters (
    id BIGSERIAL PRIMARY KEY,
    agenda_id BIGINT NOT NULL REFERENCES research_agendas(id),
    cluster_key TEXT NOT NULL,
    signature_json TEXT NOT NULL,
    evidence_json TEXT NOT NULL,
    occurrence_count INTEGER NOT NULL CHECK (occurrence_count > 0),
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE (agenda_id, cluster_key)
);

CREATE TABLE IF NOT EXISTS harness_evaluation_runs (
    id BIGSERIAL PRIMARY KEY,
    agenda_id BIGINT NOT NULL REFERENCES research_agendas(id),
    candidate_id BIGINT NOT NULL REFERENCES harness_candidates(id),
    patch_id BIGINT NOT NULL REFERENCES harness_patches(id),
    suite TEXT NOT NULL CHECK (suite IN ('held_in', 'held_out', 'canary')),
    evaluator_ref TEXT NOT NULL,
    evaluator_hash TEXT NOT NULL,
    status TEXT NOT NULL,
    result_json TEXT,
    artifact_manifest_json TEXT,
    started_at TIMESTAMPTZ,
    completed_at TIMESTAMPTZ,
    UNIQUE (candidate_id, patch_id, suite)
);

CREATE TABLE IF NOT EXISTS harness_regression_reports (
    id BIGSERIAL PRIMARY KEY,
    agenda_id BIGINT NOT NULL REFERENCES research_agendas(id),
    candidate_id BIGINT NOT NULL REFERENCES harness_candidates(id),
    held_in_run_id BIGINT NOT NULL REFERENCES harness_evaluation_runs(id),
    held_out_run_id BIGINT NOT NULL REFERENCES harness_evaluation_runs(id),
    canary_run_id BIGINT NOT NULL REFERENCES harness_evaluation_runs(id),
    decision TEXT NOT NULL CHECK (decision IN ('reject', 'awaiting_approval', 'approved')),
    reviewer TEXT,
    approved_at TIMESTAMPTZ,
    report_json TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS harness_archives (
    id BIGSERIAL PRIMARY KEY,
    agenda_id BIGINT NOT NULL REFERENCES research_agendas(id),
    source_commit TEXT NOT NULL,
    source_tree_hash TEXT NOT NULL,
    policy_hash TEXT NOT NULL,
    evaluator_hash TEXT NOT NULL,
    holdout_hash TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE (agenda_id, source_commit, policy_hash, evaluator_hash, holdout_hash)
);

ALTER TABLE IF EXISTS experiment_runs
    ADD COLUMN IF NOT EXISTS scientific_evidence_state TEXT NOT NULL DEFAULT 'planned';
ALTER TABLE IF EXISTS experiment_runs
    ADD COLUMN IF NOT EXISTS scientific_reviewer_approved_by TEXT;
ALTER TABLE IF EXISTS experiment_runs
    ADD COLUMN IF NOT EXISTS scientific_reviewer_approved_at TIMESTAMPTZ;

CREATE TABLE IF NOT EXISTS evidence_state_transitions (
    id BIGSERIAL PRIMARY KEY,
    agenda_id BIGINT NOT NULL REFERENCES research_agendas(id),
    experiment_run_id BIGINT NOT NULL,
    from_state TEXT NOT NULL,
    to_state TEXT NOT NULL,
    actor TEXT NOT NULL,
    context_json TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_evidence_state_transition_run
    ON evidence_state_transitions(agenda_id, experiment_run_id, created_at, id);

CREATE TABLE IF NOT EXISTS reviewer_approval_records (
    id BIGSERIAL PRIMARY KEY,
    agenda_id BIGINT NOT NULL REFERENCES research_agendas(id),
    purpose TEXT NOT NULL,
    subject TEXT NOT NULL,
    reviewer_id TEXT NOT NULL,
    key_id TEXT NOT NULL,
    issued_at TIMESTAMPTZ NOT NULL,
    signature_hash TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE (purpose, subject)
);
CREATE INDEX IF NOT EXISTS idx_reviewer_approval_agenda
    ON reviewer_approval_records(agenda_id, purpose, created_at, id);

CREATE TABLE IF NOT EXISTS evidence_audit_records (
    id BIGSERIAL PRIMARY KEY,
    agenda_id BIGINT NOT NULL REFERENCES research_agendas(id),
    experiment_run_id BIGINT NOT NULL REFERENCES experiment_runs(id),
    raw_artifacts_hash TEXT NOT NULL,
    claim_ledger_hash TEXT NOT NULL,
    benchmark_contract_hash TEXT NOT NULL,
    evaluator_ref TEXT NOT NULL,
    evaluator_hash TEXT NOT NULL,
    holdout_ref TEXT NOT NULL,
    holdout_hash TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE (agenda_id, experiment_run_id)
);

CREATE TABLE IF NOT EXISTS scientific_decision_records (
    id BIGSERIAL PRIMARY KEY,
    agenda_id BIGINT NOT NULL REFERENCES research_agendas(id),
    experiment_run_id BIGINT NOT NULL REFERENCES experiment_runs(id),
    evidence_audit_record_id BIGINT NOT NULL REFERENCES evidence_audit_records(id),
    verdict TEXT NOT NULL
        CHECK (verdict IN ('supported', 'refuted', 'inconclusive')),
    verdict_hash TEXT NOT NULL,
    evidence_decision_json TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE (agenda_id, experiment_run_id)
);

CREATE INDEX IF NOT EXISTS idx_scientific_decision_scope
    ON scientific_decision_records(agenda_id, verdict, created_at, id);

-- The guarded migration runner writes the journal row with the file checksum
-- in the same transaction. This SQL file does not self-record an unverifiable
-- checksum.
