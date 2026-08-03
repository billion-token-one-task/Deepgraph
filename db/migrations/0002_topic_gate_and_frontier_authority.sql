-- Additive PostgreSQL migration for the topic gate and the Frontier-evaluator
-- bootstrap authority.
-- DO NOT run on production directly. Validate twice against an isolated
-- restore. No destructive table operation, no destructive type change, no
-- implicit backlog assignment, and no data backfill.

-- Topic gate: the recorded three-question answers and the minimum
-- falsification experiment for one candidate. Input only; the gate decision
-- itself is recomputed deterministically and stored with the selection.
ALTER TABLE IF EXISTS deep_insights
    ADD COLUMN IF NOT EXISTS topic_gate_json TEXT;

-- One narrowly scoped authority to produce exactly one Frontier assessment.
-- This is not a ResourceGrant: it has no backend allowlist, no GPU column, and
-- no decision packet, so it can never be widened into one.
CREATE TABLE IF NOT EXISTS frontier_evaluation_authorities (
    id BIGSERIAL PRIMARY KEY,
    agenda_id BIGINT NOT NULL REFERENCES research_agendas(id),
    research_problem_id BIGINT NOT NULL REFERENCES research_problems(id),
    token_cap BIGINT NOT NULL CHECK (token_cap > 0 AND token_cap <= 20000),
    issued_at TIMESTAMPTZ NOT NULL,
    expires_at TIMESTAMPTZ NOT NULL,
    idempotency_key TEXT NOT NULL,
    provider TEXT NOT NULL,
    model TEXT NOT NULL,
    model_family TEXT NOT NULL,
    prompt_version TEXT NOT NULL,
    evaluator TEXT NOT NULL,
    issued_by TEXT NOT NULL,
    issue_reason TEXT NOT NULL,
    reservation_id BIGINT NOT NULL REFERENCES agenda_resource_ledger(id),
    status TEXT NOT NULL DEFAULT 'active'
        CHECK (status IN ('active', 'consumed', 'expired', 'revoked')),
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    closed_at TIMESTAMPTZ,
    CHECK (expires_at > issued_at),
    UNIQUE (agenda_id, idempotency_key)
);

CREATE INDEX IF NOT EXISTS idx_frontier_authorities_admission
    ON frontier_evaluation_authorities(
        agenda_id, research_problem_id, status, expires_at, id
    );

-- Append-only usage ledger. One row per attempt, successful or not, so a
-- reviewer can verify what the authority actually spent and produced.
CREATE TABLE IF NOT EXISTS frontier_authority_usage (
    id BIGSERIAL PRIMARY KEY,
    authority_id BIGINT NOT NULL REFERENCES frontier_evaluation_authorities(id),
    agenda_id BIGINT NOT NULL REFERENCES research_agendas(id),
    research_problem_id BIGINT NOT NULL REFERENCES research_problems(id),
    operation TEXT NOT NULL CHECK (operation = 'frontier_assessment'),
    provider TEXT NOT NULL,
    model TEXT NOT NULL,
    model_family TEXT NOT NULL,
    prompt_version TEXT NOT NULL,
    input_tokens BIGINT NOT NULL DEFAULT 0 CHECK (input_tokens >= 0),
    output_tokens BIGINT NOT NULL DEFAULT 0 CHECK (output_tokens >= 0),
    cost_usd DOUBLE PRECISION CHECK (cost_usd IS NULL OR cost_usd >= 0),
    status TEXT NOT NULL CHECK (status IN ('succeeded', 'failed')),
    failure_reason TEXT,
    frontier_packet_id BIGINT REFERENCES frontier_packets(id),
    evidence_query_ref TEXT NOT NULL DEFAULT '',
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CHECK (status <> 'succeeded' OR frontier_packet_id IS NOT NULL)
);

CREATE INDEX IF NOT EXISTS idx_frontier_authority_usage_scope
    ON frontier_authority_usage(agenda_id, authority_id, status, id);

-- Which authority produced a packet, so the Frontier gate record stays
-- independently verifiable.
ALTER TABLE IF EXISTS frontier_packets
    ADD COLUMN IF NOT EXISTS frontier_evaluation_authority_id BIGINT
    REFERENCES frontier_evaluation_authorities(id);
