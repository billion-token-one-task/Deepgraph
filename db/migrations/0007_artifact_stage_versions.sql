-- Append-only stage versions for runner artifacts.
--
-- A pilot and its later full benchmark reuse results/<name>. Registering that
-- mutable path once left the pilot SHA in metadata after the bytes had been
-- replaced by the full benchmark. New rows identify their producing stage and
-- content version; the application points `path` at a content-addressed copy.
-- Legacy rows remain untouched and readable through the original columns.

ALTER TABLE IF EXISTS experiment_artifacts
    ADD COLUMN IF NOT EXISTS artifact_stage TEXT;
ALTER TABLE IF EXISTS experiment_artifacts
    ADD COLUMN IF NOT EXISTS artifact_version INTEGER NOT NULL DEFAULT 1;
ALTER TABLE IF EXISTS experiment_artifacts
    ADD COLUMN IF NOT EXISTS content_sha256 TEXT;

CREATE INDEX IF NOT EXISTS idx_experiment_artifacts_stage
    ON experiment_artifacts(run_id, artifact_type, artifact_stage, artifact_version);

CREATE UNIQUE INDEX IF NOT EXISTS idx_experiment_artifacts_stage_content
    ON experiment_artifacts(run_id, artifact_type, artifact_stage, content_sha256)
    WHERE artifact_stage IS NOT NULL AND content_sha256 IS NOT NULL;
