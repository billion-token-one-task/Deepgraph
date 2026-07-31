# Durable queue validation

No command in this document has been run on the current host.

## Scoped ingestion

Against a disposable PostgreSQL restore, enqueue only paper IDs already in
`papers` through `POST /api/meta-harness/v1/ingestion/jobs`. Verify exact
agenda/idea/stage/grant matching, `llm` allowlisting and idempotent replay.
Crash the worker after claim and after one paper checkpoint. After lease
expiry, confirm checkpoint resume without duplicate scientific rows, bounded
retry, and `manual_reconciliation` or `failed` after exhaustion. Expire or
revoke the grant between papers and confirm no later provider call occurs.

## Colab

Use a synthetic CLI first. Enqueue through
`POST /api/meta-harness/v1/compute/colab/jobs`; verify the work row and
`compute_jobs_v1` claim exist before any `new` invocation. Inject crashes:

1. after work-row insert but before compute claim;
2. after compute bind but before work-row bind;
3. after worker claim but before `new`;
4. after `new`, upload, exec and download;
5. during stop and artifact hashing.

Safe bind gaps may be rebound. Any lost claimed/remote session must become
`manual_reconciliation` plus compute `usage_unknown`, never an automatic
resubmission or success. Only complete grant-named artifacts and bounded
measured wall/GPU usage may settle `succeeded`.

For local/SSH PostgreSQL transport, call the legacy queue without a durable
idempotency identity and confirm it is rejected. Then submit through
`ComputeScheduler` and verify the identical key is persisted on both
`compute_jobs_v1` and `gpu_jobs`.
