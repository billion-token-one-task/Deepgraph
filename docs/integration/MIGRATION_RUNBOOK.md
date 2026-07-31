# Isolated PostgreSQL migration runbook

This runbook is for a disposable PostgreSQL restore in isolated CI or a
dedicated canary machine. It is not authorization to touch production. There
is no destructive down-migration.

## Preconditions

1. Obtain explicit approval for the database write.
2. Confirm the target database name contains `test`, `ci`, `canary`,
   `sandbox`, `staging`, `restore`, or `shadow`.
3. Confirm its URL differs from `DEEPGRAPH_DATABASE_URL`.
4. Restore a fresh production backup without reading secrets into logs.
5. Record the candidate's full commit hash and the backup identifier.
6. Stop if any application process points at the disposable database.

The production backup is expected to lack GitHub-line
`research_problems`, `experimental_evidence_edges`, and
`benchmark_harness_jobs`. Migration `0001` creates those tables additively
before indexing or scoping them.

## Read-only plan

This command reads only the SQL file and does not import DeepGraph or access a
database:

```bash
python3 scripts/meta_harness_migration.py
```

Review `sha256`, byte count, statement count, and require
`destructive_tokens=[]`. Pin that checksum in the change ticket.

## Pre-migration evidence

In the isolated restore only, save:

- schema-only dump;
- `information_schema.columns` inventory;
- `SELECT COUNT(*)` for every pre-existing application table;
- count of non-null `agenda_id` values where that column already exists;
- database name, server version, backup identifier, and candidate commit.

Do not copy business row contents into CI logs or acceptance artifacts.

## Apply once

Set the credential in the isolated secret store, not in TOML or shell history:

```bash
export DEEPGRAPH_MIGRATION_DATABASE_URL='postgresql://.../deepgraph_canary_restore'
python3 scripts/meta_harness_migration.py \
  --apply \
  --confirm-isolated-restore I_UNDERSTAND_THIS_WRITES_AN_ISOLATED_RESTORE \
  --source-commit '<40-character-candidate-commit>'
```

The runner uses one transaction, a five-second lock timeout, a 120-second
statement timeout, and a checksum journal. Any error must roll back the whole
migration.

## Apply twice and verify

Run the same command again. It must report `already_applied` with the same
checksum. Then verify:

- every pre-existing table count is unchanged;
- old `deep_insights` and `auto_research_jobs` did not acquire `agenda_id`;
- new tables and indexes exist;
- no `DROP`, `TRUNCATE`, delete, rename, or destructive type change occurred;
- journal contains exactly one row for `0001_meta_harness_v1`.

The isolated integration test automates these assertions:

```bash
DEEPGRAPH_ALLOW_ISOLATED_INTEGRATION_TESTS=1 \
DEEPGRAPH_ISOLATED_POSTGRES_URL='postgresql://.../deepgraph_ci_restore' \
META_HARNESS_CANDIDATE_COMMIT='<40-character-candidate-commit>' \
pytest -q tests/integration/test_meta_harness_postgres.py
```

This command is forbidden on the current production-adjacent host.

After it passes, the durable compute lifecycle test may run in a separate
process against the same disposable restore:

```bash
DEEPGRAPH_ALLOW_ISOLATED_INTEGRATION_TESTS=1 \
DEEPGRAPH_ISOLATED_POSTGRES_URL='postgresql://.../deepgraph_ci_restore' \
META_HARNESS_CANDIDATE_COMMIT='<40-character-candidate-commit>' \
pytest -q tests/integration/test_compute_repository_postgres.py
```

It writes and removes synthetic Agenda/grant/compute rows only in that
isolated database. It verifies live-job reuse after repository restart,
unknown-submission quarantine, and artifact/usage-gated success. It does not
contact a real compute backend.

Run the evidence persistence case in another process:

```bash
DEEPGRAPH_ALLOW_ISOLATED_INTEGRATION_TESTS=1 \
DEEPGRAPH_ISOLATED_POSTGRES_URL='postgresql://.../deepgraph_ci_restore' \
META_HARNESS_CANDIDATE_COMMIT='<40-character-candidate-commit>' \
pytest -q tests/integration/test_evidence_repository_postgres.py
```

That case writes and removes synthetic scoped records. It verifies
stage-specific grants, content-addressed raw/claim/benchmark/evaluator/holdout
inputs, the M1/M4 decision contract, and reviewer-gated manuscript permission.

## Failure handling

On a failed first apply, keep the logs and discard the restore. Do not repair
the database in place. Fix the migration, create a new migration key/checksum
if an earlier checksum was accepted anywhere, restore the backup again, and
repeat from preflight.
