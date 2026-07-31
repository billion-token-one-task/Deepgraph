# Schema delta and migration policy

This inventory is based on source SQL and Git diffs only. No migration was
executed and no production business rows were read.

## Cross-line observations

- GitHub master contains the newer `research_problems`,
  `experimental_evidence_edges`, `benchmark_harness_jobs`, and pipeline event
  structures.
- The production snapshot contains Agenda schema in separate
  `schema_agenda*.sql` files and Agenda behavior in startup migrations, but
  lacks parts of GitHub's newer problem-first/benchmark schema.
- Production added `agenda_id` primarily to `deep_insights`; the v1 requirement
  is broader: problems, ideas, jobs, runs, artifacts, grants, decisions,
  outcomes, and ledger entries all require agenda scope.
- Existing production row counts supplied by the operator are migration inputs,
  not test fixtures. They were not independently queried in this session.
- Tree inspection additionally proved that production `7d0b42a` has no
  `research_problems`, `experimental_evidence_edges`, or
  `benchmark_harness_jobs`, while GitHub master has all three. Migration `0001`
  therefore creates those GitHub-line tables additively before adding indexes;
  relying on target application startup would make backup migration order
  ambiguous.

## Additive migration set

`0001_meta_harness_v1.sql` must:

1. create the three missing GitHub problem/evidence/benchmark tables plus
   Agenda, Frontier, portfolio, grant, outcome, route-observation,
   provider-cooldown, compute-job, evidence-audit/decision and
   harness-evolution tables with checks and foreign keys;
2. add nullable `agenda_id` to legacy core tables so existing rows survive;
3. create indexes concurrently only in a separately approved deployment step,
   or use ordinary indexes in an isolated restored database;
4. avoid delete, truncate, rename, destructive type conversion, and automatic
   backlog assignment;
5. record a schema-migration journal row with a checksum;
6. remain safe to execute twice.

New v1 rows use application and database constraints to require `agenda_id`.
Legacy tables need nullable columns for add-only compatibility; the runtime
rejects `NULL` for all newly created scoped objects. `legacy_scope_imports`
is the only mechanism that can bind an old row to an agenda.

## Migration verification (isolated PostgreSQL only)

1. Restore a production backup into a disposable database/namespace.
2. Run preflight counts and foreign-key orphan queries.
3. Render and review the migration plan with credentials redacted.
4. Execute the migration once; compare all pre-existing table counts.
5. Execute it a second time; assert no changes other than migration audit
   timestamps explicitly designed to change.
6. Run new constraint, grant, queue isolation, and restart recovery tests.
7. Roll back by restoring the disposable pre-migration snapshot. The production
   rollback strategy is version switch plus database restore, not destructive
   down-migration.

SQLite cannot substitute for this verification. If retained, it is a fast
developer compatibility lane only.

## Current static migration plan

After the missing-table correction, the side-effect-free planner reported:

- 84 statements;
- 24,759 bytes;
- SHA-256
  `dd64219c5b4189093deb4ace3f87a3a658696a07695d279487f32eba5b7e38de`;
- no destructive token;
- `database_accessed=false`.

This is an intermediate working-tree checksum. Regenerate it from the final
candidate commit; never apply a migration whose checksum differs from the
reviewed ticket.
