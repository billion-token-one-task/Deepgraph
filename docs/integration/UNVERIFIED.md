# Unverified and incomplete items

The implementation is not eligible to replace master. The following are
deliberately not claimed as working:

- no pytest or application test was run;
- no module import smoke or application startup was run;
- the system Python is 3.9 and has neither `tomllib` nor `tomli`, so TOML
  runtime parsing remains for the pinned CI/application interpreter;
- no PostgreSQL migration or query was executed;
- no restored production backup was started;
- no SQLite meta-harness acceptance was attempted;
- no provider, CPU experiment, GPU, SSH, Colab, build or dependency install ran;
- no candidate worktree was created and no held-in/out/canary suite ran;
- Colab CLI upload/exec/download argument compatibility is unverified;
- `colab_work_requests_v1`, its claim-before-session worker, restart
  quarantine and artifact/usage settlement have not run against PostgreSQL or
  a synthetic/real CLI;
- legacy scheduler worker internals still contain transport branches, although
  new CPU/local/SSH admission and durable settlement now pass through
  `ComputeScheduler`, and PostgreSQL direct queue insertion requires the
  persisted durable identity; runtime restart behavior is unverified;
- proposal, ingestion/enrichment, benchmark, forge, validation, manuscript
  revision and plain-review core LLM paths are granted/role-routed; the direct
  caller inventory is statically complete; the scoped ingestion API, durable
  lease/checkpoint worker and bounded retry code now exist, but their
  PostgreSQL/ledger/provider-failure behavior has not run;
- canonical scientific-state SQL writes pass a static authority audit, but
  legacy operational/verdict semantics are not end-to-end verified;
- durable compute claim/reuse/quarantine code and isolated PostgreSQL tests
  exist; compute recovery is ordered before auto-research startup, but
  exactly-once behavior across real process/backend crash windows is not
  end-to-end verified;
- non-success compute usage settlement is implemented, but no isolated
  PostgreSQL/backend fault test has verified it;
- the agenda mutation AST audit is clean, but PostgreSQL concurrency/fault
  execution has not proved cross-agenda isolation at runtime;
- signed reviewer approval is implemented, but reviewer identity/key issuance,
  rotation and external authentication have not been exercised;
- the hash-pinned bubblewrap evaluator runner and mutation route are written,
  but bubblewrap availability, mount isolation, candidate immutability and all
  three suites have not been exercised;
- durable provider cooldown persistence exists but restart behavior has not
  run against isolated PostgreSQL;
- calibration reports have no real OutcomeRecord sample;
- minimal API registration is statically reviewed but never served;
- legacy Web 410 and agenda-scoped read behavior has test material but was not
  exercised;
- production rollback rehearsal has not occurred;
- single-operator/deployment-quiescence control was not established, so no
  remote mutation was attempted.

The current safe evidence is limited to Git/ref inspection, source diff review,
AST parsing, text scans, migration-plan rendering, documentation, and test
code authored for later isolated execution.
