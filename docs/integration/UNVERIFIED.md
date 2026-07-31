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
- legacy scheduler worker internals still contain transport branches, although
  new CPU/local/SSH admission and durable settlement now pass through
  `ComputeScheduler`; runtime restart behavior is unverified;
- proposal, benchmark, forge, validation, manuscript revision and plain-review
  core LLM paths are granted/role-routed, but the remaining direct caller
  inventory is not yet fully classified or removed;
- canonical scientific-state SQL writes pass a static authority audit, but
  legacy operational/verdict semantics are not end-to-end verified;
- durable compute claim/reuse/quarantine code and isolated PostgreSQL tests
  exist, but service startup wiring and exactly-once behavior across real
  process/backend crash windows are not end-to-end verified;
- non-success compute usage settlement is implemented, but no isolated
  PostgreSQL/backend fault test has verified it;
- the agenda mutation AST audit is clean, but PostgreSQL concurrency/fault
  execution has not proved cross-agenda isolation at runtime;
- signed reviewer approval is implemented, but reviewer identity/key issuance,
  rotation and external authentication have not been exercised;
- durable provider cooldown persistence exists but restart behavior has not
  run against isolated PostgreSQL;
- calibration reports have no real OutcomeRecord sample;
- minimal API registration is statically reviewed but never served;
- production rollback rehearsal has not occurred;
- single-operator/deployment-quiescence control was not established, so no
  remote mutation was attempted.

The current safe evidence is limited to Git/ref inspection, source diff review,
AST parsing, text scans, migration-plan rendering, documentation, and test
code authored for later isolated execution.
