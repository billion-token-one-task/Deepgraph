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
- legacy scheduler transport branches are not fully replaced by
  `ComputeScheduler`;
- forge and validation repair LLM paths are granted/role-routed, but
  pre-idea discovery and manuscript/refinement callers are not exhaustively
  moved to `call_llm_for_role`; a pre-candidate grant contract remains open;
- legacy operational/scientific state writes are not exhaustively routed
  through the canonical evidence transition repository;
- durable compute claim/reuse/quarantine code and isolated PostgreSQL tests
  exist, but service startup wiring and exactly-once behavior across real
  process/backend crash windows are not end-to-end verified;
- the operator API assembles successful-run OutcomeRecord values from durable
  metering/artifacts/decisions, but failed compute jobs do not yet persist
  enough usage data for trusted automatic settlement;
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
