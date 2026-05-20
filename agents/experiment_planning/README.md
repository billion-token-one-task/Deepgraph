# Experiment Planning Agent

Owns benchmark contracts, experiment scaffolding, evidence review, and paper artifact audits.

Primary legacy modules:

- `agents.experiment_forge`
- `agents.experiment_supervisor`
- `agents.experiment_progress_monitor` — live progress from artifacts/processes
- `agents.experiment_feasibility_agent` — viability verdict before/during runs
- `agents.experiment_tuning_agent` — scale/time/retry/finalize decisions
- `agents.experiment_watchdog` — coordinates the three agents in production
- `agents.experiment_review`
- `agents.benchmark_audit`
- `agents.result_interpreter`
- `agents.evosci_requirements`

Primary scripts:

- `scripts.audit_cggr_shard_contract`
- `scripts.audit_paper_benchmark_artifacts`
- `scripts.materialize_audited_cggr_results`
- `scripts.merge_cggr_method_shards`
- `scripts.prepare_cggr_top_venue_baseline_shard`
- `scripts.triage_cggr_audit_failure`

Configuration lives in `deepgraph.toml` under `experiment`, `codex`, `gpu`, `tracking`, and `paths`.

