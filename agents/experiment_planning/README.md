# Experiment Planning Agent

Owns benchmark contracts, experiment scaffolding, evidence review, and paper artifact audits.

Primary legacy modules:

- `agents.experiment_forge`
- `agents.experiment_supervisor`
- `agents.experiment_review`
- `agents.benchmark_audit`
- `agents.result_interpreter`
- `agents.evosci_requirements`

Configuration lives in `deepgraph.toml` under `experiment`, `codex`, `gpu`, `tracking`, and `paths`.
