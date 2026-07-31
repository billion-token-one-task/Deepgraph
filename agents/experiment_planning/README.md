# Experiment Planning Agent

Owns benchmark contracts, experiment scaffolding, evidence review, and paper artifact audits.

Primary legacy modules:

- `agents.experiment_forge`
- `agents.experiment_supervisor`
- `agents.experiment_review`
- `agents.benchmark_audit`
- `agents.result_interpreter`
- `agents.evosci_requirements`

Primary scripts:

- Topic-specific benchmark auditors are opt-in examples under
  `plugins/examples/cggr/scripts`; the generic planning boundary does not
  register them.

Topic-specific CGGR planning utilities are isolated under
`plugins/examples/cggr` and are not part of this generic boundary.

Configuration lives in `deepgraph.toml` under `experiment`, `codex`, `gpu`, `tracking`, and `paths`.
