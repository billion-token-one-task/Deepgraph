# Experiment Execution Agent

Owns validation-loop execution, local and SSH GPU workers, remote shards, health checks, and merge watchers.

Primary legacy modules:

- `agents.validation_loop`
- `agents.codex_executor`
- `orchestrator.gpu_scheduler`
- `orchestrator.ssh_gpu_backend`
- `orchestrator.benchmark_completion`
- `orchestrator.tracking`

Primary scripts:

- `scripts.run_gpu_scheduler_forever`

Configuration lives in `deepgraph.toml` under `experiment`, `gpu`, `runtime`, `tracking`, and `paths`.
