# Experiment Execution Agent

Owns validation-loop execution, local and SSH GPU workers, remote shards, health checks, and merge watchers.

Primary legacy modules:

- `agents.validation_loop`
- `agents.codex_executor`
- `agents.experiment_executor`
- `agents.metric_parser`
- `orchestrator.gpu_scheduler`
- `orchestrator.benchmark_completion`
- `orchestrator.tracking`

Primary scripts:

- `scripts.run_gpu_scheduler_forever`
- `scripts.stage_and_launch_cggr_top_venue_baseline_shard`
- `scripts.watch_and_merge_cggr_shards`
- `scripts.watch_cggr_live_health`

Configuration lives in `deepgraph.toml` under `experiment`, `gpu`, `runtime`, `tracking`, and `paths`.

