# Orchestration Agent

Owns the end-to-end pipeline, background scheduling, workspace layout, web service, and deployment hooks.

Primary legacy modules:

- `agents.workspace_layout`
- `orchestrator.auto_research`
- `orchestrator.discovery_scheduler`
- `orchestrator.manuscript_watchdog`
- `orchestrator.paper_worker`
- `orchestrator.pipeline`
- `web.app`

Primary scripts:

- `scripts.backfill_idea_workspaces`
- `scripts.migrate_sqlite_to_postgres`
- `scripts.run_pipeline_forever`
- `scripts.run_tunnel_forever`
- `scripts.run_web_forever`

Configuration lives in `deepgraph.toml` under `app`, `auto_research`, `pipeline`, `web`, `paths`, and `database`.

