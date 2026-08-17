# Web data recovery deployment — 2026-08-17

Release directory: `/home/billion-token/releases/deepgraph-v1-webfix-20260817`

The protected source snapshot at `/home/billion-token/Deepgraph` is never
modified.  Its `.env`, Python environment, logs, and durable workspace remain
the configured runtime dependencies while application code runs from the
immutable release directory.

## Pre-cutover gates

1. `deepgraph-postgres.service` is active and a read-only `SELECT 1` succeeds.
2. No experiment, GPU job, Colab request, resource reservation, or GPU broker
   lease is running.
3. Candidate `/api/health/data`, `/api/stats`, `/api/processing`, and
   `/api/agent_office` return structured JSON against production PostgreSQL.
4. Record the candidate commit, old unit hash, new unit hash, and a copy of the
   installed old unit.

## Cutover

1. Copy the committed candidate tree to the release directory without `.git`.
2. Install `deploy/deepgraph-web.service` and run `systemctl daemon-reload`.
3. Restart only `deepgraph-web.service`.
4. Require `active/running`, then verify the four data endpoints.  The homepage
   alone is not a health check.

## Automatic rollback trigger

Restore the saved unit and restart `deepgraph-web.service` if any critical data
endpoint is non-200, non-JSON, reports unhealthy/unavailable, or lacks its
required structured payload after the bounded startup window.  The database is
not migrated by this change, so rollback is a version/unit switch only.

## Public statistics presentation

The public dashboard presents a research funnel rather than merging records
with different scientific meanings:

- corpus total, analyzed papers, awaiting-analysis papers, and errors;
- extracted structured results and literature insights;
- graph entities and graph relations;
- generated paper ideas, experiment runs with completed/failed breakdown, and
  formally adjudicated findings.

Large-table counts use PostgreSQL planner estimates and are prefixed with `~`.
Small operational tables use cached exact counts. `/api/health/data` fails
closed if a table is non-empty while its public count is zero. Stale database
rows no longer make `/api/processing` claim that the worker is running.

## LLM credential override

Run `sudo scripts/configure_llm_runtime.sh` from the immutable release checkout.
The script reads the key with hidden terminal input, checks `/models` by
default, stores it in root-only `/etc/deepgraph/runtime.env`, installs Web and
execution-worker systemd drop-ins, and reloads systemd. It deliberately does
not restart either service; use the normal no-active-work gate first.
