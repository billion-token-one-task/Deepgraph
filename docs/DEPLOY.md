# Deployment guide

Everything needed to run DeepGraph, from a laptop demo to a production box
with remote GPU workers.

## Requirements

- Python 3.12+ (`pyproject.toml`)
- An LLM API key (any OpenAI-compatible endpoint)
- PostgreSQL (for the full control plane; the engine and dashboard start
  without it)
- Optional: GROBID for high-quality PDF parsing (Docker compose file included:
  `docker-compose.grobid.yml`)

## Five-minute start

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env    # set at least DEEPGRAPH_LLM_API_KEY
export $(grep -v '^#' .env | xargs)
python3.12 main.py
```

Open `http://localhost:8080`. This runs ingestion, the graph, discovery, and
the dashboard in one process.

## PostgreSQL and the control plane

The meta-harness control plane (grants, evidence ladder, agendas) needs
PostgreSQL. Migrations are additive-only.

1. Preview, then apply the migration (rehearse against a disposable restore
   first -- see [internal/MIGRATION_RUNBOOK.md](internal/MIGRATION_RUNBOOK.md);
   never point a rehearsal at a production URL):

   ```bash
   python3.12 scripts/meta_harness_migration.py            # dry-run plan
   python3.12 scripts/meta_harness_migration.py \
     --apply \
     --confirm-isolated-restore I_UNDERSTAND_THIS_WRITES_AN_ISOLATED_RESTORE \
     --source-commit "$(git rev-parse HEAD)"
   ```

2. Start with the CPU backend and an operator token from your secret store:

   ```bash
   export DEEPGRAPH_COMPUTE_BACKENDS=cpu
   export DEEPGRAPH_META_HARNESS_OPERATOR_TOKEN='<secret-store-injected>'
   python3.12 main.py
   ```

3. Verify the control plane (returns counts only, never business rows):

   ```bash
   curl http://localhost:8080/api/meta-harness/v1/status
   ```

Mutating endpoints under `/api/meta-harness/v1/*` require the
`X-DeepGraph-Operator-Token` header; if the token env var is unset, the
mutation API is disabled entirely. Missing scope, expired grants, and unknown
backend states fail closed.

## Remote GPU workers (SSH mode)

Training compute runs on SSH-reachable GPU machines; the control-plane host
needs no GPU of its own (an idle local `nvidia-smi` is normal and expected).

```bash
export DEEPGRAPH_COMPUTE_BACKENDS=cpu,ssh_gpu
export DEEPGRAPH_GPU_MODE=ssh
export DEEPGRAPH_GPU_REMOTE_SSH_HOST='<approved-host>'
export DEEPGRAPH_GPU_REMOTE_SSH_USER='<approved-user>'
export DEEPGRAPH_COMPUTE_SSH_TARGET_REF=env:DEEPGRAPH_SSH_TARGET
export DEEPGRAPH_COMPUTE_SSH_CREDENTIAL_REF=env:DEEPGRAPH_SSH_CREDENTIAL
export DEEPGRAPH_SSH_KNOWN_HOSTS='<reviewed-known-hosts-file>'
```

Credentials are secret *references*, never stored values, and host identity is
pinned via known-hosts.

A freshly configured backend is `unknown`, not `enabled`: it becomes
schedulable only after an operator records a successful canary in
`DEEPGRAPH_COMPUTE_VERIFIED_BACKENDS`. Audit backend state read-only with
`python3.12 scripts/meta_harness_backend_audit.py`.

Execution flow per run: the local workdir is rsynced to the remote base dir,
the command runs under `CUDA_VISIBLE_DEVICES`, and artifacts are rsynced back
(`orchestrator/ssh_gpu_backend.py`). Remote dependency install is automatic by
default (`DEEPGRAPH_GPU_REMOTE_AUTO_PIP_INSTALL=true`,
`DEEPGRAPH_GPU_REMOTE_SETUP_TIMEOUT_SECONDS=3600`); for production, prefer a
pre-built conda/venv on the GPU host, point `DEEPGRAPH_GPU_REMOTE_PYTHON` at
it, and set auto-install to `false`.

### GPU debugging checklist

1. Remember which machine computes: check `gpu_jobs.assigned_worker` and
   `gpu_workers.metadata` for the SSH target before staring at local
   `nvidia-smi`.
2. Read `{workdir}/run.log` -- remote stdout/stderr aggregates there. Search
   for `ModuleNotFoundError`, `CUDA`, `timeout`, `rsync`.
3. Compare `gpu_jobs.status` against `experiment_iterations` growth: `running`
   with frozen iterations means stuck in sync/install/first step.
4. Reproduce by hand on the GPU host inside
   `<remote_base_dir>/runs/run_<id>/code` with the same command.

## Production (systemd)

Systemd units, the Caddyfile, and SHA-pinned deployment manifests live in
`deploy/`. The manifests record source, target, and SHA256 for every deployed
file, plus rollback artifacts -- deploy from the manifest, not by hand. The
self-heal watchdog (`deploy/deepgraph-selfheal.timer`) restarts the web
service on sustained health-probe failure and holds in every ambiguous case;
reason codes and operator responses:
[internal/runbooks/SELFHEAL.md](internal/runbooks/SELFHEAL.md).

Runbooks for recovery baselines, canaries, and rollback are in
[internal/runbooks/](internal/runbooks/RECOVERY_2026-08-03.md) and
[internal/](internal/ROLLBACK_RUNBOOK.md).

## Configuration reference

Defaults live in `deepgraph.toml`; environment variables and `.env` override
TOML. The most important variables:

- `DEEPGRAPH_LLM_API_KEY` -- required; LLM key for extraction and generation
- `DEEPGRAPH_LLM_SECONDARY_*` -- optional second OpenAI-compatible route
- `DEEPGRAPH_LLM_EXTRA_PROVIDERS_JSON` -- optional additional routes
- `DEEPGRAPH_DATABASE_URL` -- PostgreSQL DSN
- `DEEPGRAPH_META_HARNESS_OPERATOR_TOKEN` -- enables the operator mutation API
- `DEEPGRAPH_COMPUTE_BACKENDS` -- offered backends, e.g. `cpu,ssh_gpu`
- `DEEPGRAPH_COMPUTE_VERIFIED_BACKENDS` -- canaried backends; only these schedule
- `DEEPGRAPH_AUTO_RESEARCH_ENABLED` / `DEEPGRAPH_AUTO_PIPELINE_ENABLED` --
  global autonomy switches for discovery and ingestion
- `DEEPGRAPH_TOPIC_GATE_*` -- gate thresholds (surprise bits, confidence
  ceiling); the gate itself has no off switch
- `DEEPGRAPH_PROFILE` -- `machine_learning` (default) or `open_science`
- `DEEPGRAPH_WEB_PORT` -- dashboard port, default 8080
- `DEEPGRAPH_BULK_*` -- discovery pipeline tuning

Full policy and secret-reference documentation:
[internal/CONFIGURATION.md](internal/CONFIGURATION.md).

The `open_science` profile widens taxonomy coverage beyond ML to mathematics,
physics, chemistry, life sciences, medicine, earth science, and engineering:

```bash
export DEEPGRAPH_PROFILE=open_science
export DEEPGRAPH_ROOT_NODE_ID=science
python3.12 main.py
```

## Tests and audits

```bash
python3.12 -m unittest discover -s tests     # 90 test modules
python3.12 scripts/meta_harness_scope_audit.py
python3.12 scripts/meta_harness_sql_audit.py
python3.12 scripts/meta_harness_static_audit.py
python3.12 scripts/meta_harness_state_authority_audit.py
python3.12 scripts/meta_harness_llm_caller_audit.py
```

The audit scripts are read-only and safe to run against a live deployment.
